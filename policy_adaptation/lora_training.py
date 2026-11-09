"""Minimal LoRA layers and parameter-efficient model adaptation.

A frozen linear weight W is augmented by (alpha / r) BA. The implementation
supports nested projection injection, trainable-parameter counting and a
numerically checked merge operation. The executable example loads Qwen2.5-0.5B;
importing the module does not download a model or start training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


# ---------------------------
# 1) LoRA Linear
# ---------------------------
class LoRALinear(nn.Module):
    """
    Minimal LoRA: add a low-rank (A: in->r, B: r->out) side path
    on top of an existing nn.Linear.

    Forward (not merged):
        out = x @ W^T (+ bias) + ( x -> A -> B ) * (alpha / r)

    Forward (merged):
        out = x @ (W + delta_W)^T (+ bias), bypass disabled.

    The base parameters are shared with the original layer. A is initialized
    with Kaiming uniform and B with zeros, so injection preserves its output.
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be an nn.Linear")
        if isinstance(r, bool) or not isinstance(r, int) or r < 0:
            raise ValueError("r must be a non-negative integer")
        if not math.isfinite(float(alpha)):
            raise ValueError("alpha must be finite")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        self.lora_A = None
        self.lora_B = None
        if r > 0:
            factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}
            self.lora_A = nn.Linear(self.in_features, r, bias=False, **factory_kwargs)
            self.lora_B = nn.Linear(r, self.out_features, bias=False, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False
        self.reset_parameters()
        self.train(base_linear.training)

    def reset_parameters(self):
        """Initialize A with Kaiming uniform and B with zeros."""
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    @torch.no_grad()
    def merge(self):
        """Add (alpha / r) BA to the base weight once and disable the bypass."""
        if self.merged:
            return
        if self.r > 0:
            delta_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.weight.add_(delta_weight)
        self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the base layer and the unmerged low-rank update."""
        out = F.linear(x, self.weight, self.bias)
        if not self.merged and self.r > 0:
            out = out + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return out


# ----------------------------------------
# 2) Inject LoRA into target Linear layers
# ----------------------------------------
TARGET_LINEAR_KEYWORDS = [
    # Common attention/MLP projection names (Qwen/LLaMA style)
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

def inject_lora_modules(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> int:
    """
    Replace specific nn.Linear submodules (whose qualified names end with any of
    TARGET_LINEAR_KEYWORDS) with LoRALinear wrappers.

    Preserve device, dtype and training mode; return the number of replacements.
    """
    replaced = 0
    # Snapshot the traversal before replacing any child modules.
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.endswith(tuple(TARGET_LINEAR_KEYWORDS)):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            lora_module = LoRALinear(module, r, alpha, dropout)
            lora_module = lora_module.to(device=module.weight.device, dtype=module.weight.dtype)
            setattr(parent, child_name, lora_module)
            replaced += 1
    return replaced


# -------------------------------------------------------
# 3) Model parameter statistics
# -------------------------------------------------------
def count_model_params(model: nn.Module) -> Tuple[int, int, float]:
    """Return (trainable_params, total_params, trainable / total)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total > 0 else 0.0
    return trainable, total, ratio


# -------------------------------------------------------
# Helper: mark only LoRA parameters as trainable
# -------------------------------------------------------
def mark_only_lora_as_trainable(model: nn.Module) -> int:
    """
    Freeze all params by default, then unfreeze LoRA matrices (A and B).
    Returns the number of trainable parameters after marking.
    """
    for p in model.parameters():
        p.requires_grad = False

    trainable = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if hasattr(m, "lora_A") and hasattr(m.lora_A, "weight"):
                m.lora_A.weight.requires_grad = True
                trainable += m.lora_A.weight.numel()
            if hasattr(m, "lora_B") and hasattr(m.lora_B, "weight"):
                m.lora_B.weight.requires_grad = True
                trainable += m.lora_B.weight.numel()
    return trainable


# -------------------------------------------
# 4) An optional model-loading smoke test
# -------------------------------------------
def select_device_dtype():
    """Use BF16 only when the CUDA device supports it natively."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    native_bf16 = device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    return device, torch.bfloat16 if native_bf16 else torch.float32


def main():
    device, dtype = select_device_dtype()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, torch_dtype=dtype,
    ).to(device)
    model.eval()

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    print("\n=== INJECT ===")
    replaced = inject_lora_modules(model, r=8, alpha=16, dropout=0.05)
    if replaced == 0:
        raise RuntimeError("No target projection layers were found for LoRA injection.")
    print(f"[LoRA] Replaced linear layers: {replaced}")

    # Mark LoRA params as trainable
    print("\n=== TRAINABLE MARK ===")
    _ = mark_only_lora_as_trainable(model)

    # Parameter stats
    print("\n=== PARAM STATS ===")
    trainable, total, ratio = count_model_params(model)
    print(f"[Params] trainable={trainable:,}, total={total:,}, ratio={ratio:.4%}")

    print("\n=== FORWARD SMOKE TEST ===")
    sample = tokenizer("Hello Tokyo in autumn!", return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**sample)
    if not torch.isfinite(output.logits).all().item():
        raise RuntimeError("Forward pass produced non-finite logits.")
    print("Forward pass succeeded.")

    print("\n=== MERGE ===")
    merged = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()
            merged += 1
    print(f"[LoRA] merged modules: {merged}")


if __name__ == "__main__":
    torch.manual_seed(1234)
    main()
