"""Minimal LoRA layers and parameter-efficient model adaptation.

A frozen linear weight W is augmented by (alpha / r) BA. The implementation
supports nested projection injection, trainable-parameter counting and a
numerically checked merge operation. The executable example loads Qwen2.5-0.5B;
importing the module does not download a model or start training.
"""

# pip install torch transformers accelerate datasets
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        out = x @ (W + 螖W)^T (+ bias), bypass disabled.

    implementations you must complete:
      - __init__: store shapes, create lora_A / lora_B, scaling, dropout, and
                  keep references to base weight/bias. Track merged-state flag.
      - reset_parameters: initialize A with Kaiming uniform, B to zeros.
      - merge: add 螖W back to base weight in-place and disable bypass thereafter.
      - forward: implement both the base forward and the LoRA bypass (when not merged).
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        # implementation: Validate base_linear is nn.Linear; set in/out features
        # implementation: Save r, alpha and compute scaling = alpha / r
        # implementation: Keep references to base_linear.weight and base_linear.bias (DO NOT deep copy)
        # implementation: Create lora_A (in_features -> r, bias=False) and lora_B (r -> out_features, bias=False)
        # implementation: Create dropout module (nn.Dropout if dropout>0 else nn.Identity)
        # implementation: Track merged-state flag (e.g., self.merged = False)
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
        """
        implementation:
        - Initialize lora_A with Kaiming uniform (nn.init.kaiming_uniform_ with a=sqrt(5))
        - Initialize lora_B to all zeros so the initial 螖W is zero
        """
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    @torch.no_grad()
    def merge(self):
        """
        implementation:
        - If already merged, return immediately.
        - Compute 螖W = (lora_B.weight @ lora_A.weight) * scaling
          Shapes note (nn.Linear uses [out, in]):
             lora_A.weight: [r, in]
             lora_B.weight: [out, r]
             resulting 螖W:  [out, in]
        - Add 螖W in-place to the base weight.
        - (Optional but recommended) zero out lora_A/B weights after merging.
        - Set merged flag so forward() stops using the bypass.
        """
        if self.merged:
            return
        if self.r > 0:
            delta_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.weight.add_(delta_weight)
        self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        implementation:
        - Compute base output: F.linear(x, base_weight, base_bias)
        - If NOT merged and r>0:
            * apply dropout to x
            * compute LoRA bypass: lora_B(lora_A(x)) * scaling
            * add to the base output
        - Return the final tensor
        """
        out = F.linear(x, self.weight, self.bias)
        if not self.merged and self.r > 0:
            out = out + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return out


# ----------------------------------------
# 2) Inject LoRA: replace target Linear layers with LoRALinear (fill the implementation)
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

    implementation:
    - Iterate over named_modules().
    - If a module name endswith any keyword in TARGET_LINEAR_KEYWORDS and the module is nn.Linear:
        * Get the parent module via model.get_submodule(parent_name)
        * Create a LoRALinear(module, r, alpha, dropout) and move it to the same device/dtype as the base weight
        * setattr(parent, child_attr_name, lora_module)
        * Count how many replacements were made
    - Return the number of replaced modules.
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
# 3) Model parameter statistics (fill the implementation)
# -------------------------------------------------------
def count_model_params(model: nn.Module) -> Tuple[int, int, float]:
    """
    Return (trainable_params, total_params, ratio)

    implementation:
    - total = sum of numel() for all parameters
    - trainable = sum of numel() for parameters with requires_grad == True
    - ratio = trainable / total if total > 0 else 0.0
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total > 0 else 0.0
    return trainable, total, ratio


# -------------------------------------------------------
# Helper: mark only LoRA params as trainable (already implemented for you)
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
# 4) A minimal smoke test (do not modify this section)
# -------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"  # base (non-instruct) for causal LM
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Try LoRA injection
    print("\n=== INJECT ===")
    try:
        replaced = inject_lora_modules(model, r=8, alpha=16, dropout=0.05)
        print(f"[LoRA] Replaced linear layers: {replaced}")
    except NotImplementedError as e:
        print(f"NotImplementedError in inject_lora_modules: {e}")

    # Mark LoRA params as trainable
    print("\n=== TRAINABLE MARK ===")
    _ = mark_only_lora_as_trainable(model)

    # Parameter stats
    print("\n=== PARAM STATS ===")
    try:
        trainable, total, ratio = count_model_params(model)
        print(f"[Params] trainable={trainable:,}, total={total:,}, ratio={ratio:.4%}")
    except NotImplementedError as e:
        print(f"NotImplementedError in count_model_params: {e}")

    # Forward smoke test (will raise until you implement LoRALinear)
    print("\n=== FORWARD SMOKE TEST ===")
    sample = tokenizer("Hello Tokyo in autumn!", return_tensors="pt").to(device)
    try:
        with torch.no_grad():
            _ = model(**sample)
        print("Forward pass succeeded.")
    except Exception as e:
        print(f"Forward encountered an exception (expected until you finish implementations): {e}")

    # Attempt to merge (will raise until you implement merge)
    print("\n=== MERGE ===")
    try:
        merged = 0
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.merge()
                merged += 1
        print(f"[LoRA] merged modules: {merged}")
    except NotImplementedError as e:
        print(f"NotImplementedError in LoRALinear.merge: {e}")


if __name__ == "__main__":
    torch.manual_seed(1234)
    main()
