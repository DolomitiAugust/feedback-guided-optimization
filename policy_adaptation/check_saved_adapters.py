"""Check saved Actor/Critic adapters and reload the Actor on the cloud GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pretrain", required=True)
    args = parser.parse_args()
    import torch
    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("Run this adapter reload check on the Linux CUDA GPU.")
    assert (args.run_dir / "exit_code.txt").read_text().strip() == "0"
    from safetensors.torch import load_file
    checks = {}
    for role, folder in (("actor", "actor_adapter"), ("critic", "actor_adapter_critic")):
        path = args.run_dir / folder / "adapter_model.safetensors"
        state = load_file(str(path), device="cpu")
        assert all(torch.isfinite(value).all().item() for value in state.values())
        bs = [value for name, value in state.items() if "lora_B" in name]
        nonzero = sum(torch.count_nonzero(value).item() > 0 for value in bs)
        assert bs and nonzero
        checks[role] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "all_finite": True, "b_tensors": len(bs), "nonzero_b_tensors": nonzero}
        del state, bs
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False)
             else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.pretrain, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, str(args.run_dir / "actor_adapter")).to("cuda")
    model.eval()
    from prepare_ppo_data import make_rows
    prompt = make_rows()[0][0]["input"]
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        assert torch.isfinite(model(**inputs).logits).all().item()
        output = model.generate(**inputs, do_sample=False, max_new_tokens=8,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    checks["actor_reload_forward_finite"] = True
    checks["reloaded_response"] = tokenizer.decode(output[0, inputs.input_ids.shape[1]:],
                                                   skip_special_tokens=True)
    (args.run_dir / "adapter_reload_check.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
