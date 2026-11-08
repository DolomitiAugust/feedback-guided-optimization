"""Evaluate the base model or a trained actor adapter on the held-out split."""
import argparse
import json
from pathlib import Path

from ppo_reward import score_response

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, help="Saved actor adapter; omit for baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrain", default="Qwen/Qwen2.5-0.5B")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; select another file to preserve earlier results")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = args.pretrain
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if device == "cuda" and
             torch.cuda.is_bf16_supported(including_emulation=False) else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter)).to(device)
    model.eval()
    rows = [json.loads(line) for line in (ROOT / "data" / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
    results = []
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            inputs = tokenizer(row["input"], return_tensors="pt", add_special_tokens=False).to(device)
            generated = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            score = score_response(response, row["label"])
            results.append({**row, "response": response, "score": score})
            print(f"{index}/{len(rows)} correct={sum(r['score'] for r in results):.0f}", flush=True)
    payload = {"model": model_name, "adapter": str(args.adapter) if args.adapter else None,
               "precision": str(dtype), "metric": "first_line_integer_accuracy_v2",
               "decoding": "greedy, max_new_tokens=32", "n": len(results),
               "accuracy": sum(r["score"] for r in results) / len(results), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Held-out accuracy: {payload['accuracy']:.2%}")


if __name__ == "__main__":
    main()
