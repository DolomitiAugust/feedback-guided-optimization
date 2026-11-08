"""Validate a real cloud run and collect compact, inspectable evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from ppo_reward import score_response

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "ppo_outputs")
    args = parser.parse_args()
    run = args.run_dir
    if (run / "exit_code.txt").read_text(encoding="ascii").strip() != "0":
        raise RuntimeError("The training process did not finish successfully.")
    from safetensors.torch import load_file
    import torch
    adapter_file = run / "actor_adapter" / "adapter_model.safetensors"
    state = load_file(str(adapter_file), device="cpu")
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise RuntimeError("Adapter contains non-finite weights.")
    b_weights = {key: value for key, value in state.items() if "lora_B" in key}
    changed = sum(torch.count_nonzero(value).item() > 0 for value in b_weights.values())
    if not b_weights or not changed:
        raise RuntimeError("No nonzero LoRA B weights; training update is not established.")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = list((run / "tensorboard").rglob("events.out.tfevents.*"))
    if not events:
        raise RuntimeError("No TensorBoard training events found.")
    scalars = []
    for event_file in events:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                scalars.append({"tag": tag, "step": event.step, "value": event.value,
                                "wall_time": event.wall_time})
    scalars.sort(key=lambda row: (row["tag"], row["step"], row["wall_time"]))
    if not scalars:
        raise RuntimeError("TensorBoard contains no scalar metrics.")
    if not all(math.isfinite(row["value"]) for row in scalars):
        raise RuntimeError("Training metrics contain a non-finite value.")
    tags = {row["tag"] for row in scalars}
    required_tags = {"train/reward", "train/policy_loss", "train/critic_loss"}
    if not required_tags <= tags:
        raise RuntimeError(f"Missing training metrics: {required_tags - tags}")
    reward_steps = {row["step"] for row in scalars if row["tag"] == "train/reward"}
    if reward_steps != set(range(1, 33)):
        raise RuntimeError(f"Expected all 32 rollout batches; observed {sorted(reward_steps)}")
    with (ROOT / "cloud_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["tag", "step", "value", "wall_time"])
        writer.writeheader()
        writer.writerows(scalars)
    before = json.loads((ROOT / "baseline_validation.json").read_text(encoding="utf-8"))
    after = json.loads((ROOT / "trained_validation.json").read_text(encoding="utf-8"))
    assert before["n"] == after["n"] == 128
    assert [(r["input"], r["label"]) for r in before["results"]] == [
        (r["input"], r["label"]) for r in after["results"]]
    assert before["decoding"] == after["decoding"]
    assert before["precision"] == after["precision"]
    assert before["metric"] == after["metric"] == "first_line_integer_accuracy_v2"
    for result in (before, after):
        assert len(result["results"]) == result["n"]
        assert all(row["score"] == score_response(row["response"], row["label"])
                   for row in result["results"])
        accuracy = sum(row["score"] for row in result["results"]) / result["n"]
        assert abs(accuracy - result["accuracy"]) < 1e-12
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    summary = {
        "training_executed": True, "training_exit_code": 0,
        "gpu": config["gpu"], "precision": config["precision"],
        "openrlhf_commit": config["openrlhf_commit"],
        "actor_adapter_sha256": hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
        "adapter_all_finite": True, "lora_B_tensor_count": len(b_weights),
        "nonzero_lora_B_tensors": changed,
        "lora_B_l2_norm": sum(v.float().square().sum().item() for v in b_weights.values()) ** 0.5,
        "logged_rollout_steps": max(row["step"] for row in scalars),
        "validation_n": 128, "baseline_accuracy": before["accuracy"],
        "trained_accuracy": after["accuracy"],
        "accuracy_difference": after["accuracy"] - before["accuracy"],
        "metric": "first_line_integer_accuracy_v2",
        "interpretation": "Measured first-line integer accuracy; a positive change is not assumed.",
        "metric_tags": sorted({row["tag"] for row in scalars}),
    }
    (ROOT / "cloud_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
