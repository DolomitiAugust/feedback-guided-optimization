"""Validate a real cloud run and collect compact, inspectable evidence."""
import argparse
import csv
import hashlib
import io
import json
import math
from contextlib import ExitStack
from pathlib import Path
from ppo_reward import score_response

ROOT = Path(__file__).resolve().parent
METRIC = "first_line_integer_accuracy_v2"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _model_reference(value):
    """Canonicalize an existing local model; keep remote repository IDs intact."""
    _require(isinstance(value, str) and bool(value), "Missing model reference.")
    path = Path(value)
    return str(path.resolve()) if path.exists() else value


def validate_evaluations(before, after, config, run, trained_path):
    """Check legacy evaluation records without claiming a missing weight digest.

    Legacy relative adapter paths are interpreted against either the evaluation
    file's directory or the current working directory. Neither basename matching
    nor a newly computed digest proves which weights a past evaluation loaded.
    """
    model = _model_reference(before.get("model"))
    _require(model == _model_reference(after.get("model")),
             "Baseline and trained evaluations use different models.")
    _require(model == _model_reference(config.get("pretrain")),
             "Evaluation model does not match the training run.")
    _require("adapter" in before and before["adapter"] is None,
             "Baseline must be evaluated without an adapter.")
    adapter = after.get("adapter")
    _require(isinstance(adapter, str) and bool(adapter), "Trained evaluation has no adapter path.")
    adapter = Path(adapter)
    expected = (Path(run) / "actor_adapter").resolve()
    candidates = {adapter.resolve()}
    if not adapter.is_absolute():
        candidates.add((Path(trained_path).resolve().parent / adapter).resolve())
    _require(expected in candidates, "Trained evaluation adapter does not belong to this run.")
    _require(before.get("n") == after.get("n") == 128, "Expected 128 paired validation rows.")
    _require(before.get("decoding") and before["decoding"] == after.get("decoding"),
             "Evaluation decoding settings differ or are missing.")
    _require(before.get("precision") == after.get("precision"), "Evaluation precisions differ.")
    precision = {"bf16": "torch.bfloat16", "fp32": "torch.float32"}.get(config.get("precision"))
    _require(precision is not None and before.get("precision") == precision,
             "Evaluation precision does not match the training run.")
    _require(before.get("metric") == after.get("metric") == METRIC, "Unexpected evaluation metric.")
    for name, result in (("baseline", before), ("trained", after)):
        rows = result.get("results", [])
        _require(len(rows) == result["n"], f"Incorrect {name} row count.")
        _require(all(row["score"] == score_response(row["response"], row["label"]) for row in rows),
                 f"Recorded {name} scores disagree with the responses.")
        accuracy = sum(row["score"] for row in rows) / result["n"]
        _require(abs(accuracy - result["accuracy"]) < 1e-12, f"Incorrect {name} accuracy.")
    _require([(r["input"], r["label"]) for r in before["results"]] ==
             [(r["input"], r["label"]) for r in after["results"]],
             "Baseline and trained evaluations do not contain the same ordered questions.")


def write_results(output_dir, scalars, summary):
    """Publish an already validated pair of files; never replace existing evidence."""
    output_dir = Path(output_dir)
    targets = (output_dir / "cloud_metrics.csv", output_dir / "cloud_results.json")
    if any(path.exists() for path in targets):
        raise FileExistsError("Collected output already exists; choose another --output-dir.")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["tag", "step", "value", "wall_time"])
    writer.writeheader()
    writer.writerows(scalars)
    contents = (buffer.getvalue(), json.dumps(summary, indent=2, allow_nan=False) + "\n")
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with ExitStack() as stack:
            handles = []
            for path in targets:
                handles.append(stack.enter_context(path.open("x", newline="", encoding="utf-8")))
                created.append(path)
            for handle, content in zip(handles, contents):
                handle.write(content)
    except BaseException:
        # Remove only files this invocation created, retaining existing outputs.
        for path in created:
            path.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "ppo_outputs")
    parser.add_argument("--baseline", type=Path, default=ROOT / "baseline_validation.json")
    parser.add_argument("--trained", type=Path, default=ROOT / "trained_validation.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="Destination for cloud_metrics.csv and cloud_results.json; neither may exist")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    before = json.loads(args.baseline.read_text(encoding="utf-8"))
    after = json.loads(args.trained.read_text(encoding="utf-8"))
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    validate_evaluations(before, after, config, run, args.trained)
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
        "metric": METRIC,
        "evaluation_binding": "model reference and adapter path checked; evaluation files contain no weight digest",
        "interpretation": "Measured first-line integer accuracy; a positive change is not assumed.",
        "metric_tags": sorted({row["tag"] for row in scalars}),
    }
    write_results(args.output_dir, scalars, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
