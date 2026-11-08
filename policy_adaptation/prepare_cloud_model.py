"""Download and stage the verified Qwen checkpoint on a Linux CUDA runtime."""
import json
import os
from pathlib import Path
import sys

MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
ROOT = Path(__file__).resolve().parent


def main():
    import torch
    if sys.platform != "linux" or not torch.cuda.is_available():
        raise SystemExit("Run this download step in a Linux cloud GPU runtime, not on the local PC.")
    from huggingface_hub import snapshot_download
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    precision = "bf16" if native_bf16 else "fp32"
    snapshot = Path(snapshot_download(
        MODEL, revision=REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    ))
    target = Path(os.environ.get("research_MODEL_DIR", str(ROOT / ".cloud_model")))
    target.mkdir(parents=True, exist_ok=True)
    # Reuse cached weight bytes through symlinks; do not duplicate a 1 GB checkpoint.
    for source in snapshot.iterdir():
        if not source.is_file() or source.name == "config.json":
            continue
        destination = target / source.name
        if destination.exists():
            if destination.resolve() != source.resolve():
                raise FileExistsError(f"Different existing model file: {destination}")
        else:
            destination.symlink_to(source.resolve())
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    # OpenRLHF uses torch_dtype='auto' when --bf16 is absent. Explicitly set
    # the config to FP32 for T4/P100, rather than inheriting Qwen's BF16 default.
    config["torch_dtype"] = "bfloat16" if precision == "bf16" else "float32"
    (target / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    record = {
        "model": MODEL, "revision": REVISION, "pretrain": str(target),
        "precision": precision, "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda, "torch": torch.__version__,
        "native_bf16_supported": native_bf16,
        "bf16_including_emulation": torch.cuda.is_bf16_supported(),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "model_weights_modified": False,
        "config_change": {"torch_dtype": config["torch_dtype"]},
    }
    (ROOT / "cloud_model.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
