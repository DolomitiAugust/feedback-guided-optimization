"""Launch actual LoRA-PPO through the pinned OpenRLHF v0.6.0 entry point.

Without --run, print and validate the configuration only. Linux + a compatible
NVIDIA GPU and CUDA toolchain are required for --run. This is PPO with a critic
and GAE, not GRPO or REINFORCE. See README.md for installation and evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from ppo_reward_server import LocalRewardServer

ROOT = Path(__file__).resolve().parent
OPENRLHF_COMMIT = "bfa85f6fa8cd6214707f93883fac5d08334f4f06"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj", "score"]


def normalize_pretrain(reference):
    """Keep local paths valid when DeepSpeed changes its working directory."""
    path = Path(reference)
    return str(path.resolve()) if path.exists() else reference


def build_training_args(data: Path, output: Path, max_samples=512,
                        reward_url="http://127.0.0.1:8001/reward",
                        pretrain="Qwen/Qwen2.5-0.5B", bf16=True):
    if max_samples <= 0 or max_samples > 512 or max_samples % 16:
        raise ValueError("max_samples must be a multiple of 16 between 16 and 512")
    return [
        "--pretrain", str(pretrain),
        "--critic_pretrain", str(pretrain),
        "--remote_rm_url", reward_url,
        "--prompt_data", str(data), "--input_key", "input", "--label_key", "label",
        "--save_path", str(output / "actor_adapter"),
        "--ckpt_path", str(output / "checkpoints"),
        "--save_value_network", "--disable_ds_ckpt",
        "--save_steps", "8", "--save_hf_ckpt", "--logging_steps", "1", "--eval_steps", "-1",
        "--num_episodes", "1", "--max_epochs", "1",
        "--max_samples", str(max_samples),
        "--rollout_batch_size", "16", "--micro_rollout_batch_size", "1",
        "--train_batch_size", "8", "--micro_train_batch_size", "1",
        "--prompt_max_len", "192", "--generate_max_len", "32",
        "--actor_learning_rate", "1e-5", "--critic_learning_rate", "5e-5",
        "--lr_warmup_ratio", "0.03", "--advantage_estimator", "gae",
        "--eps_clip", "0.2", "--value_clip", "0.2",
        "--gamma", "1.0", "--lambd", "0.95", "--init_kl_coef", "0.01",
        "--lora_rank", "8", "--lora_alpha", "16", "--lora_dropout", "0.0",
        "--target_modules", *TARGETS,
        "--zero_stage", "2", *(["--bf16"] if bf16 else []), "--gradient_checkpointing",
        "--seed", "1234", "--use_tensorboard", str(output / "tensorboard"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute real GPU training")
    parser.add_argument("--output", type=Path, default=ROOT / "ppo_outputs")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--pretrain", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16",
                        help="Use fp32 with a staged FP32 config on free T4/P100 GPUs")
    args = parser.parse_args()
    args.pretrain = normalize_pretrain(args.pretrain)
    data = ROOT / "data" / "train.jsonl"
    if not data.is_file():
        parser.error("Run prepare_ppo_data.py first")
    training_args = build_training_args(data.resolve(), args.output.resolve(), args.max_samples,
                                        pretrain=args.pretrain, bf16=args.precision == "bf16")
    command = ["deepspeed", "--num_gpus", "1", "--module", "openrlhf.cli.train_ppo", *training_args]
    print("OpenRLHF commit:", OPENRLHF_COMMIT)
    print("Mode:", "REAL GPU TRAINING" if args.run else "DRY RUN; NO TRAINING")
    print(shlex.join(command), flush=True)
    if not args.run:
        return
    if sys.platform != "linux":
        parser.error("This pinned OpenRLHF setup requires Linux; use a Linux GPU machine")
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; no PPO training was started")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        parser.error("Use --precision fp32 and the model staged by prepare_cloud_model.py on this GPU")
    if args.precision == "fp32":
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.pretrain)
        if config.torch_dtype != torch.float32:
            parser.error("FP32 requires a staged float32 model config; run prepare_cloud_model.py")
    if shutil.which("deepspeed") is None:
        parser.error("DeepSpeed is missing; follow setup_ppo.sh")
    from importlib.metadata import distribution
    installed = distribution("openrlhf")
    origin = json.loads(installed.read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != OPENRLHF_COMMIT:
        parser.error("Install the exact OpenRLHF Git commit from requirements_ppo.txt")
    from patch_openrlhf import apply_compatibility_patch
    patch_record = apply_compatibility_patch()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new --output path")
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NO_PROXY"] = env.get("NO_PROXY", "") + ",127.0.0.1,localhost"
    env["no_proxy"] = env["NO_PROXY"]
    # Use one GPU. DeepSpeed applies gradient accumulation to reach batch size 8.
    with LocalRewardServer() as reward_server, (args.output / "train.log").open("w", encoding="utf-8") as log:
        command[command.index("--remote_rm_url") + 1] = reward_server.url
        (args.output / "run_config.json").write_text(json.dumps({
            "openrlhf_commit": OPENRLHF_COMMIT, "argv": command,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "precision": args.precision, "pretrain": args.pretrain,
            "compatibility_patch": patch_record,
            "status": "started; consult train.log and exit code for completion",
        }, indent=2), encoding="utf-8")
        print("Reward API:", reward_server.url, flush=True)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    (args.output / "exit_code.txt").write_text(str(returncode) + "\n", encoding="ascii")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
