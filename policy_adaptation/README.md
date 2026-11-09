# Low-rank policy adaptation

A minimal, tested LoRA implementation is paired with a pinned OpenRLHF PPO experiment on `Qwen/Qwen2.5-0.5B`. This is a small arithmetic-development experiment, not a benchmark of general reasoning.

[English](README.md) / [简体中文](README.zh-CN.md)

## What is measured

The A10/BF16 run completed 512 prompts in 32 rollout batches (16 prompts each). On the same 128 held-out development questions with greedy decoding, first-line integer accuracy changed from 117/128 (91.40625%) to 122/128 (95.3125%), five corrected answers and no newly wrong answers. The exact split, responses and scalar values are committed under `data/` and `results/`.

Actor and Critic adapters were saved. The Actor was actually reloaded on GPU and passed finite forward/generation checks; the Critic was checked for finite and nonzero weights but was not reloaded. Full optimizer/resume checkpoints are absent. A historical T4/FP32 run completed its 32-batch loop but failed final model serialization (exit 1); its rounded diagnostics remain in `logs/observed_attempt/` and are not used as the A10 baseline.

The Critic safetensors contain `score.lora_A.weight` and `score.lora_B.weight`, but not the underlying randomly initialized value-head weight. The exported Critic adapter therefore cannot independently restore the complete trained value function. The original files are preserved as recorded, not retroactively completed.

## CPU behavior checks

Use an isolated CPU environment. These dependencies differ from the pinned PPO environment:

```bash
python -m pip install -r policy_adaptation/requirements_lora.txt
python tests/test_policy.py
```

Nine tests cover LoRA numerical equivalence and merge, real adapter updates with the base frozen, projection injection, edge cases, deterministic disjoint data splits, response-only scoring and the local reward HTTP contract. Tests do not download Qwen or run PPO.

## GPU reproduction

The measured configuration is recorded in `results/ppo_outputs/run_config.json`; the pinned OpenRLHF commit is `bfa85f6fa8cd6214707f93883fac5d08334f4f06`. Use Linux, Python 3.10 and a compatible NVIDIA CUDA toolchain. The historical A10 used CUDA toolkit 12.1 with PyTorch 2.5.1+cu124; the installation performed an actual CUDA optimizer check. A compatible toolkit is required; the installer is a packaged recipe, not a new clean-machine experiment.

```bash
cd policy_adaptation
python3.10 -m venv .venv-eager
source .venv-eager/bin/activate
bash setup_eager_ppo.sh
python prepare_cloud_model.py  # stages the exact revision from the report
python prepare_ppo_data.py
python train_ppo_lora.py --pretrain .cloud_model --precision bf16  # configuration preview only
python evaluate_ppo.py --pretrain .cloud_model --output baseline_validation.json
python train_ppo_lora.py --run --pretrain .cloud_model --precision bf16 --max-samples 16 --output new_smoke
python check_saved_adapters.py --run-dir new_smoke --pretrain .cloud_model
python train_ppo_lora.py --run --pretrain .cloud_model --precision bf16 --output new_full_run
python check_saved_adapters.py --run-dir new_full_run --pretrain .cloud_model
python evaluate_ppo.py --pretrain .cloud_model --adapter new_full_run/actor_adapter --output trained_validation.json
python collect_ppo_results.py --run-dir new_full_run --baseline baseline_validation.json --trained trained_validation.json --output-dir ../local_outputs/collected
```

Run each command separately and stop on error. Use fresh output names. The 16-prompt check is independent of the fresh full experiment. For a paired BF16 baseline, use a BF16-capable GPU and the same base model/precision as the adapted evaluation. The evaluator's saved `precision` field is authoritative. The setup does not start training.

The collector checks model references, the adapted evaluation's run path, precision, decoding and the ordered question/answer pairs before writing results; it rejects existing collection files. Historical evaluation records contain no evaluation-time weight digest, so matching references and paths does not by itself prove which bytes a past evaluation loaded. The published Actor hash and reload evidence are separate records. The public copy omits raw TensorBoard event files, so its committed CSV is the retained scalar evidence; collection from events is for a newly completed run.

The staging helper pins model revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. All model-consuming commands above use that staged directory instead of a mutable remote default. Check that `cloud_model.json` reports `precision: bf16` before reproducing the A10 comparison; an FP32 run is a different experiment.

The eager route delays three otherwise-unused FlashAttention imports in exact upstream source files; packing and ring attention remain disabled. It preserves the explicit package-metadata exception: OpenRLHF declares `flash-attn`, which is absent. Actual forward/backward and FusedAdam updates passed. `setup_ppo.sh` is an optional Flash-enabled installation route, not a separately measured result.

## Replot saved data

```bash
python policy_adaptation/plot_successful_ppo.py
python policy_adaptation/plot_paired_validation.py
python policy_adaptation/plot_ppo_diagnostics.py
```

These scripts read saved numerical records and write figures to the repository's `figures/`. No training or inference is performed. Public adapter configuration paths are changed to the base model ID for portability; safetensors bytes and hashes are unchanged. Base model weights/tokenizers are loaded from the original model repository and are not duplicated here.

## Limitations

One seed, a small arithmetic distribution and development-validation scoring are insufficient to establish general reasoning gains or statistical robustness. The scoring rule was refined after inspecting responses; there is no untouched final test set. This repository does not claim an ablation, complete PPO state recovery or improved generation quality beyond the documented first-line metric.
