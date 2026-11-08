# Reproducibility and check boundaries

## Saved-record checks

```sh
python -m unittest discover -s tests -p "test_records.py" -v
```

This standard-library-only command independently grades all paired responses, checks 320 finite scalars across ten metrics and 32 batches, distinguishes the historical failure, validates the saved local circle geometry and verifies the Actor weight hash. It does not contact a service or load a model.

## CPU geometry and plotting

The historical local geometry record used Python 3.14.2, NumPy 2.4.4 and SciPy 1.17.1. Create a separate analysis environment:

```sh
python -m pip install -r program_search/requirements.txt
python -m pip install -r figure_tools/requirements.txt
python program_search/solve.py
python program_search/plot_publication_figures.py
python policy_adaptation/plot_successful_ppo.py
python policy_adaptation/plot_paired_validation.py
python figure_tools/plot_research_mechanisms.py
```

Small numerical differences across platforms and dependency versions are possible. Newly executed geometry is a new local record; do not overwrite the preserved online score.

## Manual LoRA checks

See [the policy module](../policy_adaptation/README.md). Its lightweight behavior tests check layer equivalence, injection/merging, frozen base weights, a real optimizer update and the reward HTTP interface. They do not demonstrate full PPO training.

## Full PPO reproduction

The measured run used Python 3.10, NVIDIA A10, PyTorch 2.5.1+cu124, Transformers 4.48.3, PEFT 0.14.0, DeepSpeed 0.16.3, Ray 2.42.0 and NumPy 1.26.4. OpenRLHF is pinned to `bfa85f6fa8cd6214707f93883fac5d08334f4f06`. Use a separate Linux GPU environment; do not merge the geometry NumPy version into it.

See the policy README for setup, dry-run, baseline, independent save/reload preflight, full training, adapter reload and paired evaluation. The eager setup retains one explicit metadata exception: upstream declares FlashAttention, while the restricted non-packing, non-ring path does not install it. Actual historical CUDA forward/backward and optimizer checks passed. The freshly packaged clean-machine installer itself has not been rerun on a clean GPU machine.

Full training is not part of the release smoke checks. Free cloud GPU availability is outside this repository's control. No complete optimizer/resume state is included.

## Rebuild reports

Install Tectonic or a XeLaTeX distribution with `fontspec`, `xeCJK`, and the fonts named in the source. Compile from `reports/source`:

```sh
tectonic -X compile --outdir .. report-en.tex
tectonic -X compile --outdir .. report-zh.tex
```

The reports reference shared repository figure files. The supplied PDF files can be read without installing LaTeX. Alternative font selections may change pagination.
