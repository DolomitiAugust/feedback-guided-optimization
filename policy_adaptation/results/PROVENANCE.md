# Measurement and portability notes

These records describe the successful A10/BF16 experiment on 8 November 2026. A separately recorded invocation nine seconds after completion was refused by the non-overwrite guard; its failure does not negate the completed run. The successful completion and later refusal are preserved separately under `logs/`.

Public copies normalize text line endings and replace local workspace/model paths with portable references. Numeric responses, scalar values, sample order and configuration hyperparameters are unchanged. The two safetensors files preserve their exact original bytes. `adapter_config.json` changes only `base_model_name_or_path` to `Qwen/Qwen2.5-0.5B`; load tokenizers and base-model weights from that original public model.

Raw cloud export archives, notebooks, machine-specific installation logs and original local audit files remain in the private project archive. Hashes that identify such historical artifacts are provenance digests, not claims that those private files are included in this public repository. This public package retains the measured results and relevant implementation, not every operational artifact.
