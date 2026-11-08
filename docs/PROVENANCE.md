# Selection and provenance

The release is a new, curated copy. Private originals and historical exports are retained outside this repository. Previous commit history, cloud access links, screenshots, installation transcripts and obsolete recovery notebooks are not imported.

## Policy experiment

Core implementation and the successful A10 experiment were compared against their recorded export. The selected implementation incorporates the sampling padding and PEFT save repairs needed for the measured run. The restricted eager route delays unused FlashAttention imports on a pinned upstream revision; it is not a general substitute for FlashAttention in packing or ring-attention workflows.

The release retains the 32-batch scalar series, the exact arithmetic prompts and responses, configuration, completion and reload evidence, and final Actor/Critic adapter weights. Weight tensors retain their original bytes. Private model paths in adapter JSON are replaced with `Qwen/Qwen2.5-0.5B`; other machine paths in published records become portable placeholders. These are editorial changes to metadata, not new experiments.

The earlier T4 FP32 series remains a separate failed-save observation. Its rounded console metrics are not silently upgraded to full-precision TensorBoard measurements. The original export later recorded a refused duplicate invocation after a successful A10 run; the publication distinguishes that refusal from the earlier completed experiment.

## Program-search experiment

The selected source is the observed online winner, candidate `4705b8e92e1b`. Five valid evaluation records are retained alongside the stated seven-proposal total and the failed/pending distinction. The local initial-constructor geometry and the locally re-executed selected program remain separate from server scores. Saved numeric arrays are preserved; figures read them directly with equal aspect ratio.

The upstream initial constructor and evaluator are attributed to LLM4AD_Next commit `dfde353e753e9021fffe88b5b034bf0c84d90b60`, with its BSD notice. The selected program is attributed to the recorded LLM-assisted search. Its measured local score does not replace its online score.

## Publication transformations

- Select the complete, independently checked experiment rather than simply choosing files by modification time.
- Use descriptive module names and research sections; omit educational submission mappings and obsolete alternatives.
- Regenerate the bilingual PDFs from identity-free LaTeX. Personal identifiers are removed from content and metadata, not covered with overlays.
- Exclude account screenshots, notebooks with outputs, raw platform archives, credentials, user-specific paths and private source-map files.
- Retain shared editable code and figure sources; use separate environments for GPU policy learning and CPU geometry/analysis.

The release includes no complete base language-model weights or optimizer-resume checkpoint, and no new full training run was performed while preparing it.
