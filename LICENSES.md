# Attribution and reuse

This repository does not apply a blanket license to all materials. Project-specific code and reports are publicly inspectable; no additional reuse grant is inferred from public hosting.

- The original circle-packing constructor and evaluator are from `Optima-CityU/LLM4AD_Next`, commit `dfde353e753e9021fffe88b5b034bf0c84d90b60`. Their BSD 3-Clause notice is retained in `program_search/upstream/LICENSE`.
- The selected packing constructor was generated during the documented LLM-assisted online search. Its source and historical numerical records are retained; no new algorithmic originality is claimed.
- The adapter weights require the separately obtained `Qwen/Qwen2.5-0.5B` base model. Its upstream model card specifies Apache 2.0 terms. The base weights are not redistributed. Refer to the [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B) and its license before downstream reuse.
- The PPO integration targets OpenRLHF commit `bfa85f6fa8cd6214707f93883fac5d08334f4f06`. The repository provides compatibility patching code and links to upstream rather than redistributing a full checkout. Upstream software retains its own terms.
- References, package names and method names identify their respective authors and projects. Reports distinguish implementation work, recorded measurements, and borrowed methods.
