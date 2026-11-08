# Reproducing figures

Numerical figures read the recorded CSV, response pairs and circle geometry. They do not start model training. Circle centers and radii are never inferred from an image.

Run from the repository root in the analysis environment:

```sh
python policy_adaptation/plot_successful_ppo.py
python policy_adaptation/plot_paired_validation.py
python policy_adaptation/plot_ppo_diagnostics.py
python program_search/plot_publication_figures.py
python figure_tools/plot_research_mechanisms.py
```

The successful and historical PPO curves use separate sources. Online search scores and locally reproduced packing geometry use separate records. The feedback-loop and LoRA branch drawings are conceptual explanations, not measurements. All scripts are editable; outputs include PDF and PNG.
