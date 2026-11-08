# Evaluated program search for circle packing

[English](README.md) / [简体中文](README.zh-CN.md)

The task is to place exactly 26 non-overlapping circles inside the unit square and maximize the sum of radii. The selected LLM-generated program uses deterministic starting layouts, SLSQP joint refinement and linear-programming radius allocation. It is not claimed to be globally optimal or a novel solver.

## Evidence

Five of seven saved online proposals completed valid evaluations. The best online score was **2.62001380699489**, candidate `4705b8e92e1b`, already found in initialization (generation 0). The observed generation-1 children did not exceed it. The run was manually stopped; convergence or a completed full search budget is not claimed.

An independent local re-execution of that same source produced **2.6206092130170973** on Python 3.14.2, NumPy 2.4.4 and SciPy 1.17.1. This is a different result record and does not replace the online score. The local initial constructor scored 0.9597642169962063; it is not a paired online baseline.

## Run and check

```bash
python -m pip install -r program_search/requirements.txt
python program_search/solve.py
python tests/test_search.py
```

The numerical pins match the historical local environment (Python 3.14.2); compatible Python versions with available wheels are required. The solver is a small CPU optimization, makes no model/API calls and prints circles as JSON. A weaker fallback exists when SciPy is missing; install SciPy for the intended method. Numeric results may differ by runtime.

The tests inspect the **saved** circles and event data without rerunning search. They check all boundaries and circle pairs, exact recorded sums, and separation of online and local scores.

```bash
python program_search/plot_publication_figures.py
```

Plotting reads committed data only and writes vector PDFs and PNGs under `figures/`. Saved centers and radii are used verbatim with equal axis scale.

## Files

- `solve.py`: selected exported candidate; its inherited initial-candidate docstring is retained.
- `config.yaml`: portable copy of the historical platform configuration; provider credentials are blank. It requires a compatible LLM4AD Next service and is not a standalone runner.
- `upstream/`: official initial constructor, original evaluator and BSD 3-Clause license with source commit recorded in `PROVENANCE.md`.
- `evidence/online_evaluations.csv`: the five valid online events, with the original full-precision scores.
- `evidence/best_local_packing.json`: saved geometry of the independent local best-program execution.
- `evidence/local_packing.json`: saved geometry of the local initial constructor.

The online service did not export original geometry for the winning score. The displayed best-program geometry therefore belongs to the separate local re-execution. Source attribution is preserved; upstream code is not presented as original research.
