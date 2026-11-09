"""Reproduce the editable program-search framework schematic.

This is a conceptual architecture diagram, not a trace of the recorded run.
Only Matplotlib is used: no candidate is executed and no service is contacted.
The original layout and default PNG resolution are retained.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]


def build_figure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
    ax.set(xlim=(0, 10), ylim=(0, 4.8))
    ax.axis("off")

    def box(x, y, w, h, label, fill="#EDF3F6"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1, edgecolor="#234D66", facecolor=fill))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                fontsize=9, color="#182B36")

    def arrow(a, b, rad=0):
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=11,
            color="#234D66", connectionstyle=f"arc3,rad={rad}"))

    box(.15, 3.8, 2.0, .7, "Task constraints\nInitial code + checks")
    box(3.05, 3.8, 3.3, .7, "Orchestrator\nBudget, parents, survivors")
    box(7.3, 3.8, 2.4, .7, "State + memory\nScores, code, failure traces")
    arrow((2.2, 4.15), (3.0, 4.15))
    arrow((6.4, 4.15), (7.25, 4.15))
    for x, label in [(0.4, "Planner\nPropose an idea"),
                     (3.7, "Coder\nImplement a candidate"),
                     (7.0, "Evaluator\nRun and check validity")]:
        box(x, 1.9, 2.5, .8, label)
    arrow((2.95, 2.3), (3.65, 2.3))
    arrow((6.25, 2.3), (6.95, 2.3))
    arrow((4.7, 3.75), (1.65, 2.75))
    arrow((8.25, 2.75), (5.85, 3.75))
    ax.text(7.0, 3.1, "measured feedback", fontsize=8, ha="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    box(2.4, .25, 4.3, .7,
        "Provider: calls the chosen LLM\nUsed for idea and code generation", "#F0F0F0")
    arrow((1.65, 1.85), (3.05, 1.0))
    arrow((4.95, 1.85), (5.55, 1.0))
    ax.text(8.2, .6, "Repeat until the\nsearch budget is reached",
            ha="center", fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "program_search" / "figures")
    parser.add_argument("--dpi", type=int, default=170,
                        help="PNG resolution; 170 reproduces the original export")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(args.output_dir / "framework.pdf")
    fig.savefig(args.output_dir / "framework.png", dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote framework.pdf and framework.png to {args.output_dir}")


if __name__ == "__main__":
    main()
