"""Plot the completed A10 PPO run from preserved TensorBoard scalars.

No model is loaded and no training, evaluation, smoothing or data editing occurs.
Run from any directory. PDF/PNG use the report's final 146 x 87 mm dimensions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results"
SOURCE = DATA / "cloud_metrics.csv"
SUMMARY = DATA / "cloud_results.json"
SIZE_MM = (146.0, 87.0)
BLUE, INK, MUTED, SUCCESS = "#0072B2", "#27323A", "#5C6670", "#27614A"


def read_metrics():
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 320, "Expected ten TensorBoard metrics at 32 batches"
    grouped = {}
    for row in rows:
        value = float(row["value"])
        step = int(row["step"])
        assert math.isfinite(value) and math.isfinite(float(row["wall_time"]))
        grouped.setdefault(row["tag"], []).append((step, value))
    for tag, points in grouped.items():
        points.sort()
        assert [step for step, _ in points] == list(range(1, 33)), tag
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["training_exit_code"] == 0 and summary["training_executed"] is True
    assert summary["gpu"] == "NVIDIA A10" and summary["precision"] == "bf16"
    assert summary["logged_rollout_steps"] == 32 and summary["adapter_all_finite"] is True
    assert summary["validation_n"] == 128
    assert summary["baseline_accuracy"] == 117 / 128
    assert summary["trained_accuracy"] == 122 / 128
    return grouped


def build_figure():
    grouped = read_metrics()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9.0, "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#7E878E", "axes.linewidth": .65,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=tuple(mm / 25.4 for mm in SIZE_MM))
    fig.subplots_adjust(left=.095, right=.975, bottom=.18, top=.80, wspace=.26, hspace=.52)
    specs = [
        ("train/reward", "a  Sampled training reward", (0., 1.08), [0, .25, .5, .75, 1.], "%.2f"),
        ("train/policy_loss", "b  Actor policy loss", (-.42, .42), [-.4, -.2, 0, .2, .4], "%.2f"),
        ("train/critic_loss", "c  Critic value loss", (0., .27), [0, .08, .16, .24], "%.2f"),
        ("train/kl", "d  Sampled KL estimate", (-.035, .60), [0, .2, .4, .6], "%.1f"),
    ]
    series = {}
    for ax, (tag, title, limits, ticks, formatter) in zip(axes.flat, specs):
        steps, values = zip(*grouped[tag])
        series[tag] = list(values)
        ax.plot(steps, values, color=BLUE, linewidth=1.05, marker="o", markersize=2.8,
                markerfacecolor="white", markeredgewidth=.65, clip_on=False)
        ax.plot(steps[-1], values[-1], marker="D", markersize=4., color=BLUE,
                markeredgewidth=.65, markeredgecolor="white", clip_on=False)
        ax.set(title=title, xlim=(.4, 33.0), ylim=limits,
               xticks=[1, 8, 16, 24, 32], yticks=ticks)
        ax.title.set_fontweight("bold")
        ax.title.set_horizontalalignment("left")
        ax.title.set_position((0, 1.0))
        ax.title.set_y(1.02)
        ax.yaxis.set_major_formatter(FormatStrFormatter(formatter))
        ax.grid(axis="y", color="#E3E7EA", linewidth=.55)
        ax.set_axisbelow(True)
        if tag in {"train/policy_loss", "train/kl"}:
            ax.axhline(0, color="#8C959D", linewidth=.7, linestyle=(0, (3, 3)), zorder=0)
        assert list(ax.lines[0].get_xdata()) == list(range(1, 33))
        assert list(ax.lines[0].get_ydata()) == list(values)
        assert min(values) > limits[0] and max(values) < limits[1]
    fig.text(.095, .955, "Completed PPO training diagnostics", fontsize=11.0,
             fontweight="bold", ha="left", va="top")
    fig.text(.095, .89, "32 rollout batches | TensorBoard scalars | one A10 BF16 run",
             fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.text(.535, .095, "Completed rollout batch", fontsize=9., ha="center")
    fig.text(.095, .025, "Final adapters saved; actor reloaded; paired evaluation completed.",
             fontsize=8.0, color=SUCCESS, ha="left", fontweight="bold")
    fig.canvas.draw()
    audit = {
        "source": SOURCE.relative_to(ROOT.parent).as_posix(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
        "source_rows": 320, "metrics": 10, "points_per_metric": 32,
        "steps": list(range(1, 33)), "plotted_series": series,
        "smoothing": False, "confidence_intervals": False,
        "metric_label_location": "Panel headings; all four quantities are dimensionless",
        "marker_note": "Circles show every recorded batch; diamond marks the final batch.",
        "kl_note": "The sampled KL estimator includes a small negative value, retained without clipping.",
        "experiment_status": "Full run exited 0; final Actor/Critic adapters saved and checked; Actor reloaded; paired evaluation completed",
        "not_claimed": "Critic reload, multi-run uncertainty, significance or general reasoning improvement",
        "figure_size_mm": SIZE_MM, "minimum_font_pt": 8.0,
        "data_scope": "New completed ModelScope A10 run; not the historical rounded T4 run",
    }
    return fig, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "figures")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--qa-scripts", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, audit = build_figure()
    if args.qa_scripts:
        sys.path.insert(0, str(args.qa_scripts))
        from visual_qa import audit_layout
        issues = audit_layout(fig)
        audit["scipilot_layout_issues"] = issues
        if any(severity == "FAIL" for severity, _ in issues):
            raise RuntimeError(f"Layout audit failed: {issues}")
    base = args.output_dir / "ppo_training_success"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    if not args.preview_only:
        fig.savefig(base.with_suffix(".pdf"), metadata={
            "Title": "Completed A10 PPO training diagnostics",
            "Subject": "32 raw TensorBoard batches; saved adapters and matched evaluation verified",
        })
    with Image.open(base.with_suffix(".png")) as im:
        im.convert("L").save(base.with_name(base.name + "_grayscale").with_suffix(".png"), dpi=(300, 300))
    base.with_name(base.name + "_plot_audit").with_suffix(".json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"points_per_metric": 32, "size_mm": SIZE_MM,
                      "layout_issues": audit.get("scipilot_layout_issues", [])}))


if __name__ == "__main__":
    main()
