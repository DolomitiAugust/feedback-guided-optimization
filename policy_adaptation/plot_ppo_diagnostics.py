"""Plot preserved PPO console metrics without running training or editing evidence.

Run from any directory. The default output is the report's package-root figures/.
Optional --qa-scripts points to scipilot-figure-skill/scripts for its layout audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "logs" / "observed_attempt" / "rounded_console_metrics.csv"
SUMMARY = ROOT / "logs" / "observed_attempt" / "summary.json"
SIZE_MM = (146.0, 87.0)
BLUE = "#0072B2"
INK = "#27323A"
MUTED = "#5C6670"
FAILURE = "#875A00"


def read_metrics():
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8", newline="")))
    steps = [int(row["completed_rollout_batch"]) for row in rows]
    assert steps == list(range(1, 33)), "Expected 32 preserved rollout batches"
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["training_exit_code"] == 1
    assert summary["adapter_export_verified"] is False
    assert summary["post_training_evaluation_available"] is False
    assert float(rows[-1]["reward"]) == 0.938
    assert rows[-1]["elapsed_console"] == "15:40"
    return rows, steps


def build_figure():
    rows, steps = read_metrics()
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
        ("reward", "a  Sampled training reward", (0., 1.08), [0, .25, .5, .75, 1.], "%.2f"),
        ("policy_loss", "b  Actor policy loss", (-.62, .48), [-.5, -.25, 0, .25], "%.2f"),
        ("critic_loss", "c  Critic value loss", (0., .125), [0, .04, .08, .12], "%.2f"),
        ("kl", "d  KL estimate to reference", (0., 1.30), [0, .4, .8, 1.2], "%.1f"),
    ]
    series = {}
    for ax, (field, title, limits, ticks, formatter) in zip(axes.flat, specs):
        values = [float(row[field]) for row in rows]
        series[field] = values
        ax.plot(steps, values, color=BLUE, linewidth=1.05, marker="o", markersize=2.8,
                markerfacecolor="white", markeredgewidth=.65, clip_on=False)
        ax.plot(steps[-1], values[-1], marker="D", markersize=4., color=BLUE,
                markeredgewidth=.65, markeredgecolor="white", clip_on=False)
        ax.set(title=title, xlim=(.4, 33.0), ylim=limits, xticks=[1, 8, 16, 24, 32], yticks=ticks)
        ax.title.set_fontweight("bold")
        ax.title.set_horizontalalignment("left")
        ax.title.set_position((0, 1.0))
        ax.title.set_y(1.02)
        ax.yaxis.set_major_formatter(FormatStrFormatter(formatter))
        ax.grid(axis="y", color="#E3E7EA", linewidth=.55)
        ax.set_axisbelow(True)
        if field == "policy_loss":
            ax.axhline(0, color="#8C959D", linewidth=.7, linestyle=(0, (3, 3)), zorder=0)
        assert len(ax.lines[0].get_xdata()) == 32
        assert list(ax.lines[0].get_ydata()) == values
    fig.text(.095, .955, "Historical PPO training diagnostics", fontsize=11.0,
             fontweight="bold", ha="left", va="top")
    fig.text(.095, .89, "32 completed rollout batches | rounded console values | one run",
             fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.text(.535, .095, "Completed rollout batch", fontsize=9., ha="center")
    fig.text(.095, .025, "Final model export failed (exit 1); no post-training evaluation.",
             fontsize=8.0, color=FAILURE, ha="left", fontweight="bold")
    fig.canvas.draw()
    audit = {
        "source": SOURCE.relative_to(ROOT.parent).as_posix(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
        "rows": len(rows), "steps": steps, "plotted_series": series,
        "smoothing": False, "confidence_intervals": False,
        "marker_note": "Diamond marks the last recorded batch; all circles show raw console values.",
        "experiment_status": "Final save failed, exit 1; no verified saved adapter or post-training accuracy",
        "figure_size_mm": SIZE_MM, "minimum_font_pt": 8.0,
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
    base = args.output_dir / "observed_attempt"
    # No bbox_inches='tight': these are the exact final report dimensions.
    fig.savefig(base.with_suffix(".png"), dpi=300)
    if not args.preview_only:
        fig.savefig(base.with_suffix(".pdf"), metadata={
            "Title": "Historical PPO diagnostics: final model saving failed",
            "Subject": "32 raw rounded console records; no post-training evaluation",
        })
    with Image.open(base.with_suffix(".png")) as im:
        im.convert("L").save(base.with_name(base.name + "_grayscale").with_suffix(".png"), dpi=(300, 300))
    base.with_name(base.name + "_plot_audit").with_suffix(".json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"rows": audit["rows"], "size_mm": SIZE_MM,
                      "layout_issues": audit.get("scipilot_layout_issues", [])}))


if __name__ == "__main__":
    main()
