"""Explain paired validation changes from preserved before/after responses.

Every tile is one of the same 128 questions in its original saved order.
Scores are independently recalculated from the documented first-line integer
rule. This script never loads the model or reruns evaluation/training.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results"
SIZE_MM = (146.0, 80.0)
INK, MUTED = "#27323A", "#5C6670"
COLORS = {"correct_to_correct": "#0072B2", "wrong_to_correct": "#E69F00",
          "wrong_to_wrong": "#E3E7EB", "correct_to_wrong": "#CC79A7"}
ANSWER = re.compile(r"[ \t]*([+-]?[0-9]+)[ \t]*")


def grade(response, label):
    match = ANSWER.fullmatch(response.split("\n", 1)[0].rstrip("\r"))
    return int(match is not None and int(match.group(1)) == int(label))


def read_pairs():
    before = json.loads((DATA / "baseline_validation.json").read_text(encoding="utf-8"))
    after = json.loads((DATA / "trained_validation.json").read_text(encoding="utf-8"))
    for key in ("model", "precision", "metric", "decoding", "n"):
        assert before[key] == after[key], "Unmatched paired evaluation: " + key
    assert before["n"] == 128 and before["precision"] == "torch.bfloat16"
    assert before["metric"] == "first_line_integer_accuracy_v2"
    assert before["adapter"] is None and after["adapter"]
    assert len(before["results"]) == len(after["results"]) == 128
    counts = {key: 0 for key in COLORS}
    rows = []
    for index, (old, new) in enumerate(zip(before["results"], after["results"]), 1):
        for key in ("input", "label", "a", "b"):
            assert old[key] == new[key], "Changed question or ordering"
        assert int(old["label"]) == old["a"] + old["b"]
        old_score, new_score = grade(old["response"], old["label"]), grade(new["response"], new["label"])
        assert old_score == old["score"] and new_score == new["score"]
        outcome = ("correct" if old_score else "wrong") + "_to_" + ("correct" if new_score else "wrong")
        counts[outcome] += 1
        rows.append({"saved_question_index": index, "a": old["a"], "b": old["b"],
                     "label": old["label"], "before_first_line": old["response"].split("\n", 1)[0],
                     "after_first_line": new["response"].split("\n", 1)[0],
                     "before_score": old_score, "after_score": new_score, "outcome": outcome})
    old_correct, new_correct = sum(r["before_score"] for r in rows), sum(r["after_score"] for r in rows)
    assert old_correct / 128 == before["accuracy"] and new_correct / 128 == after["accuracy"]
    return rows, counts, old_correct, new_correct


def build_figure():
    rows, counts, old_correct, new_correct = read_pairs()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.2, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#7E878E", "axes.linewidth": .65,
        "xtick.major.size": 3., "ytick.major.size": 0.,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=tuple(mm / 25.4 for mm in SIZE_MM))
    tiles = fig.add_axes([.055, .315, .52, .475])
    tiles.set(xlim=(0, 16), ylim=(8, 0), aspect="equal")
    tiles.set_axis_off()
    centers = []
    for index, row in enumerate(rows):
        col, line = index % 16, index // 16
        center = (col + .5, line + .5)
        centers.append(center)
        tiles.add_patch(Rectangle((col + .065, line + .065), .87, .87,
                                  facecolor=COLORS[row["outcome"]], edgecolor="white", linewidth=.35))
        if row["outcome"] in {"wrong_to_correct", "correct_to_wrong", "wrong_to_wrong"}:
            marker = "+" if row["outcome"] == "wrong_to_correct" else "x"
            tiles.plot(*center, marker=marker, color=INK, markersize=4.2, markeredgewidth=.8)
    fig.text(.055, .965, "Paired validation outcomes", fontsize=11.0, fontweight="bold", va="top")
    fig.text(.055, .895, "Same 128 questions | BF16 | greedy decoding | first-line integer accuracy",
             color=MUTED, fontsize=8.1, va="top")
    fig.text(.055, .81, "a  Each tile is one question", fontsize=9.0, fontweight="bold", va="bottom")
    fig.text(.725, .81, "b  Correct answers", fontsize=9.0, fontweight="bold", va="bottom")
    fig.text(.315, .285, "Original saved order: left to right, then down", fontsize=7.8,
             color=MUTED, ha="center")
    ax = fig.add_axes([.725, .405, .245, .345])
    for value, y, marker in ((old_correct, 1, "o"), (new_correct, 0, "D")):
        ax.scatter([value], [y], s=29, marker=marker, color=COLORS["correct_to_correct"], zorder=3)
        ax.annotate(f"{value}/128 ({value / 128:.2%})", (value, y), xytext=(-5, 6),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8.)
    ax.set(xlim=(0, 128), ylim=(-.45, 1.5), xticks=[0, 32, 64, 96, 128],
           yticks=[1, 0], yticklabels=["Before PPO", "After PPO"])
    ax.tick_params(labelsize=7.5)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#E3E7EA", linewidth=.55)
    ax.set_axisbelow(True)
    ax.set_xlabel("Correct answers (out of 128)", fontsize=7.8)
    net = new_correct - old_correct
    fig.text(.725, .245, f"{net:+d} correct answers", fontsize=9.0, fontweight="bold")
    fig.text(.725, .195, f"{net / 128 * 100:+.2f} percentage points", fontsize=8.0, color=MUTED)
    handles = [Patch(facecolor=COLORS[key], edgecolor="white", label=label) for key, label in (
        ("correct_to_correct", f"Correct to correct ({counts['correct_to_correct']})"),
        ("wrong_to_correct", f"Wrong to correct ({counts['wrong_to_correct']}; +)"),
        ("wrong_to_wrong", f"Wrong to wrong ({counts['wrong_to_wrong']}; x)"),
    )]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.045, .115), frameon=False,
               ncol=3, fontsize=7.6, handlelength=1.0, handletextpad=.4, columnspacing=1.1)
    fig.text(.055, .082, f"Correct to wrong: {counts['correct_to_wrong']}/128. One run and one validation split.", fontsize=7.8, color=MUTED)
    fig.text(.055, .025, "Descriptive paired results; no statistical significance or generalization claim.", fontsize=7.8, color=MUTED)
    fig.canvas.draw()
    audit = {
        "sources": {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                    for name in ("baseline_validation.json", "trained_validation.json")},
        "n_paired_questions": 128, "same_questions_order_precision_decoding": True,
        "every_response_score_recomputed": True, "transition_counts": counts,
        "before_correct": old_correct, "after_correct": new_correct,
        "before_accuracy": old_correct / 128, "after_accuracy": new_correct / 128,
        "net_correct": net, "accuracy_difference_percentage_points": net / 128 * 100,
        "tiles": [{"saved_question_index": row["saved_question_index"], "center": center,
                   "outcome": row["outcome"]} for row, center in zip(rows, centers)],
        "tile_order": "Original saved response order, row-major 16 columns by 8 rows",
        "correct_count_axis_limits": [0, 128], "confidence_intervals": False,
        "significance_claim": False, "generalization_claim": False,
        "figure_size_mm": SIZE_MM, "minimum_font_pt": 7.5,
        "redundant_encoding": "Improved tiles have +; persistently incorrect tiles have x; no regressions observed",
    }
    return fig, audit, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "figures")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--qa-scripts", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, audit, rows = build_figure()
    if args.qa_scripts:
        sys.path.insert(0, str(args.qa_scripts))
        from visual_qa import audit_layout
        audit["scipilot_layout_issues"] = audit_layout(fig)
        if any(level == "FAIL" for level, _ in audit["scipilot_layout_issues"]):
            raise RuntimeError("Figure layout audit failed")
    base = args.output_dir / "paired_validation_outcomes"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    if not args.preview_only:
        fig.savefig(base.with_suffix(".pdf"), metadata={"Title": "Paired validation outcomes on 128 fixed questions"})
    with Image.open(base.with_suffix(".png")) as im:
        im.convert("L").save(base.with_name(base.name + "_grayscale").with_suffix(".png"), dpi=(300, 300))
    with base.with_name(base.name + "_points").with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    base.with_name(base.name + "_audit").with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"transition_counts": audit["transition_counts"], "layout_issues": audit.get("scipilot_layout_issues", [])}))


if __name__ == "__main__":
    main()
