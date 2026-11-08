"""Redraw program figures from saved evidence without rerunning any candidate.

Default output is program_search/figures. Use --output-dir to stage a separate review set.
Original JSON/CSV files and platform exports are read-only inputs. The two
packing figures use saved centers/radii verbatim; no numerical optimization,
radius repair, or coordinate adjustment is performed here.

Optional --qa-skill-dir enables SciPilot's style, profiling, layout audit and
export helpers. They are not required for the portable reproduction path.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MM = 1 / 25.4
BLUE = "#0072B2"
ORANGE = "#D55E00"
INK = "#25313B"
GRAY = "#525B63"
FILL = "#D9EAF4"
BEST_ID = "4705b8e92e1b"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_data():
    csv_path = ROOT / "evidence/online_evaluations.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("generation", "island"):
            row[key] = int(row[key])
        for key in ("score", "elapsed_minutes"):
            row[key] = float(row[key])
    rows.sort(key=lambda r: r["elapsed_minutes"])
    online = read_json(ROOT / "evidence/online_result_summary.json")
    local = read_json(ROOT / f"evidence/verified_{BEST_ID}.json")
    initial = read_json(ROOT / "evidence/local_baseline.json")
    assert len(rows) == online["valid_evaluated_candidates"] == 5
    best = max(rows, key=lambda r: r["score"])
    assert best["candidate_id"] == online["best_candidate_id"] == BEST_ID
    assert best["score"] == online["best_online_score"] == local["online_score"]
    saved_packing = ROOT / local["packing_file"].replace("\\", "/")
    packing = read_json(saved_packing)
    initial_packing = read_json(ROOT / "evidence/local_packing.json")
    return rows, online, local, initial, packing, initial_packing, saved_packing


def setup(qa_dir):
    helpers = {}
    if qa_dir is not None:
        sys.path.insert(0, str(qa_dir / "scripts"))
        from setup_style import setup_style
        from visual_qa import audit_layout, render_preview
        from export_figure import export_figure
        setup_style(journal="general", lang="en", use_sciplots=False)
        helpers = dict(audit_layout=audit_layout, render_preview=render_preview,
                       export_figure=export_figure)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": .65,
        "xtick.major.width": .65, "ytick.major.width": .65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": False, "savefig.facecolor": "white",
        "figure.constrained_layout.use": False,
    })
    return helpers


def search_figure(rows):
    fig = plt.figure(figsize=(166 * MM, 74 * MM))
    axes = [fig.add_axes([.085, .235, .35, .53]),
            fig.add_axes([.565, .235, .40, .53])]
    xs = np.array([r["elapsed_minutes"] for r in rows])
    ys = np.array([r["score"] for r in rows])
    best = np.maximum.accumulate(ys)
    for ax in axes:
        ax.step(xs, best, where="post", color=GRAY, linewidth=1.2, zorder=2)
        for generation, marker, color in [(0, "o", BLUE), (1, "s", ORANGE)]:
            chosen = [i for i, row in enumerate(rows) if row["generation"] == generation]
            ax.scatter(xs[chosen], ys[chosen], marker=marker, color=color, s=22,
                       edgecolors="white", linewidths=.45, zorder=4)
        ax.set_xlim(0, 45)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_xlabel("Elapsed time (min)", labelpad=4)
        ax.grid(axis="y", color="#E3E7EA", linewidth=.55, zorder=0)
    axes[0].set(ylim=(0, 3), yticks=[0, 1, 2, 3], ylabel="Sum of 26 radii")
    axes[1].set(ylim=(2.43, 2.66), yticks=[2.45, 2.50, 2.55, 2.60, 2.65])
    axes[0].set_title("a  Full score scale", loc="left", fontweight="bold", pad=7)
    axes[1].set_title("b  Expanded y-axis", loc="left", fontweight="bold", pad=7)
    # Break marks explicitly identify the non-zero origin of the detail view.
    for height in [-.017, .017]:
        axes[1].plot([-.012, .012], [height-.012, height+.012],
                     transform=axes[1].transAxes, color=INK, linewidth=.65,
                     clip_on=False)
    i = int(np.argmax(ys))
    axes[1].annotate(f"Best online: {ys[i]:.9f}", xy=(xs[i], ys[i]),
                     xytext=(1.6, 2.642), fontsize=8, color=INK,
                     arrowprops={"arrowstyle": "-", "color": GRAY, "lw": .65},
                     ha="left", va="center")
    handles = [Line2D([0], [0], marker="o", linestyle="none", color=BLUE,
                      markersize=4.2, label="Generation 0 (initialization)"),
               Line2D([0], [0], marker="s", linestyle="none", color=ORANGE,
                      markersize=4.2, label="Generation 1"),
               Line2D([0], [0], color=GRAY, lw=1.2, drawstyle="steps-post",
                      label="Best observed online score")]
    fig.text(.085, .965, "program online search: five valid evaluations", fontsize=10,
             fontweight="bold", va="top")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.075, .917),
               ncol=3, frameon=False, handlelength=1.7, handletextpad=.4,
               columnspacing=1.2, borderaxespad=0, fontsize=7.5)
    fig.text(.085, .075, "Time from 12:38:01 (UTC+08:00), 31 Oct 2026; only recorded evaluations shown.",
             fontsize=7.5, va="center", color=GRAY)
    fig.text(.085, .025, "Same data in both panels. Failed and unfinished evaluations are omitted; run stopped manually.",
             fontsize=7.5, va="center", color=GRAY)
    return fig, {"n_valid_online": len(rows), "x_minutes": xs.tolist(),
                 "scores": ys.tolist(), "running_best": best.tolist(),
                 "no_result_before_first_or_after_last_evaluation": True,
                 "local_reproduction_values_included": False,
                 "full_y_limits": [0, 3], "detail_y_limits": [2.43, 2.66],
                 "generation_marker_encoding": {"0": "blue circles", "1": "orange squares"},
                 "island_ids_remain_in_source_csv_and_report_table": True}


def packing_figure(packing, summary, name):
    centers = np.asarray(packing["centers"], dtype=float)
    radii = np.asarray(packing["radii"], dtype=float)
    assert centers.shape == (26, 2) and radii.shape == (26,)
    assert np.isfinite(centers).all() and np.isfinite(radii).all() and (radii >= 0).all()
    is_initial = name == "initial_packing"
    recorded_sum = summary["sum_radii"] if is_initial else summary["local_score"]
    assert abs(float(np.sum(radii)) - recorded_sum) < 1e-12
    fig = plt.figure(figsize=(75 * MM, 75 * MM))
    ax = fig.add_axes([.19, .20, .69, .69])
    patches = []
    for center, radius in zip(packing["centers"], packing["radii"]):
        patch = Circle(tuple(center), radius, facecolor=FILL, edgecolor=BLUE, linewidth=.65)
        ax.add_patch(patch)
        patches.append(patch)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="x", ylabel="y")
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([0, .25, .5, .75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.spines[["top", "right"]].set_visible(True)
    ax.tick_params(pad=3)
    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 3
    title = "Initial packing | local preparation" if is_initial else "Best candidate | local reproduction"
    fig.text(.5, .98, title, fontsize=8.5, fontweight="bold", ha="center", va="top")
    fig.text(.5, .925, f"Sum of radii = {recorded_sum:.9f}", fontsize=8, ha="center", va="top")
    zeros = int(np.count_nonzero(radii == 0))
    foot = f"26 circles; {zeros} zero-radius circles retained." if is_initial else "26 circles; saved local geometry only."
    fig.text(.5, .035, foot, fontsize=7, ha="center", va="center", color=GRAY)
    # Assert exact binary floats and count at the artist level, not merely image appearance.
    assert len(patches) == 26
    for i, patch in enumerate(patches):
        assert tuple(patch.center) == tuple(packing["centers"][i])
        assert patch.radius == packing["radii"][i]
    fig.canvas.draw()
    ratio = (ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]) / (
        ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1])
    assert abs(ratio - 1) < 1e-12
    return fig, {"circle_count": 26, "zero_radius_count": zeros,
                 "all_artist_centers_and_radii_equal_saved_json": True,
                 "x_limits": list(ax.get_xlim()), "y_limits": list(ax.get_ylim()),
                 "axis_aspect": ax.get_aspect(), "data_unit_pixel_ratio_x_over_y": ratio,
                 "recorded_local_sum_not_recomputed_score": recorded_sum,
                 "source": "local preparation" if is_initial else "saved local reproduction",
                 "online_score_used_to_label_geometry": False,
                 "optimizer_or_candidate_executed": False}


def export(fig, basename, helpers, preview_only):
    paths = []
    fig.canvas.draw()
    issues = helpers["audit_layout"](fig) if helpers else []
    assert not [x for x in issues if x[0] == "FAIL"], issues
    if preview_only:
        fig.savefig(str(basename) + ".png", dpi=300, bbox_inches=None)
        paths.append(str(basename) + ".png")
    elif helpers:
        paths.extend(helpers["export_figure"](fig, str(basename), formats=["pdf", "png"],
                     dpi=300, size_inches=tuple(fig.get_size_inches()),
                     grayscale_preview=False, tight=False))
    else:
        for extension in ["pdf", "png"]:
            path = str(basename) + "." + extension
            fig.savefig(path, dpi=300, bbox_inches=None)
            paths.append(path)
    with Image.open(str(basename) + ".png") as image:
        gray = image.convert("L")
        gray_path = str(basename) + "_grayscale.png"
        gray.save(gray_path, dpi=(300, 300))
        paths.append(gray_path)
    return {"issues": issues, "size_inches": fig.get_size_inches().tolist(),
            "size_mm": (fig.get_size_inches() * 25.4).tolist(),
            "min_text_size_pt": min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text)
                                    if t.get_visible() and t.get_text()),
            "dpi": 300, "tight_crop": False, "files": paths,
            "preview_only": preview_only}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--qa-skill-dir", type=Path)
    parser.add_argument("--preview-only", action="store_true",
                        help="Make PNG review images first; defer vector PDF export.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / "evidence/online_evaluations.csv",
              ROOT / "evidence/online_result_summary.json",
              ROOT / f"evidence/verified_{BEST_ID}.json",
              ROOT / "evidence/local_baseline.json",
              ROOT / "evidence/local_packing.json",
              ROOT / "evidence/best_local_packing.json"]
    before = {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs}
    rows, online, local, initial, packing, initial_packing, _ = load_data()
    helpers = setup(args.qa_skill_dir)
    figures = [(*search_figure(rows), "online_progress"),
               (*packing_figure(initial_packing, initial, "initial_packing"), "initial_packing"),
               (*packing_figure(packing, local, "online_best_reproduced"), "online_best_reproduced")]
    output = {}
    for fig, semantic, name in figures:
        output[name] = dict(semantic=semantic, formal=export(fig, args.output_dir / name,
                                                          helpers, args.preview_only))
        plt.close(fig)
    after = {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs}
    assert before == after
    report = {"input_sha256_before": before, "input_sha256_after": after,
              "all_historical_data_unchanged": True, "matplotlib": matplotlib.__version__,
              "figures": output, "no_model_download_or_training": True,
              "original_plotting_scripts_unchanged": True,
              "notes": ["General A4 academic-report style; not a specific journal submission.",
                        "Generation and island columns are categorical IDs, not Boolean measurements.",
                        "Five observed candidates are not statistical replicates; no CI or regression is claimed.",
                        "Circle sums are recorded local values; the search plot contains online values only."]}
    (args.output_dir / "figure_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir),
                      "figures": list(output), "historical_data_unchanged": True}))


if __name__ == "__main__":
    main()

