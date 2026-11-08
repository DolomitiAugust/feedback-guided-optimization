"""Editable vector explanations; schematic geometry is not experimental data."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
BLUE, ORANGE, GREEN, INK, MUTED = "#0072B2", "#D55E00", "#009E73", "#26343E", "#5B6570"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})

def box(ax, x, y, width, height, text, color, fill):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.008,rounding_size=0.016",
                          linewidth=0.85, edgecolor=color, facecolor=fill)
    ax.add_patch(patch)
    ax.text(x + width/2, y + height/2, text, ha="center", va="center", color=INK, linespacing=1.4)

def arrow(ax, start, end, color=INK, style="-", connection="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                linewidth=0.9, color=color, linestyle=style,
                                connectionstyle=connection, shrinkA=1, shrinkB=1))

def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / (stem + ".pdf"))
    fig.savefig(OUT / (stem + ".png"), dpi=300)
    plt.close(fig)

def feedback_overview():
    fig, ax = plt.subplots(figsize=(166/25.4, 72/25.4))
    fig.subplots_adjust(0.02, 0.035, 0.98, 0.98)
    ax.set(xlim=(0,1), ylim=(0,1)); ax.axis("off")
    groups = [
        (0.63, BLUE, "#EDF5FA", "a  Parameter adaptation", [
            "Arithmetic\ntraining prompts", "Actor responses\n+ Critic values",
            "Correctness reward\n+ reference penalty", "GAE + clipped PPO\nupdate adapters"]),
        (0.18, ORANGE, "#FCF1EA", "b  Program search", [
            "26-circle task\nand fixed constraints", "Planner + Coder\npropose a program",
            "Geometry evaluator\nvalidity + sum of radii", "Select survivors\nretain feedback"]),
    ]
    xs, width, height = [0.015, 0.275, 0.535, 0.795], 0.19, 0.19
    for y, color, fill, title, labels in groups:
        ax.text(0.015, y+0.265, title, color=color, weight="bold", fontsize=9.5, va="center")
        for x, text in zip(xs, labels): box(ax, x, y, width, height, text, color, fill)
        for left, right in zip(xs, xs[1:]): arrow(ax, (left+width, y+height/2), (right, y+height/2))
        # Return feedback to the generator, rather than directly to the task.
        ax.plot([0.89,0.89,0.37,0.37], [y,y-0.055,y-0.055,y-0.015],
                color=color, linewidth=0.9, linestyle=(0,(3,2)))
        arrow(ax, (0.37,y-0.015), (0.37,y), color=color)
        text = "Updated object: LoRA parameters" if color == BLUE else "Updated object: candidate source code"
        ax.text(0.62,y-0.105,text,ha="center",va="center",color=color,fontsize=7.6)
    save(fig,"feedback_mechanisms")

def lora_branch():
    config = json.loads((ROOT / "policy_adaptation/results/ppo_outputs/run_config.json").read_text())
    argv = config["argv"]
    rank = int(argv[argv.index("--lora_rank")+1])
    alpha = int(argv[argv.index("--lora_alpha")+1])
    scale = alpha/rank
    fig, ax = plt.subplots(figsize=(146/25.4, 57/25.4))
    fig.subplots_adjust(0.02, 0.04, 0.98, 0.98)
    ax.set(xlim=(0,1), ylim=(0,1)); ax.axis("off")
    ax.text(0.01,0.95,"Low-rank adaptation of one linear projection",fontsize=9.5,weight="bold",color=INK)
    ax.text(0.045,0.54,r"$\mathbf{x}$",fontsize=13,ha="center",va="center")
    ax.plot([0.085,0.15,0.15],[0.54,0.54,0.76],color=INK,linewidth=0.9)
    arrow(ax,(0.15,0.76),(0.31,0.76))
    box(ax,0.31,0.65,0.30,0.22,r"Frozen base $W_0$",MUTED,"#F1F3F4")
    ax.plot([0.61,0.84,0.84],[0.76,0.76,0.595],color=INK,linewidth=0.9)
    arrow(ax,(0.84,0.64),(0.84,0.59))
    ax.plot([0.15,0.15],[0.54,0.31],color=INK,linewidth=0.9)
    arrow(ax,(0.15,0.31),(0.24,0.31),BLUE)
    box(ax,0.24,0.20,0.18,0.22,r"Trainable $A$"+"\n"+fr"rank $r={rank}$",BLUE,"#EDF5FA")
    arrow(ax,(0.42,0.31),(0.51,0.31),BLUE)
    box(ax,0.51,0.20,0.18,0.22,r"Trainable $B$"+"\n"+r"initialized at $0$",BLUE,"#EDF5FA")
    ax.plot([0.69,0.84,0.84],[0.31,0.31,0.485],color=BLUE,linewidth=0.9)
    arrow(ax,(0.84,0.44),(0.84,0.49),BLUE)
    ax.text(0.77,0.24,fr"$\times\alpha/r={scale:g}$",ha="center",color=BLUE,fontsize=8)
    ax.add_patch(Circle((0.84,0.54),0.037,facecolor="white",edgecolor=INK,linewidth=1))
    ax.text(0.84,0.54,"+",ha="center",va="center",fontsize=11)
    arrow(ax,(0.88,0.54),(0.96,0.54))
    ax.text(0.98,0.54,r"$\mathbf{y}$",fontsize=13,ha="center",va="center")
    ax.text(0.50,0.08,r"$\mathbf{y}=W_0\mathbf{x}+(\alpha/r)BA\mathbf{x}$",
            ha="center",va="center",fontsize=11,color=INK)
    save(fig,"lora_mechanism")

if __name__ == "__main__":
    feedback_overview()
    lora_branch()
    print("Wrote editable vector PDF and 300 dpi PNG schematics.")
