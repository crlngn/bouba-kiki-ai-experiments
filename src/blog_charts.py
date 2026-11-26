"""Redraw the three experiment charts in the blog's palette.

Usage: uv run --project <repo> python blog_charts.py <repo> <out_dir> [--bg '#ffffff'] [--font 'Helvetica Neue']
Reuses the repo's analyze.py for the numbers; only the drawing is new.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import pandas as pd
from PIL import Image as PILImage

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("out")
ap.add_argument("--bg", default="#ffffff")
ap.add_argument("--font", default="Helvetica Neue")
args = ap.parse_args()

REPO = Path(args.repo).resolve()
OUT = Path(args.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
import analyze as A  # noqa: E402

# Palette: site ink + accent, two non-purple series at the accent's saturation/lightness.
INK = "#1c1623"          # --color-charcoal
INK2 = "#5b5563"         # secondary text
MUTED = "#8d8793"        # axis labels, quadrant corner notes
RULE = "#d3c8df"         # --color-accent at 30% on paper (grid, reference lines)
RULE2 = "#b9a9cc"        # a bit stronger, for the human/chance lines
ACCENT = "#6c469c"
SERIES = {"ollama:gemma3:27b": ("#1689A8", "Gemma 3 27B"),
          "ollama:gemma3:4b": ("#C46A33", "Gemma 3 4B")}
BG = args.bg

plt.rcParams.update({
    "font.family": args.font,
    "font.size": 12,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": RULE2,
    "axes.linewidth": 0.8,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 14,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 8,
    "ytick.major.pad": 6,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "legend.labelcolor": INK2,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "savefig.dpi": 200,
})


TITLE = {"fontfamily": "Raleway", "fontsize": 16, "fontweight": "medium", "color": INK, "loc": "left", "pad": 16}
_set_title = plt.Axes.set_title
plt.Axes.set_title = lambda ax, s, **kw: _set_title(ax, s, **{**TITLE, **kw})


def strip(ax, left=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_color(RULE2)
    ax.spines["left"].set_color(RULE2)
    ax.set_axisbelow(True)


def label(model):
    return SERIES[model][1]


def color(model):
    return SERIES[model][0]


# ---------------------------------------------------------------- main bars
def plot_main(summary, path):
    labels = [c for c, _ in A.CONDITIONS if c in set(summary["condition"])]
    models = [m for m in SERIES if m in set(summary["model"])]
    x = np.arange(len(labels))
    n = len(models)
    width = 0.66 / n
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.yaxis.grid(True, color=RULE, lw=0.8)
    for i, model in enumerate(models):
        s = summary[summary["model"] == model].set_index("condition").reindex(labels)
        pos = x - 0.33 + width * (i + 0.5)
        err = np.vstack([s["rate"] - s["ci_lo"], s["ci_hi"] - s["rate"]]) * 100
        ax.bar(pos, s["rate"] * 100, width * 0.92, color=color(model), label=label(model), zorder=3)
        ax.errorbar(pos, s["rate"] * 100, yerr=err, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, capthick=0.9,
                    alpha=0.7, zorder=4)
    ax.axhline(A.HUMAN_BASELINE * 100, ls=(0, (6, 4)), c=INK2, lw=1, zorder=2)
    ax.text(len(labels) - 0.55, A.HUMAN_BASELINE * 100 + 1.5, "people, bouba/kiki", ha="right", va="bottom",
            fontsize=10.5, color=INK2)
    ax.axhline(50, ls=(0, (2, 3)), c=INK2, lw=1, zorder=2)
    ax.text(len(labels) - 0.55, 51.5, "chance", ha="right", va="bottom", fontsize=10.5, color=INK2)
    ax.set_xticks(x, [l.replace(", ", ",\n").replace(" ", "\n") if len(l) > 11 else l for l in labels])
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_ylabel("Words matched the way people match them", labelpad=10)
    ax.set_title("Do models match sounds to shapes the way people do?", pad=40)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.005, 1.01), ncol=2, handlelength=1.1, handleheight=1.1,
              columnspacing=1.6, borderaxespad=0, handletextpad=0.6)
    strip(ax, left=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- quadrant
def plot_quadrant(q, path):
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.axvline(50, c=RULE2, ls=(0, (2, 3)), lw=1)
    ax.axhline(50, c=RULE2, ls=(0, (2, 3)), lw=1)
    ax.plot([0, 100], [0, 100], c=RULE, ls=(0, (6, 4)), lw=1)
    hb = A.HUMAN_BASELINE * 100
    ax.scatter([hb], [hb], marker="*", s=220, c=INK, zorder=4, linewidths=0)
    ax.annotate("people, bouba/kiki", (hb, hb), xytext=(-10, 6), textcoords="offset points", ha="right",
                fontsize=10.5, color=INK2)
    for _, r in q.iterrows():
        xerr = [[(r.x_rate - r.x_ci_lo) * 100], [(r.x_ci_hi - r.x_rate) * 100]]
        yerr = [[(r.y_rate - r.y_ci_lo) * 100], [(r.y_ci_hi - r.y_rate) * 100]]
        c = color(r.model)
        ax.errorbar(r.x_rate * 100, r.y_rate * 100, xerr=xerr, yerr=yerr, fmt="none", ecolor=c, elinewidth=1.2,
                    capsize=3, capthick=1.2, alpha=0.8, zorder=3)
        ax.scatter([r.x_rate * 100], [r.y_rate * 100], s=90, c=c, edgecolors=BG, linewidths=2, zorder=5)
        ax.annotate(label(r.model), (r.x_rate * 100, r.y_rate * 100), xytext=(0, -20), textcoords="offset points",
                    ha="center", va="top", fontsize=11.5, color=c, fontweight="medium")
    for x, y, t in [(97, 53, "matches people,\nnot only via letters"), (97, 3, "letter-shape\nshortcut"),
                    (3, 3, "no effect"), (3, 97, "only without\nletters")]:
        ax.text(x, y, t, ha="right" if x > 50 else "left", va="top" if y > 60 else "bottom", fontsize=10.5,
                color=MUTED, linespacing=1.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Correct matches, spelled words", labelpad=10)
    ax.set_ylabel("Correct matches, no letter-shape cues\n(sound descriptions and katakana)", labelpad=10)
    ax.set_title("Is the effect carried by letters or by sound?")
    strip(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- shape rating
def plot_shape_rating(t, path):
    shapes = pd.read_csv(REPO / "stimuli" / "shapes.csv").set_index("shape_id")["file"]
    order = t.drop_duplicates("shape_id").sort_values("human_mean")
    x = {sid: i for i, sid in enumerate(order["shape_id"])}
    xs = list(x.values())
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.yaxis.grid(True, color=RULE, lw=0.8)
    ax.plot(xs, order["human_mean"], c=INK, lw=2, marker="s", ms=7, markeredgecolor=BG, markeredgewidth=1.5,
            label="People (thesis survey, n = 24)", zorder=5)
    for model in [m for m in SERIES if m in set(t["model"])]:
        g = t[t["model"] == model].assign(x=lambda d: d["shape_id"].map(x)).sort_values("x")
        rho = g["spearman_rho"].iloc[0]
        ax.plot(g["x"], g["mean"], marker="o", ms=7, lw=2, c=color(model), markeredgecolor=BG, markeredgewidth=1.5,
                label=f"{label(model)}, rank agreement {rho:.2f}", zorder=4)
    for sid, i in x.items():
        img = PILImage.open(REPO / shapes[sid]).convert("RGBA")
        # recolor the black outline to ink and drop the white paper
        arr = np.asarray(img).astype(float)
        lum = arr[..., :3].mean(axis=2)
        alpha = np.clip((255 - lum) / 255 * 2.2, 0, 1)
        rgb = np.array([int(INK[1:3], 16), int(INK[3:5], 16), int(INK[5:7], 16)], dtype=float)
        out = np.zeros_like(arr)
        out[..., :3] = rgb
        out[..., 3] = alpha * 255
        tile = PILImage.fromarray(out.astype(np.uint8)).resize((110, 110), PILImage.LANCZOS)
        ab = AnnotationBbox(OffsetImage(np.asarray(tile), zoom=0.5), (i, 0), xycoords=("data", "axes fraction"),
                            box_alignment=(0.5, 1.18), frameon=False)
        ax.add_artist(ab)
    ax.set_xticks(xs, [""] * len(xs))
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(0.7, 5.3)
    ax.set_yticks([1, 2, 3, 4, 5], ["1  not at all", "2", "3  somewhat", "4", "5  very much"])
    ax.set_title("How sharp or hard does this shape feel?", pad=44)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.01), ncol=3, handlelength=1.6, borderaxespad=0, columnspacing=1.6)
    strip(ax, left=False)
    ax.spines["bottom"].set_visible(False)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


df = A.load(False)
plot_main(A.summarise_match(df), OUT / "main.png")
plot_quadrant(A.summarise_quadrant(df), OUT / "quadrant.png")
plot_shape_rating(A.summarise_shape_rating(df), OUT / "shape_rating.png")
print("wrote", OUT)
