"""Summarise phase 1 logs into tables and the main chart.

Samples of the same item are not independent, so confidence intervals are bootstrapped over items
(words), and the binomial test counts items whose majority choice matches the human pattern.

Outputs (results/):
  summary_match.csv       % round-congruent matches per model x condition, item-bootstrap 95% CI, binomial p
  summary_2x2.csv         P(round shape) per consonant class x vowel class (fresh words, spelled, text)
  summary_rating.csv      mean sharpness rating (1 round .. 7 sharp) per word class
  summary_crossmodal.csv  % round-congruent choices per dimension
  summary_classic.csv     % congruent choices per classic correspondence
  summary_shape_rating.csv  model vs. thesis participants' sharpness ratings of the six thesis shapes
  summary_graded.csv      logistic weights of P(sharp name) on frequency, amplitude, spikiness, per language
  position_bias.csv       share of answers picking the first option (0.5 = no bias)
  screening_flags.csv     words a model said it recognises
  figures/main.png        the main chart

Usage: uv run python src/analyze.py [--include-mock]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

HUMAN_BASELINE = 0.95  # Ramachandran & Hubbard (2001) report 95-98%; check the paper before citing.

CONDITIONS = [
    ("Classic words", dict(word_set="famous", rendering="spelled", shape_mode="text")),
    ("New pseudowords", dict(word_set="fresh", rendering="spelled", shape_mode="text")),
    ("IPA", dict(word_set="fresh", rendering="ipa", shape_mode="text")),
    ("Sounds-like", dict(word_set="fresh", rendering="sounds_like", shape_mode="text")),
    ("Articulatory", dict(word_set="fresh", rendering="articulatory", shape_mode="text")),
    ("Katakana", dict(word_set="fresh", rendering="katakana", shape_mode="text")),
    ("Image", dict(word_set="fresh", rendering="spelled", shape_mode="image", shape_set="classic")),
    ("Image, mixed shapes", dict(word_set="fresh", rendering="spelled", shape_mode="image", shape_set="mixed")),
]


def load(include_mock):
    """Load all logs, keeping the latest successful record per model, trial and sample."""
    rows = []
    for p in sorted((RES / "raw").glob("*.jsonl")):
        if p.stem.endswith("-mock") and not include_mock:
            continue
        rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No logs found in results/raw")
    df = pd.DataFrame(rows)
    df = df[df["error"].isna()]
    if "shape_set" not in df:
        df["shape_set"] = None
    old_image = (df["task"] == "match") & (df["shape_mode"] == "image") & df["shape_set"].isna()
    df.loc[old_image, "shape_set"] = "mixed"
    df.loc[old_image, "trial_id"] = df.loc[old_image, "trial_id"].str.replace("match|image|", "match|image-mixed|", regex=False)
    return df.drop_duplicates(["model", "trial_id", "sample"], keep="last")


def item_rate(d, col, value, n_boot=2000, seed=0):
    """Rate of col == value with an item-bootstrap CI and a binomial test on item majorities."""
    d = d[d["choice"].notna()]
    if d.empty:
        return None
    per_item = d.assign(hit=(d[col] == value)).groupby("word")["hit"].mean()
    rng = np.random.default_rng(seed)
    boots = [rng.choice(per_item.values, len(per_item)).mean() for _ in range(n_boot)]
    majority = int((per_item > 0.5).sum())
    decided = int((per_item != 0.5).sum())
    p = binomtest(majority, decided, 0.5).pvalue if decided else np.nan
    return {"rate": d.assign(hit=(d[col] == value))["hit"].mean(), "ci_lo": np.percentile(boots, 2.5),
            "ci_hi": np.percentile(boots, 97.5), "items": len(per_item), "samples": len(d),
            "items_majority": majority, "binom_p": p}


def summarise_match(df):
    """Congruent-match rate per model and condition, using only pure round/sharp words."""
    m = df[(df["task"] == "match") & (df["cell"] == "pure")].copy()
    m["congruent"] = m["choice"] == m["word_class"]
    out = []
    for model, dm in m.groupby("model"):
        for label, f in CONDITIONS:
            sel = dm
            for k, v in f.items():
                sel = sel[sel[k] == v]
            r = item_rate(sel.assign(c=sel["congruent"]), "c", True)
            if r:
                out.append({"model": model, "condition": label, **r})
    return pd.DataFrame(out)


def summarise_2x2(df):
    """P(round shape chosen) by consonant and vowel class, fresh spelled words in text mode."""
    m = df[(df["task"] == "match") & (df["word_set"] == "fresh") & (df["rendering"] == "spelled")
           & (df["shape_mode"] == "text") & df["choice"].notna()]
    if m.empty:
        return pd.DataFrame()
    m = m.assign(round_choice=m["choice"] == "round")
    return m.groupby(["model", "consonant_class", "vowel_class"])["round_choice"].agg(["mean", "count"]).reset_index()


def summarise_rating(df):
    """Mean sharpness rating per model, rendering and word class."""
    m = df[(df["task"] == "rating") & df["choice"].notna()]
    if m.empty:
        return pd.DataFrame()
    m = m.assign(sharpness=m["choice"].astype(float))
    return m.groupby(["model", "rendering", "consonant_class", "vowel_class"])["sharpness"].agg(["mean", "std", "count"]).reset_index()


def summarise_crossmodal(df):
    """Congruent rate per model, dimension and rendering (pure words only)."""
    m = df[(df["task"] == "crossmodal") & (df["cell"] == "pure")].copy()
    if m.empty:
        return pd.DataFrame()
    out = []
    for (model, dim, rend), d in m.groupby(["model", "dimension", "rendering"]):
        r = item_rate(d.assign(c=d["choice"] == d["word_class"]), "c", True)
        if r:
            out.append({"model": model, "dimension": dim, "rendering": rend, **r})
    return pd.DataFrame(out)


def summarise_classic(df):
    """Congruent rate per model and classic item."""
    m = df[(df["task"] == "classic") & df["choice"].notna()]
    if m.empty:
        return pd.DataFrame()
    m = m.assign(congruent=m["choice"] == "congruent")
    return m.groupby(["model", "item", "source"])["congruent"].agg(["mean", "count"]).reset_index()


def summarise_graded(df):
    """Logistic regression of choosing the sharp name on standardised shape parameters."""
    import statsmodels.api as sm
    m = df[(df["task"] == "graded") & df["choice"].notna()]
    if m.empty:
        return pd.DataFrame()
    out = []
    if "shape_set" in m:
        m = m.assign(shape_set=m["shape_set"].fillna("mixed"))
    for (model, shape_set, lang), d in m.groupby(["model", "shape_set", "lang"]):
        y = (d["choice"] == "sharp").astype(int)
        if y.nunique() < 2:
            continue
        X = d[["frequency", "amplitude", "spikiness"]].astype(float)
        X = sm.add_constant((X - X.mean()) / X.std())
        try:
            fit = sm.Logit(y, X).fit(disp=0)
        except Exception as e:
            out.append({"model": model, "shape_set": shape_set, "lang": lang, "error": str(e)})
            continue
        row = {"model": model, "shape_set": shape_set, "lang": lang, "n": len(d), "p_sharp": y.mean()}
        for k in ["const", "frequency", "amplitude", "spikiness"]:
            row[f"b_{k}"], row[f"se_{k}"] = fit.params[k], fit.bse[k]
        out.append(row)
    return pd.DataFrame(out)


def position_bias(df):
    """Share of two-choice answers that picked the first option, per model and task (0.5 = no bias)."""
    m = df[df["answer_type"].isin(["letter", "word"]) & df["parsed"].notna()].copy()
    if "first_option" not in m:
        m["first_option"] = None
    m = m[(m["answer_type"] == "letter") | m["first_option"].notna()]
    if m.empty:
        return pd.DataFrame()
    first = m["parsed"] == m["first_option"].where(m["answer_type"] == "word", "A")
    return m.assign(first=first).groupby(["model", "task"])["first"].agg(["mean", "count"]).reset_index()


def summarise_shape_rating(df):
    """Model mean sharpness (1-5) per thesis shape next to the thesis participants' mean, with Spearman rho."""
    from scipy.stats import spearmanr
    m = df[(df["task"] == "shape_rating") & df["choice"].notna()]
    if m.empty:
        return pd.DataFrame()
    human = pd.read_csv(ROOT / "stimuli" / "thesis_shape_ratings.csv")[["shape_id", "mean"]].rename(columns={"mean": "human_mean"})
    t = m.assign(r=m["choice"].astype(float)).groupby(["model", "shape_id"])["r"].agg(["mean", "count"]).reset_index()
    t = t.merge(human, on="shape_id")
    rho = {model: spearmanr(g["mean"], g["human_mean"]).statistic for model, g in t.groupby("model")}
    t["spearman_rho"] = t["model"].map(rho)
    return t


def plot_shape_rating(t, path):
    """For each thesis shape (pictured, softest to sharpest by people), people's and each model's mean rating."""
    if t.empty:
        return
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from PIL import Image as PILImage
    shapes = pd.read_csv(ROOT / "stimuli" / "shapes.csv").set_index("shape_id")["file"]
    order = t.drop_duplicates("shape_id").sort_values("human_mean")
    x = {sid: i for i, sid in enumerate(order["shape_id"])}
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(x.values()), order["human_mean"], c="k", lw=1.5, marker="s", ms=8, label="People (PerForm thesis, n = 24)", zorder=4)
    for model, g in t.groupby("model"):
        g = g.assign(x=g["shape_id"].map(x)).sort_values("x")
        rho = g["spearman_rho"].iloc[0]
        ax.plot(g["x"], g["mean"], marker="o", ms=7, lw=1, alpha=0.9,
                label=f"{model.split(':', 1)[1]} (same order as people: rho = {rho:.2f})", zorder=3)
    for sid, i in x.items():
        img = PILImage.open(ROOT / shapes[sid]).convert("L").resize((90, 90))
        ab = AnnotationBbox(OffsetImage(np.asarray(img), cmap="gray", zoom=0.55), (i, 0), xycoords=("data", "axes fraction"),
                            box_alignment=(0.5, 1.15), frameon=False)
        ax.add_artist(ab)
    ax.set_xticks(list(x.values()), [""] * len(x))
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0.7, 5.3)
    ax.set_yticks([1, 2, 3, 4, 5], ["1 not at all", "2", "3 somewhat", "4", "5 very much"])
    ax.set_ylabel("Mean rating")
    ax.set_title("How sharp / hard does this shape feel?", pad=12)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def screening_flags(df):
    """Words any model claimed to recognise, with what it said."""
    m = df[(df["task"] == "screen") & (df["parsed"] == "KNOWN")]
    return m[["word", "model", "raw"]].drop_duplicates()


def plot_main(summary, path):
    """Grouped bars of % round-congruent matches by condition, one colour per model."""
    if summary.empty:
        return
    labels = [c for c, _ in CONDITIONS if c in set(summary["condition"])]
    models = sorted(summary["model"].unique())
    x = np.arange(len(labels))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(models):
        s = summary[summary["model"] == model].set_index("condition").reindex(labels)
        pos = x - 0.4 + width * (i + 0.5)
        err = np.vstack([s["rate"] - s["ci_lo"], s["ci_hi"] - s["rate"]]) * 100
        ax.bar(pos, s["rate"] * 100, width, label=model, yerr=err, capsize=3)
    ax.axhline(HUMAN_BASELINE * 100, ls="--", c="k", lw=1, label="Human baseline (bouba/kiki)")
    ax.axhline(50, ls=":", c="gray", lw=1, label="Chance")
    ax.set_xticks(x, [label.replace(" ", "\n") for label in labels])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% round-sounding words matched to round shape\n(and sharp to sharp)")
    ax.set_title("Do models match sounds to shapes the way people do?")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def summarise_quadrant(df):
    """Per model: congruent rate with letters (spelled) vs without letter-shape cues (articulatory + katakana)."""
    m = df[(df["task"] == "match") & (df["cell"] == "pure") & (df["word_set"] == "fresh")
           & (df["shape_mode"] == "text")].copy()
    m["c"] = m["choice"] == m["word_class"]
    out = []
    for model, d in m.groupby("model"):
        x = item_rate(d[d["rendering"] == "spelled"], "c", True)
        y = item_rate(d[d["rendering"].isin(["articulatory", "katakana"])], "c", True)
        if x and y:
            out.append({"model": model, **{f"x_{k}": v for k, v in x.items()}, **{f"y_{k}": v for k, v in y.items()}})
    return pd.DataFrame(out)


def plot_quadrant(q, path):
    """Each model as a point: score with letters (x) against score without letter-shape cues (y)."""
    if q.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.axvline(50, c="gray", ls=":", lw=1)
    ax.axhline(50, c="gray", ls=":", lw=1)
    ax.plot([0, 100], [0, 100], c="lightgray", ls="--", lw=1)
    ax.scatter([HUMAN_BASELINE * 100], [HUMAN_BASELINE * 100], marker="*", s=160, c="k", zorder=3)
    ax.annotate("people (bouba/kiki)", (HUMAN_BASELINE * 100, HUMAN_BASELINE * 100), xytext=(-8, 8),
                textcoords="offset points", ha="right", fontsize=8)
    for _, r in q.iterrows():
        xerr = [[(r.x_rate - r.x_ci_lo) * 100], [(r.x_ci_hi - r.x_rate) * 100]]
        yerr = [[(r.y_rate - r.y_ci_lo) * 100], [(r.y_ci_hi - r.y_rate) * 100]]
        ax.errorbar(r.x_rate * 100, r.y_rate * 100, xerr=xerr, yerr=yerr, fmt="o", capsize=3, label=r.model)
    for x, y, t in [(97, 53, "matches people,\nnot only via letters"), (97, 3, "letter-shape\nshortcut"),
                    (3, 3, "no effect"), (3, 97, "only without\nletters")]:
        ax.text(x, y, t, ha="right" if x > 50 else "left", va="top" if y > 60 else "bottom", fontsize=8, c="gray")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% correct matches, spelled words")
    ax.set_ylabel("% correct matches, no letter-shape cues\n(sound descriptions + katakana)")
    ax.set_title("Is the effect carried by letters or by sound?")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.62), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    """Write every summary table and the main chart."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-mock", action="store_true")
    args = ap.parse_args()
    df = load(args.include_mock)
    (RES / "figures").mkdir(parents=True, exist_ok=True)
    tables = {
        "summary_match": summarise_match(df), "summary_2x2": summarise_2x2(df),
        "summary_rating": summarise_rating(df), "summary_crossmodal": summarise_crossmodal(df),
        "summary_classic": summarise_classic(df), "summary_graded": summarise_graded(df),
        "summary_shape_rating": summarise_shape_rating(df),
        "position_bias": position_bias(df), "screening_flags": screening_flags(df),
    }
    for name, t in tables.items():
        if not t.empty:
            t.to_csv(RES / f"{name}.csv", index=False)
            print(f"\n== {name} ==\n{t.head(20).to_string(index=False)}")
    tables["summary_quadrant"] = summarise_quadrant(df)
    plot_main(tables["summary_match"], RES / "figures" / "main.png")
    plot_quadrant(tables["summary_quadrant"], RES / "figures" / "quadrant.png")
    plot_shape_rating(tables["summary_shape_rating"], RES / "figures" / "shape_rating.png")
    print(f"\nChart: {RES / 'figures' / 'main.png'}")


if __name__ == "__main__":
    main()
