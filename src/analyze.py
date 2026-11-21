"""Summarise phase 1 logs into tables and the main chart.

Samples of the same item are not independent, so confidence intervals are bootstrapped over items
(words), and the binomial test counts items whose majority choice matches the human pattern.

Outputs (results/):
  summary_match.csv       % round-congruent matches per model x condition, item-bootstrap 95% CI, binomial p
  summary_2x2.csv         P(round shape) per consonant class x vowel class (fresh words, spelled, text)
  summary_rating.csv      mean sharpness rating (1 round .. 7 sharp) per word class
  summary_crossmodal.csv  % round-congruent choices per dimension
  summary_classic.csv     % congruent choices per classic correspondence
  summary_graded.csv      logistic weights of P(sharp name) on frequency, amplitude, spikiness, per language
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
    ("Image", dict(word_set="fresh", rendering="spelled", shape_mode="image")),
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
    out = []
    for (model, lang), d in m.groupby(["model", "lang"]):
        y = (d["choice"] == "sharp").astype(int)
        if y.nunique() < 2:
            continue
        X = d[["frequency", "amplitude", "spikiness"]].astype(float)
        X = sm.add_constant((X - X.mean()) / X.std())
        try:
            fit = sm.Logit(y, X).fit(disp=0)
        except Exception as e:
            out.append({"model": model, "lang": lang, "error": str(e)})
            continue
        row = {"model": model, "lang": lang, "n": len(d), "p_sharp": y.mean()}
        for k in ["const", "frequency", "amplitude", "spikiness"]:
            row[f"b_{k}"], row[f"se_{k}"] = fit.params[k], fit.bse[k]
        out.append(row)
    return pd.DataFrame(out)


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
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        s = summary[summary["model"] == model].set_index("condition").reindex(labels)
        pos = x - 0.4 + width * (i + 0.5)
        err = np.vstack([s["rate"] - s["ci_lo"], s["ci_hi"] - s["rate"]]) * 100
        ax.bar(pos, s["rate"] * 100, width, label=model, yerr=err, capsize=3)
    ax.axhline(HUMAN_BASELINE * 100, ls="--", c="k", lw=1, label="Human baseline (bouba/kiki)")
    ax.axhline(50, ls=":", c="gray", lw=1, label="Chance")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% round-sounding words matched to round shape\n(and sharp to sharp)")
    ax.set_title("Do models match sounds to shapes the way people do?")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
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
        "screening_flags": screening_flags(df),
    }
    for name, t in tables.items():
        if not t.empty:
            t.to_csv(RES / f"{name}.csv", index=False)
            print(f"\n== {name} ==\n{t.head(20).to_string(index=False)}")
    plot_main(tables["summary_match"], RES / "figures" / "main.png")
    print(f"\nChart: {RES / 'figures' / 'main.png'}")


if __name__ == "__main__":
    main()
