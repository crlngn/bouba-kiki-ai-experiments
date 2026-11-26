"""Summarise phase 1 logs into tables and the main chart.

Samples of the same item are not independent, so confidence intervals are bootstrapped over items
(words), and the binomial test counts items whose majority choice matches the human pattern.

Outputs (results/):
  summary_match.csv       % round-congruent matches per model x condition, item-bootstrap 95% CI, binomial p
  summary_2x2.csv         P(round shape) per consonant class x vowel class (fresh words, spelled, text)
  summary_rating.csv      mean sharpness rating (1 round .. 7 sharp) per word class
  summary_concept.csv     sharp-side score per concept for round/sharp words and round/sharp shapes
  summary_concept_route.csv  word-to-picture matches predicted from concept profiles vs. direct answers
  summary_classic.csv     % congruent choices per classic correspondence
  summary_shape_rating.csv  model vs. thesis participants' sharpness ratings of the six thesis shapes
  summary_graded.csv      logistic weights of P(sharp name) on frequency, amplitude, spikiness, per language
  position_bias.csv       share of answers picking the first option (0.5 = no bias)
  summary_human_items.csv model vs. human item means for the SoS PASS pseudowords (per rendering) and shapes
  summary_rec_match.csv   recorded words matched to shapes, per rendering (ipa, audio, heard): congruent rate,
                          and for SoS PASS the correlation of P(sharp) with people's pointedness
  summary_spelling_check.csv  per task, how closely answers to the audio follow answers to the model's own spelling
  summary_sidhu.csv       model vs. human nonword means on each of Sidhu et al.'s 25 scales
  summary_sidhu_route.csv concept route on the 25 scales: word-to-shape matches predicted from profile distance
  summary_sidhu_shapes.csv  model Sharp-Round ratings of SoS PASS shapes vs. people's pointedness
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


CONCEPT_DIMS = ["emotion", "scent", "taste", "sound", "touch", "sharp_hard"]
CONCEPT_LABELS = {"emotion": "calm → excited", "scent": "vanilla → lemon", "taste": "sweet → sour",
                  "sound": "hum → harsh beep", "touch": "soft → hard", "sharp_hard": "sharp / hard\n(1-5)"}


def concept_scores(df):
    """One row per concept sample with 'sharp_score' in 0..1: 1 for the sharp-side choice, or (rating-1)/4."""
    m = df[(df["task"] == "concept") & df["choice"].notna()].copy()
    if m.empty:
        return m
    rating = m["dimension"] == "sharp_hard"
    m["sharp_score"] = np.where(rating, (pd.to_numeric(m["choice"], errors="coerce") - 1) / 4,
                                (m["choice"] == "sharp").astype(float))
    m["item"] = np.where(m["stimulus"] == "word", m["word"], m["shape_id"])
    return m


def summarise_concept(df):
    """Mean sharp-side score per model, stimulus type, class and concept (fresh/famous words and classic shapes)."""
    m = concept_scores(df)
    if m.empty:
        return pd.DataFrame()
    m = m[(m["stimulus"] == "image") & (m["shape_set"] == "classic") | (m["stimulus"] == "word") & (m["cell"] == "pure")]
    g = m.groupby(["model", "stimulus", "word_class", "dimension", "item"])["sharp_score"].mean().reset_index()
    return g.groupby(["model", "stimulus", "word_class", "dimension"])["sharp_score"].agg(["mean", "sem", "count"]).reset_index()


def summarise_concept_route(df):
    """Per model: how often picking the picture with the closer concept profile matches the human pattern, vs. asking directly."""
    m = concept_scores(df)
    if m.empty:
        return pd.DataFrame()
    prof = m.groupby(["model", "stimulus", "item", "dimension"])["sharp_score"].mean().unstack("dimension")
    prof = prof.reindex(columns=CONCEPT_DIMS)
    shapes = pd.read_csv(ROOT / "stimuli" / "shapes.csv")
    pairs = {p: dict(zip(g["role"], g["shape_id"])) for p, g in shapes[shapes["set"] == "classic"].groupby("pair")}
    direct = df[(df["task"] == "match") & (df["shape_set"] == "classic") & (df["word_set"] == "fresh")
                & (df["cell"] == "pure") & df["choice"].notna()]
    out = []
    for model, dm in direct.groupby("model"):
        if (model, "word") not in prof.index.droplevel([2]).unique():
            continue
        hits = []
        for (word, pair, cls), d in dm.groupby(["word", "shape_pair", "word_class"]):
            try:
                pw = prof.loc[(model, "word", word)].values
                pr, ps = prof.loc[(model, "image", pairs[pair]["round"])].values, prof.loc[(model, "image", pairs[pair]["sharp"])].values
            except KeyError:
                continue
            ok = ~(np.isnan(pw) | np.isnan(pr) | np.isnan(ps))
            dr, ds = np.linalg.norm((pw - pr)[ok]), np.linalg.norm((pw - ps)[ok])
            if dr == ds:
                continue
            predicted = "round" if dr < ds else "sharp"
            hits.append({"concept_route": predicted == cls, "direct": (d["choice"] == cls).mean()})
        if hits:
            h = pd.DataFrame(hits)
            out.append({"model": model, "words": len(h), "concept_route": h["concept_route"].mean(), "direct": h["direct"].mean()})
    return pd.DataFrame(out)


def plot_concepts(summary, path):
    """Per model and concept: how much further sharp items land on the sharp side than round items, for words and shapes."""
    if summary.empty:
        return
    models = sorted(summary["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.6), sharey=True, squeeze=False)
    x = np.arange(len(CONCEPT_DIMS))
    for ax, model in zip(axes[0], models):
        sm = summary[summary["model"] == model].set_index(["stimulus", "word_class", "dimension"])["mean"]
        for k, (stim, label, color) in enumerate([("word", "made-up words", "#6a51a3"), ("image", "shapes", "#9e9ac8")]):
            gap = [sm.get((stim, "sharp", d), np.nan) - sm.get((stim, "round", d), np.nan) for d in CONCEPT_DIMS]
            ax.bar(x + (k - 0.5) * 0.38, np.array(gap) * 100, 0.38, color=color, label=label)
        ax.axhline(0, c="k", lw=0.8)
        ax.set_xticks(x, [CONCEPT_LABELS[d].replace(" → ", "\n→ ") for d in CONCEPT_DIMS], fontsize=8)
        ax.set_ylim(-20, 100)
        ax.set_title(model.split(":", 1)[1])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Sharp items minus round items\n(percentage points toward the sharp side)")
    axes[0][-1].legend(fontsize=8, loc="upper left", frameon=False)
    fig.suptitle("Do sharp-sounding words and sharp shapes push the same concepts the same way?")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


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


def _external_ok():
    """True when the external datasets are downloaded, so human-comparison tables can be built."""
    return (ROOT / "data" / "external" / "sospass").exists() and (ROOT / "data" / "external" / "sidhu2022").exists()


def summarise_rec_match(df):
    """Per model, word set and rendering: congruent rate on scorable words, and r(P(sharp), human pointedness) for SoS PASS."""
    m = df[(df["task"] == "rec_match") & df["choice"].notna()].copy()
    if m.empty:
        return pd.DataFrame()
    m["c"] = m["choice"] == m["word_class"]
    out = []
    for (model, ws, rend), d in m.groupby(["model", "word_set", "rendering"]):
        scored = d[d["cell"] != "mixed"]
        r = item_rate(scored, "c", True) or {}
        row = {"model": model, "word_set": ws, "rendering": rend, **r}
        if ws == "sospass":
            p = d.assign(s=d["choice"] == "sharp").groupby("item").agg(p_sharp=("s", "mean"), human=("human_mean", "first"))
            row["r_with_human"] = p["p_sharp"].corr(p["human"]) if p["p_sharp"].nunique() > 1 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def summarise_spelling_check(df):
    """Do answers to the recording follow answers to the model's own spelling of it? Item-level r, audio vs heard vs ipa."""
    out = []
    for task, keys in (("rec_match", ("word_set", "item")), ("sos_word", ("item",)), ("sidhu", ("scale", "item"))):
        m = df[(df["task"] == task) & df["choice"].notna() & df["rendering"].isin(["audio", "heard", "ipa"])].copy()
        if m.empty:
            continue
        m["v"] = (m["choice"] == "sharp").astype(float) if task == "rec_match" else pd.to_numeric(m["choice"], errors="coerce")
        wide = m.pivot_table(index=["model", *keys], columns="rendering", values="v", aggfunc="mean").reset_index()
        for model, w in wide.groupby("model"):
            row = {"model": model, "task": task, "items": len(w)}
            for a, b in (("audio", "heard"), ("audio", "ipa"), ("heard", "ipa")):
                if a in w and b in w and w[[a, b]].dropna().shape[0] > 2:
                    row[f"r_{a}_{b}"] = w[a].corr(w[b])
            if len(row) > 3:
                out.append(row)
    t = df[(df["task"] == "transcribe") & df["choice"].notna() & (df["word_set"] == "sidhu")]
    for model, d in t.groupby("model"):
        out.append({"model": model, "task": "transcribe (Sidhu)", "items": d["item"].nunique(),
                    "exact_spelling_rate": (d["choice"] == d["item"]).mean()})
    return pd.DataFrame(out)


def item_means(df, task, keys=("item",)):
    """Model mean rating per model and item for a rating task."""
    m = df[(df["task"] == task) & df["choice"].notna()]
    if m.empty:
        return pd.DataFrame()
    return m.assign(r=pd.to_numeric(m["choice"], errors="coerce")).groupby(["model", *keys])["r"].mean().reset_index()


def summarise_human_items(df):
    """Correlation of model and human item means for the SoS PASS words and shapes, with the human split-half ceiling."""
    if not _external_ok():
        return pd.DataFrame()
    import external as X
    from scipy.stats import pearsonr, spearmanr
    out = []
    for task, loader, ceiling in (("sos_word", X.sos_words, X.sos_words_reliability), ("sos_shape", X.sos_shapes, None)):
        sub = df[df["task"] == task].copy()
        if sub.empty:
            continue
        sub["rendering"] = sub["rendering"].replace("", "image").fillna("image")
        mm = item_means(sub, task, keys=("rendering", "item"))
        human = loader()[["item", "human_mean"]]
        ceil = ceiling() if ceiling else np.nan
        for (model, rend), g in mm.merge(human, on="item").groupby(["model", "rendering"]):
            r, p = pearsonr(g["r"], g["human_mean"])
            out.append({"model": model, "task": task, "rendering": rend, "items": len(g), "pearson_r": r, "p": p,
                        "spearman_rho": spearmanr(g["r"], g["human_mean"]).statistic, "human_split_half": ceil})
    return pd.DataFrame(out)


def summarise_sidhu(df):
    """Per model and scale: correlation across the 40 nonwords between model and human means, with the human ceiling."""
    if not _external_ok():
        return pd.DataFrame()
    import external as X
    mm = item_means(df, "sidhu", keys=("rendering", "scale", "item"))
    if mm.empty:
        return pd.DataFrame()
    _, human = X.sidhu_words()
    rel = X.sidhu_reliability()
    g = mm.merge(human, on=["item", "scale"])
    out = []
    for (model, rend, scale), d in g.groupby(["model", "rendering", "scale"]):
        r = d["r"].corr(d["human_mean"]) if d["r"].nunique() > 1 else np.nan
        out.append({"model": model, "rendering": rend, "scale": scale, "items": len(d), "pearson_r": r,
                    "human_split_half": rel[scale], "model_mean": d["r"].mean(), "human_mean": d["human_mean"].mean()})
    return pd.DataFrame(out)


def _sidhu_profiles(df, task, keys):
    """Model profiles on the 25 scales: one row per (model, *keys), one column per scale, centred within the stimulus type."""
    mm = item_means(df, task, keys=(*keys, "scale"))
    if mm.empty:
        return pd.DataFrame()
    prof = mm.pivot_table(index=["model", *keys], columns="scale", values="r")
    return prof - prof.groupby(level="model").transform("mean")


def _route_score(word_prof, word_class, shape_prof, pairs):
    """Share of (word, round shape, sharp shape) triples where the shape with the nearer profile is the human-congruent one.

    Returns per-word scores so the CI can be bootstrapped over words. Ties count as half."""
    out = {}
    for word, cls in word_class.items():
        if word not in word_prof.index:
            continue
        w = word_prof.loc[word].values
        hits = []
        for rnd, shp in pairs:
            dr = np.linalg.norm(w - shape_prof.loc[rnd].values)
            ds = np.linalg.norm(w - shape_prof.loc[shp].values)
            hits.append(0.5 if dr == ds else float((dr < ds) == (cls == "round")))
        out[word] = np.mean(hits)
    return pd.Series(out)


def summarise_sidhu_route(df):
    """Concept route on Sidhu et al.'s 25 human-rated scales.

    Words are the 40 Sidhu nonwords; the outer thirds on people's Sharp-Round ratings are the sharp and round words.
    Shapes are the classic pairs (round vs sharp member) and the SoS PASS shapes (outer thirds on people's
    rounded/pointed means, every round x pointed pair). A word goes with the shape whose 25-scale profile is nearer,
    each scale centred within words and within shapes. 'model words' uses the model's own word profiles;
    'human words' uses people's word profiles with the model's shape profiles."""
    if not _external_ok():
        return pd.DataFrame()
    import external as X
    shapes = _sidhu_profiles(df, "sidhu_shape", ("shape_set", "item"))
    if shapes.empty:
        return pd.DataFrame()
    words = _sidhu_profiles(df[df["rendering"] == "spelled"], "sidhu", ("item",))
    _, human = X.sidhu_words()
    hprof = human.pivot_table(index="item", columns="scale", values="human_mean")
    hprof = hprof - hprof.mean()
    sr = hprof["Sharp-Round"]
    lo, hi = sr.quantile([1 / 3, 2 / 3])
    word_class = {w: "sharp" for w in sr[sr <= lo].index} | {w: "round" for w in sr[sr >= hi].index}
    sos = X.sos_shapes().set_index("item")["human_mean"]
    slo, shi = sos.quantile([1 / 3, 2 / 3])
    classic = pd.read_csv(ROOT / "stimuli" / "shapes.csv").query("set == 'classic'")
    shape_pairs = {
        "classic": [(("classic", g.set_index("role").loc["round", "shape_id"]), ("classic", g.set_index("role").loc["sharp", "shape_id"]))
                    for _, g in classic.groupby("pair")],
        "sospass": [(("sospass", r), ("sospass", s)) for r in sos[sos <= slo].index for s in sos[sos >= shi].index],
    }
    out = []
    for model, sp in shapes.groupby(level="model"):
        sp = sp.droplevel("model")
        for label, wp in (("model words", words.xs(model, level="model") if model in words.index.get_level_values(0) else None),
                          ("human words", hprof)):
            if wp is None:
                continue
            cols = sp.columns.intersection(wp.columns)
            for set_name, pairs in shape_pairs.items():
                pairs = [p for p in pairs if p[0] in sp.index and p[1] in sp.index]
                if not pairs:
                    continue
                s = _route_score(wp[cols], word_class, sp[cols], pairs)
                boot = np.random.default_rng(0).choice(s.values, (2000, len(s))).mean(1)
                out.append({"model": model, "word_profiles": label, "shapes": set_name, "words": len(s),
                            "shape_pairs": len(pairs), "concept_route": s.mean(),
                            "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5)})
    return pd.DataFrame(out)


def summarise_sidhu_shapes(df):
    """Check that the model's shape ratings mean something: its Sharp-Round rating of SoS shapes vs people's pointedness."""
    if not _external_ok():
        return pd.DataFrame()
    import external as X
    from scipy.stats import spearmanr
    mm = item_means(df[df["shape_set"] == "sospass"], "sidhu_shape", keys=("item", "scale"))
    if mm.empty:
        return pd.DataFrame()
    g = mm[mm["scale"] == "Sharp-Round"].merge(X.sos_shapes()[["item", "human_mean"]], on="item")
    return pd.DataFrame([{"model": m, "items": len(d), "spearman_rho_pointed": spearmanr(8 - d["r"], d["human_mean"]).statistic}
                         for m, d in g.groupby("model")])


def plot_human_items(df, path):
    """Scatter of human vs model item means: SoS pseudowords and shapes, one row per model."""
    if not _external_ok():
        return
    import external as X
    df = df[(df["task"] != "sos_word") | (df["rendering"] == "ipa")]
    panels = [("sos_word", X.sos_words, "Pseudowords (SoS PASS, IPA)"), ("sos_shape", X.sos_shapes, "Shapes (SoS PASS)")]
    data = {t: item_means(df, t).merge(loader()[["item", "human_mean"]], on="item") for t, loader, _ in panels
            if not item_means(df, t).empty}
    if not data:
        return
    models = sorted(set().union(*[set(d["model"]) for d in data.values()]))
    fig, axes = plt.subplots(len(models), len(panels), figsize=(5 * len(panels), 4.4 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        for j, (task, _, title) in enumerate(panels):
            ax = axes[i][j]
            d = data.get(task, pd.DataFrame())
            d = d[d["model"] == model] if not d.empty else d
            if d.empty:
                ax.set_visible(False)
                continue
            ax.scatter(d["human_mean"], d["r"], s=10 if task == "sos_word" else 22, alpha=0.5, color="#6a51a3")
            ax.plot([1, 7], [1, 7], c="lightgray", ls="--", lw=1)
            r = d["r"].corr(d["human_mean"])
            ax.set_title(f"{title}\n{model.split(':', 1)[1]}: r = {r:.2f}", fontsize=10)
            ax.set_xlim(1, 7)
            ax.set_ylim(0.8, 7.2)
            ax.set_xlabel("People: mean rating (1 rounded, 7 pointed)")
            ax.set_ylabel("Model: mean rating")
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


def plot_sidhu(t, path):
    """Per scale, how well each model's nonword ratings track people's, against the human split-half ceiling."""
    if t.empty:
        return
    t = t[t["rendering"] == "spelled"]
    order = t.groupby("scale")["human_split_half"].first().sort_values().index
    fig, ax = plt.subplots(figsize=(7.5, 8))
    y = np.arange(len(order))
    ax.scatter(t.groupby("scale")["human_split_half"].first().reindex(order), y, marker="|", s=200, c="k",
               label="People vs people (split-half ceiling)", zorder=4)
    for k, (model, g) in enumerate(t.groupby("model")):
        g = g.set_index("scale").reindex(order)
        ax.barh(y + (k - 0.5) * 0.38, g["pearson_r"], 0.38, label=model.split(":", 1)[1])
    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks(y, [s.replace("-", " – ") for s in order], fontsize=8)
    ax.set_xlim(-0.6, 1)
    ax.set_xlabel("Correlation with people across the 40 nonwords")
    ax.set_title("Do models rate made-up words like people do?\n(Sidhu, Vigliocco & Pexman, 2022; 25 scales)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.07), ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
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
        "summary_rating": summarise_rating(df), "summary_concept": summarise_concept(df),
        "summary_concept_route": summarise_concept_route(df),
        "summary_classic": summarise_classic(df), "summary_graded": summarise_graded(df),
        "summary_shape_rating": summarise_shape_rating(df),
        "summary_human_items": summarise_human_items(df), "summary_sidhu": summarise_sidhu(df),
        "summary_rec_match": summarise_rec_match(df), "summary_spelling_check": summarise_spelling_check(df),
        "summary_sidhu_route": summarise_sidhu_route(df), "summary_sidhu_shapes": summarise_sidhu_shapes(df),
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
    plot_concepts(tables["summary_concept"], RES / "figures" / "concepts.png")
    plot_human_items(df, RES / "figures" / "human_items.png")
    plot_sidhu(tables["summary_sidhu"], RES / "figures" / "sidhu_scales.png")
    print(f"\nChart: {RES / 'figures' / 'main.png'}")


if __name__ == "__main__":
    main()
