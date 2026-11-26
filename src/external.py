"""Load the external human datasets downloaded by fetch_external.py, as items with human means.

  audio16k()     Recordings are converted once to 16 kHz mono WAV under data/external/audio16k, and the
                 loaders return those paths, so every audio model hears the same files.
  sos_words()    SoS PASS (McCormick et al., 2015): recorded CVCV pseudowords with IPA and the mean of
                 52 raters on one scale, 1 = rounded .. 7 = pointed ("very rounded" ratings are flipped).
                 Only the 475 rated items that have a recording are kept, so text and audio tests match.
  sos_shapes()   SoS PASS shapes: 90 outlines with mean ratings on the same 1..7 scale. The BMP files are
                 converted once to PNG next to the originals, since not every model API takes BMP.
  sidhu_words()  Sidhu, Vigliocco & Pexman (2022): 40 recorded nonwords, mean ratings on 25 seven-point
                 scales; for a scale named "A-B", 1 = A and 7 = B.

Each loader also reports split-half reliability of the human means (random halves of raters,
Spearman-Brown corrected), the ceiling a model's correlation with people can be compared to.
"""

from functools import lru_cache
from pathlib import Path

import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"
IPA_FIX = {"I": "ɪ"}
AUDIO16K = EXT / "audio16k"


def _require(path):
    """Fail with a helpful message when the external data hasn't been downloaded."""
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: uv run python src/fetch_external.py")
    return path


def audio16k(path):
    """Convert a recording once to 16 kHz mono WAV (the input format Gemma 4 expects) and return its repo path.

    Every audio model gets the converted file, so all providers hear the same signal. Needs ffmpeg.
    """
    src = Path(path)
    out = AUDIO16K / src.parent.parent.name / f"{src.stem}.wav"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(out)],
                       check=True)
    return str(out.relative_to(ROOT))


def split_half(trials, item, rater, value, n_iter=200, seed=0):
    """Mean Spearman-Brown corrected correlation between item means from random halves of raters."""
    rng = np.random.default_rng(seed)
    raters = trials[rater].unique()
    rs = []
    for _ in range(n_iter):
        half = set(rng.choice(raters, len(raters) // 2, replace=False))
        in_half = trials[rater].isin(half)
        a = trials[in_half].groupby(item)[value].mean()
        b = trials[~in_half].groupby(item)[value].mean()
        r = a.corr(b.reindex(a.index))
        rs.append(2 * r / (1 + r))
    return float(np.nanmean(rs))


@lru_cache(maxsize=None)
def _sos_trials():
    """Useable SoS PASS trials with ratings on the 1 = rounded .. 7 = pointed scale (spreadsheet cached as CSV)."""
    cache = EXT / "sospass" / "trials_cache.csv"
    if cache.exists():
        p = pd.read_csv(cache)
    else:
        p = pd.read_excel(_require(EXT / "sospass" / "SoS Pseudoword Database.xlsx"))
        p.to_csv(cache, index=False)
    p = p[p["useableTrial"] == 1].copy()
    p["pointed"] = np.where(p["RightOption"] == "very rounded", 8 - p["LikertRESPONSE"], p["LikertRESPONSE"])
    return p


def sos_words():
    """SoS PASS pseudowords with a recording: sound file, IPA, features, human mean and audio path."""
    p = _sos_trials()
    audio = EXT / "sospass" / "Pseudowords"
    have = {f.stem for f in _require(audio).glob("*.wav")}
    p = p[p["SoundFile"].isin(have)]
    cols = ["Position1_C1", "Position2_V1", "Position3_C2", "Position4_V2"]
    items = p.drop_duplicates("SoundFile")[["SoundFile", *cols, "ConsonantSonorObstru", "ConsonantManner",
                                             "ConsonantVoicing", "VowelRoundedness", "VowelBackness"]].copy()
    items["ipa"] = items[cols].apply(lambda r: "".join(IPA_FIX.get(s, s) for s in r), axis=1)
    means = p.groupby("SoundFile")["pointed"].agg(human_mean="mean", human_n="count")
    items = items.merge(means, left_on="SoundFile", right_index=True)
    items["audio"] = items["SoundFile"].map(lambda s: audio16k(audio / f"{s}.wav"))
    return items.rename(columns={"SoundFile": "item"}).reset_index(drop=True)


def sos_words_reliability():
    """Split-half reliability of the SoS PASS pseudoword means."""
    p = _sos_trials()
    return split_half(p[p["SoundFile"].isin(set(sos_words()["item"]))], "SoundFile", "Subject", "pointed")


def sos_shapes():
    """SoS PASS shapes with protuberance features, human mean and a PNG path."""
    s = pd.read_excel(_require(EXT / "sospass" / "SoS Shape Stimuli.xlsx"))
    s = s[s["Filename"].notna()].rename(columns={"Mean rating recoded to single scale": "human_mean",
                                                 "Proturberance #": "protuberances", "Protuberance Depth": "depth",
                                                 "Protuberance Contour": "contour"})
    s = s[["Filename", "protuberances", "depth", "contour", "human_mean"]].copy()
    src = {f.stem.upper(): f for f in (EXT / "sospass" / "Shapes").iterdir()}
    png_dir = EXT / "sospass" / "png"
    png_dir.mkdir(exist_ok=True)
    paths = []
    for name in s["Filename"]:
        out = png_dir / f"{name}.png"
        if not out.exists():
            from PIL import Image
            Image.open(src[str(name).upper()]).convert("RGB").save(out)
        paths.append(str(out.relative_to(ROOT)))
    s["file"] = paths
    s["human_mean"] = s["human_mean"].astype(float)
    return s.rename(columns={"Filename": "item"}).reset_index(drop=True)


@lru_cache(maxsize=None)
def _sidhu_trials():
    """Sidhu et al. trial-level nonword ratings."""
    return pd.read_csv(_require(EXT / "sidhu2022" / "Data" / "AllData_Long.csv"), encoding="utf-8-sig")


def sidhu_scales():
    """The 25 scales as (name, first pole, second pole), poles lower-cased."""
    names = sorted(_sidhu_trials()["Scale_Nice"].unique())
    return [(n, *[p.lower().replace("nonsolid", "non-solid") for p in n.split("-")]) for n in names]


def sidhu_words():
    """The 40 nonwords with IPA, consonant class and audio path, plus a long table of human means per scale."""
    d = _sidhu_trials()
    props = pd.read_csv(_require(EXT / "sidhu2022" / "Data" / "Item Properties.csv"), encoding="utf-8-sig")
    props = props.rename(columns={"Row Labels": "item", "IPA": "ipa", "Consonant": "consonant_class"})
    audio_dir = EXT / "sidhu2022" / "Audio Stimuli"
    audio = {f.stem: audio16k(f) for f in audio_dir.iterdir() if f.suffix.lower() in (".wav", ".mp3")}
    props["audio"] = props["item"].map(audio)
    means = d.groupby(["Word", "Scale_Nice"])["Rating"].agg(human_mean="mean", human_n="count").reset_index()
    means = means.rename(columns={"Word": "item", "Scale_Nice": "scale"})
    return props[["item", "ipa", "consonant_class", "audio"]], means


def sidhu_reliability():
    """Split-half reliability of the nonword means for each of the 25 scales."""
    d = _sidhu_trials()
    return {s: split_half(g, "Word", "PPT", "Rating", n_iter=100) for s, g in d.groupby("Scale_Nice")}
