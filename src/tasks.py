"""Build the phase 1 trials from the stimulus files.

Tasks:
  match      Two-choice: which of two shapes (text descriptions or images) has this name?
  rating     1-7 roundness/sharpness rating of a name (scale direction counterbalanced).
  crossmodal Which of two things (emotion, scent, taste, sound) has this name?
  classic    Pitch/loudness with brightness and size (Marks 1974; Mondloch 2004). No pseudowords.
  graded     One radial-frequency shape, two names: which name fits? English and Mandarin prompts.
             Uses the Chen et al. (2016) grid by default, or the "mixed" round-and-spiky set.
  shape_rating  How sharp / hard does this shape feel (1-5)? The six PerForm thesis shapes, so model
             ratings can be compared with the 24 thesis participants.
  screen     Does this word already mean something? Used to drop words with a prior meaning.

Every trial records which option is the round-associated one, so analysis can score a choice as
"round" or "sharp" whatever the answer format. Answer order is swapped on alternate trials.
"""

import csv
import random
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STIM = ROOT / "stimuli"

RENDERINGS = ["spelled", "ipa", "sounds_like", "articulatory", "katakana"]

SHAPE_TEXT = {
    "round": "a smooth, rounded, blob-like outline with gentle curves",
    "sharp": "a jagged, spiky, star-like outline with sharp points",
}

CROSSMODAL = {
    "emotion": {"round": "a calm feeling", "sharp": "an excited feeling"},
    "scent": {"round": "the smell of vanilla", "sharp": "the smell of lemon"},
    "taste": {"round": "a sweet taste", "sharp": "a sour taste"},
    "sound": {"round": "a soft hum", "sharp": "a harsh beep"},
}

CLASSIC = [
    {"item": "pitch_bright", "q": "Which sound goes better with a bright light?",
     "congruent": "a high-pitched tone", "incongruent": "a low-pitched tone", "source": "Marks (1974)"},
    {"item": "pitch_dim", "q": "Which sound goes better with a dim light?",
     "congruent": "a low-pitched tone", "incongruent": "a high-pitched tone", "source": "Marks (1974)"},
    {"item": "loud_bright", "q": "Which sound goes better with a bright light?",
     "congruent": "a loud tone", "incongruent": "a quiet tone", "source": "Marks (1974)"},
    {"item": "pitch_small", "q": "Which sound would a small ball make when it bounces?",
     "congruent": "a high-pitched squeak", "incongruent": "a low-pitched thud", "source": "Mondloch (2004)"},
    {"item": "pitch_large", "q": "Which sound would a large ball make when it bounces?",
     "congruent": "a low-pitched thud", "incongruent": "a high-pitched squeak", "source": "Mondloch (2004)"},
    {"item": "pitch_white", "q": "Which sound goes better with a white ball?",
     "congruent": "a high-pitched tone", "incongruent": "a low-pitched tone", "source": "Mondloch (2004)"},
    {"item": "pitch_black", "q": "Which sound goes better with a black ball?",
     "congruent": "a low-pitched tone", "incongruent": "a high-pitched tone", "source": "Mondloch (2004)"},
]

GRADED_PROMPT = {
    "en": "This shape has a name in a made-up language. Is it called {a} or {b}? Answer with the name only.",
    "zh": "这个形状在一种虚构的语言里有一个名字。它叫 {a} 还是 {b}？只回答名字。",
}


def load_words(include_dropped=False):
    """Read pseudowords.csv, skipping words marked keep=0 unless asked."""
    with open(STIM / "pseudowords.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if include_dropped or r["keep"] == "1"]


def load_shapes(shape_set=None):
    """Read shapes.csv, optionally one set only, with numeric parameters where present."""
    with open(STIM / "shapes.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if shape_set is None or r["set"] == shape_set]
    for r in rows:
        if r["frequency"]:
            r["frequency"], r["amplitude"], r["spikiness"] = int(r["frequency"]), float(r["amplitude"]), int(r["spikiness"])
    return rows


def classic_pairs():
    """The classic round/sharp image pairs, as a list of (pair name, {role: row})."""
    pairs = {}
    for r in load_shapes("classic"):
        pairs.setdefault(r["pair"], {})[r["role"]] = r
    return sorted(pairs.items())


def word_phrase(w, rendering):
    """Present a word in one rendering, or None if that rendering isn't available for it."""
    value = w.get(rendering, "")
    if not value:
        return None
    if rendering == "spelled":
        return f"“{value}”"
    if rendering == "ipa":
        return f"{value} (written in IPA)"
    if rendering == "sounds_like":
        return f"a word pronounced “{value}”"
    if rendering == "articulatory":
        return f"a word made of these sounds, in order ({value})"
    if rendering == "katakana":
        return f"「{value}」 (written in Japanese katakana)"
    raise ValueError(rendering)


def _word_fields(w):
    """Trial fields that describe the word."""
    return {"word": w["word"], "word_set": w["set"], "word_class": w["class"], "cell": w["cell"],
            "consonant_class": w["consonant_class"], "vowel_class": w["vowel_class"]}


def match_trials(words, renderings, shape_mode):
    """Two-choice word-to-shape matching, in text mode or with a classic image pair."""
    pairs = classic_pairs()
    trials = []
    for w in words:
        for rendering in renderings:
            phrase = word_phrase(w, rendering)
            if phrase is None:
                continue
            for order in (0, 1):
                first, second = ("round", "sharp") if order == 0 else ("sharp", "round")
                t = {"task": "match", "rendering": rendering, "shape_mode": shape_mode, "order": order, "lang": "en",
                     "shape_set": "classic" if shape_mode == "image" else "",
                     "options": {"A": first, "B": second}, "answer_type": "letter", **_word_fields(w)}
                if shape_mode == "text":
                    t["prompt"] = (f"Here are two shapes.\nA: {SHAPE_TEXT[first]}\nB: {SHAPE_TEXT[second]}\n\n"
                                   f"In a made-up language, one of these shapes is called {phrase}. "
                                   "Which shape is it? Answer with a single letter, A or B.")
                    t["images"] = []
                else:
                    name, imgs = pairs[zlib.crc32((w["word"] + rendering).encode()) % len(pairs)]
                    t["images"] = [imgs[first]["file"], imgs[second]["file"]]
                    t["shape_pair"] = name
                    t["prompt"] = ("The first image is shape A and the second image is shape B. "
                                   f"In a made-up language, one of these shapes is called {phrase}. "
                                   "Which shape is it? Answer with a single letter, A or B.")
                t["trial_id"] = f"match|{shape_mode}{'-classic' if shape_mode == 'image' else ''}|{rendering}|{w['word']}|{order}"
                trials.append(t)
    return trials


def rating_trials(words, renderings):
    """1-7 rating; order 0 puts 'very round' at 1, order 1 puts 'very sharp' at 1."""
    trials = []
    for w in words:
        for rendering in renderings:
            phrase = word_phrase(w, rendering)
            if phrase is None:
                continue
            for order in (0, 1):
                lo, hi = ("very round", "very sharp") if order == 0 else ("very sharp", "very round")
                trials.append({
                    "task": "rating", "rendering": rendering, "shape_mode": "", "order": order, "lang": "en",
                    "options": {"1": lo, "7": hi}, "answer_type": "number", "scale_max": 7, "images": [],
                    **_word_fields(w),
                    "prompt": (f"In a made-up language there is a word: {phrase}. On a scale from 1 ({lo}) "
                               f"to 7 ({hi}), how does this word sound? Answer with a single number."),
                    "trial_id": f"rating|{rendering}|{w['word']}|{order}",
                })
    return trials


def crossmodal_trials(words, renderings):
    """Two-choice matching of a word to emotions, scents, tastes and sounds."""
    trials = []
    for w in words:
        for rendering in renderings:
            phrase = word_phrase(w, rendering)
            if phrase is None:
                continue
            for dim, pair in CROSSMODAL.items():
                for order in (0, 1):
                    first, second = ("round", "sharp") if order == 0 else ("sharp", "round")
                    trials.append({
                        "task": "crossmodal", "dimension": dim, "rendering": rendering, "shape_mode": "",
                        "order": order, "lang": "en", "options": {"A": first, "B": second},
                        "answer_type": "letter", "images": [], **_word_fields(w),
                        "prompt": (f"In a made-up language, {phrase} is the name for one of these two things.\n"
                                   f"A: {pair[first]}\nB: {pair[second]}\n\n"
                                   "Which one? Answer with a single letter, A or B."),
                        "trial_id": f"crossmodal|{dim}|{rendering}|{w['word']}|{order}",
                    })
    return trials


def classic_trials():
    """Pitch, loudness, brightness and size correspondences, scored as congruent or not."""
    trials = []
    for c in CLASSIC:
        for order in (0, 1):
            first, second = ("congruent", "incongruent") if order == 0 else ("incongruent", "congruent")
            trials.append({
                "task": "classic", "item": c["item"], "source": c["source"], "rendering": "", "shape_mode": "",
                "order": order, "lang": "en", "options": {"A": first, "B": second}, "answer_type": "letter",
                "images": [], "word": "", "word_set": "", "word_class": "", "cell": "",
                "prompt": f"{c['q']}\nA: {c[first]}\nB: {c[second]}\n\nAnswer with a single letter, A or B.",
                "trial_id": f"classic|{c['item']}|{order}",
            })
    return trials


def graded_trials(words, shapes, shape_set, n_pairs=4, langs=("en", "zh"), seed=0):
    """Each grid shape with a round and a sharp fresh word; scored as the sharp word being chosen."""
    rng = random.Random(seed)
    rnd = [w for w in words if w["set"] == "fresh" and w["class"] == "round"]
    shp = [w for w in words if w["set"] == "fresh" and w["class"] == "sharp"]
    pairs = list(zip(rng.sample(rnd, n_pairs), rng.sample(shp, n_pairs)))
    trials = []
    for s in shapes:
        for i, (r, k) in enumerate(pairs):
            for lang in langs:
                for order in (0, 1):
                    a, b = (r["word"], k["word"]) if order == 0 else (k["word"], r["word"])
                    trials.append({
                        "task": "graded", "shape_set": shape_set, "shape_id": s["shape_id"], "frequency": s["frequency"],
                        "amplitude": s["amplitude"], "spikiness": s["spikiness"], "word_pair": i,
                        "rendering": "spelled", "shape_mode": "image", "order": order, "lang": lang,
                        "options": {r["word"]: "round", k["word"]: "sharp"}, "answer_type": "word", "first_option": a,
                        "images": [s["file"]], "word": f"{r['word']}/{k['word']}", "word_set": "fresh",
                        "word_class": "", "cell": "pure",
                        "prompt": GRADED_PROMPT[lang].format(a=a, b=b),
                        "trial_id": f"graded|{shape_set}|{s['shape_id']}|{i}|{lang}|{order}",
                    })
    return trials


def shape_rating_trials():
    """The thesis question about each thesis shape, with the 1-5 scale direction counterbalanced."""
    trials = []
    for s in load_shapes("thesis"):
        for order in (0, 1):
            lo, hi = ("not at all", "very much") if order == 0 else ("very much", "not at all")
            trials.append({
                "task": "shape_rating", "shape_set": "thesis", "shape_id": s["shape_id"], "rendering": "",
                "shape_mode": "image", "order": order, "lang": "en", "options": {"1": lo, "5": hi},
                "answer_type": "number", "scale_max": 5, "images": [s["file"]], "word": "", "word_set": "",
                "word_class": "", "cell": "",
                "prompt": f"How sharp / hard does this shape feel? Answer with a number from 1 ({lo}) to 5 ({hi}).",
                "trial_id": f"shape_rating|{s['shape_id']}|{order}",
            })
    return trials


def screen_trials(words):
    """Ask whether each fresh word already has a meaning."""
    return [{
        "task": "screen", "rendering": "spelled", "shape_mode": "", "order": 0, "lang": "en", "images": [],
        "options": {"KNOWN": "known", "UNKNOWN": "unknown"}, "answer_type": "screen", **_word_fields(w),
        "prompt": (f"Is “{w['word']}” a real word, name, brand or place in any language? If you are confident "
                   "it is, answer KNOWN: followed by what it means. Otherwise answer UNKNOWN."),
        "trial_id": f"screen|{w['word']}",
    } for w in words if w["set"] == "fresh"]


def build(task, renderings=RENDERINGS, shape_mode="text", include_dropped=False, shape_set="chen"):
    """Return the trials for one task. shape_set picks the graded task's shapes (chen or mixed)."""
    words = load_words(include_dropped=include_dropped or task == "screen")
    if task == "match":
        return match_trials(words, renderings, shape_mode)
    if task == "rating":
        return rating_trials(words, renderings)
    if task == "crossmodal":
        return crossmodal_trials(words, renderings)
    if task == "classic":
        return classic_trials()
    if task == "graded":
        return graded_trials(words, load_shapes(shape_set), shape_set)
    if task == "shape_rating":
        return shape_rating_trials()
    if task == "screen":
        return screen_trials(words)
    raise SystemExit(f"Unknown task {task!r}")
