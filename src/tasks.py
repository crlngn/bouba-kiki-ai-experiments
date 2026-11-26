"""Build the phase 1 trials from the stimulus files.

Tasks:
  match      Two-choice: which of two shapes (text descriptions or images) has this name?
  rating     1-7 roundness/sharpness rating of a name (scale direction counterbalanced).
  concept    Concept profiles: the same questions about each word and each picture. Five two-choice
             sensory concepts (calm/excited, vanilla/lemon, sweet/sour, soft hum/harsh beep, soft/hard
             to touch) plus the PerForm survey question "How sharp / hard does it feel?" (1-5).
             Words: fresh round/sharp words and famous words. Pictures: classic pairs and thesis shapes.
  classic    Pitch/loudness with brightness and size (Marks 1974; Mondloch 2004). No pseudowords.
  graded     One radial-frequency shape, two names: which name fits? English and Mandarin prompts.
             Uses the Chen et al. (2016) grid by default, or the "mixed" round-and-spiky set.
  shape_rating  How sharp / hard does this shape feel (1-5)? The six PerForm thesis shapes, so model
             ratings can be compared with the 24 thesis participants.
  screen     Does this word already mean something? Used to drop words with a prior meaning.

External human datasets (src/external.py; download first with src/fetch_external.py):
  sos_word   How rounded or pointed does this pseudoword sound (1-7)? 475 SoS PASS pseudowords, in IPA.
  sos_shape  How rounded or pointed does this shape look (1-7)? The 90 SoS PASS shapes.
  sidhu      Where does this nonword fall between two adjectives (1-7)? Sidhu et al.'s 40 nonwords on
             their 25 scales, with the same instruction to rate the impression the word gives.
All three use the human scale direction, with the poles swapped on alternate trials.

Recorded words (the SoS PASS and Sidhu recordings, so models can hear what people heard):
  rec_match  Two-choice shape matching for the recorded words. Sidhu words are scored by consonant
             class (sonorants round, voiceless stops sharp, the rest "mixed"); SoS PASS words carry
             people's mean pointedness rating, so analysis can correlate P(sharp) with it.
  transcribe Spell the word in the recording with English letters. The spelling check: if a model's
             answers to the audio follow its own spelling, it may be matching letters, not sound.
Renderings for recorded words: spelled (Sidhu only), ipa, audio (the recording, word never written),
and heard (the model's own most common transcription, from a transcribe log passed as transcripts).
sidhu and sos_word take the same renderings.

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
RECORDED_RENDERINGS = ["spelled", "ipa", "audio", "heard"]
SIDHU_CLASS = {"son": "round", "vls": "sharp"}

SHAPE_TEXT = {
    "round": "a smooth, rounded, blob-like outline with gentle curves",
    "sharp": "a jagged, spiky, star-like outline with sharp points",
}

CONCEPTS = {
    "emotion": {"round": "a calm feeling", "sharp": "an excited feeling"},
    "scent": {"round": "the smell of vanilla", "sharp": "the smell of lemon"},
    "taste": {"round": "a sweet taste", "sharp": "a sour taste"},
    "sound": {"round": "a soft hum", "sharp": "a harsh beep"},
    "touch": {"round": "something soft to touch", "sharp": "something hard to touch"},
}
CONCEPT_RATING = "sharp_hard"

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


def _concept_items(stimulus, subject, fields, key):
    """Two-choice concept trials and the 1-5 sharp/hard rating for one word or picture."""
    trials = []
    for dim, pair in CONCEPTS.items():
        for order in (0, 1):
            first, second = ("round", "sharp") if order == 0 else ("sharp", "round")
            trials.append({**fields, "task": "concept", "stimulus": stimulus, "dimension": dim, "order": order,
                           "lang": "en", "options": {"A": first, "B": second}, "answer_type": "letter",
                           "prompt": (f"{subject} Which of these does it go with better?\nA: {pair[first]}\n"
                                      f"B: {pair[second]}\n\nAnswer with a single letter, A or B."),
                           "trial_id": f"concept|{stimulus}|{dim}|{key}|{order}"})
    for order in (0, 1):
        lo, hi = ("not at all", "very much") if order == 0 else ("very much", "not at all")
        trials.append({**fields, "task": "concept", "stimulus": stimulus, "dimension": CONCEPT_RATING, "order": order,
                       "lang": "en", "options": {"1": lo, "5": hi}, "answer_type": "number", "scale_max": 5,
                       "prompt": (f"{subject} How sharp / hard does it feel? "
                                  f"Answer with a number from 1 ({lo}) to 5 ({hi})."),
                       "trial_id": f"concept|{stimulus}|{CONCEPT_RATING}|{key}|{order}"})
    return trials


def concept_trials(words, renderings):
    """Concept profile trials for pure-cell words and for the classic and thesis pictures."""
    trials = []
    for w in words:
        for rendering in renderings:
            phrase = word_phrase(w, rendering)
            if phrase is None:
                continue
            fields = {"rendering": rendering, "shape_mode": "", "images": [], "shape_set": "", "shape_id": "",
                      **_word_fields(w)}
            trials += _concept_items("word", f"Here is a made-up word: {phrase}.", fields, f"{rendering}|{w['word']}")
    for s in load_shapes("classic") + load_shapes("thesis"):
        word_class = s["role"] if s["set"] == "classic" else ""
        fields = {"rendering": "", "shape_mode": "image", "images": [s["file"]], "shape_set": s["set"],
                  "shape_id": s["shape_id"], "shape_pair": s["pair"], "word": "", "word_set": "",
                  "word_class": word_class, "cell": ""}
        trials += _concept_items("image", "Here is a shape.", fields, s["shape_id"])
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


def load_transcripts(path):
    """Most common spelling per recorded item from a transcribe log, keyed by 'word_set|item'."""
    import json
    from collections import Counter
    counts = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("task") == "transcribe" and r.get("parsed"):
            counts.setdefault(f"{r['word_set']}|{r['item']}", Counter())[r["parsed"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in counts.items()}


def recorded_words():
    """The recorded SoS PASS and Sidhu words as dicts: item, word_set, spelling, ipa, audio, class fields."""
    import external as X
    words = []
    for _, w in X.sos_words().iterrows():
        words.append({"item": w["item"], "word_set": "sospass", "spelled": "", "ipa": w["ipa"], "audio": w["audio"],
                      "word_class": "round" if w["human_mean"] < 4 else "sharp", "cell": "",
                      "human_mean": float(w["human_mean"])})
    items, _ = X.sidhu_words()
    for _, w in items.iterrows():
        cls = SIDHU_CLASS.get(w["consonant_class"])
        words.append({"item": w["item"], "word_set": "sidhu", "spelled": w["item"], "ipa": w["ipa"], "audio": w["audio"],
                      "word_class": cls or w["consonant_class"], "cell": "pure" if cls else "mixed",
                      "consonant_class": w["consonant_class"]})
    return words


def recorded_phrase(w, rendering, transcripts=None):
    """(intro sentence, name phrase, audio list) for a recorded word, or None if the rendering isn't available."""
    if rendering == "audio":
        return "The recording is a made-up word.", "the word spoken in the recording", [w["audio"]]
    if rendering == "heard":
        value = (transcripts or {}).get(f"{w['word_set']}|{w['item']}")
    elif rendering == "spelled":
        value = w["spelled"]
    elif rendering == "ipa":
        value = w["ipa"]
    else:
        raise ValueError(rendering)
    if not value:
        return None
    shown = f"/{value}/ (written in IPA)" if rendering == "ipa" else f"“{value}”"
    return f"Here is a made-up word: {shown}.", shown, []


def rec_match_trials(renderings, shape_mode, transcripts=None):
    """Two-choice shape matching for the recorded words, in any recorded rendering."""
    pairs = classic_pairs()
    trials = []
    for w in recorded_words():
        for rendering in renderings:
            shown = recorded_phrase(w, rendering, transcripts)
            if shown is None:
                continue
            _, phrase, audio = shown
            for order in (0, 1):
                first, second = ("round", "sharp") if order == 0 else ("sharp", "round")
                t = {"task": "rec_match", "item": w["item"], "word": w["item"], "word_set": w["word_set"],
                     "word_class": w["word_class"], "cell": w["cell"], "human_mean": w.get("human_mean"),
                     "consonant_class": w.get("consonant_class", ""), "rendering": rendering,
                     "shape_mode": shape_mode, "shape_set": "classic" if shape_mode == "image" else "",
                     "order": order, "lang": "en", "options": {"A": first, "B": second}, "answer_type": "letter",
                     "audio": audio}
                ask = (f"In a made-up language, one of these shapes is called {phrase}. "
                       "Which shape is it? Answer with a single letter, A or B.")
                if shape_mode == "text":
                    t["images"] = []
                    t["prompt"] = f"Here are two shapes.\nA: {SHAPE_TEXT[first]}\nB: {SHAPE_TEXT[second]}\n\n{ask}"
                else:
                    name, imgs = pairs[zlib.crc32((w["item"] + rendering).encode()) % len(pairs)]
                    t["images"] = [imgs[first]["file"], imgs[second]["file"]]
                    t["shape_pair"] = name
                    t["prompt"] = f"The first image is shape A and the second image is shape B. {ask}"
                t["trial_id"] = f"rec_match|{shape_mode}|{rendering}|{w['word_set']}|{w['item']}|{order}"
                trials.append(t)
    return trials


def transcribe_trials():
    """Ask for an English spelling of each recording."""
    return [{"task": "transcribe", "item": w["item"], "word": w["item"], "word_set": w["word_set"],
             "word_class": w["word_class"], "cell": w["cell"], "rendering": "audio", "shape_mode": "", "order": 0,
             "lang": "en", "options": {}, "answer_type": "spelling", "images": [], "audio": [w["audio"]],
             "prompt": ("The recording is a made-up word. Spell it with ordinary English letters, the way an English "
                        "speaker would write it. Answer with the spelling only."),
             "trial_id": f"transcribe|{w['word_set']}|{w['item']}"} for w in recorded_words()]


def _rating(prompt_fn, lo, hi, order, scale_max=7):
    """Prompt text and options for a rating whose poles swap on order 1."""
    a, b = (lo, hi) if order == 0 else (hi, lo)
    return prompt_fn(a, b), {"1": a, str(scale_max): b}


def sos_word_trials(renderings=("ipa",), transcripts=None):
    """SoS PASS pseudowords rated 1 (very rounded) to 7 (very pointed), in IPA, as audio or as the model's spelling."""
    trials = []
    for w in [w for w in recorded_words() if w["word_set"] == "sospass"]:
        for rendering in renderings:
            shown = recorded_phrase(w, rendering, transcripts)
            if shown is None:
                continue
            intro, _, audio = shown
            if rendering == "ipa":
                intro = f"Here is a made-up word, written in IPA: /{w['ipa']}/."
            for order in (0, 1):
                prompt, options = _rating(lambda a, b: (f"{intro} How rounded or pointed does it sound? "
                                                        f"Answer with a number from 1 ({a}) to 7 ({b})."),
                                          "very rounded", "very pointed", order)
                key = "" if rendering == "ipa" else f"{rendering}|"
                trials.append({"task": "sos_word", "item": w["item"], "ipa": w["ipa"], "rendering": rendering,
                               "shape_mode": "", "order": order, "lang": "en", "options": options,
                               "answer_type": "number", "scale_max": 7, "images": [], "audio": audio,
                               "word": w["ipa"], "word_set": "sospass", "word_class": "", "cell": "",
                               "prompt": prompt, "trial_id": f"sos_word|{key}{w['item']}|{order}"})
    return trials


def sos_shape_trials():
    """SoS PASS shapes, rated 1 (very rounded) to 7 (very pointed)."""
    import external as X
    trials = []
    for _, sh in X.sos_shapes().iterrows():
        for order in (0, 1):
            prompt, options = _rating(lambda a, b: ("How rounded or pointed does this shape look? "
                                                    f"Answer with a number from 1 ({a}) to 7 ({b})."),
                                      "very rounded", "very pointed", order)
            trials.append({"task": "sos_shape", "item": sh["item"], "shape_set": "sospass", "rendering": "",
                           "shape_mode": "image", "order": order, "lang": "en", "options": options,
                           "answer_type": "number", "scale_max": 7, "images": [sh["file"]], "word": "", "word_set": "",
                           "word_class": "", "cell": "", "prompt": prompt, "trial_id": f"sos_shape|{sh['item']}|{order}"})
    return trials


def sidhu_trials(renderings=("spelled",), transcripts=None):
    """Sidhu et al.'s nonwords on their 25 scales; 1 = first adjective of the scale name, 7 = second."""
    import external as X
    trials = []
    for w in [w for w in recorded_words() if w["word_set"] == "sidhu"]:
        for rendering in renderings:
            shown = recorded_phrase(w, rendering, transcripts)
            if shown is None:
                continue
            intro, _, audio = shown
            for scale, lo, hi in X.sidhu_scales():
                for order in (0, 1):
                    prompt, options = _rating(lambda a, b: (f"{intro} Even though it doesn't mean anything, rate the "
                                                            f"impression it gives on a scale from 1 ({a}) to 7 ({b}). "
                                                            "Answer with a single number."),
                                              lo, hi, order)
                    trials.append({"task": "sidhu", "item": w["item"], "scale": scale, "rendering": rendering,
                                   "shape_mode": "", "order": order, "lang": "en", "options": options,
                                   "answer_type": "number", "scale_max": 7, "images": [], "audio": audio,
                                   "word": w["item"], "word_set": "sidhu", "word_class": w["consonant_class"],
                                   "cell": "", "prompt": prompt,
                                   "trial_id": f"sidhu|{rendering}|{scale}|{w['item']}|{order}"})
    return trials


def sidhu_shape_trials():
    """The SoS PASS shapes and the classic pairs on Sidhu et al.'s 25 scales, so shapes get the same profile as words."""
    import external as X
    shapes = [("sospass", sh["item"], sh["file"]) for _, sh in X.sos_shapes().iterrows()]
    shapes += [("classic", s["shape_id"], s["file"]) for s in load_shapes("classic")]
    trials = []
    for shape_set, item, file in shapes:
        for scale, lo, hi in X.sidhu_scales():
            for order in (0, 1):
                prompt, options = _rating(lambda a, b: ("Here is a shape. Rate the impression it gives on a scale from "
                                                        f"1 ({a}) to 7 ({b}). Answer with a single number."), lo, hi, order)
                trials.append({"task": "sidhu_shape", "item": item, "scale": scale, "shape_set": shape_set,
                               "rendering": "", "shape_mode": "image", "order": order, "lang": "en",
                               "options": options, "answer_type": "number", "scale_max": 7, "images": [file],
                               "word": "", "word_set": "", "word_class": "", "cell": "", "prompt": prompt,
                               "trial_id": f"sidhu_shape|{shape_set}|{scale}|{item}|{order}"})
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


def build(task, renderings=RENDERINGS, shape_mode="text", include_dropped=False, shape_set="chen", transcripts=None):
    """Return the trials for one task. shape_set picks the graded task's shapes (chen or mixed);
    transcripts (from load_transcripts) supplies the 'heard' rendering of recorded words."""
    words = load_words(include_dropped=include_dropped or task == "screen")
    if task == "match":
        return match_trials(words, renderings, shape_mode)
    if task == "rating":
        return rating_trials(words, renderings)
    if task == "concept":
        return concept_trials(words, renderings)
    if task == "classic":
        return classic_trials()
    if task == "graded":
        return graded_trials(words, load_shapes(shape_set), shape_set)
    if task == "shape_rating":
        return shape_rating_trials()
    recorded = [r for r in renderings if r in RECORDED_RENDERINGS]
    if task == "sos_word":
        return sos_word_trials(recorded or ["ipa"], transcripts)
    if task == "sos_shape":
        return sos_shape_trials()
    if task == "sidhu":
        return sidhu_trials(recorded or ["spelled"], transcripts)
    if task == "rec_match":
        return rec_match_trials(recorded or ["ipa", "audio"], shape_mode, transcripts)
    if task == "transcribe":
        return transcribe_trials()
    if task == "sidhu_shape":
        return sidhu_shape_trials()
    if task == "screen":
        return screen_trials(words)
    raise SystemExit(f"Unknown task {task!r}")
