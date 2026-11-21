"""Generate the phase 1 stimuli: pseudowords in several renderings and radial-frequency shapes.

Pseudowords follow a 2x2 design (round or sharp consonants x round or sharp vowels), built from
CV syllables at random with a fixed seed. Each word gets five renderings that remove letter-shape
cues to different degrees: spelled, IPA, sounds-like, articulatory description (no letters at all)
and Japanese katakana. Famous words (bouba/kiki, takete/maluma) form a separate "memorized" set.

Shapes are radial frequency patterns (Chen, Huang, Woods & Spence, 2016): r(t) = r0 * (1 + A * g(t)),
where g is a sinusoid of frequency F whose peaks are sharpened by raising it to the power k
(spikiness). Every outline is scaled to the same area.

Usage: uv run python src/make_stimuli.py [--seed 20261121] [--per-cell 50]
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
STIM = ROOT / "stimuli"

CONSONANTS = {"round": ["m", "b", "l", "n", "w"], "sharp": ["k", "t", "p", "z", "s"]}
VOWELS = {"round": ["o", "u"], "sharp": ["i", "e"]}

SOUNDS_LIKE_VOWEL = {"o": "oh", "u": "oo", "i": "ee", "e": "eh"}

ARTICULATORY_CONSONANT = {
    "m": "a voiced nasal made with both lips",
    "b": "a voiced stop made with both lips",
    "l": "a voiced lateral made with the tongue tip",
    "n": "a voiced nasal made with the tongue tip",
    "w": "a voiced glide made with both lips",
    "k": "a voiceless stop made with the back of the tongue",
    "t": "a voiceless stop made with the tongue tip",
    "p": "a voiceless stop made with both lips",
    "z": "a voiced fricative made with the tongue tip",
    "s": "a voiceless fricative made with the tongue tip",
}
ARTICULATORY_VOWEL = {
    "o": "a mid back vowel",
    "u": "a high back vowel",
    "i": "a high front vowel",
    "e": "a mid front vowel",
}

KATAKANA = {
    "mo": "モ", "mu": "ム", "mi": "ミ", "me": "メ",
    "bo": "ボ", "bu": "ブ", "bi": "ビ", "be": "ベ",
    "lo": "ロ", "lu": "ル", "li": "リ", "le": "レ",
    "no": "ノ", "nu": "ヌ", "ni": "ニ", "ne": "ネ",
    "wo": "ウォ", "wu": "ウ", "wi": "ウィ", "we": "ウェ",
    "ko": "コ", "ku": "ク", "ki": "キ", "ke": "ケ",
    "to": "ト", "tu": "トゥ", "ti": "ティ", "te": "テ",
    "po": "ポ", "pu": "プ", "pi": "ピ", "pe": "ペ",
    "zo": "ゾ", "zu": "ズ", "zi": "ズィ", "ze": "ゼ",
    "so": "ソ", "su": "ス", "si": "スィ", "se": "セ",
}

BLOCKLIST = {
    "lomo", "maluma", "takete", "bouba", "kiki", "lobo", "lobe", "mole", "mule", "memo", "nemo",
    "mono", "molo", "limo", "lime", "mile", "nile", "bile", "mine", "wine", "wise", "bike", "mike",
    "kobe", "kite", "tote", "poke", "pete", "sete", "sumo", "suki", "tiki", "piki", "pipe", "zine",
    "tomo", "toko", "tuko", "sushi", "seme", "keto", "kepi", "tipi", "pise", "sepe", "tese", "zeke",
    "bono", "bolo", "bulu", "nomu", "numo", "lumo", "wolo", "momo", "mimi", "bubu", "lulu", "nono",
    "bibi", "lili", "nini", "titi", "pepe", "sisi", "zizi", "koko", "tete", "pipi", "kuku", "tutu",
    "kimono", "tokio", "tokyo", "kosovo", "bolivia", "molokai", "numenor", "namibia", "kisumu",
}

FAMOUS = [
    {"word": "bouba", "class": "round", "source": "Ramachandran & Hubbard (2001)"},
    {"word": "kiki", "class": "sharp", "source": "Ramachandran & Hubbard (2001)"},
    {"word": "maluma", "class": "round", "source": "Köhler (1929/1947)"},
    {"word": "takete", "class": "sharp", "source": "Köhler (1929/1947)"},
]
FAMOUS_IPA = {"bouba": "/ˈbu.ba/", "kiki": "/ˈki.ki/", "maluma": "/maˈlu.ma/", "takete": "/taˈke.te/"}

FREQUENCIES = [3, 5, 7, 9, 11]
AMPLITUDES = [0.05, 0.10, 0.15, 0.20, 0.25]
SPIKINESS = [1, 2, 4, 8, 16]
IMAGE_SIZE = 384
SUPERSAMPLE = 4
TARGET_AREA_FRACTION = 0.22
ANCHOR_PAIRS = [((5, 0.15, 1), (11, 0.25, 16)), ((3, 0.20, 1), (9, 0.25, 8)), ((7, 0.10, 1), (7, 0.25, 16))]


def syllables(word):
    """Split a CV-only pseudoword into its syllables."""
    return [word[i:i + 2] for i in range(0, len(word), 2)]


def render(word):
    """Return the IPA, sounds-like, articulatory and katakana renderings of a CV pseudoword."""
    sylls = syllables(word)
    if len(sylls) == 2:
        ipa = "/ˈ" + ".".join(sylls) + "/"
    else:
        ipa = "/" + sylls[0] + ".ˈ" + ".".join(sylls[1:]) + "/"
    sounds_like = "-".join(s[0] + SOUNDS_LIKE_VOWEL[s[1]] for s in sylls)
    parts = []
    for i, s in enumerate(sylls, 1):
        parts.append(f"syllable {i}: {ARTICULATORY_CONSONANT[s[0]]} followed by {ARTICULATORY_VOWEL[s[1]]}")
    articulatory = f"{len(sylls)} syllables; " + "; ".join(parts)
    katakana = "".join(KATAKANA[s] for s in sylls)
    return {"ipa": ipa, "sounds_like": sounds_like, "articulatory": articulatory, "katakana": katakana}


def make_word(rng, c_class, v_class):
    """Draw one CVCV or CVCVCV word with no repeated adjacent syllable and no repeated consonant run."""
    while True:
        n = rng.choice([2, 3])
        sylls = [rng.choice(CONSONANTS[c_class]) + rng.choice(VOWELS[v_class]) for _ in range(n)]
        if any(a == b for a, b in zip(sylls, sylls[1:])):
            continue
        if len({s[0] for s in sylls}) < 2:
            continue
        return "".join(sylls)


def make_pseudowords(rng, per_cell):
    """Generate per_cell unique words in each of the four consonant x vowel cells, plus the famous set."""
    rows, seen = [], set(BLOCKLIST)
    for c_class in ("round", "sharp"):
        for v_class in ("round", "sharp"):
            count = 0
            while count < per_cell:
                w = make_word(rng, c_class, v_class)
                if w in seen:
                    continue
                seen.add(w)
                count += 1
                cell = "pure" if c_class == v_class else "mixed"
                rows.append({
                    "word": w, "set": "fresh", "consonant_class": c_class, "vowel_class": v_class,
                    "class": c_class if cell == "pure" else "mixed", "cell": cell,
                    "syllables": len(syllables(w)), "spelled": w, **render(w),
                    "keep": 1, "screen_notes": "",
                })
    for f in FAMOUS:
        rows.append({
            "word": f["word"], "set": "famous", "consonant_class": f["class"], "vowel_class": f["class"],
            "class": f["class"], "cell": "pure", "syllables": "", "spelled": f["word"],
            "ipa": FAMOUS_IPA[f["word"]], "sounds_like": "", "articulatory": "", "katakana": "",
            "keep": 1, "screen_notes": f["source"],
        })
    return rows


def outline(freq, amp, spike, n=2048):
    """Return unit-area-normalised (x, y) points of a radial frequency outline."""
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    g = 2 * ((1 + np.sin(freq * t)) / 2) ** spike - 1
    g = g - g.mean()
    g = g / np.abs(g).max()
    r = 1 + amp * g
    x, y = r * np.cos(t), r * np.sin(t)
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return x / math.sqrt(area), y / math.sqrt(area)


def draw_shape(freq, amp, spike, path):
    """Render one black outline-filled shape on white with matched area."""
    size = IMAGE_SIZE * SUPERSAMPLE
    x, y = outline(freq, amp, spike)
    scale = math.sqrt(TARGET_AREA_FRACTION * size * size)
    pts = list(zip(size / 2 + scale * x, size / 2 - scale * y))
    img = Image.new("L", (size, size), 255)
    ImageDraw.Draw(img).polygon(pts, fill=0)
    img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).convert("RGB").save(path)


def shape_id(freq, amp, spike):
    """Stable file stem for a shape's parameters."""
    return f"rf_f{freq:02d}_a{int(round(amp * 100)):02d}_k{spike:02d}"


def make_shapes():
    """Draw the full frequency x amplitude x spikiness grid and mark the anchor pairs."""
    out_dir = STIM / "shapes"
    out_dir.mkdir(parents=True, exist_ok=True)
    anchors = {}
    for i, (rnd, shp) in enumerate(ANCHOR_PAIRS):
        anchors[shape_id(*rnd)] = f"anchor_round_{i}"
        anchors[shape_id(*shp)] = f"anchor_sharp_{i}"
    rows = []
    for f in FREQUENCIES:
        for a in AMPLITUDES:
            for k in SPIKINESS:
                sid = shape_id(f, a, k)
                path = out_dir / f"{sid}.png"
                draw_shape(f, a, k, path)
                rows.append({"shape_id": sid, "file": str(path.relative_to(ROOT)), "frequency": f,
                             "amplitude": a, "spikiness": k, "role": anchors.get(sid, "grid")})
    return rows


def write_csv(path, rows):
    """Write a list of dicts to CSV using the first row's keys."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    """Generate pseudowords and shapes, and record the generation parameters."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--per-cell", type=int, default=50)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    STIM.mkdir(exist_ok=True)
    words = make_pseudowords(rng, args.per_cell)
    write_csv(STIM / "pseudowords.csv", words)
    shapes = make_shapes()
    write_csv(STIM / "shapes.csv", shapes)
    meta = {"seed": args.seed, "per_cell": args.per_cell, "consonants": CONSONANTS, "vowels": VOWELS,
            "frequencies": FREQUENCIES, "amplitudes": AMPLITUDES, "spikiness": SPIKINESS,
            "anchor_pairs": ANCHOR_PAIRS, "image_size": IMAGE_SIZE,
            "target_area_fraction": TARGET_AREA_FRACTION}
    (STIM / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(words)} words, {len(shapes)} shapes written to {STIM}")


if __name__ == "__main__":
    main()
