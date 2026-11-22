"""Control for the image condition: can the model tell which image in each classic pair is spiky?

If a model fails the image matching task but passes this check, it sees the difference between the
shapes and keeps "shape A" and "shape B" straight; it just doesn't link the made-up word to it.

Usage: uv run python src/check_perception.py --model ollama:gemma3:27b [--samples 4]
"""

import argparse
import csv
from pathlib import Path

import tasks as T
from models import load_model

ROOT = Path(__file__).resolve().parent.parent
PROMPT = ("The first image is shape A and the second image is shape B. "
          "Which shape has spiky, pointed edges? Answer with a single letter, A or B.")


def main():
    """Ask the spikiness question for every anchor pair in both orders and append the score to a CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()
    model = load_model(args.model)
    correct = total = 0
    for _, pair in T.classic_pairs():
        rnd, shp = str(ROOT / pair["round"]["file"]), str(ROOT / pair["sharp"]["file"])
        for images, spiky in (([rnd, shp], "B"), ([shp, rnd], "A")):
            for _ in range(args.samples):
                answer = model.generate(PROMPT, images, 1.0, 8).strip()[:1].upper()
                correct += answer == spiky
                total += 1
    print(f"{args.model}: picked the spiky shape {correct}/{total}")
    out = ROOT / "results" / "perception_check.csv"
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["model", "shape_set", "correct", "total"])
        w.writerow([args.model, "classic", correct, total])


if __name__ == "__main__":
    main()
