"""Run phase 1 behavioral tasks against one model and log every sample to JSONL.

Each trial is sampled --samples times at --temperature, giving a choice probability per item.
Runs resume: samples already in the output file are skipped, so an interrupted run can be restarted
with the same command.

Examples:
  uv run python src/run_behavioral.py --model mock:random --tasks match --limit 20
  uv run python src/run_behavioral.py --model ollama:gemma3:27b --tasks match,rating,crossmodal,classic
  uv run python src/run_behavioral.py --model anthropic:claude-sonnet-5-5 --tasks match --shape-mode image
  uv run python src/run_behavioral.py --model gemini:<model id> --tasks screen --samples 1 --temperature 0
"""

import argparse
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import tasks as T
from models import SYSTEM_PROMPT, load_model

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"


def parse(trial, text):
    """Map raw output to an option key, or None when it can't be parsed."""
    s = text.strip()
    kind = trial["answer_type"]
    if kind == "letter":
        m = re.match(r"^[\s\*\(\"'“]*([AB])\b", s, re.I) or re.search(r"\b(?:shape|answer)?\s*([AB])\b", s)
        return m.group(1).upper() if m else None
    if kind == "number":
        m = re.search(r"\b([1-7])\b", s)
        return m.group(1) if m else None
    if kind == "word":
        low = s.lower()
        hits = [(low.find(w.lower()), w) for w in trial["options"] if w.lower() in low]
        return min(hits)[1] if hits else None
    if kind == "screen":
        u = s.upper()
        if u.startswith("UNKNOWN") or "UNKNOWN" in u[:20]:
            return "UNKNOWN"
        if "KNOWN" in u:
            return "KNOWN"
        return None
    raise ValueError(kind)


def choice_of(trial, parsed):
    """Turn a parsed option into 'round'/'sharp' (or the rating value, or congruent/incongruent)."""
    if parsed is None:
        return None
    if trial["answer_type"] == "number":
        n = int(parsed)
        return n if trial["order"] == 0 else 8 - n
    return trial["options"][parsed]


def out_path(spec, suffix):
    """Default JSONL path for a model spec."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", spec)
    return RAW / f"{safe}{suffix}.jsonl"


def done_keys(path):
    """Set of (trial_id, sample) already logged without error."""
    keys = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not r.get("error"):
                keys.add((r["trial_id"], r["sample"]))
    return keys


def main():
    """Build trials, sample the model, and append results to the log."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="provider:model_id")
    ap.add_argument("--tasks", default="match", help="comma list: match,rating,crossmodal,classic,graded,screen")
    ap.add_argument("--renderings", default=None,
                    help=f"comma list of {','.join(T.RENDERINGS)}; default: all for match, spelled for the rest")
    ap.add_argument("--shape-mode", default="text", choices=["text", "image"])
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="random subset of trials per task (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print trial counts and one prompt per task")
    args = ap.parse_args()

    trials = []
    for task in args.tasks.split(","):
        default = T.RENDERINGS if task == "match" else ["spelled"]
        renderings = args.renderings.split(",") if args.renderings else default
        ts = T.build(task, renderings=renderings, shape_mode=args.shape_mode)
        if args.limit and len(ts) > args.limit:
            ts = random.Random(args.seed).sample(ts, args.limit)
        trials += ts
        print(f"{task}: {len(ts)} trials x {args.samples} samples")
        if args.dry_run and ts:
            print("  e.g.", ts[0]["prompt"].replace("\n", " | "), ts[0]["images"])
    if args.dry_run:
        return

    model = load_model(args.model)
    path = args.out or out_path(args.model, "-mock" if model.provider == "mock" else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(path)
    jobs = [(t, s) for t in trials for s in range(args.samples) if (t["trial_id"], s) not in done]
    print(f"{len(jobs)} samples to run, {len(done)} already logged -> {path}")
    lock = threading.Lock()

    def run(trial, sample):
        """Sample once and return the log record."""
        rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "model": args.model,
               "provider": model.provider, "model_id": model.model_id, "sample": sample,
               "temperature": args.temperature, "system_prompt": SYSTEM_PROMPT, **trial}
        try:
            raw = model.generate(trial["prompt"], trial["images"], args.temperature, args.max_tokens,
                                 options=list(trial["options"]))
            rec["raw"] = raw
            rec["parsed"] = parse(trial, raw)
            rec["choice"] = choice_of(trial, rec["parsed"])
            rec["error"] = None
        except Exception as e:
            rec.update(raw=None, parsed=None, choice=None, error=f"{type(e).__name__}: {e}")
        return rec

    failures = unparsed = 0
    with open(path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(run, t, s) for t, s in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            rec = f.result()
            failures += bool(rec["error"])
            unparsed += rec["error"] is None and rec["parsed"] is None
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            if i % 100 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)}  errors={failures}  unparsed={unparsed}")
    if failures:
        print(f"{failures} samples failed; rerun the same command to retry them.")


if __name__ == "__main__":
    main()
