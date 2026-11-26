"""Download the external human datasets from OSF into data/external/ (gitignored, not redistributed).

  sidhu2022  Sidhu, Vigliocco & Pexman (2022), "Higher order factors of sound symbolism",
             Journal of Memory and Language 125, 104323. https://osf.io/gruqs/
             Spoken nonwords rated on 25 semantic scales (sharp-round, exciting-calming, hard-soft, ...).
  sospass    "SoS PASS: Auditory pseudoword and visual shape stimuli to investigate sound symbolism"
             (McCormick, Lacey, Sathian et al.). https://osf.io/ekpgh/
             Recorded pseudowords with roundedness/pointedness ratings, and shape stimuli.

Neither OSF project states a licence, so the files stay local and are cited, not committed.

Usage: uv run python src/fetch_external.py [--list] [sidhu2022 sospass]
"""

import argparse
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "external"
PROJECTS = {"sidhu2022": "gruqs", "sospass": "ekpgh"}
API = "https://api.osf.io/v2"


def walk(session, url, prefix=""):
    """Yield (relative path, size, download url) for every file under an OSF folder listing, following pages."""
    while url:
        page = session.get(url, timeout=60).json()
        for item in page["data"]:
            a = item["attributes"]
            path = f"{prefix}{a['name']}"
            if a["kind"] == "folder":
                yield from walk(session, item["relationships"]["files"]["links"]["related"]["href"], path + "/")
            else:
                yield path, a.get("size") or 0, item["links"]["download"]
        url = page["links"].get("next")


def main():
    """List or download each requested project's files, skipping files already present at the right size."""
    ap = argparse.ArgumentParser()
    ap.add_argument("projects", nargs="*", default=list(PROJECTS))
    ap.add_argument("--list", action="store_true", help="only list files and sizes")
    args = ap.parse_args()
    session = requests.Session()
    for name in args.projects:
        node = PROJECTS[name]
        files = list(walk(session, f"{API}/nodes/{node}/files/osfstorage/"))
        total = sum(s for _, s, _ in files)
        print(f"{name} (osf.io/{node}): {len(files)} files, {total / 1e6:.1f} MB")
        if args.list:
            for path, size, _ in files:
                print(f"  {size / 1e3:9.1f} kB  {path}")
            continue
        for path, size, url in files:
            dest = OUT / name / path
            if dest.exists() and dest.stat().st_size == size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(8):
                with session.get(url, stream=True, timeout=120) as r:
                    if r.status_code == 429:
                        time.sleep(int(r.headers.get("Retry-After", 0)) or 5 * 2 ** attempt)
                        continue
                    r.raise_for_status()
                    with open(dest, "wb") as fh:
                        for chunk in r.iter_content(1 << 16):
                            fh.write(chunk)
                    break
            else:
                raise SystemExit(f"OSF kept rate-limiting {path}; rerun later to resume")
            time.sleep(0.5)
        print(f"  saved to {OUT / name}", file=sys.stderr)


if __name__ == "__main__":
    main()
