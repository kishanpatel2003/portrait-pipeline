#!/usr/bin/env python3
"""Admit voted golden-input candidates into the Kontext pool.

Human approval is the gate: a "✓ me" identity vote on the G1_golden_candidates
run (localhost:3777/eval) admits that crop into experiments/kontext_inputs/.
Anything else (close / no / unvoted) is left alone. Re-runnable: already-admitted
photos are skipped by perceptual hash.

  uv run --project pipeline python pipeline/scripts/admit_golden.py
"""
import json
import shutil
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments" / "runs" / "G1_golden_candidates"
POOL = ROOT / "experiments" / "kontext_inputs"
VOTES = ROOT / "experiments" / "votes.json"
DUP_HAMMING = 6


def main():
    votes = json.loads(VOTES.read_text()) if VOTES.exists() else {}
    approved = sorted(
        k.split("/", 1)[1] for k, v in votes.items()
        if k.startswith("G1_golden_candidates/") and v.get("identity") == "me")
    if not approved:
        print("no '✓ me' votes on G1_golden_candidates yet — vote at http://localhost:3777/eval")
        return

    pool = sorted(POOL.glob("cand_*.jpg"))
    pool_hashes = [imagehash.phash(Image.open(p)) for p in pool]
    next_idx = max((int(p.stem.split("_")[1]) for p in pool), default=-1) + 1

    admitted, skipped = [], []
    for f in approved:
        src = RUN / f
        if not src.exists():
            continue
        h = imagehash.phash(Image.open(src))
        if any(h - ph <= DUP_HAMMING for ph in pool_hashes):
            skipped.append(f)
            continue
        dst = POOL / f"cand_{next_idx:02d}.jpg"
        shutil.copy2(src, dst)
        pool_hashes.append(h)
        admitted.append(f"{f} → {dst.name}")
        next_idx += 1

    for line in admitted:
        print("admitted", line)
    if skipped:
        print(f"skipped {len(skipped)} already in pool: {', '.join(skipped)}")
    print(f"pool size now {len(list(POOL.glob('cand_*.jpg')))} "
          f"({len(admitted)} new of {len(approved)} approved)")


if __name__ == "__main__":
    main()
