#!/usr/bin/env python3
"""Recommend the next blind-voting batch: the unvoted images the critic is
least sure about (|p_real - 0.5| smallest) — votes there teach it the most.

  uv run --project pipeline python pipeline/critic/queue_uncertain.py [--top 15]

Unvoted = present in a run dir (top-level *.jpg only; contact sheets in
subdirs don't count) with no "real"/"ai" label in votes.json — an
identity-only vote still needs a realism vote. Read-only: never touches
votes.json, meta.json or images; only the shared feature cache is written.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from score import load_model, score_paths  # noqa: E402

RUNS = features.ROOT / "experiments" / "runs"
VOTES = features.ROOT / "experiments" / "votes.json"


def unvoted_keys():
    votes = json.loads(VOTES.read_text()) if VOTES.exists() else {}
    keys = []
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        for img in sorted(run_dir.glob("*.jpg")):
            key = f"{run_dir.name}/{img.name}"
            if votes.get(key, {}).get("real") not in ("real", "ai"):
                keys.append(key)
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=15,
                    help="how many images to recommend (default 15)")
    args = ap.parse_args()

    keys = unvoted_keys()
    if not keys:
        print("nothing unvoted — every run image has a real/ai label")
        return
    print(f"{len(keys)} unvoted images across "
          f"{len({k.split('/')[0] for k in keys})} runs; scoring...")

    bundle = load_model()
    p_real = score_paths(bundle, [RUNS / k for k in keys])
    order = sorted(range(len(keys)), key=lambda i: abs(p_real[i] - 0.5))

    print(f"\nnext voting batch — top {args.top} by uncertainty:")
    print("      p_real  |p-0.5|  image")
    for rank, i in enumerate(order[:args.top], 1):
        print(f"{rank:4d}  {p_real[i]:.4f}  {abs(p_real[i] - 0.5):.4f}   {keys[i]}")


if __name__ == "__main__":
    main()
