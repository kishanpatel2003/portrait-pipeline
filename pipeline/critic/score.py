#!/usr/bin/env python3
"""Score images with the trained realism critic (rejection sampling / triage
only — never a training signal).

  uv run --project pipeline python pipeline/critic/score.py G2b_graft_sharp
  uv run --project pipeline python pipeline/critic/score.py some/img.jpg [...]

Run mode writes ONE sidecar, experiments/runs/<runId>/critic_scores.json
({"<file>.jpg": {"p_real": 0.87}, ...}); meta.json, votes.json and the
images are never touched. Explicit image paths are scored and printed only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402

RUNS = features.ROOT / "experiments" / "runs"
MODEL = Path(__file__).resolve().parent / "model.joblib"


def load_model(path=MODEL):
    if not Path(path).exists():
        raise SystemExit(f"no model at {path} — run pipeline/critic/train.py first")
    import joblib
    bundle = joblib.load(path)
    if bundle["feature_version"] != features.FEATURE_VERSION:
        raise SystemExit(
            f"model was trained on feature v{bundle['feature_version']} but "
            f"features.py is v{features.FEATURE_VERSION} — retrain first")
    return bundle


def score_paths(bundle, paths):
    X = features.featurize_many(paths)
    return bundle["model"].predict_proba(X)[:, 1]


def print_table(names, p_real, votes=None):
    order = sorted(range(len(names)), key=lambda i: -p_real[i])
    print(f"\nrank  p_real  vote  file")
    for rank, i in enumerate(order, 1):
        vote = (votes or {}).get(names[i], {}).get("real", "-")
        print(f"{rank:4d}  {p_real[i]:.4f}  {vote:4s}  {names[i]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("targets", nargs="+",
                    help="a runId under experiments/runs, or image paths")
    args = ap.parse_args()

    bundle = load_model()
    run_dir = RUNS / args.targets[0]
    if len(args.targets) == 1 and run_dir.is_dir():
        run_id = args.targets[0]
        imgs = sorted(run_dir.glob("*.jpg"))
        if not imgs:
            raise SystemExit(f"no .jpg images in {run_dir}")
        p_real = score_paths(bundle, imgs)
        sidecar = run_dir / "critic_scores.json"
        sidecar.write_text(json.dumps(
            {img.name: {"p_real": round(float(p), 4)}
             for img, p in zip(imgs, p_real)}, indent=1))
        votes_file = features.ROOT / "experiments" / "votes.json"
        votes = json.loads(votes_file.read_text()) if votes_file.exists() else {}
        votes = {k.split("/", 1)[1]: v for k, v in votes.items()
                 if k.startswith(run_id + "/")}
        print_table([i.name for i in imgs], p_real, votes)
        print(f"\n{len(imgs)} scores -> {sidecar}")
    else:
        paths = [Path(t) for t in args.targets]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f"not a runId and not image files: "
                             f"{', '.join(map(str, missing))}")
        p_real = score_paths(bundle, paths)
        print_table([str(p) for p in paths], p_real)


if __name__ == "__main__":
    main()
