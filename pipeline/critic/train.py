#!/usr/bin/env python3
"""Train the Tier-1 personal realism critic on the blind-vote labels. $0.

Predicts p(the subject votes "real") per image, for rejection sampling and vote
triage ONLY — this model must never feed a generator loss or prompt loop
(a generator optimized against its judge can exploit that judge).

Labels come straight from experiments/votes.json ("real"/"ai" field; entries
without it stay unlabeled), so re-running this script IS the retrain loop.
Validation is grouped CV with runId as the group (images inside a run share
a generation method and would leak under random CV): GroupKFold x5, ROC-AUC
per fold, C chosen by mean AUC over a small grid. The shipped head is
StandardScaler -> L2 LogisticRegression (balanced), Platt-calibrated on the
same group folds.

  uv run --project pipeline python pipeline/critic/train.py

Writes pipeline/critic/model.joblib + metrics.json (both gitignored). If the
mean AUC lands under the 0.70 floor the model is NOT written — the printed
per-family ablation is the deliverable instead of a bad gate.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402

VOTES = features.ROOT / "experiments" / "votes.json"
RUNS = features.ROOT / "experiments" / "runs"
MODEL = Path(__file__).resolve().parent / "model.joblib"
METRICS = Path(__file__).resolve().parent / "metrics.json"

MIN_LABELED = 60
MIN_PER_CLASS = 20
AUC_FLOOR = 0.70
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
N_FOLDS = 5


def load_labeled(votes_path=VOTES):
    """{'<runId>/<file>.jpg': 0|1} for every vote with a real/ai label."""
    votes = json.loads(Path(votes_path).read_text())
    labeled = {}
    for key, vote in sorted(votes.items()):
        if vote.get("real") not in ("real", "ai"):
            continue
        if not (RUNS / key).exists():
            print(f"  warning: voted file missing on disk, skipped: {key}")
            continue
        labeled[key] = 1 if vote["real"] == "real" else 0
    return labeled


def check_labels(labeled):
    n, pos = len(labeled), sum(labeled.values())
    if n < MIN_LABELED:
        raise SystemExit(
            f"aborting: only {n} labeled images (need >= {MIN_LABELED}) — "
            "vote more images before training")
    if min(pos, n - pos) < MIN_PER_CLASS:
        raise SystemExit(
            f"aborting: class too small — {pos} real / {n - pos} ai "
            f"(need >= {MIN_PER_CLASS} each)")
    return pos, n - pos


def build_matrix(labeled):
    keys = sorted(labeled)
    X = features.featurize_many([RUNS / k for k in keys])
    y = np.array([labeled[k] for k in keys])
    groups = np.array([k.split("/")[0] for k in keys])
    return keys, X, y, groups


def make_head(C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def group_folds(X, y, groups):
    """GroupKFold splits whose test side has both classes (AUC/Platt need
    that; a fold of all-ai runs would be undefined)."""
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    kept = [(tr, te) for tr, te in folds if len(set(y[te])) == 2]
    if len(kept) < len(folds):
        print(f"  note: {len(folds) - len(kept)} single-class fold(s) skipped")
    return kept


def cv_aucs(X, y, groups, C):
    from sklearn.metrics import roc_auc_score
    return [
        roc_auc_score(y[te], make_head(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
        for tr, te in group_folds(X, y, groups)
    ]


def select_C(X, y, groups):
    results = {C: cv_aucs(X, y, groups, C) for C in C_GRID}
    best = max(results, key=lambda C: float(np.mean(results[C])))
    return best, results


def ablation(X, y, groups):
    """Best grouped-CV AUC per feature family alone — the diagnostic the
    plan asks for when the combined gate misses the floor."""
    out = {}
    for fam, sl in features.FEATURE_SLICES.items():
        best, results = select_C(X[:, sl], y, groups)
        out[fam] = {"best_C": best,
                    "mean_auc": round(float(np.mean(results[best])), 4)}
    return out


def fit_final(X, y, groups, C):
    """Calibrated head on all data: Platt on the same group folds, the
    resulting 5 calibrated models are averaged at predict time."""
    from sklearn.calibration import CalibratedClassifierCV
    model = CalibratedClassifierCV(make_head(C), method="sigmoid",
                                   cv=group_folds(X, y, groups))
    return model.fit(X, y)


def main():
    labeled = load_labeled()
    pos, neg = check_labels(labeled)
    runs = sorted({k.split("/")[0] for k in labeled})
    print(f"{len(labeled)} labeled images ({pos} real / {neg} ai) "
          f"across {len(runs)} runs")

    _, X, y, groups = build_matrix(labeled)
    best_C, results = select_C(X, y, groups)
    aucs = results[best_C]
    mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))

    print(f"\ngrouped 5-fold CV (runId groups), C grid {C_GRID}:")
    for C in C_GRID:
        marker = "  <-- selected" if C == best_C else ""
        print(f"  C={C:<6g} mean AUC {np.mean(results[C]):.4f}{marker}")
    print(f"\nper-fold AUC @ C={best_C:g}: "
          + "  ".join(f"{a:.4f}" for a in aucs))
    print(f"mean {mean_auc:.4f} +/- {std_auc:.4f}  "
          f"(floor {AUC_FLOOR}, target 0.80)")

    fam_aucs = ablation(X, y, groups)
    print("\nfeature-family ablation (each family alone):")
    for fam, r in fam_aucs.items():
        print(f"  {fam:5s} mean AUC {r['mean_auc']:.4f}  (C={r['best_C']:g})")

    shipped = mean_auc >= AUC_FLOOR
    METRICS.write_text(json.dumps({
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "labels": {"total": len(labeled), "real": pos, "ai": neg,
                   "runs": len(runs)},
        "feature_dim": features.FEATURE_DIM,
        "feature_version": features.FEATURE_VERSION,
        "chosen_C": best_C,
        "fold_aucs": [round(a, 4) for a in aucs],
        "mean_auc": round(mean_auc, 4),
        "std_auc": round(std_auc, 4),
        "auc_floor": AUC_FLOOR,
        "ablation": fam_aucs,
        "shipped": shipped,
    }, indent=1))
    print(f"\nmetrics -> {METRICS}")

    if not shipped:
        raise SystemExit(
            f"mean AUC {mean_auc:.3f} < floor {AUC_FLOOR} — NOT writing a "
            "model. The ablation above says which family is pulling weight; "
            "fix features or gather more votes, then retrain.")

    import joblib
    joblib.dump({
        "model": fit_final(X, y, groups, best_C),
        "feature_dim": features.FEATURE_DIM,
        "feature_version": features.FEATURE_VERSION,
        "mean_auc": mean_auc,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_labels": len(labeled),
    }, MODEL)
    print(f"model   -> {MODEL}")


if __name__ == "__main__":
    main()
