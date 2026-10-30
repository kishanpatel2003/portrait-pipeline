#!/usr/bin/env python3
"""Mine the scored corpus for Kontext golden-input candidates. $0, fully local.

Golden-input profile (validated by blind evals): close-up, sharp, near-frontal,
warm light, strong identity. Ranks dataset/reports/corpus_scores.json with a
pure fitness function (tested in pipeline/tests/test_mine_golden.py), then
loads only the head of the queue to measure warmth, dedupe by phash against
the training set + existing pool, crop to a head-and-shoulders portrait, and
re-verify each crop against the CURRENT identity centroid. Output is a normal
experiment run so admission happens through the eval UI:

  uv run --project pipeline python pipeline/scripts/mine_golden.py
  → experiments/runs/G1_golden_candidates/   (vote "✓ me" to admit, then run
    admit_golden.py)

Calibration is checked on every run: the corpus rows matching the proven
kontext_inputs pool (by phash) must rank in the top decile, or the weights
have drifted from ground truth.
"""
import json
import sys
from math import log10
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "dataset" / "reports"
POOL = ROOT / "experiments" / "kontext_inputs"
OUT = ROOT / "experiments" / "runs" / "G1_golden_candidates"
_args = [a for a in sys.argv[1:] if a != "--force"]
CORPUS = Path(_args[0]) if _args else Path("dataset/corpus")

DETECT_HEAD = 120   # how many top-ranked rows get the expensive image pass
KEEP = 24           # candidates published for voting
DUP_HAMMING = 6     # phash distance at or below which two photos are "the same"
CROP_PASS = 0.50    # crop must re-verify vs current centroid at least this


def _clamp(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def gate(row) -> bool:
    """Hard floors — anything failing these is not a candidate at all."""
    return bool(
        row.get("subject")
        and row.get("identity_sim", 0) >= 0.5
        and row.get("face_px", 0) >= 180
        and row.get("sharpness", 0) >= 40
        and abs(row.get("yaw", 90)) <= 50
        and row.get("det_score", 0) >= 0.6
    )


def fitness(row) -> float:
    """Pure ranking score mirroring the validated golden-input profile.

    Calibrated against the proven kontext_inputs photos: those are dim, warm,
    modestly sharp night candids, so identity + frontality + croppable face
    resolution dominate, while brightness and Laplacian sharpness get only
    mild weight (studio-bright ≠ golden).
    """
    ident = _clamp((row["identity_sim"] - 0.45) / 0.40)
    res = _clamp(row["face_px"] / 400)
    frontal = _clamp(1 - abs(row["yaw"]) / 45) * _clamp(1 - abs(row.get("pitch", 0)) / 45)
    sharp = _clamp((log10(max(row["sharpness"], 1)) - 1.4) / 1.1)
    lm = row.get("light_mean", 0)
    light = 1.0 if 70 <= lm <= 170 else _clamp(1 - (min(abs(lm - 70), abs(lm - 170))) / 60)
    closeup = _clamp((row.get("face_frac", 0) - 0.005) / 0.045)
    prom = row.get("prominence")
    solo = 1.0 if prom is None else _clamp((prom - 0.8) / 1.2)
    return (3.5 * ident + 1.0 * res + 1.5 * frontal + 0.15 * sharp
            + 0.75 * light + 0.75 * closeup + 0.75 * solo)


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def main():
    import cv2
    import imagehash
    import numpy as np
    from PIL import Image, ImageOps
    from insightface.app import FaceAnalysis

    # never silently replace candidates that already have votes — the votes
    # are keyed by filename and would point at different photos
    votes_file = ROOT / "experiments" / "votes.json"
    if votes_file.exists() and any(
            k.startswith("G1_golden_candidates/")
            for k in json.loads(votes_file.read_text())):
        if "--force" not in sys.argv:
            sys.exit("G1_golden_candidates already has votes. Run admit_golden.py first, "
                     "then re-mine with --force (this replaces the candidate set).")

    rows = json.loads((REPORTS / "corpus_scores.json").read_text())
    ranked = sorted((r for r in rows if gate(r)), key=fitness, reverse=True)
    print(f"{len(ranked)} gated candidates of {len(rows)} corpus photos")

    # ---- calibration: proven pool photos must sit in the top decile ----
    pool_files = sorted(POOL.glob("cand_*.jpg"))
    pool_hashes = [str(imagehash.phash(Image.open(p))) for p in pool_files]
    anchor_idx = [i for i, r in enumerate(ranked)
                  if any(hamming(r["phash"], h) <= 8 for h in pool_hashes)]
    decile = max(1, len(ranked) // 10)
    print(f"calibration: {len(anchor_idx)} pool anchors found in corpus, "
          f"ranks {sorted(anchor_idx)[:10]} (top decile = {decile})")
    if anchor_idx and min(anchor_idx) >= decile:
        print("WARNING: no proven golden input ranks in the top decile — "
              "fitness weights disagree with ground truth. Aborting.")
        sys.exit(1)

    # phashes already spoken for: training set + current pool
    taken = list(pool_hashes)
    for p in sorted((ROOT / "dataset" / "train").glob("*.jpg")):
        taken.append(str(imagehash.phash(Image.open(p))))

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    centroid = np.load(REPORTS / "current_centroid.npy")

    OUT.mkdir(parents=True, exist_ok=True)
    picked, provenance = [], {}
    for r in ranked[:DETECT_HEAD]:
        if len(picked) >= KEEP:
            break
        if any(hamming(r["phash"], h) <= DUP_HAMMING for h in taken):
            continue
        src = CORPUS / r["file"]
        try:
            pil = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        except Exception:
            continue
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        scale = min(1.0, 1280 / max(h, w))
        det = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
        faces = app.get(det)
        if not faces:
            continue
        face = max(faces, key=lambda f: float(f.normed_embedding @ centroid))
        x1, y1, x2, y2 = np.clip((face.bbox / scale).astype(int), 0, [w, h, w, h])
        fw, fh = x2 - x1, y2 - y1

        # warmth measured on the face itself (R/B ratio); warm candids are the target
        fc = img[y1:y2, x1:x2].astype(np.float32)
        warmth = float(fc[..., 2].mean() / max(fc[..., 0].mean(), 1.0)) if fc.size else 1.0

        # head-and-shoulders portrait crop around the face
        cx = (x1 + x2) / 2
        cx1, cx2 = int(cx - 1.4 * fw), int(cx + 1.4 * fw)
        cy1, cy2 = int(y1 - 0.9 * fh), int(y2 + 1.6 * fh)
        cx1, cy1 = max(cx1, 0), max(cy1, 0)
        cx2, cy2 = min(cx2, w), min(cy2, h)
        crop = pil.crop((cx1, cy1, cx2, cy2))

        # crop must still read as him to the CURRENT centroid
        cimg = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
        cfaces = app.get(cimg)
        csim = max((float(f.normed_embedding @ centroid) for f in cfaces), default=0.0)
        if csim < CROP_PASS:
            continue

        crop.thumbnail((1536, 1536))
        name = f"cand_{len(picked):03d}.jpg"
        crop.save(OUT / name, quality=93)
        taken.append(r["phash"])
        picked.append(name)
        provenance[name] = {
            "source": r["file"], "fitness": round(fitness(r), 3),
            "identity_sim": r["identity_sim"], "crop_sim": round(csim, 3),
            "warmth": round(warmth, 3), "face_px": r["face_px"],
            "yaw": r["yaw"], "sharpness": r["sharpness"],
        }
        print(f"  {name}  sim {csim:.2f}  warmth {warmth:.2f}  ← {r['file']}")

    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=1))
    (OUT / "meta.json").write_text(json.dumps({
        "title": "G1 — golden-input candidates (local corpus mine, $0)",
        "desc": "Vote '✓ me' to admit a photo into the Kontext golden-input pool "
                "(then run admit_golden.py). These are cropped candidates matching the "
                "validated profile: close-up, sharp, near-frontal, warm, high identity. "
                "Local mine — no API spend.",
        "status": f"awaiting votes ({len(picked)} candidates)",
        "local": True, "cost_usd": 0,
    }, indent=1))
    print(f"published {len(picked)} candidates → {OUT}")


if __name__ == "__main__":
    main()
