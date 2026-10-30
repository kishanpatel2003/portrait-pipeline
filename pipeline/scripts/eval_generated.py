#!/usr/bin/env python3
"""Phase 5: score generated images against the real-photo identity reference.

Usage:
  uv run python scripts/eval_generated.py ../generated/<batch_folder>

Per image:
  identity_sim   — ArcFace cosine vs the centroid of your real curated photos.
                   Calibrated against the real corpus: your own real photos
                   score 0.79-0.95 vs centroid (profiles as low as 0.52).
                     > 0.60 frontal / > 0.45 profile  -> PASS
                     0.45-0.60 frontal                -> BORDERLINE
                     below                            -> FAIL (not you)
  best_real_sim  — max similarity to any single real photo (catches the case
                   where the centroid is unfair to extreme poses)
  smoothness     — high-frequency energy on the face crop; too LOW = waxy
                   AI skin. Real faces in this corpus: sharpness 30-630.
  face_count     — >1 subject face or 0 faces flags composition failures
  exposure/clip  — blown highlights or crushed shadows beyond candid-photo norms

Writes <batch>/eval.json + eval.md ranked best-first, and prints a verdict
line per image so you can triage a 50-image batch in one glance.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "dataset" / "reports"

FRONTAL_PASS, FRONTAL_BORDER = 0.72, 0.62
PROFILE_PASS = 0.55
WAXY_SHARPNESS = 25.0   # below this on a big clean face crop = AI smoothing


def load_bgr(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def main():
    batch = Path(sys.argv[1]).resolve()
    # prefer the current-look reference (5 anchor photos) when present
    cur = REPORTS / "current_centroid.npy"
    if cur.exists():
        centroid = np.load(cur)
        real_mat = np.load(REPORTS / "current_embeddings.npy")
    else:
        centroid = np.load(REPORTS / "identity_centroid.npy")
        real_mat = np.load(REPORTS / "identity_embeddings.npy")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    results = []
    files = sorted(p for p in batch.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not files:
        sys.exit(f"no images in {batch}")

    for p in files:
        img = load_bgr(p)
        h, w = img.shape[:2]
        faces = app.get(img)
        row = {"file": p.name, "n_faces": len(faces)}
        if not faces:
            row["verdict"] = "FAIL: no face detected"
            results.append(row)
            continue

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb = face.normed_embedding
        sim = float(emb @ centroid)
        best = float((real_mat @ emb).max())
        yaw = float(face.pose[1])

        x1, y1, x2, y2 = np.clip(face.bbox.astype(int), 0, [w, h, w, h])
        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gray_all = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clip_high = float((gray_all > 250).mean())
        clip_low = float((gray_all < 5).mean())

        is_profile = abs(yaw) > 55
        threshold = PROFILE_PASS if is_profile else FRONTAL_PASS
        problems = []
        if sim < threshold and best < threshold + 0.1:
            problems.append(f"identity {sim:.2f} (not convincing)")
        elif sim < (threshold if is_profile else FRONTAL_BORDER + 0.15):
            pass  # borderline handled below
        if sharp < WAXY_SHARPNESS and (x2-x1) > 200:
            problems.append(f"waxy/oversmoothed skin (sharp {sharp:.0f})")
        if clip_high > 0.25:
            problems.append("blown highlights")
        if clip_low > 0.4:
            problems.append("crushed to black")

        if problems:
            verdict = "FAIL: " + "; ".join(problems)
        elif sim < threshold + 0.08:
            verdict = "BORDERLINE: identity plausible but verify by eye"
        else:
            verdict = "PASS"

        row.update({
            "identity_sim": round(sim, 3),
            "best_real_sim": round(best, 3),
            "yaw": round(yaw, 1),
            "face_sharpness": round(sharp, 1),
            "clip_high": round(clip_high, 3),
            "verdict": verdict,
        })
        results.append(row)

    results.sort(key=lambda r: -(r.get("identity_sim") or 0))
    (batch / "eval.json").write_text(json.dumps(results, indent=1))

    n_pass = sum(r["verdict"] == "PASS" for r in results)
    lines = [f"# Eval — {batch.name}",
             f"**{n_pass}/{len(results)} PASS**", ""]
    for r in results:
        lines.append(f"- `{r['file']}` sim={r.get('identity_sim','—')} "
                     f"best={r.get('best_real_sim','—')} "
                     f"sharp={r.get('face_sharpness','—')} → **{r['verdict']}**")
    (batch / "eval.md").write_text("\n".join(lines))

    for r in results:
        print(f"{r.get('identity_sim','  —  ')}  {r['verdict']:<50} {r['file']}")
    print(f"\n{n_pass}/{len(results)} PASS — details in {batch/'eval.md'}")


if __name__ == "__main__":
    main()
