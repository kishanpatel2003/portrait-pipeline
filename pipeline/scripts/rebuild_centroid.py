#!/usr/bin/env python3
"""Rebuild the current-look identity centroid from reference photos.

Drop new phone references into a folder and re-run. Rolling current-look
policy: only the folders passed (or the two current defaults) contribute —
older looks are excluded, not down-weighted. Each reference's mutual
similarity to the others is reported; outliers below the floor are excluded
from the centroid so one bad photo can't dilute the identity signal.
Previous centroid files are backed up alongside with a timestamp suffix.

  uv run --project pipeline python pipeline/scripts/rebuild_centroid.py [ref_dir ...]
"""
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "dataset" / "reports"
DEFAULT_DIRS = [
    ROOT / "dataset" / "reference_current",
    ROOT / "experiments" / "train_v3_identity_current_2026-07-06" / "reference_current_v3",
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# a reference whose BEST match to any sibling falls below this is a different
# look or a bad photo. Measured on the current 10-ref set: genuine references
# bottom out at 0.773 (v3 refs vs bathroom refs), the one true outlier at 0.695.
OUTLIER_FLOOR = 0.75


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or DEFAULT_DIRS
    files = sorted(p for d in dirs for p in d.glob("*")
                   if p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))
    if len(files) < 3:
        sys.exit(f"need at least 3 reference photos, found {len(files)}")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    names, embs = [], []
    for p in files:
        pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        faces = app.get(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        if not faces:
            print(f"  {p.name}: NO FACE — skipped")
            continue
        f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        names.append(str(p.relative_to(ROOT)))
        embs.append(f.normed_embedding)

    mat = np.stack(embs)
    sims = mat @ mat.T
    np.fill_diagonal(sims, np.nan)
    keep = []
    print(f"\n{'reference':58} mean   best")
    for i, n in enumerate(names):
        mean_s, best_s = np.nanmean(sims[i]), np.nanmax(sims[i])
        flag = "" if best_s >= OUTLIER_FLOOR else "  ← OUTLIER, excluded"
        print(f"{n:58} {mean_s:.3f}  {best_s:.3f}{flag}")
        if best_s >= OUTLIER_FLOOR:
            keep.append(i)
    if len(keep) < 3:
        sys.exit("fewer than 3 references survive the outlier floor — not rebuilding")

    kept = mat[keep]
    centroid = kept.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    for f in ("current_centroid.npy", "current_embeddings.npy", "current_reference_files.json"):
        src = REPORTS / f
        if src.exists():
            shutil.copy2(src, REPORTS / f"{f}.bak_{stamp}")
    np.save(REPORTS / "current_centroid.npy", centroid)
    np.save(REPORTS / "current_embeddings.npy", kept)
    (REPORTS / "current_reference_files.json").write_text(
        json.dumps([names[i] for i in keep], indent=1))
    print(f"\ncentroid rebuilt from {len(keep)}/{len(names)} references "
          f"(backups *.bak_{stamp})")


if __name__ == "__main__":
    main()
