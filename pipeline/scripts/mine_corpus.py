#!/usr/bin/env python3
"""Mine a large unfiltered photo corpus for training-worthy images.

Unlike score_dataset.py (which assumes the biggest face is the subject),
this matches EVERY detected face against the saved identity reference and
keeps the best-matching one — so group shots resolve to the subject, not to
whoever is closest to the camera. Photos where no face clears the identity
floor are binned as not_subject.

Usage:
  uv run python scripts/mine_corpus.py dataset/corpus

Writes dataset/reports/corpus_scores.json. Selection happens in
select_from_corpus.py so thresholds can be tuned without re-running 25 min
of face detection.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS
from tqdm import tqdm

from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "dataset" / "reports"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# identity floors: real photos of the subject score 0.52-0.95 vs centroid
# (extreme profiles lowest). Anything below 0.35 on BOTH metrics is treated
# as another person.
FLOOR_CENTROID = 0.35
FLOOR_BEST = 0.40


def exif_date(im: Image.Image) -> str | None:
    try:
        exif = im.getexif()
        for tag_id in (36867, 306):  # DateTimeOriginal, DateTime
            v = exif.get(tag_id)
            if v:
                return datetime.strptime(str(v)[:19], "%Y:%m:%d %H:%M:%S").isoformat()
    except Exception:
        pass
    return None


def main():
    corpus = Path(sys.argv[1]).resolve()
    centroid = np.load(REPORTS / "identity_centroid.npy")
    real_mat = np.load(REPORTS / "identity_embeddings.npy")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    files = sorted(p for p in corpus.rglob("*")
                   if p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))
    print(f"Mining {len(files)} images…", flush=True)

    rows = []
    for p in tqdm(files, mininterval=15):
        row = {"file": str(p.relative_to(corpus))}
        try:
            pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        except Exception:
            row["error"] = "unreadable"
            rows.append(row)
            continue
        row["date"] = exif_date(Image.open(p))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        row["width"], row["height"] = w, h

        scale = min(1.0, 1280 / max(h, w))
        det_img = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
        faces = app.get(det_img)
        row["n_faces"] = len(faces)
        if not faces:
            rows.append(row)
            continue

        # find the subject among all faces by identity match
        best_face, best_key = None, (-1.0, -1.0)
        for f in faces:
            emb = f.normed_embedding
            c_sim = float(emb @ centroid)
            b_sim = float((real_mat @ emb).max())
            if (c_sim, b_sim) > best_key:
                best_key, best_face = (c_sim, b_sim), f
        c_sim, b_sim = best_key
        row["identity_sim"] = round(c_sim, 3)
        row["best_real_sim"] = round(b_sim, 3)
        if c_sim < FLOOR_CENTROID and b_sim < FLOOR_BEST:
            row["subject"] = False
            rows.append(row)
            continue
        row["subject"] = True

        f = best_face
        bbox = (f.bbox / scale).astype(int)
        x1, y1, x2, y2 = np.clip(bbox, 0, [w, h, w, h])
        face_px = max(x2 - x1, y2 - y1)
        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else np.zeros((1, 1), np.uint8)

        # how prominent is he vs. the biggest other face?
        others = [g for g in faces if g is not f]
        biggest_other = max(
            ((g.bbox[2]-g.bbox[0]) * (g.bbox[3]-g.bbox[1]) for g in others),
            default=0.0)
        own = (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])
        row.update({
            "face_frac": round(float((x2-x1)*(y2-y1)) / (w*h), 4),
            "face_px": int(face_px),
            "prominence": round(float(own / max(biggest_other, 1e-6)), 2) if others else None,
            "yaw": round(float(f.pose[1]), 1),
            "pitch": round(float(f.pose[0]), 1),
            "det_score": round(float(f.det_score), 3),
            "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
            "light_mean": round(float(gray.mean()), 1),
            "mouth_eye_ratio": round(
                float(np.linalg.norm(f.kps[3]-f.kps[4]) /
                      max(np.linalg.norm(f.kps[0]-f.kps[1]), 1e-6)), 3),
            # face top near frame top + tiny face_frac ⇒ likely full body below
            "face_top_frac": round(float(y1) / h, 3),
            "phash": str(imagehash.phash(pil)),
        })
        rows.append(row)

    out = REPORTS / "corpus_scores.json"
    out.write_text(json.dumps(rows, indent=1))
    n_subj = sum(1 for r in rows if r.get("subject"))
    print(f"done: {len(rows)} images, {n_subj} contain the subject → {out}")


if __name__ == "__main__":
    main()
