#!/usr/bin/env python3
"""Phase 1: score every raw photo for LoRA-training usefulness.

Outputs dataset/reports/scores.json with per-image metrics:
  face detection/visibility, sharpness, lighting, pose (yaw/pitch/roll),
  expression proxy (mouth openness/smile from landmarks), perceptual-hash
  duplicate clusters, and identity-embedding distance from the corpus centroid
  (flags photos where the detected face may not be the subject).
"""
import json
import os
import sys
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from insightface.app import FaceAnalysis

RAW_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def load_bgr(path: Path) -> np.ndarray | None:
    """Load with EXIF orientation applied (phone photos need this)."""
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def sharpness_score(gray_face: np.ndarray) -> float:
    """Variance of Laplacian, normalized on the face crop only —
    a blurry background with a sharp face should still pass."""
    if gray_face.size == 0:
        return 0.0
    v = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    return float(v)


def lighting_score(gray_face: np.ndarray) -> dict:
    if gray_face.size == 0:
        return {"mean": 0, "clip_low": 1.0, "clip_high": 0.0, "contrast": 0}
    mean = float(gray_face.mean())
    clip_low = float((gray_face < 10).mean())
    clip_high = float((gray_face > 245).mean())
    contrast = float(gray_face.std())
    return {"mean": mean, "clip_low": clip_low, "clip_high": clip_high, "contrast": contrast}


def expression_proxy(kps: np.ndarray) -> dict:
    """From 5-point kps: mouth width vs eye distance is a crude smile proxy."""
    eye_d = float(np.linalg.norm(kps[0] - kps[1]))
    mouth_w = float(np.linalg.norm(kps[3] - kps[4]))
    return {"mouth_eye_ratio": round(mouth_w / max(eye_d, 1e-6), 3)}


def main():
    root = Path(__file__).resolve().parents[2]
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("PORTRAIT_INPUT_DIR", root / "dataset" / "raw"))
    if not raw.is_dir():
        raise SystemExit(f"Input directory missing: {raw}")
    report_dir = root / "dataset" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    files = sorted(
        p for p in raw.iterdir()
        if p.suffix.lower() in RAW_EXTS and not p.name.startswith(".")
    )
    print(f"Scoring {len(files)} images…")

    rows = []
    embeddings = {}
    for p in tqdm(files):
        img = load_bgr(p)
        row = {"file": p.name}
        if img is None:
            row["error"] = "unreadable"
            rows.append(row)
            continue
        h, w = img.shape[:2]
        row["width"], row["height"] = w, h

        # detect on a bounded copy for speed; map boxes back to full res
        scale = min(1.0, 1600 / max(h, w))
        det_img = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
        faces = app.get(det_img)
        row["n_faces"] = len(faces)
        if not faces:
            rows.append(row)
            continue

        # biggest face = presumed subject
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        bbox = (face.bbox / scale).astype(int)
        x1, y1, x2, y2 = np.clip(bbox, 0, [w, h, w, h])
        face_frac = ((x2 - x1) * (y2 - y1)) / (w * h)
        face_px = max(x2 - x1, y2 - y1)

        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else np.zeros((1, 1), np.uint8)

        row.update({
            "det_score": round(float(face.det_score), 3),
            "face_frac": round(float(face_frac), 4),
            "face_px": int(face_px),
            "yaw": round(float(face.pose[1]), 1),
            "pitch": round(float(face.pose[0]), 1),
            "roll": round(float(face.pose[2]), 1),
            "sharpness": round(sharpness_score(gray), 1),
            "lighting": lighting_score(gray),
            "expression": expression_proxy(face.kps / scale),
        })
        emb = face.normed_embedding
        embeddings[p.name] = emb

        # perceptual hash for dedup (whole image)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        row["phash"] = str(imagehash.phash(pil))
        rows.append(row)

    # identity centroid + per-image similarity (finds mislabeled/other-person faces)
    if embeddings:
        names = list(embeddings)
        mat = np.stack([embeddings[n] for n in names])
        centroid = mat.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        sims = mat @ centroid
        sim_map = dict(zip(names, sims))
        for row in rows:
            if row["file"] in sim_map:
                row["identity_sim"] = round(float(sim_map[row["file"]]), 3)

    # duplicate clustering by phash hamming distance
    hashed = [(r["file"], imagehash.hex_to_hash(r["phash"])) for r in rows if "phash" in r]
    DUP_THRESH = 8
    cluster_of = {}
    clusters = []
    for name, hsh in hashed:
        placed = False
        for ci, (rep_hash, members) in enumerate(clusters):
            if hsh - rep_hash <= DUP_THRESH:
                members.append(name)
                cluster_of[name] = ci
                placed = True
                break
        if not placed:
            cluster_of[name] = len(clusters)
            clusters.append((hsh, [name]))
    for row in rows:
        if row["file"] in cluster_of:
            row["dup_cluster"] = cluster_of[row["file"]]

    # embedding centroid saved for the Phase 5 eval loop
    if embeddings:
        np.save(report_dir / "identity_centroid.npy", centroid)
        np.save(report_dir / "identity_embeddings.npy", mat)
        (report_dir / "identity_embedding_files.json").write_text(json.dumps(names))

    out = report_dir / "scores.json"
    out.write_text(json.dumps(rows, indent=1))
    n_clusters = len(clusters)
    print(f"Wrote {out} — {len(rows)} images, {n_clusters} visual clusters")


if __name__ == "__main__":
    main()
