#!/usr/bin/env python3
"""Phase 3a: turn dataset/curated/ into dataset/train/ ready for LoRA training.

- applies EXIF orientation, converts to sRGB JPEG q95
- caps longest side at 1536px (trainer buckets down from there; bigger just
  slows uploads)
- if the subject's face is tiny (<1.5% of frame), crops to ~2.8x the face box
  so the identity signal isn't lost in downscaling
- renames to stable ascii names (img_001.jpg …) — the Unicode narrow-space
  filenames in this corpus break some trainer UIs
- writes dataset/train/manifest.json mapping new -> original names + metrics
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
CURATED = ROOT / "dataset" / "curated"
TRAIN = ROOT / "dataset" / "train"
REPORTS = ROOT / "dataset" / "reports"

MAX_SIDE = 1536
CROP_FACE_FRAC = 0.015
CROP_SCALE = 2.8


def main():
    TRAIN.mkdir(parents=True, exist_ok=True)
    for old in TRAIN.glob("img_*.jpg"):
        old.unlink()

    scores = {r["file"]: r for r in json.loads((REPORTS / "scores.json").read_text())}
    files = sorted(p for p in CURATED.iterdir() if not p.name.startswith("."))
    manifest = {}

    for i, p in enumerate(files, 1):
        r = scores.get(p.name, {})
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        w, h = im.size

        if r.get("face_frac", 1) < CROP_FACE_FRAC and "face_px" in r:
            # re-detect not needed: scores.json has no bbox, so approximate a
            # center crop at CROP_SCALE * face_px around image center of mass
            side = min(max(int(r["face_px"] * CROP_SCALE), 512), min(w, h))
            cx, cy = w // 2, h // 3  # faces sit in the upper third of candids
            x1 = np.clip(cx - side // 2, 0, w - side)
            y1 = np.clip(cy - side // 2, 0, h - side)
            im = im.crop((int(x1), int(y1), int(x1) + side, int(y1) + side))
            w, h = im.size

        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

        name = f"img_{i:03d}.jpg"
        im.save(TRAIN / name, quality=95, subsampling=1)
        manifest[name] = {
            "original": p.name,
            "yaw": r.get("yaw"), "pitch": r.get("pitch"),
            "face_frac": r.get("face_frac"),
            "lighting_mean": r.get("lighting", {}).get("mean"),
            "mouth_eye_ratio": r.get("expression", {}).get("mouth_eye_ratio"),
            "studio": p.name.startswith(os.environ.get("PORTRAIT_STUDIO_PREFIX", "studio_")),
        }

    (TRAIN / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} training images to {TRAIN}")


if __name__ == "__main__":
    main()
