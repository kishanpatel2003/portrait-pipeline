#!/usr/bin/env python3
"""Phase 3b: generate .txt captions for every training image.

Captioning strategy for identity LoRAs: caption everything that VARIES
(clothing, background, lighting, pose, framing) and never describe the face
itself. What the captions don't explain, the model attributes to the trigger
token — which is exactly where the identity should live. This also stops the
LoRA from welding your identity to the black tee / white studio wall that
dominates the corpus.

Captions are built from the metrics in manifest.json, so they're grounded,
not hallucinated. Review them by eye afterwards — a small batch takes a few minutes
and caption errors are the #1 silent quality killer.
"""
import json
import os
from pathlib import Path

TRIGGER = os.environ.get("PORTRAIT_TRIGGER", "portrait_subject")
ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "dataset" / "train"


def pose_phrase(yaw, pitch):
    if yaw is None:
        return "portrait"
    a = abs(yaw)
    side = "left" if yaw < 0 else "right"
    if a >= 65:
        base = f"{side} side profile"
    elif a >= 35:
        base = f"looking to the {side}"
    elif a >= 12:
        base = f"face turned slightly {side}"
    else:
        base = "facing the camera"
    if pitch is not None and pitch > 25:
        base += ", head tilted back"
    return base


def expression_phrase(ratio):
    if ratio is None:
        return ""
    if ratio >= 0.80:
        return "smiling"
    if ratio >= 0.62:
        return "soft smile"
    return "neutral expression"


def scene_phrase(meta):
    if meta["studio"]:
        return "black t-shirt, textured white studio backdrop, soft even studio lighting"
    lm = meta.get("lighting_mean") or 128
    if lm < 80:
        return "indoors at a social event, dim warm indoor lighting, candid photo"
    return "casual setting, natural indoor lighting, candid photo"


def framing_phrase(face_frac):
    if face_frac is None:
        return ""
    if face_frac > 0.08:
        return "close-up headshot"
    if face_frac > 0.015:
        return "waist-up shot"
    return "full body shot"


def main():
    manifest = json.loads((TRAIN / "manifest.json").read_text())
    for name, meta in manifest.items():
        parts = [
            f"photo of {TRIGGER} person",
            framing_phrase(meta.get("face_frac")),
            pose_phrase(meta.get("yaw"), meta.get("pitch")),
            expression_phrase(meta.get("mouth_eye_ratio")),
            scene_phrase(meta),
        ]
        caption = ", ".join(x for x in parts if x)
        (TRAIN / name).with_suffix(".txt").write_text(caption + "\n")
    print(f"wrote {len(manifest)} captions (trigger: '{TRIGGER}')")
    print("REVIEW THEM MANUALLY — fix any wrong clothing/scene guesses.")


if __name__ == "__main__":
    main()
