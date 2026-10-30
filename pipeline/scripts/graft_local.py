#!/usr/bin/env python3
"""Local graft rescue — $0, no API. Attacks the two known failures of the
recall pipeline on already-generated near-misses:

1. identity gap: local inswapper (if ~/.insightface/models/inswapper_128.onnx
   exists) re-copies the face geometry from a real reference;
2. glossy skin: frequency-separation transplant replaces the face's
   high-frequency luminance texture with REAL skin texture from the reference
   (realism-inheritance law: real pixel statistics in → real look out).

Targets are picked from your own votes: every image voted "~ close" in the
runs given (default the copy/recall workhorses). Output is a normal run for
blind voting, marked local/free.

  uv run --project pipeline python pipeline/scripts/graft_local.py \
      [--runs X1b,X0,X7] [--ref experiments/kontext_inputs/cand_00.jpg]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "runs"
VOTES = ROOT / "experiments" / "votes.json"
SWAPPER = Path.home() / ".insightface" / "models" / "inswapper_128.onnx"


def load_bgr(path):
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def face_of(app, img, centroid=None):
    faces = app.get(img)
    if not faces:
        return None
    if centroid is None:
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    return max(faces, key=lambda f: float(f.normed_embedding @ centroid))


def texture_transplant(img, tgt_face, ref_img, ref_face, strength=0.85,
                       mid_strength=0.0, sharpen=0.0):
    """Replace high-frequency luminance inside the face with the reference's
    real skin texture. Low frequencies (identity structure, lighting) stay.

    The inswapper paste is a 128px upscale, so it reads soft/"video game".
    Two counters, both from real pixels or the image itself:
      - `sharpen`: unsharp mask on the face region (restores local contrast)
      - `mid_strength`: a gentle BANDPASS transplant of the reference's
        mid-scale structure (beard strands, skin variation) between the pore
        cutoff and ~fw/45. Full-band transplant at fw/45 destroys identity
        (0.70→0.18 measured), so the mid band must stay moderate.
    """
    x1, y1, x2, y2 = tgt_face.bbox.astype(int)
    fw, fh = x2 - x1, y2 - y1
    if fw < 40 or fh < 40:
        return img, False

    # align reference face onto the target's face geometry via the 5 keypoints
    M, _ = cv2.estimateAffinePartial2D(ref_face.kps.astype(np.float32),
                                       tgt_face.kps.astype(np.float32))
    if M is None:
        return img, False
    ref_warp = cv2.warpAffine(ref_img, M, (img.shape[1], img.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # frequency separation on luminance only (no color shifts).
    # Cutoff must stay at true pore scale: measured on real near-misses,
    # fw/45 wrecks ArcFace identity (0.70→0.18) while fw/120 preserves it.
    sigma = max(1.2, fw / 120)
    sigma_mid = max(sigma * 1.5, fw / 45)
    ycc_t = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ycc_r = cv2.cvtColor(ref_warp, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_t, y_r = ycc_t[..., 0], ycc_r[..., 0]

    if sharpen > 0:  # unsharp mask before texture, so texture lands on a crisp base
        y_t = y_t + sharpen * (y_t - cv2.GaussianBlur(y_t, (0, 0), sigma_mid))

    high_r = y_r - cv2.GaussianBlur(y_r, (0, 0), sigma)
    low_t = cv2.GaussianBlur(y_t, (0, 0), sigma)
    y_new = low_t + strength * high_r + (1 - strength) * (y_t - low_t)
    if mid_strength > 0:  # reference mid band ADDED (not replacing target's)
        mid_r = cv2.GaussianBlur(y_r, (0, 0), sigma) - cv2.GaussianBlur(y_r, (0, 0), sigma_mid)
        y_new = y_new + mid_strength * mid_r

    # feathered elliptical mask over the inner face (skin, not hair/edges)
    mask = np.zeros(img.shape[:2], np.float32)
    cv2.ellipse(mask, (int((x1+x2)/2), int((y1+y2)/2 + 0.05*fh)),
                (int(0.42*fw), int(0.55*fh)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), fw / 12)

    ycc_t[..., 0] = mask * y_new + (1 - mask) * ycc_t[..., 0]
    out = cv2.cvtColor(np.clip(ycc_t, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return out, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="X1b,X0,X7")
    ap.add_argument("--ref", default=str(ROOT / "experiments/kontext_inputs/cand_00.jpg"))
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--mid", type=float, default=0.0,
                    help="mid-band texture strength (counters the 128px swap softness)")
    ap.add_argument("--sharpen", type=float, default=0.0,
                    help="unsharp amount on the face region before texture")
    ap.add_argument("--out", default="G2_local_graft", help="output run id")
    args = ap.parse_args()
    wanted = args.runs.split(",")

    votes = json.loads(VOTES.read_text()) if VOTES.exists() else {}
    targets = sorted(k for k, v in votes.items()
                     if v.get("identity") == "close" and k.split("/")[0] in wanted)
    if not targets:
        print(f"no '~ close' votes found in runs {wanted} — nothing to rescue")
        return

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    centroid = np.load(ROOT / "dataset/reports/current_centroid.npy")

    swapper = None
    if SWAPPER.exists():
        from insightface.model_zoo import get_model
        swapper = get_model(str(SWAPPER), providers=["CPUExecutionProvider"])
    mode = "swap+texture" if swapper else "texture-only (inswapper_128.onnx not found)"
    OUT = RUNS / args.out
    print(f"mode: {mode}; ref: {Path(args.ref).name}; {len(targets)} targets; "
          f"strength {args.strength} mid {args.mid} sharpen {args.sharpen} → {args.out}")

    ref_img = load_bgr(args.ref)
    ref_face = face_of(app, ref_img, centroid)
    if ref_face is None:
        raise SystemExit("no face found in reference photo")

    OUT.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for key in targets:
        src = RUNS / key
        if not src.exists():
            continue
        img = load_bgr(src)
        tgt = face_of(app, img, centroid)
        if tgt is None:
            print(f"  {key}: no face — skipped")
            continue
        sim_before = float(tgt.normed_embedding @ centroid)

        if swapper is not None:
            img = swapper.get(img, tgt, ref_face, paste_back=True)
            tgt = face_of(app, img, centroid) or tgt
        img, ok = texture_transplant(img, tgt, ref_img, ref_face, args.strength,
                                     args.mid, args.sharpen)

        tgt2 = face_of(app, img, centroid)
        sim_after = float(tgt2.normed_embedding @ centroid) if tgt2 is not None else 0.0
        name = key.replace("/", "_")
        cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        provenance[name] = {"source": key, "ref": Path(args.ref).name, "mode": mode,
                            "texture": ok, "sim_before": round(sim_before, 3),
                            "sim_after": round(sim_after, 3)}
        print(f"  {name}  sim {sim_before:.2f} → {sim_after:.2f}")

    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=1))
    (OUT / "meta.json").write_text(json.dumps({
        "title": f"{args.out} — local graft rescue ({mode}, "
                 f"str {args.strength}/mid {args.mid}/sharp {args.sharpen}, $0)",
        "desc": "Your '~ close' near-misses re-grafted locally: inswapper identity pass "
                "(when available) + real-skin high-frequency texture transplant from a "
                "golden reference. Attacks the glossy-skin tell with real pixel "
                "statistics. Local, no API spend.",
        "status": f"awaiting votes ({len(provenance)} images)",
        "local": True, "cost_usd": 0,
    }, indent=1))
    print(f"published {len(provenance)} → {OUT}")


if __name__ == "__main__":
    main()
