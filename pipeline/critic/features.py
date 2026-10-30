#!/usr/bin/env python3
"""Feature extraction for the Tier-1 personal realism critic. $0, fully local.

One image in → one fixed FEATURE_DIM float32 vector out (pure per path).
Frozen embeddings + no-reference IQA scores, no fine-tuning anywhere:

  clip  OpenCLIP ViT-L/14 (laion2b_s32b_b82k) on the full frame and on the
        subject's face crop                                          2 x 768
  dino  DINOv2-base (timm, lvd142m weights) on the face crop — texture and
        local-structure sensitivity (waxy skin / GTA smoothness)         768
  iqa   pyiqa no-reference metrics on face crop then full frame       2 x 5
        (topiq_nr-face, musiq, liqe, niqe, clipiqa — blur, halos, grain)

The face crop is the subject's face bbox expanded ~1.6x, the face chosen by
max similarity to the CURRENT identity centroid (same pattern as
graft_local.face_of), falling back to the full frame when nothing is
detected. IQA inputs are clamped to [224, 2048] px on the short/long side so
NIQE's block statistics stay defined and huge frames stay cheap.

Vectors are cached in pipeline/critic/cache/*.npz keyed by (path, mtime,
FEATURE_VERSION) — retraining is seconds once a corpus is featurized. Bump
FEATURE_VERSION when anything above changes.

Heavy models load lazily and run on MPS when available; every pyiqa metric
drops to CPU by itself if its ops are missing on MPS (correctness over
speed). Set CRITIC_DEVICE=cpu to force CPU for everything.
"""
import hashlib
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "cache"
CENTROID = ROOT / "dataset" / "reports" / "current_centroid.npy"

FEATURE_VERSION = 1
CLIP_DIM = 768
DINO_DIM = 768
IQA_METRICS = ("topiq_nr-face", "musiq", "liqe", "niqe", "clipiqa")
FEATURE_SLICES = {
    "clip": slice(0, 2 * CLIP_DIM),
    "dino": slice(2 * CLIP_DIM, 2 * CLIP_DIM + DINO_DIM),
    "iqa": slice(2 * CLIP_DIM + DINO_DIM,
                 2 * CLIP_DIM + DINO_DIM + 2 * len(IQA_METRICS)),
}
FEATURE_DIM = 2 * CLIP_DIM + DINO_DIM + 2 * len(IQA_METRICS)

FACE_EXPAND = 1.6   # bbox -> crop scale factor
IQA_MIN_SIDE = 224  # below this NIQE's 96px block stats break down
IQA_MAX_SIDE = 2048

_models = {}


def _device():
    import torch
    if "device" not in _models:
        if os.environ.get("CRITIC_DEVICE"):
            _models["device"] = torch.device(os.environ["CRITIC_DEVICE"])
        else:
            _models["device"] = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu")
    return _models["device"]


def _clip():
    if "clip" not in _models:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k")
        _models["clip"] = (model.eval().to(_device()), preprocess)
    return _models["clip"]


def _dino():
    if "dino" not in _models:
        import timm
        model = timm.create_model("vit_base_patch14_dinov2.lvd142m",
                                  pretrained=True, num_classes=0)
        cfg = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**cfg, is_training=False)
        _models["dino"] = (model.eval().to(_device()), preprocess)
    return _models["dino"]


def _face_app():
    if "face" not in _models:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _models["face"] = (app, np.load(CENTROID))
    return _models["face"]


def load_rgb(path):
    from PIL import Image, ImageOps
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def face_crop(pil):
    """Subject's face crop (centroid-max face, bbox x FACE_EXPAND).

    Returns (crop, True) or (full frame, False) when no face is detected.
    """
    import cv2
    app, centroid = _face_app()
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    scale = min(1.0, 1280 / max(h, w))
    det = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
    faces = app.get(det)
    if not faces:
        return pil, False
    face = max(faces, key=lambda f: float(f.normed_embedding @ centroid))
    x1, y1, x2, y2 = face.bbox / scale
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * FACE_EXPAND / 2, (y2 - y1) * FACE_EXPAND / 2
    box = (max(int(cx - hw), 0), max(int(cy - hh), 0),
           min(int(cx + hw), w), min(int(cy + hh), h))
    return pil.crop(box), True


def _embed_clip(pils):
    import torch
    model, preprocess = _clip()
    batch = torch.stack([preprocess(p) for p in pils]).to(_device())
    with torch.no_grad():
        emb = model.encode_image(batch)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.float().cpu().numpy()


def _embed_dino(pil):
    import torch
    model, preprocess = _dino()
    batch = preprocess(pil)[None].to(_device())
    with torch.no_grad():
        emb = model(batch)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.float().cpu().numpy()[0]


def _iqa_ready(pil):
    """Resize copy for the IQA metrics: short side >= IQA_MIN_SIDE (NIQE
    needs room for its 96px blocks), long side <= IQA_MAX_SIDE."""
    from PIL import Image
    w, h = pil.size
    scale = max(1.0, IQA_MIN_SIDE / min(w, h))
    scale = min(scale, IQA_MAX_SIDE / max(w, h))
    if scale == 1.0:
        return pil
    return pil.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def _iqa_score(name, pil):
    import torch
    tensor = torch.from_numpy(
        np.asarray(pil, np.float32) / 255.0).permute(2, 0, 1)[None]
    key = ("iqa", name)
    if key not in _models:
        import pyiqa
        try:
            _models[key] = pyiqa.create_metric(name, device=_device())
        except Exception:
            _models[key] = pyiqa.create_metric(name, device="cpu")
    try:
        with torch.no_grad():
            return float(_models[key](tensor))
    except Exception:
        import pyiqa  # op not implemented on MPS etc. — redo on CPU
        _models[key] = pyiqa.create_metric(name, device="cpu")
        with torch.no_grad():
            return float(_models[key](tensor))


def _cache_file(path):
    st = path.stat()
    key = hashlib.sha1(
        f"{path.resolve()}|{st.st_mtime_ns}|v{FEATURE_VERSION}".encode()).hexdigest()
    return CACHE / f"{key}.npz"


def featurize(path, use_cache=True):
    """The one entry point: FEATURE_DIM float32 vector for one image path."""
    path = Path(path)
    cfile = _cache_file(path)
    if use_cache and cfile.exists():
        vec = np.load(cfile)["vec"]
        if vec.shape == (FEATURE_DIM,):
            return vec
    pil = load_rgb(path)
    crop, has_face = face_crop(pil)
    clip_full, clip_face = _embed_clip([pil, crop])
    dino_face = _embed_dino(crop)
    crop_iqa, full_iqa = _iqa_ready(crop), _iqa_ready(pil)
    iqa = ([_iqa_score(n, crop_iqa) for n in IQA_METRICS]
           + [_iqa_score(n, full_iqa) for n in IQA_METRICS])
    vec = np.concatenate([clip_full, clip_face, dino_face,
                          np.asarray(iqa, np.float32)]).astype(np.float32)
    assert vec.shape == (FEATURE_DIM,), vec.shape
    if use_cache:
        CACHE.mkdir(exist_ok=True)
        np.savez_compressed(cfile, vec=vec, path=str(path), has_face=has_face)
    return vec


def featurize_many(paths, progress=True):
    from tqdm import tqdm
    it = tqdm(paths, desc="featurize", unit="img") if progress else paths
    return np.stack([featurize(p) for p in it])
