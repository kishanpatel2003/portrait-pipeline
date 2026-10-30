#!/usr/bin/env python3
"""Phase 1b: apply curation rules to scores.json.

Copies keepers to dataset/curated/, rejects to dataset/rejected/<reason>/,
and writes dataset/reports/curation_report.md with gap analysis.

A single studio session can dominate and overfit a LoRA. Cap matching
filenames with PORTRAIT_STUDIO_PREFIX and PORTRAIT_STUDIO_CAP, using diversity
selection over pose and expression buckets. These defaults are heuristics;
review the report and calibrate them on your own dataset.
"""
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "dataset" / "reports"
CURATED = ROOT / "dataset" / "curated"
REJECTED = ROOT / "dataset" / "rejected"

RAW = Path(os.environ.get("PORTRAIT_INPUT_DIR", ROOT / "dataset" / "raw"))
STUDIO_PREFIX = os.environ.get("PORTRAIT_STUDIO_PREFIX", "studio_")
STUDIO_CAP = int(os.environ.get("PORTRAIT_STUDIO_CAP", "18"))          # max studio-session images in final set
MIN_SHARP = 12.0         # variance-of-Laplacian on face crop
MIN_FACE_PX = 250        # below this AND soft sharpness -> reject
SOFT_SHARP = 30.0


def yaw_bucket(yaw: float) -> str:
    a = abs(yaw)
    side = "L" if yaw < 0 else "R"
    if a < 12: return "frontal"
    if a < 35: return f"quarter-{side}"
    if a < 65: return f"half-{side}"
    return f"profile-{side}"


def expr_bucket(ratio: float) -> str:
    # mouth-width / eye-distance: wide mouth ~ smiling
    if ratio < 0.62: return "neutral"
    if ratio < 0.80: return "soft"
    return "smile"


def pitch_bucket(pitch: float) -> str:
    if pitch < -12: return "down"
    if pitch > 25: return "up"
    return "level"


def main():
    rows = json.loads((REPORTS / "scores.json").read_text())
    for d in (CURATED, REJECTED):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    keep, reject = [], []  # (row, reason)

    for r in rows:
        if r.get("error") or r.get("n_faces", 0) == 0:
            reject.append((r, "no_face"))
            continue
        if r["sharpness"] < MIN_SHARP:
            reject.append((r, "blurry"))
            continue
        if r["face_px"] < MIN_FACE_PX and r["sharpness"] < SOFT_SHARP:
            reject.append((r, "small_and_soft"))
            continue
        lit = r["lighting"]
        if lit["clip_high"] > 0.30 or (lit["mean"] < 35 and r["sharpness"] < SOFT_SHARP):
            reject.append((r, "bad_lighting"))
            continue
        keep.append(r)

    # exact/near-duplicate collapse: best sharpness per phash cluster
    by_cluster = defaultdict(list)
    for r in keep:
        by_cluster[r.get("dup_cluster", r["file"])].append(r)
    deduped = []
    for members in by_cluster.values():
        members.sort(key=lambda r: -r["sharpness"])
        deduped.append(members[0])
        for m in members[1:]:
            reject.append((m, "near_duplicate"))

    # cap the studio session with diversity-greedy selection
    studio = [r for r in deduped if r["file"].startswith(STUDIO_PREFIX)]
    other = [r for r in deduped if not r["file"].startswith(STUDIO_PREFIX)]
    picked = []
    if len(studio) > STUDIO_CAP:
        buckets = defaultdict(list)
        for r in studio:
            key = (yaw_bucket(r["yaw"]), pitch_bucket(r["pitch"]),
                   expr_bucket(r["expression"]["mouth_eye_ratio"]))
            buckets[key].append(r)
        # round-robin: one best-sharpness image per bucket, repeat until cap
        ordered = sorted(buckets.values(), key=len)
        for b in ordered:
            b.sort(key=lambda r: -r["sharpness"])
        i = 0
        while len(picked) < STUDIO_CAP and any(ordered):
            for b in ordered:
                if b and len(picked) < STUDIO_CAP:
                    picked.append(b.pop(0))
            ordered = [b for b in ordered if b]
            i += 1
            if i > 50: break
        for b in buckets.values():
            for m in b:
                if m not in picked:
                    reject.append((m, "studio_redundant"))
        studio = picked
    final = studio + other

    for r in final:
        src = RAW / r["file"]
        shutil.copy2(src, CURATED / r["file"])
    for r, reason in reject:
        d = REJECTED / reason
        d.mkdir(exist_ok=True)
        src = RAW / r["file"]
        if src.exists():
            shutil.copy2(src, d / r["file"])

    # ---- gap analysis ----
    yaws = Counter(yaw_bucket(r["yaw"]) for r in final)
    exprs = Counter(expr_bucket(r["expression"]["mouth_eye_ratio"]) for r in final)
    body = Counter(
        "headshot" if r["face_frac"] > 0.08 else
        "waist-up" if r["face_frac"] > 0.015 else "full-body"
        for r in final)
    envs = Counter("studio" if r["file"].startswith(STUDIO_PREFIX) else "real-world"
                   for r in final)

    gaps = []
    if yaws.get("profile-R", 0) + yaws.get("half-R", 0) == 0:
        gaps.append("**No right-side profiles** — all profile shots face left. Shoot a few right-side 45° and 90° angles.")
    if body.get("full-body", 0) < 3:
        gaps.append("**Almost no full-body shots** — the model will guess your build/height/posture. Add 5-8 full-body photos (standing, walking, sitting) in different outfits.")
    if exprs.get("smile", 0) < 4:
        gaps.append("Few open smiles — add laughing/candid-smile shots or generations will default to a neutral face.")
    if envs["real-world"] < envs["studio"]:
        gaps.append("Studio still outweighs real-world environments — every extra outdoor/indoor candid photo helps break the studio-backdrop association.")
    gaps.append("Only ~2 lighting regimes present (studio softbox, dim indoor). Add golden-hour outdoor, harsh sunlight, and window light.")
    gaps.append("Hairstyle varies across the corpus (short curls vs longer perm/mullet). Decide which era to train on and drop or caption the other, otherwise the LoRA averages them.")

    lines = [
        "# Curation report", "",
        f"- Input images: **{len(rows)}**",
        f"- Curated (kept): **{len(final)}**  → `dataset/curated/`",
        f"- Rejected: **{len(reject)}**  → `dataset/rejected/<reason>/`", "",
        "## Rejection reasons",
        *[f"- {reason}: {n}" for reason, n in
          Counter(reason for _, reason in reject).most_common()], "",
        "## Curated-set composition",
        f"- Yaw: {dict(yaws)}",
        f"- Expression: {dict(exprs)}",
        f"- Framing: {dict(body)}",
        f"- Environment: {dict(envs)}", "",
        "## Dataset gaps (in priority order)",
        *[f"{i+1}. {g}" for i, g in enumerate(gaps)], "",
        "## Kept files",
        *[f"- {r['file']}  (yaw {r['yaw']}, sharp {r['sharpness']}, sim {r.get('identity_sim')})"
          for r in sorted(final, key=lambda r: r["file"])],
    ]
    (REPORTS / "curation_report.md").write_text("\n".join(lines))
    print(f"kept {len(final)} ({envs['studio']} studio / {envs['real-world']} real-world), "
          f"rejected {len(reject)}")
    print(f"report: {REPORTS / 'curation_report.md'}")


if __name__ == "__main__":
    main()
