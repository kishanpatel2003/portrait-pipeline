# Portrait Pipeline

Reusable tools for curating a consented portrait dataset, checking identity
consistency, and reviewing generated images. The pipeline combines face metrics,
perceptual deduplication, a human admission loop, and an optional learned realism
critic. No portraits, face embeddings, model weights, hosted model links, or
original experiment records are included.

## What is here

- `pipeline/scripts/score_dataset.py`: score input images for face quality,
  identity similarity, pose, sharpness, and perceptual duplicates
- `curate.py`, `preprocess.py`, `caption.py`: prepare a training corpus and
  heuristic captions for manual review
- `mine_corpus.py`, `mine_golden.py`: rank candidates from your local corpus
- `admit_golden.py`, `rebuild_centroid.py`: turn explicit human votes into an
  admitted reference pool and rebuilt identity centroid
- `graft_local.py`: local face-region refinement experiments
- `eval_generated.py`: compare generated images against a reference centroid
- `pipeline/critic/`: frozen image features, grouped cross-validation, and a
  calibrated linear classifier trained on your own realism labels
- `budget_guard.mjs`: persistent estimate/debit accounting, defaulting to $0

## Start without photos or API access

Node 20+ and Python 3.11+:

```sh
npm test
python3 -m unittest discover -s pipeline/tests -p 'test_*_synthetic.py' -v
python3 -m compileall -q pipeline
```

These checks use synthetic metadata and a temporary budget ledger. They require
no personal images, weights, account credentials, or paid API calls.

## Run on your own dataset

The Python environment in `pipeline/pyproject.toml` lists the image/ML libraries
used by the original implementation. Install compatible versions for your
hardware (the full environment is substantial). Model loaders may download
third-party weights on first use; review their individual terms separately.

```sh
uv sync --project pipeline
# Put only your own or explicitly consented images in dataset/raw.
uv run --project pipeline python pipeline/scripts/score_dataset.py
uv run --project pipeline python pipeline/scripts/curate.py
uv run --project pipeline python pipeline/scripts/preprocess.py
PORTRAIT_TRIGGER=portrait_subject uv run --project pipeline python pipeline/scripts/caption.py
```

Scripts resolve dataset, report, and generated-output paths relative to the repo;
inspect each script's input contract before running a later stage. The score and curation
scripts use `dataset/raw/` by default; set `PORTRAIT_INPUT_DIR` consistently to
choose a different input folder. Studio diversity caps use the `studio_` filename
prefix by default (`PORTRAIT_STUDIO_PREFIX` / `PORTRAIT_STUDIO_CAP`). Corpus mining
accepts a corpus path. This is a research toolkit, not a one-command hosted product.
Original training-service runners and experiment-specific presets are excluded.

## Evaluation limits

Face-recognition similarity is not a guarantee that a photo looks right to a
human. The realism critic needs a sufficiently large, balanced set of labels
from your own image runs; no trained classifier is bundled. Report grouped
validation rather than claiming a general-purpose authenticity detector.
Generated scenes are not documentary evidence. Keep private identity data,
training artifacts, and model URLs outside version control.

MIT applies to the code in this repository. External models and datasets retain
their own licenses. The public edition contains no original private history.
