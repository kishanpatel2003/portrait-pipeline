# Portrait Pipeline

Curate portrait datasets, check identity consistency, and review generated images. The pipeline combines face metrics, perceptual deduplication, human review, and a learned realism critic.

## Tools

- `pipeline/scripts/score_dataset.py`: face quality, identity similarity, pose, sharpness, and duplicate scoring
- `curate.py`, `preprocess.py`, `caption.py`: training corpus preparation and captions
- `mine_corpus.py`, `mine_golden.py`: candidate ranking from a local corpus
- `admit_golden.py`, `rebuild_centroid.py`: approved reference pool and identity centroid
- `graft_local.py`: face-region refinement
- `eval_generated.py`: generated-image comparison against a reference centroid
- `pipeline/critic/`: frozen features, grouped cross-validation, and a calibrated realism classifier
- `budget_guard.mjs`: persistent spend estimates and budget accounting

## Setup

Python 3.11+, Node 20+, and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --project pipeline
# Add images to dataset/raw
uv run --project pipeline python pipeline/scripts/score_dataset.py
uv run --project pipeline python pipeline/scripts/curate.py
uv run --project pipeline python pipeline/scripts/preprocess.py
PORTRAIT_TRIGGER=portrait_subject uv run --project pipeline python pipeline/scripts/caption.py
```

Install model dependencies appropriate for your hardware. Model loaders may download weights on first use; bring your own images and review each model's license.

Scripts resolve dataset, report, and output paths relative to the repo. Scoring and curation use `dataset/raw/` by default; set `PORTRAIT_INPUT_DIR` to choose another folder. Studio diversity caps use the `studio_` filename prefix (`PORTRAIT_STUDIO_PREFIX` / `PORTRAIT_STUDIO_CAP`). Corpus mining accepts a corpus path.

## Tests

```sh
npm test
python3 -m unittest discover -s pipeline/tests -p 'test_*.py' -v
python3 -m compileall -q pipeline
```

These checks run with example metadata and a temporary budget ledger, without images or model downloads.

## Evaluation

Identity similarity and realism scores guide human review. Train the critic on balanced labels from your own image runs and evaluate it with grouped validation. No trained classifier or portrait model is bundled.

Keep datasets, embeddings, weights, and generated output outside version control.

[MIT](LICENSE). External models and datasets retain their own licenses.
