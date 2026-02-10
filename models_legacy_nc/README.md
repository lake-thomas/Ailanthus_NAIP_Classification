# `models_legacy_nc/` (Legacy North Carolina workflow)

This directory preserves the **legacy NC-only** model code that predates the US-scale pipeline.

Use this when reproducing earlier NC experiments or comparing old/new results.

## Key files

- `main.py` - legacy training entry point.
- `model.py` - legacy architecture definitions.
- `datasets.py` - legacy data loader implementation.
- `train_utils.py` - legacy fit/checkpoint utilities.
- `eval_utils.py` - legacy evaluation and plotting.
- `random_forest_classifier.py` - legacy baseline model.

## Important context

- The newer US-scale implementation lives in `models/`.
- This legacy path is intentionally retained; no new features should be developed here unless needed for reproducibility.
