# `models/` (Current US-scale modeling code)

This directory contains the **current** model training and evaluation code used for US-scale Ailanthus classification.

## Key files

- `main.py` - primary training entry point.
- `model.py` - model architectures (`image_climate`, `image_only`, `climate_only`).
- `datasets.py` - PyTorch dataset for NAIP chips + environmental predictors.
- `train_utils.py` - training loop, checkpointing, early stopping.
- `eval_utils.py` - confusion matrix/metrics and error mapping helpers.
- `eval_model_us.py` - US-scale model evaluation workflow.
- `transforms.py` - image augmentations for 4-band NAIP chips.

## Typical command

```bash
python models/main.py --config configs_sweeps/model_config.json
```

## Notes

- The code expects CSV metadata with `split`, `presence`, `chip_path`, and environmental feature columns.
- The dataset class currently uses a fixed optimized environmental feature subset documented in `datasets.py`.
