# `data_prep/` (Current US-scale data preparation)

Scripts in this folder prepare the US-wide training dataset.

## What these scripts do

- collect presence/background occurrence records,
- query NAIP image URLs,
- download imagery,
- build training CSVs with split assignments and predictor variables.

## Typical sequence

1. Build occurrence/background table.
2. Generate NAIP URL manifest.
3. Download missing imagery.
4. Create train/val/test dataset CSV.

> Some scripts are long-running and intended for HPC or batch execution.
