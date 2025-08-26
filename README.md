# Ailanthus NAIP Classification

This repository contains code, configuration files, and workflows for classifying **Tree-of-Heaven (Ailanthus altissima)** presence using **NAIP aerial imagery** combined with environmental predictors (climate, DEM, human modification).  

Models are trained and evaluated in **Python/ PyTorch** with support for reproducible runs, inference, and experiment tracking through WandB.

---

## 📂 Repository Structure

README.md — Documentation (this file)

* configs_sweeps/ — Config files for model training and hyperparameter sweeps
  * model_config.json — Config for training CNNs on NAIP + environmental predictors (used by main.py)
  * launch_sweep_wandb.py — Launch W&B hyperparameter sweep for CNN training
  * sweep.yaml — Sweep config example
  * sweep_cnn_hyperparameters.yaml — Sweep config example

* data_prep/ — Scripts for dataset creation and preprocessing
  * species_occurrences_inat.py — Query species points from iNaturalist
  * species_occurrences_gbif.py — Query species points from GBIF
  * species_occurrences_filtering.py — Distance-based point filtering
  * naip_imagery_downloader.py — Download NAIP imagery in North Carolina from S3 bucket
  * naip_imagery_manifest_metadata.py — Query NAIP imagery in North Carolina
  * naip_imagery_download_status.py — Check download status
  * species_train_val_test_random_sampling.py — Randomly split points + image + env data
  * species_train_val_test_stratified_sampling.py — Stratified split for model training
  * host_image_climate_dataset_sampling_ttv_unif_spatialcv.py — Create random/stratified data + spatial CV

* inference/ — Inference and prediction workflows
  * tiled_inference_serial.py — Serial inference on image + env data; outputs predicted probabilities
  * tiled_inference_parallel.py — Parallel inference on image + env data; outputs predicted probabilities
  * tiled_inference_serial_uncertainty.py — Inference with dropout; outputs probability and uncertainty

* models/ — CNN model training and evaluation for image + env data
  * main.py — Train and evaluate CNNs in PyTorch for classification
  * model.py — HostImageClimate, HostImageOnly, HostClimateOnly PyTorch models
  * datasets.py — PyTorch Dataset classes
  * train_utils.py — Fit, load, and save models
  * eval_utils.py — Evaluate on withheld data and compute metrics
  * transforms.py — Image transformations (rotations, flips, etc.)
  * test_transforms.py — Visualize image transformations
  * logging_utils.py — Logging boilerplate
  * random_forst_classifier.py — Random Forest baseline to compare with CNNs

## 🌍 Usage

### Clone the repo
- git clone https://github.com/lake-thomas/Ailanthus_NAIP_Classification.git

### Create a conda environment
- conda create -n <your_environment_name> python=3.10

### Install dependencies
- Python (3.10), Pytorch (Torch 2.4, Torchvision compatable with your CUDA version), and standard libraries (Numpy, Pandas, GeoPandas, Shapely ...)

### Download and organize data
- Run scripts in data_prep to obtain species occurrence points and 4-band NAIP imagery rasters in North Carolina.
- NAIP imagery for North Carolina also available from: https://www.fisheries.noaa.gov/inport/item/70313
- Download climate data (Worldclim.org) and Global Human Footprint (https://doi.org/10.1111/gcb.14549) raster data.

### Train and evaluate models
python models/main.py --config configs_sweeps/model_config.json

### Inference
python inference/tiled_inference_serial.py 

If you use this repo, =cite:
Lake, T. (2025). Classifying Tree-of-Heaven with NAIP imagery and environmental predictors. In prep.



