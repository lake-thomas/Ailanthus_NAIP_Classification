# Random Forest Classification and Inference for Host Species Presence

import os
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt

# Set Conda Env Path with GDAL and PROJ libraries
conda_env = r"C:\Users\tomla\anaconda3\envs\naip_ailanthus"

os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

# Import rasterio after setting environment variables because GDAL and PROJ need to be configured
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask

# ------------- CONFIG ----------------
csv_path = r'D:\Ailanthus_NAIP_Classification\Datasets\Ailanthus_Uniform_PA_NAIP_256_July25\Ailanthus_Train_Val_Test.csv'
model_save_path = r"D:\Ailanthus_NAIP_Classification\rf_model\rf_model_ailanthus_nc_env_dem_ghm_lat_lon.pkl"
output_path = r"D:\Ailanthus_NAIP_Classification\rf_model\rf_inference.tif"
worldclim_dir = r'D:\Ailanthus_NAIP_Classification\Env_Data\Worldclim'
ghm_path = r'D:\Ailanthus_NAIP_Classification\Env_Data\Global_Human_modification\ghm_wgs84.tif'
shapefile_path = r'D:\Ailanthus_NAIP_Classification\Env_Data\tl_2024_us_state\tl_2024_us_state_wgs84.shp'
bioclim_files = [f"wc2.1_30s_bio_{i}.tif" for i in range(1, 20)]
raster_paths = [os.path.join(worldclim_dir, fname) for fname in bioclim_files] + [ghm_path]
raster_names = [f'bio{i}' for i in range(1, 20)] + ['ghm']

# ------------- Helper: sample rasters at points ----------------
def sample_rasters_at_points(lat, lon, raster_paths, raster_names):
    coords = list(zip(lon, lat))
    features = []
    for path in raster_paths:
        with rasterio.open(path) as src:
            sampled = [v[0] if v is not None else np.nan for v in src.sample(coords)]
            features.append(sampled)
    feature_array = np.stack(features, axis=1) # shape: (n_points, n_rasters)
    feature_df = pd.DataFrame(feature_array, columns=raster_names)
    return feature_df

# ------------- MAIN WORKFLOW ----------------
print("Loading data...")
df = pd.read_csv(csv_path)
train_df, val_df, test_df = df[df.split == 'train'], df[df.split == 'val'], df[df.split == 'test']

print("Sampling raster values for TRAIN set...")
X_train = sample_rasters_at_points(
    lat=train_df['lat'], lon=train_df['lon'],
    raster_paths=raster_paths, raster_names=raster_names
)
y_train = train_df['presence'].values

print("Sampling raster values for VAL set...")
X_val = sample_rasters_at_points(
    lat=val_df['lat'], lon=val_df['lon'],
    raster_paths=raster_paths, raster_names=raster_names
)
y_val = val_df['presence'].values

print("Sampling raster values for TEST set...")
X_test = sample_rasters_at_points(
    lat=test_df['lat'], lon=test_df['lon'],
    raster_paths=raster_paths, raster_names=raster_names
)
y_test = test_df['presence'].values

# Drop rows with missing raster data (in any column)
def drop_nans(X, y):
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]

X_train, y_train = drop_nans(X_train.values, y_train)
X_val, y_val     = drop_nans(X_val.values, y_val)
X_test, y_test   = drop_nans(X_test.values, y_test)

# ------------- TRAIN RANDOM FOREST ----------------
param_grid = {
    'n_estimators': [500],
    'max_features': ['sqrt'],
    'max_depth': [5, 10, None],
    'min_samples_split': [10],
    'min_samples_leaf': [10],
    'bootstrap': [True],
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='f1', n_jobs=4, verbose=1
)
print("Fitting model (grid search)...")
grid.fit(X_train, y_train)
print(f"Best parameters: {grid.best_params_}")

best_model = grid.best_estimator_
joblib.dump(best_model, model_save_path)

for split, X_split, y_split in [
    ('Validation', X_val, y_val),
    ('Test', X_test, y_test)
]:
    print(f"{split}:")
    print(classification_report(y_split, best_model.predict(X_split)))

# ------------- FEATURE IMPORTANCE ----------------
importances = best_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[sorted_idx])
plt.xticks(range(len(importances)), [raster_names[i] for i in sorted_idx], rotation=90)
plt.tight_layout()
plt.show()

# ------------- RESPONSE CURVES ----------------
for i in range(min(5, len(raster_names))):
    feature = raster_names[sorted_idx[i]]
    plt.figure(figsize=(8, 4))
    plt.title(f"Response for {feature}")
    plt.scatter(X_train[:, sorted_idx[i]], y_train, alpha=0.3, label='Train')
    plt.scatter(X_val[:, sorted_idx[i]], y_val, alpha=0.3, label='Val', color='orange')
    plt.xlabel(feature)
    plt.ylabel('Presence (1) / Absence (0)')
    plt.legend()
    plt.show()

# ------------- INFERENCE: PREDICT RASTER ----------------
states = gpd.read_file(shapefile_path)
nc_geom = [g.__geo_interface__ for g in states[states['STUSPS'] == 'NC'].geometry.values]

stacked_arrays = []
raster_meta = None

print("Loading and stacking rasters for prediction...")

# First, stack Worldclim/bioclim variables
stacked_arrays = []
raster_meta = None

for i, path in enumerate(raster_paths):
    print(f"Loading {os.path.basename(path)} ...")
    with rasterio.open(path) as src:
        if raster_names[i] != 'ghm':
            data, out_transform = mask(src, nc_geom, crop=True)
            arr = data[0].astype(np.float32)
            if raster_meta is None:
                # Store output metadata from the first raster
                raster_meta = src.meta.copy()
                raster_meta.update({
                    "count": 1,
                    "dtype": "float32",
                    "height": arr.shape[0],
                    "width": arr.shape[1],
                    "transform": out_transform,
                    "compress": "lzw"
                })
            if i == 0:
                # Save these for resampling GHM
                reference_shape = arr.shape
                reference_transform = out_transform
            stacked_arrays.append(arr)
        else:
            # --- GHM: mask, then reproject/resample to reference shape and transform ---
            ghm_masked, _ = mask(src, nc_geom, crop=True)
            ghm_arr = ghm_masked[0].astype(np.float32)
            ghm_resampled = np.empty(reference_shape, dtype=np.float32)
            rasterio.warp.reproject(
                source=ghm_arr,
                destination=ghm_resampled,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference_transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear
            )
            stacked_arrays.append(ghm_resampled)

stacked_array = np.stack(stacked_arrays, axis=-1)
print(f"Stacked array shape: {stacked_array.shape}")
n_rows, n_cols, n_features = stacked_array.shape

# Reshape for model input
X_infer = stacked_array.reshape(-1, n_features)
valid_mask = ~np.any(np.isnan(X_infer), axis=1)
X_valid = X_infer[valid_mask]

print("Running prediction for raster...")
probs = np.zeros(X_infer.shape[0], dtype=np.float32)
probs[valid_mask] = best_model.predict_proba(X_valid)[:, 1]
prob_raster = probs.reshape(n_rows, n_cols)

print(f"Saving inference raster to: {output_path}")
with rasterio.open(output_path, "w", **raster_meta) as dst:
    dst.write(prob_raster, 1)

# ----------- TEST SET RASTER SAMPLING AND FINAL EVALUATION ----------
print("Sampling predictions at test locations...")
with rasterio.open(output_path) as src:
    coords = list(zip(test_df['lon'], test_df['lat']))
    sampled_preds = np.array([v[0] for v in src.sample(coords)])

valid = ~np.isnan(sampled_preds)
y_true = y_test[valid]
y_pred_prob = sampled_preds[valid]

plt.figure(figsize=(10, 6))
plt.hist(y_pred_prob, bins=50)
plt.title("Histogram of Predicted Probabilities at Test Points")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.show()

threshold = 0.5
y_pred_binary = (y_pred_prob >= threshold).astype(int)
print("Performance at Test Locations (Raster Sampling):")
print("Accuracy:", accuracy_score(y_true, y_pred_binary))
print("Precision:", precision_score(y_true, y_pred_binary))
print("Recall:", recall_score(y_true, y_pred_binary))
print("F1 Score:", f1_score(y_true, y_pred_binary))
print("ROC AUC:", roc_auc_score(y_true, y_pred_prob))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_binary))