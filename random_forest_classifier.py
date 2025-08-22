# Random Forest Classification and Inference for Host Species Presence

import os
import joblib
import glob
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

import warnings
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but RandomForestClassifier was fitted without feature names"
)

# Set Conda Env Path with GDAL and PROJ libraries
conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"

os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

# Import rasterio after setting environment variables because GDAL and PROJ need to be configured
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask

# ------------- CONFIG ----------------

# CSV file containing training, validation, and test data in spatial cross validation splits
cv_paths = [
   r"D:\Ailanthus_NAIP_Classification\Datasets\Ailanthus_Uniform_PA_1300m_thin_NAIP_256_Aug2125\Ailanthus_Train_Val_Test_Uniform_Random_NC.csv", # Uniform Random Split
   r"D:\Ailanthus_NAIP_Classification\Datasets\Ailanthus_CrossVal_PA_1300m_thin_NAIP_256_Aug2125\CV_1\Ailanthus_Train_Val_Test_CV_1.csv", # CV 1
   r"D:\Ailanthus_NAIP_Classification\Datasets\Ailanthus_CrossVal_PA_1300m_thin_NAIP_256_Aug2125\CV_2\Ailanthus_Train_Val_Test_CV_2.csv", # CV 2
   r"D:\Ailanthus_NAIP_Classification\Datasets\Ailanthus_CrossVal_PA_1300m_thin_NAIP_256_Aug2125\CV_3\Ailanthus_Train_Val_Test_CV_3.csv" # CV 3
]
split_names = ["uniform", "cv1", "cv2", "cv3"]

output_path = r"D:\Ailanthus_NAIP_Classification\rf_model\rf_inference_map_aug2125.tif"
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

# Drop rows with missing raster data (in any column)
def drop_nans(X, y):
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]

# ------------- INFERENCE: Run model on stacked rasters ----------------
def run_model_inference(model, output_path, test_df, shapefile_path):
    print("Loading and stacking rasters for prediction...")
    states = gpd.read_file(shapefile_path)
    nc_geom = [g.__geo_interface__ for g in states[states['STUSPS'] == 'NC'].geometry.values]

    stacked_arrays = []
    raster_meta = None

    for j, path in enumerate(raster_paths):
        print(f"Loading {os.path.basename(path)} ...")
        with rasterio.open(path) as src:
            if raster_names[j] != 'ghm':
                data, out_transform = mask(src, nc_geom, crop=True)
                arr = data[0].astype(np.float32)
                if raster_meta is None:
                    raster_meta = src.meta.copy()
                    raster_meta.update({
                        "count": 1,
                        "dtype": "float32",
                        "height": arr.shape[0],
                        "width": arr.shape[1],
                        "transform": out_transform,
                        "compress": "lzw"
                    })
                if j == 0:
                    reference_shape = arr.shape
                    reference_transform = out_transform
                stacked_arrays.append(arr)
            else:
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
    n_rows, n_cols, n_features = stacked_array.shape
    X_infer = stacked_array.reshape(-1, n_features)
    valid_mask = ~np.any(np.isnan(X_infer), axis=1)
    X_valid = X_infer[valid_mask]

    print("Running prediction for raster...")
    probs = np.zeros(X_infer.shape[0], dtype=np.float32)
    probs[valid_mask] = model.predict_proba(X_valid)[:, 1]
    prob_raster = probs.reshape(n_rows, n_cols)

    print(f"Saving inference raster to: {output_path}")
    with rasterio.open(output_path, "w", **raster_meta) as dst:
        dst.write(prob_raster, 1)

    return prob_raster


# ------------- MAIN WORKFLOW ----------------
results = []

for split_name, csv_path in zip(split_names, cv_paths):
    print(f"\n=== Processing Split: {split_name} ===")

    df = pd.read_csv(csv_path)
    
    train_df, val_df, test_df = (
        df[df.split == 'train'], 
        df[df.split == 'val'], 
        df[df.split == 'test']
    )

    # Sample raster values for train, validation, and test sets
    X_train = sample_rasters_at_points(train_df['lat'], train_df['lon'], raster_paths, raster_names)
    y_train = train_df['presence'].values

    X_val = sample_rasters_at_points(val_df['lat'], val_df['lon'], raster_paths, raster_names)
    y_val = val_df['presence'].values

    X_test = sample_rasters_at_points(test_df['lat'], test_df['lon'], raster_paths, raster_names)
    y_test = test_df['presence'].values

    # Drop NaNs
    X_train, y_train = drop_nans(X_train.values, y_train)
    X_val, y_val = drop_nans(X_val.values, y_val)
    X_test, y_test = drop_nans(X_test.values, y_test)
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # ------------- TRAIN RANDOM FOREST ----------------
    param_grid = {
        'n_estimators': [500],
        'max_features': ['sqrt'],
        'max_depth': [5, ],
        'min_samples_split': [5],
        'min_samples_leaf': [5],
        'bootstrap': [True],
    }

    print("Grid search for hyperparameter tuning...")
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring='f1', n_jobs=4, verbose=1
    )

    print("Fitting model (grid search)...")
    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")

    best_model = grid.best_estimator_
    model_save_path = fr"D:\Ailanthus_NAIP_Classification\rf_model\rf_model_{split_name}_aug2125.pkl"
    joblib.dump(best_model, model_save_path)

    # ------------- PARTIAL DEPENDENCE PLOTS ----------------
    from sklearn.inspection import PartialDependenceDisplay

    # Make a directory to save PDPs if not exists
    pdp_dir = r"D:\Ailanthus_NAIP_Classification\rf_model\pdp_plots"
    os.makedirs(pdp_dir, exist_ok=True)

    # Create partial dependence plots for all predictors
    print("Generating partial dependence plots...")

    # Select top N features if you want fewer plots
    features_to_plot = raster_names  # or e.g., raster_names[:10]

    # Scikit-learn requires the *training data* used to fit
    # (here, X_train with column names)
    X_train_df = pd.DataFrame(X_train, columns=raster_names)

    for feature in features_to_plot:
        print(f" - {feature}")
        fig, ax = plt.subplots(figsize=(6, 4))
        disp = PartialDependenceDisplay.from_estimator(
            best_model,
            X_train_df,
            [feature],
            kind="average",       # "average" (default) ≈ Maxent response curves
            grid_resolution=50,   # more points = smoother curves
            ax=ax
        )
        ax.set_title(f"Partial Dependence: {feature}")
        fig.tight_layout()
        fig.savefig(os.path.join(pdp_dir, f"pdp_{feature}_{split_name}.png"), dpi=300)
        plt.close(fig)

    # ------------- INFERENCE: PREDICT RASTER ---------------
    output_raster_path = fr"D:\Ailanthus_NAIP_Classification\rf_model\rf_inference_{split_name}_aug2125.tif"
    prob_raster = run_model_inference(best_model, output_raster_path, test_df, shapefile_path)

    # Sample raster at test points
    with rasterio.open(output_raster_path) as src:
        coords = list(zip(test_df['lon'], test_df['lat']))
        sampled_preds = np.array([v[0] for v in src.sample(coords)])

    valid = ~np.isnan(sampled_preds)
    y_true = y_test[valid]
    y_pred_prob = sampled_preds[valid]
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)

    # Plot histogram of predicted probabilities
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_prob, bins=50)
    plt.title("Histogram of Predicted Probabilities at Test Points")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.savefig(fr"D:\Ailanthus_NAIP_Classification\rf_model\rf_pred_prob_histogram_{split_name}_aug2125.png", dpi=300)
    plt.close()

    # Evaluate performance at test locations
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

    # ------------- FORMATTED CLASSIFICATION REPORT ----------------
    print("\nGenerating formatted classification report...")
    target_names = ['Negative', 'Positive']

    report_dict = classification_report(
        y_true, y_pred_binary, target_names=target_names, output_dict=True
    )

    report_df = pd.DataFrame(report_dict).T
    report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
    report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].round(4)
    report_df['support'] = report_df['support'].fillna(0).astype(int)

    # Set support for accuracy explicitly
    report_df.loc['accuracy', 'support'] = len(y_true)

    print("\nFormatted Classification Report:")
    print(report_df.to_string())

    report_csv_path = fr"D:\Ailanthus_NAIP_Classification\rf_model\rf_classification_report_{split_name}_aug2125.csv"
    report_df.to_csv(report_csv_path, index=True)







