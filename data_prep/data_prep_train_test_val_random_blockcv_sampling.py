# Dataset Sampling Script for Ailanthus altissima in the US
# Uniform Random Train/ Val/ Test Split and Block Cross-validation Split
# Thomas Lake, January 2026

# Imports
import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import random # noqa: E402
import numpy as np # noqa: E402
import pandas as pd # noqa: E402
import geopandas as gpd # noqa: E402
import rasterio # noqa: E402
from rasterio.windows import Window # noqa: E402
from rasterio.transform import from_origin # noqa: E402
from rasterio.merge import merge # noqa: E402
from rasterio.mask import mask # noqa: E402
from shapely.geometry import Point, box # noqa: E402
import tqdm # noqa: E402


# ---------------------- Load Paths ---------------------- #

# Set random seed for reproducibility
random.seed(42)

# Root Path to Save Uniform/ Random Dataset Split
uniform_dataset_name = "Ailanthus_US_Uniform_PA_NAIP_256_jan26"
uniform_dataset_root = rf"Y:\Ailanthus_NAIP_SDM\Datasets\{uniform_dataset_name}"
os.makedirs(uniform_dataset_root, exist_ok=True)  # Create root directory if it doesn't exist

# Root Path to Save Block Cross-validation Dataset Split
crossval_dataset_name = "Ailanthus_US_BlockCV_PA_NAIP_256_jan26"
crossval_dataset_root = rf"Y:\Ailanthus_NAIP_SDM\Datasets\{crossval_dataset_name}"
os.makedirs(crossval_dataset_root, exist_ok=True)  # Create root directory if it doesn't exist

# Paths for NAIP data
tileindex_fp = r"Y:\Ailanthus_NAIP_SDM\NAIP_Imagery_Tile_Indices\NAIP_US_State_Tile_Indices_URL_Paths_jan26.shp"
naip_folder = r"Y:\Ailanthus_NAIP_SDM\NAIP_Imagery_Ailanthus_PB_US_2m"

# Paths for Host Occurrence Data, containing Ailanthus presence and background points with lon, lat columns and presence (0/1 column)
# Occurrences obtained from iNaturalist, GBIF, and State/ Federal Agency Sources (APHIS)
host_occurrence_fp = r"Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Occurrences_Background_iNat_GBIF_Agency_Processed\Ailanthus_US_iNat_GBIF_APHIS_State_P_B_Filtered_Sampled_NAIP_Tiles_113k.csv"

# Paths for Worldclim Climate Data
worldclim_folder = r"Y:\Ailanthus_NAIP_SDM\Env_Data\Worldclim"
wc_raster_fp = r"Y:\Ailanthus_NAIP_SDM\Env_Data\Worldclim\wc2.1_30s_bio_1.tif"

# Climate variable names list
WC_VARS = [f"wc2.1_30s_bio_{i}" for i in range(1, 20)] + ["ghm"]

# Path for Human Modification Data
ghm_raster_fp = r"Y:\Ailanthus_NAIP_SDM\Env_Data\Global_Human_modification\gHM_WGS84.tif"

# States shapefile path
states_shapefile_path = r"Y:\Ailanthus_NAIP_SDM\Env_Data\tl_2024_us_state\tl_2024_us_state.shp"
states = gpd.read_file(states_shapefile_path)

# Block Cross-validation shapefile (blocks defined from blockCV in R)
blocks_shapefile_path = r"Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Occurrences_Background_iNat_GBIF_Agency_Processed\cv_blocks_jan26\Ailanthus_US_113k_SpatialBlocks.shp"
blocks = gpd.read_file(blocks_shapefile_path).to_crs('EPSG:4326')


# ------------------------ Load Data ------------------------ #

# Load NAIP data from paths
tileindex = gpd.read_file(tileindex_fp).to_crs('EPSG:4326')
tileindex['filename'] = tileindex['filename'].apply(os.path.basename)
naip_files = set(os.listdir(naip_folder))
tileindex['status'] = tileindex['filename'].apply(lambda x: "Downloaded" if x in naip_files else "Not Downloaded")
downloaded_tiles = tileindex[tileindex['status'] == 'Downloaded']
# downloaded_union = downloaded_tiles.unary_union # Shapefile union of all downloaded NAIP tiles
downloaded_union = downloaded_tiles.geometry.union_all()

# Load climate raster
with rasterio.open(wc_raster_fp) as src:
    wc_transform = src.transform
    wc_crs = src.crs # EPSG:4326

# Load Host Occurrence Data as GeoDataFrame
pa_df = pd.read_csv(host_occurrence_fp)

pa_data = gpd.GeoDataFrame(
    pa_df,
    geometry=gpd.points_from_xy(pa_df.lon, pa_df.lat),
    crs="EPSG:4326"
)

print("Initial Host Occurrence Data")
print(pa_data['presence'].value_counts())

# Check for duplicate occurrence records by location (lon, lat) and drop duplicates
# Duplicates introduced during NAIP downloading (one point intersecting multiple tiles)
pa_data = pa_data.drop_duplicates(subset=['lon', 'lat'])

print("Deduplicated Host Occurrence Data")
print(pa_data['presence'].value_counts())

# Check data format
print(pa_data.head())

# Keep a copy of the original data before splitting for later use in cross-validation splits
pa_data_og = pa_data.copy()  # Keep original for reference before assigning train/val/test sets


# ------------------------- Helper Functions ------------------------- #

# Function to compute mean and standard deviation of WorldClim variables within a study area
def compute_worldclim_stats(worldclim_folder, study_geom, buffer=0.01):
    """
    Compute mean and standard deviation of WorldClim variables within a study area.
    Normalization step prior to training models
    """
    stats = {}
    # Buffer study area slightly
    study_geom_proj = gpd.GeoSeries([study_geom], crs='EPSG:4326').buffer(buffer).__geo_interface__['features'][0]['geometry']

    for n in range(1, 20):  # Loop through each WorldClim variable
        varname = f"wc2.1_30s_bio_{n}"
        raster_fp = os.path.join(worldclim_folder, f"{varname}.tif")
        with rasterio.open(raster_fp) as src:
            out_image, _ = mask(src, [study_geom_proj], crop=True)
            data = out_image[0]  # Get the first band
            if src.nodata is not None:
                data = np.ma.masked_equal(data, src.nodata)
            else:
                data = np.ma.masked_where((data < -1e5) | (data > 1e5), data)

            # Compute stats only on valid data
            mean = data.mean()
            std = data.std()

            stats[varname] = {"mean": float(mean), "std": float(std)}
            print(f"{varname}: mean={mean:.4f}, std={std:.4f}")

    return stats

def extract_worldclim_vars_for_point(lon, lat, wc_datasets, normalization_stats):
    """ Extract normalized Worldclim variables for a given point (lon, lat). """
    vals = {}
    for varname, ds in wc_datasets.items():
        value = list(ds.sample([(lon, lat)]))[0][0]

        mean = normalization_stats[varname]["mean"]
        std = normalization_stats[varname]["std"]

        vals[varname] = (value - mean) / std

    return vals

def is_valid_climate_record(wc_vars, min_val=-100, max_val=100):
    """
    Returns False if any climate variable is NaN, inf,
    or absurdly large/small after normalization.
    """
    for v in WC_VARS:
        val = wc_vars.get(v, np.nan)

        if not np.isfinite(val):
            return False

        if val < min_val or val > max_val:
            return False

    return True

def extract_ghm_for_point(lon, lat, ghm_ds):
    """ Extract GHM value using pre-opened rasterio dataset. """
    value = list(ghm_ds.sample([(lon, lat)]))[0][0]

    # Clamp to [0, 1]
    return min(max(value, 0.0), 1.0)

# Extract NAIP chips for each point in the dataset
def extract_naip_chip_for_point(lon, lat, out_fp, chip_size, tileindex, naip_folder):
    """
    Extract a chip of size `chip_size` around the point (lon, lat) from NAIP tiles.
    Based on the tile index, it finds the appropriate NAIP tiles that cover the point.
    If multiple tiles are needed to cover the chip area, it tries to mosaic them.
    Saves the chip as a GeoTIFF to `out_fp`.
    Returns True if successful, False if point is not covered by NAIP.
    """

    # WGS84 point (lon, lat)
    point_wgs84 = Point(lon, lat)

    # Find *any* matching tile to get image CRS (all NAIP images should be in same CRS per region)
    matches = tileindex[tileindex.geometry.intersects(point_wgs84)]

    if matches.empty:
        raise ValueError(f"Point {lon},{lat} not in any NAIP tile.")
    
    # Use the tile matching to select a NAIP image based on filename and check if it exists
    naip_fp = None
    for idx, row in matches.iterrows():
        candidate_fp = os.path.join(naip_folder, row['filename'])
        if os.path.exists(candidate_fp):
            naip_fp = candidate_fp
            break

    if naip_fp is None:
        raise FileNotFoundError(f"No downloaded NAIP tile found for point {lon},{lat}.")

    # Open the NAIP image to get its CRS
    with rasterio.open(naip_fp) as src:
        dst_crs = src.crs

    # Project point to the CRS of the NAIP image
    point_proj = gpd.GeoSeries([point_wgs84], crs="EPSG:4326").to_crs(dst_crs).iloc[0]

    # Work out chip window in map coordinates
    with rasterio.open(naip_fp) as src:
        fx, fy = src.index(point_proj.x, point_proj.y, op=None) # Get pixel coordinates
        half = chip_size / 2 # Half chip size in pixels
        row_off = fy - half # Row offset in pixels
        col_off = fx - half # Column offset in pixels
        chip_win = Window(row_off, col_off, chip_size, chip_size) # Create a window for the chip based on pixel coordinates
        chip_transform = src.window_transform(chip_win)
        chip_left, chip_top = chip_transform * (0, 0) # Upper-left corner of the chip
        chip_right, chip_bottom = chip_transform * (chip_size, chip_size) # Lower-right corner of the chip
        # Create a bounding box for the chip in map coordinates to find any intersecting NAIP tiles
        bbox = box(min(chip_left, chip_right), min(chip_top, chip_bottom),
                   max(chip_left, chip_right), max(chip_top, chip_bottom))
        bbox_geom = gpd.GeoDataFrame(geometry=[bbox], crs=dst_crs)

    # Find all intersecting/overlapping NAIP tiles for this chip (some chips may span multiple tiles)
    bbox4326 = bbox_geom.to_crs(4326)
    overlapping_tiles = tileindex[tileindex.geometry.intersects(bbox4326.geometry.iloc[0])]

    if overlapping_tiles.empty:
        print(f"Warning: {lon},{lat} chip area not covered by NAIP.")
        return False

    naip_fps = [] # List of NAIP file paths to open
    for _, r in overlapping_tiles.iterrows(): # Open all overlapping tiles that exist
        fp = os.path.join(naip_folder, r['filename'])
        if os.path.exists(fp):
            naip_fps.append(fp) # Only add if file exists
    
    if not naip_fps:
        print(f"Warning, no downloaded NAIP tiles found for chip at {lon},{lat}.")
        return False
    
    # Open NAIP files that exist that overlap with chip area
    datasets = [rasterio.open(fp) for fp in naip_fps]

    # Enforce single-CRS mosaics (skip cross-UTM chips for simplicity, this is a very rare case)
    crs_set = set()
    for fp in naip_fps:
        with rasterio.open(fp) as ds:
            crs_set.add(ds.crs.to_string())
    
    if len(crs_set) > 1:
        print(f"Warning: Multiple CRS detected for chip at {lon},{lat}, skipping chip extraction.")
        for ds in datasets:
            ds.close()
        raise ValueError(f"Mixed CRS NAIP tiles for chip: {crs_set}")

    # Mosaic the NAIP tiles to cover the chip area
    try:
        mosaic, mosaic_transform = merge(
            datasets,
            bounds=(min(chip_left, chip_right), min(chip_top, chip_bottom),
                    max(chip_left, chip_right), max(chip_top, chip_bottom)),
            res=(datasets[0].res[0], datasets[0].res[1]),
            nodata=0
        )
    finally:
        # Close datasets after merging
        for ds in datasets:
            ds.close()

    # Extract chip sized window (should be 0,0 upper-left)
    chip = mosaic[:, 0:chip_size, 0:chip_size]
    chip_transform = from_origin(min(chip_left, chip_right), max(chip_top, chip_bottom),
                                 datasets[0].res[0], datasets[0].res[1])

    # Padding with 0s if necessary
    pad_y = chip_size - chip.shape[1]
    pad_x = chip_size - chip.shape[2]
    if pad_y > 0 or pad_x > 0:
        chip_padded = np.zeros((mosaic.shape[0], chip_size, chip_size), dtype=chip.dtype)
        chip_padded[:, :chip.shape[1], :chip.shape[2]] = chip
        chip = chip_padded

    # Save the image chip to a GeoTIFF file
    with rasterio.open(
        out_fp, 'w', driver='GTiff',
        height=chip.shape[1], width=chip.shape[2], count=chip.shape[0],
        dtype=chip.dtype, crs=dst_crs, transform=chip_transform
    ) as dst:
        dst.write(chip)
    return True

# ------------------------ Sample Uniform Train, Val, Test Splits ------------------------ #

# Split the presence/ pseudoabsence data into train, validation, and test sets
train_data = pa_data.sample(frac=0.7, random_state=42) # 70% for training
val_data = pa_data.drop(train_data.index).sample(frac=0.5, random_state=42) # 15% for validation
test_data = pa_data.drop(train_data.index).drop(val_data.index) # 15% for testing

# Add columns for train, val, test
train_data['set'] = 'train'
val_data['set'] = 'val'
test_data['set'] = 'test'

# Print train, val, test counts for presence/background points
print("Train/ Val/ Test Counts:")
print(train_data['presence'].value_counts())
print(val_data['presence'].value_counts())
print(test_data['presence'].value_counts())

# Combine all training, testing, and validation sets in a Geodataframe
geom_col = pa_data.geometry.name

pa_data_split = gpd.GeoDataFrame(
    pd.concat([train_data, val_data, test_data], ignore_index=True),
    geometry=geom_col,
    crs=pa_data.crs
)

print(pa_data_split.head())

# Save presence and background data to CSV with train/val/test splits
csv_out_fp = os.path.join(uniform_dataset_root, "Ailanthus_Pres_Bg_US_Uniform_Train_Val_Test_Points.csv")
pa_data_split.to_csv(csv_out_fp, index=False)
print(f"Saved CSV to {csv_out_fp}")




# ------------------------ Extract NAIP Image Chips ----------------------- #

# Create the output folder for NAIP image chips
output_naip_chips_folder = os.path.join(uniform_dataset_root, "images")
os.makedirs(output_naip_chips_folder, exist_ok=True)

chip_size = 256  # Size of the chip in pixels (256x256)

# Load tileindex of NAIP tiles
tileindex = gpd.read_file(tileindex_fp).to_crs(epsg=4326)

# Iterate over each point in the presence/ pseudo-absence data and extract NAIP chips
print("Extracting Image Chips from Points ... ")

for idx, row in pa_data_split.iterrows():
    lat, lon = row['lat'], row['lon']
    cell_id = idx # Use index as cell_id
    is_presence = row['presence']
    set_type = row['set']
    
    # Use a filename like "chip_{cellid}_pres_train.tif"
    chip_fn = f"chip_{cell_id}_{'pres' if is_presence else 'abs'}_{set_type}.tif"
    out_fp = os.path.join(output_naip_chips_folder, chip_fn)
    
    try:
        # Exract the chip for this point
        result = extract_naip_chip_for_point(lon, lat, out_fp, chip_size, tileindex, naip_folder)
        if result:
            continue
        else:
            print(f"Missing image chip coverage for point: {lon}, {lat}")
    except Exception as e:
        print(f"Failed image chip sampling for point: {lon}, {lat}: {e}")


# ------------------------ UNIFORM RANDOM: Sample Dataset with Chips and WorldClim Data ----------------------- #

# Initialize a list to hold the records
master_records = []

# Compute mean, std statistics for normalizing Worldclim data
print("Computing Worldclim Normalization Statistics ...")
wc_normalization_stats = compute_worldclim_stats(worldclim_folder, downloaded_union)

# Iterate over each point in the presence/ pseudo-absence data and create records based on the chips and WorldClim data
print("Extracting WorldClim and GHM variables for each point ...")

for idx, row in pa_data_split.iterrows():
    lat, lon = row['lat'], row['lon']
    cell_id = idx # Use index as cell_id
    presence = int(row['presence'])
    set_type = row['set']  # e.g. 'train', 'test', 'val'
    source = row.get('source', '')
    sample_id = f"{cell_id}_{presence}_{set_type}"

    # ---- NAIP Image Path (chip extraction should already have saved this chip) ----
    chip_fn = f"chip_{cell_id}_{'pres' if presence else 'abs'}_{set_type}.tif"
    chip_path = os.path.join(output_naip_chips_folder, chip_fn)

    # ---- WorldClim and GHM variables for this point ----
    wc_vars = extract_worldclim_vars_for_point(lon, lat, worldclim_folder, wc_normalization_stats)
    ghm_vars = extract_ghm_for_point(lon, lat, ghm_raster_fp)
    wc_vars['ghm'] = ghm_vars  # Add gHM value to WorldClim variables

    # Drop invalid climate records
    if not is_valid_climate_record(wc_vars):
        print(f"Skipping point {lon},{lat} due to invalid climate data.")
        continue

    # ---- Record ----- #
    record = {
        'sample_id': sample_id,
        'chip_path': os.path.relpath(chip_path, os.path.dirname(output_naip_chips_folder)), # relative path from .csv file
        'split': set_type,
        'presence': presence,
        'lat': lat,
        'lon': lon,
        'source': source
    }
    record.update(wc_vars)
    master_records.append(record)

# To DataFrame:
df_master = pd.DataFrame(master_records)

# Save the master DataFrame to CSV
csv_out_fp = os.path.join(uniform_dataset_root, "Ailanthus_Pres_Bg_US_Uniform_Train_Val_Test_Dataset.csv")

# Verify chip paths were created successfully
print("Verifying chip paths exist ...")
df_master['chip_path_abs'] = df_master['chip_path'].apply(lambda x: os.path.join(os.path.dirname(csv_out_fp), x))
df_master = df_master[df_master['chip_path_abs'].apply(os.path.exists)].drop(columns=['chip_path_abs'])

# Add columns: normalized lat, lon
df_master['lat_norm'] = (df_master['lat'] - df_master['lat'].mean()) / df_master['lat'].std()
df_master['lon_norm'] = (df_master['lon'] - df_master['lon'].mean()) / df_master['lon'].std()

# Lat/ Lon Mean/ Std Vars
print("Latitude Mean:", df_master['lat'].mean(), "Std:", df_master['lat'].std())
print("Longitude Mean:", df_master['lon'].mean(), "Std:", df_master['lon'].std())

# Save the DataFrame to CSV
df_master.to_csv(csv_out_fp, index=False)
print(f"Saved CSV to {csv_out_fp}")





# ------------------------ BLOCK SPATIAL CV Dataset with Chips and WorldClim Data ----------------------- #

chip_size = 256  # Size of the chip in pixels (256x256)

pa_data = pa_data_og # Use original data before train/val/test split

# Spatial join to get blocks for each point
pa_data = gpd.sjoin(pa_data, blocks[['block_id', 'folds', 'geometry']], how='inner', predicate='within').drop(columns=['index_right'])

# Normalize "folds" to be "fold" as integer column
pa_data["fold"] = pa_data["folds"].astype(int)
print(pa_data['fold'].value_counts())
print(pa_data.head(20))

# 5-fold CV config from block folds (1..5)
unique_folds = sorted(pa_data["fold"].unique())  # expect [1,2,3,4,5] if K=5
cv_configs = []
for f in unique_folds:
    train_folds = [x for x in unique_folds if x != f]
    cv_configs.append({
        "train_folds": train_folds,
        "test_fold": f,
        "cv_round": f,              # keep round number == held-out fold
        "test_label": f"blocks_fold_{f}"
    })

print(cv_configs)

# Create datasets for each CV round
cv_datasets = []
for config in cv_configs:
    train_val = pa_data[pa_data["fold"].isin(config["train_folds"])].copy()
    test = pa_data[pa_data["fold"] == config["test_fold"]].copy()

    # Random split of the training folds into train/val
    train = train_val.sample(frac=0.7, random_state=42)
    val = train_val.drop(train.index)

    # Label sets and carry round metadata
    train["set"] = "train"
    val["set"] = "val"
    test["set"] = "test"
    for df in (train, val, test):
        df["cv_round"] = config["cv_round"]
        df["test_block_fold"] = config["test_label"]

    cv_datasets.append(pd.concat([train, val, test], ignore_index=True))


# CV Dataset
pa_data_split_cv = pd.concat(cv_datasets, ignore_index=True)

# Save to CSV
csv_out_fp = os.path.join(crossval_dataset_root, "Ailanthus_Pres_Bg_US_SpatialCV_Train_Val_Test_Points.csv")
pa_data_split_cv.to_csv(csv_out_fp, index=False)
print(f"Saved CSV to {csv_out_fp}")

# Save separate CSV files for each CV round
for round_num in unique_folds: # e.g. 1,2,3,4,5
    round_df = pa_data_split_cv[pa_data_split_cv["cv_round"] == round_num]
    out_fp = os.path.join(
        crossval_dataset_root,
        f"Ailanthus_Pres_Bg_US_SpatialCV_Blocks_Fold_{round_num}_Jan26.csv"
    )
    round_df.to_csv(out_fp, index=False)
    print(f"Saved CV Fold {round_num} CSV to {out_fp}")
    
# Summary
fold_counts = pa_data.groupby(["fold", "presence"]).size().unstack(fill_value=0)
fold_counts.columns = ["Absence (0)", "Presence (1)"]
fold_counts["Total"] = fold_counts.sum(axis=1)
print("Presence/Pseudoabsence counts by block fold:\n", fold_counts)


### Sample NAIP chips and WorldClim variables for each point in the CV dataset ###

# Load tileindex and compute Worldclim normalization stats once
tileindex = gpd.read_file(tileindex_fp).to_crs(epsg=4326)
wc_normalization_stats = compute_worldclim_stats(worldclim_folder, downloaded_union)

# Open Climate Rasters
wc_datasets = {v: rasterio.open(os.path.join(worldclim_folder, f"{v}.tif")) for v in WC_VARS if v != 'ghm'}
# Open gHM raster
ghm_ds = rasterio.open(ghm_raster_fp)

# Loop through each CV round and extract data
for cv_round in [1, 2, 3, 4, 5]:
    print(f"\n=== Processing CV Round {cv_round} ===")

    # Load CSV for this round
    gdf = pd.read_csv(os.path.join(crossval_dataset_root, f"Ailanthus_Pres_Bg_US_SpatialCV_Blocks_Fold_{cv_round}_Jan26.csv"))

    # Create CV round-specific folder
    cv_dir = os.path.join(crossval_dataset_root, f"CV_{cv_round}")
    chip_dir = os.path.join(cv_dir, "images")
    os.makedirs(chip_dir, exist_ok=True)

    master_records = []

    for idx, row in tqdm.tqdm(gdf.iterrows(), total=len(gdf), desc=f"CV{cv_round}"):
        lat, lon = row['lat'], row['lon']
        cell_id = idx # Use index as cell_id
        presence = int(row['presence'])
        set_type = row['set']
        source = row.get('source', '')
        sample_id = f"{cell_id}_{presence}_{set_type}_CV{cv_round}"

        # Define chip output path
        chip_fn = f"chip_{cell_id}_{'pres' if presence else 'abs'}_{set_type}_CV{cv_round}.tif"
        chip_path = os.path.join(chip_dir, chip_fn)

        # Extract NAIP chip
        try:
            success = extract_naip_chip_for_point(lon, lat, chip_path, chip_size, tileindex, naip_folder)
        except Exception as e:
            print(f"Error extracting chip for point {lon}, {lat}: {e}")
            continue

        if not success:
            print(f"Missing chip for point {lon}, {lat}")
            continue

        # Extract WorldClim variables
        wc_vars = extract_worldclim_vars_for_point(lon, lat, wc_datasets, wc_normalization_stats)
        ghm_vars = extract_ghm_for_point(lon, lat, ghm_ds)
        wc_vars['ghm'] = ghm_vars  # Add gHM value to WorldClim variables

            # Drop invalid climate records
        if not is_valid_climate_record(wc_vars):
            print(f"Skipping point {lon},{lat} due to invalid climate data.")
            continue

        # Create sample record
        record = {
            'sample_id': sample_id,
            'chip_path': os.path.relpath(chip_path, cv_dir),
            'split': set_type,
            'presence': presence,
            'lat': lat,
            'lon': lon,
            'source': source,
            'cv_round': cv_round
        }
        record.update(wc_vars)
        master_records.append(record)

    # Save master CSV in CV round folder
    df_master = pd.DataFrame(master_records)
    csv_out_fp = os.path.join(cv_dir, f"Ailanthus_Train_Val_Test_US_BlockCV_{cv_round}_Jan26.csv")

    # Verify chip paths were created successfully
    print("Verifying chip paths exist ...")
    df_master['chip_path_abs'] = df_master['chip_path'].apply(lambda x: os.path.join(cv_dir, x))
    print(df_master[['chip_path', 'chip_path_abs']].head())
    df_master = df_master[df_master['chip_path_abs'].apply(os.path.exists)].drop(columns=['chip_path_abs'])

    # Add columns: normalized lat, lon
    df_master['lat_norm'] = (df_master['lat'] - df_master['lat'].mean()) / df_master['lat'].std()
    df_master['lon_norm'] = (df_master['lon'] - df_master['lon'].mean()) / df_master['lon'].std()

    df_master.to_csv(csv_out_fp, index=False)
    print(f"✓ Saved CV{cv_round} dataset to: {csv_out_fp}")


# -------------------- Close Raster Datasets -------------------- #

for ds in wc_datasets.values():
    ds.close()

ghm_ds.close()


# EOF