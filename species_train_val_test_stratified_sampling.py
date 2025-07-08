# This script samples presence and pseudo-absence points for Ailanthus altissima based on NAIP tile coverage and WorldClim raster data.
# It ensures one presence and one pseudo-absence per unique climate cell.

# Imports
import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol, from_origin
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

### Paths for NAIP, Climate, and Host Occurrence data ###

# Paths for NAIP data
tileindex_fp = r"D:\Ailanthus_NAIP_Classification\tileindex_NC_NAIP_2022\tileindex_NC_NAIP_2022.shp"
naip_folder = r"D:\Ailanthus_NAIP_Classification\NAIP_NC_4Band"

# Paths for Host Occurrence Data
host_occurrence_fp = r"D:\Ailanthus_NAIP_Classification\Ailanthus_Occurrences\ailanthus_nc_inat_gbif_dedup_2015_2025.csv"

# Paths for Climate Data
worldclim_folder = r"D:\Ailanthus_NAIP_Classification\Env_Data\Worldclim"
wc_raster_fp = r"D:\Ailanthus_NAIP_Classification\Env_Data\Worldclim\wc2.1_30s_bio_1.tif"

# States shapefile path
states_shapefile_path = r"D:\Ailanthus_NAIP_Classification\Env_Data\tl_2024_us_state\tl_2024_us_state.shp"
states = gpd.read_file(states_shapefile_path)
nc = states[states['NAME'] == 'North Carolina']  # Filter for study area (North Carolina)

# Load NAIP data from paths
tileindex = gpd.read_file(tileindex_fp).to_crs('EPSG:4326')
tileindex['filename'] = tileindex['filename'].apply(os.path.basename)
naip_files = set(os.listdir(naip_folder))
tileindex['status'] = tileindex['filename'].apply(lambda x: "Downloaded" if x in naip_files else "Not Downloaded")
downloaded_tiles = tileindex[tileindex['status'] == 'Downloaded']
downloaded_union = downloaded_tiles.unary_union # Shapefile union of all downloaded NAIP tiles

# Load climate raster
with rasterio.open(wc_raster_fp) as src:
    wc_transform = src.transform
    wc_crs = src.crs # EPSG:4326

### Presence and Pseudo-absence Sampling ###

# Load host occurrence points
presence = pd.read_csv(host_occurrence_fp)
presence['geometry'] = gpd.points_from_xy(presence['lon'], presence['lat'])
presence = gpd.GeoDataFrame(presence, geometry='geometry', crs='EPSG:4326')

# Filter occurrences to those with NAIP coverage
presence = presence[presence.geometry.within(downloaded_union)]
presence['species'] = 'Ailanthus altissima'
presence['presence'] = 1
presence = presence.to_crs('EPSG:4326') # Reproject to (EPSG:4326)

# Function to compute unique cell ID for WorldClim raster
def get_cell_id(point):
    row, col = rowcol(wc_transform, point.x, point.y)
    return f"{row}_{col}"

# Assign climate cell ID and drop duplicate occurrences
# This will ensure only one presence or pseudo-absence point per unique climate cell
presence['cell_id'] = presence.geometry.apply(get_cell_id)
presence_unique = presence.drop_duplicates(subset='cell_id')
print(f"Filtered to {len(presence_unique)} unique presence cells.")

# Sample pseudo-absences within NAIP tile coverage
n_absences = len(presence_unique) # Sample equal number of absences as presences
used_cells = set(presence_unique['cell_id'])
absences = []
attempts = 0
max_attempts = n_absences * 20  # fail-safe

# Randomly sample points within the area of downloaded NAIP tiles
# Ensure we do not sample the same climate cell as a presence
with rasterio.open(wc_raster_fp) as wc_src:
    wc_nodata = wc_src.nodata

    # Iterate until we have enough pseudo-absences
    while len(absences) < n_absences and attempts < max_attempts:
        # Sample within bounds of the downloaded NAIP tiles
        x = random.uniform(downloaded_union.bounds[0], downloaded_union.bounds[2])
        y = random.uniform(downloaded_union.bounds[1], downloaded_union.bounds[3])
        pt = Point(x, y)
        if downloaded_union.contains(pt):
            # Sample climate data at the point to verify the data are valid
            value = list(wc_src.sample([(x, y)]))[0][0]
            # print(value)
            if (value == wc_nodata or np.isnan(value) or value < -1e5 or value > 1e5):
                attempts +=1
                # print("Found Bad Value - Not Using That Point")
                continue

            # Check climate cell ID and add if unique
            r, c = rowcol(wc_transform, pt.x, pt.y)
            cell_id = f"{r}_{c}"
            if cell_id not in used_cells:
                absences.append({
                    'geometry': pt,
                    'species': 'Ailanthus altissima',
                    'presence': 0,
                    'cell_id': cell_id,
                    'source': 'pseudo-absence',
                })
                used_cells.add(cell_id)
        attempts += 1

# Convert absences to GeoDataFrame
absence = gpd.GeoDataFrame(absences, crs='EPSG:4326')

# Combine presence and pseudoabsence data, ensure it's in 'EPSG:4326'
pa_data = pd.concat([presence_unique, absence], ignore_index=True)
pa_data = pa_data.to_crs('EPSG:4326')

# Add lat/lon columns
pa_data['lon'] = pa_data.geometry.x
pa_data['lat'] = pa_data.geometry.y

# Assign cross-validation folds based on longitude (approximate ranges for NC)
def assign_longitude_fold(lon):
    if lon < -81:        # Western NC
        return 0
    elif lon < -78:      # Central NC
        return 1
    else:                # Eastern NC
        return 2

pa_data['fold'] = pa_data['lon'].apply(assign_longitude_fold)

# --- Create 3 cross-validation rounds using fixed training regions ---
cv_configs = [
    {'train_folds': [0, 1], 'test_fold': 2, 'cv_round': 1},
    {'train_folds': [1, 2], 'test_fold': 0, 'cv_round': 2},
    {'train_folds': [0, 2], 'test_fold': 1, 'cv_round': 3}
]

cv_datasets = []

for config in cv_configs:
    train_folds = config['train_folds']
    test_fold = config['test_fold']
    round_num = config['cv_round']
    
    # Split data
    train_val = pa_data[pa_data['fold'].isin(train_folds)].copy()
    test = pa_data[pa_data['fold'] == test_fold].copy()
    
    # Randomly split train_val into train and validation sets
    train = train_val.sample(frac=0.7, random_state=42)
    val = train_val.drop(train.index)

    # Assign set labels
    train['set'] = 'train'
    val['set'] = 'val'
    test['set'] = 'test'

    # Add CV round index
    for df in [train, val, test]:
        df['cv_round'] = round_num

    # Combine
    cv_datasets.append(pd.concat([train, val, test], ignore_index=True))

# Concatenate all CV rounds into one DataFrame
pa_data_split = pd.concat(cv_datasets, ignore_index=True)
print(pa_data_split.head(10))

# Save to GeoJSON
pa_data_split.to_file(
    r"D:\Ailanthus_NAIP_Classification\Ailanthus_Occurrences\Ailanthus_Pres_Pseudoabs_NC_July25_FixedTrainCV.geojson",
    driver="GeoJSON"
)

# Save separate GeoJSON files for each CV round
for round_num in [1, 2, 3]:
    round_df = pa_data_split[pa_data_split["cv_round"] == round_num]
    out_fp = rf"D:\Ailanthus_NAIP_Classification\Ailanthus_Occurrences\Ailanthus_Pres_Pseudoabs_NC_CV{round_num}.geojson"
    round_df.to_file(out_fp, driver="GeoJSON")
    print(f"Saved CV Round {round_num} to: {out_fp}")

# --- Summary ---
print("Sampled Presence/ Pseudoabsence Data with 3 CV rounds")
print(pa_data_split.head())

# --- Count presence and absence per fold for reference ---
fold_counts = pa_data.groupby(['fold', 'presence']).size().unstack(fill_value=0)
fold_counts.columns = ['Absence (0)', 'Presence (1)']
fold_counts['Total'] = fold_counts.sum(axis=1)
fold_counts = fold_counts.rename_axis("Fold").reset_index()
print("Presence/Pseudoabsence counts by fold:\n")
print(fold_counts.to_string(index=False))

# # Define consistent colors
# set_colors = {
#     'train': '#1f77b4',  # blue
#     'val': '#ff7f0e',    # orange
#     'test': '#2ca02c'    # green
# }

# # Plot each CV round
# fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharex=True, sharey=True)

# for i, round_num in enumerate([1, 2, 3]):
#     ax = axes[i]
#     cv_subset = pa_data_split[pa_data_split['cv_round'] == round_num].copy()

#     # Force lowercase set labels to avoid key errors
#     cv_subset['set'] = cv_subset['set'].str.lower()

#     for set_type in ['train', 'val', 'test']:
#         if set_type in cv_subset['set'].unique():
#             points = cv_subset[cv_subset['set'] == set_type]
#             points.plot(ax=ax,
#                         markersize=4,
#                         alpha=0.7,
#                         label=set_type.capitalize(),
#                         color=set_colors[set_type])
    
#     # Optional: Plot state boundary for context
#     nc.boundary.plot(ax=ax, color='black', linewidth=0.5)

#     ax.set_title(f"CV Round {round_num}", fontsize=14)
#     ax.set_xlabel("Longitude")
#     if i == 0:
#         ax.set_ylabel("Latitude")
#     ax.grid(False)
#     ax.legend(title="Dataset", loc='upper right')

# plt.suptitle("Spatial CV Assignments for Ailanthus Classification", fontsize=18)
# plt.tight_layout()
# plt.show()


### Extract Climate and NAIP Data for Each Point ###

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

# Extract WorldClim data for each point in the dataset
def extract_worldclim_vars_for_point(lon, lat, worldclim_folder, normalization_stats=None):
    """
    Extract WorldClim variables for a given point (lon, lat).
    """
    vals = {}
    for n in range(1, 20): # Open each of 19 bioclim variables and extract value at point
        # Construct the file path for the WorldClim raster
        fp = os.path.join(worldclim_folder, f"wc2.1_30s_bio_{n}.tif")
        varname = f"wc2.1_30s_bio_{n}"
        with rasterio.open(fp) as src:
            coords = (lon, lat)
            value = list(src.sample([coords]))[0][0]  # Get the first band value
            # Normalize the value using precomputed mean and std of each WC band
            mean = normalization_stats[varname]['mean']
            std = normalization_stats[varname]['std']
            value_norm = (value - mean) / std # Normalize
            vals[varname] = value_norm
    return vals

# Extract NAIP chips for each point in the dataset
def extract_chip_for_point(lon, lat, out_fp, chip_size, tileindex, naip_folder):
    """
    Extract a chip of size `chip_size` around the point (lon, lat) from NAIP tiles.
    Based on the tile index, it finds the appropriate NAIP tiles that cover the point.
    Saves the chip as a GeoTIFF to `out_fp`.
    Returns True if successful, False if point is not covered by NAIP.
    """
    # WGS84 point (lon, lat)
    point_wgs84 = Point(lon, lat)

    # Find *any* matching tile to get image CRS (all NAIP images should be in same CRS per region)
    matches = tileindex[tileindex.geometry.contains(point_wgs84)]
    if matches.empty:
        raise ValueError(f"Point {lon},{lat} not in any NAIP tile.")
    
    # Use the first matching tile to get the CRS
    # To Do: Handle case where one point can match tiles with multiple CRS (e.g., UTM17 and UTM18)
    naip_fp = os.path.join(naip_folder, matches.iloc[0]['filename'])
    
    # Open the NAIP image to get its CRS
    with rasterio.open(naip_fp) as src:
        dst_crs = src.crs
        # print(dst_crs) # Print the CRS of the NAIP image

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

    # Find all intersecting NAIP tiles for this chip
    bbox4326 = bbox_geom.to_crs(4326)
    overlapping_tiles = tileindex[tileindex.geometry.intersects(bbox4326.geometry.iloc[0])]
    if overlapping_tiles.empty:
        print(f"Warning: {lon},{lat} chip area not covered by NAIP.")
        return False

    # Load/mosaic the overlapping NAIP tiles to sample the chip
    # Use rasterio.merge to combine the overlapping tiles
    naip_fps = [os.path.join(naip_folder, r['filename']) for _, r in overlapping_tiles.iterrows()]
    datasets = [rasterio.open(fp) for fp in naip_fps]
    mosaic, mosaic_transform = merge(
        datasets,
        bounds=(min(chip_left, chip_right), min(chip_top, chip_bottom),
                max(chip_left, chip_right), max(chip_top, chip_bottom)),
        res=(datasets[0].res[0], datasets[0].res[1]),
        nodata=0
    )
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

### Sample NAIP chips and WorldClim variables for each point in the CV dataset ###
# Paths
cv_geojson_base = r"D:\Ailanthus_NAIP_Classification\Ailanthus_Occurrences"
output_base = r"D:\Ailanthus_NAIP_Classification"
chip_size = 256

# Load tileindex and compute Worldclim normalization stats once
tileindex = gpd.read_file(tileindex_fp).to_crs(epsg=4326)
wc_stats = compute_worldclim_stats(worldclim_folder, downloaded_union)

# Loop through each CV round and extract data
for cv_round in [1, 2, 3]:
    print(f"\n=== Processing CV Round {cv_round} ===")

    # Load GeoJSON for this round
    gdf = gpd.read_file(os.path.join(cv_geojson_base, f"Ailanthus_Pres_Pseudoabs_NC_CV{cv_round}.geojson"))

    # Create CV round-specific folder
    cv_dir = os.path.join(output_base, f"CV{cv_round}")
    chip_dir = os.path.join(cv_dir, "chips")
    os.makedirs(chip_dir, exist_ok=True)

    master_records = []

    for idx, row in tqdm.tqdm(gdf.iterrows(), total=len(gdf), desc=f"CV{cv_round}"):
        lat, lon = row['lat'], row['lon']
        cell_id = row['cell_id']
        presence = int(row['presence'])
        set_type = row['set']
        source = row.get('source', '')
        sample_id = f"{cell_id}_{presence}_{set_type}_CV{cv_round}"

        # Define chip output path
        chip_fn = f"chip_{cell_id}_{'pres' if presence else 'abs'}_{set_type}_CV{cv_round}.tif"
        chip_path = os.path.join(chip_dir, chip_fn)

        # Extract NAIP chip
        try:
            success = extract_chip_for_point(lon, lat, chip_path, chip_size, tileindex, naip_folder)
        except Exception as e:
            print(f"Error extracting chip for point {lon}, {lat}: {e}")
            continue

        if not success:
            print(f"Missing chip for point {lon}, {lat}")
            continue

        # Extract WorldClim variables
        wc_vars = extract_worldclim_vars_for_point(lon, lat, worldclim_folder, wc_stats)

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
    df_out = pd.DataFrame(master_records)
    out_csv = os.path.join(cv_dir, f"Ailanthus_NC_CV{cv_round}_Master.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"✓ Saved CV{cv_round} dataset to: {out_csv}")



# EOF