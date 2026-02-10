# This script samples presence and pseudo-absence data for Ailanthus altissima in North Carolina,
# creates random splits, and extracts NAIP image chips and WorldClim variables for each point.

# Thomas Lake, July 2025

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
import tqdm

### Paths for NAIP, Climate, Env, and Host Occurrence data ###

# Root Paths to Save Dataset 
dataset_name = "Ailanthus_Uniform_PA_NAIP_256_July25" 
dataset_root = rf"D:\Ailanthus_NAIP_Classification\Datasets\{dataset_name}"
os.makedirs(dataset_root, exist_ok=True)  # Create root directory if it doesn't exist

# Paths for NAIP data
tileindex_fp = r"D:\Ailanthus_NAIP_Classification\tileindex_NC_NAIP_2022\tileindex_NC_NAIP_2022.shp"
naip_folder = r"D:\Ailanthus_NAIP_Classification\NAIP_NC_4Band_1m"

# Paths for Host Occurrence Data
host_occurrence_fp = r"D:\Ailanthus_NAIP_Classification\Ailanthus_Occurrences\ailanthus_nc_inat_gbif_dedup_2015_2025.csv"

# Paths for Climate Data
worldclim_folder = r"D:\Ailanthus_NAIP_Classification\Env_Data\Worldclim"
wc_raster_fp = r"D:\Ailanthus_NAIP_Classification\Env_Data\Worldclim\wc2.1_30s_bio_1.tif"

# Path for Elevation DEM Data
elevation_raster_fp = r"D:\Ailanthus_NAIP_Classification\Env_Data\DEM_SRTM\nc_dem_srtm.tif"

# Path for Human Modification Data
ghm_raster_fp = r"D:\Ailanthus_NAIP_Classification\Env_Data\Global_Human_modification\gHM_WGS84.tif"

# States shapefile path
states_shapefile_path = r"D:\Ailanthus_NAIP_Classification\Env_Data\tl_2024_us_state\tl_2024_us_state.shp"
states = gpd.read_file(states_shapefile_path)
nc = states[states['NAME'] == 'NC']  # Filter for study area (North Carolina)

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
# And check that the sampled points are valid in both WorldClim and GHM rasters
print("Sampling Pseudo-Absences ...")
with rasterio.open(wc_raster_fp) as wc_src, rasterio.open(ghm_raster_fp) as ghm_src:
    wc_nodata = wc_src.nodata
    ghm_nodata = ghm_src.nodata
    wc_transform = wc_src.transform
    ghm_transform = ghm_src.transform

    while len(absences) < n_absences and attempts < max_attempts:
        x = random.uniform(downloaded_union.bounds[0], downloaded_union.bounds[2])
        y = random.uniform(downloaded_union.bounds[1], downloaded_union.bounds[3])
        pt = Point(x, y)

        if downloaded_union.contains(pt):
            wc_val = list(wc_src.sample([(x, y)]))[0][0]
            ghm_val = list(ghm_src.sample([(x, y)]))[0][0]

            # Check WC raster valid
            wc_invalid = (
                wc_val == wc_nodata or
                np.isnan(wc_val) or
                wc_val < -1e5 or
                wc_val > 1e5
            )

            # Check GHM raster valid (including tiny values near zero)
            ghm_invalid = (
                ghm_val is None or
                np.isnan(ghm_val) or
                (ghm_nodata is not None and ghm_val == ghm_nodata) or
                abs(ghm_val) < 1e-10
            )

            if wc_invalid or ghm_invalid:
                attempts += 1
                continue

            # Check climate cell uniqueness (you could choose either transform)
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

# Add lat/lon columns to pa_data from geometry
pa_data['lon'] = pa_data.geometry.x
pa_data['lat'] = pa_data.geometry.y

# Split the presence/ pseudoabsence data into train, validation, and test sets
train_data = pa_data.sample(frac=0.7, random_state=42) # 70% for training
val_data = pa_data.drop(train_data.index).sample(frac=0.5, random_state=42) # 15% for validation
test_data = pa_data.drop(train_data.index).drop(val_data.index) # 15% for testing

# Add columns for train, val, test
train_data['set'] = 'train'
val_data['set'] = 'val'
test_data['set'] = 'test'

# Combine all training, testing, and validation sets
pa_data_split = pd.concat([train_data, val_data, test_data], ignore_index=True)

# Save presence and pseudo-absence data to GeoJSON with train/val/test splits
geojson_out_fp = os.path.join(dataset_root, "Ailanthus_Pres_Pseudoabs_NC.geojson")
pa_data_split.to_file(geojson_out_fp, driver="GeoJSON")

print("Sampled Presence/ Pseudoabsence Data")
print(pa_data_split.head())


### Extract Data for Each Point ###

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

# Extract Global Human Modification (ghm) value for a given point
def extract_ghm_for_point(lon, lat, ghm_raster_fp):
    """
    Extract Global Human Modification (ghm) value for a given point (lon, lat) from the ghm raster.
    Values are already normalized to be between 0 and 1.
    0 = no human modification, 1 = maximum human modification.
    """
    with rasterio.open(ghm_raster_fp) as src:
        # Reproject raster to WGS84
        if src.crs.to_string() != 'EPSG:4326':
            src = src.reproject('EPSG:4326')
        # Get pixel coordinates of the point
        coords = (lon, lat)
        value = list(src.sample([coords]))[0][0]  # Get the first band value
        return value

def compute_dem_stats(dem_raster_fp):
    """
    Compute mean and standard deviation of the DEM raster.
    Normalization step prior to training models
    """
    with rasterio.open(dem_raster_fp) as src:
        data = src.read(1)  # Read the first band
        if src.nodata is not None:
            data = np.ma.masked_equal(data, src.nodata)
        else:
            data = np.ma.masked_where((data < -1e5) | (data > 1e5), data)

        mean = data.mean()
        std = data.std()

    return {"mean": float(mean), "std": float(std)}

# Extract DEM value for a given point
def extract_dem_for_point(lon, lat, dem_raster_fp, normalization_stats=None):
    """
    Extract DEM value for a given point (lon, lat) from the DEM raster.
    Normalizes the value using precomputed mean and std of the DEM band.
    """
    with rasterio.open(dem_raster_fp) as src:
        # Reproject raster to WGS84
        if src.crs.to_string() != 'EPSG:4326':
            src = src.reproject('EPSG:4326')
        # Get pixel coordinates of the point
        coords = (lon, lat)
        value = list(src.sample([coords]))[0][0]  # Get the first band value
        # Normalize the value using precomputed mean and std of the DEM band
        mean = normalization_stats['mean']
        std = normalization_stats['std']
        value_norm = (value - mean) / std  # Normalize
        return value_norm

# Extract NAIP chips for each point in the dataset
def extract_naip_chip_for_point(lon, lat, out_fp, chip_size, tileindex, naip_folder):
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

# Create the output folder for NAIP image chips
output_naip_chips_folder = os.path.join(dataset_root, "images")
os.makedirs(output_naip_chips_folder, exist_ok=True)

chip_size = 256  # Size of the chip in pixels (256x256)

# Load tileindex of NAIP tiles
tileindex = gpd.read_file(tileindex_fp).to_crs(epsg=4326)

# Iterate over each point in the presence/ pseudo-absence data and extract NAIP chips
print("Extracting Image Chips from Points ... ")
for idx, row in pa_data_split.iterrows():
    lat, lon = row['lat'], row['lon']
    cell_id = row['cell_id'] # Unique ID for the climate cell
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


### Create Master Dataset with Chips and WorldClim Data ###
# This section creates a master dataset that combines the NAIP image chips and WorldClim data for each point.

# Initialize a list to hold the records
master_records = []

# Compute mean, std statistics for normalizing Worldclim data
print("Computing Worldclim Normalization Statistics ...")
wc_normalization_stats = compute_worldclim_stats(worldclim_folder, downloaded_union)
dem_normalization_stats = compute_dem_stats(elevation_raster_fp)

# Iterate over each point in the presence/ pseudo-absence data and create records based on the chips and WorldClim data
print("Extracting WorldClim and GHM variables for each point ...")
for idx, row in tqdm.tqdm(pa_data_split.iterrows(), total=len(pa_data_split)):
    lat, lon = row['lat'], row['lon']
    cell_id = row['cell_id']
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
    dem_vars = extract_dem_for_point(lon, lat, elevation_raster_fp, dem_normalization_stats)
    wc_vars['ghm'] = ghm_vars  # Add gHM value to WorldClim variables
    wc_vars['dem'] = dem_vars  # Add DEM value to WorldClim variables

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
csv_out_fp = os.path.join(dataset_root, "Ailanthus_Train_Val_Test.csv")

# Filter out rows where the chip file does not exist (edge cases where UTM zones overlap and chip extraction failed)
print("Verifying chip paths exist ...")
df_master['chip_path_abs'] = df_master['chip_path'].apply(lambda x: os.path.join(os.path.dirname(csv_out_fp), x))
df_master = df_master[df_master['chip_path_abs'].apply(os.path.exists)].drop(columns=['chip_path_abs'])

# Save the DataFrame to CSV
df_master.to_csv(csv_out_fp, index=False)
print(f"Saved CSV to {csv_out_fp}")


# EOF
