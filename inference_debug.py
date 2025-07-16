# Simple script to debug inference of HostImageryClimateModel


import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.transform import from_origin
from shapely.geometry import Point, box
import geopandas as gpd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import HostImageryClimateModel
from train_utils import get_default_device, load_model_from_checkpoint


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

# ---------------------------
# Sample Input Data
# ---------------------------
image_path = r"D:\Ailanthus_NAIP_Classification\Datasets\Ailanthus_Uniform_PA_NAIP_256_July25\images\chip_6501_12107_pres_train.tif"
env_vector = [
    35.820835135, -79.1031097994, 0.105885657, 0.618426961, 0.193383286, 0.616368778, 0.487954217, -0.118006432, 0.79391867,
    0.426719939, -0.121785098, 0.254240254, -0.021807718, -0.604288304, -1.071244666,
    -0.167968716, -0.770878302, -0.994858261, -0.162620649, -0.957474927, -0.384644353,
    0.51435894, -0.171986967
]
label = 1  # True label (for sanity check)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load Image
# ---------------------------
def load_image(image_path):
    with rasterio.open(image_path) as src:
        img = src.read()  # shape: (bands, H, W)
    img = img.astype(np.float32) / 255.0  # normalize to 0-1
    return torch.tensor(img)

# ---------------------------
# Load Model and Weights
# ---------------------------

checkpoint_path = r"D:\Ailanthus_NAIP_Classification\NAIP_Host_Model\outputs\ailanthus_naip_test\checkpoints\checkpoint_epoch_4.tar"
model, _ = load_model_from_checkpoint(checkpoint_path, env_vector)
model.eval()

# ---------------------------
# Run Inference
# ---------------------------
img_tensor = load_image(image_path).unsqueeze(0).to(device)  # shape: (1, 4, 256, 256)
env_tensor = torch.tensor(env_vector, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, 22)

with torch.no_grad():
    output = model(img_tensor, env_tensor)
    prob = torch.sigmoid(output).item()
    pred = int(prob > 0.5)

print(f"Known label: {label}")
print(f"Model output (logit): {output.item():.4f}")
print(f"Model output (probability): {prob:.4f}")
print(f"Predicted label: {pred}")
