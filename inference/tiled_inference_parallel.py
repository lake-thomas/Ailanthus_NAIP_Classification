# Inference script for HostImageryClimateModel 
# Parallelized Tiled Inference for Raster-Based Results
# Thomas Lake, July 2025

import os
import numpy as np
import time
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
from rasterio.transform import from_origin
from shapely.geometry import Point, box
import geopandas as gpd
from multiprocessing import Pool, cpu_count

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

# Run model inference on a single NAIP tile
def run_inference_on_tile(tile_fp, output_dir, model, device, image_transform, 
                          wc_stats, dem_stats, worldclim_folder, ghm_raster_fp, 
                          dem_raster_fp, chip_size=256, stride=64):
    """
    Run tiled inference on a NAIP tile using the provided model.
    Args:
        tile_fp (str): Path to the NAIP tile file.
        output_dir (str): Directory to save the output predictions.
        model (torch.nn.Module): The trained model for inference.
        device (torch.device): Device to run the model on (CPU or GPU).
        image_transform (callable): Transform to apply to the NAIP images.
        wc_stats (dict): Normalization statistics for WorldClim variables.
        dem_stats (dict): Normalization statistics for DEM.
        worldclim_folder (str): Folder containing WorldClim raster files.
        ghm_raster_fp (str): Path to the Global Human Modification raster file.
        dem_raster_fp (str): Path to the DEM raster file.
        chip_size (int): Size of the chips to extract from the tile.
        stride (int): Stride for moving the chip window across the tile.
    """
    # Track inference time per NAIP tile
    start_time = time.time()

    tile_name = os.path.splitext(os.path.basename(tile_fp))[0]
    output_fp = os.path.join(output_dir, f"{tile_name}_predictions.tif")

    # Read the NAIP tile
    with rasterio.open(tile_fp) as src:
        profile = src.profile
        width, height = src.width, src.height
        transform = src.transform
        crs = src.crs

        # Initialize arrays to accumulate predictions
        # sum_array will hold the sum of predictions, count_array will hold the number of chips contributing to each pixel
        # This allows us to average the predictions later
        sum_array = np.zeros((height, width), dtype=np.float32)
        count_array = np.zeros((height, width), dtype=np.uint16)

        # Define rows and columns for the sliding window based on chip size and stride
        # This will create a grid of chips across the tile
        rows = range(0, height - chip_size + 1, stride)
        cols = range(0, width - chip_size + 1, stride)
        total_chips = len(rows) * len(cols)

        chip_iter = tqdm(
            [(row, col) for row in rows for col in cols],
            desc=f"[{tile_name}]",
            total=total_chips,
            leave=False
        )

        # Iterate over each chip in the tile
        for row, col in chip_iter:
            # Create a window to read the chip
            window = Window(col_off=col, row_off=row, width=chip_size, height=chip_size)
            chip = src.read(window=window)

            # Get the center coordinates of the chip (lat, lon)
            center_x, center_y = transform * (col + chip_size // 2, row + chip_size // 2)
            lon, lat = rasterio.warp.transform(crs, "EPSG:4326", [center_x], [center_y])
            lon, lat = lon[0], lat[0]

            # Extract environmental variables for the chip
            try:
                wc_vars = extract_worldclim_vars_for_point(lon, lat, worldclim_folder, wc_stats)
                ghm = extract_ghm_for_point(lon, lat, ghm_raster_fp)
                dem = extract_dem_for_point(lon, lat, dem_raster_fp, dem_stats)
                env_tensor = torch.tensor([[lat, lon, *wc_vars.values(), ghm, dem]], dtype=torch.float32)

                # Normalize the chip and covert to tensor
                chip = chip.astype(np.float32) / 255.0

                chip_tensor = image_transform(np.moveaxis(chip, 0, -1)).unsqueeze(0)

                chip_tensor = chip_tensor.to(device)
                env_tensor = env_tensor.to(device)

                # Run inference
                with torch.no_grad():
                    logit = model(chip_tensor, env_tensor)
                    prob = torch.sigmoid(logit).item()
                    #print(f"Prob: {prob:.3f}, Lat: {lat:.4f}, Lon: {lon:.4f}, GHM: {ghm:.2f}, DEM: {dem:.2f}, WC1: {wc_vars['wc2.1_30s_bio_2']:.2f}")

                # Update the sum and count arrays
                # This allows us to average the predictions later
                sum_array[row:row + chip_size, col:col + chip_size] += prob
                count_array[row:row + chip_size, col:col + chip_size] += 1

            except Exception as e:
                print(f"[{tile_name}] Error at row={row}, col={col}: {e}")
                continue
        
        # Average the predictions where count > 0
        # This avoids division by zero and ensures we only average valid predictions
        avg_array = np.divide(
            sum_array,
            count_array,
            out=np.full_like(sum_array, -9999, dtype=np.float32),
            where=(count_array > 0)
        )

        out_profile = profile.copy()
        out_profile.update({
            "dtype": "float32",
            "count": 1,
            "nodata": -9999
        })
        
        # Save the averaged predictions to a new raster file
        with rasterio.open(output_fp, "w", **out_profile) as dst:
            dst.write(avg_array, 1)

        # print(f"[{tile_name}] Saved prediction to {output_fp}")

        elapsed = time.time() - start_time
        # print(f"[{tile_name}] Saved prediction to {output_fp} in {elapsed:.2f} seconds")

def run_tile_worker(args):
    tile_fp, output_dir, model_path, env_vars, wc_stats, dem_stats, worldclim_folder, ghm_raster_fp, dem_raster_fp, chip_size, stride = args
    
    device = get_default_device()
    image_transform = transforms.Compose([transforms.ToTensor()])
    
    # Load model fresh for each process (models are not fork-safe!)
    model, _ = load_model_from_checkpoint(model_path, env_vars)
    model.to(device).eval()
    
    try:
        run_inference_on_tile(
            tile_fp=tile_fp,
            output_dir=output_dir,
            model=model,
            device=device,
            image_transform=image_transform,
            wc_stats=wc_stats,
            dem_stats=dem_stats,
            worldclim_folder=worldclim_folder,
            ghm_raster_fp=ghm_raster_fp,
            dem_raster_fp=dem_raster_fp,
            chip_size=chip_size,
            stride=stride
        )
    except Exception as e:
        print(f"[ERROR] Failed on {os.path.basename(tile_fp)}: {e}")

def main():

    # Paths
    naip_dir = r"D:\Ailanthus_NAIP_Classification\NAIP_NC_4Band_1m" # NAIP tiles for inference
    # naip_folder = r"D:\Ailanthus_NAIP_Classification\NAIP_NC_4Band_1m" # All NAIP files for calculating normalization stats
    output_dir = r"D:\Ailanthus_NAIP_Classification\model_inference"
    os.makedirs(output_dir, exist_ok=True)

    tileindex_fp = r"D:\Ailanthus_NAIP_Classification\tileindex_NC_NAIP_2022\tileindex_NC_NAIP_2022.shp"
    worldclim_folder = r"D:\Ailanthus_NAIP_Classification\Env_Data\WorldClim"
    ghm_raster_fp = r"D:\Ailanthus_NAIP_Classification\Env_Data\Global_Human_Modification\gHM_WGS84.tif"
    dem_raster_fp = r"D:\Ailanthus_NAIP_Classification\Env_Data\DEM_SRTM\nc_dem_srtm.tif"

    model_path = r"D:\Ailanthus_NAIP_Classification\NAIP_Host_Model\outputs\ailanthus_naip_climate_dem_ghm_lat_lon_30ep_july1725\checkpoints\checkpoint_epoch_29.tar"
    env_vars = ["lat", "lon"] + [f"wc2.1_30s_bio_{i}" for i in range(1, 20)] + ["ghm", "dem"]

    # Get all NAIP .tif files
    naip_files = [os.path.join(naip_dir, f) for f in os.listdir(naip_dir) if f.endswith(".tif")]
    print(f"Found {len(naip_files)} NAIP tiles to process.")

    # ###### Restart infernece with only missing NAIP .tif files #####
    # # Get all original NAIP tile base names (e.g., m_3307701_ne_18_060_20220923_20221207)
    # naip_tiles = [os.path.splitext(f)[0] for f in os.listdir(naip_dir) if f.endswith(".tif")]
    # # Get all completed prediction tile base names (remove _predictions suffix)
    # predicted_tiles = [f.replace("_predictions", "").replace(".tif", "") for f in os.listdir(output_dir) if f.endswith("_predictions.tif")]
    # # Find which tiles are missing
    # missing_tiles = sorted(list(set(naip_tiles) - set(predicted_tiles)))
    # print(f"Missing {len(missing_tiles)} tiles")
    # # Get full paths for the missing tiles
    # missing_tile_paths = [os.path.join(naip_dir, f + ".tif") for f in missing_tiles]
    # print(f"Running inference on {len(missing_tile_paths)} missing tiles")

    # Load NAIP data from paths to compute normalization stats by area
    tileindex = gpd.read_file(tileindex_fp).to_crs('EPSG:4326')
    tileindex['filename'] = tileindex['filename'].apply(os.path.basename)
    all_naip_files = set(os.listdir(naip_dir))
    tileindex['status'] = tileindex['filename'].apply(lambda x: "Downloaded" if x in all_naip_files else "Not Downloaded")
    downloaded_tiles = tileindex[tileindex['status'] == 'Downloaded']
    downloaded_union = downloaded_tiles.unary_union # Shapefile union of all downloaded NAIP tiles

    # Precompute global stats (these are passed to each process)
    wc_stats = compute_worldclim_stats(worldclim_folder, downloaded_union)
    dem_stats = compute_dem_stats(dem_raster_fp)

    # Build argument list  
    chip_size = 256
    stride = 256

    args_list = [
        (
            tile_fp, output_dir, model_path, env_vars,
            wc_stats, dem_stats, worldclim_folder,
            ghm_raster_fp, dem_raster_fp,
            chip_size, stride
        )
        for tile_fp in missing_tile_paths ##### Change to naip_files if you want to run all NAIP tiles
    ]

    # Parallel execution
    print(f"Running inference with {cpu_count()} CPU cores...")
    
    with Pool(processes=14) as pool:
        pool.map(run_tile_worker, args_list)


if __name__ == "__main__":
    main()