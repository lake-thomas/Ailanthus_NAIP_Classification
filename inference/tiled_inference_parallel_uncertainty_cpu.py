# Inference script for Host Imagery + Climate Models
# Using Tiled Inference for Raster-Based Results
# Inference using Monte Carlo Dropout for Uncertainty Estimation
# Models: HostImageryClimateModel, HostClimateOnlyModel, HostImageryOnlyModel
# Version: CPU on HPC
# Thomas Lake, August 2025

import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import numpy as np #noqa: E402
import time #noqa: E402
import rasterio #noqa: E402
from rasterio.windows import Window #noqa: E402
from rasterio.mask import mask #noqa: E402
from rasterio.transform import from_origin #noqa: E402
from rasterio.warp import reproject, Resampling, calculate_default_transform #noqa: E402
from shapely.geometry import Point, box #noqa: E402
import geopandas as gpd #noqa: E402
from multiprocessing import Pool, cpu_count #noqa: E402

import torch #noqa: E402
from torch.utils.data import DataLoader #noqa: E402
from torchvision import transforms #noqa: E402

from model import HostImageryClimateModel, HostClimateOnlyModel, HostImageryOnlyModel #noqa: E402
from train_utils import get_default_device, load_model_from_checkpoint #noqa: E402

torch.set_num_threads(1)  # Each worker uses 1 thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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

# Enable dropout layers during inference for MC Dropout
def enable_mc_dropout(model):
    """Enable dropout layers during inference"""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

# Run model inference on a single NAIP tile
def run_inference_on_tile(tile_fp, output_dir, model, device, image_transform, wc_stats, worldclim_folder, ghm_raster_fp, chip_size=256, stride=64):
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
        sum_array = np.zeros((height, width), dtype=np.float32) # for mean prediction
        std_sum_array = np.zeros((height, width), dtype=np.float32) # For uncertainty estimation
        count_array = np.zeros((height, width), dtype=np.uint16)

        # Define rows and columns for the sliding window based on chip size and stride
        # This will create a grid of chips across the tile
        rows = range(0, height - chip_size + 1, stride)
        cols = range(0, width - chip_size + 1, stride)
        total_chips = len(rows) * len(cols)

        chip_iter = [(row, col) for row in rows for col in cols]

        # Iterate over each chip in the tile
        for row, col in chip_iter:
            # Create a window to read the chip
            window = Window(col_off=col, row_off=row, width=chip_size, height=chip_size)
            chip = src.read(window=window)

            # Get the center coordinates of the chip (lat, lon)
            center_x, center_y = transform * (col + chip_size // 2, row + chip_size // 2)
            lon, lat = rasterio.warp.transform(crs, "EPSG:4326", [center_x], [center_y])
            lon, lat = lon[0], lat[0] # Coordinates of the center of the chip

            # Extract environmental variables for the chip
            try:
                # Create tensors for chip and environment features
                chip_tensor = None
                env_tensor = None

                # Normalize and prepare environmental tensor if needed
                if isinstance(model, (HostImageryClimateModel, HostClimateOnlyModel)):
                    wc_vars = extract_worldclim_vars_for_point(lon, lat, worldclim_folder, wc_stats)
                    ghm = extract_ghm_for_point(lon, lat, ghm_raster_fp)
                    # dem = extract_dem_for_point(lon, lat, dem_raster_fp, dem_stats)
                    env_tensor = torch.tensor([[*wc_vars.values(), ghm]], dtype=torch.float32).to(device)

                # Normalize and prepare image tensor if needed
                if isinstance(model, (HostImageryClimateModel, HostImageryOnlyModel)):
                    chip = chip.astype(np.float32) / 255.0
                    chip_tensor = image_transform(np.moveaxis(chip, 0, -1)).unsqueeze(0).to(device)

                # Run inference with uncertainty (dropout enabled)
                T = 10 # Number of stochastic forward passes for uncertainty estimation
                probs = []

                with torch.no_grad():
                    for _ in range(T):
                        if isinstance(model, HostImageryClimateModel):
                            prob = model(chip_tensor, env_tensor)
                        elif isinstance(model, HostImageryOnlyModel):
                            prob = model(chip_tensor)
                        elif isinstance(model, HostClimateOnlyModel):
                            prob = model(env_tensor)
                        else:
                            raise NotImplementedError("Unknown model type used in inference.")
                        probs.append(prob.cpu().item()) # Store the probability for one forward pass

                probs = np.array(probs)
                mean_prob = probs.mean()
                std_prob = probs.std()

                sum_array[row:row + chip_size, col:col + chip_size] += float(mean_prob)
                std_sum_array[row:row + chip_size, col:col + chip_size] += float(std_prob)
                count_array[row:row + chip_size, col:col + chip_size] += 1

            except Exception as e:
                print(f"[{tile_name}] Error at row={row}, col={col}: {e}")
                continue
        
                
        # --- Compute mean and std ---
        avg_array = np.divide(
            sum_array,
            count_array,
            out=np.full_like(sum_array, np.nan, dtype=np.float32),
            where=(count_array > 0)
        )

        std_avg_array = np.divide(
            std_sum_array,
            count_array,
            out=np.full_like(std_sum_array, np.nan, dtype=np.float32),
            where=(count_array > 0)
        )

        # --- Reproject setup ---
        dst_crs = "EPSG:5070"
        dst_transform, dst_width, dst_height = calculate_default_transform(
            crs, dst_crs, src.width, src.height, *src.bounds
        )

        # --- Reproject average prediction to float32 ---
        reprojected_avg_array = np.full((dst_height, dst_width), -9999, dtype=np.float32)

        reproject(
            source=avg_array,
            destination=reprojected_avg_array,
            src_transform=src.transform,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=-9999
        )

        # --- Reproject standard deviation to float32 ---
        reprojected_std_array = np.full((dst_height, dst_width), -9999, dtype=np.float32)

        reproject(
            source=std_avg_array,
            destination=reprojected_std_array,
            src_transform=src.transform,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=-9999
        )

        # --- Common raster profile ---
        float_profile = profile.copy()
        float_profile.update({
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "width": dst_width,
            "height": dst_height,
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": -9999,
            "compress": "lzw"
        })

        # --- Save mean prediction raster ---
        with rasterio.open(output_fp, "w", **float_profile) as dst:
            dst.write(reprojected_avg_array, 1)

        print(f"[{tile_name}] Saved float32 prediction to {output_fp}")

        # --- Save standard deviation (uncertainty) raster ---
        std_output_fp = os.path.join(output_dir, f"{tile_name}_uncertainty.tif")

        with rasterio.open(std_output_fp, "w", **float_profile) as dst:
            dst.write(reprojected_std_array, 1)

        print(f"[{tile_name}] Saved float32 uncertainty to {std_output_fp}")

        elapsed = time.time() - start_time
        print(f"[{tile_name}] Saved prediction to {output_fp} in {elapsed:.2f} seconds", flush=True)


def run_tile_worker(args):
    tile_fp, output_dir, env_vars, wc_stats,  worldclim_folder, ghm_raster_fp, chip_size, stride = args
    
    image_transform = transforms.Compose([transforms.ToTensor()])

    checkpoint_path = r"D:\Ailanthus_NAIP_Classification\NAIP_Host_Model\outputs\ailanthus_image_climate_pa_1300m_thin_aug2125\checkpoints\checkpoint_epoch_28.tar"
    
    # Load model fresh for each process (models are not fork-safe!)
    model, _ = load_model_from_checkpoint(checkpoint_path, env_vars, hidden_dim=256, dropout=0.25)
    model.eval() # Keeps BatchNorm layers in eval mode
    enable_mc_dropout(model) # Activates dropout for uncertainty estimation
    
    # Set device to CPU for inference
    device = torch.device("cpu")
    model.to(device)
    print(f"Loaded model type: {model.__class__.__name__}")
    print(f"Loaded model from {checkpoint_path} on device {device}")
    
    try:
        run_inference_on_tile(
            tile_fp=tile_fp,
            output_dir=output_dir,
            model=model,
            device=device,
            image_transform=image_transform,
            wc_stats=wc_stats,
            worldclim_folder=worldclim_folder,
            ghm_raster_fp=ghm_raster_fp,
            chip_size=chip_size,
            stride=stride
        )
    except Exception as e:
        print(f"[ERROR] Failed on {os.path.basename(tile_fp)}: {e}")

def main():

    # Paths
    tileindex_fp = r"D:\Ailanthus_NAIP_Classification\tileindex_NC_NAIP_2022\tileindex_NC_NAIP_2022.shp"
    naip_folder = r"D:\Ailanthus_NAIP_Classification\NAIP_NC_4Band_1m"
    worldclim_folder = r"D:\Ailanthus_NAIP_Classification\Env_Data\WorldClim"
    ghm_raster_fp = r"D:\Ailanthus_NAIP_Classification\Env_Data\Global_Human_Modification\gHM_WGS84.tif"
    output_dir = r"C:\Users\talake2\Desktop\NAIP_Inference_Parallel_Test"

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load NAIP data from paths to compute normalization statistics based on available data
    tileindex = gpd.read_file(tileindex_fp).to_crs('EPSG:4326')
    tileindex['filename'] = tileindex['filename'].apply(os.path.basename)
    naip_files = set(os.listdir(naip_folder))
    tileindex['status'] = tileindex['filename'].apply(lambda x: "Downloaded" if x in naip_files else "Not Downloaded")
    downloaded_tiles = tileindex[tileindex['status'] == 'Downloaded']
    downloaded_union = downloaded_tiles.unary_union # Shapefile union of all downloaded NAIP tiles

    # Environment variables used by the model
    # env_vars = ["lat", "lon"] + [f"wc2.1_30s_bio_{i}" for i in range(1, 20)] + ["ghm", "dem"]
    env_vars = [f"wc2.1_30s_bio_{i}" for i in range(1, 20)] + ["ghm"]
    print("Length of env_vars:", len(env_vars))

    # Compute mean, std statistics for normalizing Worldclim features
    print("Computing Worldclim Normalization Statistics ...")
    wc_stats = compute_worldclim_stats(worldclim_folder, downloaded_union)

    # Get all NAIP .tif files
    naip_files = [os.path.join(naip_folder, f) for f in os.listdir(naip_folder) if f.endswith(".tif")]
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

    # Build argument list  
    chip_size = 256
    stride = 64

    args_list = [
        (
            tile_fp, output_dir,
            env_vars, wc_stats,  worldclim_folder,
            ghm_raster_fp, chip_size, stride
        )
        for tile_fp in naip_files ##### Change to missing_tiles if needed
    ]

    # Parallel execution
    p = 8 # num cores to use 
    print(f"Running inference with {p} of {cpu_count()} available CPU cores...")
    
    with Pool(processes=p) as pool:
        pool.map(run_tile_worker, args_list)


if __name__ == "__main__":
    main()