# Inference script CNN-SDM Models
# Using Batch, Tiled Inference for Raster-Based Results
# Thomas Lake, Sept 2025

# Imports

import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import numpy as np #noqa: E402
from tqdm import tqdm #noqa: E402
import rasterio #noqa: E402
from rasterio.windows import Window #noqa: E402
from rasterio.merge import merge #noqa: E402
from rasterio.mask import mask #noqa: E402
from rasterio.transform import from_origin #noqa: E402
from rasterio.warp import transform, reproject, Resampling, calculate_default_transform #noqa: E402
from shapely.geometry import Point, box #noqa: E402
import geopandas as gpd #noqa: E402
import time #noqa: E402

import torch #noqa: E402
from torchvision import transforms #noqa: E402

from model import HostImageryClimateModel, HostClimateOnlyModel, HostImageryOnlyModel #noqa: E402
from train_utils import get_default_device, load_model_from_checkpoint #noqa: E402


# --- Hard-coded WorldClim normalization stats ---
WORLDCLIM_STATS = {
    "wc2.1_30s_bio_1":  {"mean": 14.8231, "std": 1.7885},
    "wc2.1_30s_bio_2":  {"mean": 12.7210, "std": 0.8688},
    "wc2.1_30s_bio_3":  {"mean": 38.9281, "std": 1.5398},
    "wc2.1_30s_bio_4":  {"mean": 758.8810, "std": 30.6425},
    "wc2.1_30s_bio_5":  {"mean": 30.9731, "std": 1.8995},
    "wc2.1_30s_bio_6":  {"mean": -1.6820, "std": 1.8469},
    "wc2.1_30s_bio_7":  {"mean": 32.6552, "std": 1.4420},
    "wc2.1_30s_bio_8":  {"mean": 21.0870, "std": 6.6312},
    "wc2.1_30s_bio_9":  {"mean": 10.8631, "std": 3.1185},
    "wc2.1_30s_bio_10": {"mean": 24.0236, "std": 1.8737},
    "wc2.1_30s_bio_11": {"mean": 5.2723, "std": 1.7850},
    "wc2.1_30s_bio_12": {"mean": 1265.9795, "std": 155.5210},
    "wc2.1_30s_bio_13": {"mean": 138.3653, "std": 23.6783},
    "wc2.1_30s_bio_14": {"mean": 81.0823, "std": 12.3971},
    "wc2.1_30s_bio_15": {"mean": 16.5062, "std": 6.5023},
    "wc2.1_30s_bio_16": {"mean": 380.1464, "std": 57.4418},
    "wc2.1_30s_bio_17": {"mean": 264.5838, "std": 40.4858},
    "wc2.1_30s_bio_18": {"mean": 371.6132, "std": 52.8611},
    "wc2.1_30s_bio_19": {"mean": 296.3508, "std": 45.1086},
}

# Function to compute mean and standard deviation of WorldClim variables within a study area
def compute_worldclim_stats(worldclim_folder, study_geom, buffer=0.01):
    """
    Compute mean and standard deviation of WorldClim variables within a study area polygon.
    Normalization step prior to training models to ensure variables are mean 0 std 1.
    0.01 degree buffer added to study area to ensure full coverage.
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

def enable_mc_dropout(model):
    """Enable dropout layers during inference"""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

# Extract Global Human Modification (ghm) value for a given point
def extract_ghm_for_point(lon, lat, ghm_raster_fp):
    """
    Extract Global Human Modification (ghm) value for a given point (lon, lat) from the ghm raster.
    Values are already normalized to be between 0 and 1.
    Where: 0 = no human modification, 1 = maximum human modification.
    """
    with rasterio.open(ghm_raster_fp) as src:
        # Reproject raster to WGS84
        if src.crs.to_string() != 'EPSG:4326':
            src = src.reproject('EPSG:4326')
        # Get pixel coordinates of the point
        coords = (lon, lat)
        value = list(src.sample([coords]))[0][0]  # Get the first band value
        return value

@torch.inference_mode() # Disable gradient calculations for inference-only
def run_inference_on_tile(
    tile_fp, output_dir, model, device, image_transform,
    wc_stats, worldclim_folder, ghm_raster_fp, 
    chip_size=256, stride=256
):
    
    # Track inference time
    start_time = time.time()

    # Output file path
    tile_name = os.path.splitext(os.path.basename(tile_fp))[0]
    output_fp = os.path.join(output_dir, f"{tile_name}_predictions.tif")

    with rasterio.open(tile_fp) as src:
        profile = src.profile
        width, height = src.width, src.height
        transform = src.transform
        crs = src.crs

        # Init prediction accumulation arrays
        sum_array = np.zeros((height, width), dtype=np.float32)
        std_sum_array = np.zeros((height, width), dtype=np.float32)
        count_array = np.zeros((height, width), dtype=np.uint16)

        # Chip grid
        rows = range(0, height - chip_size + 1, stride)
        cols = range(0, width - chip_size + 1, stride)
        chip_iter = [(r, c) for r in rows for c in cols]

        # Batch buffers
        chip_batch, env_batch, locs = [], [], []

        def flush_batch():
            '''
            Run model with batch inference.
            '''

            chips = torch.cat(chip_batch, dim=0).to(device) if chip_batch else None
            envs = torch.cat(env_batch, dim=0).to(device) if env_batch else None

            # Monte Carlo Dropout
            T = 2 # Number of stochastic forward passes for uncertainty estimation
            probs = []
            for _ in range(T):
                if isinstance(model, HostImageryClimateModel):
                    p = model(chips, envs)
                elif isinstance(model, HostImageryOnlyModel):
                    p = model(chips)
                elif isinstance(model, HostClimateOnlyModel):
                    p = model(envs)
                else:
                    raise NotImplementedError("Unknown model type used in inference.")
                
                probs.append(p.detach().cpu().numpy()) # Store the probability for one forward pass

            probs = np.stack(probs, axis=0) # Shape: (T, batch_size, 1)
            mean_probs = probs.mean(axis=0).squeeze() # Shape: (batch_size,)
            std_probs = probs.std(axis=0).squeeze() # Shape: (batch_size,)

            # Write batch back to accumulators
            for (row, col), mean_p, std_p in zip(locs, mean_probs, std_probs):
                sum_array[row:row + chip_size, col:col + chip_size] += float(mean_p)
                std_sum_array[row:row + chip_size, col:col + chip_size] += float(std_p)
                count_array[row:row + chip_size, col:col + chip_size] += 1

            # Clear batch buffers
            chip_batch.clear()
            env_batch.clear()
            locs.clear()

        # Iterate over chips for tiled inference
        for row, col in chip_iter:
            window = Window(col_off=col, row_off=row, width=chip_size, height=chip_size)
            chip = src.read(window=window).astype(np.float32) / 255.0 # Normalize to [0, 1]
            chip_tensor = image_transform(np.moveaxis(chip, 0, -1)).unsqueeze(0) # Tensor Shape: (1, C, H, W)

            # Center coords -> pixel indices for env rasters
            cx, cy = src.transform * (col + chip_size // 2, row + chip_size // 2)
            lon, lat = rasterio.warp.transform(crs, "EPSG:4326", [cx], [cy])
            lon, lat = lon[0], lat[0]

            # Collect env features
            wc_vars = extract_worldclim_vars_for_point(lon, lat, worldclim_folder, wc_stats)
            ghm = extract_ghm_for_point(lon, lat, ghm_raster_fp)
            env_tensor = torch.tensor([[*wc_vars.values(), ghm]], dtype=torch.float32).to(device)

            # Accumulate in batch
            chip_batch.append(chip_tensor)
            env_batch.append(env_tensor)
            locs.append((row, col))

            if len(chip_batch) >= 128: # Batch size
                flush_batch()
        
        # Flush any remaining chips in batch
        flush_batch()

        # Compute final averages
        avg_array = np.divide(sum_array, count_array, out=np.full_like(sum_array, np.nan), where=(count_array > 0))
        std_avg_array = np.divide(std_sum_array, count_array, out=np.full_like(std_sum_array, np.nan), where=(count_array > 0))

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

        # Track inference time
        end_time = time.time()
        print(f"[{tile_name}] Inference completed in {end_time - start_time:.2f} seconds.")



def main():
    # Paths
    tileindex_fp = r"D:\Ailanthus_NAIP_Classification\tileindex_NC_NAIP_2022\tileindex_NC_NAIP_2022.shp"
    naip_folder = r"D:\Ailanthus_NAIP_Classification\NAIP_NC_4Band_1m"
    worldclim_folder = r"D:\Ailanthus_NAIP_Classification\Env_Data\WorldClim"
    ghm_raster_fp = r"D:\Ailanthus_NAIP_Classification\Env_Data\Global_Human_Modification\gHM_WGS84.tif"
    output_dir = r"C:\Users\talake2\Desktop\NAIP_Opt_test"

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load NAIP data from paths
    naip_files = set(os.listdir(naip_folder))
    tileindex = gpd.read_file(tileindex_fp).to_crs('EPSG:4326')
    tileindex['filename'] = tileindex['filename'].apply(os.path.basename)
    tileindex['status'] = tileindex['filename'].apply(lambda x: "Downloaded" if x in naip_files else "Not Downloaded")
    downloaded_tiles = tileindex[tileindex['status'] == 'Downloaded']
    downloaded_union = downloaded_tiles.unary_union # Shapefile union of all downloaded NAIP tiles

    # Environment variables used by the model
    # env_vars = ["lat", "lon"] + [f"wc2.1_30s_bio_{i}" for i in range(1, 20)] + ["ghm", "dem"]
    env_vars = [f"wc2.1_30s_bio_{i}" for i in range(1, 20)] + ["ghm"]
    print("Length of env_vars:", len(env_vars))

    # Load model from checkpoint
    checkpoint_path = r"D:\Ailanthus_NAIP_Classification\NAIP_Host_Model\outputs\ailanthus_climate_only_pa_1300m_10ep_blockcv_uniform_sept0225\checkpoints\checkpoint_epoch_8.tar"
    model, _ = load_model_from_checkpoint(checkpoint_path, env_vars, hidden_dim=256, dropout=0.25)

    model.eval()  # Keeps BatchNorm layers in eval mode
    enable_mc_dropout(model)  # Activates dropout for uncertainty estimation

    device = get_default_device() # Use GPU
    print(f"Loaded model type: {model.__class__.__name__}")
    print(f"Loaded model from {checkpoint_path} on device {device}")

    image_transform = transforms.Compose([transforms.ToTensor()])

    # Compute mean, std statistics for normalizing Worldclim features
    print("Computing Worldclim Normalization Statistics ...")
    # wc_normalization_stats = compute_worldclim_stats(worldclim_folder, downloaded_union)
    wc_normalization_stats = WORLDCLIM_STATS # Use hard-coded stats to save re-calculating time
    
    # Get all NAIP .tif files
    naip_files = [os.path.join(naip_folder, f) for f in os.listdir(naip_folder) if f.endswith(".tif")]
    print(f"Found {len(naip_files)} NAIP tiles to process.")

    ##### Restart infernece with only missing NAIP .tif files #####
    # Get all original NAIP tile base names (e.g., m_3307701_ne_18_060_20220923_20221207)
    naip_tiles = [os.path.splitext(f)[0] for f in os.listdir(naip_folder) if f.endswith(".tif")]
    # Get all completed prediction tile base names (remove _predictions suffix)
    predicted_tiles = [f.replace("_predictions", "").replace(".tif", "") for f in os.listdir(output_dir) if f.endswith("_predictions.tif")]
    print(len(predicted_tiles), "tiles already predicted")
    # Find which tiles are missing
    missing_tiles = sorted(list(set(naip_tiles) - set(predicted_tiles)))
    print(f"Missing {len(missing_tiles)} tiles")
    # Get full paths for the missing tiles
    missing_tile_paths = [os.path.join(naip_folder, f + ".tif") for f in missing_tiles]
    # print(missing_tile_paths)
    print(f"Running inference on {len(missing_tile_paths)} missing tiles")

    # Loop over each NAIP tile and run inference
    for tile_fp in tqdm(missing_tile_paths, desc="Processing Tiles"):
        try:
            run_inference_on_tile(
                tile_fp, output_dir, model, device, image_transform,
                wc_normalization_stats, 
                worldclim_folder, ghm_raster_fp, 
                chip_size=256, stride=256
            )
        except Exception as e:
            print(f"[ERROR] Failed on {tile_fp}: {e}")


if __name__ == "__main__":
    main()


# EOF