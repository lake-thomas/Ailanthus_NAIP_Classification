# Inference script for HostImageryClimateModel
# Thomas Lake, July 2025

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
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


def main():
    # Paths
    tileindex_fp = r"D:\Ailanthus_NAIP_Classification\tileindex_NC_NAIP_2022\tileindex_NC_NAIP_2022.shp"
    naip_folder = r"D:\Ailanthus_NAIP_Classification\NAIP_NC_4Band_1m"
    worldclim_folder = r"D:\Ailanthus_NAIP_Classification\Env_Data\WorldClim"
    ghm_raster_fp = r"D:\Ailanthus_NAIP_Classification\Env_Data\Global_Human_Modification\gHM_WGS84.tif"
    elevation_raster_fp = r"D:\Ailanthus_NAIP_Classification\Env_Data\DEM_SRTM\nc_dem_srtm.tif"

    # Load NAIP data from paths
    tileindex = gpd.read_file(tileindex_fp).to_crs('EPSG:4326')
    tileindex['filename'] = tileindex['filename'].apply(os.path.basename)
    naip_files = set(os.listdir(naip_folder))
    tileindex['status'] = tileindex['filename'].apply(lambda x: "Downloaded" if x in naip_files else "Not Downloaded")
    downloaded_tiles = tileindex[tileindex['status'] == 'Downloaded']
    downloaded_union = downloaded_tiles.unary_union # Shapefile union of all downloaded NAIP tiles

    # Output directory
    os.makedirs("model_inference", exist_ok=True)

    # Environment variables used by the model
    env_vars = ["lat", "lon"] + [f"wc2.1_30s_bio_{i}" for i in range(1, 20)] + ["ghm", "dem"]
    print("Length of env_vars:", len(env_vars))

    # Load model from checkpoint
    checkpoint_path = r"D:\Ailanthus_NAIP_Classification\NAIP_Host_Model\outputs\ailanthus_naip_test\checkpoints\checkpoint_epoch_4.tar"
    model, _ = load_model_from_checkpoint(checkpoint_path, env_vars)
    model.eval()
    print(model)
    for param in model.parameters():
        if param.requires_grad:
            print(param.mean(), param.std())
    device = get_default_device()
    print(f"Loaded model from {checkpoint_path} on device {device}")

    # Image transform (NAIP: 4-band uint8 → float tensor)
    image_transforms = transforms.Compose([transforms.ToTensor()])

    # Compute mean, std statistics for normalizing Worldclim and DEM features
    print("Computing Worldclim Normalization Statistics ...")
    wc_normalization_stats = compute_worldclim_stats(worldclim_folder, downloaded_union)
    dem_normalization_stats = compute_dem_stats(elevation_raster_fp)

    # Inference points
    inference_points_df = pd.read_csv(r"D:\Ailanthus_NAIP_Classification\point_grid_100m.csv")

    print("Loaded inference points:\n", inference_points_df)

    records = []
    probs = []

    for idx, row in tqdm(inference_points_df.iterrows(), total=len(inference_points_df), desc="Running Inference"):
        lat, lon = row['lat'], row['lon']
        # print(f"Processing point {idx+1}/{len(inference_points_df)}: Lat {lat}, Lon {lon}")
        record = {'lat': lat, 'lon': lon}

        try:
            # Extract NAIP image chip
            chip_out_fp = f"temp_chip_{idx}.tif"
            success = extract_naip_chip_for_point(
                lon, lat, chip_out_fp, chip_size=256,
                tileindex=tileindex, naip_folder=naip_folder
            )
            if not success:
                record['error'] = 'Missing NAIP chip'
                records.append(record)
                continue

            # Load NAIP chip as tensor
            with rasterio.open(chip_out_fp) as src:
                chip_img = src.read() # shape: (bands, height, width), NAIP is 4bands (RGB + NIR)
                img_norm = chip_img.astype(np.float32) / 255.0 # Convert NAIP image (0-255) to float32 and normalize to [0, 1]
                image_tensor = image_transforms(np.moveaxis(img_norm, 0, -1)).unsqueeze(0).to(device)  # shape: [1, C, H, W]
                src.close()
            if os.path.exists(chip_out_fp):
                os.remove(chip_out_fp) # Clean up temporary chip file
            
            # Extract environmental features
            wc_vars = extract_worldclim_vars_for_point(lon, lat, worldclim_folder, wc_normalization_stats)
            ghm = extract_ghm_for_point(lon, lat, ghm_raster_fp)
            dem = extract_dem_for_point(lon, lat, elevation_raster_fp, dem_normalization_stats)
            env_tensor = torch.tensor([[lat, lon, *wc_vars.values(), ghm, dem]], dtype=torch.float32).to(device)

            # print("Image tensor:", image_tensor.shape, image_tensor.dtype, image_tensor.device)
            # print(image_tensor)
            # print("Env tensor:", env_tensor.shape, env_tensor.dtype, env_tensor.device)
            # print(env_tensor)

            # Model inference
            with torch.no_grad():
                output = model(image_tensor, env_tensor)
                # print("Model output:", output.shape, output.dtype, output.device)
                #print(output) # Logit
                prob = torch.sigmoid(output).item()
                # print(f"Probability for point {idx+1}: {prob:.4f}")
                prediction = int(prob > 0.5)
                # print(f"Prediction for point {idx+1}: {prediction} (Threshold: 0.5)")

            record.update({'probability': prob, 'predicted_label': prediction})
            probs.append(prob)

        except Exception as e:
            print(f"Error processing point {idx+1}: {e}")
            record['error'] = str(e)

        records.append(record)

    # Save results
    results_df = pd.DataFrame(records)
    results_df.to_csv("model_inference/inference_results.csv", index=False)
    print("Inference complete. Results saved to model_inference/inference_results.csv")
    
    # Prediction summary
    if probs:
        probs_np = np.array(probs)
        print("\nPrediction Statistics:")
        print(f"  Mean:   {probs_np.mean():.4f}")
        print(f"  Median: {np.median(probs_np):.4f}")
        print(f"  Min:    {probs_np.min():.4f}")
        print(f"  Max:    {probs_np.max():.4f}")
    else:
        print("No valid predictions to summarize.")


if __name__ == "__main__":
    main()
