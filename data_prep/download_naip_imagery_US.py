# Download NAIP imagery from NOAA webservers
# Source: https://www.fisheries.noaa.gov/inport/hierarchy?select=49403
# Author: Thomas Lake, January 2026

# Imports
import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import time # noqa: E402
import requests # noqa: E402
from tqdm import tqdm # noqa: E402
import geopandas as gpd # noqa: E402
import rasterio # noqa: E402
from rasterio.enums import Resampling # noqa: E402
from rasterio.warp import calculate_default_transform, reproject # noqa: E402
import matplotlib.pyplot as plt # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed # noqa: E402
import logging # noqa: E402
from datetime import datetime # noqa: E402

# --------------------
# Config
# --------------------
URL_DIR = r"Y:\Ailanthus_NAIP_SDM\NAIP_Imagery_Tile_Indices\urllist_tileindex_naip_us_states"  # Text file with list of NAIP imagery URLs
OUTPUT_DIR = r"Y:\Ailanthus_NAIP_SDM\NAIP_South_Central_2m" # Directory to save downloaded imagery
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
MAX_WORKERS = 12
TIMEOUT = 60
CHUNK_SIZE = 1024 * 1024  # 1 MB per chunk

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --------------------
# Logging Setup
# --------------------
log_file = os.path.join(LOG_DIR, f"naip_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# --------------------
# Existing NAIP tiles (US train/ val/ test splits) to skip downloading
# --------------------
EXISTING_NAIP_DIR = r"Y:\Ailanthus_NAIP_SDM\NAIP_Imagery_Ailanthus_PB_US_2m" # Directory with existing NAIP tiles to skip

def build_existing_naip_index(root_dir):
    """Build a set of existing NAIP tile filenames to skip during download."""
    existing = set()
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith('.tif'):
                existing.add(f)
    logging.info(f"Indexed {len(existing)} existing NAIP tiles to skip.")
    return existing

EXISTING_NAIP_FILES = build_existing_naip_index(EXISTING_NAIP_DIR)

# --------------------
# Downloading Functions
# --------------------

def resample_to_2m(src_path, dst_path, target_res=2.0, delete_original=True):
    """
    Resample a GeoTIFF to 2m resolution and save to disk.
    """
    with rasterio.open(src_path) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            src.crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=(target_res, target_res),
        )

        dst_meta = src.meta.copy()
        dst_meta.update({
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        })

        with rasterio.open(dst_path, "w", **dst_meta) as dst:
            for band in range(1, src.count + 1):
                dst.write(
                    src.read(
                        band,
                        out_shape=(dst_height, dst_width),
                        resampling=Resampling.average,
                    ),
                    band,
                )

    if delete_original:
        os.remove(src_path)


def download_file(url, output_folder):
    """Download a single .tif file, resample to 2m, and save to disk."""
    if not url.lower().endswith('.tif'):
        return None

    base_name = os.path.basename(url)

    # Skip if file already exists in existing NAIP set
    if base_name in EXISTING_NAIP_FILES:
        logging.debug(f"Skipping existing NAIP tile - it's already downloaded: {base_name}")
        return None

    raw_path = os.path.join(output_folder, base_name)
    resampled_path = raw_path.replace(".tif", "_2m.tif")

    # Skip if 2m product already exists
    if os.path.exists(resampled_path):
        logging.debug(f"Skipping existing 2m file: {resampled_path}")
        return resampled_path

    try:
        response = requests.get(url, stream=True, timeout=TIMEOUT)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(raw_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=base_name,
            leave=False
        ) as pbar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        # --- NEW: resample to 2m ---
        resample_to_2m(
            src_path=raw_path,
            dst_path=resampled_path,
            target_res=2.0,
            delete_original=True,
        )

        logging.info(f"✅ SUCCESS (2m): {url}")
        return resampled_path

    except Exception as e:
        logging.error(f"❌ FAILED: {url} — {e}")
        return None


def download_state(state_name, url_list_file):
    """Download all .tif URLs for a given state and log summary."""
    logging.info(f"=== Starting downloads for {state_name} ===")

    output_folder = os.path.join(OUTPUT_DIR, state_name)
    os.makedirs(output_folder, exist_ok=True)

    with open(url_list_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip().endswith('.tif')]

    start_time = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, url, output_folder): url for url in urls}
        for future in tqdm(as_completed(futures), total=len(urls), desc=f"{state_name} Progress"):
            result = future.result()
            if result:
                completed += 1
            else:
                failed += 1

    elapsed = time.time() - start_time
    avg_time = elapsed / max(1, completed)
    logging.info(
        f"{state_name}: {completed}/{len(urls)} success, {failed} failed "
        f"in {elapsed/60:.1f} min (~{avg_time:.2f}s per successful file)"
    )

    # Write per-state summary to CSV for easy tracking
    summary_file = os.path.join(LOG_DIR, "state_download_summary.csv")
    header_needed = not os.path.exists(summary_file)
    with open(summary_file, "a") as sf:
        if header_needed:
            sf.write("timestamp,state,success,failed,total,time_minutes\n")
        sf.write(
            f"{datetime.now().isoformat()},{state_name},{completed},{failed},{len(urls)},{elapsed/60:.2f}\n"
        )

    return elapsed, completed, len(urls)


# --------------------
# Main Execution
# --------------------

# Find all state URL files with links to .tif imagery
# Ex: https://coastalimagery.blob.core.windows.net/digitalcoast/NJ_NAIP_2019_9544/m_3807401_ne_18_060_20190730.tif
url_files = [f for f in os.listdir(URL_DIR) if f.startswith("urllist_") and f.endswith(".txt")] #Ex; urllist_2019_4BandImagery_Nevada_m9543.txt

# Select a specific state from the url_files list for downloading
# url_files = [f for f in url_files if "Ohio" in f]  # Example: only download one state

# Select multiple states to download
# States to download (match naming used in URL list filenames)
SC_STATES = {
    "Louisiana",
    "Arkansas",
    "Texas",
    "Oklahoma",
    "Missouri",
    "Kansas"
}

url_files = [
    f for f in url_files
    if any(state in f for state in SC_STATES)
]

total_est_time = 0
total_files = 0
total_completed = 0

logging.info(f"Found {len(url_files)} state URL files in {URL_DIR}")

for url_file in url_files:
    try:
        state_name = url_file.split('_')[3]
    except IndexError:
        logging.warning(f"Skipping unrecognized file format: {url_file}")
        continue

    url_path = os.path.join(URL_DIR, url_file)
    elapsed, completed, total = download_state(state_name, url_path)
    total_est_time += elapsed
    total_completed += completed
    total_files += total

    logging.info(
        f"Progress Summary: {total_completed}/{total_files} total tiles "
        f"({total_est_time/3600:.2f} hours elapsed)"
    )

logging.info("=== All downloads complete ===")