# Download NAIP imagery from NOAA webservers
# Source: https://www.fisheries.noaa.gov/inport/hierarchy?select=49403
# Author: Thomas Lake, October 2025

# Imports
import os
import time
import requests
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime

# --------------------
# Config
# --------------------
URL_DIR = r"Y:\NAIP_state_tile_indices_url_lists"  # Text file with list of NAIP imagery URLs
OUTPUT_DIR = r"Y:\NAIP_Imagery_Raw" # Directory to save downloaded imagery
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
MAX_WORKERS = 10
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
# Functions
# --------------------

def download_file(url, output_folder):
    """Download a single .tif file if not already present."""
    if not url.lower().endswith('.tif'):
        return None

    filename = os.path.join(output_folder, os.path.basename(url))
    if os.path.exists(filename):
        logging.debug(f"Skipping existing file: {filename}")
        return filename  # skip already downloaded files

    try:
        response = requests.get(url, stream=True, timeout=TIMEOUT)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(filename, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=os.path.basename(url), leave=False
        ) as pbar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        logging.info(f"✅ SUCCESS: {url}")
        return filename
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
url_files = [f for f in os.listdir(URL_DIR) if f.startswith("urllist_") and f.endswith(".txt")]

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