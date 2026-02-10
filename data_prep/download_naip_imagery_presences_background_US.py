# Download NAIP imagery from NOAA webservers
# Source: https://www.fisheries.noaa.gov/inport/hierarchy?select=49403
# Author: Thomas Lake, October 2025

# Imports
import os
import time
import requests
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime

# --------------------
# Config
# --------------------
# CSV File contains locations of presence and background points with associated NAIP tile URLs from https://www.fisheries.noaa.gov
CSV_FILE = r"Y:\Ailanthus_NAIP_SDM\Ailanthus_US_All_Sources_NAIP_Tile_URLs.csv"
OUTPUT_DIR = r"Y:\Ailanthus_NAIP_SDM\NAIP_Imagery_Ailanthus_PB_Raw"
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
    if not url.lower().endswith(".tif"):
        return None

    filename = os.path.join(output_folder, os.path.basename(url))
    if os.path.exists(filename):
        logging.debug(f"Skipping existing file: {filename}")
        return filename  # Skip already downloaded

    try:
        response = requests.get(url, stream=True, timeout=TIMEOUT)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(filename, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(url),
            leave=False,
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


# --------------------
# Main Execution
# --------------------
# if __name__ == "__main__":
logging.info(f"Reading CSV file: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

if "url" not in df.columns:
    raise ValueError("CSV file must contain a column named 'url'.")

urls = df["url"].dropna().unique().tolist()
print(f"Total unique URLs found: {len(urls)}")
logging.info(f"Found {len(urls)} unique NAIP image URLs to download.")

start_time = time.time()
completed, failed = 0, 0

# Check status of imagery after resampling - do not re-download if already resampled
RESAMPLED_DIR = r"Y:\Ailanthus_NAIP_SDM\NAIP_Imagery_Ailanthus_2m_Resampled"

# Get set of all already-processed filenames (basename only)
existing_resampled = {os.path.basename(f) for f in os.listdir(RESAMPLED_DIR) if f.lower().endswith(".tif")}

# Filter URLS to only those NOT yet processed
urls_to_download = [url for url in urls if os.path.basename(url) not in existing_resampled]

print(f"Already processed/ resampled: {len(existing_resampled)}")
print(f"Remaining to download: {len(urls_to_download)}")
logging.info(
    f"Skipping {len(existing_resampled)} files already resampled. "
    f"{len(urls_to_download)} remain to download."
)

# Download with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(download_file, url, OUTPUT_DIR): url for url in urls_to_download}
    for future in tqdm(as_completed(futures), total=len(urls), desc="Downloading NAIP imagery"):
        result = future.result()
        if result:
            completed += 1
        else:
            failed += 1

elapsed = time.time() - start_time
avg_time = elapsed / max(1, completed)

logging.info(
    f"✅ Download complete: {completed}/{len(urls)} succeeded, {failed} failed "
    f"in {elapsed/60:.1f} minutes (~{avg_time:.2f}s per successful file)."
)

# Save summary CSV
summary_file = os.path.join(LOG_DIR, "naip_download_summary.csv")
header_needed = not os.path.exists(summary_file)
with open(summary_file, "a") as sf:
    if header_needed:
        sf.write("timestamp,success,failed,total,time_minutes\n")
    sf.write(
        f"{datetime.now().isoformat()},{completed},{failed},{len(urls)},{elapsed/60:.2f}\n"
    )

logging.info("=== All downloads complete ===")