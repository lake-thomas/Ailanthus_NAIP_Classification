# NAIP SDM Analyses: Verify Downloaded NAIP Tile URLs from Ailanthus Points against NAIP Polygons
# January 2026
# ------------------------------------------------------

# Imports
import os
import random

# Set up GDAL and PROJ environment variables for geopandas compatibility
conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

# Imports after setting GDAL environment variables
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# # # ------------------------------------------------------

# Verify NAIP tile download compared to expected URLs from Ailanthus records

# Unique NAIP tiles downloaded from p/b ailanthsu records. From: naip_tile_urls_from_polygon_ailanthus_records.py
CSV_FILE = r"Y:\Ailanthus_NAIP_SDM\Ailanthus_US_All_Sources_NAIP_Tile_URLs_Downloaded.csv" 

# Directory where NAIP tiles have been downloaded
NAIP_DIR = r"Y:\Ailanthus_NAIP_SDM\NAIP_Imagery_Ailanthus_PB_US_2m"

# Read in CSV with expected NAIP tile URLs
df = pd.read_csv(CSV_FILE)
expected_files = set(df["filename"].dropna().apply(os.path.basename))
print(f"Expected NAIP tiles: {len(expected_files)}")

downloaded_files = {f for f in os.listdir(NAIP_DIR) if f.lower().endswith(".tif")}
print(f"Downloaded NAIP tiles: {len(downloaded_files)}")

# # # ------------------------------------------------------

# Verify spatial coverage of downloaded NAIP tiles against Ailanthus points

NAIP_POLYGONS_FP = r"Y:\Ailanthus_NAIP_SDM\NAIP_Imagery_Tile_Indices\NAIP_US_State_Tile_Indices_URL_Paths_Jan25.shp" # All NAIP tiles in US with URLs

naip_polys = gpd.read_file(NAIP_POLYGONS_FP)
if naip_polys.crs is None:
    naip_polys = naip_polys.set_crs(epsg=4326)

naip_polys["filename"] = naip_polys["url"].apply(os.path.basename)

naip_downloaded = naip_polys[naip_polys["filename"].isin(downloaded_files)].copy() # Filter to only downloaded tiles

print(f"NAIP polygons corresponding to downloads: {len(naip_downloaded)}")

# # # ------------------------------------------------------

# Check coverage of Ailanthus points by downloaded NAIP tiles (within)

ailanthus_pb_points = gpd.read_file(r"Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Occurrences_Background_iNat_GBIF_Agency_Processed\Ailanthus_US_iNat_GBIF_APHIS_State_P_B_Filtered_Sampled_NAIP_Tiles.shp")

if ailanthus_pb_points.crs is None:
    ailanthus_pb_points = ailanthus_pb_points.set_crs(epsg=4326)

print(f"Total Ailanthus presence/background points: {len(ailanthus_pb_points)}")

points_in_tiles_i = gpd.sjoin(
    ailanthus_pb_points,
    naip_downloaded[["geometry"]],
    how="left",
    predicate="intersects"
)

coverage_rate_i = points_in_tiles_i["index_right"].notnull().mean()
print(f"Point coverage by downloaded NAIP tiles (intersects): {coverage_rate_i:.3%}") # 100% Coverage Rate - the downloaded NAIP tiles intersect all Ailanthus presence/ background points

# # # ------------------------------------------------------

# EOF









































