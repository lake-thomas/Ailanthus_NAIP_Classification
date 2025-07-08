# Imports
import os
import geopandas as gpd
import matplotlib.pyplot as plt

# Define paths for NAIP tile index and folder containing NAIP TIFF files
tileindex_fp = r"D:\Ailanthus_NAIP_Classification\tileindex_NC_NAIP_2022\tileindex_NC_NAIP_2022.shp"
naip_folder = r"D:\Ailanthus_NAIP_Classification\NAIP_NC_4Band"

# Load NAIP tile index shapefile
tileindex = gpd.read_file(tileindex_fp)
print(tileindex.columns)
print(f"Tile index loaded with {len(tileindex)} tiles.")

# Get list of downloaded NAIP TIFF files
NAIP_files = set(os.listdir(naip_folder))

# Assign downloaded status to each tile
def check_downloaded(row):
    tif_name = os.path.basename(row['filename'])
    return "Downloaded" if tif_name in NAIP_files else "Not Downloaded"

# Update tile index field with download status
tileindex['status'] = tileindex.apply(check_downloaded, axis=1)

# Plot the tile index with download status
fig, ax = plt.subplots(figsize=(10, 10))
tileindex[tileindex['status'] == 'Downloaded'].plot(ax=ax, color='green', label='Downloaded')
tileindex[tileindex['status'] == 'Not Downloaded'].plot(ax=ax, color='red', label='Not Downloaded')
plt.title('NAIP Tile Download Status')
plt.legend()
plt.axis('off')
plt.tight_layout()
plt.show()
