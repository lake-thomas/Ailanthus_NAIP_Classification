# Identify all NAIP images from polygons given three sets of point records:
# 1) Ailanthus occurrences and background points from iNaturalist and GBIF
# 2) Ailanthus occurrences and background points from US agencies
# 3) Ailanthus occurrences from both sources combined

# Thomas Lake, Nov 2025

# Imports
import os
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt


# --------------------
# Path to NAIP polygons shapefile
# --------------------
NAIP_polygons_fp = r"Y:\Ailanthus_NAIP_SDM\NAIP_Imagery_Tile_Indices\NAIP_US_State_Tile_Indices_URL_Paths_Nov625.shp" # NAIP polygons with URLs for downloading

# --------------------
# Paths to occurrence and background point records
# --------------------

# 1) Ailanthus occurrences and background points from iNaturalist and GBIF
ailanthus_inat_gbif_pb_fp = r"Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Occurrences_Background_iNat_GBIF_Agency_Processed\Ailanthus_US_iNat_GBIF_P_B_Filtered_Sampled_NAIP_Tiles.shp"

# 2) Ailanthus occurrences and background points from US agencies
ailanthus_agency_pb_fp = r"Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Occurrences_Background_iNat_GBIF_Agency_Processed\Ailanthus_US_APHIS_State_P_B_Filtered_Sampled_NAIP_Tiles.shp"

# 3) Ailanthus occurrences from both sources combined
ailanthus_inat_gbif_agency_pb_fp = r"Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Occurrences_Background_iNat_GBIF_Agency_Processed\Ailanthus_US_iNat_GBIF_APHIS_State_P_B_Filtered_Sampled_NAIP_Tiles.shp"

# Output directory for CSV files with NAIP image URLs
output_dir = r"Y:\Ailanthus_NAIP_SDM"

# --------------------
# Extract NAIP tile URLs for each set of points
# --------------------

# Load NAIP polygons
naip_gdf = gpd.read_file(NAIP_polygons_fp)

# Check CRS (Coordinate Reference System) and assign to EPSG:4326 if undefined
if naip_gdf.crs is None:
    naip_gdf.set_crs(epsg=4326, inplace=True)


# --------------------
# Load point dataset: Ailanthus occurrences and background points from iNaturalist and GBIF
# --------------------
ailanthus_inat_gbif_p_b = gpd.read_file(ailanthus_inat_gbif_pb_fp)

# Check CRS and assign to EPSG:4326 if undefined
if ailanthus_inat_gbif_p_b.crs is None:
    ailanthus_inat_gbif_p_b.set_crs(epsg=4326, inplace=True)

# Assign source dataset identifier
ailanthus_inat_gbif_p_b['source_dataset'] = 'Ailanthus_iNat_GBIF_P_B'

# Spatial Join between points and NAIP polygons to get URLs
joined_dataset_1 = gpd.sjoin(ailanthus_inat_gbif_p_b, naip_gdf, how="inner", predicate='within')
joined_dataset_1 = joined_dataset_1.reset_index(drop=True)

# Joined datset has two URLs after spatial join: 'url_left' from points and 'url_right' from NAIP polygons
# Drop 'url_left' and 'index_right' and rename 'url_right' to 'url'
joined_dataset_1 = joined_dataset_1.drop(columns=['url_left', 'index_right'])
joined_dataset_1 = joined_dataset_1.rename(columns={'url_right': 'url'})

print(joined_dataset_1.head())

# --------------------
# Load point dataset: Ailanthus occurrences and background points from US agencies
# --------------------
ailanthus_agency_p_b = gpd.read_file(ailanthus_agency_pb_fp)

# Check CRS and assign to EPSG:4326 if undefined
if ailanthus_agency_p_b.crs is None:
    ailanthus_agency_p_b.set_crs(epsg=4326, inplace=True)

# Assign source dataset identifier
ailanthus_agency_p_b['source_dataset'] = 'Ailanthus_Agency_P_B'

# Spatial Join between points and NAIP polygons to get URLs
joined_dataset_2 = gpd.sjoin(ailanthus_agency_p_b, naip_gdf, how="inner", predicate='within')
joined_dataset_2 = joined_dataset_2.reset_index(drop=True)

# Joined datset has two URLs after spatial join: 'url_left' from points and 'url_right' from NAIP polygons
# Drop 'url_left' and 'index_right' and rename 'url_right' to 'url'
joined_dataset_2 = joined_dataset_2.drop(columns=['url_left', 'index_right'])
joined_dataset_2 = joined_dataset_2.rename(columns={'url_right': 'url'})

print(joined_dataset_2.head())

# --------------------
# Load point dataset: Ailanthus occurrences from both sources combined
# --------------------
ailanthus_inat_gbif_agency_p_b = gpd.read_file(ailanthus_inat_gbif_agency_pb_fp)

# Check CRS and assign to EPSG:4326 if undefined
if ailanthus_inat_gbif_agency_p_b.crs is None:
    ailanthus_inat_gbif_agency_p_b.set_crs(epsg=4326, inplace=True)

# Assign source dataset identifier
ailanthus_inat_gbif_agency_p_b['source_dataset'] = 'Ailanthus_iNat_GBIF_Agency_P_B'

# Spatial Join between points and NAIP polygons to get URLs
joined_dataset_3 = gpd.sjoin(ailanthus_inat_gbif_agency_p_b, naip_gdf, how="inner", predicate='within')
joined_dataset_3 = joined_dataset_3.reset_index(drop=True)

# Joined datset has two URLs after spatial join: 'url_left' from points and 'url_right' from NAIP polygons
# Drop 'url_left' and 'index_right' and rename 'url_right' to 'url'
joined_dataset_3 = joined_dataset_3.drop(columns=['url_left', 'index_right'])
joined_dataset_3 = joined_dataset_3.rename(columns={'url_right': 'url'})

print(joined_dataset_3.head())

# --------------------
# Combine all joined datasets and extract unique URLs for downloading
# --------------------

combined_datasets = pd.concat([joined_dataset_1, joined_dataset_2, joined_dataset_3], ignore_index=True)
combined_datasets = combined_datasets.reset_index(drop=True)

print(combined_datasets.head())

print(f"Total records in combined dataset: {len(combined_datasets)}")

# Drop duplicate URLs in combined dataset and keep geodataframe structure and retain only one NAIP image per unique URL
unique_urls_gdf = combined_datasets.drop_duplicates(subset=['url']).reset_index(drop=True)

print(f"Total unique NAIP tile URLs across all datasets: {len(unique_urls_gdf)}")

print(unique_urls_gdf.head())

# Save URLs to CSV
output_csv_fp = os.path.join(output_dir, "Ailanthus_US_All_Sources_NAIP_Tile_URLs.csv")
unique_urls_gdf.to_csv(output_csv_fp, index=False)


# EOF