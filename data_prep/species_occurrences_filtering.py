# This script filters and deduplicates species occurrence data from iNaturalist and GBIF for Ailanthus in North Carolina, 2015-2025.
# It combines the datasets, removes duplicates within a 10-meter radius, and saves the cleaned data to a CSV file.

# Imports
import pandas as pd
import geopandas as gpd

# Load datasets
inat_df = pd.read_csv(r'C:/Users/tomla/OneDrive/Desktop/Ailanthus_NAIP_Classification/ailanthus_occurrences/ailanthus_nc_inat_filtered_2015_2025.csv')
gbif_df = pd.read_csv(r'C:/Users/tomla/OneDrive/Desktop/Ailanthus_NAIP_Classification/ailanthus_occurrences/ailanthus_nc_gbif_filtered_2015_2025.csv')

# Standardize column names
inat_df = inat_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
gbif_df = gbif_df.rename(columns={'decimalLatitude': 'lat', 'decimalLongitude': 'lon'})

# Add source column for tracking
inat_df['source'] = 'iNat'
gbif_df['source'] = 'GBIF'

# Combine into one DataFrame
df_combined = pd.concat([inat_df[['lat', 'lon', 'source']], gbif_df[['lat', 'lon', 'source']]], ignore_index=True)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df_combined, geometry=gpd.points_from_xy(df_combined.lon, df_combined.lat), crs='EPSG:4326')

# Project to meters (UTM or Web Mercator) for distance calculation
gdf_m = gdf.to_crs(epsg=3857)  # Web Mercator in meters

# Spatial deduplication: remove points within 10 meters of a previous one
gdf_m['keep'] = True
geometry_index = gdf_m.geometry.reset_index(drop=True)

# Loop through and mark duplicates
kept_geometries = []
for idx, geom in enumerate(geometry_index):
    is_duplicate = any(geom.distance(g) < 10 for g in kept_geometries)
    if is_duplicate:
        gdf_m.loc[idx, 'keep'] = False
    else:
        kept_geometries.append(geom)

# Keep only deduplicated points
gdf_cleaned = gdf_m[gdf_m['keep']].drop(columns=['keep'])

# Reproject back to lat/lon
gdf_cleaned = gdf_cleaned.to_crs(epsg=4326)

# Save to CSV
gdf_cleaned[['lat', 'lon', 'source']].to_csv('ailanthus_nc_inat_gbif_2015_2025_deduplicated.csv', index=False)

print(f"Final cleaned dataset has {len(gdf_cleaned)} points.")
print(gdf_cleaned[['lat', 'lon', 'source']].head())

