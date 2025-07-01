# Download species occurrence data from iNaturalist using pyinaturalist
# Filters: Ailanthus altissima in North Carolina, 2015–2025
# Author: Thomas Lake
# Date: June 30, 2025

# Imports
from pyinaturalist import get_observations
import pandas as pd
import time

# iNat parameters
params = {
    'taxon_id': 57278,            # Ailanthus altissima
    'place_id': 30,               # North Carolina
    'verifiable': True,           # Needs ID or Research Grade
    'quality_grade': 'research',  # High-confidence observations only
    'geo': True,                  # Must have coordinates
    'acc_below': 120, # Coordinate accuracy below 100m
    'd1': '2015-01-01',
    'd2': '2025-12-31',
    'per_page': 200
}

all_results = []
page = 1

while True:
    print(f"Fetching page {page}...")
    response = get_observations(**params, page=page)
    results = response.get('results', [])

    if not results:
        break

    all_results.extend(results)
    page += 1
    time.sleep(1)

print(f"Retrieved {len(all_results)} Ailanthus altissima observations from iNaturalist in North Carolina.")

# Extract coordinates from geojson
coords = [
    {'longitude': obs['geojson']['coordinates'][0],
     'latitude': obs['geojson']['coordinates'][1]}
    for obs in all_results
    if obs.get('geojson') and 'coordinates' in obs['geojson']
]

df_coords = pd.DataFrame(coords)
df_coords.to_csv("ailanthus_nc_inat_filtered_2015_2025.csv", index=False)

print(f"Saved {len(df_coords)} coordinates to 'ailanthus_nc_inat_filtered_2015_2025.csv'")
print(df_coords.head())
# EOF