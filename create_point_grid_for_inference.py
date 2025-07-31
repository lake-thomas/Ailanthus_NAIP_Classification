import numpy as np
import pandas as pd
from pyproj import Transformer

# Define bounding box (degrees)
west, east = -79.61869, -76.68688
south, north = 34.82255, 36.53291

# Transformer to convert between lat/lon and a projected CRS for meter-based spacing
trans = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Project corners
x_min, y_min = trans.transform(west, south)
x_max, y_max = trans.transform(east, north)

# Create grid in projected meters
spacing = 1000  # meters
xs = np.arange(x_min, x_max, spacing)
ys = np.arange(y_min, y_max, spacing)

# Unproject grid back to lat/lon
lons, lats = trans.transform(xs.repeat(len(ys)), np.tile(ys, len(xs)), direction="INVERSE")

grid_df = pd.DataFrame({"lat": lats, "lon": lons})
grid_df.to_csv("point_grid_1000m.csv", index=False)
print(f"Generated {len(grid_df)} points.")
