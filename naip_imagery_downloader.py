# A script to download NAIP imagery and metadata from AWS S3 based on a manifest file.
# This script filters for North Carolina (NC) imagery from the most recent year available.
# Author: Thomas Lake
# Date: June 30, 2025


import pandas as pd
import boto3
import os
from tqdm import tqdm

# --- PARAMETERS ----------------------------------------------------
manifest_file = r'C:/Users/tomla/OneDrive/Desktop/Ailanthus_NAIP_Classification/naip_manifest_aws_june3025.txt'
output_dir = r'C:/Users/tomla/OneDrive/Desktop/Ailanthus_NAIP_Classification/NAIP_downloaded'
bucket = 'naip-analytic'
prefix = 'nc/'  # North Carolina

# --- READ AND PARSE MANIFEST FILE ----------------------------------
df = pd.read_csv(manifest_file, header=None, delimiter='\t', names=['path'])

# Split manifest path into parts
df['parts'] = df['path'].str.split('/')
df['state'] = df['parts'].apply(lambda x: x[0])
df['year'] = df['parts'].apply(lambda x: x[1] if len(x)>1 else None)
df['resolution'] = df['parts'].apply(lambda x: x[2] if len(x)>2 else None)

# --- SELECT NC FILES AND LATEST YEAR -------------------------------
nc_df = df[df['state'] == 'nc']
years = pd.to_numeric(nc_df['year'], errors='coerce')
latest_year = years.max()
print(f'Most recent year in manifest for NC: {latest_year}')

nc_df_latest = nc_df[nc_df['year'] == str(latest_year)]

# Set default resolution for index file lookup (most common)
default_resolution = nc_df_latest['resolution'].mode()[0]
print(f"Using default resolution for index file lookup: {default_resolution}")

# Filter files based on extensions 
exts = ['.tif']
image_df = nc_df_latest[nc_df_latest['path'].str.endswith(tuple(exts))]

print(image_df.shape[0], 'files found for NC in the latest year:', latest_year)
print(image_df.head(10))


# --- DOWNLOAD NAIP INDEX SHAPEFILE -------------------------
index_files = [
    "NAIP_20_NC.cpg",
    "NAIP_20_NC.dbf",
    "NAIP_20_NC.prj",
    "NAIP_20_NC.sbn",
    "NAIP_20_NC.sbx",
    "NAIP_20_NC.shp",
    "NAIP_20_NC.shx"
]

index_dir = os.path.join(output_dir, "index_files")
os.makedirs(index_dir, exist_ok=True)

s3_client = boto3.client("s3")

for fname in index_files:
    s3_key = f"{prefix}{latest_year}/{default_resolution}/index/{fname}"
    local_path = os.path.join(index_dir, fname)

    if os.path.exists(local_path):
        print(f"Already exists: {local_path}")
        continue

    try:
        print(f"Downloading index file: {s3_key}")
        s3_client.download_file(
            Bucket=bucket,
            Key=s3_key,
            Filename=local_path,
            ExtraArgs={'RequestPayer': 'requester'}
        )
        print(f"Saved: {local_path}")
    except Exception as e:
        print(f"Failed to download {s3_key}: {e}")



# --- DOWNLOAD NAIP IMAGES FROM S3 --------------------------

def download_from_s3(bucket, s3_key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(bucket, s3_key, local_path, ExtraArgs={'RequestPayer': 'requester'})
    except Exception as e:
        print(f"Failed to download {s3_key}: {e}")

# Download all filtered files
print(f"Number of files to download: {len(image_df)}")

for rel_key in tqdm(image_df['path'].tolist()):
    s3_key = rel_key  # path in manifest is S3 key
    local_dest = os.path.join(output_dir, s3_key.replace('/', '_'))
    if os.path.isfile(local_dest):
        continue
    download_from_s3(bucket, s3_key, local_dest)

print("Download complete!")

# EOF