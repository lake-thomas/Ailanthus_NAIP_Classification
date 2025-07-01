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

# --- SELECT NC FILES AND LATEST YEAR -------------------------------
nc_df = df[df['state'] == 'nc']
years = pd.to_numeric(nc_df['year'], errors='coerce')
latest_year = years.max()
print(f'Most recent year in manifest for NC: {latest_year}')

nc_df_latest = nc_df[nc_df['year'] == str(latest_year)]

# Filter files based on extensions 
exts = ['.tif']
image_df = nc_df_latest[nc_df_latest['path'].str.endswith(tuple(exts))]

print(image_df.shape[0], 'files found for NC in the latest year:', latest_year)
print(image_df.head(20))

# --- DOWNLOAD FROM S3 ----------------------------------------------

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