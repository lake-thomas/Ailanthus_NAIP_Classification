#!/bin/bash

# Set the path to your URL list file
url_list="urllist_2022_4BandImagery_NAIP_NorthCarolina_m9864.txt"

# Directory containing already-downloaded NAIP TIFFs
download_dir="/d/Ailanthus_NAIP_Classification/NAIP_NC_4Band"

# Make sure the output directory exists
mkdir -p "$download_dir"

# Loop through each URL in the list
while IFS= read -r url; do
  # Extract just the filename
  filename=$(basename "$url")
  filepath="$download_dir/$filename"

  # If file doesn't exist or is 0 KB, re-download
  if [[ ! -s "$filepath" ]]; then
    echo "Downloading: $filename"
    curl -L --retry 5 --retry-delay 10 -o "${filepath}.part" "$url" && mv "${filepath}.part" "$filepath"
  else
    echo "Already exists and is non-zero: $filename"
  fi
done < "$url_list"
