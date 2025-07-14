# Pytorch data classes for host tree classification using NAIP imagery and environmental variables
# Thomas Lake, July 2025

import os
import pandas as pd
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# Create Pytorch Dataset for NAIP imagery and Environmental Variables

class HostNAIPDataset(Dataset):
    def __init__(self, csv_path, image_base_dir, split='train', environment_features=None, transform=None):
        """
        csv_path: Path to the CSV file with meatadata like NAIP images and environmental features
        image_base_dir: Base directory where NAIP images are stored, corresponds to the chip_path column in the CSV
        split: 'train', 'val', or 'test' to specify the dataset split and filter the DataFrame accordingly
        environment_features: List of environmental feature columns to include in the dataset
        transform: Optional torchvision transforms to apply to the images
        """
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True) 
        self.image_base_dir = image_base_dir

        # Get all columns starting with 'wc2.1_30s' and dem and ghm
        self.environment_features = [col for col in self.df.columns if col.startswith('wc2.1_30s') or col in ['dem', 'ghm']]

        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load NAIP chips: path is relative to image_base_dir
        img_path = os.path.join(self.image_base_dir, row['chip_path'])

        with rasterio.open(img_path) as src:
            img = src.read() # shape: (bands, height, width), NAIP is 4bands (RGB + NIR)
            img = img.astype(np.float32) / 255.0 # Convert NAIP image (0-255) to float32 and normalize to [0, 1]

        if self.transform:
            # torchvision transforms expect (C, H, W) format from PIL or tensor, but rasterio gives np array
            # so convert numpy to tensor first (C,H,W)
            img = torch.from_numpy(img)
            img = self.transform(img)
        else:
            # If no transform, ensure it's a tensor
            img = torch.from_numpy(img)

        # Load environmental features
        env_features = row[self.environment_features].values.astype(np.float32)
        env_features = torch.tensor(env_features, dtype=torch.float32)

        label = torch.tensor(row['presence'], dtype=torch.float32)  # 'presence' is the label column (0/1)

        return img, env_features, label

# EOF