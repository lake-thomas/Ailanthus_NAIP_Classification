# Description



# Imports
import os
import pandas as pd
import geopandas as gpd
import torch
from torch.utils.data import DataLoader, Dataset
import rasterio
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn


#### Define Pytorch Dataset and Models for NAIP Imagery and Environmental Variables ####

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

        if environment_features is None:
            # Get all columsn starting with 'wc2.1_30s' and dem and ghm
            environment_features = [col for col in self.df.columns if col.startswith('wc2.1_30s') or col in ['dem', 'ghm']]
        self.environment_features = environment_features

        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load NAIP chips: path is relative to image_base_dir
        img_path = os.path.join(self.image_base_dir, row['chip_path'])

        with rasterio.open(img_path) as src:
            img = src.read() # shape: (bands, height, width), NAIP is 4bands (RGB + NIR)
            img = img.astype(np.float32) / 255.0 # Convert to float32 and normalize to [0, 1]

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

        label = torch.tensor(row['presence'], dtype=torch.float32)  #'presence' is the label column (0/1)

        return img, env_features, label
    

def get_resnet_model(pretrained=True):
    """
    Create a ResNet model that accepts 4-channel input (NAIP RGB + NIR)
    """
    model = models.resnet18(pretrained=pretrained)
    # Modify the first convolution layer to accept 4 channels instead of 3
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(in_channels=4,
                            out_channels=original_conv.out_channels,
                            kernel_size=original_conv.kernel_size,
                            stride=original_conv.stride,
                            padding=original_conv.padding,
                            bias=original_conv.bias is not None)
    
    # Initalize the new conv layer weights for the 4th channel as the mean of the first 3 channels
    with torch.no_grad():
        model.conv1.weight[:, :3, :, :] = original_conv.weight
        model.conv1.weight[:, 3, :, :] = original_conv.weight.mean(dim=1)

    return model


class HostImageryClimateModel(nn.Module):
    def __init__(self, num_env_features, hidden_dim=256):
        super().__init__()
        self.resnet = get_resnet_model(pretrained=True)
        self.resnet.fc = nn.Identity() # Remove the final fully connected layer

        self.climate_mlp = nn.Sequential(
            nn.Linear(num_env_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, hidden_dim),  # 512 from Resnet18 or 2048 from ResNet50 + 64 from climate features
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 1),  # Binary classification
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, img, env):
        img_feat = self.resnet(img)  # Shape: (batch_size, 512) for ResNet18
        env_feat = self.climate_mlp(env)  # Shape: (batch_size, 64)
        fused = torch.cat((img_feat, env_feat), dim=1)
        out = self.classifier(fused) # Shape: (batch_size, 1)
        return out.squeeze(1) # Return shape (batch_size,)



#### Define DataLoader and Training Loop ####

if __name__ == "__main__":
    print("Running Host CNN NAIP Climate Pytorch Training Script...")
    
    # Paths
    csv_path = r"D:\Ailanthus_NAIP_Classification\Datasets_Occurrences_NAIP_256_Climate\Ailanthus_NC_Pres_Pseudoabs_NAIP_WC_Train_Val_Test_June25.csv"
    image_base_dir = r"D:\Ailanthus_NAIP_Classification"

    # Define environmental features list
    env_features = [col for col in pd.read_csv(csv_path).columns if col.startswith('wc2.1_30s') or col in ['dem', 'ghm']]
    print(env_features)

    # Define transformations for NAIP images (horizontal flip, vertical flip)
    def naip_transforms(x):
        # Placholder - to do
        return x

    # Create datasets for train, validation, and test splits
    train_dataset = HostNAIPDataset(csv_path, image_base_dir, split='train', environment_features=env_features)
    val_dataset = HostNAIPDataset(csv_path, image_base_dir, split='val', environment_features=env_features)
    test_dataset = HostNAIPDataset(csv_path, image_base_dir, split='test', environment_features=env_features)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize the model and move to device (GPU if available)
    model = HostImageryClimateModel(num_env_features=len(env_features))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    loss_fn = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50  # Number of training epochs

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        for imgs, envs, labels in train_loader:
            imgs = imgs.to(device)
            envs = envs.to(device)
            labels = labels.to(device)

            outputs = model(imgs, envs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")






