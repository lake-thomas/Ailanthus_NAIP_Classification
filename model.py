# Pytorch model classes for host tree classification using NAIP imagery and environmental variables
# Thomas Lake, July 2025

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

class HostImageClimateModelBase(torch.nn.Module):
    """
    Base class for Imagery Climate Model
    """
    def training_step(self, batch):
        """
        Perform a training step on the model
        """
        images, envs, labels = batch
        device = next(self.parameters()).device
        images = images.to(device)
        envs = envs.to(device)
        labels = labels.to(device)
        out = self(images, envs)
        loss = F.binary_cross_entropy_with_logits(out, labels)
        return loss
    
    def validation_step(self, batch):
        """
        Perform a validation step on the model
        """
        images, envs, labels = batch
        device = next(self.parameters()).device
        images = images.to(device)
        envs = envs.to(device)
        labels = labels.to(device)
        out = self(images, envs)
        loss = F.binary_cross_entropy_with_logits(out, labels)
        preds = (out > 0.5).float()  # Convert probabilities to binary predictions
        acc = (preds == labels).float().mean() # Calculate validation accuracy based on binary predictions
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}
    
    def validation_epoch_end(self, outputs):
        """
        Aggregate validation results at the end of an epoch
        """
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        return {'val_loss': loss.item(), 'val_acc': acc.item()}
    
    def epoch_end(self, epoch, result):
        """
        Print the results at the end of an epoch
        """
        print(f"Epoch [{epoch+1}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")


def get_resnet_model(pretrained=True):
    """
    Create a ResNet model that accepts 4-channel input (NAIP RGB + NIR)
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
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


class HostImageryClimateModel(HostImageClimateModelBase):
    """
    Inherits from HostImageClimateModelBase and combines NAIP imagery with environmental variables
    to predict the presence of a species.
    Args:
        num_env_features (int): Number of environmental features.
        hidden_dim (int): Dimension of the hidden layer in the classifier.
    """
    def __init__(self, num_env_features, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.resnet = get_resnet_model(pretrained=True)
        self.resnet.fc = nn.Identity() # Remove the final fully connected layer

        # MLP from Gillespie et al., 2024
        # self.climate_mlp = nn.Sequential(
        #     nn.Linear(num_env_features, 1000),
        #     nn.ELU(),
        #     nn.Linear(1000, 1000),
        #     nn.ELU(),
        #     nn.Linear(1000, 2000),
        #     nn.ELU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(2000, 2000),
        #     nn.ELU()
        # )

        # self.classifier = nn.Sequential(
        #     nn.Linear(512 + 2000, hidden_dim),  # 512 from Resnet18 or 2048 from ResNet50 + 64 from climate features
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1),  # Binary classification
        #     nn.Sigmoid()  # Output between 0 and 1
        # )

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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Binary classification
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, img, env):
        img_feat = self.resnet(img)  # Shape: (batch_size, 512) for ResNet18
        env_feat = self.climate_mlp(env)  # Shape: (batch_size, 64)
        fused = torch.cat((img_feat, env_feat), dim=1)
        out = self.classifier(fused) # Shape: (batch_size, 1)
        return out.squeeze(1) # Return shape (batch_size,)


# Testing
# if __name__ == "__main__":
#     model = HostImageryClimateModel(num_env_features=17)
#     img = torch.randn(4, 4, 256, 256)  # batch of 4 RGB+NIR images
#     env = torch.randn(4, 17)
#     output = model(img, env)
#     print("Output shape:", output.shape)  # Should be (4,)