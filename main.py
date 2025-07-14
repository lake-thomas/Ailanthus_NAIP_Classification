# CNN Model for Host Tree Classification using NAIP Imagery and Environmental Variables
# Thomas Lake, July 2025

# Imports
import os
import time
import logging
import argparse
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model import HostImageryClimateModel
from datasets import HostNAIPDataset
from transforms import RandomAugment4Band
from eval_utils import test_model, plot_accuracies, plot_losses
from train_utils import fit
from logging_utils import setup_logging

import wandb

def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description="Train Host NAIP Imagery and Environmental Variables Model for Ailanthus Classification")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Init and override config with W&B sweep values for hyperparameter search
    wandb.init(entity="talake2-ncsu",
               project="naip_climate_classification")
    if wandb.run:
        config['batch_size'] = wandb.config.get('batch_size', config['batch_size'])
        config['learning_rate'] = wandb.config.get('learning_rate', config['learning_rate'])
        sweep_run_id = wandb.run.name  # e.g., "sweep-1"
        config["experiment"] += f"_{sweep_run_id}"
        config['dropout'] = wandb.config.get('dropout', config.get('dropout', 0.25))
        config['hflip_prob'] = wandb.config.get('hflip_prob', config.get('hflip_prob', 0.5))
        config['vflip_prob'] = wandb.config.get('vflip_prob', config.get('vflip_prob', 0.5))
        config['rotation_degrees'] = wandb.config.get('rotation_degrees', config.get('rotation_degrees', 45))

    # Create experiment subdirectory
    experiment_name = config.get('experiment')
    experiment_dir = os.path.join(config['output_dir'], experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Create experiment-specific checkpoint directory
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(experiment_dir, f"{experiment_name}.log")
    setup_logging(log_path)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image Transformations
    image_transform = RandomAugment4Band(
    rotation_degrees=config['rotation_degrees'],
    hflip_prob=config['hflip_prob'],
    vflip_prob=config['vflip_prob']
    )

    # Dataset
    env_vars = config['env_features'] # List of environmental variables
    train_ds = HostNAIPDataset(config['csv_path'], config['image_dir'], 'train', env_vars, transform=image_transform)
    val_ds = HostNAIPDataset(config['csv_path'], config['image_dir'], 'val', env_vars)
    test_ds = HostNAIPDataset(config['csv_path'], config['image_dir'], 'test', env_vars)

    # Dataloaders
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Model
    dropout = config.get('dropout', 0.25)  # Default dropout if not specified
    model = HostImageryClimateModel(num_env_features=len(env_vars), dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Log the Model Arguments 
    logging.info(f"Model Arguments: {json.dumps(config, indent=4)}")
    wandb.config.update({"dropout": dropout}, allow_val_change=True)


    start_time = time.time()

    # Fit Model ( Training Loop )
    history = fit(
        config['epochs'],
        config['learning_rate'],
        model,
        train_dl,
        val_dl,
        optimizer,
        checkpoint_dir,
        lr_patience=config['lr_patience'],
        es_patience=config['es_patience'],
    )

    elapsed_time = time.time() - start_time
    logging.info(f"Training completed in {elapsed_time:.2f} seconds")

    # Save Model Training History    
    output_csv = os.path.join(experiment_dir, "training_history.csv")
    print("Saving training history to:", output_csv)
    pd.DataFrame(history).to_csv(output_csv, index=False)
    logging.info(f"Training history saved to {output_csv}")

    # Plot Accuracies and Losses
    plot_accuracies(history, experiment_dir)
    plot_losses(history, experiment_dir)
                                
    # Model Evaluation and Confusion Matrix
    test_model(model, test_dl, device, experiment_dir)


if __name__ == "__main__":
    main()