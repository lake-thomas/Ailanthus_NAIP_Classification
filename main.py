# Description
# Author


# Imports
import time
import logging
import argparse
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model import HostImageryClimateModel
from datasets import HostNAIPDataset
from eval_utils import evaluate, test_model, plot_accuracies, plot_losses
from train_utils import fit, save_checkpoint, load_model_from_checkpoint
from logging_utils import setup_logging



def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description="Train Host NAIP Imagery and Environmental Variables Model for Ailanthus Classification")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Logging
    setup_logging(config['log_path'])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    env_vars = config['env_features']
    train_ds = HostNAIPDataset(config['csv_path'], config['image_dir'], 'train', env_vars)
    val_ds = HostNAIPDataset(config['csv_path'], config['image_dir'], 'val', env_vars)
    test_ds = HostNAIPDataset(config['csv_path'], config['image_dir'], 'test', env_vars)

    # Dataloaders
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Model
    model = HostImageryClimateModel(num_env_features=len(env_vars)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Log the Model Arguments 
    logging.info(f"Model Arguments: {json.dumps(config, indent=4)}")

    start_time = time.time()

    # Fit Model ( Training Loop )
    history = fit(
        config['epochs'],
        config['learning_rate'],
        model,
        train_dl,
        val_dl,
        optimizer,
        config['checkpoint_dir'],
        config.get('lr_patience', 5),
        config.get('early_stopping_patience', 10),
    )

    elapsed_time = time.time() - start_time
    logging.info(f"Training completed in {elapsed_time:.2f} seconds")

    # Save Model Training History
    pd.DataFrame(history).to_csv(config['output_dir'], index=False)
    logging.info(f"Training history saved to {config['output_dir']}")

    # Plot Accuracies and Losses
    plot_accuracies(history, config['output_dir'])
    plot_losses(history, config["output_dir"])
                                
    # Model Evaluation and Confusion Matrix
    test_model(model, test_dl, device, config['output_dir'])


if __name__ == "__main__":
    main()
