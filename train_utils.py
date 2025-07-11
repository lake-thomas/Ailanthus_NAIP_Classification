# Description
# Author

import os
import torch
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import HostImageryClimateModel

from eval_utils import evaluate


def get_default_device():
    """ Set Device to GPU or CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def save_checkpoint(model, epoch, optimizer, path="checkpoints"):
    """
    Save the model weights and optimizer state to the specified path
    """
    os.makedirs(path, exist_ok=True)
    filename = f"checkpoint_epoch_{epoch}.tar"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(path, filename))
    print(f"Checkpoint saved to {os.path.join(path, filename)}")
    

def load_model_from_checkpoint(checkpoint_path: str, env_vars: list) -> torch.nn.Module:
    """
    Load model and optimizer state from a checkpoint file for evaluation
    """
    device = get_default_device()
    checkpoint = torch.load(checkpoint_path)
    model = HostImageryClimateModel(num_env_features=len(env_vars)).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model, optimizer


def fit(epochs, lr, model, train_loader, val_loader, optimizer, outpath, lr_patience=5, es_patience=10, use_fp16=True):
    """
    Train the model for a specified number of epochs with learning rate scheduling and early stopping
    """

    # Initialize history and best validation loss
    history = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=lr_patience, verbose=True)
    scaler = amp.GradScaler()
    
    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = model.training_step(batch)
                train_losses.append(loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Validation Step
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

        # Learning rate adjustment
        scheduler.step(result['val_loss'])

        # Print the current learning rate
        for param_group in optimizer.param_groups:
            print(f"Learning Rate: {param_group['lr']:.6f}")

        # Checkpoint and early stopping
        if result['val_loss'] < best_val_loss:
            best_val_loss = result['val_loss']
            early_stop_counter = 0
            save_checkpoint(model, epoch, optimizer, path=outpath)
        else:
            early_stop_counter += 1
            if early_stop_counter >= es_patience:
                print(f"Early stopping at epoch {epoch}.")
                save_checkpoint(model, epoch, optimizer, path=outpath)
                break

    return history
