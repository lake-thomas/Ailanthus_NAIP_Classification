# Evaluation utilities for NAIP imagery and environmental variables model
# Thomas Lake, July 2025

import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


@torch.no_grad()
def test_model(model, test_loader, device, out_dir="model_results"):
    """
    Evaluate the model on the testing dataset and save results
    """

    model.to(device)
    model.eval # Set model to evaluation mode for testing
    y_pred = []
    y_true = []

    for batch in test_loader:
        images, envs, labels = batch
        images = images.to(device)
        envs = envs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images, envs)
            preds = (outputs > 0.5).float()  # Convert probabilities to binary predictions
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Save confusion matrix plot
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    os.makedirs(out_dir, exist_ok=True)
    plt.title("Binary Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True, target_names=["Negative", "Positive"])
    report_df = pd.DataFrame(report).transpose()
    print("Classification Report:\n", report_df)

    # Save report
    report_df.to_csv(os.path.join(out_dir, "classification_report.csv"))


def plot_accuracies(history, outpath):
    """
    Plot history of model accuracy
    """
    outpath = os.path.join(outpath, 'model_accuracy.png')
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.autoscale()
    plt.margins(0.2)
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_losses(history, outpath):
    """
    Plot history of model losses
    """
    outpath = os.path.join(outpath, 'model_losses.png')
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.autoscale()
    plt.margins(0.2)
    plt.savefig(outpath, dpi=300)
    plt.close()