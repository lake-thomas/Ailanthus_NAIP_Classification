# Evaluation utilities for NAIP imagery and environmental variables model
# Thomas Lake, July 2025

import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from model import HostImageryClimateModel, HostImageryOnlyModel, HostClimateOnlyModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, roc_auc_score


@torch.no_grad()
def test_model(model, test_loader, device, out_dir="model_results"):
    """
    Evaluate the model on the testing dataset and save results
    """
    model.to(device)
    model.eval()

    y_pred = []
    y_true = []

    for batch in test_loader:
        images, envs, labels = batch
        images = images.to(device)
        envs = envs.to(device)
        labels = labels.to(device)

        # Dynamically handle model input
        if isinstance(model, HostImageryClimateModel):
            outputs = model(images, envs)
        elif isinstance(model, HostImageryOnlyModel):
            outputs = model(images)
        elif isinstance(model, HostClimateOnlyModel):
            outputs = model(envs)
        else:
            raise NotImplementedError("Unknown model type for test_model")

        preds = (outputs > 0.5).float()
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

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

    report_df.to_csv(os.path.join(out_dir, "classification_report.csv"))

    # ROC Curve (Binary Classification) and AUC
    plot_roc_curve(y_true=y_true, y_pred=y_pred, out_dir=out_dir)

@torch.no_grad()
def map_model_errors(model, test_loader, device, out_dir="model_results"):
    """
    Map model errors on the test dataset
    """

    model.to(device)
    model.eval()

    y_pred = []
    y_true = []
    lat_list = []
    lon_list = []

    for batch in test_loader:
        images, envs, labels = batch
        images = images.to(device)
        envs = envs.to(device)
        labels = labels.to(device)

        # Dynamically handle model input
        if isinstance(model, HostImageryClimateModel):
            outputs = model(images, envs)
        elif isinstance(model, HostImageryOnlyModel):
            outputs = model(images)
        elif isinstance(model, HostClimateOnlyModel):
            outputs = model(envs)
        else:
            raise NotImplementedError("Unknown model type for map_model_errors")

        preds = (outputs > 0.5).float()
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        
        # Collect lat/lon from the batch
        lat = envs[:, 0].cpu().numpy()  # Lat is the 5th column in the environment features
        lon = envs[:, 1].cpu().numpy()  # Lon is the 6th column in the environment features
        lat_list.extend(lat)
        lon_list.extend(lon)

    # Build Dataframe
    df = pd.DataFrame({
        "lat": lat_list,
        "lon": lon_list,
        "true_label": y_true,
        "predicted_label": y_pred})
    
    # Label Prediction Types by Error
    df['error_type'] = "UNDEF"
    df.loc[(df.true_label == 1) & (df.predicted_label == 1), "error_type"] = "TP"
    df.loc[(df.true_label == 0) & (df.predicted_label == 0), "error_type"] = "TN"
    df.loc[(df.true_label == 0) & (df.predicted_label == 1), "error_type"] = "FP"
    df.loc[(df.true_label == 1) & (df.predicted_label == 0), "error_type"] = "FN"

    # Save DataFrame to CSV
    df.to_csv(os.path.join(out_dir, "spatial_predictions.csv"), index=False)

    # Plot Errors on Map
    color_map = {"TP": "green", "TN": "blue", "FP": "red", "FN": "orange"}
    plt.figure(figsize=(8, 6))
    for err_type, color in color_map.items():
        subset = df[df["error_type"] == err_type]
        plt.scatter(subset["lon"], subset["lat"], c=color, label=err_type, alpha=0.6, s=10)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatial Distribution of Prediction Errors")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "spatial_errors.png"), dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_pred, out_dir="model_results"):
    """
    Plot ROC curve and save the figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)

    os.makedirs(out_dir, exist_ok=True)
    roc_path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

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