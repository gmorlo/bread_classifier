from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training and testing loss curves.
    
    Args: 
        results history dictionary containing lists of values:
        {"train_loss": [...],
        "train_accuracy": [...],
        "test_loss": [...],
        "test_accuracy": [...]}
    """
    
    # Get the values from the results dictionary
    train_loss = results["train_loss"]
    train_accuracy = results["train_acc"]
    test_loss = results["test_loss"]
    test_accuracy = results["test_acc"]

    # Figure how many epochs we trained for
    epochs = range(len(results["train_loss"]))

    # Set up the plot
    plt.figure(figsize=(12, 6))

    # Plot training and testing loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, test_accuracy, label="Test Accuracy")
    plt.title("Accuracy Curves")
    plt.xlabel("Epochs")
    plt.legend()


def evaluate_model(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   class_names: List[str],
                   device: torch.device):
    """Creates a confusion matrix for the model's predictions.
    
    Evaluates the model on the given dataloader and plots the confusion matrix.

    Args:
        model: A PyTorch model to evaluate.
        dataloader: A PyTorch DataLoader for the evaluation dataset.
        class_names: List[str]: The names of the classes.
        device: A PyTorch device (e.g. "cuda" or "cpu") to compute on.
    """

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()