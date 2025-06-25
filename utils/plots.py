from typing import Dict, List
import matplotlib.pyplot as plt

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


