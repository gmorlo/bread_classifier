import os
import random
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision import transforms
from PIL import Image
from pathlib import Path


def load_image(image_path: str):
    """Loads an image from a file path and preprocesses it for model input."""

    img = Image.open(image_path).convert("RGB")
    img = transforms.Resize((64, 64))(img)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img

def get_class_label(preds: torch.Tensor) -> int:
    """Returns the class label with the highest score from model predictions."""

    if not isinstance(preds, torch.Tensor):
        raise TypeError("Predictions must be a torch.Tensor.")
    class_idx = torch.argmax(preds, dim=1).item()
    return class_idx

def get_conv_layer(model: torch.nn.Module, 
                   conv_layer_name: str) -> torch.nn.Module:
    """Extracts convolutional layer from a model."""

    for name, layer in model.named_modules():
        if name == conv_layer_name:
            return layer
    raise ValueError(f"Convolutional layer '{conv_layer_name}' not found in the model.")

def compute_heatmap(model: torch.nn.Module, 
                    image_tensor: torch.Tensor, 
                    class_index: int,
                    conv_layer_name: str) -> np.ndarray:
    """Computes a heatmap for the input image using the specified convolutional layer (Grad-CAM)."""

    model.eval()
    conv_layer = get_conv_layer(model, conv_layer_name)
    
    activation = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activation
        activation = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    # Register hooks
    forward_handle = conv_layer.register_forward_hook(forward_hook)
    backward_handle = conv_layer.register_backward_hook(backward_hook)

    image_tensor.requires_grad_(True)
    preds = model(image_tensor)
    loss = preds[:, class_index]
    model.zero_grad()
    loss.backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Gradients and activations
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    activation = activation[0]

    for i in range(activation.shape[0]):
        activation[i, ...] *= pooled_grads[i]

    heatmap = activation.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap

def overlay_heatmap(image_path: str, 
                    heatmap: np.ndarray, 
                    alpha: float = 0.5) -> np.ndarray:
    """Overlays the heatmap on the original image."""
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    overlayed_image = cv2.addWeighted(original_image, alpha, heatmap_colored, 1 - alpha, 0)

    return overlayed_image

def generate_gradcam_overlay(model: torch.nn.Module, 
                            data_set: str = 'dataset_1', 
                            conv_layer_name: str = 'conv_block_1', 
                            alpha: float = 0.5,
                            class_names: list = None,
                            plot: bool = False) -> Tuple[np.ndarray, int, str]:
    """Generates a Grad-CAM overlay for the specified image and class index.

    Args:
        model: A PyTorch model to use for generating the heatmap.
        data_set: The dataset to use for selecting a random image.
        conv_layer_name: The name of the convolutional layer to use for Grad-CAM.
        alpha: The transparency factor for overlaying the heatmap on the image.
        class_names: A list of class names corresponding to the model's output.
        plot: If True, displays the overlayed image using matplotlib.

    Returns:
        A tuple containing the overlayed image, predicted label, and true label.
    """


    # Get a random image path from the dataset
    base_dir = Path("data/")
    class_folders = [folder for folder in (base_dir / data_set / 'train').iterdir() if folder.is_dir()]
    val_random_dir = random.choice(class_folders)
    img_files = list(val_random_dir.glob("*.jpg"))
    image_path = random.choice(img_files)
    print(image_path)
    true_label = image_path.parent.name


    # Ensure the model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the image and model
    image_tensor = load_image(image_path).to(device)
    model = model.to(device)
    model.eval()

    # Get predictions for the image
    with torch.no_grad():
        preds = model(image_tensor)

    # Get the class index from the predictions
    class_index = get_class_label(preds)
    predicted_label = class_names[class_index]

    # Compute the heatmap
    heatmap = compute_heatmap(model, image_tensor, class_index, conv_layer_name)

    # Overlay the heatmap on the original image
    overlayed_image = overlay_heatmap(image_path, heatmap, alpha)

    # If plot is True, display the overlayed image
    if plot:
        output_image_rgb = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
        plt.imshow(output_image_rgb)
        plt.axis('off')
        plt.title("Grad-CAM Heatmap Overlay" \
                  f"\nPredicted: {predicted_label}, True: {true_label}")
        plt.show()

    return overlayed_image, predicted_label, true_label
