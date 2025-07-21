import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models

def load_image(image_path: str):
    """Loads an image from a file path and preprocesses it for model input."""
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                    image: torch.Tensor, 
                    conv_layer_name: str) -> np.ndarray:
    """Computes a heatmap for the input image using the specified convolutional layer."""

    model.eval()
    conv_layer = get_conv_layer(model, conv_layer_name)
    
    # Forward pass
    with torch.no_grad():
        features = conv_layer(image)
        preds = model(image)

    # Get class index
    class_idx = get_class_label(preds)

    # Get gradients
    features.register_hook(lambda grad: grad)
    preds[0, class_idx].backward()

    # Compute heatmap
    heatmap = features.grad.data.abs().mean(dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap