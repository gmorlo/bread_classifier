import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision import transforms
from PIL import Image

def load_image(image_path: str):
    """Loads an image from a file path and preprocesses it for model input."""

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image to use .convert("RGB")
    img = Image.fromarray(img).convert("RGB")
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

# def compute_heatmap(model: torch.nn.Module, 
#                     image_tensor: torch.Tensor, 
#                     class_index: int,
#                     conv_layer_name: str) -> np.ndarray:
#     """Computes a heatmap for the input image using the specified convolutional layer."""

#     model.eval()
#     conv_layer = get_conv_layer(model, conv_layer_name)
    
#     activation = None

#     def get_activation_hook(module, input, output):
#         nonlocal activation
#         activation = output

#     hook = conv_layer.register_forward_hook(get_activation_hook)

#     image_tensor.requires_grad_(True)
#     preds = model(image_tensor)
#     loss = preds[:, class_index]
#     model.zero_grad()
#     loss.backward()

#     grads = image_tensor.grad.cpu().numpy()
#     pooled_grads = np.mean(grads, axis=(0, 2, 3))

#     hook.remove()

#     activation = activation.detach().cpu().numpy()[0]
#     for i in range(len(pooled_grads.shape[0])):
#         activation[i, ...] *= pooled_grads[i]

#     heatmap = np.mean(activation, axis=0)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)

#     return heatmap

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