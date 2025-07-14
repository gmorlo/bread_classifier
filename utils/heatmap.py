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
    class_idx = torch.argmax(preds, dim=1).item()
    return class_idx

