# 🥐 Bread Classifier – Rozpoznawanie Typów Pieczywa z CNN

This project implements a convolutional neural network to classify different types of bread based on images.
It also integrates interpretability through Grad-CAM heatmaps to highlight which parts of the image influenced the model’s predictions.

The model is trained from scratch using a custom TinyVGG architecture and prepared for deployment via Hugging Face Spaces.

---

## Overview

- Custom CNN model: **TinyVGG**
- Image classification: e.g. *croissant*, *pretzel*, *garlic bread*, etc.
- Data loading, augmentation and training pipeline
- Grad-CAM heatmap visualization for model interpretability
- Notebook-based training and evaluation
- Model export (`.pth`) + integration-ready for **Gradio/Hugging Face**

---

## 📁 Project Structure

```bash
├── app/ # (To be used for deployment / HF Space)
├── data/ # Datasets
│ ├── dataset_1/
│ └── zip/data.zip
├── models/
│ ├── model_builder.py # TinyVGG model definition
│ └── tiny_vgg_model_1.pth # Trained weights
├── notebooks/
│ ├── dataset_creation.ipynb # Data organization
├── utils/
│ ├── dataloader.py # Dataloader creation
│ ├── engine.py # Training/validation loop
│ ├── heatmap.py # Grad-CAM generation - TODO
│ ├── plots.py # Metric plotting
│ ├── save_load.py # Save/load utils
│ └── transforms.py # Image transforms - TODO
├── requirements.txt
├── README.md
├── tiny_vgg_model_training.ipynb # Model training
└── tiny_vgg_metrics_plots.ipynb # Evaluation & metrics
```

---

## Features

- ✅ Image classification with CNN  
- ✅ Modular and reusable architecture (`utils/`, `models/`)  
- ✅ Custom training/validation loop with accuracy tracking  
- ✅ Pretrained model saved as `.pth`  
- 🟡 Notebook-based experimentation and visualization (planned)
- 🟡 Grad-CAM heatmaps (planned)  
- 🛠️ Hugging Face demo interface (planned)

---

## 📦 Dataset

```
data/dataset_1/
├── train/
│ ├── french_toast/
│ └── garlic_bread/
├── val/
│ ├── french_toast/
│ └── garlic_bread/
└── test/
├── french_toast/
└── garlic_bread/
```

### 📁 Structure Summary

- `train/` — used for model training  
- `val/` — used for validation during training  
- `test/` — used for final evaluation  

This structure is compatible with PyTorch’s `ImageFolder` and allows for easy loading using custom dataloaders from `utils/dataloader.py`.

---

You can modify or extend the dataset by adding new bread classes and images inside the respective folders.
