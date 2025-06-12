### 🥐 Bread Classifier – Rozpoznawanie Typów Pieczywa z CNN

Mój nowy projekt oparty na konwolucyjnych sieciach neuronowych (CNN) służy do automatycznego rozpoznawania rodzaju pieczywa na zdjęciach. 
Model uczy się na podstawie obrazów m.in. croissantów, precli i bagietek, a następnie dokonuje klasyfikacji nowych zdjęć. 
Projekt zawiera również element interpretowalności modelu przy użyciu techniki Grad-CAM, co pozwala lepiej zrozumieć, jakie cechy obrazu wpłynęły na decyzję sieci neuronowej.



```
bread_classifier/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── cnn_model.pth
│   └── model_builder.py
│
├── notebooks/
│   └── model_training.ipynb
│   └── explainability.ipynb
│   └── dataset_creation.ipynb
│
├── utils/
│   └── dataloader.py
│   └── save_load.py
│   └── training.py
│   └── transforms.py
│
├── app/
│   └── streamlit_app.py
│
├── README.md
└── requirements.txt
```