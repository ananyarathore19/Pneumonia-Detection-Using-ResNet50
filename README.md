# Pneumonia Detection Using ResNet50

Deep learning project for automated pneumonia detection from chest X-ray images using transfer learning with ResNet50 and PyTorch.

## Project Overview

This project implements a convolutional neural network based on ResNet50 architecture to classify chest X-ray images as normal or pneumonia-infected. The model uses transfer learning to leverage pre-trained weights from ImageNet, significantly reducing training time and improving performance.

## Features

- Transfer learning with ResNet50 pre-trained model
- PyTorch implementation
- Data preprocessing and augmentation
- Model training and evaluation
- Visualization of results
- Web application interface for inference

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architecture
│   ├── training/          # Training scripts
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for exploration
├── app/                   # Web application (Flask/Streamlit)
├── outputs/               # Model outputs and results
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ananyarathore19/Pneumonia-Detection-Using-ResNet50.git
cd Pneumonia-Detection-Using-ResNet50
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python src/training/train.py
```

### Running the Web Application
```bash
python app/app.py
```

## Dataset

Source:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The project uses chest X-ray images for pneumonia detection. The data should be organized as follows:
```
data/
├── train/
│   ├── normal/
│   └── pneumonia/
├── test/
│   ├── normal/
│   └── pneumonia/
└── val/
    ├── normal/
    └── pneumonia/
```

## Project Workflow

Chest X-Ray
      │
      ▼
Convert to RGB
      │
      ▼
Resize (224 × 224)
      │
      ▼
Normalize (ImageNet Statistics)
      │
      ▼
Transfer Learning (ResNet50)
      │
      ▼
Binary Classification Layer
      │
      ▼
Training
      │
      ▼
Evaluation
      │
      ▼
Prediction

## Model Architecture

Pretrained ResNet50
ImageNet Weights
Frozen Feature Extractor
Custom Fully Connected Layer
Binary Classification
BCEWithLogitsLoss
Adam Optimizer

## Preprocessing

The following preprocessing steps were applied to every image:

Convert grayscale images to RGB
Resize images to 224 × 224
Normalize using ImageNet mean and standard deviation

Training images were also shuffled using PyTorch's DataLoader.

## Results

| Metric                | Score      |
| --------------------- | ---------- |
| Accuracy              | **75.64%** |
| Precision (Pneumonia) | **72%**    |
| Recall / Sensitivity  | **98.97%** |
| Specificity           | **36.75%** |
| ROC-AUC               | **0.9073** |


- Model accuracy and performance metrics are saved in `outputs/`
- Visualizations and plots are generated during training

## Author

Ananya Rathore
