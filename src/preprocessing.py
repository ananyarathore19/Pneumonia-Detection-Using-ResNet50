# preprocess.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = "../data/chest_xray"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# Training Transform
# -----------------------------

train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),   # Convert grayscale to RGB
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),              # Data Augmentation
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Validation / Test Transform
# -----------------------------

test_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load Datasets
# -----------------------------

train_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/val",
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/test",
    transform=test_transform
)

# -----------------------------
# Create DataLoaders
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# Print Dataset Information
# -----------------------------

if __name__ == "__main__":

    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(val_dataset)}")
    print(f"Testing Images    : {len(test_dataset)}")

    images, labels = next(iter(train_loader))

    print("\nBatch Shape:", images.shape)
    print("Labels Shape:", labels.shape)
    print("Classes:", train_dataset.classes)