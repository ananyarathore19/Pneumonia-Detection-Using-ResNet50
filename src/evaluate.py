import os
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

import matplotlib.pyplot as plt

# ---------------------------------------------------
# Device Configuration
# ---------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

model = resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    1
)

model.load_state_dict(
    torch.load(
        "../src/models/best_model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

# ---------------------------------------------------
# Image Transform
# ---------------------------------------------------

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------------------------
# Test Dataset
# ---------------------------------------------------

test_dataset = ImageFolder(
    root="../data/chest_xray/test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

print("\nClass Mapping:")
print(test_dataset.class_to_idx)

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs).view(-1)

        predictions = (probabilities >= 0.5).long()

        all_labels.extend(labels.numpy())

        all_predictions.extend(predictions.cpu().numpy())

        all_probabilities.extend(probabilities.cpu().numpy())

# ---------------------------------------------------
# Accuracy
# ---------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\nOverall Accuracy")
print(f"{accuracy*100:.2f}%")

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix")

print(cm)

# ---------------------------------------------------
# Classification Report
# ---------------------------------------------------

print("\nClassification Report\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes
    )
)

# ---------------------------------------------------
# Sensitivity & Specificity
# ---------------------------------------------------

tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)

specificity = tn / (tn + fp)

print(f"Sensitivity : {sensitivity:.4f}")

print(f"Specificity : {specificity:.4f}")

# ---------------------------------------------------
# ROC-AUC
# ---------------------------------------------------

auc = roc_auc_score(
    all_labels,
    all_probabilities
)

print(f"ROC-AUC      : {auc:.4f}")

# ---------------------------------------------------
# Misclassified Images
# ---------------------------------------------------

print("\nMisclassified Images\n")

for i, (true, pred) in enumerate(zip(all_labels, all_predictions)):

    if true != pred:

        path, _ = test_dataset.samples[i]

        confidence = all_probabilities[i]

        if pred == 0:
            confidence = 1 - confidence

        print(
            f"{os.path.basename(path)}"
            f" | Actual: {test_dataset.classes[true]}"
            f" | Predicted: {test_dataset.classes[pred]}"
            f" | Confidence: {confidence*100:.2f}%"
        )

# ---------------------------------------------------
# Plot Confusion Matrix
# ---------------------------------------------------

plt.figure(figsize=(5,5))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks([0,1], test_dataset.classes)

plt.yticks([0,1], test_dataset.classes)

plt.xlabel("Predicted")

plt.ylabel("Actual")

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="white",
            fontsize=14
        )

plt.tight_layout()

os.makedirs("../outputs", exist_ok=True)

plt.savefig(
    "../outputs/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved to outputs/confusion_matrix.png")