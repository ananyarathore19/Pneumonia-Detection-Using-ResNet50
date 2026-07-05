import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet50, ResNet50_Weights

from preprocessing import train_loader, val_loader

# ---------------------------------------------------
# Device Configuration
# ---------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------
# Load Pretrained ResNet50
# ---------------------------------------------------

model = resnet50(weights=ResNet50_Weights.DEFAULT)

# ---------------------------------------------------
# Freeze Feature Extractor
# ---------------------------------------------------

for param in model.parameters():
    param.requires_grad = False

# ---------------------------------------------------
# Replace Final Layer
# ---------------------------------------------------

model.fc = nn.Linear(
    model.fc.in_features,
    1
)

model = model.to(device)

# ---------------------------------------------------
# Loss Function
# ---------------------------------------------------

criterion = nn.BCEWithLogitsLoss()

# ---------------------------------------------------
# Optimizer
# ---------------------------------------------------

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=1e-4
)

# ---------------------------------------------------
# Training Configuration
# ---------------------------------------------------

EPOCHS = 5

best_val_loss = float("inf")

os.makedirs("models", exist_ok=True)

# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------

for epoch in range(EPOCHS):

    # -------------------------
    # Training
    # -------------------------

    model.train()

    running_train_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)

    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    running_val_loss = 0

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            predictions = torch.sigmoid(outputs)

            predictions = (predictions >= 0.5).float()

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    avg_val_loss = running_val_loss / len(val_loader)

    val_accuracy = 100 * correct / total

    # -------------------------
    # Print Results
    # -------------------------

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    print(f"Training Loss   : {avg_train_loss:.4f}")
    print(f"Validation Loss : {avg_val_loss:.4f}")
    print(f"Validation Acc  : {val_accuracy:.2f}%")

    # -------------------------
    # Save Best Model
    # -------------------------

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(
            model.state_dict(),
            "models/best_model.pth"
        )

        print("Best model saved!")

print("\nTraining Complete!")