import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

# ---------------------------------------------------
# Device Configuration
# ---------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ---------------------------------------------------
# Load Model Architecture
# ---------------------------------------------------

model = resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    1
)

# ---------------------------------------------------
# Load Trained Weights
# ---------------------------------------------------

model.load_state_dict(
    torch.load("models/best_model.pth", map_location=device)
)

model.to(device)

model.eval()

# ---------------------------------------------------
# Preprocessing
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
# Image Path
# ---------------------------------------------------

image_path = input("Enter image path: ")

# ---------------------------------------------------
# Load Image
# ---------------------------------------------------

image = Image.open(image_path)

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

with torch.no_grad():

    output = model(image)

    probability = torch.sigmoid(output)

    confidence = probability.item()

# ---------------------------------------------------
# Display Prediction
# ---------------------------------------------------

if confidence >= 0.5:

    print("\nPrediction : PNEUMONIA")
    print(f"Confidence : {confidence*100:.2f}%")

else:

    print("\nPrediction : NORMAL")
    print(f"Confidence : {(1-confidence)*100:.2f}%")