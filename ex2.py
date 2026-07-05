from pathlib import Path
from PIL import Image

widths = []
heights = []

for img_path in Path("data/chest_xray").rglob("*.jpeg"):
    img = Image.open(img_path)

    widths.append(img.width)
    heights.append(img.height)

print(f"Minimum size : {min(widths)} x {min(heights)}")
print(f"Maximum size : {max(widths)} x {max(heights)}")
print(f"Average width: {sum(widths)/len(widths):.1f}")
print(f"Average height: {sum(heights)/len(heights):.1f}")