from pathlib import Path
from PIL import Image
from collections import Counter

counter = Counter()

for img_path in Path("data/chest_xray").rglob("*.jpeg"):
    img = Image.open(img_path)
    counter[img.mode] += 1

print(counter)