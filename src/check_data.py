"""Counts images per split/class and shows one image."""
import cv2
import matplotlib.pyplot as plt
from src.config import DATA_DIR, CLASS_NAMES

for split in ["train", "val", "test"]:
    for cls in CLASS_NAMES:
        n = len(list((DATA_DIR / split / cls).glob("*.jp*g")))
        print(f"{split:5s} {cls:10s} {n}")

sample = next((DATA_DIR / "train" / "PNEUMONIA").glob("*.jp*g"))
img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
print("Sample shape:", img.shape)
plt.imshow(img, cmap="gray")
plt.title("Sample X-ray")
plt.axis("off")
plt.show()