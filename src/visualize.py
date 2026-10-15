"""All plots used as GitHub proof."""
import random
import cv2
import matplotlib.pyplot as plt
from src.config import *


def _random_files(folder, n):
    files = list(folder.glob("*.jp*g"))
    random.Random(SEED).shuffle(files)
    return files[:n]


def show_sample_images(n=4):
    """Grid of raw dataset images, one row per class."""
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for r, cls in enumerate(CLASS_NAMES):
        for c, f in enumerate(_random_files(TRAIN_DIR / cls, n)):
            axes[r, c].imshow(cv2.imread(str(f), 0), cmap="gray")
            axes[r, c].set_title(cls)
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_images.png", dpi=150)
    plt.close()


def show_preprocessing():
    """Original -> resized -> contrast-enhanced (CLAHE) for one image."""
    f = _random_files(TRAIN_DIR / "PNEUMONIA", 1)[0]
    orig = cv2.imread(str(f), 0)
    resized = cv2.resize(orig, IMG_SIZE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    titles = [f"Original {orig.shape}", "Resized 224x224", "CLAHE enhanced"]
    for a, im, t in zip(ax, [orig, resized, clahe], titles):
        a.imshow(im, cmap="gray")
        a.set_title(t)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "preprocessing.png", dpi=150)
    plt.close()
def prediction_grid(n_per_class=4):
    """Actual vs predicted on test images. Green = correct, red = wrong."""
    import tensorflow as tf
    from src.predict import predict_image
    model = tf.keras.models.load_model(MODEL_PATH)
    cases = [(f, cls) for cls in CLASS_NAMES
             for f in _random_files(TEST_DIR / cls, n_per_class)]
    fig, axes = plt.subplots(2, n_per_class, figsize=(3.2 * n_per_class, 7))
    for ax, (f, actual) in zip(axes.ravel(), cases):
        label, conf, _ = predict_image(f, model)
        ax.imshow(cv2.imread(str(f), 0), cmap="gray")
        ax.axis("off")
        ax.set_title(f"Actual: {actual}\nPred: {label} ({conf:.0%})", fontsize=9,
                     color="green" if label == actual else "red")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_grid.png", dpi=150)
    plt.close()