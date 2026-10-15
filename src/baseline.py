"""Baseline CNN to compare against MobileNetV2."""
import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import *
from src.data_loader import load_datasets, compute_class_weights


def build_baseline():
    m = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        layers.Rescaling(1 / 255),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])
    m.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return m


if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_datasets()
    model = build_baseline()
    model.fit(train_ds, validation_data=val_ds, epochs=8,
              class_weight=compute_class_weights())
    loss, acc = model.evaluate(test_ds)
    print(f"Baseline test accuracy: {acc:.1%}")
    model.save(ROOT / "models" / "baseline_cnn.keras")