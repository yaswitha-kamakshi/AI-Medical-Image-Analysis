"""Loads images from folders into tf.data pipelines."""
import tensorflow as tf
from src.config import *


def load_datasets():
    """Return train, validation and test datasets.

    The Kaggle val folder has only 16 images (too small to trust),
    so we carve 20% of the training set as validation instead.
    """
    common = dict(image_size=IMG_SIZE, batch_size=BATCH_SIZE,
                  label_mode="binary", class_names=CLASS_NAMES)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=0.2, subset="training", seed=SEED, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=0.2, subset="validation", seed=SEED, **common)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, shuffle=False, **common)   # no shuffle -> labels stay aligned

    auto = tf.data.AUTOTUNE
    return (train_ds.prefetch(auto), val_ds.prefetch(auto), test_ds.prefetch(auto))


def compute_class_weights():
    """Balance the classes: pneumonia images outnumber normal ones about 3 to 1."""
    counts = [len(list((TRAIN_DIR / c).glob("*.jp*g"))) for c in CLASS_NAMES]
    total = sum(counts)
    return {i: total / (2 * n) for i, n in enumerate(counts)}