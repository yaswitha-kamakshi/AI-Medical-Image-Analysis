"""Predict a single X-ray image."""
import numpy as np
import tensorflow as tf
from src.config import *


def predict_image(image_path, model=None):
    """Returns (label, confidence, probability_of_pneumonia)."""
    model = model or tf.keras.models.load_model(MODEL_PATH)
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = np.expand_dims(tf.keras.utils.img_to_array(img), 0)
    p = float(model.predict(arr, verbose=0)[0][0])
    label = CLASS_NAMES[int(p >= 0.5)]
    confidence = p if p >= 0.5 else 1 - p
    return label, confidence, p