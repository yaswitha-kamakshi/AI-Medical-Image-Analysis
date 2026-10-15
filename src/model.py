"""MobileNetV2 transfer-learning classifier."""
import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import IMG_SIZE


def build_model():
    """Returns (model, backbone). Backbone starts frozen."""
    inputs = layers.Input(shape=IMG_SIZE + (3,))

    # Augmentation: only active during training
    x = layers.RandomRotation(0.05)(inputs)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)

    # Scale pixels 0..255 -> -1..1 (what MobileNetV2 expects)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    backbone.trainable = False
    x = backbone(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs), backbone