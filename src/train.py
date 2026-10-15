"""Two-stage training: head first, then light fine-tuning."""
import matplotlib.pyplot as plt
import tensorflow as tf
from src.config import *
from src.data_loader import load_datasets, compute_class_weights
from src.model import build_model


def _compile(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Recall(name="recall")])


def _plot_history(hist):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, key in zip(ax, ["accuracy", "loss"]):
        a.plot(hist[key], label="train")
        a.plot(hist["val_" + key], label="validation")
        a.set_title(key.capitalize())
        a.set_xlabel("Epoch")
        a.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close()


def train():
    train_ds, val_ds, _ = load_datasets()
    weights = compute_class_weights()
    print("Class weights:", weights)

    model, backbone = build_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss",
                                           save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=1),
        tf.keras.callbacks.CSVLogger(str(OUTPUT_DIR / "training_log.csv"), append=True),
    ]

    # Stage 1: train only the new classification head
    _compile(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                   class_weight=weights, callbacks=callbacks)

    # Stage 2: unfreeze the last 30 backbone layers, tiny learning rate
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    _compile(model, 1e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
                   class_weight=weights, callbacks=callbacks)

    history = {k: h1.history[k] + h2.history[k] for k in h1.history}
    _plot_history(history)
    print(f"Model saved to {MODEL_PATH}")