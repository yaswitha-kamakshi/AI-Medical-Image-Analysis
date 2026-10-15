"""Baseline vs MobileNetV2 on the same test set."""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.config import *
from src.data_loader import load_datasets

_, _, test_ds = load_datasets()
y_true = np.concatenate([y.numpy() for _, y in test_ds]).ravel()

rows = []
for name, path in [("Baseline CNN", ROOT / "models" / "baseline_cnn.keras"),
                   ("MobileNetV2", MODEL_PATH)]:
    prob = tf.keras.models.load_model(path).predict(test_ds).ravel()
    pred = (prob >= 0.5).astype(int)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, pred),
        "Pneumonia recall": recall_score(y_true, pred),
        "Pneumonia precision": precision_score(y_true, pred),
        "NORMAL recall": recall_score(y_true, pred, pos_label=0),
        "ROC-AUC": roc_auc_score(y_true, prob),
    })

df = pd.DataFrame(rows).round(3)
df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
print(df.to_string(index=False))