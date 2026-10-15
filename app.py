"""Flask API:  python app.py  ->  POST an X-ray to /predict"""
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from src.config import MODEL_PATH, IMG_SIZE, CLASS_NAMES

app = Flask(__name__)
model = tf.keras.models.load_model(MODEL_PATH)
ALLOWED = {".jpg", ".jpeg", ".png"}


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Upload a file with key 'image'"}), 400
    file = request.files["image"]
    if not any(file.filename.lower().endswith(e) for e in ALLOWED):
        return jsonify({"error": "Only JPG/PNG images allowed"}), 400

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not read image"}), 400
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE).astype("float32")

    p = float(model.predict(img[None], verbose=0)[0][0])
    label = CLASS_NAMES[int(p >= 0.5)]
    return jsonify({
        "prediction": label,
        "probability_pneumonia": round(p, 4),
        "confidence": round(p if p >= 0.5 else 1 - p, 4),
        "disclaimer": "Educational demo. Not a medical diagnosis.",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)