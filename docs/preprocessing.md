# Preprocessing Decisions

- **Resize to 224x224:** MobileNetV2's native input size.
- **RGB (3 channels):** the pretrained backbone expects 3 channels, so grayscale X-rays are converted.
- **Scaling to [-1, 1]:** what MobileNetV2 expects, done inside the model.
- **Augmentation:** small rotation, zoom and contrast. No horizontal flip, because flipping a chest X-ray moves the heart to the wrong side.
- **CLAHE:** shown for visualization only; the model pipeline does not use it.