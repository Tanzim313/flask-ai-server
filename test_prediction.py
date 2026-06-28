import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import json

model_path = os.path.join("models", "plant_disease_final_model.h5")
labels_path = os.path.join("models", "labels.json")
test_image_path = os.path.join("..", "server", "test.png")

if not os.path.exists(test_image_path):
    print(f"Test image not found at {test_image_path}")
    exit(1)

# Load model and labels
model = tf.keras.models.load_model(model_path, compile=False)
with open(labels_path, 'r', encoding='utf-8') as f:
    labels = json.load(f)

# Load and preprocess image
img = Image.open(test_image_path).convert('RGB')
img = ImageOps.exif_transpose(img)
img = img.resize((224, 224), Image.Resampling.LANCZOS)
img_array = np.asarray(img, dtype=np.float32)

# Case 1: Preprocessed with division by 255.0
input_div_255 = np.expand_dims(img_array / 255.0, axis=0)
pred_div_255 = model.predict(input_div_255, verbose=0)[0]
idx_div = np.argmax(pred_div_255)
conf_div = pred_div_255[idx_div]

# Case 2: Preprocessed without division by 255.0 (0-255 range)
input_raw = np.expand_dims(img_array, axis=0)
pred_raw = model.predict(input_raw, verbose=0)[0]
idx_raw = np.argmax(pred_raw)
conf_raw = pred_raw[idx_raw]

print("\n=== PREDICTION RESULTS ===")
print("Case 1 (Divided by 255.0):")
print(f"  Class: {labels[idx_div]} (index {idx_div})")
print(f"  Confidence/Raw Output: {conf_div:.6f}")
print(f"  Raw Softmax sum: {np.sum(pred_div_255):.4f}")

print("\nCase 2 (Raw 0-255 values):")
print(f"  Class: {labels[idx_raw]} (index {idx_raw})")
print(f"  Confidence/Raw Output: {conf_raw:.6f}")
print(f"  Raw Softmax sum: {np.sum(pred_raw):.4f}")
