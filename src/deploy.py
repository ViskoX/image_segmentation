import tensorflow as tf
import numpy as np
import os
from src.utils import *  # your helper functions

config = Config()

MODEL_PATH = config.MODEL_PATH
IMAGE_DIR = config.IMAGE_DIR
OUTPUT_DIR = config.OUTPUT_DIR
IMG_HEIGHT, IMG_WIDTH = config.IMG_HEIGHT, config.IMG_WIDTH

# Load model normally
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# Load & preprocess images
def load_and_preprocess_images(img_dir):
    images, filenames = [], sorted(os.listdir(img_dir))
    for f in filenames:
        img = load_image(os.path.join(img_dir, f), target_size=(IMG_HEIGHT, IMG_WIDTH))
        img = preprocess_image(img)
        images.append(img)
    return np.array(images), filenames

X_new, filenames = load_and_preprocess_images(IMAGE_DIR)

# Predict
predictions = model.predict(X_new, batch_size=8)

# Save predictions
os.makedirs(OUTPUT_DIR, exist_ok=True)
for i, pred in enumerate(predictions):
    pred_img = (pred.squeeze() * 255).astype("uint8")
    if pred_img.ndim == 2:
        pred_img = np.expand_dims(pred_img, axis=-1)  # shape -> (H, W, 1)
    
    tf.keras.preprocessing.image.save_img(
        os.path.join(OUTPUT_DIR, filenames[i]),
        pred_img
    )

print(f"Predictions saved to {OUTPUT_DIR}")