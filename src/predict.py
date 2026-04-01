import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

# HARD disable GPU at runtime
tf.config.set_visible_devices([], 'GPU')


import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model.h5"
TEST_DIR = "/Users/pratik/Documents/Finalized Projects/Distracted_Driver_detection/dataset/imgs/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Get image list
image_files = sorted([
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

predictions_list = []

# Batch processing (memory safe)
for i in range(0, len(image_files), BATCH_SIZE):
    batch_files = image_files[i:i+BATCH_SIZE]
    batch_images = []

    for file in batch_files:
        img_path = os.path.join(TEST_DIR, file)
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img = image.img_to_array(img) / 255.0
        batch_images.append(img)

    batch_images = np.array(batch_images)

    preds = model.predict(batch_images)
    predictions_list.append(preds)

# Combine all predictions
predictions = np.vstack(predictions_list)

# Create CSV
num_classes = predictions.shape[1]
columns = [f'c{i}' for i in range(num_classes)]

df = pd.DataFrame(predictions, columns=columns)
df.insert(0, "img", image_files)

df.to_csv("predictions.csv", index=False)

print("✅ predictions.csv created successfully")