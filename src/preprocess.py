import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
from sklearn.model_selection import train_test_split
import os

# path is relative to the project root, one level up from src/
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def load_data(img_size=128):
    img_size = (img_size, img_size)

    full_dataset, info = tfds.load(
        "tf_flowers",
        split="train",
        as_supervised=True,
        with_info=True,
        data_dir=DATA_DIR
    )
    num_classes = info.features["label"].num_classes
    class_names = info.features["label"].names

    images, labels = [], []
    for img, label in full_dataset:
        img = Image.fromarray(img.numpy()).resize(img_size)
        img = np.array(img, dtype=np.float32) / 255.0
        images.append(img)
        labels.append(label.numpy())

    return np.array(images), np.array(labels), class_names, num_classes

def split_data(images, labels):
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_test_images(folder, class_names, img_size=128):
    """Load labeled images from folder/class_name/*.jpg structure."""
    img_size = (img_size, img_size)
    images, labels = [], []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(class_dir, fname)
            try:
                img = Image.open(img_path).convert("RGB").resize(img_size)
            except Exception as e:
                print(f"  Skipping {img_path}: {e}")
                continue
            img = np.array(img, dtype=np.float32) / 255.0
            images.append(img)
            labels.append(label_idx)
    return np.array(images), np.array(labels)