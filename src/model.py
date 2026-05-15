from keras import Input, layers, models

def build_cnn(input_shape, num_classes, num_blocks=4, filters_start=32, dropout_rate=0.5, dense_units=512):
    model_layers = [
        Input(shape=input_shape),

        # augmentation (only active during training)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2, value_range=(0, 1)),
        layers.RandomContrast(0.2),
    ]

    for i in range(num_blocks):
        filters = filters_start * (2 ** i)
        model_layers += [
            layers.Conv2D(filters, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
        ]

    model_layers += [
        layers.Flatten(),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax"),
    ]

    return models.Sequential(model_layers)

import json
import os
import sys
import yaml
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD, RMSprop

OPTIMIZERS = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}
from preprocess import load_data, split_data
from model import build_cnn

# accept optional config path: python src/train.py configs/shallow_sgd.yaml
config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/baseline.yaml"
run_name = os.path.splitext(os.path.basename(config_path))[0]

with open(config_path) as f:
    cfg = yaml.safe_load(f)

# load and split data
img_size = cfg["image_size"]
images, labels, class_names, num_classes = load_data(img_size=img_size)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, labels)

# one-hot encode labels
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat   = to_categorical(y_val,   num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)

# build model
model = build_cnn(
    input_shape=(img_size, img_size, 3),
    num_classes=num_classes,
    num_blocks=cfg.get("num_blocks", 4),
    filters_start=cfg.get("filters_start", 32),
    dropout_rate=cfg.get("dropout_rate", 0.5),
    dense_units=cfg.get("dense_units", 512),
)
optimizer_cls = OPTIMIZERS[cfg["optimizer"]]
model.compile(
    optimizer=optimizer_cls(learning_rate=cfg["learning_rate"]),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

os.makedirs("models", exist_ok=True)

# callbacks — outputs named after config so runs don't overwrite each other
callbacks = [
    ModelCheckpoint(f"models/{run_name}_best_model.keras", save_best_only=True, monitor="val_accuracy"),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5),
]

# train
history = model.fit(
    X_train, y_train_cat,
    batch_size=cfg["batch_size"],
    validation_data=(X_val, y_val_cat),
    epochs=cfg["epochs"],
    callbacks=callbacks,
)

# evaluate on held-out test set
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest accuracy : {test_acc:.4f}")
print(f"Test loss     : {test_loss:.4f}")

# save history for local analysis
with open(f"models/{run_name}_history.json", "w") as f:
    json.dump(history.history, f)
