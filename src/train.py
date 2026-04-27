import json
import os
import yaml
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD, RMSprop

OPTIMIZERS = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}
from preprocess import load_data, split_data
from model import build_cnn

# load config
with open("configs/baseline.yaml") as f:
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
model = build_cnn(input_shape=(img_size, img_size, 3), num_classes=num_classes)
optimizer_cls = OPTIMIZERS[cfg["optimizer"]]
model.compile(
    optimizer=optimizer_cls(learning_rate=cfg["learning_rate"]),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

os.makedirs("models", exist_ok=True)

# callbacks
callbacks = [
    ModelCheckpoint("models/best_model.keras", save_best_only=True, monitor="val_accuracy"),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

# train
history = model.fit(
    X_train, y_train_cat,
    batch_size=cfg["batch_size"],
    validation_data=(X_val, y_val_cat),
    epochs=cfg["epochs"],
    callbacks=callbacks
)

# evaluate on held-out test set
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest accuracy : {test_acc:.4f}")
print(f"Test loss     : {test_loss:.4f}")

# save history for local analysis
with open("models/history.json", "w") as f:
    json.dump(history.history, f)