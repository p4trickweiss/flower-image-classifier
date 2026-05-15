import sys
import os
import yaml
import numpy as np
import joblib
from keras.models import load_model
from keras import Model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_data, split_data

config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/cnn_svm.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

run_name      = os.path.splitext(os.path.basename(config_path))[0]
backbone_path = f"models/{cfg['backbone']}_best_model.keras"

# Build feature extractor: output of Dense(512) layer, before Dropout + softmax
backbone  = load_model(backbone_path)
extractor = Model(inputs=backbone.inputs, outputs=backbone.layers[-3].output)
print(f"Backbone  : {backbone_path}")
print(f"Embedding : {extractor.output_shape[1]}-dim\n")

# Load tf_flowers and split identically to training
images, labels, class_names, num_classes = load_data(img_size=cfg["image_size"])
X_train, X_val, _, y_train, y_val, _    = split_data(images, labels)

# Extract CNN embeddings
print("Extracting features ...")
X_train_feat = extractor.predict(X_train, verbose=0)
X_val_feat   = extractor.predict(X_val,   verbose=0)
print(f"  Train : {X_train_feat.shape}")
print(f"  Val   : {X_val_feat.shape}\n")

# Train SVM with StandardScaler (CNN embeddings vary in scale)
print("Training SVM ...")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(C=cfg["svm_C"], kernel=cfg["svm_kernel"], probability=True)),
])
clf.fit(X_train_feat, y_train)

val_acc = clf.score(X_val_feat, y_val)
print(f"Val accuracy : {val_acc:.4f}  ({val_acc:.1%})\n")

# Save classifier (backbone file is shared, no copy needed)
os.makedirs("models", exist_ok=True)
out_path = f"models/{run_name}_classifier.pkl"
joblib.dump(clf, out_path)
print(f"Saved : {out_path}")
print(f"Run   : python src/train_hybrid.py configs/cnn_svm.yaml")
