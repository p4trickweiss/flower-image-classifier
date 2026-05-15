import sys
import os
import yaml
import numpy as np
import joblib
from keras.models import load_model
from keras import Model
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_test_images

CLASS_NAMES = ["dandelion", "daisy", "tulips", "sunflowers", "roses"]

# accept optional config and custom test folder as arguments:
#   python src/evaluate.py configs/baseline.yaml data/test-images
config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/baseline.yaml"
test_folder = sys.argv[2] if len(sys.argv) > 2 else "data/test-images"

run_name = os.path.splitext(os.path.basename(config_path))[0]

with open(config_path) as f:
    cfg = yaml.safe_load(f)

print(f"Loading images from '{test_folder}' ...")
images, labels = load_test_images(test_folder, CLASS_NAMES, img_size=cfg["image_size"])
images = np.array(images)
labels = np.array(labels)
print(f"  {len(images)} images loaded\n")

clf_path = f"models/{run_name}_classifier.pkl"
if os.path.exists(clf_path):
    # Hybrid: CNN feature extractor + SVM
    backbone  = load_model(f"models/{cfg['backbone']}_best_model.keras")
    extractor = Model(inputs=backbone.inputs, outputs=backbone.layers[-3].output)
    feats     = extractor.predict(images, verbose=0)
    preds     = joblib.load(clf_path).predict(feats)
    print(f"Model: {cfg['backbone']}_best_model.keras  +  {clf_path}\n")
else:
    # Pure CNN
    model = load_model(f"models/{run_name}_best_model.keras")
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    print(f"Model: models/{run_name}_best_model.keras\n")

accuracy = np.mean(preds == labels)
print(f"Accuracy: {accuracy:.1%}  ({np.sum(preds == labels)}/{len(labels)})\n")
print(classification_report(labels, preds, target_names=CLASS_NAMES))

cm = confusion_matrix(labels, preds)
print("Confusion matrix (rows=true, cols=predicted):")
header = f"{'':12}" + "".join(f"{n:>12}" for n in CLASS_NAMES)
print(header)
for i, row in enumerate(cm):
    print(f"{CLASS_NAMES[i]:12}" + "".join(f"{v:>12}" for v in row))
