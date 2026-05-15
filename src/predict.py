import sys
import os
import numpy as np
from keras.models import load_model
from PIL import Image

CLASS_NAMES = ["dandelion", "daisy", "tulips", "sunflowers", "roses"]

def predict(image_path, model_path):
    model = load_model(model_path)

    img = Image.open(image_path).convert("RGB").resize((128, 128))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    probs = model.predict(img, verbose=0)[0]
    predicted_class = CLASS_NAMES[np.argmax(probs)]
    confidence = probs.max()

    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.1%}")
    print()
    for name, prob in zip(CLASS_NAMES, probs):
        bar = "█" * int(prob * 30)
        print(f"  {name:<12} {prob:.1%}  {bar}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path> [config_name]")
        print("  config_name defaults to 'baseline'")
        sys.exit(1)
    image_path = sys.argv[1]
    config_name = os.path.splitext(os.path.basename(sys.argv[2]))[0] if len(sys.argv) > 2 else "baseline"
    model_path = f"models/{config_name}_best_model.keras"
    predict(image_path, model_path)
