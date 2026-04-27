import sys
import numpy as np
from keras.models import load_model
from PIL import Image

CLASS_NAMES = ["dandelion", "daisy", "tulips", "sunflowers", "roses"]

def predict(image_path):
    model = load_model("models/best_model.keras")

    img = Image.open(image_path).convert("RGB").resize((128, 128))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # shape: (1, 128, 128, 3)

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
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
