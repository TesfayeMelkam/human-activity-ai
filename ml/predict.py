"""
Command-line prediction.

    python ml/predict.py path/to/image.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "ml" / "models" / "activity_model.keras"
LABELS_PATH = BASE_DIR / "ml" / "models" / "labels.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify a single image.")
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    import numpy as np
    import tensorflow as tf
    from PIL import Image

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Train it first:  python ml/training/train.py --model mobilenetv2")
        return 1

    model = tf.keras.models.load_model(MODEL_PATH)
    classes = json.loads(LABELS_PATH.read_text(encoding="utf-8"))["classes"] \
        if LABELS_PATH.exists() else []

    img = Image.open(args.image).convert("RGB").resize((224, 224))
    batch = np.asarray(img, dtype="float32")[np.newaxis, ...]  # model does /255

    probs = model.predict(batch, verbose=0)[0]
    order = np.argsort(probs)[::-1]

    print(f"\nImage: {args.image}\n")
    for i in order[: args.top]:
        name = classes[i] if i < len(classes) else f"class_{i}"
        print(f"  {name:<18} {probs[i]:.4f}  ({probs[i] * 100:.2f} %)")

    best = order[0]
    print(f"\nPrediction: {classes[best] if best < len(classes) else best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())