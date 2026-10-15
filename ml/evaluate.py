"""
Evaluate the trained model on a held-out folder.

    python ml/evaluate.py
    python ml/evaluate.py --data ml/dataset --batch-size 32

Metrics explained
-----------------
* Accuracy  = (TP + TN) / total          - overall hit rate.
* Precision = TP / (TP + FP)             - of everything we called "walking",
                                            how much really was walking?
* Recall    = TP / (TP + FN)             - of all real "walking" samples, how
                                            many did we find?
* F1-score  = 2 * (P * R) / (P + R)      - harmonic mean of precision and recall,
                                            a single balanced number.
* Confusion matrix - rows = true class, columns = predicted class. The diagonal
                     is correct, everything else is a mistake.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "ml" / "models" / "activity_model.keras"
LABELS_PATH = BASE_DIR / "ml" / "models" / "labels.json"
DATASET_DIR = BASE_DIR / "ml" / "dataset"
REPORT_PATH = BASE_DIR / "ml" / "models" / "evaluation.json"

IMG_SIZE = (224, 224)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATASET_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", type=float, default=0.2,
                        help="Fraction of the data used as the validation set.")
    args = parser.parse_args()

    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return 1

    model = tf.keras.models.load_model(MODEL_PATH)

    ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        validation_split=args.split,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    class_names = ds.class_names

    y_true, y_pred = [], []
    for images, labels in ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())

    print("\n=== Classification report ===")
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    print(report)

    print("=== Confusion matrix (rows = true, cols = predicted) ===")
    cm = confusion_matrix(y_true, y_pred)
    header = "".join(f"{name[:10]:>12}" for name in class_names)
    print(f"{'':>12}{header}")
    for name, row in zip(class_names, cm):
        print(f"{name[:10]:>12}" + "".join(f"{v:>12}" for v in row))

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    print(f"\nOverall accuracy: {accuracy:.4f}")

    payload = {
        "classes": class_names,
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "report": classification_report(
            y_true, y_pred, target_names=class_names, digits=4,
            zero_division=0, output_dict=True,
        ),
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())