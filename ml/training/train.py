"""
Train an activity-classification model.

Usage
-----
    python ml/training/train.py --model cnn --epochs 15
    python ml/training/train.py --model mobilenetv2 --epochs 15
    python ml/training/train.py --model efficientnet --epochs 20 --batch-size 32

Pipeline
--------
Dataset -> Preprocessing -> Training -> Validation -> Evaluation -> Save model
        -> Django prediction

Key AI concepts
---------------
* Weights / Bias    : the learnable parameters of the network.
* Convolution       : Z = X * K + b, detects local patterns.
* ReLU              : max(0, x), introduces non-linearity.
* Pooling           : downsamples feature maps (keeps the strongest response).
* Softmax           : converts logits into probabilities.
* Loss function     : categorical cross-entropy,
                      L = -sum(y_i * log(p_i))
                      It measures how wrong the prediction is.
* Backpropagation   : the chain rule computes dL/dW for every weight.
* Gradient descent  : W <- W - lr * dL/dW, step by step towards lower loss.
* Transfer learning : start from ImageNet weights. The early layers already
                      detect edges and textures, so we only retrain the head.
                      Faster, and far more accurate when data is limited.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (standalone script - it does NOT need Django to be running)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "ml" / "dataset"
MODEL_DIR = BASE_DIR / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
AUTOTUNE = None  # set after TensorFlow import


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_custom_cnn(num_classes: int, augmentation=None):
    """
    Baseline CNN:

    Input 224x224x3
      -> Rescaling(1/255)      x_normalized = x / 255
      -> Conv2D(32) + ReLU     Z = X * K + b ; ReLU(Z) = max(0, Z)
      -> MaxPooling            halves width/height
      -> Conv2D(64) + ReLU
      -> MaxPooling
      -> Conv2D(128) + ReLU
      -> GlobalAveragePooling  averages each feature map -> compact vector
      -> Dense(128) + ReLU
      -> Dropout(0.4)          randomly zeroes 40 % of activations (regulariser)
      -> Dense(num_classes) + Softmax
                               P(class_i) = e^z_i / sum_j e^z_j
    """
    import tensorflow as tf
    from tensorflow import keras

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")
    x = augmentation(inputs) if augmentation is not None else inputs

    # Normalisation layer: x / 255, learned once and stored in the .keras file
    x = keras.layers.Rescaling(1.0 / 255.0, name="normalize")(x)

    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(x)
    x = keras.layers.MaxPooling2D(name="pool1")(x)

    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = keras.layers.MaxPooling2D(name="pool2")(x)

    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)

    x = keras.layers.Dense(128, activation="relu", name="fc1")(x)
    x = keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name="custom_cnn")


def build_transfer_model(num_classes: int, backbone: str = "mobilenetv2", augmentation=None):
    """
    MobileNetV2 / EfficientNetB0 with ImageNet weights.

    Why transfer learning?
    ----------------------
    ImageNet has 1.2 M images and 1000 classes. The convolutional filters the
    network learned there already encode edges, corners, textures and object
    parts - exactly the kind of low-level features a human-activity classifier
    also needs. We freeze those filters (base.trainable = False) and only train
    the new classification head. Result: excellent accuracy with a few thousand
    images instead of millions, and training finishes in minutes.
    """
    import tensorflow as tf
    from tensorflow import keras

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")
    x = augmentation(inputs) if augmentation is not None else inputs

    if backbone == "efficientnet":
        from tensorflow.keras.applications import EfficientNetB0

        # EfficientNet uses the same [-1, 1] scaling as MobileNet
        x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="normalize")(x)
        try:
            base = EfficientNetB0(
                include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
            )
        except Exception as exc:  # offline fallback
            print(f"[warn] Could not download ImageNet weights ({exc}). Using random init.")
            base = EfficientNetB0(
                include_top=False, weights=None, input_shape=(*IMG_SIZE, 3)
            )
    else:
        from tensorflow.keras.applications import MobileNetV2

        x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="normalize")(x)
        try:
            base = MobileNetV2(
                include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
            )
        except Exception as exc:
            print(f"[warn] Could not download ImageNet weights ({exc}). Using random init.")
            base = MobileNetV2(
                include_top=False, weights=None, input_shape=(*IMG_SIZE, 3)
            )

    base.trainable = False                     # freeze the pretrained filters

    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name=f"{backbone}_transfer")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_datasets(batch_size: int, validation_split: float, seed: int = 42):
    import tensorflow as tf

    global AUTOTUNE
    AUTOTUNE = tf.data.AUTOTUNE

    if not DATASET_DIR.is_dir():
        raise SystemExit(f"Dataset folder not found: {DATASET_DIR}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    # Prefetch keeps the GPU/CPU busy while the next batch is being decoded
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Train a human activity classifier.")
    parser.add_argument("--model", choices=["cnn", "mobilenetv2", "efficientnet"],
                        default="mobilenetv2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--fine-tune", action="store_true",
                        help="Unfreeze the last 30 layers of the backbone and "
                             "continue training with a very small learning rate.")
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow import keras

    print("TensorFlow:", tf.__version__)
    print("Dataset  :", DATASET_DIR)
    print("Model    :", args.model)

    train_ds, val_ds, class_names = load_datasets(
        args.batch_size, args.validation_split
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    if num_classes < 2:
        raise SystemExit(
            "Need at least 2 activity classes with images. "
            f"Add pictures to {DATASET_DIR}/<activity>/ and retry."
        )

    from ml.preprocessing import build_augmentation  # noqa: E402

    sys.path.insert(0, str(BASE_DIR))
    augmentation = build_augmentation()

    if args.model == "cnn":
        model = build_custom_cnn(num_classes, augmentation)
    else:
        model = build_transfer_model(num_classes, args.model, augmentation)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        # Categorical cross-entropy: L = -sum(y_i * log(p_i))
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_checkpoint.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Optional fine-tuning: unfreeze the top of the backbone and train gently
    if args.fine_tune and args.model != "cnn":
        print("\n--- Fine-tuning the top layers of the backbone ---")
        for layer in model.layers:
            if hasattr(layer, "layers"):        # the frozen base model
                layer.trainable = True
                for sub_layer in layer.layers[:-30]:
                    sub_layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate / 100),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=max(args.epochs // 2, 3),
            callbacks=callbacks,
        )

    # ---- Evaluation -----------------------------------------------------
    loss, accuracy = model.evaluate(val_ds)
    print(f"\nValidation accuracy: {accuracy:.4f}")
    print(f"Validation loss    : {loss:.4f}")

    # ---- Save -----------------------------------------------------------
    model_path = MODEL_DIR / "activity_model.keras"
    model.save(model_path)

    labels_path = MODEL_DIR / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "classes": class_names,
                "input_size": list(IMG_SIZE),
                "model_type": args.model,
                "val_accuracy": float(accuracy),
                "val_loss": float(loss),
            },
            fh,
            indent=2,
        )

    print(f"\nSaved model  -> {model_path}")
    print(f"Saved labels -> {labels_path}")
    print("\nRestart the Django server (or POST /api/train/) to load the new model.")
    return 0


if __name__ == "__main__":
    # Make `from ml.preprocessing import ...` work when run as a script
    sys.path.insert(0, str(BASE_DIR))
    raise SystemExit(main())