"""
Image preprocessing utilities.

Mathematics used here
---------------------
1. Resize            : bilinear interpolation maps every output pixel back to
                       the source image and blends the 4 nearest neighbours.
2. Normalisation     : x_normalized = x / 255
                       Pixel values live in [0, 255]; dividing by 255 maps them
                       to [0, 1]. Neural networks converge much faster when
                       their inputs are small and centred.
3. Standardisation   : x_std = (x - mean) / std   (optional, for EfficientNet
                       style preprocessing).
4. Augmentation      : random flip / rotation / zoom artificially enlarges the
                       dataset. This is a regulariser - it reduces overfitting
                       (memorising the training set) and improves generalisation.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

IMG_SIZE: Tuple[int, int] = (224, 224)


# ---------------------------------------------------------------------------
# Basic NumPy / OpenCV helpers
# ---------------------------------------------------------------------------
def load_image_cv2(path: str | Path, size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Read an image with OpenCV, resize it and normalise it to [0, 1].

    Returns a float32 array of shape (224, 224, 3) in **RGB** order.
    """
    import cv2  # local import keeps `import ml.preprocessing` cheap

    image = cv2.imread(str(path))                 # BGR, uint8
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, Keras RGB
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    # x_normalized = x / 255
    image = image.astype("float32") / 255.0
    return image


def normalize_array(image: np.ndarray) -> np.ndarray:
    """x_normalized = x / 255 for an already-loaded uint8 array."""
    return image.astype("float32") / 255.0


def standardize_array(image: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """x_std = (x - mean) / std   -> roughly zero-mean, unit-variance input."""
    return (image.astype("float32") / 255.0 - mean) / std


# ---------------------------------------------------------------------------
# Keras augmentation pipeline
# ---------------------------------------------------------------------------
def build_augmentation():
    """
    Returns a Keras Sequential model of augmentation layers.

    These layers are ACTIVE ONLY during training (Keras preprocessing layers
    are no-ops in inference mode), so they never slow down prediction.

    * RandomFlip("horizontal") : mirrors the image - a person walking left is
      still walking when they walk right.
    * RandomRotation(0.1)      : rotates by up to +/- 36 degrees (0.1 * 360).
    * RandomZoom(0.15)         : zooms in/out by up to 15 %.
    * RandomContrast(0.1)      : small lighting variation.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
def list_dataset_classes(dataset_dir: str | Path) -> List[str]:
    root = Path(dataset_dir)
    if not root.is_dir():
        return []
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def count_dataset_images(dataset_dir: str | Path) -> dict:
    root = Path(dataset_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts = {}
    for folder in sorted([d for d in root.iterdir() if d.is_dir()]) if root.is_dir() else []:
        counts[folder.name] = sum(
            1 for p in folder.iterdir() if p.suffix.lower() in extensions
        )
    return counts


def split_indices(n_samples: int, validation_split: float = 0.2, seed: int = 42):
    """
    Return (train_idx, val_idx) using a reproducible random shuffle.

    Training set   : used to update the weights via gradient descent.
    Validation set : never used for weight updates - it tells us whether the
                     model generalises or is just memorising.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    cut = int(n_samples * (1.0 - validation_split))
    return indices[:cut], indices[cut:]