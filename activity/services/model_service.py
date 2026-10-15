"""
Loads the trained Keras model once and keeps it in memory.

Key AI ideas explained here
---------------------------
* Convolution  : a small filter (kernel) slides over the image computing
                 Z = X * K + b  ->  it learns edges, shapes and body poses.
* Weights (K)  : the numbers the network learns during training.
* Bias (b)     : a learnable offset added after the convolution.
* ReLU         : ReLU(x) = max(0, x) -> adds non-linearity.
* Pooling      : shrinks the feature map, keeps the strongest activations.
* Softmax      : P(class_i) = e^z_i / sum(e^z_j) -> turns scores into
                 probabilities that sum to 1.
* Transfer learning: reuse a network already trained on ImageNet. The early
                 layers already know edges/textures, so we only need to teach
                 the last layers our 7 activities. This needs far less data
                 and trains much faster than starting from scratch.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from django.conf import settings

# Demo classes used when no trained model exists yet
DEFAULT_CLASSES: List[str] = list(settings.DEFAULT_ACTIVITIES)


class ModelService:
    """
    Thread-safe singleton wrapper around the Keras model.

    If no trained model file exists we fall back to a transparent geometric
    heuristic so that the whole web application is still demonstrable.
    The API always reports ``demo_mode: true`` in that case.
    """

    _instance: Optional["ModelService"] = None
    _lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Singleton plumbing
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        self.model = None
        self.classes: List[str] = list(DEFAULT_CLASSES)
        self.model_name: str = "demo-heuristic"
        self.loaded: bool = False
        self.load_error: Optional[str] = None
        self.img_size: Tuple[int, int] = tuple(settings.ML_IMG_SIZE)  # type: ignore

    @classmethod
    def instance(cls) -> "ModelService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self, force: bool = False) -> bool:
        """Load the .keras model and labels.json. Returns True on success."""
        if self.loaded and not force:
            return True

        model_path = Path(settings.ML_MODEL_PATH)
        labels_path = Path(settings.ML_LABELS_PATH)

        if not model_path.exists():
            self.load_error = (
                f"No trained model found at {model_path}. "
                "Run: python ml/training/train.py --model mobilenetv2"
            )
            self.loaded = False
            self.model = None
            self.model_name = "demo-heuristic"
            return False

        try:
            # Imported lazily: TensorFlow takes ~5 s to import and we do not
            # want that cost on every Django request.
            import tensorflow as tf  # noqa: WPS433

            self.model = tf.keras.models.load_model(model_path)
            self.model_name = model_path.stem

            if labels_path.exists():
                with open(labels_path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                self.classes = payload.get("classes", DEFAULT_CLASSES)
            else:
                self.classes = list(DEFAULT_CLASSES)

            self.loaded = True
            self.load_error = None
            return True
        except Exception as exc:  # pragma: no cover - depends on environment
            self.load_error = f"Failed to load model: {exc}"
            self.model = None
            self.loaded = False
            self.model_name = "demo-heuristic"
            return False

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(self, rgb_image: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Parameters
        ----------
        rgb_image : np.ndarray  (H, W, 3) uint8, RGB order

        Returns
        -------
        (activity, confidence, {class: probability})
        """
        if self.model is None and not self.loaded:
            self.load()

        if self.model is None:
            return self._heuristic(rgb_image)

        import cv2  # noqa: WPS433

        resized = cv2.resize(rgb_image, self.img_size, interpolation=cv2.INTER_AREA)
        # The model contains a Rescaling layer (x / 255) so we feed 0..255.
        batch = resized.astype("float32")[np.newaxis, ...]

        probs = self.model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        activity = self.classes[idx] if idx < len(self.classes) else "unknown"

        prob_map = {
            (self.classes[i] if i < len(self.classes) else f"class_{i}"): float(p)
            for i, p in enumerate(probs)
        }
        return activity, confidence, prob_map

    # ------------------------------------------------------------------ #
    # Fallback "demo mode" heuristic
    # ------------------------------------------------------------------ #
    def _heuristic(self, rgb_image: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        A deliberately simple silhouette-geometry classifier used only when no
        trained model is available. It estimates the background from the image
        border, thresholds the difference to get a foreground mask, then looks
        at the bounding-box aspect ratio and area ratio.

        This is NOT a replacement for the CNN - it exists so the dashboard,
        API and camera pages are testable before training.
        """
        import cv2  # noqa: WPS433

        img = cv2.resize(rgb_image, (224, 224), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        background = float(np.median(border))

        diff = np.abs(gray.astype("float32") - background)
        mask = (diff > 25).astype("uint8") * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "uncertain", 0.20, {}

        biggest = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(biggest)
        if w == 0 or h == 0:
            return "uncertain", 0.20, {}

        aspect = h / float(w)                      # tall vs wide silhouette
        area_ratio = (w * h) / float(224 * 224)    # how much of the frame is body

        if area_ratio < 0.04:
            activity, confidence = "uncertain", 0.25
        elif aspect <= 0.95:
            activity, confidence = "lying", 0.62
        elif aspect >= 2.0 and area_ratio >= 0.45:
            activity, confidence = "standing", 0.58
        elif area_ratio >= 0.45:
            activity, confidence = "walking", 0.52
        elif area_ratio >= 0.18:
            activity, confidence = "sitting", 0.55
        else:
            activity, confidence = "uncertain", 0.35

        return activity, confidence, {activity: confidence}

    # ------------------------------------------------------------------ #
    # Metadata for /api/model-status/
    # ------------------------------------------------------------------ #
    def status(self) -> Dict:
        return {
            "loaded": self.loaded,
            "demo_mode": not self.loaded,
            "model_name": self.model_name,
            "classes": self.classes,
            "input_size": list(self.img_size),
            "confidence_threshold": settings.ML_CONFIDENCE_THRESHOLD,
            "model_path": str(settings.ML_MODEL_PATH),
            "error": self.load_error,
        }