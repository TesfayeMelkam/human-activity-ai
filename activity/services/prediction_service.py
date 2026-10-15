"""Ties the model, the image helpers and the database together."""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from PIL import Image
import io

from ..models import Prediction
from .model_service import ModelService


def predict_array(
    rgb_image: np.ndarray,
    source: str = "camera",
    save: bool = True,
    image_bytes: Optional[bytes] = None,
    image_name: Optional[str] = None,
) -> Dict:
    """
    Run inference on an RGB array and (optionally) store the result.

    Returns a JSON-serialisable dictionary:
        {
            "activity": "walking",
            "confidence": 0.94,
            "processing_time_ms": 42,
            "uncertain": False,
            "demo_mode": False,
            "model": "activity_model",
            "probabilities": {...}   # only when a real model is loaded
        }
    """
    service = ModelService.instance()

    started = time.perf_counter()
    activity, confidence, probabilities = service.predict(rgb_image)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Never force a low-confidence guess: report "uncertain" instead.
    uncertain = confidence < settings.ML_CONFIDENCE_THRESHOLD
    label = "uncertain" if uncertain else activity

    result = {
        "activity": label,
        "raw_activity": activity,
        "confidence": round(float(confidence), 4),
        "confidence_percent": round(float(confidence) * 100, 2),
        "processing_time_ms": round(elapsed_ms, 2),
        "uncertain": bool(uncertain),
        "demo_mode": not service.loaded,
        "model": service.model_name,
        "threshold": settings.ML_CONFIDENCE_THRESHOLD,
    }
    if probabilities:
        result["probabilities"] = {
            k: round(v, 4)
            for k, v in sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        }

    if save:
        record = Prediction(
            activity=label,
            confidence=float(confidence),
            processing_time=elapsed_ms,
            source=source if source in dict(Prediction.SOURCE_CHOICES) else "api",
            model_name=service.model_name,
            is_uncertain=bool(uncertain),
        )
        if image_bytes and image_name:
            record.image.save(image_name, ContentFile(image_bytes), save=False)
        record.save()
        result["prediction_id"] = record.id
        result["created_at"] = record.created_at.isoformat()

    return result


def predict_uploaded_file(django_file, source: str = "upload") -> Dict:
    """Full upload pipeline: validate -> decode -> resize -> normalise -> predict."""
    from .camera_service import uploaded_file_to_rgb

    rgb = uploaded_file_to_rgb(django_file)

    # Save a small JPEG copy so the history page can show a thumbnail.
    thumb = Image.fromarray(rgb)
    thumb.thumbnail((480, 480))
    buffer = io.BytesIO()
    thumb.convert("RGB").save(buffer, format="JPEG", quality=85)
    image_bytes = buffer.getvalue()

    name = getattr(django_file, "name", "upload.jpg")
    safe_name = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

    return predict_array(
        rgb,
        source=source,
        save=True,
        image_bytes=image_bytes,
        image_name=safe_name,
    )