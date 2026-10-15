"""Helpers that turn browser / upload data into NumPy RGB arrays."""
from __future__ import annotations

import base64
import binascii
import io

import numpy as np
from django.conf import settings
from PIL import Image, UnidentifiedImageError


def data_url_to_rgb(data_url: str) -> np.ndarray:
    """
    Convert a browser canvas data-URL
    ("data:image/jpeg;base64,....") into an (H, W, 3) uint8 RGB array.
    """
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Invalid image payload.")

    header, encoded = data_url.split(",", 1)
    if "base64" not in header:
        raise ValueError("Only base64 data URLs are supported.")

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image data is not valid base64.") from exc

    if not raw:
        raise ValueError("Empty image payload.")

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise ValueError(f"Image exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit.")

    return bytes_to_rgb(raw)


def bytes_to_rgb(raw: bytes) -> np.ndarray:
    """Decode raw image bytes into an RGB NumPy array."""
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img = img.convert("RGB")          # normalise colour space
            return np.asarray(img, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The file could not be read as an image.") from exc


def uploaded_file_to_rgb(django_file) -> np.ndarray:
    """Convert an InMemoryUploadedFile / TemporaryUploadedFile into RGB."""
    django_file.seek(0)
    return bytes_to_rgb(django_file.read())


def frame_to_jpeg_bytes(rgb: np.ndarray, quality: int = 85) -> bytes:
    """Encode an RGB array as JPEG bytes (useful for tests or saving)."""
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()