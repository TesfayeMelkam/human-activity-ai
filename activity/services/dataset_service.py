"""Filesystem-backed dataset management (ml/dataset/<class>/<image>.jpg)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from django.conf import settings
from django.core.files.storage import default_storage
from django.utils.text import slugify

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def dataset_root() -> Path:
    root = Path(settings.ML_DATASET_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_default_classes() -> None:
    for name in settings.DEFAULT_ACTIVITIES:
        (dataset_root() / name).mkdir(parents=True, exist_ok=True)


def count_images(class_dir: Path) -> int:
    if not class_dir.is_dir():
        return 0
    return sum(1 for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def list_classes() -> List[Dict]:
    root = dataset_root()
    rows = []
    for folder in sorted([d for d in root.iterdir() if d.is_dir()]):
        rows.append(
            {
                "name": folder.name,
                "slug": slugify(folder.name),
                "count": count_images(folder),
                "path": str(folder),
            }
        )
    return rows


def total_images() -> int:
    return sum(row["count"] for row in list_classes())


def create_class(name: str) -> Path:
    clean = slugify(name)
    if not clean:
        raise ValueError("Class name must contain letters or digits.")
    folder = dataset_root() / clean
    if folder.exists():
        raise ValueError(f"Class '{clean}' already exists.")
    folder.mkdir(parents=True)
    return folder


def _resolve_inside_root(relative_path: str) -> Path:
    """Prevent path-traversal attacks (../../etc/passwd style)."""
    root = dataset_root().resolve()
    target = (root / relative_path).resolve()
    if root not in target.parents and target != root:
        raise ValueError("Invalid path.")
    return target


def save_images(class_name: str, files) -> int:
    """Persist uploaded images into ml/dataset/<class_name>/."""
    folder = _resolve_inside_root(class_name)
    if not folder.is_dir():
        raise ValueError(f"Unknown activity class '{class_name}'.")

    saved = 0
    for index, uploaded in enumerate(files):
        ext = Path(uploaded.name).suffix.lower()
        if ext not in IMAGE_EXTS:
            continue
        if uploaded.size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            continue

        stem = SAFE_NAME.sub("_", Path(uploaded.name).stem)[:40] or "image"
        filename = f"{stem}_{index}{ext}"
        target = folder / filename
        counter = 1
        while target.exists():
            target = folder / f"{stem}_{index}_{counter}{ext}"
            counter += 1

        with open(target, "wb") as fh:
            for chunk in uploaded.chunks():
                fh.write(chunk)
        saved += 1
    return saved


def list_images(class_name: str, limit: int = 200) -> List[Dict]:
    folder = _resolve_inside_root(class_name)
    if not folder.is_dir():
        return []
    items = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [
        {
            "name": p.name,
            "relative_path": str(p.relative_to(dataset_root())).replace("\\", "/"),
            "size_kb": round(p.stat().st_size / 1024, 1),
            "url": f"/api/dataset/image/?path={p.relative_to(dataset_root()).as_posix()}",
        }
        for p in items[:limit]
    ]


def delete_image(relative_path: str) -> None:
    target = _resolve_inside_root(relative_path)
    if target.suffix.lower() not in IMAGE_EXTS:
        raise ValueError("Not an image file.")
    if not target.exists():
        raise ValueError("Image not found.")
    target.unlink()


def delete_class(class_name: str) -> None:
    folder = _resolve_inside_root(class_name)
    if not folder.is_dir():
        raise ValueError("Class not found.")
    if count_images(folder) > 0:
        raise ValueError("Class folder is not empty - delete the images first.")
    folder.rmdir()