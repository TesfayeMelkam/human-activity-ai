"""Page views + REST API endpoints."""
from __future__ import annotations

import threading
from datetime import timedelta

from django.conf import settings
from django.contrib import messages
from django.db.models import Avg, Count
from django.db.models.functions import TruncDate
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework.response import Response

from .forms import ImageUploadForm
from .models import ActivityClass, Prediction, TrainingRun
from .serializers import (
    ActivityClassSerializer,
    PredictionSerializer,
    TrainingRunSerializer,
)
from .services import dataset_service
from .services.audio_service import phrase_for
from .services.camera_service import data_url_to_rgb
from .services.model_service import ModelService
from .services.prediction_service import predict_array, predict_uploaded_file

# ===========================================================================
# HTML pages
# ===========================================================================


def home(request):
    context = {
        "activities": settings.DEFAULT_ACTIVITIES,
        "total_predictions": Prediction.objects.count(),
        "model_ready": ModelService.instance().load(),
    }
    return render(request, "home.html", context)


@ensure_csrf_cookie
def camera(request):
    """Live webcam page. ensure_csrf_cookie guarantees the JS can POST."""
    return render(
        request,
        "camera.html",
        {
            "activities": settings.DEFAULT_ACTIVITIES,
            "threshold": settings.ML_CONFIDENCE_THRESHOLD,
        },
    )


def upload(request):
    form = ImageUploadForm()
    return render(request, "upload.html", {"form": form})


def dashboard(request):
    return render(request, "dashboard.html")


def history(request):
    return render(request, "history.html")


def dataset(request):
    dataset_service.ensure_default_classes()
    classes = dataset_service.list_classes()
    return render(
        request,
        "dataset.html",
        {
            "classes": classes,
            "total_images": sum(c["count"] for c in classes),
            "runs": TrainingRun.objects.all()[:10],
            "is_staff": request.user.is_staff,
        },
    )


def about(request):
    return render(request, "about.html")


# ===========================================================================
# REST API
# ===========================================================================


@api_view(["POST"])
@permission_classes([AllowAny])
def api_predict(request):
    """
    POST /api/predict/
    Body (JSON)   : {"image": "data:image/jpeg;base64,...", "source": "camera"}
    Body (multipart): image=<file>
    """
    image_payload = request.data.get("image")

    # --- Multipart upload path -------------------------------------------
    if not image_payload and "image" in request.FILES:
        return _handle_upload(request, source="upload")

    if not image_payload:
        return Response(
            {"error": "No image provided. Send 'image' as a base64 data URL."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        rgb = data_url_to_rgb(image_payload)
    except ValueError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    try:
        result = predict_array(rgb, source=request.data.get("source", "camera"))
    except Exception as exc:  # pragma: no cover
        return Response(
            {"error": f"Prediction failed: {exc}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    result["speech"] = phrase_for(result["activity"], result["confidence"])
    return Response(result, status=status.HTTP_200_OK)


def _handle_upload(request, source: str = "upload") -> Response:
    form = ImageUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return Response(
            {"error": "Invalid upload.", "details": form.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )
    try:
        result = predict_uploaded_file(form.cleaned_data["image"], source=source)
    except ValueError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as exc:  # pragma: no cover
        return Response(
            {"error": f"Prediction failed: {exc}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    result["speech"] = phrase_for(result["activity"], result["confidence"])
    return Response(result, status=status.HTTP_200_OK)


@api_view(["POST"])
@permission_classes([AllowAny])
def api_upload(request):
    """POST /api/upload/  (multipart form-data, field name: image)"""
    return _handle_upload(request, source="upload")


@api_view(["GET", "DELETE"])
@permission_classes([AllowAny])
def api_history(request):
    """
    GET    /api/history/?page=1&page_size=10&activity=walking&search=run&source=camera
    DELETE /api/history/?id=12       -> delete one row
    DELETE /api/history/?all=true    -> delete everything (staff only)
    """
    if request.method == "DELETE":
        if request.data.get("all") or request.query_params.get("all"):
            if not request.user.is_staff:
                return Response(
                    {"error": "Only staff members can clear the whole history."},
                    status=status.HTTP_403_FORBIDDEN,
                )
            deleted, _ = Prediction.objects.all().delete()
            return Response({"deleted": deleted})

        pk = request.data.get("id") or request.query_params.get("id")
        if not pk:
            return Response({"error": "Provide 'id' or 'all=true'."}, status=400)
        record = get_object_or_404(Prediction, pk=pk)
        record.delete()
        return Response({"deleted": 1})

    queryset = Prediction.objects.all()

    activity = request.query_params.get("activity")
    if activity:
        queryset = queryset.filter(activity=activity)

    source = request.query_params.get("source")
    if source:
        queryset = queryset.filter(source=source)

    search = request.query_params.get("search")
    if search:
        queryset = queryset.filter(activity__icontains=search)

    try:
        page = max(int(request.query_params.get("page", 1)), 1)
        page_size = min(max(int(request.query_params.get("page_size", 10)), 1), 100)
    except ValueError:
        page, page_size = 1, 10

    total = queryset.count()
    start = (page - 1) * page_size
    rows = queryset[start:start + page_size]

    serializer = PredictionSerializer(rows, many=True, context={"request": request})
    return Response(
        {
            "count": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max((total + page_size - 1) // page_size, 1),
            "results": serializer.data,
        }
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def api_classes(request):
    """GET /api/classes/ - activity classes from the DB plus dataset counts."""
    db_classes = list(ActivityClass.objects.filter(is_active=True))
    if not db_classes:
        # Bootstrap from the filesystem so the endpoint is never empty
        payload = [
            {"id": i, "name": c["name"], "slug": c["slug"], "description": "",
             "is_active": True, "created_at": None, "image_count": c["count"]}
            for i, c in enumerate(dataset_service.list_classes(), start=1)
        ]
        return Response(payload)

    counts = {c["slug"]: c["count"] for c in dataset_service.list_classes()}
    data = ActivityClassSerializer(db_classes, many=True).data
    for row in data:
        row["image_count"] = counts.get(row["slug"], 0)
    return Response(data)


@api_view(["GET"])
@permission_classes([AllowAny])
def api_model_status(request):
    """GET /api/model-status/"""
    service = ModelService.instance()
    service.load()
    payload = service.status()
    payload["last_training_run"] = TrainingRunSerializer(
        TrainingRun.objects.first()
    ).data if TrainingRun.objects.exists() else None
    payload["total_predictions"] = Prediction.objects.count()
    return Response(payload)


@api_view(["GET"])
@permission_classes([AllowAny])
def api_stats(request):
    """Aggregated numbers + chart series for the dashboard."""
    days = int(request.query_params.get("days", 14))
    since = timezone.now() - timedelta(days=days)

    queryset = Prediction.objects.all()
    recent = queryset.filter(created_at__gte=since)

    total = queryset.count()
    avg_confidence = queryset.aggregate(v=Avg("confidence"))["v"] or 0.0
    avg_time = queryset.aggregate(v=Avg("processing_time"))["v"] or 0.0

    by_activity = list(
        queryset.values("activity").annotate(count=Count("id")).order_by("-count")
    )
    most_detected = by_activity[0] if by_activity else None

    over_time = list(
        recent.annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(count=Count("id"), avg_confidence=Avg("confidence"))
        .order_by("day")
    )

    return Response(
        {
            "total_predictions": total,
            "avg_confidence": round(avg_confidence, 4),
            "avg_processing_time": round(avg_time, 2),
            "most_detected": most_detected,
            "by_activity": by_activity,
            "over_time": [
                {
                    "date": row["day"].isoformat() if row["day"] else None,
                    "count": row["count"],
                    "avg_confidence": round(row["avg_confidence"] or 0, 4),
                }
                for row in over_time
            ],
            "sources": list(
                queryset.values("source").annotate(count=Count("id")).order_by("-count")
            ),
        }
    )


# ===========================================================================
# Dataset management API
# ===========================================================================


@api_view(["GET", "POST", "DELETE"])
@permission_classes([AllowAny])
def api_dataset(request):
    """
    GET    /api/dataset/                 -> list classes with counts
    GET    /api/dataset/?class=walking   -> list images of one class
    POST   /api/dataset/                 -> multipart: class_name + images[]
    DELETE /api/dataset/?path=walking/x.jpg  (staff only)
    """
    if request.method == "GET":
        class_name = request.query_params.get("class")
        if class_name:
            try:
                return Response(
                    {
                        "class": class_name,
                        "images": dataset_service.list_images(class_name),
                    }
                )
            except ValueError as exc:
                return Response({"error": str(exc)}, status=400)
        return Response(
            {
                "classes": dataset_service.list_classes(),
                "total": dataset_service.total_images(),
                "root": str(settings.ML_DATASET_DIR),
            }
        )

    if request.method == "DELETE":
        if not request.user.is_staff:
            return Response(
                {"error": "Staff privileges required."}, status=status.HTTP_403_FORBIDDEN
            )
        relative_path = request.data.get("path") or request.query_params.get("path")
        if not relative_path:
            return Response({"error": "Provide 'path'."}, status=400)
        try:
            dataset_service.delete_image(relative_path)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=400)
        return Response({"deleted": relative_path})

    # POST ---------------------------------------------------------------
    if not request.user.is_staff:
        return Response(
            {"error": "Staff privileges required to upload training images."},
            status=status.HTTP_403_FORBIDDEN,
        )

    class_name = request.data.get("class_name") or request.data.get("activity")
    if not class_name:
        return Response({"error": "Provide 'class_name'."}, status=400)

    files = request.FILES.getlist("images") or request.FILES.getlist("image")
    if not files:
        return Response({"error": "No images were uploaded."}, status=400)

    try:
        saved = dataset_service.save_images(class_name, files)
    except ValueError as exc:
        return Response({"error": str(exc)}, status=400)

    return Response(
        {"saved": saved, "class": class_name, "classes": dataset_service.list_classes()}
    )


@api_view(["POST"])
@permission_classes([AllowAny])
def api_dataset_create_class(request):
    if not request.user.is_staff:
        return Response({"error": "Staff privileges required."}, status=403)
    name = request.data.get("name", "").strip()
    if not name:
        return Response({"error": "Provide a class 'name'."}, status=400)
    try:
        dataset_service.create_class(name)
    except ValueError as exc:
        return Response({"error": str(exc)}, status=400)
    return Response({"created": name, "classes": dataset_service.list_classes()}, status=201)


def api_dataset_image(request):
    """Stream a dataset image (used for the management-grid thumbnails)."""
    relative = request.GET.get("path", "")
    try:
        target = dataset_service._resolve_inside_root(relative)
    except ValueError:
        raise Http404("Invalid path")
    if not target.is_file():
        raise Http404("Image not found")
    return FileResponse(open(target, "rb"))


# ===========================================================================
# Training API
# ===========================================================================


@api_view(["GET", "POST"])
@permission_classes([AllowAny])
def api_train(request):
    """
    GET  /api/train/  -> status of training runs
    POST /api/train/  -> start a background training job (staff only)
    """
    if request.method == "GET":
        runs = TrainingRun.objects.all()[:20]
        return Response({"runs": TrainingRunSerializer(runs, many=True).data})

    if not request.user.is_staff:
        return Response(
            {"error": "Staff privileges required to start training."},
            status=status.HTTP_403_FORBIDDEN,
        )

    model_type = request.data.get("model_type", "mobilenetv2")
    if model_type not in dict(TrainingRun.MODEL_CHOICES):
        return Response({"error": f"Unknown model_type '{model_type}'."}, status=400)

    try:
        epochs = min(max(int(request.data.get("epochs", 10)), 1), 100)
    except (TypeError, ValueError):
        epochs = 10

    run = TrainingRun.objects.create(model_type=model_type, epochs=epochs, status="pending")

    thread = threading.Thread(target=_run_training_job, args=(run.id,), daemon=True)
    thread.start()

    return Response(
        {"message": "Training started in the background.", "run": TrainingRunSerializer(run).data},
        status=status.HTTP_202_ACCEPTED,
    )


def _run_training_job(run_id: int) -> None:
    """Executed in a daemon thread so the HTTP request returns immediately."""
    import subprocess
    import sys
    from pathlib import Path

    run = TrainingRun.objects.filter(pk=run_id).first()
    if not run:
        return

    run.status = "running"
    run.save(update_fields=["status"])

    script = Path(settings.BASE_DIR) / "ml" / "training" / "train.py"
    command = [
        sys.executable,
        str(script),
        "--model", run.model_type,
        "--epochs", str(run.epochs),
    ]

    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=60 * 60
        )
        run.notes = (completed.stdout or "")[-4000:]
        if completed.returncode == 0:
            run.status = "completed"
            # Reload the freshly trained model
            ModelService.instance().load(force=True)
        else:
            run.status = "failed"
            run.notes += "\n\nSTDERR:\n" + (completed.stderr or "")[-2000:]
    except Exception as exc:  # pragma: no cover
        run.status = "failed"
        run.notes = f"{exc}"
    finally:
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "notes", "finished_at"])


# ===========================================================================
# Friendly error handlers
# ===========================================================================


def handler404(request, exception):
    if request.path.startswith("/api/"):
        return JsonResponse({"error": "Not found."}, status=404)
    return render(request, "base.html", {"error_message": "Page not found."}, status=404)


def handler500(request):
    if request.path.startswith("/api/"):
        return JsonResponse({"error": "Internal server error."}, status=500)
    return render(
        request, "base.html", {"error_message": "Something went wrong."}, status=500
    )