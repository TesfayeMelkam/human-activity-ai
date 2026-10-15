from django.contrib import admin

from .models import ActivityClass, Prediction, TrainingRun


@admin.register(ActivityClass)
class ActivityClassAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "is_active", "created_at")
    list_filter = ("is_active",)
    search_fields = ("name",)
    prepopulated_fields = {"slug": ("name",)}


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("activity", "confidence", "processing_time", "source", "created_at")
    list_filter = ("activity", "source", "is_uncertain")
    search_fields = ("activity",)
    date_hierarchy = "created_at"


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ("model_type", "status", "epochs", "accuracy", "started_at", "finished_at")
    list_filter = ("model_type", "status")