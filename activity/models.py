from django.db import models
from django.utils.text import slugify


class ActivityClass(models.Model):
    """
    A human activity the AI can recognise, e.g. 'walking'.

    Keeping this in the database lets an administrator add new activity
    classes without touching code.
    """

    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=120, unique=True, blank=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]
        verbose_name_plural = "Activity classes"

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return self.name


class Prediction(models.Model):
    """One single inference result, stored so we can build history + charts."""

    SOURCE_CHOICES = [
        ("camera", "Live camera"),
        ("upload", "Uploaded image"),
        ("api", "External API"),
    ]

    activity = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)                 # 0.0 .. 1.0
    processing_time = models.FloatField(default=0.0)            # milliseconds
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default="camera")
    model_name = models.CharField(max_length=80, blank=True, default="")
    is_uncertain = models.BooleanField(default=False)
    image = models.ImageField(upload_to="uploads/%Y/%m/%d/", blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.activity} ({self.confidence:.0%})"

    @property
    def confidence_percent(self) -> float:
        return round(self.confidence * 100, 2)


class TrainingRun(models.Model):
    """Bookkeeping for a model-training job started from the web UI."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    MODEL_CHOICES = [
        ("cnn", "Custom CNN"),
        ("mobilenetv2", "MobileNetV2"),
        ("efficientnet", "EfficientNetB0"),
    ]

    model_type = models.CharField(max_length=30, choices=MODEL_CHOICES, default="mobilenetv2")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    epochs = models.PositiveIntegerField(default=10)
    accuracy = models.FloatField(null=True, blank=True)
    loss = models.FloatField(null=True, blank=True)
    notes = models.TextField(blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return f"{self.get_model_type_display()} - {self.status}"