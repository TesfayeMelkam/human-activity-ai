from rest_framework import serializers

from .models import ActivityClass, Prediction, TrainingRun


class ActivityClassSerializer(serializers.ModelSerializer):
    class Meta:
        model = ActivityClass
        fields = ["id", "name", "slug", "description", "is_active", "created_at"]


class PredictionSerializer(serializers.ModelSerializer):
    confidence_percent = serializers.SerializerMethodField()
    image_url = serializers.SerializerMethodField()

    class Meta:
        model = Prediction
        fields = [
            "id",
            "activity",
            "confidence",
            "confidence_percent",
            "processing_time",
            "source",
            "model_name",
            "is_uncertain",
            "image_url",
            "created_at",
        ]

    def get_confidence_percent(self, obj) -> float:
        return round(obj.confidence * 100, 2)

    def get_image_url(self, obj) -> str | None:
        if not obj.image:
            return None
        request = self.context.get("request")
        url = obj.image.url
        return request.build_absolute_uri(url) if request else url


class TrainingRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingRun
        fields = [
            "id",
            "model_type",
            "status",
            "epochs",
            "accuracy",
            "loss",
            "notes",
            "started_at",
            "finished_at",
        ]