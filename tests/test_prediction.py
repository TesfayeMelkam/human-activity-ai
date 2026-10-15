"""Tests for the inference service (works with or without a trained model)."""
import numpy as np
from django.test import TestCase

from activity.models import Prediction
from activity.services.model_service import ModelService
from activity.services.prediction_service import predict_array


class PredictionServiceTests(TestCase):
    def setUp(self):
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.image[100:400, 250:380] = 255  # a tall bright rectangle

    def test_predict_returns_expected_keys(self):
        result = predict_array(self.image, source="api", save=False)
        for key in ("activity", "confidence", "processing_time_ms", "uncertain"):
            self.assertIn(key, result)

    def test_confidence_is_a_probability(self):
        result = predict_array(self.image, source="api", save=False)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_prediction_is_saved(self):
        before = Prediction.objects.count()
        predict_array(self.image, source="api", save=True)
        self.assertEqual(Prediction.objects.count(), before + 1)

    def test_low_confidence_becomes_uncertain(self):
        result = predict_array(np.zeros((224, 224, 3), dtype=np.uint8), source="api", save=False)
        if result["confidence"] < result["threshold"]:
            self.assertEqual(result["activity"], "uncertain")

    def test_model_status_shape(self):
        status = ModelService.instance().status()
        self.assertIn("loaded", status)
        self.assertIn("classes", status)
        self.assertIsInstance(status["classes"], list)