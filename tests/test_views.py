import base64
import io
import json

from django.test import TestCase
from django.urls import reverse
from PIL import Image

from activity.models import Prediction


def make_data_url(width=320, height=240, color=(120, 160, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"


class PageTests(TestCase):
    def test_all_pages_render(self):
        for name in ("home", "camera", "upload", "dashboard", "history", "dataset", "about"):
            with self.subTest(page=name):
                response = self.client.get(reverse(f"activity:{name}"))
                self.assertEqual(response.status_code, 200)


class ApiTests(TestCase):
    def test_predict_endpoint(self):
        response = self.client.post(
            reverse("activity:api_predict"),
            data=json.dumps({"image": make_data_url(), "source": "api"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("activity", payload)
        self.assertIn("confidence", payload)

    def test_predict_without_image_returns_400(self):
        response = self.client.post(
            reverse("activity:api_predict"),
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_with_garbage_returns_400(self):
        response = self.client.post(
            reverse("activity:api_predict"),
            data=json.dumps({"image": "not-a-data-url"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_history_endpoint(self):
        Prediction.objects.create(activity="walking", confidence=0.9, processing_time=40)
        response = self.client.get(reverse("activity:api_history"))
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(response.json()["count"], 1)

    def test_stats_endpoint(self):
        response = self.client.get(reverse("activity:api_stats"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("total_predictions", response.json())

    def test_classes_endpoint(self):
        response = self.client.get(reverse("activity:api_classes"))
        self.assertEqual(response.status_code, 200)

    def test_model_status_endpoint(self):
        response = self.client.get(reverse("activity:api_model_status"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("classes", response.json())

    def test_dataset_upload_requires_staff(self):
        response = self.client.post(reverse("activity:api_dataset"), data={})
        self.assertEqual(response.status_code, 403)