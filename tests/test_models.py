from django.test import TestCase

from activity.models import ActivityClass, Prediction


class ActivityClassTests(TestCase):
    def test_slug_is_generated(self):
        obj = ActivityClass.objects.create(name="Raising Hands")
        self.assertEqual(obj.slug, "raising-hands")

    def test_str(self):
        obj = ActivityClass.objects.create(name="Walking")
        self.assertEqual(str(obj), "Walking")


class PredictionTests(TestCase):
    def test_confidence_percent(self):
        p = Prediction.objects.create(activity="walking", confidence=0.9412)
        self.assertEqual(p.confidence_percent, 94.12)

    def test_default_ordering_is_newest_first(self):
        Prediction.objects.create(activity="sitting", confidence=0.5)
        Prediction.objects.create(activity="running", confidence=0.9)
        self.assertEqual(Prediction.objects.first().activity, "running")