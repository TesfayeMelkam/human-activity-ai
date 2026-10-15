from django.urls import path

from . import views

app_name = "activity"

urlpatterns = [
    # Pages
    path("", views.home, name="home"),
    path("camera/", views.camera, name="camera"),
    path("upload/", views.upload, name="upload"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("history/", views.history, name="history"),
    path("dataset/", views.dataset, name="dataset"),
    path("about/", views.about, name="about"),

    # REST API
    path("api/predict/", views.api_predict, name="api_predict"),
    path("api/upload/", views.api_upload, name="api_upload"),
    path("api/history/", views.api_history, name="api_history"),
    path("api/classes/", views.api_classes, name="api_classes"),
    path("api/model-status/", views.api_model_status, name="api_model_status"),
    path("api/stats/", views.api_stats, name="api_stats"),

    # Dataset management
    path("api/dataset/", views.api_dataset, name="api_dataset"),
    path("api/dataset/create-class/", views.api_dataset_create_class, name="api_dataset_create_class"),
    path("api/dataset/image/", views.api_dataset_image, name="api_dataset_image"),

    # Training
    path("api/train/", views.api_train, name="api_train"),
]