from pathlib import Path

from django import forms
from django.conf import settings


class ImageUploadForm(forms.Form):
    """Validates that an uploaded file really is a small, allowed image."""

    image = forms.ImageField(
        help_text="JPG, JPEG, PNG, BMP or WEBP. Maximum size is set in settings."
    )

    def clean_image(self):
        image = self.cleaned_data["image"]

        # 1. Extension whitelist
        ext = Path(image.name).suffix.lower()
        if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
            raise forms.ValidationError(
                f"Unsupported file type '{ext}'. Allowed: "
                f"{', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}"
            )

        # 2. Size limit
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if image.size > max_bytes:
            raise forms.ValidationError(
                f"File too large ({image.size / 1024 / 1024:.1f} MB). "
                f"Maximum is {settings.MAX_UPLOAD_SIZE_MB} MB."
            )

        # 3. Content type sanity check
        if image.content_type and not image.content_type.startswith("image/"):
            raise forms.ValidationError("The uploaded file is not an image.")

        return image