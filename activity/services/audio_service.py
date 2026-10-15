"""Builds the short sentences the browser speaks with speechSynthesis."""
from __future__ import annotations

PHRASES = {
    "standing": "Standing detected.",
    "sitting": "Sitting detected.",
    "walking": "Walking detected.",
    "running": "Running detected.",
    "jumping": "Jumping detected.",
    "lying": "Lying down detected.",
    "raising_hands": "Raising hands detected.",
    "uncertain": "Not sure what the activity is.",
}

# Activities whose names need a friendlier spoken form
SPOKEN_NAMES = {
    "lying": "lying down",
    "raising_hands": "raising hands",
}


def phrase_for(activity: str, confidence: float | None = None) -> str:
    """Return a natural sentence for the given activity label."""
    key = (activity or "").strip().lower()
    if key in PHRASES:
        return PHRASES[key]

    spoken = SPOKEN_NAMES.get(key, key.replace("_", " "))
    if not spoken:
        return "Activity not recognised."
    return f"{spoken.capitalize()} detected."