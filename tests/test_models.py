from __future__ import annotations

from app.models import FruitDetectionResponse


def test_fruit_detection_response_accepts_fruits_key_alias() -> None:
    payload = {
        "image_id": "img-1",
        "width": 320,
        "height": 320,
        "fruits": [
            {
                "fruit_id": "f-1",
                "class": "apple",
                "confidence": 0.91,
                "bbox": [10, 20, 100, 200],
            }
        ],
    }

    parsed = FruitDetectionResponse.model_validate(payload)
    assert len(parsed.detections) == 1
    assert parsed.detections[0].fruit_class == "apple"
