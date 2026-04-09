from __future__ import annotations

from io import BytesIO

from PIL import Image

from app.core.image_ops import crop_to_jpeg_bytes, defect_crop_to_jpeg_bytes, defect_padded_crop_rect
from app.models import BBox


def _dummy_image_bytes(*, width: int = 100, height: int = 80) -> bytes:
    image = Image.new("RGB", (width, height), color=(100, 140, 180))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _jpeg_size(image_bytes: bytes) -> tuple[int, int]:
    image = Image.open(BytesIO(image_bytes))
    return image.size


def test_defect_padded_crop_rect_expands_center_bbox() -> None:
    bbox = BBox(x_min=20.0, y_min=10.0, x_max=60.0, y_max=50.0)
    crop_rect = defect_padded_crop_rect(
        100,
        80,
        bbox,
        padding_x_ratio=0.15,
        padding_y_ratio=0.15,
    )
    assert crop_rect == (14, 4, 66, 56)


def test_defect_padded_crop_rect_clamps_left_top() -> None:
    bbox = BBox(x_min=2.0, y_min=1.0, x_max=12.0, y_max=11.0)
    crop_rect = defect_padded_crop_rect(
        100,
        80,
        bbox,
        padding_x_ratio=0.20,
        padding_y_ratio=0.20,
    )
    assert crop_rect == (0, 0, 14, 13)


def test_defect_padded_crop_rect_clamps_right_bottom() -> None:
    bbox = BBox(x_min=90.0, y_min=70.0, x_max=99.0, y_max=79.0)
    crop_rect = defect_padded_crop_rect(
        100,
        80,
        bbox,
        padding_x_ratio=0.20,
        padding_y_ratio=0.20,
    )
    assert crop_rect == (88, 68, 100, 80)


def test_defect_crop_preparation_does_not_mutate_original_bbox() -> None:
    image_bytes = _dummy_image_bytes()
    bbox = BBox(x_min=20.0, y_min=10.0, x_max=60.0, y_max=50.0)
    original = bbox.to_xyxy()

    defect_crop_to_jpeg_bytes(
        image_bytes,
        bbox,
        padding_x_ratio=0.15,
        padding_y_ratio=0.15,
    )

    assert bbox.to_xyxy() == original


def test_defect_crop_zero_padding_matches_legacy_crop_dimensions() -> None:
    image_bytes = _dummy_image_bytes()
    bbox = BBox(x_min=10.2, y_min=10.7, x_max=31.8, y_max=39.9)

    legacy_crop_bytes = crop_to_jpeg_bytes(image_bytes, bbox)
    defect_crop_bytes, _ = defect_crop_to_jpeg_bytes(
        image_bytes,
        bbox,
        padding_x_ratio=0.0,
        padding_y_ratio=0.0,
    )

    assert _jpeg_size(defect_crop_bytes) == _jpeg_size(legacy_crop_bytes)
