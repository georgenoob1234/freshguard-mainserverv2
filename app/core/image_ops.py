from __future__ import annotations

import math
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.models import BBox


class ImageDecodeError(RuntimeError):
    pass


def decode_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError("Failed to decode image bytes") from exc


def _bbox_to_crop_rect(image_width: int, image_height: int, bbox: BBox) -> tuple[int, int, int, int]:
    x1 = max(0, min(image_width - 1, int(bbox.x_min)))
    y1 = max(0, min(image_height - 1, int(bbox.y_min)))
    x2 = max(x1 + 1, min(image_width, int(bbox.x_max)))
    y2 = max(y1 + 1, min(image_height, int(bbox.y_max)))
    return x1, y1, x2, y2


def defect_padded_crop_rect(
    image_width: int,
    image_height: int,
    fruit_bbox: BBox,
    *,
    padding_x_ratio: float,
    padding_y_ratio: float,
) -> tuple[int, int, int, int]:
    if padding_x_ratio == 0.0 and padding_y_ratio == 0.0:
        return _bbox_to_crop_rect(image_width, image_height, fruit_bbox)

    width = fruit_bbox.x_max - fruit_bbox.x_min
    height = fruit_bbox.y_max - fruit_bbox.y_min

    pad_x = width * padding_x_ratio
    pad_y = height * padding_y_ratio

    x1 = max(0, math.floor(fruit_bbox.x_min - pad_x))
    y1 = max(0, math.floor(fruit_bbox.y_min - pad_y))
    x2 = min(image_width, math.ceil(fruit_bbox.x_max + pad_x))
    y2 = min(image_height, math.ceil(fruit_bbox.y_max + pad_y))

    x1 = min(image_width - 1, x1)
    y1 = min(image_height - 1, y1)
    x2 = max(x1 + 1, min(image_width, x2))
    y2 = max(y1 + 1, min(image_height, y2))
    return x1, y1, x2, y2


def _crop_jpeg_bytes_from_rect(
    image: Image.Image,
    crop_rect: tuple[int, int, int, int],
    *,
    jpeg_quality: int,
) -> bytes:
    cropped = image.crop(crop_rect)
    buffer = BytesIO()
    cropped.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()


def crop_to_jpeg_bytes(image_bytes: bytes, bbox: BBox, *, jpeg_quality: int = 95) -> bytes:
    image = decode_image(image_bytes)
    width, height = image.size

    crop_rect = _bbox_to_crop_rect(width, height, bbox)
    return _crop_jpeg_bytes_from_rect(image, crop_rect, jpeg_quality=jpeg_quality)


def defect_crop_to_jpeg_bytes(
    image_bytes: bytes,
    fruit_bbox: BBox,
    *,
    padding_x_ratio: float,
    padding_y_ratio: float,
    jpeg_quality: int = 95,
) -> tuple[bytes, tuple[int, int, int, int]]:
    image = decode_image(image_bytes)
    width, height = image.size
    crop_rect = defect_padded_crop_rect(
        width,
        height,
        fruit_bbox,
        padding_x_ratio=padding_x_ratio,
        padding_y_ratio=padding_y_ratio,
    )
    crop_bytes = _crop_jpeg_bytes_from_rect(image, crop_rect, jpeg_quality=jpeg_quality)
    return crop_bytes, crop_rect
