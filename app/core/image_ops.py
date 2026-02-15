from __future__ import annotations

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


def crop_to_jpeg_bytes(image_bytes: bytes, bbox: BBox, *, jpeg_quality: int = 95) -> bytes:
    image = decode_image(image_bytes)
    width, height = image.size

    x1 = max(0, min(width - 1, int(bbox.x_min)))
    y1 = max(0, min(height - 1, int(bbox.y_min)))
    x2 = max(x1 + 1, min(width, int(bbox.x_max)))
    y2 = max(y1 + 1, min(height, int(bbox.y_max)))

    cropped = image.crop((x1, y1, x2, y2))
    buffer = BytesIO()
    cropped.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()
