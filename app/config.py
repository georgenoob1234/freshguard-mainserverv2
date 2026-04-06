from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def max_fruit_imgsz_as_capture_resolution(primary: int | str, fallback: int | str) -> str:
    """Larger WxH for camera /capture; supports int (square N) or 'WxH' / 'W' strings from env."""

    def to_wh(value: int | str) -> tuple[int, int]:
        if isinstance(value, int):
            if value < 1:
                raise ValueError(f"imgsz must be >= 1, got {value}")
            return value, value
        s = str(value).strip().lower()
        parts = s.split("x", 1)
        if len(parts) == 2:
            return int(parts[0].strip()), int(parts[1].strip())
        n = int(s)
        if n < 1:
            raise ValueError(f"imgsz must be >= 1, got {value!r}")
        return n, n

    pw, ph = to_wh(primary)
    fw, fh = to_wh(fallback)
    if pw * ph >= fw * fh:
        return f"{pw}x{ph}"
    return f"{fw}x{fh}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    OPERATING_MODE: Literal["scale", "shelf"] = "scale"

    SERVICE_HOST: str = "0.0.0.0"
    SERVICE_PORT: int = 8000

    WEIGHT_PUSH_PATH: str = "/weight"
    ENABLE_WEIGHT_POLLING: bool = False
    SHELF_SCAN_INTERVAL_SECONDS: float = Field(default=5.0, gt=0)
    SHELF_SOURCE_ID: str | None = None
    # Compatibility payload value when shelf mode has no real scale weight.
    SHELF_PUBLISH_WEIGHT_GRAMS: float = Field(default=0.0, ge=0)

    # Locked weight semantics:
    # - IDLE -> ACTIVE when grams >= ENTER_ACTIVE_WEIGHT
    # - ACTIVE -> IDLE when grams < EXIT_ACTIVE_WEIGHT
    # Hysteresis prevents rapid bouncing around one threshold.
    ENTER_ACTIVE_WEIGHT: float = 30.0
    EXIT_ACTIVE_WEIGHT: float = 25.0

    # Backward-compatible alias still used by non-state-machine components.
    MIN_FRUIT_WEIGHT: float = 30.0
    SIGNIFICANT_DELTA: float = 0.0

    CAMERA_SERVICE_URL: str = "http://localhost:8200"
    FRUIT_DETECTOR_URL: str = "http://localhost:8300"
    FRUIT_DETECT_PATH: str = "/detect-fruits"
    DEFECT_DETECTOR_URL: str = "http://localhost:8400"
    UI_SERVICE_URL: str = "http://localhost:8500"
    MAIN_SERVER_URL: str = "http://localhost:8600"

    UI_PUBLISH_PATH: str = "/update"
    MAIN_SERVER_PUBLISH_PATH: str = "/update"

    HTTP_TIMEOUT_SECONDS: float = 8.0
    HTTP_RETRIES: int = 1

    ALLOWED_FRUIT_CLASSES: list[str] = Field(
        default_factory=lambda: ["apple", "banana", "tomato", "lemon", "cucumber"]
    )
    CLASS_CONFIDENCE_THRESHOLDS: dict[str, float] = Field(
        default_factory=lambda: {
            # Contract values from UpdateFruitDetectionPrompt.txt
            "apple": 0.40,
            "banana": 0.50,
            "tomato": 0.45,
            "lemon": 0.35,
            "cucumber": 0.40
        }
    )
    DEFAULT_CLASS_CONFIDENCE_THRESHOLD: float = 0.50

    FRUIT_PRIMARY_IMGSZ: int = 320
    FRUIT_FALLBACK_IMGSZ: int = 416
    FRUIT_LOW_CONFIDENCE_FALLBACK_THRESHOLD: float = 0.30
    FRUIT_TINY_BBOX_AREA_RATIO: float = 0.005

    CAMERA_USE_EXTRA_DEFAULT: bool = True
    CAMERA_CAPTURE_FORMAT: Literal["jpeg", "png"] = "jpeg"
    CAMERA_CAPTURE_QUALITY: int = Field(default=95, ge=1, le=100)
    AGGREGATION_POLICY: Literal["vote", "average", "best_frame_plus_vote"] = "average"

    DEFECT_MAX_PARALLEL: int = 6

    ENABLE_DUPLICATE_SUPPRESSION: bool = True
    DUPLICATE_SUPPRESSION_WINDOW_MS: int = 3000
    DUPLICATE_WEIGHT_BUCKET_GRAMS: float = 5.0

    JOURNAL_PATH: Path = Path("data/journal/events.jsonl")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
