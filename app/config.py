from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    OPERATING_MODE: Literal["scale", "shelf"] = "shelf"

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
        default_factory=lambda: ["apple", "banana", "tomato"]
    )
    CLASS_CONFIDENCE_THRESHOLDS: dict[str, float] = Field(
        default_factory=lambda: {
            # Contract values from UpdateFruitDetectionPrompt.txt
            "apple": 0.40,
            "banana": 0.50,
            "tomato": 0.45,
        }
    )
    DEFAULT_CLASS_CONFIDENCE_THRESHOLD: float = 0.50

    FRUIT_PRIMARY_IMGSZ: int = 320
    FRUIT_FALLBACK_IMGSZ: int = 416
    FRUIT_LOW_CONFIDENCE_FALLBACK_THRESHOLD: float = 0.30
    FRUIT_TINY_BBOX_AREA_RATIO: float = 0.005

    CAMERA_USE_EXTRA_DEFAULT: bool = True
    AGGREGATION_POLICY: Literal["vote", "average", "best_frame_plus_vote"] = "average"

    DEFECT_MAX_PARALLEL: int = 6

    ENABLE_DUPLICATE_SUPPRESSION: bool = True
    DUPLICATE_SUPPRESSION_WINDOW_MS: int = 3000
    DUPLICATE_WEIGHT_BUCKET_GRAMS: float = 5.0

    JOURNAL_PATH: Path = Path("data/journal/events.jsonl")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
