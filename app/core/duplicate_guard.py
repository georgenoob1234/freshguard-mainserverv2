from __future__ import annotations

import hashlib
import json
from datetime import datetime

from app.models import ScanResult


class DuplicateResultGuard:
    """Suppress repeated equivalent scan publications in a short time window."""

    def __init__(self, *, window_ms: int, weight_bucket_grams: float) -> None:
        self._window_ms = window_ms
        self._weight_bucket_grams = weight_bucket_grams
        self._last_hash: str | None = None
        self._last_ts: datetime | None = None

    def should_suppress(self, result: ScanResult, now: datetime) -> tuple[bool, str]:
        result_hash = self.compute_result_hash(result)
        if self._last_hash is not None and self._last_ts is not None:
            elapsed_ms = (now - self._last_ts).total_seconds() * 1000
            if result_hash == self._last_hash and elapsed_ms <= self._window_ms:
                return True, result_hash

        self._last_hash = result_hash
        self._last_ts = now
        return False, result_hash

    def compute_result_hash(self, result: ScanResult) -> str:
        weight_bucket = self._bucket_weight(result.weight_grams)
        normalized_fruits = []
        for fruit in result.fruits:
            normalized_fruits.append(
                {
                    "fruit_class": fruit.fruit_class,
                    "confidence": round(fruit.confidence, 3),
                    "bbox": [
                        round(fruit.bbox.x_min, 1),
                        round(fruit.bbox.y_min, 1),
                        round(fruit.bbox.x_max, 1),
                        round(fruit.bbox.y_max, 1),
                    ],
                    "defects": sorted(
                        [
                            {
                                "type": defect.type,
                                "confidence": round(defect.confidence, 3),
                            }
                            for defect in fruit.defects
                        ],
                        key=lambda item: (item["type"], item["confidence"]),
                    ),
                }
            )

        normalized_fruits.sort(
            key=lambda item: (
                item["fruit_class"],
                item["bbox"],
                item["confidence"],
                len(item["defects"]),
            )
        )
        digest_payload = {
            "weight_bucket": weight_bucket,
            "fruits": normalized_fruits,
        }
        raw = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _bucket_weight(self, grams: float) -> float:
        if self._weight_bucket_grams <= 0:
            return round(grams, 2)
        bucketed = round(grams / self._weight_bucket_grams) * self._weight_bucket_grams
        return round(bucketed, 2)
