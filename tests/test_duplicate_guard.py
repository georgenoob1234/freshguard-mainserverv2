from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.duplicate_guard import DuplicateResultGuard
from app.models import BBox, ScanFruit, ScanResult


def _result(weight: float, fruit_class: str = "apple") -> ScanResult:
    return ScanResult(
        session_id="session-1",
        image_id="img-1",
        weight_grams=weight,
        fruits=[
            ScanFruit(
                fruit_id="fruit-1",
                fruit_class=fruit_class,
                confidence=0.9,
                bbox=BBox(x_min=1, y_min=1, x_max=10, y_max=10),
                defects=[],
            )
        ],
    )


def test_duplicate_result_suppressed_within_window() -> None:
    guard = DuplicateResultGuard(window_ms=2000, weight_bucket_grams=5.0)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    first = _result(100.0)
    second = _result(101.0)  # same 5g bucket

    suppressed_first, _ = guard.should_suppress(first, now)
    suppressed_second, _ = guard.should_suppress(second, now + timedelta(milliseconds=500))

    assert suppressed_first is False
    assert suppressed_second is True


def test_different_result_not_suppressed() -> None:
    guard = DuplicateResultGuard(window_ms=2000, weight_bucket_grams=5.0)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    first = _result(100.0, fruit_class="apple")
    second = _result(100.0, fruit_class="banana")

    guard.should_suppress(first, now)
    suppressed_second, _ = guard.should_suppress(second, now + timedelta(milliseconds=500))

    assert suppressed_second is False
