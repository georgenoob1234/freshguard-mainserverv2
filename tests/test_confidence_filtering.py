from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import pytest
from PIL import Image

from app.config import Settings
from app.core.orchestrator import ScanOrchestrator, filter_detections_by_confidence
from app.journal import EventJournal
from app.models import (
    CameraCaptureResponse,
    DefectDetectionResponse,
    FruitDetection,
    FruitDetectionResponse,
    ScanTriggerContext,
)


class _StubCameraClient:
    def __init__(self, image_bytes: bytes) -> None:
        self._image_bytes = image_bytes

    async def capture(self, *, use_extra: bool) -> CameraCaptureResponse:  # noqa: ARG002
        return CameraCaptureResponse(
            image_id="img-1",
            image_url_or_path="/api/images/img-1.jpg",
            timestamp=datetime.now(timezone.utc),
        )

    async def fetch_image_bytes(self, image_url_or_path: str) -> bytes:  # noqa: ARG002
        return self._image_bytes

    async def close(self) -> None:
        return None


class _StubFruitClient:
    async def detect(self, *, image_bytes: bytes, imgsz: int) -> FruitDetectionResponse:  # noqa: ARG002
        return FruitDetectionResponse(
            image_id="img-1",
            width=320,
            height=320,
            detections=[
                FruitDetection(fruit_id="low", **{"class": "banana"}, confidence=0.30, bbox=(10, 10, 80, 80)),
                FruitDetection(fruit_id="high", **{"class": "banana"}, confidence=0.90, bbox=(100, 100, 200, 200)),
            ],
        )

    async def close(self) -> None:
        return None


class _StubDefectClient:
    def __init__(self) -> None:
        self.calls = 0

    async def detect(self, *, image_bytes: bytes, image_id: str, fruit_id: str) -> DefectDetectionResponse:  # noqa: ARG002
        self.calls += 1
        return DefectDetectionResponse(image_id=image_id, fruit_id=fruit_id, defects=[])

    async def close(self) -> None:
        return None


class _StubDefectClientWithPolygon:
    def __init__(self) -> None:
        self.crop_size: tuple[int, int] | None = None

    async def detect(self, *, image_bytes: bytes, image_id: str, fruit_id: str) -> DefectDetectionResponse:  # noqa: ARG002
        image = Image.open(BytesIO(image_bytes))
        self.crop_size = image.size
        return DefectDetectionResponse(
            image_id=image_id,
            fruit_id=fruit_id,
            defects=[
                {
                    "type": "defect",
                    "confidence": 0.9,
                    "segmentation": {
                        "polygon": [(5.0, 7.0), (15.0, 7.0), (15.0, 17.0)],
                    },
                }
            ],
        )

    async def close(self) -> None:
        return None


class _StubPublisher:
    async def publish(self, *, payload, path: str) -> None:  # noqa: ANN001, ARG002
        return None

    async def close(self) -> None:
        return None


def _dummy_image_bytes() -> bytes:
    image = Image.new("RGB", (320, 320), color=(120, 180, 120))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_filter_detections_by_confidence_basic_cases() -> None:
    detections = [
        FruitDetection(fruit_id="f1", **{"class": "apple"}, confidence=0.39, bbox=(0, 0, 10, 10)),
        FruitDetection(fruit_id="f2", **{"class": "apple"}, confidence=0.60, bbox=(0, 0, 10, 10)),
        FruitDetection(fruit_id="f3", **{"class": "dragonfruit"}, confidence=0.99, bbox=(0, 0, 10, 10)),
    ]
    valid, dropped = filter_detections_by_confidence(
        detections=detections,
        image_id="img-1",
        allowed_classes={"apple", "banana", "tomato"},
        class_thresholds={"apple": 0.40},
        default_threshold=0.50,
    )

    assert [item.fruit_id for item in valid] == ["f2"]
    assert len(dropped) == 2
    assert {item.reason for item in dropped} == {"below_class_threshold", "unknown_class"}


@pytest.mark.asyncio
async def test_low_confidence_fruit_does_not_reach_defect_detector_and_is_logged(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = Settings(
        CLASS_CONFIDENCE_THRESHOLDS={"banana": 0.50},
        ALLOWED_FRUIT_CLASSES=["banana"],
        JOURNAL_PATH=tmp_path / "journal.jsonl",
    )

    camera_client = _StubCameraClient(_dummy_image_bytes())
    fruit_client = _StubFruitClient()
    defect_client = _StubDefectClient()
    orchestrator = ScanOrchestrator(
        settings=settings,
        journal=EventJournal(settings.JOURNAL_PATH),
        camera_client=camera_client,
        fruit_client=fruit_client,
        defect_client=defect_client,
        ui_publisher=_StubPublisher(),
        main_publisher=_StubPublisher(),
    )

    context = ScanTriggerContext(
        session_id="session-1",
        scan_id="session-1-0001",
        weight_grams=120.0,
        trigger_reason="test",
    )
    caplog.set_level("INFO")
    frame = await orchestrator._process_single_image(
        context=context,
        frame_id=0,
        image_id="img-1",
        image_bytes=_dummy_image_bytes(),
    )

    assert defect_client.calls == 1
    assert len(frame.fruits) == 1
    assert frame.fruits[0].source_fruit_id == "high"
    assert "Fruit dropped due to confidence/class filter" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("padding_ratio", "expected_polygon", "expected_crop_size"),
    [
        (0.15, [(90.0, 92.0), (100.0, 92.0), (100.0, 102.0)], (130, 130)),
        (0.0, [(105.0, 107.0), (115.0, 107.0), (115.0, 117.0)], (100, 100)),
    ],
)
async def test_defect_polygon_translated_to_full_image_coordinates(
    tmp_path,
    padding_ratio: float,
    expected_polygon: list[tuple[float, float]],
    expected_crop_size: tuple[int, int],
) -> None:
    settings = Settings(
        CLASS_CONFIDENCE_THRESHOLDS={"banana": 0.50},
        ALLOWED_FRUIT_CLASSES=["banana"],
        DEFECT_CROP_PADDING_RATIO=padding_ratio,
        JOURNAL_PATH=tmp_path / "journal.jsonl",
    )
    defect_client = _StubDefectClientWithPolygon()

    orchestrator = ScanOrchestrator(
        settings=settings,
        journal=EventJournal(settings.JOURNAL_PATH),
        camera_client=_StubCameraClient(_dummy_image_bytes()),
        fruit_client=_StubFruitClient(),
        defect_client=defect_client,
        ui_publisher=_StubPublisher(),
        main_publisher=_StubPublisher(),
    )
    context = ScanTriggerContext(
        session_id="session-1",
        scan_id="session-1-0001",
        weight_grams=120.0,
        trigger_reason="test",
    )

    frame = await orchestrator._process_single_image(
        context=context,
        frame_id=0,
        image_id="img-1",
        image_bytes=_dummy_image_bytes(),
    )

    assert len(frame.fruits) == 1
    assert frame.fruits[0].bbox.to_xyxy() == (100.0, 100.0, 200.0, 200.0)
    assert defect_client.crop_size == expected_crop_size
    defect = frame.fruits[0].defects[0]
    assert defect.segmentation is not None
    assert defect.segmentation.polygon == expected_polygon
