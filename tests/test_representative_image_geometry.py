from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import pytest
from PIL import Image

from app.config import Settings
from app.core.orchestrator import ScanOrchestrator
from app.journal import EventJournal
from app.models import (
    CameraCaptureImage,
    CameraCaptureResponse,
    DefectDetectionResponse,
    FruitDetection,
    FruitDetectionResponse,
    ScanResult,
    ScanTriggerContext,
)


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (320, 320), color=color)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class _StubCameraClient:
    def __init__(self, *, capture_response: CameraCaptureResponse, image_by_path: dict[str, bytes]) -> None:
        self._capture_response = capture_response
        self._image_by_path = image_by_path

    async def capture(self, *, use_extra: bool) -> CameraCaptureResponse:  # noqa: ARG002
        return self._capture_response

    async def fetch_image_bytes(self, image_url_or_path: str) -> bytes:
        return self._image_by_path[image_url_or_path]

    async def close(self) -> None:
        return None


class _StubFruitClient:
    def __init__(self, response_by_bytes: dict[bytes, FruitDetectionResponse]) -> None:
        self._response_by_bytes = response_by_bytes

    async def detect(self, *, image_bytes: bytes, imgsz: int) -> FruitDetectionResponse:  # noqa: ARG002
        return self._response_by_bytes[image_bytes]

    async def close(self) -> None:
        return None


class _RecordingDefectClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def detect(self, *, image_bytes: bytes, image_id: str, fruit_id: str) -> DefectDetectionResponse:  # noqa: ARG002
        self.calls.append((image_id, fruit_id))
        return DefectDetectionResponse(image_id=image_id, fruit_id=fruit_id, defects=[])

    async def close(self) -> None:
        return None


class _RecordingPublisher:
    def __init__(self) -> None:
        self.payloads: list[ScanResult] = []

    async def publish(self, *, payload: ScanResult, path: str) -> None:  # noqa: ARG002
        self.payloads.append(payload.model_copy(deep=True))

    async def close(self) -> None:
        return None


def _context() -> ScanTriggerContext:
    return ScanTriggerContext(
        session_id="session-1",
        scan_id="session-1-0001",
        weight_grams=100.0,
        trigger_reason="test",
    )


def _settings(tmp_path, *, aggregation_policy: str = "vote") -> Settings:
    return Settings(
        JOURNAL_PATH=tmp_path / "journal.jsonl",
        ENABLE_DUPLICATE_SUPPRESSION=False,
        CAMERA_USE_EXTRA_DEFAULT=True,
        ALLOWED_FRUIT_CLASSES=["apple"],
        CLASS_CONFIDENCE_THRESHOLDS={"apple": 0.40},
        DEFAULT_CLASS_CONFIDENCE_THRESHOLD=0.40,
        MIN_FRUIT_WEIGHT=1000.0,
        AGGREGATION_POLICY=aggregation_policy,
    )


def _build_orchestrator(
    *,
    settings: Settings,
    capture_response: CameraCaptureResponse,
    image_by_path: dict[str, bytes],
    response_by_bytes: dict[bytes, FruitDetectionResponse],
    defect_client: _RecordingDefectClient,
    ui_publisher: _RecordingPublisher,
    main_publisher: _RecordingPublisher,
) -> ScanOrchestrator:
    return ScanOrchestrator(
        settings=settings,
        journal=EventJournal(settings.JOURNAL_PATH),
        camera_client=_StubCameraClient(capture_response=capture_response, image_by_path=image_by_path),
        fruit_client=_StubFruitClient(response_by_bytes=response_by_bytes),
        defect_client=defect_client,
        ui_publisher=ui_publisher,
        main_publisher=main_publisher,
    )


def _capture_with_three_images() -> CameraCaptureResponse:
    return CameraCaptureResponse(
        image_id="img-main",
        image_url_or_path="/images/main.jpg",
        timestamp=datetime.now(timezone.utc),
        images=[
            CameraCaptureImage(index=0, image_id="img-main", image_url_or_path="/images/main.jpg"),
            CameraCaptureImage(index=1, image_id="img-extra-1", image_url_or_path="/images/extra-1.jpg"),
            CameraCaptureImage(index=2, image_id="img-extra-2", image_url_or_path="/images/extra-2.jpg"),
        ],
    )


def _detection_response(*, image_id: str, detections: list[FruitDetection]) -> FruitDetectionResponse:
    return FruitDetectionResponse(
        image_id=image_id,
        width=320,
        height=320,
        detections=detections,
    )


@pytest.mark.asyncio
async def test_main_detection_wins_and_geometry_matches_main(tmp_path) -> None:
    main_bytes = _jpeg_bytes((200, 0, 0))
    extra1_bytes = _jpeg_bytes((0, 200, 0))
    extra2_bytes = _jpeg_bytes((0, 0, 200))
    capture_response = _capture_with_three_images()

    main_detection = FruitDetection(
        fruit_id="m1",
        **{"class": "apple"},
        confidence=0.70,
        bbox=(10, 10, 60, 60),
    )
    extra_detection = FruitDetection(
        fruit_id="e1",
        **{"class": "apple"},
        confidence=0.95,
        bbox=(120, 120, 180, 180),
    )
    response_by_bytes = {
        main_bytes: _detection_response(image_id="img-main", detections=[main_detection]),
        extra1_bytes: _detection_response(image_id="img-extra-1", detections=[extra_detection]),
        extra2_bytes: _detection_response(image_id="img-extra-2", detections=[]),
    }
    defect_client = _RecordingDefectClient()
    ui_publisher = _RecordingPublisher()
    main_publisher = _RecordingPublisher()
    orchestrator = _build_orchestrator(
        settings=_settings(tmp_path),
        capture_response=capture_response,
        image_by_path={
            "/images/main.jpg": main_bytes,
            "/images/extra-1.jpg": extra1_bytes,
            "/images/extra-2.jpg": extra2_bytes,
        },
        response_by_bytes=response_by_bytes,
        defect_client=defect_client,
        ui_publisher=ui_publisher,
        main_publisher=main_publisher,
    )

    result = await orchestrator.run_scan(_context())

    assert result is not None
    assert result.image_id == "img-main"
    assert len(result.fruits) == 1
    assert result.fruits[0].bbox.to_xyxy() == main_detection.bbox
    assert all(call[0] == "img-main" for call in defect_client.calls)
    assert ui_publisher.payloads[0].image_id == "img-main"
    assert ui_publisher.payloads[0].fruits[0].bbox.to_xyxy() == main_detection.bbox


@pytest.mark.asyncio
async def test_fallback_to_first_extra_with_detection_when_main_has_none(tmp_path) -> None:
    main_bytes = _jpeg_bytes((200, 0, 0))
    extra1_bytes = _jpeg_bytes((0, 200, 0))
    extra2_bytes = _jpeg_bytes((0, 0, 200))
    capture_response = _capture_with_three_images()

    extra1_detection = FruitDetection(
        fruit_id="e1",
        **{"class": "apple"},
        confidence=0.88,
        bbox=(40, 40, 90, 90),
    )
    extra2_detection = FruitDetection(
        fruit_id="e2",
        **{"class": "apple"},
        confidence=0.99,
        bbox=(130, 130, 200, 200),
    )
    response_by_bytes = {
        main_bytes: _detection_response(image_id="img-main", detections=[]),
        extra1_bytes: _detection_response(image_id="img-extra-1", detections=[extra1_detection]),
        extra2_bytes: _detection_response(image_id="img-extra-2", detections=[extra2_detection]),
    }
    defect_client = _RecordingDefectClient()
    ui_publisher = _RecordingPublisher()
    main_publisher = _RecordingPublisher()
    orchestrator = _build_orchestrator(
        settings=_settings(tmp_path),
        capture_response=capture_response,
        image_by_path={
            "/images/main.jpg": main_bytes,
            "/images/extra-1.jpg": extra1_bytes,
            "/images/extra-2.jpg": extra2_bytes,
        },
        response_by_bytes=response_by_bytes,
        defect_client=defect_client,
        ui_publisher=ui_publisher,
        main_publisher=main_publisher,
    )

    result = await orchestrator.run_scan(_context())

    assert result is not None
    assert result.image_id == "img-extra-1"
    assert len(result.fruits) == 1
    assert result.fruits[0].bbox.to_xyxy() == extra1_detection.bbox
    assert all(call[0] == "img-extra-1" for call in defect_client.calls)
    assert ui_publisher.payloads[0].image_id == "img-extra-1"
    assert ui_publisher.payloads[0].fruits[0].bbox.to_xyxy() == extra1_detection.bbox


@pytest.mark.asyncio
async def test_best_frame_plus_vote_still_keeps_image_and_bbox_from_same_frame(tmp_path) -> None:
    main_bytes = _jpeg_bytes((200, 0, 0))
    extra1_bytes = _jpeg_bytes((0, 200, 0))
    extra2_bytes = _jpeg_bytes((0, 0, 200))
    capture_response = _capture_with_three_images()

    extra1_detection = FruitDetection(
        fruit_id="e1",
        **{"class": "apple"},
        confidence=0.97,
        bbox=(55, 55, 120, 120),
    )
    extra2_detection = FruitDetection(
        fruit_id="e2",
        **{"class": "apple"},
        confidence=0.62,
        bbox=(150, 150, 210, 210),
    )
    response_by_bytes = {
        main_bytes: _detection_response(image_id="img-main", detections=[]),
        extra1_bytes: _detection_response(image_id="img-extra-1", detections=[extra1_detection]),
        extra2_bytes: _detection_response(image_id="img-extra-2", detections=[extra2_detection]),
    }
    defect_client = _RecordingDefectClient()
    ui_publisher = _RecordingPublisher()
    main_publisher = _RecordingPublisher()
    orchestrator = _build_orchestrator(
        settings=_settings(tmp_path, aggregation_policy="best_frame_plus_vote"),
        capture_response=capture_response,
        image_by_path={
            "/images/main.jpg": main_bytes,
            "/images/extra-1.jpg": extra1_bytes,
            "/images/extra-2.jpg": extra2_bytes,
        },
        response_by_bytes=response_by_bytes,
        defect_client=defect_client,
        ui_publisher=ui_publisher,
        main_publisher=main_publisher,
    )

    result = await orchestrator.run_scan(_context())

    assert result is not None
    assert result.image_id == "img-extra-1"
    assert len(result.fruits) == 1
    assert result.fruits[0].bbox.to_xyxy() == extra1_detection.bbox
    assert all(call[0] == "img-extra-1" for call in defect_client.calls)
