from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import pytest
from PIL import Image

from app.config import Settings
from app.core.orchestrator import ScanOrchestrator
from app.journal import EventJournal
from app.models import CameraCaptureImage, CameraCaptureResponse, FruitDetectionResponse, ScanTriggerContext
from app.services.clients import ServiceCallError


class _StubCameraClient:
    def __init__(
        self,
        *,
        capture_response: CameraCaptureResponse,
        image_bytes: bytes,
        fail_paths: set[str] | None = None,
    ) -> None:
        self._capture_response = capture_response
        self._image_bytes = image_bytes
        self._fail_paths = fail_paths or set()
        self.capture_calls = 0
        self.capture_use_extra_values: list[bool] = []

    async def capture(self, *, use_extra: bool) -> CameraCaptureResponse:
        self.capture_calls += 1
        self.capture_use_extra_values.append(use_extra)
        return self._capture_response

    async def fetch_image_bytes(self, image_url_or_path: str) -> bytes:
        if image_url_or_path in self._fail_paths:
            raise ServiceCallError("camera-service", "download failed")
        return self._image_bytes

    async def close(self) -> None:
        return None


class _StubFruitClient:
    async def detect(self, *, image_bytes: bytes, imgsz: int) -> FruitDetectionResponse:  # noqa: ARG002
        return FruitDetectionResponse(
            image_id="img-any",
            width=320,
            height=320,
            detections=[],
        )

    async def close(self) -> None:
        return None


class _StubDefectClient:
    async def detect(self, *, image_bytes: bytes, image_id: str, fruit_id: str):  # noqa: ANN001, ARG002
        raise AssertionError("Defect detector should not be called in these tests")

    async def close(self) -> None:
        return None


class _StubPublisher:
    def __init__(self) -> None:
        self.calls = 0

    async def publish(self, *, payload, path: str) -> None:  # noqa: ANN001, ARG002
        self.calls += 1

    async def close(self) -> None:
        return None


def _dummy_image_bytes() -> bytes:
    image = Image.new("RGB", (320, 320), color=(60, 100, 200))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _scan_context() -> ScanTriggerContext:
    return ScanTriggerContext(
        session_id="session-1",
        scan_id="session-1-0001",
        weight_grams=200.0,
        trigger_reason="test",
    )


def _make_orchestrator(
    *,
    settings: Settings,
    camera_client: _StubCameraClient,
    ui_publisher: _StubPublisher,
    main_publisher: _StubPublisher,
) -> ScanOrchestrator:
    return ScanOrchestrator(
        settings=settings,
        journal=EventJournal(settings.JOURNAL_PATH),
        camera_client=camera_client,
        fruit_client=_StubFruitClient(),
        defect_client=_StubDefectClient(),
        ui_publisher=ui_publisher,
        main_publisher=main_publisher,
    )


@pytest.mark.asyncio
async def test_single_capture_old_schema_fallback_and_use_extra_flag(tmp_path) -> None:
    capture_response = CameraCaptureResponse(
        image_id="img-main",
        image_url_or_path="/images/main.jpg",
        timestamp=datetime.now(timezone.utc),
    )
    camera_client = _StubCameraClient(capture_response=capture_response, image_bytes=_dummy_image_bytes())
    settings = Settings(
        CAMERA_USE_EXTRA_DEFAULT=False,
        JOURNAL_PATH=tmp_path / "journal.jsonl",
    )
    ui_publisher = _StubPublisher()
    main_publisher = _StubPublisher()
    orchestrator = _make_orchestrator(
        settings=settings,
        camera_client=camera_client,
        ui_publisher=ui_publisher,
        main_publisher=main_publisher,
    )

    result = await orchestrator.run_scan(_scan_context())

    assert result is not None
    assert result.image_id == "img-main"
    assert camera_client.capture_calls == 1
    assert camera_client.capture_use_extra_values == [False]
    assert ui_publisher.calls == 1
    assert main_publisher.calls == 1


@pytest.mark.asyncio
async def test_extra_image_download_failure_logs_warning_and_continues(tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    capture_response = CameraCaptureResponse(
        image_id="img-main",
        image_url_or_path="/images/main.jpg",
        timestamp=datetime.now(timezone.utc),
        images=[
            CameraCaptureImage(index=0, image_id="img-main", image_url_or_path="/images/main.jpg"),
            CameraCaptureImage(index=1, image_id="img-extra", image_url_or_path="/images/extra.jpg"),
        ],
    )
    camera_client = _StubCameraClient(
        capture_response=capture_response,
        image_bytes=_dummy_image_bytes(),
        fail_paths={"/images/extra.jpg"},
    )
    settings = Settings(
        CAMERA_USE_EXTRA_DEFAULT=True,
        JOURNAL_PATH=tmp_path / "journal.jsonl",
    )
    ui_publisher = _StubPublisher()
    main_publisher = _StubPublisher()
    orchestrator = _make_orchestrator(
        settings=settings,
        camera_client=camera_client,
        ui_publisher=ui_publisher,
        main_publisher=main_publisher,
    )

    caplog.set_level("WARNING")
    result = await orchestrator.run_scan(_scan_context())

    assert result is not None
    assert result.image_id == "img-main"
    assert camera_client.capture_calls == 1
    assert camera_client.capture_use_extra_values == [True]
    assert "Extra image download failed; continuing scan" in caplog.text
    assert ui_publisher.calls == 1
    assert main_publisher.calls == 1


@pytest.mark.asyncio
async def test_main_image_download_failure_aborts_scan(tmp_path) -> None:
    capture_response = CameraCaptureResponse(
        image_id="img-main",
        image_url_or_path="/images/main.jpg",
        timestamp=datetime.now(timezone.utc),
    )
    camera_client = _StubCameraClient(
        capture_response=capture_response,
        image_bytes=_dummy_image_bytes(),
        fail_paths={"/images/main.jpg"},
    )
    settings = Settings(
        CAMERA_USE_EXTRA_DEFAULT=True,
        JOURNAL_PATH=tmp_path / "journal.jsonl",
    )
    ui_publisher = _StubPublisher()
    main_publisher = _StubPublisher()
    orchestrator = _make_orchestrator(
        settings=settings,
        camera_client=camera_client,
        ui_publisher=ui_publisher,
        main_publisher=main_publisher,
    )

    result = await orchestrator.run_scan(_scan_context())

    assert result is None
    assert camera_client.capture_calls == 1
    assert ui_publisher.calls == 0
    assert main_publisher.calls == 0


@pytest.mark.asyncio
async def test_images_contract_mismatch_fails_scan(tmp_path) -> None:
    capture_response = CameraCaptureResponse(
        image_id="img-top",
        image_url_or_path="/images/top.jpg",
        timestamp=datetime.now(timezone.utc),
        images=[
            CameraCaptureImage(index=0, image_id="img-0", image_url_or_path="/images/zero.jpg"),
        ],
    )
    camera_client = _StubCameraClient(
        capture_response=capture_response,
        image_bytes=_dummy_image_bytes(),
    )
    settings = Settings(
        CAMERA_USE_EXTRA_DEFAULT=True,
        JOURNAL_PATH=tmp_path / "journal.jsonl",
    )
    ui_publisher = _StubPublisher()
    main_publisher = _StubPublisher()
    orchestrator = _make_orchestrator(
        settings=settings,
        camera_client=camera_client,
        ui_publisher=ui_publisher,
        main_publisher=main_publisher,
    )

    result = await orchestrator.run_scan(_scan_context())

    assert result is None
    assert camera_client.capture_calls == 1
    assert ui_publisher.calls == 0
    assert main_publisher.calls == 0
