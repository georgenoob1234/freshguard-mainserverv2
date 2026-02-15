from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from time import perf_counter

from app.config import CameraConfig, Settings
from app.core.aggregation import ScanAggregator
from app.core.duplicate_guard import DuplicateResultGuard
from app.core.image_ops import ImageDecodeError, crop_to_jpeg_bytes
from app.journal import EventJournal
from app.logging import get_logger
from app.models import (
    AggregatedScan,
    BBox,
    DefectInfo,
    DroppedDetection,
    FrameEvidence,
    FruitDetection,
    FruitDetectionResponse,
    FruitEvidence,
    ScanResult,
    ScanTriggerContext,
    Segmentation,
    utc_now,
)
from app.services.clients import (
    CameraServiceClient,
    DefectDetectorClient,
    PublisherClient,
    ServiceCallError,
    ServiceValidationError,
)


def filter_detections_by_confidence(
    *,
    detections: list[FruitDetection],
    image_id: str,
    allowed_classes: set[str],
    class_thresholds: dict[str, float],
    default_threshold: float,
) -> tuple[list[FruitDetection], list[DroppedDetection]]:
    valid: list[FruitDetection] = []
    dropped: list[DroppedDetection] = []

    for detection in detections:
        fruit_class = detection.fruit_class
        if fruit_class not in allowed_classes:
            dropped.append(
                DroppedDetection(
                    image_id=image_id,
                    fruit_id=detection.fruit_id,
                    fruit_class=fruit_class,
                    confidence=detection.confidence,
                    threshold=class_thresholds.get(fruit_class, default_threshold),
                    reason="unknown_class",
                )
            )
            continue

        threshold = class_thresholds.get(fruit_class, default_threshold)
        if detection.confidence < threshold:
            dropped.append(
                DroppedDetection(
                    image_id=image_id,
                    fruit_id=detection.fruit_id,
                    fruit_class=fruit_class,
                    confidence=detection.confidence,
                    threshold=threshold,
                    reason="below_class_threshold",
                )
            )
            continue
        valid.append(detection)

    return valid, dropped


def translate_defects_to_image_coordinates(*, defects: list[DefectInfo], fruit_bbox: BBox) -> list[DefectInfo]:
    """Translate defect polygons from crop space into full-image coordinates."""
    translated: list[DefectInfo] = []
    x_offset = fruit_bbox.x_min
    y_offset = fruit_bbox.y_min

    for defect in defects:
        if defect.segmentation is None:
            translated.append(defect.model_copy(deep=True))
            continue

        translated_polygon = [
            (float(point_x + x_offset), float(point_y + y_offset))
            for point_x, point_y in defect.segmentation.polygon
        ]
        translated.append(
            DefectInfo(
                type=defect.type,
                confidence=defect.confidence,
                segmentation=Segmentation(polygon=translated_polygon),
            )
        )

    return translated


class ScanOrchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        journal: EventJournal,
        camera_client: CameraServiceClient,
        fruit_client,
        defect_client: DefectDetectorClient,
        ui_publisher: PublisherClient,
        main_publisher: PublisherClient,
    ) -> None:
        self._settings = settings
        self._journal = journal
        self._camera_client = camera_client
        self._fruit_client = fruit_client
        self._defect_client = defect_client
        self._ui_publisher = ui_publisher
        self._main_publisher = main_publisher

        self._logger = get_logger("brain.orchestrator")
        self._aggregator = ScanAggregator()
        self._duplicate_guard = DuplicateResultGuard(
            window_ms=settings.DUPLICATE_SUPPRESSION_WINDOW_MS,
            weight_bucket_grams=settings.DUPLICATE_WEIGHT_BUCKET_GRAMS,
        )
        self._defect_semaphore = asyncio.Semaphore(max(1, settings.DEFECT_MAX_PARALLEL))
        self._scan_lock = asyncio.Lock()

    async def close(self) -> None:
        await asyncio.gather(
            self._camera_client.close(),
            self._fruit_client.close(),
            self._defect_client.close(),
            self._ui_publisher.close(),
            self._main_publisher.close(),
        )

    async def run_scan(self, context: ScanTriggerContext) -> ScanResult | None:
        async with self._scan_lock:
            await self._journal.write_event(
                "scan_triggered",
                session_id=context.session_id,
                scan_id=context.scan_id,
                weight_grams=context.weight_grams,
                reason=context.trigger_reason,
            )
            self._logger.info(
                "Scan triggered",
                extra={
                    "session_id": context.session_id,
                    "scan_id": context.scan_id,
                    "weight_grams": context.weight_grams,
                    "reason": context.trigger_reason,
                },
            )

            frame_evidences: list[FrameEvidence] = []
            for camera in self._settings.CAMERAS:
                camera_frames = await self._process_camera_frames(context=context, camera=camera)
                frame_evidences.extend(camera_frames)

            aggregated = self._aggregator.aggregate(
                frame_evidences,
                policy=self._settings.AGGREGATION_POLICY,
            )
            result = self._build_scan_result(context=context, aggregated=aggregated)

            await self._journal.write_event(
                "scan_aggregated",
                session_id=context.session_id,
                scan_id=context.scan_id,
                image_id=result.image_id,
                fruit_count=len(result.fruits),
                frames=len(frame_evidences),
            )

            if self._settings.ENABLE_DUPLICATE_SUPPRESSION:
                suppress, result_hash = self._duplicate_guard.should_suppress(result, utc_now())
                if suppress:
                    await self._journal.write_event(
                        "anti_duplicate_suppressed",
                        session_id=context.session_id,
                        scan_id=context.scan_id,
                        result_hash=result_hash,
                    )
                    self._logger.info(
                        "Suppressed duplicate scan result",
                        extra={
                            "session_id": context.session_id,
                            "scan_id": context.scan_id,
                            "result_hash": result_hash,
                        },
                    )
                    return None

            await self._publish_result(result=result, context=context)
            return result

    async def _process_camera_frames(
        self,
        *,
        context: ScanTriggerContext,
        camera: CameraConfig,
    ) -> list[FrameEvidence]:
        frames, interval_ms = self._settings.frames_for_role(camera.role)
        evidences: list[FrameEvidence] = []

        for frame_id in range(max(1, frames)):
            evidence = await self._process_single_frame(
                context=context,
                camera=camera,
                frame_id=frame_id,
            )
            evidences.append(evidence)
            if frame_id < frames - 1:
                await asyncio.sleep(max(0.0, interval_ms / 1000))

        return evidences

    async def _process_single_frame(
        self,
        *,
        context: ScanTriggerContext,
        camera: CameraConfig,
        frame_id: int,
    ) -> FrameEvidence:
        capture_started = perf_counter()
        await self._journal.write_event(
            "capture_started",
            session_id=context.session_id,
            scan_id=context.scan_id,
            camera_id=camera.camera_id,
            frame_id=frame_id,
        )
        try:
            capture_response = await self._camera_client.capture()
        except ServiceCallError as exc:
            await self._journal.write_event(
                "service_call_failed",
                session_id=context.session_id,
                scan_id=context.scan_id,
                camera_id=camera.camera_id,
                frame_id=frame_id,
                service=exc.service,
                http_status=exc.status_code,
                error=str(exc),
            )
            return FrameEvidence(
                camera_id=camera.camera_id,
                camera_role=camera.role,
                frame_id=frame_id,
                image_id=f"{camera.camera_id}-frame-{frame_id}-capture-error",
                frame_error=str(exc),
            )

        capture_duration_ms = int((perf_counter() - capture_started) * 1000)
        await self._journal.write_event(
            "capture_finished",
            session_id=context.session_id,
            scan_id=context.scan_id,
            camera_id=camera.camera_id,
            frame_id=frame_id,
            image_id=capture_response.image_id,
            duration_ms=capture_duration_ms,
        )
        self._logger.info(
            "Camera capture finished",
            extra={
                "session_id": context.session_id,
                "scan_id": context.scan_id,
                "camera_id": camera.camera_id,
                "frame_id": frame_id,
                "image_id": capture_response.image_id,
                "duration_ms": capture_duration_ms,
                "service": "camera",
            },
        )

        try:
            image_bytes = await self._camera_client.fetch_image_bytes(capture_response.image_url_or_path)
        except ServiceCallError as exc:
            await self._journal.write_event(
                "service_call_failed",
                session_id=context.session_id,
                scan_id=context.scan_id,
                camera_id=camera.camera_id,
                frame_id=frame_id,
                image_id=capture_response.image_id,
                service=exc.service,
                http_status=exc.status_code,
                error=str(exc),
            )
            return FrameEvidence(
                camera_id=camera.camera_id,
                camera_role=camera.role,
                frame_id=frame_id,
                image_id=capture_response.image_id,
                frame_error=str(exc),
            )

        try:
            fruit_response = await self._detect_fruits_with_optional_fallback(
                context=context,
                camera=camera,
                frame_id=frame_id,
                image_id=capture_response.image_id,
                image_bytes=image_bytes,
                weight_grams=context.weight_grams,
            )
        except (ServiceCallError, ServiceValidationError) as exc:
            event_type = "validation_failed" if isinstance(exc, ServiceValidationError) else "service_call_failed"
            await self._journal.write_event(
                event_type,
                session_id=context.session_id,
                scan_id=context.scan_id,
                camera_id=camera.camera_id,
                frame_id=frame_id,
                image_id=capture_response.image_id,
                service=getattr(exc, "service", "fruit-detector"),
                error=str(exc),
            )
            return FrameEvidence(
                camera_id=camera.camera_id,
                camera_role=camera.role,
                frame_id=frame_id,
                image_id=capture_response.image_id,
                frame_error=str(exc),
            )

        valid_detections, dropped = filter_detections_by_confidence(
            detections=fruit_response.detections,
            image_id=capture_response.image_id,
            allowed_classes=set(self._settings.ALLOWED_FRUIT_CLASSES),
            class_thresholds=self._settings.CLASS_CONFIDENCE_THRESHOLDS,
            default_threshold=self._settings.DEFAULT_CLASS_CONFIDENCE_THRESHOLD,
        )

        for item in dropped:
            self._logger.info(
                "Fruit dropped due to confidence/class filter",
                extra={
                    "session_id": context.session_id,
                    "scan_id": context.scan_id,
                    "camera_id": camera.camera_id,
                    "frame_id": frame_id,
                    "image_id": item.image_id,
                    "fruit_id": item.fruit_id,
                    "class_name": item.fruit_class,
                    "confidence": item.confidence,
                    "threshold": item.threshold,
                    "drop_reason": item.reason,
                },
            )
            await self._journal.write_event(
                "detection_dropped",
                session_id=context.session_id,
                scan_id=context.scan_id,
                camera_id=camera.camera_id,
                frame_id=frame_id,
                image_id=item.image_id,
                fruit_id=item.fruit_id,
                fruit_class=item.fruit_class,
                confidence=item.confidence,
                threshold=item.threshold,
                drop_reason=item.reason,
            )

        fruit_evidences = await self._collect_defect_evidence(
            context=context,
            camera=camera,
            frame_id=frame_id,
            image_id=capture_response.image_id,
            image_bytes=image_bytes,
            detections=valid_detections,
        )
        return FrameEvidence(
            camera_id=camera.camera_id,
            camera_role=camera.role,
            frame_id=frame_id,
            image_id=capture_response.image_id,
            fruits=fruit_evidences,
        )

    async def _detect_fruits_with_optional_fallback(
        self,
        *,
        context: ScanTriggerContext,
        camera: CameraConfig,
        frame_id: int,
        image_id: str,
        image_bytes: bytes,
        weight_grams: float,
    ) -> FruitDetectionResponse:
        primary_imgsz = self._settings.FRUIT_PRIMARY_IMGSZ
        fallback_imgsz = self._settings.FRUIT_FALLBACK_IMGSZ

        await self._journal.write_event(
            "fruit_detect_started",
            session_id=context.session_id,
            scan_id=context.scan_id,
            camera_id=camera.camera_id,
            frame_id=frame_id,
            image_id=image_id,
            imgsz=primary_imgsz,
        )
        started = perf_counter()
        primary = await self._fruit_client.detect(image_bytes=image_bytes, imgsz=primary_imgsz)
        primary_duration_ms = int((perf_counter() - started) * 1000)
        await self._journal.write_event(
            "fruit_detect_finished",
            session_id=context.session_id,
            scan_id=context.scan_id,
            camera_id=camera.camera_id,
            frame_id=frame_id,
            image_id=image_id,
            imgsz=primary_imgsz,
            duration_ms=primary_duration_ms,
            detections=len(primary.detections),
        )
        self._logger.info(
            "Fruit detection finished",
            extra={
                "session_id": context.session_id,
                "scan_id": context.scan_id,
                "camera_id": camera.camera_id,
                "frame_id": frame_id,
                "image_id": image_id,
                "imgsz": primary_imgsz,
                "duration_ms": primary_duration_ms,
                "detections": len(primary.detections),
                "service": "fruit_detector",
            },
        )

        fallback_reason = self._fallback_reason(primary=primary, weight_grams=weight_grams)
        if fallback_reason is None:
            return primary

        self._logger.info(
            "Running fallback fruit detection",
            extra={
                "session_id": context.session_id,
                "scan_id": context.scan_id,
                "camera_id": camera.camera_id,
                "frame_id": frame_id,
                "image_id": image_id,
                "fallback_reason": fallback_reason,
                "primary_imgsz": primary_imgsz,
                "fallback_imgsz": fallback_imgsz,
            },
        )

        await self._journal.write_event(
            "fruit_detect_started",
            session_id=context.session_id,
            scan_id=context.scan_id,
            camera_id=camera.camera_id,
            frame_id=frame_id,
            image_id=image_id,
            imgsz=fallback_imgsz,
            fallback_reason=fallback_reason,
        )
        started_fallback = perf_counter()
        try:
            fallback = await self._fruit_client.detect(image_bytes=image_bytes, imgsz=fallback_imgsz)
        except (ServiceCallError, ServiceValidationError):
            # If fallback fails, keep primary result as bounded behavior.
            return primary

        fallback_duration_ms = int((perf_counter() - started_fallback) * 1000)
        await self._journal.write_event(
            "fruit_detect_finished",
            session_id=context.session_id,
            scan_id=context.scan_id,
            camera_id=camera.camera_id,
            frame_id=frame_id,
            image_id=image_id,
            imgsz=fallback_imgsz,
            duration_ms=fallback_duration_ms,
            detections=len(fallback.detections),
            fallback_reason=fallback_reason,
        )
        self._logger.info(
            "Fruit detection fallback finished",
            extra={
                "session_id": context.session_id,
                "scan_id": context.scan_id,
                "camera_id": camera.camera_id,
                "frame_id": frame_id,
                "image_id": image_id,
                "imgsz": fallback_imgsz,
                "duration_ms": fallback_duration_ms,
                "detections": len(fallback.detections),
                "fallback_reason": fallback_reason,
                "service": "fruit_detector",
            },
        )
        return fallback

    def _fallback_reason(self, *, primary: FruitDetectionResponse, weight_grams: float) -> str | None:
        detections = primary.detections
        if weight_grams >= self._settings.MIN_FRUIT_WEIGHT and not detections:
            return "no_detections_with_weight"

        if detections and all(
            detection.confidence < self._settings.FRUIT_LOW_CONFIDENCE_FALLBACK_THRESHOLD
            for detection in detections
        ):
            return "all_low_confidence"

        if weight_grams >= self._settings.MIN_FRUIT_WEIGHT * 2 and len(detections) <= 1:
            return "weight_suggests_multiple_fruits"

        if detections:
            image_area = float(primary.width * primary.height)
            tiny = all(
                (detection.area / image_area) <= self._settings.FRUIT_TINY_BBOX_AREA_RATIO
                for detection in detections
            )
            if tiny:
                return "tiny_suspicious_bboxes"

        return None

    async def _collect_defect_evidence(
        self,
        *,
        context: ScanTriggerContext,
        camera: CameraConfig,
        frame_id: int,
        image_id: str,
        image_bytes: bytes,
        detections: list[FruitDetection],
    ) -> list[FruitEvidence]:
        tasks = [
            asyncio.create_task(
                self._detect_single_fruit_defects(
                    context=context,
                    camera=camera,
                    frame_id=frame_id,
                    image_id=image_id,
                    image_bytes=image_bytes,
                    detection=detection,
                )
            )
            for detection in detections
        ]
        if not tasks:
            return []
        return await asyncio.gather(*tasks)

    async def _detect_single_fruit_defects(
        self,
        *,
        context: ScanTriggerContext,
        camera: CameraConfig,
        frame_id: int,
        image_id: str,
        image_bytes: bytes,
        detection: FruitDetection,
    ) -> FruitEvidence:
        bbox = BBox.from_xyxy(detection.bbox)
        try:
            crop_bytes = crop_to_jpeg_bytes(image_bytes, bbox)
        except ImageDecodeError as exc:
            await self._journal.write_event(
                "service_call_failed",
                session_id=context.session_id,
                scan_id=context.scan_id,
                camera_id=camera.camera_id,
                frame_id=frame_id,
                image_id=image_id,
                fruit_id=detection.fruit_id,
                service="image_ops",
                error=str(exc),
            )
            return FruitEvidence(
                source_fruit_id=detection.fruit_id,
                fruit_class=detection.fruit_class,
                confidence=detection.confidence,
                bbox=bbox,
                defects=[],
                note=str(exc),
            )

        started = perf_counter()
        await self._journal.write_event(
            "defect_detect_started",
            session_id=context.session_id,
            scan_id=context.scan_id,
            camera_id=camera.camera_id,
            frame_id=frame_id,
            image_id=image_id,
            fruit_id=detection.fruit_id,
        )

        async with self._defect_semaphore:
            try:
                response = await self._defect_client.detect(
                    image_bytes=crop_bytes,
                    image_id=image_id,
                    fruit_id=detection.fruit_id,
                )
            except ServiceCallError as exc:
                await self._journal.write_event(
                    "service_call_failed",
                    session_id=context.session_id,
                    scan_id=context.scan_id,
                    camera_id=camera.camera_id,
                    frame_id=frame_id,
                    image_id=image_id,
                    fruit_id=detection.fruit_id,
                    service=exc.service,
                    http_status=exc.status_code,
                    error=str(exc),
                )
                return FruitEvidence(
                    source_fruit_id=detection.fruit_id,
                    fruit_class=detection.fruit_class,
                    confidence=detection.confidence,
                    bbox=bbox,
                    defects=[],
                    note=f"defect_service_error:{exc}",
                )
            except ServiceValidationError as exc:
                await self._journal.write_event(
                    "validation_failed",
                    session_id=context.session_id,
                    scan_id=context.scan_id,
                    camera_id=camera.camera_id,
                    frame_id=frame_id,
                    image_id=image_id,
                    fruit_id=detection.fruit_id,
                    service=exc.service,
                    error=str(exc),
                )
                return FruitEvidence(
                    source_fruit_id=detection.fruit_id,
                    fruit_class=detection.fruit_class,
                    confidence=detection.confidence,
                    bbox=bbox,
                    defects=[],
                    note=f"defect_validation_error:{exc}",
                )

        defect_duration_ms = int((perf_counter() - started) * 1000)
        await self._journal.write_event(
            "defect_detect_finished",
            session_id=context.session_id,
            scan_id=context.scan_id,
            camera_id=camera.camera_id,
            frame_id=frame_id,
            image_id=image_id,
            fruit_id=detection.fruit_id,
            defects=len(response.defects),
            duration_ms=defect_duration_ms,
        )
        self._logger.info(
            "Defect detection finished",
            extra={
                "session_id": context.session_id,
                "scan_id": context.scan_id,
                "camera_id": camera.camera_id,
                "frame_id": frame_id,
                "image_id": image_id,
                "fruit_id": detection.fruit_id,
                "duration_ms": defect_duration_ms,
                "defects": len(response.defects),
                "service": "defect_detector",
            },
        )
        defects_in_image_space = translate_defects_to_image_coordinates(
            defects=response.defects,
            fruit_bbox=bbox,
        )
        return FruitEvidence(
            source_fruit_id=detection.fruit_id,
            fruit_class=detection.fruit_class,
            confidence=detection.confidence,
            bbox=bbox,
            defects=defects_in_image_space,
        )

    def _build_scan_result(self, *, context: ScanTriggerContext, aggregated: AggregatedScan) -> ScanResult:
        image_id = aggregated.representative_image_id or "unknown"
        return ScanResult(
            session_id=context.session_id,
            image_id=image_id,
            timestamp=datetime.now(timezone.utc),
            weight_grams=context.weight_grams,
            fruits=aggregated.fruits,
        )

    async def _publish_result(self, *, result: ScanResult, context: ScanTriggerContext) -> None:
        try:
            started = perf_counter()
            await self._ui_publisher.publish(payload=result, path=self._settings.UI_PUBLISH_PATH)
            duration_ms = int((perf_counter() - started) * 1000)
            await self._journal.write_event(
                "scan_published_ui",
                session_id=context.session_id,
                scan_id=context.scan_id,
                image_id=result.image_id,
                fruit_count=len(result.fruits),
                duration_ms=duration_ms,
            )
            self._logger.info(
                "Published to UI",
                extra={
                    "session_id": context.session_id,
                    "scan_id": context.scan_id,
                    "image_id": result.image_id,
                    "duration_ms": duration_ms,
                    "service": "ui",
                },
            )
        except ServiceCallError as exc:
            await self._journal.write_event(
                "service_call_failed",
                session_id=context.session_id,
                scan_id=context.scan_id,
                service=exc.service,
                http_status=exc.status_code,
                error=str(exc),
                publish_target="ui",
            )

        try:
            started = perf_counter()
            await self._main_publisher.publish(payload=result, path=self._settings.MAIN_SERVER_PUBLISH_PATH)
            duration_ms = int((perf_counter() - started) * 1000)
            await self._journal.write_event(
                "scan_published_main_server",
                session_id=context.session_id,
                scan_id=context.scan_id,
                image_id=result.image_id,
                fruit_count=len(result.fruits),
                duration_ms=duration_ms,
            )
            self._logger.info(
                "Published to main server",
                extra={
                    "session_id": context.session_id,
                    "scan_id": context.scan_id,
                    "image_id": result.image_id,
                    "duration_ms": duration_ms,
                    "service": "main_server",
                },
            )
        except ServiceCallError as exc:
            await self._journal.write_event(
                "service_call_failed",
                session_id=context.session_id,
                scan_id=context.scan_id,
                service=exc.service,
                http_status=exc.status_code,
                error=str(exc),
                publish_target="main_server",
            )
