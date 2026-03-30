from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter

from app.config import Settings
from app.core.aggregation import ScanAggregator
from app.core.duplicate_guard import DuplicateResultGuard
from app.core.image_ops import ImageDecodeError, crop_to_jpeg_bytes
from app.journal import EventJournal
from app.logging import get_logger
from app.models import (
    AggregatedScan,
    BBox,
    CameraCaptureImage,
    CameraCaptureResponse,
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


@dataclass(frozen=True)
class DownloadedImage:
    image: CameraCaptureImage
    image_bytes: bytes


@dataclass(frozen=True)
class ImageDownloadFailure:
    image: CameraCaptureImage
    error: ServiceCallError


@dataclass(frozen=True)
class DetectedImage:
    image: CameraCaptureImage
    image_bytes: bytes
    valid_detections: list[FruitDetection]
    frame_error: str | None = None


@dataclass(frozen=True)
class RepresentativeSelection:
    representative: DetectedImage | None
    main_has_detection: bool
    reason: str


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
        self._image_download_semaphore = asyncio.Semaphore(6)
        self._image_pipeline_semaphore = asyncio.Semaphore(6)
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
                operating_mode=context.operating_mode,
            )
            self._logger.info(
                "Scan triggered",
                extra={
                    "session_id": context.session_id,
                    "scan_id": context.scan_id,
                    "weight_grams": context.weight_grams,
                    "reason": context.trigger_reason,
                    "operating_mode": context.operating_mode,
                },
            )

            capture_use_extra = bool(self._settings.CAMERA_USE_EXTRA_DEFAULT)
            capture_response = await self._capture_once(
                context=context,
                capture_use_extra=capture_use_extra,
            )
            if capture_response is None:
                return None

            try:
                captured_images = self._normalize_capture_images(capture_response=capture_response)
            except ServiceValidationError as exc:
                await self._journal.write_event(
                    "validation_failed",
                    session_id=context.session_id,
                    scan_id=context.scan_id,
                    image_id=capture_response.image_id,
                    service=exc.service,
                    error=str(exc),
                )
                self._logger.error(
                    "Camera capture contract violation",
                    extra={
                        "session_id": context.session_id,
                        "scan_id": context.scan_id,
                        "image_id": capture_response.image_id,
                        "error": str(exc),
                        "service": "camera",
                    },
                )
                return None

            self._logger.info(
                "Camera capture parsed",
                extra={
                    "session_id": context.session_id,
                    "scan_id": context.scan_id,
                    "capture_use_extra": capture_use_extra,
                    "number_of_images_returned": len(captured_images),
                    "service": "camera",
                },
            )

            downloaded_images = await self._download_captured_images(
                context=context,
                captured_images=captured_images,
            )
            if downloaded_images is None:
                return None

            detected_images = await self._detect_downloaded_images(
                context=context,
                downloaded_images=downloaded_images,
            )
            preliminary_frame_evidences = self._build_detection_only_frame_evidences(detected_images=detected_images)

            selection = self._select_representative_image(detected_images=detected_images)
            preliminary_aggregated = self._aggregator.aggregate(
                preliminary_frame_evidences,
                policy=self._settings.AGGREGATION_POLICY,
            )
            selection = self._maybe_adjust_representative_for_aggregated_classes(
                detected_images=detected_images,
                selection=selection,
                aggregated=preliminary_aggregated,
            )

            self._logger.info(
                "Representative image selected",
                extra={
                    "session_id": context.session_id,
                    "scan_id": context.scan_id,
                    "main_has_detection": selection.main_has_detection,
                    "representative_image_index": (
                        selection.representative.image.index if selection.representative is not None else 0
                    ),
                    "reason": selection.reason,
                    "service": "camera",
                },
            )

            frame_evidences = await self._build_frame_evidences_with_representative(
                context=context,
                detected_images=detected_images,
                selection=selection,
            )

            aggregated = self._aggregator.aggregate(
                frame_evidences,
                policy=self._settings.AGGREGATION_POLICY,
            )
            representative_image_id = (
                selection.representative.image.image_id
                if selection.representative is not None
                else captured_images[0].image_id
            )
            aggregated = aggregated.model_copy(update={"representative_image_id": representative_image_id})
            result = self._build_scan_result(context=context, aggregated=aggregated)
            if selection.representative is not None:
                representative_frame = self._find_representative_frame(
                    frame_evidences=frame_evidences,
                    representative=selection.representative,
                )
                if representative_frame is not None:
                    result = self._enforce_representative_geometry(
                        context=context,
                        result=result,
                        representative_frame=representative_frame,
                    )

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

    async def _capture_once(
        self,
        *,
        context: ScanTriggerContext,
        capture_use_extra: bool,
    ) -> CameraCaptureResponse | None:
        capture_started = perf_counter()
        await self._journal.write_event(
            "capture_started",
            session_id=context.session_id,
            scan_id=context.scan_id,
            capture_use_extra=capture_use_extra,
        )
        try:
            capture_response = await self._camera_client.capture(use_extra=capture_use_extra)
        except (ServiceCallError, ServiceValidationError) as exc:
            event_type = "validation_failed" if isinstance(exc, ServiceValidationError) else "service_call_failed"
            await self._journal.write_event(
                event_type,
                session_id=context.session_id,
                scan_id=context.scan_id,
                service=exc.service,
                http_status=getattr(exc, "status_code", None),
                error=str(exc),
            )
            self._logger.error(
                "Camera capture failed",
                extra={
                    "session_id": context.session_id,
                    "scan_id": context.scan_id,
                    "capture_use_extra": capture_use_extra,
                    "error": str(exc),
                    "service": "camera",
                },
            )
            return None

        capture_duration_ms = int((perf_counter() - capture_started) * 1000)
        await self._journal.write_event(
            "capture_finished",
            session_id=context.session_id,
            scan_id=context.scan_id,
            image_id=capture_response.image_id,
            duration_ms=capture_duration_ms,
            capture_use_extra=capture_use_extra,
        )
        self._logger.info(
            "Camera capture finished",
            extra={
                "session_id": context.session_id,
                "scan_id": context.scan_id,
                "image_id": capture_response.image_id,
                "duration_ms": capture_duration_ms,
                "capture_use_extra": capture_use_extra,
                "service": "camera",
            },
        )
        return capture_response

    def _normalize_capture_images(self, *, capture_response: CameraCaptureResponse) -> list[CameraCaptureImage]:
        if capture_response.images is None:
            return [
                CameraCaptureImage(
                    index=0,
                    image_id=capture_response.image_id,
                    image_url_or_path=capture_response.image_url_or_path,
                )
            ]

        if not capture_response.images:
            raise ServiceValidationError("camera-service", "Camera response contains empty images[]")

        main_image = capture_response.images[0]
        if main_image.index != 0:
            raise ServiceValidationError("camera-service", "Camera response images[0].index must be 0")
        if (
            main_image.image_id != capture_response.image_id
            or main_image.image_url_or_path != capture_response.image_url_or_path
        ):
            raise ServiceValidationError(
                "camera-service",
                "Camera response contract violation: top-level image fields must match images[0]",
            )
        return capture_response.images

    async def _download_single_image(
        self,
        *,
        image: CameraCaptureImage,
    ) -> DownloadedImage | ImageDownloadFailure:
        async with self._image_download_semaphore:
            try:
                image_bytes = await self._camera_client.fetch_image_bytes(image.image_url_or_path)
                return DownloadedImage(image=image, image_bytes=image_bytes)
            except ServiceCallError as exc:
                return ImageDownloadFailure(image=image, error=exc)

    async def _download_captured_images(
        self,
        *,
        context: ScanTriggerContext,
        captured_images: list[CameraCaptureImage],
    ) -> list[DownloadedImage] | None:
        tasks = [
            asyncio.create_task(self._download_single_image(image=image))
            for image in captured_images
        ]
        results = await asyncio.gather(*tasks)

        downloaded_images: list[DownloadedImage] = []
        for result in results:
            if isinstance(result, ImageDownloadFailure):
                await self._journal.write_event(
                    "service_call_failed",
                    session_id=context.session_id,
                    scan_id=context.scan_id,
                    frame_id=result.image.index,
                    image_id=result.image.image_id,
                    service=result.error.service,
                    http_status=result.error.status_code,
                    error=str(result.error),
                )
                if result.image.index == 0:
                    self._logger.error(
                        "Main image download failed; aborting scan",
                        extra={
                            "session_id": context.session_id,
                            "scan_id": context.scan_id,
                            "frame_id": result.image.index,
                            "image_id": result.image.image_id,
                            "error": str(result.error),
                            "service": "camera",
                        },
                    )
                    return None

                self._logger.warning(
                    "Extra image download failed; continuing scan",
                    extra={
                        "session_id": context.session_id,
                        "scan_id": context.scan_id,
                        "frame_id": result.image.index,
                        "image_id": result.image.image_id,
                        "error": str(result.error),
                        "service": "camera",
                    },
                )
                continue

            downloaded_images.append(result)

        self._logger.info(
            "Image downloads completed",
            extra={
                "session_id": context.session_id,
                "scan_id": context.scan_id,
                "number_of_images_returned": len(captured_images),
                "number_of_images_downloaded": len(downloaded_images),
                "service": "camera",
            },
        )
        return downloaded_images

    async def _detect_downloaded_images(
        self,
        *,
        context: ScanTriggerContext,
        downloaded_images: list[DownloadedImage],
    ) -> list[DetectedImage]:
        tasks = [
            asyncio.create_task(
                self._detect_single_image_with_limit(
                    context=context,
                    image=item.image,
                    image_bytes=item.image_bytes,
                )
            )
            for item in downloaded_images
        ]
        return await asyncio.gather(*tasks)

    async def _detect_single_image_with_limit(
        self,
        *,
        context: ScanTriggerContext,
        image: CameraCaptureImage,
        image_bytes: bytes,
    ) -> DetectedImage:
        async with self._image_pipeline_semaphore:
            return await self._detect_single_image(
                context=context,
                image=image,
                image_bytes=image_bytes,
            )

    async def _detect_single_image(
        self,
        *,
        context: ScanTriggerContext,
        image: CameraCaptureImage,
        image_bytes: bytes,
    ) -> DetectedImage:
        frame_id = image.index
        image_id = image.image_id
        try:
            fruit_response = await self._detect_fruits_with_optional_fallback(
                context=context,
                frame_id=frame_id,
                image_id=image_id,
                image_bytes=image_bytes,
                weight_grams=context.weight_grams,
            )
        except (ServiceCallError, ServiceValidationError) as exc:
            event_type = "validation_failed" if isinstance(exc, ServiceValidationError) else "service_call_failed"
            await self._journal.write_event(
                event_type,
                session_id=context.session_id,
                scan_id=context.scan_id,
                frame_id=frame_id,
                image_id=image_id,
                service=getattr(exc, "service", "fruit-detector"),
                error=str(exc),
            )
            return DetectedImage(
                image=image,
                image_bytes=image_bytes,
                valid_detections=[],
                frame_error=str(exc),
            )

        valid_detections, dropped = filter_detections_by_confidence(
            detections=fruit_response.detections,
            image_id=image_id,
            allowed_classes=set(self._settings.ALLOWED_FRUIT_CLASSES),
            class_thresholds=self._settings.CLASS_CONFIDENCE_THRESHOLDS,
            default_threshold=self._settings.DEFAULT_CLASS_CONFIDENCE_THRESHOLD,
        )
        await self._log_dropped_detections(
            context=context,
            frame_id=frame_id,
            dropped=dropped,
        )
        return DetectedImage(
            image=image,
            image_bytes=image_bytes,
            valid_detections=valid_detections,
        )

    async def _process_single_image(
        self,
        *,
        context: ScanTriggerContext,
        frame_id: int,
        image_id: str,
        image_bytes: bytes,
    ) -> FrameEvidence:
        """Compatibility helper used by unit tests."""
        detected = await self._detect_single_image(
            context=context,
            image=CameraCaptureImage(index=frame_id, image_id=image_id, image_url_or_path=f"/images/{image_id}.jpg"),
            image_bytes=image_bytes,
        )
        if detected.frame_error is not None:
            return FrameEvidence(
                frame_id=frame_id,
                image_id=image_id,
                frame_error=detected.frame_error,
            )
        fruits = await self._collect_defect_evidence(
            context=context,
            frame_id=frame_id,
            image_id=image_id,
            image_bytes=image_bytes,
            detections=detected.valid_detections,
        )
        return FrameEvidence(
            frame_id=frame_id,
            image_id=image_id,
            fruits=fruits,
        )

    async def _log_dropped_detections(
        self,
        *,
        context: ScanTriggerContext,
        frame_id: int,
        dropped: list[DroppedDetection],
    ) -> None:
        for item in dropped:
            self._logger.info(
                "Fruit dropped due to confidence/class filter",
                extra={
                    "session_id": context.session_id,
                    "scan_id": context.scan_id,
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
                frame_id=frame_id,
                image_id=item.image_id,
                fruit_id=item.fruit_id,
                fruit_class=item.fruit_class,
                confidence=item.confidence,
                threshold=item.threshold,
                drop_reason=item.reason,
            )

    def _build_detection_only_fruits(self, *, detections: list[FruitDetection]) -> list[FruitEvidence]:
        return [
            FruitEvidence(
                source_fruit_id=detection.fruit_id,
                fruit_class=detection.fruit_class,
                confidence=detection.confidence,
                bbox=BBox.from_xyxy(detection.bbox),
                defects=[],
            )
            for detection in detections
        ]

    def _build_detection_only_frame_evidences(
        self,
        *,
        detected_images: list[DetectedImage],
    ) -> list[FrameEvidence]:
        ordered = sorted(detected_images, key=lambda item: item.image.index)
        return [
            FrameEvidence(
                frame_id=item.image.index,
                image_id=item.image.image_id,
                fruits=self._build_detection_only_fruits(detections=item.valid_detections),
                frame_error=item.frame_error,
            )
            for item in ordered
        ]

    def _select_representative_image(
        self,
        *,
        detected_images: list[DetectedImage],
    ) -> RepresentativeSelection:
        ordered = sorted(detected_images, key=lambda item: item.image.index)
        main = next((item for item in ordered if item.image.index == 0), None)
        main_has_detection = main is not None and bool(main.valid_detections)
        if main_has_detection and main is not None:
            return RepresentativeSelection(
                representative=main,
                main_has_detection=True,
                reason="main_detected",
            )

        for item in ordered:
            if item.image.index > 0 and item.valid_detections:
                return RepresentativeSelection(
                    representative=item,
                    main_has_detection=False,
                    reason="fallback_to_extra",
                )
        return RepresentativeSelection(
            representative=None,
            main_has_detection=False,
            reason="no_detections",
        )

    def _maybe_adjust_representative_for_aggregated_classes(
        self,
        *,
        detected_images: list[DetectedImage],
        selection: RepresentativeSelection,
        aggregated: AggregatedScan,
    ) -> RepresentativeSelection:
        representative = selection.representative
        if representative is None or not aggregated.fruits:
            return selection

        representative_classes = {detection.fruit_class for detection in representative.valid_detections}
        missing_classes = [
            fruit.fruit_class
            for fruit in aggregated.fruits
            if fruit.fruit_class not in representative_classes
        ]
        if not missing_classes:
            return selection

        for item in sorted(detected_images, key=lambda image_item: image_item.image.index):
            if not item.valid_detections:
                continue
            classes_in_item = {detection.fruit_class for detection in item.valid_detections}
            if any(missing_class in classes_in_item for missing_class in missing_classes):
                if item.image.index != representative.image.index:
                    return RepresentativeSelection(
                        representative=item,
                        main_has_detection=selection.main_has_detection,
                        reason="fallback_to_extra",
                    )
                break
        return selection

    async def _build_frame_evidences_with_representative(
        self,
        *,
        context: ScanTriggerContext,
        detected_images: list[DetectedImage],
        selection: RepresentativeSelection,
    ) -> list[FrameEvidence]:
        ordered = sorted(detected_images, key=lambda item: item.image.index)
        representative_index = (
            selection.representative.image.index if selection.representative is not None else None
        )
        frame_evidences: list[FrameEvidence] = []
        for item in ordered:
            if representative_index is not None and item.image.index == representative_index:
                fruits = await self._collect_defect_evidence(
                    context=context,
                    frame_id=item.image.index,
                    image_id=item.image.image_id,
                    image_bytes=item.image_bytes,
                    detections=item.valid_detections,
                )
            else:
                fruits = self._build_detection_only_fruits(detections=item.valid_detections)
            frame_evidences.append(
                FrameEvidence(
                    frame_id=item.image.index,
                    image_id=item.image.image_id,
                    fruits=fruits,
                    frame_error=item.frame_error,
                )
            )
        return frame_evidences

    def _find_representative_frame(
        self,
        *,
        frame_evidences: list[FrameEvidence],
        representative: DetectedImage,
    ) -> FrameEvidence | None:
        for frame in frame_evidences:
            if frame.frame_id == representative.image.index and frame.image_id == representative.image.image_id:
                return frame
        return None

    def _enforce_representative_geometry(
        self,
        *,
        context: ScanTriggerContext,
        result: ScanResult,
        representative_frame: FrameEvidence,
    ) -> ScanResult:
        representative_by_class: dict[str, list[FruitEvidence]] = defaultdict(list)
        for fruit in representative_frame.fruits:
            representative_by_class[fruit.fruit_class].append(fruit)
        for fruit_list in representative_by_class.values():
            fruit_list.sort(key=lambda item: (item.bbox.x_min, item.bbox.y_min))

        consumed_per_class: dict[str, int] = defaultdict(int)
        aligned_fruits = []
        for fruit in result.fruits:
            class_name = fruit.fruit_class
            class_index = consumed_per_class[class_name]
            candidates = representative_by_class.get(class_name, [])
            if class_index >= len(candidates):
                self._logger.warning(
                    "Dropping aggregated fruit without representative geometry",
                    extra={
                        "session_id": context.session_id,
                        "scan_id": context.scan_id,
                        "image_id": result.image_id,
                        "fruit_id": fruit.fruit_id,
                        "fruit_class": class_name,
                        "representative_image_id": representative_frame.image_id,
                    },
                )
                continue

            representative_fruit = candidates[class_index]
            consumed_per_class[class_name] += 1
            aligned_fruits.append(
                fruit.model_copy(
                    update={
                        "bbox": representative_fruit.bbox,
                        "defects": representative_fruit.defects,
                    }
                )
            )
        return result.model_copy(update={"fruits": aligned_fruits})

    async def _detect_fruits_with_optional_fallback(
        self,
        *,
        context: ScanTriggerContext,
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
        frame_id: int,
        image_id: str,
        image_bytes: bytes,
        detections: list[FruitDetection],
    ) -> list[FruitEvidence]:
        tasks = [
            asyncio.create_task(
                self._detect_single_fruit_defects(
                    context=context,
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
