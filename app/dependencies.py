from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from app.config import Settings, max_fruit_imgsz_as_capture_resolution
from app.core.orchestrator import ScanOrchestrator
from app.core.state_machine import WeightStateMachine
from app.journal import EventJournal
from app.logging import get_logger
from app.models import MachineState, ScanTriggerContext, WeightEvent, WeightIngressResponse
from app.services.clients import (
    CameraServiceClient,
    DefectDetectorClient,
    FruitDetectorClient,
    PublisherClient,
)


@dataclass
class BrainContainer:
    settings: Settings
    journal: EventJournal
    state_machine: WeightStateMachine
    orchestrator: ScanOrchestrator
    logger_name: str = "brain.container"
    _scan_tasks: set[asyncio.Task] = field(default_factory=set)
    _shelf_task: asyncio.Task | None = None
    _shelf_scan_task: asyncio.Task | None = None
    _shelf_stop_event: asyncio.Event | None = None
    _shelf_tick_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _shelf_session_id: str | None = None
    _shelf_scan_counter: int = 0

    def _is_shelf_mode(self) -> bool:
        return self.settings.OPERATING_MODE == "shelf"

    async def start(self) -> None:
        if not self._is_shelf_mode() or self._shelf_task is not None:
            return

        logger = get_logger(self.logger_name)
        self._shelf_stop_event = asyncio.Event()
        self._shelf_session_id = str(uuid4())
        self._shelf_scan_counter = 0
        self._shelf_scan_task = None
        await self.journal.write_event(
            "shelf_loop_started",
            session_id=self._shelf_session_id,
            operating_mode="shelf",
            interval_seconds=self.settings.SHELF_SCAN_INTERVAL_SECONDS,
            source_id=self.settings.SHELF_SOURCE_ID,
        )
        logger.info(
            "Shelf loop started",
            extra={
                "session_id": self._shelf_session_id,
                "operating_mode": "shelf",
                "interval_seconds": self.settings.SHELF_SCAN_INTERVAL_SECONDS,
                "source_id": self.settings.SHELF_SOURCE_ID,
            },
        )
        self._shelf_task = asyncio.create_task(self._run_shelf_loop())

    async def handle_weight_event(self, event: WeightEvent) -> WeightIngressResponse:
        logger = get_logger(self.logger_name)
        if self._is_shelf_mode():
            await self.journal.write_event(
                "weight_ignored_shelf_mode",
                grams=event.grams,
                source_id=event.source_id,
                seq=event.seq,
                operating_mode="shelf",
            )
            logger.info(
                "Ignored weight event in shelf mode",
                extra={
                    "grams": event.grams,
                    "source_id": event.source_id,
                    "seq": event.seq,
                    "operating_mode": "shelf",
                },
            )
            return WeightIngressResponse(
                state=MachineState.IDLE,
                session_id=None,
                scan_id=None,
                triggered_scan=False,
                reason="shelf_mode_weight_ignored",
            )

        decision = self.state_machine.handle_weight_event(event)

        await self.journal.write_event(
            "weight_event_received",
            session_id=decision.session_id,
            scan_id=decision.scan_id,
            grams=event.grams,
            source_id=event.source_id,
            seq=event.seq,
            state=decision.state.value,
            reason=decision.reason,
        )

        if decision.transition is not None:
            await self.journal.write_event(
                "state_transition",
                session_id=decision.session_id,
                scan_id=decision.scan_id,
                from_state=decision.transition.from_state.value,
                to_state=decision.transition.to_state.value,
                reason=decision.reason,
            )

        if decision.triggered_scan and decision.session_id and decision.scan_id:
            trigger_context = ScanTriggerContext(
                session_id=decision.session_id,
                scan_id=decision.scan_id,
                weight_grams=decision.stable_weight if decision.stable_weight is not None else event.grams,
                trigger_reason=decision.reason,
                operating_mode="scale",
                triggered_at=datetime.now(timezone.utc),
            )
            task = asyncio.create_task(self.orchestrator.run_scan(trigger_context))
            self._scan_tasks.add(task)
            task.add_done_callback(self._scan_tasks.discard)
            logger.info(
                "Scheduled scan task",
                extra={
                    "session_id": decision.session_id,
                    "scan_id": decision.scan_id,
                    "reason": decision.reason,
                },
            )

        return WeightIngressResponse(
            state=decision.state,
            session_id=decision.session_id,
            scan_id=decision.scan_id,
            triggered_scan=decision.triggered_scan,
            reason=decision.reason,
        )

    async def shutdown(self) -> None:
        if self._shelf_stop_event is not None:
            self._shelf_stop_event.set()
        if self._shelf_task is not None:
            await asyncio.gather(self._shelf_task, return_exceptions=True)
            self._shelf_task = None
        if self._shelf_scan_task is not None:
            await asyncio.gather(self._shelf_scan_task, return_exceptions=True)
            self._shelf_scan_task = None
        self._shelf_stop_event = None
        self._shelf_session_id = None

        if self._scan_tasks:
            await asyncio.gather(*self._scan_tasks, return_exceptions=True)
        await self.orchestrator.close()

    async def _run_shelf_loop(self) -> None:
        logger = get_logger(self.logger_name)
        stop_event = self._shelf_stop_event
        session_id = self._shelf_session_id
        if stop_event is None or session_id is None:
            return

        try:
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=self.settings.SHELF_SCAN_INTERVAL_SECONDS,
                    )
                    break
                except asyncio.TimeoutError:
                    pass

                tick_at = datetime.now(timezone.utc)
                await self.journal.write_event(
                    "shelf_interval_tick",
                    session_id=session_id,
                    operating_mode="shelf",
                    trigger_reason="interval",
                    source_id=self.settings.SHELF_SOURCE_ID,
                )
                logger.info(
                    "Shelf interval tick",
                    extra={
                        "session_id": session_id,
                        "operating_mode": "shelf",
                        "trigger_reason": "interval",
                        "source_id": self.settings.SHELF_SOURCE_ID,
                    },
                )

                if self._shelf_scan_task is not None and not self._shelf_scan_task.done():
                    await self.journal.write_event(
                        "shelf_scan_tick_skipped",
                        session_id=session_id,
                        operating_mode="shelf",
                        trigger_reason="interval",
                        reason="scan_in_progress",
                    )
                    logger.info(
                        "Skipped shelf interval tick because scan is still running",
                        extra={
                            "session_id": session_id,
                            "operating_mode": "shelf",
                            "trigger_reason": "interval",
                        },
                    )
                    continue

                scan_id = self._next_shelf_scan_id()
                context = ScanTriggerContext(
                    session_id=session_id,
                    scan_id=scan_id,
                    weight_grams=self.settings.SHELF_PUBLISH_WEIGHT_GRAMS,
                    trigger_reason="interval",
                    operating_mode="shelf",
                    triggered_at=tick_at,
                )
                await self.journal.write_event(
                    "shelf_scan_started",
                    session_id=session_id,
                    scan_id=scan_id,
                    operating_mode="shelf",
                    trigger_reason="interval",
                    source_id=self.settings.SHELF_SOURCE_ID,
                    weight_grams=self.settings.SHELF_PUBLISH_WEIGHT_GRAMS,
                )
                logger.info(
                    "Shelf scan started from interval",
                    extra={
                        "session_id": session_id,
                        "scan_id": scan_id,
                        "operating_mode": "shelf",
                        "trigger_reason": "interval",
                        "weight_grams": self.settings.SHELF_PUBLISH_WEIGHT_GRAMS,
                    },
                )
                self._shelf_scan_task = asyncio.create_task(self._run_shelf_scan(context))
        finally:
            await self.journal.write_event(
                "shelf_loop_stopped",
                session_id=session_id,
                operating_mode="shelf",
            )
            logger.info(
                "Shelf loop stopped",
                extra={
                    "session_id": session_id,
                    "operating_mode": "shelf",
                },
            )

    def _next_shelf_scan_id(self) -> str:
        if self._shelf_session_id is None:
            raise RuntimeError("Shelf session is not initialized")
        self._shelf_scan_counter += 1
        return f"{self._shelf_session_id}-{self._shelf_scan_counter:04d}"

    async def _run_shelf_scan(self, context: ScanTriggerContext) -> None:
        logger = get_logger(self.logger_name)
        try:
            async with self._shelf_tick_lock:
                await self.orchestrator.run_scan(context)
        except Exception:
            logger.exception(
                "Shelf scan task failed",
                extra={
                    "session_id": context.session_id,
                    "scan_id": context.scan_id,
                    "operating_mode": "shelf",
                    "trigger_reason": context.trigger_reason,
                },
            )
        finally:
            task = asyncio.current_task()
            if self._shelf_scan_task is task:
                self._shelf_scan_task = None


def create_container(settings: Settings) -> BrainContainer:
    journal = EventJournal(settings.JOURNAL_PATH)
    state_machine = WeightStateMachine(settings)

    camera_client = CameraServiceClient(
        service_name="camera-service",
        base_url=settings.CAMERA_SERVICE_URL,
        timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
        retries=settings.HTTP_RETRIES,
        capture_resolution=max_fruit_imgsz_as_capture_resolution(
            settings.FRUIT_PRIMARY_IMGSZ,
            settings.FRUIT_FALLBACK_IMGSZ,
        ),
        capture_format=settings.CAMERA_CAPTURE_FORMAT,
        capture_quality=settings.CAMERA_CAPTURE_QUALITY,
    )
    fruit_client = FruitDetectorClient(
        service_name="fruit-detector",
        base_url=settings.FRUIT_DETECTOR_URL,
        timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
        retries=settings.HTTP_RETRIES,
        detect_path=settings.FRUIT_DETECT_PATH,
    )
    defect_client = DefectDetectorClient(
        service_name="defect-detector",
        base_url=settings.DEFECT_DETECTOR_URL,
        timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
        retries=settings.HTTP_RETRIES,
    )
    ui_publisher = PublisherClient(
        service_name="ui-service",
        base_url=settings.UI_SERVICE_URL,
        timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
        retries=settings.HTTP_RETRIES,
    )
    main_publisher = PublisherClient(
        service_name="main-server",
        base_url=settings.MAIN_SERVER_URL,
        timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
        retries=settings.HTTP_RETRIES,
    )

    orchestrator = ScanOrchestrator(
        settings=settings,
        journal=journal,
        camera_client=camera_client,
        fruit_client=fruit_client,
        defect_client=defect_client,
        ui_publisher=ui_publisher,
        main_publisher=main_publisher,
    )
    return BrainContainer(
        settings=settings,
        journal=journal,
        state_machine=state_machine,
        orchestrator=orchestrator,
    )
