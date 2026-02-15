from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config import Settings
from app.core.orchestrator import ScanOrchestrator
from app.core.state_machine import WeightStateMachine
from app.journal import EventJournal
from app.logging import get_logger
from app.models import ScanTriggerContext, WeightEvent, WeightIngressResponse
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

    async def handle_weight_event(self, event: WeightEvent) -> WeightIngressResponse:
        logger = get_logger(self.logger_name)
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
        if self._scan_tasks:
            await asyncio.gather(*self._scan_tasks, return_exceptions=True)
        await self.orchestrator.close()


def create_container(settings: Settings) -> BrainContainer:
    journal = EventJournal(settings.JOURNAL_PATH)
    state_machine = WeightStateMachine(settings)

    camera_client = CameraServiceClient(
        service_name="camera-service",
        base_url=settings.CAMERA_SERVICE_URL,
        timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
        retries=settings.HTTP_RETRIES,
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
