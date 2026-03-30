from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.config import Settings
from app.core.duplicate_guard import DuplicateResultGuard
from app.dependencies import BrainContainer
from app.journal import EventJournal
from app.models import (
    BBox,
    MachineState,
    ScanFruit,
    ScanResult,
    ScanTriggerContext,
    StateMachineDecision,
    WeightEvent,
)


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def _event_types(journal_path: Path) -> list[str]:
    if not journal_path.exists():
        return []
    lines = [line for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line)["event_type"] for line in lines]


class _FailStateMachine:
    def handle_weight_event(self, event: WeightEvent) -> StateMachineDecision:  # noqa: ARG002
        raise AssertionError("State machine must not be called in shelf mode")


class _StaticStateMachine:
    def __init__(self, decision: StateMachineDecision) -> None:
        self._decision = decision
        self.called = False

    def handle_weight_event(self, event: WeightEvent) -> StateMachineDecision:  # noqa: ARG002
        self.called = True
        return self._decision


class _StubOrchestrator:
    def __init__(self, *, scan_delay_seconds: float = 0.0) -> None:
        self.scan_delay_seconds = scan_delay_seconds
        self.contexts: list[ScanTriggerContext] = []
        self.closed = False
        self.max_concurrent = 0
        self._inflight = 0

    async def run_scan(self, context: ScanTriggerContext) -> None:
        self.contexts.append(context)
        self._inflight += 1
        self.max_concurrent = max(self.max_concurrent, self._inflight)
        try:
            if self.scan_delay_seconds > 0:
                await asyncio.sleep(self.scan_delay_seconds)
        finally:
            self._inflight -= 1

    async def close(self) -> None:
        self.closed = True


def _make_shelf_container(
    *,
    tmp_path: Path,
    interval_seconds: float = 0.02,
    scan_delay_seconds: float = 0.0,
) -> tuple[BrainContainer, _StubOrchestrator, Path]:
    journal_path = tmp_path / "events.jsonl"
    settings = Settings(
        OPERATING_MODE="shelf",
        SHELF_SCAN_INTERVAL_SECONDS=interval_seconds,
        SHELF_PUBLISH_WEIGHT_GRAMS=0.0,
        JOURNAL_PATH=journal_path,
    )
    orchestrator = _StubOrchestrator(scan_delay_seconds=scan_delay_seconds)
    container = BrainContainer(
        settings=settings,
        journal=EventJournal(journal_path),
        state_machine=_FailStateMachine(),
        orchestrator=orchestrator,
    )
    return container, orchestrator, journal_path


@pytest.mark.asyncio
async def test_scale_mode_weight_path_still_uses_state_machine(tmp_path: Path) -> None:
    journal_path = tmp_path / "events.jsonl"
    settings = Settings(
        OPERATING_MODE="scale",
        JOURNAL_PATH=journal_path,
    )
    decision = StateMachineDecision(
        state=MachineState.IDLE,
        triggered_scan=False,
        reason="idle_below_enter_threshold",
        stable_weight=5.0,
    )
    state_machine = _StaticStateMachine(decision)
    orchestrator = _StubOrchestrator()
    container = BrainContainer(
        settings=settings,
        journal=EventJournal(journal_path),
        state_machine=state_machine,
        orchestrator=orchestrator,
    )

    response = await container.handle_weight_event(WeightEvent(grams=5.0, timestamp=_ts(), seq=1))

    assert state_machine.called is True
    assert response.reason == "idle_below_enter_threshold"
    assert response.triggered_scan is False
    await container.shutdown()
    assert orchestrator.closed is True


@pytest.mark.asyncio
async def test_shelf_mode_starts_periodic_loop_on_startup(tmp_path: Path) -> None:
    container, orchestrator, journal_path = _make_shelf_container(tmp_path=tmp_path)

    await container.start()
    await asyncio.sleep(0.07)
    await container.shutdown()

    assert len(orchestrator.contexts) >= 1
    assert all(context.trigger_reason == "interval" for context in orchestrator.contexts)
    assert all(context.operating_mode == "shelf" for context in orchestrator.contexts)
    assert "shelf_loop_started" in _event_types(journal_path)
    assert "shelf_loop_stopped" in _event_types(journal_path)


@pytest.mark.asyncio
async def test_shelf_mode_triggers_scans_without_weight_events(tmp_path: Path) -> None:
    container, orchestrator, _ = _make_shelf_container(tmp_path=tmp_path, interval_seconds=0.02)

    await container.start()
    await asyncio.sleep(0.06)
    await container.shutdown()

    assert len(orchestrator.contexts) >= 1
    assert all(context.weight_grams == 0.0 for context in orchestrator.contexts)


@pytest.mark.asyncio
async def test_shelf_mode_skips_ticks_while_scan_in_progress(tmp_path: Path) -> None:
    container, orchestrator, journal_path = _make_shelf_container(
        tmp_path=tmp_path,
        interval_seconds=0.02,
        scan_delay_seconds=0.08,
    )

    await container.start()
    await asyncio.sleep(0.14)
    await container.shutdown()

    assert orchestrator.max_concurrent == 1
    assert "shelf_scan_tick_skipped" in _event_types(journal_path)


@pytest.mark.asyncio
async def test_shelf_mode_shutdown_stops_new_interval_scans(tmp_path: Path) -> None:
    container, orchestrator, _ = _make_shelf_container(
        tmp_path=tmp_path,
        interval_seconds=0.02,
        scan_delay_seconds=0.03,
    )

    await container.start()
    await asyncio.sleep(0.07)
    await container.shutdown()
    scans_before_wait = len(orchestrator.contexts)
    await asyncio.sleep(0.06)

    assert len(orchestrator.contexts) == scans_before_wait
    assert container._shelf_task is None
    assert container._shelf_scan_task is None


@pytest.mark.asyncio
async def test_weight_events_are_ignored_in_shelf_mode(tmp_path: Path) -> None:
    container, orchestrator, journal_path = _make_shelf_container(tmp_path=tmp_path)

    response = await container.handle_weight_event(WeightEvent(grams=42.0, timestamp=_ts(), seq=1))
    await container.shutdown()

    assert response.state == MachineState.IDLE
    assert response.triggered_scan is False
    assert response.reason == "shelf_mode_weight_ignored"
    assert orchestrator.contexts == []
    assert "weight_ignored_shelf_mode" in _event_types(journal_path)


def test_duplicate_suppression_still_applies_for_shelf_weight() -> None:
    guard = DuplicateResultGuard(window_ms=2000, weight_bucket_grams=5.0)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = ScanResult(
        session_id="shelf-session",
        image_id="img-1",
        weight_grams=0.0,
        fruits=[
            ScanFruit(
                fruit_id="fruit-1",
                fruit_class="apple",
                confidence=0.9,
                bbox=BBox(x_min=1, y_min=1, x_max=10, y_max=10),
                defects=[],
            )
        ],
    )

    suppressed_first, _ = guard.should_suppress(result, now)
    suppressed_second, _ = guard.should_suppress(result, now)

    assert suppressed_first is False
    assert suppressed_second is True
