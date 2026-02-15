from __future__ import annotations

from datetime import datetime, timezone

from app.config import Settings
from app.core.state_machine import WeightStateMachine
from app.models import MachineState, WeightEvent


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_idle_to_active_immediately_triggers_first_scan() -> None:
    settings = Settings(
        ENTER_ACTIVE_WEIGHT=30.0,
        EXIT_ACTIVE_WEIGHT=25.0,
        SIGNIFICANT_DELTA=20.0,
    )
    machine = WeightStateMachine(settings)

    decision = machine.handle_weight_event(WeightEvent(grams=30.0, timestamp=_ts(), seq=1))

    assert decision.state == MachineState.ACTIVE
    assert decision.triggered_scan is True
    assert decision.session_id is not None
    assert decision.scan_id is not None
    assert decision.reason == "entered_active_initial_scan"


def test_active_small_delta_does_not_trigger_scan() -> None:
    settings = Settings(
        ENTER_ACTIVE_WEIGHT=30.0,
        EXIT_ACTIVE_WEIGHT=25.0,
        SIGNIFICANT_DELTA=20.0,
    )
    machine = WeightStateMachine(settings)

    initial = machine.handle_weight_event(WeightEvent(grams=35.0, timestamp=_ts(), seq=1))
    followup = machine.handle_weight_event(WeightEvent(grams=40.0, timestamp=_ts(), seq=2))

    assert initial.triggered_scan is True
    assert followup.state == MachineState.ACTIVE
    assert followup.session_id == initial.session_id
    assert followup.triggered_scan is False
    assert followup.reason == "active_no_significant_change"


def test_active_significant_delta_triggers_scan_same_session() -> None:
    settings = Settings(
        ENTER_ACTIVE_WEIGHT=30.0,
        EXIT_ACTIVE_WEIGHT=25.0,
        SIGNIFICANT_DELTA=20.0,
    )
    machine = WeightStateMachine(settings)

    initial = machine.handle_weight_event(WeightEvent(grams=35.0, timestamp=_ts(), seq=1))
    second = machine.handle_weight_event(WeightEvent(grams=60.0, timestamp=_ts(), seq=2))

    assert initial.triggered_scan is True
    assert second.triggered_scan is True
    assert second.reason == "significant_delta"
    assert second.session_id == initial.session_id
    assert second.scan_id != initial.scan_id


def test_active_to_idle_when_below_exit_threshold() -> None:
    settings = Settings(
        ENTER_ACTIVE_WEIGHT=30.0,
        EXIT_ACTIVE_WEIGHT=25.0,
        SIGNIFICANT_DELTA=20.0,
    )
    machine = WeightStateMachine(settings)

    initial = machine.handle_weight_event(WeightEvent(grams=35.0, timestamp=_ts(), seq=1))
    closed = machine.handle_weight_event(WeightEvent(grams=10.0, timestamp=_ts(), seq=2))

    assert initial.state == MachineState.ACTIVE
    assert closed.state == MachineState.IDLE
    assert closed.close_session is True
    assert closed.triggered_scan is False
    assert closed.reason == "exited_active_below_exit_threshold"


def test_same_seq_events_are_still_processed_as_weight_snapshots() -> None:
    settings = Settings(
        ENTER_ACTIVE_WEIGHT=30.0,
        EXIT_ACTIVE_WEIGHT=25.0,
        SIGNIFICANT_DELTA=20.0,
    )
    machine = WeightStateMachine(settings)

    first = machine.handle_weight_event(WeightEvent(grams=35.0, timestamp=_ts(), source_id="s1", seq=1))
    second = machine.handle_weight_event(WeightEvent(grams=60.0, timestamp=_ts(), source_id="s1", seq=1))

    assert first.triggered_scan is True
    assert second.triggered_scan is True
    assert second.reason == "significant_delta"
