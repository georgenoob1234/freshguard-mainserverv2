from __future__ import annotations

from uuid import uuid4

from app.config import Settings
from app.models import MachineState, StateMachineDecision, StateTransition, WeightEvent


class WeightStateMachine:
    """Deterministic weight-driven state machine for session/scan triggering."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        self._state: MachineState = MachineState.IDLE
        self._session_id: str | None = None
        self._scan_counter: int = 0

        self._last_scan_weight: float | None = None

    @property
    def state(self) -> MachineState:
        return self._state

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def handle_weight_event(self, event: WeightEvent) -> StateMachineDecision:
        current_weight = event.grams

        if self._state == MachineState.IDLE:
            if current_weight >= self._settings.ENTER_ACTIVE_WEIGHT:
                transition = StateTransition(
                    from_state=MachineState.IDLE,
                    to_state=MachineState.ACTIVE,
                )
                self._state = MachineState.ACTIVE
                self._session_id = str(uuid4())
                self._scan_counter = 0
                scan_id = self._next_scan_id()
                self._last_scan_weight = current_weight
                return StateMachineDecision(
                    state=self._state,
                    session_id=self._session_id,
                    scan_id=scan_id,
                    triggered_scan=True,
                    reason="entered_active_initial_scan",
                    transition=transition,
                    stable_weight=current_weight,
                )

            return StateMachineDecision(
                state=self._state,
                session_id=self._session_id,
                reason="idle_below_enter_threshold",
                stable_weight=current_weight,
            )

        # ACTIVE state logic
        if current_weight < self._settings.EXIT_ACTIVE_WEIGHT:
            closing_session_id = self._session_id
            transition = StateTransition(
                from_state=MachineState.ACTIVE,
                to_state=MachineState.IDLE,
            )
            self._state = MachineState.IDLE
            self._session_id = None
            self._scan_counter = 0
            self._last_scan_weight = None
            return StateMachineDecision(
                state=self._state,
                session_id=closing_session_id,
                triggered_scan=False,
                reason="exited_active_below_exit_threshold",
                transition=transition,
                stable_weight=current_weight,
                close_session=True,
            )

        if self._last_scan_weight is None:
            self._last_scan_weight = current_weight

        delta = abs(current_weight - self._last_scan_weight)
        if delta >= self._settings.SIGNIFICANT_DELTA:
            scan_id = self._next_scan_id()
            self._last_scan_weight = current_weight
            return StateMachineDecision(
                state=self._state,
                session_id=self._session_id,
                scan_id=scan_id,
                triggered_scan=True,
                reason="significant_delta",
                stable_weight=current_weight,
            )

        return StateMachineDecision(
            state=self._state,
            session_id=self._session_id,
            reason="active_no_significant_change",
            stable_weight=current_weight,
        )

    def _next_scan_id(self) -> str:
        if self._session_id is None:
            raise RuntimeError("Cannot allocate scan_id without active session_id")
        self._scan_counter += 1
        return f"{self._session_id}-{self._scan_counter:04d}"
