from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventJournal:
    """Append-only JSONL event journal."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def write_event(
        self,
        event_type: str,
        *,
        session_id: str | None = None,
        scan_id: str | None = None,
        **context: Any,
    ) -> None:
        event: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "session_id": session_id,
            "scan_id": scan_id,
            **context,
        }

        line = json.dumps(event, ensure_ascii=True, separators=(",", ":"))
        async with self._lock:
            with self._path.open("a", encoding="utf-8") as file_handle:
                file_handle.write(f"{line}\n")
