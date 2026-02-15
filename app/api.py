from __future__ import annotations

from fastapi import APIRouter, Request

from app.dependencies import BrainContainer
from app.models import WeightEvent, WeightIngressResponse

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/weight", response_model=WeightIngressResponse)
async def ingest_weight(event: WeightEvent, request: Request) -> WeightIngressResponse:
    container: BrainContainer = request.app.state.container
    return await container.handle_weight_event(event)
