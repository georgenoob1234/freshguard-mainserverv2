from __future__ import annotations

from fastapi import FastAPI

from app.api import router
from app.config import get_settings
from app.dependencies import create_container
from app.logging import setup_logging


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    app = FastAPI(title="Brain Service", version="2.0.0")
    app.include_router(router)

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.container = create_container(settings)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.container.shutdown()

    return app


app = create_app()
