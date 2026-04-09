# syntax=docker/dockerfile:1.7
# Brain / MainServer v2 — CPU-only FastAPI orchestrator (binds 0.0.0.0).

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=8000

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

RUN mkdir -p /app/data/journal && chown -R appuser:appuser /app/data

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import os,urllib.request; p=os.environ.get('SERVICE_PORT','8000'); urllib.request.urlopen('http://127.0.0.1:%s/healthz' % p, timeout=3)"

CMD ["sh", "-c", "uvicorn app.main:app --host ${SERVICE_HOST} --port ${SERVICE_PORT}"]
