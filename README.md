# Brain Service (MainServer v2)

Production-oriented Brain orchestrator for a modular smart-scale computer vision system.

The Brain is the only active orchestrator. It accepts pushed weight events, decides when to scan, coordinates camera/detector calls, aggregates multi-frame evidence, suppresses duplicate outputs, and publishes final scan results.

## Scope

This repository implements the **Brain/MainServer only**.

External services are treated as passive dependencies and are not implemented here:
- Camera service
- Fruit detector service
- Defect detector service
- UI service
- Main online server
- Weight producer (future push sender)

## Implemented Features

- `POST /weight` push ingress for weight events (no polling path implemented)
- Deterministic weight state machine with explicit `IDLE <-> ACTIVE` transitions
- Session and scan identity model:
  - `session_id`: lifecycle from fruit placement to fruit removal
  - `scan_id`: unique per scan within a session
- Multi-frame scan support with per-camera-role configuration:
  - main camera frame count and interval
  - auxiliary camera frame count and interval
- Bounded fruit-detection fallback from `imgsz=320` to `imgsz=416`
- Class whitelist + class-specific confidence filtering before cropping
- Parallel defect detection with configurable semaphore limits
- Aggregation layer for frame evidence -> single final scan result
- Anti-duplicate suppression via stable result hashing + time window
- Structured runtime JSON logs to stdout
- Append-only JSONL event journal for analytics and auditing
- Strict Pydantic validation for inbound/outbound contracts
- Pytest coverage for state machine, aggregation, duplicate guard, and confidence filtering

## High-Level Flow

1. Weight producer sends `POST /weight`.
2. Brain validates and forwards event into state machine.
3. State machine may trigger scan when:
   - event weight enters active range (`grams >= ENTER_ACTIVE_WEIGHT`), or
   - active-session event delta meets `SIGNIFICANT_DELTA`.
4. For each configured camera and frame:
   - capture image (`POST /capture`)
   - fetch bytes from camera image path (`GET /api/images/...`)
   - fruit detection (`POST /detect-fruits?imgsz=320` by default)
   - optional fallback (`POST /detect-fruits?imgsz=416`) once
   - filter detections by allowed classes and class-specific thresholds
   - crop fruits and call defect detection in parallel (`POST /detect-defects`)
5. Aggregator combines all frame evidence into one deterministic result.
6. Duplicate guard suppresses repeated equivalent outputs in the configured window.
7. Brain publishes result to UI and main server endpoints.
8. Runtime logs + journal events are written throughout.

## API

### `GET /healthz`

Simple liveness endpoint.

Response:

```json
{"status":"ok"}
```

### `POST /weight`

Receives pushed weight samples from a weight producer.

Request body:

```json
{
  "grams": 145.2,
  "timestamp": "2026-02-12T10:15:23.123456+00:00",
  "source_id": "scale-1",
  "seq": 1242
}
```

Fields:
- `grams` (float, required, `>= 0`)
- `timestamp` (ISO8601 datetime, required)
- `source_id` (string, optional)
- `seq` (int, optional metadata)

Response body:

```json
{
  "status": "accepted",
  "state": "ACTIVE",
  "session_id": "0ea6a40f-f7d9-48f7-bf1f-29f04f9db73c",
  "scan_id": "0ea6a40f-f7d9-48f7-bf1f-29f04f9db73c-0001",
  "triggered_scan": true,
  "reason": "entered_active_initial_scan"
}
```

## External Service Contracts Used

### Camera
- `POST /capture` -> `image_id`, `image_url_or_path`, `timestamp`
- `GET {image_url_or_path}` -> image bytes

### Fruit Detector
- `POST /detect-fruits` (multipart `file`, query `imgsz`) by default
- Expected schema:
  - `image_id`, `width`, `height`, and detections list under `detections[]` or `fruits[]`
  - each detection: `fruit_id`, `class`, `confidence`, `bbox=[x1,y1,x2,y2]`

### Defect Detector
- `POST /detect-defects` (multipart `image`, form fields `image_id`, `fruit_id`)
- Expected schema:
  - `image_id`, `fruit_id`, `defects[]`

### Publishers
- UI publish target: `POST {UI_SERVICE_URL}{UI_PUBLISH_PATH}`
- Main server publish target: `POST {MAIN_SERVER_URL}{MAIN_SERVER_PUBLISH_PATH}`

## Configuration

Configuration is managed through environment variables (`.env` supported by `pydantic-settings`).

### Core service

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `dev` | Runtime environment |
| `LOG_LEVEL` | `INFO` | Logging level |
| `SERVICE_HOST` | `0.0.0.0` | Bind host |
| `SERVICE_PORT` | `8000` | Bind port |
| `WEIGHT_PUSH_PATH` | `/weight` | Weight ingress path constant |
| `ENABLE_WEIGHT_POLLING` | `false` | Reserved toggle; polling not implemented |

### State machine and triggers

| Variable | Default | Description |
|---|---|---|
| `ENTER_ACTIVE_WEIGHT` | `30.0` | IDLE -> ACTIVE threshold |
| `EXIT_ACTIVE_WEIGHT` | `25.0` | ACTIVE -> IDLE threshold (hysteresis) |
| `MIN_FRUIT_WEIGHT` | `30.0` | Fallback helper threshold in orchestration |
| `SIGNIFICANT_DELTA` | `20.0` | Grams delta threshold for re-scan |

### Service URLs and networking

| Variable | Default |
|---|---|
| `CAMERA_SERVICE_URL` | `http://localhost:8200` |
| `FRUIT_DETECTOR_URL` | `http://localhost:8300` |
| `FRUIT_DETECT_PATH` | `/detect-fruits` |
| `DEFECT_DETECTOR_URL` | `http://localhost:8400` |
| `UI_SERVICE_URL` | `http://localhost:8500` |
| `MAIN_SERVER_URL` | `http://localhost:8600` |
| `UI_PUBLISH_PATH` | `/update` |
| `MAIN_SERVER_PUBLISH_PATH` | `/update` |
| `HTTP_TIMEOUT_SECONDS` | `8.0` |
| `HTTP_RETRIES` | `1` |

### Detection filtering

| Variable | Default |
|---|---|
| `ALLOWED_FRUIT_CLASSES` | `["apple","banana","tomato"]` |
| `CLASS_CONFIDENCE_THRESHOLDS` | `{"apple":0.40,"banana":0.50,"tomato":0.45}` |
| `DEFAULT_CLASS_CONFIDENCE_THRESHOLD` | `0.50` |
| `FRUIT_PRIMARY_IMGSZ` | `320` |
| `FRUIT_FALLBACK_IMGSZ` | `416` |
| `FRUIT_LOW_CONFIDENCE_FALLBACK_THRESHOLD` | `0.30` |
| `FRUIT_TINY_BBOX_AREA_RATIO` | `0.005` |

### Multi-frame and cameras

| Variable | Default | Description |
|---|---|---|
| `MAIN_CAMERA_FRAMES` | `3` | Frames per scan for main camera |
| `MAIN_CAMERA_FRAME_INTERVAL_MS` | `150` | Delay between main frames |
| `AUX_CAMERA_FRAMES` | `1` | Frames per scan for aux cameras |
| `AUX_CAMERA_FRAME_INTERVAL_MS` | `150` | Delay between aux frames |
| `AGGREGATION_POLICY` | `vote` | `vote`, `average`, `best_frame_plus_vote` |
| `CAMERAS` | `[{"camera_id":"camera-main","role":"main"}]` | Camera list |
| `DEFECT_MAX_PARALLEL` | `6` | Max concurrent defect requests |

### Duplicate suppression and journal

| Variable | Default | Description |
|---|---|---|
| `ENABLE_DUPLICATE_SUPPRESSION` | `true` | Suppress repeated equivalent outputs |
| `DUPLICATE_SUPPRESSION_WINDOW_MS` | `3000` | Suppression window |
| `DUPLICATE_WEIGHT_BUCKET_GRAMS` | `5.0` | Weight bucket in result hash |
| `JOURNAL_PATH` | `data/journal/events.jsonl` | Append-only JSONL journal path |

## Event Journal

The journal is append-only JSONL (`one JSON object per line`), intended for analytics and auditing.

Primary event types currently emitted:
- `weight_event_received`
- `state_transition`
- `scan_triggered`
- `capture_started`, `capture_finished`
- `fruit_detect_started`, `fruit_detect_finished`
- `detection_dropped`
- `defect_detect_started`, `defect_detect_finished`
- `scan_aggregated`
- `scan_published_ui`
- `scan_published_main_server`
- `anti_duplicate_suppressed`
- `service_call_failed`
- `validation_failed`

Each event includes `ts`, `event_type`, `session_id`, `scan_id`, and contextual fields (for example: `camera_id`, `frame_id`, `image_id`, `fruit_id`, `duration_ms`, `http_status`).

## Runtime Logging

- JSON-structured logs to stdout
- Correlation fields included when available (`session_id`, `scan_id`, camera/frame/image/fruit context)
- Raw image bytes are never logged

## Project Layout

```text
app/
  api.py
  config.py
  dependencies.py
  journal.py
  logging.py
  main.py
  models.py
  core/
    aggregation.py
    duplicate_guard.py
    image_ops.py
    orchestrator.py
    state_machine.py
  services/
    clients.py
tests/
  test_aggregation.py
  test_confidence_filtering.py
  test_duplicate_guard.py
  test_state_machine.py
```

## Installation

### Option A: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: Conda Python (as used in this environment)

```bash
/opt/anaconda/bin/python -m pip install -r requirements.txt
```

## Running

```bash
/opt/anaconda/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/healthz
```

Example weight event:

```bash
curl -X POST "http://localhost:8000/weight" \
  -H "Content-Type: application/json" \
  -d '{
    "grams": 120.5,
    "timestamp": "2026-02-12T12:30:00+00:00",
    "source_id": "scale-1",
    "seq": 1
  }'
```

## Testing

```bash
/opt/anaconda/bin/python -m pytest -q
```

Current baseline:
- `12 passed`

## Notes

- The service does not store images long-term; camera remains image source of truth.
- If individual defect calls fail, scan continues with partial evidence.
- If fallback detection fails, the pipeline keeps bounded behavior and continues with primary results.
- If all frames fail, result still aggregates deterministically (possibly empty fruit list) and remains observable via logs/journal.
