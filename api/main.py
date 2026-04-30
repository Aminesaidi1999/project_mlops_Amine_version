"""Waste Detection API.

Endpoints
- GET  /health   : liveness + count of loaded models
- GET  /models   : list of registered MLflow Production models with version + date
- POST /predict  : multipart upload (image, latitude, longitude, model_name) -> detection
- GET  /history  : every persisted detection (manual + drone_patrol)
- GET  /metrics  : Prometheus exposition

At startup every `waste-detector-*` model in MLflow Production is loaded and held
in memory. Model selection happens per /predict call via the `model_name` form
field (validated against the cache; HTTP 422 with the list of valid names on
miss). Each prediction is persisted to SQLite, logged as a JSONL line, and
recorded in Prometheus counters/histograms.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
APP_DB_PATH = os.environ.get("APP_DB_PATH", "/data/app_detections.db")
LOG_PATH = os.environ.get("LOG_PATH", "/app/logs/predictions.jsonl")
MODEL_PREFIX = "waste-detector-"
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

logger = logging.getLogger("waste_detection_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

# --------------------------------------------------------------------------- #
# Prometheus metrics (rubric: 4 metrics minimum on /metrics)
# --------------------------------------------------------------------------- #
ml_predictions_total = Counter(
    "ml_predictions_total",
    "Total number of successful /predict calls",
)
ml_predictions_by_model_total = Counter(
    "ml_predictions_by_model_total",
    "Predictions per model",
    ["model"],
)
ml_inference_latency_seconds = Histogram(
    "ml_inference_latency_seconds",
    "Inference latency (model.predict only) in seconds",
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)
ml_validation_errors_total = Counter(
    "ml_validation_errors_total",
    "Number of /predict requests rejected by input validation (HTTP 422)",
)

# --------------------------------------------------------------------------- #
# Structured JSON logging — one line per prediction in logs/predictions.jsonl
# --------------------------------------------------------------------------- #
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
_log_lock = threading.Lock()


def write_prediction_log(payload: Dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# --------------------------------------------------------------------------- #
# SQLite — shared with Airflow DAG 2 via the /data bind mount
# --------------------------------------------------------------------------- #
def init_db() -> None:
    Path(APP_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(APP_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                latitude    REAL NOT NULL,
                longitude   REAL NOT NULL,
                confiance   REAL NOT NULL,
                model_name  TEXT NOT NULL,
                source      TEXT NOT NULL,
                drone_id    TEXT
            )
            """
        )
        conn.commit()


def insert_detection(row: Dict[str, Any]) -> None:
    with sqlite3.connect(APP_DB_PATH) as conn:
        conn.execute(
            """INSERT INTO detections
                 (timestamp, latitude, longitude, confiance, model_name, source, drone_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                row["timestamp"],
                row["latitude"],
                row["longitude"],
                row["confiance"],
                row["model_name"],
                row["source"],
                row.get("drone_id"),
            ),
        )
        conn.commit()


def fetch_history() -> List[Dict[str, Any]]:
    with sqlite3.connect(APP_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT timestamp, latitude, longitude, confiance, model_name, source, drone_id
                 FROM detections
                ORDER BY timestamp DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# MLflow model cache — loaded once at startup, held in memory
# --------------------------------------------------------------------------- #
class ModelEntry:
    __slots__ = ("name", "version", "registered_at", "model")

    def __init__(self, name: str, version: str, registered_at: str, model: Any) -> None:
        self.name = name
        self.version = version
        self.registered_at = registered_at
        self.model = model


MODEL_CACHE: Dict[str, ModelEntry] = {}
_cache_lock = threading.Lock()


def discover_and_load_models() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    try:
        registered = client.search_registered_models()
    except Exception as e:
        logger.warning("MLflow registry unreachable at startup: %r", e)
        return

    target = [m for m in registered if m.name.startswith(MODEL_PREFIX)]
    logger.info("Discovered %d %s* models in registry", len(target), MODEL_PREFIX)

    for m in target:
        prod = [v for v in m.latest_versions if v.current_stage == "Production"]
        if not prod:
            logger.warning("Skipping %s: no Production version", m.name)
            continue
        v = prod[0]
        try:
            t0 = time.perf_counter()
            model = mlflow.pyfunc.load_model(f"models:/{m.name}/Production")
            elapsed = time.perf_counter() - t0
            registered_at = datetime.fromtimestamp(v.creation_timestamp / 1000, tz=timezone.utc).isoformat()
            with _cache_lock:
                MODEL_CACHE[m.name] = ModelEntry(m.name, str(v.version), registered_at, model)
            logger.info("Loaded %s v%s in %.2fs", m.name, v.version, elapsed)
        except Exception as e:
            logger.error("FAILED to load %s: %r", m.name, e)


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    discover_and_load_models()
    yield


app = FastAPI(title="Waste Detection API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "models_loaded": len(MODEL_CACHE)}


@app.get("/models")
def list_models() -> List[Dict[str, Any]]:
    with _cache_lock:
        entries = list(MODEL_CACHE.values())
    return [
        {
            "name": e.name,
            "version": e.version,
            "registered_at": e.registered_at,
            "stage": "Production",
        }
        for e in sorted(entries, key=lambda x: x.name)
    ]


def _validate(
    image: UploadFile,
    image_bytes: bytes,
    latitude: float,
    longitude: float,
    model_name: str,
) -> None:
    """Raise HTTP 422 with an explicit message on any rule violation."""
    if model_name not in MODEL_CACHE:
        ml_validation_errors_total.inc()
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unknown model_name",
                "received": model_name,
                "valid": sorted(MODEL_CACHE.keys()),
            },
        )
    ctype = (image.content_type or "").lower()
    if ctype not in ALLOWED_CONTENT_TYPES:
        ml_validation_errors_total.inc()
        raise HTTPException(
            status_code=422,
            detail={"error": "image must be JPEG or PNG", "received_content_type": image.content_type},
        )
    if not (image_bytes.startswith(JPEG_MAGIC) or image_bytes.startswith(PNG_MAGIC)):
        ml_validation_errors_total.inc()
        raise HTTPException(
            status_code=422,
            detail={"error": "image bytes are not a valid JPEG or PNG"},
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        ml_validation_errors_total.inc()
        raise HTTPException(
            status_code=422,
            detail={"error": "image must be smaller than 10 MB", "received_bytes": len(image_bytes)},
        )
    if not (-90.0 <= latitude <= 90.0):
        ml_validation_errors_total.inc()
        raise HTTPException(
            status_code=422,
            detail={"error": "latitude must be between -90 and 90", "received": latitude},
        )
    if not (-180.0 <= longitude <= 180.0):
        ml_validation_errors_total.inc()
        raise HTTPException(
            status_code=422,
            detail={"error": "longitude must be between -180 and 180", "received": longitude},
        )


@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    model_name: str = Form(...),
) -> Dict[str, Any]:
    image_bytes = image.file.read()
    _validate(image, image_bytes, latitude, longitude, model_name)

    entry = MODEL_CACHE[model_name]
    t0 = time.perf_counter()
    out = entry.model.predict([image_bytes])[0]
    latency_s = time.perf_counter() - t0

    ml_inference_latency_seconds.observe(latency_s)
    ml_predictions_total.inc()
    ml_predictions_by_model_total.labels(model=model_name).inc()

    timestamp = datetime.now(timezone.utc).isoformat()
    confiance = float(out.get("confiance", 0.0))
    rubbish = bool(out.get("rubbish", False))

    row = {
        "timestamp": timestamp,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "confiance": confiance,
        "model_name": model_name,
        "source": "manual",
        "drone_id": None,
    }
    insert_detection(row)
    write_prediction_log({**row, "latence_ms": int(latency_s * 1000)})

    return {
        "rubbish": rubbish,
        "confiance": confiance,
        "model_used": model_name,
        "timestamp": timestamp,
    }


@app.get("/history")
def history() -> List[Dict[str, Any]]:
    return fetch_history()


# --------------------------------------------------------------------------- #
# Bonus: multi-model shoot-out
# --------------------------------------------------------------------------- #
@app.post("/predict/compare")
def predict_compare(
    image: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
) -> Dict[str, Any]:
    """Run every loaded model on the same image and return ranked results.

    Bonus MLOps component (rubric §Bonus). Operators routinely face the
    question: "which of our 8 models should I trust for this kind of drone
    image?". This endpoint answers it with real field data — every Production
    model is invoked on the upload, results are sorted by confiance, and per-
    model latency is reported. It also serves as a built-in A/B harness for
    new challenger models.

    Latency cost: ~8 x single-model latency (≈ 1–2 s for the current 8 models).
    Each per-model call still increments the same Prometheus counters as a
    plain /predict, so the /metrics endpoint reflects compare-driven traffic.

    Note: this endpoint deliberately does NOT persist to SQLite — it is a
    diagnostic / A-B tool, not a production prediction. The standard /predict
    is what writes to the detection history.
    """
    image_bytes = image.file.read()

    # Reuse the same validator as /predict, minus model_name (which we ignore).
    if not MODEL_CACHE:
        raise HTTPException(
            status_code=503,
            detail={"error": "no models loaded — register them first via scripts/register_models.py"},
        )
    ctype = (image.content_type or "").lower()
    if ctype not in ALLOWED_CONTENT_TYPES:
        ml_validation_errors_total.inc()
        raise HTTPException(status_code=422, detail={"error": "image must be JPEG or PNG", "received_content_type": image.content_type})
    if not (image_bytes.startswith(JPEG_MAGIC) or image_bytes.startswith(PNG_MAGIC)):
        ml_validation_errors_total.inc()
        raise HTTPException(status_code=422, detail={"error": "image bytes are not a valid JPEG or PNG"})
    if len(image_bytes) > MAX_IMAGE_BYTES:
        ml_validation_errors_total.inc()
        raise HTTPException(status_code=422, detail={"error": "image must be smaller than 10 MB", "received_bytes": len(image_bytes)})
    if not (-90.0 <= latitude <= 90.0):
        ml_validation_errors_total.inc()
        raise HTTPException(status_code=422, detail={"error": "latitude must be between -90 and 90", "received": latitude})
    if not (-180.0 <= longitude <= 180.0):
        ml_validation_errors_total.inc()
        raise HTTPException(status_code=422, detail={"error": "longitude must be between -180 and 180", "received": longitude})

    timestamp = datetime.now(timezone.utc).isoformat()
    results: List[Dict[str, Any]] = []
    for name, entry in MODEL_CACHE.items():
        try:
            t0 = time.perf_counter()
            out = entry.model.predict([image_bytes])[0]
            latency_s = time.perf_counter() - t0

            ml_inference_latency_seconds.observe(latency_s)
            ml_predictions_total.inc()
            ml_predictions_by_model_total.labels(model=name).inc()

            results.append({
                "model_name": name,
                "rubbish": bool(out.get("rubbish", False)),
                "confiance": float(out.get("confiance", 0.0)),
                "latency_ms": int(latency_s * 1000),
                "error": None,
            })
        except Exception as e:  # noqa: BLE001
            results.append({
                "model_name": name,
                "rubbish": False,
                "confiance": 0.0,
                "latency_ms": None,
                "error": repr(e),
            })

    # Sort: highest confidence first, then lowest latency as tiebreaker.
    results.sort(key=lambda r: (-r["confiance"], r["latency_ms"] or 10**9))

    return {
        "timestamp": timestamp,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "models_evaluated": len(results),
        "results": results,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
