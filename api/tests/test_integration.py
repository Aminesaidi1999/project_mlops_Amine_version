"""Integration tests — exercise the live API running in Docker.

The rubric requires at least one integration test that "starts the API in
Docker and sends a real request to /predict, verifies the response". The CI
workflow brings the stack up with `docker compose up -d` and runs:

    docker compose exec api pytest /app/tests/test_integration.py -v

Locally:
    docker compose up -d
    docker compose exec api python /app/scripts/register_models.py   # one-time
    docker compose exec api pytest /app/tests/test_integration.py -v

Or from outside the container (CI alternate):
    API_URL=http://localhost:8000 pytest api/tests/test_integration.py -v

A synthesized JPEG is used to avoid any external file dependency.
"""

from __future__ import annotations

import io
import os

import pytest
import requests
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT = 30


def _synthetic_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color=(120, 80, 40)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def test_health_endpoint_live():
    r = requests.get(f"{API_URL}/health", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] >= 1


def test_predict_endpoint_returns_valid_response_for_real_request():
    r = requests.post(
        f"{API_URL}/predict",
        files={"image": ("img.jpg", _synthetic_jpeg(), "image/jpeg")},
        data={
            "latitude": "48.8566",
            "longitude": "2.3522",
            "model_name": "waste-detector-yolov8",
        },
        timeout=TIMEOUT,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body.get("rubbish"), bool)
    confiance = body.get("confiance")
    assert isinstance(confiance, (int, float))
    assert 0.0 <= confiance <= 1.0
    assert body.get("model_used") == "waste-detector-yolov8"
    assert body.get("timestamp")
