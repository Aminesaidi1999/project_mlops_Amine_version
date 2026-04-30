"""Unit tests — no MLflow, no real models. The lifespan's discover-and-load
function is patched out and a fake model is injected into MODEL_CACHE.

Covers (rubric requires >=3):
- model sanity check on the shared pyfunc wrapper (stub branch)
- /predict happy-path with a synthesized JPEG
- /predict invalid-input (HTTP 422) cases: unknown model, lat out of range,
  non-image content type
- /health
"""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Import main once at module load — re-importing would crash the prometheus_client
# global REGISTRY (Counter/Histogram are module-level singletons). conftest.py
# already redirected APP_DB_PATH and LOG_PATH to tempfiles before this import.
import main  # noqa: E402

# Short-circuit MLflow loading at lifespan startup. The function lookup happens
# at lifespan-execution time, so patching the module attribute is sufficient.
main.discover_and_load_models = lambda: None


def _jpeg_bytes(size: tuple[int, int] = (32, 32), color: tuple[int, int, int] = (200, 50, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def app_client():
    """TestClient with one fake model freshly injected into MODEL_CACHE."""
    main.MODEL_CACHE.clear()

    fake = MagicMock()
    fake.predict.return_value = [
        {"rubbish": True, "confiance": 0.87, "model_name": "waste-detector-yolov8"}
    ]
    main.MODEL_CACHE["waste-detector-yolov8"] = main.ModelEntry(
        name="waste-detector-yolov8",
        version="1",
        registered_at="2026-01-01T00:00:00+00:00",
        model=fake,
    )

    with TestClient(main.app) as client:
        yield client, fake

    main.MODEL_CACHE.clear()


# --------------------------------------------------------------------------- #
# 1. Model sanity check (rubric: "model sanity check")
# --------------------------------------------------------------------------- #
def test_pyfunc_stub_returns_canonical_shape(tmp_path):
    """The shared pyfunc 'stub' branch must always return {rubbish, confiance, model_name}."""
    # Path differs in container (/app/scripts) vs CI runner (<repo>/scripts).
    from pathlib import Path as _P
    for candidate in [_P("/app/scripts"), _P(__file__).resolve().parents[2] / "scripts"]:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break
    from waste_detector_pyfunc import WasteDetectorPyFunc

    cfg = tmp_path / "config.json"
    cfg.write_text('{"model_name": "test-stub", "framework": "stub"}')
    ctx = MagicMock()
    ctx.artifacts = {"config": str(cfg)}

    m = WasteDetectorPyFunc()
    m.load_context(ctx)
    out = m.predict(None, [_jpeg_bytes()])

    assert len(out) == 1
    r = out[0]
    assert set(r.keys()) == {"rubbish", "confiance", "model_name"}
    assert isinstance(r["rubbish"], bool)
    assert 0.0 <= r["confiance"] <= 1.0
    assert r["model_name"] == "test-stub"


# --------------------------------------------------------------------------- #
# 2. /health
# --------------------------------------------------------------------------- #
def test_health_returns_ok_with_models_loaded(app_client):
    client, _ = app_client
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == 1


# --------------------------------------------------------------------------- #
# 3. /predict — happy path
# --------------------------------------------------------------------------- #
def test_predict_valid_image_returns_canonical_response(app_client):
    client, fake = app_client
    r = client.post(
        "/predict",
        files={"image": ("img.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"latitude": "48.8566", "longitude": "2.3522", "model_name": "waste-detector-yolov8"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["rubbish"] is True
    assert body["confiance"] == 0.87
    assert body["model_used"] == "waste-detector-yolov8"
    assert body["timestamp"]
    fake.predict.assert_called_once()


# --------------------------------------------------------------------------- #
# 4-6. /predict — validation (HTTP 422)
# --------------------------------------------------------------------------- #
def test_predict_unknown_model_returns_422(app_client):
    client, _ = app_client
    r = client.post(
        "/predict",
        files={"image": ("img.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"latitude": "10", "longitude": "10", "model_name": "does-not-exist"},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "unknown model_name"
    assert detail["received"] == "does-not-exist"
    assert "waste-detector-yolov8" in detail["valid"]


def test_predict_invalid_latitude_returns_422(app_client):
    client, _ = app_client
    r = client.post(
        "/predict",
        files={"image": ("img.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"latitude": "999", "longitude": "10", "model_name": "waste-detector-yolov8"},
    )
    assert r.status_code == 422
    assert "latitude" in r.json()["detail"]["error"].lower()


def test_predict_non_image_content_type_returns_422(app_client):
    client, _ = app_client
    r = client.post(
        "/predict",
        files={"image": ("note.txt", b"this is not an image", "text/plain")},
        data={"latitude": "10", "longitude": "10", "model_name": "waste-detector-yolov8"},
    )
    assert r.status_code == 422
    assert "JPEG" in r.json()["detail"]["error"] or "PNG" in r.json()["detail"]["error"]
