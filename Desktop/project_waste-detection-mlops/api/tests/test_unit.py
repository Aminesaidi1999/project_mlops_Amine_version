"""
Foundation stub — only smoke-tests /health.
Teammate A adds: model sanity check, /predict valid image, /predict invalid input.
"""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
