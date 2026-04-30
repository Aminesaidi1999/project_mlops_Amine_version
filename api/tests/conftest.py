"""Pytest config — redirects the API's SQLite DB and JSONL log to tempfiles
so unit tests don't touch /data or /app/logs. Loaded by pytest before any
test module is imported, so `main`'s module-level constants pick these up."""

import os
import tempfile
from pathlib import Path

_TMPDIR = Path(tempfile.gettempdir())
os.environ["APP_DB_PATH"] = str(_TMPDIR / "waste_test_detections.db")
os.environ["LOG_PATH"] = str(_TMPDIR / "waste_test_predictions.jsonl")
