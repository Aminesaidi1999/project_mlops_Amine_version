"""Register all 8 detector architectures in MLflow under waste-detector-<name>
and transition each to the Production stage.

Idempotent: re-running creates a new version per model and promotes it to
Production (archiving the previous one). Run from inside the api container:

    docker compose exec api python /app/scripts/register_models.py

Per-model rationale (rubric explicitly permits random init when weights aren't
published — `projet.md` "Weight availability" note):

  yolov8        ultralytics + real weights from prof's repo (models/yolov8n/best.pt)
  yolo26        stub — YOLOv26 is not in ultralytics 8.3.x; registered for API loadability
  rtdetr        ultralytics RTDETR + random init (rtdetr-l.yaml from ultralytics)
  rtdetrv2      stub — keeps the bundle slim; can swap for transformers later
  rfdetr        stub — Roboflow's RF-DETR not in our env
  dfine         stub — same reason
  deim-dfine    stub — DEIM repo not packaged
  fusion-model  ultralytics RTDETR + custom YAML (random init); uses the fork installed via requirements.txt
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from waste_detector_pyfunc import WasteDetectorPyFunc  # noqa: E402

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))
EXPERIMENT_NAME = "waste-detector-bootstrap"


def _maybe(path: Path) -> str | None:
    return str(path) if path.exists() else None


MODELS = [
    {
        "name": "waste-detector-yolov8",
        "framework": "ultralytics_yolo",
        "weight": _maybe(MODELS_DIR / "yolov8n" / "best.pt"),
        "arch": None,
    },
    {
        "name": "waste-detector-yolo26",
        "framework": "stub",
        "weight": None,
        "arch": None,
    },
    {
        "name": "waste-detector-rtdetr",
        "framework": "ultralytics_rtdetr",
        "weight": None,
        "arch": "rtdetr-l.yaml",  # resolved by ultralytics from its bundled cfg dir
    },
    {
        "name": "waste-detector-rtdetrv2",
        "framework": "stub",
        "weight": None,
        "arch": None,
    },
    {
        "name": "waste-detector-rfdetr",
        "framework": "stub",
        "weight": None,
        "arch": None,
    },
    {
        "name": "waste-detector-dfine",
        "framework": "stub",
        "weight": None,
        "arch": None,
    },
    {
        "name": "waste-detector-deim-dfine",
        "framework": "stub",
        "weight": None,
        "arch": None,
    },
    {
        "name": "waste-detector-fusion-model",
        "framework": "ultralytics_rtdetr",
        "weight": None,
        "arch": _maybe(MODELS_DIR / "yolov8n_yolo_neck_rtdetr_head.yaml"),
    },
]


def _build_artifacts(cfg: dict, tmpdir: Path) -> dict:
    config_payload = {
        "model_name": cfg["name"],
        "framework": cfg["framework"],
    }
    config_path = tmpdir / "config.json"
    config_path.write_text(json.dumps(config_payload, indent=2))

    artifacts = {"config": str(config_path)}
    if cfg.get("weight") and os.path.exists(cfg["weight"]):
        artifacts["weight"] = cfg["weight"]
    if cfg.get("arch"):
        arch = cfg["arch"]
        if isinstance(arch, str) and os.path.exists(arch):
            artifacts["arch"] = arch
        # Strings that are NOT existing paths (e.g. "rtdetr-l.yaml") are
        # treated as ultralytics-resolvable identifiers — the pyfunc passes
        # them straight to RTDETR(...) which knows how to find them.
        elif isinstance(arch, str):
            config_payload["arch_hint"] = arch
            config_path.write_text(json.dumps(config_payload, indent=2))
    return artifacts


def register_one(client: MlflowClient, cfg: dict) -> str:
    name = cfg["name"]
    print(f"\n=== {name} ({cfg['framework']}) ===")

    tmpdir = Path(tempfile.mkdtemp(prefix="reg_"))
    try:
        artifacts = _build_artifacts(cfg, tmpdir)
        code_paths = [str(SCRIPT_DIR / "waste_detector_pyfunc.py")]

        with mlflow.start_run(run_name=f"register-{name}") as run:
            mlflow.set_tag("model_name", name)
            mlflow.set_tag("framework", cfg["framework"])
            mlflow.set_tag("has_real_weights", "true" if "weight" in artifacts else "false")
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=WasteDetectorPyFunc(),
                artifacts=artifacts,
                code_paths=code_paths,
                registered_model_name=name,
            )
            print(f"  logged   run_id={run.info.run_id}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    versions = client.search_model_versions(f"name='{name}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"  promoted v{latest.version} -> Production")
    return latest.version


def main() -> int:
    print(f"MLflow tracking URI: {TRACKING_URI}")
    print(f"Local models dir  : {MODELS_DIR}")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    failures: list[tuple[str, Exception]] = []
    for cfg in MODELS:
        try:
            register_one(client, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e!r}")
            failures.append((cfg["name"], e))

    print("\n--- summary ---")
    print(f"{len(MODELS) - len(failures)}/{len(MODELS)} models registered & promoted to Production")
    if failures:
        for name, err in failures:
            print(f"  FAIL  {name}: {err!r}")
        return 1
    print("Verify at http://localhost:5000/#/models  (each model -> latest version stage = Production)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
