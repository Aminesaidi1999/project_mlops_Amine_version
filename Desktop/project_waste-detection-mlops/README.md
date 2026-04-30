# Waste Detection MLOps — Drone Patrol Stack

[![CI](https://github.com/YoussefRais12/waste-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/YoussefRais12/waste-detection-mlops/actions/workflows/ci.yml)

End-to-end MLOps stack for the urban waste-detection drone scenario described in [`projet.md`](projet.md). One `docker compose up -d` brings every component online: 8 detection models in MLflow, a FastAPI inference server, a Streamlit operator UI with a live Folium map, an Airflow ETL pipeline that promotes drone-patrol detections into the operator app, and a full Prometheus + Grafana + Alertmanager observability lane.

- **Reference repo (provided weights & test image)**: <https://github.com/sinaayyy/project_mlops>
- **This repo**: [<https://github.com/YoussefRais12/waste-detection-mlops>](https://github.com/Aminesaidi1999/waste-detection-mlops/tree/master)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        docker-compose.yml (one command)                      │
│                                                                              │
│  ┌──────────┐  POST /predict     ┌───────────────────┐  ┌────────────────┐  │
│  │Streamlit │ + model_name       │  FastAPI (api)     │  │  MLflow Reg.   │  │
│  │   app    │ ─────────────────► │  loads 8 models    │◄─┤  8 models      │  │
│  │ + Folium │ ◄───────────────── │  /predict /models  │  │  Production    │  │
│  │ map +    │   {rubbish, conf,  │  /history /metrics │  └────────────────┘  │
│  │ filters  │    model_used}     │  /predict/compare  │                       │
│  └──────────┘                    └────┬───────────────┘                       │
│                                       │                                       │
│                                       │ writes /data/app_detections.db        │
│                                       ▼                                       │
│  ┌──────────────┐    extract       ┌───────────┐    transform    ┌─────────┐ │
│  │ generate_    │ ────► drone_     │ Airflow   │ ─────► confiance│ load    │ │
│  │ patrol_db.py │       patrol.db  │ DAG 1+DAG2│        >= 0.65  │ into app│ │
│  │ (DAG 1)      │                  │ chained   │                 │ DB      │ │
│  └──────────────┘                  └───────────┘                 └─────────┘ │
│                                                                              │
│  ┌────────────┐    ┌──────────┐    ┌──────────────┐                          │
│  │ Prometheus │───►│ Grafana  │    │ Alertmanager │  ←  alerts.yml rules    │
│  │  /metrics  │    │ 4 panels │    │   2 rules    │                          │
│  └────────────┘    └──────────┘    └──────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

Manual uploads land **red** on the map; drone-patrol detections land **orange**.

---

## Project structure

```
waste-detection-mlops/
├── .github/workflows/ci.yml         # CI: tests + build + push to GHCR
├── api/                             # FastAPI inference service
│   ├── main.py                      #   /predict /history /models /metrics /predict/compare
│   ├── Dockerfile
│   ├── requirements.txt
│   └── tests/
│       ├── conftest.py              # redirects DB/log to tempfiles
│       ├── test_unit.py             # 6 unit tests
│       └── test_integration.py      # 2 integration tests vs the live API
├── app/                             # Streamlit operator UI
│   ├── app.py                       #   Predict, Map & History, Compare Models tabs
│   ├── Dockerfile
│   └── requirements.txt
├── dags/
│   ├── drone_mission_simulator_dag.py   # */5 cron, BashOp + TriggerDagRunOperator
│   └── drone_patrol_sync_dag.py         # extract -> transform -> load
├── models/
│   ├── yolov8n/best.pt              # real weights from prof's repo (6 MB)
│   └── yolov8n_yolo_neck_rtdetr_head.yaml   # fusion model architecture
├── monitoring/
│   ├── prometheus.yml               # scrape targets
│   ├── alerts.yml                   # 2 alert rules
│   ├── alertmanager.yml             # routing
│   └── grafana/
│       ├── dashboard.json           # 4 panels (versioned)
│       └── provisioning/            # auto-load datasource + dashboard
├── scripts/
│   ├── waste_detector_pyfunc.py     # mlflow.pyfunc.PythonModel (3 frameworks)
│   └── register_models.py           # registers 8 models -> Production
├── docker-compose.yml               # 7 services
├── generate_patrol_db.py            # provided — simulates drone missions
├── test_image.jpg                   # provided — real drone waste image
├── projet.md                        # the rubric
├── requirements.txt                 # master env (for pip-only setup)
└── README.md                        # this file
```

---

## Prerequisites

- **Docker Engine ≥ 24** with **Docker Compose v2** (Docker Desktop on macOS/Windows ships both).
- **`curl`** for the verification commands. Optional: **Python 3.11+** if you want to invoke the test suite from your host.

> No GPU required — the API image installs CPU-only PyTorch wheels (saves ~2.2 GB and cuts the build to ~4 min).

---

## 1. Setup

```bash
git clone https://github.com/Aminesaidi1999/waste-detection-mlops/tree/master
cd waste-detection-mlops

# Seed the drone DB once so DAG 1's first run isn't required to see data on the map.
# (Airflow re-runs this every 5 minutes anyway — it's just a head start.)
python generate_patrol_db.py
mv drone_patrol.db data/   # or wait — DAG 1 will create it inside /data automatically
```

Expected output of `generate_patrol_db.py`:
```
✓ Mission simulated — drone_patrol.db updated
  47 new detections inserted
  Average confidence       : 0.71
  Below threshold 0.65     : ~12
  Will pass filter         : ~35
  Total cumulative in DB   : 47
```

---

## 2. Stack startup

```bash
docker compose up -d --build
```

After ~1 minute (image cache → faster on subsequent boots), all 7 containers should be healthy:

```bash
docker compose ps
```

Expected — **`(healthy)`** on api, app, mlflow:
```
NAME           STATUS
airflow        Up
alertmanager   Up
api            Up (healthy)
app            Up (healthy)
grafana        Up
mlflow         Up (healthy)
prometheus     Up
```

### Register the 8 models in MLflow (one-time bootstrap)

The MLflow registry starts empty. Run this **once** after the stack is up — it populates every `waste-detector-<name>` slot in **Production**:

```bash
docker compose exec api python /app/scripts/register_models.py
```

Expected tail:
```
8/8 models registered & promoted to Production
Verify at http://localhost:5000/#/models
```

The api container auto-reloads its in-memory cache on next restart:

```bash
docker compose restart api
curl -s http://localhost:8000/health
# {"status":"ok","models_loaded":8}
```

---

## 3. API verification (Chap. 2 + Chap. 3)

### `/health`
```bash
curl -s http://localhost:8000/health
# {"status":"ok","models_loaded":8}
```

### `/models` — registry list with version + date (Chap. 2 — 0.25 pt)
```bash
curl -s http://localhost:8000/models | python -m json.tool
```
Expected: 8 entries, each with `name`, `version`, `registered_at`, `stage="Production"`.

### `/predict` — golden path (Chap. 3 — 0.75 pt)
```bash
curl -s -X POST http://localhost:8000/predict \
  -F "image=@test_image.jpg;type=image/jpeg" \
  -F "latitude=48.8566" \
  -F "longitude=2.3522" \
  -F "model_name=waste-detector-yolov8"
```
Expected: `{"rubbish":true,"confiance":0.6438...,"model_used":"waste-detector-yolov8","timestamp":"..."}`.

### `/predict` — model selection 422 (Chap. 3 — 0.5 pt)
```bash
curl -i -X POST http://localhost:8000/predict \
  -F "image=@test_image.jpg;type=image/jpeg" \
  -F "latitude=10" -F "longitude=10" \
  -F "model_name=does-not-exist"
```
Expected: `HTTP 422` with body `{"detail":{"error":"unknown model_name","received":"does-not-exist","valid":[...8 names...]}}`.

### `/predict` — input validation 422 (Chap. 3 — 0.5 pt)
```bash
# latitude out of range
curl -i -X POST http://localhost:8000/predict \
  -F "image=@test_image.jpg;type=image/jpeg" \
  -F "latitude=999" -F "longitude=10" \
  -F "model_name=waste-detector-yolov8"

# wrong content-type (not JPEG/PNG)
echo "not an image" > /tmp/bad.txt
curl -i -X POST http://localhost:8000/predict \
  -F "image=@/tmp/bad.txt;type=text/plain" \
  -F "latitude=10" -F "longitude=10" \
  -F "model_name=waste-detector-yolov8"
```
Both expected: `HTTP 422` with explicit messages.

### `/history`
```bash
curl -s http://localhost:8000/history | python -m json.tool | head -20
```
Expected: list of every prediction (manual + drone) with `timestamp`, `latitude`, `longitude`, `confiance`, `model_name`, `source`, `drone_id`.

---

## 4. Automated tests (Chap. 3 — 0.75 + 0.5 pt)

```bash
docker compose exec api pytest /app/tests/ -v -W ignore::DeprecationWarning
```
Expected: **8 passed** (6 unit + 2 integration).

Run only one suite:
```bash
docker compose exec api pytest /app/tests/test_unit.py        -v   # 6 unit
docker compose exec api pytest /app/tests/test_integration.py -v   # 2 integration (live API)
```

---

## 5. Airflow ETL pipeline (ETL — /3 + 0.5 bonus)

Airflow webserver: <http://localhost:8080> (login `airflow` / `airflow`).

Both DAGs auto-detect on startup. If they don't appear, force a reparse:
```bash
docker compose exec airflow airflow dags reserialize
```

Unpause both:
```bash
docker compose exec airflow airflow dags unpause drone_mission_simulator
docker compose exec airflow airflow dags unpause drone_patrol_sync
```

Trigger a mission immediately (vs waiting up to 5 min for the cron):
```bash
docker compose exec airflow airflow dags trigger drone_mission_simulator -r "manual_$(date +%s)"
```

Verify the chain ran:
```bash
docker compose exec airflow airflow tasks states-for-dag-run drone_mission_simulator <run_id>
docker compose exec airflow airflow dags list-runs -d drone_patrol_sync | head -5
```
Expected: `simulate_mission` + `trigger_sync` `success`, then a fresh `drone_patrol_sync` run with `extract` + `transform` + `load` all `success`.

Inspect the data pipeline result:
```bash
docker compose exec airflow python -c "
import sqlite3
c = sqlite3.connect('/data/drone_patrol.db'); c.row_factory = sqlite3.Row
print('drone_patrol.db total :', c.execute('SELECT COUNT(*) FROM drone_detections').fetchone()[0])
print('  processed=1         :', c.execute('SELECT COUNT(*) FROM drone_detections WHERE processed=1').fetchone()[0])
print('  confiance >= 0.65   :', c.execute('SELECT COUNT(*) FROM drone_detections WHERE confiance >= 0.65').fetchone()[0])

a = sqlite3.connect('/data/app_detections.db')
for source, n in a.execute(\"SELECT source, COUNT(*) FROM detections GROUP BY source\"):
    print(f'app_detections.db {source}: {n}')
"
```

**Advanced level (+0.5 bonus)**: DAG 1 ends with a `TriggerDagRunOperator` that immediately fires DAG 2 — see `dags/drone_mission_simulator_dag.py`. DAG 2 has `schedule_interval=None` (only triggered).

---

## 6. Streamlit interface (Chap. 3 — 1.5 pt)

Open <http://localhost:8501>.

| Feature | Where | Rubric |
|---|---|---|
| Model dropdown fed by `GET /models` | **Predict** tab | model selection |
| Image upload + GPS lat/lon → result panel | **Predict** tab | upload + GPS + result |
| Folium map with **red (manual)** vs **orange (drone)** markers + popups | **Map & History** tab | distinct sources |
| Filters by source / model / time window | **Map & History** tab | filters |
| **Multi-model shoot-out** (bonus) | **Compare Models** tab | bonus |

The "API status" sidebar shows `OK · 8 models loaded` if the bootstrap step ran.

---

## 7. Observability (Chap. 5 — /2)

### Prometheus metrics
```bash
curl -s http://localhost:8000/metrics | grep -E '^ml_'
```
Exposed metrics (rubric requires ≥ 4):
- `ml_predictions_total` — total successful /predict calls
- `ml_predictions_by_model_total{model="..."}` — per-model counter
- `ml_inference_latency_seconds` — histogram
- `ml_validation_errors_total` — input-validation rejects

Prometheus UI: <http://localhost:9090> · query e.g. `rate(ml_predictions_total[5m])`.

### Structured JSON logging
Every prediction lands as a JSON line in `logs/predictions.jsonl`:
```bash
docker compose exec api tail -3 /app/logs/predictions.jsonl
# {"timestamp": "...", "source": "manual", "latitude": 48.85, "longitude": 2.35, "confiance": 0.64, "model_name": "waste-detector-yolov8", "latence_ms": 203, ...}
```

### Grafana dashboard (versioned at `monitoring/grafana/dashboard.json`)
Open <http://localhost:3000> (anonymous viewer enabled, or login `admin` / `admin`). Dashboard "Waste Detection — API" auto-provisioned with **4 panels**:

1. Requests per minute
2. Inference latency (p95)
3. Detections per model
4. Validation error rate

### Alertmanager rules
File: `monitoring/alerts.yml` · Alertmanager UI: <http://localhost:9093>.

| Alert | Condition |
|---|---|
| `APIDown` | `up{job="waste-detection-api"} == 0` for 30s (critical) |
| `HighValidationErrorRate` | validation-error rate > 5 % over 5 min (warning) |

Demonstrate `APIDown` firing:
```bash
docker compose stop api
sleep 45
curl -s 'http://localhost:9090/api/v1/alerts' | python -m json.tool | head -30
docker compose start api
```

---

## 8. CI / CD (Chap. 4 — /3)

Workflow file: [`.github/workflows/ci.yml`](.github/workflows/ci.yml). Runs on every push to `main` and on every PR.

| Stage | What |
|---|---|
| **unit-tests** | Sets up Python 3.11, installs slim deps, runs `pytest api/tests/test_unit.py` |
| **integration-and-publish** | `docker compose build` → start MLflow → `register_models.py` → start full stack → wait for 8 models → run `pytest api/tests/test_integration.py` inside the api container → on push to `main`: log in to GHCR and push `ghcr.io/youssefrais12/waste-detection-api:latest` and `:<sha>` |

Status badge above the README header is the canonical "is it green?" signal.

**Pull the published image** (after CI runs):
```bash
docker pull ghcr.io/youssefrais12/waste-detection-api:latest
```

---

## 9. Bonus — Multi-Model Shoot-Out (`/predict/compare`)

**Component name**: Multi-Model Shoot-Out — built-in champion/challenger comparison endpoint.

**Why it's relevant (drone waste detection specifically)**: Operators have 8 detection models registered (yolov8, yolo26, rtdetr, rtdetrv2, rfdetr, dfine, deim-dfine, fusion). Drone images vary by city, lighting, time of day, and waste type. Picking the right model per image style is otherwise guesswork. This endpoint runs *every* loaded model on a single upload and returns them ranked by confidence with per-model latency, so the ops team can:

1. Pick the most reliable model for a given image style (urban vs. peri-urban, day vs. dusk).
2. A/B-test a new challenger against the incumbents using real field data — promote it only if it consistently beats the production model on the same images.
3. Detect silent regressions: if a previously top-3 model suddenly ranks last on familiar inputs, that's a signal to investigate (data drift, weight rot).

**Technical implementation**:
- New endpoint `POST /predict/compare` in `api/main.py` (~80 LOC). Same input schema as `/predict` minus `model_name`. Loops over `MODEL_CACHE`, increments the existing per-model Prometheus counters (so Grafana's "Detections per model" panel naturally reflects shoot-out traffic), and returns `{models_evaluated, results: [{model_name, rubbish, confiance, latency_ms, error}, ...]}` sorted by confidence desc, latency asc as tiebreaker.
- Per-model failures are caught and reported with `error: "<repr>"` in the response — one bad model never breaks the batch.
- Streamlit "Compare Models" tab in `app/app.py` (~70 LOC) — uploads + GPS form, calls the endpoint, renders a winner-card + two horizontal bar charts (confidence and latency) + a raw-results table.
- The endpoint deliberately **does not** persist to `app_detections.db` — it's a diagnostic tool, not a production prediction. The standard `/predict` is what writes history.

**Demonstration command**:
```bash
curl -s -X POST http://localhost:8000/predict/compare \
  -F "image=@test_image.jpg;type=image/jpeg" \
  -F "latitude=48.8566" \
  -F "longitude=2.3522" | python -m json.tool
```
Expected: 8 model entries ranked, e.g.
```json
{
  "models_evaluated": 8,
  "results": [
    {"model_name": "waste-detector-yolov8", "rubbish": true, "confiance": 0.6438, "latency_ms": 23, "error": null},
    {"model_name": "waste-detector-fusion-model", "rubbish": true, "confiance": 0.4597, "latency_ms": 31, "error": null},
    ...
  ]
}
```
UI demonstration: Streamlit → "Compare Models (Bonus)" tab → upload `test_image.jpg` → click **Run shoot-out**.

---

## A note on the model weights

Per `projet.md`, weights ship progressively from the professor's repo. As of the submission:

| Model | Source | Status in this repo |
|---|---|---|
| **yolov8** | `models/yolov8n/best.pt` (committed, 6 MB) | **Real weights** |
| rtdetr | ultralytics-bundled YAML, random init | Architecture loaded |
| fusion-model | `models/yolov8n_yolo_neck_rtdetr_head.yaml` from the `sialaoui/ultralytics@feat/yolo-rtdetr` fork, random init | Custom architecture loaded |
| yolo26, rtdetrv2, rfdetr, dfine, deim-dfine | Stub branch of the shared pyfunc — deterministic synthetic output | Loadable, returns canonical `{rubbish, confiance, model_name}` |

This matches the rubric note explicitly permitting random-init when prof weights aren't published yet. Re-running `scripts/register_models.py` after new weights drop bumps every model's version and re-promotes it to Production atomically.

---

## Quick verification checklist (for the grader)

```bash
# 1. up
docker compose up -d --build && docker compose ps
docker compose exec api python /app/scripts/register_models.py
docker compose restart api

# 2. API
curl -s http://localhost:8000/health
curl -s http://localhost:8000/models | python -m json.tool | head
curl -s -X POST http://localhost:8000/predict \
  -F "image=@test_image.jpg;type=image/jpeg" \
  -F "latitude=48.85" -F "longitude=2.35" \
  -F "model_name=waste-detector-yolov8"

# 3. tests
docker compose exec api pytest /app/tests/ -v -W ignore::DeprecationWarning

# 4. Airflow
open http://localhost:8080   # airflow / airflow
docker compose exec airflow airflow dags trigger drone_mission_simulator -r demo

# 5. Streamlit
open http://localhost:8501

# 6. Observability
curl -s http://localhost:8000/metrics | grep ml_predictions_total
open http://localhost:3000   # Grafana
open http://localhost:9093   # Alertmanager

# 7. Bonus
curl -s -X POST http://localhost:8000/predict/compare \
  -F "image=@test_image.jpg;type=image/jpeg" \
  -F "latitude=48.85" -F "longitude=2.35" | python -m json.tool
```
