# Final Project — MLOps
## Drone Waste Detection App

**Module**: MLOps — MSc Data Management M2
**Teams**: Pairs
**Submission**: Private GitHub repository — invite the professor before the deadline
**Evaluated branch**: `main`
**Deadline**: `31 April, 2026`

---
Source repo : https://github.com/sinaayyy/project_mlops

## Context

You are joining a team developing an urban waste detection system using drones. A computer vision model has already been trained and drones are patrolling cities, continuously collecting detections.

**Your mission**: design and deploy the complete MLOps infrastructure around this system — from reproducibility to production observability, including a data synchronization pipeline from the field.

The final deliverable is an application usable by field operators, displaying in real time the waste deposit zones detected by drones on an interactive map.

---

## Provided Resources

### 1. Pre-trained Models

Several detection models are to be retrieved. Each takes an image as input and returns:
- `class`: always `rubbish` (binary detection — waste or not)
- `confiance`: score between 0 and 1

All models are registered in MLflow Registry under distinct names and must be loaded via `mlflow.pyfunc.load_model(...)`.

> **Weight availability**: pre-trained weight files (`.pt`) are made available progressively in the professor's GitHub repository. Check it regularly. If weights for a model are not yet available at submission time, instantiate that model **without pre-trained weights** (random weights) so that the architecture is still registered in MLflow and loadable via the API. As soon as weights are published on the professor's repo, update your MLflow registry and your repo.

| UI display name | MLflow Registry name | Architecture | Description | Framework |
|---|---|---|---|---|
| YOLOv8 | `models:/waste-detector-yolov8/Production` | YOLOv8n | Real-time detection, good speed/accuracy trade-off | ultralytics |
| YOLO26 | `models:/waste-detector-yolo26/Production` | YOLOv26 | Advanced version, improved accuracy on small objects | ultralytics |
| RT-DETR | `models:/waste-detector-rtdetr/Production` | RT-DETR-S | Real-time transformer detector (Baidu), high accuracy | ultralytics |
| RT-DETRv2 | `models:/waste-detector-rtdetrv2/Production` | RT-DETRv2-N | v2 of RT-DETR | Hugging Face |
| RF-DETR | `models:/waste-detector-rfdetr/Production` | RF-DETR-N | DETR variant focused on robustness/accuracy | Roboflow |
| D-FINE | `models:/waste-detector-dfine/Production` | D-FINE-N | Fine box regression, excellent localization | Hugging Face |
| DEIM-DFINE | `models:/waste-detector-deim-dfine/Production` | DEIM-D-FINE-L | Training method added to D-FINE for faster convergence | [DEIM repo](https://github.com/Intellindust-AI-Lab/DEIM) |
| Yolov8-Rtdetr-decoder-head | `models:/waste-detector-fusion-model/Production` | yolov8n_yolo_neck_rtdetr_head.yaml | YOLOv8 backbone + RT-DETR decoder head for global context | [custom ultralytics](https://github.com/sialaoui/ultralytics.git@feat/yolo-rtdetr) |

### Using the `Yolov8-Rtdetr-decoder-head` model

This model uses a hybrid architecture combining:
- a **YOLOv8n backbone** for fast feature extraction,
- a **YOLO neck** for multi-scale fusion,
- an **RT-DETR decoder head** for global visual context modeling.

It is implemented in a custom fork of the Ultralytics framework:

```
https://github.com/sialaoui/ultralytics.git@feat/yolo-rtdetr
```

The architecture file is:
```
ultralytics/cfg/models/models/custom/yolov8n_yolo_neck_rtdetr_head.yaml
```

Copy this YAML file into your project repo:
```
waste-detection-mlops/
├── models/
│   └── yolov8n_yolo_neck_rtdetr_head.yaml
```

Install the custom framework inside your Docker image (add to `requirements.txt` or `Dockerfile`):
```bash
pip install git+https://github.com/sialaoui/ultralytics.git@feat/yolo-rtdetr
```

**Inference only** — this model is used for prediction, not training:
```python
from ultralytics import RTDETR

model = RTDETR("models/yolov8n_yolo_neck_rtdetr_head.yaml")
results = model.predict(image)
```

### 2. Drone patrol database (simulated)

A `generate_patrol_db.py` script is provided. It simulates a **drone returning from a mission**: each call inserts a random number of detections (between 20 and 100) into the SQLite database `drone_patrol.db`, all marked `processed=0`.

The script can be called **multiple times** to simulate successive mission returns. This continuous flow is what the Airflow pipeline must process automatically.

**Schema of the `drone_detections` table:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `drone_id` | TEXT | Drone identifier (`drone_001`, `drone_002`, `drone_003`) |
| `timestamp` | TEXT | Detection date and time (ISO 8601) |
| `latitude` | REAL | GPS latitude |
| `longitude` | REAL | GPS longitude |
| `ville` | TEXT | Patrolled city |
| `zone` | TEXT | District/zone |
| `classe` | TEXT | Always `rubbish` |
| `confiance` | REAL | Model confidence score |
| `image_filename` | TEXT | Image file name |
| `processed` | INTEGER | `0` = not processed, `1` = already loaded into the app |

> **Important**: each call to `generate_patrol_db.py` simulates a drone returning from a mission and depositing its data. The Airflow pipeline (every 10 min) must detect new entries (`processed = 0`), filter them, load them into the app, then mark them as processed (`processed = 1`). The map updates automatically after each new mission.

### 3. Test image

A sample drone waste detection image (`test_image.jpg`) is provided in the professor's repository. Place it at the root of your repo. It is used to run the verification commands in your README.

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   docker-compose.yml                    │
│                                                         │
│  ┌──────────┐  POST /predict    ┌───────────────────┐  │
│  │Streamlit │  + model_name     │    FastAPI (API)   │  │
│  │   App    │ ────────────────► │   + MLflow load    │  │
│  │+ Folium  │ ◄──────────────── │  (multi-model)     │  │
│  │   Map    │  {rubbish, conf,  └────────┬──────────┘  │
│  │[dropdown │   model_used}              │              │
│  │ model]   │               ┌────────────┴──────────┐  │
│  └────┬─────┘               │    MLflow Registry     │  │
│       │                     │  YOLOv8 | YOLO26       │  │
│       │                     │  RT-DETR | D-FINE | …  │  │
│       │                     └───────────────────────┘  │
│       │                            App DB (SQLite)      │
│       │                                  ▲              │
│       │ GET /history                     │ ETL          │
│       └──────────────────────────────────┤              │
│                                          │              │
│  ┌───────────┐   Airflow DAG    ┌────────┴──────────┐  │
│  │ Drone     │ ───────────────► │  ETL Pipeline     │  │
│  │ Patrol DB │  extract→filter  │  (your DAG)       │  │
│  │(provided) │  →load→mark      └───────────────────┘  │
│  └───────────┘                                          │
│                                                         │
│  ┌────────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Prometheus │  │ Grafana  │  │  Alertmanager     │   │
│  │  /metrics  │  │dashboard │  │  alert rules      │   │
│  └────────────┘  └──────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Two data sources on the map:**
- 🔴 Manual upload (operator uploads an image via the interface)
- 🟠 Drone patrol (data from the Airflow ETL pipeline)

---

## Expected Repository Structure

```
waste-detection-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── api/
│   ├── main.py
│   ├── Dockerfile
│   └── tests/
│       ├── test_unit.py
│       └── test_integration.py
├── app/
│   ├── app.py
│   └── Dockerfile
├── dags/
│   ├── drone_mission_simulator_dag.py
│   └── drone_patrol_sync_dag.py
├── monitoring/
│   ├── prometheus.yml
│   ├── alertmanager.yml
│   └── grafana/
│       └── dashboard.json
├── logs/
│   └── predictions.jsonl
├── models/
│   ├── yolov8n_yolo_neck_rtdetr_head.yaml   ← architecture file (fusion model)
│   └── *.pt                                 ← weight files (downloaded from prof's repo)
├── generate_patrol_db.py  ← provided
├── test_image.jpg         ← provided
├── docker-compose.yml
├── requirements.txt
└── README.md              ← grading entry point (mandatory)
```

#### Mandatory README content

> ⚠️ **The README is the primary grading tool.** The grader clones the repo and follows the commands in your README to validate each criterion. A criterion with no verification command in the README, or whose command fails = **0 points on that criterion**. The code will also be read — inconsistencies between the README and the code, non-functional code, or code that is clearly not understood may result in a grade revision.

The `README.md` must allow the full project to be tested from the root of the repo. It must cover in this order:

1. **Prerequisites** — Docker, Python versions, etc.
2. **Setup** — `git clone`, drone database generation (`python generate_patrol_db.py`)
3. **Stack startup** — `docker compose up -d` + verify all containers are UP
4. **API verification** — `curl` commands for `/health`, `/models`, `/predict` (with expected responses), HTTP 422 tests
5. **Automated tests** — `pytest` commands for unit and integration tests
6. **Airflow pipeline** — how to trigger DAGs manually and verify execution
7. **Streamlit interface** — URL and features to test
8. **Observability** — Prometheus metrics, logs, Grafana dashboard, alerts
9. **CI/CD** — GitHub Actions pipeline link and status badge
10. *(if bonus)* **Additional component** — demonstration command

A README template is provided with the project (`README_template.md`). Complete it and submit it in your repo.

---

## Deliverables & Grading — /18

> **Important**: the README is the primary grading tool, but **the source code will also be reviewed**. Inconsistencies between the README and the code, non-functional code, or code that is clearly not understood may result in a grade revision. **Oral presentations may be requested** for some teams — it is therefore essential that each member fully understands and masters everything that was produced.

---

### Chapter 2 — Packaging, Reproducibility & Model Management — `/4`

#### Environment & Containerisation

- A `requirements.txt` file allowing environment reproduction with a single command (`pip install -r requirements.txt`)
- A `Dockerfile` for the API and a `Dockerfile` for the Streamlit app — both must build without error
- A `docker-compose.yml` that orchestrates the **entire stack** (API, App, Airflow, Prometheus, Grafana, Alertmanager) and starts it in one command: `docker compose up`

#### MLflow — Model Registry (multi-model)

No model may be loaded from a file path. Each model must be:
1. Registered in the MLflow server (included in `docker-compose.yml`) under its own registry name
2. Loaded dynamically via `mlflow.pyfunc.load_model("models:/waste-detector-<name>/Production")`

The API must load **all models at startup** and keep them in memory. Model selection happens at each `/predict` call via the `model_name` parameter.

The `GET /models` endpoint must list all available models with, for each: name, MLflow version and registration date.

| Criterion | Points |
|-----------|--------|
| `requirements.txt` functional | 0.25 |
| Dockerfile API + Dockerfile App — build without error | 0.75 |
| `docker-compose.yml` — full stack in one command | 0.75 |
| MLflow registry — each model registered and loaded via `mlflow.pyfunc.load_model()` | **0.25 pt / model** (8 models → **2 pts max**) |
| `GET /models` lists all models with MLflow version + date | 0.25 |

---

### Chapter 3 — Production Application — `/5`

#### FastAPI

The API exposes the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Receives an image + GPS coordinates + `model_name`, returns `rubbish`, `confiance`, `model_used`, `timestamp` |
| `/history` | GET | Returns all stored detections (upload + drone) |
| `/models` | GET | Lists all available models with MLflow version and date |
| `/health` | GET | API status |

**Model selection**: the `model_name` parameter (form field or query param) indicates which model to use for inference. If `model_name` is missing or unknown, return `HTTP 422` with the list of valid models.

**Input validation** (`/predict`): before calling the model, verify that:
- The received file is an image (JPEG or PNG)
- The size is less than 10 MB
- Latitude is between -90 and 90, longitude between -180 and 180
- Return `HTTP 422` with an explicit message if a condition is not met

**Storage**: each prediction (manual upload AND drone) is persisted in a SQLite database `app_detections.db` with the following columns:

| Column | Description |
|--------|-------------|
| `timestamp` | Detection datetime |
| `latitude` | GPS latitude |
| `longitude` | GPS longitude |
| `confiance` | Model confidence score |
| `model_name` | Name of the model used |
| `source` | `manual` or `drone_patrol` |
| `drone_id` | Drone identifier (if applicable) |

The `classe` column is not needed — all detections are `rubbish`.

#### Tests

- `tests/test_unit.py` — minimum 3 tests: model sanity check, test `/predict` with a valid image, validation test with an invalid input
- `tests/test_integration.py` — minimum 1 test: start the API in Docker, send a real request to `/predict`, verify the response

#### Streamlit Interface

The interface must allow an operator to:
1. **Select a model** via a dropdown (fed by `GET /models`) before launching a prediction
2. Upload an image and enter GPS coordinates → display the prediction result (confidence, model used)
3. Visualize a Folium interactive map with all historical detections (upload + drone), differentiated by color or icon
4. Filter the map by source, model, or time period

| Criterion | Points |
|-----------|--------|
| Endpoints `/predict`, `/history`, `/health` functional | 0.75 |
| Model selection — `model_name` forwarded to `/predict`, HTTP 422 if unknown | 0.5 |
| Input validation — explicit HTTP 422 if invalid | 0.5 |
| `GET /models` lists all models with MLflow info | 0.25 |
| DB storage — `model_name` tracked for each prediction | 0.25 |
| Unit tests (min. 3) | 0.75 |
| Integration test (min. 1, via Docker) | 0.5 |
| Streamlit interface — model dropdown + upload + result + Folium map + filters | 1.5 |

---

### ETL Pipeline — Drone → Map Synchronisation — `/3`

This pipeline relies on **two Airflow DAGs**. Two implementation levels are possible, with different grading.

#### DAG 1 — `drone_mission_simulator`

Simulates a drone returning from a mission: calls `generate_patrol_db.py` which inserts between 20 and 100 random new detections.

**Schedule:**

| Mode | `schedule_interval` | Usage |
|------|---------------------|-------|
| Test / demo | `*/5 * * * *` | One mission every 5 min |
| Production | `0 20 * * *` | One simulated mission every evening at 8pm |

**The repo must be submitted with test mode (`*/5 * * * *`) active** to allow automatic grading.

#### DAG 2 — `drone_patrol_sync`

Contains **3 chained tasks**:

```
extract → transform → load
```

- **`extract`**: read all rows from `drone_detections` where `processed = 0`
- **`transform`**: keep only detections with `confiance >= 0.65`
- **`load`**: insert filtered detections into `app_detections.db`, then mark source rows as `processed = 1` in `drone_patrol.db`

#### Basic level — Independent DAGs

Both DAGs run on their own schedule, with no explicit link between them. DAG 2 runs every 10 minutes (`*/10 * * * *`) and processes available detections.

```
DAG 1 : simulate_mission            (*/5 * * * *)
DAG 2 : extract → transform → load  (*/10 * * * *)
```

#### Advanced level — Chained DAGs *(+0.5 pt)*

DAG 1 **immediately triggers** DAG 2 at the end of each mission via a `TriggerDagRunOperator`. DAG 2 has no schedule of its own (`schedule_interval=None`) and only runs when triggered.

```
DAG 1 : simulate_mission → trigger_sync   (*/5 * * * *)
DAG 2 : extract → transform → load        (triggered by DAG 1)
```

Airflow must be included in `docker-compose.yml` and **share a volume** with the API to access both SQLite databases.

| Criterion | Points |
|-----------|--------|
| DAG 1 `drone_mission_simulator` — runs on schedule, generates data | 0.5 |
| DAG 2 `drone_patrol_sync` — 3 tasks `extract → transform → load` without error | 1 |
| Filter `confiance >= 0.65` in `transform` + flag `processed = 1` after loading | 0.5 |
| Airflow UI accessible — both DAGs visible with execution history | 0.5 |
| Streamlit map displays drone detections distinct from manual uploads | 0.5 |
| **[Bonus]** DAG 2 triggered by DAG 1 via `TriggerDagRunOperator` | +0.5 |

---

### Chapter 4 — CI/CD — `/3`

The GitHub Actions pipeline triggers on every push to `main`.

**Expected steps:**

1. Checkout code
2. Install dependencies
3. Run unit tests (`pytest tests/test_unit.py`)
4. Run integration test (`pytest tests/test_integration.py`) — starts the API in Docker inside the pipeline
5. Build the API Docker image
6. Push the image to Docker Hub or GitHub Container Registry (GHCR)

The pipeline must be **green** (`✅`) at the deadline. A status badge must be visible in `README.md`.

| Criterion | Points |
|-----------|--------|
| Unit tests pass in the pipeline | 0.75 |
| Integration test runs in the pipeline (API started in Docker) | 0.75 |
| Build + push Docker image to a public registry | 1 |
| Pipeline green at deadline — badge in README | 0.5 |

---

### Chapter 5 — Observability — `/2`

#### Prometheus Metrics

The API must expose a `/metrics` endpoint (scraped by Prometheus) containing at minimum:

- Total number of predictions (`ml_predictions_total`)
- Inference latency in seconds — histogram (`ml_inference_latency_seconds`)
- Number of detections per model — counter by label (`ml_predictions_by_model_total{model="..."}`)
- Number of validation errors (`ml_validation_errors_total`)

#### Structured Logging

Each prediction (manual upload and drone) must be logged in `logs/predictions.jsonl` in JSON format, one entry per line:

```json
{"timestamp": "2025-03-04T14:23:01Z", "source": "manual", "latitude": 48.83, "longitude": 2.35, "confiance": 0.91, "model_name": "yolov8", "latence_ms": 43}
```

#### Grafana Dashboard

A Grafana dashboard must be versioned in the repo (`monitoring/grafana/dashboard.json`). It must contain **at minimum 4 panels**:

1. Requests per minute
2. Inference latency (p95)
3. Detections per model
4. Validation error rate

#### Alerting

At least **one alert rule** must be defined in `monitoring/alertmanager.yml`. Accepted examples:

- Average confidence score < 0.60 over the last 5 minutes
- Error rate > 5% over the last 5 minutes
- API unreachable for more than 30 seconds

| Criterion | Points |
|-----------|--------|
| Prometheus metrics — 4 metrics exposed on `/metrics` | 0.5 |
| Structured JSON logging — each prediction logged in `predictions.jsonl` | 0.5 |
| Grafana dashboard — versioned JSON file, min. 4 panels | 0.5 |
| Alerting — at least one rule defined and active in `alertmanager.yml` | 0.5 |

---

### Git & Quality — `/1`

| Criterion | Points |
|-----------|--------|
| Regular and descriptive commits — both members visibly contributing on `git log` | 0.5 |
| Correct `.gitignore` — no unwanted files (`__pycache__/`, `.venv/`, `*.db` if large, models > 100 MB) | 0.25 |
| Professor invited to the private repo **before** the deadline | 0.25 |

---

### Bonus — Additional MLOps Component — `/+2`

To achieve a grade above 18/20, you must **propose, implement and justify** an additional MLOps component that concretely improves the project.

**The proposal must include** (in `README.md`, "Bonus" section):
- The name and description of the chosen component
- Why it is relevant for this specific project (waste detection + drone)
- Technical implementation details
- A demonstration command

An undocumented or non-functional component earns 0 bonus points.

| Criterion | Points |
|-----------|--------|
| Written and justified proposal in the README | +0.5 |
| Correct technical implementation (runs without error) | +1 |
| Demonstration via a command in the verification procedure | +0.5 |

---

## MLOps Components Summary

| Component | Chapter | Tool |
|-----------|---------|------|
| Reproducible environment | 2 | `requirements.txt` |
| Containerisation | 2 | Docker |
| Service orchestration | 2 | Docker Compose |
| Model Registry | 2 | MLflow |
| Inference API | 3 | FastAPI |
| Input validation | 3 | Pydantic |
| Detection storage | 3 | SQLite |
| User interface | 3 | Streamlit |
| Interactive map | 3 | Folium |
| Unit tests | 3 | pytest |
| Integration tests | 3/4 | pytest + Docker |
| CI/CD | 4 | GitHub Actions |
| Drone mission simulation | ETL | Apache Airflow (`drone_mission_simulator`) |
| ETL pipeline + DAG chaining | ETL | Apache Airflow (`drone_patrol_sync`) |
| Metrics | 5 | Prometheus |
| Structured logging | 5 | JSON / jsonlines |
| Dashboard | 5 | Grafana |
| Alerting | 5 | Alertmanager |

---

## Tips

- **Start with `docker-compose.yml`** — if the full stack starts in one command, you have the foundation. Everything else builds on top of it.
- **The API must work from Docker**, not just locally. The integration test and the grader will verify from the container.
- **Version all artifacts**: Grafana dashboard, Alertmanager rules file. An unversioned file = not graded.
- **The Airflow pipeline must actually run**: a defined but never executed DAG will not be graded. Execution history in the Airflow UI is checked.
- **CI must be green** at the deadline. A failing pipeline = 0 on the CI/CD section, regardless of code quality.
- **Both members must code** — a repo with a single contributor on `git log` will be penalised.

---

## Grading Summary

| Section | Points |
|---------|--------|
| Chap. 2 — Packaging, Reproducibility & MLflow | /4 |
| Chap. 3 — Production application | /5 |
| ETL Pipeline — Airflow | /3 |
| Chap. 4 — CI/CD | /3 |
| Chap. 5 — Observability | /2 |
| Git & Quality | /1 |
| **Total** | **/18** |
| Bonus — Additional MLOps component | **/+2** |
| **Maximum** | **/20** |
