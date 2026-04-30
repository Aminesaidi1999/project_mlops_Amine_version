# Waste Detection MLOps — [Group Name]

All the informations about project are in the project.md file

**Based Repo**: https://github.com/sinaayyy/project_mlops


**Members**: [First Last 1] · [First Last 2]
**Repo**: `https://github.com/<your-repo>`

![CI/CD](https://github.com/<your-repo>/actions/workflows/ci.yml/badge.svg)

> **For the grader**: clone the repo, generate the drone database, start the stack, then follow the commands section by section. Criteria marked `[VISUAL]` are evaluated through the UI or the GitHub repository.

---

## Setup

```bash
git clone https://github.com/<your-repo>.git
cd waste-detection-mlops
python generate_patrol_db.py
```

Expected output:
```
✓ Mission simulated — drone_patrol.db updated
  XX new detections inserted
```

---

## Chap. 2 — Packaging & MLflow `/4`

### `requirements.txt` present `0.25 pt`

```bash
ls requirements.txt
# File must exist at the root of the repo
```

### Dockerfile API + Dockerfile App — build without error `0.75 pt`

```bash
docker build -t waste-api ./api
docker build -t waste-app ./app
```

### `docker-compose.yml` — full stack in one command `0.75 pt`

```bash
docker compose up -d
docker compose ps
```

Expected output — all services `running`:
```
NAME          STATUS    PORTS
api           running   0.0.0.0:8000->8000/tcp
app           running   0.0.0.0:8501->8501/tcp
airflow       running   0.0.0.0:8080->8080/tcp
mlflow        running   0.0.0.0:5000->5000/tcp
prometheus    running   0.0.0.0:9090->9090/tcp
grafana       running   0.0.0.0:3000->3000/tcp
alertmanager  running   0.0.0.0:9093->9093/tcp
```

### MLflow registry — each model loaded `0.25 pt / model (8 models = 2 pts max)`

```bash
curl -s http://localhost:8000/models | python -m json.tool
```

Expected output — 1 entry per loaded model (0.25 pt each):
```json
[
  {"name": "yolov8",       "version": "1", "registered_at": "..."},
  {"name": "yolo26",       "version": "1", "registered_at": "..."},
  {"name": "rtdetr",       "version": "1", "registered_at": "..."},
  {"name": "rtdetrv2",     "version": "1", "registered_at": "..."},
  {"name": "rfdetr",       "version": "1", "registered_at": "..."},
  {"name": "dfine",        "version": "1", "registered_at": "..."},
  {"name": "deim-dfine",   "version": "1", "registered_at": "..."},
  {"name": "fusion-model", "version": "1", "registered_at": "..."}
]
```

### `GET /models` — version + MLflow registration date `0.25 pt`

```bash
curl -s http://localhost:8000/models | python -m json.tool
# Each entry must contain: name, version, registered_at
```

---

## Chap. 3 — Production Application `/5`

> `test_image.jpg` is provided in the professor's repo — place it at the root of your repo.

### Endpoints `/predict`, `/history`, `/health` working `0.75 pt`

```bash
curl -s http://localhost:8000/health | python -m json.tool
# Expected: {"status": "ok"}

curl -s -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "latitude=48.8566" \
  -F "longitude=2.3522" \
  -F "model_name=yolov8" \
  | python -m json.tool
# Expected: {"rubbish": ..., "confiance": 0.XX, "model_used": "yolov8", "timestamp": "..."}

curl -s http://localhost:8000/history | python -m json.tool
# Expected: list of detections
```

### Model selection — `model_name` forwarded + HTTP 422 if unknown `0.5 pt`

```bash
# Verify that model_used changes based on the requested model
curl -s -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" -F "latitude=48.8566" -F "longitude=2.3522" \
  -F "model_name=rtdetr" | python -m json.tool
# Expected: "model_used": "rtdetr"

# Unknown model -> 422
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" -F "latitude=48.8566" -F "longitude=2.3522" \
  -F "model_name=unknown_model"
# Expected: 422
```

### Input validation — explicit HTTP 422 `0.5 pt`

```bash
# Non-image file
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict \
  -F "file=@requirements.txt" -F "latitude=48.8566" -F "longitude=2.3522" \
  -F "model_name=yolov8"
# Expected: 422

# Invalid GPS coordinates
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" -F "latitude=999" -F "longitude=2.3522" \
  -F "model_name=yolov8"
# Expected: 422

# Verify the error message is explicit
curl -s -X POST http://localhost:8000/predict \
  -F "file=@requirements.txt" -F "latitude=48.8566" -F "longitude=2.3522" \
  -F "model_name=yolov8" | python -m json.tool
# Expected: {"detail": "...explicit message..."}
```

### `GET /models` — MLflow info `0.25 pt`

```bash
curl -s http://localhost:8000/models | python -m json.tool
# Verified by the same command as Chap. 2
```

### DB storage — `model_name` tracked `0.25 pt`

```bash
docker compose exec api \
  sqlite3 /data/app_detections.db \
  "SELECT timestamp, model_name, source FROM app_detections ORDER BY timestamp DESC LIMIT 5;"
# Expected: rows with model_name filled in
```

### Unit tests (min. 3) `0.75 pt`

```bash
pytest api/tests/test_unit.py -v
```

Expected output:
```
api/tests/test_unit.py::test_model_loads          PASSED
api/tests/test_unit.py::test_predict_valid_image  PASSED
api/tests/test_unit.py::test_predict_invalid_file PASSED
```

### Integration test (min. 1, via Docker) `0.5 pt`

```bash
pytest api/tests/test_integration.py -v
```

Expected output:
```
api/tests/test_integration.py::test_api_end_to_end PASSED
```

### Streamlit interface `1.5 pt` `[VISUAL]` — http://localhost:8501

- [ ] Model selection dropdown fed by `GET /models`
- [ ] Image upload + GPS input → result displayed (confidence, model used)
- [ ] Folium map with historical detections
- [ ] Filters by source, model, time period
- [ ] Distinct markers: red (manual upload) / orange (drone patrol)

---

## ETL Pipeline — Airflow `/3`

### DAG 1 `drone_mission_simulator` — automatic execution `0.5 pt`

```bash
# Check automatic runs (schedule every 5 min)
docker compose exec airflow airflow dags list-runs \
  --dag-id drone_mission_simulator --output table
# Expected: at least one run with state=success

# Verify that data was generated
docker compose exec airflow \
  sqlite3 /data/drone_patrol.db \
  "SELECT COUNT(*) FROM drone_detections;"
# Expected: count > 0
```

### DAG 2 `drone_patrol_sync` — 3 tasks without error `1 pt`

```bash
# Trigger manually if needed
docker compose exec airflow airflow dags trigger drone_patrol_sync

# Check overall status
docker compose exec airflow airflow dags list-runs \
  --dag-id drone_patrol_sync --output table
# Expected: state=success

# Get the latest run_id
RUN_ID=$(docker compose exec airflow airflow dags list-runs \
  --dag-id drone_patrol_sync --output json | python -m json.tool \
  | python -c "import sys,json; runs=json.load(sys.stdin); print(runs[0]['run_id'])")

# Check each of the 3 tasks
docker compose exec airflow airflow tasks states-for-dag-run \
  drone_patrol_sync "$RUN_ID" --output table
# Expected:
# extract   | success
# transform | success
# load      | success
```

### Filter `confiance >= 0.65` + flag `processed = 1` `0.5 pt`

```bash
# Verify minimum confidence of loaded detections
docker compose exec api \
  sqlite3 /data/app_detections.db \
  "SELECT MIN(confiance) FROM app_detections WHERE source='drone_patrol';"
# Expected: value >= 0.65

# Verify processed flag in source database
docker compose exec airflow \
  sqlite3 /data/drone_patrol.db \
  "SELECT processed, COUNT(*) FROM drone_detections GROUP BY processed;"
# Expected:
# 0 | N  (confidence < 0.65, not loaded)
# 1 | M  (loaded into app, M > 0)
```

### Airflow UI accessible `0.5 pt`

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080
# Expected: 200 or 302
```

`[VISUAL]` http://localhost:8080 — both DAGs visible with execution history

### Streamlit map — drone detections distinct `0.5 pt` `[VISUAL]` — http://localhost:8501

- [ ] Detections with `source=drone_patrol` visible on the map
- [ ] Visually distinct from manual uploads (different color or icon)

### [Bonus] DAG 2 triggered by DAG 1 via `TriggerDagRunOperator` `+0.5 pt`

```bash
docker compose exec airflow airflow dags list-runs \
  --dag-id drone_patrol_sync --output table
# The run_type column must show drone_mission_simulator as the trigger origin
```

---

## Chap. 4 — CI/CD `/3`

### Unit tests + integration test pass in pipeline `0.75 + 0.75 pt`

```bash
gh run view --repo <your-repo> --log | grep -E "(test_unit|test_integration|PASSED|FAILED)"
# Expected: pytest steps successful
```

`[VISUAL]` https://github.com/<your-repo>/actions — `pytest test_unit.py` and `pytest test_integration.py` steps in green

### Build + push Docker image to public registry `1 pt`

```bash
docker pull <registry>/<image>:latest
# Must download without error
```

### Pipeline green + badge in README `0.5 pt`

```bash
gh run list --repo <your-repo> --limit 1
# Expected: conclusion=success on main branch
```

`[VISUAL]` CI badge at the top of this README shows `passing`

---

## Chap. 5 — Observability `/2`

### Prometheus metrics — 4 metrics on `/metrics` `0.5 pt`

```bash
# Generate a few predictions first
for i in 1 2 3; do
  curl -s -X POST http://localhost:8000/predict \
    -F "file=@test_image.jpg" -F "latitude=48.8566" -F "longitude=2.3522" \
    -F "model_name=yolov8" > /dev/null
done

# Verify the 4 expected metrics
curl -s http://localhost:8000/metrics | grep "^ml_"
# Expected:
# ml_predictions_total X
# ml_inference_latency_seconds_count X
# ml_predictions_by_model_total{model="yolov8"} X
# ml_validation_errors_total X

# Verify Prometheus is scraping the API
curl -s "http://localhost:9090/api/v1/query?query=ml_predictions_total" \
  | python -m json.tool
# Expected: non-empty result
```

### Structured JSON logging `0.5 pt`

```bash
tail -5 logs/predictions.jsonl

# Verify each line is valid JSON with the required fields
python -c "
import json
with open('logs/predictions.jsonl') as f:
    lines = [l for l in f if l.strip()]
last = json.loads(lines[-1])
required = {'timestamp', 'model_name', 'confiance', 'source', 'latence_ms'}
assert required.issubset(last.keys()), f'Missing fields: {required - last.keys()}'
print(f'OK — {len(lines)} valid entries, last: {last}')
"
```

### Grafana dashboard — versioned JSON file + 4 panels `0.5 pt`

```bash
ls monitoring/grafana/dashboard.json
# File must exist

# Verify the dashboard has at least 4 panels
python -c "
import json
d = json.load(open('monitoring/grafana/dashboard.json'))
panels = d.get('panels', d.get('dashboard', {}).get('panels', []))
print(f'{len(panels)} panels found')
assert len(panels) >= 4, 'Less than 4 panels'
print('OK')
"
```

`[VISUAL]` http://localhost:3000 — "Waste Detection" dashboard with data on all 4 panels

### Alerting — rule defined and active `0.5 pt`

```bash
ls monitoring/alertmanager.yml
# File must exist

# Verify rules are loaded in Prometheus
curl -s http://localhost:9090/api/v1/rules | python -m json.tool
# Expected: at least one rule group with at least one rule

# Verify Alertmanager status
curl -s http://localhost:9093/api/v2/status | python -m json.tool
# Expected: {"cluster": {"status": "ready", ...}}
```

---

## Git & Quality `/1`

### Regular commits — both members contributing `0.5 pt`

```bash
git shortlog -sn
# Expected: both members with a significant number of commits

git log --oneline -15
```

### Correct `.gitignore` `0.25 pt`

```bash
# Verify required entries
grep -E "(__pycache__|\.venv|\.pt|\.db)" .gitignore

# Verify no forbidden artifacts are tracked
git ls-files | grep -E "(__pycache__|\.venv|\.pt$)"
# Expected: no output
```

### Professor invited to the private repo `0.25 pt` `[VISUAL]`

> GitHub → Settings → Collaborators → professor present before the deadline

---

## Bonus — Additional MLOps component `/+2`

**Component chosen:** [name]

**Justification:** [why relevant for this project]

**Implementation:** [technical description]

```bash
# Demonstration command
# [to be filled in]
```

---

## Quick grading checklist

```
COMMANDS
[ ] docker compose ps                              -> all containers running
[ ] docker build ./api && docker build ./app       -> build without error
[ ] curl /health                                   -> {"status": "ok"}
[ ] curl /models                                   -> 8 models with version + date
[ ] curl /predict (yolov8)                         -> rubbish + confiance + model_used
[ ] curl /predict (rtdetr)                         -> model_used = "rtdetr"
[ ] curl /predict (unknown_model)                  -> HTTP 422
[ ] curl /predict (requirements.txt)               -> HTTP 422
[ ] curl /predict (GPS=999)                        -> HTTP 422
[ ] curl /history                                  -> list of detections
[ ] sqlite3 app_detections.db (model_name)         -> model_name tracked
[ ] pytest test_unit.py                            -> PASSED (min. 3)
[ ] pytest test_integration.py                     -> PASSED (min. 1)
[ ] airflow dags list-runs drone_mission_sim       -> success
[ ] airflow tasks states drone_patrol_sync         -> extract+transform+load success
[ ] sqlite3 app_detections.db MIN(confiance)       -> >= 0.65
[ ] sqlite3 drone_patrol.db processed              -> 1 for loaded rows
[ ] curl http://localhost:8080                     -> 200/302
[ ] docker pull <registry>/<image>                 -> image available
[ ] gh run list                                    -> success on main
[ ] curl /metrics | grep ml_                       -> 4 metrics present
[ ] curl prometheus /api/v1/query                  -> ml_predictions_total non-empty
[ ] tail logs/predictions.jsonl                    -> valid JSON
[ ] python validate dashboard.json                 -> >= 4 panels
[ ] curl prometheus /api/v1/rules                  -> at least one rule
[ ] curl alertmanager /api/v2/status               -> ready
[ ] git shortlog -sn                               -> 2 contributors
[ ] git ls-files | grep pycache                    -> no output

VISUAL [VISUAL]
[ ] Streamlit http://localhost:8501                -> dropdown + map + filters + 2 colors
[ ] Airflow UI http://localhost:8080               -> 2 DAGs with run history
[ ] MLflow UI http://localhost:5000                -> 8 models in Production
[ ] Grafana http://localhost:3000                  -> 4 panels with data
[ ] GitHub Actions                                 -> badge + all steps green
[ ] GitHub Settings                                -> professor invited
```
