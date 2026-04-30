"""
Foundation stub — only /health is wired. Teammate A replaces this with:
  - /predict (POST: image + GPS + model_name → rubbish, confiance, model_used, timestamp)
  - /history (GET)
  - /models (GET — lists MLflow Production models)
  - /metrics (Prometheus exposition)
  - SQLite persistence at $APP_DB_PATH
  - JSON logging at logs/predictions.jsonl
"""

from fastapi import FastAPI

app = FastAPI(title="Waste Detection API", version="0.0.1-foundation")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
