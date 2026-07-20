"""FastAPI application.

Health, manual transaction submission, history + stats queries, and the
model retrain endpoint.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import joblib
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from alerts.telegram import send_fraud_alert
from api.deps import get_db, get_model, reset_model
from data.models import ScoredTransaction
from data.queries import (
    init_db,
    log_scored_transaction,
    recent_transactions,
    transaction_stats,
)
from model.score import score_transaction
from model.train import DATASET_PATH, MODEL_PATH
from model.train import load_dataset as load_training_dataset
from model.train import save_model
from model.train import train as train_model

logger = logging.getLogger("api")

LABEL_COLUMN = "Class"


def fraud_threshold() -> float:
    return float(os.environ.get("FRAUD_THRESHOLD", "0.7"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
        logger.info("Database tables ensured")
    except Exception:
        # The consumer also runs init_db on startup; don't take /health
        # down just because the DB was briefly unreachable at boot.
        logger.exception("Could not initialize database tables")
    yield


app = FastAPI(title="fraud-detection-api", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    model_file_exists = MODEL_PATH.exists()
    model_loaded = False
    model_error = None
    if model_file_exists:
        try:
            joblib.load(MODEL_PATH)
            model_loaded = True
        except Exception as exc:  # a corrupt artifact must not take /health down
            model_error = str(exc)

    payload = {
        "status": "ok",
        "model_file_exists": model_file_exists,
        "model_loaded": model_loaded,
    }
    if model_error:
        payload["model_error"] = model_error
    return payload


def _serialize(record: ScoredTransaction) -> dict:
    return {
        "id": record.id,
        "amount": record.amount,
        "risk_score": record.risk_score,
        "predicted_fraud": record.predicted_fraud,
        "actual_label": record.actual_label,
        "scored_at": record.scored_at.isoformat() if record.scored_at else None,
        "payload": record.payload,
    }


@app.post("/transactions", status_code=201)
def submit_transaction(
    payload: dict,
    db: Session = Depends(get_db),
    model=Depends(get_model),
) -> dict:
    """Score one manually submitted transaction — same path as the
    consumer: score, log to DB regardless of outcome, alert if fraud."""
    if not payload:
        raise HTTPException(status_code=400, detail="Transaction body must be a non-empty JSON object")
    try:
        numeric = {key: float(value) for key, value in payload.items()}
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="All transaction values must be numeric")

    actual_label = numeric.pop(LABEL_COLUMN, None)
    risk_score = score_transaction(numeric, model)
    threshold = fraud_threshold()
    predicted_fraud = risk_score > threshold

    record = log_scored_transaction(
        db,
        payload=payload,
        amount=numeric.get("Amount", 0.0),
        risk_score=risk_score,
        predicted_fraud=predicted_fraud,
        actual_label=int(actual_label) if actual_label is not None else None,
    )

    alert_sent = False
    if predicted_fraud:
        try:
            send_fraud_alert(payload, risk_score)
            alert_sent = True
        except Exception:
            logger.exception("Fraud alert failed for txn id=%d (already saved)", record.id)

    return {
        "id": record.id,
        "risk_score": risk_score,
        "predicted_fraud": predicted_fraud,
        "threshold": threshold,
        "alert_sent": alert_sent,
    }


@app.post("/model/retrain")
def retrain_model() -> dict:
    """Re-train against the current dataset and atomically swap the model
    file in (save_model's os.replace — no request ever sees a half-written
    file). Synchronous: training this dataset takes seconds, not minutes."""
    try:
        df = load_training_dataset(DATASET_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.monotonic()
    model = train_model(df)
    saved_to = save_model(model, MODEL_PATH)
    elapsed = time.monotonic() - started

    # Drop the cached model so the next request loads the freshly saved file.
    reset_model()

    return {
        "status": "retrained",
        "training_rows": len(df),
        "seconds_taken": round(elapsed, 2),
        "model_path": str(saved_to),
    }


@app.get("/transactions/stats")
def get_stats(db: Session = Depends(get_db)) -> dict:
    return transaction_stats(db)


@app.get("/transactions")
def list_transactions(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> dict:
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    records = recent_transactions(db, limit=limit, offset=offset)
    return {
        "limit": limit,
        "offset": offset,
        "transactions": [_serialize(record) for record in records],
    }
