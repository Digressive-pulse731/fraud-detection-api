"""Tests for POST /model/retrain: FastAPI TestClient + in-memory SQLite +
a tiny synthetic dataset file on disk (no Postgres, no real Kaggle CSV)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.deps as deps
import api.main as main
from api.deps import get_db
from api.main import app
from data.models import Base


@pytest.fixture
def retrain_client(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "V1": rng.normal(0, 1, 50),
            "V2": rng.normal(0, 1, 50),
            "Amount": rng.exponential(50, 50),
            "Class": 0,
        }
    )
    dataset_path = tmp_path / "creditcard.csv"
    df.to_csv(dataset_path, index=False)
    model_path = tmp_path / "model.joblib"

    # These are plain module attributes read at call time (not bound
    # function defaults), so monkeypatching them redirects both the
    # retrain endpoint and get_model()'s cache-miss load.
    monkeypatch.setattr(main, "DATASET_PATH", dataset_path)
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(deps, "MODEL_PATH", model_path)

    monkeypatch.setenv("FRAUD_THRESHOLD", "0.7")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    def override_get_db():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    deps._model = None  # start with a clean cache for each test

    with TestClient(app) as test_client:
        yield test_client, dataset_path, model_path

    app.dependency_overrides.clear()
    deps._model = None


def test_retrain_succeeds_and_updates_the_cached_model(retrain_client):
    client, _dataset_path, model_path = retrain_client

    # Simulate a stale model already cached from a prior request.
    deps._model = "stale-sentinel-not-a-real-model"

    response = client.post("/model/retrain")
    assert response.status_code == 200

    # The retrain call must have invalidated the cache immediately.
    assert deps._model is None
    assert model_path.exists()

    # The next request that needs the model must load the fresh one —
    # not the stale sentinel — proven by successfully scoring with it.
    score_response = client.post("/transactions", json={"V1": 0.1, "V2": -0.2, "Amount": 10.0})
    assert score_response.status_code == 201
    assert isinstance(deps._model, IsolationForest)
    assert set(deps._model.feature_names_in_) == {"V1", "V2", "Amount"}


def test_retrain_with_missing_dataset_returns_clear_400(retrain_client, tmp_path):
    client, _dataset_path, _model_path = retrain_client

    missing_path = tmp_path / "does-not-exist.csv"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(main, "DATASET_PATH", missing_path)
        response = client.post("/model/retrain")

    assert response.status_code == 400
    assert "Kaggle" in response.json()["detail"] or "not found" in response.json()["detail"]


def test_retrain_response_includes_expected_summary_fields(retrain_client):
    client, dataset_path, model_path = retrain_client
    response = client.post("/model/retrain")
    body = response.json()

    assert body["status"] == "retrained"
    assert body["training_rows"] == 50  # matches the synthetic dataset written above
    assert body["seconds_taken"] >= 0
    assert body["model_path"] == str(model_path)
