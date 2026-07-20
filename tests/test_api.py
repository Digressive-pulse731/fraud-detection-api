"""API endpoint tests: FastAPI TestClient + in-memory SQLite + a small
synthetic model — no Postgres, Kafka, or the real model file needed."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.deps import get_db, get_model
from api.main import app
from data.models import Base
from model.train import train


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "V1": rng.normal(0, 1, 300),
            "V2": rng.normal(0, 1, 300),
            "Amount": rng.exponential(50, 300),
        }
    )
    return train(df)


@pytest.fixture
def client(model, monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
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
    app.dependency_overrides[get_model] = lambda: model
    monkeypatch.setenv("FRAUD_THRESHOLD", "0.7")
    # Never let a fraud-path test reach the real Telegram API
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    # lifespan's init_db targets Postgres; the TestClient context below
    # would run it, so keep it harmless by pointing tests at raise-free env
    with TestClient(app, raise_server_exceptions=True) as test_client:
        yield test_client
    app.dependency_overrides.clear()


TXN = {"V1": 0.1, "V2": -0.3, "Amount": 25.0, "Class": 0}


def test_submit_transaction_scores_and_returns_result(client):
    response = client.post("/transactions", json=TXN)
    assert response.status_code == 201
    body = response.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["predicted_fraud"] is False
    assert body["threshold"] == 0.7
    assert body["id"] == 1
    assert body["alert_sent"] is False


def test_list_transactions_returns_logged_records(client):
    client.post("/transactions", json=TXN)
    client.post("/transactions", json={**TXN, "Amount": 99.0})

    response = client.get("/transactions")
    assert response.status_code == 200
    body = response.json()
    assert len(body["transactions"]) == 2
    # Most recent first
    assert body["transactions"][0]["amount"] == 99.0
    assert body["transactions"][1]["amount"] == 25.0
    assert body["transactions"][0]["payload"]["V1"] == 0.1


def test_pagination(client):
    for i in range(5):
        client.post("/transactions", json={**TXN, "Amount": float(i)})

    page = client.get("/transactions", params={"limit": 2, "offset": 1}).json()
    assert page["limit"] == 2 and page["offset"] == 1
    amounts = [t["amount"] for t in page["transactions"]]
    assert amounts == [3.0, 2.0]  # newest is Amount=4.0, skipped by offset=1


def test_stats_aggregates_correctly(client, monkeypatch):
    client.post("/transactions", json=TXN)
    # Force one guaranteed fraud by dropping the threshold to 0
    monkeypatch.setenv("FRAUD_THRESHOLD", "0.0")
    client.post("/transactions", json={**TXN, "Amount": 50.0})

    stats = client.get("/transactions/stats").json()
    assert stats["total_scored"] == 2
    assert stats["total_flagged"] == 1
    assert 0.0 < stats["avg_risk_score"] < 1.0


def test_malformed_transaction_returns_400(client):
    response = client.post("/transactions", json={"V1": "not-a-number"})
    assert response.status_code == 400
    assert "numeric" in response.json()["detail"]

    response = client.post("/transactions", json={})
    assert response.status_code == 400
