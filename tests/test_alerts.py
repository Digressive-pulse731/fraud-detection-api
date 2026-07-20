"""Tests for Telegram alerting — all HTTP calls are mocked."""

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import alerts.telegram
import streaming.consumer
from alerts.telegram import TelegramAlertError, send_fraud_alert
from model.train import train
from streaming.consumer import handle_message


@pytest.fixture(autouse=True)
def telegram_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")


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


def fake_response(ok=True, status_code=200, text="ok"):
    response = MagicMock()
    response.ok = ok
    response.status_code = status_code
    response.text = text
    return response


def test_successful_alert_send(monkeypatch):
    post = MagicMock(return_value=fake_response())
    monkeypatch.setattr(alerts.telegram.requests, "post", post)

    send_fraud_alert({"Amount": 999.99, "Time": 3600.0, "V1": -5.2}, risk_score=0.91)

    post.assert_called_once()
    url = post.call_args.args[0]
    body = post.call_args.kwargs["json"]
    assert "bottest-token/sendMessage" in url
    assert body["chat_id"] == "12345"
    assert "0.910" in body["text"]
    assert "$999.99" in body["text"]


def test_missing_credentials_raises_clear_error(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    with pytest.raises(TelegramAlertError, match="TELEGRAM_BOT_TOKEN"):
        send_fraud_alert({"Amount": 10.0}, risk_score=0.9)

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.delenv("TELEGRAM_CHAT_ID")
    with pytest.raises(TelegramAlertError, match="TELEGRAM_CHAT_ID"):
        send_fraud_alert({"Amount": 10.0}, risk_score=0.9)


def test_telegram_api_failure_raises_clear_error(monkeypatch):
    post = MagicMock(return_value=fake_response(ok=False, status_code=502, text="bad gateway"))
    monkeypatch.setattr(alerts.telegram.requests, "post", post)

    with pytest.raises(TelegramAlertError, match="502"):
        send_fraud_alert({"Amount": 10.0}, risk_score=0.9)


def test_consumer_saves_transaction_even_when_alert_fails(monkeypatch, model):
    """The critical invariant: a failing alert must neither crash the
    consumer loop nor prevent the DB write (which happens first)."""
    failing_alert = MagicMock(side_effect=TelegramAlertError("boom"))
    monkeypatch.setattr(streaming.consumer, "send_fraud_alert", failing_alert)

    session = MagicMock()
    session.commit.side_effect = lambda: setattr(session.add.call_args.args[0], "id", 1)
    raw = json.dumps({"V1": 0.1, "V2": -0.3, "Amount": 25.0, "Class": 1}).encode()
    # threshold below any possible score -> guaranteed fraud -> alert path taken
    record = handle_message(raw, model, session, threshold=-0.1)

    assert record is not None
    assert record.predicted_fraud is True
    failing_alert.assert_called_once()          # alert was attempted...
    session.add.assert_called_once_with(record)  # ...but the txn was saved anyway
    session.commit.assert_called_once()


def test_consumer_sends_alert_only_above_threshold(monkeypatch, model):
    alert = MagicMock()
    monkeypatch.setattr(streaming.consumer, "send_fraud_alert", alert)

    raw = json.dumps({"V1": 0.1, "V2": -0.3, "Amount": 25.0, "Class": 0}).encode()
    handle_message(raw, model, MagicMock(), threshold=1.1)  # can never exceed
    alert.assert_not_called()

    handle_message(raw, model, MagicMock(), threshold=-0.1)  # always exceeds
    alert.assert_called_once()
