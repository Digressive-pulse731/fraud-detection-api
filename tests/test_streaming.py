"""Tests for the producer/consumer logic — all Kafka and DB interactions
are mocked; no broker or database is needed."""

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from streaming.consumer import handle_message
from streaming.producer import LABEL_COLUMN, row_to_message, stream
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
def session():
    """Fake SQLAlchemy session: records what would be persisted and
    assigns an id on commit, like the real autoincrement PK would."""
    fake = MagicMock()

    def fake_commit():
        for call in fake.add.call_args_list:
            record = call.args[0]
            if record.id is None:
                record.id = 1

    fake.commit.side_effect = fake_commit
    return fake


def sample_row() -> pd.Series:
    return pd.Series({"V1": 0.5, "V2": -1.2, "Amount": 42.5, LABEL_COLUMN: 1})


def test_row_to_message_shape():
    message = row_to_message(sample_row())
    assert message == {"V1": 0.5, "V2": -1.2, "Amount": 42.5, LABEL_COLUMN: 1}
    assert isinstance(message[LABEL_COLUMN], int)  # ground-truth label kept as int
    json.dumps(message)  # must be JSON-serializable as-is


def test_producer_respects_limit():
    df = pd.DataFrame([sample_row()] * 20)
    producer = MagicMock()
    sent = stream(df, producer, limit=5, interval=0)
    assert sent == 5
    assert producer.send.call_count == 5
    producer.flush.assert_called_once()


def test_producer_limit_zero_streams_everything():
    df = pd.DataFrame([sample_row()] * 7)
    producer = MagicMock()
    assert stream(df, producer, limit=0, interval=0) == 7
    assert producer.send.call_count == 7


def test_consumer_scores_and_logs_valid_message(model, session):
    raw = json.dumps({"V1": 0.1, "V2": -0.3, "Amount": 25.0, LABEL_COLUMN: 0}).encode()
    record = handle_message(raw, model, session, threshold=0.7)

    assert record is not None
    assert 0.0 <= record.risk_score <= 1.0
    assert record.amount == 25.0
    assert record.actual_label == 0
    session.add.assert_called_once_with(record)
    session.commit.assert_called_once()


def test_consumer_skips_malformed_message(model, session):
    for raw in (b"not json at all", b'"a bare string"', b"\xff\xfe"):
        assert handle_message(raw, model, session, threshold=0.7) is None
    session.add.assert_not_called()
    session.commit.assert_not_called()


def test_predicted_fraud_follows_threshold(model, session):
    raw = json.dumps({"V1": 0.1, "V2": -0.3, "Amount": 25.0, LABEL_COLUMN: 0}).encode()

    # Threshold below any possible score -> always flagged
    flagged = handle_message(raw, model, session, threshold=-0.1)
    assert flagged.predicted_fraud is True

    # Threshold above any possible score -> never flagged
    clean = handle_message(raw, model, session, threshold=1.1)
    assert clean.predicted_fraud is False
