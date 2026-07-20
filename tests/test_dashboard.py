"""Tests for the dashboard's pure data-transform functions — no Dash
app, no HTTP, just plain dicts in, chart/table-ready shapes out."""

from dashboard.transforms import (
    confusion_counts,
    cumulative_fraud_series,
    format_table_rows,
    risk_scores,
)

TRANSACTIONS = [
    {
        "id": 3,
        "amount": 12.5,
        "risk_score": 0.92,
        "predicted_fraud": True,
        "actual_label": 1,
        "scored_at": "2026-07-20T10:02:00+00:00",
        "payload": {},
    },
    {
        "id": 2,
        "amount": 8.0,
        "risk_score": 0.05,
        "predicted_fraud": False,
        "actual_label": 0,
        "scored_at": "2026-07-20T10:01:00+00:00",
        "payload": {},
    },
    {
        "id": 1,
        "amount": 40.25,
        "risk_score": 0.81,
        "predicted_fraud": True,
        "actual_label": 0,  # false positive
        "scored_at": "2026-07-20T10:00:00+00:00",
        "payload": {},
    },
]


def test_risk_scores_extracts_in_given_order():
    assert risk_scores(TRANSACTIONS) == [0.92, 0.05, 0.81]


def test_cumulative_fraud_series_sorts_chronologically_and_accumulates():
    times, counts = cumulative_fraud_series(TRANSACTIONS)
    # Input is newest-first; output must be oldest-first
    assert times == [
        "2026-07-20T10:00:00+00:00",
        "2026-07-20T10:01:00+00:00",
        "2026-07-20T10:02:00+00:00",
    ]
    # txn 1 (fraud) -> 1, txn 2 (legit) -> stays 1, txn 3 (fraud) -> 2
    assert counts == [1, 1, 2]


def test_confusion_counts_breaks_down_predictions_correctly():
    counts = confusion_counts(TRANSACTIONS)
    assert counts == {
        "true_positive": 1,   # id=3: predicted fraud, actually fraud
        "false_positive": 1,  # id=1: predicted fraud, actually legit
        "false_negative": 0,
        "true_negative": 1,   # id=2: predicted legit, actually legit
    }


def test_confusion_counts_ignores_unlabeled_records():
    unlabeled = TRANSACTIONS + [
        {
            "id": 4,
            "amount": 5.0,
            "risk_score": 0.5,
            "predicted_fraud": True,
            "actual_label": None,
            "scored_at": "2026-07-20T10:03:00+00:00",
            "payload": {},
        }
    ]
    counts = confusion_counts(unlabeled)
    assert sum(counts.values()) == 3  # the unlabeled record contributes nothing


def test_format_table_rows_labels_flag_and_actual_without_relying_on_color():
    rows = format_table_rows(TRANSACTIONS)
    assert rows[0]["flag"] == "\U0001f6a9 Fraud"
    assert rows[0]["actual_label"] == "fraud"
    assert rows[1]["flag"] == "OK"
    assert rows[1]["actual_label"] == "legit"
    assert rows[0]["risk_score"] == 0.92
    assert rows[0]["scored_at"] == "2026-07-20 10:02:00"
