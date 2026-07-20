"""Tests for model/train.py using a small synthetic dataset — the real
Kaggle CSV is never needed here, keeping the suite fast."""

import joblib
import numpy as np
import pandas as pd
import pytest

from model.score import score_transaction
from model.train import LABEL_COLUMN, load_dataset, save_model, train


def synthetic_transactions(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "V1": rng.normal(0, 1, n),
            "V2": rng.normal(0, 1, n),
            "Amount": rng.exponential(50, n),
            LABEL_COLUMN: np.zeros(n, dtype=int),
        }
    )


def test_training_produces_model_that_scores(tmp_path):
    df = synthetic_transactions()
    model = train(df)

    # Label column must not be used as a feature
    assert LABEL_COLUMN not in model.feature_names_in_

    # Round-trip through joblib, the same way production loads it
    saved = save_model(model, tmp_path / "model.joblib")
    loaded = joblib.load(saved)

    typical = {"V1": 0.1, "V2": -0.2, "Amount": 40.0}
    risk = score_transaction(typical, loaded)
    assert 0.0 <= risk <= 1.0

    # A wildly out-of-distribution transaction must score as riskier
    outlier = {"V1": 50.0, "V2": -60.0, "Amount": 100_000.0}
    assert score_transaction(outlier, loaded) > risk


def test_missing_dataset_raises_clear_error(tmp_path):
    missing = tmp_path / "creditcard.csv"
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_dataset(missing)
