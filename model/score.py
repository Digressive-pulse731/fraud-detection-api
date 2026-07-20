"""Load model.joblib and score a single transaction.

score_transaction is a pure function (transaction dict + model in,
risk score out) so it is testable without Kafka or a running consumer.
"""

import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

# Steepness of the sigmoid that maps Isolation Forest decision values
# (roughly -0.1..0.3 on the Kaggle data, negative = anomalous) onto a
# 0..1 risk score. Calibrated at 25 so the default FRAUD_THRESHOLD=0.7
# catches the most anomalous frauds (~7.5% of labeled frauds) at a
# ~0.03% false-positive rate on legitimate transactions.
_SIGMOID_SCALE = 25.0


def load_model(path: Path = MODEL_PATH) -> IsolationForest:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Train it first: python train_model.py"
        )
    return joblib.load(path)


def score_transaction(transaction: dict, model: IsolationForest) -> float:
    """Return a fraud risk score in [0, 1]; higher means more anomalous.

    Missing features default to 0.0; extra keys in the transaction are ignored.
    """
    feature_names = list(getattr(model, "feature_names_in_", transaction.keys()))
    row = pd.DataFrame([{name: transaction.get(name, 0.0) for name in feature_names}])
    decision = float(model.decision_function(row)[0])
    return 1.0 / (1.0 + math.exp(_SIGMOID_SCALE * decision))
