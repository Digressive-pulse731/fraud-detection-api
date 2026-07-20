"""Isolation Forest training logic.

Used by train_model.py (standalone CLI) and, from M5 on, by the
POST /model/retrain endpoint. The model file is a build artifact:
always regenerate it via `python train_model.py`, never hand-edit.
"""

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "model.joblib"
LABEL_COLUMN = "Class"
KAGGLE_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

# Known fraud rate of the Kaggle dataset (~0.17%); used as the expected
# anomaly proportion so the decision threshold lands near the real base rate.
CONTAMINATION = 0.0017


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Download the Kaggle Credit Card Fraud dataset from {KAGGLE_URL} "
            f"(file: creditcard.csv) and place it at {path}, then re-run."
        )
    return pd.read_csv(path)


def train(df: pd.DataFrame) -> IsolationForest:
    """Fit an Isolation Forest on the feature columns (label dropped if present)."""
    features = df.drop(columns=[LABEL_COLUMN], errors="ignore")
    model = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features)
    return model


def save_model(model: IsolationForest, path: Path = MODEL_PATH) -> Path:
    """Serialize the model, swapping into place atomically so a reader
    never sees a half-written file (the contract the M5 retrain endpoint
    depends on)."""
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    joblib.dump(model, tmp_path)
    os.replace(tmp_path, path)
    return path


def main(dataset_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH) -> Path:
    df = load_dataset(dataset_path)
    model = train(df)
    saved_to = save_model(model, model_path)

    features = df.drop(columns=[LABEL_COLUMN], errors="ignore")
    flagged = int((model.predict(features) == -1).sum())
    print(f"Trained on {len(df):,} transactions ({features.shape[1]} features)")
    print(f"Self-check: {flagged:,} flagged as anomalies ({flagged / len(df):.2%})")
    if LABEL_COLUMN in df.columns:
        print(f"Labeled frauds in dataset: {int(df[LABEL_COLUMN].sum()):,}")
    print(f"Model saved to {saved_to}")
    return saved_to
