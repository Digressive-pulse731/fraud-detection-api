"""Shared FastAPI dependencies (DB session, model)."""

from collections.abc import Generator

from fastapi import HTTPException
from sqlalchemy.orm import Session

from data.db import get_session_factory
from model.score import load_model
from model.train import MODEL_PATH

_model = None


def get_db() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def get_model():
    """Load the model once and cache it for the process lifetime.
    Invalidated by reset_model() after a successful retrain."""
    global _model
    if _model is None:
        try:
            _model = load_model(MODEL_PATH)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet — run: python train_model.py",
            ) from exc
    return _model


def reset_model() -> None:
    """Drop the cached model so the next get_model() call reloads it from
    disk — called by POST /model/retrain after a successful atomic swap."""
    global _model
    _model = None
