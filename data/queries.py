"""Transaction/scoring history queries."""

from sqlalchemy import case, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from data.db import get_engine
from data.models import Base, ScoredTransaction


def init_db(engine: Engine | None = None) -> None:
    """Create all tables if they don't exist. Called on API startup and
    consumer startup, so whichever process comes up first wins."""
    Base.metadata.create_all(engine or get_engine())


def log_scored_transaction(
    session: Session,
    *,
    payload: dict,
    amount: float,
    risk_score: float,
    predicted_fraud: bool,
    actual_label: int | None,
) -> ScoredTransaction:
    record = ScoredTransaction(
        amount=amount,
        payload=payload,
        risk_score=risk_score,
        predicted_fraud=predicted_fraud,
        actual_label=actual_label,
    )
    session.add(record)
    session.commit()
    return record


def recent_transactions(
    session: Session, *, limit: int = 50, offset: int = 0
) -> list[ScoredTransaction]:
    """Most recent first — what the M4 dashboard feed consumes."""
    stmt = (
        select(ScoredTransaction)
        .order_by(ScoredTransaction.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(session.execute(stmt).scalars().all())


def transaction_stats(session: Session) -> dict:
    total, flagged, avg_risk = session.execute(
        select(
            func.count(ScoredTransaction.id),
            func.coalesce(
                func.sum(case((ScoredTransaction.predicted_fraud, 1), else_=0)), 0
            ),
            func.avg(ScoredTransaction.risk_score),
        )
    ).one()
    return {
        "total_scored": int(total),
        "total_flagged": int(flagged),
        "avg_risk_score": float(avg_risk) if avg_risk is not None else 0.0,
    }
