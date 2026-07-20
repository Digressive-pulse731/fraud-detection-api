"""SQLAlchemy table models."""

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ScoredTransaction(Base):
    """One row per scored transaction — every transaction is logged
    regardless of fraud status, so the dashboard can show full history."""

    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_fraud: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # Ground-truth label carried through from the dataset for offline
    # evaluation only — a real feed would not have this.
    actual_label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
