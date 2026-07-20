"""Kafka consumer: scores transactions and logs them to PostgreSQL.

Subscribes to the "transactions" topic, scores each message via
model/score.py, and logs every transaction (fraud or not) so the
dashboard can show full history. Reads and scores only — never imports
producer internals.

Run standalone:  python -m streaming.consumer
"""

import json
import logging
import os

from kafka import KafkaConsumer
from sqlalchemy.orm import Session

from alerts.telegram import send_fraud_alert
from data.queries import init_db, log_scored_transaction
from model.score import load_model, score_transaction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("consumer")

TOPIC = "transactions"
GROUP_ID = "fraud-scoring"
LABEL_COLUMN = "Class"

DEFAULT_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.7"))


def handle_message(raw: bytes, model, session: Session, threshold: float = DEFAULT_THRESHOLD):
    """Score one raw Kafka message and log it to the database.

    Returns the persisted record, or None if the message was malformed —
    a single bad message must never crash the consumer loop.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
        actual_label = payload.get(LABEL_COLUMN)
        features = {k: v for k, v in payload.items() if k != LABEL_COLUMN}
        risk_score = score_transaction(features, model)
    except Exception:
        logger.exception("Skipping malformed message: %.200r", raw)
        return None

    record = log_scored_transaction(
        session,
        payload=payload,
        amount=float(payload.get("Amount", 0.0)),
        risk_score=risk_score,
        predicted_fraud=risk_score > threshold,
        actual_label=int(actual_label) if actual_label is not None else None,
    )
    logger.info(
        "Scored txn id=%s amount=%.2f risk=%.3f fraud=%s label=%s",
        record.id,
        record.amount,
        record.risk_score,
        record.predicted_fraud,
        record.actual_label,
    )

    # The transaction is already persisted above — an alert failure must
    # never crash the loop or undo the DB write (apiwatch/AutoReport pattern).
    if record.predicted_fraud:
        try:
            send_fraud_alert(payload, risk_score)
            logger.info("Fraud alert sent for txn id=%s", record.id)
        except Exception:
            logger.exception("Fraud alert failed for txn id=%s (already saved)", record.id)
    return record


def main() -> None:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    threshold = DEFAULT_THRESHOLD

    init_db()
    model = load_model()
    from data.db import get_session_factory  # imported late: needs env configured

    session_factory = get_session_factory()

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=bootstrap,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    logger.info("Consuming '%s' from %s (threshold=%.2f)", TOPIC, bootstrap, threshold)
    try:
        for message in consumer:
            session = session_factory()
            try:
                handle_message(message.value, model, session, threshold)
            except Exception:
                # DB or other unexpected failure: log and keep consuming.
                logger.exception("Failed to process message at offset %d", message.offset)
                session.rollback()
            finally:
                session.close()
    except KeyboardInterrupt:
        logger.info("Interrupted — closing consumer")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
