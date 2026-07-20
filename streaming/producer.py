"""Kafka producer: simulates a live transaction feed.

Reads data/creditcard.csv and publishes each row as a JSON message to
the "transactions" topic on an interval. Publishes only — never imports
consumer internals.

Run standalone:  python -m streaming.producer [--limit N] [--interval S]
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("producer")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
TOPIC = "transactions"
LABEL_COLUMN = "Class"

DEFAULT_LIMIT = int(os.environ.get("PRODUCER_LIMIT", "100"))  # 0 = full dataset
DEFAULT_INTERVAL = float(os.environ.get("PRODUCER_INTERVAL", "1.0"))


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    # Deliberately self-contained (not imported from model/train.py):
    # the producer stays decoupled from training internals per convention.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from {KAGGLE_URL} "
            f"and place creditcard.csv there."
        )
    return pd.read_csv(path)


def row_to_message(row: pd.Series) -> dict:
    """Convert a dataset row to a JSON-safe message dict.

    The original Class label is included as ground truth so we can later
    evaluate model predictions against it (testing/dashboard). A real
    production feed would not carry this field.
    """
    message = {name: float(value) for name, value in row.items()}
    if LABEL_COLUMN in message:
        message[LABEL_COLUMN] = int(message[LABEL_COLUMN])
    return message


def stream(
    df: pd.DataFrame,
    producer: KafkaProducer,
    *,
    topic: str = TOPIC,
    limit: int = DEFAULT_LIMIT,
    interval: float = DEFAULT_INTERVAL,
) -> int:
    """Publish rows to Kafka; returns the number of messages sent.

    limit=0 streams the entire DataFrame.
    """
    to_send = df if limit <= 0 else df.head(limit)
    sent = 0
    for _, row in to_send.iterrows():
        message = row_to_message(row)
        producer.send(topic, json.dumps(message).encode("utf-8"))
        sent += 1
        if sent % 25 == 0 or sent == len(to_send):
            logger.info("Published %d/%d transactions", sent, len(to_send))
        if interval > 0:
            time.sleep(interval)
    producer.flush()
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream transactions to Kafka")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max transactions to send, 0 = full dataset (default {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"Seconds between messages (default {DEFAULT_INTERVAL})",
    )
    parser.add_argument("--topic", default=TOPIC)
    args = parser.parse_args()

    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    df = load_dataset()
    producer = KafkaProducer(bootstrap_servers=bootstrap)
    logger.info(
        "Streaming %s transactions to '%s' on %s every %.2fs",
        "all" if args.limit <= 0 else args.limit,
        args.topic,
        bootstrap,
        args.interval,
    )
    try:
        sent = stream(df, producer, topic=args.topic, limit=args.limit, interval=args.interval)
        logger.info("Done: %d transactions published", sent)
    except KeyboardInterrupt:
        logger.info("Interrupted — flushing and exiting")
        producer.flush()
    finally:
        producer.close()


if __name__ == "__main__":
    main()
