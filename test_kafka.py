"""Smoke test: verify the Kafka broker on localhost:9092 is reachable.

Creates "test-topic" if missing, publishes one message, consumes it back.
Run with the Docker Compose Kafka service up: python test_kafka.py
"""

import json
import time
import uuid

from kafka import KafkaAdminClient, KafkaConsumer, KafkaProducer
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError

BROKER = "localhost:9092"
TOPIC = "test-topic"


def ensure_topic() -> None:
    admin = KafkaAdminClient(bootstrap_servers=BROKER)
    try:
        admin.create_topics([NewTopic(name=TOPIC, num_partitions=1, replication_factor=1)])
        print(f"[1/3] Created topic '{TOPIC}'")
    except TopicAlreadyExistsError:
        print(f"[1/3] Topic '{TOPIC}' already exists")
    finally:
        admin.close()


def publish() -> dict:
    message = {"id": str(uuid.uuid4()), "sent_at": time.time(), "text": "kafka smoke test"}
    producer = KafkaProducer(
        bootstrap_servers=BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    result = producer.send(TOPIC, message).get(timeout=10)
    producer.flush()
    producer.close()
    print(f"[2/3] Published to {result.topic} partition {result.partition} offset {result.offset}: {message}")
    return message


def consume(expected: dict) -> None:
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKER,
        group_id=f"smoke-test-{uuid.uuid4()}",
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=15000,
    )
    for record in consumer:
        if record.value.get("id") == expected["id"]:
            consumer.close()
            print(f"[3/3] Consumed it back from offset {record.offset}: {record.value}")
            print("\nSUCCESS: broker reachable, publish + consume round-trip works.")
            return
    consumer.close()
    raise SystemExit("FAILED: timed out without seeing the published message back.")


if __name__ == "__main__":
    ensure_topic()
    consume(publish())
