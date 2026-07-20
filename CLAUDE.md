# fraud-detection-api — Project Context

## Overview
Real-time fraud detection system: a Kafka-based streaming pipeline simulates
incoming credit card transactions, an Isolation Forest model scores each
transaction for fraud risk in real time, high-risk transactions trigger a
Telegram alert, and a Plotly Dash dashboard visualizes the live feed and
detection results. Level: Hard-Expert. Sixth project in the roadmap and the
first to involve machine learning — reuses Telegram alert and Docker
patterns from apiwatch/AutoReport/realtime-analytics-platform, but the
Kafka streaming layer and ML model are new territory.

## Tech Stack
- Python 3.12+
- FastAPI (REST endpoints — health, manual transaction submission, model
  retrain trigger)
- Kafka (KRaft mode — no Zookeeper) via kafka-python or confluent-kafka
- scikit-learn (Isolation Forest — unsupervised anomaly detection)
- Pandas (data loading, feature prep)
- joblib (model serialization — save/load the trained model file)
- PostgreSQL + SQLAlchemy (transaction/scoring history log)
- Plotly Dash (live dashboard — reuse WebSocket/callback patterns from
  realtime-analytics-platform where they fit)
- Telegram Bot API via requests (fraud alerts — reuse apiwatch/AutoReport
  pattern)
- Docker + Docker Compose (kafka, db, app, dashboard services)
- pytest

## Commands
- Run via Docker: `docker compose up --build`
- Train the model standalone: `python train_model.py`
- Test: `pytest`
- Lint: `ruff check .`

## Project Structure
```
fraud-detection-api/
├── train_model.py               # Standalone script: trains Isolation Forest
│                                   on the Kaggle dataset, saves model.joblib
├── app.py                          # Plotly Dash dashboard entry point
├── api/
│   ├── main.py                       # FastAPI: health, manual submit, retrain
│   └── deps.py                          # shared dependencies (DB session, etc.)
├── streaming/
│   ├── producer.py                       # Kafka producer: reads dataset, publishes
│   │                                        to "transactions" topic on an interval
│   └── consumer.py                          # Kafka consumer: reads topic, scores
│                                               via the model, logs + alerts
├── model/
│   ├── train.py                                # Training logic (used by train_model.py
│   │                                              and the retrain endpoint)
│   ├── score.py                                  # Load model.joblib, score a transaction
│   └── model.joblib                                # Trained model artifact (git-ignored;
│                                                      regenerate via train_model.py)
├── alerts/
│   └── telegram.py                                   # Fraud alert delivery (reuse pattern)
├── data/
│   ├── db.py                                            # PostgreSQL connection (SQLAlchemy)
│   ├── models.py                                          # ScoredTransaction table (SQLAlchemy)
│   └── queries.py                                           # Transaction/scoring history queries
├── dashboard/
│   ├── api_client.py                                          # HTTP client the dashboard uses
│   │                                                             to reach the API — never
│   │                                                             queries the DB directly
│   └── transforms.py                                            # Pure API-response -> chart/
│                                                                    table data functions
├── tests/
├── docker-compose.yml                                        # kafka + db + app + producer +
│                                                                consumer + dashboard
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Conventions
- The trained model is a build artifact, not source code — train_model.py
  must be runnable standalone to reproduce model.joblib from the Kaggle
  dataset; never hand-edit the model file
- Producer and consumer are separate processes/services — the producer
  only publishes to Kafka, the consumer only reads and scores; neither
  imports the other's internals
- Scoring logic (model/score.py) is a pure function (transaction dict in,
  risk score out) — testable without Kafka or a running consumer
- Every scored transaction is logged to PostgreSQL (data/queries.py) 
  regardless of fraud status, so the dashboard can show full history, not
  just flagged ones
- Telegram alerts fire only above the configured fraud threshold; a failed
  alert send must never crash the consumer loop — log and continue, same
  pattern as apiwatch/AutoReport
- Never hardcode credentials (DB, Kafka broker address, Telegram bot
  token) — everything from .env
- The retrain endpoint re-runs model/train.py against the current dataset
  (or an updated one) and atomically replaces model.joblib — the API must
  not serve requests mid-swap with a half-written model file

## Milestones
- **M1** — Project skeleton, Docker Compose with Kafka (KRaft mode) + 
  Postgres + app services healthy together; train_model.py produces a
  working model.joblib from the Kaggle dataset
- **M2** — Kafka producer (streams transactions from the dataset on an
  interval) + consumer (reads the topic, scores via the model, logs every
  transaction to Postgres)
- **M3** — Telegram fraud alerting wired into the consumer (fires above
  threshold), FastAPI endpoints (health, manual transaction submission for
  testing, basic transaction/history query)
- **M4** — Plotly Dash dashboard: live transaction feed, risk score
  distribution, fraud count over time, updating without manual refresh
- **M5** — Retrain endpoint (POST /model/retrain, atomic model swap), final
  Docker verification (fresh `docker compose up` works end-to-end: Kafka
  healthy, model loads, producer/consumer flowing, dashboard live)

## Current Milestone
All 5 milestones complete. Final feature set:
- Docker Compose brings up six services healthy with no manual
  intervention: kafka (KRaft), db (Postgres), app (FastAPI), producer,
  consumer, dashboard
- End-to-end pipeline: producer streams the Kaggle dataset onto the
  "transactions" Kafka topic on an interval -> consumer scores each via
  the Isolation Forest and logs every transaction to Postgres -> fraud
  above FRAUD_THRESHOLD fires a Telegram alert -> the Dash dashboard
  polls the API every few seconds and shows the live feed, risk
  distribution, cumulative fraud count, and (since this dataset carries
  ground truth) a true/false positive breakdown
- FastAPI: GET /health, POST /transactions (manual scoring — same path
  as the consumer), GET /transactions + GET /transactions/stats (what
  the dashboard consumes), POST /model/retrain (re-trains against the
  current dataset, atomically swaps model.joblib via os.replace, and
  invalidates the API's cached model so the next request uses the fresh
  one)
- One-time manual prerequisite (documented in README.md): download the
  Kaggle dataset to data/creditcard.csv and run `python train_model.py`
  on the host before the API can score anything — the dataset is
  .dockerignore'd and never auto-downloaded

## Data Source
Kaggle Credit Card Fraud dataset (anonymized/PCA-transformed transaction
features + a binary fraud label) — used both to train the Isolation
Forest offline (train_model.py) and as the source the Kafka producer
streams from to simulate "real-time" transactions arriving.
