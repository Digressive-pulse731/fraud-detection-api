# 🚨 Fraud Detection API — Real-Time ML Fraud Scoring

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white)
![Kafka](https://img.shields.io/badge/Apache_Kafka-Streaming-231F20?style=flat-square&logo=apachekafka&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation_Forest-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-26_passing-brightgreen?style=flat-square)

---

## ❓ Problem

Fraud detection needs to happen the instant a transaction occurs — not hours or days later in a batch report. Most fraud datasets are also severely imbalanced (real fraud is often <0.2% of transactions), which rules out simple supervised approaches and makes false-alarm control a genuine engineering challenge, not just a modeling afterthought.

---

## 💡 Solution

A streaming pipeline that scores every transaction the moment it arrives, using unsupervised anomaly detection rather than a label-hungry classifier:

- **Kafka-based streaming** — transactions flow through a message broker, decoupling "who generates transactions" from "who scores them" — the same pattern real fraud systems use at scale
- **Isolation Forest** for anomaly scoring — an unsupervised model well-suited to rare, hard-to-label fraud patterns, calibrated against the real Kaggle Credit Card Fraud dataset (empirically tuned so ~7.5% of real frauds are caught at a threshold that keeps false alarms to ~0.03%)
- **Instant alerting** — high-risk transactions trigger a Telegram alert the moment the consumer scores them
- **Full observability** — every transaction is logged (fraud or not) for a live dashboard and later evaluation against ground truth

---

## ✨ Features

- ⚡ **Real-time Kafka pipeline** — producer simulates a live transaction feed, consumer scores and logs every message as it arrives
- 🧠 **Isolation Forest ML model** — unsupervised anomaly detection, trained on 284,807 real credit card transactions
- 📊 **Live Plotly Dash dashboard** — auto-refreshing risk score distribution, cumulative fraud detections, and a full confusion matrix (true/false positives) evaluated against ground truth
- 🚨 **Telegram fraud alerts** — instant notification the moment a transaction crosses the risk threshold
- 🛠️ **Manual scoring endpoint** — submit any transaction via REST API for immediate scoring, no Kafka required for testing
- 🔄 **On-demand retraining** — `POST /model/retrain` re-trains and atomically swaps the model with zero downtime
- 🛡️ **Fails safe** — a malformed message or failed alert never crashes the pipeline; every failure is logged and the system keeps running
- 🐳 **Six-service Docker stack** — Kafka, PostgreSQL, API, producer, consumer, and dashboard, all orchestrated with one command

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Streaming | Apache Kafka (KRaft mode) | Real-time transaction feed, decoupled producer/consumer |
| ML | scikit-learn (Isolation Forest) | Unsupervised fraud anomaly scoring |
| API | FastAPI | Manual scoring, history queries, retrain trigger |
| Database | PostgreSQL | Every scored transaction logged for dashboard + evaluation |
| Dashboard | Plotly Dash | Live-updating charts and confusion matrix |
| Alerts | Telegram Bot API | Instant fraud notifications |
| Deployment | Docker Compose | 6 services orchestrated together |
| Testing | pytest | 26 tests — training, streaming, alerts, API, dashboard transforms |

---

## 📡 Architecture

```
data/creditcard.csv → producer → Kafka topic "transactions"
                                        │
                                        ▼
                                    consumer ──► IsolationForest (score)
                                        │              │
                                        ▼              ▼ (if risky)
                                  PostgreSQL      Telegram alert
                                        ▲
                                        │
                              FastAPI (GET /transactions, /stats)
                                        ▲
                                        │  polls every 4s
                                  Plotly Dash dashboard
```

Manual path: `POST /transactions` scores a single transaction through the exact same model, immediately — no Kafka required for testing.

---

## ⚡ Quick Start

```bash
git clone https://github.com/rizalcodes/fraud-detection-api.git
cd fraud-detection-api
```

**One-time step — download the dataset and train the model:**
1. Download `creditcard.csv` from the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place it at `data/creditcard.csv`
3. Train:
```bash
pip install -r requirements.txt
python train_model.py
```

**Run the full stack:**
```bash
cp .env.example .env   # set DB_PASSWORD, optionally TELEGRAM_BOT_TOKEN/CHAT_ID
docker compose up --build
```

- **Dashboard:** http://localhost:8050
- **API docs (Swagger):** http://localhost:8000/docs

### Tests

```bash
pytest
```

26 tests, all fast — synthetic data and mocked HTTP calls, no Kafka/Postgres/real model needed.

---

## 📁 Project Structure

```
fraud-detection-api/
├── train_model.py          # Standalone training entry point
├── app.py                    # Plotly Dash dashboard
├── api/
│   ├── main.py                 # FastAPI: health, submit, history, retrain
│   └── deps.py                    # Cached model + DB session dependencies
├── streaming/
│   ├── producer.py                  # Publishes dataset rows to Kafka
│   └── consumer.py                    # Scores + logs + alerts on every message
├── model/
│   ├── train.py                         # Isolation Forest training logic
│   └── score.py                           # Pure scoring function
├── dashboard/
│   ├── api_client.py                        # Dashboard's HTTP client to the API
│   └── transforms.py                          # Pure chart/table data transforms
├── alerts/telegram.py                            # Fraud alert delivery
├── tests/                                          # 26 tests
├── docker-compose.yml                                # 6 services
└── Dockerfile
```

---

## 👤 Author

**Rizal**

[![Portfolio](https://img.shields.io/badge/Portfolio-rizalcodes.github.io-0A66C2?style=flat-square)](https://rizalcodes.github.io)
[![GitHub](https://img.shields.io/badge/GitHub-rizalcodes-181717?style=flat-square&logo=github)](https://github.com/rizalcodes)
[![Twitter/X](https://img.shields.io/badge/X-@rizalcodes_-000000?style=flat-square&logo=x)](https://x.com/rizalcodes_)

---

*Built with Kafka, Isolation Forest, and the conviction that fraud detection only matters if it happens before the transaction clears.*
