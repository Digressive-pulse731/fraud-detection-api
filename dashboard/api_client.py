"""Thin HTTP client for the fraud-detection API.

Keeps app.py free of raw `requests` calls, same pattern as AutoReport's
api_client.py. A down/unreachable API must not crash the dashboard — every
function catches request errors, logs, and returns a safe empty value so
callbacks can render a "data unavailable" state instead of raising.
"""

import logging
import os

import requests

logger = logging.getLogger("dashboard.api_client")

# In Docker, API_URL is set to http://app:8000 (service name resolves on
# the compose network); on the host it defaults to localhost.
API_URL = os.environ.get("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 5


def get_stats() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/transactions/stats", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        logger.exception("Failed to fetch stats from %s", API_URL)
        return None


def get_transactions(limit: int = 50, offset: int = 0) -> list[dict]:
    try:
        response = requests.get(
            f"{API_URL}/transactions",
            params={"limit": limit, "offset": offset},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("transactions", [])
    except requests.RequestException:
        logger.exception("Failed to fetch transactions from %s", API_URL)
        return []
