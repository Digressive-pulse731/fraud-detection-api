"""Telegram fraud alert delivery.

Same pattern as apiwatch/AutoReport: Bot API via requests, credentials
from the environment. Raises TelegramAlertError on any failure — callers
(consumer, API) must catch it and continue; a failed alert must never
stop transaction processing.
"""

import os
from datetime import datetime, timezone

import requests

API_URL = "https://api.telegram.org/bot{token}/sendMessage"
REQUEST_TIMEOUT = 10


class TelegramAlertError(Exception):
    """Raised when a fraud alert could not be delivered."""


def _format_alert(transaction: dict, risk_score: float) -> str:
    threshold = os.environ.get("FRAUD_THRESHOLD", "0.7")
    lines = [
        "\U0001f6a8 FRAUD ALERT \U0001f6a8",
        f"Risk score: {risk_score:.3f} (threshold {threshold})",
        f"Amount: ${float(transaction.get('Amount', 0.0)):,.2f}",
        f"Scored at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    ]
    if "Time" in transaction:
        lines.append(f"Dataset time offset: {transaction['Time']:.0f}s")
    # A few leading PCA features for identification/debugging
    details = [f"{k}={transaction[k]:.3f}" for k in ("V1", "V2", "V3", "V4") if k in transaction]
    if details:
        lines.append("Features: " + ", ".join(details))
    return "\n".join(lines)


def send_fraud_alert(transaction: dict, risk_score: float) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise TelegramAlertError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set to send alerts"
        )

    try:
        response = requests.post(
            API_URL.format(token=token),
            json={"chat_id": chat_id, "text": _format_alert(transaction, risk_score)},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise TelegramAlertError(f"Telegram request failed: {exc}") from exc

    if not response.ok:
        raise TelegramAlertError(
            f"Telegram API returned {response.status_code}: {response.text[:200]}"
        )
