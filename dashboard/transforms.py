"""Pure data-transformation functions between the API's transaction list
and chart/table-ready shapes. No Dash or network imports here, so these
are testable with plain dicts."""


def risk_scores(transactions: list[dict]) -> list[float]:
    """Risk scores for the histogram, in whatever order they arrived."""
    return [t["risk_score"] for t in transactions]


def cumulative_fraud_series(transactions: list[dict]) -> tuple[list[str], list[int]]:
    """(timestamps, running fraud count) in chronological order, for the
    fraud-detections-over-time line chart. Input may be newest-first
    (as returned by GET /transactions) — this re-sorts to oldest-first."""
    ordered = sorted(transactions, key=lambda t: t["scored_at"])
    timestamps: list[str] = []
    counts: list[int] = []
    running = 0
    for txn in ordered:
        if txn["predicted_fraud"]:
            running += 1
        timestamps.append(txn["scored_at"])
        counts.append(running)
    return timestamps, counts


def confusion_counts(transactions: list[dict]) -> dict:
    """True/false positive/negative counts against actual_label.

    Records with no actual_label (a real unlabeled feed) are excluded —
    this breakdown is only meaningful because this dataset carries
    ground truth for evaluation.
    """
    counts = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    for txn in transactions:
        label = txn.get("actual_label")
        if label is None:
            continue
        predicted = bool(txn["predicted_fraud"])
        actual = bool(label)
        if predicted and actual:
            counts["true_positive"] += 1
        elif predicted and not actual:
            counts["false_positive"] += 1
        elif not predicted and actual:
            counts["false_negative"] += 1
        else:
            counts["true_negative"] += 1
    return counts


def format_table_rows(transactions: list[dict]) -> list[dict]:
    """Round/label fields for direct display in the recent-transactions
    table, so risk isn't conveyed by row color alone."""
    rows = []
    for txn in transactions:
        label = txn.get("actual_label")
        rows.append(
            {
                "id": txn["id"],
                "amount": round(txn["amount"], 2),
                "risk_score": round(txn["risk_score"], 3),
                "flag": "\U0001f6a9 Fraud" if txn["predicted_fraud"] else "OK",
                "predicted_fraud": txn["predicted_fraud"],
                "actual_label": "fraud" if label == 1 else ("legit" if label == 0 else "n/a"),
                "scored_at": (
                    str(txn["scored_at"])
                    .replace("T", " ")
                    .split(".")[0]
                    .split("+")[0]
                ),
            }
        )
    return rows
