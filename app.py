"""Plotly Dash dashboard: live transaction feed, risk score distribution,
fraud detections over time, and (since this dataset carries ground
truth) a prediction-accuracy breakdown.

Consumes the FastAPI service's GET /transactions and GET
/transactions/stats endpoints — never queries the database directly, so
the dashboard stays decoupled from the API's storage details.

Run standalone:  python app.py   (serves on 0.0.0.0:8050)
"""

import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from dashboard.api_client import get_stats, get_transactions
from dashboard.transforms import (
    confusion_counts,
    cumulative_fraud_series,
    format_table_rows,
    risk_scores,
)

REFRESH_MS = 4000
TABLE_ROWS = 50

# Validated default palette (dataviz skill, references/palette.md) — light mode.
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BORDER = "rgba(11,11,11,0.10)"
BLUE = "#2a78d6"
BLUE_SOFT = "#6da7ec"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARNING = "#fab219"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

app = Dash(__name__)
app.title = "Fraud Detection Dashboard"


def stat_card(label: str, value_id: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"color": INK_SECONDARY, "fontSize": "13px"}),
            html.Div(
                id=value_id,
                style={
                    "color": INK_PRIMARY,
                    "fontSize": "32px",
                    "fontWeight": 600,
                    "fontVariantNumeric": "tabular-nums",
                },
            ),
        ],
        style={
            "background": SURFACE,
            "border": f"1px solid {BORDER}",
            "borderRadius": "8px",
            "padding": "16px 20px",
            "flex": "1",
            "minWidth": "160px",
        },
    )


def confusion_card(label: str, value_id: str, color: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"color": INK_SECONDARY, "fontSize": "13px"}),
            html.Div(
                id=value_id,
                style={
                    "color": color,
                    "fontSize": "24px",
                    "fontWeight": 600,
                    "fontVariantNumeric": "tabular-nums",
                },
            ),
        ],
        style={
            "background": SURFACE,
            "border": f"1px solid {BORDER}",
            "borderRadius": "8px",
            "padding": "12px 16px",
            "flex": "1",
            "minWidth": "140px",
        },
    )


app.layout = html.Div(
    [
        dcc.Interval(id="interval", interval=REFRESH_MS, n_intervals=0),
        html.Div(
            [
                html.H1("Fraud Detection Dashboard", style={"color": INK_PRIMARY, "marginBottom": "2px"}),
                html.Div(
                    "Live view of scored transactions — refreshes every "
                    f"{REFRESH_MS // 1000}s",
                    style={"color": INK_MUTED, "fontSize": "13px", "marginBottom": "20px"},
                ),
                html.Div(id="api-error-banner"),
                # Summary cards
                html.Div(
                    [
                        stat_card("Total scored", "stat-total"),
                        stat_card("Flagged as fraud", "stat-flagged"),
                        stat_card("Average risk score", "stat-avg-risk"),
                        stat_card("Fraud detection rate", "stat-rate"),
                    ],
                    style={"display": "flex", "gap": "12px", "marginBottom": "24px"},
                ),
                # Charts
                html.Div(
                    [
                        html.Div(dcc.Graph(id="risk-histogram"), style={"flex": "1", "minWidth": "320px"}),
                        html.Div(dcc.Graph(id="fraud-over-time"), style={"flex": "1", "minWidth": "320px"}),
                    ],
                    style={"display": "flex", "gap": "12px", "marginBottom": "24px", "flexWrap": "wrap"},
                ),
                # Evaluation-only confusion breakdown
                html.Div(
                    [
                        html.H3("Prediction accuracy (evaluation only)", style={"color": INK_PRIMARY, "marginBottom": "2px"}),
                        html.Div(
                            "Uses the dataset's known fraud label to check predictions — "
                            "not available with real unlabeled data.",
                            style={"color": INK_MUTED, "fontSize": "12px", "marginBottom": "10px"},
                        ),
                        html.Div(
                            [
                                confusion_card("True positives (caught fraud)", "conf-tp", GOOD),
                                confusion_card("False positives (false alarm)", "conf-fp", CRITICAL),
                                confusion_card("False negatives (missed fraud)", "conf-fn", WARNING),
                                confusion_card("True negatives (correct, legit)", "conf-tn", BLUE),
                            ],
                            style={"display": "flex", "gap": "12px"},
                        ),
                    ],
                    style={"marginBottom": "24px"},
                ),
                # Recent transactions table
                html.H3(f"Recent transactions (last {TABLE_ROWS})", style={"color": INK_PRIMARY}),
                dash_table.DataTable(
                    id="transactions-table",
                    columns=[
                        {"name": "ID", "id": "id"},
                        {"name": "Amount", "id": "amount"},
                        {"name": "Risk score", "id": "risk_score"},
                        {"name": "Prediction", "id": "flag"},
                        {"name": "Actual label", "id": "actual_label"},
                        {"name": "Scored at", "id": "scored_at"},
                    ],
                    data=[],
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "fontFamily": FONT_FAMILY,
                        "fontSize": "13px",
                        "padding": "6px 10px",
                        "textAlign": "left",
                    },
                    style_header={
                        "backgroundColor": PAGE,
                        "color": INK_SECONDARY,
                        "fontWeight": 600,
                        "border": "none",
                        "borderBottom": f"1px solid {GRIDLINE}",
                    },
                    style_data={"backgroundColor": SURFACE, "color": INK_PRIMARY, "border": "none"},
                    style_data_conditional=[
                        {
                            "if": {"filter_query": "{predicted_fraud} = true"},
                            "backgroundColor": "rgba(208,59,59,0.08)",
                        },
                    ],
                ),
            ],
            style={"maxWidth": "1100px", "margin": "0 auto", "padding": "24px"},
        ),
    ],
    style={"background": PAGE, "minHeight": "100vh", "fontFamily": FONT_FAMILY},
)


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        annotations=[{"text": message, "showarrow": False, "font": {"color": INK_MUTED}}],
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        height=320,
    )
    return fig


def _base_layout(title: str) -> dict:
    return {
        "template": "plotly_white",
        "title": {"text": title, "font": {"color": INK_PRIMARY, "size": 15}},
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"family": FONT_FAMILY, "color": INK_SECONDARY},
        "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
        "height": 320,
        "xaxis": {"gridcolor": GRIDLINE, "linecolor": GRIDLINE},
        "yaxis": {"gridcolor": GRIDLINE, "linecolor": GRIDLINE},
    }


@app.callback(
    Output("api-error-banner", "children"),
    Output("stat-total", "children"),
    Output("stat-flagged", "children"),
    Output("stat-avg-risk", "children"),
    Output("stat-rate", "children"),
    Output("risk-histogram", "figure"),
    Output("fraud-over-time", "figure"),
    Output("conf-tp", "children"),
    Output("conf-fp", "children"),
    Output("conf-fn", "children"),
    Output("conf-tn", "children"),
    Output("transactions-table", "data"),
    Input("interval", "n_intervals"),
)
def refresh(_n_intervals):
    stats = get_stats()
    transactions = get_transactions(limit=TABLE_ROWS)

    banner = ""
    if stats is None:
        banner = html.Div(
            "⚠ Could not reach the API — showing the last successful data, if any.",
            style={
                "background": "rgba(208,59,59,0.10)",
                "color": CRITICAL,
                "padding": "8px 12px",
                "borderRadius": "6px",
                "marginBottom": "16px",
                "fontSize": "13px",
            },
        )
        stats = {"total_scored": 0, "total_flagged": 0, "avg_risk_score": 0.0}

    total = stats["total_scored"]
    flagged = stats["total_flagged"]
    rate = (flagged / total) if total else 0.0

    histogram = go.Figure(
        go.Histogram(x=risk_scores(transactions), nbinsx=20, marker_color=BLUE)
    )
    histogram.update_layout(**_base_layout("Risk score distribution (recent)"))
    histogram.update_xaxes(title_text="Risk score", range=[0, 1])
    histogram.update_yaxes(title_text="Count")
    if not transactions:
        histogram = _empty_figure("No transactions yet")

    times, counts = cumulative_fraud_series(transactions)
    fraud_line = go.Figure(
        go.Scatter(x=times, y=counts, mode="lines", line={"color": CRITICAL, "width": 2})
    )
    fraud_line.update_layout(**_base_layout("Cumulative fraud detections (recent)"))
    fraud_line.update_xaxes(title_text="Time")
    fraud_line.update_yaxes(title_text="Cumulative flagged count")
    if not transactions:
        fraud_line = _empty_figure("No transactions yet")

    conf = confusion_counts(transactions)

    return (
        banner,
        f"{total:,}",
        f"{flagged:,}",
        f"{stats['avg_risk_score']:.3f}",
        f"{rate:.1%}",
        histogram,
        fraud_line,
        f"{conf['true_positive']:,}",
        f"{conf['false_positive']:,}",
        f"{conf['false_negative']:,}",
        f"{conf['true_negative']:,}",
        format_table_rows(transactions),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
