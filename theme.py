"""Shared palette and Plotly styling.

.streamlit/config.toml is the single source of truth for the app's colours.
Reading the values back here means the Matplotlib gauge and every Plotly chart
use exactly the same palette as the widgets - change a colour in config.toml
and it propagates through the entire app.
"""
import streamlit as st


def _theme(option, fallback):
    """A theme option from config.toml, or `fallback` if it is not set."""
    value = st.get_option(f"theme.{option}")
    return fallback if value is None else value


PRIMARY = _theme("primaryColor", "#023E8A")
TEXT = _theme("textColor", "#1F2430")
GRAY = _theme("grayColor", "#6B7280")
GREEN = _theme("greenColor", "#16A34A")
AMBER = _theme("yellowColor", "#F59E0B")
RED = _theme("redColor", "#DC2626")
BORDER = _theme("borderColor", "#E3E7F0")
SURFACE = _theme("secondaryBackgroundColor", "#F5F6FA")

CATEGORICAL = _theme("chartCategoricalColors", [PRIMARY, "#00B4D8", AMBER, GREEN, RED])
SEQUENTIAL = _theme("chartSequentialColors", ["#F0F9FF", "#00B4D8", "#023E8A"])
DIVERGING = _theme("chartDivergingColors", [RED, "#FFFFFF", PRIMARY])

# Pale band fills for the gauge dial - tinted versions of the semantic colours.
ZONE_TINTS = {"low": "#DCFCE7", "medium": "#FEF3C7", "high": "#FEE2E2"}

# Churned vs retained. Red/green matches the report's own figures (the mosaic
# plots and Figure 9c), which keeps the app and the write-up consistent.
# Note: red/green alone is hard for colour-blind readers, so every chart that
# uses this pair also labels the two groups directly rather than relying on
# colour on its own.
CHURN = RED
RETAIN = GREEN
CHURN_FILL = "#FEE2E2"
RETAIN_FILL = "#DCFCE7"

# Row highlights for the model-comparison table.
BASELINE_ROW_STYLE = "background-color: #DCEBFF; color: #0B2E59; font-weight: 700;"
BEST_ROW_STYLE = "background-color: #DDF5E5; color: #123D24; font-weight: 700;"

# Risk-band cut-offs, shared by the gauge, the readout and the callout.
MEDIUM_RISK_AT = 33.3
HIGH_RISK_AT = 66.6

# One colour per model, reused across the ROC, CV and comparison charts so a
# model keeps the same identity everywhere in the dashboard.
MODEL_COLORS = {
    "Random Forest": PRIMARY,
    "XGBoost": "#00B4D8",
    "Logistic Regression": AMBER,
    "Decision Tree": "#7C3AED",
}

# Green -> amber -> red, for "how bad is this segment" heatmaps. A churn rate
# is not a neutral quantity: low is good and high is bad, and a single-hue blue
# ramp hides that. Anchored 0-60% so the colour of a cell means the same thing
# no matter which two features are being crossed.
RISK_SCALE = [
    [0.00, "#DCFCE7"], [0.25, "#BBF7D0"], [0.45, "#FEF3C7"],
    [0.65, "#FDBA74"], [0.85, "#F87171"], [1.00, "#DC2626"],
]

# Zoom / pan / reset / "download as PNG" stay enabled - they are the point of
# an interactive chart. Only the selection tools are dropped as unused here.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "responsive": True,
}


def risk_band(pct):
    """Map a churn probability (0-100) to its (band name, colour) pair, so the
    needle, the readout and the callout box can never disagree."""
    if pct >= HIGH_RISK_AT:
        return "high", RED
    if pct >= MEDIUM_RISK_AT:
        return "medium", AMBER
    return "low", GREEN


def model_color(name):
    """Stable colour for a model, falling back to the categorical palette."""
    return MODEL_COLORS.get(name, CATEGORICAL[0])


def style(fig, height=380, title=None, legend=True, margin=None):
    """Apply the shared chart look. Called by every figure builder so the whole
    dashboard reads as one design rather than a pile of default Plotly charts."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(color=TEXT, size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=margin or dict(l=10, r=20, t=54 if title else 24, b=44),
        # Explicit hover font colour - Plotly otherwise picks a washed-out grey
        # that is hard to read against the white tooltip background.
        hoverlabel=dict(bgcolor="white", bordercolor=BORDER,
                        font=dict(color=TEXT, size=12.5)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left",
                                     font=dict(size=15, color=TEXT)))
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    return fig
