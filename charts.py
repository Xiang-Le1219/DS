"""Interactive Plotly figures for the Telco churn dashboard.

Every figure here replaces one of the notebook's static PNG exports. Because
they are built live, the reader can hover for exact values, zoom, toggle series
from the legend and export any chart as an image - none of which a PNG allows.

All colours come from theme.py, which in turn reads .streamlit/config.toml, so
the charts and the widgets can never drift apart.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from plotly.subplots import make_subplots

import analysis as A
import theme as T


def _pct_axis(fig, axis="x", title=None):
    """Percent-formatted axis with a consistent title."""
    update = fig.update_xaxes if axis == "x" else fig.update_yaxes
    update(ticksuffix="%", title=title)
    return fig


# ---------------------------------------------------------------------------
# Exploratory data analysis
# ---------------------------------------------------------------------------
def churn_donut():
    """Replaces fig1_churn_pie.png."""
    df = A.load_cleaned()
    counts = df["Churn"].value_counts()
    fig = go.Figure(go.Pie(
        labels=["Retained", "Churned"],
        values=[int(counts.get("No", 0)), int(counts.get("Yes", 0))],
        marker=dict(colors=[T.RETAIN, T.CHURN],
                    line=dict(color="white", width=2)),
        textinfo="label+percent", textfont=dict(size=13),
        hovertemplate="%{label}<br>%{value:,} customers (%{percent})<extra></extra>",
        sort=False,
    ))
    rate = (df["Churn"] == "Yes").mean()
    return T.style(fig, height=340, legend=False)


def churn_rate_bar(column):
    """Churn rate per category, with segment sizes on hover.

    Replaces fig2_mosaic_categorical.png - a mosaic plot cannot be read
    precisely, whereas this gives the exact rate and the exact n per segment.
    """
    data = A.churn_rate_by(column)
    baseline = (A.load_cleaned()["Churn"] == "Yes").mean()
    fig = go.Figure(go.Bar(
        x=data[column].astype(str), y=data["ChurnRate"] * 100,
        marker=dict(color=data["ChurnRate"] * 100, colorscale=T.SEQUENTIAL,
                    line=dict(color=T.BORDER, width=1)),
        text=[f"{v:.1%}" for v in data["ChurnRate"]], textposition="outside",
        customdata=data["Customers"],
        hovertemplate="%{x}<br>Churn rate: %{y:.1f}%<br>%{customdata:,} customers<extra></extra>",
    ))
    fig.add_hline(y=baseline * 100, line_dash="dash", line_color=T.GRAY,
                  annotation_text=f"Overall {baseline:.1%}",
                  annotation_position="top right",
                  annotation_font=dict(color=T.GRAY, size=11))
    fig.update_yaxes(range=[0, max(data["ChurnRate"].max() * 130, 10)])
    return _pct_axis(T.style(fig, height=380, legend=False), "y", "Churn rate")


def preparation_evidence():
    """Replaces fig0_preparation_validation.png.

    Three panels: SMOTE rebalanced the training set (and only the training set),
    and the StandardScaler turned raw tenure into a mean-0 / std-1 distribution.
    SMOTE is actually re-run to produce these counts, not quoted.

    Raw months and z-scores get their own panels rather than being overlaid on a
    shared axis - they span completely different ranges, so one axis would
    squash the z-scores into a spike and misrepresent the check.
    """
    balance = A.smote_balance()
    raw, scaled = A.scaling_check("tenure")

    fig = make_subplots(
        rows=1, cols=3, column_widths=[0.4, 0.3, 0.3],
        subplot_titles=("SMOTE on the training set (train only)",
                        "Raw `tenure`",
                        f"Scaled: mean {scaled.mean():.2f}, std {scaled.std():.2f}"),
        horizontal_spacing=0.09,
    )
    for label, color in [("Before", "#9DC3E6"), ("After", T.PRIMARY)]:
        fig.add_trace(go.Bar(
            x=balance["Class"], y=balance[label], name=label,
            marker_color=color, text=balance[label], textposition="outside",
            textfont=dict(size=11),
            hovertemplate="%{x}<br>" + label + " SMOTE: %{y:,} rows<extra></extra>",
        ), row=1, col=1)

    # Histograms, not KDE curves: tenure has hard boundaries (0 to 72 months,
    # discrete integer values), and a Gaussian KDE does not know that - it leaks
    # a meaningful share of its density past both edges (measured at ~6% below 0
    # and ~5% above 72 with this data), which is a real distortion, not just a
    # style choice. A histogram also keeps the real spike of zero-tenure
    # customers visible instead of smoothing it into a rounded bump - and this
    # chart exists to show *evidence*, so the exact counts matter more than a
    # smooth aesthetic.
    fig.add_trace(go.Histogram(
        x=raw, marker_color="#9DC3E6", nbinsx=36, showlegend=False,
        hovertemplate="tenure %{x:.0f} months<br>%{y:,} customers<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Histogram(
        x=scaled, marker_color=T.CHURN, nbinsx=36, showlegend=False,
        hovertemplate="z-score %{x:.2f}<br>%{y:,} customers<extra></extra>",
    ), row=1, col=3)
    # The two markers the check is actually about.
    fig.add_vline(x=0, line_dash="dash", line_color=T.TEXT, line_width=1.5,
                  row=1, col=3)

    before_rate = balance["Before"][1] / balance["Before"].sum()
    after_rate = balance["After"][1] / balance["After"].sum()
    fig.update_yaxes(title="Training rows", row=1, col=1,
                     range=[0, balance["After"].max() * 1.18])
    fig.update_xaxes(title=f"churn {before_rate:.1%} -> {after_rate:.1%}", row=1, col=1)
    fig.update_xaxes(title="months", row=1, col=2)
    fig.update_xaxes(title="z-score", row=1, col=3)
    fig.update_yaxes(title="Customers", row=1, col=2)
    fig.update_layout(barmode="group")
    styled = T.style(fig, height=420, margin=dict(l=10, r=20, t=54, b=76))
    # Legend below the panels: the shared default sits above the plot area, where
    # the three subplot titles already are.
    styled.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.20,
                                     xanchor="left", x=0))
    styled.update_annotations(font_size=12)
    return styled


def mosaic(column):
    """Replaces fig2_mosaic_categorical.png as a true mosaic plot: column width
    encodes customer volume, and the red segment height is the churn rate.

    Unlike the static original, hovering gives the exact count and rate - a
    mosaic is otherwise impossible to read precisely.
    """
    data = A.mosaic_data(column)
    total = data["Customers"].sum()
    widths = (data["Customers"] / total).to_numpy()
    # Bar centres, so each column's width is proportional to its customer count.
    centres = np.cumsum(widths) - widths / 2

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centres, y=data["ChurnRate"] * 100, width=widths * 0.98,
        name="Churned", marker=dict(color=T.CHURN, line=dict(color="white", width=1)),
        text=[f"{v:.1%}" for v in data["ChurnRate"]], textposition="inside",
        textfont=dict(color="white", size=12),
        customdata=np.stack([data[column].astype(str), data["Customers"]], axis=-1),
        hovertemplate=("%{customdata[0]}<br>Churned: %{y:.1f}%"
                       "<br>%{customdata[1]:,} customers<extra></extra>"),
    ))
    fig.add_trace(go.Bar(
        x=centres, y=(1 - data["ChurnRate"]) * 100, width=widths * 0.98,
        name="Retained", marker=dict(color=T.RETAIN, line=dict(color="white", width=1)),
        customdata=np.stack([data[column].astype(str), data["Customers"]], axis=-1),
        hovertemplate=("%{customdata[0]}<br>Retained: %{y:.1f}%"
                       "<br>%{customdata[1]:,} customers<extra></extra>"),
    ))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(
        tickvals=centres,
        ticktext=[f"{c}<br>(n={n:,})" for c, n in
                  zip(data[column].astype(str), data["Customers"])],
        title=f"{column} - column width = customer volume",
    )
    fig.update_yaxes(title="Share of customers", ticksuffix="%", range=[0, 100])
    return T.style(fig, height=420)


def cramers_v_lollipop():
    """Replaces fig3_cramers_v_lollipop.png."""
    data = A.cramers_v_scores().sort_values("CramersV")
    fig = go.Figure()
    for _, row in data.iterrows():
        fig.add_trace(go.Scatter(
            x=[0, row["CramersV"]], y=[row["Feature"], row["Feature"]],
            mode="lines", line=dict(color="#9DB2CC", width=3),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=data["CramersV"], y=data["Feature"], mode="markers",
        marker=dict(size=12, color=data["CramersV"], colorscale=T.SEQUENTIAL,
                    line=dict(color=T.PRIMARY, width=1)),
        hovertemplate="%{y}<br>Cramer's V: %{x:.3f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_xaxes(title="Cramer's V (association strength with churn)")
    return T.style(fig, height=520, legend=False, margin=dict(l=10, r=30, t=24, b=44))


def numeric_violin(feature):
    """Replaces fig4_numeric_violin.png."""
    df = A.load_cleaned()
    fig = go.Figure()
    for label, fill, line in [("No", T.RETAIN_FILL, T.RETAIN),
                              ("Yes", T.CHURN_FILL, T.CHURN)]:
        subset = df[df["Churn"] == label][feature]
        fig.add_trace(go.Violin(
            y=subset, name="Retained" if label == "No" else "Churned",
            box_visible=False, meanline_visible=True, opacity=0.85,
            fillcolor=fill, line=dict(color=line), points=False,
            hovertemplate=f"{feature}: %{{y:.2f}}<extra></extra>",
        ))
    # No box: with the mean line plus the median annotation below, an inset
    # quartile box was a third layer of central-tendency marks on only two
    # groups - the KDE shape already carries the spread, so the box was adding
    # clutter rather than information.
    for i, label in enumerate(["No", "Yes"]):
        med = float(df[df["Churn"] == label][feature].median())
        fig.add_annotation(
            x=i, y=med, text=f"median {med:,.1f}", showarrow=True, arrowhead=0,
            arrowcolor=T.TEXT, ax=48, ay=0, xanchor="left",
            font=dict(size=11, color=T.TEXT),
            bgcolor="rgba(255,255,255,0.85)",
        )
    fig.update_yaxes(title=feature)
    return T.style(fig, height=420)


def ecdf(feature):
    """Replaces fig5_ecdf.png."""
    df = A.load_cleaned()
    fig = go.Figure()
    for label, name, color in [("No", "Retained", T.RETAIN), ("Yes", "Churned", T.CHURN)]:
        values = np.sort(df[df["Churn"] == label][feature].to_numpy())
        y = np.arange(1, len(values) + 1) / len(values)
        fig.add_trace(go.Scatter(
            x=values, y=y * 100, mode="lines", name=name,
            line=dict(color=color, width=2.5),
            hovertemplate=f"{feature}: %{{x:.1f}}<br>%{{y:.1f}}%% at or below<extra></extra>",
        ))
    # The 50% line and where each group crosses it - that crossing IS the median,
    # and it is the actionable number the report reads off this chart.
    fig.add_hline(y=50, line_dash="dash", line_color=T.GRAY, line_width=1.5,
                  annotation_text="50% line", annotation_position="top right",
                  annotation_font=dict(color=T.GRAY, size=10))
    for label, color in [("No", T.RETAIN), ("Yes", T.CHURN)]:
        med = float(df[df["Churn"] == label][feature].median())
        fig.add_vline(x=med, line_dash="dot", line_color=color, line_width=1.5)
        fig.add_annotation(x=med, y=4, text=f"<b>{med:,.0f}</b>", showarrow=False,
                           font=dict(size=12, color=color), xanchor="left",
                           bgcolor="rgba(255,255,255,0.85)")
    fig.update_xaxes(title=feature)
    return _pct_axis(T.style(fig, height=400), "y", "Cumulative share of customers")


def interaction_heatmap(row_feature, col_feature):
    """Replaces fig6_interaction_heatmap.png - now with any pair of features."""
    rate, count = A.interaction_rates(row_feature, col_feature)
    fig = go.Figure(go.Heatmap(
        z=rate.to_numpy() * 100,
        x=[str(c) for c in rate.columns], y=[str(i) for i in rate.index],
        colorscale=T.RISK_SCALE, zmin=0, zmax=60,
        colorbar=dict(title="Churn %", ticksuffix="%"),
        customdata=count.to_numpy(),
        text=[[f"{v:.0%}" for v in row] for row in rate.to_numpy()],
        texttemplate="%{text}", textfont=dict(size=12),
        hovertemplate=(f"{row_feature}: %{{y}}<br>{col_feature}: %{{x}}"
                       "<br>Churn rate: %{z:.1f}%<br>%{customdata:,} customers<extra></extra>"),
    ))
    fig.update_xaxes(title=col_feature)
    # Reversed so the first category sits at the top, matching the table order
    # a reader expects rather than Plotly's bottom-up default.
    fig.update_yaxes(title=row_feature, autorange="reversed")
    return T.style(fig, height=420, legend=False)


def services_dual_axis():
    """Replaces fig7_num_services_dual_axis.png."""
    data = A.services_vs_churn()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    baseline = (A.load_cleaned()["Churn"] == "Yes").mean()
    fig.add_trace(go.Bar(
        x=data["num_services"], y=data["Customers"], name="Customers",
        marker=dict(color="#CFE3F5", line=dict(color=T.BORDER, width=1)),
        text=[f"{v:,}" for v in data["Customers"]], textposition="outside",
        textfont=dict(size=10, color=T.GRAY),
        hovertemplate="%{x} services<br>%{y:,} customers<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=data["num_services"], y=data["ChurnRate"] * 100, name="Churn rate",
        mode="lines+markers+text", line=dict(color=T.PRIMARY, width=3),
        marker=dict(size=9),
        text=[f"{v:.1%}" for v in data["ChurnRate"]], textposition="top center",
        textfont=dict(size=10, color=T.PRIMARY),
        hovertemplate="%{x} services<br>Churn rate: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)
    fig.add_hline(y=baseline * 100, line_dash="dash", line_color=T.GRAY,
                  secondary_y=True, annotation_text=f"Overall {baseline:.1%}",
                  annotation_position="bottom right",
                  annotation_font=dict(color=T.GRAY, size=11))
    fig.update_xaxes(title="Number of subscribed services", dtick=1)
    fig.update_yaxes(title="Customers", secondary_y=False,
                     range=[0, data["Customers"].max() * 1.18])
    fig.update_yaxes(title="Churn rate", ticksuffix="%", secondary_y=True,
                     showgrid=False, range=[0, 45])
    return T.style(fig, height=420)


def scatter_matrix(features=None, sample=1200):
    """Replaces fig8_scatter_matrix.png.

    Built by hand rather than with go.Splom: Splom's diagonal plots each feature
    against ITSELF, which is just a 45-degree line and tells you nothing. The
    report's Figure 8 puts a density curve there instead, so the diagonal here
    is a real KDE per churn group and the off-diagonal cells are scatters.
    """
    df = A.load_cleaned()
    features = list(features) if features else ["tenure", "MonthlyCharges", "TotalCharges"]
    n = len(features)
    points = df.sample(min(sample, len(df)), random_state=A.RANDOM_STATE)
    groups = [("No", "Retained", T.RETAIN), ("Yes", "Churned", T.CHURN)]

    fig = make_subplots(rows=n, cols=n, horizontal_spacing=0.035,
                        vertical_spacing=0.035)
    first = True
    for i, f_y in enumerate(features):
        for j, f_x in enumerate(features):
            row, col = i + 1, j + 1
            if i == j:
                # Diagonal: the feature's own distribution, split by churn.
                for label, name, color in groups:
                    values = df[df["Churn"] == label][f_x].to_numpy()
                    grid = np.linspace(values.min(), values.max(), 160)
                    # Weight each curve by how much of the data that group is.
                    # Without this both curves integrate to 1, so the smaller
                    # churned group peaks higher than retained - which reads as
                    # "more churners", the opposite of the truth.
                    share = len(values) / len(df)
                    density = gaussian_kde(values)(grid) * share
                    fig.add_trace(go.Scatter(
                        x=grid, y=density, mode="lines", name=name,
                        line=dict(color=color, width=2), fill="tozeroy",
                        fillcolor=color.replace(")", ", 0.25)").replace("rgb", "rgba")
                        if color.startswith("rgb") else None,
                        opacity=0.55, showlegend=False, legendgroup=name,
                        hovertemplate=f"{f_x}: %{{x:.1f}}<extra>{name}</extra>",
                    ), row=row, col=col)
            else:
                for label, name, color in groups:
                    subset = points[points["Churn"] == label]
                    fig.add_trace(go.Scattergl(
                        x=subset[f_x], y=subset[f_y], mode="markers", name=name,
                        marker=dict(size=4, color=color, opacity=0.5,
                                    line=dict(width=0)),
                        showlegend=first, legendgroup=name,
                        hovertemplate=(f"{f_x}: %{{x:.1f}}<br>{f_y}: %{{y:.1f}}"
                                       f"<extra>{name}</extra>"),
                    ), row=row, col=col)
                first = False

            # Labels only on the outer edge, as in the report's figure.
            if col == 1:
                fig.update_yaxes(title_text=f_y, title_font=dict(size=11),
                                 row=row, col=col)
            if row == n:
                fig.update_xaxes(title_text=f_x, title_font=dict(size=11),
                                 row=row, col=col)
            if i == j:
                # Density is not on the same scale as the scatters sharing this
                # row, so its ticks would mislead.
                fig.update_yaxes(showticklabels=False, row=row, col=col)

    fig.update_xaxes(tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    styled = T.style(fig, height=190 * n + 90, legend=True,
                     margin=dict(l=10, r=20, t=46, b=50))
    styled.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                     xanchor="left", x=0))
    return styled


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
def correlation_heatmap(top_n=15):
    """Replaces fig9a/fig9b correlation matrices, with an adjustable size."""
    matrix = A.correlation_matrix(top_n)
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index),
        colorscale=T.DIVERGING, zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="r"),
        hovertemplate="%{y}<br>%{x}<br>r = %{z:.3f}<extra></extra>",
    ))
    # Reversed so the diagonal runs top-left to bottom-right, as a correlation
    # matrix is normally printed.
    fig.update_yaxes(autorange="reversed")
    return T.style(fig, height=max(420, 26 * len(matrix)), legend=False,
                   margin=dict(l=10, r=20, t=24, b=44))


def correlation_ranked(top_n=None):
    """Replaces fig9c_correlation_ranked.png.

    Shows all 27 encoded features by default, exactly like the report's figure.
    Passing top_n keeps the strongest relationships at each end without ever
    double-counting a feature in the middle.
    """
    corr = A.churn_correlations()
    if top_n is None or top_n >= len(corr):
        ranked = corr.sort_values()
    else:
        keep = corr.abs().sort_values(ascending=False).head(top_n).index
        ranked = corr[keep].sort_values()
    colors = [T.CHURN if v > 0 else T.PRIMARY for v in ranked]
    fig = go.Figure(go.Bar(
        x=ranked.to_numpy(), y=list(ranked.index), orientation="h",
        marker=dict(color=colors),
        hovertemplate="%{y}<br>Correlation with churn: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=T.GRAY, line_width=1)
    fig.update_xaxes(title="Correlation with churn (red = raises risk)")
    return T.style(fig, height=max(420, 24 * len(ranked)), legend=False)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------
def model_comparison(metrics, threshold=0.5):
    """Replaces fig10_model_comparison.png."""
    data = A.live_metrics(threshold)[list(metrics)]
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric, x=list(data.index), y=data[metric],
            marker_color=T.CATEGORICAL[i % len(T.CATEGORICAL)],
            text=[f"{v:.3f}" for v in data[metric]], textposition="outside",
            hovertemplate="%{x}<br>" + metric + ": %{y:.4f}<extra></extra>",
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Score", range=[0, 1.08])
    return T.style(fig, height=440)


def roc_curves():
    """Replaces fig11_roc_curves.png - click the legend to isolate a model."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
        line=dict(color=T.GRAY, dash="dash", width=1.5), hoverinfo="skip",
    ))
    for name, (fpr, tpr, auc) in A.roc_points().items():
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.4f})",
            line=dict(color=T.model_color(name), width=2.5),
            hovertemplate=("False positive rate: %{x:.3f}"
                           "<br>True positive rate: %{y:.3f}<extra></extra>"),
        ))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return T.style(fig, height=480)


def confusion_heatmap(model_name, threshold):
    """Replaces fig12_confusion_matrices.png, recomputed at any threshold."""
    result = A.metrics_at_threshold(model_name, threshold)
    matrix = result["matrix"]
    labels = ["Predicted<br>retained", "Predicted<br>churn"]
    rows = ["Actually<br>retained", "Actually<br>churned"]
    total = matrix.sum()
    text = [[f"<b>{matrix[i][j]:,}</b><br>{matrix[i][j] / total:.1%}"
             for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=rows, colorscale=T.SEQUENTIAL, showscale=False,
        text=text, texttemplate="%{text}", textfont=dict(size=15),
        hovertemplate="%{y}<br>%{x}<br>%{z:,} customers<extra></extra>",
    ))
    # Plotly places y[0] at the bottom; reverse it so the matrix reads in the
    # conventional order, with the "actually retained" row on top.
    fig.update_yaxes(autorange="reversed")
    return T.style(fig, height=360, legend=False,
                   margin=dict(l=10, r=10, t=24, b=10))


def threshold_tradeoff(model_name, threshold):
    """Precision, recall and F1 across every threshold - the chart that shows
    *why* moving the cut-off is a trade-off rather than a free win."""
    data = A.threshold_sweep(model_name)
    fig = go.Figure()
    for metric, color in [("Precision", T.AMBER), ("Recall", T.PRIMARY), ("F1", T.GREEN)]:
        # connectgaps stays off so the NaN tail leaves a genuine gap.
        fig.add_trace(go.Scatter(
            x=data["Threshold"], y=data[metric], mode="lines", name=metric,
            line=dict(color=color, width=2.5),
            hovertemplate="Threshold %{x:.2f}<br>" + metric + ": %{y:.3f}<extra></extra>",
        ))
    fig.add_vline(x=threshold, line_color=T.TEXT, line_width=2,
                  annotation_text=f"{threshold:.2f}", annotation_position="top",
                  annotation_font=dict(color=T.TEXT, size=11))

    # Read off the value where each curve crosses the selected threshold - the
    # number the reader actually wants when they move the slider, rather than
    # having to eyeball it against the y axis.
    nearest = data.loc[(data["Threshold"] - threshold).abs().idxmin()]
    for metric, color in [("Precision", T.AMBER), ("Recall", T.PRIMARY),
                          ("F1", T.GREEN)]:
        value = nearest[metric]
        if pd.isna(value):
            continue          # precision is undefined once nothing is flagged
        fig.add_trace(go.Scatter(
            x=[nearest["Threshold"]], y=[value], mode="markers",
            marker=dict(size=10, color=color, line=dict(color="white", width=2)),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_annotation(
            x=nearest["Threshold"], y=value, text=f"<b>{value:.3f}</b>",
            showarrow=False, xanchor="left", xshift=9,
            font=dict(size=11, color=color), bgcolor="rgba(255,255,255,0.88)",
        )
    # Past the model's highest predicted probability nothing is flagged at all,
    # so precision is undefined rather than zero. Shade that region and stop the
    # precision line there instead of letting it drop to the floor.
    ceiling = A.max_probability(model_name)
    if ceiling < 0.95:
        fig.add_vrect(x0=ceiling, x1=0.95, fillcolor=T.GRAY, opacity=0.10,
                      layer="below", line_width=0,
                      annotation_text="flags nobody", annotation_position="top left",
                      annotation_font=dict(color=T.GRAY, size=10))
    fig.update_xaxes(title="Decision threshold")
    fig.update_yaxes(title="Score", range=[0, 1.02])
    return T.style(fig, height=360)


def cv_stability(metric="f1", show_folds=False):
    """Replaces fig13_cv_stability.png.

    Mean score per model with a +/- 1 standard deviation error bar, matching
    Figure 13 in the report - a shorter bar means a more reliable model. The
    individual folds can be overlaid, which the static original could not do.
    """
    data = A.cv_results(metric)
    # ddof=0 (population std) matches numpy's default, which is what Figure 13
    # used; pandas' default ddof=1 would inflate every error bar by sqrt(5/4).
    summary = data.groupby("Model")["Score"].agg(
        mean="mean", std=lambda x: x.std(ddof=0)).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary["Model"], y=summary["mean"],
        marker=dict(color=[T.model_color(m) for m in summary["Model"]],
                    line=dict(color="white", width=1)),
        error_y=dict(type="data", array=summary["std"], color=T.TEXT,
                     thickness=1.6, width=8),
        hovertemplate=("%{x}<br>mean " + metric + ": %{y:.4f}"
                       "<extra></extra>"),
        showlegend=False,
    ))
    # Labels as annotations rather than bar text: bar text sits at the bar top,
    # where the error-bar cap already is, so the two collide.
    headroom = float((summary["mean"] + summary["std"]).max())
    for _, row in summary.iterrows():
        fig.add_annotation(
            x=row["Model"], y=row["mean"] + row["std"] + headroom * 0.06,
            text=f"<b>{row['mean']:.3f}</b><br>(+/-{row['std']:.3f})",
            showarrow=False, font=dict(size=11, color=T.TEXT), align="center",
        )
    if show_folds:
        for name in data["Model"].unique():
            scores = data[data["Model"] == name]["Score"]
            fig.add_trace(go.Scatter(
                x=[name] * len(scores), y=scores, mode="markers",
                marker=dict(color=T.TEXT, size=7, symbol="circle-open",
                            line=dict(width=1.6)),
                name="Individual folds", showlegend=False,
                hovertemplate=f"{name}<br>fold {metric}: %{{y:.4f}}<extra></extra>",
            ))
    fig.update_yaxes(title=f"{metric} ({A.CV_FOLDS}-fold CV mean +/- std)",
                     range=[0, float((summary["mean"] + summary["std"]).max()) * 1.30])
    return T.style(fig, height=440, legend=False)


def overfitting_check():
    """Replaces fig14_overfitting_check.png."""
    data = A.train_vs_test()
    fig = go.Figure()
    for label, color in [("Train", "#9DC3E6"), ("Test", T.PRIMARY)]:
        fig.add_trace(go.Bar(
            name=label, x=data["Model"], y=data[label],
            marker=dict(color=color, line=dict(color=T.BORDER, width=1)),
            text=[f"{v:.3f}" for v in data[label]], textposition="outside",
            customdata=data["Gap"],
            hovertemplate="%{x}<br>" + label +
                          " ROC-AUC: %{y:.4f}<br>Train-test gap: %{customdata:.4f}<extra></extra>",
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="ROC-AUC", range=[0, 1.1])
    return T.style(fig, height=420)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def importance_bar(model_name, top_n=15):
    """Replaces fig15_feature_importance.png.

    Tree importances are always positive, but Logistic Regression coefficients
    are signed - a negative coefficient means the feature *lowers* churn risk.
    Ranking is by magnitude either way, and the sign is kept so that direction
    is not thrown away, matching Figure 15 in the report.
    """
    data, label = A.builtin_importance(model_name)
    data = data.head(top_n).sort_values("Importance", key=abs)
    signed = data["Importance"].min() < 0
    colors = ([T.CHURN if v > 0 else T.PRIMARY for v in data["Importance"]]
              if signed else data["Importance"])
    fig = go.Figure(go.Bar(
        x=data["Importance"], y=data["Feature"], orientation="h",
        marker=(dict(color=colors) if signed
                else dict(color=colors, colorscale=T.SEQUENTIAL)),
        text=[f"{v:+.3f}" if signed else f"{v:.3f}" for v in data["Importance"]],
        textposition="outside", textfont=dict(size=10),
        cliponaxis=False,
        hovertemplate="%{y}<br>" + label + ": %{x:.4f}<extra></extra>",
    ))
    if signed:
        fig.add_vline(x=0, line_color=T.GRAY, line_width=1)
        label += "  (red = raises churn risk, blue = lowers it)"
    fig.update_xaxes(title=label)
    return T.style(fig, height=max(380, 26 * len(data)), legend=False,
                   margin=dict(l=10, r=60, t=24, b=44))


def importance_grid(top_n=10):
    """Figure 15's four-panel view: the top features for every model at once,
    so the reader can see where the models agree."""
    everything = A.all_importances()
    names = list(everything.keys())
    fig = make_subplots(rows=2, cols=2, subplot_titles=names,
                        horizontal_spacing=0.22, vertical_spacing=0.14)
    for i, name in enumerate(names):
        frame, label = everything[name]
        frame = frame.head(top_n).sort_values("Importance")
        fig.add_trace(go.Bar(
            x=frame["Importance"], y=frame["Feature"], orientation="h",
            marker_color=T.model_color(name), showlegend=False,
            hovertemplate="%{y}<br>" + label + ": %{x:.4f}<extra></extra>",
        ), row=i // 2 + 1, col=i % 2 + 1)
    fig.update_yaxes(tickfont=dict(size=9))
    fig.update_xaxes(tickfont=dict(size=9))
    return T.style(fig, height=210 + 62 * top_n, legend=False,
                   margin=dict(l=10, r=20, t=40, b=30))


def permutation_bar(model_name, top_n=15):
    """Replaces fig17_permutation_importance.png, with error bars."""
    data = A.permutation_scores(model_name).head(top_n).sort_values("Importance")
    fig = go.Figure(go.Bar(
        x=data["Importance"], y=data["Feature"], orientation="h",
        error_x=dict(type="data", array=data["Std"], color=T.TEXT, thickness=1.2),
        marker=dict(color=T.PRIMARY),
        hovertemplate="%{y}<br>ROC-AUC drop: %{x:.4f} (+/-%{error_x.array:.4f})<extra></extra>",
    ))

    # place value labels past the tip of each error bar, not the bar itself
    pad = (data["Importance"] + data["Std"]).max() * 0.02
    fig.add_trace(go.Scatter(
        x=data["Importance"] + data["Std"] + pad, y=data["Feature"],
        mode="text", text=[f"{v:.3f}" for v in data["Importance"]],
        textposition="middle right", textfont=dict(size=10, color=T.TEXT),
        hoverinfo="skip", showlegend=False,
    ))

    fig.update_xaxes(title="Drop in ROC-AUC when the feature is shuffled")
    return T.style(fig, height=max(380, 26 * len(data)), legend=False,
                   margin=dict(l=10, r=70, t=24, b=44))


def correlation_vs_importance(model_name, top_n=20):
    """Replaces fig16_correlation_vs_importance.png (report title:
    "Correlation vs. Learned Importance").

    A sanity check: if the model's importance ranking agrees with the raw
    correlations, it is picking up genuine signal rather than noise.
    """
    importance, label = A.builtin_importance(model_name)
    corr = A.churn_correlations().abs()          # match |Correlation| used in Figure 9
    merged = importance.head(top_n).copy()
    merged["Correlation"] = merged["Feature"].map(corr)
    merged = merged.dropna(subset=["Correlation"])

    # rank shift, same as the notebook's "consistency" table
    merged["Corr rank"] = merged["Correlation"].rank(ascending=False).astype(int)
    merged["RF rank"] = merged["Importance"].abs().rank(ascending=False).astype(int)
    merged["Rank shift"] = merged["Corr rank"] - merged["RF rank"]

    def _tag(row):
        shift = row["Rank shift"]
        return f"{row['Feature']} ({'+' if shift > 0 else ''}{shift})" if shift != 0 else row["Feature"]

    fig = go.Figure(go.Scatter(
        x=merged["Correlation"], y=merged["Importance"], mode="markers+text",
        text=merged.apply(_tag, axis=1), textposition="top center",
        textfont=dict(size=9, color=T.GRAY),
        marker=dict(size=12, color=merged["Importance"], colorscale=T.SEQUENTIAL,
                    line=dict(color=T.PRIMARY, width=1)),
        customdata=merged[["Corr rank", "RF rank", "Rank shift"]],
        hovertemplate="%{text}<br>|Correlation|: %{x:.3f}<br>" + label + ": %{y:.4f}"
                      "<br>Corr rank: %{customdata[0]} | RF rank: %{customdata[1]}"
                      "<br>Rank shift: %{customdata[2]:+d}<extra></extra>",
    ))
    fig.update_xaxes(title="Correlation with churn", rangemode="tozero")
    fig.update_yaxes(title=label)
    return T.style(fig, height=480, legend=False)


# ---------------------------------------------------------------------------
# Risk scorer
# ---------------------------------------------------------------------------
def risk_factor_bar(flags, rates, baseline):
    """Historical churn rate for each risk factor present in a scored profile."""
    axis_max = max(list(rates) + [baseline]) * 100 * 1.3
    fig = go.Figure(go.Bar(
        x=[r * 100 for r in rates], y=list(flags), orientation="h",
        marker_color=T.PRIMARY, cliponaxis=False,
        text=[f"{r:.1%}" for r in rates], textposition="outside",
        hovertemplate="%{y}<br>Historical churn rate: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=baseline * 100, line_dash="dash", line_color=T.GRAY)
    fig.add_annotation(
        x=baseline * 100, y=1, xref="x", yref="paper",
        text=f"Overall baseline {baseline:.1%}", showarrow=True, arrowhead=2,
        arrowsize=1, arrowwidth=1.5, arrowcolor=T.GRAY, ax=0, ay=-30,
        xanchor="center", yanchor="bottom", font=dict(size=11, color=T.GRAY),
        bgcolor="rgba(255,255,255,0.85)",
    )
    fig.update_xaxes(range=[0, axis_max],
                     title="Historical churn rate for this segment (%)")
    fig.update_yaxes(autorange="reversed")
    return T.style(fig, height=95 + 55 * len(flags), legend=False,
                   margin=dict(l=10, r=30, t=60, b=40))


# ---------------------------------------------------------------------------
# What-if analysis (new - no PNG equivalent in the notebook)
# ---------------------------------------------------------------------------
def sensitivity(raw, model_name, feature, current_value):
    """How this customer's risk moves as one feature is swept end to end."""
    if feature == "tenure":
        values = list(range(0, 73, 2))
        axis_title = "Tenure (months)"
    else:
        values = [round(v, 1) for v in np.arange(18.0, 120.1, 2.0)]
        axis_title = "Monthly charges (RM)"

    data = A.sensitivity_curve(raw, model_name, feature, values)
    risk = data["Risk"] * 100

    fig = go.Figure()
    for lower, upper, color in [(0, T.MEDIUM_RISK_AT, T.ZONE_TINTS["low"]),
                                (T.MEDIUM_RISK_AT, T.HIGH_RISK_AT, T.ZONE_TINTS["medium"]),
                                (T.HIGH_RISK_AT, 100, T.ZONE_TINTS["high"])]:
        fig.add_hrect(y0=lower, y1=upper, fillcolor=color, opacity=0.55,
                      layer="below", line_width=0)
    fig.add_trace(go.Scatter(
        x=data[feature], y=risk, mode="lines", name="Predicted risk",
        line=dict(color=T.PRIMARY, width=3),
        hovertemplate=axis_title + ": %{x}<br>Churn risk: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[current_value],
        y=[float(np.interp(current_value, data[feature], risk))],
        mode="markers", name="This customer",
        marker=dict(size=14, color=T.TEXT, line=dict(color="white", width=2)),
        hovertemplate="This customer<br>Churn risk: %{y:.1f}%<extra></extra>",
    ))
    fig.update_xaxes(title=axis_title)
    fig.update_yaxes(title="Predicted churn risk", ticksuffix="%", range=[0, 100])
    return T.style(fig, height=380)


def contribution_chart(breakdown, actual_risk):
    """Which attributes push this customer's risk up or down, versus a typical
    customer. Red bars add risk, blue bars remove it."""
    colors = [T.CHURN if v > 0 else T.PRIMARY for v in breakdown["Contribution"]]
    fig = go.Figure(go.Bar(
        x=breakdown["Contribution"] * 100, y=breakdown["Field"], orientation="h",
        marker=dict(color=colors), cliponaxis=False,
        text=[f"{v * 100:+.1f}" for v in breakdown["Contribution"]],
        textposition="outside", textfont=dict(size=10),
        customdata=breakdown["Value"].astype(str),
        hovertemplate="%{y}: %{customdata}<br>%{x:+.1f} pts vs a typical customer"
                      "<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=T.GRAY, line_width=1)
    fig.update_xaxes(title="Effect on churn risk (percentage points)", ticksuffix=" pts")
    return T.style(fig, height=max(300, 30 * len(breakdown) + 90), legend=False,
                   margin=dict(l=10, r=60, t=24, b=48))


def lever_chart(levers, current_risk, top_n=6):
    """The changes that would move this customer's risk the most."""
    data = levers.head(top_n).sort_values("Reduction")
    labels = [f"{row['Field']} -> {row['Change to']}" for _, row in data.iterrows()]
    colors = [T.GREEN if v > 0 else T.CHURN for v in data["Reduction"]]
    fig = go.Figure(go.Bar(
        x=data["Reduction"] * 100, y=labels, orientation="h",
        marker=dict(color=colors), cliponaxis=False,
        text=[f"{v * 100:+.1f}" for v in data["Reduction"]],
        textposition="outside", textfont=dict(size=10),
        customdata=data["New risk"] * 100,
        hovertemplate="%{y}<br>Risk would become %{customdata:.1f}%"
                      "<br>Change: %{x:+.1f} pts<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=T.GRAY, line_width=1)
    fig.update_xaxes(title="Risk reduction (percentage points)", ticksuffix=" pts")
    return T.style(fig, height=max(280, 34 * len(data) + 90), legend=False,
                   margin=dict(l=10, r=60, t=24, b=48))


def categorical_whatif_chart(data, field_label, current_value):
    """Risk under each option of one categorical field, current option outlined."""
    band_colors = [T.risk_band(v * 100)[1] for v in data["Risk"]]
    outlines = [T.TEXT if opt == current_value else "white" for opt in data["Option"]]
    widths = [3 if opt == current_value else 1 for opt in data["Option"]]
    fig = go.Figure(go.Bar(
        x=data["Option"].astype(str), y=data["Risk"] * 100,
        marker=dict(color=band_colors,
                    line=dict(color=outlines, width=widths)),
        text=[f"{v:.1%}" + ("  (current)" if o == current_value else "")
              for v, o in zip(data["Risk"], data["Option"])],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="%{x}<br>Churn risk: %{y:.1f}%<extra></extra>",
    ))
    for lower, upper, color in [(0, T.MEDIUM_RISK_AT, T.ZONE_TINTS["low"]),
                                (T.MEDIUM_RISK_AT, T.HIGH_RISK_AT, T.ZONE_TINTS["medium"]),
                                (T.HIGH_RISK_AT, 100, T.ZONE_TINTS["high"])]:
        fig.add_hrect(y0=lower, y1=upper, fillcolor=color, opacity=0.45,
                      layer="below", line_width=0)
    fig.update_xaxes(title=field_label)
    fig.update_yaxes(title="Predicted churn risk", ticksuffix="%", range=[0, 100])
    return T.style(fig, height=360, legend=False)
