"""
BMDS2003 Data Science - Deployment Prototype
Telco Customer Churn Risk Scorer
Run locally:  streamlit run streamlit_app.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend - no GUI/display available in Streamlit
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Telco Churn Risk Scorer", page_icon="signal", layout="wide")
st.title("Telco Customer Churn Risk Scorer")
st.caption("BMDS2003 Data Science - Group Assignment Deployment Prototype")

APP_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_bundle():
    with open("trained_models.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    return pd.read_csv("telco_churn_cleaned.csv")


bundle = load_bundle()
df = load_data()

# Overall churn rate, used as the baseline reference line in the result tab.
BASELINE_CHURN = float((df["Churn"] == "Yes").mean())


@st.cache_data
def compute_factor_rates(_df):
    """Historical churn rate for each risk-factor segment, computed from the data
    (leading underscore keeps Streamlit from trying to hash the DataFrame)."""
    def rate(mask):
        segment = _df[mask]
        return float((segment["Churn"] == "Yes").mean()) if len(segment) else 0.0

    return {
        "Month-to-month contract": rate(_df["Contract"] == "Month-to-month"),
        "Tenure in first 12 months": rate(_df["tenure"] <= 12),
        "Fiber optic internet": rate(_df["InternetService"] == "Fiber optic"),
        "Electronic check payment": rate(_df["PaymentMethod"] == "Electronic check"),
        "No online security & no tech support": rate(
            (_df["OnlineSecurity"] == "No") & (_df["TechSupport"] == "No")
        ),
    }


FACTOR_RATES = compute_factor_rates(df)


def render_speedometer(pct, bar_color, baseline_pct):
    """Build a genuine speedometer-style gauge: a semicircular green/amber/red
    dial with a needle pointing at the predicted churn risk, plus a short dark
    tick marking the overall baseline churn rate for comparison.

    Plotly's own gauge (go.Indicator) can only show the value as a filled arc
    or a number - it has no real rotating needle - so this is drawn with
    Matplotlib instead, which supports rotating pointer needles directly.
    """
            
    zones = [(0, 33.3, "#DCFCE7"), (33.3, 66.6, "#FEF3C7"), (66.6, 100, "#FEE2E2")]
    r_outer, r_inner = 1.0, 0.72

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)

    # Colour-zone ring (green / amber / red), each drawn as a donut wedge.
    # Angle convention: value 0% -> 180 deg (left), value 100% -> 0 deg (right),
    # sweeping over the top - the usual speedometer layout.
    for lo, hi, color in zones:
        theta_lo = 180 - (lo / 100) * 180
        theta_hi = 180 - (hi / 100) * 180
        wedge = mpatches.Wedge((0, 0), r_outer, theta_hi, theta_lo, width=r_outer - r_inner,
                                facecolor=color, edgecolor="black", linewidth=2.0)
        ax.add_patch(wedge)

    # Value-fill arc: a saturated, bar_color-tinted overlay from 0% up to the
    # predicted value, drawn on top of the pale zone bands - this is the same
    # "how far does it reach" shading the old Plotly gauge showed via its
    # `bar` property, kept here alongside the needle rather than instead of it.
    if pct > 0:
        fill_theta_lo = 180 - (min(pct, 100) / 100) * 180
        fill_wedge = mpatches.Wedge((0, 0), r_outer, fill_theta_lo, 180, width=r_outer - r_inner,
                                     facecolor=bar_color, edgecolor="none", zorder=2)
        ax.add_patch(fill_wedge)

    # Tick labels at the zone boundaries.
    for v in [0, 33, 66, 100]:
        ang = np.radians(180 - (v / 100) * 180)
        ax.text(1.14 * np.cos(ang), 1.14 * np.sin(ang), f"{v}%",
                 ha="center", va="center", fontsize=9, color="#4B5563")

    # Baseline reference: a short dark tick just outside the ring.
    base_ang = np.radians(180 - (baseline_pct / 100) * 180)
    ax.plot([r_outer * np.cos(base_ang), 1.08 * np.cos(base_ang)],
             [r_outer * np.sin(base_ang), 1.08 * np.sin(base_ang)],
             color="#1F2430", linewidth=2.5, solid_capstyle="round")

    # Needle - a slim kite/triangle pivoting at the hub, pointing at pct.
    needle_ang = np.radians(180 - (pct / 100) * 180)
    tip = (r_inner * 0.98 * np.cos(needle_ang), r_inner * 0.98 * np.sin(needle_ang))
    base_l = (0.09 * np.cos(needle_ang + np.pi / 2), 0.09 * np.sin(needle_ang + np.pi / 2))
    base_r = (0.09 * np.cos(needle_ang - np.pi / 2), 0.09 * np.sin(needle_ang - np.pi / 2))
    ax.add_patch(mpatches.Polygon([base_l, tip, base_r], closed=True,
                                   facecolor="#1F2430", edgecolor="#1F2430", zorder=5))
    ax.add_patch(mpatches.Circle((0, 0), 0.09, facecolor="#1F2430", edgecolor="white",
                                  linewidth=1.5, zorder=6))

    # Value readout, colour-matched to the risk zone.
    ax.text(0, -0.32, f"{pct:.1f}%", ha="center", va="center",
             fontsize=22, fontweight="bold", color=bar_color)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.5, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    return fig


# The pickle bundle stores every trained model, not just the winner, so the
# user can pick any of them at runtime instead of always using Random Forest.
BEST_MODEL_NAME = bundle["best_model_name"]
AVAILABLE_MODELS = list(bundle["models"].keys())
FEATURES = bundle["feature_names"]

# Keep the original model keys for prediction, but show clearer names in the UI.
MODEL_DISPLAY_NAMES = {
    "Random Forest": "Random Forest (Best Model)",
    "Logistic Regression": "Logistic Regression (Baseline Model)",
}


def format_model_name(model_name):
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


# Place Random Forest first and make it the default model in the scorer.
DEFAULT_MODEL_NAME = "Random Forest" if "Random Forest" in AVAILABLE_MODELS else BEST_MODEL_NAME
ORDERED_MODELS = [DEFAULT_MODEL_NAME] + [
    model_name for model_name in AVAILABLE_MODELS if model_name != DEFAULT_MODEL_NAME
]

# Style the model selector so all four cards fill the complete content width.
st.markdown(
    """
    <style>
    /* Full-width four-card model selector, spanning the whole page content. */
    div.st-key-model_selector {
        width: 100%;
    }

    div.st-key-model_selector div[data-testid="stHorizontalBlock"] {
        width: 100% !important;
        display: flex !important;
        align-items: stretch !important;
        gap: 1rem !important;
    }

    div.st-key-model_selector div[data-testid="stColumn"] {
        flex: 1 1 0 !important;
        width: 0 !important;
        min-width: 0 !important;
    }

    div.st-key-model_selector div[data-testid="stButton"] {
        width: 100% !important;
        height: 100% !important;
    }

    div.st-key-model_selector button {
        width: 100% !important;
        min-height: 4rem !important;
        height: 100% !important;
        padding: 0.65rem 0.75rem !important;
        border-radius: 0.75rem !important;
        white-space: normal !important;
        line-height: 1.25 !important;
        font-weight: 600 !important;
    }

    div.st-key-model_selector button p {
        width: 100% !important;
        margin: 0 !important;
        text-align: center !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }

    /* Stack into a 2x2 grid, then a single column, on narrower screens. */
    @media (max-width: 900px) {
        div.st-key-model_selector div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }

        div.st-key-model_selector div[data-testid="stColumn"] {
            flex: 1 1 calc(50% - 0.5rem) !important;
            width: calc(50% - 0.5rem) !important;
        }
    }

    @media (max-width: 480px) {
        div.st-key-model_selector div[data-testid="stColumn"] {
            flex-basis: 100% !important;
            width: 100% !important;
        }
    }

    /* ========================================================== */
    /* FORCEFULLY REMOVE THE 1PX BORDER ON ALL CONTAINERS         */
    /* ========================================================== */
    [data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 0px !important;
        border-style: none !important;
        border-color: transparent !important;
        border-width: 0px !important;
    }

    /* Tighten the gap between the Customer profile and Result columns. */
    div.st-key-scorer_row div[data-testid="stHorizontalBlock"] {
        gap: 1.25rem !important;
        align-items: stretch !important;
    }

    /* Force the columns to stretch to the same height */
    div.st-key-scorer_row div[data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
    }

    /* Force the bordered container inside the column to fill the column's height */
    div.st-key-scorer_row div[data-testid="stColumn"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        flex: 1 1 auto !important;
        height: 100% !important;
    }

    /* "3. Result" panel only - give it a soft shadow so it still stands out */
    div.st-key-result_panel div[data-testid="stVerticalBlockBorderWrapper"] {
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08) !important;
    }

    div.st-key-result_panel div[data-testid="stVerticalBlock"] {
        padding: 0.25rem 0.4rem !important;
        gap: 0.9rem !important;
    }

    /* Speedometer gauge (Matplotlib image) - center it and cap its width */
    div.st-key-gauge_chart div[data-testid="stVerticalBlock"] {
        display: flex;
        justify-content: center;
    }

    div.st-key-gauge_chart div[data-testid="stElementContainer"] {
        max-width: 380px;
        width: 100% !important;
        margin: 0 auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



def select_model(model_name):
    """Store the selected model so the highlighted choice persists after reruns."""
    st.session_state["selected_model_name"] = model_name


# Each entry is (filename, image caption, presentation note).
EDA_FIGS = [
    ("fig0_preparation_validation.png", "Data preparation & validation checks",
     "Confirms the dataset is clean before modelling: no duplicate rows, correct data types, "
     "and missing values limited to the expected zero-tenure TotalCharges cases."),
    ("fig1_churn_pie.png", "Overall customer churn distribution",
     "About 1 in 4 customers (~26.5%) churned, so the target is imbalanced - this is why "
     "Recall and F1 matter more than plain Accuracy for this problem."),
    ("fig2_mosaic_categorical.png", "Categorical feature relationships (mosaic plot)",
     "The churn share (colour) clearly shifts across contract type and internet service tiles, "
     "flagging both as strong categorical predictors."),
    ("fig3_cramers_v_lollipop.png", "Association strength of categorical features (Cramer's V)",
     "Contract, OnlineSecurity, TechSupport and InternetService rank highest, making them the "
     "categorical features worth highlighting first in the presentation."),
    ("fig4_numeric_violin.png", "Distribution of numeric features by churn status",
     "Churners cluster at low tenure and higher MonthlyCharges, while retained customers spread "
     "across much longer tenures."),
    ("fig5_ecdf.png", "Empirical cumulative distribution functions",
     "The ECDF curves for churners sit further left on tenure and further right on "
     "MonthlyCharges, backing up the pattern seen in the violin plots with cumulative evidence."),
    ("fig6_interaction_heatmap.png", "Feature interaction heatmap",
     "Risk compounds when features combine: month-to-month contract plus fiber optic internet "
     "is the single riskiest combination in the heatmap."),
    ("fig7_num_services_dual_axis.png", "Number of subscribed services vs churn rate",
     "Churn rate drops steadily as customers add more services, even though most customers sit "
     "at the low end of the service count."),
    ("fig8_scatter_matrix.png", "Scatter matrix of numeric features",
     "Numeric features alone only partly separate churners from non-churners, which is why the "
     "categorical and engineered features are still needed for the model."),
]

CORR_FIGS = [
    ("fig9a_correlation_full.png", "Full correlation matrix (all features)",
     "Highlights expected collinearity (e.g. TotalCharges with tenure) that was considered "
     "during feature engineering."),
    ("fig9b_correlation_top15.png", "Top 15 correlated features",
     "Contract- and tenure-related engineered features dominate the top 15 correlations with "
     "churn."),
    ("fig9c_correlation_ranked.png", "Features ranked by correlation with churn",
     "Two-year contracts and long tenure correlate negatively with churn, while fiber optic and "
     "month-to-month correlate positively - two ends of the same story."),
]

MODEL_FIGS = [
    ("fig11_roc_curves.png", "ROC curves for all models",
     "All four models clear the diagonal comfortably; Random Forest and XGBoost post the "
     "highest AUC, meaning they rank churners vs non-churners best."),
    ("fig12_confusion_matrices.png", "Confusion matrices for all models",
     "Decision Tree and XGBoost catch slightly more true churners (higher recall) but at the "
     "cost of more false alarms compared to Random Forest."),
    ("fig13_cv_stability.png", "Cross-validation stability across folds",
     "Random Forest and XGBoost show tight, consistent scores across folds, indicating the "
     "results generalise rather than depending on one lucky data split."),
    ("fig14_overfitting_check.png", "Train vs test performance (overfitting check)",
     "The train-test gap stays small for Random Forest, suggesting limited overfitting compared "
     "to the wider gap on the Decision Tree."),
]

IMPORTANCE_FIGS = [
    ("fig15_feature_importance.png", "Feature importance (best model)",
     "Contract, tenure and TotalCharges carry the most weight when the Random Forest splits "
     "churners from non-churners."),
    ("fig16_correlation_vs_importance.png", "Correlation vs model-based feature importance",
     "Model-based importance mostly agrees with the raw correlation ranking, which is a good "
     "sanity check that the model is picking up genuine signal rather than noise."),
    ("fig17_permutation_importance.png", "Permutation importance",
     "Permutation importance confirms Contract and tenure as the features whose removal hurts "
     "predictive performance the most."),
]


def show_gallery(fig_list, columns=2):
    """Render figures in a responsive grid and flag any missing image files."""
    cols = st.columns(columns)
    for i, (filename, caption, note) in enumerate(fig_list):
        path = APP_DIR / filename
        with cols[i % columns]:
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
                st.caption(note)
            else:
                st.warning(f"Missing file: {filename}")


def highlight_model_rows(row):
    """Add distinct colours to the baseline and best-model rows."""
    if row["Model"] == "Logistic Regression (Baseline Model)":
        style = "background-color: #DCEBFF; color: #0B2E59; font-weight: 700;"
        return [style] * len(row)
    if row["Model"] == "Random Forest (Best Model)":
        style = "background-color: #DDF5E5; color: #123D24; font-weight: 700;"
        return [style] * len(row)
    return [""] * len(row)


tab_scorer, tab_model_evaluation, tab_eda = st.tabs([
    "Risk Scorer",
    "Model Evaluation",
    "EDA",
])

# ----------------------------------------------------------------------
# TAB 1: Risk Scorer
# ----------------------------------------------------------------------
with tab_scorer:
    # ------------------------------------------------------------------
    # ROW 1: model choice - full page width, no scrolling needed to reach it
    # ------------------------------------------------------------------
    with st.container(border=True, key="model_panel"):
        st.markdown("#### 1. Choose a model")
        st.caption("Model used to score this customer")

        # Initialise the default once, then keep the user's latest choice in session state.
        if st.session_state.get("selected_model_name") not in ORDERED_MODELS:
            st.session_state["selected_model_name"] = DEFAULT_MODEL_NAME

        # Keyed container so the four model cards get the full-width row styling above.
        with st.container(key="model_selector"):
            model_columns = st.columns([1, 1, 1, 1], gap="small")
            for index, (column, model_name) in enumerate(zip(model_columns, ORDERED_MODELS)):
                is_selected = st.session_state["selected_model_name"] == model_name
                with column:
                    st.button(
                        format_model_name(model_name),
                        key=f"model_choice_{index}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True,
                        on_click=select_model,
                        args=(model_name,),
                        help=f"Use {format_model_name(model_name)} to calculate the churn risk.",
                    )

    selected_model_name = st.session_state["selected_model_name"]
    model = bundle["models"][selected_model_name]

    # ------------------------------------------------------------------
    # ROW 2: Customer profile (left) | Result (right)
    # ------------------------------------------------------------------
    with st.container(key="scorer_row"):
        left, right = st.columns([1, 1], gap="medium")

        with left:
            with st.container(border=True, key="profile_panel"):
                st.markdown("#### 2. Customer profile")
                c1, c2 = st.columns(2)
                with c1:
                    tenure = st.slider("Tenure (months)", 0, 72, 12)
                    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
                    payment = st.selectbox("Payment method", sorted(df["PaymentMethod"].unique()))
                    monthly = st.slider("Monthly charges (RM)", 18.0, 120.0, 70.0, step=0.5)
                with c2:
                    gender = st.selectbox("Gender", ["Female", "Male"])
                    senior = st.selectbox("Senior citizen", ["No", "Yes"])
                    partner = st.selectbox("Has partner", ["No", "Yes"])
                    dependents = st.selectbox("Has dependents", ["No", "Yes"])
                    paperless = st.selectbox("Paperless billing", ["No", "Yes"])


                st.markdown("**Services**")
                s1, s2 = st.columns(2)
                with s1:
                    phone = st.checkbox("Phone service", value=True)
                    multiple = st.checkbox("Multiple lines")
                    security = st.checkbox("Online security")
                    backup = st.checkbox("Online backup")
                with s2:
                    protection = st.checkbox("Device protection")
                    support = st.checkbox("Tech support")
                    tv = st.checkbox("Streaming TV")
                    movies = st.checkbox("Streaming movies")

                yn = lambda b: "Yes" if b else "No"
                total = round(tenure * monthly, 2)

                raw = {
                    "gender": gender, "SeniorCitizen": senior, "Partner": partner,
                    "Dependents": dependents, "tenure": tenure, "PhoneService": yn(phone),
                    "MultipleLines": yn(multiple), "InternetService": internet,
                    "OnlineSecurity": yn(security), "OnlineBackup": yn(backup),
                    "DeviceProtection": yn(protection), "TechSupport": yn(support),
                    "StreamingTV": yn(tv), "StreamingMovies": yn(movies),
                    "Contract": contract, "PaperlessBilling": paperless,
                    "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
                }

                st.caption(f"Estimated total charges: RM {total:,.2f}  (tenure x monthly charges)")

                if st.button("Score this customer", type="primary", use_container_width=True):
                    row = pd.DataFrame([raw])

                    # Rebuild the engineered features exactly as in Part 2 of the notebook.
                    row["tenure_group"] = pd.cut(
                        row["tenure"],
                        bins=[-1, 12, 24, 48, 72],
                        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
                    )
                    row["num_services"] = (row[bundle["service_cols"]] == "Yes").sum(axis=1)

                    for c in bundle["binary_cols"]:
                        row[c] = row[c].map({"No": 0, "Yes": 1, "Female": 0, "Male": 1})
                    row = pd.get_dummies(
                        row,
                        columns=bundle["nominal_cols"],
                        drop_first=False,
                        dtype=int,
                    )

                    # Align to the exact training columns (missing dummies -> 0, extras dropped).
                    row = row.reindex(columns=FEATURES, fill_value=0).astype(float)

                    # Only Logistic Regression was trained on scaled numeric features.
                    if selected_model_name in bundle["needs_scaling"]:
                        row[bundle["numeric_cols"]] = bundle["scaler"].transform(
                            row[bundle["numeric_cols"]]
                        )

                    proba = float(model.predict_proba(row)[0, 1])
                    pred = int(model.predict(row)[0])

                    # Which of the known risk factors this profile matches.
                    flags = []
                    if contract == "Month-to-month":
                        flags.append("Month-to-month contract")
                    if tenure <= 12:
                        flags.append("Tenure in first 12 months")
                    if internet == "Fiber optic":
                        flags.append("Fiber optic internet")
                    if payment == "Electronic check":
                        flags.append("Electronic check payment")
                    if not security and not support:
                        flags.append("No online security & no tech support")

                    peers = df[(df["Contract"] == contract) & (df["InternetService"] == internet)]
                    peer_rate = float((peers["Churn"] == "Yes").mean()) if len(peers) > 0 else 0.0

                    # Store the result in session_state so it survives reruns and renders
                    # in the right-hand column (which is built before this button's code runs).
                    st.session_state["result"] = {
                        "model_name": format_model_name(selected_model_name),
                        "proba": proba,
                        "pred": pred,
                        "flags": flags,
                        "factor_rates": FACTOR_RATES,
                        "peer_count": len(peers),
                        "peer_rate": peer_rate,
                        "contract": contract,
                        "internet": internet,
                    }

        # --------------------------------------------------------------
        # RIGHT: result (gauge chart on top, risk-factor detail below)
        # --------------------------------------------------------------
        with right:
            with st.container(border=True, key="result_panel"):
                st.markdown("#### 3. Result")
                result = st.session_state.get("result")

                if result:
                    st.caption(f"Prediction generated using **{result['model_name']}**")

                    pct = result["proba"] * 100
                    bar_color = "#DC2626" if pct >= 66.6 else ("#F59E0B" if pct >= 33.3 else "#16A34A")
                    gauge_fig = render_speedometer(pct, bar_color, BASELINE_CHURN * 100)
                    with st.container(key="gauge_chart"):
                        st.pyplot(gauge_fig, use_container_width=True)
                    plt.close(gauge_fig)
                    st.caption(f"Dark tick marks the overall baseline churn rate ({BASELINE_CHURN:.1%}).")

                    if pct >= 66.6:
                        st.error("**HIGH RISK - flag for retention contact.**")
                    elif pct >= 33.3:
                        st.warning("**MEDIUM RISK - monitor customer closely.**")
                    else:
                        st.success("**LOW RISK - no retention action required.**")
                    flags = result["flags"]
                    
                    if flags:
                        st.markdown("**Risk factors present in this profile** (from the notebook's EDA):")
                        rates = [result["factor_rates"][f] for f in flags]
                        colors = ["#DC2626" if r > BASELINE_CHURN else "#16A34A" for r in rates]

                        # Fixed top margin (in pixels, not a fraction) so the baseline
                        # label always gets the same dedicated space up top, whether
                        # there's 1 risk factor or 5 - it never has to fight the bars
                        # for room the way the old inline annotation did.
                        max_rate_pct = max(rates + [BASELINE_CHURN]) * 100
                        axis_max = max_rate_pct * 1.3

                        bar = go.Figure(go.Bar(
                            x=[r * 100 for r in rates], y=flags, orientation="h",
                            marker_color=colors,
                            text=[f"{r:.1%}" for r in rates], textposition="outside",
                            cliponaxis=False,
                            hovertemplate="%{y}<br>Historical churn rate: %{x:.1f}%<extra></extra>",
                        ))
                        bar.add_vline(x=BASELINE_CHURN * 100, line_dash="dash", line_color="#6B7280")
                        bar.add_annotation(
                            x=BASELINE_CHURN * 100, y=1, xref="x", yref="paper",
                            text=f"Overall baseline {BASELINE_CHURN:.1%}",
                            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                            arrowcolor="#6B7280", ax=0, ay=-30,
                            xanchor="center", yanchor="bottom",
                            font=dict(size=11, color="#6B7280"),
                            bgcolor="rgba(255,255,255,0.85)",  
                        )
                        bar.update_layout(
                            height=95 + 55 * len(flags), margin=dict(l=10, r=30, t=60, b=30),
                            xaxis=dict(range=[0, axis_max]),
                            xaxis_title="Historical churn rate for this segment (%)",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})
                    else:
                        st.info("No major risk factor from the top drivers identified in the EDA.")

                    if result["peer_count"] > 0:
                        st.caption(
                            f"📌 Historical benchmark: {result['peer_count']:,} customers in the data "
                            f"share this contract + internet combination ({result['contract']} / "
                            f"{result['internet']}), and {result['peer_rate']:.1%} of them churned."
                        )
                else:
                    st.info(
                        "Fill in the customer profile on the left and click **Score this customer** "
                        "to see the risk result here."
                    )

    st.divider()
    st.caption(
        "Academic prototype trained on 7,043 historical customer records. It outputs a churn "
        "RISK SCORE for retention triage, not a prediction about an individual's intentions. "
        "The associations shown are correlational, not causal, and the model must be retrained "
        "as customer behaviour and tariffs change. Not for real commercial or financial decisions."
    )

# ----------------------------------------------------------------------
# TAB 2: Model Evaluation
# ----------------------------------------------------------------------
with tab_model_evaluation:
    st.caption("Performance comparison and diagnostic charts for all four trained models")

    results_path = APP_DIR / "model_comparison_results.csv"
    if results_path.exists():
        st.subheader("Metrics table")
        results_df = pd.read_csv(results_path, index_col=0)

        results_df = results_df.rename(index={
            "Logistic Regression": "Logistic Regression (Baseline Model)",
            "Random Forest": "Random Forest (Best Model)",
        })

        display_results = results_df.reset_index()
        display_results = display_results.rename(columns={display_results.columns[0]: "Model"})
        styled_results = display_results.style.apply(highlight_model_rows, axis=1).format(precision=3)

        st.dataframe(
            styled_results,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Missing file: model_comparison_results.csv")

    # The overall comparison is full-width directly below the metrics table.
    fig10_path = APP_DIR / "fig10_model_comparison.png"
    if fig10_path.exists():
        st.image(
            str(fig10_path),
            caption="Comparison of all four models",
            use_container_width=True,
        )
        st.caption(
            "Random Forest and XGBoost lead on F1 and ROC-AUC; Random Forest is the "
            "notebook's deployed default for its balance of precision and recall."
        )
    else:
        st.warning("Missing file: fig10_model_comparison.png")

    show_gallery(MODEL_FIGS)

# ----------------------------------------------------------------------
# TAB 3: EDA
# ----------------------------------------------------------------------
with tab_eda:
    st.caption("Exploratory analysis, correlation analysis and feature-importance charts")

    sub1, sub2, sub3 = st.tabs([
        "Exploratory Data Analysis",
        "Correlation Analysis",
        "Feature Importance",
    ])

    with sub1:
        show_gallery(EDA_FIGS)

    with sub2:
        show_gallery(CORR_FIGS)

    with sub3:
        show_gallery(IMPORTANCE_FIGS)
