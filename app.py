"""
BMDS2003 Data Science - Deployment Prototype
Telco Customer Churn Risk Scorer
Run locally:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import pickle
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Telco Churn Risk Scorer", page_icon="📡", layout="wide")

APP_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# Global styling - fonts, hero banner, card polish, gallery image sizing.
# Colours for buttons / sliders / radio / checkboxes come from
# .streamlit/config.toml (primaryColor) rather than being hard-coded here,
# so alert colours (info/success/error) stay correctly semantic.
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
/* font-family is inherited by default, so setting it once on html/body is
   enough and - unlike a class/attribute wildcard - can never out-specificity
   Streamlit's own icon-font rules (sidebar collapse arrow, expander chevron). */
html, body { font-family: 'Manrope', sans-serif; }

.hero-banner {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    border-radius: 18px;
    padding: 1.9rem 2.2rem;
    color: white;
    margin-bottom: 1.1rem;
}
.hero-banner h1 { margin: 0 0 0.35rem 0; font-size: 2rem; font-weight: 800; color: white; }
.hero-banner p { margin: 0; opacity: 0.92; font-size: 1.02rem; }

/* Gallery images (inside st.columns grids) get a consistent height so mixed
   aspect-ratio figures line up in the grid instead of jagged rows. The
   full-width fig10 chart lives outside any column, so it is unaffected. */
div[data-testid="stColumn"] div[data-testid="stImage"] img {
    height: 230px;
    width: 100%;
    object-fit: contain;
    background: #ffffff;
    border-radius: 8px;
    padding: 4px;
}

details.insight { margin: -0.3rem 0 0.6rem 0; font-size: 0.88rem; }
details.insight summary { cursor: pointer; font-weight: 600; color: #4F46E5; }
details.insight p { color: #4B5563; margin: 0.4rem 0 0 0; }

/* Keep longer metric values (e.g. a model name) from being clipped with an
   ellipsis inside the narrow stat cards. */
[data-testid="stMetricValue"] {
    font-size: 1.5rem;
    overflow: visible;
    white-space: normal;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    with open(APP_DIR / "trained_models.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(APP_DIR / "telco_churn_cleaned.csv")


bundle = load_bundle()
df = load_data()

BEST_MODEL_NAME = bundle["best_model_name"]
AVAILABLE_MODELS = list(bundle["models"].keys())
FEATURES = bundle["feature_names"]
BASELINE_CHURN = (df["Churn"] == "Yes").mean()

# ----------------------------------------------------------------------
# Sidebar - model choice + context, kept separate from the input form so
# the main area stays focused on "fill in a customer, get a score".
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📡 Model settings")
    selected_model_name = st.radio(
        "Model used to score this customer",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(BEST_MODEL_NAME),
        help="All 4 models from the notebook (Part 5) are bundled in trained_models.pkl. "
             "Random Forest was selected as the notebook's best model, but you can compare "
             "predictions from any of them here.",
    )
    model = bundle["models"][selected_model_name]
    metrics = bundle["results"].loc[selected_model_name]
    is_best = selected_model_name == BEST_MODEL_NAME
    badge = "notebook's best model" if is_best else "not the notebook's best model"
    st.info(f"**{selected_model_name}** _({badge})_\n\n"
            f"F1 = {metrics['F1']:.3f}  \nRecall = {metrics['Recall']:.3f}  \n"
            f"ROC-AUC = {metrics['ROC-AUC']:.3f}")

    with st.expander("ℹ️ About this prototype"):
        st.caption(
            "Academic prototype trained on 7,043 historical customer records. It outputs a "
            "churn RISK SCORE for retention triage, not a prediction about an individual's "
            "intentions. The associations shown are correlational, not causal, and the model "
            "must be retrained as customer behaviour and tariffs change. Not for real "
            "commercial or financial decisions.")

# ----------------------------------------------------------------------
# Hero header + quick dataset stats
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>Telco Customer Churn Risk Scorer</h1>
    <p>BMDS2003 Data Science - Group Assignment Deployment Prototype</p>
</div>
""", unsafe_allow_html=True)

stat1, stat2, stat3, stat4 = st.columns(4)
with stat1:
    with st.container(border=True):
        st.metric("Training records", f"{len(df):,}")
with stat2:
    with st.container(border=True):
        st.metric("Historical churn rate", f"{BASELINE_CHURN:.1%}")
with stat3:
    with st.container(border=True):
        st.metric("Best model", BEST_MODEL_NAME)
with stat4:
    with st.container(border=True):
        st.metric("Best model ROC-AUC", f"{bundle['results'].loc[BEST_MODEL_NAME, 'ROC-AUC']:.3f}")

tab_scorer, tab_report = st.tabs(["🧮 Risk Scorer", "📊 EDA & Model Report"])

# ----------------------------------------------------------------------
# TAB 1: Risk Scorer
# ----------------------------------------------------------------------
with tab_scorer:
    with st.container(border=True):
        st.markdown("#### 1. Customer profile")
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            payment = st.selectbox("Payment method", sorted(df["PaymentMethod"].unique()))
            monthly = st.slider("Monthly charges (RM)", 18.0, 120.0, 70.0, step=0.5)
            total = round(tenure * monthly, 2)
            st.caption(f"Estimated total charges: RM {total:,.2f}  (tenure × monthly charges)")
        with c2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has partner", ["No", "Yes"])
            dependents = st.selectbox("Has dependents", ["No", "Yes"])
            paperless = st.selectbox("Paperless billing", ["No", "Yes"])

    with st.container(border=True):
        st.markdown("#### 2. Services")
        s1, s2 = st.columns(2)
        with s1:
            phone = st.checkbox("☎️ Phone service", value=True)
            multiple = st.checkbox("📞 Multiple lines")
            security = st.checkbox("🔒 Online security")
            backup = st.checkbox("💾 Online backup")
        with s2:
            protection = st.checkbox("🛡️ Device protection")
            support = st.checkbox("🛠️ Tech support")
            tv = st.checkbox("📺 Streaming TV")
            movies = st.checkbox("🎬 Streaming movies")

    yn = lambda b: "Yes" if b else "No"
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

    btn_col, clear_col = st.columns([4, 1])
    with btn_col:
        score_clicked = st.button("⚡ Score this customer", type="primary", width="stretch")
    with clear_col:
        if st.session_state.get("last_result") and st.button("✕ Clear", width="stretch"):
            st.session_state.pop("last_result", None)
            st.rerun()

    if score_clicked:
        row = pd.DataFrame([raw])

        # Rebuild the engineered features exactly as in Part 2 of the notebook
        row["tenure_group"] = pd.cut(row["tenure"], bins=[-1, 12, 24, 48, 72],
                                     labels=["0-12m", "13-24m", "25-48m", "49-72m"])
        row["num_services"] = (row[bundle["service_cols"]] == "Yes").sum(axis=1)

        for c in bundle["binary_cols"]:
            row[c] = row[c].map({"No": 0, "Yes": 1, "Female": 0, "Male": 1})
        row = pd.get_dummies(row, columns=bundle["nominal_cols"], drop_first=False, dtype=int)

        # Align to the exact training columns (missing dummies -> 0, extras dropped)
        row = row.reindex(columns=FEATURES, fill_value=0).astype(float)

        # Only Logistic Regression was trained on scaled numeric features -
        # scale here only if the currently selected model needs it.
        if selected_model_name in bundle["needs_scaling"]:
            row[bundle["numeric_cols"]] = bundle["scaler"].transform(row[bundle["numeric_cols"]])

        proba = float(model.predict_proba(row)[0, 1])
        pred = int(model.predict(row)[0])

        # Data-driven risk factors: (label, this-segment's historical churn rate)
        factor_rates = {
            "Month-to-month contract": 0.427,
            "Tenure in the first 12 months": 0.474,
            "Fiber optic internet": 0.419,
            "Electronic check payment": 0.453,
            "No online security and no tech support": 0.490,
        }
        flags = []
        if contract == "Month-to-month":
            flags.append("Month-to-month contract")
        if tenure <= 12:
            flags.append("Tenure in the first 12 months")
        if internet == "Fiber optic":
            flags.append("Fiber optic internet")
        if payment == "Electronic check":
            flags.append("Electronic check payment")
        if not security and not support:
            flags.append("No online security and no tech support")

        peers = df[(df["Contract"] == contract) & (df["InternetService"] == internet)]
        peer_rate = float((peers["Churn"] == "Yes").mean()) if len(peers) else None

        st.session_state["last_result"] = {
            "proba": proba, "pred": pred, "model_name": selected_model_name,
            "flags": flags, "factor_rates": factor_rates,
            "peer_count": len(peers), "peer_rate": peer_rate,
            "contract": contract, "internet": internet,
        }

    result = st.session_state.get("last_result")
    if result:
        with st.container(border=True):
            st.markdown("#### 3. Result")
            st.caption(f"Prediction generated using **{result['model_name']}**")

            gcol, rcol = st.columns([1, 1.3])
            with gcol:
                pct = result["proba"] * 100
                bar_color = "#DC2626" if pct >= 50 else ("#F59E0B" if pct >= 30 else "#16A34A")
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pct,
                    number={"suffix": "%", "font": {"size": 38}},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar": {"color": bar_color},
                        "steps": [
                            {"range": [0, 30], "color": "#DCFCE7"},
                            {"range": [30, 60], "color": "#FEF3C7"},
                            {"range": [60, 100], "color": "#FEE2E2"},
                        ],
                        "threshold": {"line": {"color": "#1F2430", "width": 3},
                                       "thickness": 0.85, "value": pct},
                    },
                ))
                gauge.update_layout(height=230, margin=dict(l=25, r=25, t=15, b=10),
                                     paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})

                if result["pred"] == 1:
                    st.error("**HIGH RISK - flag for retention contact.**")
                else:
                    st.success("**LOW RISK - no retention action required.**")

            with rcol:
                flags = result["flags"]
                if flags:
                    st.markdown("**Risk factors present in this profile** (from the notebook's EDA):")
                    rates = [result["factor_rates"][f] for f in flags]
                    colors = ["#DC2626" if r > BASELINE_CHURN else "#16A34A" for r in rates]
                    bar = go.Figure(go.Bar(
                        x=[r * 100 for r in rates], y=flags, orientation="h",
                        marker_color=colors,
                        text=[f"{r:.1%}" for r in rates], textposition="outside",
                        hovertemplate="%{y}<br>Historical churn rate: %{x:.1f}%<extra></extra>",
                    ))
                    bar.add_vline(x=BASELINE_CHURN * 100, line_dash="dash", line_color="#6B7280",
                                  annotation_text=f"Overall baseline {BASELINE_CHURN:.1%}",
                                  annotation_position="bottom right")
                    bar.update_layout(
                        height=70 + 55 * len(flags), margin=dict(l=10, r=10, t=35, b=30),
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
                        f"{result['internet']}), and {result['peer_rate']:.1%} of them churned.")

# ----------------------------------------------------------------------
# TAB 2: EDA & Model Report (all 20 figures)
# ----------------------------------------------------------------------
with tab_report:
    st.caption("All charts produced during the CRISP-DM analysis (Parts 1-6 of the notebook)")

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
        """Render figures as bordered cards in a responsive grid, skipping/flagging
        any that are missing instead of letting one bad path crash the whole tab."""
        cols = st.columns(columns)
        for i, (filename, caption, note) in enumerate(fig_list):
            path = APP_DIR / filename
            with cols[i % columns]:
                with st.container(border=True):
                    if path.exists():
                        st.image(str(path), width="stretch")
                        st.markdown(f"**{caption}**")
                        st.markdown(
                            f"<details class='insight'><summary>💡 Why this matters</summary>"
                            f"<p>{note}</p></details>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning(f"Missing file: {filename}")

    sub1, sub2, sub3, sub4 = st.tabs([
        "Exploratory Data Analysis",
        "Correlation Analysis",
        "Model Evaluation",
        "Feature Importance",
    ])

    with sub1:
        show_gallery(EDA_FIGS)

    with sub2:
        show_gallery(CORR_FIGS)

    with sub3:
        results_path = APP_DIR / "model_comparison_results.csv"
        if results_path.exists():
            st.markdown("#### Metrics table")
            results_df = pd.read_csv(results_path, index_col=0)

            if "Logistic Regression" in results_df.index:
                results_df = results_df.rename(index={"Logistic Regression": "Logistic Regression (baseline)"})

            st.dataframe(
                results_df.style.background_gradient(cmap="Purples", axis=0).format("{:.4f}"),
                width="stretch",
            )

        fig10_path = APP_DIR / "fig10_model_comparison.png"
        if fig10_path.exists():
            with st.container(border=True):
                st.image(str(fig10_path), width="stretch")
                st.markdown("**Comparison of all four models**")
                st.caption("Random Forest and XGBoost lead on F1 and ROC-AUC; Random Forest is the "
                           "notebook's deployed default for its balance of precision and recall.")
        else:
            st.warning("Missing file: fig10_model_comparison.png")

        show_gallery(MODEL_FIGS)

    with sub4:
        show_gallery(IMPORTANCE_FIGS)
