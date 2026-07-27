"""
BMDS2003 Data Science - Deployment Prototype
Telco Customer Churn Risk Scorer
Run locally:  streamlit run streamlit_app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
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

# The pickle bundle stores every trained model, not just the winner, so the
# user can pick any of them at runtime instead of always using Random Forest.
BEST_MODEL_NAME = bundle["best_model_name"]
AVAILABLE_MODELS = list(bundle["models"].keys())
FEATURES = bundle["feature_names"]

tab_scorer, tab_report = st.tabs(["Risk Scorer", "EDA & Model Report"])

# ----------------------------------------------------------------------
# TAB 1: Risk Scorer
# ----------------------------------------------------------------------
with tab_scorer:
    st.subheader("1. Customer profile")
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

    st.subheader("2. Services")
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

    submitted = st.button("Score this customer", type="primary", use_container_width=True)

    # 预处理数据和风险分析（独立于模型，提交后展示一次）
    if submitted:
        # Rebuild the engineered features exactly as in Part 2 of the notebook
        row_base = pd.DataFrame([raw])
        row_base["tenure_group"] = pd.cut(row_base["tenure"], bins=[-1, 12, 24, 48, 72],
                                     labels=["0-12m", "13-24m", "25-48m", "49-72m"])
        row_base["num_services"] = (row_base[bundle["service_cols"]] == "Yes").sum(axis=1)

        for c in bundle["binary_cols"]:
            row_base[c] = row_base[c].map({"No": 0, "Yes": 1, "Female": 0, "Male": 1})
        row_base = pd.get_dummies(row_base, columns=bundle["nominal_cols"], drop_first=False, dtype=int)

        # Align to the exact training columns (missing dummies -> 0, extras dropped)
        row_base = row_base.reindex(columns=FEATURES, fill_value=0).astype(float)

        st.subheader("3. Risk Analysis")
        st.markdown("**Risk factors present in this profile** (from the notebook's EDA):")
        flags = []
        if contract == "Month-to-month":
            flags.append("Month-to-month contract - churn rate ~42.7% vs ~2.8% on two-year contracts")
        if tenure <= 12:
            flags.append(f"Tenure only {tenure} months - the first year carries ~47.4% churn")
        if internet == "Fiber optic":
            flags.append("Fiber optic internet - ~41.9% churn despite being the premium product")
        if payment == "Electronic check":
            flags.append("Electronic check payment - ~45.3% churn, roughly triple automatic methods")
        if not security and not support:
            flags.append("No online security and no tech support - both are associated with retention")
        
        if flags:
            for fl in flags:
                st.write(f"- {fl}")
        else:
            st.write("- No major risk factor from the top drivers identified in the EDA.")

        peers = df[(df["Contract"] == contract) & (df["InternetService"] == internet)]
        if len(peers) > 0:
            actual = (peers["Churn"] == "Yes").mean()
            st.caption(f"Historical benchmark: {len(peers):,} customers in the data share this "
                       f"contract + internet combination, and {actual:.1%} of them churned.")

    # 选项卡：展示所有可用的模型 (取代了原本的 st.selectbox)
    st.subheader("4. Model Predictions")
    st.caption("Select a model tab below to view its specific prediction and performance metrics.")
    model_tabs = st.tabs(AVAILABLE_MODELS)

    # 遍历所有模型并为其生成独立的选项卡内容
    for idx, m_name in enumerate(AVAILABLE_MODELS):
        with model_tabs[idx]:
            model = bundle["models"][m_name]
            metrics = bundle["results"].loc[m_name]

            is_best = m_name == BEST_MODEL_NAME
            badge = " (notebook's best model)" if is_best else ""
            st.info(f"**{m_name}**{badge}  |  Test F1 = {metrics['F1']:.3f}  |  "
                    f"Recall = {metrics['Recall']:.3f}  |  ROC-AUC = {metrics['ROC-AUC']:.3f}")

            # 只有在点击提交按钮后，才会渲染预测结果
            if submitted:
                row_model = row_base.copy()
                
                # Only Logistic Regression was trained on scaled numeric features -
                # scale here only if the currently selected model needs it.
                if m_name in bundle["needs_scaling"]:
                    row_model[bundle["numeric_cols"]] = bundle["scaler"].transform(row_model[bundle["numeric_cols"]])

                proba = float(model.predict_proba(row_model)[0, 1])
                pred = int(model.predict(row_model)[0])

                st.metric("Churn probability", f"{proba:.1%}")
                st.progress(min(proba, 1.0))

                if pred == 1:
                    st.error("**HIGH RISK - flag for retention contact.**")
                else:
                    st.success("**LOW RISK - no retention action required.**")

    st.divider()
    st.caption(
        "Academic prototype trained on 7,043 historical customer records. It outputs a churn "
        "RISK SCORE for retention triage, not a prediction about an individual's intentions. "
        "The associations shown are correlational, not causal, and the model must be retrained "
        "as customer behaviour and tariffs change. Not for real commercial or financial decisions.")

# ----------------------------------------------------------------------
# TAB 2: EDA & Model Report (all 20 figures)
# ----------------------------------------------------------------------
with tab_report:
    st.caption("All charts produced during the CRISP-DM analysis (Parts 1-6 of the notebook)")

    # Each entry is (filename, image caption, presentation note).
    # The presentation note is the short "talking point" line shown below the
    # image so it's easier to explain the figure out loud during presentation.
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
        """Render figures in a responsive grid, skipping/flagging any that are missing
        instead of letting one bad path crash the whole tab. Each image gets a small
        note underneath (a short talking point) to make presenting the report easier."""
        cols = st.columns(columns)
        for i, (filename, caption, note) in enumerate(fig_list):
            path = APP_DIR / filename
            with cols[i % columns]:
                if path.exists():
                    st.image(str(path), caption=caption, use_container_width=True)
                    st.caption(note)
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
            st.subheader("Metrics table")
            st.dataframe(pd.read_csv(results_path, index_col=0), use_container_width=True)

        # fig10 (overall model comparison) is shown full-width, right below the
        # metrics table, so it matches the table's width instead of being squeezed
        # into the 2-column grid with the rest of the model evaluation figures.
        fig10_path = APP_DIR / "fig10_model_comparison.png"
        if fig10_path.exists():
            st.image(str(fig10_path), caption="Comparison of all four models", use_container_width=True)
            st.caption("Random Forest and XGBoost lead on F1 and ROC-AUC; Random Forest is the "
                       "notebook's deployed default for its balance of precision and recall.")
        else:
            st.warning("Missing file: fig10_model_comparison.png")

        show_gallery(MODEL_FIGS)

    with sub4:
        show_gallery(IMPORTANCE_FIGS)
