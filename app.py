import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Credit Risk App", layout="wide")

# ======================
# Sidebar Navigation
# ======================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Risk Scorer", "EDA & Model Report", "Model Comparison"]
)

# ======================
# 1. Risk Scorer
# ======================
if page == "Risk Scorer":

    st.title("📊 Credit Risk Scorer")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="risk_tenure")
        monthly_charges = st.slider("Monthly Charges", 0, 200, 70, key="risk_charge")

    with col2:
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0, key="risk_total")

    if st.button("Predict Risk"):
        risk_score = np.random.rand()
        st.success(f"Predicted Risk Score: {risk_score:.2f}")

# ======================
# 2. EDA & Model Report
# ======================
elif page == "EDA & Model Report":

    st.title("📈 EDA & Model Report")

    st.write("### Dataset Preview")
    df = pd.DataFrame({
        "Tenure": np.random.randint(1, 72, 100),
        "MonthlyCharges": np.random.randint(20, 150, 100),
    })

    st.dataframe(df)

    st.write("### Summary Statistics")
    st.write(df.describe())

    st.write("### Model Performance (Example)")
    st.write({
        "Logistic Regression": "AUC = 0.78",
        "Random Forest": "AUC = 0.85",
        "XGBoost": "AUC = 0.88"
    })

# ======================
# 3. Model Comparison ⭐（重点改造）
# ======================
elif page == "Model Comparison":

    st.title("🤖 Model Comparison")

    st.write("Compare multiple models side by side")

    # ===== 输入区域 =====
    st.subheader("Input Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="cmp_tenure")

    with col2:
        monthly_charges = st.slider("Monthly Charges", 0, 200, 70, key="cmp_charge")

    with col3:
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0, key="cmp_total")

    # ===== 模型选择（多选）=====
    st.subheader("Select Models")

    models = st.multiselect(
        "Choose models to compare",
        ["Logistic Regression", "Random Forest", "XGBoost"],
        default=["Logistic Regression", "Random Forest"]
    )

    # ===== 预测 =====
    if st.button("Run Comparison"):

        results = []

        for model in models:
            score = np.random.rand()
            results.append({
                "Model": model,
                "Risk Score": round(score, 3)
            })

        result_df = pd.DataFrame(results)

        st.subheader("Results")
        st.dataframe(result_df)

        st.subheader("Best Model")

        best_model = result_df.sort_values("Risk Score").iloc[0]
        st.success(f"Best Model: {best_model['Model']} (Score: {best_model['Risk Score']})")
