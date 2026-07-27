"""
BMDS2003 Data Science - Deployment Prototype
Telco Customer Churn Risk Scorer (TAB VERSION)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Telco Churn Risk Scorer", layout="wide")
st.title("Telco Customer Churn Risk Scorer")
st.caption("BMDS2003 Data Science - Group Assignment Deployment Prototype")

APP_DIR = Path(__file__).resolve().parent

# -----------------------------
# Load model bundle + data
# -----------------------------
@st.cache_resource
def load_bundle():
    with open("trained_models.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("telco_churn_cleaned.csv")

bundle = load_bundle()
df = load_data()

BEST_MODEL_NAME = bundle["best_model_name"]
AVAILABLE_MODELS = list(bundle["models"].keys())
FEATURES = bundle["feature_names"]

# -----------------------------
# MAIN TABS
# -----------------------------
tab_scorer, tab_report = st.tabs(["Risk Scorer", "EDA & Model Report"])

# =========================================================
# TAB 1: RISK SCORER
# =========================================================
with tab_scorer:

    st.subheader("1. Choose a model")

    # 👉 平行 model tabs
    model_tabs = st.tabs(AVAILABLE_MODELS)

    # =====================================================
    # 每个 model 一个 tab
    # =====================================================
    for i, model_name in enumerate(AVAILABLE_MODELS):

        with model_tabs[i]:

            model = bundle["models"][model_name]
            metrics = bundle["results"].loc[model_name]

            st.markdown(f"## {model_name}")

            if model_name == BEST_MODEL_NAME:
                st.success("⭐ Recommended (Best Model)")

            st.info(
                f"Test F1 = {metrics['F1']:.3f} | "
                f"Recall = {metrics['Recall']:.3f} | "
                f"ROC-AUC = {metrics['ROC-AUC']:.3f}"
            )

            # -----------------------------
            # CUSTOMER INPUT
            # -----------------------------
            st.subheader("2. Customer profile")

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

            st.subheader("3. Services")

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
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": yn(phone),
                "MultipleLines": yn(multiple),
                "InternetService": internet,
                "OnlineSecurity": yn(security),
                "OnlineBackup": yn(backup),
                "DeviceProtection": yn(protection),
                "TechSupport": yn(support),
                "StreamingTV": yn(tv),
                "StreamingMovies": yn(movies),
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }

            st.caption(f"Estimated total charges: RM {total:,.2f}")

            # -----------------------------
            # PREDICT BUTTON
            # -----------------------------
            if st.button(f"Score using {model_name}", use_container_width=True):

                row = pd.DataFrame([raw])

                # Feature engineering
                row["tenure_group"] = pd.cut(
                    row["tenure"],
                    bins=[-1, 12, 24, 48, 72],
                    labels=["0-12m", "13-24m", "25-48m", "49-72m"]
                )

                row["num_services"] = (
                    row[bundle["service_cols"]] == "Yes"
                ).sum(axis=1)

                # Binary encode
                for c in bundle["binary_cols"]:
                    row[c] = row[c].map({
                        "No": 0, "Yes": 1,
                        "Female": 0, "Male": 1
                    })

                # One-hot
                row = pd.get_dummies(
                    row,
                    columns=bundle["nominal_cols"],
                    drop_first=False,
                    dtype=int
                )

                # Align features
                row = row.reindex(columns=FEATURES, fill_value=0).astype(float)

                # Scaling (only for LR)
                if model_name in bundle["needs_scaling"]:
                    row[bundle["numeric_cols"]] = bundle["scaler"].transform(
                        row[bundle["numeric_cols"]]
                    )

                # Prediction
                proba = float(model.predict_proba(row)[0, 1])
                pred = int(model.predict(row)[0])

                # -----------------------------
                # RESULT
                # -----------------------------
                st.subheader("4. Result")

                st.metric("Churn probability", f"{proba:.1%}")
                st.progress(min(proba, 1.0))

                if pred == 1:
                    st.error("HIGH RISK")
                else:
                    st.success("LOW RISK")

    st.divider()

# =========================================================
# TAB 2: REPORT
# =========================================================
with tab_report:
    st.caption("EDA & Model Report (same as before)")
    st.write("👉 这里保留你原本的20张图代码")
