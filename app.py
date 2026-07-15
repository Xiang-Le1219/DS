import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Dashboard")

# -----------------------------
# Load Model Bundle
# (bundle = model + scaler + label encoders + exact feature order,
#  all saved together by train_model.py so they can never drift apart)
# -----------------------------
@st.cache_resource
def load_bundle():
    with open("model_bundle.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
label_encoders = bundle["label_encoders"]
FEATURE_ORDER = bundle["feature_order"]
CATEGORICAL_COLS = bundle["categorical_cols"]
CATEGORY_OPTIONS = bundle["category_options"]

# -----------------------------
# Sidebar Inputs
# All 19 fields the model was actually trained on — nothing skipped.
# -----------------------------
st.sidebar.header("🧾 Customer Input")

gender = st.sidebar.selectbox("Gender", CATEGORY_OPTIONS["gender"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", CATEGORY_OPTIONS["Partner"])
dependents = st.sidebar.selectbox("Dependents", CATEGORY_OPTIONS["Dependents"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

phone = st.sidebar.selectbox("Phone Service", CATEGORY_OPTIONS["PhoneService"])
if phone == "No":
    multiple = "No phone service"
    st.sidebar.selectbox("Multiple Lines", ["No phone service"], disabled=True)
else:
    multiple = st.sidebar.selectbox(
        "Multiple Lines", [o for o in CATEGORY_OPTIONS["MultipleLines"] if o != "No phone service"]
    )

internet = st.sidebar.selectbox("Internet Service", CATEGORY_OPTIONS["InternetService"])

def internet_dependent_select(label, key):
    if internet == "No":
        st.sidebar.selectbox(label, ["No internet service"], disabled=True, key=key)
        return "No internet service"
    return st.sidebar.selectbox(label, ["No", "Yes"], key=key)

online_security = internet_dependent_select("Online Security", "os")
online_backup = internet_dependent_select("Online Backup", "ob")
device_protection = internet_dependent_select("Device Protection", "dp")
tech_support = internet_dependent_select("Tech Support", "ts")
streaming_tv = internet_dependent_select("Streaming TV", "stv")
streaming_movies = internet_dependent_select("Streaming Movies", "smv")

contract = st.sidebar.selectbox("Contract", CATEGORY_OPTIONS["Contract"])
paperless = st.sidebar.selectbox("Paperless Billing", CATEGORY_OPTIONS["PaperlessBilling"])
payment = st.sidebar.selectbox("Payment Method", CATEGORY_OPTIONS["PaymentMethod"])

monthly = st.sidebar.number_input("Monthly Charges", value=50.0, min_value=0.0)
total = st.sidebar.number_input("Total Charges", value=1000.0, min_value=0.0)

predict_btn = st.sidebar.button("🚀 Predict")

# -----------------------------
# Preprocessing — mirrors training EXACTLY:
# label-encode each categorical column with its fitted encoder, then scale
# with the fitted StandardScaler, in the exact column order used in training.
# -----------------------------
def build_raw_row():
    return {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

def preprocess(raw_row: dict) -> np.ndarray:
    df = pd.DataFrame([raw_row])[FEATURE_ORDER]  # enforce exact training column order

    for col in CATEGORICAL_COLS:
        le = label_encoders[col]
        # Guard against a category never seen in training (shouldn't happen since
        # dropdowns are built from the same source data, but fail safely if it does)
        val = df.at[0, col]
        if val not in le.classes_:
            raise ValueError(f"'{val}' is not a recognised value for {col}")
        df[col] = le.transform(df[col])

    scaled = scaler.transform(df)
    return scaled  # shape (1, 19), matches model.n_features_in_

# -----------------------------
# Dashboard Layout
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Explainability", "📈 What-if"])

# -----------------------------
# Run Prediction
# -----------------------------
if predict_btn:
    raw_row = build_raw_row()
    input_scaled = preprocess(raw_row)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    churn_prob = proba[1]

    # =========================
    # TAB 1: Overview
    # =========================
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            if pred == 1:
                st.metric("Churn Risk", "High ⚠️", f"{churn_prob:.1%}")
            else:
                st.metric("Churn Risk", "Low ✅", f"{churn_prob:.1%}")

        with col2:
            st.metric("Confidence", f"{max(proba):.1%}")

        st.subheader("📊 Feature Importance")
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            df_imp = pd.DataFrame({"Feature": FEATURE_ORDER, "Importance": imp})
            df_imp = df_imp.sort_values("Importance", ascending=True)

            fig, ax = plt.subplots()
            ax.barh(df_imp["Feature"], df_imp["Importance"])
            ax.set_title("Model Feature Importance (overall, from training)")
            st.pyplot(fig)

    # =========================
    # TAB 2: Explainability
    # =========================
    with tab2:
        st.subheader("🔍 Why this prediction?")
        st.caption(
            "For each feature, we set it to this customer's mean-encoded '0' baseline "
            "and see how much the churn probability shifts."
        )

        base_prob = churn_prob
        impact = {}

        for i, col in enumerate(FEATURE_ORDER):
            temp = input_scaled.copy()
            temp[0, i] = 0  # zero out in *scaled* space (≈ dataset average for that feature)
            new_prob = model.predict_proba(temp)[0][1]
            impact[col] = base_prob - new_prob

        df_imp = pd.DataFrame.from_dict(impact, orient="index", columns=["Impact"])
        df_imp = df_imp.sort_values(by="Impact", ascending=True)

        fig, ax = plt.subplots()
        ax.barh(df_imp.index, df_imp["Impact"])
        ax.set_title("Feature Impact on This Prediction")
        st.pyplot(fig)

        st.subheader("🧠 Model Explanation")
        top_features = df_imp.reindex(df_imp["Impact"].abs().sort_values(ascending=False).index).head(3)

        explanation = "This customer's churn prediction is most influenced by:\n"
        for feature, row in top_features.iterrows():
            direction = "increases" if row["Impact"] > 0 else "decreases"
            explanation += f"- **{feature}** {direction} churn risk\n"

        st.info(explanation)

    # =========================
    # TAB 3: What-if Analysis
    # =========================
    with tab3:
        st.subheader("📈 Feature Sensitivity")

        feature = st.selectbox("Choose Feature", ["tenure", "MonthlyCharges"])

        if feature == "tenure":
            values = np.arange(0, 73)
        else:
            values = np.linspace(0, 150, 50)

        probs = []
        for v in values:
            temp_row = dict(raw_row)
            temp_row[feature] = v
            temp_scaled = preprocess(temp_row)
            probs.append(model.predict_proba(temp_scaled)[0][1])

        fig, ax = plt.subplots()
        ax.plot(values, probs)
        ax.set_xlabel(feature)
        ax.set_ylabel("Churn Probability")
        ax.set_title("What-if Analysis")
        st.pyplot(fig)
else:
    st.info("👈 Fill in the customer details in the sidebar and click **Predict** to see results.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("🚀 Stable Dashboard • Full 19-feature preprocessing • Ready for Presentation")
