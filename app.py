"""
BMDS2003 Data Science - Deployment Prototype
Telco Customer Churn Risk Scorer
Run locally:  streamlit run streamlit_app.py
"""
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Churn Risk Scorer", page_icon="signal", layout="centered")
st.title("Telco Customer Churn Risk Scorer")
st.caption("BMDS2003 Data Science - Group Assignment Deployment Prototype")


@st.cache_resource
def load_bundle():
    with open("trained_models.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    return pd.read_csv("telco_churn_cleaned.csv")


bundle = load_bundle()
df = load_data()

MODEL_NAME = bundle["best_model_name"]
model = bundle["models"][MODEL_NAME]
FEATURES = bundle["feature_names"]
metrics = bundle["results"].loc[MODEL_NAME]

st.info(f"Model in use: **{MODEL_NAME}**  |  Test F1 = {metrics['F1']:.3f}  |  "
        f"Recall = {metrics['Recall']:.3f}  |  ROC-AUC = {metrics['ROC-AUC']:.3f}")

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

if st.button("Score this customer", type="primary", use_container_width=True):
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

    if MODEL_NAME in bundle["needs_scaling"]:
        row[bundle["numeric_cols"]] = bundle["scaler"].transform(row[bundle["numeric_cols"]])

    proba = float(model.predict_proba(row)[0, 1])
    pred = int(model.predict(row)[0])

    st.subheader("3. Result")
    st.metric("Churn probability", f"{proba:.1%}")
    st.progress(min(proba, 1.0))

    if pred == 1:
        st.error("**HIGH RISK - flag for retention contact.**")
    else:
        st.success("**LOW RISK - no retention action required.**")

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

st.divider()
st.caption(
    "Academic prototype trained on 7,043 historical customer records. It outputs a churn "
    "RISK SCORE for retention triage, not a prediction about an individual's intentions. "
    "The associations shown are correlational, not causal, and the model must be retrained "
    "as customer behaviour and tariffs change. Not for real commercial or financial decisions.")
