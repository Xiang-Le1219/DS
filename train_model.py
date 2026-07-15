"""
Train the churn model and save a COMPLETE bundle: encoders + scaler + model +
feature order. This replaces the old approach of pickling only the bare
RandomForestClassifier, which is what caused the Streamlit app's silent
failures (the app had no way to know how raw inputs were encoded/scaled).

Run this in the SAME environment (same scikit-learn version) that will run
the Streamlit app, to avoid the version-mismatch segfault.
"""
import pandas as pd
import numpy as np
import pickle
import sklearn

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

print(f"Training with scikit-learn {sklearn.__version__}")

# -----------------------------
# Load & clean (same as notebook)
# -----------------------------
df = pd.read_csv("Telco_Cusomer_Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df.drop("customerID", axis=1, inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# -----------------------------
# Label-encode every categorical column, and KEEP the encoders
# -----------------------------
categorical_cols = df.select_dtypes(include="object").columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# Split X/y, scale, KEEP the scaler and the exact column order
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]
feature_order = X.columns.tolist()  # <-- this is the ground truth order, 19 cols

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train the model (Random Forest, matching the notebook's chosen "best_model")
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------
# Build lookup of raw category options for each categorical column,
# so the Streamlit UI can build dropdowns that are guaranteed valid.
# -----------------------------
raw_df = pd.read_csv("Telco_Cusomer_Churn.csv")
category_options = {
    col: sorted(raw_df[col].dropna().unique().tolist()) for col in categorical_cols
}

# -----------------------------
# Save EVERYTHING needed to reproduce preprocessing at inference time
# -----------------------------
bundle = {
    "model": rf,
    "scaler": scaler,
    "label_encoders": label_encoders,
    "feature_order": feature_order,       # exact column order model was trained on
    "categorical_cols": categorical_cols, # which of those columns need label-encoding
    "category_options": category_options, # valid raw string options per categorical col
    "sklearn_version": sklearn.__version__,
}

with open("model_bundle.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\nSaved model_bundle.pkl")
print("Feature order:", feature_order)
