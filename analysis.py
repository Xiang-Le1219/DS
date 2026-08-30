"""Data loading and model analysis for the Telcom churn dashboard.

Everything the interactive charts need is computed here, live from the CSVs and
the trained models, so the dashboard no longer depends on the notebook's static
PNG exports.

The train/test split reproduces the notebook's split exactly
(test_size=0.2, random_state=42, stratify=y). This was verified by recomputing
all four models' test metrics and matching model_comparison_results.csv to four
decimal places, so every number shown in the app agrees with the report.

All expensive work is wrapped in Streamlit's cache: the whole module costs
about ten seconds on first load and is instant afterwards.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split

APP_DIR = Path(__file__).resolve().parent

RANDOM_STATE = 42          # matches RANDOM_STATE in the notebook
TEST_SIZE = 0.2
CV_FOLDS = 5

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "num_services"]
CATEGORICAL_FEATURES = [
    "Contract", "InternetService", "PaymentMethod", "tenure_group", "gender",
    "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaperlessBilling",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    with open(APP_DIR / "trained_models.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_cleaned():
    return pd.read_csv(APP_DIR / "telco_churn_cleaned.csv")


@st.cache_data
def load_model_ready():
    return pd.read_csv(APP_DIR / "telco_churn_model_ready.csv")


@st.cache_data
def load_published_metrics():
    """The notebook's own results table, used to cross-check the live numbers."""
    path = APP_DIR / "model_comparison_results.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


@st.cache_data
def get_split():
    """The notebook's exact train/test split, rebuilt from the encoded matrix."""
    bundle = load_bundle()
    frame = load_model_ready()
    X = frame[bundle["feature_names"]].astype(float)
    y = frame["Churn"].astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


def scaled(X, model_name):
    """Logistic Regression is the only model trained on scaled features."""
    bundle = load_bundle()
    if model_name not in bundle["needs_scaling"]:
        return X
    X = X.copy()
    X[bundle["numeric_cols"]] = bundle["scaler"].transform(X[bundle["numeric_cols"]])
    return X


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------
@st.cache_data
def test_predictions():
    """Predicted churn probabilities on the held-out test set, per model."""
    bundle = load_bundle()
    _, X_test, _, y_test = get_split()
    probas = {name: model.predict_proba(scaled(X_test, name))[:, 1]
              for name, model in bundle["models"].items()}
    return probas, y_test.to_numpy()


@st.cache_data
def roc_points():
    """False/true positive rates and AUC for every model."""
    probas, y_test = test_predictions()
    out = {}
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        out[name] = (fpr, tpr, roc_auc_score(y_test, proba))
    return out


@st.cache_data
def live_metrics(threshold=0.5):
    """Headline metrics for all models at a given decision threshold."""
    probas, y_test = test_predictions()
    rows = []
    for name, proba in probas.items():
        pred = (proba >= threshold).astype(int)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba),
        })
    return pd.DataFrame(rows).set_index("Model")


@st.cache_data
def metrics_at_threshold(model_name, threshold):
    """Confusion matrix and metrics for one model at any cut-off.

    The published metrics all assume the default 0.50 threshold; exposing the
    threshold makes the precision/recall trade-off visible instead of implicit.
    """
    probas, y_test = test_predictions()
    pred = (probas[model_name] >= threshold).astype(int)
    return {
        "matrix": confusion_matrix(y_test, pred),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }


@st.cache_data
def threshold_sweep(model_name):
    """Precision, recall and F1 across thresholds from 0.05 to 0.95.

    Above a model's highest predicted probability it flags nobody as a churn
    risk, so precision becomes 0/0 - undefined, not zero. Those points are left
    as NaN so the precision line simply stops rather than plunging to zero and
    implying the model suddenly got everything wrong. Recall and F1 are
    genuinely 0 there (no churner is caught), so they are plotted as 0.
    """
    probas, y_test = test_predictions()
    proba = probas[model_name]
    rows = []
    for cut in np.arange(0.05, 0.96, 0.01):
        pred = (proba >= cut).astype(int)
        flagged = int(pred.sum())
        rows.append({
            "Threshold": cut,
            "Precision": (precision_score(y_test, pred, zero_division=0)
                          if flagged else np.nan),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "Flagged": flagged,
        })
    return pd.DataFrame(rows)


@st.cache_data
def max_probability(model_name):
    """The model's highest predicted probability on the test set - the point
    beyond which it stops flagging anyone at all."""
    probas, _ = test_predictions()
    return float(probas[model_name].max())


@st.cache_data
def cv_results(metric="f1"):
    """Per-fold cross-validation scores - the evidence that the model ranking is
    stable rather than an artefact of one lucky split.

    Cached per metric so the first load only pays for the metric actually shown
    (about 8s) instead of refitting for all of them up front.
    """
    bundle = load_bundle()
    X_train, _, y_train, _ = get_split()
    # cv=CV_FOLDS (an int, so StratifiedKFold WITHOUT shuffling) is what the
    # notebook used. Passing a shuffled StratifiedKFold instead gives visibly
    # different fold scores - with this setting all four models reproduce
    # Figure 13's means and standard deviations to three decimal places.
    rows = []
    for name, model in bundle["models"].items():
        scores = cross_val_score(model, scaled(X_train, name), y_train,
                                 cv=CV_FOLDS, scoring=metric, n_jobs=-1)
        for fold, score in enumerate(scores, start=1):
            rows.append({"Model": name, "Fold": fold, "Score": score})
    return pd.DataFrame(rows)


@st.cache_data
def train_vs_test():
    """Train and test ROC-AUC side by side, to expose any overfitting gap."""
    bundle = load_bundle()
    X_train, X_test, y_train, y_test = get_split()
    rows = []
    for name, model in bundle["models"].items():
        train_proba = model.predict_proba(scaled(X_train, name))[:, 1]
        test_proba = model.predict_proba(scaled(X_test, name))[:, 1]
        rows.append({"Model": name,
                     "Train": roc_auc_score(y_train, train_proba),
                     "Test": roc_auc_score(y_test, test_proba)})
    frame = pd.DataFrame(rows)
    frame["Gap"] = frame["Train"] - frame["Test"]
    return frame


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
@st.cache_data
def builtin_importance(model_name):
    """Model-native importance - Gini for the trees, coefficients for LR.

    Logistic Regression coefficients are kept SIGNED: a negative coefficient
    means the feature lowers churn risk (long tenure, two-year contracts), and
    throwing the sign away with abs() would discard the direction, which is half
    of what the coefficient tells you. Ranking is by magnitude either way, so
    the ordering is unchanged - this only preserves the direction.
    """
    bundle = load_bundle()
    model = bundle["models"][model_name]
    if hasattr(model, "feature_importances_"):
        values, label = model.feature_importances_, "Gini importance"
    else:
        values, label = model.coef_[0], "Coefficient"
    frame = pd.DataFrame({"Feature": bundle["feature_names"], "Importance": values})
    return (frame.reindex(frame["Importance"].abs().sort_values(ascending=False).index),
            label)


@st.cache_data
def permutation_scores(model_name, n_repeats=5):
    """Permutation importance on the test set: the ROC-AUC lost when each
    feature is shuffled. Slower than the built-in kind, but model-agnostic and
    measured on data the model never saw."""
    bundle = load_bundle()
    _, X_test, _, y_test = get_split()
    result = permutation_importance(
        bundle["models"][model_name], scaled(X_test, model_name), y_test,
        n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1,
        scoring="roc_auc",
    )
    return pd.DataFrame({
        "Feature": bundle["feature_names"],
        "Importance": result.importances_mean,
        "Std": result.importances_std,
    }).sort_values("Importance", ascending=False)


# ---------------------------------------------------------------------------
# Exploratory statistics
# ---------------------------------------------------------------------------
@st.cache_data
def cramers_v_scores():
    """Cramer's V for every categorical feature against Churn."""
    df = load_cleaned()
    rows = []
    for col in CATEGORICAL_FEATURES:
        table = pd.crosstab(df[col], df["Churn"])
        chi2 = chi2_contingency(table)[0]
        n = table.to_numpy().sum()
        min_dim = min(table.shape) - 1
        rows.append({"Feature": col,
                     "CramersV": float(np.sqrt(chi2 / (n * min_dim))) if min_dim else 0.0})
    return pd.DataFrame(rows).sort_values("CramersV", ascending=False)


@st.cache_data
def churn_correlations():
    """Correlation of every encoded feature with Churn, most positive first."""
    frame = load_model_ready()
    corr = frame.corr(numeric_only=True)["Churn"].drop("Churn")
    return corr.sort_values(ascending=False)


@st.cache_data
def correlation_matrix(top_n=15):
    """Correlation matrix of the features most associated with Churn."""
    frame = load_model_ready()
    ranked = churn_correlations().abs().sort_values(ascending=False)
    keep = list(ranked.head(top_n).index) + ["Churn"]
    return frame[keep].corr(numeric_only=True)


@st.cache_data
def churn_rate_by(column):
    """Churn rate and customer count for each level of a categorical column."""
    df = load_cleaned()
    grouped = df.groupby(column, observed=True)["Churn"]
    out = pd.DataFrame({
        "Customers": grouped.size(),
        "ChurnRate": grouped.apply(lambda s: (s == "Yes").mean()),
    }).reset_index()
    return out.sort_values("ChurnRate", ascending=False)


@st.cache_data
def interaction_rates(row_feature, col_feature):
    """Churn rate for every combination of two categorical features, plus the
    segment sizes - risk compounds, and this is where that shows up."""
    df = load_cleaned()
    rate = df.pivot_table(index=row_feature, columns=col_feature, values="Churn",
                          aggfunc=lambda s: (s == "Yes").mean(), observed=True)
    count = df.pivot_table(index=row_feature, columns=col_feature, values="Churn",
                           aggfunc="size", observed=True)
    return rate, count


@st.cache_data
def services_vs_churn():
    """Churn rate against the number of subscribed services."""
    df = load_cleaned()
    grouped = df.groupby("num_services", observed=True)["Churn"]
    return pd.DataFrame({
        "Customers": grouped.size(),
        "ChurnRate": grouped.apply(lambda s: (s == "Yes").mean()),
    }).reset_index()


@st.cache_data
def smote_balance():
    """Training-set class balance before and after SMOTE.

    SMOTE is actually run here rather than assumed, so the counts are evidence
    rather than a restatement of the notebook. It is applied to the training
    split only - never the test set - which is what keeps the evaluation honest.
    """
    from imblearn.over_sampling import SMOTE
    X_train, _, y_train, _ = get_split()
    before = y_train.value_counts().sort_index()
    _, y_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    after = pd.Series(y_resampled).value_counts().sort_index()
    return pd.DataFrame({
        "Class": ["No churn (0)", "Churn (1)"],
        "Before": [int(before.get(0, 0)), int(before.get(1, 0))],
        "After": [int(after.get(0, 0)), int(after.get(1, 0))],
    })


@st.cache_data
def scaling_check(column="tenure"):
    """Raw vs standardised values for one numeric column, to show that the
    StandardScaler behaved as specified (mean 0, standard deviation 1)."""
    bundle = load_bundle()
    X_train, _, _, _ = get_split()
    raw = X_train[column].to_numpy()
    scaled_all = bundle["scaler"].transform(X_train[bundle["numeric_cols"]])
    idx = bundle["numeric_cols"].index(column)
    return raw, scaled_all[:, idx]


@st.cache_data
def mosaic_data(column):
    """Segment sizes and churn split per category, for a true mosaic plot where
    column width encodes customer volume."""
    df = load_cleaned()
    grouped = df.groupby(column, observed=True)["Churn"]
    frame = pd.DataFrame({
        "Customers": grouped.size(),
        "ChurnRate": grouped.apply(lambda s: (s == "Yes").mean()),
    }).reset_index()
    return frame.sort_values("Customers", ascending=False)


@st.cache_data
def all_importances():
    """Top features for every model at once, mirroring Figure 15 in the report."""
    bundle = load_bundle()
    out = {}
    for name in bundle["models"]:
        out[name] = builtin_importance(name)
    return out


@st.cache_data
def data_quality():
    """The validation checks behind the notebook's data-preparation figure."""
    raw_path = APP_DIR / "Telco_Cusomer_Churn.csv"
    clean = load_cleaned()
    raw = pd.read_csv(raw_path) if raw_path.exists() else None
    return {
        "raw_rows": len(raw) if raw is not None else None,
        "clean_rows": len(clean),
        "columns": clean.shape[1],
        "missing": int(clean.isna().sum().sum()),
        "duplicates": int(clean.duplicated().sum()),
        "churn_rate": float((clean["Churn"] == "Yes").mean()),
    }


# ---------------------------------------------------------------------------
# What-if analysis
# ---------------------------------------------------------------------------
def encode_profiles(raws, bundle):
    """Turn a list of raw customer dicts into the model's feature matrix,
    rebuilding the engineered features exactly as Part 2 of the notebook does.

    Encoding the whole batch in one pass (rather than one row at a time) is what
    keeps the what-if curve interactive - it turns 37 encode+predict round trips
    into a single one.
    """
    rows = pd.DataFrame(list(raws))
    rows["tenure_group"] = pd.cut(rows["tenure"], bins=[-1, 12, 24, 48, 72],
                                  labels=["0-12m", "13-24m", "25-48m", "49-72m"])
    rows["num_services"] = (rows[bundle["service_cols"]] == "Yes").sum(axis=1)
    for col in bundle["binary_cols"]:
        rows[col] = rows[col].map({"No": 0, "Yes": 1, "Female": 0, "Male": 1})
    rows = pd.get_dummies(rows, columns=bundle["nominal_cols"], drop_first=False,
                          dtype=int)
    return rows.reindex(columns=bundle["feature_names"], fill_value=0).astype(float)


def encode_profile(raw, bundle):
    """Feature vector for a single customer."""
    return encode_profiles([raw], bundle)


def sensitivity_curve(raw, model_name, feature, values):
    """Re-score one customer across a range of values for a single feature.

    This is the what-if view: it holds the rest of the profile fixed and shows
    how the predicted risk responds, turning a one-off score into something the
    reader can actually explore.
    """
    bundle = load_bundle()
    variants = []
    for value in values:
        variant = dict(raw)
        variant[feature] = value
        if feature == "tenure":
            variant["TotalCharges"] = round(value * variant["MonthlyCharges"], 2)
        elif feature == "MonthlyCharges":
            variant["TotalCharges"] = round(variant["tenure"] * value, 2)
        variants.append(variant)
    encoded = scaled(encode_profiles(variants, bundle), model_name)
    proba = bundle["models"][model_name].predict_proba(encoded)[:, 1]
    return pd.DataFrame({feature: values, "Risk": proba})


# ---------------------------------------------------------------------------
# Per-customer explanation
# ---------------------------------------------------------------------------
# Fields a retention team could realistically influence. Demographics and
# tenure are excluded - you cannot offer a customer a different age.
ACTIONABLE_FIELDS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)",
                      "Electronic check", "Mailed check"],
    "OnlineSecurity": ["No", "Yes"],
    "TechSupport": ["No", "Yes"],
    "OnlineBackup": ["No", "Yes"],
    "DeviceProtection": ["No", "Yes"],
    "PaperlessBilling": ["No", "Yes"],
}

# Human wording for the explanation charts.
FIELD_LABELS = {
    "Contract": "Contract type", "InternetService": "Internet service",
    "PaymentMethod": "Payment method", "OnlineSecurity": "Online security",
    "TechSupport": "Tech support", "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection", "PaperlessBilling": "Paperless billing",
    "tenure": "Tenure", "MonthlyCharges": "Monthly charges",
    "SeniorCitizen": "Senior citizen", "Partner": "Has partner",
    "Dependents": "Has dependents", "gender": "Gender",
    "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
}


@st.cache_data
def reference_profile():
    """A 'typical' customer: the most common value for every categorical field
    and the median for the numeric ones. Contributions are measured against
    this, so 'what makes THIS customer different' has a concrete meaning."""
    df = load_cleaned()
    reference = {}
    for column in df.columns:
        if column in ("customerID", "Churn", "tenure_group", "num_services",
                      "has_internet"):
            continue
        if df[column].dtype.kind in "if":
            reference[column] = float(df[column].median())
        else:
            reference[column] = df[column].mode().iloc[0]
    reference["TotalCharges"] = round(reference["tenure"] * reference["MonthlyCharges"], 2)
    return reference


def _with_total(profile):
    """Keep TotalCharges consistent after changing tenure or monthly charges."""
    profile = dict(profile)
    profile["TotalCharges"] = round(profile["tenure"] * profile["MonthlyCharges"], 2)
    return profile


def contribution_breakdown(raw, model_name):
    """How much each attribute adds to (or removes from) this customer's risk.

    For every field, the customer is re-scored with just that one field swapped
    to the 'typical' value. The difference is that field's contribution. This is
    a leave-one-out attribution against a fixed reference - not SHAP, and it does
    not split interaction effects, but it is honest, exact for the model as used,
    and costs a single batched prediction.
    """
    bundle = load_bundle()
    reference = reference_profile()
    fields = [f for f in FIELD_LABELS if f in raw]

    variants = [_with_total(raw)]
    for field in fields:
        swapped = dict(raw)
        swapped[field] = reference[field]
        variants.append(_with_total(swapped))

    encoded = scaled(encode_profiles(variants, bundle), model_name)
    probas = bundle["models"][model_name].predict_proba(encoded)[:, 1]

    actual = float(probas[0])
    rows = []
    for i, field in enumerate(fields):
        rows.append({
            "Field": FIELD_LABELS[field],
            "Value": raw[field],
            "Contribution": actual - float(probas[i + 1]),
        })
    frame = pd.DataFrame(rows)
    frame = frame[frame["Contribution"].abs() > 1e-6]
    return frame.reindex(frame["Contribution"].abs().sort_values().index), actual


def retention_levers(raw, model_name):
    """Every single change a retention team could make, ranked by how much it
    would move this customer's risk. The top row is the 'biggest lever'."""
    bundle = load_bundle()
    current = _with_total(raw)

    variants, meta = [current], []
    for field, options in ACTIONABLE_FIELDS.items():
        for option in options:
            if option == raw.get(field):
                continue
            changed = dict(raw)
            changed[field] = option
            variants.append(_with_total(changed))
            meta.append((field, option))

    encoded = scaled(encode_profiles(variants, bundle), model_name)
    probas = bundle["models"][model_name].predict_proba(encoded)[:, 1]

    current_risk = float(probas[0])
    rows = [{
        "Field": FIELD_LABELS.get(field, field),
        "Change to": option,
        "New risk": float(probas[i + 1]),
        "Reduction": current_risk - float(probas[i + 1]),
    } for i, (field, option) in enumerate(meta)]
    return pd.DataFrame(rows).sort_values("Reduction", ascending=False), current_risk


def categorical_whatif(raw, model_name, field):
    """Risk under every option of one categorical field, everything else fixed."""
    bundle = load_bundle()
    options = ACTIONABLE_FIELDS[field]
    variants = []
    for option in options:
        changed = dict(raw)
        changed[field] = option
        variants.append(_with_total(changed))
    encoded = scaled(encode_profiles(variants, bundle), model_name)
    probas = bundle["models"][model_name].predict_proba(encoded)[:, 1]
    return pd.DataFrame({"Option": options, "Risk": probas})
