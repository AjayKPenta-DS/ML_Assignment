# app.py — resilient Streamlit app that auto-retrains models that fail to unpickle.
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Base estimators
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Optional XGBoost with safe fallback
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

st.set_page_config(page_title="Credit Card Prediction", page_icon="💳", layout="wide")
st.title("Customer Credit Card Prediction — ML Assignment 2")

# ------------------------------------------------------------------------------
# Paths & files
# ------------------------------------------------------------------------------
DATA_PATH = Path("data_cleaned_customer_financial_info.csv")  # local training data
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
FEATURES_DIR = Path("features")
FEATURES_DIR.mkdir(exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree": MODEL_DIR / "decision_tree.pkl",
    "kNN": MODEL_DIR / "knn.pkl",
    "Naive Bayes": MODEL_DIR / "naive_bayes.pkl",
    "Random Forest (Ensemble)": MODEL_DIR / "random_forest_ensemble.pkl",
    "XGBoost (Ensemble)": MODEL_DIR / "xgboost_ensemble.pkl",
}
FEATURE_JSON = {
    name: FEATURES_DIR / (MODEL_FILES[name].stem + "_features.json")
    for name in MODEL_FILES
}

st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def clean_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # drop identifiers and target if they appear
    for col in ["Customer_ID", "ZIP_Code", "Has_CreditCard"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def load_training_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if not DATA_PATH.exists():
        st.error(
            f"Training file '{DATA_PATH.name}' not found. "
            "Place it in the app folder or upload a CSV to train."
        )
        st.stop()
    df = pd.read_csv(DATA_PATH)
    # Ensure consistent cleaning as in training script
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    if "Years_Experience" in df.columns:
        df["Years_Experience"] = df["Years_Experience"].clip(lower=0)

    if "Has_CreditCard" not in df.columns:
        st.error("Target column 'Has_CreditCard' not found in training CSV.")
        st.stop()

    y = df["Has_CreditCard"]
    X = df.drop(columns=["Has_CreditCard", "Customer_ID", "ZIP_Code"], errors="ignore")
    feature_cols = list(X.columns)
    return X, y, feature_cols

def build_estimator(name: str):
    """
    Build an estimator WITHOUT ColumnTransformer.
    For LR/kNN/NB we add a plain StandardScaler in a Pipeline.
    For tree/forest we use the raw estimator (they don't need scaling).
    For XGB we add no ColumnTransformer either.
    """
    if name == "Logistic Regression":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))
        ])
    elif name == "kNN":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15))
        ])
    elif name == "Naive Bayes":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())
        ])
    elif name == "Decision Tree":
        pipe = DecisionTreeClassifier(random_state=42)
    elif name == "Random Forest (Ensemble)":
        pipe = RandomForestClassifier(n_estimators=300, random_state=42)
    elif name == "XGBoost (Ensemble)":
        if XGB_AVAILABLE:
            pipe = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=42, n_jobs=4, eval_metric="logloss"
            )
        else:
            st.warning("XGBoost not available in this environment. "
                       "Using RandomForest as a fallback.")
            pipe = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        st.error(f"Unknown model name: {name}")
        st.stop()
    return pipe

def save_feature_order(name: str, feature_cols: list[str]):
    FEATURES_DIR.mkdir(exist_ok=True)
    with open(FEATURE_JSON[name], "w", encoding="utf-8") as f:
        json.dump({"feature_order": feature_cols}, f)

def load_feature_order(name: str):
    try:
        with open(FEATURE_JSON[name], "r", encoding="utf-8") as f:
            return json.load(f)["feature_order"]
    except Exception:
        return None

@st.cache_resource(show_spinner=True)
def load_or_train_model(name: str):
    """
    1) Try to load existing model pickle.
    2) If loading fails with AttributeError (legacy pickle), retrain the model
       from local training CSV using a no-ColumnTransformer setup and save it.
    3) Always return (model, feature_order).
    """
    pkl = str(MODEL_FILES[name])
    # Step 1 — try joblib.load
    try:
        model = joblib.load(pkl)
        # Try loading stored feature order
        feature_cols = load_feature_order(name)
        if feature_cols is None:
            # Fallback: ensure we have feature order from training data
            X, y, feature_cols = load_training_data()
        return model, feature_cols
    except Exception as e:
        st.info(f"Reloading '{name}' failed ({type(e).__name__}: {e}). "
                f"Retraining '{name}' now without ColumnTransformer...")

    # Step 2 — retrain without ColumnTransformer
    X, y, feature_cols = load_training_data()
    model = build_estimator(name)
    # For speed, you can fit on full data (fast enough for this dataset)
    model.fit(X, y)

    # Save for future runs
    joblib.dump(model, pkl)
    save_feature_order(name, feature_cols)
    st.success(f"Re-trained and saved new '{name}' model (no ColumnTransformer).")
    return model, feature_cols

def align_columns_for_inference(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    # ensure same columns and order; any missing columns -> fill with 0
    current = set(df.columns)
    needed = set(expected_cols)
    missing = list(needed - current)
    if missing:
        st.warning(f"Uploaded CSV is missing columns: {missing}. "
                   "They will be filled with 0 for inference.")
    extra = list(current - needed)
    if extra:
        st.info(f"Ignoring extra columns not used by model: {extra}")
    aligned = df.reindex(columns=expected_cols, fill_value=0)
    return aligned

# ------------------------------------------------------------------------------
# Load model (or retrain if needed)
# ------------------------------------------------------------------------------
model, feature_order = load_or_train_model(model_name)

# ------------------------------------------------------------------------------
# UI: inference
# ------------------------------------------------------------------------------
st.sidebar.header("Upload Test CSV")
upload = st.sidebar.file_uploader(
    "Upload CSV with same schema as training (except target)",
    type=["csv"]
)

if upload is not None:
    data = pd.read_csv(upload)
    data = clean_feature_columns(data)
    data = align_columns_for_inference(data, feature_order)
    st.write("### Preview", data.head())

    try:
        preds = model.predict(data)
        prob = model.predict_proba(data)[:, 1] if hasattr(model, "predict_proba") else None
        out = pd.DataFrame({"prediction": preds})
        if prob is not None:
            out["probability"] = prob
        st.write("### Predictions", out.head())
    except Exception as e:
        st.error(
            "Prediction failed. This can happen if your CSV doesn’t match the model’s "
            "expected features or due to environment issues.\n\n"
            f"Details: {type(e).__name__}: {e}"
        )
else:
    st.info("Upload a CSV to get predictions. For demo, no data uploaded.")

st.caption(
    f"Python: {sys.version.split()[0]} | "
    f"sklearn pickles from older versions may break on newer versions. "
    f"This app auto-retrains models that can't be unpickled, saving compatible pickles "
    f"without ColumnTransformer."
)