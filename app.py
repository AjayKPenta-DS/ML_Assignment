
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Credit Card Prediction", page_icon="💳", layout="wide")
st.title("Customer Credit Card Prediction — ML Assignment 2")

st.sidebar.header("Model Selection")
model_files = {
    'Logistic Regression': 'model/logistic_regression.pkl',
    'Decision Tree': 'model/decision_tree.pkl',
    'kNN': 'model/knn.pkl',
    'Naive Bayes': 'model/naive_bayes.pkl',
    'Random Forest (Ensemble)': 'model/random_forest_ensemble.pkl',
    'XGBoost (Ensemble)': 'model/xgboost_ensemble.pkl',
}
model_name = st.sidebar.selectbox("Choose a model", list(model_files.keys()))

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model_path = model_files[model_name]
model = load_model(model_path)

st.sidebar.header("Upload Test CSV")
upload = st.sidebar.file_uploader("Upload CSV with the same schema as training (except target)", type=['csv'])

if upload is not None:
    data = pd.read_csv(upload)
    # Basic column cleaning similar to training
    data.columns = [c.strip().replace(' ', '_') for c in data.columns]
    for col in ['Customer_ID','ZIP_Code','Has_CreditCard']:
        if col in data.columns:
            data = data.drop(columns=[col])
    st.write("### Preview", data.head())
    preds = model.predict(data)
    prob = model.predict_proba(data)[:,1] if hasattr(model, 'predict_proba') else None
    out = pd.DataFrame({'prediction': preds})
    if prob is not None:
        out['probability'] = prob
    st.write("### Predictions", out.head())
else:
    st.info("Upload a CSV to get predictions. For demo, no data uploaded.")
