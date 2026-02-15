# ML Assignment 2 — Classification + Streamlit App

## Problem Statement
Predict whether a customer holds a **credit card** (target: `Has_CreditCard`) from financial and demographic attributes.

## Dataset Description
- Source: provided CSV `Customer_Financial_Info-checkpoint.csv`
- Instances: 5000
- Features used: 11 (after dropping identifiers)

### Feature List
- Age
- Years_Experience
- Annual_Income
- Family_size
- Avg_Spending
- Education_Level
- Mortgage
- Has_Consumer_Loan
- Has_Securities_Account
- Has_CD_Account
- Uses_Online_Banking

## Models and Metrics
| ML Model Name            |   Accuracy |      AUC |   Precision |   Recall |       F1 |      MCC |
|:-------------------------|-----------:|---------:|------------:|---------:|---------:|---------:|
| Logistic Regression      |     0.7352 | 0.623884 |    0.78125  | 0.13624  | 0.232019 | 0.248744 |
| XGBoost (Ensemble)       |     0.7384 | 0.611644 |    0.857143 | 0.13079  | 0.22695  | 0.267988 |
| Random Forest (Ensemble) |     0.708  | 0.611474 |    0.507463 | 0.185286 | 0.271457 | 0.162724 |
| kNN                      |     0.716  | 0.598326 |    0.561224 | 0.149864 | 0.236559 | 0.171399 |
| Naive Bayes              |     0.7352 | 0.560955 |    0.78125  | 0.13624  | 0.232019 | 0.248744 |
| Decision Tree            |     0.6368 | 0.556263 |    0.376771 | 0.362398 | 0.369444 | 0.114566 |

## Observations
- **Logistic Regression**: Acc=0.735, AUC=0.624, F1=0.232.
- **XGBoost (Ensemble)**: Acc=0.738, AUC=0.612, F1=0.227.
- **Random Forest (Ensemble)**: Acc=0.708, AUC=0.611, F1=0.271.
- **kNN**: Acc=0.716, AUC=0.598, F1=0.237.
- **Naive Bayes**: Acc=0.735, AUC=0.561, F1=0.232.
- **Decision Tree**: Acc=0.637, AUC=0.556, F1=0.369.

## How to Run Locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
