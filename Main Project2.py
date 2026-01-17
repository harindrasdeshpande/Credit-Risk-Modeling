"""
Goal
Create a synthetic loan dataset suitable for credit risk analysis.
"""
# Goal 1: Credit Risk Modeling (Industry-Grade)

import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = {
    "Age": np.random.randint(21, 65, n),
    "Income": np.random.randint(20000, 150000, n),
    "Loan_Amount": np.random.randint(50000, 1000000, n),
    "Credit_Score": np.random.randint(300, 850, n),
    "Loan_Tenure_Months": np.random.choice([12, 24, 36, 48, 60], n),
    "Past_Defaults": np.random.choice([0, 1, 2, 3], n),
}

df = pd.DataFrame(data)

# Default probability logic (risk-based)
df["Default"] = (
    (df["Credit_Score"] < 600).astype(int)
    | (df["Past_Defaults"] > 1).astype(int)
    | (df["Loan_Amount"] > 700000).astype(int)
)

print(df.head())
print(df["Default"].value_counts())

# Goal 2: Risk Segmentation & EDA (bank-style analysis)


print("\n--- DATA OVERVIEW ---")
print(df.describe())                                    # Basic Data Checks

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())                                # Basic Data Checks 


default_rate = df["Default"].mean()
print("\nOverall Default Rate:", round(default_rate, 3))             # Default Rate Analysis 


credit_bins = pd.cut(
    df["Credit_Score"],
    bins=[300, 550, 650, 750, 850],
    labels=["Very Low", "Low", "Medium", "High"],
    include_lowest=True
)

df["Credit_Risk_Band"] = credit_bins

risk_summary = (
    df.groupby("Credit_Risk_Band", observed=True)["Default"]
    .mean()
    .sort_index()
)

print("\nDefault Rate by Credit Score Band:")
print(risk_summary)                                                   # Risk Segmentation 


# Goal 3: Logistic Regression PD Model 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

X = df.drop(columns=["Default", "Credit_Risk_Band"])
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- MODEL PERFORMANCE ---")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", round(roc_auc, 3))

# Goal 4: Feature Importance & Risk Interpretation

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\n--- FEATURE IMPORTANCE (LOGISTIC COEFFICIENTS) ---")
print(feature_importance)                                                     # Extract Feature Importance 


feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()
feature_importance.sort_values("Abs_Coefficient", ascending=False, inplace=True)

print("\n--- SORTED BY RISK IMPACT ---")
print(feature_importance)                                                     # Normalization Coefficent 

"""
CONCLUSION
Built a probability-of-default (PD) model using logistic regression
Performed risk segmentation using credit score bands
Evaluated model performance using ROC-AUC
Interpreted risk drivers using model coefficients
"""