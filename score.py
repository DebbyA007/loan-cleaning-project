"""
Approval Likelihood Scorecard
------------------------------
raw_data.csv has NO historical approval/denial outcome, so a trained ML
classifier cannot be built from this file alone. Instead, this script builds
a transparent, rule-based scorecard using the same factors underwriters
typically weigh manually. This mirrors how real lenders score applicants
BEFORE they have enough labeled history to train a model.

Weights are documented and adjustable -- they are expert-assigned, not
learned from data. If a real Loan_Status (Approved/Denied) column becomes
available, replace this scorecard with a trained logistic regression /
gradient boosting model using the same features.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("cleaned_data.csv")

def credit_score(status):
    return {"Has Credit History": 40, "Unknown/Not on File": 20, "No Credit History": 8}[status]

def ratio_score(ratio):
    if pd.isna(ratio):
        return 12
    if ratio <= 18:
        return 25
    elif ratio <= 26:
        return 18
    elif ratio <= 35:
        return 10
    else:
        return 3

def employment_score(row):
    # Salaried income is easier to verify than self-employed income
    return 10 if row["Self_Employed"] == "No" else 6

def stability_score(row):
    pts = 0
    pts += 5 if row["Married"] == "Yes" else 2          # dual-income / shared liability
    pts += 5 if row["Education"] == "Graduate" else 2    # proxy for income stability
    pts += 5 if row["Dependents"] == "0" else 2          # fewer financial obligations
    pts += 5 if row["Loan_Amount_Term"] >= 360 else 3    # longer term = smaller installment burden
    return pts  # max 20

df["Score_CreditHistory"] = df["Credit_History_Status"].apply(credit_score)          # 0-40
df["Score_LoanToIncome"] = df["Loan_to_Income_Ratio"].apply(ratio_score)             # 0-25
df["Score_Employment"] = df.apply(employment_score, axis=1)                          # 0-10
df["Score_Stability"] = df.apply(stability_score, axis=1)                            # 0-20
# max possible = 40+25+10+20 = 95; normalize to 0-100
raw_total = df["Score_CreditHistory"] + df["Score_LoanToIncome"] + df["Score_Employment"] + df["Score_Stability"]
df["Approval_Likelihood_Score"] = (raw_total / 95 * 100).round(1)

def band(score):
    if score >= 70:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

df["Likelihood_Band"] = df["Approval_Likelihood_Score"].apply(band)

df.to_csv("scored_data.csv", index=False)

print("Score summary:")
print(df["Approval_Likelihood_Score"].describe().round(1))
print("\nBand distribution:")
print(df["Likelihood_Band"].value_counts())
print("\nBand distribution %:")
print((df["Likelihood_Band"].value_counts(normalize=True) * 100).round(1))
print("\nAvg score by Credit History status:")
print(df.groupby("Credit_History_Status")["Approval_Likelihood_Score"].mean().round(1))
print("\nAvg score by Property Area:")
print(df.groupby("Property_Area")["Approval_Likelihood_Score"].mean().round(1))
print("\nAvg score by Self_Employed:")
print(df.groupby("Self_Employed")["Approval_Likelihood_Score"].mean().round(1))
print("\nBand vs Property Area (counts):")
print(pd.crosstab(df["Property_Area"], df["Likelihood_Band"]))