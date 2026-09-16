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

    return 10 if row["Self_Employed"] == "No" else 6

def stability_score(row):
    pts = 0
    pts += 5 if row["Married"] == "Yes" else 2          
    pts += 5 if row["Education"] == "Graduate" else 2   
    pts += 5 if row["Dependents"] == "0" else 2          
    pts += 5 if row["Loan_Amount_Term"] >= 360 else 3    
    return pts  

df["Score_CreditHistory"] = df["Credit_History_Status"].apply(credit_score)          
df["Score_LoanToIncome"] = df["Loan_to_Income_Ratio"].apply(ratio_score)            
df["Score_Employment"] = df.apply(employment_score, axis=1)                          
df["Score_Stability"] = df.apply(stability_score, axis=1)                            

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