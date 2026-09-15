"""
EDA — Loan Applicant Risk & Segmentation Analysis
Reads cleaned_data.csv and prints/saves the key numbers used in the presentation.
Run: python3 eda.py
"""
import pandas as pd

df = pd.read_csv("cleaned_data.csv")

print("=" * 60)
print("1. APPLICANT SEGMENTATION BY PROPERTY AREA")
print("=" * 60)
print(df["Property_Area"].value_counts())
print("\nMedian total income by area:")
print(df.groupby("Property_Area")["TotalIncome"].median())

print("\n" + "=" * 60)
print("2. CREDIT HISTORY LANDSCAPE")
print("=" * 60)
print(df["Credit_History_Status"].value_counts())
print("Share of applicants with no usable credit history: {:.1f}%".format(
    100 * (df["Credit_History_Status"] != "Has Credit History").mean()
))

print("\n" + "=" * 60)
print("3. RISK SIGNAL — LOAN-TO-INCOME RATIO BY CREDIT STATUS")
print("=" * 60)
print(df.groupby("Credit_History_Status")["Loan_to_Income_Ratio"].mean().round(2))

print("\n" + "=" * 60)
print("4. INCOME VS. LOAN AMOUNT RELATIONSHIP")
print("=" * 60)
print("Correlation (TotalIncome vs LoanAmount):",
      round(df["TotalIncome"].corr(df["LoanAmount_Actual"]), 3))

print("\n" + "=" * 60)
print("5. EMPLOYMENT & MARITAL STATUS VS. INCOME")
print("=" * 60)
print("Median income by employment type:")
print(df.groupby("Self_Employed")["TotalIncome"].median())
print("\nMedian income by marital status:")
print(df.groupby("Married")["TotalIncome"].median())

print("\n" + "=" * 60)
print("6. FULL CORRELATION MATRIX (numeric fields)")
print("=" * 60)
num_cols = ["ApplicantIncome", "CoapplicantIncome", "TotalIncome",
            "LoanAmount_Actual", "Loan_Amount_Term", "Loan_to_Income_Ratio"]
print(df[num_cols].corr().round(2))