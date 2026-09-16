import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("cleaned_data.csv")
os.makedirs("charts", exist_ok=True)

def save_bar_chart(filename, question, answer, labels, values, ylabel, color="#2E1A47"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, values, color=color)
    ax.set_title(question, fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel(ylabel)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    plt.xticks(rotation=20, ha="right")
    fig.text(0.5, -0.02, "Answer: " + answer, ha="center", fontsize=10,
              wrap=True, style="italic", color="#333333")
    plt.tight_layout()
    plt.savefig(f"charts/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved charts/{filename}")

# ---------- Analysis 1: Credit history rate by profile ----------
rate_dep = (df.groupby("Dependents")["Credit_History"].mean() * 100).round(1)
save_bar_chart(
    "analysis1_dependents.png",
    "Question: Does number of dependents relate to credit history?",
    "Applicants with 3 or more dependents have the lowest credit history rate.",
    rate_dep.index.tolist(), rate_dep.values.tolist(), "Percent with good credit history"
)

rate_edu = (df.groupby("Education")["Credit_History"].mean() * 100).round(1)
save_bar_chart(
    "analysis1_education.png",
    "Question: Does education relate to credit history?",
    "Graduates have a somewhat higher credit history rate than non-graduates.",
    rate_edu.index.tolist(), rate_edu.values.tolist(), "Percent with good credit history"
)

# ---------- Analysis 2: Income vs loan burden by credit status ----------
ratio = df.groupby("Credit_History_Status")["LoanIncomeRatio"].mean().round(2)
save_bar_chart(
    "analysis2_loan_burden.png",
    "Question: Do applicants with no credit history carry a heavier loan burden?",
    "Yes. No-credit-history applicants have the highest loan-to-income ratio.",
    ratio.index.tolist(), ratio.values.tolist(), "Loan amount vs income (ratio)",
    color="#D4A843"
)

# ---------- Analysis 3: Credit history and money by area ----------
rate_area = (df.groupby("Property_Area")["Credit_History"].mean() * 100).round(1)
save_bar_chart(
    "analysis3_area.png",
    "Question: Does the area someone lives in relate to credit history?",
    "Not much. Credit history rates are close across Rural, Semiurban, and Urban.",
    rate_area.index.tolist(), rate_area.values.tolist(), "Percent with good credit history"
)

# ---------- Analysis 4: Loan amount and term patterns ----------
avg_loan = df.groupby("Credit_History_Status")["LoanAmount_Actual"].mean().round(0)
save_bar_chart(
    "analysis4_loan_amount.png",
    "Question: Do applicants with good credit history take smaller loans?",
    "No. Average loan amount is nearly the same across all credit history groups.",
    avg_loan.index.tolist(), avg_loan.values.tolist(), "Average loan amount (rupees)"
)

# ---------- Analysis 5: Does a co-applicant lower risk ----------
df["Coapplicant_Label"] = df["Has_Coapplicant"].map({0: "No Co-applicant", 1: "Has Co-applicant"})
rate_coapp = (df.groupby("Coapplicant_Label")["Credit_History"].mean() * 100).round(1)
save_bar_chart(
    "analysis5_coapplicant.png",
    "Question: Does having a co-applicant mean better credit history?",
    "No. Credit history rate is about the same, slightly lower with a co-applicant.",
    rate_coapp.index.tolist(), rate_coapp.values.tolist(), "Percent with good credit history",
    color="#D4A843"
)

# ---------- Analysis 6: Education and self-employment ----------
df["edu_se"] = df["Education"] + "\n" + df["Self_Employed"].map({"Yes": "Self-Employed", "No": "Salaried"})
rate_edu_se = (df.groupby("edu_se")["Credit_History"].mean() * 100).round(1)
save_bar_chart(
    "analysis6_education_selfemployed.png",
    "Question: Are self-employed applicants riskier than salaried ones?",
    "No. Self-employed applicants show an equal or better credit history rate.",
    rate_edu_se.index.tolist(), rate_edu_se.values.tolist(), "Percent with good credit history"
)

print("\nAll charts saved in the 'charts' folder.")