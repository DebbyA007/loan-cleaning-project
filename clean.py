import pandas as pd
import numpy as np

df = pd.read_csv('raw_data.csv')
print("RAW SHAPE:", df.shape)

log = []

# 1. Strip whitespace from string columns
str_cols = df.select_dtypes(include='object').columns.tolist() + ['Loan_ID']
str_cols = list(dict.fromkeys(str_cols))
for c in str_cols:
    if df[c].dtype == object or str(df[c].dtype) == 'str':
        df[c] = df[c].astype('string').str.strip()

# 2. Fix Gender inconsistent labels (m/Male, f/Female)
before = df['Gender'].value_counts(dropna=False).to_dict()
df['Gender'] = df['Gender'].replace({'m': 'Male', 'f': 'Female'})
log.append(f"Gender: standardized {before} -> {df['Gender'].value_counts(dropna=False).to_dict()}")

# 3. Fix Property_Area typos -> map to 3 canonical categories
area_map = {
    'Urban': 'Urban', 'Uban': 'Urban', 'Urben': 'Urban', 'Urbun': 'Urban', 'Urbn': 'Urban',
    'Semiurban': 'Semiurban', 'Semiurbn': 'Semiurban', 'Semiurben': 'Semiurban', 'Semurban': 'Semiurban',
    'Rural': 'Rural', 'Rurl': 'Rural', 'Rurel': 'Rural', 'Rurall': 'Rural', 'Rual': 'Rural'
}
before_n = df['Property_Area'].nunique()
df['Property_Area'] = df['Property_Area'].map(area_map)
log.append(f"Property_Area: collapsed {before_n} misspelled variants -> {df['Property_Area'].nunique()} clean categories")

# 4. Dependents: keep as category, but also make a numeric version (3+ -> 3)
df['Dependents'] = df['Dependents'].astype('string')
df['Dependents_numeric'] = df['Dependents'].replace({'3+': '3'}).astype('Float64')

# 5. Missing value handling
missing_before = df.isnull().sum()

# Categorical -> fill with mode (the most common value)
for c in ['Gender', 'Married', 'Self_Employed', 'Dependents']:
    mode_val = df[c].mode(dropna=True)[0]
    df[c] = df[c].fillna(mode_val)

# Numeric -> fill with median (the middle value, resistant to outliers)
for c in ['LoanAmount', 'Loan_Amount_Term']:
    med = df[c].median()
    df[c] = df[c].fillna(med)

df['Dependents_numeric'] = df['Dependents'].replace({'3+': '3'}).astype('Float64')

# Credit_History: missing here is meaningful, so keep it as its own category
df['Credit_History_Status'] = df['Credit_History'].map({1.0: 'Has Credit History', 0.0: 'No Credit History'})
df['Credit_History_Status'] = df['Credit_History_Status'].fillna('Unknown/Not on File')

log.append(f"Missing values before: {missing_before.to_dict()}")
log.append(f"Missing values after: {df.isnull().sum().to_dict()}")

# 6. Duplicates
dupes = df.duplicated(subset=['Loan_ID']).sum()
df = df.drop_duplicates(subset=['Loan_ID'])
log.append(f"Duplicate Loan_IDs removed: {dupes}")

# 7. Data type fixes
df['ApplicantIncome'] = df['ApplicantIncome'].astype(float)
df['CoapplicantIncome'] = df['CoapplicantIncome'].astype(float)
df['LoanAmount'] = df['LoanAmount'].astype(float)
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype(float)

# 8. Sanity check for impossible values
bad_income = (df['ApplicantIncome'] < 0).sum()
bad_loan = (df['LoanAmount'] <= 0).sum()
log.append(f"Negative income rows: {bad_income}, non-positive loan amount rows: {bad_loan}")

# 9. New columns useful for analysis later
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['LoanAmount_Actual'] = df['LoanAmount'] * 1000  # dataset stores this in thousands
df['Loan_to_Income_Ratio'] = df['LoanAmount_Actual'] / df['TotalIncome'].replace(0, np.nan)

print("CLEAN SHAPE:", df.shape)
df.to_csv('cleaned_data.csv', index=False)

with open('cleaning_log.txt', 'w') as f:
    f.write("\n".join(log))

print("\n".join(log))