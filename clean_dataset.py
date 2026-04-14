import pandas as pd
import numpy as np

# Load dataset (if already loaded, skip this line)
df = pd.read_csv("cardio_data_processed.csv")
# -----------------------------
# 1. Remove duplicates
# -----------------------------
df = df.drop_duplicates()

# -----------------------------
# 2. Handle missing values
# -----------------------------
df = df.dropna()

# -----------------------------
# 3. Fix age (convert from days if needed)
# -----------------------------
if df["age"].max() > 1000:
    df["age_years"] = (df["age"] / 365).astype(int)
else:
    df["age_years"] = df["age_years"].astype(int)

# -----------------------------
# 4. Recompute BMI (ensure correctness)
# -----------------------------
df["height_m"] = df["height"] / 100
df["bmi"] = df["weight"] / (df["height_m"] ** 2)
df = df.drop(columns=["height_m"])

# -----------------------------
# 5. Fix invalid height/weight values (outlier removal)
# -----------------------------
df = df[(df["height"] >= 120) & (df["height"] <= 220)]
df = df[(df["weight"] >= 30) & (df["weight"] <= 200)]

# -----------------------------
# 6. Fix blood pressure inconsistencies
# -----------------------------
df.loc[df["ap_lo"] > df["ap_hi"], ["ap_lo", "ap_hi"]] = df.loc[
    df["ap_lo"] > df["ap_hi"], ["ap_hi", "ap_lo"]
].values

# Remove extreme BP values
df = df[(df["ap_hi"] >= 70) & (df["ap_hi"] <= 250)]
df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 150)]

# -----------------------------
# 7. Ensure binary columns are clean
# -----------------------------
binary_cols = ["gender", "smoke", "alco", "active", "cardio"]
for col in binary_cols:
    df[col] = df[col].astype(int)

# -----------------------------
# 8. Standardize categorical columns
# -----------------------------
df["cholesterol"] = df["cholesterol"].astype(int)
df["gluc"] = df["gluc"].astype(int)

# -----------------------------
# 9. Drop redundant columns
# -----------------------------
if "bp_category_encoded" in df.columns:
    df = df.drop(columns=["bp_category_encoded"])

# -----------------------------
# 10. Reset index
# -----------------------------
df = df.reset_index(drop=True)


# -----------------------------
# Save final cleaned dataset
# -----------------------------
df.to_csv("cardio_data_processed_cleaned.csv", index=False)

print("Cleaned dataset saved successfully!")