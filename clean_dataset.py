import pandas as pd
import numpy as np

# =============================
# LOAD DATA
# =============================
df = pd.read_csv("cardio_data_processed.csv")

print("Original shape:", df.shape)

# =============================
# 1. Remove duplicates
# =============================
df = df.drop_duplicates()

# =============================
# 2. Remove missing values
# =============================
df = df.dropna()

# =============================
# 3. Fix Age (days → years)
# =============================
if df["age"].max() > 1000:
    df["age_years"] = (df["age"] / 365).astype(int)

# Validate age
df = df[
    (df["age_years"] >= 18) &
    (df["age_years"] <= 100)
]

# =============================
# 4. Recompute BMI
# =============================
df["height_m"] = df["height"] / 100

df["bmi"] = df["weight"] / (df["height_m"] ** 2)

df = df.drop(columns=["height_m"])

# Validate BMI
df = df[
    (df["bmi"] >= 10) &
    (df["bmi"] <= 60)
]

# =============================
# 5. Validate Height & Weight
# =============================
df = df[
    (df["height"] >= 120) &
    (df["height"] <= 220)
]

df = df[
    (df["weight"] >= 30) &
    (df["weight"] <= 200)
]

# =============================
# 6. Fix Blood Pressure
# =============================

# Swap if diastolic > systolic
df.loc[
    df["ap_lo"] > df["ap_hi"],
    ["ap_lo", "ap_hi"]
] = df.loc[
    df["ap_lo"] > df["ap_hi"],
    ["ap_hi", "ap_lo"]
].values

# Remove extreme BP
df = df[
    (df["ap_hi"] >= 70) &
    (df["ap_hi"] <= 250)
]

df = df[
    (df["ap_lo"] >= 40) &
    (df["ap_lo"] <= 150)
]

# =============================
# 6.5 Pulse Pressure Check
# =============================
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

df = df[
    (df["pulse_pressure"] >= 20) &
    (df["pulse_pressure"] <= 100)
]

# =============================
# 6.6 Mean Arterial Pressure
# =============================
df["map"] = df["ap_lo"] + (
    (df["ap_hi"] - df["ap_lo"]) / 3
)

df = df[
    (df["map"] >= 60) &
    (df["map"] <= 150)
]

# Drop temporary validation columns
df = df.drop(columns=[
    "pulse_pressure",
    "map"
])

# =============================
# 7. Validate Cholesterol
# =============================
df = df[
    df["cholesterol"].isin([1, 2, 3])
]

# =============================
# 8. Validate Glucose
# =============================
df = df[
    df["gluc"].isin([1, 2, 3])
]

# =============================
# 9. Binary Column Cleanup
# =============================
binary_cols = [
    "gender",
    "smoke",
    "alco",
    "active",
    "cardio"
]

for col in binary_cols:
    df[col] = df[col].astype(int)

# =============================
# 10. IQR Outlier Removal
# =============================

def remove_outliers_iqr(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return data[
        (data[column] >= lower) &
        (data[column] <= upper)
    ]

for col in ["height", "weight", "bmi"]:
    df = remove_outliers_iqr(df, col)

# =============================
# 11. Drop redundant columns
# =============================
if "bp_category_encoded" in df.columns:
    df = df.drop(columns=["bp_category_encoded"])

# =============================
# 12. Reset index
# =============================
df = df.reset_index(drop=True)

print("Final cleaned shape:", df.shape)

# =============================
# SAVE CLEANED DATASET
# =============================
df.to_csv(
    "cardio_data_processed_cleaned.csv",
    index=False
)

print("\n✅ Cleaned dataset saved successfully!")