import pandas as pd
import json

print("🚀 Script started")

df = pd.read_csv("cardio_data_processed_cleaned.csv")

print("📊 Dataset loaded:", df.shape)

numerical_defaults = {
    "age_years": float(df["age_years"].median()),
    "height": float(df["height"].median()),
    "weight": float(df["weight"].median()),
    "ap_hi": float(df["ap_hi"].median()),
    "ap_lo": float(df["ap_lo"].median()),
    "bmi": float(df["bmi"].median())
}

categorical_defaults = {
    "gender": int(df["gender"].mode()[0]),
    "cholesterol": int(df["cholesterol"].mode()[0]),
    "gluc": int(df["gluc"].mode()[0]),
    "smoke": int(df["smoke"].mode()[0]),
    "alco": int(df["alco"].mode()[0]),
    "active": int(df["active"].mode()[0])
}

defaults = {**numerical_defaults, **categorical_defaults}

print("\n📊 DEFAULT VALUES:")
print(defaults)

with open("defaults.json", "w") as f:
    json.dump(defaults, f, indent=4)

print("\n💾 Saved to defaults.json")