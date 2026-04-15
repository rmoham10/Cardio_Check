import pandas as pd
import json

print("🚀 Script started")

df = pd.read_csv("cardio_data_processed.csv")

print("📊 Dataset loaded:", df.shape)

# ─────────────────────────────────────────────
# NUMERICAL DEFAULTS (MEDIAN = BEST PRACTICE)
# ─────────────────────────────────────────────
numerical_cols = [
    "age_years", "height", "weight",
    "ap_hi", "ap_lo", "bmi"
]

numerical_defaults = {
    col: float(df[col].median()) for col in numerical_cols
}

# ─────────────────────────────────────────────
# CATEGORICAL DEFAULTS (HEALTHY BASELINE FIX)
# ─────────────────────────────────────────────

categorical_defaults = {
    "gender": int(df["gender"].mode()[0]),

    # SAFE CLINICAL DEFAULTS (NOT RAW MODE)
    "cholesterol": 1,  # normal
    "gluc": 1,         # normal
    "smoke": 0,        # non-smoker
    "alco": 0,         # non-alcohol
    "active": 1        # physically active
}

# ─────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────
defaults = {**numerical_defaults, **categorical_defaults}

# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────
print("\n📊 DEFAULT VALUES (PRODUCTION SAFE):")
for k, v in defaults.items():
    print(f"{k}: {v}")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
with open("defaults.json", "w") as f:
    json.dump(defaults, f, indent=4)

print("\n💾 Saved to defaults.json")