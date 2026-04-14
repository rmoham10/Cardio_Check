"""
Heart Disease Training with
Feature Engineering + Model Comparison + GradientBoosting Tuning
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import numpy as np


# ==========================
# FILE PATHS
# ==========================

DATA_PATH = "cardio_data_processed_cleaned.csv"

MODEL_PATH = "best_model.pkl"

REPORT_PATH = "training_report.txt"


# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(DATA_PATH)

TARGET = "cardio"


# ==========================
# FEATURE ENGINEERING
# ==========================

print("\n⚙️ Creating engineered features...")

df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

df["mean_arterial_pressure"] = (
    df["ap_lo"]
    + (df["ap_hi"] - df["ap_lo"]) / 3
)

df["bmi_age"] = df["bmi"] * df["age_years"]

df["chol_bmi"] = df["cholesterol"] * df["bmi"]

df["bp_ratio"] = df["ap_hi"] / df["ap_lo"]


# ==========================
# FEATURE LIST
# ==========================

FEATURES = [
    "age_years",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "bmi",

    # NEW FEATURES
    "pulse_pressure",
    "mean_arterial_pressure",
    "bmi_age",
    "chol_bmi",
    "bp_ratio"
]


X = df[FEATURES]

y = df[TARGET]


# ==========================
# SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    stratify=y,

    random_state=42
)


# ==========================
# PREPROCESSOR
# ==========================

preprocessor = ColumnTransformer([

    ("num", Pipeline([

        ("imputer", KNNImputer(n_neighbors=5)),

        ("scaler", StandardScaler())

    ]), FEATURES)

])


# ==========================
# FEATURE SELECTION
# ==========================

selector_model = RandomForestClassifier(

    n_estimators=300,

    random_state=42
)

selector = SelectFromModel(

    selector_model,

    threshold="median"
)


# ==========================
# PIPELINE FACTORY
# ==========================

def make_pipeline(model):

    return ImbPipeline([

        ("preprocessor", preprocessor),

        ("feature_selection", selector),

        ("smote", SMOTE(random_state=42)),

        ("model", model)

    ])


# ==========================
# MODELS
# ==========================

models = {

    "LogisticRegression":

        make_pipeline(

            LogisticRegression(

                max_iter=2000,

                class_weight="balanced"

            )

        ),


    "RandomForest":

        make_pipeline(

            RandomForestClassifier(

                n_estimators=300,

                random_state=42

            )

        ),


    "ExtraTrees":

        make_pipeline(

            ExtraTreesClassifier(

                n_estimators=300,

                random_state=42

            )

        ),


    "GradientBoosting":

        make_pipeline(

            GradientBoostingClassifier()

        ),


    "XGBoost":

        make_pipeline(

            XGBClassifier(

                n_estimators=300,

                learning_rate=0.05,

                max_depth=6,

                eval_metric="logloss",

                random_state=42

            )

        ),


    "LightGBM":

        make_pipeline(

            LGBMClassifier(

                n_estimators=300,

                learning_rate=0.05,

                verbosity=-1,

                random_state=42

            )

        ),


    "SVM":

        make_pipeline(

            SVC(

                probability=True,

                kernel="rbf",

                class_weight="balanced"

            )

        )
}


# ==========================
# CROSS VALIDATION
# ==========================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)

scores = {}

print("\n🔍 Comparing models...\n")

for name, model in models.items():

    auc = cross_val_score(

        model,

        X_train,

        y_train,

        cv=cv,

        scoring="roc_auc"

    )

    scores[name] = auc.mean()

    print(f"{name}: {auc.mean():.4f}")


# ==========================
# BEST MODEL
# ==========================

best_model_name = max(

    scores,

    key=scores.get
)

best_model = models[best_model_name]

print("\n🏆 BEST MODEL:", best_model_name)


# ==========================
# TUNE GRADIENT BOOSTING
# ==========================

if best_model_name == "GradientBoosting":

    print("\n⚙️ Tuning GradientBoosting...")

    param_dist = {

        "model__n_estimators":

            [200,300,400],

        "model__learning_rate":

            [0.01,0.05,0.1],

        "model__max_depth":

            [3,4,5]

    }

    tuner = RandomizedSearchCV(

        best_model,

        param_distributions=param_dist,

        n_iter=10,

        cv=3,

        scoring="roc_auc",

        random_state=42,

        n_jobs=-1

    )

    tuner.fit(X_train,y_train)

    best_model = tuner.best_estimator_


# ==========================
# FINAL TRAIN
# ==========================

best_model.fit(X_train,y_train)


# ==========================
# EVALUATE
# ==========================

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:,1]


accuracy = accuracy_score(

    y_test,

    y_pred
)

auc = roc_auc_score(

    y_test,

    y_prob
)

f1 = f1_score(

    y_test,

    y_pred
)

report = classification_report(

    y_test,

    y_pred
)


# ==========================
# SAVE MODEL
# ==========================

joblib.dump(

    best_model,

    MODEL_PATH
)


# ==========================
# SAVE REPORT
# ==========================

with open(REPORT_PATH,"w") as f:

    f.write("HEART DISEASE TRAINING REPORT\n")

    f.write("="*60 + "\n\n")

    f.write("MODEL COMPARISON:\n")

    for name,score in scores.items():

        f.write(f"{name}: {score:.4f}\n")

    f.write("\nBEST MODEL:\n")

    f.write(best_model_name + "\n\n")

    f.write("FINAL METRICS:\n")

    f.write(f"Accuracy: {accuracy:.4f}\n")

    f.write(f"AUC: {auc:.4f}\n")

    f.write(f"F1 Score: {f1:.4f}\n\n")

    f.write("CLASSIFICATION REPORT:\n")

    f.write(report)


print("\n✅ Training Complete")
print("📄 Report saved:", REPORT_PATH)
print("💾 Model saved:", MODEL_PATH)