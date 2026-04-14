from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder="static", static_url_path="")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")
pipeline = None

def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model load error: {e}")

load_model()

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        age     = float(data["age"])
        gender  = int(data["gender"])      # 1=female, 2=male
        height  = float(data["height"])
        weight  = float(data["weight"])
        ap_hi   = float(data["ap_hi"])
        ap_lo   = float(data["ap_lo"])
        chol    = int(data["cholesterol"]) # 1/2/3
        gluc    = int(data["gluc"])        # 1/2/3
        smoke   = int(data["smoke"])       # 0/1
        alco    = int(data["alco"])        # 0/1
        active  = int(data["active"])      # 0/1

        # Engineer features
        bmi    = weight / (height / 100) ** 2
        pp     = ap_hi - ap_lo
        map_   = ap_lo + (ap_hi - ap_lo) / 3
        bmi_age   = bmi * age
        chol_bmi  = chol * bmi
        bp_ratio  = ap_hi / ap_lo if ap_lo != 0 else 1.5

        import pandas as pd
        features = pd.DataFrame([{
            "age_years": age,
            "gender":    gender,
            "height":    height,
            "weight":    weight,
            "ap_hi":     ap_hi,
            "ap_lo":     ap_lo,
            "cholesterol": chol,
            "gluc":      gluc,
            "smoke":     smoke,
            "alco":      alco,
            "active":    active,
            "bmi":       bmi,
            "pulse_pressure":        pp,
            "mean_arterial_pressure": map_,
            "bmi_age":   bmi_age,
            "chol_bmi":  chol_bmi,
            "bp_ratio":  bp_ratio,
        }])

        prediction = int(pipeline.predict(features)[0])
        proba      = float(pipeline.predict_proba(features)[0][1])

        return jsonify({
            "prediction": prediction,
            "probability": round(proba * 100, 1),
            "bmi": round(bmi, 1),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
