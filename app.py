from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
load_dotenv()
from flask_cors import CORS
import joblib
import pandas as pd
import os
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ─────────────────────────────────────────────
# EMAIL CONFIG (Render ENV VARS)
# ─────────────────────────────────────────────
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
TO_EMAIL = os.environ.get("TO_EMAIL")

# ─────────────────────────────────────────────
# LOAD MODEL BUNDLE
# ─────────────────────────────────────────────
BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "cardio_bundle_v7.pkl")

bundle = joblib.load(BUNDLE_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
THRESHOLD = bundle["threshold"]

print("✅ Model loaded")

# ─────────────────────────────────────────────
# FRONTEND ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "threshold": float(THRESHOLD)
    })


# ─────────────────────────────────────────────
# PREDICT API (ML MODEL)
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        age_years = float(data.get("age", 50))
        gender = int(data.get("gender", 1))
        height = float(data.get("height", 165))
        weight = float(data.get("weight", 70))
        ap_hi = float(data.get("ap_hi", 120))
        ap_lo = float(data.get("ap_lo", 80))
        cholesterol = int(data.get("cholesterol", 1))
        gluc = int(data.get("gluc", 1))
        smoke = int(data.get("smoke", 0))
        alco = int(data.get("alco", 0))
        active = int(data.get("active", 1))

        # Feature engineering
        bmi = weight / (height / 100) ** 2
        pulse_pressure = ap_hi - ap_lo
        map_val = ap_lo + pulse_pressure / 3
        age_bp = age_years * ap_hi
        bmi_age = bmi * age_years

        features = pd.DataFrame([{
            "age_years": age_years,
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active,
            "bmi": bmi,
            "pulse_pressure": pulse_pressure,
            "map": map_val,
            "age_bp": age_bp,
            "bmi_age": bmi_age,
        }])[FEATURES]

        proba = float(model.predict_proba(features)[0][1])
        prediction = int(proba >= THRESHOLD)

        if proba < 0.35:
            risk = "Low Risk"
        elif proba < 0.55:
            risk = "Moderate Risk"
        elif proba < 0.70:
            risk = "Elevated Risk"
        else:
            risk = "High Risk"

        return jsonify({
            "prediction": prediction,
            "probability": round(proba * 100, 1),
            "bmi": round(bmi, 1),
            "risk_level": risk,
            "threshold": float(THRESHOLD)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────────
# CONTACT FORM (EMAIL SENDING)
# ─────────────────────────────────────────────
@app.route("/api/contact", methods=["POST"])
def contact():
    try:
        data = request.get_json()

        name = data.get("name")
        email = data.get("email")
        subject = data.get("subject", "General Question")
        msg = data.get("msg")

        if not name or not email or not msg:
            return jsonify({"success": False, "error": "Missing fields"}), 400

        # HTML EMAIL (professional format)
        html_body = f"""
        <html>
          <body style="font-family: Arial; background:#f4f4f4; padding:20px;">
            <div style="max-width:600px; margin:auto; background:white; padding:20px; border-radius:10px;">
              
              <h2 style="color:#e63946;"> New Query Submission</h2>

              <p><b>Name:</b> {name}</p>
              <p><b>Email:</b> {email}</p>
              <p><b>Subject:</b> {subject}</p>

              <hr>

              <h3>Message:</h3>
              <p style="white-space: pre-line;">{msg}</p>

              <hr>

              <p style="font-size:12px; color:gray;">
                Sent from Cardio App
              </p>

            </div>
          </body>
        </html>
        """

        message = MIMEText(html_body, "html")
        message["Subject"] = f"[CardioCheck] {subject} - {name}"
        message["From"] = EMAIL_USER
        message["To"] = TO_EMAIL
        message["Reply-To"] = email

        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, TO_EMAIL, message.as_string())
        server.quit()

        return jsonify({"success": True})

    except Exception as e:
        print("Email error:", e)
        return jsonify({"success": False}), 500


# ─────────────────────────────────────────────
# RUN (RENDER READY)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True)