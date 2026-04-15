# 🫀 CardioCheck — AI Heart Risk Predictor

CardioCheck is a full-stack machine learning web application that predicts cardiovascular disease risk in under 60 seconds using user-provided health and lifestyle data.

Built with a Flask backend + JavaScript frontend, the app uses a trained Gradient Boosting / XGBoost-based pipeline on 68,000+ patient records to deliver real-time predictions and personalized health guidance.

-Live_Link: https://cardio-check-x9b1.onrender.com/

---

## 🚀 Live Features

- ⚡ Instant Risk Prediction
- 🧠 ML Model with ~0.80 ROC-AUC
- 📊 Engineered Health Metrics (BMI, MAP, etc.)
- 💊 Personalized Diet & Lifestyle Advice
- 🔒 100% Private (No data stored)
- 📱 Fully Responsive — No app needed

---

## 🛠️ Tech Stack

### 🖥️ Frontend
- HTML5
- CSS3 (Custom UI styling)
- JavaScript (Vanilla JS)

### ⚙️ Backend
- Python
- Flask (REST API)

---

## 🧠 Model Details

- Dataset: Kaggle Cardiovascular Disease Dataset (~68k records)
- Model: Gradient Boosting / XGBoost pipeline

### Evaluation Metrics
- Accuracy: ~70%
- ROC-AUC: ~0.80
- Recall: ~0.84
- Precision: ~0.70

---

## 📥 Input Features

- Age  
- Gender  
- Height  
- Weight  
- Systolic BP  
- Diastolic BP  
- Cholesterol  
- Glucose  
- Smoking  
- Alcohol Intake  
- Physical Activity  

---

## 🔬 Feature Engineering

- BMI = weight / (height/100)²  
- Pulse Pressure = systolic − diastolic  
- MAP (Mean Arterial Pressure)  
- Age × BP  
- BMI × Age  

---

## ⚙️ Setup

```bash
git clone https://github.com/your-username/cardiocheck.git
cd cardiocheck
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

App runs at: http://localhost:3000

---

## 🔌 API

POST /predict

Example:
{
  "age_years": 45,
  "gender": 2,
  "height": 170,
  "weight": 75,
  "ap_hi": 130,
  "ap_lo": 85,
  "cholesterol": 2,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1
}

---

## ⚠️ Disclaimer

This is an educational project and not a medical tool. Always consult a doctor.

---

## 👨‍💻 Author

Riyaz Mohammad
