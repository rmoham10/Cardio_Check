# 🫀 CardioCheck — Heart Risk Predictor

A Gen Z-aesthetic multi-page web app that predicts cardiovascular risk using a
GradientBoosting ML model trained on 68,000+ patient records.

---

## 📁 Project Structure

```
cardiorisk/
├── app.py                    # Flask backend + /predict API
├── best_model.pkl            # Trained GradientBoosting pipeline
├── requirements.txt          # Python dependencies
├── render.yaml               # Render.com deployment config
├── README.md
└── static/
    └── index.html            # Full multi-page frontend (Home, Checker, About, Contact)
```

---

## ⚙️ Local Setup (VS Code)

### 1. Open the folder in VS Code
```
File → Open Folder → cardiorisk/
```

### 2. Create a virtual environment
Open the integrated terminal (`Ctrl + `` ` ``):
```bash
python -m venv .venv
```

Activate it:
```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

VS Code will detect the `.venv` and prompt you to select it as the Python interpreter.
Click **Yes**, or press `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `.venv`.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask app
```bash
python app.py
```

You should see:
```
✅ Model loaded successfully
 * Running on http://127.0.0.1:5000
```

### 5. Open in browser
Go to: **http://localhost:5000**

---

## 🌐 Deploy to Render.com

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "CardioCheck initial commit"
git remote add origin https://github.com/YOUR_USERNAME/cardiocheck.git
git branch -M main
git push -u origin main
```

> ⚠️ Make sure `best_model.pkl` is committed — it's needed at runtime.

### 2. Create a Render Web Service
1. Go to **https://render.com** → New → Web Service
2. Connect your GitHub repo
3. Set these fields:

| Field | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT` |

4. Click **Deploy** — live in ~5 minutes at `https://cardiocheck.onrender.com`

---

## 🧠 How the Prediction Works

```
User fills the form (11 inputs)
        ↓
JavaScript sends POST /predict with JSON
        ↓
Flask computes 6 engineered features:
  bmi, pulse_pressure, MAP, bmi_age, chol_bmi, bp_ratio
        ↓
Builds a 17-feature DataFrame
        ↓
best_model.pkl pipeline:
  KNNImputer → StandardScaler → SelectFromModel → SMOTE → GradientBoosting
        ↓
Returns: prediction (0/1) + probability (%)
        ↓
Frontend renders result + personalised advice
```

---

## 🎛️ User Inputs & Defaults

| Field | Type | Default if blank |
|---|---|---|
| Age | Number | required |
| Gender | Dropdown | required |
| Height (cm) | Number | required |
| Weight (kg) | Number | required |
| Systolic BP | Number | 120 |
| Diastolic BP | Number | 80 |
| Cholesterol | Dropdown | Normal (1) |
| Glucose | Dropdown | Normal (1) |
| Smoking | Toggle | No (0) |
| Alcohol | Toggle | No (0) |
| Physical Activity | Toggle | Yes (1) |
