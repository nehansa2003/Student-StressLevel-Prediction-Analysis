from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load("student_stress_rf_model.pkl")
scaler = joblib.load("stress_scaler.pkl")

def stress_class(level):
    if level <= 0.25:
        return "Healthy"
    elif level == 0.50:
        return "Moderate"
    else:
        return "High"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    sleep_quality = float(data["sleep_quality"])
    headaches = float(data["headaches"])
    academic_performance = float(data["academic_performance"])
    study_load = float(data["study_load"])
    extracurricular = float(data["extracurricular"])

    # Feature engineering
    study_efficiency = academic_performance / (study_load + 1)
    sleep_stress_risk = headaches / (sleep_quality + 1)
    study_burden_index = study_load * (6 - academic_performance)
    study_load_sleep_balance = study_load / (sleep_quality + 1)

    input_df = pd.DataFrame([{
        "Sleep_Quality": sleep_quality,
        "Headaches": headaches,
        "Academic_Performance": academic_performance,
        "Study_Load": study_load,
        "Extracurricular": extracurricular,
        "Study_Efficiency": study_efficiency,
        "Sleep_Stress_Risk": sleep_stress_risk,
        "Study_Burden_Index": study_burden_index,
        "Study_load_sleep_balance": study_load_sleep_balance
    }])

    input_scaled = scaler.transform(input_df)
    stress_value = model.predict(input_scaled)[0]
    stress_label = stress_class(stress_value)

    return jsonify({
        "stress_value": round(stress_value, 3),
        "stress_level": stress_label
    })

if __name__ == "__main__":
    app.run(debug=True)