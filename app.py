import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model and encoders
try:
    model = joblib.load("model.pkl")
    le_health = joblib.load("le_health.pkl")
    le_promotion = joblib.load("le_promotion.pkl")
except FileNotFoundError as e:
    print(f"Error loading model or encoders: {e}")
    raise

def preprocess_input(data):
    try:
        data["health_condition"] = le_health.transform([data["health_condition"]])[0]
        df = pd.DataFrame([data], columns=[
            "years_of_service", "NAAS_publication_factor", "office_punctuality",
            "training_attendance", "health_condition"
        ])
        return df
    except ValueError as e:
        return f"Invalid input value: {str(e)}"
    except Exception as e:
        return f"Preprocessing error: {str(e)}"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            "name": request.form.get("name"),
            "years_of_service": int(request.form.get("years_of_service", 0)),
            "NAAS_publication_factor": float(request.form.get("NAAS_publication_factor", 0.0)),
            "office_punctuality": int(request.form.get("office_punctuality", 0)),
            "training_attendance": int(request.form.get("training_attendance", 0)),
            "health_condition": request.form.get("health_condition"),
        }

        required_fields = ["name", "health_condition"]
        missing_fields = [field for field in required_fields if not data[field]]
        if missing_fields:
            return render_template("index.html", prediction_text=f"Error: Missing fields - {', '.join(missing_fields)}")

        input_data = preprocess_input(data)
        if isinstance(input_data, str):
            return render_template("index.html", prediction_text=input_data)

        prediction = model.predict(input_data)[0]
        prediction_label = le_promotion.inverse_transform([prediction])[0]

        return render_template("index.html", prediction_text=f"Predicted Promotion Level for {data['name']}: {prediction_label}")
    except ValueError as e:
        return render_template("index.html", prediction_text=f"Invalid input: {str(e)}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)