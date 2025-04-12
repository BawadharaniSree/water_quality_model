from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from flask import jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Load model and scaler
model = joblib.load('model/water_quality_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Classify based on thresholds
thresholds = {
    'pH': {'min': 6.5, 'max': 8.5},
    'Dissolved Oxygen': {'min': 5.0, 'max': 14.0},
    'Turbidity': {'min': 0.0, 'max': 5.0},
    'Chlorophyll': {'min': 0.0, 'max': 10.0},
    'Temperature': {'min': 20.0, 'max': 25.0},
    'Salinity': {'min': 0.0, 'max': 0.5}
}

def classify_quality(sample):
    for param, value in sample.items():
        if value < thresholds[param]['min'] or value > thresholds[param]['max']:
            return 'WORST'
    return 'GOOD'

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            sample = {
                'Chlorophyll': float(request.form['chlorophyll']),
                'Temperature': float(request.form['temperature']),
                'Dissolved Oxygen': float(request.form['do']),
                'pH': float(request.form['ph']),
                'Salinity': float(request.form['salinity']),
                'Turbidity': float(request.form['turbidity'])
            }

            df = pd.DataFrame([sample])
            scaled = scaler.transform(df)
            prediction = model.predict(scaled)[0]

            if prediction == 1:
                result = "✅ Normal - Water Quality is GOOD"
            else:
                result = "❌ Anomaly Detected - Water Quality is WORST"
        except Exception as e:
            result = f"Error: {e}"

    return render_template('index.html', result=result)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        sample = {
            'Chlorophyll': float(data['Chlorophyll']),
            'Temperature': float(data['Temperature']),
            'Dissolved Oxygen': float(data['Dissolved Oxygen']),
            'pH': float(data['pH']),
            'Salinity': float(data['Salinity']),
            'Turbidity': float(data['Turbidity'])
        }

        df = pd.DataFrame([sample])
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)[0]
        prediction_text = "✅ Normal - Water Quality is GOOD" if prediction == 1 else "❌ Anomaly Detected - Water Quality is WORST"
        #result = "✅ Normal - Water Quality is GOOD" if prediction == 1 else "❌ Anomaly Detected - Water Quality is WORST"
        threshold_quality = classify_quality(sample)

        return jsonify({
            'prediction': prediction_text,
            'threshold_quality': threshold_quality
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
