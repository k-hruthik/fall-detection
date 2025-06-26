from flask import Flask, request, jsonify
import numpy as np
from joblib import load
from preprocess import normalize_data
from features import extract_features
import pandas as pd

app = Flask(__name__)
model = load("../model/fall_detector_svm.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Expecting a list of dicts with 'x', 'y', 'z'
    if not data or not isinstance(data, list):
        return jsonify({"error": "Input must be a list of dicts with keys 'x', 'y', 'z'"}), 400

    df = pd.DataFrame(data)
    if not all(axis in df.columns for axis in ['x', 'y', 'z']):
        return jsonify({"error": "Missing required accelerometer keys 'x', 'y', 'z'"}), 400

    df = normalize_data(df)
    segment = df[['x', 'y', 'z']].values
    segment = np.expand_dims(segment, axis=0)
    features = extract_features(segment)
    prediction = model.predict(features)[0]

    return jsonify({"prediction": int(prediction), "label": "FALL" if prediction == 1 else "NO FALL"})


if __name__ == "__main__":
    app.run(debug=True)
