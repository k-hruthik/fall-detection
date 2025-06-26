import time
import numpy as np
import pandas as pd
from joblib import load
from src.preprocess import normalize_data
from src.features import extract_features


def simulate_streaming_data(df, window_size=100, step_size=50, delay=0.1):
    """Simulate real-time data streaming and fall detection."""
    model = load("model/fall_detector_svm.joblib")
    df = normalize_data(df)

    for start in range(0, len(df) - window_size, step_size):
        end = start + window_size
        window = df[['x', 'y', 'z']].iloc[start:end].values
        segment = np.expand_dims(window, axis=0)  # shape: (1, window_size, 3)
        features = extract_features(segment)
        prediction = model.predict(features)[0]

        print(f"Window {start}-{end}: Prediction => {'FALL' if prediction == 1 else 'NO FALL'}")
        time.sleep(delay)


if __name__ == "__main__":
    df = pd.read_csv("data/accelerometer.csv")
    simulate_streaming_data(df)