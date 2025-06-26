# Directory: fall-detection/src/preprocess.py

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def load_data(file_path):
    """Load raw accelerometer data from CSV file."""
    return pd.read_csv(file_path)


def normalize_data(df, columns=['x', 'y', 'z']):
    """Apply z-score normalization to x, y, z columns."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def segment_data(df, window_size=100, step_size=50):
    """Segment data using sliding window."""
    segments = []
    labels = []
    for start in range(0, len(df) - window_size, step_size):
        end = start + window_size
        segment = df[['x', 'y', 'z']].iloc[start:end].values
        window_labels = df['label'].iloc[start:end]
        fall_ratio = window_labels.sum() / len(window_labels)
        label = 1 if fall_ratio >= 0.4 else 0
        segments.append(segment)
        labels.append(label)
    segments, labels = shuffle(segments, labels, random_state=42)
    return np.array(segments), np.array(labels)


# Directory: fall-detection/src/realtime_simulation.py

import os
import pandas as pd
import numpy as np
from joblib import load
from preprocess import normalize_data
from features import extract_features


def simulate_streaming_data(df, window_size=100, step_size=50):
    df = normalize_data(df)

    # Try loading RF model, fallback to SVM if unavailable
    model_path = "model/fall_detector_rf.joblib"
    if not os.path.exists(model_path):
        print("⚠️  Random Forest model not found. Using fallback SVM model.")
        model_path = "model/fall_detector_svm.joblib"

    model = load(model_path)

    fall_count = 0
    no_fall_count = 0

    for i in range(0, len(df) - window_size, step_size):
        window = df.iloc[i:i + window_size][['x', 'y', 'z']].values
        segment = np.expand_dims(window, axis=0)
        features = extract_features(segment)
        prediction = model.predict(features)[0]

        if prediction == 1:
            fall_count += 1
            status = "FALL"
        else:
            no_fall_count += 1
            status = "NO FALL"

        print(f"Window {i}-{i + window_size}: Prediction => {status}")

    print(f"\nTotal FALL predictions: {fall_count}")
    print(f"Total NO FALL predictions: {no_fall_count}")


if __name__ == "__main__":
    df = pd.read_csv("data/accelerometer.csv")
    simulate_streaming_data(df)
