from preprocess import load_data, normalize_data, segment_data
from features import extract_features
from model import train_model
import numpy as np
import pandas as pd

def main():
    data_path = "data/accelerometer.csv"
    df = load_data(data_path)
    df = normalize_data(df)

    print("Label distribution:\n", df['label'].value_counts())

    X, y = segment_data(df)
    unique, counts = np.unique(y, return_counts=True)
    print("Segmented labels:", dict(zip(unique, counts)))

    if len(np.unique(y)) < 2:
        print("❌ Error: Only one class found after segmentation. Please adjust the `fall_ratio` threshold.")
        return

    # ✅ Extract time + frequency domain features
    X_features = extract_features(X)
    print("Feature shape:", X_features.shape)

    train_model(X_features, y)

if __name__ == "__main__":
    main()
