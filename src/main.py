from preprocess import load_data, normalize_data, segment_data
from features import extract_features
from model import train_model
import numpy as np
import pandas as pd


def main():
    data_path = "data/accelerometer.csv"
    df = load_data(data_path)
    df = normalize_data(df)

    # Check class balance
    label_counts = df['label'].value_counts()
    print("Label distribution:\n", label_counts)

    if len(label_counts) < 2:
        raise ValueError("❌ Dataset must contain at least two classes (fall and no fall) to train the model.")

    X_seg, y = segment_data(df)

    # Debug segmented label distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Segmented labels:", dict(zip(unique, counts)))

    # Optional: save to CSV to inspect manually
    pd.Series(y).to_csv("segmented_labels.csv", index=False)

    X = extract_features(X_seg)
    train_model(X, y)


if __name__ == "__main__":
    main()