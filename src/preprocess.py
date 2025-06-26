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