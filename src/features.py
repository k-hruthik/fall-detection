import numpy as np
import scipy.fftpack as fft


def extract_time_features(segment):
    """Extract time-domain features from a window segment."""
    features = []
    for axis in range(segment.shape[1]):
        data = segment[:, axis]
        features.extend([
            np.mean(data),
            np.std(data),
            np.max(data),
            np.min(data),
            np.ptp(data),  # peak-to-peak
            np.sqrt(np.mean(data ** 2))  # RMS
        ])
    return features


def extract_freq_features(segment):
    """Extract frequency-domain features using FFT."""
    features = []
    for axis in range(segment.shape[1]):
        freq_data = np.abs(fft.fft(segment[:, axis]))
        features.extend([
            np.mean(freq_data),
            np.std(freq_data),
            np.argmax(freq_data[:len(freq_data)//2])  # dominant freq bin
        ])
    return features


def extract_features(segments):
    """Extract combined time and frequency domain features for all segments."""
    all_features = []
    for segment in segments:
        time_feats = extract_time_features(segment)
        freq_feats = extract_freq_features(segment)
        all_features.append(time_feats + freq_feats)
    return np.array(all_features)


if __name__ == "__main__":
    from preprocess import load_data, normalize_data, segment_data
    df = load_data("../data/accelerometer.csv")
    df = normalize_data(df)
    X, y = segment_data(df)
    features = extract_features(X)
    print(f"Extracted Features: {features.shape}")