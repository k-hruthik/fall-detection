import pandas as pd
import numpy as np

np.random.seed(42)
num_samples = 5000
fall_ratio = 0.4  # 40% falls

# Simulate accelerometer readings
x = np.random.normal(0, 1, num_samples)
y = np.random.normal(0, 1, num_samples)
z = np.random.normal(9.8, 1, num_samples)

# Assign fall labels
labels = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])

# Add extreme motion for fall samples
fall_indices = np.where(labels == 1)[0]
x[fall_indices] += np.random.normal(0, 4, len(fall_indices))
y[fall_indices] += np.random.normal(0, 4, len(fall_indices))
z[fall_indices] += np.random.normal(0, 4, len(fall_indices))

# Save DataFrame
df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'label': labels})
df.to_csv("data/accelerometer.csv", index=False)
print(df['label'].value_counts())
