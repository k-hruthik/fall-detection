# Directory: fall-detection/README.md

# Fall Detection for Elderly using Accelerometer Data

This project implements a robust fall detection system using accelerometer data and machine learning (SVM classifier). It is designed for real-time prediction and includes feature extraction, model tuning, performance evaluation, and a REST API for inference.

## 🚀 Features

- ✅ Data normalization & segmentation
- ✅ Time & frequency domain feature extraction
- ✅ SVM with GridSearchCV for hyperparameter tuning
- ✅ Evaluation metrics & confusion matrix
- ✅ Real-time fall detection simulation
- ✅ Model comparison (SVM, Random Forest, k-NN)
- ✅ Lightweight Flask API for inference
- ✅ EDA Jupyter notebook
- ✅ Modular Python codebase with good structure

## 📁 Project Structure

```
fall-detection/
├── data/                    # Put accelerometer.csv here
├── model/                   # Trained models saved here
├── notebooks/
│   └── EDA.ipynb            # Data visualization and exploration
├── src/
│   ├── preprocess.py        # Data loading, normalization, segmentation
│   ├── features.py          # Feature extraction
│   ├── model.py             # SVM, RF, kNN training & evaluation
│   ├── realtime_simulation.py  # Live fall detection simulation
│   ├── main.py              # Pipeline entry point
│   └── api.py               # Flask API for prediction
├── requirements.txt
└── README.md
```

## 🧠 How It Works

1. **Preprocessing**: Normalize accelerometer data (x, y, z), then segment using a sliding window.
2. **Feature Engineering**: Extract statistical (mean, std, RMS) and frequency features (FFT, spectral stats).
3. **Modeling**: Compare SVM (GridSearchCV), Random Forest, and k-NN classifiers.
4. **Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix.
5. **Simulation**: Emulates real-time fall detection from continuous data.
6. **API**: Accepts live accelerometer input via POST and returns prediction.
7. **EDA**: Visualizes trends, feature correlations, and label distribution.

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/fall-detection.git
cd fall-detection
pip install -r requirements.txt
```

## 📈 Run the Full Pipeline

```bash
python src/main.py
```

## 🧪 Run Real-Time Simulation

```bash
python src/realtime_simulation.py
```

## 🔌 Run the Flask API

```bash
python src/api.py
```

Then use Postman or PowerShell:
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method POST -ContentType "application/json" -Body '[{"x":0.1,"y":0.2,"z":9.8}, ...]'
```

## 📊 Run EDA Notebook

```bash
jupyter notebook notebooks/EDA.ipynb
```

## 📬 Submission

Upload this project to GitHub and submit the link to [DevifyX Submission Form](https://forms.gle/UEUafAtyPUEm2ZjbA)

## 👨‍💻 Author

**K Hruthik Varma**  
Machine Learning Enthusiast  
[hruthikthebest@gmail.com](mailto:hruthikthebest@gmail.com)
