import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import dump
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def train_model(X, y):
    """Train and compare SVM, RandomForest, and KNN classifiers."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # -- SVM --
    print("\n=== Training SVM ===")
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
        'kernel': ['rbf']
    }
    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_weighted')
    grid.fit(X_train, y_train)
    best_svm = grid.best_estimator_
    evaluate_model(best_svm, X_test, y_test, "SVM")

    # Save SVM model
    os.makedirs("../model", exist_ok=True)
    dump(best_svm, "../model/fall_detector_svm.joblib")

    # -- Random Forest --
    print("\n=== Training Random Forest ===")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    evaluate_model(rf, X_test, y_test, "Random Forest")

    # -- k-NN --
    print("\n=== Training k-NN ===")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    evaluate_model(knn, X_test, y_test, "k-NN")


def evaluate_model(model, X_test, y_test, name):
    """Evaluate model and print performance metrics."""
    y_pred = model.predict(X_test)
    print(f"\n[{name}] Accuracy:", accuracy_score(y_test, y_pred))
    print(f"[{name}] Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=1))
    print(f"[{name}] Recall:", recall_score(y_test, y_pred, average='weighted'))
    print(f"[{name}] F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=1))
    plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {name}")


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()
