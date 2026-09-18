"""
modelling.py (untuk MLProject / Workflow CI)

Script training yang dipanggil oleh MLflow Project (`mlflow run`) di dalam
GitHub Actions workflow untuk melakukan retraining model secara otomatis
setiap kali trigger terpantik (push / manual dispatch).

Menerima parameter `data_path` sesuai definisi entry point di file `MLProject`.
"""

import argparse

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def load_data(data_path: str):
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="namadataset_preprocessing",
        help="Path ke folder berisi train.csv dan test.csv hasil preprocessing.",
    )
    args = parser.parse_args()

    # Autolog cukup untuk kebutuhan CI retraining (fokus kriteria ini ada di
    # otomatisasi pipeline & docker image, bukan di eksperimen model itu sendiri).
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_data(args.data_path)

    with mlflow.start_run(run_name="ci_retrain_random_forest"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"[CI Retrain] Akurasi: {acc:.4f} | F1-score: {f1:.4f}")


if __name__ == "__main__":
    main()
