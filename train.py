"""
Training script for the Telco Customer Churn classification task.

Loads the raw dataset, cleans and encodes it, trains five classification
models, evaluates them on a held-out test split, and persists the models
together with the fitted preprocessing objects so app.py can reproduce the
same transformation at inference time.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "WA_FnUseC_TelcoCustomerChurn.csv"
MODEL_DIR = "model"
TEST_DATA_PATH = "test_data.csv"
METRICS_PATH = "metrics.csv"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[ID_COLUMN])

    # TotalCharges is read as object because a handful of new-customer rows
    # (tenure == 0) contain blank strings instead of numeric values.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # SeniorCitizen arrives as 0/1 integers; cast to string so it goes
    # through the same categorical label-encoding path as the rest.
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    return df


def encode_features(df, encoders=None, fit=True):
    df = df.copy()
    if encoders is None:
        encoders = {}

    for column in CATEGORICAL_COLUMNS:
        if fit:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
            encoders[column] = encoder
        else:
            encoder = encoders[column]
            df[column] = encoder.transform(df[column])

    return df, encoders


def encode_target(series, fit=True, encoder=None):
    if fit:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(series)
        return encoded, encoder
    return encoder.transform(series), encoder


def build_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_and_clean(DATA_PATH)

    # Raw (pre-encoding) split is what gets saved as test_data.csv, so the
    # Streamlit app can accept the same kind of file a user would export
    # from the source dataset and run it through the saved pipeline.
    train_raw, test_raw = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    test_raw.to_csv(TEST_DATA_PATH, index=False)

    X_train_raw = train_raw.drop(columns=[TARGET_COLUMN])
    X_test_raw = test_raw.drop(columns=[TARGET_COLUMN])

    X_train_enc, encoders = encode_features(X_train_raw, fit=True)
    X_test_enc, _ = encode_features(X_test_raw, encoders=encoders, fit=False)

    y_train, target_encoder = encode_target(train_raw[TARGET_COLUMN], fit=True)
    y_test, _ = encode_target(test_raw[TARGET_COLUMN], fit=False, encoder=target_encoder)

    scaler = StandardScaler()
    X_train_enc[NUMERIC_COLUMNS] = scaler.fit_transform(X_train_enc[NUMERIC_COLUMNS])
    X_test_enc[NUMERIC_COLUMNS] = scaler.transform(X_test_enc[NUMERIC_COLUMNS])

    feature_columns = X_train_enc.columns.tolist()

    models = build_models()
    results = []

    for name, model in models.items():
        model.fit(X_train_enc[feature_columns], y_train)
        metrics = evaluate(model, X_test_enc[feature_columns], y_test)
        metrics["Model"] = name
        results.append(metrics)
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
        print(f"{name}: {metrics}")

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.joblib"))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.joblib"))
    joblib.dump(NUMERIC_COLUMNS, os.path.join(MODEL_DIR, "numeric_columns.joblib"))

    metrics_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    metrics_df.to_csv(METRICS_PATH, index=False)
    print("\nSaved models, preprocessing objects, test_data.csv and metrics.csv")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
