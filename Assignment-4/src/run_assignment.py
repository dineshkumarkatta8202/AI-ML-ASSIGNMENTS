from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
TRACKING_DB = ROOT / "mlflow.db"
SEED = 42
MAX_LEGIT_ROWS = 120_000


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    timestamp = pd.to_datetime(data["trans_date_trans_time"], errors="coerce")
    dob = pd.to_datetime(data["dob"], errors="coerce")
    data["hour"] = timestamp.dt.hour
    data["day_of_week"] = timestamp.dt.dayofweek
    data["month"] = timestamp.dt.month
    data["age"] = ((timestamp - dob).dt.days / 365.25).clip(lower=0, upper=110)
    data["gender_code"] = data["gender"].map({"F": 0, "M": 1}).fillna(-1)
    data["category_code"] = pd.Categorical(data["category"]).codes
    # Keep numeric transaction and location signals; discard identifiers and direct identity fields.
    columns = ["amt", "lat", "long", "city_pop", "unix_time", "merch_lat",
               "merch_long", "hour", "day_of_week", "month", "age",
               "gender_code", "category_code"]
    X = data[columns].replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median(numeric_only=True)).astype("float32")


def metrics_for(y_true, pred, prob) -> dict:
    report = classification_report(y_true, pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "recall_class_0": float(report["0"]["recall"]),
        "recall_class_1": float(report["1"]["recall"]),
        "precision_class_1": float(precision_score(y_true, pred, zero_division=0)),
        "f1_score_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
    }


def main() -> None:
    train_path = DATA_DIR / "fraudTrain.csv"
    test_path = DATA_DIR / "fraudTest.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Place fraudTrain.csv in {DATA_DIR}")
    if not test_path.exists():
        raise FileNotFoundError(f"Place fraudTest.csv in {DATA_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(train_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    # Retain all fraud rows and a reproducible random sample of legitimate rows.
    fraud = df[df["is_fraud"] == 1]
    legitimate = df[df["is_fraud"] == 0].sample(n=min(MAX_LEGIT_ROWS, (df["is_fraud"] == 0).sum()), random_state=SEED)
    sampled = pd.concat([fraud, legitimate], ignore_index=True).sample(frac=1, random_state=SEED)
    X = prepare_features(sampled)
    y = sampled["is_fraud"].astype("int8")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    smote = SMOTE(random_state=SEED, sampling_strategy=0.5, k_neighbors=5)
    X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
    mlflow.set_registry_uri(f"sqlite:///{TRACKING_DB}")
    mlflow.set_experiment("Fraud Detection Model Comparison")

    candidates = [
        ("Logistic Regression", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED), X_train_scaled, y_train, X_val_scaled),
        ("Random Forest", RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), X_train, y_train, X_val),
        ("XGBoost", XGBClassifier(n_estimators=140, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", n_jobs=4, random_state=SEED), X_train, y_train, X_val),
        ("XGBoost + SMOTE", XGBClassifier(n_estimators=140, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", n_jobs=4, random_state=SEED), X_smote, y_smote, X_val_scaled),
    ]

    summary = []
    client = MlflowClient()
    for name, model, fit_x, fit_y, predict_x in candidates:
        with mlflow.start_run(run_name=name) as run:
            model.fit(fit_x, fit_y)
            pred = model.predict(predict_x)
            prob = model.predict_proba(predict_x)[:, 1]
            metrics = metrics_for(y_val, pred, prob)
            mlflow.log_param("training_rows", len(fit_y))
            mlflow.log_param("validation_rows", len(y_val))
            mlflow.log_param("resampling", "SMOTE" if "SMOTE" in name else "none")
            mlflow.log_param("sampling_seed", SEED)
            mlflow.log_metrics(metrics)
            if "SMOTE" in name:
                mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
            elif name == "Logistic Regression":
                mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
            else:
                mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
            summary.append({"model": name, "run_id": run.info.run_id, **metrics,
                            "confusion_matrix": confusion_matrix(y_val, pred).tolist()})
            print(name, json.dumps(metrics))

    # Choose by fraud recall, then macro F1, and save the corresponding candidate model.
    best = sorted(summary, key=lambda r: (r["recall_class_1"], r["f1_score_macro"]), reverse=True)[0]
    runs = mlflow.search_runs(filter_string=f"attributes.run_id = '{best['run_id']}'")
    best_model = mlflow.sklearn.load_model(f"runs:/{best['run_id']}/model")
    joblib.dump({"model": best_model, "scaler": scaler, "feature_columns": list(X.columns)}, MODEL_DIR / "fraud_model.joblib")
    results = {"dataset_rows": int(len(df)), "sampled_rows": int(len(sampled)),
               "training_rows": int(len(X_train)), "validation_rows": int(len(X_val)),
               "best_model": best["model"], "selection_rule": "highest fraud recall, then macro F1",
               "models": summary}
    (OUTPUT_DIR / "model_comparison.json").write_text(json.dumps(results, indent=2))

    # Final holdout check using fraudTest.csv, with same feature preparation.
    holdout = pd.read_csv(test_path)
    holdout = holdout.drop(columns=["Unnamed: 0"], errors="ignore")
    X_holdout = prepare_features(holdout)
    X_holdout = X_holdout.reindex(columns=X.columns, fill_value=0)
    best_model_name = best["model"]
    best_index = [x[0] for x in candidates].index(best_model_name)
    if best_model_name in ("Logistic Regression", "XGBoost + SMOTE"):
        holdout_input = scaler.transform(X_holdout)
    else:
        holdout_input = X_holdout
    best_pred = best_model.predict(holdout_input)
    best_prob = best_model.predict_proba(holdout_input)[:, 1]
    holdout_metrics = metrics_for(holdout["is_fraud"], best_pred, best_prob)
    (OUTPUT_DIR / "holdout_metrics.json").write_text(json.dumps(holdout_metrics, indent=2))
    (OUTPUT_DIR / "holdout_classification_report.json").write_text(json.dumps(
        classification_report(holdout["is_fraud"], best_pred, output_dict=True, zero_division=0), indent=2))
    print("Best model:", best_model_name)
    print("FraudTest metrics:", json.dumps(holdout_metrics, indent=2))
    print("MLflow tracking database:", TRACKING_DB)


if __name__ == "__main__":
    main()
