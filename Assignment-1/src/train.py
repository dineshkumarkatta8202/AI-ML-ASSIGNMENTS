from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "kc_house_data.csv"
MODEL_PATH = ROOT / "model" / "house_price_pipeline.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if "price" not in df:
        raise ValueError("Dataset must contain a 'price' target column")

    # The identifier and sale date are excluded; zipcode is treated as a category.
    X = df.drop(columns=["price", "id", "date"], errors="ignore")
    if "zipcode" in X:
        X["zipcode"] = X["zipcode"].astype(str)
    y = df["price"]

    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "test_rows": int(len(y_test)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
        "features": list(X.columns),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
