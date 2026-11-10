from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "kc_house_data.csv"
MODEL_PATH = ROOT / "model" / "house_price_pipeline.joblib"
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["price", "id", "date"], errors="ignore")
    if "zipcode" in X:
        X["zipcode"] = X["zipcode"].astype(str)
    y = df["price"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load(MODEL_PATH)
    predicted = model.predict(X_test)
    results = {
        "rmse": float(mean_squared_error(y_test, predicted) ** 0.5),
        "r2": float(r2_score(y_test, predicted)),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "evaluation.json").write_text(json.dumps(results, indent=2))

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predicted, alpha=0.35, s=12)
    low = min(y_test.min(), predicted.min())
    high = max(y_test.max(), predicted.max())
    plt.plot([low, high], [low, high], "r--", label="Perfect prediction")
    plt.xlabel("Actual price ($)")
    plt.ylabel("Predicted price ($)")
    plt.title("Actual vs. predicted house prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    print(json.dumps(results, indent=2))
    print(f"Saved chart to {OUTPUT_DIR / 'actual_vs_predicted.png'}")


if __name__ == "__main__":
    main()
