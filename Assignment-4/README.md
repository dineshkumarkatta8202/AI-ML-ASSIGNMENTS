# Fraud Detection with MLflow

This project compares Logistic Regression, Random Forest, XGBoost, and XGBoost trained with SMOTE. It records model parameters and fraud-sensitive metrics in a local MLflow tracking store, selects a model by fraud recall and macro F1, and evaluates that model on `fraudTest.csv`.

## Data

Place the Kaggle files in `data/`:

- `fraudTrain.csv`
- `fraudTest.csv`

The script uses a reproducible sample of up to 120,000 legitimate training rows plus every fraud row to keep runtime manageable. It never resamples the validation data. The CSVs are not included in this package because they are large.

## Run

Install dependencies, then run from this directory:

```bash
python -m pip install -r requirements.txt
python src/run_assignment.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://127.0.0.1:5000` to inspect the experiment and model runs. Results are written to `outputs/`; the selected model bundle is saved in `models/`.

## Metrics

Accuracy is included for comparison, but fraud recall, macro F1, ROC-AUC, and average precision are more informative for this imbalanced classification task. Results can differ from other reports because the sampling strategy, feature set, split, and model settings affect performance. The dataset is synthetic and its metrics should not be treated as production fraud performance.
