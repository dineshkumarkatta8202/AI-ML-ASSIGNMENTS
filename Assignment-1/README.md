# House Price Prediction

This assignment trains a linear regression model to estimate King County home sale prices. The CSV is the King County house sales dataset (`kc_house_data.csv`).

## Workflow

1. Load the CSV and use `price` as the value to predict.
2. Exclude the record ID and sale date; treat `zipcode` as a category.
3. Split the data into 80% training and 20% test rows with a fixed seed.
4. Impute missing values, scale numeric inputs, and one-hot encode categorical inputs in a scikit-learn pipeline.
5. Train Linear Regression and evaluate with RMSE and R².

RMSE is reported in dollars and measures the typical prediction error size. R² measures how much of the variation in sale prices the model explains; it can be below zero if predictions are worse than predicting the test-set mean.

## Run

From this directory:

```bash
python -m pip install pandas scikit-learn joblib matplotlib
python src/train.py
python src/evaluate.py
```

The trained pipeline is saved under `model/`. Metrics and the actual-versus-predicted plot are saved under `outputs/`.
