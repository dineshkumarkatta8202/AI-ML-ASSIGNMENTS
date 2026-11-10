# Fraud Detection Results

Training source rows: 1,296,675. For runtime, training used 127,506 rows, including every fraud case and a fixed-seed sample of legitimate transactions. Validation rows: 38,252.

Models were selected by fraud recall, then macro F1. The selected model was **Random Forest**.

## Validation model comparison

| Model | Accuracy | Fraud recall | Fraud precision | Macro F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9365 | 0.7518 | 0.4753 | 0.7740 | 0.8589 | 0.5912 |
| Random Forest | 0.9885 | 0.9174 | 0.8901 | 0.9487 | 0.9943 | 0.9626 |
| XGBoost | 0.9928 | 0.9134 | 0.9630 | 0.9669 | 0.9979 | 0.9814 |
| XGBoost + SMOTE | 0.9901 | 0.9085 | 0.9229 | 0.9552 | 0.9964 | 0.9720 |

## Separate fraudTest.csv evaluation

Accuracy: 0.9940; fraud recall: 0.8713; fraud precision: 0.3799; macro F1: 0.7630; ROC-AUC: 0.9905; average precision: 0.7850.

The dataset is simulated. These metrics depend on the feature set, sample, split, and model settings and should not be treated as production fraud performance.
