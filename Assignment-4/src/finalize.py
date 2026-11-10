from pathlib import Path
import json
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "mlflow.db"
OUT = ROOT / "outputs"
mlflow.set_tracking_uri(f"sqlite:///{DB}")
mlflow.set_registry_uri(f"sqlite:///{DB}")
client = MlflowClient()
results = json.loads((OUT / "model_comparison.json").read_text())
best = next(r for r in results["models"] if r["model"] == results["best_model"])
model_uri = f"runs:/{best['run_id']}/model"
registered = mlflow.register_model(model_uri, "fraud-detection-model")
client.set_registered_model_alias("fraud-detection-model", "challenger", registered.version)
client.set_registered_model_alias("fraud-detection-model", "champion", registered.version)
results["registered_model"] = {
    "name": "fraud-detection-model",
    "version": str(registered.version),
    "aliases": ["challenger", "champion"],
}
(OUT / "model_comparison.json").write_text(json.dumps(results, indent=2))

names = [r["model"] for r in results["models"]]
recall = [r["recall_class_1"] for r in results["models"]]
macro = [r["f1_score_macro"] for r in results["models"]]
fig, ax = plt.subplots(figsize=(9, 5))
pos = list(range(len(names)))
ax.bar([x - 0.18 for x in pos], recall, width=0.36, label="Fraud recall")
ax.bar([x + 0.18 for x in pos], macro, width=0.36, label="Macro F1")
ax.set_xticks(pos, names, rotation=12)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Validation performance by model")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "validation_model_comparison.png", dpi=160)
plt.close(fig)

holdout = json.loads((OUT / "holdout_metrics.json").read_text())
report = f'''# Fraud Detection Results\n\nTraining source rows: {results['dataset_rows']:,}. For runtime, training used {results['sampled_rows']:,} rows, including every fraud case and a fixed-seed sample of legitimate transactions. Validation rows: {results['validation_rows']:,}.\n\nModels were selected by fraud recall, then macro F1. The selected model was **{results['best_model']}**.\n\n## Validation model comparison\n\n| Model | Accuracy | Fraud recall | Fraud precision | Macro F1 | ROC-AUC | Average precision |\n|---|---:|---:|---:|---:|---:|---:|\n'''
for r in results['models']:
    report += f"| {r['model']} | {r['accuracy']:.4f} | {r['recall_class_1']:.4f} | {r['precision_class_1']:.4f} | {r['f1_score_macro']:.4f} | {r['roc_auc']:.4f} | {r['average_precision']:.4f} |\n"
report += f"\n## Separate fraudTest.csv evaluation\n\nAccuracy: {holdout['accuracy']:.4f}; fraud recall: {holdout['recall_class_1']:.4f}; fraud precision: {holdout['precision_class_1']:.4f}; macro F1: {holdout['f1_score_macro']:.4f}; ROC-AUC: {holdout['roc_auc']:.4f}; average precision: {holdout['average_precision']:.4f}.\n\nThe dataset is simulated. These metrics depend on the feature set, sample, split, and model settings and should not be treated as production fraud performance.\n"
(OUT / "results_report.md").write_text(report)
print("Registered", registered.name, "version", registered.version, "with champion and challenger aliases")
print("Wrote chart and results report")
