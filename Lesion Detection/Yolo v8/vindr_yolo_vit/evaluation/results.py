import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_vit/evaluation/vit_evaluation")

TXT_FILE = BASE_DIR / "results.txt"
JSON_FILE = BASE_DIR / "metrics.json"


# ============================================================
# LOAD JSON
# ============================================================

with open(JSON_FILE, "r") as f:
    metrics = json.load(f)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 60)
print("                    ViT EVALUATION")
print("=" * 60)

print(f"\nModel:      {metrics.get('model', 'ViT-Base-Patch16-224')}")
print(f"Checkpoint: {metrics.get('checkpoint', 'N/A')}")

print("\n" + "-" * 60)
print("DATASET")
print("-" * 60)

print(f"Test images:       {metrics['test_images']}")
print(f"Positive images:   {metrics['positive_images']}")
print(f"Negative images:   {metrics['negative_images']}")

print("\n" + "-" * 60)
print("CLASSIFICATION METRICS")
print("-" * 60)

print(f"Loss:               {metrics['loss']:.4f}")
print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
print(f"PR-AUC:             {metrics['pr_auc']:.4f}")
print(f"Accuracy:           {metrics['accuracy']:.4f}")
print(f"Precision:          {metrics['precision']:.4f}")
print(f"Recall:             {metrics['recall']:.4f}")
print(f"Sensitivity:        {metrics['sensitivity']:.4f}")
print(f"Specificity:        {metrics['specificity']:.4f}")
print(f"F1-score:           {metrics['f1']:.4f}")

print("\n" + "-" * 60)
print("CONFUSION MATRIX")
print("-" * 60)

print(f"True Negatives:     {metrics['tn']}")
print(f"False Positives:    {metrics['fp']}")
print(f"False Negatives:    {metrics['fn']}")
print(f"True Positives:     {metrics['tp']}")

print("\n" + "=" * 60)