#!/usr/bin/env python3
"""ViT size ablation (positives regime): ViT-Base vs ViT-Small, with Swin-Tiny as reference.

Same paired study-level bootstrap as bootstrap_ci.py (same METRICS, n_boot 2000, seed 0),
on the predictions of Test/outputs/{vit_positives, vit_base_positives, swin_positives}.

Outputs (comparison/ablation/):
    ablation_ci.csv          estimate + 95% CI per model and metric
    ablation_paired_diff.csv ViT-Base - ViT-Small and Swin-Tiny - ViT-Base: delta, 95% CI, p
    ablation_froc.png        FROC (IoU 0.30) of the three models

Run eval_vit_base_positives.py, the positives evaluations and bootstrap_ci.py first.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # comparison/, for bootstrap_ci.py
sys.path.insert(0, str(HERE.parents[2]))       # Test/, for compare_all.py
from bootstrap_ci import METRICS, detection_tables, load_test_studies, metric_values  # noqa: E402

MODELS = {"vit_positives": "ViT-S · positives", "vit_base_positives": "ViT-B · positives (ablation)",
          "swin_positives": "Swin-T · positives"}
PAIRS = [("vit_base_positives", "vit_positives"), ("swin_positives", "vit_base_positives")]
N_BOOT, SEED, ALPHA = 2000, 0, 0.025

images, studies, gt_per_study, images_per_study = load_test_studies()
iou_thresholds = sorted({iou for _, iou, _, _ in METRICS})
tables = {k: detection_tables(k, images, iou_thresholds) for k in MODELS}
ones = np.ones(len(studies))
estimates = {k: metric_values(tables[k], ones, gt_per_study, images_per_study) for k in MODELS}

rng = np.random.default_rng(SEED)
weights = np.stack([np.bincount(rng.integers(0, len(studies), len(studies)), minlength=len(studies))
                    for _ in range(N_BOOT)]).astype(float)
samples = {k: np.zeros((N_BOOT, len(METRICS))) for k in MODELS}
for b in tqdm(range(N_BOOT), desc="Bootstrap", unit="rep"):
    for k in MODELS:
        v = metric_values(tables[k], weights[b], gt_per_study, images_per_study)
        samples[k][b] = [v[name] for name, *_ in METRICS]

ci = [{"experiment": k, "label": MODELS[k], "metric": name, "estimate": estimates[k][name],
       "ci_low": np.quantile(samples[k][:, m], ALPHA), "ci_high": np.quantile(samples[k][:, m], 1 - ALPHA)}
      for k in MODELS for m, (name, *_) in enumerate(METRICS)]
diff = []
for a, b in PAIRS:
    for m, (name, *_) in enumerate(METRICS):
        d = samples[a][:, m] - samples[b][:, m]
        diff.append({"model": MODELS[a], "minus": MODELS[b], "metric": name,
                     "delta": estimates[a][name] - estimates[b][name],
                     "ci_low": np.quantile(d, ALPHA), "ci_high": np.quantile(d, 1 - ALPHA),
                     "p_two_sided": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
pd.DataFrame(ci).to_csv(HERE / "ablation_ci.csv", index=False)
diff = pd.DataFrame(diff); diff.to_csv(HERE / "ablation_paired_diff.csv", index=False)
print(diff[diff.metric.isin(["TPR@IoU0.30_1FPPI", "TPR@IoU0.50_1FPPI", "AP@0.30"])].round(3).to_string(index=False))

# FROC at IoU 0.30
n_images, total_gt = len(images), gt_per_study.sum()
fig, ax = plt.subplots(figsize=(8, 5.5))
for k, color, ls in (("swin_positives", "#1baf7a", "-"), ("vit_positives", "#1baf7a", "--"), ("vit_base_positives", "#7a4fd6", "--")):
    tp, _ = tables[k][0.30]
    x = np.concatenate([[0], np.cumsum(~tp) / n_images]); y = np.concatenate([[0], np.cumsum(tp) / total_gt])
    ax.step(x, y, where="post", color=color, ls=ls, lw=2.2, label=f"{MODELS[k]}  ({estimates[k]['TPR@IoU0.30_1FPPI']:.3f} @1 FPPI)")
ax.axvline(1, color="#999", ls=":"); ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.grid(alpha=.3)
ax.set_xlabel("Mean number of false positives per image"); ax.set_ylabel("Sensitivity (TPR)")
ax.set_title("ViT size ablation · positives · IoU 0.30", loc="left"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(HERE / "ablation_froc.png", dpi=110)
