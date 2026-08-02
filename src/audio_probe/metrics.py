from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCALAR_METRIC_KEYS = ("accuracy", "f1", "precision", "recall", "roc_auc")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None

    report = classification_report(
        y_true,
        y_pred,
        target_names=["real", "fake"],
        output_dict=True,
        zero_division=0,
    )
    return {
        **metrics,
        "per_class": {
            "real": report["real"],
            "fake": report["fake"],
        },
    }


def aggregate_cv_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean / std of scalar metrics across CV folds."""
    mean: dict[str, float | None] = {}
    std: dict[str, float | None] = {}
    for key in SCALAR_METRIC_KEYS:
        vals = [m[key] for m in fold_metrics if m.get(key) is not None]
        if not vals:
            mean[key] = None
            std[key] = None
            continue
        arr = np.asarray(vals, dtype=np.float64)
        mean[key] = float(arr.mean())
        std[key] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": mean, "std": std}
