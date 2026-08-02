from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from audio_probe.metrics import aggregate_cv_metrics, compute_metrics


def fit_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
) -> Pipeline:
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)
    return clf


def predict_probe(clf: Pipeline, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = clf.predict(x)
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(x)[:, 1]
    else:
        prob = pred.astype(np.float32)
    return pred, prob


def run_stratified_kfold(
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Stratified k-fold CV over precomputed features (probe only)."""
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if len(y) < n_splits:
        raise ValueError(f"Need at least {n_splits} samples for {n_splits}-fold CV, got {len(y)}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(x, y)):
        clf = fit_linear_probe(x[train_idx], y[train_idx], seed=seed)
        pred, prob = predict_probe(clf, x[test_idx])
        metrics = compute_metrics(y[test_idx], pred, prob)
        folds.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                **metrics,
            }
        )

    agg = aggregate_cv_metrics(folds)
    return {
        "n_splits": n_splits,
        "seed": seed,
        "folds": folds,
        "mean": agg["mean"],
        "std": agg["std"],
    }


class LinearHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)
