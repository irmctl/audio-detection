from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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


class LinearHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)
