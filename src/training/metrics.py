from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    f1_macro: float
    confusion: np.ndarray

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": float(self.accuracy),
            "f1_macro": float(self.f1_macro),
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    return ClassificationMetrics(accuracy=float(acc), f1_macro=float(f1), confusion=cm)
