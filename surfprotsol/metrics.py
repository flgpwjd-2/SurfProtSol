"""Classification metrics with probability-based AUROC."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as_arrays(
    labels: Iterable[int] | np.ndarray,
    probabilities: Iterable[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"labels and probabilities differ: {y.shape} vs {p.shape}")
    if y.size == 0:
        raise ValueError("metrics require at least one sample")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(p).all():
        raise ValueError("probabilities contain NaN or infinity")
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return y, p


def classification_metrics(
    labels: Iterable[int] | np.ndarray,
    probabilities: Iterable[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute probability AUROC and threshold-dependent binary metrics."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    y, p = _as_arrays(labels, probabilities)
    prediction = (p >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    auroc = float("nan")
    if np.unique(y).size == 2:
        auroc = float(roc_auc_score(y, p))
    return {
        "auroc": auroc,
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n": int(y.size),
        "positive": int(y.sum()),
        "negative": int(y.size - y.sum()),
    }


def metric_value(metrics: dict[str, float | int], name: str) -> float:
    aliases = {"acc": "accuracy", "roc_auc": "auroc"}
    key = aliases.get(name.lower(), name.lower())
    if key not in metrics:
        raise KeyError(f"Unsupported metric {name!r}; available: {sorted(metrics)}")
    return float(metrics[key])
