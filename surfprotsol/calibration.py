"""Validation-only calibration and probability-level branch fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .metrics import classification_metrics, metric_value


def _probability_array(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("probabilities must be non-empty and finite")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return values


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Apply binary temperature scaling in log-odds space."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    p = np.clip(_probability_array(probabilities), epsilon, 1.0 - epsilon)
    log_odds = np.log(p) - np.log1p(-p)
    scaled = log_odds / float(temperature)
    return 1.0 / (1.0 + np.exp(-scaled))


def binary_nll(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.clip(_probability_array(probabilities), 1e-6, 1.0 - 1e-6)
    if y.shape != p.shape:
        raise ValueError("labels and probabilities must have equal length")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def fit_temperature(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum: float = 0.05,
    maximum: float = 5.0,
    steps: int = 200,
) -> tuple[float, float]:
    """Select temperature with minimum validation negative log likelihood."""
    if minimum <= 0 or maximum < minimum or steps < 2:
        raise ValueError("Invalid temperature grid")
    candidates = np.linspace(minimum, maximum, steps, dtype=np.float64)
    best_temperature = 1.0
    best_nll = float("inf")
    for candidate in candidates:
        calibrated = apply_temperature(probabilities, float(candidate))
        nll = binary_nll(labels, calibrated)
        if nll < best_nll:
            best_temperature = float(candidate)
            best_nll = nll
    return best_temperature, best_nll


def fuse_probabilities(
    cross_probabilities: np.ndarray,
    concat_probabilities: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    cross = _probability_array(cross_probabilities)
    concat = _probability_array(concat_probabilities)
    if cross.shape != concat.shape:
        raise ValueError("Branch probability arrays must have equal shape")
    return alpha * cross + (1.0 - alpha) * concat


@dataclass(frozen=True)
class OperatingPoint:
    temperature: float
    alpha: float
    threshold: float
    optimize: str
    validation_score: float
    validation_nll: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def select_operating_point(
    labels: np.ndarray,
    concat_probabilities: np.ndarray,
    cross_probabilities: np.ndarray,
    optimize: str = "accuracy",
    temperature_min: float = 0.05,
    temperature_max: float = 5.0,
    temperature_steps: int = 200,
    alpha_step: float = 0.1,
    threshold_step: float = 0.005,
) -> OperatingPoint:
    """Fit temperature, alpha, and threshold using validation data only."""
    if not 0.0 < alpha_step <= 1.0:
        raise ValueError("alpha_step must lie in (0, 1]")
    if not 0.0 < threshold_step < 1.0:
        raise ValueError("threshold_step must lie in (0, 1)")

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    concat = _probability_array(concat_probabilities)
    cross = _probability_array(cross_probabilities)
    if not (y.shape == concat.shape == cross.shape):
        raise ValueError("Labels and both branch predictions must align")

    temperature, nll = fit_temperature(
        y,
        cross,
        minimum=temperature_min,
        maximum=temperature_max,
        steps=temperature_steps,
    )
    calibrated_cross = apply_temperature(cross, temperature)

    alphas = np.unique(
        np.round(np.arange(0.0, 1.0 + 1e-12, alpha_step), decimals=12)
    )
    thresholds = np.unique(
        np.concatenate(
            (
                np.asarray([0.5]),
                np.arange(threshold_step, 1.0, threshold_step),
                np.asarray([1.0 - 1e-6]),
            )
        )
    )

    best_alpha = 0.5
    best_threshold = 0.5
    best_score = -float("inf")
    for alpha in alphas:
        fused = fuse_probabilities(calibrated_cross, concat, float(alpha))
        for threshold in thresholds:
            metrics = classification_metrics(y, fused, float(threshold))
            score = metric_value(metrics, optimize)
            if score > best_score:
                best_alpha = float(alpha)
                best_threshold = float(threshold)
                best_score = score

    return OperatingPoint(
        temperature=temperature,
        alpha=best_alpha,
        threshold=best_threshold,
        optimize=optimize,
        validation_score=best_score,
        validation_nll=nll,
    )
