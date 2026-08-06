#!/usr/bin/env python3
"""Recompute the paper metrics from released probabilities and upstream labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from surfprotsol.calibration import (
    apply_temperature,
    fuse_probabilities,
    select_operating_point,
)
from surfprotsol.metrics import classification_metrics
from surfprotsol.reproduction import (
    join_prediction_labels,
    probability_column,
    read_label_table,
)


CONTROL_MODELS = {
    "Frozen surface": "frozen_surface",
    "Shuffled surface": "shuffled_surface",
    "Chemistry-only": "chemistry_only",
    "Geometry-only": "geometry_only",
}


def _prediction_dir(root: Path) -> Path:
    candidates = (root / "predictions", root / "artifacts" / "predictions")
    match = next((path for path in candidates if path.is_dir()), None)
    if match is None:
        raise FileNotFoundError(
            f"Could not find predictions/ or artifacts/predictions/ under {root}"
        )
    return match


def _artifact_file(predictions: Path, stem: str, split: str) -> Path:
    suffix = "valid" if split == "validation" else split
    path = predictions / f"{stem}_{suffix}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _labeled_probability(
    path: Path,
    labels: pd.DataFrame,
    joined_dir: Path,
) -> pd.DataFrame:
    frame = join_prediction_labels(path, labels)
    frame.to_csv(joined_dir / path.name, index=False)
    probability = probability_column(frame)
    return frame[["name", "label", probability]].rename(
        columns={probability: "probability"}
    )


def _align_branches(concat: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    aligned = concat.rename(columns={"probability": "prob_concat"}).merge(
        cross.rename(columns={"label": "label_cross", "probability": "prob_cross"}),
        on="name",
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) != len(concat) or len(aligned) != len(cross):
        raise ValueError("Concat and cross-attention IDs do not match")
    if not np.array_equal(aligned["label"], aligned["label_cross"]):
        raise ValueError("Concat and cross-attention labels disagree")
    return aligned.drop(columns="label_cross")


def _metric_row(model: str, split: str, metrics: dict[str, float | int]) -> dict[str, object]:
    return {"model": model, "split": split, **metrics}


def _compare_metrics(
    observed: pd.DataFrame,
    expected_path: Path,
    tolerance: float,
) -> tuple[pd.DataFrame, bool]:
    expected = pd.read_csv(expected_path)
    keys = ["model", "split"]
    observed_lookup = observed.set_index(keys)
    report: list[dict[str, object]] = []
    passed = True
    for _, row in expected.iterrows():
        key = (row["model"], row["split"])
        if key not in observed_lookup.index:
            report.append(
                {
                    "model": key[0],
                    "split": key[1],
                    "metric": "row",
                    "expected": "present",
                    "observed": "missing",
                    "absolute_error": "",
                    "status": "FAIL",
                }
            )
            passed = False
            continue
        actual = observed_lookup.loc[key]
        for metric in expected.columns:
            if metric in keys or pd.isna(row[metric]):
                continue
            expected_value = float(row[metric])
            observed_value = float(actual[metric])
            error = abs(observed_value - expected_value)
            status = "PASS" if error <= tolerance else "FAIL"
            passed = passed and status == "PASS"
            report.append(
                {
                    "model": key[0],
                    "split": key[1],
                    "metric": metric,
                    "expected": expected_value,
                    "observed": observed_value,
                    "absolute_error": error,
                    "status": status,
                }
            )
    return pd.DataFrame(report), passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--validation-labels", required=True)
    parser.add_argument("--test-labels", required=True)
    parser.add_argument("--external-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-metrics")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    root = Path(args.artifact_root).resolve()
    predictions = _prediction_dir(root)
    output_dir = Path(args.output_dir).resolve()
    joined_dir = output_dir / "joined_predictions"
    joined_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "validation": read_label_table(args.validation_labels),
        "test": read_label_table(args.test_labels),
        "external": read_label_table(args.external_labels),
    }

    branch_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for split in ("validation", "test", "external"):
        concat = _labeled_probability(
            _artifact_file(predictions, "final_concat", split), labels[split], joined_dir
        )
        cross = _labeled_probability(
            _artifact_file(predictions, "final_cross", split), labels[split], joined_dir
        )
        branch_frames[split] = {"aligned": _align_branches(concat, cross)}

    valid = branch_frames["validation"]["aligned"]
    point = select_operating_point(
        valid["label"].to_numpy(),
        valid["prob_concat"].to_numpy(),
        valid["prob_cross"].to_numpy(),
        optimize="accuracy",
        temperature_min=0.05,
        temperature_max=5.0,
        temperature_steps=200,
        alpha_step=0.1,
        threshold_step=0.005,
    )

    rows: list[dict[str, object]] = []
    for split in ("validation", "test", "external"):
        frame = branch_frames[split]["aligned"].copy()
        frame["prob_cross_calibrated"] = apply_temperature(
            frame["prob_cross"].to_numpy(), point.temperature
        )
        frame["prob_fused"] = fuse_probabilities(
            frame["prob_cross_calibrated"].to_numpy(),
            frame["prob_concat"].to_numpy(),
            point.alpha,
        )
        frame["prediction"] = (frame["prob_fused"] >= point.threshold).astype(int)
        frame.to_csv(output_dir / f"surfprotsol_{split}_predictions.csv", index=False)
        rows.append(
            _metric_row(
                "SurfProtSol",
                split,
                classification_metrics(
                    frame["label"].to_numpy(),
                    frame["prob_fused"].to_numpy(),
                    point.threshold,
                ),
            )
        )

    for model, stem in CONTROL_MODELS.items():
        for split in ("test", "external"):
            frame = _labeled_probability(
                _artifact_file(predictions, stem, split), labels[split], joined_dir
            )
            rows.append(
                _metric_row(
                    model,
                    split,
                    classification_metrics(
                        frame["label"].to_numpy(), frame["probability"].to_numpy(), 0.5
                    ),
                )
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "recomputed_metrics.csv", index=False)
    expected = (
        Path(args.expected_metrics).resolve()
        if args.expected_metrics
        else root / "results" / "reported_metrics.csv"
    )
    if not expected.is_file():
        fallback = Path(__file__).resolve().parents[1] / "validation" / "reported_metrics.csv"
        expected = fallback
    report, passed = _compare_metrics(metrics, expected, args.tolerance)
    report.to_csv(output_dir / "regression_report.csv", index=False)
    summary = {
        "status": "PASS" if passed else "FAIL",
        "tolerance": args.tolerance,
        "artifact_root": str(root),
        "expected_metrics": str(expected),
        "operating_point": point.to_dict(),
        "metric_rows": len(metrics),
        "comparisons": len(report),
        "failed_comparisons": int((report["status"] == "FAIL").sum()),
    }
    (output_dir / "reproduction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"{summary['status']}: {summary['comparisons']} comparisons, "
        f"{summary['failed_comparisons']} failures"
    )
    print(
        f"Selected T={point.temperature:.10f}, alpha={point.alpha:.3f}, "
        f"threshold={point.threshold:.4f}"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
