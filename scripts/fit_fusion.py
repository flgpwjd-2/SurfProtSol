#!/usr/bin/env python3
"""Fit validation-only temperature/fusion parameters from branch predictions."""

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
from surfprotsol.data import canonical_protein_id
from surfprotsol.metrics import classification_metrics


def read_predictions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_column = next(
        (column for column in ("name", "protein_id") if column in frame),
        None,
    )
    probability_column = next(
        (
            column
            for column in ("prob_pos", "probability", "prob_fused")
            if column in frame
        ),
        None,
    )
    if id_column is None or "label" not in frame or probability_column is None:
        raise ValueError(
            f"{path} needs name/protein_id, label, and prob_pos/probability"
        )
    output = pd.DataFrame(
        {
            "name": frame[id_column].map(canonical_protein_id),
            "label": frame["label"].astype(int),
            "probability": frame[probability_column].astype(float),
        }
    )
    if output["name"].duplicated().any():
        raise ValueError(f"Duplicate protein IDs in {path}")
    return output


def align_branches(
    concat_path: str | Path,
    cross_path: str | Path,
) -> pd.DataFrame:
    concat = read_predictions(concat_path).rename(
        columns={"label": "label_concat", "probability": "prob_concat"}
    )
    cross = read_predictions(cross_path).rename(
        columns={"label": "label_cross", "probability": "prob_cross"}
    )
    aligned = concat.merge(cross, on="name", how="inner", validate="one_to_one")
    if len(aligned) != len(concat) or len(aligned) != len(cross):
        raise ValueError("Branch prediction files contain different protein ID sets")
    if not np.array_equal(aligned["label_concat"], aligned["label_cross"]):
        raise ValueError("Branch prediction labels disagree")
    return aligned.rename(columns={"label_concat": "label"}).drop(
        columns="label_cross"
    )


def evaluate_aligned(
    frame: pd.DataFrame,
    temperature: float,
    alpha: float,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    output = frame.copy()
    output["prob_cross_calibrated"] = apply_temperature(
        output["prob_cross"].to_numpy(),
        temperature,
    )
    output["prob_fused"] = fuse_probabilities(
        output["prob_cross_calibrated"].to_numpy(),
        output["prob_concat"].to_numpy(),
        alpha,
    )
    output["prediction"] = (output["prob_fused"] >= threshold).astype(int)
    metrics = classification_metrics(
        output["label"].to_numpy(),
        output["prob_fused"].to_numpy(),
        threshold,
    )
    return output, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concat-valid", required=True)
    parser.add_argument("--cross-valid", required=True)
    parser.add_argument("--concat-test", required=True)
    parser.add_argument("--cross-test", required=True)
    parser.add_argument("--concat-external")
    parser.add_argument("--cross-external")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--optimize", default="accuracy", choices=("accuracy", "mcc", "f1"))
    parser.add_argument("--temperature-min", type=float, default=0.05)
    parser.add_argument("--temperature-max", type=float, default=5.0)
    parser.add_argument("--temperature-steps", type=int, default=200)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    args = parser.parse_args()

    valid = align_branches(args.concat_valid, args.cross_valid)
    point = select_operating_point(
        valid["label"].to_numpy(),
        valid["prob_concat"].to_numpy(),
        valid["prob_cross"].to_numpy(),
        optimize=args.optimize,
        temperature_min=args.temperature_min,
        temperature_max=args.temperature_max,
        temperature_steps=args.temperature_steps,
        alpha_step=args.alpha_step,
        threshold_step=args.threshold_step,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"operating_point": point.to_dict(), "metrics": {}}

    split_paths = {
        "validation": (args.concat_valid, args.cross_valid),
        "test": (args.concat_test, args.cross_test),
    }
    if bool(args.concat_external) != bool(args.cross_external):
        raise ValueError("Provide both external branch files or neither")
    if args.concat_external:
        split_paths["external"] = (args.concat_external, args.cross_external)

    for split, (concat_path, cross_path) in split_paths.items():
        aligned = align_branches(concat_path, cross_path)
        predictions, metrics = evaluate_aligned(
            aligned,
            point.temperature,
            point.alpha,
            point.threshold,
        )
        predictions.to_csv(output_dir / f"predictions_{split}.csv", index=False)
        summary["metrics"][split] = metrics

    (output_dir / "fusion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"T={point.temperature:.10f} alpha={point.alpha:.3f} "
        f"threshold={point.threshold:.4f}"
    )


if __name__ == "__main__":
    main()
