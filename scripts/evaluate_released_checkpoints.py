#!/usr/bin/env python3
"""Evaluate the two independent Zenodo branch checkpoints on feature bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from surfprotsol.calibration import apply_temperature, fuse_probabilities
from surfprotsol.checkpoints import load_released_branches
from surfprotsol.data import FeatureBundleDataset, collate_feature_batch
from surfprotsol.metrics import classification_metrics
from surfprotsol.train import choose_device


@torch.no_grad()
def evaluate_bundle(
    concat: torch.nn.Module,
    cross: torch.nn.Module,
    bundle: str | Path,
    *,
    device: torch.device,
    temperature: float,
    alpha: float,
    threshold: float,
    batch_size: int,
    num_workers: int,
    allow_prepooled: bool,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    expected_dims = (
        int(concat.sequence_structure_dim),
        int(concat.physicochemical_dim),
        int(concat.surface_dim),
    )
    dataset = FeatureBundleDataset(bundle, expected_dims=expected_dims)
    if not dataset.ragged_sequence_structure and not allow_prepooled:
        raise ValueError(
            "Exact released-checkpoint evaluation requires h_values.npy and "
            "h_offsets.npy. A pooled h.npy bypasses the checkpoint's learned "
            "residue-attention layer; pass --allow-prepooled only for a non-equivalent audit."
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_feature_batch,
    )
    concat.to(device).eval()
    cross.to(device).eval()
    names: list[str] = []
    labels: list[int] = []
    concat_probabilities: list[float] = []
    cross_probabilities: list[float] = []
    for raw in loader:
        sequence = raw["sequence_structure"].to(device)
        physico = raw["physicochemical"].to(device)
        surface = raw["surface"].to(device)
        mask = raw.get("residue_mask")
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device)
        concat_logits = concat(sequence, physico, surface, mask)
        cross_logits = cross(sequence, surface, mask)
        concat_probability = torch.softmax(concat_logits, dim=-1)[:, 1]
        cross_probability = torch.softmax(cross_logits, dim=-1)[:, 1]
        names.extend(str(value) for value in raw["name"])
        labels.extend(int(value) for value in raw["label"].tolist())
        concat_probabilities.extend(concat_probability.cpu().tolist())
        cross_probabilities.extend(cross_probability.cpu().tolist())
    concat_values = np.asarray(concat_probabilities, dtype=np.float64)
    cross_values = np.asarray(cross_probabilities, dtype=np.float64)
    calibrated = apply_temperature(cross_values, temperature)
    fused = fuse_probabilities(calibrated, concat_values, alpha)
    frame = pd.DataFrame(
        {
            "name": names,
            "label": labels,
            "prob_concat": concat_values,
            "prob_cross": cross_values,
            "prob_cross_calibrated": calibrated,
            "prob_fused": fused,
            "prediction": (fused >= threshold).astype(int),
        }
    )
    metrics = classification_metrics(frame["label"], frame["prob_fused"], threshold)
    dataset.close()
    return frame, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--fusion-parameters", required=True)
    parser.add_argument("--validation-bundle")
    parser.add_argument("--test-bundle", required=True)
    parser.add_argument("--external-bundle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-prepooled", action="store_true")
    args = parser.parse_args()

    concat, cross = load_released_branches(args.checkpoint_dir)
    parameters = json.loads(Path(args.fusion_parameters).read_text(encoding="utf-8"))
    device = choose_device(args.device)
    bundles = {
        "validation": args.validation_bundle,
        "test": args.test_bundle,
        "external": args.external_bundle,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for split, bundle in bundles.items():
        if not bundle:
            continue
        frame, metrics = evaluate_bundle(
            concat,
            cross,
            bundle,
            device=device,
            temperature=float(parameters["temperature"]),
            alpha=float(parameters["alpha"]),
            threshold=float(parameters["threshold"]),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            allow_prepooled=args.allow_prepooled,
        )
        frame.to_csv(output_dir / f"predictions_{split}.csv", index=False)
        all_metrics[split] = metrics
    (output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Evaluated {', '.join(all_metrics)} with released branch checkpoints")


if __name__ == "__main__":
    main()
