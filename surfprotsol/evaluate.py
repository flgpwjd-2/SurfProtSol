"""Evaluate a trained clean SurfProtSol model with fixed validation parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .calibration import OperatingPoint
from .config import build_model, load_config, model_dimensions, resolve_project_path
from .data import FeatureBundleDataset, collate_feature_batch
from .train import _prediction_payload, choose_device


def run_evaluation(
    config_path: str | Path,
    checkpoint_dir: str | Path,
    splits: list[str],
) -> Path:
    config, project_root = load_config(config_path)
    checkpoint = Path(checkpoint_dir).resolve()
    device = choose_device(str(config.get("device", "auto")))
    model = build_model(config)
    model.load_state_dict(torch.load(checkpoint / "model_state.pt", map_location="cpu"))
    parameters = json.loads(
        (checkpoint / "fusion_parameters.json").read_text(encoding="utf-8")
    )
    operating_point = OperatingPoint(**parameters)
    model.set_operating_point(
        operating_point.temperature,
        operating_point.alpha,
        operating_point.threshold,
    )
    model.to(device).eval()

    output_dir = checkpoint / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for split in splits:
        split_config = config["data"].get(split)
        if split_config is None:
            raise ValueError(f"Split {split!r} is absent from the configuration")
        bundle = resolve_project_path(project_root, split_config["bundle"])
        dataset = FeatureBundleDataset(bundle, expected_dims=model_dimensions(config))
        loader = DataLoader(
            dataset,
            batch_size=int(config["data"].get("batch_size", 64)),
            shuffle=False,
            num_workers=int(config["data"].get("num_workers", 0)),
            collate_fn=collate_feature_batch,
        )
        frame, metrics = _prediction_payload(model, loader, device, operating_point)
        frame.to_csv(output_dir / f"predictions_{split}.csv", index=False)
        all_metrics[split] = metrics
    (output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["test", "external"])
    args = parser.parse_args()
    run_evaluation(args.config, args.checkpoint_dir, args.splits)


if __name__ == "__main__":
    main()
