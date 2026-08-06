"""Train the two SurfProtSol branches and select validation-only fusion parameters."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .calibration import (
    OperatingPoint,
    apply_temperature,
    fuse_probabilities,
    select_operating_point,
)
from .config import build_model, load_config, model_dimensions, resolve_project_path
from .data import FeatureBundleDataset, collate_feature_batch
from .metrics import classification_metrics, metric_value


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _loader(
    dataset: FeatureBundleDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) % batch_size == 1,
        generator=generator,
        collate_fn=collate_feature_batch,
    )


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _branch_logits(
    model: nn.Module,
    batch: dict[str, Any],
    branch: str,
) -> torch.Tensor:
    if branch == "concat":
        return model.concat_branch(
            batch["sequence_structure"],
            batch["physicochemical"],
            batch["surface"],
            batch.get("residue_mask"),
        )
    if branch == "cross":
        return model.cross_branch(
            batch["sequence_structure"],
            batch["surface"],
            batch.get("residue_mask"),
        )
    raise KeyError(branch)


@torch.no_grad()
def predict_branch(
    model: nn.Module,
    loader: DataLoader,
    branch: str,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    names: list[str] = []
    labels: list[int] = []
    probabilities: list[float] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        logits = _branch_logits(model, batch, branch)
        probability = torch.softmax(logits, dim=-1)[:, 1]
        names.extend(str(name) for name in raw_batch["name"])
        labels.extend(int(value) for value in raw_batch["label"].tolist())
        probabilities.extend(float(value) for value in probability.cpu().tolist())
    return names, np.asarray(labels, dtype=np.int64), np.asarray(probabilities)


def train_branch(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    branch: str,
    device: torch.device,
    *,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    monitor: str,
) -> list[dict[str, float | int | str]]:
    module = model.concat_branch if branch == "concat" else model.cross_branch
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.CrossEntropyLoss()
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int | str]] = []

    for epoch in range(1, epochs + 1):
        module.train()
        total_loss = 0.0
        sample_count = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = _branch_logits(model, batch, branch)
            loss = loss_function(logits, batch["label"])
            loss.backward()
            optimizer.step()
            batch_size = int(batch["label"].shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            sample_count += batch_size

        _, valid_labels, valid_probabilities = predict_branch(
            model,
            validation_loader,
            branch,
            device,
        )
        valid_metrics = classification_metrics(valid_labels, valid_probabilities)
        score = metric_value(valid_metrics, monitor)
        row: dict[str, float | int | str] = {
            "branch": branch,
            "epoch": epoch,
            "train_loss": total_loss / max(sample_count, 1),
            **{
                key: value
                for key, value in valid_metrics.items()
                if key in ("auroc", "accuracy", "f1", "mcc")
            },
        }
        history.append(row)
        print(
            f"[{branch}] epoch={epoch:02d} loss={row['train_loss']:.6f} "
            f"valid_{monitor}={score:.6f}"
        )

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(module.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint was selected for {branch}")
    module.load_state_dict(best_state)
    return history


def _load_datasets(
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, FeatureBundleDataset]:
    expected_dims = model_dimensions(config)
    datasets: dict[str, FeatureBundleDataset] = {}
    for split in ("train", "validation", "test", "external"):
        split_config = config["data"].get(split)
        if split_config is None:
            continue
        bundle = resolve_project_path(project_root, split_config["bundle"])
        datasets[split] = FeatureBundleDataset(bundle, expected_dims=expected_dims)
    for required in ("train", "validation", "test"):
        if required not in datasets:
            raise ValueError(f"Missing required data split: {required}")
    return datasets


def _prediction_payload(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    operating_point: OperatingPoint,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    names_concat, labels_concat, concat = predict_branch(model, loader, "concat", device)
    names_cross, labels_cross, cross = predict_branch(model, loader, "cross", device)
    if names_concat != names_cross or not np.array_equal(labels_concat, labels_cross):
        raise RuntimeError("Branch predictions are not aligned")
    calibrated_cross = apply_temperature(cross, operating_point.temperature)
    fused = fuse_probabilities(calibrated_cross, concat, operating_point.alpha)
    prediction = (fused >= operating_point.threshold).astype(np.int64)
    frame = pd.DataFrame(
        {
            "name": names_concat,
            "label": labels_concat,
            "prob_concat": concat,
            "prob_cross": cross,
            "prob_cross_calibrated": calibrated_cross,
            "prob_fused": fused,
            "prediction": prediction,
        }
    )
    return frame, classification_metrics(
        labels_concat,
        fused,
        operating_point.threshold,
    )


def run_training(config_path: str | Path) -> Path:
    config, project_root = load_config(config_path)
    seed = int(config.get("seed", 3407))
    set_random_seed(seed)
    device = choose_device(str(config.get("device", "auto")))
    output_dir = resolve_project_path(project_root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = _load_datasets(config, project_root)
    batch_size = int(config["data"].get("batch_size", 64))
    num_workers = int(config["data"].get("num_workers", 0))
    loaders = {
        split: _loader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=split == "train",
            seed=seed,
        )
        for split, dataset in datasets.items()
    }

    model = build_model(config).to(device)
    training = config["training"]
    histories = []
    histories.extend(
        train_branch(
            model,
            loaders["train"],
            loaders["validation"],
            "concat",
            device,
            learning_rate=float(training["concat_learning_rate"]),
            weight_decay=float(training.get("weight_decay", 0.01)),
            epochs=int(training.get("epochs", 30)),
            patience=int(training.get("patience", 5)),
            monitor=str(training.get("concat_monitor", "mcc")),
        )
    )
    histories.extend(
        train_branch(
            model,
            loaders["train"],
            loaders["validation"],
            "cross",
            device,
            learning_rate=float(training["cross_learning_rate"]),
            weight_decay=float(training.get("weight_decay", 0.01)),
            epochs=int(training.get("epochs", 30)),
            patience=int(training.get("patience", 5)),
            monitor=str(training.get("cross_monitor", "accuracy")),
        )
    )

    valid_names, valid_labels, valid_concat = predict_branch(
        model, loaders["validation"], "concat", device
    )
    cross_names, cross_labels, valid_cross = predict_branch(
        model, loaders["validation"], "cross", device
    )
    if valid_names != cross_names or not np.array_equal(valid_labels, cross_labels):
        raise RuntimeError("Validation branch predictions are not aligned")
    selection = config["selection"]
    operating_point = select_operating_point(
        valid_labels,
        valid_concat,
        valid_cross,
        optimize=str(selection.get("optimize", "accuracy")),
        temperature_min=float(selection.get("temperature_min", 0.05)),
        temperature_max=float(selection.get("temperature_max", 5.0)),
        temperature_steps=int(selection.get("temperature_steps", 200)),
        alpha_step=float(selection.get("alpha_step", 0.1)),
        threshold_step=float(selection.get("threshold_step", 0.005)),
    )
    model.set_operating_point(
        operating_point.temperature,
        operating_point.alpha,
        operating_point.threshold,
    )

    torch.save(model.state_dict(), output_dir / "model_state.pt")
    pd.DataFrame(histories).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "fusion_parameters.json").write_text(
        json.dumps(operating_point.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics: dict[str, dict[str, float | int]] = {}
    for split in ("validation", "test", "external"):
        if split not in loaders:
            continue
        frame, split_metrics = _prediction_payload(
            model,
            loaders[split],
            device,
            operating_point,
        )
        frame.to_csv(output_dir / f"predictions_{split}.csv", index=False)
        metrics[split] = split_metrics
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Selected on validation: "
        f"T={operating_point.temperature:.10f}, "
        f"alpha={operating_point.alpha:.3f}, "
        f"threshold={operating_point.threshold:.4f}"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
