"""Configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .dual_fusion import DualFusionModel


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping")
    for key in ("data", "model", "training", "selection"):
        if key not in payload or not isinstance(payload[key], dict):
            raise ValueError(f"Configuration needs a {key!r} mapping")
    project_root = config_path.parent.parent
    return payload, project_root


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def model_dimensions(config: dict[str, Any]) -> tuple[int, int, int]:
    model = config["model"]
    return (
        int(model["sequence_structure_dim"]),
        int(model["physicochemical_dim"]),
        int(model["surface_dim"]),
    )


def build_model(config: dict[str, Any]) -> DualFusionModel:
    model = config["model"]
    return DualFusionModel(
        sequence_structure_dim=int(model["sequence_structure_dim"]),
        physicochemical_dim=int(model["physicochemical_dim"]),
        surface_dim=int(model["surface_dim"]),
        num_prototypes=int(model.get("num_prototypes", 8)),
        num_attention_heads=int(model.get("num_attention_heads", 8)),
        dropout=float(model.get("dropout", 0.1)),
        ffn_multiplier=int(model.get("ffn_multiplier", 2)),
    )
