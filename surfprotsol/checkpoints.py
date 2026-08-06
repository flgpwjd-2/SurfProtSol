"""Load the release-safe SurfProtSol branch checkpoint format."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dual_fusion import DescriptorAugmentationBranch, SurfaceGuidedCrossAttentionBranch


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint with weights-only mode when supported by PyTorch."""
    source = Path(path)
    parameters = inspect.signature(torch.load).parameters
    if "weights_only" in parameters:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    else:
        warnings.warn(
            "This PyTorch version does not support weights_only=True. Only load "
            "the checksum-verified SurfProtSol release checkpoints.",
            RuntimeWarning,
            stacklevel=2,
        )
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {source} must contain a dictionary payload")
    required = {"model_type", "model_config", "state_dict"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint {source} is missing {sorted(missing)}")
    if not isinstance(payload["model_config"], dict) or not isinstance(
        payload["state_dict"], dict
    ):
        raise ValueError(f"Checkpoint {source} has an invalid schema")
    return payload


def load_released_branch(path: str | Path) -> nn.Module:
    """Construct one public branch and strictly load its released state dict."""
    payload = load_checkpoint_payload(path)
    model_type = str(payload["model_type"])
    config = dict(payload["model_config"])
    if model_type == "descriptor_augmentation":
        branch: nn.Module = DescriptorAugmentationBranch(**config)
    elif model_type == "surface_guided_cross_attention":
        branch = SurfaceGuidedCrossAttentionBranch(**config)
    else:
        raise ValueError(f"Unsupported released model type: {model_type}")
    branch.load_state_dict(payload["state_dict"], strict=True)
    return branch


def load_released_branches(
    checkpoint_dir: str | Path,
) -> tuple[DescriptorAugmentationBranch, SurfaceGuidedCrossAttentionBranch]:
    """Load both branch checkpoints from the Zenodo directory layout."""
    root = Path(checkpoint_dir)
    concat = load_released_branch(root / "descriptor_augmentation.weights.pt")
    cross = load_released_branch(root / "surface_guided_cross_attention.weights.pt")
    if not isinstance(concat, DescriptorAugmentationBranch) or not isinstance(
        cross, SurfaceGuidedCrossAttentionBranch
    ):
        raise TypeError("Released checkpoint model types are inconsistent")
    if concat.sequence_structure_dim != cross.sequence_structure_dim:
        raise ValueError("Branch sequence-structure dimensions disagree")
    if concat.surface_dim != cross.surface_dim:
        raise ValueError("Branch surface dimensions disagree")
    return concat, cross
