"""Protein-level readout for variable-length surface point embeddings."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

PoolingMode = Literal["mean", "meanmax", "attention"]


def _validate_points(points: torch.Tensor, batch: torch.Tensor) -> int:
    if points.ndim != 2:
        raise ValueError("points must have shape [N, E]")
    if batch.ndim != 1 or batch.shape[0] != points.shape[0]:
        raise ValueError("batch must have shape [N] and align with points")
    if points.shape[0] == 0:
        raise ValueError("At least one surface point is required")
    if batch.dtype != torch.long:
        raise ValueError("batch must use torch.long")
    if int(batch.min()) < 0:
        raise ValueError("batch indices must be non-negative")
    group_count = int(batch.max().item()) + 1
    observed = torch.bincount(batch, minlength=group_count)
    if torch.any(observed == 0):
        raise ValueError("batch indices must form a contiguous range")
    return group_count


def segment_mean(points: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    group_count = _validate_points(points, batch)
    sums = points.new_zeros((group_count, points.shape[1]))
    sums.index_add_(0, batch, points)
    counts = torch.bincount(batch, minlength=group_count).to(points.dtype).unsqueeze(1)
    return sums / counts


def segment_max(points: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    group_count = _validate_points(points, batch)
    return torch.stack([points[batch == index].max(dim=0).values for index in range(group_count)])


class SurfaceToProteinReadout(nn.Module):
    """Aggregate point embeddings and project them to a fixed surface vector."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        pooling: PoolingMode = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "meanmax", "attention"):
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling = pooling
        pooled_dim = input_dim * 2 if pooling == "meanmax" else input_dim
        self.attention_gate = (
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(input_dim, 1),
            )
            if pooling == "attention"
            else None
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, output_dim),
        )

    def _attention_pool(self, points: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        group_count = _validate_points(points, batch)
        if self.attention_gate is None:
            raise RuntimeError("attention gate is unavailable")
        scores = self.attention_gate(points).squeeze(-1)
        pooled = []
        for index in range(group_count):
            mask = batch == index
            weights = torch.softmax(scores[mask], dim=0)
            pooled.append(torch.sum(points[mask] * weights.unsqueeze(1), dim=0))
        return torch.stack(pooled)

    def forward(self, points: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if points.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected point dimension {self.input_dim}, found {points.shape[-1]}"
            )
        if self.pooling == "mean":
            pooled = segment_mean(points, batch)
        elif self.pooling == "meanmax":
            pooled = torch.cat((segment_mean(points, batch), segment_max(points, batch)), dim=-1)
        else:
            pooled = self._attention_pool(points, batch)
        return self.projection(pooled)
