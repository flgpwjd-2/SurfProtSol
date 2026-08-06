"""License-safe interface between an external surface encoder and SurfProtSol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .surface_readout import SurfaceToProteinReadout


@dataclass(frozen=True)
class SurfacePointBatch:
    coordinates: torch.Tensor
    normals: torch.Tensor
    features: torch.Tensor
    batch: torch.Tensor

    def validate(self) -> None:
        point_count = self.coordinates.shape[0]
        if self.coordinates.shape != (point_count, 3):
            raise ValueError("coordinates must have shape [N, 3]")
        if self.normals.shape != (point_count, 3):
            raise ValueError("normals must have shape [N, 3]")
        if self.features.ndim != 2 or self.features.shape[0] != point_count:
            raise ValueError("features must have shape [N, C]")
        if self.batch.shape != (point_count,) or self.batch.dtype != torch.long:
            raise ValueError("batch must be a torch.long tensor with shape [N]")
        devices = {
            self.coordinates.device,
            self.normals.device,
            self.features.device,
            self.batch.device,
        }
        if len(devices) != 1:
            raise ValueError("All surface tensors must be on the same device")


class ExternalSurfaceEncoder(Protocol):
    """Protocol implemented by user-side adapters around separately obtained code."""

    def __call__(self, surface: SurfacePointBatch) -> torch.Tensor:
        """Return point embeddings with shape ``[N, E]``."""


def encode_surface_to_protein(
    encoder: ExternalSurfaceEncoder,
    readout: SurfaceToProteinReadout,
    surface: SurfacePointBatch,
) -> torch.Tensor:
    surface.validate()
    point_embeddings = encoder(surface)
    if point_embeddings.ndim != 2:
        raise ValueError("External surface encoder must return [N, E]")
    if point_embeddings.shape[0] != surface.coordinates.shape[0]:
        raise ValueError("Surface encoder changed the number of aligned points")
    return readout(point_embeddings, surface.batch)
