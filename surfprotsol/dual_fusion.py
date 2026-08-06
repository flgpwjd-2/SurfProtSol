"""Surface-aware dual-fusion classifier.

This module consumes precomputed protein-level representations. It deliberately
contains no sequence encoder, structure encoder, or molecular-surface engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _positive_probability(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected two-class logits [B, 2], got {tuple(logits.shape)}")
    return torch.softmax(logits, dim=-1)[:, 1]


class ClassificationHead(nn.Module):
    """One-hidden-layer classifier used by each independently trained branch."""

    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(input_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class ResidueAttentionPooling(nn.Module):
    """Learn one scalar attention score per residue and return a protein vector."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(feature_dim, 1)

    def forward(
        self,
        residue_features: torch.Tensor,
        residue_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if residue_features.ndim != 3:
            raise ValueError("residue_features must have shape [B, L, H]")
        logits = self.score(residue_features).squeeze(-1)
        if residue_mask is not None:
            if residue_mask.shape != logits.shape:
                raise ValueError("residue_mask must have shape [B, L]")
            if not torch.all(residue_mask.any(dim=1)):
                raise ValueError("Every protein needs at least one valid residue")
            logits = logits.masked_fill(~residue_mask.bool(), -torch.inf)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(residue_features * weights.unsqueeze(-1), dim=1)


def _pool_sequence_structure(
    features: torch.Tensor,
    residue_mask: torch.Tensor | None,
    pooling: ResidueAttentionPooling,
    expected_dim: int,
) -> torch.Tensor:
    if features.ndim == 2:
        if residue_mask is not None:
            raise ValueError("residue_mask is only valid for residue-level features")
        pooled = features
    elif features.ndim == 3:
        pooled = pooling(features, residue_mask)
    else:
        raise ValueError("sequence_structure must have shape [B, H] or [B, L, H]")
    if pooled.shape[1] != expected_dim:
        raise ValueError("Unexpected sequence-structure feature dimension")
    return pooled


class DescriptorAugmentationBranch(nn.Module):
    """Classify the concatenation of sequence-structure, physico, and surface vectors."""

    def __init__(
        self,
        sequence_structure_dim: int = 1280,
        physicochemical_dim: int = 42,
        surface_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sequence_structure_dim = sequence_structure_dim
        self.physicochemical_dim = physicochemical_dim
        self.surface_dim = surface_dim
        self.sequence_pool = ResidueAttentionPooling(sequence_structure_dim)
        auxiliary_dim = physicochemical_dim + surface_dim
        self.auxiliary_norm = nn.BatchNorm1d(auxiliary_dim)
        self.classifier = ClassificationHead(
            sequence_structure_dim + auxiliary_dim,
            dropout=dropout,
        )

    def forward(
        self,
        sequence_structure: torch.Tensor,
        physicochemical: torch.Tensor,
        surface: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sequence_structure = _pool_sequence_structure(
            sequence_structure,
            residue_mask,
            self.sequence_pool,
            self.sequence_structure_dim,
        )
        _validate_modalities(
            sequence_structure,
            physicochemical,
            surface,
            self.sequence_structure_dim,
            self.physicochemical_dim,
            self.surface_dim,
        )
        auxiliary = self.auxiliary_norm(torch.cat((physicochemical, surface), dim=-1))
        return self.classifier(torch.cat((sequence_structure, auxiliary), dim=-1))


class SurfaceGuidedCrossAttentionBranch(nn.Module):
    """Use the molecular-surface vector to query global protein context."""

    def __init__(
        self,
        sequence_structure_dim: int = 1280,
        surface_dim: int = 64,
        num_prototypes: int = 8,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if sequence_structure_dim % num_attention_heads:
            raise ValueError(
                "sequence_structure_dim must be divisible by num_attention_heads"
            )
        if num_prototypes < 1:
            raise ValueError("num_prototypes must be positive")

        self.sequence_structure_dim = sequence_structure_dim
        self.surface_dim = surface_dim
        self.sequence_pool = ResidueAttentionPooling(sequence_structure_dim)
        self.surface_norm = nn.BatchNorm1d(surface_dim)
        self.query_projection = nn.Linear(surface_dim, sequence_structure_dim)
        self.prototype_tokens = nn.Parameter(
            torch.empty(1, num_prototypes, sequence_structure_dim)
        )
        nn.init.normal_(self.prototype_tokens, mean=0.0, std=0.02)

        self.query_norm = nn.LayerNorm(sequence_structure_dim)
        self.context_norm = nn.LayerNorm(sequence_structure_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=sequence_structure_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_norm = nn.LayerNorm(sequence_structure_dim)

        ffn_dim = sequence_structure_dim * ffn_multiplier
        self.feed_forward = nn.Sequential(
            nn.Linear(sequence_structure_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, sequence_structure_dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.classifier = ClassificationHead(sequence_structure_dim, dropout=dropout)

    def forward(
        self,
        sequence_structure: torch.Tensor,
        surface: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sequence_structure = _pool_sequence_structure(
            sequence_structure,
            residue_mask,
            self.sequence_pool,
            self.sequence_structure_dim,
        )
        if surface.ndim != 2:
            raise ValueError("surface must have shape [B, Z]")
        if sequence_structure.shape[0] != surface.shape[0]:
            raise ValueError("All modalities must use the same batch size")
        if sequence_structure.shape[1] != self.sequence_structure_dim:
            raise ValueError("Unexpected sequence-structure feature dimension")
        if surface.shape[1] != self.surface_dim:
            raise ValueError("Unexpected surface feature dimension")

        batch_size = sequence_structure.shape[0]
        surface_query = self.surface_norm(surface)
        query = self.query_norm(self.query_projection(surface_query)).unsqueeze(1)

        prototypes = self.prototype_tokens.expand(batch_size, -1, -1)
        context = torch.cat((sequence_structure.unsqueeze(1), prototypes), dim=1)
        context = self.context_norm(context)

        attended, _ = self.cross_attention(
            query=query,
            key=context,
            value=context,
            need_weights=False,
        )
        attended = self.attention_dropout(attended.squeeze(1))
        fused = self.residual_norm(sequence_structure + attended)
        fused = fused + self.feed_forward_dropout(self.feed_forward(fused))
        return self.classifier(fused)


@dataclass
class DualFusionOutput:
    concat_logits: torch.Tensor
    cross_logits: torch.Tensor
    concat_probability: torch.Tensor
    cross_probability: torch.Tensor
    calibrated_cross_probability: torch.Tensor
    fused_probability: torch.Tensor


class DualFusionModel(nn.Module):
    """Container for independently trained branches and validation-selected fusion."""

    def __init__(
        self,
        sequence_structure_dim: int = 1280,
        physicochemical_dim: int = 42,
        surface_dim: int = 64,
        num_prototypes: int = 8,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        ffn_multiplier: int = 2,
        temperature: float = 1.0,
        alpha: float = 0.5,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.concat_branch = DescriptorAugmentationBranch(
            sequence_structure_dim=sequence_structure_dim,
            physicochemical_dim=physicochemical_dim,
            surface_dim=surface_dim,
            dropout=dropout,
        )
        self.cross_branch = SurfaceGuidedCrossAttentionBranch(
            sequence_structure_dim=sequence_structure_dim,
            surface_dim=surface_dim,
            num_prototypes=num_prototypes,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
        )
        self.set_operating_point(temperature, alpha, threshold)

    def set_operating_point(
        self,
        temperature: float,
        alpha: float,
        threshold: float,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.threshold = float(threshold)

    def forward(
        self,
        sequence_structure: torch.Tensor,
        physicochemical: torch.Tensor,
        surface: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> DualFusionOutput:
        concat_logits = self.concat_branch(
            sequence_structure,
            physicochemical,
            surface,
            residue_mask,
        )
        cross_logits = self.cross_branch(sequence_structure, surface, residue_mask)
        concat_probability = _positive_probability(concat_logits)
        cross_probability = _positive_probability(cross_logits)
        calibrated_cross_probability = _positive_probability(
            cross_logits / self.temperature
        )
        fused_probability = (
            self.alpha * calibrated_cross_probability
            + (1.0 - self.alpha) * concat_probability
        )
        return DualFusionOutput(
            concat_logits=concat_logits,
            cross_logits=cross_logits,
            concat_probability=concat_probability,
            cross_probability=cross_probability,
            calibrated_cross_probability=calibrated_cross_probability,
            fused_probability=fused_probability,
        )

    def predict(
        self,
        sequence_structure: torch.Tensor,
        physicochemical: torch.Tensor,
        surface: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self(sequence_structure, physicochemical, surface, residue_mask)
        return (output.fused_probability >= self.threshold).to(torch.long)


def _validate_modalities(
    sequence_structure: torch.Tensor,
    physicochemical: torch.Tensor,
    surface: torch.Tensor,
    sequence_structure_dim: int,
    physicochemical_dim: int,
    surface_dim: int,
) -> None:
    tensors = (sequence_structure, physicochemical, surface)
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("All modalities must be rank-two [B, D] tensors")
    batch_sizes = {tensor.shape[0] for tensor in tensors}
    if len(batch_sizes) != 1:
        raise ValueError("All modalities must use the same batch size")
    expected = (sequence_structure_dim, physicochemical_dim, surface_dim)
    observed = tuple(tensor.shape[1] for tensor in tensors)
    if observed != expected:
        raise ValueError(f"Expected modality dimensions {expected}, got {observed}")
