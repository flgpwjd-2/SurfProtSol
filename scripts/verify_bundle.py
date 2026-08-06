#!/usr/bin/env python3
"""Validate feature-bundle dimensions, labels, IDs, and finite values."""

from __future__ import annotations

import argparse
import json

import numpy as np

from surfprotsol.data import FeatureBundleDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--sequence-structure-dim", type=int, default=1280)
    parser.add_argument("--physicochemical-dim", type=int, default=42)
    parser.add_argument("--surface-dim", type=int, default=64)
    args = parser.parse_args()

    dataset = FeatureBundleDataset(
        args.bundle,
        expected_dims=(
            args.sequence_structure_dim,
            args.physicochemical_dim,
            args.surface_dim,
        ),
    )
    labels = np.asarray(dataset.labels)
    summary = {
        "bundle": args.bundle,
        "rows": len(dataset),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "sequence_structure_dim": int(dataset.sequence_structure.shape[1]),
        "sequence_structure_level": (
            "residue" if dataset.ragged_sequence_structure else "protein"
        ),
        "physicochemical_dim": int(dataset.physicochemical.shape[1]),
        "surface_dim": int(dataset.surface.shape[1]),
        "first_id": dataset.ids[0] if dataset.ids else None,
        "last_id": dataset.ids[-1] if dataset.ids else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
