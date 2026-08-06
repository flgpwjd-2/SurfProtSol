#!/usr/bin/env python3
"""Align labels and three protein-level feature stores into one split bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from surfprotsol.data import (
    PHYSICOCHEMICAL_COLUMNS,
    FeatureStore,
    load_sequence_structure_store,
    read_labels,
    write_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="CSV with name and label columns")
    parser.add_argument("--sequence-structure", required=True, help="1280-D feature store")
    parser.add_argument("--physicochemical", required=True, help="42-D CSV/feature store")
    parser.add_argument("--surface", required=True, help="64-D surface_z CSV/feature store")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id-column", default="name")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--sequence-structure-dim", type=int, default=1280)
    parser.add_argument("--physicochemical-dim", type=int, default=42)
    parser.add_argument("--surface-dim", type=int, default=64)
    parser.add_argument(
        "--metadata-json",
        help="Optional JSON object merged into metadata.json",
    )
    args = parser.parse_args()

    ids, labels = read_labels(
        args.labels,
        id_column=args.id_column,
        label_column=args.label_column,
    )
    sequence_structure = load_sequence_structure_store(
        args.sequence_structure,
        expected_dim=args.sequence_structure_dim,
        id_column=args.id_column,
    )
    physicochemical = FeatureStore.from_path(
        args.physicochemical,
        expected_dim=args.physicochemical_dim,
        id_column=args.id_column,
        columns=PHYSICOCHEMICAL_COLUMNS,
    )
    surface = FeatureStore.from_path(
        args.surface,
        expected_dim=args.surface_dim,
        id_column=args.id_column,
        prefix="z_",
    )
    metadata = json.loads(args.metadata_json) if args.metadata_json else {}
    metadata.update(
        {
            "labels_source": Path(args.labels).name,
            "sequence_structure_source": Path(args.sequence_structure).name,
            "physicochemical_source": Path(args.physicochemical).name,
            "surface_source": Path(args.surface).name,
        }
    )
    write_bundle(
        args.output_dir,
        ids,
        labels,
        sequence_structure.aligned(ids),
        physicochemical.aligned(ids),
        surface.aligned(ids),
        metadata=metadata,
    )
    print(f"Wrote {args.output_dir} with {len(ids)} aligned proteins")


if __name__ == "__main__":
    main()
