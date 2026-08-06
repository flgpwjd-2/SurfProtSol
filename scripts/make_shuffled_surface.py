#!/usr/bin/env python3
"""Shuffle complete surface-vector rows while preserving protein IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def surface_columns(frame: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in frame.columns if str(column).startswith("z_")]
    return sorted(
        columns,
        key=lambda column: int(column[2:]) if column[2:].isdigit() else 10**9,
    )


def derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    if size < 2:
        raise ValueError("A shuffled-surface control needs at least two proteins")
    for _ in range(100):
        permutation = rng.permutation(size)
        if np.all(permutation != np.arange(size)):
            return permutation
    offset = int(rng.integers(1, size))
    return np.roll(np.arange(size), offset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--allow-fixed-points",
        action="store_true",
        help="Use an unconstrained permutation rather than a derangement",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    columns = surface_columns(frame)
    if not columns:
        raise ValueError("No z_0, z_1, ... surface columns were found")
    rng = np.random.default_rng(args.seed)
    permutation = (
        rng.permutation(len(frame))
        if args.allow_fixed_points
        else derangement(len(frame), rng)
    )
    output = frame.copy()
    output.loc[:, columns] = frame.iloc[permutation][columns].to_numpy()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    fixed = int(np.sum(permutation == np.arange(len(frame))))
    print(
        f"Wrote {destination}: rows={len(frame)}, surface_dim={len(columns)}, "
        f"seed={args.seed}, fixed_points={fixed}"
    )


if __name__ == "__main__":
    main()
