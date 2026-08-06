#!/usr/bin/env python3
"""Compute standard probability AUROC and threshold-dependent metrics."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from surfprotsol.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--probability-column", default="probability")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output")
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    metrics = classification_metrics(
        frame[args.label_column].to_numpy(),
        frame[args.probability_column].to_numpy(),
        args.threshold,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
