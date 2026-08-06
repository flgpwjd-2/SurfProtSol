#!/usr/bin/env python3
"""Join legally obtained benchmark labels to release-safe prediction CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from surfprotsol.reproduction import join_prediction_labels, read_label_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="Official CSV containing IDs and labels")
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id-column", default="name")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()

    labels = read_label_table(
        args.labels,
        id_column=args.id_column,
        label_column=args.label_column,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_value in args.predictions:
        source = Path(source_value)
        joined = join_prediction_labels(source, labels)
        destination = output_dir / source.name
        joined.to_csv(destination, index=False)
        print(f"PASS {source.name}: joined {len(joined)} rows -> {destination}")


if __name__ == "__main__":
    main()
