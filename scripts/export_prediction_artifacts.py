#!/usr/bin/env python
"""Export ID-and-probability files while withholding upstream labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


INDEX_FIELDS = (
    "artifact_id",
    "model",
    "split",
    "file",
    "rows",
    "sha256",
    "source_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spec(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Prediction specification must be a JSON list")
    required = {"artifact_id", "model", "split", "source"}
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Prediction entry {index} must be an object")
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Prediction entry {index} is missing {sorted(missing)}")
        artifact_id = str(item["artifact_id"])
        if artifact_id in seen:
            raise ValueError(f"Duplicate artifact_id: {artifact_id}")
        seen.add(artifact_id)
    return payload


def resolve_source(root: Path, relative_source: str) -> Path:
    source = (root / relative_source).resolve()
    if not source.is_relative_to(root):
        raise ValueError(f"Prediction source escapes the root: {relative_source}")
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def export_prediction(source: Path, destination: Path) -> int:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    seen: set[str] = set()
    rows = 0
    with source.open("r", encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        required = {"name", "prob_pos"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source} is missing columns {sorted(missing)}")
        with temporary.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=("name", "prob_pos"),
                lineterminator="\n",
            )
            writer.writeheader()
            for row in reader:
                name = str(row["name"]).strip()
                if not name:
                    raise ValueError(f"{source} contains an empty protein identifier")
                if name in seen:
                    raise ValueError(f"{source} contains duplicate identifier {name!r}")
                seen.add(name)
                probability = float(row["prob_pos"])
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        f"{source} contains invalid probability for {name!r}"
                    )
                writer.writerow(
                    {"name": name, "prob_pos": format(probability, ".17g")}
                )
                rows += 1
    temporary.replace(destination)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for item in load_spec(args.spec):
        source = resolve_source(root, str(item["source"]))
        output = destination / f"{item['artifact_id']}.csv"
        rows = export_prediction(source, output)
        index.append(
            {
                "artifact_id": item["artifact_id"],
                "model": item["model"],
                "split": item["split"],
                "file": output.name,
                "rows": rows,
                "sha256": sha256_file(output),
                "source_sha256": sha256_file(source),
            }
        )

    index_path = destination / "prediction_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=INDEX_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(index)
    print(f"Exported {len(index)} prediction artifacts to {destination}")


if __name__ == "__main__":
    main()
