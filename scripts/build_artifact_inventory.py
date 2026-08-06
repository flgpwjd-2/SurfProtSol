#!/usr/bin/env python
"""Build a checksum and schema inventory without copying private artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "artifact_id",
    "kind",
    "source",
    "bytes",
    "sha256",
    "rows",
    "columns",
    "release_channel",
    "redistribution_status",
    "notes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
        except StopIteration:
            return 0, []
        rows = sum(1 for _ in reader)
    return rows, columns


def load_spec(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Artifact specification must be a JSON list")
    required = {
        "artifact_id",
        "kind",
        "source",
        "release_channel",
        "redistribution_status",
    }
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Artifact entry {index} must be an object")
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Artifact entry {index} is missing {sorted(missing)}")
        artifact_id = str(item["artifact_id"])
        if artifact_id in seen:
            raise ValueError(f"Duplicate artifact_id: {artifact_id}")
        seen.add(artifact_id)
    return payload


def resolve_source(root: Path, relative_source: str) -> Path:
    source = (root / relative_source).resolve()
    if not source.is_relative_to(root):
        raise ValueError(f"Artifact escapes the source root: {relative_source}")
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def build_inventory(root: Path, spec: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in spec:
        relative_source = Path(str(item["source"])).as_posix()
        source = resolve_source(root, relative_source)
        rows: int | str = ""
        columns: list[str] = []
        if source.suffix.lower() in {".csv", ".tsv"}:
            rows, columns = csv_shape(source)
        records.append(
            {
                "artifact_id": item["artifact_id"],
                "kind": item["kind"],
                "source": relative_source,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "rows": rows,
                "columns": "|".join(columns),
                "release_channel": item["release_channel"],
                "redistribution_status": item["redistribution_status"],
                "notes": item.get("notes", ""),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    records = build_inventory(root, load_spec(args.spec))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} artifact records to {args.output}")


if __name__ == "__main__":
    main()
