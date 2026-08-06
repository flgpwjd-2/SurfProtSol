#!/usr/bin/env python3
"""Build a SHA-256 file manifest for the clean public repository."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = ROOT / "RELEASE_MANIFEST.csv"
OUTPUT_JSON = ROOT / "RELEASE_MANIFEST_SUMMARY.json"
EXCLUDED_DIRECTORIES = {
    ".git",
    "__pycache__",
    "data",
    "outputs",
    "checkpoints",
    "vendor",
}
EXCLUDED_FILES = {OUTPUT_CSV.name, OUTPUT_JSON.name}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_metadata(relative: Path) -> tuple[str, str, str]:
    if relative.parts[:2] == ("artifacts", "predictions"):
        return (
            "generated_prediction",
            "SurfProtSol generated probabilities; upstream identifiers retained",
            "CC-BY-4.0; upstream identifiers excluded",
        )
    if relative.as_posix() in {
        "manifests/autodl_artifact_inventory.csv",
        "manifests/autodl_environment.json",
        "validation/paper_fusion_regression.json",
    }:
        return (
            "generated_research_metadata",
            "SurfProtSol generated metadata or result",
            "CC-BY-4.0",
        )
    if relative.parts[0] == "surfprotsol":
        return "source", "SurfProtSol original", "BSD-3-Clause"
    if relative.parts[0] == "scripts":
        return "utility", "SurfProtSol original", "BSD-3-Clause"
    if relative.parts[0] == "tests":
        return "test", "SurfProtSol original", "BSD-3-Clause"
    if relative.parts[0] == "configs":
        return "configuration", "SurfProtSol original", "BSD-3-Clause"
    if relative.suffix in {".md", ".cff"} or relative.name == "LICENSE":
        return "documentation", "SurfProtSol original", "BSD-3-Clause"
    return "project_metadata", "SurfProtSol original", "BSD-3-Clause"


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if relative.name in EXCLUDED_FILES or relative.suffix == ".pyc":
            continue
        category, copyright_scope, license_name = release_metadata(relative)
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "category": category,
                "copyright_scope": copyright_scope,
                "license": license_name,
                "release_action": "include",
            }
        )
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "files": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "all_release_actions": sorted({str(row["release_action"]) for row in rows}),
        "excluded_directories": sorted(EXCLUDED_DIRECTORIES),
    }
    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_CSV} with {len(rows)} files")


if __name__ == "__main__":
    main()
