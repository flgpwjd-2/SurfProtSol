#!/usr/bin/env python
"""Audit the clean repository for common publication blockers."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_DIRECTORIES = {
    ".git",
    "__pycache__",
    "build",
    "dist",
    "checkpoints",
    "data",
    "outputs",
    "vendor",
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".pdb", ".npy", ".npz"}
FORBIDDEN_DIRECTORY_NAMES = {"dMaSIF-master", "ProtSolM-main"}
MAX_GIT_FILE_BYTES = 50 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def scan_content(files: list[Path]) -> dict[str, list[str]]:
    private_patterns = {
        "autodl_absolute_path": b"/root/" + b"autodl-tmp",
        "windows_user_path": re.compile(rb"[A-Za-z]:\\Users\\[^\\\r\n]+"),
        "local_school_path": b"D:" + b"\\school",
    }
    secret_patterns = {
        "private_key": b"-----BEGIN " + b"PRIVATE KEY-----",
        "rsa_private_key": b"-----BEGIN RSA " + b"PRIVATE KEY-----",
        "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
        "openai_style_key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    }
    private_hits: list[str] = []
    secret_hits: list[str] = []
    for path in files:
        if path.name in {"RELEASE_MANIFEST.csv", "RELEASE_MANIFEST_SUMMARY.json"}:
            continue
        content = path.read_bytes()
        relative = path.relative_to(ROOT).as_posix()
        for name, pattern in private_patterns.items():
            matched = pattern.search(content) if hasattr(pattern, "search") else pattern in content
            if matched:
                private_hits.append(f"{relative}:{name}")
        for name, pattern in secret_patterns.items():
            matched = pattern.search(content) if hasattr(pattern, "search") else pattern in content
            if matched:
                secret_hits.append(f"{relative}:{name}")
    return {"private_path_hits": private_hits, "secret_hits": secret_hits}


def verify_manifest() -> list[str]:
    manifest = ROOT / "RELEASE_MANIFEST.csv"
    if not manifest.is_file():
        return ["RELEASE_MANIFEST.csv is missing"]
    failures: list[str] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        relative = Path(row["path"])
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            failures.append(f"manifest path escapes root: {relative.as_posix()}")
            continue
        if not path.is_file():
            failures.append(f"manifest file is missing: {relative.as_posix()}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            failures.append(f"manifest size mismatch: {relative.as_posix()}")
        if sha256_file(path) != row["sha256"]:
            failures.append(f"manifest hash mismatch: {relative.as_posix()}")
    return failures


def main() -> None:
    files = release_files()
    relative_files = [path.relative_to(ROOT) for path in files]
    content_scan = scan_content(files)
    prediction_label_headers = []
    prediction_root = ROOT / "artifacts" / "predictions"
    if prediction_root.is_dir():
        for path in sorted(prediction_root.glob("*.csv")):
            with path.open("rb") as handle:
                header = handle.readline().strip()
            if header.startswith(b"name,label"):
                prediction_label_headers.append(path.relative_to(ROOT).as_posix())

    report: dict[str, Any] = {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files_over_50_mib": [
            relative.as_posix()
            for path, relative in zip(files, relative_files)
            if path.stat().st_size > MAX_GIT_FILE_BYTES
        ],
        "forbidden_asset_files": [
            relative.as_posix()
            for relative in relative_files
            if relative.suffix.lower() in FORBIDDEN_SUFFIXES
        ],
        "forbidden_upstream_directories": sorted(
            {
                part
                for relative in relative_files
                for part in relative.parts
                if part in FORBIDDEN_DIRECTORY_NAMES
            }
        ),
        "symlinks": [
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_symlink()
        ],
        "prediction_files_with_label_column": prediction_label_headers,
        "manifest_failures": verify_manifest(),
        **content_scan,
    }
    failures = {
        key: value
        for key, value in report.items()
        if isinstance(value, list) and value
    }
    report["passed"] = not failures
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
