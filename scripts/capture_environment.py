#!/usr/bin/env python
"""Capture a path-free software and accelerator environment record."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def installed_packages() -> list[dict[str, str]]:
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    return [
        {"name": name, "version": packages[name]}
        for name in sorted(packages, key=str.casefold)
    ]


def accelerator_record() -> dict[str, Any]:
    record: dict[str, Any] = {}
    try:
        import torch

        record["torch_version"] = torch.__version__
        record["torch_cuda_version"] = torch.version.cuda
        record["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            record["torch_device_count"] = torch.cuda.device_count()
            record["torch_devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        record["torch_available"] = False

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        record["nvidia_smi"] = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
    except (FileNotFoundError, subprocess.SubprocessError):
        record["nvidia_smi"] = []
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "schema_version": 1,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "accelerator": accelerator_record(),
        "packages": installed_packages(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote environment record to {args.output}")


if __name__ == "__main__":
    main()
