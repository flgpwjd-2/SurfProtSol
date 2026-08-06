#!/usr/bin/env python3
"""Fetch pinned upstream source into ignored vendor/ after explicit license acceptance."""

from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

UPSTREAMS = {
    "protsolm": {
        "commit": "3ae6593826851a3c2642023e03d22a3ecc72c727",
        "url": "https://github.com/tyang816/ProtSolM",
        "license": "CC BY-NC-ND 4.0",
    },
    "dmasif": {
        "commit": "0dcc26c3c218a39d5fe26beb2e788b95fb028896",
        "url": "https://github.com/FreyrS/dMaSIF",
        "license": "CC BY-NC-ND 4.0",
    },
}


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination != target and destination not in target.parents:
            raise ValueError(f"Unsafe archive path: {member.filename}")
    archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=sorted(UPSTREAMS))
    parser.add_argument("--destination", default="vendor")
    parser.add_argument(
        "--accept-upstream-license",
        action="store_true",
        help="Confirm that you reviewed and accept the upstream license terms",
    )
    args = parser.parse_args()
    if not args.accept_upstream_license:
        raise SystemExit("Refusing download without --accept-upstream-license")

    metadata = UPSTREAMS[args.project]
    owner_repo = metadata["url"].removeprefix("https://github.com/")
    archive_url = (
        f"https://github.com/{owner_repo}/archive/{metadata['commit']}.zip"
    )
    destination = Path(args.destination) / args.project
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {args.project} at {metadata['commit']} under "
        f"{metadata['license']}..."
    )
    with urllib.request.urlopen(archive_url) as response:
        content = response.read()
    temporary = destination.parent / f".{args.project}-extract"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            safe_extract(archive, temporary)
        roots = [path for path in temporary.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("Unexpected upstream archive layout")
        roots[0].rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Fetched to {destination}. This directory is ignored by Git.")


if __name__ == "__main__":
    main()
