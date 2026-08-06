# License Boundary

## Included Code

Original files in this repository are released under BSD-3-Clause.

## Excluded Upstream Code

ProtSolM and dMaSIF both display CC BY-NC-ND 4.0 in their repositories.
SurfProtSol therefore does not distribute:

- unchanged copies of their source;
- locally modified copies;
- source patches containing their implementation text;
- upstream model checkpoints.

Users obtain pinned upstream revisions directly from their authors. The
optional `scripts/fetch_upstream.py` downloads them only after an explicit
license-acceptance flag and places them under ignored `vendor/`.

## Data

The audited Hugging Face pages display Apache-2.0 metadata for the ProtSolM
datasets. Dataset and code licenses are independent. GitHub contains only
manifests, schemas, and download/verification instructions; large data
artifacts should use Zenodo or Hugging Face with an exact revision and checksum.

## Generated Artifacts

SurfProtSol prediction files and surface vectors may be released separately
after confirming that they contain no upstream source or weights. Their
metadata must preserve attribution to the underlying dataset and models.

This document records the engineering release policy and is not legal advice.
