# SurfProtSol

SurfProtSol is a clean reference implementation of a surface-aware multimodal
protein-solubility classifier. It integrates:

- a pooled sequence-structure representation `h_p` (default 1280 dimensions);
- global physicochemical descriptors `a_p` (42 dimensions);
- a protein-level molecular-surface vector `z_p` (64 dimensions).

The final model combines a descriptor-augmentation branch and a
surface-guided cross-attention branch at the probability level. Calibration,
fusion weight, and classification threshold are selected only on the
validation split.

## Clean-Repository Boundary

This repository does not contain ProtSolM or dMaSIF source code, raw PDBSol
data, ESM2/ProtSSN weights, or pretrained upstream checkpoints. Those materials
remain under their original licenses and are obtained from their original
sources. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The stable input boundary is a set of precomputed, protein-aligned feature
bundles. This makes the released fusion, evaluation, ablation, and statistical
analysis code independently auditable.

## Installation

```bash
conda env create -f environment.yml
conda activate surfprotsol
```

Alternatively:

```bash
python -m pip install -e .
```

## Feature Bundles

For lightweight fusion audits, each split bundle contains:

```text
bundle/
|-- ids.csv
|-- labels.npy
|-- h.npy
|-- physico.npy
|-- surface.npy
`-- metadata.json
```

Expected shapes are `h=[N,1280]`, `physico=[N,42]`, and
`surface=[N,64]`. Use `scripts/prepare_feature_bundle.py` to align source
tables by protein ID. Details are in [docs/data.md](docs/data.md).

For full retraining, replace `h.npy` with `h_values.npy` and
`h_offsets.npy`. This ragged representation preserves one 1280-dimensional
vector per residue. The descriptor and cross-attention branches then learn
their own attention-pooling parameters, matching the experimental training
design.

## Training

Edit the paths in `configs/pdbsol.yaml`, then run:

```bash
surfprotsol-train --config configs/pdbsol.yaml
```

The command independently trains both branches, writes validation/test
probabilities, fits cross-branch temperature on validation data, selects the
fusion weight and threshold on validation data, and evaluates the test split
once.

## Prediction-Only Reproduction

Existing branch probabilities can reproduce the final calibration and fusion
without loading a model checkpoint. Release-safe ID-and-probability files are
stored in `artifacts/predictions/`; their source hashes and row counts are in
`prediction_index.csv`. Benchmark labels are omitted and must be joined from a
copy obtained from the original dataset provider. Run the complete paper audit
with:

```bash
python scripts/reproduce_paper_results.py \
  --artifact-root /path/to/SurfProtSol_artifacts_v1.0.0 \
  --validation-labels /path/to/PDBSol/valid.csv \
  --test-labels /path/to/PDBSol/test.csv \
  --external-labels /path/to/ExternalTest/ExternalTest.csv \
  --output-dir outputs/paper_reproduction
```

The command recomputes SurfProtSol and the Frozen, Shuffled, Chemistry-only,
and Geometry-only controls, compares 154 expected numeric values, and exits
with a nonzero status on a mismatch. `scripts/join_labels.py` is available for
individual prediction files.

For the paper run, the validation-selected parameters were:

- temperature `T = 1.5175879396984926`;
- cross-attention weight `alpha = 0.5`;
- classification threshold `t* = 0.49`.

## Tests

The core test suite uses only synthetic data:

```bash
python -m unittest discover -s tests -v
```

## Released Checkpoints and Scope

The Zenodo checkpoints contain only the two independently implemented
SurfProtSol downstream branches. They strictly load through
`surfprotsol.checkpoints.load_released_branches` and can be evaluated with
`scripts/evaluate_released_checkpoints.py`.

Checkpoint evaluation also requires residue-level sequence-structure features,
42-dimensional physicochemical descriptors, benchmark labels, and the released
surface vectors. Complete feature bundles are not included in Zenodo.
Consequently, this release supports exact prediction-level metric reproduction,
but does not claim checkpoint-level reproduction from Zenodo alone, complete
retraining, figure regeneration, or end-to-end surface generation. The precise
boundary is documented in `docs/reproduction.md`.

The path-free AutoDL software record is available in
`manifests/autodl_environment.json`. Checksums and schemas for large private
artifacts are recorded in `manifests/autodl_artifact_inventory.csv`; an
inventory entry does not imply redistribution permission.

The source repository is https://github.com/flgpwjd-2/SurfProtSol. The
reproducibility-artifact record has the reserved Zenodo DOI
https://doi.org/10.5281/zenodo.21583106. Pinned upstream revisions and the
ProtSSN checkpoint checksum are recorded in `THIRD_PARTY_NOTICES.md`. Code and
generated research outputs have separate license notices in `LICENSE` and
`DATA_LICENSE.md`.

The file-level public-release inventory is recorded in
`RELEASE_MANIFEST.csv`; regenerate it with
`python scripts/build_release_manifest.py`.

Run the same publication audit used for the release candidate with:

```bash
python scripts/audit_public_repo.py
```
