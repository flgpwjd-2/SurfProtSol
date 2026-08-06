# Reproduction Workflow

## Level 1: Prediction-Level Paper Audit

The supported public reproduction path uses the released concat-branch and
cross-attention probabilities. Benchmark labels are not redistributed. Obtain
the PDBSol validation/test and ExternalTest CSV files from the original
ProtSolM provider, then run:

```bash
python scripts/reproduce_paper_results.py \
  --artifact-root /path/to/SurfProtSol_artifacts_v1.0.0 \
  --validation-labels /path/to/PDBSol/valid.csv \
  --test-labels /path/to/PDBSol/test.csv \
  --external-labels /path/to/ExternalTest/ExternalTest.csv \
  --output-dir outputs/paper_reproduction
```

The command aligns labels by canonical protein ID, reselects temperature,
fusion weight, and threshold using validation data only, evaluates test and
external data once, and recomputes the Frozen, Shuffled, Chemistry-only, and
Geometry-only controls. It writes `regression_report.csv` with per-value
PASS/FAIL status and exits nonzero if a value differs beyond the tolerance.

Expected operating point:

```text
temperature = 1.5175879396984926
alpha = 0.5
threshold = 0.49
```

AUROC is computed from continuous soluble-class probabilities. Individual
prediction files can be labeled with `scripts/join_labels.py`.

## Level 2: Released-Checkpoint Evaluation

The public evaluator strictly loads the two independent downstream checkpoints
from the Zenodo layout:

```bash
python scripts/evaluate_released_checkpoints.py \
  --checkpoint-dir /path/to/artifacts/checkpoints \
  --fusion-parameters /path/to/artifacts/configs/fusion_parameters.json \
  --test-bundle /path/to/pdbsol_test_bundle \
  --external-bundle /path/to/external_bundle \
  --output-dir outputs/checkpoint_evaluation
```

Exact checkpoint evaluation requires residue-level sequence-structure
features, 42-dimensional physicochemical descriptors, labels, and surface
vectors. The Zenodo archive provides the surface vectors but does not
redistribute the first three inputs. This level is therefore available only to
users who independently prepare the complete bundles described in
`docs/data.md`; Zenodo alone supports Level 1, not Level 2.

## Level 3: Fusion Retraining

1. Independently prepare train, validation, test, and external feature bundles.
2. Verify every bundle with `scripts/verify_bundle.py`.
3. Set paths in `configs/pdbsol.yaml`.
4. Run `surfprotsol-train --config configs/pdbsol.yaml`.

The training command does not use the external split for checkpoint,
temperature, alpha, or threshold selection. Complete training bundles are not
redistributed, so the current archive does not claim standalone retraining.

## Level 4: Surface Regeneration

Surface regeneration requires the pinned dMaSIF source, the audited software
environment, ESMFold structures, point-level caches, and a compatible
SurfProtSol surface-readout checkpoint. The current clean release does not
include an executable dMaSIF adapter or the surface-readout checkpoint and does
not claim end-to-end `surface_z` regeneration.

## Release Preparation

The repository contains utilities for generating checksum-only private
inventories, exporting label-free predictions, and recording a path-free
software environment. See `scripts/build_artifact_inventory.py`,
`scripts/export_prediction_artifacts.py`, and `scripts/capture_environment.py`.

## Recorded Runtime

The study recorded approximately two hours for `surface_z` generation and
approximately twenty hours for a 10-13 epoch training run on the original
AutoDL setup. These values document the original run and are not presented as
a computational advantage.
