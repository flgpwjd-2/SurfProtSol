# Provenance Record

Creation date: 2026-07-23

This repository was created from an empty directory after a file-level audit of
the research workspace. No ProtSolM or dMaSIF source tree was copied into this
repository.

## Independently Authored Core

The following modules were implemented from the SurfProtSol manuscript
specification, tensor interfaces, and mathematical behavior:

- `surfprotsol/dual_fusion.py`
- `surfprotsol/data.py`
- `surfprotsol/metrics.py`
- `surfprotsol/calibration.py`
- `surfprotsol/checkpoints.py`
- `surfprotsol/reproduction.py`
- `surfprotsol/surface_readout.py`
- `surfprotsol/surface_adapter.py`
- `surfprotsol/train.py`
- `surfprotsol/evaluate.py`

The dual-fusion specification is:

1. independent trainable residue-attention pooling for each branch, followed
   by descriptor augmentation from sequence-structure, 42-dimensional
   physicochemical, and 64-dimensional surface representations;
2. surface-guided cross-attention with the surface vector as query and the
   pooled sequence-structure token plus learned prototype tokens as key/value;
3. validation-only temperature calibration of the cross-attention branch;
4. validation-only probability integration and threshold selection.

## Reviewed Utility Migration

Small research utilities were not copied verbatim. Their required behavior was
reauthored behind stable command-line interfaces:

- alignment of physicochemical and `surface_z` tables;
- construction of memory-mappable feature bundles;
- protein-wise shuffled-surface controls;
- prediction-only calibration, fusion, and metric recomputation.
- label-safe prediction alignment and paper-value regression checks;
- strict loading and evaluation of independently released branch checkpoints.

## Intentionally Excluded

- modified `run_ft.py`, `eval.py`, `get_feature.py`, and `src/models.py`;
- modified or unmodified dMaSIF source;
- upstream datasets and model weights;
- local result trees, caches, checkpoints, and machine-specific launchers.

The pre-release audit is retained outside this repository under
`release_audit/` in the research workspace.
