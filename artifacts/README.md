# Release Artifacts

The `predictions/` directory contains release-safe outputs with the benchmark
protein identifier and continuous soluble-class probability. Benchmark labels
are intentionally omitted. Obtain PDBSol and ExternalTest labels from the
original provider and join them by identifier before evaluation.

Use `scripts/join_labels.py` for individual files or
`scripts/reproduce_paper_results.py` for the complete paper metric audit. The
latter recomputes final SurfProtSol predictions and the four released surface
controls and compares them with `validation/reported_metrics.csv`.

Large artifacts are archived separately on Zenodo under the reserved DOI
https://doi.org/10.5281/zenodo.21583106. The archive contains:

- 64-dimensional `surface_z` tables;
- portable SurfProtSol-only downstream branch checkpoints;
- source and checksum manifests;
- final fusion parameters and exact metric-regression targets;
- environment and provenance metadata.

The prediction files are covered by `DATA_LICENSE.md`. That license does not
grant rights to upstream benchmark labels, sequences, structures, or model
weights. `prediction_index.csv` records each file's row count and SHA-256.
