# Data and Feature Interface

## Stable Model Inputs

The clean fusion model does not parse sequences, PDB files, residue graphs, or
dMaSIF caches. It consumes three aligned protein-level matrices:

| Symbol | File | Default shape | Description |
|---|---|---:|---|
| `h_p` | `h.npy` | `[N, 1280]` | optional pre-pooled sequence-structure representation for lightweight fusion audits |
| `h_{p,i}` | `h_values.npy` + `h_offsets.npy` | `[sum L_p, 1280]` | residue-level sequence-structure representation for full retraining |
| `a_p` | `physico.npy` | `[N, 42]` | global physicochemical descriptors |
| `z_p` | `surface.npy` | `[N, 64]` | protein-level molecular-surface vector |

`ids.csv` defines the protein order and `labels.npy` contains binary labels.
When residue-level inputs are used, `h_offsets.npy` has shape `[N+1]` and
identifies each protein's slice in `h_values.npy`.

## Building a Bundle

```bash
python scripts/prepare_feature_bundle.py \
  --labels /path/to/train.csv \
  --sequence-structure /path/to/train_h.npz \
  --physicochemical /path/to/PDBSol_feature.csv \
  --surface /path/to/pdbsol_surface_z.csv \
  --output-dir data/bundles/pdbsol_train
```

Accepted sequence-structure feature-store formats:

- CSV with an ID column and numeric feature columns;
- NPZ with `ids`/`names` and `values`/`features`/`h_p`;
- directory containing `ids.csv` and `values.npy`.
- ragged NPZ with `ids`, `values`, and `offsets`;
- ragged directory containing `ids.csv`, `values.npy`, and `offsets.npy`.

The physicochemical table is selected using the explicit 42-column schema in
`surfprotsol.data.PHYSICOCHEMICAL_COLUMNS`. Surface columns must be named
`z_0` through `z_63`.

## Sequence-Structure Export

For full training reproduction, the server must export the residue-level fused
representation before the trainable attention-pooling layer. The clean concat
and cross-attention branches each own an independent pooling layer. A pooled
`h_p` archive is still useful for lightweight classifier and fusion audits but
does not reproduce branch-specific pooling training.

Because the upstream code license does not permit redistribution of modified
source, the residue exporter is not bundled here. The released feature archive
should include:

- protein IDs;
- residue-level float32 values and offsets, plus an optional pooled `h_p`;
- source split;
- ESM2 model revision;
- ProtSSN checkpoint SHA-256;
- pooling/checkpoint identifier;
- exporter script SHA-256 retained in the private provenance record.

## Released-Checkpoint Input Boundary

The Zenodo archive contains the two SurfProtSol downstream branch checkpoints,
but it does not contain complete inference bundles. Exact checkpoint inference
requires, for each evaluated split:

- `ids.csv` and legally obtained `labels.npy`;
- residue-level `h_values.npy` and `h_offsets.npy` generated with the pinned
  ESM2 and ProtSSN resources;
- `physico.npy` with the 42 documented descriptor columns;
- `surface.npy` aligned to the same IDs.

The public `surface_z` tables provide only the last item. A protein-level
`h.npy` can be used with `--allow-prepooled` for interface testing, but it
bypasses the learned residue-attention pooling and is not an exact reproduction
of the reported checkpoint outputs.

## Surface Export

The surface artifact should contain protein IDs and `z_0` through `z_63`.
Archive metadata must record:

- source PDB manifest and revision;
- dMaSIF revision;
- point-feature configuration;
- readout configuration;
- surface checkpoint SHA-256;
- geometry/chemistry ablation setting.

Raw point clouds and point-level caches are not required for prediction-only
reproduction.

## Integrity Requirements

Bundle construction fails on:

- duplicate or missing protein IDs;
- label values outside `{0, 1}`;
- unexpected dimensions;
- NaN or infinite features;
- row-count mismatches.

Run `scripts/verify_bundle.py` after transferring each server artifact.
