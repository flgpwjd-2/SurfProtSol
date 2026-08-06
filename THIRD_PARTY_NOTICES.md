# Third-Party Notices

The BSD-3-Clause license in this repository applies only to original
SurfProtSol files in this repository. It does not relicense any upstream code,
dataset, predicted structure, pretrained model, or checkpoint.

## ProtSolM

- Source: https://github.com/tyang816/ProtSolM
- Reference revision: `3ae6593826851a3c2642023e03d22a3ecc72c727`
- Upstream license: CC BY-NC-ND 4.0
- Policy: ProtSolM source is not vendored or modified in this repository.
  Users obtain it from the original source when upstream feature generation is
  required.

## dMaSIF

- Source: https://github.com/FreyrS/dMaSIF
- Reference revision: `0dcc26c3c218a39d5fe26beb2e788b95fb028896`
- Upstream license: CC BY-NC-ND 4.0
- Policy: dMaSIF source is not vendored or modified in this repository.
  SurfProtSol consumes exported surface representations through a documented
  file interface.

## ProtSolM Data

- Labels and metadata: https://huggingface.co/datasets/AI4Protein/ProtSolM
  - Reference revision: `bb6ec84db77588288481ba9871898cf1e35ddf81`
- ESMFold-derived data:
  https://huggingface.co/datasets/AI4Protein/ProtSolM_ESMFold
  - Reference revision: `e5c869f334433ae77105173b7a38d51a427a052f`
- ESMFold PDB files:
  https://huggingface.co/datasets/AI4Protein/ProtSolM_ESMFold_PDB
  - Reference revision: `121ce4edc933289ed0727cc8ee1ffddb997e6113`
- Dataset license metadata observed during the 2026-07-23 audit: Apache-2.0
- Policy: raw datasets are not mirrored in this Git repository. Release
  manifests must pin the exact source revision and checksums.

## ESM2 and ProtSSN

- ESM2 model identifier: `facebook/esm2_t33_650M_UR50D`
- ESM2 reference revision: `08e4846e537177426273712802403f7ba8261b6c`
- ProtSSN checkpoint used in the study: `protssn_k20_h512.pt`
- ProtSSN checkpoint SHA-256:
  `1ab7896a3e02e8da6819e50aef9a23df4fccc1e3848ac973f74ab6eca52cae4e`
- Policy: pretrained weights are not distributed here.

See `docs/licenses.md` and `PROVENANCE.md` for the release boundary.
