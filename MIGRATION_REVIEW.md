# Allowlist Migration Review

Review date: 2026-07-23

No candidate file was copied verbatim. A candidate was migrated only by
reauthoring its required behavior behind the clean repository interfaces.

## Core Candidates

| Research-workspace candidate | Decision | Clean replacement or reason |
|---|---|---|
| `src/dmasif_encoder.py` | Deferred | Tightly coupled to upstream dMaSIF internals and contained AutoDL path assumptions; use `surface_adapter.py` protocol until a separately reviewed plugin is written |
| `src/dmasif_surface_pooling.py` | Reauthored | `surfprotsol/surface_readout.py`, implemented without PyG/PyKeOps/torch-scatter |
| `src/utils/supervise_features.py` | Reauthored | `surfprotsol/data.py` with strict ID alignment and explicit 42-feature schema |
| `script/build_prediction_shift_figure.py` | Deferred | Paper-figure utility; migrate after final artifact schema is frozen |
| `script/build_standard_auroc_artifacts.py` | Replaced | Metrics are implemented in `surfprotsol/metrics.py`; manuscript-specific Excel writing remains outside core release |
| `script/cache_embedding_to_surface_z_csv.py` | Deferred | Depends on historical surface checkpoints and cache schemas not present locally |
| `script/check_external_prereqs.py` | Reauthored | `scripts/verify_bundle.py` validates all clean split bundles |
| `script/eval_ensemble_external.py` | Reauthored | `surfprotsol/evaluate.py` applies fixed validation parameters to external data |
| `script/eval_physico42_z_external.py` | Replaced | Clean evaluator covers branch and final fused predictions |
| `script/export_dmasif_z.py` | Deferred | Requires separate upstream dMaSIF integration and server checkpoint provenance |
| `script/extract_dmasif_protein_vectors.py` | Deferred | Contains AutoDL layout assumptions and direct upstream loading |
| `script/install_dmasif_stack_cu121.sh` | Deferred | Hardware-specific installer; publish a tested lock file after AutoDL environment capture |
| `script/make_shuffled_surface_z_features.py` | Reauthored | `scripts/make_shuffled_surface.py` uses deterministic whole-row shuffling |
| `script/merge_physico_surface_z_csv.py` | Reauthored | `scripts/prepare_feature_bundle.py` aligns all modalities by protein ID |
| `script/prepare_external_test_prereqs.sh` | Replaced | Dataset acquisition is documented; immutable revision/checksum still needs server metadata |
| `script/run_ensemble_physico42_cross.sh` | Replaced | `surfprotsol-train` performs branch training and validation-only fusion |
| `script/run_eval_ensemble_physico42_external.sh` | Replaced | `surfprotsol-evaluate --splits external` |
| `script/run_eval_physico42_z_external.sh` | Replaced | Clean evaluator and prediction-only metric script |
| `script/train_dmasif_readout_cached.py` | Deferred | Surface-stage training will be a separate plugin after server cache/checkpoint audit |
| `script/train_dmasif_stage2_finetune.py` | Deferred | Direct upstream coupling and 771-line experimental scope need separate review |
| `script/tune_ensemble_fusion.py` | Reauthored | `surfprotsol/calibration.py` and `scripts/fit_fusion.py` |
| `script/tune_threshold_physico42_z.sh` | Replaced | Threshold selection is part of the validation-only calibration API |

## Optional Multi-Seed Candidates

| Candidate | Decision |
|---|---|
| `script/compare_multiseed_groups.py` | Deferred until released prediction schema is final |
| `script/ensemble_multiseed_probs.py` | Deferred; can be implemented as a generic probability ensemble |
| `script/export_ours2_seed_probs.py` | Excluded from core; historical layout-specific exporter |
| `script/run_multiseed_protsolm_ours2.sh` | Excluded; AutoDL/upstream launcher |
| `script/summarize_multiseed_results.py` | Deferred until server artifacts are transferred |
| `script/tune_multiseed_ensemble.py` | Deferred; no need for the first reproducible release |

## Current Result

- Reauthored or replaced by clean generic functionality: 16 candidates.
- Deferred pending server artifacts or a separate plugin: 10 candidates.
- Excluded from the core repository: 2 candidates.
- Verbatim migrated files: 0.

This review should be repeated after the AutoDL artifact transfer.
