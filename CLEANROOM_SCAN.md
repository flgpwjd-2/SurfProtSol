# Clean-Room Similarity Scan

Scan date: 2026-07-23

The clean Python sources were compared against Python/shell files in the
research workspace using exact normalized-line containment. The check is a
review aid, not a copyright determination.

## Core Modules

| Clean module | Highest observed containment | Review |
|---|---:|---|
| `surfprotsol/dual_fusion.py` | 4.46% | Shared lines were generic imports, constructor signatures, dropout calls, and dimension arguments |
| `surfprotsol/train.py` | 5.26% | Generic training/CLI statements |
| `surfprotsol/calibration.py` | 5.98% | Generic NumPy validation and function structure |
| `surfprotsol/surface_adapter.py` | 6.82% | Generic tensor validation |
| `surfprotsol/surface_readout.py` | 15.07% | Shared lines were generic PyTorch imports, `super()`, pooling-mode branches, and dropout declarations |
| `surfprotsol/data.py` | 7.48% | Mostly the factual names of the 42 published feature columns |

No old class/function body was copied into `dual_fusion.py`. The nine exact
lines shared with the historical model file were generic syntax such as
`import torch`, constructor declarations, dropout calls, and dimension
arguments.

## Utilities

Small utilities show higher percentages because their total line counts are
small and common CLI/import statements dominate. For example, the shuffled
surface utility shares standard `argparse`, NumPy, pandas, and `__main__`
boilerplate plus the random-number-generator initialization. The algorithm was
reauthored and additionally enforces a derangement by default.

## Release Decision

The observed overlap is consistent with shared APIs, mathematical operations,
factual column names, and conventional Python boilerplate. Core release files
remain classified as original SurfProtSol code under the repository license.
