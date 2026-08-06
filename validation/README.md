# Local Validation

`paper_fusion_regression.json` records the original final-model regression.
`reported_metrics.csv` records exact expected metrics for SurfProtSol and the
four released surface controls. No benchmark labels or private server paths are
stored here.

The regression must reproduce:

- `T = 1.5175879396984926`;
- `alpha = 0.5`;
- `threshold = 0.49`;
- PDBSol test AUROC `0.8895794980083506`;
- external AUROC `0.6570711966672655`.

Run `scripts/reproduce_paper_results.py` with labels obtained from the original
ProtSolM provider. The command writes a comparison for every reported numeric
field and exits with a nonzero status if any value exceeds the configured
tolerance.
