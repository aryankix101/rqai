# Minimal-change LLM contribution report

Start with [`REPORT.md`](REPORT.md). It contains the findings, matched return tables, added-versus-dropped attribution, baseline comparison, methods, limitations, and an explanation of the unrecorded backend revision.

## Folder contents

- `tables/` has the comparison tables, date-level return lifts, attribution outputs, and analysis summary.
- `inputs/replicas/` has each replica's run configuration, weekly returns, selections, and compressed candidate scores/features/outcomes.
- `inputs/matched_trail5/` has the matched Trail5 returns, selections, and candidate panel.
- `inputs/tabular_baselines/` has the prior-only model predictions, leakage audit, and model-work summary.
- `inputs/minimal_change_prompt.txt` is the exact prompt text used by the three runs.
- `code/` has a snapshot of the analysis script. The same working copy is also present at `scripts/analyze_minimal_change_replicas.py` at the repository root.

The `signal_checks.csv.gz` and `walk_forward_predictions.csv.gz` files are gzip-compressed CSVs. They can be read directly with pandas using `pd.read_csv("file.csv.gz")`.

The report package is intended to be committed with the repository. The analysis script's original defaults read the historical source runs under `backtest_results/`, which is git-ignored; the copied inputs here preserve the key data used for review, but a fresh end-to-end rerun from the package alone is not currently wired up.
