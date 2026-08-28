# Mean-Reversion LLM Research

## Research Dashboards

[Open the published research dashboard](https://aryankix101.github.io/rqai/)

The dashboard hub provides one-click access to the prompt dashboard,
autoresearch report, future experiments, out-of-sample testing, latest prompt
campaign, and A/B/C/D interaction update. The tracked static site lives in
`docs/research/`; large generated outputs and private caches remain ignored.

Refresh the published copies after regenerating local dashboards:

```bash
./.venv/bin/python scripts/export_research_site.py
```

GitHub Pages deploys the bundle after it is pushed to `main`. On first use,
select **GitHub Actions** under **Settings → Pages → Build and deployment**.

## Current Pipeline

The active experiment runner is `mean_reversion_llm_v2.py`. The current lab
configuration is a 60/60 candidate pool, a 30/30 book, fixed five-trading-day
gross C2C accounting, and paired Trail5 controls.

Core runtime modules:

- `mean_reversion_llm_v2.py`: prompt variants, LLM caching, replay, and reports.
- `mean_reversion_llm_v1.py`: point-in-time feature and portfolio utilities used by V2.
- `mean_reversion_llm_v25_memory.py`: optional V2.5 historical-analogue fields.
- `run_backtest_v3.py`: holding-window accounting.
- `technical_llm.py`: provider and JSON-response plumbing.
- `market_regime.py`: point-in-time SPY/VIX context helpers.
- `russell_data_loader.py` and `data_source_config.py`: Russell data loading.

Current research and reporting tools:

- `mean_reversion_llm_v2_autoresearch.py`: staged prompt search and campaign management.
- `mean_reversion_llm_v2_consensus.py`: offline 3-of-4 V2 rank-consensus experiment.
- `mean_reversion_llm_v2_side_experiments.py`: dedicated-side specs, hybrid evaluation, and side-campaign evaluator.
- `mean_reversion_llm_v2_phase2_validation.py`: development/OOS comparison tracker.
- `mean_reversion_research_diagnostics.py`: leakage-safe offline diagnostics and prompt-suite declarations.
- `run_mean_reversion_prompt_interaction.py` and `finalize_mean_reversion_prompt_interaction.py`: resumable prompt-interaction supervision and packaging.
- `mean_reversion_llm_v2_sleeve_splice_experiments.py`: frozen sleeve-splice specifications and offline evaluation.
- `run_mean_reversion_sleeve_splice_suite.py` and `run_mean_reversion_sleeve_splice_oos.py`: guarded in-sample/OOS sleeve supervisors.
- `mean_reversion_prompt_dashboard.py` and `update_sleeve_splice_phase2_dashboard.py`: static report generation and sleeve-result updates.
- `presentations/mean_reversion_llm_research_review.html`: self-contained browser presentation of the research.

## Results Layout

- `backtest_results/llm_reversal_signal_v2_2009_2014/`: development report hub.
- `backtest_results/llm_reversal_signal_v2_anon_validation_2015_2020/`: validation period.
- `backtest_results/llm_reversal_signal_v2_anon_final_2020_2025/`: final-period artifacts.
- `backtest_results/llm_reversal_signal_v2_anon_offset2_exploratory_2020_2025/`: offset-2 exploratory run.
- `backtest_results/composite_scores_russell/`: active deterministic score input; do not archive.
- `backtest_results/past/`: retained historical outputs, failed runs, and paid caches no longer on the active path.

Historical prompt variants remain at their original paths because the combined
dashboard reads them directly. Completed standalone experiments and legacy
documentation are kept as local historical snapshots and excluded from Git.

## Repository Hygiene

Git includes runnable source, its tests, documentation, small reference inputs,
and the research-review presentation. Generated backtests, caches, credentials,
virtual environments, OS metadata, and local historical snapshots are excluded
through `.gitignore` so `git add .` does not publish them accidentally.

## Other Retained Pipelines

The deterministic and original portfolio-constructor modules remain available
because they are coherent runnable pipelines with their own tests. They are not
part of the active V2 execution path.

## Consensus And Dedicated-Side Experiments

The consensus experiment is entirely offline. It validates the four frozen
current-epoch V2 caches, excludes the two no-regime paired controls, and writes
all 3-of-4 triplets plus the one-through-four-run ensemble-size curve:

```bash
./.venv/bin/python mean_reversion_llm_v2_consensus.py
```

The fixed dedicated-side test deliberately reuses the completed joint
minimal-change book. First create the two immutable prompt specs, then score
each side once with `mean_reversion_llm_v2.py --phase llm-cache` using the same
2009–2014 anchor-3 manifest and a shared cache directory. The cache identity
includes both `side_scope` and `replicate_id`.

```bash
./.venv/bin/python mean_reversion_llm_v2_side_experiments.py init-specs
./.venv/bin/python mean_reversion_llm_v2.py --phase llm-cache --start 2009-01-01 --end 2014-01-01 --schedule anchors --anchor-offsets 3 --pool-sizes 60 --book-sizes 30 --llm-pool-size 60 --model deepseek-v4-flash --temperature 1 --thinking-mode disabled --anonymize-tickers --cost-bps-levels 0 --experimental-override --experimental-prompt-spec backtest_results/v2_dedicated_side_prompts_2009_2014/prompt_specs/dedicated_long_r1.json --output-dir backtest_results/v2_dedicated_side_prompts_2009_2014/long --cache-dir backtest_results/v2_dedicated_side_prompts_2009_2014/cache
./.venv/bin/python mean_reversion_llm_v2.py --phase llm-cache --start 2009-01-01 --end 2014-01-01 --schedule anchors --anchor-offsets 3 --pool-sizes 60 --book-sizes 30 --llm-pool-size 60 --model deepseek-v4-flash --temperature 1 --thinking-mode disabled --anonymize-tickers --cost-bps-levels 0 --experimental-override --experimental-prompt-spec backtest_results/v2_dedicated_side_prompts_2009_2014/prompt_specs/dedicated_short_r1.json --output-dir backtest_results/v2_dedicated_side_prompts_2009_2014/short --cache-dir backtest_results/v2_dedicated_side_prompts_2009_2014/cache
./.venv/bin/python mean_reversion_llm_v2_side_experiments.py evaluate
```

Only sides listed in `fixed_prompt_result.json` under `autoresearch_sides` should
be searched. Initialize one campaign per qualifying side with its dedicated
parent cache:

```bash
./.venv/bin/python mean_reversion_llm_v2_autoresearch.py campaign init long-side-v1 --side-scope long --side-parent-cache /absolute/path/to/dedicated-long-r1-cache.json
./.venv/bin/python mean_reversion_llm_v2_autoresearch.py campaign start long-side-v1
```

Smoke, mini, and medium candidates use one scorer run. The sole full-stage
finalist uses three independent replicas and mean-rank consensus. Manual
`approve-validation` freezes the winner, all three validation replicas, the
dedicated parent, lineage, manifests, and code/harness fingerprints before any
2015–2019 call. Use the batch command for the actual manual reveal approval;
pass one campaign if only one side won, or both campaigns if both won:

```bash
./.venv/bin/python mean_reversion_llm_v2_side_experiments.py freeze-validation backtest_results/autoresearch_campaigns/long-side-v1 --output backtest_results/autoresearch_campaigns/side-validation-batch.json
./.venv/bin/python mean_reversion_llm_v2_autoresearch.py campaign start long-side-v1
./.venv/bin/python mean_reversion_llm_v2_side_experiments.py combine-campaigns backtest_results/autoresearch_campaigns/long-side-v1 --validation
```

The one-side combined result retains the existing minimal scorer on the other
side. With two winning campaigns, `combine-campaigns` takes the long
contribution from the frozen long winner and the short contribution from the
frozen short winner. The 2020-and-later firewall remains unchanged.
