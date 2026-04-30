# Today Change Summary

Date: 2026-03-20

## Scope

This summary reflects the current working tree as of today. It covers:

- Modified today, existing files:
  - `composite_score_calculator.py`
  - `factor_backtest_engine.py`
  - `interactive_portfolio_dashboard.py`
  - `llm_ab_comparison.py`
  - `llm_ab_test_runner.sh`
  - `llm_composite_writer.py`
  - `llm_portfolio_constructor.py`
  - `run_pipeline.sh`
  - `hit_rate_validator.py`
  - `technical_llm.py`
- New files added today:
  - `adaptive_regime_sweep.py`
  - `data_source_config.py`
  - `llm_loss_memory.py`
  - `portfolio_policies.py`
  - `russell_data_loader.py`
  - `russell_signal_dump.py`
  - `test_llm_experiment_modes.py`
  - `test_llm_loss_memory.py`
  - `test_portfolio_policies.py`
  - `test_russell_pipeline.py`

## 1. Russell Dataset Migration

Main theme: added a Russell-backed stock-universe path using `RusselData` as the primary stock-history source, while still keeping SPY external for the market calendar and regime logic.

Files:

- `data_source_config.py`
- `russell_data_loader.py`
- `russell_signal_dump.py`
- `composite_score_calculator.py`
- `factor_backtest_engine.py`
- `llm_portfolio_constructor.py`
- `llm_composite_writer.py`
- `hit_rate_validator.py`
- `test_russell_pipeline.py`

What changed:

- Added shared dataset path/config constants for Russell signal dumps, Russell composite outputs, Russell LLM composite outputs, and the external SPY path.
- Added a Russell loader that:
  - reads the wide panel from parquet when parquet support is available
  - falls back to streaming the Excel workbook XML otherwise
  - parses suffixed and unsuffixed ticker columns
  - preserves `asset_id` / `source_column` provenance
- Implemented deterministic base-ticker collapsing:
  - prefer the unsuffixed/current column when present
  - otherwise prefer the latest suffixed historical column
  - carry `collision_count` for visibility
- Added a Russell signal-dump builder that outputs close-based stock histories and metadata including:
  - `date`
  - `ticker`
  - `asset_id`
  - `source_column`
  - `is_historical_suffix`
  - `adj_close`
  - close-based forward returns
  - dataset metadata fields
- Added close-to-close execution support in `hit_rate_validator.py` via `close_to_close_hold`.
- Updated `factor_backtest_engine.py` to:
  - default to the Russell composite directory
  - accept composite files with only `adj_close`
  - auto-detect close-only vs legacy open-based datasets
  - use `adj_close` as the execution price when `adj_open` is absent
- Updated `llm_portfolio_constructor.py` to:
  - default to the Russell composite directory
  - keep SPY external
  - include `asset_id` / `source_column` in the LLM universe payload
  - record dataset metadata in cache files
- Updated `llm_composite_writer.py` to validate the Russell dataset version before writing LLM composite outputs.
- Added focused regression coverage for:
  - source-column parsing
  - ticker-collision resolution
  - close-to-close forward labels
  - close-only backtest mode detection

Important note:

- `composite_score_calculator.py` now writes Russell-aware outputs and no longer requires `adj_open`.
- The full original direction/magnitude score definitions were restored in `composite_score_calculator.py`.
- However, the Russell upstream is still a close-only path today, so not every old OHLCV-dependent raw feature is recreated yet.

## 2. LLM Prompting / Two-Stage Portfolio Construction

Main theme: the LLM portfolio flow now supports explicit `signal_only` vs `turnover_aware` research modes, and the actual portfolio construction path was further refactored into a two-stage pipeline:

- stage 1: alpha discovery / candidate ranking
- stage 2: optional turnover overlay / keep-vs-replace implementation
- final weights: deterministic code, not LLM-sized

Files:

- `technical_llm.py`
- `llm_portfolio_constructor.py`
- `llm_ab_test_runner.sh`

What changed:

- Added two explicit prompt modes:
  - `turnover_aware`
  - `signal_only`
- Refactored the old one-shot final-basket prompt into two separate prompt stages:
  - signal-stage prompt: current cross-sectional edge only
  - turnover-overlay prompt: incumbents vs challengers only
- Stage 1 now returns ranked `long_candidates` / `short_candidates` with:
  - `edge_score`
  - `confidence`
  - short rationale text
- Stage 2 now returns final selected names only, without weights, using:
  - previous holdings
  - stage-1 challengers
  - cost-aware keep-vs-replace rules
- Previous holdings are no longer shown in stage 1, even when `turnover_aware=True`.
- Candidate presentation remains centered on clear long and clear short pools.
- Final portfolio weights are now built deterministically in code from stage-1 candidate utility, with:
  - utility-weighted sizing when `use_equal_weight=False`
  - equal-weight execution override when `use_equal_weight=True`
- The final returned portfolio object now also carries intermediate artifacts:
  - `preliminary_portfolio`
  - `signal_stage`
  - `overlay_stage`
- Portfolio transition tracking now derives `kept_from_previous` and `dropped_from_previous` from the actual executed baskets, including treating long/short flips as drops from the prior side.

Why it matters:

- The cost-free experiment remains a first-class configuration instead of a fragile prompt tweak.
- Alpha discovery is now separated from implementation, which makes research attribution cleaner.
- Net-turnover logic is isolated to the overlay stage instead of being mixed into the raw signal prompt.
- The final cached output is still downstream-compatible while giving much better debugging visibility into what each stage did.

## 3. Prompted Loss Memory / Loss Cache

Main theme: added a realized-loss memory system that can feed prior failure analogs back into future LLM prompts.

Files:

- `llm_loss_memory.py`
- `llm_portfolio_constructor.py`
- `technical_llm.py`
- `llm_ab_test_runner.sh`

What changed:

- Added `LossMemoryStore` to persist realized position outcomes as reusable memory cases.
- Memory cases are built from:
  - side
  - entry/exit dates
  - realized return
  - weighted contribution
  - direction/magnitude scores
  - a multi-feature signal snapshot
- Retrieval logic now:
  - filters to realized losses only
  - searches for similar prior cases by feature similarity
  - applies a small same-regime bonus
  - returns top-k matches per target
- Prompt context can now append a "similar prior loss cases" block as cautionary evidence.
- Constructor gained knobs for:
  - enabling loss memory
  - `top_k`
  - max targets per side
- Runner gained env-driven loss-memory flags and path suffixes.

Why it matters:

- This is the new "prompted loss cache" path.
- It lets the model see recurring failure patterns without hard-coding exclusions.

## 4. LLM Token Usage Review

Main theme: capture and persist model-usage data so we can inspect approximate LLM cost/size per rebalance.

Files:

- `technical_llm.py`
- `llm_portfolio_constructor.py`

What changed:

- LLM responses now capture provider-side usage metadata when present.
- Usage aggregation now supports multi-stage runs by storing stage-level usage under `llm_usage["stages"]`.
- Added stored usage fields under `llm_usage`, including:
  - `provider_usage`
  - `prompt_chars`
  - `completion_chars`
  - `approx_prompt_tokens`
  - `approx_completion_tokens`
  - `approx_total_tokens`
- In two-stage `turnover_aware` runs, the top-level usage fields now reflect the sum of:
  - signal stage
  - overlay stage
- Batch constructor now prints per-rebalance token counts when available.
- Usage is saved into each cached portfolio record.

Current status:

- Token usage is now recorded and visible for both one-stage and two-stage LLM flows.
- It is not yet rolled up into downstream dashboard/report artifacts.

## 5. Cache Schema / Compatibility Hardening

Main theme: make cache reuse safe by encoding experiment settings directly into filenames and metadata, and reject legacy/fallback caches.

Files:

- `llm_portfolio_constructor.py`
- `llm_composite_writer.py`

What changed:

- Cache filenames now encode:
  - date range
  - rebalance frequency
  - execution mode
  - prompt style
  - source dataset version
  - optional loss-memory suffix
- Cache metadata now records:
  - schema version
  - execution mode
  - prompt style
  - prompt format version
  - loss-memory settings
  - strict LLM-only source policy
  - minimum trade price used at decision time
  - source dataset version
  - active composite directory
- Resume logic now invalidates cache reuse when:
  - min trade price changed
  - schema or prompt format changed
  - prompt style changed
  - loss-memory settings changed
  - source policy is not strict LLM-only
  - source dataset changed
- Composite writer now refuses legacy/fallback caches instead of silently converting them.
- Prompt format version was bumped again for the two-stage pipeline:
  - `two_stage_signal_overlay_v3`

Why it matters:

- Different LLM experiments should no longer silently share the same cache.
- The old cache at `backtest_results/llm_portfolio_cache/llm_portfolio_2000-01-01_2025-12-31_5d.json` is effectively legacy relative to the stricter prompt-style, loss-memory, and now two-stage prompt-format checks.

## 6. LLM Composite Output / Explicit Execution Metadata

Main theme: downstream backtests now use explicit LLM intent instead of inferring long/short behavior from score sign alone.

Files:

- `llm_composite_writer.py`
- `factor_backtest_engine.py`

What changed:

- Composite writer now preserves base deterministic scores in:
  - `base_direction_score`
  - `base_magnitude_score`
- Added explicit LLM execution columns:
  - `llm_selected`
  - `llm_side`
  - `llm_weight_raw`
  - `llm_weight_exec`
  - `llm_source`
  - `llm_execution_mode`
- Unselected rows are neutralized to zero execution scores.
- Backtest engine detects explicit LLM score files from these columns.
- Explicit LLM backtests now:
  - use cached LLM decision dates
  - resolve to the latest available LLM signal date on or before the requested signal date
  - rebuild weights from explicit selection metadata
  - respect `equal_weight` vs `llm_weighted`
- In the new two-stage flow:
  - `long_basket` / `short_basket` still remain the final downstream contract
  - `weight_raw` now reflects deterministic utility-based sizing from the signal stage
  - `weight_exec` still reflects the actual executed mode
  - non-equal-weight runs remain labeled `llm_weighted` for downstream compatibility, even though sizing is now code-driven from LLM candidate output

Why it matters:

- LLM backtests are now driven by explicit LLM basket membership and side labels.
- This removes a major source of ambiguity between deterministic scores and LLM intent, while preserving comparability across the old equal-weight vs weighted execution paths.

## 7. Portfolio Policy Refactor

Main theme: review timing, hysteresis, and trade-cost math were separated into policy objects so deterministic and LLM flows can share cleaner execution logic.

Files:

- `portfolio_policies.py`
- `factor_backtest_engine.py`

What changed:

- Added reusable policy/data classes:
  - `PositionState`
  - `ReviewDecision`
  - `TradeDecision`
  - `ReviewPolicy`
  - `HysteresisPolicy`
  - `PortfolioConstructionConfig`
- Added reusable helpers for:
  - equal-notional target weights
  - minimum weight-delta filtering
  - turnover and cost calculation
  - asymmetric retention bands
  - replacement-gap logic
- Backtest engine was refactored to:
  - separate review timing from execution timing
  - support adaptive regime-based review cadence
  - support multiple hysteresis modes
  - keep structured review/trade/holdings logs

Why it matters:

- This is the core of the new deterministic-policy experimentation layer.
- It also gives the LLM backtest path a cleaner interface when replaying cached LLM baskets.

## 8. Backtest Metrics / Cost Accounting

Main theme: better net vs gross accounting and richer review/trade activity metrics.

Files:

- `factor_backtest_engine.py`
- `llm_ab_comparison.py`
- `interactive_portfolio_dashboard.py`

What changed:

- Backtest output now carries both gross and net daily return fields.
- Activity metrics now include:
  - total cost bps
  - cost drag
  - review counts
  - trade review counts
  - turnover summaries
- A/B comparison now reports:
  - net total return
  - gross total return
  - cost drag return
  - paired daily spread statistics
  - Pearson/Spearman diagnostics
  - adaptive rolling windows for shorter samples
- Interactive dashboard was updated to:
  - prefer `realized_return_pct`
  - fall back to older return-column names when needed
  - reconstruct raw stock returns correctly for shorts
  - apply turnover/cost-aware equity compounding

Why it matters:

- The reporting stack now matches the cost-aware execution model more closely.
- Net/gross separation is now visible in both analytics and comparison outputs.

## 8. Experiment Runner / Pipeline Updates

Main theme: make LLM experiment orchestration configurable and variant-aware.

Files:

- `llm_ab_test_runner.sh`
- `run_pipeline.sh`

What changed:

- A/B runner now exposes env-driven experiment switches for:
  - prompt mode (`signal_only` vs `turnover_aware`)
  - loss memory
  - adaptive review cadence inputs
  - hysteresis mode
- Cache, LLM score, and result directories now carry prompt/loss-memory tags.
- Runner uses the project venv Python automatically when available.
- A small runner-label fix was added today so run IDs now include:
  - review mode
  - hysteresis mode
  - date window

Why it matters:

- Different experiment variants are easier to keep separate.
- The new runner-label fix reduces accidental mislabeling/overwrite risk across runs.

## 9. New Research / Utility Scripts

Files:

- `adaptive_regime_sweep.py`
- `signal_half_life_experiment.py`

What they add:

- `adaptive_regime_sweep.py`
  - batch-runs multiple adaptive review-cadence combinations
  - summarizes performance, cost drag, review frequency, turnover, and holding time
- `signal_half_life_experiment.py`
  - measures structural score decay and economic decay
  - outputs half-life diagnostics for selected score types and horizons

## 10. Tests Added Today

Files:

- `test_llm_experiment_modes.py`
- `test_llm_loss_memory.py`
- `test_portfolio_policies.py`

Coverage added:

- signal-only prompt behavior
- cache compatibility across prompt styles
- cost drag metric calculation
- loss-memory recording and retrieval
- static vs adaptive review policy behavior
- hysteresis retention/replacement rules
- trade-decision no-op behavior

## 11. Validation Status

Completed:

- Targeted tests passed with the project venv:
  - `./.venv/bin/python -m unittest -q test_llm_experiment_modes.py test_llm_loss_memory.py test_portfolio_policies.py`
  - Result: 12 tests passed
