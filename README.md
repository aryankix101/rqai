# LLM Mean-Reversion Pipeline

## What this does

This program runs an LLM-assisted mean-reversion stock strategy on the Russell universe.

In plain English:

1. It finds recent losers and recent winners across the universe.
2. It shows the LLM a filtered candidate list plus technical context.
3. The LLM chooses which names to keep.
4. The system converts those choices into a portfolio and backtests the result.

For the current run setup:
- The LLM sees the top `100` long candidates and top `100` short candidates.  (for this example config)
- It must choose exactly `50` longs and `50` shorts. (for this example config)
- The executed portfolio is equal-weighted.
- `--log-llm-trace` stores both the LLM input and raw output for auditability.

## Components actually used

### 1. Data and config

- [data_source_config.py](/Users/aryankumawat/Desktop/RQA/rqai/data_source_config.py)
  Defines where the Russell data, SPY calendar data, and composite score files live.

- [indicator_feature_schema.py](/Users/aryankumawat/Desktop/RQA/rqai/indicator_feature_schema.py)
  Standardizes the technical indicator fields that are passed into the LLM prompt.

- [portfolio_constraints.py](/Users/aryankumawat/Desktop/RQA/rqai/portfolio_constraints.py)
  Applies basic trading guardrails such as minimum price and symbol-hygiene filters.

### 2. LLM portfolio construction

- [llm_portfolio_constructor.py](/Users/aryankumawat/Desktop/RQA/rqai/llm_portfolio_constructor.py)
  The main orchestrator.
  It loads the data, builds each rebalance universe, calls the strategy module, and saves the cached LLM decisions.

- [mean_reversion_llm.py](/Users/aryankumawat/Desktop/RQA/rqai/mean_reversion_llm.py)
  The strategy-specific logic for this run.
  It ranks stocks by `mean_reversion_signal = -(5-day trailing return)`, builds the prompt, and asks the LLM to choose the final `50` longs and `50` shorts.

- [technical_llm.py](/Users/aryankumawat/Desktop/RQA/rqai/technical_llm.py)
  Shared LLM plumbing.
  It handles validation, exact position-count enforcement, weighting logic, and optional trace capture.

### 3. Converting LLM output into backtest inputs

- [llm_composite_writer.py](/Users/aryankumawat/Desktop/RQA/rqai/llm_composite_writer.py)
  Converts the cached LLM portfolio file into score CSVs that the backtest engine can read.

### 4. Backtesting and execution rules

- [factor_backtest_engine.py](/Users/aryankumawat/Desktop/RQA/rqai/factor_backtest_engine.py)
  Simulates the portfolio through time and produces performance outputs such as PnL, holdings logs, and review logs.

- [portfolio_policies.py](/Users/aryankumawat/Desktop/RQA/rqai/portfolio_policies.py)
  Provides shared portfolio construction rules used by the backtest engine, such as weighting and book-sizing behavior.

## What the LLM actually sees

On each rebalance date, the LLM receives:

- the top recent losers for possible longs
- the top recent winners for possible shorts
- the 5-day trailing return and derived mean-reversion signal
- a compact set of technical indicators for context
- strict instructions to return exactly `50` longs and `50` shorts

Most indicator values are percentile-normalized to make them easier to compare across stocks.

## Main outputs

- `backtest_results/llm_portfolio_cache/...json`
  Cached LLM decisions and, when enabled, full prompt/response traces.

- `backtest_results/composite_scores_.../`
  LLM output rewritten into backtest-ready score files.

- `backtest_results/factor_backtest/.../`
  Final performance files, including returns, holdings, and logs.

## Not part of the current run

- `llm_loss_memory.py` is only used if `--enable-loss-memory` is turned on.
- The deterministic LLM strategy path is available, but not used when `--strategy-style mean_reversion` is selected.
