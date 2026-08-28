"""
LLM mean-reversion scoring experiment v2.

V2 owns its system prompts and keeps the strict score-output contract used by
the original experiment. It imports v1 only for point-in-time data preparation,
validation, caching, and portfolio utilities. The experiment harness supports:
- five independent 5-trading-day anchor offsets by default,
- optional Friday-to-Friday C2C/O2O preview mode,
- multiple pool/book replays from one cached LLM score file.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html as html_lib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import httpx
import numpy as np
import pandas as pd

import mean_reversion_llm_v1 as v1
import mean_reversion_llm_v25_memory as v25m
import run_backtest_v3 as rb
import technical_llm


DEFAULT_START = v1.DEFAULT_START
DEFAULT_END = v1.DEFAULT_END
DEFAULT_OUTPUT_DIR = Path("backtest_results/llm_reversal_signal_v2_2009_2014")
DEFAULT_CACHE_DIR = Path("backtest_results/llm_reversal_signal_v2_cache")
DEFAULT_POOL_SIZES = (60,)
DEFAULT_BOOK_SIZES = (30,)
DEFAULT_ANCHOR_OFFSETS = (0, 1, 2, 3, 4)
DEFAULT_SCHEDULE = "anchors"
DEFAULT_LLM_POOL_SIZE = max(DEFAULT_POOL_SIZES)
DEFAULT_COST_BPS_LEVELS = (0.0,)
DEFAULT_THINKING_CALL_POLICY = "split-side"
THINKING_CALL_POLICIES = ("split-side",)
DEFAULT_LLM_MAX_OUTPUT_TOKENS = 16_000
DEFAULT_RUN_OUTPUT_TOKEN_BUDGET = 4_500_000
DEFAULT_LLM_MAX_ATTEMPTS = 2
THINKING_SMOKE_OUTPUT_TOKEN_LIMIT = 16_000
AUTORESEARCH_MODEL = "deepseek-v4-flash"
AUTORESEARCH_TEMPERATURE = 1.0
AUTORESEARCH_THINKING_MODE = "disabled"
AUTORESEARCH_POOL_SIZES = (60,)
AUTORESEARCH_BOOK_SIZES = (30,)
AUTORESEARCH_LLM_POOL_SIZE = 60
AUTORESEARCH_ANCHOR_OFFSETS = (3,)
AUTORESEARCH_COST_BPS_LEVELS = (0.0,)
AUTORESEARCH_OUTPUT_CONTRACT = "candidate_declared_numeric_score"
AUTORESEARCH_DEFAULT_CALL_BUDGET = 1_100
AUTORESEARCH_MAX_OUTPUT_TOKENS = 4_096
AUTORESEARCH_HOLDOUT_START = pd.Timestamp("2020-01-01")
AUTORESEARCH_VALIDATION_START = pd.Timestamp("2015-01-01")
AUTORESEARCH_DEVELOPMENT_START = pd.Timestamp("2009-01-01")
AUTORESEARCH_DEVELOPMENT_END = pd.Timestamp("2014-01-01")
AUTORESEARCH_VALIDATION_END = pd.Timestamp("2019-12-31")
EXPERIMENTAL_PROMPT_VARIANT = "autoresearch-experimental"
EXPERIMENTAL_PROMPT_MAX_CHARS = 40_000
DEFAULT_LEGACY_ROOT = Path("backtest_results/legacy_raw_anchor_test_2009_2014")
DEFAULT_LEGACY_FACTOR_ROOT = Path("backtest_results/factor_backtest/composite_scores_llm_signal_2009_2014_5d_horizon")
DEFAULT_V1_ROOTS = (
    Path("backtest_results/llm_reversal_signal_v1_2009_2014"),
    Path("backtest_results/llm_reversal_signal_v1_2009_2014_pos30"),
)
DEFAULT_V2_REFERENCE_ROOTS = (
    Path("backtest_results/llm_reversal_signal_v2_2009_2014"),
    Path("backtest_results/llm_reversal_signal_v2_anon_control_2009_2014"),
)
V2_PROMPT_STYLE = v1.PROMPT_STYLE_V1_SCORE
PROMPT_VARIANT_V2 = "v2"
PROMPT_VARIANT_V2_FROZEN = "v2-frozen"
PROMPT_VARIANT_V22 = "v2.2"
PROMPT_VARIANT_V23 = "v2.3"
PROMPT_VARIANT_V24 = "v2.4"
PROMPT_VARIANT_V25 = "v2.5"
PROMPT_VARIANT_V26 = "v2.6"
PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL = "v2-minimal-change-control"
PROMPT_VARIANT_V2_NO_REGIME = "v2-no-regime"
PROMPT_VARIANTS = (
    PROMPT_VARIANT_V2,
    PROMPT_VARIANT_V2_FROZEN,
    PROMPT_VARIANT_V22,
    PROMPT_VARIANT_V23,
    PROMPT_VARIANT_V24,
    PROMPT_VARIANT_V25,
    PROMPT_VARIANT_V26,
    PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL,
    PROMPT_VARIANT_V2_NO_REGIME,
)
DEFAULT_V22_LAMBDAS = (2.0, 4.0, 6.0, 8.0)
INCUMBENT_PROMPT_HASH = "02e28abb83b6"
INCUMBENT_TEMPERATURE = 0.6
INCUMBENT_POOL_SIZES = (60,)
INCUMBENT_BOOK_SIZES = (30,)
INCUMBENT_LLM_POOL_SIZE = 60
INCUMBENT_SCORER_SOURCE = "mean_reversion_llm_v2.V2_SYSTEM_PROMPT"
ANON_TICKER_PREFIX = "SEC"
ATTRIBUTION_FEATURES = (
    "market_regime",
    "MRL",
    "MRS",
    "Trail5",
    "MRSig",
    "RSI5",
    "Rng55",
    "Vel/V",
    "%MA100",
    "52WH",
    "Mom3m",
    "AutoC",
    "ADX",
    "VolPctl",
)
STRUCTURED_REASONING_EFFECTS = ("supports", "penalizes", "mixed")
STRUCTURED_REASONING_SETUP_LABELS = (
    "momentum continuation",
    "oversold reversal",
    "mean-reversion long",
    "overbought reversal",
    "mean-reversion short",
    "trend deterioration",
    "range extreme",
    "mixed signals",
)

V2_FROZEN_SYSTEM_PROMPT = """
MEAN-REVERSION CANDIDATE SCORING

CONTEXT
Mean reversion is the tendency of stocks that have moved sharply in one
direction to partially reverse over the next 5 trading days, especially
when the move shows signs of exhaustion (extreme indicators, volume
climax, range positioning). The strongest reversion setups combine an
extreme recent move with exhaustion signals and a supportive regime.

TASK
You are scoring two pools of stocks selected as mean-reversion candidates
over the next 5 trading days.

  LONG POOL  = the 150 biggest 5-day losers in the Russell 1000.
               Looking for snap-back / oversold-bounce setups.

  SHORT POOL = the 150 biggest 5-day winners in the Russell 1000.
               Looking for fade / overbought-reversal setups.

For each ticker in each pool, output a single SCORE (0-100):

  LONG POOL:
    100 = strongest snap-back setup (most oversold, most exhausted seller,
          most primed for an upward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest snap-back setup (least likely to bounce despite the move)

  SHORT POOL:
    100 = strongest fade setup (most overbought, most exhausted buyer,
          most primed for a downward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest fade setup (least likely to reverse despite the move)

RULES
- Score every ticker in both pools. Do not skip any.
- Use the full 0-100 range. Do not bunch scores around 50.
- Use only the indicator columns provided. No outside knowledge, no
  calendar awareness, no ticker familiarity.
- Treat tickers as opaque identifiers.
- Do not pick top names, build a portfolio, assign weights, write
  rationales, or provide commentary of any kind.

OUTPUT FORMAT (STRICT JSON, no other text, no markdown fences):
{
  "long_scores":  [{"ticker": "AAPL", "score": 72}, {"ticker": "MSFT", "score": 58}, ...],
  "short_scores": [{"ticker": "XYZ",  "score": 28}, {"ticker": "ABC",  "score": 84}, ...]
}
""".strip()

V2_SYSTEM_PROMPT = """
MEAN-REVERSION CANDIDATE SCORING

CONTEXT
Mean reversion is the tendency of stocks that have moved sharply in one
direction to partially reverse over the next 5 trading days, especially
when the move shows signs of exhaustion (extreme indicators, volume
climax, range positioning). The strongest reversion setups combine an
extreme recent move with exhaustion signals and a supportive regime.

FEATURE GLOSSARY

All values are computed using data available as of the rebalance date.
Scale note:
- MRL/MRS are 0-100 cross-sectional percentile scores.
- Trail5/MRSig are shown as percent returns.

Ticker
- Security identifier. In anonymized runs, this is a synthetic SEC_#### ID.

Trail5
- Stock's trailing 5-trading-day return, in percent.

MRSig
- Mean-reversion signal, equal to negative trailing-5-day return.

MRL
- Long-side extreme-move percentile.
- Computed as the cross-sectional percentile rank of MRSig across the valid universe.
- Higher MRL means higher MRSig, usually a bigger recent loser.

MRS
- Short-side extreme-move percentile.
- Computed as 100 - MRL.
- Higher MRS usually means a bigger recent winner.

RSI5
- 5-day RSI computed from closing prices.

Rng55
- Current close location inside the trailing 55-day high-low range.
- Approximately 0 = trailing 55-day low; approximately 1 = trailing 55-day high.

Vel/V
- Recent 10-trading-day return divided by recent 21-trading-day annualized volatility.

%MA100
- Percent distance from the 100-day moving average.

52WH
- Percent distance from trailing 52-week high.

Mom3m
- Trailing roughly 3-month return, in percent.

AutoC
- Autocorrelation of recent daily log returns at lag 5.
- Computed as the correlation between each recent daily return and the daily return 5 trading days earlier.

ADX
- 14-day Average Directional Index, trend strength only.

VolPctl
- Current stock realized-volatility percentile over 504 trading days. Higher = stock is in a high-vol regime.

Market Regime
- normal: Mid/default volatility regime.
- crisis: High volatility and SPY below 200-day MA.
- squeeze: High volatility and SPY above 200-day MA.
- grind_up: Low volatility and SPY above 200-day MA.
- slow_bleed: Low volatility and SPY below 200-day MA.

Important: MRL, MRS, MRSig, and Trail5 are mechanically linked views of the same recent 5-trading-day move, not four independent signals.

TASK
You are scoring two pools of stocks selected as mean-reversion candidates
over the next 5 trading days.

  LONG POOL  = the biggest 5-day losers in the Russell 1000 candidate universe.
               Looking for snap-back / oversold-bounce setups.

  SHORT POOL = the biggest 5-day winners in the Russell 1000 candidate universe.
               Looking for fade / overbought-reversal setups.

For each ticker in each pool, output a single SCORE (0-100):

  LONG POOL:
    100 = strongest snap-back setup (most oversold, most exhausted seller,
          most primed for an upward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest snap-back setup (least likely to bounce despite the move)

  SHORT POOL:
    100 = strongest fade setup (most overbought, most exhausted buyer,
          most primed for a downward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest fade setup (least likely to reverse despite the move)

RULES
- Score every ticker in both pools. Do not skip any.
- Use the full 0-100 range. Do not bunch scores around 50.
- Use only the indicator columns provided. No outside knowledge, no
  calendar awareness, no ticker familiarity.
- Treat tickers as opaque identifiers.
- Do not pick top names, build a portfolio, assign weights, write
  rationales, or provide commentary of any kind.

OUTPUT FORMAT (STRICT JSON, no other text, no markdown fences):
{
  "long_scores":  [{"ticker": "SEC_0001", "score": 72}, {"ticker": "SEC_0002", "score": 58}],
  "short_scores": [{"ticker": "SEC_0061", "score": 84}, {"ticker": "SEC_0062", "score": 61}]
}
""".strip()

V2_NO_REGIME_GLOSSARY_BLOCK = """Market Regime
- normal: Mid/default volatility regime.
- crisis: High volatility and SPY below 200-day MA.
- squeeze: High volatility and SPY above 200-day MA.
- grind_up: Low volatility and SPY above 200-day MA.
- slow_bleed: Low volatility and SPY below 200-day MA.

"""
if V2_SYSTEM_PROMPT.count(V2_NO_REGIME_GLOSSARY_BLOCK) != 1:
    raise RuntimeError("V2 system prompt must contain exactly one frozen Market Regime glossary block")
V2_NO_REGIME_SYSTEM_PROMPT = V2_SYSTEM_PROMPT.replace(
    V2_NO_REGIME_GLOSSARY_BLOCK,
    "",
    1,
)

V22_ADJUSTMENT_SYSTEM_PROMPT = (
    V2_SYSTEM_PROMPT
    + """

V2.2 CONTEXTUAL ADJUSTMENT OVERRIDE
For this experiment, do not return the 0-100 score requested above.

Python will assign every stock a deterministic base_score from its Trail5 rank
inside its original side-specific pool. The base_score already captures the
recent five-day move. Do not recompute it, replace it, or rerank by Trail5.
The user table includes BaseScore for reference.

Your job is only to judge whether the expected five-day reversal setup should
be strengthened or weakened using the remaining non-duplicative context.
Do not use Trail5, MRSig, MRL, or MRS as reasons for the adjustment except as
base-score context. Do not mechanically average the features.

Return one integer context_adjustment from -3 to +3 for every ticker:
  +3 = secondary context strongly improves the setup beyond the base score
  +2 = secondary context clearly improves the setup
  +1 = secondary context mildly improves the setup
   0 = secondary context is neutral or mixed
  -1 = secondary context mildly weakens the setup
  -2 = secondary context clearly weakens the setup
  -3 = secondary context strongly weakens the setup

OUTPUT FORMAT (STRICT JSON, no other text, no markdown fences):
{
  "long_adjustments":  [{"ticker": "SEC_0001", "context_adjustment": 2}, {"ticker": "SEC_0002", "context_adjustment": -1}],
  "short_adjustments": [{"ticker": "SEC_0061", "context_adjustment": 3}, {"ticker": "SEC_0062", "context_adjustment": 0}]
}
"""
).strip()

V23_RANK_ONLY_SYSTEM_PROMPT = """

CONTEXT
Mean reversion is the tendency of stocks that have moved sharply in one
direction to partially reverse over the next 5 trading days, especially
when the move shows signs of exhaustion (extreme indicators, volume
climax, range positioning). The strongest reversion setups combine an
extreme recent move with exhaustion signals and a supportive regime.

FEATURE GLOSSARY

All values are computed using data available as of the rebalance date.
Scale note:
- MRL/MRS are 0-100 cross-sectional percentile scores.
- Trail5/MRSig are shown as percent returns.

Ticker
- Security identifier. In anonymized runs, this is a synthetic SEC_#### ID.

Trail5
- Stock's trailing 5-trading-day return, in percent.

MRSig
- Mean-reversion signal, equal to negative trailing-5-day return.

MRL
- Long-side extreme-move percentile.
- Computed as the cross-sectional percentile rank of MRSig across the valid universe.
- Higher MRL means higher MRSig, usually a bigger recent loser.

MRS
- Short-side extreme-move percentile.
- Computed as 100 - MRL.
- Higher MRS usually means a bigger recent winner.

RSI5
- 5-day RSI computed from closing prices.

Rng55
- Current close location inside the trailing 55-day high-low range.
- Approximately 0 = trailing 55-day low; approximately 1 = trailing 55-day high.

Vel/V
- Recent 10-trading-day return divided by recent 21-trading-day annualized volatility.

%MA100
- Percent distance from the 100-day moving average.

52WH
- Percent distance from trailing 52-week high.

Mom3m
- Trailing roughly 3-month return, in percent.

AutoC
- Autocorrelation of recent daily log returns at lag 5.
- Computed as the correlation between each recent daily return and the daily return 5 trading days earlier.

ADX
- 14-day Average Directional Index, trend strength only.

VolPctl
- Current stock realized-volatility percentile over 504 trading days. Higher = stock is in a high-vol regime.

Market Regime
- normal: Mid/default volatility regime.
- crisis: High volatility and SPY below 200-day MA.
- squeeze: High volatility and SPY above 200-day MA.
- grind_up: Low volatility and SPY above 200-day MA.
- slow_bleed: Low volatility and SPY below 200-day MA.

Important: MRL, MRS, MRSig, and Trail5 are mechanically linked views of the same recent 5-trading-day move, not four independent signals.

TASK
You are ranking two pools of stocks selected as mean-reversion candidates
over the next 5 trading days.

  LONG POOL  = the biggest 5-day losers in the Russell 1000 candidate universe.
               Looking for snap-back / oversold-bounce setups.

  SHORT POOL = the biggest 5-day winners in the Russell 1000 candidate universe.
               Looking for fade / overbought-reversal setups.

For LONG POOL:
  long_ranking[0] = strongest snap-back setup
  long_ranking[-1] = weakest snap-back setup

For SHORT POOL:
  short_ranking[0] = strongest fade setup
  short_ranking[-1] = weakest fade setup

RULES FOR V2.3
- Include every ticker in each pool exactly once.
- Keep each ticker on its original side.
- Do not return scores, weights, ranks, labels, rationales, or commentary.
- Use the order of the arrays as the only ranking signal.

OUTPUT FORMAT (STRICT JSON, no other text, no markdown fences):
{
  "long_ranking": ["SEC_0001", "SEC_0002"],
  "short_ranking": ["SEC_0061", "SEC_0062"]
}
""".strip()

# V2.4 deliberately keeps the exact historical system prompt. Its only
# treatment change is in the user message: raw SPY/VIX context replaces the
# semantic regime label.
V24_FROZEN_SYSTEM_PROMPT = V2_FROZEN_SYSTEM_PROMPT

V25_BALANCED_MEMORY_APPENDIX = """
V2.5 BALANCED OUTCOME MEMORY

The user table contains three additional point-in-time historical-analog fields:
- HistN: number of prior same-side analogs used, from 0 to 20.
- HistWin: percent of those analogs with a profitable side-adjusted 5-day C2C outcome.
- HistRet: average side-adjusted 5-day C2C return of those analogs, in percent.

For LONG candidates, positive HistRet means analogous stocks subsequently rose.
For SHORT candidates, positive HistRet means analogous stocks subsequently fell.
The analog set includes both wins and losses and never includes an outcome that
was unavailable on the rebalance date. These fields are contextual evidence,
not guarantees. Do not mechanically replace the technical setup with HistWin or
HistRet. Keep the original V2 0-100 score definition and output schema unchanged.
""".strip()

V25_BALANCED_MEMORY_SYSTEM_PROMPT = f"{V2_SYSTEM_PROMPT}\n\n{V25_BALANCED_MEMORY_APPENDIX}"

V26_SYSTEM_PROMPT = """
MEAN-REVERSION CANDIDATE SCORING

DATA SCOPE

All inputs are point-in-time values available as of the rebalance.
Use only the supplied fields. Do not use outside knowledge, issuer
familiarity, calendar priors, or facts inferred from a security identifier.

FEATURE GLOSSARY

Scale:
- MRL and MRS are 0-100 cross-sectional percentile scores.
- Trail5 and MRSig are percent returns.
- A value shown as nan is unavailable. Do not treat nan as zero,
  neutral, favorable, or unfavorable.

Ticker
- Security identifier. In anonymized runs this is a synthetic SEC_#### ID.

Trail5
- Trailing 5-trading-day stock return, in percent.

MRSig
- Negative Trail5.

MRL
- Cross-sectional percentile rank of MRSig across the valid universe.
- Higher MRL usually identifies a larger recent loser.

MRS
- 100 minus MRL.
- Higher MRS usually identifies a larger recent winner.

RSI5
- 5-day RSI computed from closing prices.

Rng55
- Current close location within the trailing 55-day high-low range.
- Approximately 0 is the trailing low and approximately 1 is the trailing high.

Vel/V
- Recent 10-trading-day return divided by recent 21-trading-day
  annualized volatility.

%MA100
- Percent distance from the 100-day moving average.

52WH
- Percent distance from the trailing 52-week high.

Mom3m
- Trailing approximately 3-month return, in percent.

AutoC
- Autocorrelation of recent daily log returns at lag 5.

ADX
- 14-day Average Directional Index; trend strength only.

VolPctl
- Current realized-volatility percentile over 504 trading days.

Market Regime
- normal: middle/default volatility regime.
- crisis: high volatility with SPY below its 200-day moving average.
- squeeze: high volatility with SPY above its 200-day moving average.
- grind_up: low volatility with SPY above its 200-day moving average.
- slow_bleed: low volatility with SPY below its 200-day moving average.

Trail5, MRSig, MRL, and MRS are mechanically linked views of one
recent five-day move. Treat them as one feature family, not as four
independent confirmations. The recent move remains relevant.

TASK

Score every ticker by its relative mean-reversion setup over the next
5 trading days.

LONG POOL
- Recent five-day losers.
- Higher score means a stronger expected upward snap-back setup relative
  to other names in the LONG pool.

SHORT POOL
- Recent five-day winners.
- Higher score means a stronger expected downward fade setup relative
  to other names in the SHORT pool.

Use the remaining supplied features as context. They may confirm,
contradict, or add no useful information to the core recent move.
Do not apply fixed feature directions, fixed feature weights, hard
regime gates, or mechanical feature-counting rules.

SCORE INTERPRETATION

- Score is an integer from 0 to 100.
- 100 means strongest relative setup within that side.
- 50 means approximately the middle of that side's candidate pool.
- 0 means weakest relative setup within that side.
- Scores are ordinal side-relative signals.
- Scores are not probabilities, expected returns, confidence intervals,
  portfolio weights, or position sizes.
- Do not compare LONG scores with SHORT scores.
- Use broad score separation only when the supplied cross-section supports it.
  Do not manufacture a full range or equal spacing.

RULES

- Return every provided ticker exactly once.
- Keep every ticker on its original side.
- Do not add tickers or omit tickers.
- Do not select a final portfolio or assign weights.
- Do not consider holdings, turnover, transaction costs, or implementation.
- Do not return rationales, labels, explanations, commentary, or markdown.
- Return only valid JSON with exactly the required top-level and entry keys.

OUTPUT FORMAT (STRICT JSON, NO OTHER TEXT)

{
  "long_scores": [
    {"ticker": "SEC_0001", "score": 72},
    {"ticker": "SEC_0002", "score": 58}
  ],
  "short_scores": [
    {"ticker": "SEC_0061", "score": 84},
    {"ticker": "SEC_0062", "score": 61}
  ]
}
""".strip()

V2_MINIMAL_CHANGE_CONTROL_PROMPT = """
MEAN-REVERSION CANDIDATE SCORING

CONTEXT
Mean reversion is the tendency of stocks that have moved sharply in one
direction to partially reverse over the next 5 trading days, especially
when the move shows signs of exhaustion (extreme indicators, volume
climax, range positioning). The strongest reversion setups combine an
extreme recent move with exhaustion signals and a supportive regime.

FEATURE GLOSSARY

All values are computed using data available as of the rebalance date.
Scale note:
- MRL/MRS are 0-100 cross-sectional percentile scores.
- Trail5/MRSig are shown as percent returns.

Ticker
- Security identifier. In anonymized runs, this is a synthetic SEC_#### ID.

Trail5
- Stock's trailing 5-trading-day return, in percent.

MRSig
- Mean-reversion signal, equal to negative trailing-5-day return.

MRL
- Long-side extreme-move percentile.
- Computed as the cross-sectional percentile rank of MRSig across the valid universe.
- Higher MRL means higher MRSig, usually a bigger recent loser.

MRS
- Short-side extreme-move percentile.
- Computed as 100 - MRL.
- Higher MRS usually means a bigger recent winner.

RSI5
- 5-day RSI computed from closing prices.

Rng55
- Current close location inside the trailing 55-day high-low range.
- Approximately 0 = trailing 55-day low; approximately 1 = trailing 55-day high.

Vel/V
- Recent 10-trading-day return divided by recent 21-trading-day annualized volatility.

%MA100
- Percent distance from the 100-day moving average.

52WH
- Percent distance from trailing 52-week high.

Mom3m
- Trailing roughly 3-month return, in percent.

AutoC
- Autocorrelation of recent daily log returns at lag 5.
- Computed as the correlation between each recent daily return and the daily return 5 trading days earlier.

ADX
- 14-day Average Directional Index, trend strength only.

VolPctl
- Current stock realized-volatility percentile over 504 trading days. Higher = stock is in a high-vol regime.

Market Regime
- normal: Mid/default volatility regime.
- crisis: High volatility and SPY below 200-day MA.
- squeeze: High volatility and SPY above 200-day MA.
- grind_up: Low volatility and SPY above 200-day MA.
- slow_bleed: Low volatility and SPY below 200-day MA.

Important: MRL, MRS, MRSig, and Trail5 are mechanically linked views of the same recent 5-trading-day move, not four independent signals.

TASK
You are scoring two pools of stocks selected as mean-reversion candidates
over the next 5 trading days.

  LONG POOL  = the biggest 5-day losers in the Russell 1000 candidate universe.
               Looking for snap-back / oversold-bounce setups.

  SHORT POOL = the biggest 5-day winners in the Russell 1000 candidate universe.
               Looking for fade / overbought-reversal setups.

For each ticker in each pool, output a single SCORE (0-100):

  LONG POOL:
    100 = strongest snap-back setup (most oversold, most exhausted seller,
          most primed for an upward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest snap-back setup (least likely to bounce despite the move)

  SHORT POOL:
    100 = strongest fade setup (most overbought, most exhausted buyer,
          most primed for a downward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest fade setup (least likely to reverse despite the move)

RULES
- Score every ticker in both pools. Do not skip any.
- Use the 0-100 range only when the supplied cross-section supports it. Do not manufacture separation.
- Use only the indicator columns provided. No outside knowledge, no
  calendar awareness, no ticker familiarity.
- Treat tickers as opaque identifiers.
- Do not pick top names, build a portfolio, assign weights, write
  rationales, or provide commentary of any kind.

OUTPUT FORMAT (STRICT JSON, no other text, no markdown fences):
{
  "long_scores":  [{"ticker": "SEC_0001", "score": 72}, {"ticker": "SEC_0002", "score": 58}],
  "short_scores": [{"ticker": "SEC_0061", "score": 84}, {"ticker": "SEC_0062", "score": 61}]
}
""".strip()

# Archived standalone prompt from smart_prompt_search_20260802 candidate G0_05.
G0_05_SYSTEM_PROMPT = """
MEAN-REVERSION CANDIDATE SCORING

CONTEXT
We are evaluating two pools of stocks that have experienced extreme 5-day moves, identifying those most likely to partially reverse over the next 5 trading days based on exhaustion, positioning, and regime signals.

FEATURE GLOSSARY

All values are computed using data available as of the rebalance date.
Scale note:
- MRL/MRS are 0-100 cross-sectional percentile scores.
- Trail5/MRSig are shown as percent returns.

Ticker
- Security identifier. In anonymized runs, this is a synthetic SEC_#### ID.

Trail5
- Stock's trailing 5-trading-day return, in percent.

MRSig
- Mean-reversion signal, equal to negative trailing-5-day return.

MRL
- Long-side extreme-move percentile.
- Computed as the cross-sectional percentile rank of MRSig across the valid universe.
- Higher MRL means higher MRSig, usually a bigger recent loser.

MRS
- Short-side extreme-move percentile.
- Computed as 100 - MRL.
- Higher MRS usually means a bigger recent winner.

RSI5
- 5-day RSI computed from closing prices.

Rng55
- Current close location inside the trailing 55-day high-low range.
- Approximately 0 = trailing 55-day low; approximately 1 = trailing 55-day high.

Vel/V
- Recent 10-trading-day return divided by recent 21-trading-day annualized volatility.

%MA100
- Percent distance from the 100-day moving average.

52WH
- Percent distance from trailing 52-week high.

Mom3m
- Trailing roughly 3-month return, in percent.

AutoC
- Autocorrelation of recent daily log returns at lag 5.
- Computed as the correlation between each recent daily return and the daily return 5 trading days earlier.

ADX
- 14-day Average Directional Index, trend strength only.

VolPctl
- Current stock realized-volatility percentile over 504 trading days. Higher = stock is in a high-vol regime.

Market Regime
- normal: Mid/default volatility regime.
- crisis: High volatility and SPY below 200-day MA.
- squeeze: High volatility and SPY above 200-day MA.
- grind_up: Low volatility and SPY above 200-day MA.
- slow_bleed: Low volatility and SPY below 200-day MA.

Important: MRL, MRS, MRSig, and Trail5 are mechanically linked views of the same recent 5-trading-day move, not four independent signals.

TASK
You are scoring two pools of stocks selected as mean-reversion candidates
over the next 5 trading days.

  LONG POOL  = the biggest 5-day losers in the Russell 1000 candidate universe.
               Looking for snap-back / oversold-bounce setups.

  SHORT POOL = the biggest 5-day winners in the Russell 1000 candidate universe.
               Looking for fade / overbought-reversal setups.

For each ticker in each pool, output a single SCORE (0-100):

  LONG POOL:
    100 = strongest snap-back setup (most oversold, most exhausted seller,
          most primed for an upward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest snap-back setup (least likely to bounce despite the move)

  SHORT POOL:
    100 = strongest fade setup (most overbought, most exhausted buyer,
          most primed for a downward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest fade setup (least likely to reverse despite the move)

RULES
- Score every ticker in both pools. Do not skip any.
- Use the 0-100 range only when the supplied cross-section supports it. Do not manufacture separation.
- Use only the indicator columns provided. No outside knowledge, no
  calendar awareness, no ticker familiarity.
- Treat tickers as opaque identifiers.
- Do not pick top names, build a portfolio, assign weights, write
  rationales, or provide commentary of any kind.

OUTPUT FORMAT (STRICT JSON, no other text, no markdown fences):
{
  "long_scores":  [{"ticker": "SEC_0001", "score": 72}, {"ticker": "SEC_0002", "score": 58}],
  "short_scores": [{"ticker": "SEC_0061", "score": 84}, {"ticker": "SEC_0062", "score": 61}]
}
""".strip()

# Unambiguous alias: this is the *old* smart_prompt_search_20260802 G0_05
# de-prescribed-context candidate, not the later forced-top-feature G0_05.
G0_05_DEPRESCRIBED_CONTEXT_SYSTEM_PROMPT = G0_05_SYSTEM_PROMPT

# Development winner from prompt_search_ticker_optional_resumed_20260806.
# Candidate G0_01; prompt SHA-256: 8cb4e396a72fde7ebfa2477961de5e8ab474ed8c9607f405acc715d047cae77e.
G0_01_SYSTEM_PROMPT = """
MEAN-REVERSION CANDIDATE SCORING

CONTEXT
Mean reversion is the tendency of stocks that have moved sharply in one
direction to partially reverse over the next 5 trading days, especially
when the move shows signs of exhaustion (extreme indicators, volume
climax, range positioning). The strongest reversion setups combine an
extreme recent move with exhaustion signals and a supportive regime.

FEATURE GLOSSARY

All values are computed using data available as of the rebalance date.
Scale note:
- MRL/MRS are 0-100 cross-sectional percentile scores.
- Trail5/MRSig are shown as percent returns.

Ticker
- Security identifier. In anonymized runs, this is a synthetic SEC_#### ID.

Trail5
- Stock's trailing 5-trading-day return, in percent.

MRSig
- Mean-reversion signal, equal to negative trailing-5-day return.

MRL
- Long-side extreme-move percentile.
- Computed as the cross-sectional percentile rank of MRSig across the valid universe.
- Higher MRL means higher MRSig, usually a bigger recent loser.

MRS
- Short-side extreme-move percentile.
- Computed as 100 - MRL.
- Higher MRS usually means a bigger recent winner.

RSI5
- 5-day RSI computed from closing prices.

Rng55
- Current close location inside the trailing 55-day high-low range.
- Approximately 0 = trailing 55-day low; approximately 1 = trailing 55-day high.

Vel/V
- Recent 10-trading-day return divided by recent 21-trading-day annualized volatility.

%MA100
- Percent distance from the 100-day moving average.

52WH
- Percent distance from trailing 52-week high.

Mom3m
- Trailing roughly 3-month return, in percent.

AutoC
- Autocorrelation of recent daily log returns at lag 5.
- Computed as the correlation between each recent daily return and the daily return 5 trading days earlier.

ADX
- 14-day Average Directional Index, trend strength only.

VolPctl
- Current stock realized-volatility percentile over 504 trading days. Higher = stock is in a high-vol regime.

Market Regime
- normal: Mid/default volatility regime.
- crisis: High volatility and SPY below 200-day MA.
- squeeze: High volatility and SPY above 200-day MA.
- grind_up: Low volatility and SPY above 200-day MA.
- slow_bleed: Low volatility and SPY below 200-day MA.

Important: MRL, MRS, MRSig, and Trail5 are mechanically linked views of the same recent 5-trading-day move, not four independent signals.

TASK
You are scoring two pools of stocks selected as mean-reversion candidates
over the next 5 trading days.

  LONG POOL  = the biggest 5-day losers in the Russell 1000 candidate universe.
               Looking for snap-back / oversold-bounce setups.

  SHORT POOL = the biggest 5-day winners in the Russell 1000 candidate universe.
               Looking for fade / overbought-reversal setups.

For each ticker in each pool, output a single SCORE (0-100):

  LONG POOL:
    100 = strongest snap-back setup (most oversold, most exhausted seller,
          most primed for an upward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest snap-back setup (least likely to bounce despite the move)

  SHORT POOL:
    100 = strongest fade setup (most overbought, most exhausted buyer,
          most primed for a downward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest fade setup (least likely to reverse despite the move)

RULES
- Score every ticker in both pools. Do not skip any.
- Use the full 0-100 range. Two tickers that differ on a signal must receive distinct scores proportional to the signal's magnitude.
- Use only the indicator columns provided. No outside knowledge, no
  calendar awareness, no ticker familiarity.
- Treat tickers as opaque identifiers.
- Do not pick top names, build a portfolio, assign weights, write
  rationales, or provide commentary of any kind.

OUTPUT FORMAT (STRICT JSON, no other text, no markdown fences):
{
  "long_scores":  [{"ticker": "SEC_0001", "score": 72}, {"ticker": "SEC_0002", "score": 58}],
  "short_scores": [{"ticker": "SEC_0061", "score": 84}, {"ticker": "SEC_0062", "score": 61}]
}
""".strip()

# Development candidate from prompt_search_ticker_optional_resumed_20260806.
# This is kept as a research candidate, not silently promoted to incumbent.
G0_01_PROPORTIONAL_SEPARATION_SYSTEM_PROMPT = G0_01_SYSTEM_PROMPT

_G0_02_EXPECTED_MAGNITUDE_OLD_TEXT = """For each ticker in each pool, output a single SCORE (0-100):

  LONG POOL:
    100 = strongest snap-back setup (most oversold, most exhausted seller,
          most primed for an upward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest snap-back setup (least likely to bounce despite the move)

  SHORT POOL:
    100 = strongest fade setup (most overbought, most exhausted buyer,
          most primed for a downward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest fade setup (least likely to reverse despite the move)"""
_G0_02_EXPECTED_MAGNITUDE_NEW_TEXT = """For each ticker in each pool, output a single SCORE (0-100) that estimates the expected magnitude of the next-5-day return (in basis points) given the current setup, scaled to a percentile rank within the pool. Use the indicator columns to form a conditional expectation. Do not use a fixed 50 as an 'average' reference; let the distribution of scores naturally reflect the cross-section of setups."""
if V2_MINIMAL_CHANGE_CONTROL_PROMPT.count(_G0_02_EXPECTED_MAGNITUDE_OLD_TEXT) != 1:
    raise RuntimeError("Could not materialize the archived 20260806 G0_02 prompt")
G0_02_EXPECTED_MAGNITUDE_SYSTEM_PROMPT = V2_MINIMAL_CHANGE_CONTROL_PROMPT.replace(
    _G0_02_EXPECTED_MAGNITUDE_OLD_TEXT,
    _G0_02_EXPECTED_MAGNITUDE_NEW_TEXT,
    1,
)

# Historical negative cross: 20260806 G0_01 proportional separation combined
# with the *later* forced-top-feature G0_05.  It is not the requested cross
# with the old de-prescribed-context G0_05 above.
G0_01_X_G0_05_SYSTEM_PROMPT = """
MEAN-REVERSION CANDIDATE SCORING

CONTEXT
Mean reversion is the tendency of stocks that have moved sharply in one
direction to partially reverse over the next 5 trading days, especially
when the move shows signs of exhaustion (extreme indicators, volume
climax, range positioning). The strongest reversion setups combine an
extreme recent move with exhaustion signals and a supportive regime.

FEATURE GLOSSARY

All values are computed using data available as of the rebalance date.
Scale note:
- MRL/MRS are 0-100 cross-sectional percentile scores.
- Trail5/MRSig are shown as percent returns.

Ticker
- Security identifier. In anonymized runs, this is a synthetic SEC_#### ID.

Trail5
- Stock's trailing 5-trading-day return, in percent.

MRSig
- Mean-reversion signal, equal to negative trailing-5-day return.

MRL
- Long-side extreme-move percentile.
- Computed as the cross-sectional percentile rank of MRSig across the valid universe.
- Higher MRL means higher MRSig, usually a bigger recent loser.

MRS
- Short-side extreme-move percentile.
- Computed as 100 - MRL.
- Higher MRS usually means a bigger recent winner.

RSI5
- 5-day RSI computed from closing prices.

Rng55
- Current close location inside the trailing 55-day high-low range.
- Approximately 0 = trailing 55-day low; approximately 1 = trailing 55-day high.

Vel/V
- Recent 10-trading-day return divided by recent 21-trading-day annualized volatility.

%MA100
- Percent distance from the 100-day moving average.

52WH
- Percent distance from trailing 52-week high.

Mom3m
- Trailing roughly 3-month return, in percent.

AutoC
- Autocorrelation of recent daily log returns at lag 5.
- Computed as the correlation between each recent daily return and the daily return 5 trading days earlier.

ADX
- 14-day Average Directional Index, trend strength only.

VolPctl
- Current stock realized-volatility percentile over 504 trading days. Higher = stock is in a high-vol regime.

Market Regime
- normal: Mid/default volatility regime.
- crisis: High volatility and SPY below 200-day MA.
- squeeze: High volatility and SPY above 200-day MA.
- grind_up: Low volatility and SPY above 200-day MA.
- slow_bleed: Low volatility and SPY below 200-day MA.

Important: MRL, MRS, MRSig, and Trail5 are mechanically linked views of the same recent 5-trading-day move, not four independent signals.

TASK
You are scoring two pools of stocks selected as mean-reversion candidates
over the next 5 trading days.

  LONG POOL  = the biggest 5-day losers in the Russell 1000 candidate universe.
               Looking for snap-back / oversold-bounce setups.

  SHORT POOL = the biggest 5-day winners in the Russell 1000 candidate universe.
               Looking for fade / overbought-reversal setups.

For each ticker in each pool, output a single SCORE (0-100):

  LONG POOL:
    100 = strongest snap-back setup (most oversold, most exhausted seller,
          most primed for an upward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest snap-back setup (least likely to bounce despite the move)

  SHORT POOL:
    100 = strongest fade setup (most overbought, most exhausted buyer,
          most primed for a downward reversion over the next 5 trading days)
    50  = average candidate in this pool
    0   = weakest fade setup (least likely to reverse despite the move)

RULES
- Score every ticker in both pools. Do not skip any.
- Use the full 0-100 range. Two tickers that differ on a signal must receive distinct scores proportional to the signal's magnitude.
- Use only the indicator columns provided. No outside knowledge, no
  calendar awareness, no ticker familiarity.
- Treat tickers as opaque identifiers.
- Do not pick top names, build a portfolio, assign weights, write
  rationales, or provide commentary of any kind.

OUTPUT FORMAT (STRICT JSON, no other text, no markdown fences):
{
  "long_scores":  [{"ticker": "SEC_0001", "score": 72}, {"ticker": "SEC_0002", "score": 58}],
  "short_scores": [{"ticker": "SEC_0061", "score": 84}, {"ticker": "SEC_0062", "score": 61}]
}

Additionally, inside the JSON, for each pool include a "top_feature" string field indicating which single feature most influenced the top-ranked and bottom-ranked scores. For example, "long_scores_top_feature": "RSI5 oversold", "short_scores_top_feature": "Vel/V exhaustion". This diagnostic does not change the scoring, only improves the internal reasoning by forcing explicit feature evaluation.
""".strip()

G0_01_X_FORCED_TOP_FEATURE_SYSTEM_PROMPT = G0_01_X_G0_05_SYSTEM_PROMPT

_DEPRESCRIBED_CONTEXT_OLD = """CONTEXT
Mean reversion is the tendency of stocks that have moved sharply in one
direction to partially reverse over the next 5 trading days, especially
when the move shows signs of exhaustion (extreme indicators, volume
climax, range positioning). The strongest reversion setups combine an
extreme recent move with exhaustion signals and a supportive regime."""
_DEPRESCRIBED_CONTEXT_NEW = """CONTEXT
We are evaluating two pools of stocks that have experienced extreme 5-day moves, identifying those most likely to partially reverse over the next 5 trading days based on exhaustion, positioning, and regime signals."""
_PROPORTIONAL_RULE_OLD = (
    "- Use the 0-100 range only when the supplied cross-section supports it. "
    "Do not manufacture separation."
)
_PROPORTIONAL_RULE_NEW = (
    "- Use the full 0-100 range. Two tickers that differ on a signal must receive "
    "distinct scores proportional to the signal's magnitude."
)
G0_05_DEPRESCRIBED_X_G0_01_PROPORTIONAL_SYSTEM_PROMPT = (
    V2_MINIMAL_CHANGE_CONTROL_PROMPT.replace(
        _DEPRESCRIBED_CONTEXT_OLD,
        _DEPRESCRIBED_CONTEXT_NEW,
        1,
    ).replace(
        _PROPORTIONAL_RULE_OLD,
        _PROPORTIONAL_RULE_NEW,
        1,
    )
)

# Named, immutable research lineage.  These entries are deliberately separate
# from PROMPT_VARIANTS and the guarded incumbent; promotion requires the full
# predeclared development package and a new blind-validation freeze.
RESEARCH_PROMPT_CANDIDATES = {
    "old_g0_05_deprescribed_context": {
        "prompt": G0_05_DEPRESCRIBED_CONTEXT_SYSTEM_PROMPT,
        "sha256": "384e11633b3135afba73b9b723b46ef629df807a44f0ff5c596800f313583ac0",
        "source": "smart_prompt_search_20260802:G0_05",
    },
    "g0_01_proportional_separation": {
        "prompt": G0_01_PROPORTIONAL_SEPARATION_SYSTEM_PROMPT,
        "sha256": "8cb4e396a72fde7ebfa2477961de5e8ab474ed8c9607f405acc715d047cae77e",
        "source": "prompt_search_ticker_optional_resumed_20260806:G0_01",
    },
    "g0_02_expected_magnitude": {
        "prompt": G0_02_EXPECTED_MAGNITUDE_SYSTEM_PROMPT,
        "sha256": "5345962b8852b806a5347168112eb1a5505b9ec6933136a365346d4f4e551a3b",
        "source": "prompt_search_ticker_optional_resumed_20260806:G0_02",
    },
    "deprescribed_x_proportional": {
        "prompt": G0_05_DEPRESCRIBED_X_G0_01_PROPORTIONAL_SYSTEM_PROMPT,
        "sha256": hashlib.sha256(
            G0_05_DEPRESCRIBED_X_G0_01_PROPORTIONAL_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "source": "predeclared_20260808_prompt_interaction:D",
    },
    "negative_forced_top_feature_cross": {
        "prompt": G0_01_X_FORCED_TOP_FEATURE_SYSTEM_PROMPT,
        "sha256": hashlib.sha256(
            G0_01_X_FORCED_TOP_FEATURE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "source": "v2_prompt_cross_g0_01_g0_05_2009_2014",
    },
}
for _research_prompt_name, _research_prompt in RESEARCH_PROMPT_CANDIDATES.items():
    if hashlib.sha256(_research_prompt["prompt"].encode("utf-8")).hexdigest() != _research_prompt["sha256"]:
        raise RuntimeError(f"Research prompt hash drift: {_research_prompt_name}")


LEGACY_LABELS = {
    "mean_reversion_raw_score": "Legacy Det Raw",
    "llm_raw_confidence_score": "Legacy LLM Raw Confidence",
    "llm_rank_score": "Legacy LLM Rank",
    "llm_rank_confidence_score": "Legacy LLM Rank + Confidence",
}
COMPARISON_LIQUIDITY_FILTERS = (
    f"Russell membership; price >= ${rb.MIN_TRADE_PRICE:g}; "
    f"ADV20 >= ${rb.MIN_ADV20_DOLLARS / 1_000_000:g}M"
)
SPREAD_RETURN_BASIS = "100/100 long-short spread"
CAPITAL_RETURN_BASIS = "50/50 capital-normalized"
CAPITAL_RETURN_SCALE = 0.5


class LLMSafeStop(RuntimeError):
    """A resumable LLM-cache stop with all completed work already persisted."""


class LLMResponseLengthError(RuntimeError):
    """The provider exhausted max_tokens before returning the requested JSON."""


@dataclass(frozen=True)
class ExperimentalPromptSpec:
    candidate_id: str
    generation: str
    parent_id: str
    mutation_mode: str
    old_text: str
    new_text: str
    hypothesis: str
    mechanism: str
    expected_failure: str
    prior_experiment_ids: tuple[str, ...]
    parent_prompt_text: str
    parent_prompt_sha256: str
    prompt_text: str
    prompt_sha256: str
    prompt_hash: str
    output_contract: str
    score_min: float
    score_max: float
    score_direction: str
    anonymize_tickers: bool
    side_scope: str
    replicate_id: str
    spec_sha256: str
    source_path: Path
    raw: dict[str, Any]


@dataclass(frozen=True)
class SignalDateManifest:
    dates: tuple[pd.Timestamp, ...]
    date_manifest_hash: str
    source_sha256: str
    source_path: Path
    raw: Any

    @property
    def date_set(self) -> frozenset[pd.Timestamp]:
        return frozenset(self.dates)


@dataclass
class OutputTokenBudget:
    limit: int
    used: int = 0

    def reserve_request(self, max_output_tokens: int) -> None:
        requested = int(max_output_tokens)
        if self.used + requested > self.limit:
            raise LLMSafeStop(
                "Output-token budget reached before the next request: "
                f"used={self.used:,}, request_cap={requested:,}, budget={self.limit:,}. "
                "The cache is safe to resume with a larger budget."
            )

    def record(self, output_tokens: int) -> None:
        self.used += max(0, int(output_tokens))


@dataclass
class LLMCallBudget:
    limit: int
    used: int = 0

    def reserve_request(self) -> None:
        if self.used >= self.limit:
            raise LLMSafeStop(
                "LLM-call budget reached before the next request: "
                f"used={self.used:,}, budget={self.limit:,}. The cache is safe to resume."
            )
        self.used += 1


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _matches_sha256(value: Any, expected_full_hash: str) -> bool:
    supplied = str(value or "").strip().lower()
    return len(supplied) in {12, 64} and expected_full_hash.startswith(supplied)


def _require_nonempty_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Experimental prompt spec requires non-empty {keys[0]!r}")


def _feature_glossary_block(prompt_text: str) -> str:
    start_marker = "FEATURE GLOSSARY"
    end_marker = "\nTASK\n"
    start = prompt_text.find(start_marker)
    end = prompt_text.find(end_marker, start + len(start_marker))
    if start < 0 or end < 0:
        raise ValueError("Prompt must preserve the FEATURE GLOSSARY followed by TASK")
    return prompt_text[start:end].strip()


def dedicated_side_system_prompt(side_scope: str) -> str:
    """Materialize the frozen one-side parent used by the split prompt experiment."""
    side = str(side_scope).strip().lower()
    if side not in {"long", "short"}:
        raise ValueError("Dedicated prompt side_scope must be long or short")
    if side == "long":
        job = (
            "Your sole job is identifying recent losers most likely to positively "
            "mean-revert over the next 5 trading days."
        )
        pool = "LONG"
        meaning = "100 = strongest expected upward mean-reversion; 0 = weakest."
    else:
        job = (
            "Your sole job is identifying recent winners most likely to negatively "
            "mean-revert over the next 5 trading days."
        )
        pool = "SHORT"
        meaning = "100 = strongest expected downward mean-reversion; 0 = weakest."
    glossary = _feature_glossary_block(V2_MINIMAL_CHANGE_CONTROL_PROMPT)
    return f"""MEAN-REVERSION CANDIDATE SCORING

CONTEXT
{job}

{glossary}

TASK
Score every ticker in the supplied {pool} pool relative to the other names in that pool.
{meaning}
Use only the supplied indicators. Treat tickers as opaque identifiers.
Do not select a portfolio, assign weights, write rationales, or add commentary.

OUTPUT FORMAT (STRICT JSON, NO OTHER TEXT)
{{"scores":[{{"ticker":"SEC_0001","score":72}}]}}
""".strip()


DEDICATED_LONG_SYSTEM_PROMPT = dedicated_side_system_prompt("long")
DEDICATED_SHORT_SYSTEM_PROMPT = dedicated_side_system_prompt("short")


def _score_contract_from_payload(payload: dict[str, Any]) -> tuple[float, float, str]:
    score_min = float(payload.get("score_min", 0.0))
    score_max = float(payload.get("score_max", 100.0))
    direction = str(payload.get("score_direction", "higher_is_better")).strip().lower()
    if not np.isfinite(score_min) or not np.isfinite(score_max) or score_max <= score_min:
        raise ValueError("score_min and score_max must be finite with score_max > score_min")
    if direction not in {"higher_is_better", "lower_is_better"}:
        raise ValueError("score_direction must be higher_is_better or lower_is_better")
    return score_min, score_max, direction


def load_experimental_prompt_spec(path: Path) -> ExperimentalPromptSpec:
    """Load and independently materialize one immutable score-only prompt mutation."""
    source_path = Path(path)
    source_bytes = source_path.read_bytes()
    try:
        payload = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid experimental prompt JSON at {source_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Experimental prompt spec must be a JSON object")

    candidate_id = _require_nonempty_text(payload, "candidate_id")
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in candidate_id):
        raise ValueError("candidate_id may contain only letters, numbers, dot, underscore, and hyphen")
    if candidate_id in {".", ".."} or Path(candidate_id).name != candidate_id:
        raise ValueError("candidate_id must be a safe single path component")
    generation_raw = payload.get("generation")
    generation = str(generation_raw).strip().upper()
    if generation in {"0", "G0"}:
        generation = "G0"
    elif generation in {"1", "G1"}:
        generation = "G1"
    else:
        raise ValueError("generation must be G0/0 or G1/1")
    parent_id = _require_nonempty_text(payload, "parent_id")
    mutation_mode = _require_nonempty_text(payload, "mutation_mode").lower()
    if mutation_mode not in {"additive", "substitutive", "rewrite", "format_repair"}:
        raise ValueError("mutation_mode must be additive, substitutive, rewrite, or format_repair")

    patch = payload.get("patch")
    if not isinstance(patch, dict):
        raise ValueError("Experimental prompt spec requires a patch object")
    operation = str(patch.get("operation", patch.get("op", ""))).strip()
    if operation != "replace_once":
        raise ValueError("Experimental prompt patch operation must be replace_once")
    old_text = patch.get("old_text")
    new_text = patch.get("new_text")
    if not isinstance(old_text, str) or not old_text:
        raise ValueError("replace_once old_text must be a non-empty string")
    if not isinstance(new_text, str) or old_text == new_text:
        raise ValueError("replace_once new_text must be a different string")

    side_scope = str(payload.get("side_scope", "both")).strip().lower()
    if side_scope not in {"both", "long", "short"}:
        raise ValueError("side_scope must be both, long, or short")
    replicate_id = str(payload.get("replicate_id", "r1")).strip()
    if not replicate_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for char in replicate_id
    ):
        raise ValueError("replicate_id must be a non-empty safe identifier")

    parent_prompt_text_raw = payload.get("parent_prompt_text", payload.get("parent_prompt"))
    if parent_prompt_text_raw is None and generation == "G0":
        parent_prompt_text = (
            V2_MINIMAL_CHANGE_CONTROL_PROMPT
            if side_scope == "both"
            else dedicated_side_system_prompt(side_scope)
        )
    elif isinstance(parent_prompt_text_raw, str) and parent_prompt_text_raw.strip():
        parent_prompt_text = parent_prompt_text_raw.strip()
    else:
        raise ValueError("G1 prompt specs must include parent_prompt_text")
    parent_prompt_sha256 = _sha256_bytes(parent_prompt_text.encode("utf-8"))
    supplied_parent_hash = payload.get("parent_prompt_hash", payload.get("parent_prompt_sha256"))
    if supplied_parent_hash is not None and not _matches_sha256(supplied_parent_hash, parent_prompt_sha256):
        raise ValueError("parent_prompt_hash does not match parent_prompt_text")
    if parent_prompt_text.count(old_text) != 1:
        raise ValueError("replace_once old_text must occur exactly once in the parent prompt")

    materialized_prompt = parent_prompt_text.replace(old_text, new_text, 1)
    prompt_text_raw = payload.get("prompt_text", payload.get("materialized_prompt"))
    if not isinstance(prompt_text_raw, str) or not prompt_text_raw.strip():
        raise ValueError("Experimental prompt spec requires prompt_text")
    prompt_text = prompt_text_raw.strip()
    if prompt_text != materialized_prompt:
        raise ValueError("prompt_text does not equal the independently materialized replace_once patch")
    if len(prompt_text) > EXPERIMENTAL_PROMPT_MAX_CHARS:
        raise ValueError(
            f"Experimental prompt exceeds the {EXPERIMENTAL_PROMPT_MAX_CHARS:,}-character safety limit"
        )
    if _feature_glossary_block(prompt_text) != _feature_glossary_block(parent_prompt_text):
        raise ValueError("Experimental mutations may not alter the frozen feature glossary")
    if side_scope == "both":
        if "long_scores" not in prompt_text or "short_scores" not in prompt_text:
            raise ValueError("Two-sided experimental prompts must request long_scores and short_scores arrays")
    elif '"scores"' not in prompt_text:
        raise ValueError("Side-scoped experimental prompts must request one scores array")

    output_contract = str(payload.get("output_contract", "")).strip()
    if output_contract == "score_0_100":
        score_min, score_max, score_direction = 0.0, 100.0, "higher_is_better"
    elif output_contract == AUTORESEARCH_OUTPUT_CONTRACT:
        score_min, score_max, score_direction = _score_contract_from_payload(payload)
    else:
        raise ValueError(f"output_contract must be {AUTORESEARCH_OUTPUT_CONTRACT!r}")
    prompt_sha256 = _sha256_bytes(prompt_text.encode("utf-8"))
    if not _matches_sha256(payload.get("prompt_hash", payload.get("prompt_sha256")), prompt_sha256):
        raise ValueError("prompt_hash does not match prompt_text")

    prior_ids = payload.get("prior_experiment_ids", [])
    if not isinstance(prior_ids, list) or any(not isinstance(item, str) or not item.strip() for item in prior_ids):
        raise ValueError("prior_experiment_ids must be a list of non-empty strings")
    anonymize_tickers = payload.get("anonymize_tickers", True)
    if not isinstance(anonymize_tickers, bool):
        raise ValueError("anonymize_tickers must be a JSON boolean")
    return ExperimentalPromptSpec(
        candidate_id=candidate_id,
        generation=generation,
        parent_id=parent_id,
        mutation_mode=mutation_mode,
        old_text=old_text,
        new_text=new_text,
        hypothesis=_require_nonempty_text(payload, "hypothesis"),
        mechanism=_require_nonempty_text(payload, "mechanism"),
        expected_failure=_require_nonempty_text(payload, "expected_failure", "expected_failure_mode"),
        prior_experiment_ids=tuple(item.strip() for item in prior_ids),
        parent_prompt_text=parent_prompt_text,
        parent_prompt_sha256=parent_prompt_sha256,
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha256,
        prompt_hash=prompt_sha256[:12],
        output_contract=output_contract,
        score_min=score_min,
        score_max=score_max,
        score_direction=score_direction,
        anonymize_tickers=anonymize_tickers,
        side_scope=side_scope,
        replicate_id=replicate_id,
        spec_sha256=_sha256_bytes(source_bytes),
        source_path=source_path.resolve(),
        raw=payload,
    )


def _normalize_signal_date(value: Any) -> pd.Timestamp:
    try:
        date = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid signal date {value!r}") from exc
    if date.tzinfo is not None:
        date = date.tz_convert(None)
    if date != date.normalize():
        raise ValueError(f"Signal date must not contain a time component: {value!r}")
    return date.normalize()


def load_signal_date_manifest(path: Path) -> SignalDateManifest:
    source_path = Path(path)
    source_bytes = source_path.read_bytes()
    try:
        payload = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid signal-date JSON at {source_path}: {exc}") from exc
    if isinstance(payload, list):
        raw_dates = payload
        supplied_hash = None
    elif isinstance(payload, dict):
        raw_dates = payload.get("signal_dates", payload.get("dates"))
        supplied_hash = payload.get("date_manifest_hash", payload.get("manifest_hash"))
    else:
        raise ValueError("Signal-date manifest must be a JSON list or object")
    if not isinstance(raw_dates, list) or not raw_dates:
        raise ValueError("Signal-date manifest requires a non-empty signal_dates list")
    dates = tuple(_normalize_signal_date(value) for value in raw_dates)
    if len(set(dates)) != len(dates):
        raise ValueError("Signal-date manifest contains duplicate dates")
    if dates != tuple(sorted(dates)):
        raise ValueError("Signal-date manifest dates must be sorted chronologically")
    canonical_dates = [str(date.date()) for date in dates]
    date_manifest_hash = _sha256_bytes(_canonical_json_bytes(canonical_dates))
    if supplied_hash is not None and not _matches_sha256(supplied_hash, date_manifest_hash):
        raise ValueError("date_manifest_hash does not match the canonical signal-date list")
    return SignalDateManifest(
        dates=dates,
        date_manifest_hash=date_manifest_hash,
        source_sha256=_sha256_bytes(source_bytes),
        source_path=source_path.resolve(),
        raw=payload,
    )


def append_progress_event(progress_file: Path | None, event: str, **fields: Any) -> None:
    if progress_file is None:
        return
    path = Path(progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "artifact_type": "llm_reversal_signal_v2_progress",
        "event": str(event),
        "recorded_at": pd.Timestamp.now(tz="UTC").isoformat(),
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _cached_llm_calls(cache: dict[str, Any]) -> int:
    return sum(1 for row in cache.get("llm_call_usage", []) or [] if isinstance(row, dict))


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def autoresearch_code_fingerprint() -> str:
    files = {
        "mean_reversion_llm_v2.py": Path(__file__),
        "mean_reversion_llm_v1.py": Path(v1.__file__),
        "run_backtest_v3.py": Path(rb.__file__),
        "technical_llm.py": Path(technical_llm.__file__),
    }
    payload = [(name, _file_sha256(path)) for name, path in sorted(files.items())]
    return _sha256_bytes(_canonical_json_bytes(payload))


def autoresearch_harness_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": str(args.model),
        "temperature": float(args.temperature),
        "thinking_mode": str(args.thinking_mode),
        "anonymize_tickers": bool(args.anonymize_tickers),
        "schedule": str(args.schedule),
        "anchor_offsets": list(args.anchor_offsets),
        "pool_sizes": list(args.pool_sizes),
        "book_sizes": list(args.book_sizes),
        "llm_pool_size": int(args.llm_pool_size),
        "cost_bps_levels": [float(value) for value in args.cost_bps_levels],
        "signal_lookback_days": 5,
        "holding_days": 5,
        "long_short_capital_split": {"long": 0.5, "short": 0.5},
        "within_side_weighting": "equal",
        "return_convention": "gross_fixed_horizon_c2c",
        "output_contract": AUTORESEARCH_OUTPUT_CONTRACT,
        "side_scope": str(getattr(args, "side_scope", "both")),
        "replicate_id": str(getattr(args, "replicate_id", "r1")),
    }


def autoresearch_harness_fingerprint(args: argparse.Namespace) -> str:
    return _sha256_bytes(_canonical_json_bytes(autoresearch_harness_manifest(args)))


def validate_autoresearch_args(args: argparse.Namespace) -> None:
    errors: list[str] = []
    if not bool(args.experimental_override):
        errors.append("--experimental-override is required")
    if args.phase not in {"llm-cache", "full"}:
        errors.append("autoresearch runs require --phase llm-cache or full")
    if str(args.model) != AUTORESEARCH_MODEL:
        errors.append(f"model must be {AUTORESEARCH_MODEL}")
    if not np.isclose(float(args.temperature), AUTORESEARCH_TEMPERATURE):
        errors.append(f"temperature must be {AUTORESEARCH_TEMPERATURE:g}")
    if str(args.thinking_mode) != AUTORESEARCH_THINKING_MODE:
        errors.append("thinking mode must be disabled")
    if tuple(args.pool_sizes) != AUTORESEARCH_POOL_SIZES:
        errors.append(f"pool sizes must be {AUTORESEARCH_POOL_SIZES}")
    if tuple(args.book_sizes) != AUTORESEARCH_BOOK_SIZES:
        errors.append(f"book sizes must be {AUTORESEARCH_BOOK_SIZES}")
    if int(args.llm_pool_size) != AUTORESEARCH_LLM_POOL_SIZE:
        errors.append(f"LLM pool size must be {AUTORESEARCH_LLM_POOL_SIZE}")
    if args.schedule != "anchors" or tuple(args.anchor_offsets) != AUTORESEARCH_ANCHOR_OFFSETS:
        errors.append("schedule must be anchors with anchor offset 3")
    if tuple(float(value) for value in args.cost_bps_levels) != AUTORESEARCH_COST_BPS_LEVELS:
        errors.append("cost basis must be gross only (--cost-bps-levels 0)")
    if int(args.rationale_top_k_per_side) != 0 or bool(args.structured_reasoning):
        errors.append("rationale and structured-reasoning outputs must be disabled")
    if bool(args.print_reasoning_content):
        errors.append("reasoning_content printing/storage must be disabled")
    if errors:
        raise ValueError("Autoresearch configuration guard rejected this run: " + "; ".join(errors))


def validate_autoresearch_manifest_scope(
    manifest: SignalDateManifest,
    config: rb.BenchmarkConfig,
) -> str:
    if any(date < config.start or date > config.end for date in manifest.dates):
        raise ValueError("Every signal-date manifest entry must fall within --start/--end")
    if any(date >= AUTORESEARCH_HOLDOUT_START for date in manifest.dates):
        raise ValueError("Autoresearch cannot access signal dates in 2020 or later")
    has_development = any(date < AUTORESEARCH_VALIDATION_START for date in manifest.dates)
    has_validation = any(date >= AUTORESEARCH_VALIDATION_START for date in manifest.dates)
    if has_development and has_validation:
        raise ValueError("A signal-date manifest cannot mix development and validation dates")
    stage = "validation" if has_validation else "development"
    if stage == "development" and any(
        date < AUTORESEARCH_DEVELOPMENT_START or date > AUTORESEARCH_DEVELOPMENT_END
        for date in manifest.dates
    ):
        raise ValueError("Development signal dates must stay within 2009-01-01 through 2014-01-01")
    if stage == "validation" and any(date > AUTORESEARCH_VALIDATION_END for date in manifest.dates):
        raise ValueError("Validation signal dates must stay within 2015-01-01 through 2019-12-31")
    return stage


def validate_pool_date_coverage(
    pools: list[v1.CandidatePool] | dict[str, v1.CandidatePool],
    allowed_signal_dates: frozenset[pd.Timestamp],
) -> None:
    values = pools.values() if isinstance(pools, dict) else pools
    actual = {pd.Timestamp(pool.signal_date).normalize() for pool in values}
    if actual != set(allowed_signal_dates):
        missing = sorted(set(allowed_signal_dates) - actual)
        extra = sorted(actual - set(allowed_signal_dates))
        raise ValueError(
            "Exact signal-date coverage failed: "
            f"missing={[str(date.date()) for date in missing]}, "
            f"extra={[str(date.date()) for date in extra]}"
        )


def validate_autoresearch_pool_dates(
    pools: list[v1.CandidatePool] | dict[str, v1.CandidatePool],
    stage: str,
) -> None:
    values = pools.values() if isinstance(pools, dict) else pools
    for pool in values:
        signal_date = pd.Timestamp(pool.signal_date).normalize()
        exit_date = pd.Timestamp(pool.exit_date).normalize()
        if signal_date >= AUTORESEARCH_HOLDOUT_START or exit_date >= AUTORESEARCH_HOLDOUT_START:
            raise ValueError(
                "Autoresearch cannot access signal or exit dates in 2020 or later: "
                f"signal={signal_date.date()}, exit={exit_date.date()}"
            )
        if stage == "development" and (
            signal_date < AUTORESEARCH_DEVELOPMENT_START
            or signal_date > AUTORESEARCH_DEVELOPMENT_END
            or exit_date > AUTORESEARCH_DEVELOPMENT_END
        ):
            raise ValueError("Development signal/exit dates must stay within 2009-01-01 through 2014-01-01")
        if stage == "validation" and (
            signal_date < AUTORESEARCH_VALIDATION_START
            or signal_date > AUTORESEARCH_VALIDATION_END
            or exit_date > AUTORESEARCH_VALIDATION_END
        ):
            raise ValueError("Validation signal/exit dates must stay within 2015-01-01 through 2019-12-31")


def load_and_validate_validation_freeze(
    path: Path,
    prompt_spec: ExperimentalPromptSpec,
    date_manifest: SignalDateManifest,
    args: argparse.Namespace,
) -> dict[str, Any]:
    freeze_path = Path(path)
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Validation freeze must be a JSON object")
    if payload.get("approved") is not True and str(payload.get("approval_status", "")).lower() != "approved":
        raise ValueError("Validation freeze does not contain explicit approval")
    allowed_specs = payload.get("allowed_prompt_specs")
    matched_allowed: dict[str, Any] | None = None
    if isinstance(allowed_specs, list):
        for allowed in allowed_specs:
            if not isinstance(allowed, dict):
                continue
            if (
                str(allowed.get("candidate_id", "")) == prompt_spec.candidate_id
                and str(allowed.get("side_scope", "both")) == prompt_spec.side_scope
                and str(allowed.get("replicate_id", "r1")) == prompt_spec.replicate_id
                and _matches_sha256(allowed.get("prompt_hash"), prompt_spec.prompt_sha256)
                and _matches_sha256(allowed.get("prompt_spec_hash"), prompt_spec.spec_sha256)
            ):
                matched_allowed = allowed
                break
        if matched_allowed is None:
            raise ValueError("Prompt spec is not in the validation freeze allowlist")
    elif str(payload.get("candidate_id", "")) != prompt_spec.candidate_id:
        raise ValueError("Validation freeze candidate_id does not match the prompt spec")
    if payload.get("anonymize_tickers", True) is not prompt_spec.anonymize_tickers:
        raise ValueError("Validation freeze ticker visibility does not match the prompt spec")
    if str(payload.get("side_scope", "both")) != prompt_spec.side_scope:
        raise ValueError("Validation freeze side_scope does not match the prompt spec")
    if matched_allowed is None and str(payload.get("replicate_id", "r1")) != prompt_spec.replicate_id:
        raise ValueError("Validation freeze replicate_id does not match the prompt spec")
    if matched_allowed is None and not _matches_sha256(
        payload.get("prompt_hash", payload.get("prompt_sha256")), prompt_spec.prompt_sha256
    ):
        raise ValueError("Validation freeze prompt hash does not match the prompt spec")
    if not _matches_sha256(payload.get("date_manifest_hash"), date_manifest.date_manifest_hash):
        raise ValueError("Validation freeze date-manifest hash does not match")
    if not _matches_sha256(payload.get("date_manifest_file_hash"), date_manifest.source_sha256):
        raise ValueError("Validation freeze date-manifest file hash does not match")
    if matched_allowed is None and not _matches_sha256(payload.get("prompt_spec_hash"), prompt_spec.spec_sha256):
        raise ValueError("Validation freeze prompt-spec hash does not match")
    if not str(payload.get("campaign_id", "")).strip():
        raise ValueError("Validation freeze requires a campaign_id")
    lineage = payload.get("lineage")
    if not isinstance(lineage, list) or not lineage or not isinstance(lineage[0], dict):
        raise ValueError("Validation freeze requires frozen candidate lineage")
    allowed_role = str((matched_allowed or {}).get("role", "winner"))
    if allowed_role == "winner" and (
        str(lineage[0].get("candidate_id", "")) != prompt_spec.candidate_id
        or not _matches_sha256(lineage[0].get("prompt_hash"), prompt_spec.prompt_sha256)
    ):
        raise ValueError("Validation freeze lineage does not match the candidate prompt")
    lineage_hash = _sha256_bytes(_canonical_json_bytes(lineage))
    if not _matches_sha256(payload.get("lineage_hash"), lineage_hash):
        raise ValueError("Validation freeze lineage hash does not match")
    supervisor_fingerprint = str(payload.get("supervisor_code_fingerprint", ""))
    if len(supervisor_fingerprint) not in {12, 64}:
        raise ValueError("Validation freeze requires a supervisor_code_fingerprint")
    expected_code = autoresearch_code_fingerprint()
    if not _matches_sha256(payload.get("code_fingerprint", payload.get("runner_code_fingerprint")), expected_code):
        raise ValueError("Validation freeze code fingerprint does not match the current runner/dependencies")
    expected_harness = autoresearch_harness_fingerprint(args)
    frozen_harness = (matched_allowed or {}).get(
        "harness_fingerprint", payload.get("harness_fingerprint")
    )
    if not _matches_sha256(frozen_harness, expected_harness):
        raise ValueError("Validation freeze harness fingerprint does not match")
    development_hash = str(payload.get("development_results_hash", "")).strip()
    if len(development_hash) not in {12, 64}:
        raise ValueError("Validation freeze requires a development_results_hash")
    development_path = payload.get("development_results_path")
    if not development_path:
        raise ValueError("Validation freeze requires a development_results_path")
    result_path = Path(str(development_path))
    if not result_path.is_absolute():
        result_path = freeze_path.parent / result_path
    if not result_path.is_file():
        raise ValueError("Validation freeze development-results artifact does not exist")
    actual_results_hash = _file_sha256(result_path)
    if not _matches_sha256(development_hash, actual_results_hash):
        raise ValueError("Validation freeze development-results hash does not match its artifact")
    return {
        **payload,
        "freeze_path": str(freeze_path.resolve()),
        "freeze_sha256": _file_sha256(freeze_path),
    }


def _provider_output_tokens(usage: dict[str, Any] | None, fallback: int = 0) -> int:
    usage = usage if isinstance(usage, dict) else {}
    for key in ("completion_tokens", "output_tokens"):
        if key not in usage:
            continue
        try:
            value = int(usage.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return max(0, int(fallback))


def _cached_output_tokens(cache: dict[str, Any]) -> int:
    total = 0
    for row in cache.get("llm_call_usage", []) or []:
        if isinstance(row, dict):
            total += max(0, int(row.get("output_tokens_charged", 0) or 0))
    return total


def _shared_autoresearch_budgets(
    cache_paths: Sequence[Path],
    *,
    call_limit: int,
    output_token_limit: int,
) -> tuple[LLMCallBudget, OutputTokenBudget]:
    """Build one cumulative budget across treatment and paired-control caches."""
    calls_used = 0
    output_tokens_used = 0
    seen: set[Path] = set()
    for raw_path in cache_paths:
        path = Path(raw_path).resolve()
        if path in seen:
            continue
        seen.add(path)
        cache = v1.load_cache(path)
        calls_used += _cached_llm_calls(cache)
        output_tokens_used += _cached_output_tokens(cache)
    if calls_used > int(call_limit):
        raise LLMSafeStop(
            f"Existing paired caches already use {calls_used:,} calls, above the {int(call_limit):,} limit"
        )
    if output_tokens_used > int(output_token_limit):
        raise LLMSafeStop(
            "Existing paired caches already exceed the output-token budget: "
            f"used={output_tokens_used:,}, budget={int(output_token_limit):,}"
        )
    return (
        LLMCallBudget(limit=int(call_limit), used=calls_used),
        OutputTokenBudget(limit=int(output_token_limit), used=output_tokens_used),
    )


def _is_retryable_llm_error(exc: Exception) -> bool:
    if isinstance(exc, LLMResponseLengthError):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        status = int(exc.response.status_code)
        return status in {408, 409, 429} or status >= 500
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, (ValueError, json.JSONDecodeError))


def _is_format_only_score_payload(
    raw_output: Any,
    expected_long: Sequence[str],
    expected_short: Sequence[str],
    score_min: float = 0.0,
    score_max: float = 100.0,
) -> bool:
    """Return true only when all numeric scores exist but the JSON shape is wrong.

    This deliberately records only a boolean diagnostic. Raw scores never leave the
    runner and therefore cannot become researcher feedback.
    """
    expected_by_ticker = {
        **{str(ticker).strip().upper(): "LONG" for ticker in expected_long},
        **{str(ticker).strip().upper(): "SHORT" for ticker in expected_short},
    }
    found: list[tuple[str, float, str | None]] = []

    def visit(value: Any, side_hint: str | None = None) -> None:
        if isinstance(value, dict):
            normalized = {str(key).strip().lower(): item for key, item in value.items()}
            ticker_value = next(
                (
                    normalized[key]
                    for key in ("ticker", "symbol", "security", "security_id", "id")
                    if key in normalized
                ),
                None,
            )
            score_value = next(
                (
                    normalized[key]
                    for key in ("score", "value", "rating")
                    if key in normalized
                ),
                None,
            )
            if ticker_value is not None and score_value is not None:
                try:
                    numeric = float(score_value)
                except (TypeError, ValueError):
                    numeric = np.nan
                found.append((str(ticker_value).strip().upper(), numeric, side_hint))
            for key, item in value.items():
                key_text = str(key).lower()
                child_hint = side_hint
                if "long" in key_text:
                    child_hint = "LONG"
                elif "short" in key_text:
                    child_hint = "SHORT"
                visit(item, child_hint)
        elif isinstance(value, list):
            for item in value:
                visit(item, side_hint)

    visit(raw_output)
    if len(found) != len(expected_by_ticker):
        return False
    tickers = [ticker for ticker, _, _ in found]
    if len(set(tickers)) != len(tickers) or set(tickers) != set(expected_by_ticker):
        return False
    for ticker, score, side_hint in found:
        if not np.isfinite(score) or score < float(score_min) or score > float(score_max):
            return False
        if side_hint is not None and side_hint != expected_by_ticker[ticker]:
            return False
    return True


def validate_and_canonicalize_autoresearch_scores(
    raw_output: Any,
    expected_longs: Sequence[str],
    expected_shorts: Sequence[str],
    *,
    score_min: float,
    score_max: float,
    score_direction: str,
) -> dict[str, Any]:
    """Validate a candidate's native score contract and map it to 0-100/high-is-best."""
    if not isinstance(raw_output, dict):
        raise ValueError("Autoresearch score output must be a JSON object")
    expected = {
        "long_scores": {str(value).strip().upper() for value in expected_longs},
        "short_scores": {str(value).strip().upper() for value in expected_shorts},
    }
    canonical: dict[str, list[dict[str, Any]]] = {}
    native: dict[str, list[dict[str, Any]]] = {}
    span = float(score_max) - float(score_min)
    for key, expected_tickers in expected.items():
        rows = raw_output.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"Autoresearch output requires a {key} array")
        cleaned: list[dict[str, Any]] = []
        native_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Every {key} entry must be an object")
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker or ticker in seen:
                raise ValueError(f"Every {key} ticker must be present exactly once")
            try:
                native_score = float(row.get("score"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Every {key} score must be numeric") from exc
            if not np.isfinite(native_score) or native_score < score_min or native_score > score_max:
                raise ValueError(
                    f"Score {native_score!r} is outside declared range [{score_min}, {score_max}]"
                )
            normalized = (native_score - float(score_min)) / span
            if score_direction == "lower_is_better":
                normalized = 1.0 - normalized
            canonical_score = float(np.clip(100.0 * normalized, 0.0, 100.0))
            cleaned.append({"ticker": ticker, "score": canonical_score})
            native_rows.append({"ticker": ticker, "score": native_score})
            seen.add(ticker)
        if seen != expected_tickers:
            missing = sorted(expected_tickers - seen)
            extra = sorted(seen - expected_tickers)
            raise ValueError(f"{key} ticker coverage mismatch: missing={missing}, extra={extra}")
        canonical[key] = cleaned
        native[key] = native_rows
    validated = v1.validate_llm_scores(
        canonical,
        expected_longs=list(expected_longs),
        expected_shorts=list(expected_shorts),
        prompt_style=V2_PROMPT_STYLE,
    )
    return {
        **validated,
        "native_scores": native,
        "native_score_contract": {
            "score_min": float(score_min),
            "score_max": float(score_max),
            "score_direction": score_direction,
        },
    }


def validate_and_canonicalize_autoresearch_side_scores(
    raw_output: Any,
    expected_tickers: Sequence[str],
    *,
    side_scope: str,
    score_min: float,
    score_max: float,
    score_direction: str,
) -> dict[str, Any]:
    """Validate one side-only response and map its native scores to 0-100/high-is-best."""
    side = str(side_scope).strip().lower()
    if side not in {"long", "short"}:
        raise ValueError("Side score validation requires side_scope long or short")
    if not isinstance(raw_output, dict) or not isinstance(raw_output.get("scores"), list):
        raise ValueError("Side-scoped output requires exactly one scores array")
    if set(raw_output) != {"scores"}:
        raise ValueError("Side-scoped output may contain only the scores top-level key")
    expected = {str(ticker).strip().upper() for ticker in expected_tickers}
    seen: set[str] = set()
    native_rows: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] = []
    span = float(score_max) - float(score_min)
    for row in raw_output["scores"]:
        if not isinstance(row, dict) or set(row) != {"ticker", "score"}:
            raise ValueError("Every side score must contain exactly ticker and score")
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker or ticker in seen or ticker not in expected:
            raise ValueError(f"Invalid, duplicate, or wrong-side ticker {ticker!r}")
        try:
            native_score = float(row.get("score"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{ticker} has a non-numeric score") from exc
        if not np.isfinite(native_score) or native_score < score_min or native_score > score_max:
            raise ValueError(f"{ticker} score is outside [{score_min}, {score_max}]")
        normalized = (native_score - float(score_min)) / span
        if score_direction == "lower_is_better":
            normalized = 1.0 - normalized
        native_rows.append({"ticker": ticker, "score": native_score})
        canonical_rows.append({"ticker": ticker, "score": float(np.clip(100.0 * normalized, 0.0, 100.0))})
        seen.add(ticker)
    if seen != expected or len(canonical_rows) != len(expected):
        raise ValueError(
            f"Side score ticker coverage mismatch: missing={sorted(expected - seen)}, "
            f"extra={sorted(seen - expected)}"
        )
    return {
        "scores": canonical_rows,
        "native_scores": native_rows,
        "native_score_contract": {
            "score_min": float(score_min),
            "score_max": float(score_max),
            "score_direction": score_direction,
        },
        "side_scope": side,
    }


def _model_supports_deepseek_json_mode(model: str) -> bool:
    """The legacy reasoner alias has a different output contract; V4 content can still use JSON mode."""
    return str(model or "").strip() != "deepseek-reasoner"


def _harden_json_payload(
    payload: dict[str, Any],
    model: str,
    thinking_mode: str = technical_llm.DEFAULT_THINKING_MODE,
    max_output_tokens: int | None = None,
) -> None:
    technical_llm.apply_deepseek_request_mode(payload, model, thinking_mode=thinking_mode)
    if _model_supports_deepseek_json_mode(model):
        payload["response_format"] = {"type": "json_object"}
    payload.setdefault("max_tokens", int(max_output_tokens or 8192))


def prompt_text_for_variant(prompt_variant: str) -> str:
    if prompt_variant == PROMPT_VARIANT_V2:
        return V2_SYSTEM_PROMPT
    if prompt_variant == PROMPT_VARIANT_V2_NO_REGIME:
        return V2_NO_REGIME_SYSTEM_PROMPT
    if prompt_variant == PROMPT_VARIANT_V2_FROZEN:
        return V2_FROZEN_SYSTEM_PROMPT
    if prompt_variant == PROMPT_VARIANT_V22:
        return V22_ADJUSTMENT_SYSTEM_PROMPT
    if prompt_variant == PROMPT_VARIANT_V23:
        return V23_RANK_ONLY_SYSTEM_PROMPT
    if prompt_variant == PROMPT_VARIANT_V24:
        return V24_FROZEN_SYSTEM_PROMPT
    if prompt_variant == PROMPT_VARIANT_V25:
        return V25_BALANCED_MEMORY_SYSTEM_PROMPT
    if prompt_variant == PROMPT_VARIANT_V26:
        return V26_SYSTEM_PROMPT
    if prompt_variant == PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL:
        return V2_MINIMAL_CHANGE_CONTROL_PROMPT
    raise ValueError(f"Unsupported prompt variant: {prompt_variant}")


def prompt_hash_for_variant(prompt_variant: str) -> str:
    return hashlib.sha256(prompt_text_for_variant(prompt_variant).encode("utf-8")).hexdigest()[:12]


def incumbent_configuration_manifest() -> dict[str, Any]:
    """Machine-readable production incumbent; change only through a reviewed experiment."""
    return {
        "scorer": "anonymized_v2",
        "scorer_source": INCUMBENT_SCORER_SOURCE,
        "prompt_variant": PROMPT_VARIANT_V2,
        "prompt_hash": INCUMBENT_PROMPT_HASH,
        "glossary": True,
        "temperature": INCUMBENT_TEMPERATURE,
        "candidate_pool_per_side": INCUMBENT_LLM_POOL_SIZE,
        "selected_per_side": INCUMBENT_BOOK_SIZES[0],
        "ticker_visibility": "anonymized_SEC_identifiers",
        "memory": False,
        "score_proportional_weighting": False,
        "concentrated_shorts": False,
        "within_side_weighting": "equal",
        "long_short_capital_split": {"long": 0.5, "short": 0.5},
        "reported_spread_return_convention": "long_leg_return_minus_short_leg_return",
        "output_contract": "existing_strict_json_score_0_100",
    }


def validate_incumbent_args(args: argparse.Namespace) -> None:
    """Prevent accidental drift from the production incumbent at the V2 entrypoint."""
    errors: list[str] = []
    actual_hash = prompt_hash_for_variant(PROMPT_VARIANT_V2)
    if actual_hash != INCUMBENT_PROMPT_HASH:
        errors.append(
            f"V2_SYSTEM_PROMPT hash changed ({actual_hash} != frozen {INCUMBENT_PROMPT_HASH})"
        )
    if args.prompt_variant != PROMPT_VARIANT_V2:
        errors.append(f"prompt variant must be {PROMPT_VARIANT_V2!r}")
    if tuple(args.pool_sizes) != INCUMBENT_POOL_SIZES:
        errors.append(f"pool sizes must be {INCUMBENT_POOL_SIZES}")
    if tuple(args.book_sizes) != INCUMBENT_BOOK_SIZES:
        errors.append(f"book sizes must be {INCUMBENT_BOOK_SIZES}")
    if int(args.llm_pool_size) != INCUMBENT_LLM_POOL_SIZE:
        errors.append(f"LLM pool size must be {INCUMBENT_LLM_POOL_SIZE}")
    if not bool(args.anonymize_tickers):
        errors.append("tickers must be anonymized")
    if not np.isclose(float(args.temperature), INCUMBENT_TEMPERATURE):
        errors.append(f"temperature must be {INCUMBENT_TEMPERATURE}")
    if args.thinking_mode != "disabled":
        errors.append("thinking mode must be disabled so temperature 0.6 is operative")
    if int(args.rationale_top_k_per_side) != 0:
        errors.append("rationale output must be disabled")
    if bool(args.structured_reasoning):
        errors.append("structured-reasoning treatment must be disabled")
    if errors:
        detail = "; ".join(errors)
        raise ValueError(
            f"Incumbent configuration guard rejected this run: {detail}. "
            "Use --experimental-override only for a deliberately labeled research run."
        )


def prompt_version_for_variant(prompt_variant: str) -> str:
    if prompt_variant == PROMPT_VARIANT_V2:
        return f"llm_reversal_signal_v2_{prompt_hash_for_variant(prompt_variant)}"
    if prompt_variant == PROMPT_VARIANT_V2_FROZEN:
        return f"llm_reversal_signal_v2_frozen_{prompt_hash_for_variant(prompt_variant)}"
    if prompt_variant == PROMPT_VARIANT_V22:
        return f"llm_reversal_signal_v2_2_{prompt_hash_for_variant(prompt_variant)}"
    if prompt_variant == PROMPT_VARIANT_V23:
        return f"llm_reversal_signal_v2_3_{prompt_hash_for_variant(prompt_variant)}"
    if prompt_variant == PROMPT_VARIANT_V24:
        return f"llm_reversal_signal_v2_4_{prompt_hash_for_variant(prompt_variant)}"
    if prompt_variant == PROMPT_VARIANT_V25:
        return f"llm_reversal_signal_v2_5_{prompt_hash_for_variant(prompt_variant)}"
    if prompt_variant == PROMPT_VARIANT_V26:
        return f"llm_reversal_signal_v2_6_{prompt_hash_for_variant(prompt_variant)}"
    if prompt_variant == PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL:
        return f"llm_reversal_signal_v2_minimal_change_control_{prompt_hash_for_variant(prompt_variant)}"
    if prompt_variant == PROMPT_VARIANT_V2_NO_REGIME:
        return f"llm_reversal_signal_v2_no_regime_{prompt_hash_for_variant(prompt_variant)}"
    raise ValueError(f"Unsupported prompt variant: {prompt_variant}")


def prompt_source_for_variant(prompt_variant: str, structured_reasoning: bool = False) -> str:
    if structured_reasoning:
        return "mean_reversion_llm_v2.structured_reasoning_system_prompt"
    if prompt_variant == PROMPT_VARIANT_V2:
        return "mean_reversion_llm_v2.V2_SYSTEM_PROMPT"
    if prompt_variant == PROMPT_VARIANT_V2_FROZEN:
        return "mean_reversion_llm_v2.V2_FROZEN_SYSTEM_PROMPT (historical v1 prompt)"
    if prompt_variant == PROMPT_VARIANT_V22:
        return "mean_reversion_llm_v2.V22_ADJUSTMENT_SYSTEM_PROMPT"
    if prompt_variant == PROMPT_VARIANT_V23:
        return "mean_reversion_llm_v2.V23_RANK_ONLY_SYSTEM_PROMPT"
    if prompt_variant == PROMPT_VARIANT_V24:
        return "mean_reversion_llm_v2.V24_FROZEN_SYSTEM_PROMPT + raw SPY/VIX user context"
    if prompt_variant == PROMPT_VARIANT_V25:
        return "mean_reversion_llm_v2.V2_SYSTEM_PROMPT + V25_BALANCED_MEMORY_APPENDIX"
    if prompt_variant == PROMPT_VARIANT_V26:
        return "mean_reversion_llm_v2.V26_SYSTEM_PROMPT"
    if prompt_variant == PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL:
        return "mean_reversion_llm_v2.V2_MINIMAL_CHANGE_CONTROL_PROMPT"
    if prompt_variant == PROMPT_VARIANT_V2_NO_REGIME:
        return (
            "mean_reversion_llm_v2.V2_NO_REGIME_SYSTEM_PROMPT + user input with "
            "MARKET REGIME line omitted"
        )
    raise ValueError(f"Unsupported prompt variant: {prompt_variant}")


def output_contract_for_variant(prompt_variant: str) -> str:
    if prompt_variant in {
        PROMPT_VARIANT_V2,
        PROMPT_VARIANT_V2_FROZEN,
        PROMPT_VARIANT_V25,
        PROMPT_VARIANT_V26,
        PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL,
        PROMPT_VARIANT_V2_NO_REGIME,
    }:
        return "score_0_100"
    if prompt_variant == PROMPT_VARIANT_V22:
        return "context_adjustment_minus3_to_plus3"
    if prompt_variant == PROMPT_VARIANT_V23:
        return "rank_only_complete_ordering"
    if prompt_variant == PROMPT_VARIANT_V24:
        return "score_0_100"
    raise ValueError(f"Unsupported prompt variant: {prompt_variant}")


def format_v2_no_regime_user_prompt(pool: v1.CandidatePool) -> str:
    """Remove only the date-level market-regime value from the standard V2 request."""
    base_prompt = v1.format_user_prompt(pool)
    regime_line = f"MARKET REGIME: {pool.regime}\n\n"
    if base_prompt.count(regime_line) != 1:
        raise ValueError("Standard V2 user prompt no longer contains exactly one market-regime line")
    ablated = base_prompt.replace(regime_line, "", 1)
    if "MARKET REGIME:" in ablated:
        raise ValueError("Market-regime line remained after V2 no-regime ablation")
    return ablated


def raw_market_context_from_spy(
    spy_df: pd.DataFrame | None,
    date: pd.Timestamp,
) -> dict[str, str | float | None]:
    """Return point-in-time market measurements without applying a regime taxonomy."""
    empty: dict[str, str | float | None] = {
        "spy_vs_200d_ma": "unknown",
        "spy_distance_from_200d_ma_pct": None,
        "vix_level": None,
        "vix_percentile_252d": None,
    }
    if spy_df is None or spy_df.empty or "date" not in spy_df or "adj_close" not in spy_df:
        return empty

    frame = spy_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    frame["_spy_200d_ma"] = pd.to_numeric(frame["adj_close"], errors="coerce").rolling(
        200,
        min_periods=200,
    ).mean()
    prior = frame.loc[frame["date"] <= pd.Timestamp(date)]
    if prior.empty:
        return empty

    row = prior.iloc[-1]

    def finite_or_none(value: Any) -> float | None:
        numeric = v1._safe_float(value)
        return None if pd.isna(numeric) else float(numeric)

    close = finite_or_none(row.get("adj_close"))
    ma200 = finite_or_none(row.get("_spy_200d_ma"))
    if close is None or ma200 is None or ma200 == 0.0:
        relation = "unknown"
        distance = None
    else:
        relation = "above" if close > ma200 else "below"
        distance = float((close / ma200 - 1.0) * 100.0)

    return {
        "spy_vs_200d_ma": relation,
        "spy_distance_from_200d_ma_pct": distance,
        "vix_level": finite_or_none(row.get("vix_level")),
        "vix_percentile_252d": finite_or_none(row.get("vix_percentile_252d")),
    }


def format_v24_user_prompt(pool: v1.CandidatePool) -> str:
    """Format the V2.4 score request with raw market measurements and no regime label."""
    context = pool.market_context or {}
    columns = "  ".join(v1.FEATURE_COLUMNS)

    def rows(tickers: list[str]) -> str:
        return "\n".join(
            "  ".join(v1._format_feature_value(pool.feature_rows[ticker][column]) for column in v1.FEATURE_COLUMNS)
            for ticker in tickers
        )

    def number(key: str, suffix: str = "") -> str:
        value = v1._safe_float(context.get(key))
        return "unavailable" if pd.isna(value) else f"{value:.1f}{suffix}"

    relation = str(context.get("spy_vs_200d_ma", "unknown")).strip().lower()
    if relation not in {"above", "below"}:
        relation = "unknown"

    return f"""
REBALANCE DATE: {pool.signal_date.date()}
MARKET CONTEXT (same for every candidate):
SPY vs 200-day MA: {relation}
SPY distance from 200-day MA: {number('spy_distance_from_200d_ma_pct', '%')}
VIX level: {number('vix_level')}
VIX trailing-252-day percentile: {number('vix_percentile_252d')}

LONG POOL ({len(pool.long_tickers)} names - biggest 5-day losers in Russell 1000):
Columns: {columns}
{rows(pool.long_tickers)}

SHORT POOL ({len(pool.short_tickers)} names - biggest 5-day winners in Russell 1000):
Columns: {columns}
{rows(pool.short_tickers)}

Score every ticker in both pools. Return JSON only.
""".strip()


def format_v25_user_prompt(pool: v1.CandidatePool) -> str:
    """Format unchanged V2 inputs plus deterministic balanced-memory summaries."""
    columns = (*v1.FEATURE_COLUMNS, *v25m.V25_MEMORY_COLUMNS)
    column_text = "  ".join(columns)

    def rows(tickers: list[str]) -> str:
        return "\n".join(
            "  ".join(v1._format_feature_value(pool.feature_rows[ticker][column]) for column in columns)
            for ticker in tickers
        )

    return f"""
REBALANCE DATE: {pool.signal_date.date()}
MARKET REGIME: {pool.regime}

LONG POOL ({len(pool.long_tickers)} names - biggest 5-day losers in Russell 1000):
Columns: {column_text}
{rows(pool.long_tickers)}

SHORT POOL ({len(pool.short_tickers)} names - biggest 5-day winners in Russell 1000):
Columns: {column_text}
{rows(pool.short_tickers)}

Score every ticker in both pools. Return JSON only.
""".strip()


@dataclass(frozen=True)
class V2PoolContext:
    adv20_df: pd.DataFrame
    trading_dates: pd.DatetimeIndex
    next_trade_map: dict[pd.Timestamp, pd.Timestamp]


@dataclass(frozen=True)
class V2Variant:
    pool_per_side: int
    positions_per_side: int
    anchor_offset: int | str

    @property
    def label(self) -> str:
        if self.anchor_offset == "Friday":
            return f"pool{self.pool_per_side}_pos{self.positions_per_side}_friday"
        return f"pool{self.pool_per_side}_pos{self.positions_per_side}_anchor{self.anchor_offset}"

    @property
    def output_dir_name(self) -> str:
        if self.anchor_offset == "Friday":
            return f"pool{self.pool_per_side}_pos{self.positions_per_side}/friday"
        return f"pool{self.pool_per_side}_pos{self.positions_per_side}/anchor_{self.anchor_offset}"


def parse_int_list(value: str) -> tuple[int, ...]:
    items: list[int] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(int(chunk))
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return tuple(dict.fromkeys(items))


def parse_float_list(value: str) -> tuple[float, ...]:
    items: list[float] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(float(chunk))
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one number")
    return tuple(dict.fromkeys(items))


def parse_path_list(value: str) -> tuple[Path, ...]:
    paths = [Path(chunk.strip()) for chunk in str(value).split(",") if chunk.strip()]
    if not paths:
        raise argparse.ArgumentTypeError("Expected at least one path")
    return tuple(paths)


def format_v22_lambda(value: float) -> str:
    text = f"{float(value):g}"
    return text


def v22_lambda_label(value: float) -> str:
    return "lambda" + format_v22_lambda(value).replace(".", "p").replace("-", "neg")


def v25_paired_control_cache_path(treatment_cache_path: Path) -> Path:
    return treatment_cache_path.with_name(
        f"{treatment_cache_path.stem}_paired-control-v2-{prompt_hash_for_variant(PROMPT_VARIANT_V2)}.json"
    )


def should_collect_fresh_v2_control(prompt_variant: str, v25_paired_control: bool) -> bool:
    """Only an explicit V2.5 control flag may request a fresh V2 control."""
    return prompt_variant == PROMPT_VARIANT_V25 and bool(v25_paired_control)


def autoresearch_paired_control_cache_path(treatment_cache_path: Path) -> Path:
    return treatment_cache_path.with_name(
        f"{treatment_cache_path.stem}_paired-control-v2mc-"
        f"{prompt_hash_for_variant(PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL)}.json"
    )


def variant_dir(output_dir: Path, variant: V2Variant) -> Path:
    return output_dir / variant.output_dir_name


def build_variant_grid(
    pool_sizes: tuple[int, ...],
    book_sizes: tuple[int, ...],
    anchors: tuple[int, ...],
    schedule: str = DEFAULT_SCHEDULE,
) -> list[V2Variant]:
    variants: list[V2Variant] = []
    for pool in pool_sizes:
        for book in book_sizes:
            if book > pool:
                continue
            if schedule == "friday":
                variants.append(V2Variant(pool_per_side=int(pool), positions_per_side=int(book), anchor_offset="Friday"))
            elif schedule == "anchors":
                for anchor in anchors:
                    variants.append(V2Variant(pool_per_side=int(pool), positions_per_side=int(book), anchor_offset=int(anchor)))
            else:
                raise ValueError(f"Unsupported V2 schedule: {schedule}")
    return variants


def build_v2_pool_context(market_frames: rb.MarketFrames) -> V2PoolContext:
    trading_dates = market_frames.close_df.index
    return V2PoolContext(
        adv20_df=market_frames.dollar_volume_df.rolling(20, min_periods=15).mean(),
        trading_dates=trading_dates,
        next_trade_map=rb._next_trading_day_map(trading_dates),
    )


def build_candidate_pool_for_index(
    market_frames: rb.MarketFrames,
    feature_frames: v1.FeatureFrames,
    membership_lookup: rb.MembershipLookup,
    spy_df: pd.DataFrame | None,
    signal_idx: int,
    pool_per_side: int,
    config: rb.BenchmarkConfig,
    context: V2PoolContext | None = None,
) -> v1.CandidatePool | None:
    context = context or build_v2_pool_context(market_frames)
    trading_dates = context.trading_dates
    lookback_days = int(config.signal_lookback_days)
    holding_days = int(config.resolved_holding_days())
    if signal_idx < lookback_days or signal_idx + holding_days >= len(trading_dates):
        return None

    close_df = market_frames.close_df
    unadj_close_df = market_frames.unadj_close_df
    signal_date = pd.Timestamp(trading_dates[signal_idx])
    lookback_date = pd.Timestamp(trading_dates[signal_idx - lookback_days])
    exit_date = pd.Timestamp(trading_dates[signal_idx + holding_days])

    if signal_date < config.start or signal_date > config.end or exit_date > config.end:
        return None

    r1000_universe = membership_lookup.universe_on_or_before(signal_date)
    valid: list[str] = []
    trails: list[float] = []
    for ticker in sorted(r1000_universe):
        if ticker not in close_df.columns:
            continue
        close_prev = close_df.at[lookback_date, ticker] if lookback_date in close_df.index else np.nan
        close_now = close_df.at[signal_date, ticker] if signal_date in close_df.index else np.nan
        trail = v1._pct_return(close_now, close_prev)
        if pd.isna(trail):
            continue
        raw_price = unadj_close_df.at[signal_date, ticker] if signal_date in unadj_close_df.index else np.nan
        if pd.isna(raw_price) or raw_price < config.min_trade_price:
            continue
        adv20 = context.adv20_df.at[signal_date, ticker] if signal_date in context.adv20_df.index else np.nan
        if pd.isna(adv20) or adv20 < config.min_adv20_dollars:
            continue
        valid.append(ticker)
        trails.append(float(trail))

    if len(valid) < pool_per_side * 2:
        return None

    trail_arr = np.asarray(trails, dtype=float)
    signal_arr = -trail_arr
    long_order = np.argsort(trail_arr, kind="mergesort")[:pool_per_side]
    short_order = np.argsort(-trail_arr, kind="mergesort")[:pool_per_side]
    long_tickers = [valid[idx] for idx in long_order]
    short_tickers = [valid[idx] for idx in short_order]
    if set(long_tickers) & set(short_tickers):
        return None

    mrl_scores = v1._percentile_scores(signal_arr, high_is_best=True)
    mrs_scores = 100.0 - mrl_scores
    trail_by_ticker = {ticker: float(trail_arr[idx]) for idx, ticker in enumerate(valid)}
    signal_by_ticker = {ticker: float(signal_arr[idx]) for idx, ticker in enumerate(valid)}
    feature_rows: dict[str, dict[str, float | str]] = {}
    for idx, ticker in enumerate(valid):
        if ticker not in long_tickers and ticker not in short_tickers:
            continue
        feature_rows[ticker] = v1.compute_prompt_features(
            market_frames=market_frames,
            feature_frames=feature_frames,
            ticker=ticker,
            signal_date=signal_date,
            trail_return=trail_by_ticker[ticker],
            mrl=float(mrl_scores[idx]),
            mrs=float(mrs_scores[idx]),
        )

    regime = v1.identify_regime(spy_df, signal_date) if spy_df is not None else "unknown"
    market_context = raw_market_context_from_spy(spy_df, signal_date)
    universe_hash = v1._hash_pool_payload(signal_date, regime, long_tickers, short_tickers, feature_rows)
    return v1.CandidatePool(
        signal_date=signal_date,
        lookback_date=lookback_date,
        exit_date=exit_date,
        entry_open_date=context.next_trade_map.get(signal_date),
        exit_open_date=context.next_trade_map.get(exit_date),
        valid_tickers=valid,
        long_tickers=long_tickers,
        short_tickers=short_tickers,
        trail_by_ticker=trail_by_ticker,
        signal_by_ticker=signal_by_ticker,
        feature_rows=feature_rows,
        universe_hash=universe_hash,
        regime=regime,
        market_context=market_context,
    )


def iter_anchor_signal_indices(config: rb.BenchmarkConfig, context: V2PoolContext, anchor_offset: int) -> list[int]:
    lookback_days = int(config.signal_lookback_days)
    holding_days = int(config.resolved_holding_days())
    rebalance_days = int(config.rebalance_days)
    first_idx = lookback_days + int(anchor_offset)
    max_idx = len(context.trading_dates) - holding_days - 1
    return list(range(first_idx, max_idx + 1, rebalance_days))


def llm_score_storage_key(pool: v1.CandidatePool, pool_per_side: int) -> str:
    return f"{pool.signal_date.date()}|pool{pool_per_side}"


def anonymous_ticker(index: int) -> str:
    return f"{ANON_TICKER_PREFIX}_{index:04d}"


def anonymize_pool_for_llm(pool: v1.CandidatePool) -> tuple[v1.CandidatePool, dict[str, str]]:
    """Return an LLM-facing pool with anonymous tickers plus anon->real mapping."""
    real_tickers = pool.pool_tickers
    real_to_anon = {ticker: anonymous_ticker(idx) for idx, ticker in enumerate(real_tickers, start=1)}
    anon_to_real = {anon: real for real, anon in real_to_anon.items()}

    def remap_tickers(tickers: list[str]) -> list[str]:
        return [real_to_anon[ticker] for ticker in tickers]

    feature_rows: dict[str, dict[str, float | str]] = {}
    for real_ticker, anon_ticker in real_to_anon.items():
        row = dict(pool.feature_rows[real_ticker])
        row["Ticker"] = anon_ticker
        feature_rows[anon_ticker] = row

    trail_by_ticker = {real_to_anon[ticker]: value for ticker, value in pool.trail_by_ticker.items() if ticker in real_to_anon}
    signal_by_ticker = {
        real_to_anon[ticker]: value for ticker, value in pool.signal_by_ticker.items() if ticker in real_to_anon
    }
    universe_hash = v1._hash_pool_payload(
        pool.signal_date,
        pool.regime,
        remap_tickers(pool.long_tickers),
        remap_tickers(pool.short_tickers),
        feature_rows,
    )
    if all(
        all(column in feature_rows[ticker] for column in v25m.V25_MEMORY_COLUMNS)
        for ticker in feature_rows
    ):
        universe_hash = v25m.hash_memory_enriched_pool(
            universe_hash,
            remap_tickers(pool.long_tickers),
            remap_tickers(pool.short_tickers),
            feature_rows,
        )
    anonymized = replace(
        pool,
        valid_tickers=remap_tickers(real_tickers),
        long_tickers=remap_tickers(pool.long_tickers),
        short_tickers=remap_tickers(pool.short_tickers),
        trail_by_ticker=trail_by_ticker,
        signal_by_ticker=signal_by_ticker,
        feature_rows=feature_rows,
        universe_hash=universe_hash,
    )
    return anonymized, anon_to_real


def deanonymize_llm_scores(result: dict[str, Any], anon_to_real: dict[str, str]) -> dict[str, Any]:
    remapped = dict(result)

    def remap_side(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            item = dict(row)
            anon = str(item.get("ticker", "")).strip().upper()
            if anon not in anon_to_real:
                raise ValueError(f"Unexpected anonymous ticker returned by LLM: {anon!r}")
            item["anonymous_ticker"] = anon
            item["ticker"] = anon_to_real[anon]
            output.append(item)
        return output

    def remap_ranking(rows: list[Any]) -> list[str]:
        output = []
        for row in rows:
            anon = str(row.get("ticker", "") if isinstance(row, dict) else row).strip().upper()
            if anon not in anon_to_real:
                raise ValueError(f"Unexpected anonymous ticker returned by LLM: {anon!r}")
            output.append(anon_to_real[anon])
        return output

    remapped["long_scores"] = remap_side(result.get("long_scores", []))
    remapped["short_scores"] = remap_side(result.get("short_scores", []))
    if "long_ranking" in result:
        remapped["long_ranking"] = remap_ranking(result.get("long_ranking", []))
    if "short_ranking" in result:
        remapped["short_ranking"] = remap_ranking(result.get("short_ranking", []))
    if "long_adjustments" in result:
        remapped["long_adjustments"] = remap_side(result.get("long_adjustments", []))
    if "short_adjustments" in result:
        remapped["short_adjustments"] = remap_side(result.get("short_adjustments", []))
    if "long_attributions" in result:
        remapped["long_attributions"] = remap_side(result.get("long_attributions", []))
    if "short_attributions" in result:
        remapped["short_attributions"] = remap_side(result.get("short_attributions", []))
    return remapped


def deanonymize_side_scores(result: dict[str, Any], anon_to_real: dict[str, str]) -> dict[str, Any]:
    remapped = dict(result)
    rows: list[dict[str, Any]] = []
    for row in result.get("scores", []):
        item = dict(row)
        anonymous = str(item.get("ticker", "")).strip().upper()
        if anonymous not in anon_to_real:
            raise ValueError(f"Unexpected anonymous ticker returned by side scorer: {anonymous!r}")
        item["anonymous_ticker"] = anonymous
        item["ticker"] = anon_to_real[anonymous]
        rows.append(item)
    remapped["scores"] = rows
    native_rows: list[dict[str, Any]] = []
    for row in result.get("native_scores", []):
        item = dict(row)
        anonymous = str(item.get("ticker", "")).strip().upper()
        if anonymous not in anon_to_real:
            raise ValueError(f"Unexpected anonymous native-score ticker: {anonymous!r}")
        item["anonymous_ticker"] = anonymous
        item["ticker"] = anon_to_real[anonymous]
        native_rows.append(item)
    if "native_scores" in result:
        remapped["native_scores"] = native_rows
    return remapped


def validate_llm_adjustments(
    raw_output: dict[str, Any],
    expected_longs: list[str],
    expected_shorts: list[str],
) -> dict[str, Any]:
    if not isinstance(raw_output, dict):
        raise ValueError("LLM adjustment output must be a JSON object")

    expected_long_set = {ticker.upper() for ticker in expected_longs}
    expected_short_set = {ticker.upper() for ticker in expected_shorts}
    if expected_long_set & expected_short_set:
        raise ValueError("Input pools overlap")

    def clean_side(key: str, expected: set[str], other: set[str]) -> list[dict[str, int | str]]:
        raw_entries = raw_output.get(key)
        if not isinstance(raw_entries, list):
            raise ValueError(f"Missing {key} array")
        seen: set[str] = set()
        cleaned: list[dict[str, int | str]] = []
        for entry in raw_entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{key} contains a non-object entry")
            ticker = str(entry.get("ticker", "")).strip().upper()
            if not ticker:
                raise ValueError(f"{key} contains missing ticker")
            if ticker in seen:
                raise ValueError(f"{key} contains duplicate ticker {ticker}")
            if ticker in other:
                raise ValueError(f"{ticker} appears on the wrong side")
            if ticker not in expected:
                raise ValueError(f"{key} contains unexpected ticker {ticker}")
            raw_adjustment = entry.get("context_adjustment", entry.get("adjustment"))
            if isinstance(raw_adjustment, (bool, np.bool_)) or not isinstance(
                raw_adjustment,
                (int, float, np.integer, np.floating),
            ):
                raise ValueError(f"{ticker} has non-numeric context_adjustment {raw_adjustment!r}")
            adjustment = float(raw_adjustment)
            if pd.isna(adjustment) or adjustment < -3.0 or adjustment > 3.0:
                raise ValueError(f"{ticker} has invalid context_adjustment {raw_adjustment!r}")
            rounded = int(round(adjustment))
            if abs(adjustment - rounded) > 1e-9:
                raise ValueError(f"{ticker} has non-integer context_adjustment {raw_adjustment!r}")
            seen.add(ticker)
            cleaned.append({"ticker": ticker, "context_adjustment": rounded})
        missing = expected - seen
        if missing:
            raise ValueError(f"{key} missing tickers: {', '.join(sorted(missing)[:10])}")
        if len(cleaned) != len(expected):
            raise ValueError(f"{key} expected {len(expected)} entries, got {len(cleaned)}")
        return cleaned

    return {
        "long_adjustments": clean_side("long_adjustments", expected_long_set, expected_short_set),
        "short_adjustments": clean_side("short_adjustments", expected_short_set, expected_long_set),
    }


def validate_llm_rankings(
    raw_output: dict[str, Any],
    expected_longs: list[str],
    expected_shorts: list[str],
) -> dict[str, Any]:
    if not isinstance(raw_output, dict):
        raise ValueError("LLM ranking output must be a JSON object")

    expected_long_set = {ticker.upper() for ticker in expected_longs}
    expected_short_set = {ticker.upper() for ticker in expected_shorts}
    if expected_long_set & expected_short_set:
        raise ValueError("Input pools overlap")

    def ticker_from_entry(entry: Any, key: str) -> str:
        if isinstance(entry, str):
            ticker = entry
        elif isinstance(entry, dict):
            ticker = str(entry.get("ticker", ""))
        else:
            raise ValueError(f"{key} contains a non-ticker entry")
        ticker = str(ticker).strip().upper()
        if not ticker:
            raise ValueError(f"{key} contains missing ticker")
        return ticker

    def clean_side(key: str, expected: set[str], other: set[str]) -> tuple[list[str], list[dict[str, Any]]]:
        raw_entries = raw_output.get(key)
        if not isinstance(raw_entries, list):
            raise ValueError(f"Missing {key} array")
        seen: set[str] = set()
        ranking: list[str] = []
        score_rows: list[dict[str, Any]] = []
        side_count = len(expected)
        for rank, entry in enumerate(raw_entries, start=1):
            ticker = ticker_from_entry(entry, key)
            if ticker in seen:
                raise ValueError(f"{key} contains duplicate ticker {ticker}")
            if ticker in other:
                raise ValueError(f"{ticker} appears on the wrong side")
            if ticker not in expected:
                raise ValueError(f"{key} contains unexpected ticker {ticker}")
            seen.add(ticker)
            ranking.append(ticker)
            percentile_score = 100.0 if side_count <= 1 else 100.0 * (side_count - rank) / (side_count - 1)
            score_rows.append(
                {
                    "ticker": ticker,
                    "rank": rank,
                    "rank_score": side_count - rank + 1,
                    "score": float(percentile_score),
                }
            )
        missing = expected - seen
        if missing:
            raise ValueError(f"{key} missing tickers: {', '.join(sorted(missing)[:10])}")
        if len(ranking) != len(expected):
            raise ValueError(f"{key} expected {len(expected)} entries, got {len(ranking)}")
        return ranking, score_rows

    long_ranking, long_scores = clean_side("long_ranking", expected_long_set, expected_short_set)
    short_ranking, short_scores = clean_side("short_ranking", expected_short_set, expected_long_set)
    return {
        "long_ranking": long_ranking,
        "short_ranking": short_ranking,
        "long_scores": long_scores,
        "short_scores": short_scores,
    }


def signed_scores_from_llm_rankings(validated_rankings: dict[str, Any]) -> dict[str, float]:
    signed: dict[str, float] = {}
    for entry in validated_rankings.get("long_scores", []):
        ticker = str(entry["ticker"]).upper()
        rank_score = float(entry.get("rank_score", entry.get("score", np.nan)))
        signed[ticker] = rank_score
    for entry in validated_rankings.get("short_scores", []):
        ticker = str(entry["ticker"]).upper()
        rank_score = float(entry.get("rank_score", entry.get("score", np.nan)))
        signed[ticker] = -rank_score
    return signed


def v22_base_scores_for_pool(pool: v1.CandidatePool) -> dict[str, float]:
    base_scores: dict[str, float] = {}
    long_values = np.asarray([pool.signal_by_ticker[ticker] for ticker in pool.long_tickers], dtype=float)
    short_values = np.asarray([pool.trail_by_ticker[ticker] for ticker in pool.short_tickers], dtype=float)
    for ticker, score in zip(pool.long_tickers, v1._percentile_scores(long_values, high_is_best=True), strict=True):
        base_scores[str(ticker).upper()] = float(score)
    for ticker, score in zip(pool.short_tickers, v1._percentile_scores(short_values, high_is_best=True), strict=True):
        base_scores[str(ticker).upper()] = float(score)
    return base_scores


def format_v22_user_prompt(pool: v1.CandidatePool) -> str:
    base_scores = v22_base_scores_for_pool(pool)
    columns = tuple(v1.FEATURE_COLUMNS) + ("BaseScore",)
    column_text = "  ".join(columns)

    def rows(tickers: list[str]) -> str:
        lines: list[str] = []
        for ticker in tickers:
            row = pool.feature_rows[ticker]
            values = [v1._format_feature_value(row[col]) for col in v1.FEATURE_COLUMNS]
            values.append(v1._format_feature_value(base_scores[str(ticker).upper()]))
            lines.append("  ".join(values))
        return "\n".join(lines)

    return f"""
REBALANCE DATE: {pool.signal_date.date()}
MARKET REGIME: {pool.regime}

LONG POOL ({len(pool.long_tickers)} names - biggest 5-day losers in Russell 1000):
Columns: {column_text}
{rows(pool.long_tickers)}

SHORT POOL ({len(pool.short_tickers)} names - biggest 5-day winners in Russell 1000):
Columns: {column_text}
{rows(pool.short_tickers)}

Return context_adjustment JSON only.
""".strip()


def v22_adjustment_map(validated_adjustments: dict[str, Any]) -> dict[str, int]:
    adjustments: dict[str, int] = {}
    for key in ("long_adjustments", "short_adjustments"):
        for row in validated_adjustments.get(key, []) or []:
            adjustments[str(row["ticker"]).upper()] = int(row["context_adjustment"])
    return adjustments


def v22_final_scores_for_pool(
    pool: v1.CandidatePool,
    validated_adjustments: dict[str, Any],
    adjustment_lambda: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, int], dict[str, float]]:
    base_scores = v22_base_scores_for_pool(pool)
    adjustments = v22_adjustment_map(validated_adjustments)
    missing = {ticker.upper() for ticker in pool.pool_tickers} - set(adjustments)
    if missing:
        raise ValueError(f"Missing V2.2 adjustments for pool tickers: {', '.join(sorted(missing)[:10])}")

    final_scores: dict[str, float] = {}
    signed_scores: dict[str, float] = {}
    short_set = {ticker.upper() for ticker in pool.short_tickers}
    for ticker in pool.pool_tickers:
        key = str(ticker).upper()
        final_score = float(np.clip(base_scores[key] + float(adjustment_lambda) * adjustments[key], 0.0, 100.0))
        final_scores[key] = final_score
        signed_scores[key] = -final_score if key in short_set else final_score
    return signed_scores, base_scores, adjustments, final_scores


def _book_masks_from_scores_with_base_tiebreak(
    pool_tickers: list[str],
    signed_scores: dict[str, float],
    base_scores: dict[str, float],
    long_tickers: set[str],
    short_tickers: set[str],
    positions_per_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray([signed_scores.get(str(ticker).upper(), np.nan) for ticker in pool_tickers], dtype=float)
    if np.isnan(signal).any():
        missing = [ticker for ticker, score in zip(pool_tickers, signal, strict=True) if pd.isna(score)]
        raise ValueError(f"Missing signed scores for pool tickers: {missing[:10]}")

    side_bias = np.asarray([1e-9 if ticker in long_tickers else -1e-9 for ticker in pool_tickers], dtype=float)
    long_order = sorted(
        range(len(pool_tickers)),
        key=lambda idx: (
            -(signal[idx] + side_bias[idx]),
            -float(base_scores.get(str(pool_tickers[idx]).upper(), np.nan)),
            pool_tickers[idx],
        ),
    )
    short_order = sorted(
        range(len(pool_tickers)),
        key=lambda idx: (
            signal[idx] + side_bias[idx],
            -float(base_scores.get(str(pool_tickers[idx]).upper(), np.nan)),
            pool_tickers[idx],
        ),
    )
    long_mask = np.zeros(len(pool_tickers), dtype=bool)
    short_mask = np.zeros(len(pool_tickers), dtype=bool)
    long_mask[long_order[:positions_per_side]] = True
    short_mask[short_order[:positions_per_side]] = True
    if set(np.where(long_mask)[0]) & set(np.where(short_mask)[0]):
        raise ValueError("Long and short books overlap")
    return signal, long_mask, short_mask


def attribution_prompt_appendix(top_k_per_side: int) -> str:
    allowed = ", ".join(ATTRIBUTION_FEATURES)
    return f"""

ATTRIBUTION ADD-ON
In the same JSON response, keep long_scores and short_scores exactly as specified above for every candidate.
Also include short rationale metadata only for the selected book implied by your scores:
- long_attributions: exactly the top {int(top_k_per_side)} LONG names by your long score.
- short_attributions: exactly the top {int(top_k_per_side)} SHORT names by your short score.

Each attribution entry must have:
- ticker: same ticker/id used in the score arrays.
- primary_features: 1 to 3 strings chosen only from this allowed list: {allowed}.
- rationale_label: max 6 words, no punctuation-heavy prose, no future-performance claims.

Return strict JSON only:
{{
  "long_scores": [{{"ticker": "SEC_0001", "score": 72}}, ...],
  "short_scores": [{{"ticker": "SEC_0061", "score": 84}}, ...],
  "long_attributions": [{{"ticker": "SEC_0001", "primary_features": ["Trail5", "RSI5"], "rationale_label": "extreme oversold low RSI"}}, ...],
  "short_attributions": [{{"ticker": "SEC_0061", "primary_features": ["MRSig", "Rng55"], "rationale_label": "overbought near range high"}}, ...]
}}
""".strip()


def _top_score_tickers(scores: list[dict[str, Any]], top_k: int) -> set[str]:
    return set(_top_score_ticker_list(scores, top_k))


def _top_score_ticker_list(scores: list[dict[str, Any]], top_k: int) -> list[str]:
    ordered = sorted(
        scores,
        key=lambda row: (-float(row["score"]), str(row["ticker"]).upper()),
    )
    return [str(row["ticker"]).upper() for row in ordered[: int(top_k)]]


def validate_attributions(
    raw_output: dict[str, Any],
    validated_scores: dict[str, Any],
    top_k_per_side: int,
) -> dict[str, Any]:
    """Clean optional attribution metadata without making the score call brittle.

    Attribution is diagnostic only. The scores determine the portfolio, so malformed
    or slightly off-target rationale should not kill an otherwise valid paid LLM call.
    """
    if top_k_per_side <= 0:
        return {}
    allowed = set(ATTRIBUTION_FEATURES)
    stats = {
        "missing": 0,
        "extra": 0,
        "invalid": 0,
        "duplicates": 0,
    }

    def clean_side(key: str, score_key: str) -> list[dict[str, Any]]:
        raw_entries = raw_output.get(key)
        if not isinstance(raw_entries, list):
            stats["missing"] += int(top_k_per_side)
            return []
        expected_order = _top_score_ticker_list(validated_scores[score_key], top_k_per_side)
        expected = set(expected_order)
        seen: set[str] = set()
        cleaned_by_ticker: dict[str, dict[str, Any]] = {}
        for entry in raw_entries:
            if not isinstance(entry, dict):
                stats["invalid"] += 1
                continue
            ticker = str(entry.get("ticker", "")).strip().upper()
            if not ticker:
                stats["invalid"] += 1
                continue
            if ticker in seen:
                stats["duplicates"] += 1
                continue
            seen.add(ticker)
            if ticker not in expected:
                stats["extra"] += 1
                continue
            features_raw = entry.get("primary_features")
            if not isinstance(features_raw, list) or not 1 <= len(features_raw) <= 3:
                stats["invalid"] += 1
                continue
            features: list[str] = []
            for feature in features_raw:
                feature_text = str(feature).strip()
                if feature_text not in allowed:
                    stats["invalid"] += 1
                    continue
                if feature_text not in features:
                    features.append(feature_text)
            if not features:
                continue
            label = " ".join(str(entry.get("rationale_label", "")).strip().split())
            if not label:
                stats["invalid"] += 1
                continue
            if len(label.split()) > 6:
                label = " ".join(label.split()[:6])
            cleaned_by_ticker[ticker] = {
                "ticker": ticker,
                "primary_features": features,
                "rationale_label": label,
            }
        stats["missing"] += len(expected - set(cleaned_by_ticker))
        return [cleaned_by_ticker[ticker] for ticker in expected_order if ticker in cleaned_by_ticker]

    cleaned = {
        "long_attributions": clean_side("long_attributions", "long_scores"),
        "short_attributions": clean_side("short_attributions", "short_scores"),
    }
    expected_total = int(top_k_per_side) * 2
    found_total = len(cleaned["long_attributions"]) + len(cleaned["short_attributions"])
    cleaned["attribution_status"] = "complete" if found_total == expected_total and not any(stats.values()) else "partial"
    cleaned["attribution_expected_count"] = expected_total
    cleaned["attribution_found_count"] = found_total
    cleaned["attribution_missing_count"] = stats["missing"]
    cleaned["attribution_extra_count"] = stats["extra"]
    cleaned["attribution_invalid_count"] = stats["invalid"]
    cleaned["attribution_duplicate_count"] = stats["duplicates"]
    return cleaned


def structured_reasoning_system_prompt() -> str:
    features = ", ".join(ATTRIBUTION_FEATURES)
    effects = ", ".join(STRUCTURED_REASONING_EFFECTS)
    labels = ", ".join(STRUCTURED_REASONING_SETUP_LABELS)
    return f"""
MEAN-REVERSION STRUCTURED REASONING SCORE TREATMENT, COMPACT JSON

CONTEXT
Mean reversion is the tendency of stocks that have moved sharply in one
direction to partially reverse over the next 5 trading days.

TASK
You are scoring ONE side-relative pool of stocks selected as mean-reversion
candidates over the next 5 trading days.

For each ticker, output:
- score: one integer from 0 to 100.
- pf: 1 to 3 compact [feature, effect] pairs identifying the input columns most
  important to the score.
- label: one compact label from the allowed list.

Allowed feature values: {features}
Allowed effect values: {effects}
Allowed label values: {labels}

Important interpretation rules:
- The score is not a position size.
- The score is not a probability of a positive return.
- The score is side-relative: compare names only to names in the provided pool.
- Use only the indicator columns provided. No outside knowledge, no ticker
  familiarity, no calendar awareness.
- Treat tickers as opaque identifiers.
- Use the full 0-100 range when the cross-section supports it.
- Do not produce ranks, weights, portfolio sizes, free-form prose, or commentary.
- Return every provided ticker exactly once.
- Use compact JSON. Do not pretty-print. Do not add markdown fences.

OUTPUT FORMAT (STRICT JSON, no other text, no markdown fences):
{{
  "scores": [
    {{"ticker":"SEC_0001","score":84,"pf":[["Trail5","supports"],["RSI5","supports"]],"label":"oversold reversal"}}
  ]
}}
""".strip()


def format_structured_side_user_prompt(pool: v1.CandidatePool, side: str) -> str:
    side_upper = side.upper()
    if side_upper == "LONG":
        tickers = pool.long_tickers
        description = "LONG POOL: biggest 5-day losers. Higher score means stronger oversold snap-back edge."
    elif side_upper == "SHORT":
        tickers = pool.short_tickers
        description = "SHORT POOL: biggest 5-day winners. Higher score means stronger overbought fade edge."
    else:
        raise ValueError(f"Unsupported structured reasoning side: {side}")
    columns = "  ".join(v1.FEATURE_COLUMNS)
    lines = []
    for ticker in tickers:
        row = pool.feature_rows[ticker]
        lines.append("  ".join(v1._format_feature_value(row[col]) for col in v1.FEATURE_COLUMNS))
    return f"""
REBALANCE DATE: {pool.signal_date.date()}
MARKET REGIME: {pool.regime}

{description}
NAMES: {len(tickers)}
Columns: {columns}
{chr(10).join(lines)}

Score every ticker in this one pool. Return JSON only.
""".strip()


def _feature_effect_text(features: list[Any]) -> tuple[str, str]:
    feature_names: list[str] = []
    effect_pairs: list[str] = []
    for item in features:
        if isinstance(item, dict):
            feature = str(item.get("feature", "")).strip()
            effect = str(item.get("effect", "")).strip()
        else:
            feature = str(item).strip()
            effect = ""
        if not feature:
            continue
        if feature not in feature_names:
            feature_names.append(feature)
        if effect:
            effect_pairs.append(f"{feature}:{effect}")
    return "|".join(feature_names), "|".join(effect_pairs)


def clean_structured_reasoning_metadata(
    raw_output: dict[str, Any],
    validated_scores: dict[str, Any],
) -> dict[str, Any]:
    """Attach soft-validated reasoning metadata to every scored candidate."""
    allowed_features = set(ATTRIBUTION_FEATURES)
    allowed_effects = set(STRUCTURED_REASONING_EFFECTS)
    allowed_labels = set(STRUCTURED_REASONING_SETUP_LABELS)
    stats = {"missing": 0, "invalid": 0, "partial": 0}

    def raw_by_ticker(key: str) -> dict[str, dict[str, Any]]:
        raw_entries = raw_output.get(key)
        if not isinstance(raw_entries, list):
            return {}
        mapping: dict[str, dict[str, Any]] = {}
        for entry in raw_entries:
            if not isinstance(entry, dict):
                stats["invalid"] += 1
                continue
            ticker = str(entry.get("ticker", "")).strip().upper()
            if ticker:
                mapping.setdefault(ticker, entry)
        return mapping

    def clean_features(features_raw: Any) -> tuple[list[dict[str, str]], str]:
        if not isinstance(features_raw, list) or not features_raw:
            stats["missing"] += 1
            return [], "missing"
        cleaned: list[dict[str, str]] = []
        status = "complete"
        for item in features_raw[:3]:
            if isinstance(item, dict):
                feature = str(item.get("feature", "")).strip()
                effect = str(item.get("effect", "")).strip()
            elif isinstance(item, (list, tuple)) and item:
                feature = str(item[0]).strip()
                effect = str(item[1]).strip() if len(item) > 1 else "mixed"
            else:
                feature = str(item).strip()
                effect = "mixed"
                status = "partial"
                stats["partial"] += 1
            if feature not in allowed_features:
                stats["invalid"] += 1
                status = "partial"
                continue
            if effect not in allowed_effects:
                stats["invalid"] += 1
                effect = "mixed"
                status = "partial"
            if feature not in {row["feature"] for row in cleaned}:
                cleaned.append({"feature": feature, "effect": effect})
        if not cleaned:
            stats["missing"] += 1
            return [], "missing"
        return cleaned, status

    def clean_label(label_raw: Any) -> tuple[str, str]:
        label = " ".join(str(label_raw or "").strip().split()).lower()
        if label in allowed_labels:
            return label, "complete"
        stats["invalid"] += 1
        return "mixed signals", "partial"

    def enrich_side(score_key: str) -> list[dict[str, Any]]:
        raw_lookup = raw_by_ticker(score_key)
        enriched: list[dict[str, Any]] = []
        for row in validated_scores.get(score_key, []) or []:
            ticker = str(row["ticker"]).strip().upper()
            raw_entry = raw_lookup.get(ticker, {})
            features_raw = raw_entry.get("primary_features", raw_entry.get("pf"))
            label_raw = raw_entry.get("setup_label", raw_entry.get("label"))
            features, feature_status = clean_features(features_raw)
            setup_label, label_status = clean_label(label_raw)
            status = "complete" if feature_status == "complete" and label_status == "complete" else "partial"
            enriched.append(
                {
                    **row,
                    "primary_features": features,
                    "setup_label": setup_label,
                    "reasoning_status": status,
                }
            )
        return enriched

    enriched = {
        "long_scores": enrich_side("long_scores"),
        "short_scores": enrich_side("short_scores"),
    }
    complete_count = sum(
        1
        for key in ("long_scores", "short_scores")
        for row in enriched[key]
        if row.get("reasoning_status") == "complete"
    )
    expected_count = len(enriched["long_scores"]) + len(enriched["short_scores"])
    enriched["structured_reasoning_status"] = "complete" if complete_count == expected_count else "partial"
    enriched["structured_reasoning_expected_count"] = expected_count
    enriched["structured_reasoning_complete_count"] = complete_count
    enriched["structured_reasoning_missing_count"] = stats["missing"]
    enriched["structured_reasoning_invalid_count"] = stats["invalid"]
    enriched["structured_reasoning_partial_count"] = stats["partial"]
    return enriched


async def call_llm_scores_with_structured_reasoning(
    client: httpx.AsyncClient,
    pool: v1.CandidatePool,
    model: str,
    temperature: float,
    thinking_mode: str = technical_llm.DEFAULT_THINKING_MODE,
) -> dict[str, Any]:
    url = technical_llm.get_deepseek_url()
    headers = {"Authorization": f"Bearer {technical_llm.get_deepseek_api_key()}"}
    started = time.time()

    def normalize_side_rows(raw_output: dict[str, Any], expected_tickers: list[str], side: str) -> list[dict[str, Any]]:
        raw_entries = raw_output.get("scores")
        if not isinstance(raw_entries, list):
            raise ValueError(f"Missing scores array for {side}")
        expected = {ticker.upper() for ticker in expected_tickers}
        seen: set[str] = set()
        cleaned: list[dict[str, Any]] = []
        for entry in raw_entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{side} scores contains a non-object entry")
            ticker = str(entry.get("ticker", "")).strip().upper()
            if not ticker:
                raise ValueError(f"{side} scores contains missing ticker")
            if ticker in seen:
                raise ValueError(f"{side} scores contains duplicate ticker {ticker}")
            if ticker not in expected:
                raise ValueError(f"{side} scores contains unexpected ticker {ticker}")
            score_raw = entry.get("score")
            if isinstance(score_raw, (bool, np.bool_)) or not isinstance(
                score_raw,
                (int, float, np.integer, np.floating),
            ):
                raise ValueError(f"{ticker} has non-numeric score {score_raw!r}")
            score = float(score_raw)
            if pd.isna(score) or score < 0.0 or score > 100.0:
                raise ValueError(f"{ticker} has invalid score {score_raw!r}")
            seen.add(ticker)
            cleaned.append(dict(entry, ticker=ticker, score=int(round(score))))
        missing = expected - seen
        if missing:
            raise ValueError(f"{side} scores missing tickers: {', '.join(sorted(missing)[:10])}")
        return cleaned

    async def call_side(side: str, expected_tickers: list[str]) -> dict[str, Any]:
        base_messages = [
            {"role": "system", "content": structured_reasoning_system_prompt()},
            {"role": "user", "content": format_structured_side_user_prompt(pool, side)},
        ]
        retry_message = {
            "role": "user",
            "content": (
                "Your previous response was invalid or malformed. Return ONLY compact strict JSON "
                'like {"scores":[{"ticker":"SEC_0001","score":84,"pf":[["Trail5","supports"]],"label":"oversold reversal"}]}. '
                "Include every provided ticker exactly once. No markdown. No prose. Do not truncate."
            ),
        }
        last_error: Exception | None = None
        for attempt in range(v1.LLM_SCORE_MAX_ATTEMPTS):
            messages = base_messages + ([retry_message] if last_error is not None else [])
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": False,
            }
            _harden_json_payload(payload, model, thinking_mode=thinking_mode)
            if technical_llm.model_supports_temperature(model, thinking_mode=thinking_mode):
                payload["temperature"] = temperature
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                response_payload = response.json()
                choices = response_payload.get("choices") or [{}]
                choice = choices[0] if choices else {}
                message = choice.get("message") or {}
                content = message.get("content") or ""
                finish_reason = choice.get("finish_reason")
                if not content:
                    raise v1.EmptyLLMResponseError(
                        "No content in LLM response "
                        f"(finish_reason={finish_reason!r}, "
                        f"message_keys={sorted(message.keys())}, "
                        f"usage={response_payload.get('usage', {})})"
                    )
                if finish_reason == "length":
                    raise ValueError(
                        f"{side} structured response hit max_tokens before valid JSON "
                        f"(usage={response_payload.get('usage', {})})"
                    )
                raw_output = technical_llm._parse_json_content(content)
                rows = normalize_side_rows(raw_output, expected_tickers, side)
                prompt_chars = sum(len(message["content"]) for message in messages)
                completion_chars = len(content)
                return {
                    "rows": rows,
                    "retry_count": attempt,
                    "usage": {
                        "provider_usage": response_payload.get("usage", {}),
                        "prompt_chars": prompt_chars,
                        "completion_chars": completion_chars,
                        "approx_prompt_tokens": (prompt_chars + 3) // 4,
                        "approx_completion_tokens": (completion_chars + 3) // 4,
                        "approx_total_tokens": (prompt_chars + completion_chars + 3) // 4,
                    },
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < v1.LLM_SCORE_MAX_ATTEMPTS - 1:
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"LLM structured-reasoning {side} call failed after {v1.LLM_SCORE_MAX_ATTEMPTS} attempts: {exc}"
                ) from exc
        raise RuntimeError(f"LLM structured-reasoning {side} call failed: {last_error}")

    long_result = await call_side("LONG", pool.long_tickers)
    short_result = await call_side("SHORT", pool.short_tickers)
    raw_output = {
        "long_scores": long_result["rows"],
        "short_scores": short_result["rows"],
    }
    validated = v1.validate_llm_scores(
        raw_output,
        pool.long_tickers,
        pool.short_tickers,
        prompt_style=V2_PROMPT_STYLE,
    )
    structured = clean_structured_reasoning_metadata(raw_output, validated)
    usage = {
        "provider_usage": {
            "long": long_result["usage"]["provider_usage"],
            "short": short_result["usage"]["provider_usage"],
        },
        "prompt_chars": int(long_result["usage"]["prompt_chars"] + short_result["usage"]["prompt_chars"]),
        "completion_chars": int(long_result["usage"]["completion_chars"] + short_result["usage"]["completion_chars"]),
        "approx_prompt_tokens": int(
            long_result["usage"]["approx_prompt_tokens"] + short_result["usage"]["approx_prompt_tokens"]
        ),
        "approx_completion_tokens": int(
            long_result["usage"]["approx_completion_tokens"] + short_result["usage"]["approx_completion_tokens"]
        ),
        "approx_total_tokens": int(
            long_result["usage"]["approx_total_tokens"] + short_result["usage"]["approx_total_tokens"]
        ),
    }
    return {
        **structured,
        "parse_status": "ok",
        "retry_count": int(long_result["retry_count"] + short_result["retry_count"]),
        "latency_seconds": time.time() - started,
        "structured_reasoning_call_policy": "split_by_side_compact_json",
        "llm_usage": usage,
    }


def format_score_side_user_prompt(pool: v1.CandidatePool, side: str) -> str:
    normalized_side = str(side).strip().upper()
    if normalized_side == "LONG":
        tickers = pool.long_tickers
        description = "biggest 5-day losers in Russell 1000"
    elif normalized_side == "SHORT":
        tickers = pool.short_tickers
        description = "biggest 5-day winners in Russell 1000"
    else:
        raise ValueError(f"Unsupported split-side score request: {side!r}")

    columns = "  ".join(v1.FEATURE_COLUMNS)
    rows = "\n".join(
        "  ".join(v1._format_feature_value(pool.feature_rows[ticker][column]) for column in v1.FEATURE_COLUMNS)
        for ticker in tickers
    )
    return f"""
REBALANCE DATE: {pool.signal_date.date()}
MARKET REGIME: {pool.regime}

{normalized_side} POOL ({len(tickers)} names - {description}):
Columns: {columns}
{rows}

Score every ticker in this {normalized_side} pool. Return JSON only.
""".strip()


def split_side_score_system_prompt(prompt_variant: str, side: str) -> str:
    normalized_side = str(side).strip().upper()
    return (
        prompt_text_for_variant(prompt_variant)
        + f"""

SPLIT-SIDE API ADAPTER
This request contains only the {normalized_side} pool. Apply the unchanged
{normalized_side}-pool score definition above to every supplied ticker.
Return only this compact JSON object:
{{"scores":[{{"ticker":"SEC_0001","score":72}}]}}
Do not return the absent side, commentary, markdown, or reasoning in content.
"""
    ).strip()


def validate_split_side_scores(
    raw_output: dict[str, Any],
    expected_tickers: list[str],
    side: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_output, dict):
        raise ValueError("Split-side LLM output must be a JSON object")
    rows = raw_output.get("scores")
    normalized_side = str(side).strip().upper()
    if rows is None:
        rows = raw_output.get("long_scores" if normalized_side == "LONG" else "short_scores")
    if normalized_side == "LONG":
        validated = v1.validate_llm_scores(
            {"long_scores": rows, "short_scores": []},
            expected_longs=expected_tickers,
            expected_shorts=[],
            prompt_style=V2_PROMPT_STYLE,
        )
        return validated["long_scores"]
    if normalized_side == "SHORT":
        validated = v1.validate_llm_scores(
            {"long_scores": [], "short_scores": rows},
            expected_longs=[],
            expected_shorts=expected_tickers,
            prompt_style=V2_PROMPT_STYLE,
        )
        return validated["short_scores"]
    raise ValueError(f"Unsupported split-side score response: {side!r}")


def combine_split_side_score_results(
    long_result: dict[str, Any],
    short_result: dict[str, Any],
    expected_longs: list[str],
    expected_shorts: list[str],
) -> dict[str, Any]:
    long_scores = validate_split_side_scores(
        {"scores": long_result.get("scores")}, expected_longs, "LONG"
    )
    short_scores = validate_split_side_scores(
        {"scores": short_result.get("scores")}, expected_shorts, "SHORT"
    )
    validated = v1.validate_llm_scores(
        {"long_scores": long_scores, "short_scores": short_scores},
        expected_longs=expected_longs,
        expected_shorts=expected_shorts,
        prompt_style=V2_PROMPT_STYLE,
    )
    long_usage = long_result.get("llm_usage", {}) or {}
    short_usage = short_result.get("llm_usage", {}) or {}
    return {
        **validated,
        "parse_status": "ok",
        "retry_count": int(long_result.get("retry_count", 0) or 0)
        + int(short_result.get("retry_count", 0) or 0),
        "reasoning_content_by_side": {
            "long": long_result.get("reasoning_content"),
            "short": short_result.get("reasoning_content"),
        },
        "llm_call_policy": "split_by_side_thinking",
        "llm_usage": {
            "provider_usage": {
                "long": long_usage.get("provider_usage", {}),
                "short": short_usage.get("provider_usage", {}),
            },
            "prompt_chars": int(long_usage.get("prompt_chars", 0) or 0)
            + int(short_usage.get("prompt_chars", 0) or 0),
            "completion_chars": int(long_usage.get("completion_chars", 0) or 0)
            + int(short_usage.get("completion_chars", 0) or 0),
            "approx_prompt_tokens": int(long_usage.get("approx_prompt_tokens", 0) or 0)
            + int(short_usage.get("approx_prompt_tokens", 0) or 0),
            "approx_completion_tokens": int(long_usage.get("approx_completion_tokens", 0) or 0)
            + int(short_usage.get("approx_completion_tokens", 0) or 0),
            "approx_total_tokens": int(long_usage.get("approx_total_tokens", 0) or 0)
            + int(short_usage.get("approx_total_tokens", 0) or 0),
        },
    }


async def call_llm_score_side_thinking(
    client: httpx.AsyncClient,
    pool: v1.CandidatePool,
    side: str,
    model: str,
    prompt_variant: str,
    max_output_tokens: int,
    max_attempts: int,
    token_budget: OutputTokenBudget,
    usage_callback: Callable[[dict[str, Any]], None],
    print_reasoning_content: bool = False,
) -> dict[str, Any]:
    normalized_side = str(side).strip().upper()
    expected_tickers = pool.long_tickers if normalized_side == "LONG" else pool.short_tickers
    base_messages = [
        {"role": "system", "content": split_side_score_system_prompt(prompt_variant, normalized_side)},
        {"role": "user", "content": format_score_side_user_prompt(pool, normalized_side)},
    ]
    retry_message = {
        "role": "user",
        "content": (
            "Your previous response was invalid. Return ONLY compact strict JSON as "
            '{"scores":[{"ticker":"SEC_0001","score":72}]}. '
            f"Include every provided {normalized_side} ticker exactly once. No other keys or prose."
        ),
    }
    url = technical_llm.get_deepseek_url()
    headers = {"Authorization": f"Bearer {technical_llm.get_deepseek_api_key()}"}
    started = time.time()
    last_error: Exception | None = None

    for attempt in range(int(max_attempts)):
        messages = base_messages + ([retry_message] if last_error is not None else [])
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        _harden_json_payload(
            payload,
            model,
            thinking_mode="enabled",
            max_output_tokens=max_output_tokens,
        )
        try:
            token_budget.reserve_request(max_output_tokens)
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            response_payload = response.json()
            choices = response_payload.get("choices") or [{}]
            choice = choices[0] if choices else {}
            message = choice.get("message") or {}
            finish_reason = choice.get("finish_reason")
            provider_usage = response_payload.get("usage", {}) or {}
            output_tokens = _provider_output_tokens(provider_usage, fallback=max_output_tokens)
            token_budget.record(output_tokens)
            usage_callback(
                {
                    "side": normalized_side,
                    "attempt": attempt + 1,
                    "model": model,
                    "prompt_variant": prompt_variant,
                    "prompt_hash": prompt_hash_for_variant(prompt_variant),
                    "max_output_tokens": int(max_output_tokens),
                    "provider_usage": provider_usage,
                    "output_tokens_charged": output_tokens,
                    "finish_reason": finish_reason,
                    "recorded_at": pd.Timestamp.utcnow().isoformat(),
                }
            )
            if finish_reason == "length":
                raise LLMResponseLengthError(
                    f"{normalized_side} response exhausted max_tokens={max_output_tokens:,} "
                    f"before final JSON (usage={provider_usage})"
                )
            content = message.get("content") or ""
            reasoning_content = message.get("reasoning_content")
            if not content:
                raise v1.EmptyLLMResponseError(
                    f"No {normalized_side} content in LLM response "
                    f"(finish_reason={finish_reason!r}, usage={provider_usage})"
                )
            if print_reasoning_content and reasoning_content:
                print(f"\n--- BEGIN reasoning_content {pool.signal_date.date()} {normalized_side} ---")
                print(reasoning_content)
                print(f"--- END reasoning_content {pool.signal_date.date()} {normalized_side} ---\n")
            raw_output = technical_llm._parse_json_content(content)
            scores = validate_split_side_scores(raw_output, expected_tickers, normalized_side)
            prompt_chars = sum(len(item["content"]) for item in messages)
            completion_chars = len(content)
            return {
                "side": normalized_side,
                "scores": scores,
                "parse_status": "ok",
                "retry_count": attempt,
                "latency_seconds": time.time() - started,
                "reasoning_content": reasoning_content,
                "llm_usage": {
                    "provider_usage": provider_usage,
                    "prompt_chars": prompt_chars,
                    "completion_chars": completion_chars,
                    "approx_prompt_tokens": (prompt_chars + 3) // 4,
                    "approx_completion_tokens": (completion_chars + 3) // 4,
                    "approx_total_tokens": (prompt_chars + completion_chars + 3) // 4,
                },
            }
        except LLMSafeStop:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            retryable = _is_retryable_llm_error(exc)
            if retryable and attempt < int(max_attempts) - 1:
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            detail = "not retried after max-token exhaustion" if isinstance(exc, LLMResponseLengthError) else "failed"
            raise LLMSafeStop(
                f"{normalized_side} thinking call {detail} after {attempt + 1} attempt(s): {exc}. "
                "Completed sides and token usage are saved; rerunning the same command will resume."
            ) from exc

    raise LLMSafeStop(f"{normalized_side} thinking call failed: {last_error}")


async def call_llm_scores_for_prompt_variant(
    client: httpx.AsyncClient,
    pool: v1.CandidatePool,
    model: str,
    temperature: float,
    prompt_variant: str,
    thinking_mode: str = technical_llm.DEFAULT_THINKING_MODE,
    print_reasoning_content: bool = False,
) -> dict[str, Any]:
    if prompt_variant == PROMPT_VARIANT_V22:
        user_prompt = format_v22_user_prompt(pool)
    elif prompt_variant == PROMPT_VARIANT_V24:
        user_prompt = format_v24_user_prompt(pool)
    elif prompt_variant == PROMPT_VARIANT_V25:
        user_prompt = format_v25_user_prompt(pool)
    elif prompt_variant == PROMPT_VARIANT_V2_NO_REGIME:
        user_prompt = format_v2_no_regime_user_prompt(pool)
    else:
        user_prompt = v1.format_user_prompt(pool)
    base_messages = [
        {"role": "system", "content": prompt_text_for_variant(prompt_variant)},
        {"role": "user", "content": user_prompt},
    ]
    if prompt_variant == PROMPT_VARIANT_V22:
        retry_content = (
            "Your previous response was invalid. Return ONLY strict JSON with "
            "complete long_adjustments and short_adjustments arrays. Include every provided "
            "ticker exactly once on its original side. Use one integer context_adjustment "
            "from -3 to +3 per ticker. No commentary."
        )
    elif prompt_variant == PROMPT_VARIANT_V23:
        retry_content = (
            "Your previous response was invalid. Return ONLY strict JSON with "
            "complete long_ranking and short_ranking arrays. Include every provided "
            "ticker exactly once on its original side. The arrays must be ordered from "
            "strongest setup to weakest setup. No scores, labels, rationales, or commentary."
        )
    else:
        retry_content = (
            "Your previous response was invalid. Return ONLY strict JSON with "
            "complete long_scores and short_scores arrays. Include every provided "
            "ticker exactly once on its original side. Use one numeric score from "
            "0 to 100 per ticker. No commentary."
        )
    retry_message = {
        "role": "user",
        "content": retry_content,
    }
    url = technical_llm.get_deepseek_url()
    headers = {"Authorization": f"Bearer {technical_llm.get_deepseek_api_key()}"}
    started = time.time()
    last_error: Exception | None = None
    for attempt in range(v1.LLM_SCORE_MAX_ATTEMPTS):
        messages = base_messages + ([retry_message] if last_error is not None else [])
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        _harden_json_payload(payload, model, thinking_mode=thinking_mode)
        if technical_llm.model_supports_temperature(model, thinking_mode=thinking_mode):
            payload["temperature"] = temperature
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            response_payload = response.json()
            choices = response_payload.get("choices") or [{}]
            choice = choices[0] if choices else {}
            message = choice.get("message") or {}
            content = message.get("content") or ""
            reasoning_content = message.get("reasoning_content")
            finish_reason = choice.get("finish_reason")
            if not content:
                raise v1.EmptyLLMResponseError(
                    "No content in LLM response "
                    f"(finish_reason={finish_reason!r}, "
                    f"message_keys={sorted(message.keys())}, "
                    f"usage={response_payload.get('usage', {})})"
                )
            if finish_reason == "length":
                raise ValueError(
                    "LLM response hit max_tokens before valid JSON "
                    f"(usage={response_payload.get('usage', {})})"
                )
            if print_reasoning_content and reasoning_content:
                print(f"\n--- BEGIN reasoning_content {pool.signal_date.date()} ---")
                print(reasoning_content)
                print(f"--- END reasoning_content {pool.signal_date.date()} ---\n")
            raw_output = technical_llm._parse_json_content(content)
            if prompt_variant == PROMPT_VARIANT_V22:
                validated = validate_llm_adjustments(raw_output, pool.long_tickers, pool.short_tickers)
            elif prompt_variant == PROMPT_VARIANT_V23:
                validated = validate_llm_rankings(raw_output, pool.long_tickers, pool.short_tickers)
            else:
                validated = v1.validate_llm_scores(
                    raw_output,
                    pool.long_tickers,
                    pool.short_tickers,
                    prompt_style=V2_PROMPT_STYLE,
                )
            prompt_chars = sum(len(message["content"]) for message in messages)
            completion_chars = len(content)
            return {
                **validated,
                "parse_status": "ok",
                "retry_count": attempt,
                "latency_seconds": time.time() - started,
                "reasoning_content": reasoning_content,
                "llm_usage": {
                    "provider_usage": response_payload.get("usage", {}),
                    "prompt_chars": prompt_chars,
                    "completion_chars": completion_chars,
                    "approx_prompt_tokens": (prompt_chars + 3) // 4,
                    "approx_completion_tokens": (completion_chars + 3) // 4,
                    "approx_total_tokens": (prompt_chars + completion_chars + 3) // 4,
                },
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < v1.LLM_SCORE_MAX_ATTEMPTS - 1:
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError(
                f"LLM {prompt_variant} score call failed after {v1.LLM_SCORE_MAX_ATTEMPTS} attempts: {exc}"
            ) from exc
    raise RuntimeError(f"LLM {prompt_variant} score call failed: {last_error}")


async def call_llm_scores_bounded_nonthinking(
    client: httpx.AsyncClient,
    pool: v1.CandidatePool,
    model: str,
    temperature: float,
    system_prompt: str,
    prompt_identity: str,
    prompt_hash: str,
    max_output_tokens: int,
    max_attempts: int,
    token_budget: OutputTokenBudget,
    call_budget: LLMCallBudget,
    usage_callback: Callable[[dict[str, Any]], None],
    progress_file: Path | None = None,
    score_min: float = 0.0,
    score_max: float = 100.0,
    score_direction: str = "higher_is_better",
    side_scope: str = "both",
) -> dict[str, Any]:
    """Strict score call used by autoresearch treatments and their fresh control."""
    side = str(side_scope).strip().lower()
    if side not in {"both", "long", "short"}:
        raise ValueError("side_scope must be both, long, or short")
    if side == "both":
        user_prompt = v1.format_user_prompt(pool)
        retry_content = (
            "Your previous response was invalid. Return ONLY strict JSON with complete "
            "long_scores and short_scores arrays. Include every supplied opaque ticker exactly once "
            f"on its original side, with one numeric score from {score_min:g} to {score_max:g}. "
            f"Use the declared {score_direction} direction. No other keys or prose."
        )
    else:
        user_prompt = format_score_side_user_prompt(pool, side.upper())
        retry_content = (
            "Your previous response was invalid. Return ONLY strict JSON with exactly one scores array. "
            f"Include every supplied {side.upper()} ticker exactly once with one numeric score from "
            f"{score_min:g} to {score_max:g}. Use the declared {score_direction} direction. "
            "Do not return the absent side, other keys, markdown, or prose."
        )
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    retry_message = {
        "role": "user",
        "content": retry_content,
    }
    url = technical_llm.get_deepseek_url()
    headers = {"Authorization": f"Bearer {technical_llm.get_deepseek_api_key()}"}
    started = time.time()
    last_error: Exception | None = None

    for attempt in range(int(max_attempts)):
        messages = base_messages + ([retry_message] if last_error is not None else [])
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        _harden_json_payload(
            payload,
            model,
            thinking_mode=AUTORESEARCH_THINKING_MODE,
            max_output_tokens=int(max_output_tokens),
        )
        payload["temperature"] = AUTORESEARCH_TEMPERATURE
        call_budget.reserve_request()
        token_budget.reserve_request(int(max_output_tokens))
        append_progress_event(
            progress_file,
            "llm_call_started",
            storage_key=llm_score_storage_key(pool, len(pool.long_tickers)),
            signal_date=str(pool.signal_date.date()),
            prompt_identity=prompt_identity,
            prompt_hash=prompt_hash,
            model=model,
            temperature=float(temperature),
            thinking_mode=AUTORESEARCH_THINKING_MODE,
            attempt=attempt + 1,
            call_budget_used=call_budget.used,
            call_budget_limit=call_budget.limit,
        )
        provider_usage: dict[str, Any] = {}
        output_tokens = 0
        finish_reason: Any = None
        raw_output: Any = None
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            response_payload = response.json()
            choices = response_payload.get("choices") or [{}]
            choice = choices[0] if choices else {}
            message = choice.get("message") or {}
            content = message.get("content") or ""
            finish_reason = choice.get("finish_reason")
            provider_usage = response_payload.get("usage", {}) or {}
            output_tokens = _provider_output_tokens(
                provider_usage,
                fallback=min(int(max_output_tokens), (len(content) + 3) // 4),
            )
            token_budget.record(output_tokens)
            if finish_reason == "length":
                raise LLMResponseLengthError(
                    f"Response exhausted max_tokens={max_output_tokens:,} before final JSON"
                )
            if not content:
                raise v1.EmptyLLMResponseError(
                    f"No content in autoresearch score response (finish_reason={finish_reason!r})"
                )
            raw_output = technical_llm._parse_json_content(content)
            if side == "both":
                validated = validate_and_canonicalize_autoresearch_scores(
                    raw_output,
                    expected_longs=pool.long_tickers,
                    expected_shorts=pool.short_tickers,
                    score_min=float(score_min),
                    score_max=float(score_max),
                    score_direction=score_direction,
                )
            else:
                expected = pool.long_tickers if side == "long" else pool.short_tickers
                validated = validate_and_canonicalize_autoresearch_side_scores(
                    raw_output,
                    expected,
                    side_scope=side,
                    score_min=float(score_min),
                    score_max=float(score_max),
                    score_direction=score_direction,
                )
            usage_record = {
                "attempt": attempt + 1,
                "status": "ok",
                "model": model,
                "temperature": AUTORESEARCH_TEMPERATURE,
                "thinking_mode": AUTORESEARCH_THINKING_MODE,
                "prompt_variant": prompt_identity,
                "prompt_hash": prompt_hash,
                "max_output_tokens": int(max_output_tokens),
                "provider_usage": provider_usage,
                "output_tokens_charged": int(output_tokens),
                "finish_reason": finish_reason,
                "recorded_at": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            usage_callback(usage_record)
            append_progress_event(
                progress_file,
                "llm_call_succeeded",
                signal_date=str(pool.signal_date.date()),
                prompt_identity=prompt_identity,
                prompt_hash=prompt_hash,
                side_scope=side,
                attempt=attempt + 1,
                output_tokens=int(output_tokens),
                output_token_budget_used=token_budget.used,
                output_token_budget_limit=token_budget.limit,
            )
            prompt_chars = sum(len(message_item["content"]) for message_item in messages)
            completion_chars = len(content)
            return {
                **validated,
                "parse_status": "ok",
                "retry_count": attempt,
                "latency_seconds": time.time() - started,
                "llm_usage": {
                    "provider_usage": provider_usage,
                    "prompt_chars": prompt_chars,
                    "completion_chars": completion_chars,
                    "approx_prompt_tokens": (prompt_chars + 3) // 4,
                    "approx_completion_tokens": (completion_chars + 3) // 4,
                    "approx_total_tokens": (prompt_chars + completion_chars + 3) // 4,
                },
            }
        except LLMSafeStop:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            usage_callback(
                {
                    "attempt": attempt + 1,
                    "status": "failed",
                    "model": model,
                    "temperature": AUTORESEARCH_TEMPERATURE,
                    "thinking_mode": AUTORESEARCH_THINKING_MODE,
                    "prompt_variant": prompt_identity,
                    "prompt_hash": prompt_hash,
                    "max_output_tokens": int(max_output_tokens),
                    "provider_usage": provider_usage,
                    "output_tokens_charged": int(output_tokens),
                    "finish_reason": finish_reason,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "recorded_at": pd.Timestamp.now(tz="UTC").isoformat(),
                }
            )
            append_progress_event(
                progress_file,
                "llm_call_failed",
                signal_date=str(pool.signal_date.date()),
                prompt_identity=prompt_identity,
                prompt_hash=prompt_hash,
                attempt=attempt + 1,
                retryable=bool(_is_retryable_llm_error(exc)),
                schema_error=isinstance(exc, (ValueError, json.JSONDecodeError))
                and not isinstance(exc, LLMResponseLengthError),
                transient_error=_is_retryable_llm_error(exc)
                and isinstance(exc, (httpx.TransportError, httpx.HTTPStatusError)),
                format_only_repair_eligible=(
                    _is_format_only_score_payload(
                        raw_output,
                        pool.long_tickers,
                        pool.short_tickers,
                        score_min=float(score_min),
                        score_max=float(score_max),
                    )
                    if side == "both"
                    else False
                ),
                output_tokens=int(output_tokens),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if _is_retryable_llm_error(exc) and attempt < int(max_attempts) - 1:
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            detail = "not retried after max-token exhaustion" if isinstance(exc, LLMResponseLengthError) else "failed"
            raise LLMSafeStop(
                f"Autoresearch {prompt_identity} call {detail} after {attempt + 1} attempt(s): {exc}. "
                "Completed scores and usage are saved; rerunning the same command will resume."
            ) from exc
    raise LLMSafeStop(f"Autoresearch {prompt_identity} score call failed: {last_error}")


async def call_llm_scores_with_attribution(
    client: httpx.AsyncClient,
    pool: v1.CandidatePool,
    model: str,
    temperature: float,
    top_k_per_side: int,
    thinking_mode: str = technical_llm.DEFAULT_THINKING_MODE,
) -> dict[str, Any]:
    user_prompt = v1.format_user_prompt(pool)
    base_messages = [
        {"role": "system", "content": V2_SYSTEM_PROMPT + "\n\n" + attribution_prompt_appendix(top_k_per_side)},
        {"role": "user", "content": user_prompt},
    ]
    retry_message = {
        "role": "user",
        "content": (
            "Your previous response was invalid. Return ONLY strict JSON. "
            "Include complete long_scores and short_scores for every ticker, plus "
            f"exactly {int(top_k_per_side)} long_attributions and {int(top_k_per_side)} "
            "short_attributions matching the top scores on each side. No commentary."
        ),
    }
    url = technical_llm.get_deepseek_url()
    headers = {"Authorization": f"Bearer {technical_llm.get_deepseek_api_key()}"}
    started = time.time()
    last_error: Exception | None = None
    for attempt in range(v1.LLM_SCORE_MAX_ATTEMPTS):
        messages = base_messages + ([retry_message] if last_error is not None else [])
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        _harden_json_payload(payload, model, thinking_mode=thinking_mode)
        if technical_llm.model_supports_temperature(model, thinking_mode=thinking_mode):
            payload["temperature"] = temperature
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            response_payload = response.json()
            choices = response_payload.get("choices") or [{}]
            choice = choices[0] if choices else {}
            message = choice.get("message") or {}
            content = message.get("content") or ""
            finish_reason = choice.get("finish_reason")
            if not content:
                raise v1.EmptyLLMResponseError(
                    "No content in LLM response "
                    f"(finish_reason={finish_reason!r}, "
                    f"message_keys={sorted(message.keys())}, "
                    f"usage={response_payload.get('usage', {})})"
                )
            if finish_reason == "length":
                raise ValueError(
                    "LLM attribution response hit max_tokens before valid JSON "
                    f"(usage={response_payload.get('usage', {})})"
                )
            raw_output = technical_llm._parse_json_content(content)
            validated = v1.validate_llm_scores(
                raw_output,
                pool.long_tickers,
                pool.short_tickers,
                prompt_style=V2_PROMPT_STYLE,
            )
            attributions = validate_attributions(raw_output, validated, top_k_per_side)
            prompt_chars = sum(len(message["content"]) for message in messages)
            completion_chars = len(content)
            return {
                **validated,
                **attributions,
                "parse_status": "ok",
                "retry_count": attempt,
                "latency_seconds": time.time() - started,
                "llm_usage": {
                    "provider_usage": response_payload.get("usage", {}),
                    "prompt_chars": prompt_chars,
                    "completion_chars": completion_chars,
                    "approx_prompt_tokens": (prompt_chars + 3) // 4,
                    "approx_completion_tokens": (completion_chars + 3) // 4,
                    "approx_total_tokens": (prompt_chars + completion_chars + 3) // 4,
                },
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < v1.LLM_SCORE_MAX_ATTEMPTS - 1:
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError(f"LLM attribution score call failed after {v1.LLM_SCORE_MAX_ATTEMPTS} attempts: {exc}") from exc
    raise RuntimeError(f"LLM attribution score call failed: {last_error}")


def _subset_llm_scores_for_pool(entry: dict[str, Any], pool: v1.CandidatePool) -> dict[str, Any]:
    long_set = {ticker.upper() for ticker in pool.long_tickers}
    short_set = {ticker.upper() for ticker in pool.short_tickers}
    sliced = dict(entry)
    sliced["long_scores"] = [
        row for row in entry.get("long_scores", [])
        if str(row.get("ticker", "")).strip().upper() in long_set
    ]
    sliced["short_scores"] = [
        row for row in entry.get("short_scores", [])
        if str(row.get("ticker", "")).strip().upper() in short_set
    ]
    if "long_attributions" in entry:
        sliced["long_attributions"] = [
            row for row in entry.get("long_attributions", [])
            if str(row.get("ticker", "")).strip().upper() in long_set
        ]
    if "short_attributions" in entry:
        sliced["short_attributions"] = [
            row for row in entry.get("short_attributions", [])
            if str(row.get("ticker", "")).strip().upper() in short_set
        ]
    if "long_adjustments" in entry:
        sliced["long_adjustments"] = [
            row for row in entry.get("long_adjustments", [])
            if str(row.get("ticker", "")).strip().upper() in long_set
        ]
    if "short_adjustments" in entry:
        sliced["short_adjustments"] = [
            row for row in entry.get("short_adjustments", [])
            if str(row.get("ticker", "")).strip().upper() in short_set
        ]
    if "long_ranking" in entry:
        sliced["long_ranking"] = [
            str(row.get("ticker", "") if isinstance(row, dict) else row).strip().upper()
            for row in entry.get("long_ranking", [])
            if str(row.get("ticker", "") if isinstance(row, dict) else row).strip().upper() in long_set
        ]
    if "short_ranking" in entry:
        sliced["short_ranking"] = [
            str(row.get("ticker", "") if isinstance(row, dict) else row).strip().upper()
            for row in entry.get("short_ranking", [])
            if str(row.get("ticker", "") if isinstance(row, dict) else row).strip().upper() in short_set
        ]
    return sliced


def llm_entry_for_v2_pool(
    llm_scores_by_key: dict[str, dict[str, Any]],
    pool: v1.CandidatePool,
    pool_per_side: int,
    llm_source_pool_per_side: int,
) -> dict[str, Any]:
    exact_key = llm_score_storage_key(pool, pool_per_side)
    if exact_key in llm_scores_by_key:
        return llm_scores_by_key[exact_key]
    source_key = llm_score_storage_key(pool, llm_source_pool_per_side)
    if source_key not in llm_scores_by_key:
        raise KeyError(f"Missing V2 LLM scores for {exact_key} and source {source_key}")
    return _subset_llm_scores_for_pool(llm_scores_by_key[source_key], pool)


def validate_llm_scores_for_v2_pool(
    llm_scores_by_key: dict[str, dict[str, Any]],
    pool: v1.CandidatePool,
    pool_per_side: int,
    llm_source_pool_per_side: int,
) -> dict[str, Any]:
    entry = llm_entry_for_v2_pool(llm_scores_by_key, pool, pool_per_side, llm_source_pool_per_side)
    return v1.validate_llm_scores(
        entry,
        pool.long_tickers,
        pool.short_tickers,
        prompt_style=V2_PROMPT_STYLE,
    )


def validate_llm_adjustments_for_v2_pool(
    llm_scores_by_key: dict[str, dict[str, Any]],
    pool: v1.CandidatePool,
    pool_per_side: int,
    llm_source_pool_per_side: int,
) -> dict[str, Any]:
    entry = llm_entry_for_v2_pool(llm_scores_by_key, pool, pool_per_side, llm_source_pool_per_side)
    return validate_llm_adjustments(entry, pool.long_tickers, pool.short_tickers)


def validate_llm_rankings_for_v2_pool(
    llm_scores_by_key: dict[str, dict[str, Any]],
    pool: v1.CandidatePool,
    pool_per_side: int,
    llm_source_pool_per_side: int,
) -> dict[str, Any]:
    entry = llm_entry_for_v2_pool(llm_scores_by_key, pool, pool_per_side, llm_source_pool_per_side)
    return validate_llm_rankings(entry, pool.long_tickers, pool.short_tickers)


def attribution_map_from_entry(entry: dict[str, Any]) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for key in ("long_scores", "short_scores"):
        for row in entry.get(key, []) or []:
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            features = row.get("primary_features", [])
            feature_text, effect_text = _feature_effect_text(features if isinstance(features, list) else [])
            if not feature_text and not row.get("setup_label"):
                continue
            setup_label = str(row.get("setup_label", "")).strip()
            mapping[ticker] = {
                "attribution_features": feature_text,
                "attribution_effects": effect_text,
                "setup_label": setup_label,
                "rationale_label": setup_label,
                "reasoning_status": str(row.get("reasoning_status", "")).strip(),
                "anonymous_ticker": str(row.get("anonymous_ticker", "")).strip(),
            }
    for key in ("long_attributions", "short_attributions"):
        for row in entry.get(key, []) or []:
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            features = row.get("primary_features", [])
            if isinstance(features, list):
                feature_text = "|".join(str(feature).strip() for feature in features if str(feature).strip())
            else:
                feature_text = ""
            existing = mapping.get(ticker, {})
            mapping[ticker] = {
                **existing,
                "attribution_features": feature_text,
                "attribution_effects": existing.get("attribution_effects", ""),
                "setup_label": existing.get("setup_label", ""),
                "rationale_label": str(row.get("rationale_label", "")).strip(),
                "reasoning_status": existing.get("reasoning_status", ""),
                "anonymous_ticker": str(row.get("anonymous_ticker", "")).strip(),
            }
    return mapping


def run_v2_pool_strategy(
    market_frames: rb.MarketFrames,
    feature_frames: v1.FeatureFrames,
    membership_lookup: rb.MembershipLookup,
    spy_df: pd.DataFrame | None,
    config: rb.BenchmarkConfig,
    pool_per_side: int,
    positions_per_side: int,
    anchor_offset: int | str,
    score_provider: str,
    llm_scores_by_key: dict[str, dict[str, Any]] | None = None,
    llm_source_pool_per_side: int | None = None,
    v22_lambda: float = 4.0,
    pool_overrides_by_key: dict[str, v1.CandidatePool] | None = None,
    allowed_signal_dates: frozenset[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[v1.CandidatePool]]:
    close_df = market_frames.close_df
    open_df = market_frames.open_df
    context = build_v2_pool_context(market_frames)
    friday_context = v1.build_pool_context(market_frames) if anchor_offset == "Friday" else None
    results: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    pools: list[v1.CandidatePool] = []
    prev_long_set: set[str] = set()
    prev_short_set: set[str] = set()

    if anchor_offset == "Friday":
        signal_items: list[pd.Timestamp | int] = [pd.Timestamp(date) for date in friday_context.fridays] if friday_context else []
        anchor_label: int | str = "Friday"
    else:
        signal_items = iter_anchor_signal_indices(config, context, int(anchor_offset))
        anchor_label = int(anchor_offset)

    for signal_item in signal_items:
        if anchor_offset == "Friday":
            signal_date = pd.Timestamp(signal_item)
            if signal_date < config.start or signal_date > config.end:
                continue
            if int(context.trading_dates.get_loc(signal_date)) < config.signal_lookback_days:
                continue
            pool = v1.build_candidate_pool(
                market_frames=market_frames,
                feature_frames=feature_frames,
                membership_lookup=membership_lookup,
                spy_df=spy_df,
                signal_date=signal_date,
                pool_per_side=pool_per_side,
                config=config,
                context=friday_context,
            )
        else:
            pool = build_candidate_pool_for_index(
                market_frames=market_frames,
                feature_frames=feature_frames,
                membership_lookup=membership_lookup,
                spy_df=spy_df,
                signal_idx=int(signal_item),
                pool_per_side=pool_per_side,
                config=config,
                context=context,
            )
        if pool is None:
            continue
        if (
            allowed_signal_dates is not None
            and pd.Timestamp(pool.signal_date).normalize() not in allowed_signal_dates
        ):
            continue
        if anchor_offset == "Friday" and pool.exit_date > config.end:
            continue
        if pool_overrides_by_key is not None:
            override_key = llm_score_storage_key(
                pool,
                int(llm_source_pool_per_side or pool_per_side),
            )
            override = pool_overrides_by_key.get(override_key)
            if override is None:
                raise KeyError(f"Missing pool override for {override_key}")
            if override.long_tickers != pool.long_tickers or override.short_tickers != pool.short_tickers:
                raise ValueError(f"Pool override ticker mismatch for {override_key}")
            pool = override
        pools.append(pool)
        pool_tickers = pool.pool_tickers
        long_set = set(pool.long_tickers)
        short_set = set(pool.short_tickers)

        if score_provider == "trail5":
            signed_scores = {ticker: pool.signal_by_ticker[ticker] for ticker in pool_tickers}
            attribution_by_ticker: dict[str, dict[str, str]] = {}
            base_scores_by_ticker: dict[str, float] = {}
            adjustments_by_ticker: dict[str, int] = {}
            final_scores_by_ticker: dict[str, float] = {}
        elif score_provider == "llm":
            if llm_scores_by_key is None:
                raise ValueError("llm_scores_by_key is required for score_provider='llm'")
            llm_entry = llm_entry_for_v2_pool(
                llm_scores_by_key=llm_scores_by_key,
                pool=pool,
                pool_per_side=pool_per_side,
                llm_source_pool_per_side=int(llm_source_pool_per_side or pool_per_side),
            )
            validated = validate_llm_scores_for_v2_pool(
                llm_scores_by_key=llm_scores_by_key,
                pool=pool,
                pool_per_side=pool_per_side,
                llm_source_pool_per_side=int(llm_source_pool_per_side or pool_per_side),
            )
            signed_scores = v1.signed_scores_from_llm(validated)
            attribution_by_ticker = attribution_map_from_entry(llm_entry)
            base_scores_by_ticker = {}
            adjustments_by_ticker = {}
            final_scores_by_ticker = {str(ticker).upper(): abs(score) for ticker, score in signed_scores.items()}
        elif score_provider == "llm_adjustment":
            if llm_scores_by_key is None:
                raise ValueError("llm_scores_by_key is required for score_provider='llm_adjustment'")
            llm_entry = llm_entry_for_v2_pool(
                llm_scores_by_key=llm_scores_by_key,
                pool=pool,
                pool_per_side=pool_per_side,
                llm_source_pool_per_side=int(llm_source_pool_per_side or pool_per_side),
            )
            validated_adjustments = validate_llm_adjustments_for_v2_pool(
                llm_scores_by_key=llm_scores_by_key,
                pool=pool,
                pool_per_side=pool_per_side,
                llm_source_pool_per_side=int(llm_source_pool_per_side or pool_per_side),
            )
            signed_scores, base_scores_by_ticker, adjustments_by_ticker, final_scores_by_ticker = (
                v22_final_scores_for_pool(pool, validated_adjustments, float(v22_lambda))
            )
            attribution_by_ticker = attribution_map_from_entry(llm_entry)
        elif score_provider == "llm_rank":
            if llm_scores_by_key is None:
                raise ValueError("llm_scores_by_key is required for score_provider='llm_rank'")
            llm_entry = llm_entry_for_v2_pool(
                llm_scores_by_key=llm_scores_by_key,
                pool=pool,
                pool_per_side=pool_per_side,
                llm_source_pool_per_side=int(llm_source_pool_per_side or pool_per_side),
            )
            validated_rankings = validate_llm_rankings_for_v2_pool(
                llm_scores_by_key=llm_scores_by_key,
                pool=pool,
                pool_per_side=pool_per_side,
                llm_source_pool_per_side=int(llm_source_pool_per_side or pool_per_side),
            )
            signed_scores = signed_scores_from_llm_rankings(validated_rankings)
            attribution_by_ticker = attribution_map_from_entry(llm_entry)
            base_scores_by_ticker = {}
            adjustments_by_ticker = {}
            final_scores_by_ticker = {
                str(row["ticker"]).upper(): float(row["score"])
                for key in ("long_scores", "short_scores")
                for row in validated_rankings.get(key, []) or []
            }
        elif score_provider == "histret":
            signed_scores = {}
            for ticker in pool.long_tickers:
                value = v1._safe_float(pool.feature_rows[ticker].get("HistRet"))
                signed_scores[ticker] = 0.0 if pd.isna(value) else float(value)
            for ticker in pool.short_tickers:
                value = v1._safe_float(pool.feature_rows[ticker].get("HistRet"))
                signed_scores[ticker] = -(0.0 if pd.isna(value) else float(value))
            attribution_by_ticker = {}
            base_scores_by_ticker = {}
            adjustments_by_ticker = {}
            final_scores_by_ticker = {
                str(ticker).upper(): abs(float(score)) for ticker, score in signed_scores.items()
            }
        else:
            raise ValueError(f"Unsupported score_provider: {score_provider}")

        if score_provider == "llm_adjustment":
            signal, long_mask, short_mask = _book_masks_from_scores_with_base_tiebreak(
                pool_tickers,
                signed_scores,
                base_scores_by_ticker,
                long_set,
                short_set,
                positions_per_side,
            )
        else:
            signal, long_mask, short_mask = v1._book_masks_from_scores(
                pool_tickers,
                signed_scores,
                long_set,
                short_set,
                positions_per_side,
            )
        deciles = v1._deciles_from_signal(signal)
        fwd_c2c = rb._forward_c2c_return(close_df, pool.signal_date, pool.exit_date, pool_tickers)
        fwd_o2o = rb._forward_o2o_return(open_df, pool.entry_open_date, pool.exit_open_date, pool_tickers)
        long_tickers = [pool_tickers[pos] for pos in np.where(long_mask)[0]]
        short_tickers = [pool_tickers[pos] for pos in np.where(short_mask)[0]]

        row: dict[str, Any] = {
            "date": pool.signal_date,
            "anchor_offset": anchor_label,
            "pool_per_side": int(pool_per_side),
            "positions_per_side": int(positions_per_side),
            "n_universe": len(pool.valid_tickers),
            "n_pool": len(pool_tickers),
            "n_long": len(long_tickers),
            "n_short": len(short_tickers),
            "long_ret_c2c": rb.safe_mean(fwd_c2c, long_mask),
            "short_ret_c2c": rb.safe_mean(fwd_c2c, short_mask),
            "long_ret_o2o": rb.safe_mean(fwd_o2o, long_mask),
            "short_ret_o2o": rb.safe_mean(fwd_o2o, short_mask),
            "turnover_long": rb.calc_turnover(prev_long_set, set(long_tickers)),
            "turnover_short": rb.calc_turnover(prev_short_set, set(short_tickers)),
            "ic_c2c": v1._spearman(signal, fwd_c2c),
            "ic_o2o": v1._spearman(signal, fwd_o2o),
            "regime": pool.regime,
        }
        for decile in range(1, 11):
            decile_mask = deciles == decile
            row[f"dec_{decile}_c2c"] = rb.safe_mean(fwd_c2c, decile_mask)
            row[f"dec_{decile}_o2o"] = rb.safe_mean(fwd_o2o, decile_mask)
        results.append(row)

        prev_long_set = set(long_tickers)
        prev_short_set = set(short_tickers)

        for pos, ticker in enumerate(pool_tickers):
            feature_row = pool.feature_rows[ticker]
            attribution = attribution_by_ticker.get(str(ticker).upper(), {})
            if score_provider == "llm_rank":
                llm_score_value = final_scores_by_ticker.get(str(ticker).upper(), np.nan)
            elif score_provider in {"llm", "llm_adjustment", "histret"}:
                llm_score_value = abs(signed_scores[ticker])
            else:
                llm_score_value = np.nan
            signal_rows.append(
                {
                    "date": pool.signal_date,
                    "anchor_offset": anchor_label,
                    "pool_per_side": int(pool_per_side),
                    "positions_per_side": int(positions_per_side),
                    "ticker": ticker,
                    "pool_side": "LONG" if ticker in long_set else "SHORT",
                    "score_provider": score_provider,
                    "llm_score": llm_score_value,
                    "base_score": base_scores_by_ticker.get(str(ticker).upper(), np.nan),
                    "context_adjustment": adjustments_by_ticker.get(str(ticker).upper(), np.nan),
                    "final_score": final_scores_by_ticker.get(str(ticker).upper(), np.nan),
                    "v22_lambda": float(v22_lambda) if score_provider == "llm_adjustment" else np.nan,
                    "signal": signal[pos],
                    "trail_ret": pool.trail_by_ticker[ticker],
                    "fwd_c2c": fwd_c2c[pos],
                    "fwd_o2o": fwd_o2o[pos],
                    "decile": deciles[pos],
                    "attribution_features": attribution.get("attribution_features", ""),
                    "attribution_effects": attribution.get("attribution_effects", ""),
                    "setup_label": attribution.get("setup_label", ""),
                    "rationale_label": attribution.get("rationale_label", ""),
                    "reasoning_status": attribution.get("reasoning_status", ""),
                    "anonymous_ticker": attribution.get("anonymous_ticker", ""),
                    "hist_n": feature_row.get("HistN", np.nan),
                    "hist_win_pct": feature_row.get("HistWin", np.nan),
                    "hist_ret_pct": feature_row.get("HistRet", np.nan),
                    **{f"feature_{col}": feature_row[col] for col in v1.FEATURE_COLUMNS if col != "Ticker"},
                }
            )

        for side, mask in (("LONG", long_mask), ("SHORT", short_mask)):
            selected_positions = np.where(mask)[0]
            side_weight = 1.0 / len(selected_positions) if len(selected_positions) else np.nan
            ordered_positions = sorted(
                selected_positions,
                key=lambda pos: signal[pos],
                reverse=(side == "LONG"),
            )
            for rank, pos in enumerate(ordered_positions, start=1):
                attribution = attribution_by_ticker.get(str(pool_tickers[pos]).upper(), {})
                selection_rows.append(
                    {
                        "rebalance_date": pool.signal_date,
                        "signal_date": pool.signal_date,
                        "entry_date": pool.signal_date,
                        "exit_date": pool.exit_date,
                        "anchor_offset": anchor_label,
                        "pool_per_side": int(pool_per_side),
                        "positions_per_side": int(positions_per_side),
                        "ticker": pool_tickers[pos],
                        "position_type": side,
                        "rank": rank,
                        "score": signal[pos],
                        "base_score": base_scores_by_ticker.get(str(pool_tickers[pos]).upper(), np.nan),
                        "context_adjustment": adjustments_by_ticker.get(str(pool_tickers[pos]).upper(), np.nan),
                        "final_score": final_scores_by_ticker.get(str(pool_tickers[pos]).upper(), np.nan),
                        "v22_lambda": float(v22_lambda) if score_provider == "llm_adjustment" else np.nan,
                        "trail_return_pct": pool.trail_by_ticker[pool_tickers[pos]] * 100.0,
                        "realized_return_pct": fwd_c2c[pos] * 100.0,
                        "realized_c2c_return_pct": fwd_c2c[pos] * 100.0,
                        "realized_o2o_return_pct": fwd_o2o[pos] * 100.0,
                        "weight": side_weight,
                        "decile": deciles[pos],
                        "attribution_features": attribution.get("attribution_features", ""),
                        "attribution_effects": attribution.get("attribution_effects", ""),
                        "setup_label": attribution.get("setup_label", ""),
                        "rationale_label": attribution.get("rationale_label", ""),
                        "reasoning_status": attribution.get("reasoning_status", ""),
                        "anonymous_ticker": attribution.get("anonymous_ticker", ""),
                        "hist_n": pool.feature_rows[pool_tickers[pos]].get("HistN", np.nan),
                        "hist_win_pct": pool.feature_rows[pool_tickers[pos]].get("HistWin", np.nan),
                        "hist_ret_pct": pool.feature_rows[pool_tickers[pos]].get("HistRet", np.nan),
                    }
                )

    weekly_returns = pd.DataFrame(results)
    if weekly_returns.empty:
        raise RuntimeError(f"No {score_provider} V2 pool results were produced")
    weekly_returns["date"] = pd.to_datetime(weekly_returns["date"])
    weekly_returns = weekly_returns.set_index("date").sort_index()
    return weekly_returns, pd.DataFrame(signal_rows), pd.DataFrame(selection_rows), pools


def build_v2_bundle(
    weekly_returns: pd.DataFrame,
    signal_checks: pd.DataFrame,
    portfolio_selections: pd.DataFrame,
    config: rb.BenchmarkConfig,
) -> v1.ExperimentBundle:
    return v1.build_bundle(weekly_returns, signal_checks, portfolio_selections, config)


def save_v2_bundle(bundle: v1.ExperimentBundle, output_dir: Path) -> None:
    v1.save_bundle(bundle, output_dir)


def save_attribution_outcome_summary(selections: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if selections.empty or "attribution_features" not in selections.columns:
        summary = pd.DataFrame()
        summary.to_csv(output_dir / "attribution_outcome_summary.csv", index=False)
        return summary
    rows: list[dict[str, Any]] = []
    attributed = selections[selections["attribution_features"].astype(str).str.len() > 0].copy()
    for _, row in attributed.iterrows():
        features = [feature for feature in str(row.get("attribution_features", "")).split("|") if feature]
        side = str(row.get("position_type", "")).strip().upper()
        c2c_return = pd.to_numeric(row.get("realized_c2c_return_pct"), errors="coerce")
        o2o_return = pd.to_numeric(row.get("realized_o2o_return_pct"), errors="coerce")
        side_multiplier = -1.0 if side == "SHORT" else 1.0
        side_c2c = c2c_return * side_multiplier
        side_o2o = o2o_return * side_multiplier
        for feature in features:
            rows.append(
                {
                    "feature": feature,
                    "side": side,
                    "realized_c2c_return_pct": c2c_return,
                    "realized_o2o_return_pct": o2o_return,
                    "side_adjusted_c2c_pct": side_c2c,
                    "side_adjusted_o2o_pct": side_o2o,
                    "hit_c2c": float(c2c_return > 0),
                    "hit_o2o": float(o2o_return > 0),
                    "side_hit_c2c": float(side_c2c > 0),
                    "side_hit_o2o": float(side_o2o > 0),
                }
            )
    exploded = pd.DataFrame(rows)
    if exploded.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            exploded.groupby(["feature", "side"], dropna=False)
            .agg(
                n=("feature", "size"),
                avg_c2c_return_pct=("realized_c2c_return_pct", "mean"),
                avg_o2o_return_pct=("realized_o2o_return_pct", "mean"),
                avg_side_adjusted_c2c_pct=("side_adjusted_c2c_pct", "mean"),
                avg_side_adjusted_o2o_pct=("side_adjusted_o2o_pct", "mean"),
                hit_rate_c2c_pct=("hit_c2c", lambda values: float(values.mean() * 100.0)),
                hit_rate_o2o_pct=("hit_o2o", lambda values: float(values.mean() * 100.0)),
                side_hit_rate_c2c_pct=("side_hit_c2c", lambda values: float(values.mean() * 100.0)),
                side_hit_rate_o2o_pct=("side_hit_o2o", lambda values: float(values.mean() * 100.0)),
            )
            .reset_index()
            .sort_values(["side", "avg_c2c_return_pct"], ascending=[True, False], kind="mergesort")
        )
    summary.to_csv(output_dir / "attribution_outcome_summary.csv", index=False)
    return summary


def save_v2_score_diagnostics(checks: pd.DataFrame, output_dir: Path) -> None:
    diagnostic_checks = checks
    if not checks.empty and "score_provider" in checks.columns:
        providers = set(checks["score_provider"].dropna().astype(str))
        if providers <= {"llm_adjustment", "llm_rank"}:
            diagnostic_checks = checks.copy()
            diagnostic_checks["score_provider"] = "llm"
    v1.save_score_diagnostics(diagnostic_checks, output_dir, fail_on_copy=False)


def scorecard_for_bundle(
    run_name: str,
    bundle: v1.ExperimentBundle,
    variant: V2Variant,
    scorer: str,
    is_anchor_average: bool = False,
    score_pool_per_side: int | None = None,
) -> pd.DataFrame:
    base = v1.build_comparison_scorecard({run_name: bundle})
    for idx, row in base.iterrows():
        column = v1._variant_column(str(row["execution_model"]), float(row["cost_bps"]))
        if column in bundle.weekly_returns.columns:
            for metric, value in _capital_metric_values(bundle.weekly_returns[column]).items():
                base.at[idx, metric] = value
    base["return_basis"] = SPREAD_RETURN_BASIS
    base["display_return_basis"] = CAPITAL_RETURN_BASIS
    base.insert(0, "score_pool_per_side", score_pool_per_side if score_pool_per_side is not None else np.nan)
    base.insert(0, "scorer", scorer)
    base.insert(0, "is_anchor_average", bool(is_anchor_average))
    base.insert(0, "anchor_offset", variant.anchor_offset)
    base.insert(0, "positions_per_side", int(variant.positions_per_side))
    base.insert(0, "pool_per_side", int(variant.pool_per_side))
    return base


def build_anchor_average_rows(scorecard: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "annual_return_pct",
        "annual_vol_pct",
        "sharpe",
        "total_return_pct",
        "cagr_pct",
        "max_drawdown_pct",
        "hit_rate_pct",
        "ic_mean",
        "ic_std",
        "ic_tstat",
        "decile_spread_pct",
        "avg_turnover_long",
        "avg_turnover_short",
        "avg_turnover_total",
        "capital_annual_return_pct",
        "capital_annual_vol_pct",
        "capital_total_return_pct",
        "capital_cagr_pct",
        "capital_max_drawdown_pct",
        "capital_hit_rate_pct",
        "n_periods",
    ]
    grouping = ["pool_per_side", "positions_per_side", "score_pool_per_side", "scorer", "run", "execution_model", "cost_bps"]
    rows: list[dict[str, Any]] = []
    source = scorecard[~scorecard["is_anchor_average"]].copy()
    for keys, group in source.groupby(grouping, dropna=False):
        row = {column: value for column, value in zip(grouping, keys)}
        row["anchor_offset"] = "average"
        row["is_anchor_average"] = True
        row["return_basis"] = SPREAD_RETURN_BASIS
        row["display_return_basis"] = CAPITAL_RETURN_BASIS
        for column in metric_columns:
            if column in group.columns:
                row[column] = float(pd.to_numeric(group[column], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def load_experiment_inputs(
    config: rb.BenchmarkConfig,
) -> tuple[dict[str, pd.DataFrame], rb.MembershipLookup, rb.MarketFrames, v1.FeatureFrames, pd.DataFrame | None]:
    return v1.load_experiment_inputs(config)


async def collect_llm_scores_for_v2_pools(
    pools_by_key: dict[str, v1.CandidatePool],
    cache_path: Path,
    model: str,
    temperature: float,
    rate_limit: float,
    thinking_mode: str = technical_llm.DEFAULT_THINKING_MODE,
    anonymize_tickers: bool = False,
    rationale_top_k_per_side: int = 0,
    structured_reasoning: bool = False,
    prompt_variant: str = PROMPT_VARIANT_V2,
    print_reasoning_content: bool = False,
    thinking_call_policy: str = DEFAULT_THINKING_CALL_POLICY,
    llm_max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    run_output_token_budget: int = DEFAULT_RUN_OUTPUT_TOKEN_BUDGET,
    llm_max_attempts: int = DEFAULT_LLM_MAX_ATTEMPTS,
    run_call_budget: int = AUTORESEARCH_DEFAULT_CALL_BUDGET,
    experimental_prompt_spec: ExperimentalPromptSpec | None = None,
    autoresearch_mode: bool = False,
    signal_date_manifest: SignalDateManifest | None = None,
    progress_file: Path | None = None,
    shared_call_budget: LLMCallBudget | None = None,
    shared_token_budget: OutputTokenBudget | None = None,
) -> dict[str, dict[str, Any]]:
    if experimental_prompt_spec is not None:
        prompt_identity = EXPERIMENTAL_PROMPT_VARIANT
        prompt_text = experimental_prompt_spec.prompt_text
        prompt_hash = experimental_prompt_spec.prompt_hash
        prompt_version = f"llm_reversal_signal_v2_experimental_{prompt_hash}"
        prompt_source = f"experimental_prompt_spec:{experimental_prompt_spec.candidate_id}"
        output_contract = experimental_prompt_spec.output_contract
        score_min = experimental_prompt_spec.score_min
        score_max = experimental_prompt_spec.score_max
        score_direction = experimental_prompt_spec.score_direction
        side_scope = experimental_prompt_spec.side_scope
        replicate_id = experimental_prompt_spec.replicate_id
        autoresearch_mode = True
    else:
        prompt_identity = prompt_variant
        prompt_text = prompt_text_for_variant(prompt_variant)
        prompt_hash = prompt_hash_for_variant(prompt_variant)
        prompt_version = prompt_version_for_variant(prompt_variant)
        prompt_source = prompt_source_for_variant(prompt_variant, structured_reasoning=structured_reasoning)
        output_contract = output_contract_for_variant(prompt_variant)
        score_min, score_max, score_direction = 0.0, 100.0, "higher_is_better"
        side_scope = "both"
        replicate_id = "r1"
    cache = v1.load_cache(cache_path)
    split_side_thinking = (
        thinking_mode == "enabled"
        and thinking_call_policy == "split-side"
        and not structured_reasoning
        and int(rationale_top_k_per_side) == 0
    )
    if split_side_thinking and output_contract != "score_0_100":
        raise ValueError("Split-side thinking supports only the V2 0-100 score output contract")
    if split_side_thinking and prompt_variant in {PROMPT_VARIANT_V24, PROMPT_VARIANT_V25}:
        raise ValueError(f"{prompt_variant} currently supports the single-call non-thinking path only")
    llm_call_policy = (
        "split_by_side_thinking"
        if split_side_thinking
        else ("bounded_single_call" if autoresearch_mode else "single_call")
    )
    cache.setdefault("signals", {})
    cache.setdefault("partial_signals", {})
    cache.setdefault("llm_call_usage", [])
    existing_metadata = cache.get("metadata") if isinstance(cache.get("metadata"), dict) else {}
    if autoresearch_mode and cache.get("signals"):
        expected_spec_hash = experimental_prompt_spec.spec_sha256 if experimental_prompt_spec is not None else None
        if existing_metadata.get("experimental_prompt_spec_sha256") != expected_spec_hash:
            raise ValueError("Existing cache belongs to a different experimental prompt specification")
        if str(existing_metadata.get("side_scope", "both")) != side_scope:
            raise ValueError("Existing cache belongs to a different experimental side scope")
        if str(existing_metadata.get("replicate_id", "r1")) != replicate_id:
            raise ValueError("Existing cache belongs to a different experimental replicate")
        if signal_date_manifest is None:
            raise ValueError("Bounded autoresearch cache reuse requires a signal-date manifest")
        cached_dates: set[pd.Timestamp] = set()
        for storage_key, entry in cache.get("signals", {}).items():
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid cached autoresearch entry at {storage_key!r}")
            raw_date = entry.get("rebalance_date", str(storage_key).split("|", 1)[0])
            cached_dates.add(_normalize_signal_date(raw_date))
        requested_dates = set(signal_date_manifest.date_set)
        if not (
            cached_dates.issubset(requested_dates)
            or requested_dates.issubset(cached_dates)
        ):
            extra = sorted(cached_dates - requested_dates)
            missing = sorted(requested_dates - cached_dates)
            raise ValueError(
                "Existing cache and requested signal-date manifest are not nested; "
                f"extra cached dates={[str(date.date()) for date in extra]}, "
                f"uncached requested dates={[str(date.date()) for date in missing]}"
            )
    if (shared_call_budget is None) != (shared_token_budget is None):
        raise ValueError("Shared autoresearch call and token budgets must be supplied together")
    token_budget = shared_token_budget or OutputTokenBudget(
        limit=int(run_output_token_budget),
        used=_cached_output_tokens(cache),
    )
    call_budget = shared_call_budget or LLMCallBudget(
        limit=int(run_call_budget),
        used=_cached_llm_calls(cache),
    )
    if int(call_budget.limit) != int(run_call_budget):
        raise ValueError("Shared call-budget limit does not match run_call_budget")
    if int(token_budget.limit) != int(run_output_token_budget):
        raise ValueError("Shared token-budget limit does not match run_output_token_budget")
    authorized_manifest_hashes = {
        str(value)
        for value in existing_metadata.get("authorized_date_manifest_hashes", [])
        if value
    }
    if existing_metadata.get("date_manifest_hash"):
        authorized_manifest_hashes.add(str(existing_metadata["date_manifest_hash"]))
    if signal_date_manifest is not None:
        authorized_manifest_hashes.add(signal_date_manifest.date_manifest_hash)
    cache["metadata"] = {
        "schema_version": 2,
        "artifact_type": "llm_reversal_signal_v2_scores",
        "scorer_source": prompt_source,
        "incumbent_spec": incumbent_configuration_manifest(),
        "prompt_variant": prompt_identity,
        "prompt_hash": prompt_hash,
        "prompt_version": prompt_version,
        "prompt_style": V2_PROMPT_STYLE,
        "output_contract": output_contract,
        "market_regime_input": (
            "omitted" if prompt_variant == PROMPT_VARIANT_V2_NO_REGIME else "included"
        ),
        "score_min": float(score_min),
        "score_max": float(score_max),
        "score_direction": score_direction,
        "model": model,
        "temperature": float(temperature),
        "thinking_mode": thinking_mode,
        "thinking_call_policy": thinking_call_policy,
        "llm_call_policy": llm_call_policy,
        "llm_max_output_tokens": int(llm_max_output_tokens),
        "run_output_token_budget": int(run_output_token_budget),
        "run_output_tokens_used": int(token_budget.used),
        "llm_max_attempts": int(llm_max_attempts),
        "run_call_budget": int(run_call_budget),
        "run_calls_used": int(call_budget.used),
        "pool_count": len(pools_by_key),
        "storage_key": "date|pool_per_side",
        "anonymize_tickers": bool(anonymize_tickers),
        "rationale_top_k_per_side": int(rationale_top_k_per_side),
        "structured_reasoning": bool(structured_reasoning),
        "print_reasoning_content": bool(print_reasoning_content),
        "memory_spec_hash": v25m.memory_spec_hash() if prompt_variant == PROMPT_VARIANT_V25 else None,
        "memory_spec": v25m.memory_spec() if prompt_variant == PROMPT_VARIANT_V25 else None,
        "autoresearch_mode": bool(autoresearch_mode),
        "side_scope": side_scope,
        "replicate_id": replicate_id,
        "experimental_candidate_id": (
            experimental_prompt_spec.candidate_id if experimental_prompt_spec is not None else None
        ),
        "experimental_generation": (
            experimental_prompt_spec.generation if experimental_prompt_spec is not None else None
        ),
        "experimental_parent_id": (
            experimental_prompt_spec.parent_id if experimental_prompt_spec is not None else None
        ),
        "experimental_prompt_sha256": (
            experimental_prompt_spec.prompt_sha256 if experimental_prompt_spec is not None else None
        ),
        "experimental_prompt_spec_sha256": (
            experimental_prompt_spec.spec_sha256 if experimental_prompt_spec is not None else None
        ),
        "experimental_prompt_spec_path": (
            str(experimental_prompt_spec.source_path) if experimental_prompt_spec is not None else None
        ),
        "date_manifest_hash": (
            signal_date_manifest.date_manifest_hash if signal_date_manifest is not None else None
        ),
        "date_manifest_source_sha256": (
            signal_date_manifest.source_sha256 if signal_date_manifest is not None else None
        ),
        "authorized_date_manifest_hashes": sorted(authorized_manifest_hashes),
        "signal_dates": (
            [str(date.date()) for date in signal_date_manifest.dates]
            if signal_date_manifest is not None
            else None
        ),
    }
    async with httpx.AsyncClient(timeout=technical_llm.PORTFOLIO_REQUEST_TIMEOUT) as client:
        items = sorted(pools_by_key.items(), key=lambda item: item[0])
        for idx, (storage_key, pool) in enumerate(items, start=1):
            market_context = dict(pool.market_context or {})

            def record_cache_hit() -> None:
                append_progress_event(
                    progress_file,
                    "cache_hit",
                    storage_key=storage_key,
                    signal_date=str(pool.signal_date.date()),
                    prompt_identity=prompt_identity,
                    prompt_hash=prompt_hash,
                    cache_path=str(cache_path.resolve()),
                )

            existing = cache["signals"].get(storage_key)
            if isinstance(existing, dict):
                try:
                    if bool(existing.get("anonymize_tickers", False)) != bool(anonymize_tickers):
                        raise ValueError(
                            f"cached anonymize_tickers={existing.get('anonymize_tickers', False)!r} "
                            f"does not match requested {bool(anonymize_tickers)!r}"
                        )
                    if int(existing.get("rationale_top_k_per_side", 0) or 0) != int(rationale_top_k_per_side):
                        raise ValueError(
                            f"cached rationale_top_k_per_side={existing.get('rationale_top_k_per_side', 0)!r} "
                            f"does not match requested {int(rationale_top_k_per_side)!r}"
                        )
                    if bool(existing.get("structured_reasoning", False)) != bool(structured_reasoning):
                        raise ValueError(
                            f"cached structured_reasoning={existing.get('structured_reasoning', False)!r} "
                            f"does not match requested {bool(structured_reasoning)!r}"
                        )
                    if str(existing.get("prompt_variant", PROMPT_VARIANT_V2)) != prompt_identity:
                        raise ValueError(
                            f"cached prompt_variant={existing.get('prompt_variant', PROMPT_VARIANT_V2)!r} "
                            f"does not match requested {prompt_identity!r}"
                        )
                    if str(existing.get("prompt_hash", prompt_hash_for_variant(PROMPT_VARIANT_V2))) != prompt_hash:
                        raise ValueError(
                            f"cached prompt_hash={existing.get('prompt_hash')!r} does not match requested {prompt_hash!r}"
                        )
                    if str(existing.get("thinking_mode", technical_llm.DEFAULT_THINKING_MODE)) != thinking_mode:
                        raise ValueError(
                            f"cached thinking_mode={existing.get('thinking_mode', technical_llm.DEFAULT_THINKING_MODE)!r} "
                            f"does not match requested {thinking_mode!r}"
                        )
                    if not np.isclose(float(existing.get("temperature", np.nan)), float(temperature)):
                        raise ValueError(
                            f"cached temperature={existing.get('temperature')!r} "
                            f"does not match requested {float(temperature)!r}"
                        )
                    if split_side_thinking and str(existing.get("llm_call_policy", "")) != llm_call_policy:
                        raise ValueError(
                            f"cached llm_call_policy={existing.get('llm_call_policy')!r} "
                            f"does not match requested {llm_call_policy!r}"
                        )
                    if prompt_variant == PROMPT_VARIANT_V24 and existing.get("market_context") != market_context:
                        raise ValueError("cached V2.4 raw market context does not match requested pool context")
                    if (
                        prompt_variant == PROMPT_VARIANT_V25
                        and existing.get("memory_spec_hash") != v25m.memory_spec_hash()
                    ):
                        raise ValueError("cached V2.5 memory specification does not match the current frozen spec")
                    if prompt_variant == PROMPT_VARIANT_V22:
                        expected_cache_key = v1._cache_key(pool.signal_date, prompt_version, model, pool.universe_hash)
                        if existing.get("cache_key") != expected_cache_key:
                            raise ValueError("cached V2.2 key does not match requested pool/model/prompt")
                        validate_llm_adjustments(existing, pool.long_tickers, pool.short_tickers)
                        print(f"  [{idx}/{len(items)}] {storage_key} cached")
                        record_cache_hit()
                        continue
                    if prompt_variant == PROMPT_VARIANT_V23:
                        expected_cache_key = v1._cache_key(pool.signal_date, prompt_version, model, pool.universe_hash)
                        if existing.get("cache_key") != expected_cache_key:
                            raise ValueError(f"cached {prompt_variant} key does not match requested pool/model/prompt")
                        validate_llm_rankings(existing, pool.long_tickers, pool.short_tickers)
                        print(f"  [{idx}/{len(items)}] {storage_key} cached")
                        record_cache_hit()
                        continue
                    if autoresearch_mode and side_scope in {"long", "short"}:
                        if str(existing.get("side_scope", "")) != side_scope:
                            raise ValueError("cached side_scope does not match the requested side")
                        if str(existing.get("replicate_id", "")) != replicate_id:
                            raise ValueError("cached replicate_id does not match the requested replicate")
                        expected_tickers = (
                            pool.long_tickers if side_scope == "long" else pool.short_tickers
                        )
                        validate_split_side_scores(
                            {"scores": existing.get("scores")},
                            expected_tickers,
                            side_scope.upper(),
                        )
                        print(f"  [{idx}/{len(items)}] {storage_key} cached")
                        record_cache_hit()
                        continue
                    cached = v1._entry_from_cache(existing, pool, model, prompt_version, V2_PROMPT_STYLE)
                    if structured_reasoning and cached is not None:
                        clean_structured_reasoning_metadata(existing, cached)
                    if cached is not None and rationale_top_k_per_side > 0:
                        validate_attributions(existing, cached, rationale_top_k_per_side)
                    if cached is not None:
                        print(f"  [{idx}/{len(items)}] {storage_key} cached")
                        record_cache_hit()
                        continue
                except ValueError as exc:
                    cache.setdefault("failures", {})[storage_key] = {
                        "error": f"Invalid cached V2 score entry: {exc}",
                        "parse_status": "invalid_cache",
                        "prompt_variant": prompt_identity,
                        "prompt_hash": prompt_hash,
                        "prompt_version": prompt_version,
                        "prompt_style": V2_PROMPT_STYLE,
                        "output_contract": output_contract,
                        "model": model,
                        "thinking_mode": thinking_mode,
                        "universe_hash": pool.universe_hash,
                    }
                    cache["signals"].pop(storage_key, None)
                    v1.save_cache(cache, cache_path)

            print(f"  [{idx}/{len(items)}] {storage_key} calling LLM")
            started = time.time()
            try:
                llm_pool = pool
                anon_to_real: dict[str, str] = {}
                if anonymize_tickers:
                    llm_pool, anon_to_real = anonymize_pool_for_llm(pool)
                if split_side_thinking:
                    partial = cache["partial_signals"].get(storage_key)
                    partial_identity = {
                        "storage_key": storage_key,
                        "model": model,
                        "thinking_mode": thinking_mode,
                        "llm_call_policy": llm_call_policy,
                        "prompt_variant": prompt_identity,
                        "prompt_hash": prompt_hash,
                        "universe_hash": llm_pool.universe_hash,
                        "anonymize_tickers": bool(anonymize_tickers),
                    }
                    if not isinstance(partial, dict) or any(
                        partial.get(key) != value for key, value in partial_identity.items()
                    ):
                        partial = {**partial_identity, "sides": {}}
                        cache["partial_signals"][storage_key] = partial
                        v1.save_cache(cache, cache_path)
                    partial.setdefault("sides", {})

                    def record_call_usage(record: dict[str, Any]) -> None:
                        cache["llm_call_usage"].append({"storage_key": storage_key, **record})
                        cache["metadata"]["run_output_tokens_used"] = int(token_budget.used)
                        v1.save_cache(cache, cache_path)

                    side_results: dict[str, dict[str, Any]] = {}
                    for side, expected_tickers in (
                        ("LONG", llm_pool.long_tickers),
                        ("SHORT", llm_pool.short_tickers),
                    ):
                        side_key = side.lower()
                        partial_side = partial["sides"].get(side_key)
                        if isinstance(partial_side, dict):
                            try:
                                validate_split_side_scores(
                                    {"scores": partial_side.get("scores")}, expected_tickers, side
                                )
                                side_results[side_key] = partial_side
                                print(f"      {side} side cached; resuming remaining side")
                                continue
                            except ValueError:
                                partial["sides"].pop(side_key, None)
                                v1.save_cache(cache, cache_path)

                        identical_length_failure = next(
                            (
                                row
                                for row in reversed(cache.get("llm_call_usage", []))
                                if isinstance(row, dict)
                                and row.get("storage_key") == storage_key
                                and row.get("side") == side
                                and row.get("finish_reason") == "length"
                                and row.get("model") == model
                                and row.get("prompt_hash") == prompt_hash
                                and int(row.get("max_output_tokens", 0) or 0)
                                == int(llm_max_output_tokens)
                            ),
                            None,
                        )
                        if identical_length_failure is not None:
                            raise LLMSafeStop(
                                f"{side} side previously ended with finish_reason=length at "
                                f"max_tokens={int(llm_max_output_tokens):,}. The identical call will not be "
                                "sent again. Use a new cache contract with a larger output cap or change the "
                                "request before retrying."
                            )

                        print(f"      calling {side} side")
                        side_result = await call_llm_score_side_thinking(
                            client=client,
                            pool=llm_pool,
                            side=side,
                            model=model,
                            prompt_variant=prompt_variant,
                            max_output_tokens=int(llm_max_output_tokens),
                            max_attempts=int(llm_max_attempts),
                            token_budget=token_budget,
                            usage_callback=record_call_usage,
                            print_reasoning_content=bool(print_reasoning_content),
                        )
                        partial["sides"][side_key] = side_result
                        side_results[side_key] = side_result
                        v1.save_cache(cache, cache_path)
                        if side == "LONG" and "short" not in partial["sides"]:
                            await asyncio.sleep(rate_limit)

                    result = combine_split_side_score_results(
                        side_results["long"],
                        side_results["short"],
                        llm_pool.long_tickers,
                        llm_pool.short_tickers,
                    )
                elif structured_reasoning:
                    result = await call_llm_scores_with_structured_reasoning(
                        client=client,
                        pool=llm_pool,
                        model=model,
                        temperature=temperature,
                        thinking_mode=thinking_mode,
                    )
                elif rationale_top_k_per_side > 0:
                    result = await call_llm_scores_with_attribution(
                        client=client,
                        pool=llm_pool,
                        model=model,
                        temperature=temperature,
                        top_k_per_side=rationale_top_k_per_side,
                        thinking_mode=thinking_mode,
                    )
                elif autoresearch_mode:
                    def record_bounded_call_usage(record: dict[str, Any]) -> None:
                        cache["llm_call_usage"].append({"storage_key": storage_key, **record})
                        cache["metadata"]["run_output_tokens_used"] = int(token_budget.used)
                        cache["metadata"]["run_calls_used"] = int(call_budget.used)
                        v1.save_cache(cache, cache_path)

                    result = await call_llm_scores_bounded_nonthinking(
                        client=client,
                        pool=llm_pool,
                        model=model,
                        temperature=temperature,
                        system_prompt=prompt_text,
                        prompt_identity=prompt_identity,
                        prompt_hash=prompt_hash,
                        max_output_tokens=min(int(llm_max_output_tokens), AUTORESEARCH_MAX_OUTPUT_TOKENS),
                        max_attempts=int(llm_max_attempts),
                        token_budget=token_budget,
                        call_budget=call_budget,
                        usage_callback=record_bounded_call_usage,
                        progress_file=progress_file,
                        score_min=float(score_min),
                        score_max=float(score_max),
                        score_direction=score_direction,
                        side_scope=side_scope,
                    )
                else:
                    result = await call_llm_scores_for_prompt_variant(
                        client=client,
                        pool=llm_pool,
                        model=model,
                        temperature=temperature,
                        prompt_variant=prompt_variant,
                        thinking_mode=thinking_mode,
                        print_reasoning_content=bool(print_reasoning_content),
                    )
                if anonymize_tickers:
                    result = (
                        deanonymize_side_scores(result, anon_to_real)
                        if side_scope in {"long", "short"}
                        else deanonymize_llm_scores(result, anon_to_real)
                    )
            except Exception as exc:
                cache.setdefault("failures", {})[storage_key] = {
                    "error": str(exc),
                    "parse_status": "failed",
                    "prompt_variant": prompt_identity,
                    "prompt_hash": prompt_hash,
                    "prompt_version": prompt_version,
                    "prompt_style": V2_PROMPT_STYLE,
                    "output_contract": output_contract,
                    "model": model,
                    "temperature": float(temperature),
                    "thinking_mode": thinking_mode,
                    "universe_hash": pool.universe_hash,
                    "pool_per_side": len(pool.long_tickers),
                    "regime": pool.regime,
                    "market_context": market_context,
                    "memory_spec_hash": v25m.memory_spec_hash() if prompt_variant == PROMPT_VARIANT_V25 else None,
                    "anonymize_tickers": bool(anonymize_tickers),
                    "rationale_top_k_per_side": int(rationale_top_k_per_side),
                    "structured_reasoning": bool(structured_reasoning),
                    "print_reasoning_content": bool(print_reasoning_content),
                    "failed_at": pd.Timestamp.utcnow().isoformat(),
                    "v2_storage_key": storage_key,
                    "v2_latency_seconds": time.time() - started,
                    "side_scope": side_scope,
                    "replicate_id": replicate_id,
                }
                v1.save_cache(cache, cache_path)
                raise
            cache_entry = {
                **result,
                "rebalance_date": str(pool.signal_date.date()),
                "prompt_variant": prompt_identity,
                "prompt_hash": prompt_hash,
                "prompt_version": prompt_version,
                "prompt_style": V2_PROMPT_STYLE,
                "output_contract": output_contract,
                "score_min": float(score_min),
                "score_max": float(score_max),
                "score_direction": score_direction,
                "model": model,
                "temperature": float(temperature),
                "thinking_mode": thinking_mode,
                "thinking_call_policy": thinking_call_policy,
                "llm_call_policy": llm_call_policy,
                "llm_max_output_tokens": (
                    int(llm_max_output_tokens)
                    if split_side_thinking
                    else (
                        min(int(llm_max_output_tokens), AUTORESEARCH_MAX_OUTPUT_TOKENS)
                        if autoresearch_mode
                        else 8192
                    )
                ),
                "run_output_token_budget": int(run_output_token_budget),
                "llm_max_attempts": (
                    int(llm_max_attempts)
                    if split_side_thinking or autoresearch_mode
                    else v1.LLM_SCORE_MAX_ATTEMPTS
                ),
                "run_call_budget": int(run_call_budget) if autoresearch_mode else None,
                "universe_hash": pool.universe_hash,
                "cache_key": v1._cache_key(pool.signal_date, prompt_version, model, pool.universe_hash),
                "pool_per_side": len(pool.long_tickers),
                "regime": pool.regime,
                "market_context": market_context,
                "memory_spec_hash": v25m.memory_spec_hash() if prompt_variant == PROMPT_VARIANT_V25 else None,
                "v2_storage_key": storage_key,
                "v2_latency_seconds": time.time() - started,
                "anonymize_tickers": bool(anonymize_tickers),
                "rationale_top_k_per_side": int(rationale_top_k_per_side),
                "structured_reasoning": bool(structured_reasoning),
                "print_reasoning_content": bool(print_reasoning_content),
                "side_scope": side_scope,
                "replicate_id": replicate_id,
            }
            cache["signals"][storage_key] = cache_entry
            cache["partial_signals"].pop(storage_key, None)
            cache.setdefault("failures", {}).pop(storage_key, None)
            cache["metadata"]["run_output_tokens_used"] = int(token_budget.used)
            cache["metadata"]["run_calls_used"] = int(call_budget.used)
            v1.save_cache(cache, cache_path)
            append_progress_event(
                progress_file,
                "cache_entry_saved",
                storage_key=storage_key,
                signal_date=str(pool.signal_date.date()),
                prompt_identity=prompt_identity,
                prompt_hash=prompt_hash,
                cache_path=str(cache_path.resolve()),
                completed_pools=len(cache["signals"]),
                planned_pools=len(pools_by_key),
            )
            if idx < len(items):
                await asyncio.sleep(rate_limit)
    if split_side_thinking and len(pools_by_key) == 1:
        storage_key = next(iter(pools_by_key))
        records = [
            row for row in cache.get("llm_call_usage", [])
            if isinstance(row, dict) and row.get("storage_key") == storage_key
        ]
        output_tokens = sum(int(row.get("output_tokens_charged", 0) or 0) for row in records)
        attempts = len(records)
        smoke_passed = (
            storage_key in cache["signals"]
            and attempts == 2
            and output_tokens <= THINKING_SMOKE_OUTPUT_TOKEN_LIMIT
        )
        status = "PASS" if smoke_passed else "FAIL"
        print(
            f"\nThinking smoke gate: {status} | calls={attempts} | "
            f"combined output tokens={output_tokens:,} | "
            f"limit={THINKING_SMOKE_OUTPUT_TOKEN_LIMIT:,}"
        )
        if not smoke_passed:
            print("Do not start the full paid run until this smoke gate passes.")
    if autoresearch_mode:
        # Persist manifest authorization metadata even when every requested
        # date was a cache hit (or a nested projection supplied no new pools).
        v1.save_cache(cache, cache_path)
    if autoresearch_mode and signal_date_manifest is not None:
        allowed = signal_date_manifest.date_set
        return {
            storage_key: entry
            for storage_key, entry in cache["signals"].items()
            if _normalize_signal_date(
                entry.get("rebalance_date", str(storage_key).split("|", 1)[0])
                if isinstance(entry, dict)
                else str(storage_key).split("|", 1)[0]
            )
            in allowed
        }
    return cache["signals"]


def _return_metric_values(returns: pd.Series, periods_per_year: int = rb.DEFAULT_PERIODS_PER_YEAR) -> dict[str, Any]:
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    ann_ret = float(returns.mean() * periods_per_year) if not returns.empty else np.nan
    ann_vol = float(returns.std() * np.sqrt(periods_per_year)) if len(returns) > 1 else np.nan
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    return {
        "annual_return_pct": ann_ret * 100.0 if np.isfinite(ann_ret) else np.nan,
        "annual_vol_pct": ann_vol * 100.0 if np.isfinite(ann_vol) else np.nan,
        "sharpe": sharpe,
        "total_return_pct": float(((1.0 + returns).prod() - 1.0) * 100.0) if not returns.empty else np.nan,
        "cagr_pct": rb.safe_cagr(returns, periods_per_year=periods_per_year) if not returns.empty else np.nan,
        "max_drawdown_pct": rb.compute_max_drawdown_pct(returns) if not returns.empty else np.nan,
        "hit_rate_pct": float((returns > 0).mean() * 100.0) if not returns.empty else np.nan,
        "n_periods": int(len(returns)),
    }


def _capital_metric_values(
    spread_returns: pd.Series,
    periods_per_year: int = rb.DEFAULT_PERIODS_PER_YEAR,
) -> dict[str, Any]:
    metrics = _return_metric_values(spread_returns * CAPITAL_RETURN_SCALE, periods_per_year=periods_per_year)
    return {f"capital_{key}": value for key, value in metrics.items() if key != "n_periods"}


def _comparison_metric_row(returns: pd.Series, periods_per_year: int, extra: dict[str, Any]) -> dict[str, Any]:
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        **extra,
        **_return_metric_values(returns, periods_per_year=periods_per_year),
        **_capital_metric_values(returns, periods_per_year=periods_per_year),
        "return_basis": SPREAD_RETURN_BASIS,
        "display_return_basis": CAPITAL_RETURN_BASIS,
    }


def _return_column(execution_model: str, cost_bps: float) -> str:
    prefix = "ls_c2c" if str(execution_model) == "C2C" else "ls_o2o"
    return f"{prefix}_{rb.format_bps_label(float(cost_bps))}"


def _capital_metrics_from_return_file(path: Path, execution_model: str, cost_bps: float) -> dict[str, Any]:
    if not path.exists():
        return {}
    returns = pd.read_csv(path)
    column = _return_column(execution_model, cost_bps)
    if column not in returns.columns:
        return {}
    return _capital_metric_values(pd.to_numeric(returns[column], errors="coerce"))


def _legacy_anchor_capital_metrics(
    legacy_root: Path,
    anchor: str,
    execution_model: str,
    cost_bps: float,
) -> dict[str, Any]:
    column = _return_column(execution_model, cost_bps)
    all_returns_path = legacy_root / "all_anchor_returns.csv"
    if anchor == "Average" and all_returns_path.exists():
        returns = pd.read_csv(all_returns_path)
        if "anchor" not in returns.columns or column not in returns.columns:
            return {}
        rows: list[dict[str, Any]] = []
        for _, group in returns.groupby("anchor"):
            rows.append(_capital_metric_values(pd.to_numeric(group[column], errors="coerce")))
        if not rows:
            return {}
        return {
            key: float(pd.to_numeric([row.get(key) for row in rows], errors="coerce").mean())
            for key in rows[0]
        }

    named_path = legacy_root / f"{anchor.lower()}_returns.csv"
    metrics = _capital_metrics_from_return_file(named_path, execution_model, cost_bps)
    if metrics:
        return metrics
    if all_returns_path.exists():
        returns = pd.read_csv(all_returns_path)
        if "anchor" in returns.columns and column in returns.columns:
            subset = returns[returns["anchor"].astype(str) == anchor]
            if not subset.empty:
                return _capital_metric_values(pd.to_numeric(subset[column], errors="coerce"))
    return {}


def _v1_run_dir(run: str) -> str | None:
    if run == "Baseline-Pool":
        return "baseline_pool"
    if run == "LLM-Pool v1":
        return "llm_pool_v1"
    if run == "Baseline-Universe":
        return "baseline_universe"
    return None


def _v1_capital_metrics(root: Path, run: str, execution_model: str, cost_bps: float) -> dict[str, Any]:
    run_dir = _v1_run_dir(run)
    if run_dir is None:
        return {}
    return _capital_metrics_from_return_file(root / run_dir / "weekly_returns.csv", execution_model, cost_bps)


def _display_metric(row: pd.Series, capital_column: str, spread_column: str) -> Any:
    capital = pd.to_numeric(row.get(capital_column), errors="coerce")
    if pd.notna(capital):
        return capital
    return row.get(spread_column)


def _display_basis(row: pd.Series) -> str:
    capital = pd.to_numeric(row.get("capital_cagr_pct"), errors="coerce")
    if pd.notna(capital):
        return CAPITAL_RETURN_BASIS
    return str(row.get("return_basis", SPREAD_RETURN_BASIS))


def legacy_anchor_gross_comparison_rows(legacy_root: Path) -> pd.DataFrame:
    anchor_summary_path = legacy_root / "anchor_summary.csv"
    if anchor_summary_path.exists():
        summary = pd.read_csv(anchor_summary_path)
        required = {"anchor", "execution_model", "cost_bps"}
        if required.issubset(summary.columns):
            gross_c2c = summary[
                (summary["execution_model"].astype(str) == "C2C")
                & (pd.to_numeric(summary["cost_bps"], errors="coerce") == 0.0)
            ].copy()
            preferred = gross_c2c[gross_c2c["anchor"].astype(str).isin(["Average", "Friday"])]
            if not preferred.empty:
                gross_c2c = preferred
            if not gross_c2c.empty:
                rows: list[dict[str, Any]] = []
                for _, row in gross_c2c.iterrows():
                    anchor = str(row["anchor"])
                    row_data = {
                        "family": "Legacy",
                        "run": "Legacy Det Raw Anchor Average"
                        if anchor == "Average"
                        else f"Legacy Det Raw {anchor}",
                        "harness": "legacy raw anchor replay",
                        "prompt_schema": "none: deterministic Trail5 raw score",
                        "score_source": "deterministic Trail5 raw score on row pool",
                        "candidate_pool": "60 biggest 5-day losers + 60 biggest 5-day winners",
                        "book": "30/30 equal-weight long-short",
                        "cadence": "every 5 trading days per anchor",
                        "accounting": "gross fixed 5-trading-day C2C",
                        "liquidity_filters": COMPARISON_LIQUIDITY_FILTERS,
                        "replay_cache": "offline deterministic replay; no LLM cache",
                        "basis": "5-day anchor gross C2C",
                        "pool_book": "60/60 -> 30/30",
                        "anchor_offset": anchor,
                        "scorer": "mean_reversion_raw_score",
                        "return_basis": SPREAD_RETURN_BASIS,
                        **{column: row[column] for column in gross_c2c.columns if column not in {"anchor", "run"}},
                    }
                    row_data.update(
                        _legacy_anchor_capital_metrics(
                            legacy_root=legacy_root,
                            anchor=anchor,
                            execution_model=str(row["execution_model"]),
                            cost_bps=float(row["cost_bps"]),
                        )
                    )
                    row_data["display_return_basis"] = (
                        CAPITAL_RETURN_BASIS
                        if pd.notna(pd.to_numeric(row_data.get("capital_cagr_pct"), errors="coerce"))
                        else SPREAD_RETURN_BASIS
                    )
                    rows.append(row_data)
                return pd.DataFrame(rows)
    return pd.DataFrame()


def _legacy_fixed_window_returns_from_selections(selections: pd.DataFrame) -> pd.Series:
    required = {"rebalance_date", "position_type", "realized_return_pct"}
    if not required.issubset(selections.columns):
        missing = ", ".join(sorted(required - set(selections.columns)))
        raise ValueError(f"Legacy portfolio selections missing required columns: {missing}")
    data = selections.copy()
    data["rebalance_date"] = pd.to_datetime(data["rebalance_date"])
    data["realized_return_pct"] = pd.to_numeric(data["realized_return_pct"], errors="coerce")
    grouped = data.groupby(["rebalance_date", "position_type"])["realized_return_pct"].mean().unstack()
    if "LONG" not in grouped.columns or "SHORT" not in grouped.columns:
        raise ValueError("Legacy portfolio selections must contain both LONG and SHORT rows")
    # Legacy short selection returns are already signed P&L, so add the two legs
    # to match the V1/V2 long-minus-short spread convention.
    returns = (grouped["LONG"] + grouped["SHORT"]) / 100.0
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    returns.index.name = "date"
    return returns


def legacy_factor_gross_comparison_rows(legacy_factor_root: Path, include_deterministic: bool = True) -> pd.DataFrame:
    leaderboard_path = legacy_factor_root / "compact_leaderboard.csv"
    ic_by_score: dict[str, float] = {}
    if leaderboard_path.exists():
        leaderboard = pd.read_csv(leaderboard_path)
        if {"score_type", "ic_mean"}.issubset(leaderboard.columns):
            ic_by_score = {
                str(row["score_type"]): float(row["ic_mean"])
                for _, row in leaderboard[["score_type", "ic_mean"]].dropna().iterrows()
            }

    rows: list[dict[str, Any]] = []
    for score_type, label in LEGACY_LABELS.items():
        if not include_deterministic and not score_type.startswith("llm_"):
            continue
        selections_path = legacy_factor_root / score_type / "portfolio_selections.csv"
        if not selections_path.exists():
            continue
        selections = pd.read_csv(selections_path)
        try:
            returns = _legacy_fixed_window_returns_from_selections(selections)
        except ValueError:
            continue

        review_path = legacy_factor_root / score_type / "review_log.csv"
        avg_turnover_total = np.nan
        if review_path.exists():
            review = pd.read_csv(review_path)
            if "turnover_total" in review.columns:
                avg_turnover_total = float(pd.to_numeric(review["turnover_total"], errors="coerce").mean())

        row = _comparison_metric_row(
            returns,
            periods_per_year=rb.DEFAULT_PERIODS_PER_YEAR,
            extra={
                "family": "Legacy",
                "run": label,
                "harness": "legacy factor selections replay",
                "prompt_schema": "deterministic or legacy cached LLM composite score",
                "score_source": "legacy selected book from score column",
                "candidate_pool": "60 biggest 5-day losers + 60 biggest 5-day winners",
                "book": "30/30 equal-weight long-short",
                "cadence": "every 5 trading days on legacy schedule",
                "accounting": "gross fixed 5-trading-day C2C",
                "liquidity_filters": COMPARISON_LIQUIDITY_FILTERS,
                "replay_cache": "legacy portfolio selections replayed offline",
                "basis": "legacy selected-book gross C2C",
                "pool_book": "60/60 -> 30/30",
                "anchor_offset": "legacy schedule",
                "scorer": score_type,
                "execution_model": "C2C",
                "cost_bps": 0.0,
                "ic_mean": ic_by_score.get(score_type, np.nan),
                "avg_turnover_total": avg_turnover_total,
            },
        )
        rows.append(row)
    return pd.DataFrame(rows)


def legacy_gross_comparison_rows(legacy_root: Path, legacy_factor_root: Path | None = None) -> pd.DataFrame:
    anchor_rows = legacy_anchor_gross_comparison_rows(legacy_root)
    factor_rows = pd.DataFrame()
    if legacy_factor_root is not None:
        factor_rows = legacy_factor_gross_comparison_rows(
            legacy_factor_root,
            include_deterministic=anchor_rows.empty,
        )
    frames = [frame for frame in (anchor_rows, factor_rows) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _infer_v1_pool_book(root: Path, run: str) -> str:
    if run == "Baseline-Universe":
        return "full universe -> deciles"
    name = root.name.lower()
    pool = 60 if "pool60" in name else 150
    book = 30 if "pos30" in name else 75
    return f"{pool}/{pool} -> {book}/{book}"


def _infer_v1_candidate_pool(root: Path, run: str) -> str:
    if run == "Baseline-Universe":
        return "full valid Russell universe, scored into deciles"
    name = root.name.lower()
    pool = 60 if "pool60" in name else 150
    return f"{pool} biggest 5-day losers + {pool} biggest 5-day winners"


def _infer_v1_book(root: Path, run: str) -> str:
    if run == "Baseline-Universe":
        return "D1 losers long / D10 winners short"
    name = root.name.lower()
    book = 30 if "pos30" in name else 75
    return f"{book}/{book} equal-weight long-short"


def v1_gross_comparison_rows(v1_roots: tuple[Path, ...]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for root in v1_roots:
        scorecard_path = root / "comparison_scorecard.csv"
        if not scorecard_path.exists():
            continue
        scorecard = pd.read_csv(scorecard_path)
        required = {"run", "execution_model", "cost_bps"}
        if not required.issubset(scorecard.columns):
            continue
        gross_c2c = scorecard[
            (scorecard["execution_model"].astype(str) == "C2C")
            & (pd.to_numeric(scorecard["cost_bps"], errors="coerce") == 0.0)
        ].copy()
        if gross_c2c.empty:
            continue
        gross_c2c.insert(0, "family", "V1")
        gross_c2c.insert(2, "basis", "Friday gross C2C")
        gross_c2c.insert(3, "pool_book", [_infer_v1_pool_book(root, str(run)) for run in gross_c2c["run"]])
        gross_c2c.insert(4, "anchor_offset", "Friday")
        gross_c2c.insert(5, "scorer", gross_c2c["run"].map(lambda run: "trail5" if "Baseline" in str(run) else "llm_score"))
        gross_c2c["harness"] = "v1 Friday fixed-hold harness"
        gross_c2c["prompt_schema"] = gross_c2c["run"].map(
            lambda run: "unchanged v1 0-100 score prompt" if "LLM" in str(run) else "none: deterministic Trail5 baseline"
        )
        gross_c2c["score_source"] = gross_c2c["run"].map(
            lambda run: "v1 LLM scored same row pool" if "LLM" in str(run) else "deterministic Trail5 on row pool"
        )
        gross_c2c["candidate_pool"] = [_infer_v1_candidate_pool(root, str(run)) for run in gross_c2c["run"]]
        gross_c2c["book"] = [_infer_v1_book(root, str(run)) for run in gross_c2c["run"]]
        gross_c2c["cadence"] = "Friday-only weekly schedule"
        gross_c2c["accounting"] = "gross fixed 5-trading-day C2C"
        gross_c2c["liquidity_filters"] = COMPARISON_LIQUIDITY_FILTERS
        gross_c2c["replay_cache"] = "scores cached once per Friday pool; book replay offline"
        gross_c2c["return_basis"] = SPREAD_RETURN_BASIS
        for idx, row in gross_c2c.iterrows():
            for metric, value in _v1_capital_metrics(
                root,
                str(row["run"]),
                str(row["execution_model"]),
                float(row["cost_bps"]),
            ).items():
                gross_c2c.at[idx, metric] = value
        if "capital_cagr_pct" not in gross_c2c.columns:
            gross_c2c["capital_cagr_pct"] = np.nan
        gross_c2c["display_return_basis"] = gross_c2c["capital_cagr_pct"].map(
            lambda value: CAPITAL_RETURN_BASIS if pd.notna(pd.to_numeric(value, errors="coerce")) else SPREAD_RETURN_BASIS
        )
        rows.append(gross_c2c)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    return combined.drop_duplicates(subset=["family", "run", "pool_book", "execution_model", "cost_bps"], keep="first")


def v2_gross_comparison_rows(scorecard: pd.DataFrame) -> pd.DataFrame:
    if scorecard.empty:
        return pd.DataFrame()
    required = {"execution_model", "cost_bps", "pool_per_side", "positions_per_side", "run", "is_anchor_average"}
    if not required.issubset(scorecard.columns):
        return pd.DataFrame()
    source_gross_c2c = scorecard[
        (scorecard["execution_model"].astype(str) == "C2C")
        & (pd.to_numeric(scorecard["cost_bps"], errors="coerce") == 0.0)
    ].copy()
    anchor_average = source_gross_c2c[source_gross_c2c["is_anchor_average"].astype(bool)].copy()
    if not anchor_average.empty:
        anchor_average["run"] = anchor_average["run"].map(lambda run: f"{run} Anchor Avg")
        anchor_detail = source_gross_c2c[~source_gross_c2c["is_anchor_average"].astype(bool)].copy()
        best_rows: list[pd.Series] = []
        grouping = ["pool_per_side", "positions_per_side", "scorer", "run", "execution_model", "cost_bps"]
        for _, group in anchor_detail.groupby(grouping, dropna=False):
            metric_col = "capital_cagr_pct" if "capital_cagr_pct" in group.columns else "cagr_pct"
            metric = pd.to_numeric(group[metric_col], errors="coerce")
            if metric.dropna().empty:
                continue
            best = group.loc[metric.idxmax()].copy()
            best["run"] = f"{best['run']} Best Offset"
            best_rows.append(best)
        gross_c2c = pd.concat([anchor_average, pd.DataFrame(best_rows)], ignore_index=True, sort=False)
    else:
        gross_c2c = source_gross_c2c
    if gross_c2c.empty:
        return pd.DataFrame()
    friday_only = gross_c2c["anchor_offset"].astype(str).eq("Friday").all()
    gross_c2c.insert(0, "family", "V2")
    gross_c2c.insert(
        2,
        "basis",
        gross_c2c["anchor_offset"].map(
            lambda anchor: "Friday gross C2C"
            if str(anchor) == "Friday"
            else ("5-anchor average gross C2C" if str(anchor) == "average" else "best single-offset gross C2C")
        ),
    )
    gross_c2c.insert(
        3,
        "pool_book",
        gross_c2c.apply(
            lambda row: f"{int(row['pool_per_side'])}/{int(row['pool_per_side'])} -> "
            f"{int(row['positions_per_side'])}/{int(row['positions_per_side'])}",
            axis=1,
        ),
    )
    if "scorer" not in gross_c2c.columns:
        gross_c2c.insert(5, "scorer", "")
    if "return_basis" not in gross_c2c.columns:
        gross_c2c["return_basis"] = SPREAD_RETURN_BASIS
    if "capital_cagr_pct" not in gross_c2c.columns:
        gross_c2c["capital_cagr_pct"] = np.nan
    if "display_return_basis" not in gross_c2c.columns:
        gross_c2c["display_return_basis"] = np.nan
    gross_c2c["display_return_basis"] = gross_c2c.apply(
        lambda row: (
            str(row["display_return_basis"])
            if pd.notna(row.get("display_return_basis"))
            else (
                CAPITAL_RETURN_BASIS
                if pd.notna(pd.to_numeric(row.get("capital_cagr_pct"), errors="coerce"))
                else SPREAD_RETURN_BASIS
            )
        ),
        axis=1,
    )
    gross_c2c["harness"] = "v2 Friday replay harness" if friday_only else "v2 multi-anchor replay harness"
    gross_c2c["prompt_schema"] = gross_c2c["run"].map(
        lambda run: (
            "V2.2 -3..+3 contextual adjustment prompt"
            if "v2.2" in str(run)
            else (
                "V2.4 frozen 0-100 score prompt with raw SPY/VIX context"
                if "v2.4" in str(run)
                else (
                    "V2.3 rank-only ordering prompt"
                    if "v2.3" in str(run)
                    else ("V2 0-100 score prompt" if "LLM" in str(run) else "none: deterministic Trail5 baseline")
                )
            )
        )
    )
    gross_c2c["score_source"] = gross_c2c.apply(
        lambda row: (
            f"LLM adjusted {int(row['score_pool_per_side'])}/{int(row['score_pool_per_side'])}; "
            f"lambda replay offline; row pool {int(row['pool_per_side'])}/{int(row['pool_per_side'])}"
            if "v2.2" in str(row["run"]) and pd.notna(row.get("score_pool_per_side"))
            else f"LLM ranked {int(row['score_pool_per_side'])}/{int(row['score_pool_per_side'])}; "
            f"row pool {int(row['pool_per_side'])}/{int(row['pool_per_side'])}"
            if (
                "v2.3" in str(row["run"])
                or str(row.get("scorer")) == "llm_rank_order"
            )
            and pd.notna(row.get("score_pool_per_side"))
            else f"LLM scored {int(row['score_pool_per_side'])}/{int(row['score_pool_per_side'])}; "
            f"row pool {int(row['pool_per_side'])}/{int(row['pool_per_side'])}"
            if ("LLM" in str(row["run"]) or str(row.get("scorer")) == "llm_score")
            and pd.notna(row.get("score_pool_per_side"))
            else "deterministic Trail5 on row pool"
        ),
        axis=1,
    )
    gross_c2c["candidate_pool"] = gross_c2c["pool_per_side"].map(
        lambda pool: f"{int(pool)} biggest 5-day losers + {int(pool)} biggest 5-day winners"
    )
    gross_c2c["book"] = gross_c2c["positions_per_side"].map(
        lambda book: f"{int(book)}/{int(book)} equal-weight long-short"
    )
    gross_c2c["cadence"] = (
        gross_c2c["anchor_offset"].map(
            lambda anchor: "Friday-only weekly schedule"
            if str(anchor) == "Friday"
            else (
                "five independent every-5-trading-day anchor offsets"
                if str(anchor) == "average"
                else f"best single every-5-trading-day offset {anchor}"
            )
        )
    )
    gross_c2c["accounting"] = "gross Friday-to-Friday C2C" if friday_only else "gross fixed 5-trading-day C2C"
    gross_c2c["liquidity_filters"] = COMPARISON_LIQUIDITY_FILTERS
    gross_c2c["replay_cache"] = gross_c2c.apply(
        lambda row: (
            f"LLM scores cached once per date at pool {int(row['score_pool_per_side'])}; "
            "smaller pools and books replay offline"
            if ("LLM" in str(row["run"]) or str(row.get("scorer")) == "llm_score")
            and pd.notna(row.get("score_pool_per_side"))
            else "offline deterministic replay; no LLM cache"
        ),
        axis=1,
    )
    return gross_c2c


def _v2_reference_label(root: Path) -> str:
    name = root.name.lower()
    if "anon_control" in name:
        return "Anon V2 Control"
    if name == DEFAULT_OUTPUT_DIR.name.lower() or "signal_v2_2009_2014" in name:
        return "Original V2"
    return root.name


def v2_reference_gross_comparison_rows(v2_reference_roots: tuple[Path, ...]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for root in v2_reference_roots:
        scorecard_path = root / "comparison_scorecard.csv"
        if not scorecard_path.exists():
            continue
        try:
            scorecard = pd.read_csv(scorecard_path)
        except Exception:  # noqa: BLE001
            continue
        gross = v2_gross_comparison_rows(scorecard)
        if gross.empty:
            continue
        label = _v2_reference_label(root)
        gross["family"] = "V2 Reference"
        gross["run"] = gross["run"].map(
            lambda run: f"{label} {str(run).replace(' Anchor Avg', '').replace(' Best Offset', '')}"
        )
        gross["replay_cache"] = gross["replay_cache"].map(lambda value: f"reference artifact: {root}; {value}")
        rows.append(gross)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False).drop_duplicates(
        subset=["family", "run", "pool_book", "anchor_offset", "execution_model", "cost_bps"],
        keep="first",
    )


def build_gross_system_comparison(
    scorecard: pd.DataFrame,
    legacy_root: Path,
    legacy_factor_root: Path | None,
    v1_roots: tuple[Path, ...],
    v2_reference_roots: tuple[Path, ...] = (),
) -> pd.DataFrame:
    frames = [
        legacy_gross_comparison_rows(legacy_root, legacy_factor_root),
        v1_gross_comparison_rows(v1_roots),
        v2_reference_gross_comparison_rows(v2_reference_roots),
        v2_gross_comparison_rows(scorecard),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    comparison = pd.concat(frames, ignore_index=True, sort=False)
    preferred_columns = [
        "family",
        "run",
        "harness",
        "prompt_schema",
        "score_source",
        "candidate_pool",
        "book",
        "cadence",
        "accounting",
        "liquidity_filters",
        "replay_cache",
        "display_return_basis",
        "return_basis",
        "basis",
        "pool_book",
        "anchor_offset",
        "scorer",
        "execution_model",
        "cost_bps",
        "annual_return_pct",
        "annual_vol_pct",
        "sharpe",
        "total_return_pct",
        "cagr_pct",
        "max_drawdown_pct",
        "hit_rate_pct",
        "capital_annual_return_pct",
        "capital_annual_vol_pct",
        "capital_total_return_pct",
        "capital_cagr_pct",
        "capital_max_drawdown_pct",
        "capital_hit_rate_pct",
        "ic_mean",
        "decile_spread_pct",
        "avg_turnover_total",
        "n_periods",
    ]
    for column in preferred_columns:
        if column not in comparison.columns:
            comparison[column] = np.nan
    comparison = comparison[preferred_columns + [column for column in comparison.columns if column not in preferred_columns]]
    family_order = pd.Categorical(comparison["family"], categories=["Legacy", "V1", "V2"], ordered=True)
    def row_order(row: pd.Series) -> int:
        if str(row.get("family")) != "V2":
            return 0
        run = str(row.get("run"))
        if "Anchor Avg" in run and "Baseline" in run:
            return 0
        if "Anchor Avg" in run and "LLM" in run:
            return 1
        if "Best Offset" in run and "Baseline" in run:
            return 2
        if "Best Offset" in run and "LLM" in run:
            return 3
        return 4

    comparison = comparison.assign(_family_order=family_order, _row_order=comparison.apply(row_order, axis=1)).sort_values(
        ["_family_order", "pool_book", "_row_order", "run"],
        kind="mergesort",
    )
    return comparison.drop(columns=["_family_order", "_row_order"]).reset_index(drop=True)


def write_v2_html_report(output_dir: Path, scorecard: pd.DataFrame, gross_comparison: pd.DataFrame | None = None) -> Path:
    output_path = output_dir / "llm_reversal_v2_report.html"

    def fmt_num(value: Any, digits: int = 2) -> str:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return ""
        return f"{float(numeric):.{digits}f}"

    def fmt_pct(value: Any, digits: int = 1, signed: bool = False) -> str:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return ""
        prefix = "+" if signed and float(numeric) > 0 else ""
        return f"{prefix}{float(numeric):.{digits}f}%"

    def scorer_label(value: Any) -> str:
        text = str(value)
        if text == "llm_score":
            return "LLM"
        if text == "llm_rank_order":
            return "LLM rank order"
        if text == "trail5":
            return "Trail5"
        if text.startswith("llm_"):
            return "Legacy LLM"
        return text

    def run_label(row: pd.Series) -> str:
        pool = int(row["pool_per_side"])
        book = int(row["positions_per_side"])
        return f"{pool}/{pool} -> {book}/{book}"

    def metric_value(value: Any) -> float:
        text = str(value).strip().replace("%", "").replace("+", "").replace(",", "")
        if not text:
            return np.nan
        return float(pd.to_numeric(text, errors="coerce"))

    def cell_text(value: Any) -> str:
        if isinstance(value, float | np.floating):
            if pd.isna(value):
                return ""
            return f"{float(value):.3f}"
        if isinstance(value, int | np.integer):
            return str(int(value))
        if pd.isna(value):
            return ""
        return str(value)

    def display_cagr(row: pd.Series) -> Any:
        return _display_metric(row, "capital_cagr_pct", "cagr_pct")

    def display_total(row: pd.Series) -> Any:
        return _display_metric(row, "capital_total_return_pct", "total_return_pct")

    def display_drawdown(row: pd.Series) -> Any:
        return _display_metric(row, "capital_max_drawdown_pct", "max_drawdown_pct")

    def merged_display_metric(row: pd.Series, suffix: str, capital_column: str, spread_column: str) -> float:
        capital = pd.to_numeric(row.get(f"{capital_column}{suffix}"), errors="coerce")
        if pd.notna(capital):
            return float(capital)
        spread = pd.to_numeric(row.get(f"{spread_column}{suffix}"), errors="coerce")
        return float(spread) if pd.notna(spread) else np.nan

    def simple_table(
        rows: list[dict[str, Any]],
        *,
        css_class: str = "compact-table",
        highlight_cols: set[str] | None = None,
    ) -> str:
        if not rows:
            return '<p class="muted">No rows available.</p>'
        df = pd.DataFrame(rows)
        highlight_cols = highlight_cols or set()
        best_values: dict[str, float] = {}
        worst_values: dict[str, float] = {}
        for column in highlight_cols:
            if column not in df.columns:
                continue
            values = df[column].map(metric_value).replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty or values.nunique() <= 1:
                continue
            best_values[column] = float(values.max())
            worst_values[column] = float(values.min())

        header = "".join(f"<th>{html_lib.escape(str(column))}</th>" for column in df.columns)
        body_rows: list[str] = []
        for _, row in df.iterrows():
            cells: list[str] = []
            for column in df.columns:
                value = row[column]
                classes: list[str] = []
                title = ""
                numeric = metric_value(value) if column in highlight_cols else np.nan
                if column in best_values and np.isclose(numeric, best_values[column], equal_nan=False):
                    classes.append("metric-best")
                    title = "Best in this table"
                elif column in worst_values and np.isclose(numeric, worst_values[column], equal_nan=False):
                    classes.append("metric-worst")
                    title = "Worst in this table"
                class_attr = f' class="{" ".join(classes)}"' if classes else ""
                title_attr = f' title="{html_lib.escape(title)}"' if title else ""
                cells.append(
                    f"<td{class_attr}{title_attr}>{html_lib.escape(cell_text(value))}</td>"
                )
            body_rows.append("<tr>" + "".join(cells) + "</tr>")
        return f'<table class="{html_lib.escape(css_class)}"><thead><tr>{header}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'

    zero_cost = scorecard[pd.to_numeric(scorecard["cost_bps"], errors="coerce") == 0.0].copy()
    v2_c2c = zero_cost[zero_cost["execution_model"].astype(str) == "C2C"].copy()
    v2_c2c = v2_c2c.sort_values(
        ["pool_per_side", "positions_per_side", "scorer", "run"],
        kind="mergesort",
    )
    anchor_average_mask = (
        v2_c2c["is_anchor_average"].astype(bool)
        if "is_anchor_average" in v2_c2c.columns
        else pd.Series(False, index=v2_c2c.index)
    )
    report_is_anchor_average = bool(anchor_average_mask.any())
    report_is_friday_preview = (
        not report_is_anchor_average
        and not v2_c2c.empty
        and v2_c2c["anchor_offset"].astype(str).eq("Friday").all()
    )
    headline_source = v2_c2c[anchor_average_mask].copy() if report_is_anchor_average else v2_c2c.copy()
    if not headline_source.empty and "score_pool_per_side" in headline_source.columns:
        score_pool_values = pd.to_numeric(
            headline_source.loc[headline_source["run"].astype(str).eq("LLM-Pool v2"), "score_pool_per_side"],
            errors="coerce",
        ).dropna()
    else:
        score_pool_values = pd.Series(dtype=float)
    if score_pool_values.empty and not headline_source.empty and "pool_per_side" in headline_source.columns:
        score_pool_values = pd.to_numeric(headline_source["pool_per_side"], errors="coerce").dropna()
    score_pool_side = int(score_pool_values.max()) if not score_pool_values.empty else DEFAULT_LLM_POOL_SIZE
    score_pool_phrase = f"{score_pool_side}/{score_pool_side}"
    llm_pass_note = f"one {score_pool_phrase} LLM score pass"

    if report_is_anchor_average:
        report_scope_label = "5-anchor every-5-trading-day"
        report_scope_note = "Anchor-average rows across offsets 0-4; each offset rebalances every 5 trading days."
        v2_section_title = "V2 Anchor-Average Gross C2C"
        v2_section_note = "Same accounting and fixed 5-trading-day holding window, summarized across all five trading-day offsets."
        run_shape_value = "5 anchors"
        run_shape_note = f"Offsets 0-4; {llm_pass_note} per signal date; configured books replay offline"
    elif report_is_friday_preview:
        report_scope_label = "Friday preview"
        report_scope_note = "Friday-only cost-saving preview; not the final legacy-style V2 cadence."
        v2_section_title = "V2 Friday Preview Gross C2C"
        v2_section_note = "Same accounting and same Friday dates. This is a preview slice, not the full every-5-trading-day V2."
        run_shape_value = "Friday preview"
        run_shape_note = f"Gross 0 bps; {llm_pass_note} per Friday; configured books replay offline"
    else:
        report_scope_label = "fixed 5-trading-day"
        report_scope_note = "Fixed 5-trading-day C2C rows."
        v2_section_title = "V2 Gross C2C"
        v2_section_note = "Gross fixed 5-trading-day C2C rows."
        run_shape_value = "fixed 5D"
        run_shape_note = f"{llm_pass_note} per signal date; configured books replay offline"

    headline_rows: list[dict[str, Any]] = []
    for _, row in headline_source.iterrows():
        headline_rows.append(
            {
                "View": run_label(row),
                "Scorer": scorer_label(row["scorer"]),
                "Return Basis": _display_basis(row),
                "Sharpe": fmt_num(row["sharpe"], 3),
                "CAGR": fmt_pct(display_cagr(row)),
                "Total": fmt_pct(display_total(row)),
                "Max DD": fmt_pct(display_drawdown(row)),
                "IC": fmt_num(row["ic_mean"], 4),
            }
        )
    visible_metric_cols = {"Sharpe", "CAGR", "Total", "Max DD", "IC"}
    raw_metric_cols = {
        "sharpe",
        "cagr_pct",
        "total_return_pct",
        "max_drawdown_pct",
        "hit_rate_pct",
        "ic_mean",
        "decile_spread_pct",
        "annual_return_pct",
        "capital_cagr_pct",
        "capital_total_return_pct",
        "capital_max_drawdown_pct",
        "capital_annual_return_pct",
    }
    headline_html = simple_table(headline_rows, highlight_cols=visible_metric_cols)

    strongest_offset_section = ""
    if report_is_anchor_average:
        anchor_detail_source = v2_c2c[~anchor_average_mask].copy()
        llm_anchor_rows = anchor_detail_source[anchor_detail_source["run"].astype(str) == "LLM-Pool v2"].copy()
        if not llm_anchor_rows.empty:
            llm_anchor_rows["_display_cagr_pct"] = llm_anchor_rows.apply(display_cagr, axis=1)
            strongest_llm = llm_anchor_rows.sort_values("_display_cagr_pct", ascending=False).iloc[0]
            strongest_anchor = str(strongest_llm["anchor_offset"])
            strongest_pool = int(strongest_llm["pool_per_side"])
            strongest_book = int(strongest_llm["positions_per_side"])
            strongest_rows = anchor_detail_source[
                (anchor_detail_source["anchor_offset"].astype(str) == strongest_anchor)
                & (pd.to_numeric(anchor_detail_source["pool_per_side"], errors="coerce") == strongest_pool)
                & (pd.to_numeric(anchor_detail_source["positions_per_side"], errors="coerce") == strongest_book)
            ].copy()
            strongest_rows = strongest_rows.sort_values(["scorer", "run"], kind="mergesort")
            strongest_table_rows: list[dict[str, Any]] = []
            for _, row in strongest_rows.iterrows():
                strongest_table_rows.append(
                    {
                        "Offset": strongest_anchor,
                        "View": run_label(row),
                        "Scorer": scorer_label(row["scorer"]),
                        "Return Basis": _display_basis(row),
                        "Sharpe": fmt_num(row["sharpe"], 3),
                        "CAGR": fmt_pct(display_cagr(row)),
                        "Total": fmt_pct(display_total(row)),
                        "Max DD": fmt_pct(display_drawdown(row)),
                        "IC": fmt_num(row["ic_mean"], 4),
                    }
                )
            strongest_offset_section = f"""
    <section class="section">
      <h2>Strongest Single Offset</h2>
      <p>Best individual offset by V2 LLM displayed C2C CAGR. The Trail5 row shown here is the paired baseline for the same offset, pool, book, accounting, and return basis.</p>
      <div class="legend"><span class="legend-swatch legend-best">Best</span><span class="legend-swatch legend-worst">Worst</span><span>Within this offset comparison.</span></div>
      <div class="table-wrap">{simple_table(strongest_table_rows, highlight_cols=visible_metric_cols)}</div>
    </section>
"""

    lift_cards: list[str] = []
    v2_baseline = headline_source[headline_source["run"].astype(str) == "Baseline-Pool"].copy()
    v2_llm = headline_source[headline_source["run"].astype(str) == "LLM-Pool v2"].copy()
    if not v2_baseline.empty and not v2_llm.empty:
        paired = v2_llm.merge(
            v2_baseline,
            on=["pool_per_side", "positions_per_side", "execution_model", "cost_bps"],
            suffixes=("_llm", "_base"),
        )
        if not paired.empty:
            paired["cagr_lift"] = paired.apply(
                lambda row: merged_display_metric(row, "_llm", "capital_cagr_pct", "cagr_pct")
                - merged_display_metric(row, "_base", "capital_cagr_pct", "cagr_pct"),
                axis=1,
            )
            paired["sharpe_lift"] = pd.to_numeric(paired["sharpe_llm"], errors="coerce") - pd.to_numeric(
                paired["sharpe_base"],
                errors="coerce",
            )
            best_lift = paired.sort_values("cagr_lift", ascending=False).iloc[0]
            lift_cards.append(
                f"""
      <div class="summary-item">
        <div class="summary-label">Best LLM Lift</div>
        <div class="summary-value">{int(best_lift['pool_per_side'])}/{int(best_lift['pool_per_side'])} -> {int(best_lift['positions_per_side'])}/{int(best_lift['positions_per_side'])}</div>
        <div class="summary-note">CAGR {fmt_pct(best_lift['cagr_lift'], signed=True)}, Sharpe {fmt_num(best_lift['sharpe_lift'])}</div>
      </div>
"""
            )
    if not v2_llm.empty:
        v2_llm = v2_llm.copy()
        v2_llm["_display_cagr_pct"] = v2_llm.apply(display_cagr, axis=1)
        v2_llm["_display_total_return_pct"] = v2_llm.apply(display_total, axis=1)
        best_cagr = v2_llm.sort_values("_display_cagr_pct", ascending=False).iloc[0]
        best_sharpe = v2_llm.sort_values("sharpe", ascending=False).iloc[0]
        lift_cards.extend(
            [
                f"""
      <div class="summary-item">
        <div class="summary-label">Best V2 LLM CAGR</div>
        <div class="summary-value">{run_label(best_cagr)}</div>
        <div class="summary-note">{fmt_pct(best_cagr['_display_cagr_pct'])} CAGR, {fmt_pct(best_cagr['_display_total_return_pct'])} total</div>
      </div>
""",
                f"""
      <div class="summary-item">
        <div class="summary-label">Best V2 LLM Sharpe</div>
        <div class="summary-value">{run_label(best_sharpe)}</div>
        <div class="summary-note">{fmt_num(best_sharpe['sharpe'], 3)} Sharpe, {fmt_pct(display_cagr(best_sharpe))} CAGR</div>
      </div>
""",
            ]
        )
    lift_cards.append(
        f"""
      <div class="summary-item">
        <div class="summary-label">Run Shape</div>
        <div class="summary-value">{run_shape_value}</div>
        <div class="summary-note">{run_shape_note}</div>
      </div>
"""
    )
    summary_html = "\n".join(lift_cards)
    if (
        not headline_source.empty
        and (pd.to_numeric(headline_source["pool_per_side"], errors="coerce") == 150).any()
        and (pd.to_numeric(headline_source["positions_per_side"], errors="coerce") == 75).any()
    ):
        headline_note = "Read this first: V2 LLM helped the concentrated books. The diversified 150/150 -> 75/75 book did not improve versus Trail5."
    else:
        headline_note = "Read this first: the headline compares Trail5 vs LLM under the same pool, book, cadence, accounting, and return basis."

    comparison_html = ""
    method_html = ""
    full_comparison_html = ""
    legacy_daily_context_section = ""
    legacy_alignment_note = ""
    if gross_comparison is not None and not gross_comparison.empty:
        is_daily_path = gross_comparison["execution_model"].astype(str).str.lower().eq("daily")
        main_comparison = gross_comparison[~is_daily_path].copy()
        legacy_daily_context = gross_comparison[is_daily_path].copy()
        if main_comparison["anchor_offset"].astype(str).eq("3").any():
            legacy_alignment_note = """
      <p class="note">Schedule note: V2 offset 3 is aligned with the legacy factor/LLM schedule start. Both first rebalance on 2009-01-06, a Tuesday, then continue every 5 trading days.</p>
"""

        compact_rows: list[dict[str, Any]] = []
        compact = main_comparison.copy()
        def report_row_order(row: pd.Series) -> int:
            if str(row.get("family")) != "V2":
                return 0
            run = str(row.get("run"))
            if "Anchor Avg" in run and "Baseline" in run:
                return 0
            if "Anchor Avg" in run and "LLM" in run:
                return 1
            if "Best Offset" in run and "Baseline" in run:
                return 2
            if "Best Offset" in run and "LLM" in run:
                return 3
            return 4

        compact = compact.assign(_row_order=compact.apply(report_row_order, axis=1))
        compact = compact.sort_values(["family", "pool_book", "_row_order", "run"], kind="mergesort")
        for _, row in compact.iterrows():
            compact_rows.append(
                {
                    "Family": row["family"],
                    "Run": row["run"],
                    "Anchor": row.get("anchor_offset", ""),
                    "Pool -> Book": row["pool_book"],
                    "Accounting": row["execution_model"],
                    "Scorer": scorer_label(row["scorer"]),
                    "Return Basis": _display_basis(row),
                    "Sharpe": fmt_num(row["sharpe"], 3),
                    "CAGR": fmt_pct(_display_metric(row, "capital_cagr_pct", "cagr_pct")),
                    "Total": fmt_pct(_display_metric(row, "capital_total_return_pct", "total_return_pct")),
                    "Max DD": fmt_pct(_display_metric(row, "capital_max_drawdown_pct", "max_drawdown_pct")),
                    "IC": fmt_num(row["ic_mean"], 4),
                }
            )
        comparison_html = simple_table(compact_rows, highlight_cols=visible_metric_cols)

        daily_rows: list[dict[str, Any]] = []
        if not legacy_daily_context.empty:
            legacy_daily_context = legacy_daily_context.sort_values(["family", "pool_book", "run"], kind="mergesort")
            for _, row in legacy_daily_context.iterrows():
                daily_rows.append(
                    {
                        "Family": row["family"],
                        "Run": row["run"],
                        "Pool -> Book": row["pool_book"],
                        "Accounting": row["execution_model"],
                        "Scorer": scorer_label(row["scorer"]),
                        "Return Basis": _display_basis(row),
                        "Sharpe": fmt_num(row["sharpe"], 3),
                        "CAGR": fmt_pct(_display_metric(row, "capital_cagr_pct", "cagr_pct")),
                        "Total": fmt_pct(_display_metric(row, "capital_total_return_pct", "total_return_pct")),
                        "Max DD": fmt_pct(_display_metric(row, "capital_max_drawdown_pct", "max_drawdown_pct")),
                        "IC": fmt_num(row["ic_mean"], 4),
                    }
                )
            legacy_daily_context_html = simple_table(daily_rows, highlight_cols=visible_metric_cols)
            legacy_daily_context_section = f"""
      <details>
        <summary>Legacy daily LLM context, not apples-to-apples</summary>
        <p>The rows below are old legacy factor-path rows with gross daily MTM accounting. They are kept for historical context only and are excluded from the main comparison above.</p>
        <div class="table-wrap">{legacy_daily_context_html}</div>
      </details>
"""

        method_rows: list[dict[str, Any]] = []
        method_cols = ["family", "run", "harness", "cadence", "accounting", "score_source"]
        method_source = main_comparison[method_cols].drop_duplicates().copy()
        for _, row in method_source.iterrows():
            method_rows.append(
                {
                    "Family": row["family"],
                    "Run": row["run"],
                    "Harness": row["harness"],
                    "Cadence": row["cadence"],
                    "Accounting": row["accounting"],
                    "Score Source": row["score_source"],
                }
            )
        method_html = simple_table(method_rows, css_class="method-table")

        detail_cols = [
            "family",
            "run",
            "harness",
            "prompt_schema",
            "score_source",
            "candidate_pool",
            "book",
            "cadence",
            "accounting",
            "liquidity_filters",
            "replay_cache",
            "display_return_basis",
            "return_basis",
            "basis",
            "pool_book",
            "anchor_offset",
            "scorer",
            "execution_model",
            "sharpe",
            "cagr_pct",
            "total_return_pct",
            "max_drawdown_pct",
            "capital_cagr_pct",
            "capital_total_return_pct",
            "capital_max_drawdown_pct",
            "ic_mean",
            "n_periods",
        ]
        detail_rows = gross_comparison[[col for col in detail_cols if col in gross_comparison.columns]].copy()
        full_comparison_html = simple_table(
            detail_rows.to_dict("records"),
            css_class="full-table",
            highlight_cols=raw_metric_cols,
        )

    detail_rows = zero_cost.copy()
    detail_rows["anchor_offset"] = detail_rows["anchor_offset"].astype(str)
    detail_rows = detail_rows.sort_values(
        ["execution_model", "cost_bps", "pool_per_side", "positions_per_side", "scorer", "anchor_offset", "run"],
        kind="mergesort",
    )
    detail_display_rows: list[dict[str, Any]] = []
    for _, row in detail_rows.iterrows():
        pool = int(row["pool_per_side"])
        book = int(row["positions_per_side"])
        detail_display_rows.append(
            {
                "Pool -> Book": f"{pool}/{pool} -> {book}/{book}",
                "Anchor": row["anchor_offset"],
                "Scorer": scorer_label(row["scorer"]),
                "Run": row["run"],
                "Accounting": row["execution_model"],
                "Cost bps": fmt_num(row["cost_bps"], 1),
                "Return Basis": _display_basis(row),
                "Sharpe": fmt_num(row["sharpe"], 3),
                "CAGR (50/50)": fmt_pct(display_cagr(row)),
                "Total (50/50)": fmt_pct(display_total(row)),
                "Max DD (50/50)": fmt_pct(display_drawdown(row)),
                "IC": fmt_num(row["ic_mean"], 4),
                "Turnover": fmt_num(row.get("avg_turnover_total"), 3),
                "Raw CAGR (100/100)": fmt_pct(row.get("cagr_pct")),
                "Raw Total (100/100)": fmt_pct(row.get("total_return_pct")),
                "Raw Max DD (100/100)": fmt_pct(row.get("max_drawdown_pct")),
            }
        )
    detail_metric_cols = {
        "Sharpe",
        "CAGR (50/50)",
        "Total (50/50)",
        "Max DD (50/50)",
        "IC",
        "Raw CAGR (100/100)",
        "Raw Total (100/100)",
        "Raw Max DD (100/100)",
    }
    detail_table_html = simple_table(
        detail_display_rows,
        css_class="full-table",
        highlight_cols=detail_metric_cols,
    )

    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Reversal V2 Report</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #657282;
      --line: #dce2ea;
      --head: #eef3f7;
      --accent: #1f7a68;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.42;
    }}
    main {{ max-width: 1380px; margin: 0 auto; padding: 28px; }}
    header {{ margin-bottom: 18px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 10px; font-size: 16px; letter-spacing: 0; }}
    p {{ margin: 0 0 12px; color: var(--muted); max-width: 900px; }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin: 16px 0;
      box-shadow: 0 1px 2px rgba(23, 32, 42, 0.04);
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 12px;
      margin: 16px 0;
    }}
    .summary-item {{
      background: #fbfcfd;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 104px;
    }}
    .summary-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .summary-value {{ margin-top: 7px; font-size: 22px; font-weight: 700; }}
    .summary-note {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: var(--panel); }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: right; vertical-align: top; }}
    th {{ background: var(--head); color: #273747; font-weight: 700; white-space: nowrap; }}
    tr:nth-child(even) td {{ background: #fafbfc; }}
    tr:last-child td {{ border-bottom: 0; }}
    .compact-table td:nth-child(-n+7), .compact-table th:nth-child(-n+7) {{ text-align: left; }}
    .method-table td, .method-table th {{ text-align: left; }}
    .full-table td:nth-child(-n+15), .full-table th:nth-child(-n+15) {{ text-align: left; }}
    td.metric-best {{
      background: #e8f6ef !important;
      color: #0e6245;
      font-weight: 800;
    }}
    td.metric-worst {{
      background: #fff0f0 !important;
      color: #9f2727;
      font-weight: 800;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      margin: 8px 0 12px;
    }}
    .legend-swatch {{
      display: inline-block;
      min-width: 54px;
      padding: 3px 7px;
      border-radius: 6px;
      font-weight: 800;
      text-align: center;
    }}
    .legend-best {{ background: #e8f6ef; color: #0e6245; }}
    .legend-worst {{ background: #fff0f0; color: #9f2727; }}
    .muted {{ color: var(--muted); }}
    .note {{
      border-left: 4px solid var(--accent);
      padding: 10px 12px;
      background: #f2faf7;
      color: #24483f;
      margin: 12px 0 0;
      max-width: 980px;
    }}
    details {{ margin-top: 12px; }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 700; margin-bottom: 10px; }}
    @media (max-width: 900px) {{
      main {{ padding: 18px; }}
      .summary-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>LLM Reversal V2 Report</h1>
      <p>{report_scope_label}, gross 0 bps. LLM rows score the {score_pool_phrase} pool once per signal date and replay configured books offline.</p>
      <p class="muted">Top-line CAGR, Total, and Max DD use the displayed return basis. New runs display 50/50 capital-normalized returns; raw 100/100 spread metrics stay in the audit tables.</p>
      <p class="muted">{report_scope_note}</p>
      <p class="muted">Generated {html_lib.escape(generated_at)} from cached/output CSVs.</p>
    </header>

    <section class="section">
      <h2>Apples-to-Apples Gross C2C</h2>
      <p>All rows here use gross fixed-window C2C accounting. Legacy LLM rows are recomputed from legacy selection files, not daily MTM P&amp;L. The Return Basis column separates exposure scaling from accounting.</p>
      {legacy_alignment_note}
      <div class="legend"><span class="legend-swatch legend-best">Best</span><span class="legend-swatch legend-worst">Worst</span><span>Highlighted across the visible rows in this table.</span></div>
      <div class="table-wrap">{comparison_html}</div>
    </section>

    <section class="section">
      <h2>Method Context</h2>
      <p>This is the translation layer for what each row means.</p>
      <div class="table-wrap">{method_html}</div>
    </section>

    <section class="section">
      <h2>Audit Details</h2>
      {legacy_daily_context_section}
      <details>
        <summary>Full metadata comparison table</summary>
        <div class="table-wrap">{full_comparison_html}</div>
      </details>
      <details>
        <summary>All V2 zero-cost rows, including O2O</summary>
        <div class="table-wrap">{detail_table_html}</div>
      </details>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(report_html, encoding="utf-8")
    return output_path


def collect_v2_llm_source_pools(
    market_frames: rb.MarketFrames,
    feature_frames: v1.FeatureFrames,
    membership_lookup: rb.MembershipLookup,
    spy_df: pd.DataFrame | None,
    config: rb.BenchmarkConfig,
    schedule: str,
    anchor_offsets: tuple[int, ...],
    llm_pool_size: int,
    allowed_signal_dates: frozenset[pd.Timestamp] | None = None,
) -> dict[str, v1.CandidatePool]:
    context = build_v2_pool_context(market_frames)
    pools_by_key: dict[str, v1.CandidatePool] = {}

    if schedule == "friday":
        friday_context = v1.build_pool_context(market_frames)
        signal_items: list[pd.Timestamp | int] = [pd.Timestamp(date) for date in friday_context.fridays]
    elif schedule == "anchors":
        friday_context = None
        signal_items = []
    else:
        raise ValueError(f"Unsupported V2 schedule: {schedule}")

    def add_pool(pool: v1.CandidatePool | None) -> None:
        if pool is None:
            return
        if (
            allowed_signal_dates is not None
            and pd.Timestamp(pool.signal_date).normalize() not in allowed_signal_dates
        ):
            return
        pools_by_key.setdefault(llm_score_storage_key(pool, llm_pool_size), pool)

    if schedule == "friday":
        assert friday_context is not None
        for signal_item in signal_items:
            signal_date = pd.Timestamp(signal_item)
            if signal_date < config.start or signal_date > config.end:
                continue
            if int(context.trading_dates.get_loc(signal_date)) < config.signal_lookback_days:
                continue
            pool = v1.build_candidate_pool(
                market_frames=market_frames,
                feature_frames=feature_frames,
                membership_lookup=membership_lookup,
                spy_df=spy_df,
                signal_date=signal_date,
                pool_per_side=llm_pool_size,
                config=config,
                context=friday_context,
            )
            if pool is not None and pool.exit_date <= config.end:
                add_pool(pool)
        return pools_by_key

    for anchor_offset in anchor_offsets:
        for signal_idx in iter_anchor_signal_indices(config, context, int(anchor_offset)):
            add_pool(
                build_candidate_pool_for_index(
                    market_frames=market_frames,
                    feature_frames=feature_frames,
                    membership_lookup=membership_lookup,
                    spy_df=spy_df,
                    signal_idx=int(signal_idx),
                    pool_per_side=llm_pool_size,
                    config=config,
                    context=context,
                )
            )
    return pools_by_key


def save_v25_comparison_reports(
    output_dir: Path,
    variant: V2Variant,
    bundles_by_run: dict[tuple[str, str], v1.ExperimentBundle],
    checks_by_run: dict[tuple[str, str], pd.DataFrame],
    memory_build: v25m.V25MemoryBuild,
) -> dict[str, str]:
    report_dir = output_dir / "v25_balanced_outcome_memory"
    report_dir.mkdir(parents=True, exist_ok=True)
    coverage = memory_build.coverage_by_date.copy()
    coverage["signal_date"] = pd.to_datetime(coverage["signal_date"])
    scope_dates = {
        "full": set(coverage["signal_date"]),
        "memory_ready": set(coverage.loc[coverage["memory_ready"], "signal_date"]),
        "fully_populated": set(coverage.loc[coverage["fully_populated"], "signal_date"]),
    }
    run_order = ("Trail5", "HistRet-Only", "V2 Fresh Control", "V2.5 Balanced Memory")
    return_column = _return_column("C2C", 0.0)
    metrics_rows: list[dict[str, Any]] = []
    period_returns = pd.DataFrame(index=pd.DatetimeIndex(sorted(scope_dates["full"]), name="date"))
    side_rows: list[dict[str, Any]] = []

    for run_name in run_order:
        bundle = bundles_by_run.get((variant.label, run_name))
        if bundle is None or return_column not in bundle.weekly_returns:
            continue
        weekly = bundle.weekly_returns.copy()
        weekly.index = pd.to_datetime(weekly.index)
        capital_returns = pd.to_numeric(weekly[return_column], errors="coerce") * CAPITAL_RETURN_SCALE
        period_returns[run_name] = capital_returns.reindex(period_returns.index)
        for scope, dates in scope_dates.items():
            scoped = capital_returns.loc[capital_returns.index.isin(dates)].dropna()
            values = _return_metric_values(scoped)
            long_returns = pd.to_numeric(weekly.get("long_ret_c2c"), errors="coerce")
            short_returns = -pd.to_numeric(weekly.get("short_ret_c2c"), errors="coerce")
            long_scoped = long_returns.loc[long_returns.index.isin(dates)].dropna() * CAPITAL_RETURN_SCALE
            short_scoped = short_returns.loc[short_returns.index.isin(dates)].dropna() * CAPITAL_RETURN_SCALE
            metrics_rows.append(
                {
                    "scope": scope,
                    "run": run_name,
                    "n_periods": int(len(scoped)),
                    "cagr_pct": values["cagr_pct"],
                    "sharpe": values["sharpe"],
                    "max_drawdown_pct": values["max_drawdown_pct"],
                    "total_return_pct": values["total_return_pct"],
                    "hit_rate_pct": values["hit_rate_pct"],
                    "annual_long_contribution_pct": float(long_scoped.mean() * rb.DEFAULT_PERIODS_PER_YEAR * 100.0)
                    if not long_scoped.empty
                    else np.nan,
                    "annual_short_contribution_pct": float(short_scoped.mean() * rb.DEFAULT_PERIODS_PER_YEAR * 100.0)
                    if not short_scoped.empty
                    else np.nan,
                }
            )
            side_rows.extend(
                [
                    {
                        "scope": scope,
                        "run": run_name,
                        "side": "LONG",
                        **_return_metric_values(long_scoped),
                    },
                    {
                        "scope": scope,
                        "run": run_name,
                        "side": "SHORT",
                        **_return_metric_values(short_scoped),
                    },
                ]
            )

    metrics = pd.DataFrame(metrics_rows)
    if not metrics.empty:
        for scope, group in metrics.groupby("scope"):
            trail = group.loc[group["run"] == "Trail5", "cagr_pct"]
            control = group.loc[group["run"] == "V2 Fresh Control", "cagr_pct"]
            trail_cagr = float(trail.iloc[0]) if not trail.empty else np.nan
            control_cagr = float(control.iloc[0]) if not control.empty else np.nan
            mask = metrics["scope"] == scope
            metrics.loc[mask, "cagr_lift_vs_trail5_pp"] = metrics.loc[mask, "cagr_pct"] - trail_cagr
            metrics.loc[mask, "cagr_lift_vs_v2_pp"] = metrics.loc[mask, "cagr_pct"] - control_cagr
        metrics.to_csv(report_dir / "coverage_adjusted_paired_scorecard.csv", index=False)
    pd.DataFrame(side_rows).to_csv(report_dir / "side_contribution_comparison.csv", index=False)
    period_returns["memory_ready"] = period_returns.index.isin(scope_dates["memory_ready"])
    period_returns["fully_populated"] = period_returns.index.isin(scope_dates["fully_populated"])
    period_returns.to_csv(report_dir / "paired_period_returns.csv")

    yearly_rows: list[dict[str, Any]] = []
    for run_name in [column for column in run_order if column in period_returns]:
        returns = pd.to_numeric(period_returns[run_name], errors="coerce")
        for year, group in returns.groupby(returns.index.year):
            values = _return_metric_values(group.dropna())
            yearly_rows.append({"year": int(year), "run": run_name, **values})
    pd.DataFrame(yearly_rows).to_csv(report_dir / "yearly_comparison.csv", index=False)

    treatment_checks = checks_by_run.get((variant.label, "V2.5 Balanced Memory"), pd.DataFrame())
    control_checks = checks_by_run.get((variant.label, "V2 Fresh Control"), pd.DataFrame())
    score_analysis = pd.DataFrame()
    if not treatment_checks.empty:
        keep = [
            "date",
            "ticker",
            "pool_side",
            "llm_score",
            "hist_n",
            "hist_win_pct",
            "hist_ret_pct",
            "fwd_c2c",
        ]
        score_analysis = treatment_checks[[column for column in keep if column in treatment_checks]].copy()
        score_analysis = score_analysis.rename(columns={"llm_score": "v25_score"})
        if not control_checks.empty:
            control = control_checks[["date", "ticker", "pool_side", "llm_score"]].rename(
                columns={"llm_score": "v2_control_score"}
            )
            score_analysis = score_analysis.merge(control, on=["date", "ticker", "pool_side"], how="left")
            score_analysis["score_delta_v25_minus_v2"] = (
                pd.to_numeric(score_analysis["v25_score"], errors="coerce")
                - pd.to_numeric(score_analysis["v2_control_score"], errors="coerce")
            )
        score_analysis["side_adjusted_fwd_c2c"] = np.where(
            score_analysis["pool_side"].eq("LONG"),
            pd.to_numeric(score_analysis["fwd_c2c"], errors="coerce"),
            -pd.to_numeric(score_analysis["fwd_c2c"], errors="coerce"),
        )
        score_analysis.to_csv(report_dir / "candidate_score_memory_analysis.csv", index=False)

    if metrics.empty:
        headline = "No completed V2.5 replay is available yet."
        table_html = ""
    else:
        ready = metrics[(metrics["scope"] == "memory_ready") & (metrics["run"] == "V2.5 Balanced Memory")]
        if ready.empty:
            headline = "V2.5 completed, but no memory-ready comparison was available."
        else:
            row = ready.iloc[0]
            headline = (
                f"Memory-ready V2.5: {row['cagr_pct']:.1f}% CAGR, {row['sharpe']:.2f} Sharpe, "
                f"{row.get('cagr_lift_vs_v2_pp', np.nan):+.1f}pp CAGR versus fresh V2."
            )
        display_columns = [
            "scope",
            "run",
            "n_periods",
            "cagr_pct",
            "sharpe",
            "max_drawdown_pct",
            "cagr_lift_vs_trail5_pp",
            "cagr_lift_vs_v2_pp",
        ]
        table_html = metrics[display_columns].round(3).to_html(index=False, border=0, classes="metrics")
    report_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>V2.5 Balanced Outcome Memory</title>
<style>
body{{font-family:Arial,sans-serif;margin:0;background:#f4f7f9;color:#17212b}}
main{{max-width:1200px;margin:0 auto;padding:32px}} section{{background:white;padding:24px;margin:18px 0;border:1px solid #d8e1e8;border-radius:8px}}
h1,h2{{margin-top:0}} .headline{{font-size:20px;font-weight:700;color:#116149}} .note{{color:#5c6b78;line-height:1.5}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th{{background:#eaf0f4;text-align:left}} th,td{{padding:10px;border-bottom:1px solid #d8e1e8}}
code{{background:#edf2f6;padding:2px 5px}}
</style></head><body><main>
<h1>V2.5 Balanced Outcome Memory</h1>
<section><div class="headline">{html_lib.escape(headline)}</div>
<p class="note">All rows use offset 3, 60/60 candidates, 30/30 books, gross fixed five-day C2C, and 50/50 capital normalization. Memory retrieval is regime-agnostic.</p></section>
<section><h2>Paired Results</h2>{table_html}</section>
<section><h2>Memory Contract</h2><p class="note">20 nearest prior same-side candidates, at most four per historical rebalance date, same-ticker history excluded, and only outcomes known by the current signal date. Fields sent to the LLM: <code>HistN</code>, <code>HistWin</code>, and <code>HistRet</code>.</p></section>
</main></body></html>"""
    (report_dir / "v25_balanced_memory_report.html").write_text(report_html, encoding="utf-8")
    return {
        "directory": str(report_dir),
        "scorecard": "v25_balanced_outcome_memory/coverage_adjusted_paired_scorecard.csv",
        "report": "v25_balanced_outcome_memory/v25_balanced_memory_report.html",
    }


async def run_v2_llm_cache_only(
    config: rb.BenchmarkConfig,
    anchor_offsets: tuple[int, ...],
    schedule: str,
    llm_pool_size: int,
    model: str,
    temperature: float,
    thinking_mode: str,
    cache_path: Path,
    rate_limit: float,
    anonymize_tickers: bool = False,
    rationale_top_k_per_side: int = 0,
    structured_reasoning: bool = False,
    prompt_variant: str = PROMPT_VARIANT_V2,
    print_reasoning_content: bool = False,
    thinking_call_policy: str = DEFAULT_THINKING_CALL_POLICY,
    llm_max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    run_output_token_budget: int = DEFAULT_RUN_OUTPUT_TOKEN_BUDGET,
    llm_max_attempts: int = DEFAULT_LLM_MAX_ATTEMPTS,
    output_dir: Path | None = None,
    v25_paired_control: bool = False,
    experimental_prompt_spec: ExperimentalPromptSpec | None = None,
    signal_date_manifest: SignalDateManifest | None = None,
    progress_file: Path | None = None,
    run_call_budget: int = AUTORESEARCH_DEFAULT_CALL_BUDGET,
    paired_control: bool = False,
    paired_control_cache: Path | None = None,
    autoresearch_stage: str | None = None,
) -> dict[str, dict[str, Any]]:
    _, membership_lookup, market_frames, feature_frames, spy_df = load_experiment_inputs(config)
    pools_by_key = collect_v2_llm_source_pools(
        market_frames=market_frames,
        feature_frames=feature_frames,
        membership_lookup=membership_lookup,
        spy_df=spy_df,
        config=config,
        schedule=schedule,
        anchor_offsets=anchor_offsets,
        llm_pool_size=llm_pool_size,
        allowed_signal_dates=(signal_date_manifest.date_set if signal_date_manifest is not None else None),
    )
    if signal_date_manifest is not None:
        validate_pool_date_coverage(pools_by_key, signal_date_manifest.date_set)
        validate_autoresearch_pool_dates(pools_by_key, str(autoresearch_stage))
    raw_pools_by_key = pools_by_key
    if prompt_variant == PROMPT_VARIANT_V25:
        memory_build = v25m.build_balanced_outcome_memory(raw_pools_by_key, market_frames.close_df)
        pools_by_key = memory_build.pools_by_key
        artifact_root = output_dir or cache_path.parent / f"{cache_path.stem}_artifacts"
        memory_dir = v25m.save_memory_artifacts(memory_build, artifact_root)
        ready_dates = int(memory_build.coverage_by_date["memory_ready"].sum())
        full_dates = int(memory_build.coverage_by_date["fully_populated"].sum())
        print(
            f"\nV2.5 memory audit: {ready_dates}/{len(memory_build.coverage_by_date)} ready dates; "
            f"{full_dates} fully populated"
        )
        print(f"Memory artifacts: {memory_dir}")
        if should_collect_fresh_v2_control(prompt_variant, v25_paired_control):
            control_cache = v25_paired_control_cache_path(cache_path)
            print(f"\nCollecting fresh paired V2 control first: {control_cache}")
            await collect_llm_scores_for_v2_pools(
                raw_pools_by_key,
                cache_path=control_cache,
                model=model,
                temperature=temperature,
                thinking_mode=thinking_mode,
                rate_limit=rate_limit,
                anonymize_tickers=anonymize_tickers,
                prompt_variant=PROMPT_VARIANT_V2,
            )
    shared_call_budget: LLMCallBudget | None = None
    shared_token_budget: OutputTokenBudget | None = None
    if experimental_prompt_spec is not None:
        budget_cache_paths = [cache_path]
        if paired_control:
            budget_cache_paths.append(
                paired_control_cache or autoresearch_paired_control_cache_path(cache_path)
            )
        shared_call_budget, shared_token_budget = _shared_autoresearch_budgets(
            budget_cache_paths,
            call_limit=int(run_call_budget),
            output_token_limit=int(run_output_token_budget),
        )

    if paired_control:
        control_cache = paired_control_cache or autoresearch_paired_control_cache_path(cache_path)
        append_progress_event(
            progress_file,
            "paired_control_started",
            cache_path=str(control_cache.resolve()),
            planned_pools=len(raw_pools_by_key),
        )
        await collect_llm_scores_for_v2_pools(
            raw_pools_by_key,
            cache_path=control_cache,
            model=model,
            temperature=temperature,
            thinking_mode=thinking_mode,
            rate_limit=rate_limit,
            anonymize_tickers=anonymize_tickers,
            prompt_variant=PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL,
            thinking_call_policy=thinking_call_policy,
            llm_max_output_tokens=llm_max_output_tokens,
            run_output_token_budget=run_output_token_budget,
            llm_max_attempts=llm_max_attempts,
            run_call_budget=run_call_budget,
            autoresearch_mode=True,
            signal_date_manifest=signal_date_manifest,
            progress_file=progress_file,
            shared_call_budget=shared_call_budget,
            shared_token_budget=shared_token_budget,
        )
        append_progress_event(
            progress_file,
            "paired_control_completed",
            cache_path=str(control_cache.resolve()),
            completed_pools=len(raw_pools_by_key),
        )

    print(f"\nResuming V2 LLM cache only: {len(pools_by_key)} pools")
    print(f"Cache path: {cache_path}")
    print(f"Anonymize tickers: {bool(anonymize_tickers)}")
    print(f"Rationale top-k per side: {int(rationale_top_k_per_side)}")
    print(f"Structured reasoning: {bool(structured_reasoning)}")
    print(f"Print reasoning_content: {bool(print_reasoning_content)}")
    prompt_label = experimental_prompt_spec.candidate_id if experimental_prompt_spec is not None else prompt_variant
    prompt_hash = (
        experimental_prompt_spec.prompt_hash
        if experimental_prompt_spec is not None
        else prompt_hash_for_variant(prompt_variant)
    )
    print(f"Prompt variant: {prompt_label} ({prompt_hash})")
    print(f"Model: {model} | Thinking mode: {thinking_mode}")
    if thinking_mode == "enabled":
        print(
            f"Thinking calls: {thinking_call_policy} | max output/side: {llm_max_output_tokens:,} | "
            f"run budget: {run_output_token_budget:,} | attempts: {llm_max_attempts}"
        )
    return await collect_llm_scores_for_v2_pools(
        pools_by_key,
        cache_path=cache_path,
        model=model,
        temperature=temperature,
        thinking_mode=thinking_mode,
        rate_limit=rate_limit,
        anonymize_tickers=anonymize_tickers,
        rationale_top_k_per_side=rationale_top_k_per_side,
        structured_reasoning=structured_reasoning,
        prompt_variant=prompt_variant,
        print_reasoning_content=print_reasoning_content,
        thinking_call_policy=thinking_call_policy,
        llm_max_output_tokens=llm_max_output_tokens,
        run_output_token_budget=run_output_token_budget,
        llm_max_attempts=llm_max_attempts,
        run_call_budget=run_call_budget,
        experimental_prompt_spec=experimental_prompt_spec,
        autoresearch_mode=experimental_prompt_spec is not None,
        signal_date_manifest=signal_date_manifest,
        progress_file=progress_file,
        shared_call_budget=shared_call_budget,
        shared_token_budget=shared_token_budget,
    )


async def run_v2_full(
    config: rb.BenchmarkConfig,
    output_dir: Path,
    pool_sizes: tuple[int, ...],
    book_sizes: tuple[int, ...],
    anchor_offsets: tuple[int, ...],
    schedule: str,
    llm_pool_size: int,
    legacy_root: Path,
    legacy_factor_root: Path,
    v1_roots: tuple[Path, ...],
    v2_reference_roots: tuple[Path, ...],
    model: str,
    temperature: float,
    thinking_mode: str,
    cache_path: Path,
    rate_limit: float,
    llm: bool,
    anonymize_tickers: bool = False,
    rationale_top_k_per_side: int = 0,
    structured_reasoning: bool = False,
    prompt_variant: str = PROMPT_VARIANT_V2,
    print_reasoning_content: bool = False,
    v22_lambdas: tuple[float, ...] = DEFAULT_V22_LAMBDAS,
    thinking_call_policy: str = DEFAULT_THINKING_CALL_POLICY,
    llm_max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    run_output_token_budget: int = DEFAULT_RUN_OUTPUT_TOKEN_BUDGET,
    llm_max_attempts: int = DEFAULT_LLM_MAX_ATTEMPTS,
    v25_paired_control: bool = False,
    incumbent_guarded: bool = True,
    experimental_prompt_spec: ExperimentalPromptSpec | None = None,
    signal_date_manifest: SignalDateManifest | None = None,
    progress_file: Path | None = None,
    run_call_budget: int = AUTORESEARCH_DEFAULT_CALL_BUDGET,
    paired_control: bool = False,
    paired_control_cache: Path | None = None,
    autoresearch_stage: str | None = None,
    validation_freeze: dict[str, Any] | None = None,
    autoresearch_harness_hash: str | None = None,
) -> pd.DataFrame:
    _, membership_lookup, market_frames, feature_frames, spy_df = load_experiment_inputs(config)
    variants = build_variant_grid(pool_sizes, book_sizes, anchor_offsets, schedule=schedule)
    if not variants:
        raise ValueError("No valid V2 variants after applying pool/book constraints")
    if llm and llm_pool_size not in pool_sizes:
        raise ValueError("--llm-pool-size must be included in --pool-sizes when --phase full")
    if llm and max(pool_sizes) > llm_pool_size:
        raise ValueError("--llm-pool-size must be at least as large as every replayed pool size")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    llm_pools_by_key: dict[str, v1.CandidatePool] = {}
    bundles_by_run: dict[tuple[str, str], v1.ExperimentBundle] = {}
    checks_by_run: dict[tuple[str, str], pd.DataFrame] = {}
    memory_build: v25m.V25MemoryBuild | None = None

    for variant in variants:
        print(f"\nRunning V2 baseline {variant.label}...")
        weekly, checks, selections, pools = run_v2_pool_strategy(
            market_frames=market_frames,
            feature_frames=feature_frames,
            membership_lookup=membership_lookup,
            spy_df=spy_df,
            config=config,
            pool_per_side=variant.pool_per_side,
            positions_per_side=variant.positions_per_side,
            anchor_offset=variant.anchor_offset,
            score_provider="trail5",
            allowed_signal_dates=(signal_date_manifest.date_set if signal_date_manifest is not None else None),
        )
        if signal_date_manifest is not None:
            validate_pool_date_coverage(pools, signal_date_manifest.date_set)
            validate_autoresearch_pool_dates(pools, str(autoresearch_stage))
        bundle = build_v2_bundle(weekly, checks, selections, config)
        run_dir = variant_dir(output_dir, variant) / "baseline_pool"
        save_v2_bundle(bundle, run_dir)
        all_rows.append(scorecard_for_bundle("Baseline-Pool", bundle, variant, scorer="trail5"))
        bundles_by_run[(variant.label, "Trail5")] = bundle
        checks_by_run[(variant.label, "Trail5")] = checks
        if variant.pool_per_side == llm_pool_size:
            for pool in pools:
                llm_pools_by_key.setdefault(llm_score_storage_key(pool, llm_pool_size), pool)

    raw_llm_pools_by_key = dict(llm_pools_by_key)
    if prompt_variant == PROMPT_VARIANT_V25:
        memory_build = v25m.build_balanced_outcome_memory(raw_llm_pools_by_key, market_frames.close_df)
        llm_pools_by_key = memory_build.pools_by_key
        memory_dir = v25m.save_memory_artifacts(memory_build, output_dir)
        ready_dates = int(memory_build.coverage_by_date["memory_ready"].sum())
        full_dates = int(memory_build.coverage_by_date["fully_populated"].sum())
        print(
            f"\nV2.5 memory audit: {ready_dates}/{len(memory_build.coverage_by_date)} ready dates; "
            f"{full_dates} fully populated"
        )
        print(f"Memory artifacts: {memory_dir}")
        for variant in variants:
            print(f"\nRunning V2.5 HistRet-only diagnostic {variant.label}...")
            weekly, checks, selections, _ = run_v2_pool_strategy(
                market_frames=market_frames,
                feature_frames=feature_frames,
                membership_lookup=membership_lookup,
                spy_df=spy_df,
                config=config,
                pool_per_side=variant.pool_per_side,
                positions_per_side=variant.positions_per_side,
                anchor_offset=variant.anchor_offset,
                score_provider="histret",
                llm_source_pool_per_side=llm_pool_size,
                pool_overrides_by_key=llm_pools_by_key,
                allowed_signal_dates=(signal_date_manifest.date_set if signal_date_manifest is not None else None),
            )
            bundle = build_v2_bundle(weekly, checks, selections, config)
            run_dir = variant_dir(output_dir, variant) / "histret_memory_only"
            save_v2_bundle(bundle, run_dir)
            all_rows.append(
                scorecard_for_bundle(
                    "HistRet-Only v2.5 Diagnostic",
                    bundle,
                    variant,
                    scorer="histret_memory_only",
                    score_pool_per_side=llm_pool_size,
                )
            )
            bundles_by_run[(variant.label, "HistRet-Only")] = bundle
            checks_by_run[(variant.label, "HistRet-Only")] = checks

    shared_call_budget: LLMCallBudget | None = None
    shared_token_budget: OutputTokenBudget | None = None
    if llm and experimental_prompt_spec is not None:
        budget_cache_paths = [cache_path]
        if paired_control:
            budget_cache_paths.append(
                paired_control_cache or autoresearch_paired_control_cache_path(cache_path)
            )
        shared_call_budget, shared_token_budget = _shared_autoresearch_budgets(
            budget_cache_paths,
            call_limit=int(run_call_budget),
            output_token_limit=int(run_output_token_budget),
        )

    llm_scores: dict[str, dict[str, Any]] = {}
    if llm:
        if paired_control:
            control_cache = paired_control_cache or autoresearch_paired_control_cache_path(cache_path)
            print(f"\nCollecting fresh paired minimal-change control: {len(raw_llm_pools_by_key)} pools")
            print(f"Control cache: {control_cache}")
            append_progress_event(
                progress_file,
                "paired_control_started",
                cache_path=str(control_cache.resolve()),
                planned_pools=len(raw_llm_pools_by_key),
            )
            control_scores = await collect_llm_scores_for_v2_pools(
                raw_llm_pools_by_key,
                cache_path=control_cache,
                model=model,
                temperature=temperature,
                thinking_mode=thinking_mode,
                rate_limit=rate_limit,
                anonymize_tickers=anonymize_tickers,
                prompt_variant=PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL,
                thinking_call_policy=thinking_call_policy,
                llm_max_output_tokens=llm_max_output_tokens,
                run_output_token_budget=run_output_token_budget,
                llm_max_attempts=llm_max_attempts,
                run_call_budget=run_call_budget,
                autoresearch_mode=True,
                signal_date_manifest=signal_date_manifest,
                progress_file=progress_file,
                shared_call_budget=shared_call_budget,
                shared_token_budget=shared_token_budget,
            )
            for variant in variants:
                print(f"\nRunning fresh paired minimal-change control replay {variant.label}...")
                weekly, checks, selections, _ = run_v2_pool_strategy(
                    market_frames=market_frames,
                    feature_frames=feature_frames,
                    membership_lookup=membership_lookup,
                    spy_df=spy_df,
                    config=config,
                    pool_per_side=variant.pool_per_side,
                    positions_per_side=variant.positions_per_side,
                    anchor_offset=variant.anchor_offset,
                    score_provider="llm",
                    llm_scores_by_key=control_scores,
                    llm_source_pool_per_side=llm_pool_size,
                    allowed_signal_dates=(
                        signal_date_manifest.date_set if signal_date_manifest is not None else None
                    ),
                )
                bundle = build_v2_bundle(weekly, checks, selections, config)
                run_dir = variant_dir(output_dir, variant) / "llm_pool_v2_minimal_change_fresh_control"
                save_v2_bundle(bundle, run_dir)
                save_v2_score_diagnostics(checks, run_dir)
                all_rows.append(
                    scorecard_for_bundle(
                        "LLM-Pool v2 Minimal-Change Fresh Control",
                        bundle,
                        variant,
                        scorer="llm_score",
                        score_pool_per_side=llm_pool_size,
                    )
                )
            append_progress_event(
                progress_file,
                "paired_control_completed",
                cache_path=str(control_cache.resolve()),
                completed_pools=len(control_scores),
            )

        if should_collect_fresh_v2_control(prompt_variant, v25_paired_control):
            control_cache = v25_paired_control_cache_path(cache_path)
            print(f"\nCollecting fresh paired V2 control: {len(raw_llm_pools_by_key)} pools")
            print(f"Control cache: {control_cache}")
            control_scores = await collect_llm_scores_for_v2_pools(
                raw_llm_pools_by_key,
                cache_path=control_cache,
                model=model,
                temperature=temperature,
                thinking_mode=thinking_mode,
                rate_limit=rate_limit,
                anonymize_tickers=anonymize_tickers,
                prompt_variant=PROMPT_VARIANT_V2,
            )
            for variant in variants:
                print(f"\nRunning fresh paired V2 control replay {variant.label}...")
                weekly, checks, selections, _ = run_v2_pool_strategy(
                    market_frames=market_frames,
                    feature_frames=feature_frames,
                    membership_lookup=membership_lookup,
                    spy_df=spy_df,
                    config=config,
                    pool_per_side=variant.pool_per_side,
                    positions_per_side=variant.positions_per_side,
                    anchor_offset=variant.anchor_offset,
                    score_provider="llm",
                    llm_scores_by_key=control_scores,
                    llm_source_pool_per_side=llm_pool_size,
                    allowed_signal_dates=(
                        signal_date_manifest.date_set if signal_date_manifest is not None else None
                    ),
                )
                bundle = build_v2_bundle(weekly, checks, selections, config)
                run_dir = variant_dir(output_dir, variant) / "llm_pool_v2_fresh_control"
                save_v2_bundle(bundle, run_dir)
                save_v2_score_diagnostics(checks, run_dir)
                all_rows.append(
                    scorecard_for_bundle(
                        "LLM-Pool v2 Fresh Control",
                        bundle,
                        variant,
                        scorer="llm_score",
                        score_pool_per_side=llm_pool_size,
                    )
                )
                bundles_by_run[(variant.label, "V2 Fresh Control")] = bundle
                checks_by_run[(variant.label, "V2 Fresh Control")] = checks

        prompt_note = (
            "structured reasoning treatment"
            if structured_reasoning
            else (
                "V2.2 contextual adjustment prompt"
                if prompt_variant == PROMPT_VARIANT_V22
                else (
                    "V2.3 rank-only prompt"
                    if prompt_variant == PROMPT_VARIANT_V23
                    else (
                        "V2.4 frozen score prompt with raw SPY/VIX context"
                        if prompt_variant == PROMPT_VARIANT_V24
                        else (
                        "V2.5 balanced outcome-memory prompt"
                        if prompt_variant == PROMPT_VARIANT_V25
                        else (
                            "V2.6 production-grade prompt treatment"
                            if prompt_variant == PROMPT_VARIANT_V26
                            else (
                                "V2 minimal one-line range-instruction control"
                                if prompt_variant == PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL
                                else (
                                    "V2 market-regime input ablation"
                                    if prompt_variant == PROMPT_VARIANT_V2_NO_REGIME
                                    else "V2 score prompt"
                                )
                            )
                        )
                        )
                    )
                )
            )
        )
        print(f"\nCollecting V2 LLM scores with {prompt_note}: {len(llm_pools_by_key)} pools")
        print(f"Anonymize tickers: {bool(anonymize_tickers)}")
        print(f"Rationale top-k per side: {int(rationale_top_k_per_side)}")
        print(f"Structured reasoning: {bool(structured_reasoning)}")
        print(f"Print reasoning_content: {bool(print_reasoning_content)}")
        treatment_label = (
            experimental_prompt_spec.candidate_id if experimental_prompt_spec is not None else prompt_variant
        )
        treatment_hash = (
            experimental_prompt_spec.prompt_hash
            if experimental_prompt_spec is not None
            else prompt_hash_for_variant(prompt_variant)
        )
        print(f"Prompt variant: {treatment_label} ({treatment_hash})")
        print(f"Model: {model} | Thinking mode: {thinking_mode}")
        if thinking_mode == "enabled":
            print(
                f"Thinking calls: {thinking_call_policy} | max output/side: {llm_max_output_tokens:,} | "
                f"run budget: {run_output_token_budget:,} | attempts: {llm_max_attempts}"
            )
        llm_scores = await collect_llm_scores_for_v2_pools(
            llm_pools_by_key,
            cache_path=cache_path,
            model=model,
            temperature=temperature,
            thinking_mode=thinking_mode,
            rate_limit=rate_limit,
            anonymize_tickers=anonymize_tickers,
            rationale_top_k_per_side=rationale_top_k_per_side,
            structured_reasoning=structured_reasoning,
            prompt_variant=prompt_variant,
            print_reasoning_content=print_reasoning_content,
            thinking_call_policy=thinking_call_policy,
            llm_max_output_tokens=llm_max_output_tokens,
            run_output_token_budget=run_output_token_budget,
            llm_max_attempts=llm_max_attempts,
            run_call_budget=run_call_budget,
            experimental_prompt_spec=experimental_prompt_spec,
            autoresearch_mode=experimental_prompt_spec is not None,
            signal_date_manifest=signal_date_manifest,
            progress_file=progress_file,
            shared_call_budget=shared_call_budget,
            shared_token_budget=shared_token_budget,
        )

        replay_lambdas = tuple(v22_lambdas) if prompt_variant == PROMPT_VARIANT_V22 else (np.nan,)
        for adjustment_lambda in replay_lambdas:
            lambda_label = v22_lambda_label(float(adjustment_lambda)) if prompt_variant == PROMPT_VARIANT_V22 else ""
            for variant in variants:
                replay_label = f"{variant.label}_{lambda_label}" if lambda_label else variant.label
                print(f"\nRunning V2 LLM replay {replay_label}...")
                weekly, checks, selections, _ = run_v2_pool_strategy(
                    market_frames=market_frames,
                    feature_frames=feature_frames,
                    membership_lookup=membership_lookup,
                    spy_df=spy_df,
                    config=config,
                    pool_per_side=variant.pool_per_side,
                    positions_per_side=variant.positions_per_side,
                    anchor_offset=variant.anchor_offset,
                    score_provider=(
                        "llm_adjustment"
                        if prompt_variant == PROMPT_VARIANT_V22
                        else (
                            "llm_rank"
                            if prompt_variant == PROMPT_VARIANT_V23
                            else "llm"
                        )
                    ),
                    llm_scores_by_key=llm_scores,
                    llm_source_pool_per_side=llm_pool_size,
                    v22_lambda=float(adjustment_lambda) if prompt_variant == PROMPT_VARIANT_V22 else 4.0,
                    pool_overrides_by_key=(
                        llm_pools_by_key if prompt_variant == PROMPT_VARIANT_V25 else None
                    ),
                    allowed_signal_dates=(
                        signal_date_manifest.date_set if signal_date_manifest is not None else None
                    ),
                )
                bundle = build_v2_bundle(weekly, checks, selections, config)
                if experimental_prompt_spec is not None:
                    run_subdir = "llm_pool_experimental"
                elif prompt_variant == PROMPT_VARIANT_V22:
                    run_subdir = f"llm_pool_v22_{lambda_label}"
                elif prompt_variant == PROMPT_VARIANT_V23:
                    run_subdir = "llm_pool_v23"
                elif prompt_variant == PROMPT_VARIANT_V24:
                    run_subdir = "llm_pool_v24"
                elif prompt_variant == PROMPT_VARIANT_V25:
                    run_subdir = "llm_pool_v25_balanced_memory"
                elif prompt_variant == PROMPT_VARIANT_V26:
                    run_subdir = "llm_pool_v26"
                elif prompt_variant == PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL:
                    run_subdir = "llm_pool_v2_minimal_change_control"
                elif prompt_variant == PROMPT_VARIANT_V2_NO_REGIME:
                    run_subdir = "llm_pool_v2_no_regime"
                else:
                    run_subdir = "llm_pool_v2"
                run_dir = variant_dir(output_dir, variant) / run_subdir
                save_v2_bundle(bundle, run_dir)
                save_v2_score_diagnostics(checks, run_dir)
                if rationale_top_k_per_side > 0 or structured_reasoning:
                    save_attribution_outcome_summary(selections, run_dir)
                run_name = (
                    f"LLM-Pool experimental:{experimental_prompt_spec.candidate_id}"
                    if experimental_prompt_spec is not None
                    else
                    f"LLM-Pool v2.2 lambda={format_v22_lambda(float(adjustment_lambda))}"
                    if prompt_variant == PROMPT_VARIANT_V22
                    else (
                        f"LLM-Pool {prompt_variant}"
                        if prompt_variant in {
                            PROMPT_VARIANT_V23,
                            PROMPT_VARIANT_V24,
                            PROMPT_VARIANT_V25,
                            PROMPT_VARIANT_V26,
                            PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL,
                            PROMPT_VARIANT_V2_NO_REGIME,
                        }
                        else "LLM-Pool v2"
                    )
                )
                scorer = (
                    f"base_score_plus_lambda_adjustment_{format_v22_lambda(float(adjustment_lambda))}"
                    if prompt_variant == PROMPT_VARIANT_V22
                    else (
                        "llm_rank_order"
                        if prompt_variant == PROMPT_VARIANT_V23
                        else "llm_score"
                    )
                )
                all_rows.append(
                    scorecard_for_bundle(
                        run_name,
                        bundle,
                        variant,
                        scorer=scorer,
                        score_pool_per_side=llm_pool_size,
                    )
                )
                if prompt_variant == PROMPT_VARIANT_V25:
                    bundles_by_run[(variant.label, "V2.5 Balanced Memory")] = bundle
                    checks_by_run[(variant.label, "V2.5 Balanced Memory")] = checks

    scorecard = pd.concat(all_rows, ignore_index=True)
    average_rows = build_anchor_average_rows(scorecard) if schedule == "anchors" else pd.DataFrame()
    if not average_rows.empty:
        scorecard = pd.concat([scorecard, average_rows], ignore_index=True)
    scorecard.to_csv(output_dir / "comparison_scorecard.csv", index=False)
    gross_comparison = build_gross_system_comparison(
        scorecard=scorecard,
        legacy_root=legacy_root,
        legacy_factor_root=legacy_factor_root,
        v1_roots=v1_roots,
        v2_reference_roots=v2_reference_roots,
    )
    gross_comparison.to_csv(output_dir / "gross_legacy_v1_v2_comparison.csv", index=False)
    apples_to_apples = gross_comparison[
        ~gross_comparison["execution_model"].astype(str).str.lower().eq("daily")
    ].copy()
    apples_to_apples.to_csv(output_dir / "apples_to_apples_gross_c2c_comparison.csv", index=False)

    v25_report_files: dict[str, str] = {}
    if prompt_variant == PROMPT_VARIANT_V25 and memory_build is not None:
        v25_report_files = save_v25_comparison_reports(
            output_dir=output_dir,
            variant=variants[0],
            bundles_by_run=bundles_by_run,
            checks_by_run=checks_by_run,
            memory_build=memory_build,
        )

    if experimental_prompt_spec is not None:
        manifest_prompt_variant = EXPERIMENTAL_PROMPT_VARIANT
        manifest_prompt_hash = experimental_prompt_spec.prompt_hash
        manifest_prompt_source = f"experimental_prompt_spec:{experimental_prompt_spec.candidate_id}"
        manifest_prompt_version = f"llm_reversal_signal_v2_experimental_{experimental_prompt_spec.prompt_hash}"
        manifest_output_contract = experimental_prompt_spec.output_contract
        manifest_score_min = experimental_prompt_spec.score_min
        manifest_score_max = experimental_prompt_spec.score_max
        manifest_score_direction = experimental_prompt_spec.score_direction
    else:
        manifest_prompt_variant = prompt_variant
        manifest_prompt_hash = prompt_hash_for_variant(prompt_variant)
        manifest_prompt_source = prompt_source_for_variant(
            prompt_variant,
            structured_reasoning=structured_reasoning,
        )
        manifest_prompt_version = prompt_version_for_variant(prompt_variant)
        manifest_output_contract = output_contract_for_variant(prompt_variant)
        manifest_score_min, manifest_score_max, manifest_score_direction = 0.0, 100.0, "higher_is_better"
    treatment_output_dir = (
        f"pool{int(pool_sizes[0])}_pos{int(book_sizes[0])}/anchor_{int(anchor_offsets[0])}/"
        "llm_pool_experimental"
        if experimental_prompt_spec is not None
        else None
    )
    paired_control_output_dir = (
        f"pool{int(pool_sizes[0])}_pos{int(book_sizes[0])}/anchor_{int(anchor_offsets[0])}/"
        "llm_pool_v2_minimal_change_fresh_control"
        if paired_control
        else None
    )

    manifest = {
        "artifact_type": "llm_reversal_signal_v2",
        "scorer_source": manifest_prompt_source,
        "incumbent_guarded": bool(incumbent_guarded),
        "incumbent_spec": incumbent_configuration_manifest(),
        "start": str(config.start.date()),
        "end": str(config.end.date()),
        "pool_sizes": list(pool_sizes),
        "book_sizes": list(book_sizes),
        "schedule": schedule,
        "anchor_offsets": list(anchor_offsets),
        "llm_pool_size": int(llm_pool_size),
        "llm_call_policy": (
            "split_by_side_thinking"
            if thinking_mode == "enabled" and thinking_call_policy == "split-side"
            else (
                "one adjustment call per date at llm_pool_size; smaller pools and lambda grid replay offline"
                if prompt_variant == PROMPT_VARIANT_V22
                else (
                    "one rank-only call per date at llm_pool_size; smaller pools slice those rankings"
                    if prompt_variant == PROMPT_VARIANT_V23
                    else (
                        "fresh V2 control plus one V2.5 treatment call per date"
                        if prompt_variant == PROMPT_VARIANT_V25 and v25_paired_control
                        else "one score call per date at llm_pool_size; smaller pools slice those scores"
                    )
                )
            )
        ),
        "cost_bps_levels": list(config.cost_bps_levels),
        "headline_cost_bps": 0.0,
        "legacy_comparison_root": str(legacy_root),
        "legacy_factor_comparison_root": str(legacy_factor_root),
        "v1_comparison_roots": [str(path) for path in v1_roots],
        "v2_reference_roots": [str(path) for path in v2_reference_roots],
        "gross_system_comparison": "gross_legacy_v1_v2_comparison.csv",
        "apples_to_apples_comparison": "apples_to_apples_gross_c2c_comparison.csv",
        "prompt_variant": manifest_prompt_variant,
        "prompt_hash": manifest_prompt_hash,
        "prompt_source": manifest_prompt_source,
        "prompt_style": V2_PROMPT_STYLE,
        "prompt_version": manifest_prompt_version,
        "output_contract": manifest_output_contract,
        "market_regime_input": (
            "omitted" if prompt_variant == PROMPT_VARIANT_V2_NO_REGIME else "included"
        ),
        "score_min": float(manifest_score_min),
        "score_max": float(manifest_score_max),
        "score_direction": manifest_score_direction,
        "model": model,
        "thinking_mode": thinking_mode,
        "thinking_call_policy": thinking_call_policy,
        "llm_max_output_tokens": int(llm_max_output_tokens),
        "run_output_token_budget": int(run_output_token_budget),
        "llm_max_attempts": int(llm_max_attempts),
        "run_call_budget": int(run_call_budget),
        "temperature": float(temperature),
        "v22_lambdas": list(v22_lambdas) if prompt_variant == PROMPT_VARIANT_V22 else [],
        "v25_paired_control": bool(v25_paired_control) if prompt_variant == PROMPT_VARIANT_V25 else False,
        "v25_control_cache": str(v25_paired_control_cache_path(cache_path))
        if prompt_variant == PROMPT_VARIANT_V25 and v25_paired_control
        else None,
        "v25_memory_spec_hash": v25m.memory_spec_hash() if prompt_variant == PROMPT_VARIANT_V25 else None,
        "paired_v2_control": should_collect_fresh_v2_control(prompt_variant, v25_paired_control),
        "paired_v2_control_cache": str(v25_paired_control_cache_path(cache_path))
        if should_collect_fresh_v2_control(prompt_variant, v25_paired_control)
        else None,
        "v25_memory_spec": v25m.memory_spec() if prompt_variant == PROMPT_VARIANT_V25 else None,
        "v25_reports": v25_report_files,
        "anonymize_tickers": bool(anonymize_tickers),
        "rationale_top_k_per_side": int(rationale_top_k_per_side),
        "structured_reasoning": bool(structured_reasoning),
        "print_reasoning_content": bool(print_reasoning_content),
        "llm_enabled": bool(llm),
        "autoresearch": {
            "enabled": experimental_prompt_spec is not None,
            "stage": autoresearch_stage,
            "candidate_id": (
                experimental_prompt_spec.candidate_id if experimental_prompt_spec is not None else None
            ),
            "generation": (
                experimental_prompt_spec.generation if experimental_prompt_spec is not None else None
            ),
            "parent_id": (
                experimental_prompt_spec.parent_id if experimental_prompt_spec is not None else None
            ),
            "prompt_sha256": (
                experimental_prompt_spec.prompt_sha256 if experimental_prompt_spec is not None else None
            ),
            "prompt_spec_sha256": (
                experimental_prompt_spec.spec_sha256 if experimental_prompt_spec is not None else None
            ),
            "prompt_spec_path": (
                str(experimental_prompt_spec.source_path) if experimental_prompt_spec is not None else None
            ),
            "date_manifest_hash": (
                signal_date_manifest.date_manifest_hash if signal_date_manifest is not None else None
            ),
            "date_manifest_source_sha256": (
                signal_date_manifest.source_sha256 if signal_date_manifest is not None else None
            ),
            "signal_dates": (
                [str(date.date()) for date in signal_date_manifest.dates]
                if signal_date_manifest is not None
                else None
            ),
            "code_fingerprint": autoresearch_code_fingerprint() if experimental_prompt_spec is not None else None,
            "harness_fingerprint": autoresearch_harness_hash,
            "validation_freeze": validation_freeze,
        },
        "output_resolver": {
            "baseline_dir": (
                f"pool{int(pool_sizes[0])}_pos{int(book_sizes[0])}/anchor_{int(anchor_offsets[0])}/baseline_pool"
            ),
            "treatment_dir": treatment_output_dir,
            "paired_control_dir": paired_control_output_dir,
            "weekly_returns_file": "weekly_returns.csv",
            "signal_checks_file": "signal_checks.csv",
            "portfolio_selections_file": "portfolio_selections.csv",
        },
        "paired_control": bool(paired_control),
        "paired_control_prompt_variant": (
            PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL if paired_control else None
        ),
        "paired_control_prompt_hash": (
            prompt_hash_for_variant(PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL) if paired_control else None
        ),
        "paired_control_cache": (
            str((paired_control_cache or autoresearch_paired_control_cache_path(cache_path)).resolve())
            if paired_control
            else None
        ),
    }
    (output_dir / "run_config.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_v2_html_report(output_dir, scorecard, gross_comparison)
    return scorecard


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM reversal signal v2 experiment.")
    parser.add_argument("--phase", choices=("baseline", "full", "llm-cache"), default="baseline")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--membership-path", type=Path, default=rb.DEFAULT_MEMBERSHIP_PATH)
    parser.add_argument("--cost-bps-levels", type=rb.parse_cost_bps_levels, default=DEFAULT_COST_BPS_LEVELS)
    parser.add_argument("--pool-sizes", type=parse_int_list, default=DEFAULT_POOL_SIZES)
    parser.add_argument("--book-sizes", type=parse_int_list, default=DEFAULT_BOOK_SIZES)
    parser.add_argument("--llm-pool-size", type=int, default=DEFAULT_LLM_POOL_SIZE)
    parser.add_argument("--schedule", choices=("friday", "anchors"), default=DEFAULT_SCHEDULE)
    parser.add_argument("--anchor-offsets", type=parse_int_list, default=DEFAULT_ANCHOR_OFFSETS)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--legacy-factor-root", type=Path, default=DEFAULT_LEGACY_FACTOR_ROOT)
    parser.add_argument("--v1-roots", type=parse_path_list, default=DEFAULT_V1_ROOTS)
    parser.add_argument("--v2-reference-roots", type=parse_path_list, default=DEFAULT_V2_REFERENCE_ROOTS)
    parser.add_argument("--model", default=technical_llm.DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=INCUMBENT_TEMPERATURE)
    parser.add_argument(
        "--thinking-mode",
        choices=("disabled", "enabled"),
        default=technical_llm.DEFAULT_THINKING_MODE,
        help=(
            "DeepSeek V4 mode. Use disabled to match old deepseek-chat/non-thinking behavior; "
            "enabled to match old deepseek-reasoner/thinking behavior. Temperature is ignored when enabled."
        ),
    )
    parser.add_argument(
        "--thinking-call-policy",
        choices=THINKING_CALL_POLICIES,
        default=DEFAULT_THINKING_CALL_POLICY,
        help="Thinking-mode score calls are split into separately checkpointed long and short requests.",
    )
    parser.add_argument(
        "--llm-max-output-tokens",
        type=int,
        default=DEFAULT_LLM_MAX_OUTPUT_TOKENS,
        help="Maximum output tokens for each thinking-mode side request (default: 16000).",
    )
    parser.add_argument(
        "--run-output-token-budget",
        type=int,
        default=DEFAULT_RUN_OUTPUT_TOKEN_BUDGET,
        help="Hard cumulative output-token ceiling across thinking calls and retries (default: 4500000).",
    )
    parser.add_argument(
        "--llm-max-attempts",
        type=int,
        default=DEFAULT_LLM_MAX_ATTEMPTS,
        help="Maximum attempts per thinking-mode side; max-token exhaustion is never retried (default: 2).",
    )
    parser.add_argument(
        "--run-call-budget",
        type=int,
        default=AUTORESEARCH_DEFAULT_CALL_BUDGET,
        help="Hard request-attempt ceiling for bounded autoresearch scoring (default: 1100).",
    )
    parser.add_argument("--rate-limit", type=float, default=2.0)
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default=PROMPT_VARIANT_V2,
        help=(
            "LLM prompt variant: v2 returns current glossary-backed 0-100 scores; v2-frozen reproduces "
            "the original anonymized V2 score prompt exactly; v2.2 returns -3..+3 contextual adjustments; "
            "v2.3 returns complete rank-only orderings; v2.4 keeps the exact v2-frozen 0-100 system prompt "
            "but replaces the regime label in the user input with raw SPY 200-day-MA and VIX context; "
            "v2.5 keeps V2 scoring and adds balanced past-only outcome-memory summaries; "
            "v2.6 is the production-grade rewrite treatment; v2-minimal-change-control changes only "
            "the score-range instruction in the incumbent prompt; v2-no-regime removes only the "
            "Market Regime glossary block and the date-level MARKET REGIME input line."
        ),
    )
    parser.add_argument(
        "--experimental-prompt-spec",
        type=Path,
        help=(
            "Immutable autoresearch prompt-spec JSON with a frozen feature glossary and "
            "candidate-declared numeric score range/direction."
        ),
    )
    parser.add_argument(
        "--signal-dates-file",
        type=Path,
        help="JSON manifest containing the exact sorted signal-date allowlist for this run.",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        help="Append-only JSONL progress stream for a background supervisor/dashboard.",
    )
    parser.add_argument(
        "--validation-freeze",
        type=Path,
        help="Manually approved validation-freeze JSON required for autoresearch dates in 2015-2019.",
    )
    parser.add_argument(
        "--paired-control",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect/replay a fresh minimal-change control on the identical date manifest.",
    )
    parser.add_argument(
        "--paired-control-cache",
        type=Path,
        help="Optional explicit cache path for --paired-control; otherwise derived from treatment cache.",
    )
    parser.add_argument(
        "--v25-paired-control",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly collect/replay a fresh unchanged V2 control for V2.5 (default: off).",
    )
    parser.add_argument(
        "--v22-lambdas",
        type=parse_float_list,
        default=DEFAULT_V22_LAMBDAS,
        help=(
            "Comma-separated lambda grid for --prompt-variant v2.2. "
            "These are replayed offline from one adjustment cache."
        ),
    )
    parser.add_argument(
        "--anonymize-tickers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Send anonymous SEC_#### identifiers to the LLM and map scores back "
            "to real tickers after validation (incumbent default: on)."
        ),
    )
    parser.add_argument(
        "--experimental-override",
        action="store_true",
        help=(
            "Permit a deliberately non-incumbent research configuration. Without this flag, "
            "the frozen anonymized V2 configuration is enforced before any run begins."
        ),
    )
    parser.add_argument(
        "--rationale-top-k-per-side",
        type=int,
        default=0,
        help=(
            "If > 0, ask the LLM in the scoring call for primary feature attributions "
            "and <=6-word rationale labels only for the top K scored names per side."
        ),
    )
    parser.add_argument(
        "--structured-reasoning",
        action="store_true",
        help=(
            "Run the structured-reasoning treatment: every candidate returns score, "
            "1-3 primary feature/effect pairs, and a fixed setup_label. Keeps cache files separate."
        ),
    )
    parser.add_argument(
        "--print-reasoning-content",
        action="store_true",
        help=(
            "Debug only: print DeepSeek reasoning_content for fresh LLM calls and store it in the cache entry. "
            "Use on tiny one-date runs to avoid large token/cost/log output."
        ),
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> rb.BenchmarkConfig:
    return rb.BenchmarkConfig(
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        output_dir=args.output_dir,
        membership_path=args.membership_path,
        cost_bps_levels=tuple(args.cost_bps_levels),
        cadence_mode="trading_days",
        signal_lookback_days=5,
        rebalance_days=5,
        holding_days=5,
        positions_per_side=None,
    )


def cache_path_for_args(args: argparse.Namespace) -> Path:
    start = str(pd.Timestamp(args.start).date())
    end = str(pd.Timestamp(args.end).date())
    model = str(args.model).replace("/", "-")
    temperature_token = format(float(args.temperature), "g").replace("-", "m").replace(".", "p")
    effective_thinking = technical_llm.effective_thinking_mode(args.model, args.thinking_mode)
    thinking_token = str(effective_thinking).replace("/", "-")
    pools = "-".join(str(pool) for pool in args.pool_sizes)
    if args.schedule == "friday":
        schedule_label = "friday"
    else:
        anchors = "-".join(str(anchor) for anchor in args.anchor_offsets)
        schedule_label = f"anchors{anchors}"
    name = (
        f"llm_reversal_signal_v2_{start}_{end}_pools{pools}_{schedule_label}_"
        f"scorepool{int(args.llm_pool_size)}_mdl-{model}_temp-{temperature_token}_think-{thinking_token}.json"
    )
    if effective_thinking == "enabled":
        call_policy = str(args.thinking_call_policy).replace("/", "-")
        name = name.replace(
            ".json",
            f"_call-{call_policy}_mtok{int(args.llm_max_output_tokens)}_a{int(args.llm_max_attempts)}.json",
        )
    if args.experimental_prompt_spec is not None:
        spec = load_experimental_prompt_spec(Path(args.experimental_prompt_spec))
        variant_token = f"exp-{spec.candidate_id}"
        variant_hash = spec.prompt_hash
    elif args.prompt_variant == PROMPT_VARIANT_V22:
        variant_token = "v22"
        variant_hash = prompt_hash_for_variant(args.prompt_variant)
    elif args.prompt_variant == PROMPT_VARIANT_V23:
        variant_token = "v23"
        variant_hash = prompt_hash_for_variant(args.prompt_variant)
    elif args.prompt_variant == PROMPT_VARIANT_V24:
        variant_token = "v24"
        variant_hash = prompt_hash_for_variant(args.prompt_variant)
    elif args.prompt_variant == PROMPT_VARIANT_V25:
        variant_token = "v25"
        variant_hash = prompt_hash_for_variant(args.prompt_variant)
    elif args.prompt_variant == PROMPT_VARIANT_V26:
        variant_token = "v26"
        variant_hash = prompt_hash_for_variant(args.prompt_variant)
    elif args.prompt_variant == PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL:
        variant_token = "v2mc"
        variant_hash = prompt_hash_for_variant(args.prompt_variant)
    elif args.prompt_variant == PROMPT_VARIANT_V2_NO_REGIME:
        variant_token = "v2nr"
        variant_hash = prompt_hash_for_variant(args.prompt_variant)
    elif args.prompt_variant == PROMPT_VARIANT_V2_FROZEN:
        variant_token = "v2f"
        variant_hash = prompt_hash_for_variant(args.prompt_variant)
    else:
        variant_token = "v2"
        variant_hash = prompt_hash_for_variant(args.prompt_variant)
    name = name.replace(".json", f"_{variant_token}-{variant_hash}.json")
    if args.experimental_prompt_spec is not None and (
        spec.side_scope != "both" or spec.replicate_id != "r1"
    ):
        name = name.replace(
            ".json",
            f"_side-{spec.side_scope}_rep-{spec.replicate_id}.json",
        )
    if args.prompt_variant == PROMPT_VARIANT_V25:
        name = name.replace(".json", f"_mem-{v25m.memory_spec_hash()}.json")
    if args.anonymize_tickers:
        name = name.replace(".json", "_anon.json")
    if args.structured_reasoning:
        name = name.replace(".json", "_sr.json")
    if int(args.rationale_top_k_per_side) > 0:
        name = name.replace(".json", f"_ratk{int(args.rationale_top_k_per_side)}.json")
    return args.cache_dir / name


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.thinking_mode = technical_llm.effective_thinking_mode(args.model, args.thinking_mode)
    experimental_prompt_spec = (
        load_experimental_prompt_spec(Path(args.experimental_prompt_spec))
        if args.experimental_prompt_spec is not None
        else None
    )
    args.side_scope = experimental_prompt_spec.side_scope if experimental_prompt_spec else "both"
    args.replicate_id = experimental_prompt_spec.replicate_id if experimental_prompt_spec else "r1"
    if args.paired_control and experimental_prompt_spec is None:
        raise ValueError("--paired-control requires --experimental-prompt-spec")
    if args.paired_control_cache is not None and not args.paired_control:
        raise ValueError("--paired-control-cache requires --paired-control")
    if args.validation_freeze is not None and experimental_prompt_spec is None:
        raise ValueError("--validation-freeze requires --experimental-prompt-spec")
    if experimental_prompt_spec is not None:
        if args.signal_dates_file is None:
            raise ValueError("--experimental-prompt-spec requires --signal-dates-file")
        if args.prompt_variant != PROMPT_VARIANT_V2:
            raise ValueError("--experimental-prompt-spec cannot be combined with a non-default --prompt-variant")
        if bool(args.anonymize_tickers) != experimental_prompt_spec.anonymize_tickers:
            raise ValueError(
                "Ticker visibility must match experimental prompt spec anonymize_tickers"
            )
        if experimental_prompt_spec.side_scope != "both" and args.phase != "llm-cache":
            raise ValueError(
                "Side-scoped prompt specs collect immutable scores with --phase llm-cache; "
                "evaluate hybrids with mean_reversion_llm_v2_side_experiments.py"
            )
        validate_autoresearch_args(args)
    elif not args.experimental_override:
        validate_incumbent_args(args)
    if any(pool <= 0 for pool in args.pool_sizes):
        raise ValueError("--pool-sizes must contain positive integers")
    if any(book <= 0 for book in args.book_sizes):
        raise ValueError("--book-sizes must contain positive integers")
    if args.llm_pool_size <= 0:
        raise ValueError("--llm-pool-size must be a positive integer")
    if args.llm_max_output_tokens <= 0:
        raise ValueError("--llm-max-output-tokens must be positive")
    if args.run_output_token_budget <= 0:
        raise ValueError("--run-output-token-budget must be positive")
    if args.llm_max_attempts <= 0:
        raise ValueError("--llm-max-attempts must be positive")
    if args.run_call_budget <= 0:
        raise ValueError("--run-call-budget must be positive")
    if args.rationale_top_k_per_side < 0:
        raise ValueError("--rationale-top-k-per-side must be non-negative")
    if args.rationale_top_k_per_side and args.rationale_top_k_per_side > min(args.book_sizes):
        raise ValueError("--rationale-top-k-per-side should be <= the smallest selected book size")
    if args.structured_reasoning and args.rationale_top_k_per_side:
        raise ValueError("--structured-reasoning cannot be combined with --rationale-top-k-per-side")
    if args.prompt_variant == PROMPT_VARIANT_V22 and args.structured_reasoning:
        raise ValueError("--prompt-variant v2.2 cannot be combined with --structured-reasoning")
    if args.prompt_variant == PROMPT_VARIANT_V22 and args.rationale_top_k_per_side:
        raise ValueError("--prompt-variant v2.2 cannot be combined with --rationale-top-k-per-side")
    if args.prompt_variant in {PROMPT_VARIANT_V23, PROMPT_VARIANT_V24, PROMPT_VARIANT_V25} and args.structured_reasoning:
        raise ValueError(f"--prompt-variant {args.prompt_variant} cannot be combined with --structured-reasoning")
    if args.prompt_variant in {PROMPT_VARIANT_V23, PROMPT_VARIANT_V24, PROMPT_VARIANT_V25} and args.rationale_top_k_per_side:
        raise ValueError(f"--prompt-variant {args.prompt_variant} cannot be combined with --rationale-top-k-per-side")
    if args.prompt_variant in {
        PROMPT_VARIANT_V26,
        PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL,
        PROMPT_VARIANT_V2_NO_REGIME,
    }:
        if args.structured_reasoning or args.rationale_top_k_per_side:
            raise ValueError(
                f"--prompt-variant {args.prompt_variant} must preserve the strict score-only output contract"
            )
    if any(float(value) <= 0 for value in args.v22_lambdas):
        raise ValueError("--v22-lambdas must contain positive numbers")
    if args.schedule == "anchors" and any(anchor < 0 or anchor > 4 for anchor in args.anchor_offsets):
        raise ValueError("--anchor-offsets must be integers between 0 and 4")
    if args.prompt_variant == PROMPT_VARIANT_V25:
        if tuple(args.pool_sizes) != (60,) or tuple(args.book_sizes) != (30,) or args.llm_pool_size != 60:
            raise ValueError("V2.5 is frozen to --pool-sizes 60 --book-sizes 30 --llm-pool-size 60")
        if args.schedule != "anchors" or tuple(args.anchor_offsets) != (3,):
            raise ValueError("V2.5 is frozen to --schedule anchors --anchor-offsets 3")
        if tuple(float(value) for value in args.cost_bps_levels) != (0.0,):
            raise ValueError("V2.5 is frozen to gross --cost-bps-levels 0")
        if not args.anonymize_tickers:
            raise ValueError("V2.5 requires --anonymize-tickers")
        if args.phase in {"llm-cache", "full"}:
            if args.model != "deepseek-v4-flash":
                raise ValueError("V2.5 is frozen to --model deepseek-v4-flash")
            if args.thinking_mode != "disabled":
                raise ValueError("V2.5 is frozen to --thinking-mode disabled")
            if not np.isclose(float(args.temperature), 0.6):
                raise ValueError("V2.5 is frozen to --temperature 0.6")
    if args.prompt_variant == PROMPT_VARIANT_V2_NO_REGIME:
        if pd.Timestamp(args.start) != pd.Timestamp("2009-01-01") or pd.Timestamp(args.end) != pd.Timestamp("2014-01-01"):
            raise ValueError("V2 no-regime is frozen to --start 2009-01-01 --end 2014-01-01")
        if tuple(args.pool_sizes) != (60,) or tuple(args.book_sizes) != (30,) or args.llm_pool_size != 60:
            raise ValueError("V2 no-regime is frozen to --pool-sizes 60 --book-sizes 30 --llm-pool-size 60")
        if args.schedule != "anchors" or tuple(args.anchor_offsets) != (3,):
            raise ValueError("V2 no-regime is frozen to --schedule anchors --anchor-offsets 3")
        if tuple(float(value) for value in args.cost_bps_levels) != (0.0,):
            raise ValueError("V2 no-regime is frozen to gross --cost-bps-levels 0")
        if not args.anonymize_tickers:
            raise ValueError("V2 no-regime requires --anonymize-tickers")
        if args.model != "deepseek-v4-flash":
            raise ValueError("V2 no-regime is frozen to --model deepseek-v4-flash")
        if args.thinking_mode != "disabled":
            raise ValueError("V2 no-regime is frozen to --thinking-mode disabled")
        if not any(np.isclose(float(args.temperature), value) for value in (0.6, 1.0)):
            raise ValueError("V2 no-regime supports only --temperature 0.6 or 1")
    if args.phase in {"llm-cache", "full"} and args.thinking_mode == "enabled":
        if args.model not in technical_llm.DEEPSEEK_V4_MODELS:
            raise ValueError(
                "Thinking runs must use --model deepseek-v4-flash or deepseek-v4-pro; "
                "legacy DeepSeek aliases are intentionally rejected."
            )
        if args.prompt_variant not in {
            PROMPT_VARIANT_V2,
            PROMPT_VARIANT_V2_FROZEN,
            PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL,
        }:
            raise ValueError(
                "Safe split-side thinking currently supports score-only v2, v2-frozen, "
                "or v2-minimal-change-control prompts"
            )
        if args.structured_reasoning or args.rationale_top_k_per_side:
            raise ValueError("Safe split-side thinking cannot be combined with rationale or structured-reasoning output")
    config = build_config(args)
    signal_date_manifest = (
        load_signal_date_manifest(Path(args.signal_dates_file))
        if args.signal_dates_file is not None
        else None
    )
    autoresearch_stage: str | None = None
    validation_freeze: dict[str, Any] | None = None
    if experimental_prompt_spec is not None:
        assert signal_date_manifest is not None
        autoresearch_stage = validate_autoresearch_manifest_scope(signal_date_manifest, config)
        if autoresearch_stage == "validation":
            if args.validation_freeze is None:
                raise ValueError("2015-2019 autoresearch dates require --validation-freeze")
            validation_freeze = load_and_validate_validation_freeze(
                Path(args.validation_freeze),
                experimental_prompt_spec,
                signal_date_manifest,
                args,
            )
        elif args.validation_freeze is not None:
            raise ValueError("A validation freeze may not be attached to a development run")
        append_progress_event(
            args.progress_file,
            "runner_started",
            phase=args.phase,
            stage=autoresearch_stage,
            candidate_id=experimental_prompt_spec.candidate_id,
            prompt_hash=experimental_prompt_spec.prompt_hash,
            date_manifest_hash=signal_date_manifest.date_manifest_hash,
            planned_dates=len(signal_date_manifest.dates),
            model=args.model,
            temperature=float(args.temperature),
            thinking_mode=args.thinking_mode,
        )
    if args.phase == "llm-cache":
        try:
            asyncio.run(
                run_v2_llm_cache_only(
                    config=config,
                    anchor_offsets=tuple(args.anchor_offsets),
                    schedule=args.schedule,
                    llm_pool_size=int(args.llm_pool_size),
                    model=args.model,
                    temperature=args.temperature,
                    thinking_mode=args.thinking_mode,
                    cache_path=cache_path_for_args(args),
                    rate_limit=args.rate_limit,
                    anonymize_tickers=bool(args.anonymize_tickers),
                    rationale_top_k_per_side=int(args.rationale_top_k_per_side),
                    structured_reasoning=bool(args.structured_reasoning),
                    prompt_variant=args.prompt_variant,
                    print_reasoning_content=bool(args.print_reasoning_content),
                    thinking_call_policy=args.thinking_call_policy,
                    llm_max_output_tokens=int(args.llm_max_output_tokens),
                    run_output_token_budget=int(args.run_output_token_budget),
                    llm_max_attempts=int(args.llm_max_attempts),
                    output_dir=Path(args.output_dir),
                    v25_paired_control=bool(args.v25_paired_control),
                    experimental_prompt_spec=experimental_prompt_spec,
                    signal_date_manifest=signal_date_manifest,
                    progress_file=args.progress_file,
                    run_call_budget=int(args.run_call_budget),
                    paired_control=bool(args.paired_control),
                    paired_control_cache=args.paired_control_cache,
                    autoresearch_stage=autoresearch_stage,
                )
            )
            if experimental_prompt_spec is not None:
                append_progress_event(
                    args.progress_file,
                    "runner_completed",
                    phase=args.phase,
                    stage=autoresearch_stage,
                    candidate_id=experimental_prompt_spec.candidate_id,
                    cache_path=str(cache_path_for_args(args).resolve()),
                )
        except LLMSafeStop as exc:
            append_progress_event(
                args.progress_file,
                "runner_stopped",
                phase=args.phase,
                stage=autoresearch_stage,
                candidate_id=(experimental_prompt_spec.candidate_id if experimental_prompt_spec else None),
                reason=str(exc),
            )
            print(f"\nLLM cache stopped safely: {exc}")
            print(f"Resume cache: {cache_path_for_args(args)}")
        return

    try:
        asyncio.run(
            run_v2_full(
                config=config,
                output_dir=Path(args.output_dir),
                pool_sizes=tuple(args.pool_sizes),
                book_sizes=tuple(args.book_sizes),
                anchor_offsets=tuple(args.anchor_offsets),
                schedule=args.schedule,
                llm_pool_size=int(args.llm_pool_size),
                legacy_root=Path(args.legacy_root),
                legacy_factor_root=Path(args.legacy_factor_root),
                v1_roots=tuple(args.v1_roots),
                v2_reference_roots=tuple(args.v2_reference_roots),
                model=args.model,
                temperature=args.temperature,
                thinking_mode=args.thinking_mode,
                cache_path=cache_path_for_args(args),
                rate_limit=args.rate_limit,
                llm=args.phase == "full",
                anonymize_tickers=bool(args.anonymize_tickers),
                rationale_top_k_per_side=int(args.rationale_top_k_per_side),
                structured_reasoning=bool(args.structured_reasoning),
                prompt_variant=args.prompt_variant,
                print_reasoning_content=bool(args.print_reasoning_content),
                v22_lambdas=tuple(float(value) for value in args.v22_lambdas),
                thinking_call_policy=args.thinking_call_policy,
                llm_max_output_tokens=int(args.llm_max_output_tokens),
                run_output_token_budget=int(args.run_output_token_budget),
                llm_max_attempts=int(args.llm_max_attempts),
                v25_paired_control=bool(args.v25_paired_control),
                incumbent_guarded=not bool(args.experimental_override),
                experimental_prompt_spec=experimental_prompt_spec,
                signal_date_manifest=signal_date_manifest,
                progress_file=args.progress_file,
                run_call_budget=int(args.run_call_budget),
                paired_control=bool(args.paired_control),
                paired_control_cache=args.paired_control_cache,
                autoresearch_stage=autoresearch_stage,
                validation_freeze=validation_freeze,
                autoresearch_harness_hash=(
                    autoresearch_harness_fingerprint(args) if experimental_prompt_spec is not None else None
                ),
            )
        )
        if experimental_prompt_spec is not None:
            append_progress_event(
                args.progress_file,
                "runner_completed",
                phase=args.phase,
                stage=autoresearch_stage,
                candidate_id=experimental_prompt_spec.candidate_id,
                output_dir=str(Path(args.output_dir).resolve()),
            )
    except LLMSafeStop as exc:
        append_progress_event(
            args.progress_file,
            "runner_stopped",
            phase=args.phase,
            stage=autoresearch_stage,
            candidate_id=(experimental_prompt_spec.candidate_id if experimental_prompt_spec else None),
            reason=str(exc),
        )
        print(f"\nLLM cache stopped safely: {exc}")
        print(f"Resume cache: {cache_path_for_args(args)}")


if __name__ == "__main__":
    main()
