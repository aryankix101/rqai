"""
LLM mean-reversion scoring experiment v1.

This runner follows the revised protocol: build a 150/150 reversal candidate
pool from the v3 Russell benchmark universe, ask the LLM for one 0-100 score per
name, and feed signed scores into the same C2C/O2O benchmark accounting.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from scipy import stats

import run_backtest_v3 as rb
import technical_llm
from market_regime import SPY_DATA_PATH, identify_regime
from russell_data_loader import load_russell_ohlcv_histories

PROMPT_STYLE_V1_SCORE = "v1_score"
PROMPT_STYLE_RAW_CONFIDENCE = "raw_confidence"
PROMPT_STYLE_RANK_CONFIDENCE = "rank_confidence"
DEFAULT_PROMPT_STYLE = PROMPT_STYLE_V1_SCORE
PROMPT_VERSION = "llm_reversal_signal_v1"
RAW_CONFIDENCE_PROMPT_VERSION = "llm_reversal_raw_confidence_v1"
RANK_CONFIDENCE_PROMPT_VERSION = "llm_reversal_rank_confidence_v1"
DEFAULT_START = "2009-01-01"
DEFAULT_END = "2014-01-01"
DEFAULT_POOL_PER_SIDE = 150
DEFAULT_POSITIONS_PER_SIDE = 75
DEFAULT_COST_BPS_LEVELS = (0.0, 2.5, 5.0)
DEFAULT_OUTPUT_DIR = Path("backtest_results/llm_reversal_signal_v1_2009_2014")
DEFAULT_CACHE_DIR = Path("backtest_results/llm_reversal_signal_v1_cache")
SCORE_COPY_CORR_THRESHOLD = 0.95
LLM_SCORE_MAX_ATTEMPTS = 4
FEATURE_COLUMNS = (
    "Ticker",
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


class EmptyLLMResponseError(ValueError):
    """Raised when the provider returns a choice without final message content."""


V1_SYSTEM_PROMPT = """
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

  LONG POOL  = the 60 biggest 5-day losers in the Russell 1000.
               Looking for snap-back / oversold-bounce setups.

  SHORT POOL = the 60 biggest 5-day winners in the Russell 1000.
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


RAW_CONFIDENCE_SYSTEM_PROMPT = """
MEAN-REVERSION RAW CONFIDENCE SCORING

CONTEXT
Mean reversion is the tendency of stocks that have moved sharply in one
direction to partially reverse over the next 5 trading days, especially
when the move shows signs of exhaustion (extreme indicators, volume
climax, range positioning). The strongest reversion setups combine an
extreme recent move with exhaustion signals and a supportive regime.

TASK
You are scoring two pools of stocks selected as mean-reversion candidates over
the next 5 trading days.

  LONG POOL  = biggest 5-day losers after Russell 1000, price, and ADV filters.
               Higher confidence means a stronger oversold snap-back edge
               versus other names in this same long pool.

  SHORT POOL = biggest 5-day winners after Russell 1000, price, and ADV filters.
               Higher confidence means a stronger overbought fade edge versus
               other names in this same short pool.

For each ticker in each pool, output a single SCORE from 0 to 100. In this
prompt, SCORE means:

  Confidence that this name has a better mean-reversion edge than other names
  in the same LONG or SHORT pool over the next 5 trading days.

Important interpretation rules:
- The score is not a position size.
- The score is not a probability of a positive return.
- The score is side-relative: compare long names only to long names, and short
  names only to short names.
- Use only the indicator columns provided. No outside knowledge, no ticker
  familiarity, no calendar awareness.
- Treat tickers as opaque identifiers.
- Use the full 0-100 range when the cross-section supports it.
- Do not produce ranks, weights, portfolio sizes, rationales, or commentary.

OUTPUT FORMAT (STRICT JSON, no other text, no markdown fences):
{
  "long_scores":  [{"ticker": "AAPL", "score": 72}, {"ticker": "MSFT", "score": 58}, ...],
  "short_scores": [{"ticker": "XYZ",  "score": 84}, {"ticker": "ABC", "score": 61}, ...]
}
""".strip()


RANK_CONFIDENCE_SYSTEM_PROMPT = """
MEAN-REVERSION RANK + CONFIDENCE SCORING

CONTEXT
Mean reversion is the tendency of stocks that have moved sharply in one
direction to partially reverse over the next 5 trading days, especially
when the move shows signs of exhaustion (extreme indicators, volume
climax, range positioning). The strongest reversion setups combine an
extreme recent move with exhaustion signals and a supportive regime.

TASK
You are ranking two pools of stocks selected as mean-reversion candidates over
the next 5 trading days.

  LONG POOL  = biggest 5-day losers after Russell 1000, price, and ADV filters.
               Rank 1 is the strongest oversold snap-back setup.

  SHORT POOL = biggest 5-day winners after Russell 1000, price, and ADV filters.
               Rank 1 is the strongest overbought fade setup.

For every ticker in each pool, output:
- rank: integer from 1 to N within that side, with no ties and no gaps.
- confidence_score: integer from 0 to 100 indicating confidence in that rank's
  mean-reversion edge versus other names in the same side.

Important interpretation rules:
- Rank long names only against other long names.
- Rank short names only against other short names.
- Do not rank across sides.
- Use only the indicator columns provided. No outside knowledge, no ticker
  familiarity, no calendar awareness.
- Treat tickers as opaque identifiers.
- Do not produce weights, portfolio sizes, rationales, or commentary.

OUTPUT FORMAT (STRICT JSON, no other text, no markdown fences):
{
  "long_rankings":  [{"ticker": "AAPL", "rank": 1, "confidence_score": 82}, {"ticker": "MSFT", "rank": 2, "confidence_score": 74}, ...],
  "short_rankings": [{"ticker": "XYZ",  "rank": 1, "confidence_score": 79}, {"ticker": "ABC", "rank": 2, "confidence_score": 70}, ...]
}
""".strip()


@dataclass(frozen=True)
class FeatureFrames:
    high_df: pd.DataFrame
    low_df: pd.DataFrame
    volume_df: pd.DataFrame


@dataclass(frozen=True)
class CandidatePool:
    signal_date: pd.Timestamp
    lookback_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_open_date: pd.Timestamp | None
    exit_open_date: pd.Timestamp | None
    valid_tickers: list[str]
    long_tickers: list[str]
    short_tickers: list[str]
    trail_by_ticker: dict[str, float]
    signal_by_ticker: dict[str, float]
    feature_rows: dict[str, dict[str, float | str]]
    universe_hash: str
    regime: str
    market_context: dict[str, Any] | None = None

    @property
    def pool_tickers(self) -> list[str]:
        return self.long_tickers + self.short_tickers


@dataclass(frozen=True)
class PoolBuildContext:
    adv20_df: pd.DataFrame
    fridays: pd.DatetimeIndex
    friday_set: set[pd.Timestamp]
    next_trade_map: dict[pd.Timestamp, pd.Timestamp]


@dataclass(frozen=True)
class ExperimentBundle:
    weekly_returns: pd.DataFrame
    signal_checks: pd.DataFrame
    portfolio_selections: pd.DataFrame
    headline_metrics: pd.DataFrame
    yearly_returns: pd.DataFrame
    strategy_summary: pd.DataFrame


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _pct_return(current: float, prior: float) -> float:
    if pd.isna(current) or pd.isna(prior) or current <= 0 or prior <= 0:
        return np.nan
    return float(current / prior - 1.0)


def _percentile_scores(values: np.ndarray, high_is_best: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    if len(values) == 1:
        return np.array([50.0], dtype=float)
    order = np.argsort(-values if high_is_best else values, kind="mergesort")
    positions = np.empty(len(values), dtype=float)
    positions[order] = np.arange(len(values), dtype=float)
    return 100.0 * (1.0 - positions / (len(values) - 1))


def build_feature_frames(histories: dict[str, pd.DataFrame]) -> FeatureFrames:
    high_cols: dict[str, pd.Series] = {}
    low_cols: dict[str, pd.Series] = {}
    volume_cols: dict[str, pd.Series] = {}

    for ticker, history in histories.items():
        if history.empty:
            continue
        df = history.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if df.empty:
            continue
        df = df.set_index("date")
        high = pd.to_numeric(df.get("adj_high"), errors="coerce") if "adj_high" in df.columns else pd.Series(index=df.index, dtype=float)
        low = pd.to_numeric(df.get("adj_low"), errors="coerce") if "adj_low" in df.columns else pd.Series(index=df.index, dtype=float)
        if "high" in df.columns:
            high = high.combine_first(pd.to_numeric(df["high"], errors="coerce"))
        if "low" in df.columns:
            low = low.combine_first(pd.to_numeric(df["low"], errors="coerce"))
        if high.isna().all():
            high = pd.to_numeric(df.get("adj_close", df.get("close")), errors="coerce")
        if low.isna().all():
            low = pd.to_numeric(df.get("adj_close", df.get("close")), errors="coerce")
        high_cols[ticker] = high
        low_cols[ticker] = low
        volume_cols[ticker] = pd.to_numeric(df.get("volume"), errors="coerce")

    return FeatureFrames(
        high_df=pd.DataFrame(high_cols).sort_index(),
        low_df=pd.DataFrame(low_cols).sort_index(),
        volume_df=pd.DataFrame(volume_cols).sort_index(),
    )


def build_pool_context(market_frames: rb.MarketFrames) -> PoolBuildContext:
    trading_dates = market_frames.close_df.index
    fridays = trading_dates[trading_dates.dayofweek == 4]
    return PoolBuildContext(
        adv20_df=market_frames.dollar_volume_df.rolling(20, min_periods=15).mean(),
        fridays=fridays,
        friday_set={pd.Timestamp(date) for date in fridays},
        next_trade_map=rb._next_trading_day_map(trading_dates),
    )


def _rsi_last(close: pd.Series, window: int = 5) -> float:
    clean = close.dropna()
    if len(clean) < window + 1:
        return np.nan
    delta = clean.diff().iloc[-window:]
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _autocorr_lag(close: pd.Series, lag: int = 5) -> float:
    clean = close.dropna()
    min_obs = lag * 4
    if len(clean) < min_obs:
        return np.nan
    returns = np.diff(np.log(clean.iloc[-min_obs:].to_numpy(dtype=float)))
    if len(returns) <= lag:
        return np.nan
    current = returns[lag:]
    lagged = returns[:-lag]
    if len(current) < lag or np.std(current) <= 0.0001 or np.std(lagged) <= 0.0001:
        return 0.0
    corr = np.corrcoef(current, lagged)[0, 1]
    return _safe_float(corr, 0.0)


def _adx_last(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
    frame = pd.DataFrame({"high": high, "low": low, "close": close}).dropna()
    if len(frame) < window * 2 + 1:
        return np.nan
    high_s = frame["high"]
    low_s = frame["low"]
    close_s = frame["close"]
    up_move = high_s.diff()
    down_move = -low_s.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=frame.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=frame.index,
    )
    prev_close = close_s.shift(1)
    true_range = pd.concat(
        [
            high_s - low_s,
            (high_s - prev_close).abs(),
            (low_s - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window, min_periods=window).mean()
    plus_di = 100.0 * plus_dm.rolling(window, min_periods=window).mean() / atr
    minus_di = 100.0 * minus_dm.rolling(window, min_periods=window).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window, min_periods=window).mean().iloc[-1]
    return _safe_float(adx)


def _vol_percentile_504(close: pd.Series) -> float:
    clean = close.dropna()
    if len(clean) < 526:
        return np.nan
    log_prices = np.log(clean.to_numpy(dtype=float))
    rolling_vols: list[float] = []
    for end in range(len(log_prices) - 504, len(log_prices) + 1):
        start = end - 22
        if start < 0 or end <= start:
            continue
        returns = np.diff(log_prices[start:end])
        if len(returns) >= 20:
            vol = np.std(returns, ddof=1) * np.sqrt(252)
            if np.isfinite(vol):
                rolling_vols.append(float(vol))
    if len(rolling_vols) < 50:
        return np.nan
    current_vol = rolling_vols[-1]
    hist = np.asarray(rolling_vols[:-1] or rolling_vols, dtype=float)
    return float((hist < current_vol).sum() / len(hist) * 100.0)


def compute_prompt_features(
    market_frames: rb.MarketFrames,
    feature_frames: FeatureFrames,
    ticker: str,
    signal_date: pd.Timestamp,
    trail_return: float,
    mrl: float,
    mrs: float,
) -> dict[str, float | str]:
    close = market_frames.close_df.loc[:signal_date, ticker].dropna()
    high = feature_frames.high_df.loc[:signal_date, ticker].dropna()
    low = feature_frames.low_df.loc[:signal_date, ticker].dropna()
    current = _safe_float(close.iloc[-1] if not close.empty else np.nan)

    high_55 = high.iloc[-55:].max() if len(high) >= 55 else np.nan
    low_55 = low.iloc[-55:].min() if len(low) >= 55 else np.nan
    rng55 = (current - low_55) / (high_55 - low_55) if pd.notna(high_55) and high_55 > low_55 else np.nan

    vol_21 = np.nan
    if len(close) >= 22:
        returns_21 = np.diff(np.log(close.iloc[-22:].to_numpy(dtype=float)))
        vol_21 = np.std(returns_21, ddof=1) * np.sqrt(252) if len(returns_21) >= 20 else np.nan

    vel_v = np.nan
    if len(close) >= 11 and pd.notna(vol_21) and vol_21 > 0:
        vel_v = _pct_return(current, close.iloc[-11]) / vol_21

    ma100 = close.iloc[-100:].mean() if len(close) >= 100 else np.nan
    pct_ma100 = (current / ma100 - 1.0) * 100.0 if pd.notna(ma100) and ma100 > 0 else np.nan

    high_252 = high.iloc[-253:].max() if len(high) >= 253 else np.nan
    pct_52wh = (current / high_252 - 1.0) * 100.0 if pd.notna(high_252) and high_252 > 0 else np.nan

    mom3m = _pct_return(current, close.iloc[-64]) * 100.0 if len(close) >= 64 else np.nan

    return {
        "Ticker": ticker,
        "MRL": float(mrl),
        "MRS": float(mrs),
        "Trail5": float(trail_return * 100.0),
        "MRSig": float(-trail_return * 100.0),
        "RSI5": _rsi_last(close, 5),
        "Rng55": _safe_float(rng55),
        "Vel/V": _safe_float(vel_v),
        "%MA100": _safe_float(pct_ma100),
        "52WH": _safe_float(pct_52wh),
        "Mom3m": _safe_float(mom3m),
        "AutoC": _autocorr_lag(close, 5),
        "ADX": _adx_last(high, low, close, 14),
        "VolPctl": _vol_percentile_504(close),
    }


def _format_feature_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    numeric = _safe_float(value)
    if pd.isna(numeric):
        return "nan"
    return f"{numeric:.3f}"


def prompt_version_for_style(prompt_style: str) -> str:
    if prompt_style == PROMPT_STYLE_V1_SCORE:
        return PROMPT_VERSION
    if prompt_style == PROMPT_STYLE_RAW_CONFIDENCE:
        return RAW_CONFIDENCE_PROMPT_VERSION
    if prompt_style == PROMPT_STYLE_RANK_CONFIDENCE:
        return RANK_CONFIDENCE_PROMPT_VERSION
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def system_prompt_for_style(prompt_style: str) -> str:
    if prompt_style == PROMPT_STYLE_V1_SCORE:
        return V1_SYSTEM_PROMPT
    if prompt_style == PROMPT_STYLE_RAW_CONFIDENCE:
        return RAW_CONFIDENCE_SYSTEM_PROMPT
    if prompt_style == PROMPT_STYLE_RANK_CONFIDENCE:
        return RANK_CONFIDENCE_SYSTEM_PROMPT
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def format_user_prompt(pool: CandidatePool) -> str:
    columns = "  ".join(FEATURE_COLUMNS)

    def rows(tickers: list[str]) -> str:
        lines = []
        for ticker in tickers:
            row = pool.feature_rows[ticker]
            lines.append("  ".join(_format_feature_value(row[col]) for col in FEATURE_COLUMNS))
        return "\n".join(lines)

    return f"""
REBALANCE DATE: {pool.signal_date.date()}
MARKET REGIME: {pool.regime}

LONG POOL ({len(pool.long_tickers)} names - biggest 5-day losers in Russell 1000):
Columns: {columns}
{rows(pool.long_tickers)}

SHORT POOL ({len(pool.short_tickers)} names - biggest 5-day winners in Russell 1000):
Columns: {columns}
{rows(pool.short_tickers)}

Score every ticker in both pools. Return JSON only.
""".strip()


def _features_for_hash(pool_rows: dict[str, dict[str, float | str]], tickers: list[str]) -> list[dict[str, Any]]:
    payload = []
    for ticker in tickers:
        row = pool_rows[ticker]
        payload.append({col: row[col] for col in FEATURE_COLUMNS})
    return payload


def _hash_pool_payload(
    signal_date: pd.Timestamp,
    regime: str,
    long_tickers: list[str],
    short_tickers: list[str],
    feature_rows: dict[str, dict[str, float | str]],
) -> str:
    payload = {
        "date": str(pd.Timestamp(signal_date).date()),
        "regime": regime,
        "prompt_version": PROMPT_VERSION,
        "long": _features_for_hash(feature_rows, long_tickers),
        "short": _features_for_hash(feature_rows, short_tickers),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def build_candidate_pool(
    market_frames: rb.MarketFrames,
    feature_frames: FeatureFrames,
    membership_lookup: rb.MembershipLookup,
    spy_df: pd.DataFrame | None,
    signal_date: pd.Timestamp,
    pool_per_side: int,
    config: rb.BenchmarkConfig,
    context: PoolBuildContext | None = None,
) -> CandidatePool | None:
    close_df = market_frames.close_df
    unadj_close_df = market_frames.unadj_close_df
    context = context or build_pool_context(market_frames)
    adv20_df = context.adv20_df
    fridays = context.fridays
    if signal_date not in context.friday_set:
        return None
    friday_pos = int(fridays.get_loc(signal_date))
    if friday_pos <= 0 or friday_pos + 1 >= len(fridays):
        return None
    lookback_date = pd.Timestamp(fridays[friday_pos - 1])
    exit_date = pd.Timestamp(fridays[friday_pos + 1])

    r1000_universe = membership_lookup.universe_on_or_before(signal_date)
    valid: list[str] = []
    trails: list[float] = []
    for ticker in sorted(r1000_universe):
        if ticker not in close_df.columns:
            continue
        close_prev = close_df.at[lookback_date, ticker] if lookback_date in close_df.index else np.nan
        close_now = close_df.at[signal_date, ticker] if signal_date in close_df.index else np.nan
        trail = _pct_return(close_now, close_prev)
        if pd.isna(trail):
            continue
        raw_price = unadj_close_df.at[signal_date, ticker] if signal_date in unadj_close_df.index else np.nan
        if pd.isna(raw_price) or raw_price < config.min_trade_price:
            continue
        adv20 = adv20_df.at[signal_date, ticker] if signal_date in adv20_df.index else np.nan
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

    mrl_scores = _percentile_scores(signal_arr, high_is_best=True)
    mrs_scores = 100.0 - mrl_scores
    trail_by_ticker = {ticker: float(trail_arr[idx]) for idx, ticker in enumerate(valid)}
    signal_by_ticker = {ticker: float(signal_arr[idx]) for idx, ticker in enumerate(valid)}
    feature_rows: dict[str, dict[str, float | str]] = {}
    for idx, ticker in enumerate(valid):
        if ticker not in long_tickers and ticker not in short_tickers:
            continue
        feature_rows[ticker] = compute_prompt_features(
            market_frames=market_frames,
            feature_frames=feature_frames,
            ticker=ticker,
            signal_date=signal_date,
            trail_return=trail_by_ticker[ticker],
            mrl=float(mrl_scores[idx]),
            mrs=float(mrs_scores[idx]),
        )

    regime = identify_regime(spy_df, signal_date) if spy_df is not None else "unknown"
    universe_hash = _hash_pool_payload(signal_date, regime, long_tickers, short_tickers, feature_rows)
    return CandidatePool(
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
    )


def _rank_confidence_score(rank: int, side_count: int, confidence_score: float) -> int:
    if side_count <= 0:
        return 0
    rank_component = (side_count - rank + 1) / side_count * 100.0
    score = rank_component * (confidence_score / 100.0)
    return int(round(min(100.0, max(0.0, score))))


def validate_llm_scores(
    raw_output: dict[str, Any],
    expected_longs: list[str],
    expected_shorts: list[str],
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> dict[str, Any]:
    if not isinstance(raw_output, dict):
        raise ValueError("LLM output must be a JSON object")

    expected_long_set = {ticker.upper() for ticker in expected_longs}
    expected_short_set = {ticker.upper() for ticker in expected_shorts}
    if expected_long_set & expected_short_set:
        raise ValueError("Input pools overlap")

    if prompt_style == PROMPT_STYLE_RANK_CONFIDENCE:
        def clean_ranking_side(
            ranking_key: str,
            score_key: str,
            expected: set[str],
            other: set[str],
        ) -> list[dict[str, int | str]]:
            raw_entries = raw_output.get(ranking_key)
            key = ranking_key
            if raw_entries is None:
                raw_entries = raw_output.get(score_key)
                key = score_key
            if not isinstance(raw_entries, list):
                raise ValueError(f"Missing {ranking_key} or {score_key} array")
            seen_tickers: set[str] = set()
            seen_ranks: set[int] = set()
            cleaned = []
            side_count = len(expected)
            for entry in raw_entries:
                if not isinstance(entry, dict):
                    raise ValueError(f"{key} contains a non-object entry")
                ticker = str(entry.get("ticker", "")).strip().upper()
                if not ticker:
                    raise ValueError(f"{key} contains missing ticker")
                if ticker in seen_tickers:
                    raise ValueError(f"{key} contains duplicate ticker {ticker}")
                if ticker in other:
                    raise ValueError(f"{ticker} appears on the wrong side")
                if ticker not in expected:
                    raise ValueError(f"{key} contains unexpected ticker {ticker}")

                rank_raw = entry.get("rank")
                if isinstance(rank_raw, (bool, np.bool_)) or not isinstance(rank_raw, (int, np.integer)):
                    raise ValueError(f"{ticker} has invalid rank {rank_raw!r}")
                rank = int(rank_raw)
                if rank < 1 or rank > side_count:
                    raise ValueError(f"{ticker} has rank outside 1..{side_count}: {rank}")
                if rank in seen_ranks:
                    raise ValueError(f"{key} contains duplicate rank {rank}")

                confidence_raw = entry.get("confidence_score")
                if isinstance(confidence_raw, (bool, np.bool_)) or not isinstance(
                    confidence_raw,
                    (int, float, np.integer, np.floating),
                ):
                    raise ValueError(f"{ticker} has non-numeric confidence_score {confidence_raw!r}")
                confidence = _safe_float(confidence_raw)
                if pd.isna(confidence) or confidence < 0.0 or confidence > 100.0:
                    raise ValueError(f"{ticker} has invalid confidence_score {confidence_raw!r}")

                seen_tickers.add(ticker)
                seen_ranks.add(rank)
                cleaned.append(
                    {
                        "ticker": ticker,
                        "rank": rank,
                        "confidence_score": int(round(confidence)),
                        "score": _rank_confidence_score(rank, side_count, confidence),
                    }
                )
            missing = expected - seen_tickers
            if missing:
                raise ValueError(f"{key} missing tickers: {', '.join(sorted(missing)[:10])}")
            if seen_ranks != set(range(1, side_count + 1)):
                raise ValueError(f"{key} ranks must be a complete 1..{side_count} permutation")
            return sorted(cleaned, key=lambda item: int(item["rank"]))

        long_scores = clean_ranking_side("long_rankings", "long_scores", expected_long_set, expected_short_set)
        short_scores = clean_ranking_side("short_rankings", "short_scores", expected_short_set, expected_long_set)
        return {"long_scores": long_scores, "short_scores": short_scores}

    def clean_side(key: str, expected: set[str], other: set[str]) -> list[dict[str, int | str]]:
        raw_entries = raw_output.get(key)
        if not isinstance(raw_entries, list):
            raise ValueError(f"Missing {key} array")
        seen: set[str] = set()
        cleaned = []
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
            score_raw = entry.get("score")
            if isinstance(score_raw, (bool, np.bool_)) or not isinstance(
                score_raw,
                (int, float, np.integer, np.floating),
            ):
                raise ValueError(f"{ticker} has non-numeric score {score_raw!r}")
            score = _safe_float(score_raw)
            if pd.isna(score) or score < 0.0 or score > 100.0:
                raise ValueError(f"{ticker} has invalid score {score_raw!r}")
            seen.add(ticker)
            cleaned.append({"ticker": ticker, "score": int(round(score))})
        missing = expected - seen
        if missing:
            raise ValueError(f"{key} missing tickers: {', '.join(sorted(missing)[:10])}")
        if len(cleaned) != len(expected):
            raise ValueError(f"{key} expected {len(expected)} entries, got {len(cleaned)}")
        return cleaned

    long_scores = clean_side("long_scores", expected_long_set, expected_short_set)
    short_scores = clean_side("short_scores", expected_short_set, expected_long_set)
    return {"long_scores": long_scores, "short_scores": short_scores}


def signed_scores_from_llm(validated_scores: dict[str, Any]) -> dict[str, float]:
    signed: dict[str, float] = {}
    for entry in validated_scores.get("long_scores", []):
        signed[str(entry["ticker"]).upper()] = float(entry["score"])
    for entry in validated_scores.get("short_scores", []):
        signed[str(entry["ticker"]).upper()] = -float(entry["score"])
    return signed


def _cache_key(date: pd.Timestamp, prompt_version: str, model: str, universe_hash: str) -> str:
    return f"{pd.Timestamp(date).date()}|{prompt_version}|{model}|{universe_hash}"


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"metadata": {}, "signals": {}, "failures": {}}
    with path.open("r") as handle:
        return json.load(handle)


def save_cache(cache: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
    tmp.replace(path)


async def call_llm_scores(
    client: httpx.AsyncClient,
    pool: CandidatePool,
    model: str,
    temperature: float,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    print_reasoning_content: bool = False,
) -> dict[str, Any]:
    user_prompt = format_user_prompt(pool)
    base_messages = [
        {"role": "system", "content": system_prompt_for_style(prompt_style)},
        {"role": "user", "content": user_prompt},
    ]
    retry_message = {
        "role": "user",
        "content": (
            "Your previous response was invalid. Return ONLY strict JSON with "
            "the exact arrays required by the system prompt. Include every provided "
            "ticker exactly once on its original side. Use valid numeric scores or "
            "a complete rank permutation as requested. No commentary."
        ),
    }
    url = technical_llm.get_deepseek_url()
    headers = {"Authorization": f"Bearer {technical_llm.get_deepseek_api_key()}"}
    last_error: Exception | None = None
    started = time.time()

    for attempt in range(LLM_SCORE_MAX_ATTEMPTS):
        use_retry_instruction = last_error is not None and not isinstance(last_error, EmptyLLMResponseError)
        messages = base_messages + ([retry_message] if use_retry_instruction else [])
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        technical_llm.apply_deepseek_request_mode(payload, model)
        if technical_llm.model_supports_temperature(model):
            payload["temperature"] = temperature
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            response_payload = response.json()
            choices = response_payload.get("choices") or [{}]
            choice = choices[0] if choices else {}
            message = choice.get("message") or {}
            content = message.get("content") or ""
            reasoning_content = message.get("reasoning_content") or ""
            if print_reasoning_content:
                print(f"\n--- BEGIN reasoning_content {pool.signal_date.date()} ---")
                print(reasoning_content or "[empty reasoning_content]")
                print(f"--- END reasoning_content {pool.signal_date.date()} ---\n")
            if not content:
                raise EmptyLLMResponseError(
                    "No content in LLM response "
                    f"(finish_reason={choice.get('finish_reason')!r}, "
                    f"message_keys={sorted(message.keys())}, "
                    f"usage={response_payload.get('usage', {})})"
                )
            raw_output = technical_llm._parse_json_content(content)
            validated = validate_llm_scores(raw_output, pool.long_tickers, pool.short_tickers, prompt_style)
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
        except Exception as exc:  # noqa: BLE001 - callers need one consolidated failure.
            last_error = exc
            if attempt < LLM_SCORE_MAX_ATTEMPTS - 1:
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError(f"LLM score call failed after {LLM_SCORE_MAX_ATTEMPTS} attempts: {exc}") from exc

    raise RuntimeError(f"LLM score call failed: {last_error}")


def _entry_from_cache(
    entry: dict[str, Any],
    pool: CandidatePool,
    model: str,
    prompt_version: str = PROMPT_VERSION,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> dict[str, Any] | None:
    expected_key = _cache_key(pool.signal_date, prompt_version, model, pool.universe_hash)
    if entry.get("cache_key") != expected_key:
        return None
    return validate_llm_scores(entry, pool.long_tickers, pool.short_tickers, prompt_style)


def _record_invalid_cache_entry(
    cache: dict[str, Any],
    date_key: str,
    exc: Exception,
    prompt_version: str,
    prompt_style: str,
    model: str,
    universe_hash: str,
) -> None:
    cache.setdefault("signals", {}).pop(date_key, None)
    cache.setdefault("failures", {})[date_key] = {
        "error": f"Invalid cached LLM score entry: {exc}",
        "parse_status": "invalid_cache",
        "retry_count": 0,
        "prompt_version": prompt_version,
        "prompt_style": prompt_style,
        "model": model,
        "universe_hash": universe_hash,
        "failed_at": pd.Timestamp.utcnow().isoformat(),
    }


async def ensure_llm_score_for_pool(
    client: httpx.AsyncClient,
    pool: CandidatePool,
    cache: dict[str, Any],
    cache_path: Path,
    model: str,
    temperature: float,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> dict[str, Any]:
    prompt_version = prompt_version_for_style(prompt_style)
    date_str = str(pool.signal_date.date())
    existing = cache.setdefault("signals", {}).get(date_str)
    if isinstance(existing, dict):
        try:
            cached = _entry_from_cache(existing, pool, model, prompt_version, prompt_style)
            if cached is not None:
                return cached
        except ValueError as exc:
            _record_invalid_cache_entry(
                cache,
                date_str,
                exc,
                prompt_version,
                prompt_style,
                model,
                pool.universe_hash,
            )
            save_cache(cache, cache_path)

    try:
        result = await call_llm_scores(client, pool, model=model, temperature=temperature, prompt_style=prompt_style)
        cache_entry = {
            **result,
            "rebalance_date": date_str,
            "prompt_version": prompt_version,
            "prompt_style": prompt_style,
            "model": model,
            "universe_hash": pool.universe_hash,
            "cache_key": _cache_key(pool.signal_date, prompt_version, model, pool.universe_hash),
            "pool_per_side": len(pool.long_tickers),
            "regime": pool.regime,
        }
        cache["signals"][date_str] = cache_entry
        cache.setdefault("failures", {}).pop(date_str, None)
        save_cache(cache, cache_path)
        return validate_llm_scores(cache_entry, pool.long_tickers, pool.short_tickers, prompt_style)
    except Exception as exc:
        cache.setdefault("failures", {})[date_str] = {
            "error": str(exc),
            "parse_status": "failed",
            "retry_count": 1,
            "prompt_version": prompt_version,
            "prompt_style": prompt_style,
            "model": model,
            "universe_hash": pool.universe_hash,
            "failed_at": pd.Timestamp.utcnow().isoformat(),
        }
        save_cache(cache, cache_path)
        raise


def _book_masks_from_scores(
    pool_tickers: list[str],
    signed_scores: dict[str, float],
    long_tickers: set[str],
    short_tickers: set[str],
    positions_per_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray([signed_scores.get(ticker, np.nan) for ticker in pool_tickers], dtype=float)
    if np.isnan(signal).any():
        missing = [ticker for ticker, score in zip(pool_tickers, signal) if pd.isna(score)]
        raise ValueError(f"Missing signed scores for pool tickers: {missing[:10]}")

    # Keep zero-score ties on their intended side while preserving the signed-score rule.
    side_bias = np.asarray([1e-9 if ticker in long_tickers else -1e-9 for ticker in pool_tickers], dtype=float)
    long_order = sorted(
        range(len(pool_tickers)),
        key=lambda idx: (-(signal[idx] + side_bias[idx]), pool_tickers[idx]),
    )
    short_order = sorted(
        range(len(pool_tickers)),
        key=lambda idx: (signal[idx] + side_bias[idx], pool_tickers[idx]),
    )
    long_mask = np.zeros(len(pool_tickers), dtype=bool)
    short_mask = np.zeros(len(pool_tickers), dtype=bool)
    long_mask[long_order[:positions_per_side]] = True
    short_mask[short_order[:positions_per_side]] = True
    if set(np.where(long_mask)[0]) & set(np.where(short_mask)[0]):
        raise ValueError("Long and short books overlap")
    return signal, long_mask, short_mask


def _deciles_from_signal(signal: np.ndarray) -> np.ndarray:
    _, deciles = rb._rank_signal(signal)
    return deciles


def _spearman(signal: np.ndarray, returns: np.ndarray) -> float:
    valid = (~np.isnan(signal)) & (~np.isnan(returns))
    if valid.sum() <= 20 or len(np.unique(signal[valid])) <= 1:
        return np.nan
    return _safe_float(stats.spearmanr(signal[valid], returns[valid])[0])


def run_pool_strategy(
    market_frames: rb.MarketFrames,
    feature_frames: FeatureFrames,
    membership_lookup: rb.MembershipLookup,
    spy_df: pd.DataFrame | None,
    config: rb.BenchmarkConfig,
    pool_per_side: int,
    positions_per_side: int,
    score_provider: str,
    llm_scores_by_date: dict[str, dict[str, Any]] | None = None,
    llm_prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[CandidatePool]]:
    close_df = market_frames.close_df
    open_df = market_frames.open_df
    trading_dates = close_df.index
    fridays = trading_dates[trading_dates.dayofweek == 4]
    results: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    pools: list[CandidatePool] = []
    prev_long_set: set[str] = set()
    prev_short_set: set[str] = set()
    pool_context = build_pool_context(market_frames)

    for idx, signal_date_raw in enumerate(fridays):
        signal_date = pd.Timestamp(signal_date_raw)
        if signal_date < config.start or signal_date > config.end:
            continue
        if int(trading_dates.get_loc(signal_date)) < config.signal_lookback_days:
            continue

        pool = build_candidate_pool(
            market_frames=market_frames,
            feature_frames=feature_frames,
            membership_lookup=membership_lookup,
            spy_df=spy_df,
            signal_date=signal_date,
            pool_per_side=pool_per_side,
            config=config,
            context=pool_context,
        )
        if pool is None or pool.exit_date > config.end:
            continue
        pools.append(pool)
        pool_tickers = pool.pool_tickers
        long_set = set(pool.long_tickers)
        short_set = set(pool.short_tickers)

        if score_provider == "trail5":
            signed_scores = {ticker: pool.signal_by_ticker[ticker] for ticker in pool_tickers}
        elif score_provider == "llm":
            if llm_scores_by_date is None:
                raise ValueError("llm_scores_by_date is required for score_provider='llm'")
            date_key = str(pool.signal_date.date())
            validated = validate_llm_scores(
                llm_scores_by_date[date_key],
                pool.long_tickers,
                pool.short_tickers,
                llm_prompt_style,
            )
            signed_scores = signed_scores_from_llm(validated)
        else:
            raise ValueError(f"Unsupported score_provider: {score_provider}")

        signal, long_mask, short_mask = _book_masks_from_scores(
            pool_tickers,
            signed_scores,
            long_set,
            short_set,
            positions_per_side,
        )
        deciles = _deciles_from_signal(signal)
        fwd_c2c = rb._forward_c2c_return(close_df, pool.signal_date, pool.exit_date, pool_tickers)
        fwd_o2o = rb._forward_o2o_return(open_df, pool.entry_open_date, pool.exit_open_date, pool_tickers)
        long_tickers = [pool_tickers[pos] for pos in np.where(long_mask)[0]]
        short_tickers = [pool_tickers[pos] for pos in np.where(short_mask)[0]]

        row: dict[str, Any] = {
            "date": pool.signal_date,
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
            "ic_c2c": _spearman(signal, fwd_c2c),
            "ic_o2o": _spearman(signal, fwd_o2o),
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
            raw_score = abs(signed_scores[ticker]) if score_provider == "llm" else np.nan
            signal_rows.append(
                {
                    "date": pool.signal_date,
                    "ticker": ticker,
                    "pool_side": "LONG" if ticker in long_set else "SHORT",
                    "score_provider": score_provider,
                    "llm_score": raw_score,
                    "signal": signal[pos],
                    "trail_ret": pool.trail_by_ticker[ticker],
                    "fwd_c2c": fwd_c2c[pos],
                    "fwd_o2o": fwd_o2o[pos],
                    "decile": deciles[pos],
                    **{f"feature_{col}": feature_row[col] for col in FEATURE_COLUMNS if col != "Ticker"},
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
                selection_rows.append(
                    {
                        "rebalance_date": pool.signal_date,
                        "signal_date": pool.signal_date,
                        "entry_date": pool.signal_date,
                        "exit_date": pool.exit_date,
                        "ticker": pool_tickers[pos],
                        "position_type": side,
                        "rank": rank,
                        "score": signal[pos],
                        "trail_return_pct": pool.trail_by_ticker[pool_tickers[pos]] * 100.0,
                        "realized_return_pct": fwd_c2c[pos] * 100.0,
                        "realized_c2c_return_pct": fwd_c2c[pos] * 100.0,
                        "realized_o2o_return_pct": fwd_o2o[pos] * 100.0,
                        "weight": side_weight,
                        "decile": deciles[pos],
                    }
                )

        if idx % 100 == 0:
            print(
                f"  {score_provider} {pool.signal_date.date()} "
                f"universe={len(pool.valid_tickers)} pool={len(pool_tickers)}"
            )

    weekly_returns = pd.DataFrame(results)
    if weekly_returns.empty:
        raise RuntimeError(f"No {score_provider} pool results were produced")
    weekly_returns["date"] = pd.to_datetime(weekly_returns["date"])
    weekly_returns = weekly_returns.set_index("date").sort_index()
    return weekly_returns, pd.DataFrame(signal_rows), pd.DataFrame(selection_rows), pools


def build_bundle(
    weekly_returns: pd.DataFrame,
    signal_checks: pd.DataFrame,
    portfolio_selections: pd.DataFrame,
    config: rb.BenchmarkConfig,
) -> ExperimentBundle:
    weekly_with_costs = rb.add_cost_overlays(weekly_returns, config.cost_bps_levels)
    return ExperimentBundle(
        weekly_returns=weekly_with_costs,
        signal_checks=signal_checks,
        portfolio_selections=portfolio_selections,
        headline_metrics=rb.build_headline_metrics(weekly_with_costs, periods_per_year=config.periods_per_year()),
        yearly_returns=rb.build_yearly_returns(weekly_with_costs),
        strategy_summary=rb.build_strategy_summary(weekly_with_costs, periods_per_year=config.periods_per_year()),
    )


def save_bundle(bundle: ExperimentBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle.weekly_returns.to_csv(output_dir / "weekly_returns.csv")
    bundle.signal_checks.to_csv(output_dir / "signal_checks.csv", index=False)
    bundle.portfolio_selections.to_csv(output_dir / "portfolio_selections.csv", index=False)
    bundle.headline_metrics.to_csv(output_dir / "headline_metrics.csv", index=False)
    bundle.yearly_returns.to_csv(output_dir / "yearly_returns.csv")
    bundle.strategy_summary.to_csv(output_dir / "strategy_summary.csv", index=False)


def _variant_column(execution_model: str, cost_bps: float) -> str:
    prefix = "ls_c2c" if execution_model == "C2C" else "ls_o2o"
    return f"{prefix}_{rb.format_bps_label(cost_bps)}"


def build_comparison_scorecard(bundles: dict[str, ExperimentBundle]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, bundle in bundles.items():
        weekly = bundle.weekly_returns
        for execution_model in ("C2C", "O2O"):
            ic_col = "ic_c2c" if execution_model == "C2C" else "ic_o2o"
            decile_prefix = "c2c" if execution_model == "C2C" else "o2o"
            ic_values = weekly[ic_col].dropna() if ic_col in weekly.columns else pd.Series(dtype=float)
            ic_mean = float(ic_values.mean()) if not ic_values.empty else np.nan
            ic_std = float(ic_values.std()) if len(ic_values) > 1 else np.nan
            ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_values))) if len(ic_values) > 1 and ic_std > 0 else np.nan
            decile_spread = np.nan
            d1 = f"dec_1_{decile_prefix}"
            d10 = f"dec_10_{decile_prefix}"
            if d1 in weekly.columns and d10 in weekly.columns:
                decile_spread = float((weekly[d1] - weekly[d10]).mean() * 100.0)

            for cost_bps in rb.available_cost_levels(weekly, "ls_c2c" if execution_model == "C2C" else "ls_o2o"):
                column = _variant_column(execution_model, cost_bps)
                returns = weekly[column].dropna()
                if returns.empty:
                    continue
                ann_ret = float(returns.mean() * rb.DEFAULT_PERIODS_PER_YEAR)
                ann_vol = float(returns.std() * np.sqrt(rb.DEFAULT_PERIODS_PER_YEAR))
                row: dict[str, Any] = {
                    "run": run_name,
                    "execution_model": execution_model,
                    "cost_bps": float(cost_bps),
                    "annual_return_pct": ann_ret * 100.0,
                    "annual_vol_pct": ann_vol * 100.0,
                    "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0.0,
                    "total_return_pct": float(((1.0 + returns).prod() - 1.0) * 100.0),
                    "cagr_pct": rb.safe_cagr(returns, periods_per_year=rb.DEFAULT_PERIODS_PER_YEAR),
                    "max_drawdown_pct": rb.compute_max_drawdown_pct(returns),
                    "hit_rate_pct": float((returns > 0).mean() * 100.0),
                    "ic_mean": ic_mean,
                    "ic_std": ic_std,
                    "ic_tstat": ic_tstat,
                    "decile_spread_pct": decile_spread,
                    "avg_turnover_long": float(weekly["turnover_long"].mean()),
                    "avg_turnover_short": float(weekly["turnover_short"].mean()),
                    "avg_turnover_total": float((weekly["turnover_long"] + weekly["turnover_short"]).mean()),
                    "n_periods": int(len(returns)),
                }
                for year, group in returns.groupby(returns.index.year):
                    row[f"year_{int(year)}_return_pct"] = float(((1.0 + group).prod() - 1.0) * 100.0)
                rows.append(row)
    return pd.DataFrame(rows)


def evaluate_success(comparison: pd.DataFrame) -> dict[str, Any]:
    c2c_5 = comparison[(comparison["execution_model"] == "C2C") & (comparison["cost_bps"] == 5.0)]
    gross = comparison[(comparison["execution_model"] == "C2C") & (comparison["cost_bps"] == 0.0)]
    status: dict[str, Any] = {
        "win_rule": "LLM-Pool beats Baseline-Universe on C2C Sharpe at 5 bps",
        "fail_rule": "LLM-Pool gross C2C Sharpe is more than 0.10 below Baseline-Pool gross C2C Sharpe",
    }
    try:
        llm_5 = float(c2c_5.loc[c2c_5["run"] == "LLM-Pool v1", "sharpe"].iloc[0])
        universe_5 = float(c2c_5.loc[c2c_5["run"] == "Baseline-Universe", "sharpe"].iloc[0])
        llm_gross = float(gross.loc[gross["run"] == "LLM-Pool v1", "sharpe"].iloc[0])
        pool_gross = float(gross.loc[gross["run"] == "Baseline-Pool", "sharpe"].iloc[0])
        status.update(
            {
                "llm_pool_5bps_sharpe": llm_5,
                "baseline_universe_5bps_sharpe": universe_5,
                "llm_pool_gross_sharpe": llm_gross,
                "baseline_pool_gross_sharpe": pool_gross,
                "is_win": bool(llm_5 > universe_5),
                "is_stop_failure": bool(llm_gross < pool_gross - 0.10),
            }
        )
    except (IndexError, KeyError):
        status.update({"is_win": False, "is_stop_failure": False, "error": "Missing comparison rows"})
    return status


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def copy_like_correlation_summary(corr_df: pd.DataFrame) -> dict[str, Any]:
    copy_like = corr_df[corr_df["score_trail5_spearman"].abs() > SCORE_COPY_CORR_THRESHOLD]
    sample = [
        {key: _json_safe_value(value) for key, value in row.items()}
        for row in copy_like.head(10).to_dict(orient="records")
    ]
    return {
        "copy_like_correlation_periods": int(len(copy_like)),
        "copy_like_correlation_sample": sample,
    }


def save_score_diagnostics(signal_checks: pd.DataFrame, output_dir: Path, fail_on_copy: bool = True) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    llm_rows = signal_checks[signal_checks["score_provider"] == "llm"].copy()
    if llm_rows.empty:
        raise RuntimeError("No LLM signal rows available for diagnostics")

    corr_rows = []
    for (date, side), group in llm_rows.groupby(["date", "pool_side"]):
        if group["llm_score"].nunique() <= 1 or group["trail_ret"].nunique() <= 1:
            corr = np.nan
        else:
            corr = stats.spearmanr(group["llm_score"], group["trail_ret"]).correlation
        corr_rows.append({"date": date, "pool_side": side, "score_trail5_spearman": corr})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(output_dir / "score_trail5_correlations.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, side in zip(axes, ("LONG", "SHORT")):
        values = llm_rows.loc[llm_rows["pool_side"] == side, "llm_score"].dropna()
        ax.hist(values, bins=np.arange(-0.5, 105.5, 5), color="#4c78a8" if side == "LONG" else "#f58518", alpha=0.85)
        ax.set_title(f"{side} LLM Scores")
        ax.set_xlabel("Score")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "llm_score_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if fail_on_copy:
        offenders = corr_df[corr_df["score_trail5_spearman"].abs() > SCORE_COPY_CORR_THRESHOLD]
        if not offenders.empty:
            sample = offenders.head(5).to_dict(orient="records")
            raise RuntimeError(f"LLM score appears to copy Trail5 rank on sample dates: {sample}")
    return corr_df


def load_spy_df() -> pd.DataFrame | None:
    if not SPY_DATA_PATH.exists():
        print(f"WARNING: SPY regime file not found: {SPY_DATA_PATH}; using unknown regime")
        return None
    return pd.read_csv(SPY_DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def load_experiment_inputs(config: rb.BenchmarkConfig) -> tuple[dict[str, pd.DataFrame], rb.MembershipLookup, rb.MarketFrames, FeatureFrames, pd.DataFrame | None]:
    print("Loading Russell OHLCV histories...")
    histories = load_russell_ohlcv_histories()
    print(f"   Loaded {len(histories)} ticker histories")
    print(f"Loading membership file: {config.membership_path}")
    membership_lookup = rb.load_membership_lookup(config.membership_path, end_date=config.end)
    market_frames = rb.build_market_frames(histories)
    feature_frames = build_feature_frames(histories)
    spy_df = load_spy_df()
    return histories, membership_lookup, market_frames, feature_frames, spy_df


def baseline_universe_bundle(
    market_frames: rb.MarketFrames,
    membership_lookup: rb.MembershipLookup,
    config: rb.BenchmarkConfig,
) -> ExperimentBundle:
    weekly_returns, signal_checks, selections = rb.run_weekly_strategy(market_frames, membership_lookup, config)
    return build_bundle(weekly_returns, signal_checks, selections, config)


def baseline_pool_bundle(
    market_frames: rb.MarketFrames,
    feature_frames: FeatureFrames,
    membership_lookup: rb.MembershipLookup,
    spy_df: pd.DataFrame | None,
    config: rb.BenchmarkConfig,
    pool_per_side: int,
    positions_per_side: int,
) -> tuple[ExperimentBundle, list[CandidatePool]]:
    weekly, checks, selections, pools = run_pool_strategy(
        market_frames=market_frames,
        feature_frames=feature_frames,
        membership_lookup=membership_lookup,
        spy_df=spy_df,
        config=config,
        pool_per_side=pool_per_side,
        positions_per_side=positions_per_side,
        score_provider="trail5",
    )
    return build_bundle(weekly, checks, selections, config), pools


async def collect_llm_scores_for_pools(
    pools: list[CandidatePool],
    cache_path: Path,
    model: str,
    temperature: float,
    rate_limit: float,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> dict[str, dict[str, Any]]:
    cache = load_cache(cache_path)
    prompt_version = prompt_version_for_style(prompt_style)
    cache["metadata"] = {
        "schema_version": 1,
        "artifact_type": "llm_reversal_signal_v1_scores",
        "prompt_version": prompt_version,
        "prompt_style": prompt_style,
        "model": model,
        "pool_count": len(pools),
    }
    async with httpx.AsyncClient(timeout=technical_llm.PORTFOLIO_REQUEST_TIMEOUT) as client:
        for idx, pool in enumerate(pools, start=1):
            date_key = str(pool.signal_date.date())
            if date_key in cache.setdefault("signals", {}):
                try:
                    cached = _entry_from_cache(cache["signals"][date_key], pool, model, prompt_version, prompt_style)
                    if cached is not None:
                        print(f"  [{idx}/{len(pools)}] {date_key} cached")
                        continue
                except ValueError as exc:
                    print(f"  [{idx}/{len(pools)}] {date_key} invalid cache; recalling LLM ({exc})")
                    _record_invalid_cache_entry(
                        cache,
                        date_key,
                        exc,
                        prompt_version,
                        prompt_style,
                        model,
                        pool.universe_hash,
                    )
                    save_cache(cache, cache_path)
            print(f"  [{idx}/{len(pools)}] {date_key} calling LLM")
            await ensure_llm_score_for_pool(client, pool, cache, cache_path, model, temperature, prompt_style)
            if idx < len(pools):
                await asyncio.sleep(rate_limit)
    return cache["signals"]


def run_baseline_phase(
    config: rb.BenchmarkConfig,
    output_dir: Path,
    pool_per_side: int,
    positions_per_side: int,
) -> dict[str, ExperimentBundle]:
    _, membership_lookup, market_frames, feature_frames, spy_df = load_experiment_inputs(config)
    bundles: dict[str, ExperimentBundle] = {}
    print("\nRunning Baseline-Universe...")
    bundles["Baseline-Universe"] = baseline_universe_bundle(market_frames, membership_lookup, config)
    save_bundle(bundles["Baseline-Universe"], output_dir / "baseline_universe")

    print("\nRunning Baseline-Pool...")
    bundles["Baseline-Pool"], _ = baseline_pool_bundle(
        market_frames,
        feature_frames,
        membership_lookup,
        spy_df,
        config,
        pool_per_side,
        positions_per_side,
    )
    save_bundle(bundles["Baseline-Pool"], output_dir / "baseline_pool")
    comparison = build_comparison_scorecard(bundles)
    comparison.to_csv(output_dir / "comparison_scorecard.csv", index=False)
    return bundles


async def run_sanity_phase(
    config: rb.BenchmarkConfig,
    output_dir: Path,
    pool_per_side: int,
    positions_per_side: int,
    sanity_date: str,
    model: str,
    temperature: float,
    cache_path: Path,
    prompt_style: str,
) -> None:
    _, membership_lookup, market_frames, feature_frames, spy_df = load_experiment_inputs(config)
    signal_date = pd.Timestamp(sanity_date)
    pool = build_candidate_pool(market_frames, feature_frames, membership_lookup, spy_df, signal_date, pool_per_side, config)
    if pool is None:
        raise RuntimeError(f"Could not build sanity pool for {sanity_date}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sanity_prompt.txt").write_text(format_user_prompt(pool))
    cache = load_cache(cache_path)
    async with httpx.AsyncClient(timeout=technical_llm.PORTFOLIO_REQUEST_TIMEOUT) as client:
        scores = await ensure_llm_score_for_pool(client, pool, cache, cache_path, model, temperature, prompt_style)
    llm_scores = {str(pool.signal_date.date()): scores}
    weekly, checks, selections, _ = run_pool_strategy(
        market_frames=market_frames,
        feature_frames=feature_frames,
        membership_lookup=membership_lookup,
        spy_df=spy_df,
        config=rb.BenchmarkConfig(
            start=signal_date,
            end=pool.exit_date,
            output_dir=config.output_dir,
            membership_path=config.membership_path,
            cost_bps_levels=config.cost_bps_levels,
            positions_per_side=None,
        ),
        pool_per_side=pool_per_side,
        positions_per_side=positions_per_side,
        score_provider="llm",
        llm_scores_by_date=llm_scores,
        llm_prompt_style=prompt_style,
    )
    sanity_bundle = build_bundle(weekly, checks, selections, config)
    save_bundle(sanity_bundle, output_dir)
    save_score_diagnostics(checks, output_dir, fail_on_copy=True)


async def run_full_phase(
    config: rb.BenchmarkConfig,
    output_dir: Path,
    pool_per_side: int,
    positions_per_side: int,
    model: str,
    temperature: float,
    cache_path: Path,
    rate_limit: float,
    prompt_style: str,
) -> dict[str, ExperimentBundle]:
    _, membership_lookup, market_frames, feature_frames, spy_df = load_experiment_inputs(config)
    bundles: dict[str, ExperimentBundle] = {}

    print("\nRunning Baseline-Universe...")
    bundles["Baseline-Universe"] = baseline_universe_bundle(market_frames, membership_lookup, config)
    save_bundle(bundles["Baseline-Universe"], output_dir / "baseline_universe")

    print("\nRunning Baseline-Pool...")
    bundles["Baseline-Pool"], pools = baseline_pool_bundle(
        market_frames,
        feature_frames,
        membership_lookup,
        spy_df,
        config,
        pool_per_side,
        positions_per_side,
    )
    save_bundle(bundles["Baseline-Pool"], output_dir / "baseline_pool")

    print(f"\nCollecting LLM-Pool scores ({prompt_style})...")
    llm_scores = await collect_llm_scores_for_pools(pools, cache_path, model, temperature, rate_limit, prompt_style)

    print("\nRunning LLM-Pool v1...")
    weekly, checks, selections, _ = run_pool_strategy(
        market_frames=market_frames,
        feature_frames=feature_frames,
        membership_lookup=membership_lookup,
        spy_df=spy_df,
        config=config,
        pool_per_side=pool_per_side,
        positions_per_side=positions_per_side,
        score_provider="llm",
        llm_scores_by_date=llm_scores,
        llm_prompt_style=prompt_style,
    )
    bundles["LLM-Pool v1"] = build_bundle(weekly, checks, selections, config)
    save_bundle(bundles["LLM-Pool v1"], output_dir / "llm_pool_v1")
    corr_df = save_score_diagnostics(checks, output_dir / "llm_pool_v1", fail_on_copy=False)

    comparison = build_comparison_scorecard(bundles)
    comparison.to_csv(output_dir / "comparison_scorecard.csv", index=False)
    success = evaluate_success(comparison)
    success.update(copy_like_correlation_summary(corr_df))
    with (output_dir / "success_evaluation.json").open("w") as handle:
        json.dump(success, handle, indent=2, sort_keys=True)
    return bundles


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM reversal signal v1 experiment.")
    parser.add_argument("--phase", choices=("baseline", "sanity", "full"), default="baseline")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--membership-path", type=Path, default=rb.DEFAULT_MEMBERSHIP_PATH)
    parser.add_argument("--cost-bps-levels", type=rb.parse_cost_bps_levels, default=DEFAULT_COST_BPS_LEVELS)
    parser.add_argument("--pool-per-side", type=int, default=DEFAULT_POOL_PER_SIDE)
    parser.add_argument("--positions-per-side", type=int, default=DEFAULT_POSITIONS_PER_SIDE)
    parser.add_argument("--sanity-date", default=None)
    parser.add_argument("--model", default=technical_llm.DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=technical_llm.DEFAULT_TEMPERATURE)
    parser.add_argument("--rate-limit", type=float, default=2.0)
    parser.add_argument(
        "--prompt-style",
        choices=(PROMPT_STYLE_V1_SCORE, PROMPT_STYLE_RAW_CONFIDENCE, PROMPT_STYLE_RANK_CONFIDENCE),
        default=DEFAULT_PROMPT_STYLE,
        help=(
            "LLM scoring prompt semantics. raw_confidence keeps the score schema but "
            "defines score as edge confidence; rank_confidence asks for side-relative "
            "rank plus confidence and converts them to an old-style combined score."
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
        cadence_mode="weekly_friday",
        signal_lookback_days=5,
        rebalance_days=5,
        holding_days=5,
        positions_per_side=None,
    )


def cache_path_for_args(args: argparse.Namespace) -> Path:
    start = str(pd.Timestamp(args.start).date())
    end = str(pd.Timestamp(args.end).date())
    model = str(args.model).replace("/", "-")
    if args.prompt_style == PROMPT_STYLE_V1_SCORE:
        name = f"llm_reversal_signal_v1_{start}_{end}_pool{args.pool_per_side}_mdl-{model}.json"
    elif args.prompt_style == PROMPT_STYLE_RAW_CONFIDENCE:
        name = f"llm_reversal_raw_confidence_v1_{start}_{end}_pool{args.pool_per_side}_mdl-{model}.json"
    else:
        name = f"llm_reversal_rank_confidence_v1_{start}_{end}_pool{args.pool_per_side}_mdl-{model}.json"
    return args.cache_dir / name


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.pool_per_side <= 0 or args.positions_per_side <= 0:
        raise ValueError("--pool-per-side and --positions-per-side must be positive")
    if args.positions_per_side > args.pool_per_side:
        raise ValueError("--positions-per-side cannot exceed --pool-per-side")
    config = build_config(args)
    output_dir = Path(args.output_dir)
    cache_path = cache_path_for_args(args)

    if args.phase == "baseline":
        run_baseline_phase(config, output_dir, args.pool_per_side, args.positions_per_side)
    elif args.phase == "sanity":
        if not args.sanity_date:
            raise ValueError("--phase sanity requires --sanity-date YYYY-MM-DD")
        asyncio.run(
            run_sanity_phase(
                config=config,
                output_dir=output_dir / "sanity",
                pool_per_side=args.pool_per_side,
                positions_per_side=args.positions_per_side,
                sanity_date=args.sanity_date,
                model=args.model,
                temperature=args.temperature,
                cache_path=cache_path,
                prompt_style=args.prompt_style,
            )
        )
    else:
        asyncio.run(
            run_full_phase(
                config=config,
                output_dir=output_dir,
                pool_per_side=args.pool_per_side,
                positions_per_side=args.positions_per_side,
                model=args.model,
                temperature=args.temperature,
                cache_path=cache_path,
                rate_limit=args.rate_limit,
                prompt_style=args.prompt_style,
            )
        )


if __name__ == "__main__":
    main()
