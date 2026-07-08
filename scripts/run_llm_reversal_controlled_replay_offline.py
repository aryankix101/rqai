"""Offline controlled replay audit for LLM reversal scorer artifacts.

This script intentionally does not import the LLM runner or any DeepSeek/httpx
path. It freezes the candidate pool once, audits saved scorer artifacts against
that pool, and writes explicit coverage outputs instead of filling missing LLM
scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import run_backtest_v3 as rb
from russell_data_loader import load_russell_ohlcv_histories


DEFAULT_START = "2009-01-01"
DEFAULT_END = "2014-01-01"
DEFAULT_OUTPUT_DIR = Path("backtest_results/llm_reversal_controlled_replay_2009_2014_offline")
DEFAULT_RAW_CONFIDENCE_ARTIFACT = (
    Path("backtest_results")
    / "llm_reversal_raw_confidence_v1_2009_2014_pool60_pos30"
    / "llm_pool_v1"
    / "signal_checks.csv"
)
DEFAULT_RANK_CONFIDENCE_ARTIFACT = (
    Path("backtest_results")
    / "llm_reversal_rank_confidence_v1_2009_2014_pool60_pos30"
    / "llm_pool_v1"
    / "signal_checks.csv"
)
DEFAULT_DIRECT_SCORE_CACHE = (
    Path("backtest_results")
    / "llm_reversal_signal_v1_cache"
    / "llm_reversal_signal_v1_2009-01-01_2014-01-01_pool60_mdl-deepseek-reasoner.json"
)
KNOWN_INCOMPATIBLE_DIRECT_SCORE_CACHE = (
    Path("backtest_results")
    / "llm_reversal_signal_v1_cache"
    / "llm_reversal_signal_v1_2009-01-01_2014-01-01_pool150_mdl-deepseek-reasoner.json"
)
SCORER_DETERMINISTIC = "deterministic_mr"
SCORER_RAW_CONFIDENCE = "v1_raw_confidence"
SCORER_RANK_CONFIDENCE = "v1_rank_confidence"
SCORER_DIRECT_SCORE = "v1_direct_score"
DIRECT_SCORE_PROMPT_STYLE = "v1_score"
DIRECT_SCORE_PROMPT_VERSION = "llm_reversal_signal_v1"
REQUIRED_SCORERS = (
    SCORER_DETERMINISTIC,
    SCORER_RAW_CONFIDENCE,
    SCORER_RANK_CONFIDENCE,
    SCORER_DIRECT_SCORE,
)


@dataclass(frozen=True)
class CandidateReplay:
    candidate_grid: pd.DataFrame
    pool_audit: pd.DataFrame


@dataclass(frozen=True)
class ScorerPerformance:
    weekly_returns: pd.DataFrame
    portfolio_selections: pd.DataFrame
    headline_metrics: pd.DataFrame
    yearly_returns: pd.DataFrame
    strategy_summary: pd.DataFrame


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return None if not np.isfinite(numeric) else numeric
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def candidate_pool_hash(rows: pd.DataFrame) -> str:
    cols = ["rebalance_date", "ticker", "pool_side"]
    records = (
        rows.loc[:, cols]
        .assign(rebalance_date=lambda df: pd.to_datetime(df["rebalance_date"]).dt.strftime("%Y-%m-%d"))
        .sort_values(cols)
        .to_dict("records")
    )
    return stable_hash(records)


def feature_payload_hash(rows: pd.DataFrame, feature_columns: list[str]) -> str:
    cols = ["rebalance_date", "ticker", "pool_side", *feature_columns]
    normalized = rows.loc[:, cols].copy()
    normalized["rebalance_date"] = pd.to_datetime(normalized["rebalance_date"]).dt.strftime("%Y-%m-%d")
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper().str.strip()
    normalized["pool_side"] = normalized["pool_side"].astype(str).str.upper().str.strip()
    for col in feature_columns:
        normalized[col] = normalized[col].map(_normalized_hash_value)
    records = normalized.sort_values(["rebalance_date", "pool_side", "ticker"]).to_dict("records")
    return stable_hash(records)


def _normalized_hash_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return round(numeric, 10) if np.isfinite(numeric) else None


def _pct_return(current: float, prior: float) -> float:
    if pd.isna(current) or pd.isna(prior) or current <= 0 or prior <= 0:
        return np.nan
    return float(current / prior - 1.0)


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
    return float(corr) if np.isfinite(corr) else 0.0


def _adx_last(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
    frame = pd.DataFrame({"high": high, "low": low, "close": close}).dropna()
    if len(frame) < window * 2 + 1:
        return np.nan
    high_s = frame["high"]
    low_s = frame["low"]
    close_s = frame["close"]
    up_move = high_s.diff()
    down_move = -low_s.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=frame.index)
    prev_close = close_s.shift(1)
    true_range = pd.concat(
        [high_s - low_s, (high_s - prev_close).abs(), (low_s - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window, min_periods=window).mean()
    plus_di = 100.0 * plus_dm.rolling(window, min_periods=window).mean() / atr
    minus_di = 100.0 * minus_dm.rolling(window, min_periods=window).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window, min_periods=window).mean().iloc[-1]
    return float(adx) if np.isfinite(adx) else np.nan


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


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _feature_frames(histories: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    high_cols: dict[str, pd.Series] = {}
    low_cols: dict[str, pd.Series] = {}
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
    return pd.DataFrame(high_cols).sort_index(), pd.DataFrame(low_cols).sort_index()


def _prompt_features(
    market: rb.MarketFrames,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    ticker: str,
    signal_date: pd.Timestamp,
    trail_return: float,
    mrl: float,
    mrs: float,
) -> dict[str, float | str]:
    close = market.close_df.loc[:signal_date, ticker].dropna()
    high = high_df.loc[:signal_date, ticker].dropna()
    low = low_df.loc[:signal_date, ticker].dropna()
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
        "feature_MRL": float(mrl),
        "feature_MRS": float(mrs),
        "feature_Trail5": float(trail_return * 100.0),
        "feature_MRSig": float(-trail_return * 100.0),
        "feature_RSI5": _rsi_last(close, 5),
        "feature_Rng55": _safe_float(rng55),
        "feature_Vel/V": _safe_float(vel_v),
        "feature_%MA100": _safe_float(pct_ma100),
        "feature_52WH": _safe_float(pct_52wh),
        "feature_Mom3m": _safe_float(mom3m),
        "feature_AutoC": _autocorr_lag(close, 5),
        "feature_ADX": _adx_last(high, low, close, 14),
        "feature_VolPctl": _vol_percentile_504(close),
    }


def build_candidate_replay(
    histories: dict[str, pd.DataFrame],
    membership_lookup: rb.MembershipLookup,
    config: rb.BenchmarkConfig,
    pool_per_side: int,
) -> CandidateReplay:
    market = rb.build_market_frames(histories)
    high_df, low_df = _feature_frames(histories)
    close_df = market.close_df
    unadj_close_df = market.unadj_close_df
    adv20_df = market.dollar_volume_df.rolling(20, min_periods=15).mean()
    trading_dates = close_df.index
    fridays = trading_dates[trading_dates.dayofweek == 4]
    next_trade_map = rb._next_trading_day_map(trading_dates)

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for idx in range(1, len(fridays) - 1):
        lookback_date = pd.Timestamp(fridays[idx - 1])
        signal_date = pd.Timestamp(fridays[idx])
        exit_date = pd.Timestamp(fridays[idx + 1])
        if signal_date < config.start:
            continue
        if signal_date > config.end or exit_date > config.end:
            break

        membership_idx = np.searchsorted(np.asarray(membership_lookup.dates, dtype="datetime64[ns]"), np.datetime64(signal_date), side="right") - 1
        membership_date = membership_lookup.dates[membership_idx] if membership_idx >= 0 else pd.NaT
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
            audit_rows.append(
                {
                    "rebalance_date": signal_date.date().isoformat(),
                    "audit_status": "skipped_insufficient_candidates",
                    "membership_date": "" if pd.isna(membership_date) else pd.Timestamp(membership_date).date().isoformat(),
                    "membership_universe_size": len(r1000_universe),
                    "eligible_universe_size": len(valid),
                    "expected_per_side": pool_per_side,
                    "long_count": 0,
                    "short_count": 0,
                    "long_pool_hash": "",
                    "short_pool_hash": "",
                    "candidate_pool_hash": "",
                    "feature_payload_hash": "",
                }
            )
            continue

        trail_arr = np.asarray(trails, dtype=float)
        signal_arr = -trail_arr
        long_order = np.argsort(trail_arr, kind="mergesort")[:pool_per_side]
        short_order = np.argsort(-trail_arr, kind="mergesort")[:pool_per_side]
        long_tickers = [valid[pos] for pos in long_order]
        short_tickers = [valid[pos] for pos in short_order]
        if set(long_tickers) & set(short_tickers):
            raise ValueError(f"Candidate pool overlap on {signal_date.date()}")

        mrl_scores = _percentile_scores(signal_arr, high_is_best=True)
        mrs_scores = 100.0 - mrl_scores
        fwd_c2c = rb._forward_c2c_return(close_df, signal_date, exit_date, long_tickers + short_tickers)
        fwd_o2o = rb._forward_o2o_return(
            market.open_df,
            next_trade_map.get(signal_date),
            next_trade_map.get(exit_date),
            long_tickers + short_tickers,
        )
        fwd_by_ticker = {
            ticker: {"fwd_c2c": float(fwd_c2c[pos]), "fwd_o2o": float(fwd_o2o[pos])}
            for pos, ticker in enumerate(long_tickers + short_tickers)
        }
        valid_pos = {ticker: pos for pos, ticker in enumerate(valid)}
        date_rows: list[dict[str, Any]] = []
        for side, tickers in (("LONG", long_tickers), ("SHORT", short_tickers)):
            for side_rank, ticker in enumerate(tickers, start=1):
                pos = valid_pos[ticker]
                trail = float(trail_arr[pos])
                feature_payload = _prompt_features(
                    market,
                    high_df,
                    low_df,
                    ticker,
                    signal_date,
                    trail,
                    float(mrl_scores[pos]),
                    float(mrs_scores[pos]),
                )
                signed_signal = float(signal_arr[pos]) if side == "LONG" else -float(trail)
                date_rows.append(
                    {
                        "rebalance_date": signal_date.date().isoformat(),
                        "lookback_date": lookback_date.date().isoformat(),
                        "entry_date": signal_date.date().isoformat(),
                        "exit_date": exit_date.date().isoformat(),
                        "entry_open_date": _date_or_empty(next_trade_map.get(signal_date)),
                        "exit_open_date": _date_or_empty(next_trade_map.get(exit_date)),
                        "membership_date": "" if pd.isna(membership_date) else pd.Timestamp(membership_date).date().isoformat(),
                        "membership_universe_size": len(r1000_universe),
                        "eligible_universe_size": len(valid),
                        "ticker": ticker,
                        "pool_side": side,
                        "side_rank_by_trail": side_rank,
                        "trail_ret": trail,
                        "deterministic_signal": signed_signal,
                        "fwd_c2c": fwd_by_ticker[ticker]["fwd_c2c"],
                        "fwd_o2o": fwd_by_ticker[ticker]["fwd_o2o"],
                        **feature_payload,
                    }
                )

        date_df = pd.DataFrame(date_rows)
        feature_cols = [col for col in date_df.columns if col.startswith("feature_")]
        long_df = date_df[date_df["pool_side"] == "LONG"]
        short_df = date_df[date_df["pool_side"] == "SHORT"]
        rows.extend(date_rows)
        audit_rows.append(
            {
                "rebalance_date": signal_date.date().isoformat(),
                "audit_status": "ok",
                "membership_date": "" if pd.isna(membership_date) else pd.Timestamp(membership_date).date().isoformat(),
                "membership_universe_size": len(r1000_universe),
                "eligible_universe_size": len(valid),
                "expected_per_side": pool_per_side,
                "long_count": len(long_tickers),
                "short_count": len(short_tickers),
                "long_pool_hash": candidate_pool_hash(long_df),
                "short_pool_hash": candidate_pool_hash(short_df),
                "candidate_pool_hash": candidate_pool_hash(date_df),
                "feature_payload_hash": feature_payload_hash(date_df, feature_cols),
            }
        )

    candidate_grid = pd.DataFrame(rows)
    pool_audit = pd.DataFrame(audit_rows)
    if candidate_grid.empty:
        raise RuntimeError("No candidate rows produced. Check date range, membership path, and filters.")
    return CandidateReplay(candidate_grid=candidate_grid, pool_audit=pool_audit)


def _date_or_empty(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def deterministic_scores(candidate_grid: pd.DataFrame) -> pd.DataFrame:
    rows = candidate_grid[["rebalance_date", "ticker", "pool_side", "deterministic_signal"]].copy()
    rows = rows.rename(columns={"deterministic_signal": score_column(SCORER_DETERMINISTIC)})
    return rows


def score_column(scorer: str) -> str:
    return scorer if scorer.endswith("_score") else f"{scorer}_score"


def _signed_scores_for_selection(rows: pd.DataFrame, scorer: str) -> pd.Series:
    col = score_column(scorer)
    if scorer == SCORER_DETERMINISTIC:
        return pd.to_numeric(rows[col], errors="coerce")
    raw = pd.to_numeric(rows[col], errors="coerce")
    side = rows["pool_side"].astype(str).str.upper().str.strip()
    return raw.where(side == "LONG", -raw)


def _rank_positions(
    rows: pd.DataFrame,
    signal: pd.Series,
    side: str,
    positions_per_side: int,
) -> pd.Index:
    side_mask = rows["pool_side"].astype(str).str.upper().str.strip() == side
    side_rows = rows.loc[side_mask].copy()
    side_rows["_signal"] = signal.loc[side_rows.index]
    if side_rows["_signal"].isna().any():
        missing = side_rows.loc[side_rows["_signal"].isna(), "ticker"].astype(str).head(10).tolist()
        raise ValueError(f"Missing {side} scores for {missing}")
    ascending = side == "SHORT"
    selected = side_rows.sort_values(["_signal", "ticker"], ascending=[ascending, True]).head(positions_per_side)
    if len(selected) != positions_per_side:
        raise ValueError(f"Expected {positions_per_side} {side} positions, got {len(selected)}")
    return selected.index


def build_performance_for_scorer(
    candidate_grid: pd.DataFrame,
    score_matrix: pd.DataFrame,
    scorer: str,
    positions_per_side: int,
    cost_bps_levels: tuple[float, ...],
    periods_per_year: float = 52.0,
) -> ScorerPerformance:
    key_cols = ["rebalance_date", "ticker", "pool_side"]
    col = score_column(scorer)
    if col not in score_matrix.columns:
        raise ValueError(f"Score matrix missing {col}")
    merged = candidate_grid.merge(score_matrix[key_cols + [col]], on=key_cols, how="left")
    merged["rebalance_date"] = pd.to_datetime(merged["rebalance_date"]).dt.strftime("%Y-%m-%d")
    merged["ticker"] = merged["ticker"].astype(str).str.upper().str.strip()
    merged["pool_side"] = merged["pool_side"].astype(str).str.upper().str.strip()

    results: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    prev_long_set: set[str] = set()
    prev_short_set: set[str] = set()

    for date, date_rows in merged.groupby("rebalance_date", sort=True):
        date_rows = date_rows.copy()
        signal = _signed_scores_for_selection(date_rows, scorer)
        long_idx = _rank_positions(date_rows, signal, "LONG", positions_per_side)
        short_idx = _rank_positions(date_rows, signal, "SHORT", positions_per_side)
        if set(long_idx) & set(short_idx):
            raise ValueError(f"{scorer} selected overlapping long/short rows on {date}")

        long_rows = date_rows.loc[long_idx]
        short_rows = date_rows.loc[short_idx]
        long_set = set(long_rows["ticker"])
        short_set = set(short_rows["ticker"])
        row = {
            "date": pd.Timestamp(date),
            "n_universe": int(date_rows["eligible_universe_size"].max())
            if "eligible_universe_size" in date_rows.columns
            else int(len(date_rows)),
            "n_pool": int(len(date_rows)),
            "n_long": int(len(long_rows)),
            "n_short": int(len(short_rows)),
            "long_ret_c2c": float(pd.to_numeric(long_rows["fwd_c2c"], errors="coerce").mean()),
            "short_ret_c2c": float(pd.to_numeric(short_rows["fwd_c2c"], errors="coerce").mean()),
            "long_ret_o2o": float(pd.to_numeric(long_rows["fwd_o2o"], errors="coerce").mean()),
            "short_ret_o2o": float(pd.to_numeric(short_rows["fwd_o2o"], errors="coerce").mean()),
            "turnover_long": rb.calc_turnover(prev_long_set, long_set),
            "turnover_short": rb.calc_turnover(prev_short_set, short_set),
            "ic_c2c": _spearman_np(signal.to_numpy(dtype=float), pd.to_numeric(date_rows["fwd_c2c"], errors="coerce").to_numpy(dtype=float)),
            "ic_o2o": _spearman_np(signal.to_numpy(dtype=float), pd.to_numeric(date_rows["fwd_o2o"], errors="coerce").to_numpy(dtype=float)),
        }
        _, deciles = rb._rank_signal(signal.to_numpy(dtype=float))
        fwd_c2c = pd.to_numeric(date_rows["fwd_c2c"], errors="coerce").to_numpy(dtype=float)
        fwd_o2o = pd.to_numeric(date_rows["fwd_o2o"], errors="coerce").to_numpy(dtype=float)
        for decile in range(1, 11):
            decile_mask = deciles == decile
            row[f"dec_{decile}_c2c"] = rb.safe_mean(fwd_c2c, decile_mask)
            row[f"dec_{decile}_o2o"] = rb.safe_mean(fwd_o2o, decile_mask)
        results.append(row)

        prev_long_set = long_set
        prev_short_set = short_set
        for side, selected_rows, reverse in (("LONG", long_rows, True), ("SHORT", short_rows, False)):
            ordered = selected_rows.assign(_signal=signal.loc[selected_rows.index]).sort_values(
                ["_signal", "ticker"],
                ascending=[not reverse, True],
            )
            side_weight = 1.0 / len(ordered) if len(ordered) else np.nan
            for rank, (_, selected) in enumerate(ordered.iterrows(), start=1):
                pos = int(date_rows.index.get_loc(selected.name))
                selections.append(
                    {
                        "scorer": scorer,
                        "rebalance_date": date,
                        "signal_date": date,
                        "entry_date": selected.get("entry_date", date),
                        "exit_date": selected.get("exit_date", ""),
                        "ticker": selected["ticker"],
                        "position_type": side,
                        "rank": rank,
                        "score": float(signal.loc[selected.name]),
                        "raw_score": _safe_float(selected[col]),
                        "trail_return_pct": _safe_float(selected.get("trail_ret")) * 100.0,
                        "realized_return_pct": _safe_float(selected.get("fwd_c2c")) * 100.0,
                        "realized_c2c_return_pct": _safe_float(selected.get("fwd_c2c")) * 100.0,
                        "realized_o2o_return_pct": _safe_float(selected.get("fwd_o2o")) * 100.0,
                        "weight": side_weight,
                        "decile": int(deciles[pos]),
                    }
                )

    weekly_returns = pd.DataFrame(results)
    if weekly_returns.empty:
        raise RuntimeError(f"No performance rows produced for {scorer}")
    weekly_returns["date"] = pd.to_datetime(weekly_returns["date"])
    weekly_returns = weekly_returns.set_index("date").sort_index()
    weekly_with_costs = rb.add_cost_overlays(weekly_returns, cost_bps_levels)
    return ScorerPerformance(
        weekly_returns=weekly_with_costs,
        portfolio_selections=pd.DataFrame(selections),
        headline_metrics=rb.build_headline_metrics(weekly_with_costs, periods_per_year=periods_per_year),
        yearly_returns=rb.build_yearly_returns(weekly_with_costs),
        strategy_summary=rb.build_strategy_summary(weekly_with_costs, periods_per_year=periods_per_year),
    )


def _spearman_np(signal: np.ndarray, returns: np.ndarray) -> float:
    valid = (~np.isnan(signal)) & (~np.isnan(returns))
    if valid.sum() <= 20 or len(np.unique(signal[valid])) <= 1:
        return np.nan
    try:
        from scipy import stats

        corr = stats.spearmanr(signal[valid], returns[valid])[0]
    except Exception:
        corr = np.nan
    return float(corr) if np.isfinite(corr) else np.nan


def build_controlled_performance_outputs(
    candidate_grid: pd.DataFrame,
    score_matrix: pd.DataFrame,
    coverage: dict[str, Any],
    positions_per_side: int,
    cost_bps_levels: tuple[float, ...],
) -> dict[str, ScorerPerformance]:
    valid_scorers = [
        item["scorer"]
        for item in coverage.get("scorers", [])
        if item.get("scorer") in REQUIRED_SCORERS and item.get("status") == "valid"
    ]
    outputs: dict[str, ScorerPerformance] = {}
    for scorer in valid_scorers:
        outputs[scorer] = build_performance_for_scorer(
            candidate_grid=candidate_grid,
            score_matrix=score_matrix,
            scorer=scorer,
            positions_per_side=positions_per_side,
            cost_bps_levels=cost_bps_levels,
        )
    return outputs


def save_controlled_performance_outputs(
    output_dir: Path,
    outputs: dict[str, ScorerPerformance],
    replay_status: str,
) -> pd.DataFrame:
    scorecard_rows: list[pd.DataFrame] = []
    for scorer, perf in outputs.items():
        scorer_dir = output_dir / "performance" / scorer
        scorer_dir.mkdir(parents=True, exist_ok=True)
        perf.weekly_returns.to_csv(scorer_dir / "weekly_returns.csv")
        perf.portfolio_selections.to_csv(scorer_dir / "portfolio_selections.csv", index=False)
        perf.headline_metrics.to_csv(scorer_dir / "headline_metrics.csv", index=False)
        perf.yearly_returns.to_csv(scorer_dir / "yearly_returns.csv")
        perf.strategy_summary.to_csv(scorer_dir / "strategy_summary.csv", index=False)
        if not perf.strategy_summary.empty:
            scorecard_rows.append(
                perf.strategy_summary.assign(
                    scorer=scorer,
                    replay_status=replay_status,
                    all_required_scorers_valid=replay_status == "VALID_CONTROLLED_REPLAY",
                )
            )
    scorecard = pd.concat(scorecard_rows, ignore_index=True) if scorecard_rows else pd.DataFrame()
    if not scorecard.empty:
        leading = ["scorer"]
        scorecard = scorecard[leading + [col for col in scorecard.columns if col not in leading]]
    scorecard.to_csv(output_dir / "performance_scorecard.csv", index=False)
    return scorecard


def load_signal_checks_artifact(path: Path, scorer: str, candidate_grid: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(), {
            "scorer": scorer,
            "status": "missing_offline_score",
            "path": str(path),
            "reason": "artifact_not_found",
        }
    artifact = pd.read_csv(path)
    required = {"date", "ticker", "pool_side", "llm_score"}
    missing_cols = sorted(required - set(artifact.columns))
    if missing_cols:
        return pd.DataFrame(), {
            "scorer": scorer,
            "status": "incompatible_cache_or_artifact",
            "path": str(path),
            "reason": f"missing columns: {missing_cols}",
        }
    normalized = artifact.copy()
    normalized["rebalance_date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper().str.strip()
    normalized["pool_side"] = normalized["pool_side"].astype(str).str.upper().str.strip()
    normalized = normalized[["rebalance_date", "ticker", "pool_side", "llm_score"]].rename(
        columns={"llm_score": score_column(scorer)}
    )
    audit = validate_artifact_grid(normalized, scorer, candidate_grid, path)
    return normalized if audit["status"] == "valid" else pd.DataFrame(), audit


def load_direct_score_cache(
    path: Path,
    candidate_grid: pd.DataFrame,
    expected_pool_per_side: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(), {
            "scorer": SCORER_DIRECT_SCORE,
            "status": "missing_offline_score",
            "path": str(path),
            "reason": "cache_not_found",
        }
    try:
        with path.open("r") as handle:
            cache = json.load(handle)
    except json.JSONDecodeError as exc:
        return pd.DataFrame(), {
            "scorer": SCORER_DIRECT_SCORE,
            "status": "incompatible_cache_or_artifact",
            "path": str(path),
            "reason": f"invalid json: {exc}",
        }

    signals = cache.get("signals", {})
    if not isinstance(signals, dict):
        return pd.DataFrame(), {
            "scorer": SCORER_DIRECT_SCORE,
            "status": "incompatible_cache_or_artifact",
            "path": str(path),
            "reason": "cache signals must be a dict",
        }
    metadata = cache.get("metadata", {})
    if isinstance(metadata, dict):
        prompt_style = metadata.get("prompt_style")
        prompt_version = metadata.get("prompt_version")
        if prompt_style is not None and prompt_style != DIRECT_SCORE_PROMPT_STYLE:
            return pd.DataFrame(), {
                "scorer": SCORER_DIRECT_SCORE,
                "status": "incompatible_cache_or_artifact",
                "path": str(path),
                "reason": f"wrong prompt_style {prompt_style!r}; expected {DIRECT_SCORE_PROMPT_STYLE!r}",
            }
        if prompt_version is not None and prompt_version != DIRECT_SCORE_PROMPT_VERSION:
            return pd.DataFrame(), {
                "scorer": SCORER_DIRECT_SCORE,
                "status": "incompatible_cache_or_artifact",
                "path": str(path),
                "reason": f"wrong prompt_version {prompt_version!r}; expected {DIRECT_SCORE_PROMPT_VERSION!r}",
            }

    observed_counts: set[int] = set()
    rows: list[dict[str, Any]] = []
    for date_key, entry in signals.items():
        if not isinstance(entry, dict):
            continue
        prompt_style = entry.get("prompt_style")
        prompt_version = entry.get("prompt_version")
        if prompt_style is not None and prompt_style != DIRECT_SCORE_PROMPT_STYLE:
            return pd.DataFrame(), {
                "scorer": SCORER_DIRECT_SCORE,
                "status": "incompatible_cache_or_artifact",
                "path": str(path),
                "reason": f"{date_key} has wrong prompt_style {prompt_style!r}; expected {DIRECT_SCORE_PROMPT_STYLE!r}",
            }
        if prompt_version is not None and prompt_version != DIRECT_SCORE_PROMPT_VERSION:
            return pd.DataFrame(), {
                "scorer": SCORER_DIRECT_SCORE,
                "status": "incompatible_cache_or_artifact",
                "path": str(path),
                "reason": f"{date_key} has wrong prompt_version {prompt_version!r}; expected {DIRECT_SCORE_PROMPT_VERSION!r}",
            }
        if "pool_per_side" in entry:
            observed_counts.add(int(entry["pool_per_side"]))
        long_scores = entry.get("long_scores", [])
        short_scores = entry.get("short_scores", [])
        if isinstance(long_scores, list):
            observed_counts.add(len(long_scores))
        if isinstance(short_scores, list):
            observed_counts.add(len(short_scores))

        rebalance_date = str(entry.get("rebalance_date") or date_key)
        for side, score_rows in (("LONG", long_scores), ("SHORT", short_scores)):
            if not isinstance(score_rows, list):
                continue
            for score_row in score_rows:
                if not isinstance(score_row, dict):
                    continue
                ticker = str(score_row.get("ticker", "")).upper().strip()
                if not ticker:
                    continue
                rows.append(
                    {
                        "rebalance_date": pd.Timestamp(rebalance_date).date().isoformat(),
                        "ticker": ticker,
                        "pool_side": side,
                        score_column(SCORER_DIRECT_SCORE): _safe_float(score_row.get("score")),
                    }
                )

    if observed_counts and observed_counts != {expected_pool_per_side}:
        return pd.DataFrame(), {
            "scorer": SCORER_DIRECT_SCORE,
            "status": "missing_offline_score",
            "path": str(path),
            "reason": (
                "rejected incompatible cache: "
                f"pool_per_side {sorted(observed_counts)} != expected {expected_pool_per_side}"
            ),
            "rejected_cache": True,
            "signal_count": len(signals),
            "observed_pool_per_side": sorted(observed_counts),
        }

    if not rows:
        return pd.DataFrame(), {
            "scorer": SCORER_DIRECT_SCORE,
            "status": "missing_offline_score",
            "path": str(path),
            "reason": "cache_has_no_direct_scores",
            "signal_count": len(signals),
            "observed_pool_per_side": sorted(observed_counts),
        }

    scores = pd.DataFrame(rows)
    audit = validate_artifact_grid(scores, SCORER_DIRECT_SCORE, candidate_grid, path)
    audit["signal_count"] = len(signals)
    audit["observed_pool_per_side"] = sorted(observed_counts)
    return scores if audit["status"] == "valid" else pd.DataFrame(), audit


def validate_artifact_grid(
    score_rows: pd.DataFrame,
    scorer: str,
    candidate_grid: pd.DataFrame,
    path: Path | str = "",
) -> dict[str, Any]:
    key_cols = ["rebalance_date", "ticker", "pool_side"]
    expected = (
        candidate_grid[key_cols]
        .assign(
            rebalance_date=lambda df: pd.to_datetime(df["rebalance_date"]).dt.strftime("%Y-%m-%d"),
            ticker=lambda df: df["ticker"].astype(str).str.upper().str.strip(),
            pool_side=lambda df: df["pool_side"].astype(str).str.upper().str.strip(),
        )
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    actual = (
        score_rows[key_cols]
        .assign(
            rebalance_date=lambda df: pd.to_datetime(df["rebalance_date"]).dt.strftime("%Y-%m-%d"),
            ticker=lambda df: df["ticker"].astype(str).str.upper().str.strip(),
            pool_side=lambda df: df["pool_side"].astype(str).str.upper().str.strip(),
        )
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    duplicate_count = int(actual.duplicated(key_cols).sum())
    expected_index = pd.MultiIndex.from_frame(expected)
    actual_index = pd.MultiIndex.from_frame(actual)
    missing = expected_index.difference(actual_index)
    unexpected = actual_index.difference(expected_index)
    expected_hash = candidate_pool_hash(expected)
    actual_hash = candidate_pool_hash(actual) if not actual.empty else ""
    status = "valid"
    reason = ""
    if duplicate_count:
        status = "incompatible_cache_or_artifact"
        reason = f"duplicate key rows: {duplicate_count}"
    elif len(missing) or len(unexpected):
        status = "incompatible_cache_or_artifact"
        reason = f"missing={len(missing)} unexpected={len(unexpected)}"
    elif expected_hash != actual_hash:
        status = "incompatible_cache_or_artifact"
        reason = "candidate pool hash mismatch"
    return {
        "scorer": scorer,
        "status": status,
        "path": str(path),
        "row_count": int(len(actual)),
        "date_count": int(actual["rebalance_date"].nunique()) if not actual.empty else 0,
        "expected_row_count": int(len(expected)),
        "expected_date_count": int(expected["rebalance_date"].nunique()),
        "duplicate_key_rows": duplicate_count,
        "missing_rows": int(len(missing)),
        "unexpected_rows": int(len(unexpected)),
        "candidate_pool_hash": actual_hash,
        "expected_candidate_pool_hash": expected_hash,
        "reason": reason,
    }


def inspect_direct_score_cache(path: Path, expected_pool_per_side: int) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {
            "scorer": SCORER_DIRECT_SCORE,
            "status": "missing_offline_score",
            "path": str(path),
            "reason": "cache_not_found",
        }
    with path.open("r") as handle:
        cache = json.load(handle)
    signals = cache.get("signals", {})
    observed_counts: set[int] = set()
    for entry in signals.values():
        if isinstance(entry, dict):
            if "pool_per_side" in entry:
                observed_counts.add(int(entry["pool_per_side"]))
            elif isinstance(entry.get("long_scores"), list):
                observed_counts.add(len(entry["long_scores"]))
    if observed_counts and observed_counts != {expected_pool_per_side}:
        return {
            "scorer": SCORER_DIRECT_SCORE,
            "status": "missing_offline_score",
            "path": str(path),
            "reason": f"rejected incompatible cache: pool_per_side {sorted(observed_counts)} != expected {expected_pool_per_side}",
            "rejected_cache": True,
            "signal_count": len(signals),
            "observed_pool_per_side": sorted(observed_counts),
        }
    return {
        "scorer": SCORER_DIRECT_SCORE,
        "status": "available_for_grid_audit",
        "path": str(path),
        "reason": "compatible direct-score cache exists; pass a candidate grid to load_direct_score_cache",
        "signal_count": len(signals),
        "observed_pool_per_side": sorted(observed_counts),
    }


def build_score_matrix(
    candidate_grid: pd.DataFrame,
    raw_artifact: Path,
    rank_artifact: Path,
    direct_cache: Path,
    pool_per_side: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    key_cols = ["rebalance_date", "ticker", "pool_side"]
    score_matrix = candidate_grid[key_cols].copy()
    score_matrix = score_matrix.merge(deterministic_scores(candidate_grid), on=key_cols, how="left")
    coverage: dict[str, Any] = {"scorers": []}
    coverage["scorers"].append(validate_artifact_grid(deterministic_scores(candidate_grid), SCORER_DETERMINISTIC, candidate_grid))

    for scorer, path in ((SCORER_RAW_CONFIDENCE, raw_artifact), (SCORER_RANK_CONFIDENCE, rank_artifact)):
        scores, audit = load_signal_checks_artifact(path, scorer, candidate_grid)
        coverage["scorers"].append(audit)
        if audit["status"] == "valid":
            score_matrix = score_matrix.merge(scores, on=key_cols, how="left")
        else:
            score_matrix[score_column(scorer)] = np.nan

    direct_scores, direct_audit = load_direct_score_cache(direct_cache, candidate_grid, pool_per_side)
    coverage["scorers"].append(direct_audit)
    if direct_audit["status"] == "valid":
        score_matrix = score_matrix.merge(direct_scores, on=key_cols, how="left")
    else:
        score_matrix[score_column(SCORER_DIRECT_SCORE)] = np.nan

    valid_required = {
        item["scorer"]
        for item in coverage["scorers"]
        if item.get("scorer") in REQUIRED_SCORERS and item.get("status") == "valid"
    }
    missing_required = [scorer for scorer in REQUIRED_SCORERS if scorer not in valid_required]
    invalid_required = [
        item
        for item in coverage["scorers"]
        if item.get("scorer") in REQUIRED_SCORERS and item.get("status") == "incompatible_cache_or_artifact"
    ]
    if invalid_required:
        status = "INVALID_REPLAY"
    elif missing_required:
        status = "PARTIAL_OFFLINE_REPLAY"
    else:
        status = "VALID_CONTROLLED_REPLAY"
    coverage["summary"] = {
        "status": status,
        "required_scorers": list(REQUIRED_SCORERS),
        "valid_required_scorers": sorted(valid_required),
        "missing_required_scorers": missing_required,
        "offline_only": True,
        "llm_calls_attempted": 0,
        "http_client_created": False,
    }
    return score_matrix, coverage


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline controlled replay audit for LLM reversal artifacts.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--membership-path", type=Path, default=rb.DEFAULT_MEMBERSHIP_PATH)
    parser.add_argument("--pool-per-side", type=int, default=60)
    parser.add_argument("--positions-per-side", type=int, default=30)
    parser.add_argument("--cost-bps-levels", type=rb.parse_cost_bps_levels, default=(0.0, 2.5, 5.0))
    parser.add_argument("--raw-confidence-artifact", type=Path, default=DEFAULT_RAW_CONFIDENCE_ARTIFACT)
    parser.add_argument("--rank-confidence-artifact", type=Path, default=DEFAULT_RANK_CONFIDENCE_ARTIFACT)
    parser.add_argument("--direct-score-cache", type=Path, default=DEFAULT_DIRECT_SCORE_CACHE)
    parser.add_argument(
        "--reuse-candidate-grid",
        action="store_true",
        help="Reuse candidate_grid.csv and candidate_pool_audit.csv in --output-dir instead of rebuilding local features.",
    )
    return parser.parse_args(argv)


def run_offline_replay(args: argparse.Namespace) -> dict[str, Any]:
    if args.pool_per_side <= 0 or args.positions_per_side <= 0:
        raise ValueError("--pool-per-side and --positions-per-side must be positive")
    if args.positions_per_side > args.pool_per_side:
        raise ValueError("--positions-per-side cannot exceed --pool-per-side")

    config = rb.BenchmarkConfig(
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        output_dir=Path(args.output_dir),
        membership_path=Path(args.membership_path),
        cost_bps_levels=tuple(args.cost_bps_levels),
        cadence_mode="weekly_friday",
        signal_lookback_days=5,
        rebalance_days=5,
        holding_days=5,
        positions_per_side=None,
    )
    output_dir = Path(args.output_dir)
    if args.reuse_candidate_grid:
        candidate_grid_path = output_dir / "candidate_grid.csv"
        pool_audit_path = output_dir / "candidate_pool_audit.csv"
        if not candidate_grid_path.exists() or not pool_audit_path.exists():
            raise FileNotFoundError(
                "--reuse-candidate-grid requires existing candidate_grid.csv "
                "and candidate_pool_audit.csv in --output-dir"
            )
        replay = CandidateReplay(
            candidate_grid=pd.read_csv(candidate_grid_path),
            pool_audit=pd.read_csv(pool_audit_path),
        )
    else:
        histories = load_russell_ohlcv_histories()
        membership_lookup = rb.load_membership_lookup(config.membership_path, end_date=config.end)
        replay = build_candidate_replay(histories, membership_lookup, config, int(args.pool_per_side))
    score_matrix, coverage = build_score_matrix(
        replay.candidate_grid,
        args.raw_confidence_artifact,
        args.rank_confidence_artifact,
        args.direct_score_cache,
        int(args.pool_per_side),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    replay.candidate_grid.to_csv(output_dir / "candidate_grid.csv", index=False)
    replay.pool_audit.to_csv(output_dir / "candidate_pool_audit.csv", index=False)
    score_matrix.to_csv(output_dir / "score_matrix.csv", index=False)
    coverage["candidate_grid"] = {
        "row_count": int(len(replay.candidate_grid)),
        "date_count": int(replay.candidate_grid["rebalance_date"].nunique()),
        "pool_per_side": int(args.pool_per_side),
        "positions_per_side": int(args.positions_per_side),
        "candidate_pool_hash": candidate_pool_hash(replay.candidate_grid),
        "feature_payload_hash": feature_payload_hash(
            replay.candidate_grid,
            [col for col in replay.candidate_grid.columns if col.startswith("feature_")],
        ),
    }
    performance_outputs = build_controlled_performance_outputs(
        candidate_grid=replay.candidate_grid,
        score_matrix=score_matrix,
        coverage=coverage,
        positions_per_side=int(args.positions_per_side),
        cost_bps_levels=tuple(args.cost_bps_levels),
    )
    performance_scorecard = save_controlled_performance_outputs(
        output_dir=output_dir,
        outputs=performance_outputs,
        replay_status=coverage.get("summary", {}).get("status", "UNKNOWN"),
    )
    coverage["performance"] = {
        "status": (
            "VALID_CONTROLLED_PERFORMANCE"
            if coverage.get("summary", {}).get("status") == "VALID_CONTROLLED_REPLAY"
            else "PARTIAL_PERFORMANCE_ONLY"
        ),
        "scorers_with_performance": sorted(performance_outputs),
        "scorecard_path": str(output_dir / "performance_scorecard.csv"),
        "scorecard_row_count": int(len(performance_scorecard)),
        "all_required_scorers_valid": coverage.get("summary", {}).get("status") == "VALID_CONTROLLED_REPLAY",
    }
    with (output_dir / "coverage_audit.json").open("w") as handle:
        json.dump(coverage, handle, indent=2, sort_keys=True, default=_json_default)
    write_summary(output_dir / "offline_replay_summary.md", coverage)
    return coverage


def write_summary(path: Path, coverage: dict[str, Any]) -> None:
    summary = coverage.get("summary", {})
    lines = [
        "# Offline Controlled Replay Audit",
        "",
        f"Status: {summary.get('status', 'UNKNOWN')}",
        "",
        "This run is offline-only. No LLM calls were attempted.",
        "",
        "## Scorer Coverage",
        "",
        "| Scorer | Status | Reason |",
        "| --- | --- | --- |",
    ]
    for item in coverage.get("scorers", []):
        lines.append(
            f"| {item.get('scorer', '')} | {item.get('status', '')} | {str(item.get('reason', '')).replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Grid",
            "",
            f"Rows: {coverage.get('candidate_grid', {}).get('row_count', 'n/a')}",
            f"Dates: {coverage.get('candidate_grid', {}).get('date_count', 'n/a')}",
            f"Pool hash: `{coverage.get('candidate_grid', {}).get('candidate_pool_hash', '')}`",
            f"Feature payload hash: `{coverage.get('candidate_grid', {}).get('feature_payload_hash', '')}`",
            "",
            "## Performance",
            "",
            f"Status: {coverage.get('performance', {}).get('status', 'n/a')}",
            f"Scorers: {', '.join(coverage.get('performance', {}).get('scorers_with_performance', [])) or 'n/a'}",
            f"Scorecard: {coverage.get('performance', {}).get('scorecard_path', 'n/a')}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    coverage = run_offline_replay(args)
    status = coverage.get("summary", {}).get("status", "UNKNOWN")
    print(f"Offline replay audit complete: {status}")
    print(f"Outputs: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
