"""
LLM mean-reversion scoring experiment v2.

V2 keeps the standard v1 LLM prompt/output contract unchanged, but changes the
experiment harness around it:
- five independent 5-trading-day anchor offsets by default,
- optional Friday-to-Friday C2C/O2O preview mode,
- multiple pool/book replays from one cached LLM score file.
"""

from __future__ import annotations

import argparse
import asyncio
import html as html_lib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

import mean_reversion_llm_v1 as v1
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
DEFAULT_LEGACY_ROOT = Path("backtest_results/legacy_raw_anchor_test_2009_2014")
DEFAULT_LEGACY_FACTOR_ROOT = Path("backtest_results/factor_backtest/composite_scores_llm_signal_2009_2014_5d_horizon")
DEFAULT_V1_ROOTS = (
    Path("backtest_results/llm_reversal_signal_v1_2009_2014"),
    Path("backtest_results/llm_reversal_signal_v1_2009_2014_pos30"),
)
V2_PROMPT_STYLE = v1.PROMPT_STYLE_V1_SCORE
V2_PROMPT_VERSION = v1.PROMPT_VERSION

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


def parse_path_list(value: str) -> tuple[Path, ...]:
    paths = [Path(chunk.strip()) for chunk in str(value).split(",") if chunk.strip()]
    if not paths:
        raise argparse.ArgumentTypeError("Expected at least one path")
    return tuple(paths)


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
    return sliced


def validate_llm_scores_for_v2_pool(
    llm_scores_by_key: dict[str, dict[str, Any]],
    pool: v1.CandidatePool,
    pool_per_side: int,
    llm_source_pool_per_side: int,
) -> dict[str, Any]:
    exact_key = llm_score_storage_key(pool, pool_per_side)
    if exact_key in llm_scores_by_key:
        return v1.validate_llm_scores(
            llm_scores_by_key[exact_key],
            pool.long_tickers,
            pool.short_tickers,
            prompt_style=V2_PROMPT_STYLE,
        )

    source_key = llm_score_storage_key(pool, llm_source_pool_per_side)
    if source_key not in llm_scores_by_key:
        raise KeyError(f"Missing V2 LLM scores for {exact_key} and source {source_key}")
    return v1.validate_llm_scores(
        _subset_llm_scores_for_pool(llm_scores_by_key[source_key], pool),
        pool.long_tickers,
        pool.short_tickers,
        prompt_style=V2_PROMPT_STYLE,
    )


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
        if anchor_offset == "Friday" and pool.exit_date > config.end:
            continue
        pools.append(pool)
        pool_tickers = pool.pool_tickers
        long_set = set(pool.long_tickers)
        short_set = set(pool.short_tickers)

        if score_provider == "trail5":
            signed_scores = {ticker: pool.signal_by_ticker[ticker] for ticker in pool_tickers}
        elif score_provider == "llm":
            if llm_scores_by_key is None:
                raise ValueError("llm_scores_by_key is required for score_provider='llm'")
            validated = validate_llm_scores_for_v2_pool(
                llm_scores_by_key=llm_scores_by_key,
                pool=pool,
                pool_per_side=pool_per_side,
                llm_source_pool_per_side=int(llm_source_pool_per_side or pool_per_side),
            )
            signed_scores = v1.signed_scores_from_llm(validated)
        else:
            raise ValueError(f"Unsupported score_provider: {score_provider}")

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
            signal_rows.append(
                {
                    "date": pool.signal_date,
                    "anchor_offset": anchor_label,
                    "pool_per_side": int(pool_per_side),
                    "positions_per_side": int(positions_per_side),
                    "ticker": ticker,
                    "pool_side": "LONG" if ticker in long_set else "SHORT",
                    "score_provider": score_provider,
                    "llm_score": abs(signed_scores[ticker]) if score_provider == "llm" else np.nan,
                    "signal": signal[pos],
                    "trail_ret": pool.trail_by_ticker[ticker],
                    "fwd_c2c": fwd_c2c[pos],
                    "fwd_o2o": fwd_o2o[pos],
                    "decile": deciles[pos],
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
                        "trail_return_pct": pool.trail_by_ticker[pool_tickers[pos]] * 100.0,
                        "realized_return_pct": fwd_c2c[pos] * 100.0,
                        "realized_c2c_return_pct": fwd_c2c[pos] * 100.0,
                        "realized_o2o_return_pct": fwd_o2o[pos] * 100.0,
                        "weight": side_weight,
                        "decile": deciles[pos],
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
) -> dict[str, dict[str, Any]]:
    cache = v1.load_cache(cache_path)
    cache["metadata"] = {
        "schema_version": 2,
        "artifact_type": "llm_reversal_signal_v2_scores",
        "prompt_version": V2_PROMPT_VERSION,
        "prompt_style": V2_PROMPT_STYLE,
        "model": model,
        "pool_count": len(pools_by_key),
        "storage_key": "date|pool_per_side",
    }
    cache.setdefault("signals", {})
    async with httpx.AsyncClient(timeout=technical_llm.PORTFOLIO_REQUEST_TIMEOUT) as client:
        items = sorted(pools_by_key.items(), key=lambda item: item[0])
        for idx, (storage_key, pool) in enumerate(items, start=1):
            existing = cache["signals"].get(storage_key)
            if isinstance(existing, dict):
                try:
                    cached = v1._entry_from_cache(existing, pool, model, V2_PROMPT_VERSION, V2_PROMPT_STYLE)
                    if cached is not None:
                        print(f"  [{idx}/{len(items)}] {storage_key} cached")
                        continue
                except ValueError as exc:
                    cache.setdefault("failures", {})[storage_key] = {
                        "error": f"Invalid cached V2 score entry: {exc}",
                        "parse_status": "invalid_cache",
                        "prompt_version": V2_PROMPT_VERSION,
                        "prompt_style": V2_PROMPT_STYLE,
                        "model": model,
                        "universe_hash": pool.universe_hash,
                    }
                    cache["signals"].pop(storage_key, None)
                    v1.save_cache(cache, cache_path)

            print(f"  [{idx}/{len(items)}] {storage_key} calling LLM")
            started = time.time()
            try:
                result = await v1.call_llm_scores(
                    client=client,
                    pool=pool,
                    model=model,
                    temperature=temperature,
                    prompt_style=V2_PROMPT_STYLE,
                )
            except Exception as exc:
                cache.setdefault("failures", {})[storage_key] = {
                    "error": str(exc),
                    "parse_status": "failed",
                    "prompt_version": V2_PROMPT_VERSION,
                    "prompt_style": V2_PROMPT_STYLE,
                    "model": model,
                    "universe_hash": pool.universe_hash,
                    "pool_per_side": len(pool.long_tickers),
                    "regime": pool.regime,
                    "failed_at": pd.Timestamp.utcnow().isoformat(),
                    "v2_storage_key": storage_key,
                    "v2_latency_seconds": time.time() - started,
                }
                v1.save_cache(cache, cache_path)
                raise
            cache_entry = {
                **result,
                "rebalance_date": str(pool.signal_date.date()),
                "prompt_version": V2_PROMPT_VERSION,
                "prompt_style": V2_PROMPT_STYLE,
                "model": model,
                "universe_hash": pool.universe_hash,
                "cache_key": v1._cache_key(pool.signal_date, V2_PROMPT_VERSION, model, pool.universe_hash),
                "pool_per_side": len(pool.long_tickers),
                "regime": pool.regime,
                "v2_storage_key": storage_key,
                "v2_latency_seconds": time.time() - started,
            }
            cache["signals"][storage_key] = cache_entry
            cache.setdefault("failures", {}).pop(storage_key, None)
            v1.save_cache(cache, cache_path)
            if idx < len(items):
                await asyncio.sleep(rate_limit)
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
    gross_c2c = scorecard[
        (scorecard["execution_model"].astype(str) == "C2C")
        & (pd.to_numeric(scorecard["cost_bps"], errors="coerce") == 0.0)
    ].copy()
    anchor_average = gross_c2c[gross_c2c["is_anchor_average"].astype(bool)]
    if not anchor_average.empty:
        gross_c2c = anchor_average
    if gross_c2c.empty:
        return pd.DataFrame()
    friday_only = gross_c2c["anchor_offset"].astype(str).eq("Friday").all()
    gross_c2c.insert(0, "family", "V2")
    gross_c2c.insert(2, "basis", "Friday gross C2C" if friday_only else "5-anchor average gross C2C")
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
        lambda run: "unchanged v1 0-100 score prompt" if "LLM" in str(run) else "none: deterministic Trail5 baseline"
    )
    gross_c2c["score_source"] = gross_c2c.apply(
        lambda row: (
            f"LLM scored {int(row['score_pool_per_side'])}/{int(row['score_pool_per_side'])}; "
            f"row pool {int(row['pool_per_side'])}/{int(row['pool_per_side'])}"
            if str(row["run"]) == "LLM-Pool v2" and pd.notna(row.get("score_pool_per_side"))
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
        "Friday-only weekly schedule"
        if friday_only
        else "five independent every-5-trading-day anchor offsets"
    )
    gross_c2c["accounting"] = "gross Friday-to-Friday C2C" if friday_only else "gross fixed 5-trading-day C2C"
    gross_c2c["liquidity_filters"] = COMPARISON_LIQUIDITY_FILTERS
    gross_c2c["replay_cache"] = gross_c2c.apply(
        lambda row: (
            f"LLM scores cached once per date at pool {int(row['score_pool_per_side'])}; "
            "smaller pools and books replay offline"
            if str(row["run"]) == "LLM-Pool v2" and pd.notna(row.get("score_pool_per_side"))
            else "offline deterministic replay; no LLM cache"
        ),
        axis=1,
    )
    return gross_c2c


def build_gross_system_comparison(
    scorecard: pd.DataFrame,
    legacy_root: Path,
    legacy_factor_root: Path | None,
    v1_roots: tuple[Path, ...],
) -> pd.DataFrame:
    frames = [
        legacy_gross_comparison_rows(legacy_root, legacy_factor_root),
        v1_gross_comparison_rows(v1_roots),
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
    comparison = comparison.assign(_family_order=family_order).sort_values(
        ["_family_order", "pool_book", "run"],
        kind="mergesort",
    )
    return comparison.drop(columns=["_family_order"]).reset_index(drop=True)


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
    if gross_comparison is not None and not gross_comparison.empty:
        is_daily_path = gross_comparison["execution_model"].astype(str).str.lower().eq("daily")
        main_comparison = gross_comparison[~is_daily_path].copy()
        legacy_daily_context = gross_comparison[is_daily_path].copy()

        compact_rows: list[dict[str, Any]] = []
        compact = main_comparison.copy()
        compact = compact.sort_values(["family", "pool_book", "run"], kind="mergesort")
        for _, row in compact.iterrows():
            compact_rows.append(
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

    detail_cols = [
        "pool_per_side",
        "positions_per_side",
        "anchor_offset",
        "scorer",
        "run",
        "execution_model",
        "cost_bps",
        "display_return_basis",
        "sharpe",
        "cagr_pct",
        "total_return_pct",
        "max_drawdown_pct",
        "capital_cagr_pct",
        "capital_total_return_pct",
        "capital_max_drawdown_pct",
        "ic_mean",
        "avg_turnover_total",
    ]
    detail_rows = zero_cost[[col for col in detail_cols if col in zero_cost.columns]].copy()
    detail_rows["anchor_offset"] = detail_rows["anchor_offset"].astype(str)
    detail_rows = detail_rows.sort_values(
        ["execution_model", "cost_bps", "pool_per_side", "positions_per_side", "scorer", "anchor_offset", "run"],
        kind="mergesort",
    )
    detail_table_html = simple_table(
        detail_rows.to_dict("records"),
        css_class="full-table",
        highlight_cols=raw_metric_cols,
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
    .compact-table td:nth-child(-n+6), .compact-table th:nth-child(-n+6) {{ text-align: left; }}
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
      <p>{report_scope_label}, gross 0 bps, unchanged v1 0-100 prompt. V2 scores the {score_pool_phrase} pool once per signal date and replays configured books offline.</p>
      <p class="muted">Top-line CAGR, Total, and Max DD use the displayed return basis. New runs display 50/50 capital-normalized returns; raw 100/100 spread metrics stay in the audit tables.</p>
      <p class="muted">{report_scope_note}</p>
      <p class="muted">Generated {html_lib.escape(generated_at)} from cached/output CSVs.</p>
    </header>

    <section class="section">
      <h2>Headline</h2>
      <div class="summary-grid">
        {summary_html}
      </div>
      <div class="note">{headline_note}</div>
    </section>

    <section class="section">
      <h2>{v2_section_title}</h2>
      <p>{v2_section_note}</p>
      <div class="legend"><span class="legend-swatch legend-best">Best</span><span class="legend-swatch legend-worst">Worst</span><span>Within the table. For Max DD, best means least negative drawdown.</span></div>
      <div class="table-wrap">{headline_html}</div>
    </section>

    <section class="section">
      <h2>Apples-to-Apples Gross C2C</h2>
      <p>All rows here use gross fixed-window C2C accounting. Legacy LLM rows are recomputed from legacy selection files, not daily MTM P&amp;L. The Return Basis column separates exposure scaling from accounting.</p>
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
    model: str,
    temperature: float,
    cache_path: Path,
    rate_limit: float,
    llm: bool,
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
        )
        bundle = build_v2_bundle(weekly, checks, selections, config)
        run_dir = variant_dir(output_dir, variant) / "baseline_pool"
        save_v2_bundle(bundle, run_dir)
        all_rows.append(scorecard_for_bundle("Baseline-Pool", bundle, variant, scorer="trail5"))
        if variant.pool_per_side == llm_pool_size:
            for pool in pools:
                llm_pools_by_key.setdefault(llm_score_storage_key(pool, llm_pool_size), pool)

    llm_scores: dict[str, dict[str, Any]] = {}
    if llm:
        print(f"\nCollecting V2 LLM scores with unchanged v1 prompt: {len(llm_pools_by_key)} pools")
        llm_scores = await collect_llm_scores_for_v2_pools(
            llm_pools_by_key,
            cache_path=cache_path,
            model=model,
            temperature=temperature,
            rate_limit=rate_limit,
        )

        for variant in variants:
            print(f"\nRunning V2 LLM replay {variant.label}...")
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
                llm_scores_by_key=llm_scores,
                llm_source_pool_per_side=llm_pool_size,
            )
            bundle = build_v2_bundle(weekly, checks, selections, config)
            run_dir = variant_dir(output_dir, variant) / "llm_pool_v2"
            save_v2_bundle(bundle, run_dir)
            v1.save_score_diagnostics(checks, run_dir, fail_on_copy=False)
            all_rows.append(
                scorecard_for_bundle(
                    "LLM-Pool v2",
                    bundle,
                    variant,
                    scorer="llm_score",
                    score_pool_per_side=llm_pool_size,
                )
            )

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
    )
    gross_comparison.to_csv(output_dir / "gross_legacy_v1_v2_comparison.csv", index=False)
    apples_to_apples = gross_comparison[
        ~gross_comparison["execution_model"].astype(str).str.lower().eq("daily")
    ].copy()
    apples_to_apples.to_csv(output_dir / "apples_to_apples_gross_c2c_comparison.csv", index=False)

    manifest = {
        "artifact_type": "llm_reversal_signal_v2",
        "start": str(config.start.date()),
        "end": str(config.end.date()),
        "pool_sizes": list(pool_sizes),
        "book_sizes": list(book_sizes),
        "schedule": schedule,
        "anchor_offsets": list(anchor_offsets),
        "llm_pool_size": int(llm_pool_size),
        "llm_call_policy": "one call per date at llm_pool_size; smaller pools slice those scores",
        "cost_bps_levels": list(config.cost_bps_levels),
        "headline_cost_bps": 0.0,
        "legacy_comparison_root": str(legacy_root),
        "legacy_factor_comparison_root": str(legacy_factor_root),
        "v1_comparison_roots": [str(path) for path in v1_roots],
        "gross_system_comparison": "gross_legacy_v1_v2_comparison.csv",
        "apples_to_apples_comparison": "apples_to_apples_gross_c2c_comparison.csv",
        "prompt_source": "mean_reversion_llm_v1.V1_SYSTEM_PROMPT",
        "prompt_style": V2_PROMPT_STYLE,
        "prompt_version": V2_PROMPT_VERSION,
        "llm_enabled": bool(llm),
    }
    (output_dir / "run_config.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_v2_html_report(output_dir, scorecard, gross_comparison)
    return scorecard


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM reversal signal v2 experiment.")
    parser.add_argument("--phase", choices=("baseline", "full"), default="baseline")
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
    parser.add_argument("--model", default=technical_llm.DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=technical_llm.DEFAULT_TEMPERATURE)
    parser.add_argument("--rate-limit", type=float, default=2.0)
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
    pools = "-".join(str(pool) for pool in args.pool_sizes)
    if args.schedule == "friday":
        schedule_label = "friday"
    else:
        anchors = "-".join(str(anchor) for anchor in args.anchor_offsets)
        schedule_label = f"anchors{anchors}"
    name = (
        f"llm_reversal_signal_v2_{start}_{end}_pools{pools}_{schedule_label}_"
        f"scorepool{int(args.llm_pool_size)}_mdl-{model}.json"
    )
    return args.cache_dir / name


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if any(pool <= 0 for pool in args.pool_sizes):
        raise ValueError("--pool-sizes must contain positive integers")
    if any(book <= 0 for book in args.book_sizes):
        raise ValueError("--book-sizes must contain positive integers")
    if args.llm_pool_size <= 0:
        raise ValueError("--llm-pool-size must be a positive integer")
    if args.schedule == "anchors" and any(anchor < 0 or anchor > 4 for anchor in args.anchor_offsets):
        raise ValueError("--anchor-offsets must be integers between 0 and 4")
    config = build_config(args)
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
            model=args.model,
            temperature=args.temperature,
            cache_path=cache_path_for_args(args),
            rate_limit=args.rate_limit,
            llm=args.phase == "full",
        )
    )


if __name__ == "__main__":
    main()
