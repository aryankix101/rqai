"""
LLM Portfolio Constructor — Batch Engine for A/B Testing

For each rebalance date within the test window:
  1. Reads direction_score, magnitude_score, and raw indicators from Russell close-only composite CSVs
  2. Reads composite scores from composite_scores CSVs
  3. Builds a universe dict for all tickers on that date
  4. Calls the LLM (construct_portfolio_from_universe) with dates masked
  5. Caches LLM portfolio decisions to JSON

Usage:
    python llm_portfolio_constructor.py --start 2019-01-01 --end 2019-12-31
    python llm_portfolio_constructor.py --start 2019-01-01 --end 2019-12-31 --rebalance-days 10
    python llm_portfolio_constructor.py --start 2019-01-01 --end 2019-12-31 --resume  # resume from cache
"""

import asyncio
import argparse
import json
import os
import sys
import tempfile
import time
from bisect import bisect_right
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import httpx

from data_source_config import (
    DATASET_VERSION,
    DEFAULT_COMPOSITE_SCORES_DIR,
    RUSSELL_MEMBERSHIP_PATH,
    SPY_SIGNAL_DATA_PATH,
)
from indicator_feature_schema import (
    CANONICAL_INDICATOR_FEATURES,
    extract_canonical_indicator_features,
)
from llm_loss_memory import LossMemoryStore
from portfolio_constraints import (
    MIN_TRADE_PRICE,
    filter_symbol_hygiene_frame,
    has_min_trade_price,
    row_has_clean_symbol_metadata,
)
import mean_reversion_llm
import technical_llm

# =============================================================================
# Config
# =============================================================================

COMPOSITE_SCORES_DIR = DEFAULT_COMPOSITE_SCORES_DIR
SPY_DATA_PATH = SPY_SIGNAL_DATA_PATH
CACHE_DIR = Path("backtest_results/llm_portfolio_cache")
CACHE_SCHEMA_VERSION = 3
SIGNAL_CACHE_SCHEMA_VERSION = 1
PROMPT_FORMAT_VERSION = "two_stage_signal_overlay_v6_r1000_prefilter_compact_trace"
DEFAULT_MODEL = technical_llm.DEFAULT_MODEL
DEFAULT_LLM_STRATEGY = "deterministic"
MEAN_REVERSION_TRAIL_LOOKBACK_DAYS = 5
SUPPORTED_LLM_STRATEGIES = {
    "deterministic": technical_llm,
    "mean_reversion": mean_reversion_llm,
}

EXCLUDED_TICKERS = {"CPWR", "SPY"}

LLM_INDICATOR_COLS = list(CANONICAL_INDICATOR_FEATURES)


def _resolve_llm_strategy_module(strategy_style: str):
    """Resolve the portfolio-construction module for a given strategy style."""
    normalized = str(strategy_style or "").strip().lower()
    if normalized in SUPPORTED_LLM_STRATEGIES:
        return SUPPORTED_LLM_STRATEGIES[normalized]

    supported = ", ".join(sorted(SUPPORTED_LLM_STRATEGIES))
    raise ValueError(f"Unsupported strategy style '{strategy_style}'. Expected one of: {supported}")


def _strategy_signal_source(strategy_style: str) -> str:
    """Return the signal-source label expected for a given strategy style."""
    normalized = str(strategy_style or "").strip().lower()
    if normalized == "mean_reversion":
        return f"trail_return_{MEAN_REVERSION_TRAIL_LOOKBACK_DAYS}d_rank_v1"
    return "technical_composite_v1"


# =============================================================================
# Data Loading
# =============================================================================

def load_spy_calendar(rebalance_days: int = 10) -> tuple:
    """Load SPY data and compute rebalance dates.

    Returns:
        (spy_df, all_spy_dates, rebalance_dates)
    """
    if not SPY_DATA_PATH.exists():
        raise FileNotFoundError(f"SPY data not found: {SPY_DATA_PATH}")

    spy_df = pd.read_csv(SPY_DATA_PATH, parse_dates=["date"])
    spy_df = spy_df.sort_values("date").reset_index(drop=True)
    all_dates = sorted(spy_df["date"].unique())
    rebalance_dates = all_dates[::rebalance_days]
    return spy_df, all_dates, rebalance_dates


def identify_regime(spy_df: pd.DataFrame, date: pd.Timestamp) -> str:
    """Get market regime for a specific date from SPY data."""
    spy_df = spy_df.sort_values("date").reset_index(drop=True)
    spy_df["spy_200d_ma"] = spy_df["adj_close"].rolling(200, min_periods=200).mean()

    row = spy_df[spy_df["date"] == date]
    if row.empty:
        # Find nearest prior date
        prior = spy_df[spy_df["date"] <= date]
        if prior.empty:
            return "unknown"
        row = prior.iloc[[-1]]

    row = row.iloc[0]
    close = row.get("adj_close", np.nan)
    ma200 = row.get("spy_200d_ma", np.nan)
    vix_pct = row.get("vix_percentile_252d", np.nan)

    if pd.isna(close) or pd.isna(ma200):
        return "unknown"

    trend = "uptrend" if close > ma200 else "downtrend"

    if pd.isna(vix_pct):
        return "normal"

    # Use expanding percentile thresholds
    prior_vix = spy_df.loc[spy_df["date"] <= date, "vix_percentile_252d"].dropna()
    if len(prior_vix) < 50:
        return "normal"

    thresh_high = prior_vix.quantile(0.80)
    thresh_low = prior_vix.quantile(0.30)

    if vix_pct > thresh_high:
        vol = "high_vol"
    elif vix_pct < thresh_low:
        vol = "low_vol"
    else:
        vol = "mid_vol"

    if vol == "high_vol" and trend == "downtrend":
        return "crisis"
    elif vol == "high_vol" and trend == "uptrend":
        return "squeeze"
    elif vol == "low_vol" and trend == "uptrend":
        return "grind_up"
    elif vol == "low_vol" and trend == "downtrend":
        return "slow_bleed"
    return "normal"


def _membership_flag_is_active(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) != 0.0
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _normalize_membership_symbol(symbol) -> str:
    if pd.isna(symbol):
        return ""
    return str(symbol).strip().upper().split("-", 1)[0]


def load_russell_membership_lookup(path=RUSSELL_MEMBERSHIP_PATH) -> Dict[str, Any]:
    """Load point-in-time Russell membership for prompt-universe prefiltering."""
    membership_path = Path(path).expanduser()
    if not membership_path.exists():
        raise FileNotFoundError(f"Russell membership file not found: {membership_path}")

    if membership_path.suffix.lower() == ".parquet":
        membership_df = pd.read_parquet(membership_path)
    else:
        membership_df = pd.read_csv(membership_path, parse_dates=["date"])

    required_cols = {"date", "symbol"}
    missing_cols = required_cols - set(membership_df.columns)
    if missing_cols:
        raise ValueError(
            f"Russell membership file missing required columns: {sorted(missing_cols)}"
        )

    membership_df = membership_df.copy()
    membership_df["date"] = pd.to_datetime(membership_df["date"], errors="coerce")
    membership_df["symbol"] = membership_df["symbol"].map(_normalize_membership_symbol)
    membership_df = membership_df.dropna(subset=["date"])
    membership_df = membership_df[membership_df["symbol"] != ""]
    if "in_r1000" in membership_df.columns:
        membership_df = membership_df[membership_df["in_r1000"].map(_membership_flag_is_active)]
    if membership_df.empty:
        raise ValueError(f"Russell membership file has no active rows: {membership_path}")

    grouped = (
        membership_df[["date", "symbol"]]
        .drop_duplicates()
        .groupby("date")["symbol"]
        .agg(lambda values: frozenset(values))
    )
    dates = grouped.index.to_numpy(dtype="datetime64[ns]").view("i8")
    order = np.argsort(dates)
    dates = dates[order]
    universes = tuple(grouped.iloc[int(idx)] for idx in order)
    return {
        "dates_ns": dates,
        "universes": universes,
        "path": str(membership_path),
    }


def _is_russell_member_on_or_before(
    membership_lookup: Optional[Dict[str, Any]],
    ticker: str,
    as_of_date: pd.Timestamp,
) -> bool:
    if membership_lookup is None:
        return True
    dates_ns = membership_lookup.get("dates_ns")
    universes = membership_lookup.get("universes")
    if dates_ns is None or len(dates_ns) == 0 or not universes:
        return False
    idx = bisect_right(dates_ns, pd.Timestamp(as_of_date).value) - 1
    if idx < 0:
        return False
    return _normalize_membership_symbol(ticker) in universes[idx]


def load_signal_data_preloaded(
    composite_dfs: Dict[str, pd.DataFrame],
    date: pd.Timestamp,
    strategy_style: str = DEFAULT_LLM_STRATEGY,
    membership_lookup: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build universe dict from pre-loaded composite DataFrames.

    Args:
        composite_dfs: {ticker: DataFrame} with composite scores and indicators
        date: Target date

    Returns:
        {
            ticker: {
                direction_score,
                magnitude_score,
                trail_return_5d,
                mean_reversion_signal,
                indicators: {...},
            }
        }
    """
    universe = {}
    for ticker, df in composite_dfs.items():
        if ticker in EXCLUDED_TICKERS:
            continue
        if not _is_russell_member_on_or_before(membership_lookup, ticker, date):
            continue

        row_positions = np.flatnonzero((df["date"] == date).to_numpy())
        if len(row_positions) == 0:
            continue
        row_pos = int(row_positions[0])
        row = df.iloc[row_pos]
        if not row_has_clean_symbol_metadata(row):
            continue

        direction_score_raw = row.get("direction_score", np.nan)
        magnitude_score_raw = row.get("magnitude_score", np.nan)
        require_composite_scores = str(strategy_style or "").strip().lower() != "mean_reversion"
        if require_composite_scores and (pd.isna(direction_score_raw) or pd.isna(magnitude_score_raw)):
            continue
        direction_score = float(direction_score_raw) if pd.notna(direction_score_raw) else 0.0
        magnitude_score = float(magnitude_score_raw) if pd.notna(magnitude_score_raw) else 0.0

        adj_close = row.get("adj_close", np.nan)
        if not has_min_trade_price(adj_close):
            continue

        trail_return_5d = np.nan
        if row_pos >= MEAN_REVERSION_TRAIL_LOOKBACK_DAYS:
            prior_close = df.iloc[row_pos - MEAN_REVERSION_TRAIL_LOOKBACK_DAYS].get("adj_close", np.nan)
            if pd.notna(prior_close) and prior_close > 0 and pd.notna(adj_close) and adj_close > 0:
                trail_return_5d = float(adj_close / prior_close - 1.0)

        # Extract indicator context using canonical names with percentile fallbacks.
        indicators = extract_canonical_indicator_features(row, LLM_INDICATOR_COLS)

        universe[ticker] = {
            "direction_score": float(direction_score),
            "magnitude_score": float(magnitude_score),
            "adj_close": float(adj_close),
            "trail_return_5d": float(trail_return_5d) if pd.notna(trail_return_5d) else np.nan,
            "mean_reversion_signal": float(-trail_return_5d) if pd.notna(trail_return_5d) else np.nan,
            "asset_id": row.get("asset_id", ticker),
            "source_column": row.get("source_column", row.get("asset_id", ticker)),
            "indicators": indicators,
        }

    return universe


def _count_mean_reversion_eligible_rows(universe: Dict[str, Dict[str, Any]]) -> int:
    return sum(
        1
        for payload in universe.values()
        if pd.notna(payload.get("trail_return_5d")) and pd.notna(payload.get("mean_reversion_signal"))
    )


def _minimum_required_universe_size(
    strategy_style: str,
    effective_top_n: int,
    target_positions_per_side: Optional[int],
    mean_reversion_candidate_mode: str = "fixed_top_n",
) -> int:
    minimum = 60
    normalized = str(strategy_style or "").strip().lower()
    if normalized == "mean_reversion":
        if str(mean_reversion_candidate_mode or "").strip().lower() == "decile":
            if target_positions_per_side is not None:
                return max(minimum, int(target_positions_per_side) * 10)
            return minimum
        required_short_candidates = (
            int(target_positions_per_side)
            if target_positions_per_side is not None
            else technical_llm.MIN_POSITIONS_PER_SIDE
        )
        return max(minimum, int(effective_top_n) + required_short_candidates)
    if target_positions_per_side is not None:
        return max(minimum, 2 * int(target_positions_per_side))
    return minimum


# =============================================================================
# Batch Preloading
# =============================================================================

def preload_all_data() -> Dict[str, pd.DataFrame]:
    """Preload all composite score CSVs into memory.

    Returns:
        composite_dfs: {ticker: DataFrame}
    """
    print("📂 Preloading composite score CSVs...")
    composite_dfs = {}
    filtered_rows = 0
    touched_tickers = 0
    for csv_file in sorted(COMPOSITE_SCORES_DIR.glob("*.csv")):
        try:
            parts = csv_file.stem.split("_")
            ticker = parts[2].upper()
            if ticker in EXCLUDED_TICKERS:
                continue
            df = pd.read_csv(csv_file, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            cleaned = filter_symbol_hygiene_frame(df)
            removed = len(df) - len(cleaned)
            if removed > 0:
                filtered_rows += removed
                touched_tickers += 1
            df = cleaned.reset_index(drop=True)
            if df.empty:
                continue
            composite_dfs[ticker] = df
        except Exception:
            continue
    print(f"   ✓ Loaded {len(composite_dfs)} composite score files\n")
    if filtered_rows > 0:
        print(
            "   ✓ Symbol hygiene filter removed "
            f"{filtered_rows} ambiguous/recycled rows across {touched_tickers} tickers\n"
        )

    return composite_dfs


# =============================================================================
# Cache Management
# =============================================================================

def load_cache(cache_path: Path) -> Dict[str, Any]:
    """Load existing LLM portfolio cache."""
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(
                f"⚠️  Cache file {cache_path} is corrupted or incomplete. "
                "Ignoring it and rebuilding."
            )
    return {"portfolios": {}, "metadata": {}, "failures": {}}


def save_cache(cache: Dict[str, Any], cache_path: Path):
    """Save LLM portfolio cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=cache_path.parent,
            prefix=f"{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(cache, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, cache_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def prompt_style_label(turnover_aware: bool) -> str:
    """Return the cache/output label for the prompt variant."""
    return "turnover_aware" if turnover_aware else "signal_only"


def _metadata_float_matches(metadata: Dict[str, Any], key: str, expected: Optional[float]) -> bool:
    actual = metadata.get(key)
    if actual is None and expected is None:
        return True
    try:
        return abs(float(actual) - float(expected)) <= 1e-12
    except (TypeError, ValueError):
        return False


def _sanitize_cache_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "-" for ch in str(value).strip().lower())
    token = "-".join(part for part in token.split("-") if part)
    return token or "unknown"


def _format_cache_float_token(value: Optional[float]) -> str:
    if value is None:
        return "none"
    token = f"{float(value):g}"
    return token.replace("-", "m").replace(".", "p")


def build_cache_filename(
    start_date: str,
    end_date: str,
    rebalance_days: int,
    use_equal_weight: bool,
    turnover_aware: bool,
    strategy_style: str = DEFAULT_LLM_STRATEGY,
    target_positions_per_side: Optional[int] = None,
    enable_loss_memory: bool = False,
    loss_memory_top_k: int = 3,
    loss_memory_max_targets_per_side: int = 4,
    log_llm_trace: bool = False,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_n_magnitude: Optional[int] = None,
    mean_reversion_candidate_mode: str = "fixed_top_n",
    compact_output: bool = False,
    signal_only: bool = False,
) -> str:
    """Build a cache filename that preserves prompt and execution variants."""
    execution_mode = "equal_weight" if use_equal_weight else "llm_weighted"
    prompt_style = prompt_style_label(turnover_aware)
    target_suffix = (
        f"_n{int(target_positions_per_side)}"
        if target_positions_per_side is not None
        else ""
    )
    loss_memory_suffix = ""
    if enable_loss_memory:
        loss_memory_suffix = f"_lossmem_k{int(loss_memory_top_k)}_t{int(loss_memory_max_targets_per_side)}"
    trace_suffix = "_trace" if log_llm_trace else ""
    config_suffix = "_r1000"
    if model is not None:
        config_suffix += f"_mdl-{_sanitize_cache_token(model)}"
    if top_n_magnitude is not None:
        config_suffix += f"_top{int(top_n_magnitude)}"
    if str(strategy_style or "").strip().lower() == "mean_reversion":
        config_suffix += f"_cand-{_sanitize_cache_token(mean_reversion_candidate_mode)}"
    if compact_output:
        config_suffix += "_compact"
    if signal_only:
        config_suffix += "_signalonly"
    if temperature is not None:
        config_suffix += f"_temp{_format_cache_float_token(temperature)}"
    return (
        f"llm_portfolio_{strategy_style}_{start_date}_{end_date}_{rebalance_days}d_"
        f"{execution_mode}_{prompt_style}_{DATASET_VERSION}"
        f"{target_suffix}{loss_memory_suffix}{trace_suffix}{config_suffix}.json"
    )


def _cache_is_compatible(
    cache: Dict[str, Any],
    execution_mode: str,
    turnover_aware: bool,
    strategy_style: str,
    model: str,
    temperature: float,
    top_n_magnitude: int,
    target_positions_per_side: Optional[int],
    mean_reversion_candidate_mode: str = "fixed_top_n",
    compact_output: bool = False,
    enable_loss_memory: bool = False,
    loss_memory_top_k: int = 3,
    loss_memory_max_targets_per_side: int = 4,
    log_llm_trace: bool = False,
) -> bool:
    """Reject legacy or fallback-generated caches."""
    metadata = cache.get("metadata", {})
    expected_signal_source = _strategy_signal_source(strategy_style)
    cached_signal_source = metadata.get("strategy_signal_source")
    if strategy_style == DEFAULT_LLM_STRATEGY and cached_signal_source is None:
        cached_signal_source = expected_signal_source
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False
    if metadata.get("execution_mode") != execution_mode:
        return False
    if metadata.get("turnover_aware") != turnover_aware:
        return False
    if metadata.get("strategy_style", DEFAULT_LLM_STRATEGY) != strategy_style:
        return False
    if metadata.get("model") != model:
        return False
    if not _metadata_float_matches(metadata, "temperature", temperature):
        return False
    if int(metadata.get("top_n_magnitude", -1)) != int(top_n_magnitude):
        return False
    if cached_signal_source != expected_signal_source:
        return False
    if metadata.get("prompt_style") != prompt_style_label(turnover_aware):
        return False
    if metadata.get("prompt_format_version") != PROMPT_FORMAT_VERSION:
        return False
    if metadata.get("target_positions_per_side") != target_positions_per_side:
        return False
    if strategy_style == "mean_reversion":
        if metadata.get("mean_reversion_candidate_mode", "fixed_top_n") != mean_reversion_candidate_mode:
            return False
        if bool(metadata.get("compact_output", False)) != bool(compact_output):
            return False
    if bool(metadata.get("log_llm_trace", False)) != bool(log_llm_trace):
        return False
    if bool(metadata.get("loss_memory_enabled", False)) != bool(enable_loss_memory):
        return False
    if int(metadata.get("loss_memory_top_k", 3)) != int(loss_memory_top_k):
        return False
    if int(metadata.get("loss_memory_max_targets_per_side", 4)) != int(loss_memory_max_targets_per_side):
        return False
    if metadata.get("llm_source_policy") != "strict_llm_only":
        return False
    if metadata.get("source_dataset") != DATASET_VERSION:
        return False
    if metadata.get("composite_scores_dir") != str(COMPOSITE_SCORES_DIR):
        return False
    if metadata.get("membership_gating_enabled") is not True:
        return False
    if metadata.get("membership_path") != str(RUSSELL_MEMBERSHIP_PATH):
        return False

    for portfolio in cache.get("portfolios", {}).values():
        if portfolio.get("llm_source") != "llm":
            return False
        for side_key in ("long_basket", "short_basket"):
            for entry in portfolio.get(side_key, []):
                if entry.get("llm_source") != "llm":
                    return False
                if "weight_raw" not in entry or "weight_exec" not in entry:
                    return False
    return True


# =============================================================================
# Main Batch Runner
# =============================================================================

async def run_batch(
    start_date: str,
    end_date: str,
    rebalance_days: int = 10,
    resume: bool = True,
    temperature: float = 0.15,
    top_n_magnitude: int = 60,
    model: str = DEFAULT_MODEL,
    rate_limit_delay: float = 2.0,
    use_equal_weight: bool = False,
    turnover_aware: bool = False,
    strategy_style: str = DEFAULT_LLM_STRATEGY,
    target_positions_per_side: Optional[int] = None,
    mean_reversion_candidate_mode: str = "fixed_top_n",
    compact_output: bool = False,
    signal_only: bool = False,
    enable_loss_memory: bool = False,
    loss_memory_top_k: int = 3,
    loss_memory_max_targets_per_side: int = 4,
    log_llm_trace: bool = False,
):
    """Run LLM portfolio construction for all rebalance dates in window.

    Args:
        start_date: Start of test window (ISO format)
        end_date: End of test window (ISO format)
        rebalance_days: Rebalance frequency in trading days
        resume: If True, skip dates already in cache
        temperature: LLM temperature (0.3 = semi-deterministic)
        top_n_magnitude: How many top-magnitude stocks to show LLM in detail
        model: DeepSeek model name
        rate_limit_delay: Seconds to wait between API calls
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    llm_module = _resolve_llm_strategy_module(strategy_style)
    if signal_only and strategy_style != "mean_reversion":
        raise ValueError("--signal-only currently supports --strategy-style mean_reversion only")
    target_positions_per_side = technical_llm.normalize_target_positions_per_side(
        target_positions_per_side
    )
    mean_reversion_candidate_mode = str(mean_reversion_candidate_mode or "fixed_top_n").strip().lower()
    if strategy_style == "mean_reversion":
        valid_modes = getattr(llm_module, "MEAN_REVERSION_CANDIDATE_MODES", {"fixed_top_n"})
        if mean_reversion_candidate_mode not in valid_modes:
            raise ValueError(
                f"mean_reversion_candidate_mode must be one of {sorted(valid_modes)}, "
                f"got {mean_reversion_candidate_mode!r}"
            )
    else:
        mean_reversion_candidate_mode = "fixed_top_n"
    effective_top_n = max(
        top_n_magnitude,
        technical_llm.max_positions_per_side(target_positions_per_side),
    )

    print(f"\n{'='*80}")
    print(f"LLM PORTFOLIO CONSTRUCTOR — BATCH MODE")
    print(f"Test Window: {start_date} to {end_date}")
    print(f"Rebalance: every {rebalance_days} trading days")
    temperature_label = (
        f"{temperature}"
        if technical_llm.model_supports_temperature(model)
        else f"{temperature} (ignored by model)"
    )
    print(f"Model: {model}  |  Temperature: {temperature_label}")
    print(f"Strategy style: {strategy_style}")
    print(f"Target positions per side: {target_positions_per_side or 'flex (5-30)'}")
    if strategy_style == "mean_reversion" and mean_reversion_candidate_mode == "decile":
        print("Mean-reversion candidate mode: exact D1/D10 decile buckets")
    else:
        print(f"Top-N ranked candidates shown to LLM: {effective_top_n}")
    print(f"Min decision price: > ${MIN_TRADE_PRICE:.2f}")
    print(f"Execution mode: {'equal_weight' if use_equal_weight else 'llm_weighted'}")
    print(f"Signal-only artifact: {'enabled' if signal_only else 'disabled'}")
    print(f"LLM prompt: {prompt_style_label(turnover_aware)}")
    print(f"Compact LLM output: {'enabled' if compact_output else 'disabled'}")
    print(f"Prompt/response trace logging: {'enabled' if log_llm_trace else 'disabled'}")
    if enable_loss_memory:
        print(
            "Loss memory: enabled "
            f"(top_k={loss_memory_top_k}, targets/side={loss_memory_max_targets_per_side})"
        )
    else:
        print("Loss memory: disabled")
    print(f"{'='*80}\n")

    # Load SPY calendar
    print("📊 Loading SPY calendar...")
    spy_df, all_spy_dates, all_rebalance_dates = load_spy_calendar(rebalance_days)

    # Filter to test window
    rebalance_dates = [
        d for d in all_rebalance_dates
        if start_ts <= pd.Timestamp(d) <= end_ts
    ]
    print(f"   ✓ {len(rebalance_dates)} rebalance dates in window\n")

    if not rebalance_dates:
        print("❌ No rebalance dates in the specified window")
        return

    print("📋 Loading Russell membership gate...")
    membership_lookup = load_russell_membership_lookup()
    print(f"   ✓ Membership gating enabled: {membership_lookup['path']}\n")

    # Preload composite data
    composite_dfs = preload_all_data()

    # Cache setup
    cache_filename = build_cache_filename(
        start_date=start_date,
        end_date=end_date,
        rebalance_days=rebalance_days,
        use_equal_weight=use_equal_weight,
        turnover_aware=turnover_aware,
        strategy_style=strategy_style,
        target_positions_per_side=target_positions_per_side,
        enable_loss_memory=enable_loss_memory,
        loss_memory_top_k=loss_memory_top_k,
        loss_memory_max_targets_per_side=loss_memory_max_targets_per_side,
        log_llm_trace=log_llm_trace,
        model=model,
        temperature=temperature,
        top_n_magnitude=effective_top_n,
        mean_reversion_candidate_mode=mean_reversion_candidate_mode,
        compact_output=compact_output,
        signal_only=signal_only,
    )
    cache_path = CACHE_DIR / cache_filename
    execution_mode = "equal_weight" if use_equal_weight else "llm_weighted"
    empty_cache = {"signals": {}, "metadata": {}, "failures": {}} if signal_only else {"portfolios": {}, "metadata": {}, "failures": {}}
    cache = load_cache(cache_path) if resume else empty_cache
    cached_min_price = cache.get("metadata", {}).get("min_decision_price")
    cache_entries_key = "signals" if signal_only else "portfolios"
    if resume and cache.get(cache_entries_key) and cached_min_price != MIN_TRADE_PRICE:
        print(
            "⚠️  Existing cache was built with a different minimum decision price. "
            "Rebuilding LLM portfolios with the new filter."
        )
        cache = empty_cache
    elif (
        resume
        and cache.get(cache_entries_key)
        and not signal_only
        and not _cache_is_compatible(
        cache,
        execution_mode,
        turnover_aware,
        strategy_style,
        model,
        temperature,
        effective_top_n,
        target_positions_per_side,
        mean_reversion_candidate_mode,
        compact_output,
        enable_loss_memory,
        loss_memory_top_k,
        loss_memory_max_targets_per_side,
        log_llm_trace,
    )):
        print(
            "⚠️  Existing cache uses a legacy or incompatible schema. "
            "Ignoring it and rebuilding strict LLM portfolios."
        )
        cache = empty_cache
    elif resume and cache.get(cache_entries_key) and signal_only:
        metadata = cache.get("metadata", {})
        if (
            metadata.get("artifact_type") != "llm_mean_reversion_signal_rankings"
            or metadata.get("schema_version") != SIGNAL_CACHE_SCHEMA_VERSION
            or metadata.get("model") != model
            or metadata.get("source_dataset") != DATASET_VERSION
            or metadata.get("composite_scores_dir") != str(COMPOSITE_SCORES_DIR)
        ):
            print(
                "⚠️  Existing signal cache uses an incompatible schema or source. "
                "Ignoring it and rebuilding."
            )
            cache = empty_cache

    cache["metadata"] = {
        "schema_version": SIGNAL_CACHE_SCHEMA_VERSION if signal_only else CACHE_SCHEMA_VERSION,
        "artifact_type": (
            "llm_mean_reversion_signal_rankings"
            if signal_only
            else "llm_portfolio_decisions"
        ),
        "start_date": start_date,
        "end_date": end_date,
        "rebalance_days": rebalance_days,
        "model": model,
        "temperature": temperature,
        "top_n_magnitude": effective_top_n,
        "mean_reversion_candidate_mode": mean_reversion_candidate_mode,
        "compact_output": bool(compact_output),
        "signal_only": bool(signal_only),
        "target_positions_per_side": target_positions_per_side,
        "min_decision_price": MIN_TRADE_PRICE,
        "execution_mode": execution_mode,
        "turnover_aware": turnover_aware,
        "strategy_style": strategy_style,
        "strategy_signal_source": _strategy_signal_source(strategy_style),
        "prompt_style": prompt_style_label(turnover_aware),
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "log_llm_trace": bool(log_llm_trace),
        "ambiguous_pool_in_prompt": False,
        "loss_memory_enabled": bool(enable_loss_memory),
        "loss_memory_top_k": int(loss_memory_top_k),
        "loss_memory_max_targets_per_side": int(loss_memory_max_targets_per_side),
        "llm_source_policy": "strict_llm_only",
        "source_dataset": DATASET_VERSION,
        "composite_scores_dir": str(COMPOSITE_SCORES_DIR),
        "membership_gating_enabled": True,
        "membership_path": membership_lookup["path"],
        "total_rebalance_dates": len(rebalance_dates),
    }
    cache.setdefault("failures", {})
    cache.setdefault(cache_entries_key, {})

    loss_memory_store = None
    if enable_loss_memory:
        loss_memory_store = LossMemoryStore(
            top_k=loss_memory_top_k,
            max_targets_per_side=loss_memory_max_targets_per_side,
        )

    # Track previous portfolio for continuity across rebalances
    previous_portfolio = None
    completed = 0
    skipped = 0
    failed = 0

    for i, rebal_date in enumerate(rebalance_dates):
        date_str = pd.Timestamp(rebal_date).strftime("%Y-%m-%d")

        if loss_memory_store is not None and previous_portfolio is not None:
            loss_memory_store.finalize_portfolio(
                entry_portfolio=previous_portfolio,
                exit_date=pd.Timestamp(rebal_date),
                composite_dfs=composite_dfs,
            )

        # Skip if already cached
        if resume and date_str in cache.setdefault(cache_entries_key, {}):
            # Restore previous portfolio from cache for continuity
            if not signal_only:
                previous_portfolio = cache["portfolios"][date_str]
            skipped += 1
            print(f"   ⏩ [{i+1}/{len(rebalance_dates)}] {date_str} — cached, skipping")
            continue

        print(f"\n   🔄 [{i+1}/{len(rebalance_dates)}] {date_str}")

        # Build universe for this date
        universe = load_signal_data_preloaded(
            composite_dfs,
            pd.Timestamp(rebal_date),
            strategy_style=strategy_style,
            membership_lookup=membership_lookup,
        )
        eligible_universe_size = len(universe)
        if str(strategy_style or "").strip().lower() == "mean_reversion":
            eligible_universe_size = _count_mean_reversion_eligible_rows(universe)
        min_universe_size = _minimum_required_universe_size(
            strategy_style,
            effective_top_n,
            target_positions_per_side,
            mean_reversion_candidate_mode,
        )

        if eligible_universe_size < min_universe_size:
            if str(strategy_style or "").strip().lower() == "mean_reversion":
                print(
                    "      ⚠️  Only "
                    f"{eligible_universe_size} MR-eligible stocks available "
                    f"({len(universe)} total), need ≥{min_universe_size}. Skipping."
                )
            else:
                print(
                    f"      ⚠️  Only {eligible_universe_size} stocks available, "
                    f"need ≥{min_universe_size}. Skipping."
                )
            failed += 1
            continue

        # Get regime
        regime = identify_regime(spy_df, pd.Timestamp(rebal_date))

        if str(strategy_style or "").strip().lower() == "mean_reversion":
            print(
                f"      Universe: {eligible_universe_size} MR-eligible / {len(universe)} total stocks"
                f"  |  Regime: {regime}"
            )
        else:
            print(f"      Universe: {len(universe)} stocks  |  Regime: {regime}")

        loss_memory_context = ""
        if loss_memory_store is not None:
            memory_payload = loss_memory_store.build_prompt_context(
                universe=universe,
                previous_portfolio=previous_portfolio if turnover_aware else None,
                regime=regime,
                as_of_date=pd.Timestamp(rebal_date),
                top_n_magnitude=effective_top_n,
            )
            loss_memory_context = memory_payload.get("text", "")
            if memory_payload.get("case_count", 0) > 0:
                print(
                    "      Loss memory: "
                    f"{memory_payload['target_count']} targets, "
                    f"{memory_payload['case_count']} retrieved loss cases"
                )

        # Call LLM
        try:
            t0 = time.time()
            strategy_kwargs = {}
            if signal_only:
                async with httpx.AsyncClient(timeout=technical_llm.PORTFOLIO_REQUEST_TIMEOUT) as client:
                    ranking = await llm_module.construct_signal_ranking_from_universe(
                        client=client,
                        universe=universe,
                        regime=regime,
                        rebalance_id=f"w_{abs(hash(date_str)) % 100000:05d}",
                        model=model,
                        temperature=temperature,
                        log_llm_trace=log_llm_trace,
                    )
                elapsed = time.time() - t0
                n_long = len(ranking.get("long_rankings", []))
                n_short = len(ranking.get("short_rankings", []))
                provider_usage = ranking.get("llm_usage", {}).get("provider_usage", {})
                total_tokens = provider_usage.get("total_tokens")
                if total_tokens is None:
                    total_tokens = ranking.get("llm_usage", {}).get("approx_total_tokens")
                token_segment = f" | Tokens: {int(total_tokens)}" if total_tokens is not None else ""
                signal_entry = {
                    "rebalance_date": date_str,
                    "rebalance_id": f"w_{abs(hash(date_str)) % 100000:05d}",
                    "regime": regime,
                    "llm_source": "llm",
                    "signal_schema_version": SIGNAL_CACHE_SCHEMA_VERSION,
                    **ranking,
                }
                cache["signals"][date_str] = signal_entry
                cache["failures"].pop(date_str, None)
                save_cache(cache, cache_path)
                completed += 1
                print(
                    f"      ✓ ranked {n_long} D1 longs / {n_short} D10 shorts | "
                    f"{elapsed:.1f}s"
                    f"{token_segment}"
                )
                if i < len(rebalance_dates) - 1:
                    await asyncio.sleep(rate_limit_delay)
                continue

            if strategy_style == "mean_reversion":
                strategy_kwargs["candidate_mode"] = mean_reversion_candidate_mode
                strategy_kwargs["compact_output"] = compact_output
            portfolio = await llm_module.construct_portfolio_from_universe(
                universe=universe,
                rebalance_date=date_str,
                previous_portfolio=previous_portfolio,
                regime=regime,
                mask_date=True,
                model=model,
                temperature=temperature,
                top_n_magnitude=effective_top_n,
                target_positions_per_side=target_positions_per_side,
                use_equal_weight=use_equal_weight,
                turnover_aware=turnover_aware,
                loss_memory_context=loss_memory_context,
                log_llm_trace=log_llm_trace,
                **strategy_kwargs,
            )
            elapsed = time.time() - t0

            n_long = len(portfolio.get("long_basket", []))
            n_short = len(portfolio.get("short_basket", []))
            kept = len(portfolio.get("kept_from_previous", []))
            provider_usage = portfolio.get("llm_usage", {}).get("provider_usage", {})
            total_tokens = provider_usage.get("total_tokens")
            if total_tokens is None:
                total_tokens = portfolio.get("llm_usage", {}).get("approx_total_tokens")
            token_segment = f" | Tokens: {int(total_tokens)}" if total_tokens is not None else ""

            print(
                f"      ✓ {n_long}L / {n_short}S | "
                f"Kept: {kept} positions | "
                f"{elapsed:.1f}s"
                f"{token_segment}"
            )

            # Save to cache
            cache["portfolios"][date_str] = portfolio
            cache["failures"].pop(date_str, None)
            save_cache(cache, cache_path)

            # Track for next iteration
            previous_portfolio = portfolio
            completed += 1

        except Exception as e:
            print(f"      ❌ Error: {e}")
            failed += 1
            cache["failures"][date_str] = {
                "rebalance_date": date_str,
                "error": str(e),
                "regime": regime,
                "execution_mode": execution_mode,
                "turnover_aware": turnover_aware,
                "strategy_style": strategy_style,
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            }
            save_cache(cache, cache_path)
            continue

        # Rate limiting
        if i < len(rebalance_dates) - 1:
            await asyncio.sleep(rate_limit_delay)

    # Final summary
    print(f"\n{'='*80}")
    print(f"BATCH COMPLETE")
    print(f"  Completed: {completed}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Cache saved: {cache_path}")
    print(f"{'='*80}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM Portfolio Constructor — Generate LLM portfolio decisions for A/B testing"
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--rebalance-days", type=int, default=10, help="Rebalance frequency (default: 10)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from cache (default: True)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore cache")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="LLM temperature (default: 0.6; ignored when DeepSeek thinking mode is enabled)",
    )
    parser.add_argument("--top-n", type=int, default=60, help="Top-N magnitude candidates for LLM (default: 60)")
    parser.add_argument(
        "--positions-per-side",
        type=int,
        help="Exact number of long and short names the LLM should return each rebalance",
    )
    parser.add_argument(
        "--mean-reversion-candidate-mode",
        choices=("fixed_top_n", "decile"),
        default="fixed_top_n",
        help=(
            "Mean-reversion prompt candidate bucket mode: fixed_top_n shows --top-n names per side; "
            "decile shows exact D1/D10 full-universe buckets"
        ),
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Ask compatible LLM strategy prompts to omit per-name rationales and return compact JSON",
    )
    parser.add_argument(
        "--signal-only",
        action="store_true",
        help="For mean_reversion, cache LLM D1/D10 rankings instead of portfolio baskets",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"DeepSeek model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--rate-limit", type=float, default=2.0, help="Delay between API calls in seconds")
    parser.add_argument("--equal-weight", action="store_true", help="Override LLM sizing and execute each leg equal-weighted")
    parser.add_argument(
        "--strategy-style",
        choices=sorted(SUPPORTED_LLM_STRATEGIES),
        default=DEFAULT_LLM_STRATEGY,
        help="LLM alpha-construction style to run",
    )
    parser.add_argument(
        "--turnover-aware",
        action="store_true",
        help="Tell the LLM to minimize turnover and preserve positions to save trading costs",
    )
    parser.add_argument(
        "--enable-loss-memory",
        action="store_true",
        help="Retrieve top-k similar prior loss cases and include them in the LLM prompt",
    )
    parser.add_argument(
        "--loss-memory-top-k",
        type=int,
        default=3,
        help="Number of similar loss cases to retrieve per target (default: 3)",
    )
    parser.add_argument(
        "--loss-memory-max-targets-per-side",
        type=int,
        default=4,
        help="Maximum reviewed names per side for loss-memory retrieval (default: 4)",
    )
    parser.add_argument(
        "--log-llm-trace",
        action="store_true",
        help="Store the exact prompt messages and raw JSON response for each LLM stage",
    )

    args = parser.parse_args()

    resume = not args.no_resume

    asyncio.run(
        run_batch(
            start_date=args.start,
            end_date=args.end,
            rebalance_days=args.rebalance_days,
            resume=resume,
            temperature=args.temperature,
            top_n_magnitude=args.top_n,
            model=args.model,
            rate_limit_delay=args.rate_limit,
            use_equal_weight=args.equal_weight,
            strategy_style=args.strategy_style,
            target_positions_per_side=args.positions_per_side,
            mean_reversion_candidate_mode=args.mean_reversion_candidate_mode,
            compact_output=args.compact_output,
            signal_only=args.signal_only,
            turnover_aware=args.turnover_aware,
            enable_loss_memory=args.enable_loss_memory,
            loss_memory_top_k=args.loss_memory_top_k,
            loss_memory_max_targets_per_side=args.loss_memory_max_targets_per_side,
            log_llm_trace=args.log_llm_trace,
        )
    )


if __name__ == "__main__":
    main()
