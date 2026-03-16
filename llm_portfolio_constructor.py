"""
LLM Portfolio Constructor — Batch Engine for A/B Testing

For each rebalance date within the test window:
  1. Reads direction_score, magnitude_score, and raw indicators from signal_data_dump CSVs
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
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from portfolio_constraints import MIN_TRADE_PRICE, has_min_trade_price
from technical_llm import construct_portfolio_from_universe

# =============================================================================
# Config
# =============================================================================

COMPOSITE_SCORES_DIR = Path("backtest_results/composite_scores")
SPY_DATA_PATH = COMPOSITE_SCORES_DIR.parent / "signal_data_dump" / "signal_data_dump_SPY_1993-01-01_2025-12-31.csv"
CACHE_DIR = Path("backtest_results/llm_portfolio_cache")

EXCLUDED_TICKERS = {"CPWR", "SPY"}

# The 17 indicators the LLM actually sees (from composite scores)
LLM_INDICATOR_COLS = [
    "vix_percentile_252d",
    "vol_percentile_504d",
    "vix_level",
    "vol_imbalance_20d",
    "m2_m1_ratio",
    "atr_14d_pct",
    "realized_vol_10d",
    "vol_adjusted_trend_ma100",
    "range_pos_55d",
    "pct_from_52w_high",
    "range_pct",
    "autocorr_5d",
    "mom_3m",
    "rsi_5d",
    "vel_10d_over_vol",
    "days_since_52w_high",
    "vol_adjusted_trend_ma100_slope",
    "pct_from_ma100",
]


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


def load_signal_data_preloaded(
    composite_dfs: Dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> Dict[str, Dict[str, Any]]:
    """Build universe dict from pre-loaded composite DataFrames.

    Args:
        composite_dfs: {ticker: DataFrame} with composite scores and indicators
        date: Target date

    Returns:
        {ticker: {direction_score, magnitude_score, indicators: {...}}}
    """
    universe = {}
    for ticker, df in composite_dfs.items():
        if ticker in EXCLUDED_TICKERS:
            continue

        row = df[df["date"] == date]
        if row.empty:
            continue
        row = row.iloc[0]

        direction_score = row.get("direction_score", np.nan)
        magnitude_score = row.get("magnitude_score", np.nan)
        if pd.isna(direction_score) or pd.isna(magnitude_score):
            continue

        adj_close = row.get("adj_close", np.nan)
        if not has_min_trade_price(adj_close):
            continue

        # Extract only the 17 LLM indicators from the composite CSV
        indicators = {}
        for col in LLM_INDICATOR_COLS:
            val = row.get(col, np.nan)
            if pd.notna(val):
                indicators[col] = float(val) if not isinstance(val, str) else val

        universe[ticker] = {
            "direction_score": float(direction_score),
            "magnitude_score": float(magnitude_score),
            "adj_close": float(adj_close),
            "indicators": indicators,
        }

    return universe


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
    for csv_file in sorted(COMPOSITE_SCORES_DIR.glob("*.csv")):
        try:
            parts = csv_file.stem.split("_")
            ticker = parts[2].upper()
            if ticker in EXCLUDED_TICKERS:
                continue
            df = pd.read_csv(csv_file, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            composite_dfs[ticker] = df
        except Exception:
            continue
    print(f"   ✓ Loaded {len(composite_dfs)} composite score files\n")

    return composite_dfs


# =============================================================================
# Cache Management
# =============================================================================

def load_cache(cache_path: Path) -> Dict[str, Any]:
    """Load existing LLM portfolio cache."""
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return {"portfolios": {}, "metadata": {}}


def save_cache(cache: Dict[str, Any], cache_path: Path):
    """Save LLM portfolio cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, default=str)


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
    model: str = "deepseek-chat",
    rate_limit_delay: float = 2.0,
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

    print(f"\n{'='*80}")
    print(f"LLM PORTFOLIO CONSTRUCTOR — BATCH MODE")
    print(f"Test Window: {start_date} to {end_date}")
    print(f"Rebalance: every {rebalance_days} trading days")
    print(f"Model: {model}  |  Temperature: {temperature}")
    print(f"Top-N magnitude candidates shown to LLM: {top_n_magnitude}")
    print(f"Min decision price: > ${MIN_TRADE_PRICE:.2f}")
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

    # Preload composite data
    composite_dfs = preload_all_data()

    # Cache setup
    cache_filename = f"llm_portfolio_{start_date}_{end_date}_{rebalance_days}d.json"
    cache_path = CACHE_DIR / cache_filename
    cache = load_cache(cache_path) if resume else {"portfolios": {}, "metadata": {}}
    cached_min_price = cache.get("metadata", {}).get("min_decision_price")
    if resume and cache.get("portfolios") and cached_min_price != MIN_TRADE_PRICE:
        print(
            "⚠️  Existing cache was built with a different minimum decision price. "
            "Rebuilding LLM portfolios with the new filter."
        )
        cache = {"portfolios": {}, "metadata": {}}

    cache["metadata"] = {
        "start_date": start_date,
        "end_date": end_date,
        "rebalance_days": rebalance_days,
        "model": model,
        "temperature": temperature,
        "top_n_magnitude": top_n_magnitude,
        "min_decision_price": MIN_TRADE_PRICE,
        "total_rebalance_dates": len(rebalance_dates),
    }

    # Track previous portfolio for cost-aware rebalancing
    previous_portfolio = None
    completed = 0
    skipped = 0
    failed = 0

    for i, rebal_date in enumerate(rebalance_dates):
        date_str = pd.Timestamp(rebal_date).strftime("%Y-%m-%d")

        # Skip if already cached
        if resume and date_str in cache["portfolios"]:
            # Restore previous portfolio from cache for continuity
            previous_portfolio = cache["portfolios"][date_str]
            skipped += 1
            print(f"   ⏩ [{i+1}/{len(rebalance_dates)}] {date_str} — cached, skipping")
            continue

        print(f"\n   🔄 [{i+1}/{len(rebalance_dates)}] {date_str}")

        # Build universe for this date
        universe = load_signal_data_preloaded(composite_dfs, pd.Timestamp(rebal_date))

        if len(universe) < 60:
            print(f"      ⚠️  Only {len(universe)} stocks available, need ≥60. Skipping.")
            failed += 1
            continue

        # Get regime
        regime = identify_regime(spy_df, pd.Timestamp(rebal_date))

        print(f"      Universe: {len(universe)} stocks  |  Regime: {regime}")

        # Call LLM
        try:
            t0 = time.time()
            portfolio = await construct_portfolio_from_universe(
                universe=universe,
                rebalance_date=date_str,
                previous_portfolio=previous_portfolio,
                regime=regime,
                mask_date=True,
                model=model,
                temperature=temperature,
                top_n_magnitude=top_n_magnitude,
            )
            elapsed = time.time() - t0

            n_long = len(portfolio.get("long_basket", []))
            n_short = len(portfolio.get("short_basket", []))
            conf = portfolio.get("confidence", 0)
            kept = len(portfolio.get("kept_from_previous", []))

            print(
                f"      ✓ {n_long}L / {n_short}S | "
                f"Confidence: {conf:.2f} | "
                f"Kept: {kept} positions | "
                f"{elapsed:.1f}s"
            )

            # Save to cache
            cache["portfolios"][date_str] = portfolio
            save_cache(cache, cache_path)

            # Track for next iteration
            previous_portfolio = portfolio
            completed += 1

        except Exception as e:
            print(f"      ❌ Error: {e}")
            failed += 1

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
    parser.add_argument("--temperature", type=float, default=0.3, help="LLM temperature (default: 0.3)")
    parser.add_argument("--top-n", type=int, default=60, help="Top-N magnitude candidates for LLM (default: 60)")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model name (default: deepseek-chat)")
    parser.add_argument("--rate-limit", type=float, default=2.0, help="Delay between API calls in seconds")

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
        )
    )


if __name__ == "__main__":
    main()
