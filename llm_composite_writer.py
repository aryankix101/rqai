"""
LLM Composite Writer — Convert LLM Portfolio Decisions to Backtestable CSVs

Reads the cached LLM portfolio JSON (from llm_portfolio_constructor.py) and
produces composite-score-format CSVs that factor_backtest_engine.py can consume.

The LLM's chosen weights are written as direction_score (positive for longs,
negative for shorts, scaled to [-100, +100] range). magnitude_score is set
to the absolute weight scaled to [0, 100].

Usage:
    python llm_composite_writer.py --cache backtest_results/llm_portfolio_cache/llm_portfolio_2019-01-01_2019-12-31_10d.json
    python llm_composite_writer.py --cache <path> --output-dir backtest_results/composite_scores_llm/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# =============================================================================
# Config
# =============================================================================

COMPOSITE_SCORES_DIR = Path("backtest_results/composite_scores")
DEFAULT_OUTPUT_DIR = Path("backtest_results/composite_scores_llm")


# =============================================================================
# Core Logic
# =============================================================================

def write_llm_composites(
    cache_path: str,
    output_dir: str = None,
):
    """Convert LLM portfolio cache to composite score CSVs.

    For each ticker that appears in any LLM portfolio decision, creates a
    composite-score CSV with:
    - direction_score: LLM weight × 100 (positive for long, negative for short)
    - magnitude_score: abs(weight) × 100
    - All other columns (adj_close, forward returns) copied from the
      deterministic composite CSV

    Rows are only included for rebalance dates where the ticker was selected
    by the LLM (or had valid data). Non-selected tickers on a given date
    get direction_score=0, magnitude_score=0 so the backtest engine can
    rank them below selected tickers.

    Args:
        cache_path: Path to LLM portfolio cache JSON
        output_dir: Output directory for LLM composite CSVs
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        print(f"❌ Cache file not found: {cache_path}")
        sys.exit(1)

    with open(cache_path, "r") as f:
        cache = json.load(f)

    portfolios = cache.get("portfolios", {})
    metadata = cache.get("metadata", {})

    if not portfolios:
        print("❌ No portfolio decisions found in cache")
        sys.exit(1)

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"LLM COMPOSITE WRITER")
    print(f"Cache: {cache_path}")
    print(f"Portfolio decisions: {len(portfolios)} rebalance dates")
    print(f"Output: {out_dir}")
    print(f"{'='*80}\n")

    # Collect all rebalance dates from the cache
    rebalance_dates = sorted(portfolios.keys())
    start_date = rebalance_dates[0]
    end_date = rebalance_dates[-1]
    print(f"   Date range: {start_date} to {end_date}")

    # Build a map: {ticker: {date: {weight, side}}} from LLM decisions
    llm_decisions = {}  # ticker -> {date_str: weight}  (positive=long, negative=short)

    for date_str, portfolio in portfolios.items():
        for entry in portfolio.get("long_basket", []):
            ticker = entry.get("ticker", "").upper()
            weight = abs(float(entry.get("weight", 0)))
            if ticker:
                llm_decisions.setdefault(ticker, {})[date_str] = weight

        for entry in portfolio.get("short_basket", []):
            ticker = entry.get("ticker", "").upper()
            weight = -abs(float(entry.get("weight", 0)))
            if ticker:
                llm_decisions.setdefault(ticker, {})[date_str] = weight

    all_llm_tickers = set(llm_decisions.keys())
    print(f"   Unique tickers across all decisions: {len(all_llm_tickers)}")

    # Load all deterministic composite CSVs and write LLM versions
    composite_files = sorted(COMPOSITE_SCORES_DIR.glob("*.csv"))
    written = 0
    skipped = 0

    for csv_file in composite_files:
        try:
            parts = csv_file.stem.split("_")
            ticker = parts[2].upper()
        except (IndexError, ValueError):
            continue

        # Load deterministic composite
        df = pd.read_csv(csv_file, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Filter to test window dates
        df = df[
            (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        ].copy()

        if df.empty:
            skipped += 1
            continue

        # Assign LLM scores
        ticker_decisions = llm_decisions.get(ticker, {})

        def get_llm_direction(row):
            date_str = row["date"].strftime("%Y-%m-%d")
            weight = ticker_decisions.get(date_str, 0.0)
            return weight * 100.0  # Scale to [-100, +100] range

        def get_llm_magnitude(row):
            date_str = row["date"].strftime("%Y-%m-%d")
            weight = ticker_decisions.get(date_str, 0.0)
            return abs(weight) * 100.0  # Scale to [0, 100] range

        df["direction_score"] = df.apply(get_llm_direction, axis=1)
        df["magnitude_score"] = df.apply(get_llm_magnitude, axis=1)

        # Write output CSV (same filename convention)
        out_file = out_dir / csv_file.name
        df.to_csv(out_file, index=False)
        written += 1

    print(f"\n   ✓ Written: {written} CSVs")
    print(f"   ⏩ Skipped (no data in window): {skipped}")
    print(f"   📁 Output directory: {out_dir}\n")

    # Summary of weight distributions
    all_weights = []
    for date_str, portfolio in portfolios.items():
        for entry in portfolio.get("long_basket", []):
            all_weights.append(float(entry.get("weight", 0)))
        for entry in portfolio.get("short_basket", []):
            all_weights.append(-float(entry.get("weight", 0)))

    if all_weights:
        all_weights = np.array(all_weights)
        print(f"   Weight stats across all decisions:")
        print(f"     Long  weights: mean={all_weights[all_weights>0].mean():.4f}, "
              f"max={all_weights[all_weights>0].max():.4f}, "
              f"count={np.sum(all_weights>0)}")
        print(f"     Short weights: mean={all_weights[all_weights<0].mean():.4f}, "
              f"min={all_weights[all_weights<0].min():.4f}, "
              f"count={np.sum(all_weights<0)}")

    # Write a metadata summary
    summary = {
        "cache_file": str(cache_path),
        "output_dir": str(out_dir),
        "date_range": f"{start_date} to {end_date}",
        "num_rebalance_dates": len(rebalance_dates),
        "num_unique_tickers": len(all_llm_tickers),
        "num_csvs_written": written,
        "metadata": metadata,
    }
    summary_path = out_dir / "llm_composite_metadata.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   📄 Metadata saved: {summary_path}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert LLM portfolio cache to composite score CSVs for backtesting"
    )
    parser.add_argument(
        "--cache", required=True,
        help="Path to LLM portfolio cache JSON file"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Output directory for LLM composite CSVs (default: {DEFAULT_OUTPUT_DIR})"
    )

    args = parser.parse_args()
    write_llm_composites(cache_path=args.cache, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
