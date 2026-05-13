"""
LLM Composite Writer — Convert LLM Portfolio Decisions to Backtestable CSVs

Reads the cached LLM portfolio JSON (from llm_portfolio_constructor.py) and
produces composite-score-format CSVs that factor_backtest_engine.py can consume.

Selected LLM names keep their underlying deterministic direction/magnitude
scores. Execution intent is carried in explicit metadata columns so the
backtest engine does not need to infer long/short intent from score values.

Usage:
    python llm_composite_writer.py --cache backtest_results/llm_portfolio_cache/llm_portfolio_2019-01-01_2019-12-31_10d.json
    python llm_composite_writer.py --cache <path> --output-dir backtest_results/composite_scores_llm/
"""

import argparse
import json
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from data_source_config import DATASET_VERSION, DEFAULT_COMPOSITE_SCORES_DIR, DEFAULT_LLM_COMPOSITE_SCORES_DIR

# =============================================================================
# Config
# =============================================================================

COMPOSITE_SCORES_DIR = DEFAULT_COMPOSITE_SCORES_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_LLM_COMPOSITE_SCORES_DIR
CACHE_SCHEMA_VERSION = 3


# =============================================================================
# Core Logic
# =============================================================================


def _promote_staged_directory(staging_dir: Path, output_dir: Path) -> None:
    backup_dir = output_dir.with_name(f"{output_dir.name}.__previous__")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    if output_dir.exists():
        output_dir.rename(backup_dir)

    try:
        staging_dir.rename(output_dir)
    except Exception:
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

def write_llm_composites(
    cache_path: str,
    output_dir: str = None,
):
    """Convert LLM portfolio cache to composite score CSVs.

    For each ticker that appears in the deterministic composite directory,
    creates an LLM-aware composite-score CSV with:
    - selected rows keeping deterministic direction/magnitude scores
    - unselected rows neutralized to zero execution scores
    - explicit LLM metadata columns for selection, side, source, and weights

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
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        print("❌ Cache uses a legacy schema. Rebuild the LLM cache before writing composites.")
        sys.exit(1)
    if metadata.get("llm_source_policy") != "strict_llm_only":
        print("❌ Cache source policy is not strict LLM-only. Refusing to write legacy/fallback composites.")
        sys.exit(1)
    if metadata.get("source_dataset") != DATASET_VERSION:
        print("❌ Cache dataset source is incompatible with the active Russell close-only pipeline.")
        sys.exit(1)

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}.tmp.", dir=str(out_dir.parent)))

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

    # Build a map: {ticker: {date: execution metadata}} from LLM decisions
    llm_decisions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    selected_rows = []

    for date_str, portfolio in portfolios.items():
        if portfolio.get("llm_source") != "llm":
            print(f"❌ Non-LLM portfolio found in cache for {date_str}; rebuild cache.")
            sys.exit(1)
        execution_mode = portfolio.get("execution_mode", metadata.get("execution_mode", "equal_weight"))
        for side, basket_key in (("LONG", "long_basket"), ("SHORT", "short_basket")):
            for entry in portfolio.get(basket_key, []):
                ticker = entry.get("ticker", "").upper()
                if ticker:
                    selected_rows.append(
                        {
                            "date": date_str,
                            "ticker": ticker,
                            "side": side,
                            "weight_exec": abs(float(entry.get("weight_exec", entry.get("weight", 0)))),
                        }
                    )

        for entry in portfolio.get("long_basket", []):
            ticker = entry.get("ticker", "").upper()
            weight_raw = abs(float(entry.get("weight_raw", entry.get("weight", 0))))
            weight_exec = abs(float(entry.get("weight_exec", entry.get("weight", 0))))
            if ticker:
                llm_decisions.setdefault(ticker, {})[date_str] = {
                    "llm_selected": 1,
                    "llm_side": "LONG",
                    "llm_weight_raw": weight_raw,
                    "llm_weight_exec": weight_exec,
                    "llm_source": "llm",
                    "llm_execution_mode": execution_mode,
                }

        for entry in portfolio.get("short_basket", []):
            ticker = entry.get("ticker", "").upper()
            weight_raw = abs(float(entry.get("weight_raw", entry.get("weight", 0))))
            weight_exec = abs(float(entry.get("weight_exec", entry.get("weight", 0))))
            if ticker:
                llm_decisions.setdefault(ticker, {})[date_str] = {
                    "llm_selected": 1,
                    "llm_side": "SHORT",
                    "llm_weight_raw": weight_raw,
                    "llm_weight_exec": weight_exec,
                    "llm_source": "llm",
                    "llm_execution_mode": execution_mode,
                }

    all_llm_tickers = set(llm_decisions.keys())
    print(f"   Unique tickers across all decisions: {len(all_llm_tickers)}")

    # Load all deterministic composite CSVs and write LLM versions
    composite_files = sorted(COMPOSITE_SCORES_DIR.glob("*.csv"))
    written = 0
    skipped = 0
    expected = 0

    coverage_rows = []
    selected_by_ticker: Dict[str, list[dict[str, Any]]] = {}
    for selected in selected_rows:
        selected_by_ticker.setdefault(selected["ticker"], []).append(selected)

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

        expected += 1

        # Preserve raw deterministic scores for analysis.
        df["base_direction_score"] = df["direction_score"]
        df["base_magnitude_score"] = df["magnitude_score"]

        # Assign explicit LLM execution metadata.
        ticker_decisions = llm_decisions.get(ticker, {})
        default_mode = metadata.get("execution_mode", "equal_weight")
        date_keys = df["date"].dt.strftime("%Y-%m-%d")
        df["llm_selected"] = date_keys.map(lambda d: int(ticker_decisions.get(d, {}).get("llm_selected", 0)))
        df["llm_side"] = date_keys.map(lambda d: ticker_decisions.get(d, {}).get("llm_side", ""))
        df["llm_weight_raw"] = date_keys.map(lambda d: float(ticker_decisions.get(d, {}).get("llm_weight_raw", 0.0)))
        df["llm_weight_exec"] = date_keys.map(lambda d: float(ticker_decisions.get(d, {}).get("llm_weight_exec", 0.0)))
        df["llm_source"] = date_keys.map(lambda d: ticker_decisions.get(d, {}).get("llm_source", ""))
        df["llm_execution_mode"] = date_keys.map(lambda d: ticker_decisions.get(d, {}).get("llm_execution_mode", default_mode))

        selected_mask = df["llm_selected"] == 1
        df.loc[~selected_mask, "direction_score"] = 0.0
        df.loc[~selected_mask, "magnitude_score"] = 0.0

        # Write output CSV (same filename convention)
        out_file = staging_dir / csv_file.name
        df.to_csv(out_file, index=False)
        written += 1

        selected_for_ticker = selected_by_ticker.pop(ticker, [])
        if selected_for_ticker:
            available_date_keys = set(df["date"].dt.strftime("%Y-%m-%d"))
            for selected in selected_for_ticker:
                has_composite_row = selected["date"] in available_date_keys
                coverage_rows.append(
                    {
                        **selected,
                        "has_composite_file": True,
                        "has_composite_row": has_composite_row,
                        "coverage_reason": "covered" if has_composite_row else "missing_composite_row",
                    }
                )

    for selected_for_ticker in selected_by_ticker.values():
        for selected in selected_for_ticker:
            coverage_rows.append(
                {
                    **selected,
                    "has_composite_file": False,
                    "has_composite_row": False,
                    "coverage_reason": "missing_composite_file",
                }
            )

    if coverage_rows:
        pd.DataFrame(coverage_rows).to_csv(staging_dir / "llm_composite_coverage.csv", index=False)

    if written == 0:
        raise RuntimeError(
            "LLM composite writer produced zero CSVs. "
            "Refusing to replace the previous output directory."
        )
    if expected != written:
        raise RuntimeError(
            f"LLM composite writer expected to write {expected} CSVs but only wrote {written}. "
            "Refusing to publish a partial output directory."
        )

    _promote_staged_directory(staging_dir, out_dir)

    print(f"\n   ✓ Written: {written} CSVs")
    print(f"   ⏩ Skipped (no data in window): {skipped}")
    print(f"   📁 Output directory: {out_dir}\n")

    # Summary of weight distributions
    all_weights = []
    for date_str, portfolio in portfolios.items():
        for entry in portfolio.get("long_basket", []):
            all_weights.append(float(entry.get("weight_exec", entry.get("weight", 0))))
        for entry in portfolio.get("short_basket", []):
            all_weights.append(-float(entry.get("weight_exec", entry.get("weight", 0))))

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
        "num_expected_csvs": expected,
        "num_csvs_written": written,
        "source_dataset": DATASET_VERSION,
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
