"""
LLM Signal Composite Writer.

Converts signal-only mean-reversion ranking caches into composite-score CSVs
that factor_backtest_engine.py can consume. Unlike llm_composite_writer.py,
this path does not require LLM portfolio baskets or weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_source_config import (
    DATASET_VERSION,
    DEFAULT_COMPOSITE_SCORES_DIR,
    DEFAULT_LLM_COMPOSITE_SCORES_DIR,
)

COMPOSITE_SCORES_DIR = DEFAULT_COMPOSITE_SCORES_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_LLM_COMPOSITE_SCORES_DIR.with_name(
    f"{DEFAULT_LLM_COMPOSITE_SCORES_DIR.name}_signal"
)
SIGNAL_CACHE_SCHEMA_VERSION = 1
LLM_SIGNAL_SCORE_COLUMNS = (
    "llm_raw_confidence_score",
    "llm_rank_score",
    "llm_rank_confidence_score",
)
DETERMINISTIC_MR_SCORE_COLUMNS = (
    "mean_reversion_raw_score",
    "mean_reversion_rank_score",
)


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


def _rank_component(rank: int, side_count: int) -> float:
    if side_count <= 0:
        return 0.0
    return float((side_count - int(rank) + 1) / side_count * 100.0)


def transform_signal_rankings(signals: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    """Build ticker/date score maps from signal-only LLM rankings."""
    decisions: dict[str, dict[str, dict[str, float]]] = {}
    for date_str, signal in signals.items():
        for side_key, sign in (("long_rankings", 1.0), ("short_rankings", -1.0)):
            rankings = signal.get(side_key, [])
            side_count = len(rankings)
            for entry in rankings:
                ticker = str(entry.get("ticker", "")).upper().strip()
                if not ticker:
                    continue
                rank_score = _rank_component(int(entry.get("rank", 0)), side_count)
                confidence = float(entry.get("confidence_score", 0.0))
                confidence = min(100.0, max(0.0, confidence))
                decisions.setdefault(ticker, {})[date_str] = {
                    "llm_raw_confidence_score": sign * confidence,
                    "llm_rank_score": sign * rank_score,
                    "llm_rank_confidence_score": sign * rank_score * (confidence / 100.0),
                    "llm_signal_rank": float(entry.get("rank", np.nan)),
                    "llm_confidence_score": confidence,
                    "llm_signal_side": "LONG" if sign > 0 else "SHORT",
                }
    return decisions


def _ticker_from_composite_name(path: Path) -> str:
    return path.stem.split("_")[2].upper()


def _add_deterministic_mean_reversion_raw_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = pd.to_numeric(df["adj_close"], errors="coerce")
    df["mean_reversion_raw_score"] = -(close / close.shift(5) - 1.0) * 100.0
    return df


def _add_deterministic_mean_reversion_scores(frames: dict[str, pd.DataFrame]) -> None:
    raw_rows = []
    for ticker, df in frames.items():
        scored = _add_deterministic_mean_reversion_raw_score(df)
        df["mean_reversion_raw_score"] = scored["mean_reversion_raw_score"]
        raw_rows.append(df[["date", "mean_reversion_raw_score"]].assign(ticker=ticker))

    if not raw_rows:
        return
    all_scores = pd.concat(raw_rows, ignore_index=True)
    all_scores = all_scores.dropna(subset=["mean_reversion_raw_score"])
    rank_map: dict[tuple[str, pd.Timestamp], float] = {}
    for date, group in all_scores.groupby("date"):
        ordered = group.sort_values(
            ["mean_reversion_raw_score", "ticker"],
            ascending=[False, True],
        ).reset_index(drop=True)
        n = len(ordered)
        if n == 1:
            values = [0.0]
        else:
            values = [100.0 - 200.0 * (idx / (n - 1)) for idx in range(n)]
        for (_, row), value in zip(ordered.iterrows(), values):
            rank_map[(str(row["ticker"]).upper(), pd.Timestamp(date))] = float(value)

    for ticker, df in frames.items():
        df["mean_reversion_rank_score"] = [
            rank_map.get((ticker, pd.Timestamp(date)), np.nan)
            for date in pd.to_datetime(df["date"])
        ]


def write_llm_signal_composites(cache_path: str, output_dir: str | None = None) -> None:
    cache_path_obj = Path(cache_path)
    if not cache_path_obj.exists():
        print(f"❌ Cache file not found: {cache_path_obj}")
        sys.exit(1)

    with cache_path_obj.open("r", encoding="utf-8") as handle:
        cache = json.load(handle)

    metadata = cache.get("metadata", {})
    signals = cache.get("signals", {})
    if not signals:
        print("❌ No signal rankings found in cache")
        sys.exit(1)
    if metadata.get("artifact_type") != "llm_mean_reversion_signal_rankings":
        print("❌ Cache is not a signal-only mean-reversion ranking artifact")
        sys.exit(1)
    if metadata.get("schema_version") != SIGNAL_CACHE_SCHEMA_VERSION:
        print("❌ Signal cache uses an unsupported schema version")
        sys.exit(1)
    if metadata.get("source_dataset") != DATASET_VERSION:
        print("❌ Cache dataset source is incompatible with the active pipeline")
        sys.exit(1)

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}.tmp.", dir=str(out_dir.parent)))

    signal_decisions = transform_signal_rankings(signals)
    rebalance_dates = sorted(signals)
    start_date = pd.Timestamp(rebalance_dates[0])
    end_date = pd.Timestamp(rebalance_dates[-1])

    print(f"\n{'=' * 80}")
    print("LLM SIGNAL COMPOSITE WRITER")
    print(f"Cache: {cache_path_obj}")
    print(f"Signal dates: {len(rebalance_dates)}")
    print(f"Output: {out_dir}")
    print(f"{'=' * 80}\n")

    composite_files: list[tuple[str, Path]] = []
    raw_rows = []
    for csv_file in sorted(COMPOSITE_SCORES_DIR.glob("*.csv")):
        try:
            ticker = _ticker_from_composite_name(csv_file)
            df = pd.read_csv(csv_file, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        except Exception:
            continue
        if df.empty:
            continue
        scored = _add_deterministic_mean_reversion_raw_score(df)
        window_scores = scored[
            (scored["date"] >= start_date)
            & (scored["date"] <= end_date)
        ][["date", "mean_reversion_raw_score"]].dropna(subset=["mean_reversion_raw_score"])
        if window_scores.empty:
            continue
        composite_files.append((ticker, csv_file))
        raw_rows.append(window_scores.assign(ticker=ticker))

    if not raw_rows:
        raise RuntimeError("Signal composite writer found no source rows in the cache date window")

    # Build cross-sectional deterministic MR ranks from compact rows only. This
    # keeps VM memory low while preserving full-history 5d lookbacks.
    all_scores = pd.concat(raw_rows, ignore_index=True)
    rank_map: dict[tuple[str, pd.Timestamp], float] = {}
    for date, group in all_scores.groupby("date"):
        ordered = group.sort_values(
            ["mean_reversion_raw_score", "ticker"],
            ascending=[False, True],
        ).reset_index(drop=True)
        n = len(ordered)
        if n == 1:
            values = [0.0]
        else:
            values = [100.0 - 200.0 * (idx / (n - 1)) for idx in range(n)]
        for (_, row), value in zip(ordered.iterrows(), values):
            rank_map[(str(row["ticker"]).upper(), pd.Timestamp(date))] = float(value)

    del raw_rows, all_scores

    coverage_rows = []
    written = 0
    covered_tickers = set()
    for ticker, csv_file in composite_files:
        df = pd.read_csv(csv_file, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        df = _add_deterministic_mean_reversion_raw_score(df)
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
        if df.empty:
            continue
        covered_tickers.add(ticker)
        df["mean_reversion_rank_score"] = [
            rank_map.get((ticker, pd.Timestamp(date)), np.nan)
            for date in pd.to_datetime(df["date"])
        ]
        if "direction_score" in df.columns:
            df["base_direction_score"] = df["direction_score"]
        if "magnitude_score" in df.columns:
            df["base_magnitude_score"] = df["magnitude_score"]

        ticker_decisions = signal_decisions.get(ticker, {})
        date_keys = df["date"].dt.strftime("%Y-%m-%d")
        for column in LLM_SIGNAL_SCORE_COLUMNS:
            df[column] = date_keys.map(lambda d, col=column: ticker_decisions.get(d, {}).get(col, 0.0))
        df["llm_signal_selected"] = date_keys.map(lambda d: int(d in ticker_decisions))
        df["llm_signal_side"] = date_keys.map(lambda d: ticker_decisions.get(d, {}).get("llm_signal_side", ""))
        df["llm_signal_rank"] = date_keys.map(lambda d: ticker_decisions.get(d, {}).get("llm_signal_rank", np.nan))
        df["llm_confidence_score"] = date_keys.map(lambda d: ticker_decisions.get(d, {}).get("llm_confidence_score", np.nan))

        for date_str, decision in ticker_decisions.items():
            coverage_rows.append(
                {
                    "date": date_str,
                    "ticker": ticker,
                    "side": decision.get("llm_signal_side", ""),
                    "rank": decision.get("llm_signal_rank", np.nan),
                    "confidence_score": decision.get("llm_confidence_score", np.nan),
                    "has_composite_file": True,
                    "has_composite_row": date_str in set(date_keys),
                }
            )

        df.to_csv(staging_dir / csv_file.name, index=False)
        written += 1

    missing_tickers = set(signal_decisions) - covered_tickers
    for ticker in sorted(missing_tickers):
        for date_str, decision in signal_decisions[ticker].items():
            coverage_rows.append(
                {
                    "date": date_str,
                    "ticker": ticker,
                    "side": decision.get("llm_signal_side", ""),
                    "rank": decision.get("llm_signal_rank", np.nan),
                    "confidence_score": decision.get("llm_confidence_score", np.nan),
                    "has_composite_file": False,
                    "has_composite_row": False,
                }
            )

    if written == 0:
        raise RuntimeError("Signal composite writer produced zero CSVs")
    if coverage_rows:
        pd.DataFrame(coverage_rows).to_csv(staging_dir / "llm_signal_coverage.csv", index=False)

    summary = {
        "cache_file": str(cache_path_obj),
        "output_dir": str(out_dir),
        "date_range": f"{start_date.date()} to {end_date.date()}",
        "num_signal_dates": len(rebalance_dates),
        "num_csvs_written": written,
        "score_columns": list(DETERMINISTIC_MR_SCORE_COLUMNS + LLM_SIGNAL_SCORE_COLUMNS),
        "source_dataset": DATASET_VERSION,
        "metadata": metadata,
    }
    with (staging_dir / "llm_signal_composite_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    _promote_staged_directory(staging_dir, out_dir)
    print(f"   ✓ Written: {written} CSVs")
    print(f"   📁 Output directory: {out_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LLM signal-only ranking cache to composite score CSVs"
    )
    parser.add_argument("--cache", required=True, help="Path to signal-only LLM cache JSON")
    parser.add_argument("--output-dir", default=None, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()
    write_llm_signal_composites(cache_path=args.cache, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
