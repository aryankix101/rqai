"""Guarded online fill for the controlled LLM reversal replay.

This is intentionally separate from the offline replay audit. It may import the
LLM runner and create an HTTP client, but only after the frozen candidate grid is
validated and the caller passes --confirm-online-llm-calls.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import run_backtest_v3 as rb
from scripts import run_llm_reversal_controlled_replay_offline as replay


DEFAULT_MODEL = "deepseek-reasoner"
DEFAULT_TEMPERATURE = 0.6
DEFAULT_CACHE_DIR = Path("backtest_results/llm_reversal_signal_v1_cache")
DEFAULT_CANDIDATE_GRID = replay.DEFAULT_OUTPUT_DIR / "candidate_grid.csv"
DEFAULT_PROMPT_STYLE = "v1_score"


def import_v1_runner():
    return importlib.import_module("mean_reversion_llm_v1")


def normalize_key_rows(rows: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["rebalance_date", "ticker", "pool_side"]
    normalized = rows.loc[:, key_cols].copy()
    normalized["rebalance_date"] = pd.to_datetime(normalized["rebalance_date"]).dt.strftime("%Y-%m-%d")
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper().str.strip()
    normalized["pool_side"] = normalized["pool_side"].astype(str).str.upper().str.strip()
    return normalized.sort_values(key_cols).reset_index(drop=True)


def candidate_grid_for_requested_dates(candidate_grid: pd.DataFrame, max_dates: int | None = None) -> pd.DataFrame:
    grid = candidate_grid.copy()
    grid["rebalance_date"] = pd.to_datetime(grid["rebalance_date"]).dt.strftime("%Y-%m-%d")
    dates = sorted(grid["rebalance_date"].unique())
    if max_dates is not None:
        if max_dates <= 0:
            raise ValueError("--max-dates must be positive when provided")
        dates = dates[:max_dates]
    return grid[grid["rebalance_date"].isin(dates)].copy()


def pool_key_rows(pools: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for pool in pools:
        date = pd.Timestamp(pool.signal_date).date().isoformat()
        rows.extend({"rebalance_date": date, "ticker": ticker, "pool_side": "LONG"} for ticker in pool.long_tickers)
        rows.extend({"rebalance_date": date, "ticker": ticker, "pool_side": "SHORT"} for ticker in pool.short_tickers)
    return pd.DataFrame(rows)


def validate_pools_against_grid(pools: list[Any], candidate_grid: pd.DataFrame) -> dict[str, Any]:
    actual = normalize_key_rows(pool_key_rows(pools))
    expected = normalize_key_rows(candidate_grid)
    key_cols = ["rebalance_date", "ticker", "pool_side"]
    expected_index = pd.MultiIndex.from_frame(expected)
    actual_index = pd.MultiIndex.from_frame(actual)
    missing = expected_index.difference(actual_index)
    unexpected = actual_index.difference(expected_index)
    duplicate_count = int(actual.duplicated(key_cols).sum())
    expected_hash = replay.candidate_pool_hash(expected)
    actual_hash = replay.candidate_pool_hash(actual) if not actual.empty else ""
    audit = {
        "status": "valid",
        "row_count": int(len(actual)),
        "date_count": int(actual["rebalance_date"].nunique()) if not actual.empty else 0,
        "expected_row_count": int(len(expected)),
        "expected_date_count": int(expected["rebalance_date"].nunique()),
        "duplicate_key_rows": duplicate_count,
        "missing_rows": int(len(missing)),
        "unexpected_rows": int(len(unexpected)),
        "candidate_pool_hash": actual_hash,
        "expected_candidate_pool_hash": expected_hash,
        "reason": "",
    }
    if duplicate_count:
        audit["status"] = "invalid"
        audit["reason"] = f"duplicate key rows: {duplicate_count}"
    elif len(missing) or len(unexpected):
        audit["status"] = "invalid"
        audit["reason"] = f"missing={len(missing)} unexpected={len(unexpected)}"
    elif actual_hash != expected_hash:
        audit["status"] = "invalid"
        audit["reason"] = "candidate pool hash mismatch"
    if audit["status"] != "valid":
        raise ValueError(f"Rebuilt LLM pools do not match frozen grid: {audit['reason']}")
    return audit


def cache_path_for_args(args: argparse.Namespace) -> Path:
    if args.cache_path is not None:
        return Path(args.cache_path)
    start = str(pd.Timestamp(args.start).date())
    end = str(pd.Timestamp(args.end).date())
    model = str(args.model).replace("/", "-")
    return Path(args.cache_dir) / f"llm_reversal_signal_v1_{start}_{end}_pool{args.pool_per_side}_mdl-{model}.json"


def build_config(args: argparse.Namespace) -> rb.BenchmarkConfig:
    return rb.BenchmarkConfig(
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        output_dir=Path(args.output_dir),
        membership_path=Path(args.membership_path),
        cost_bps_levels=(0.0, 2.5, 5.0),
        cadence_mode="weekly_friday",
        signal_lookback_days=5,
        rebalance_days=5,
        holding_days=5,
        positions_per_side=None,
    )


def build_matching_pools(args: argparse.Namespace, candidate_grid: pd.DataFrame) -> tuple[list[Any], dict[str, Any]]:
    v1 = import_v1_runner()
    config = build_config(args)
    _, membership_lookup, market_frames, feature_frames, spy_df = v1.load_experiment_inputs(config)
    context = v1.build_pool_context(market_frames)
    pools = []
    dates = sorted(pd.to_datetime(candidate_grid["rebalance_date"]).dt.strftime("%Y-%m-%d").unique())
    for date in dates:
        pool = v1.build_candidate_pool(
            market_frames=market_frames,
            feature_frames=feature_frames,
            membership_lookup=membership_lookup,
            spy_df=spy_df,
            signal_date=pd.Timestamp(date),
            pool_per_side=int(args.pool_per_side),
            config=config,
            context=context,
        )
        if pool is None:
            raise ValueError(f"Could not rebuild controlled candidate pool for {date}")
        pools.append(pool)
    return pools, validate_pools_against_grid(pools, candidate_grid)


def summarize_cache_coverage(
    pools: list[Any],
    cache_path: Path,
    model: str,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> dict[str, Any]:
    v1 = import_v1_runner()
    cache = v1.load_cache(cache_path)
    prompt_version = v1.prompt_version_for_style(prompt_style)
    cached_dates: list[str] = []
    missing_dates: list[str] = []
    invalid_dates: list[dict[str, str]] = []
    signals = cache.get("signals", {})
    if not isinstance(signals, dict):
        signals = {}
    for pool in pools:
        date_key = str(pd.Timestamp(pool.signal_date).date())
        entry = signals.get(date_key)
        if not isinstance(entry, dict):
            missing_dates.append(date_key)
            continue
        try:
            cached = v1._entry_from_cache(entry, pool, model, prompt_version, prompt_style)
        except ValueError as exc:
            invalid_dates.append({"date": date_key, "reason": str(exc)})
            continue
        if cached is None:
            invalid_dates.append({"date": date_key, "reason": "cache key mismatch"})
        else:
            cached_dates.append(date_key)
    return {
        "cache_path": str(cache_path),
        "prompt_style": prompt_style,
        "prompt_version": prompt_version,
        "model": model,
        "cached_count": len(cached_dates),
        "missing_count": len(missing_dates),
        "invalid_count": len(invalid_dates),
        "pending_llm_call_count": len(missing_dates) + len(invalid_dates),
        "cached_dates": cached_dates,
        "missing_dates": missing_dates,
        "invalid_dates": invalid_dates,
    }


def write_fill_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "online_fill_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=replay._json_default)


def refresh_offline_audit(args: argparse.Namespace, direct_cache: Path) -> dict[str, Any]:
    offline_args = replay.parse_args(
        [
            "--reuse-candidate-grid",
            "--output-dir",
            str(args.output_dir),
            "--membership-path",
            str(args.membership_path),
            "--pool-per-side",
            str(args.pool_per_side),
            "--positions-per-side",
            str(args.positions_per_side),
            "--direct-score-cache",
            str(direct_cache),
        ]
    )
    return replay.run_offline_replay(offline_args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill controlled replay LLM scores behind strict audit gates.")
    parser.add_argument("--start", default=replay.DEFAULT_START)
    parser.add_argument("--end", default=replay.DEFAULT_END)
    parser.add_argument("--candidate-grid", type=Path, default=DEFAULT_CANDIDATE_GRID)
    parser.add_argument("--output-dir", type=Path, default=replay.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--membership-path", type=Path, default=rb.DEFAULT_MEMBERSHIP_PATH)
    parser.add_argument("--pool-per-side", type=int, default=60)
    parser.add_argument("--positions-per-side", type=int, default=30)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--rate-limit", type=float, default=2.0)
    parser.add_argument("--prompt-style", choices=(DEFAULT_PROMPT_STYLE,), default=DEFAULT_PROMPT_STYLE)
    parser.add_argument("--max-dates", type=int, default=None, help="Optional small fill subset for a guarded smoke run.")
    parser.add_argument("--dry-run", action="store_true", help="Audit pools and cache coverage without creating an HTTP client.")
    parser.add_argument(
        "--confirm-online-llm-calls",
        action="store_true",
        help="Required to spend LLM calls for missing/invalid cache entries.",
    )
    parser.add_argument(
        "--skip-refresh-offline-audit",
        action="store_true",
        help="Do not rerun the offline scorer-matrix audit after a successful full fill.",
    )
    return parser.parse_args(argv)


def run_fill(args: argparse.Namespace) -> dict[str, Any]:
    if args.pool_per_side <= 0 or args.positions_per_side <= 0:
        raise ValueError("--pool-per-side and --positions-per-side must be positive")
    if args.positions_per_side > args.pool_per_side:
        raise ValueError("--positions-per-side cannot exceed --pool-per-side")
    if not Path(args.candidate_grid).exists():
        raise FileNotFoundError(f"Frozen candidate grid not found: {args.candidate_grid}")

    candidate_grid = candidate_grid_for_requested_dates(pd.read_csv(args.candidate_grid), args.max_dates)
    cache_path = cache_path_for_args(args)
    pools, pool_audit = build_matching_pools(args, candidate_grid)
    before = summarize_cache_coverage(pools, cache_path, args.model, args.prompt_style)

    summary: dict[str, Any] = {
        "status": "DRY_RUN" if args.dry_run else "READY",
        "candidate_grid": {
            "path": str(args.candidate_grid),
            "row_count": int(len(candidate_grid)),
            "date_count": int(candidate_grid["rebalance_date"].nunique()),
            "candidate_pool_hash": replay.candidate_pool_hash(candidate_grid),
            "subset_max_dates": args.max_dates,
        },
        "pool_audit": pool_audit,
        "cache_coverage_before": before,
        "llm_calls_attempted": 0,
        "http_client_created": False,
    }

    if args.dry_run:
        write_fill_summary(Path(args.output_dir), summary)
        return summary
    if before["pending_llm_call_count"] > 0 and not args.confirm_online_llm_calls:
        summary["status"] = "BLOCKED_CONFIRMATION_REQUIRED"
        write_fill_summary(Path(args.output_dir), summary)
        raise RuntimeError(
            "Missing/invalid LLM scores are present. Re-run with --confirm-online-llm-calls "
            "to intentionally spend calls."
        )
    if before["pending_llm_call_count"] > 0:
        v1 = import_v1_runner()
        asyncio.run(
            v1.collect_llm_scores_for_pools(
                pools=pools,
                cache_path=cache_path,
                model=args.model,
                temperature=float(args.temperature),
                rate_limit=float(args.rate_limit),
                prompt_style=args.prompt_style,
            )
        )
        summary["llm_calls_attempted"] = int(before["pending_llm_call_count"])
        summary["http_client_created"] = True
    after = summarize_cache_coverage(pools, cache_path, args.model, args.prompt_style)
    summary["cache_coverage_after"] = after
    summary["status"] = "FILLED" if after["pending_llm_call_count"] == 0 else "INCOMPLETE"

    if (
        summary["status"] == "FILLED"
        and args.max_dates is None
        and not args.skip_refresh_offline_audit
    ):
        offline_coverage = refresh_offline_audit(args, cache_path)
        summary["offline_audit_status_after_fill"] = offline_coverage.get("summary", {}).get("status")

    write_fill_summary(Path(args.output_dir), summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_fill(args)
    print(f"Controlled replay online fill: {summary['status']}")
    print(f"Cache: {cache_path_for_args(args)}")
    print(f"Pending LLM calls: {summary.get('cache_coverage_after', summary['cache_coverage_before'])['pending_llm_call_count']}")


if __name__ == "__main__":
    main()
