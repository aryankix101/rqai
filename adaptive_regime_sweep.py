#!/usr/bin/env python3
"""Batch runner for adaptive regime review cadence sweeps."""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).parent
BACKTEST_DIR = BASE_DIR / "backtest_results" / "factor_backtest"
SWEEP_DIR = BASE_DIR / "backtest_results" / "regime_sweeps"


def parse_int_list(value: str) -> list[int]:
    values = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return values


def default_python_bin() -> str:
    venv_python = BASE_DIR / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def build_variant_label(prefix: str, normal: int, grind_up: int, slow_bleed: int, unknown: int) -> str:
    return f"{prefix}_n{normal}_g{grind_up}_sb{slow_bleed}_u{unknown}"


def summarize_result(variant_label: str, forward_days: int, score_type: str) -> dict[str, object]:
    score_dir = BACKTEST_DIR / variant_label / f"{forward_days}d_horizon" / score_type
    pnl_path = score_dir / "factor_pnl.csv"
    review_path = score_dir / "review_log.csv"
    holdings_path = score_dir / "holdings_log.csv"

    if not pnl_path.exists():
        raise FileNotFoundError(f"Missing expected output: {pnl_path}")

    pnl_df = pd.read_csv(pnl_path)
    review_df = pd.read_csv(review_path) if review_path.exists() else pd.DataFrame()
    holdings_df = pd.read_csv(holdings_path) if holdings_path.exists() else pd.DataFrame()

    daily_ret = pnl_df["daily_ret_bps"] / 10000.0 if not pnl_df.empty else pd.Series(dtype=float)
    gross_daily_ret = (
        pnl_df["gross_daily_ret_bps"] / 10000.0
        if "gross_daily_ret_bps" in pnl_df.columns and not pnl_df.empty
        else daily_ret
    )

    total_ret = float(((1.0 + daily_ret).prod() - 1.0) * 100.0) if len(daily_ret) > 0 else 0.0
    gross_total_ret = float(((1.0 + gross_daily_ret).prod() - 1.0) * 100.0) if len(gross_daily_ret) > 0 else 0.0
    avg_daily_ret = float(pnl_df["daily_ret_bps"].mean()) if not pnl_df.empty else 0.0
    daily_vol = float(pnl_df["daily_ret_bps"].std()) if len(pnl_df) > 1 else 0.0
    sharpe = float((avg_daily_ret / daily_vol) * np.sqrt(252)) if daily_vol > 0 else 0.0
    total_cost_bps = float(pnl_df["cost_bps_applied"].sum()) if "cost_bps_applied" in pnl_df.columns else 0.0

    n_reviews = int(len(review_df)) if not review_df.empty else 0
    n_trade_reviews = int(review_df["executed"].fillna(False).astype(bool).sum()) if "executed" in review_df.columns else 0
    review_to_trade_rate = float(n_trade_reviews / n_reviews * 100.0) if n_reviews > 0 else 0.0
    turnover_mean = float(review_df["turnover_total"].mean()) if "turnover_total" in review_df.columns and not review_df.empty else np.nan
    avg_holding_days = float(holdings_df["holding_days"].mean()) if "holding_days" in holdings_df.columns and not holdings_df.empty else np.nan

    return {
        "variant_label": variant_label,
        "score_type": score_type,
        "total_ret": total_ret,
        "gross_total_ret": gross_total_ret,
        "cost_drag_ret_pct": gross_total_ret - total_ret,
        "total_cost_bps": total_cost_bps,
        "avg_daily_ret_bps": avg_daily_ret,
        "sharpe": sharpe,
        "n_reviews": n_reviews,
        "n_trade_reviews": n_trade_reviews,
        "review_to_trade_rate": review_to_trade_rate,
        "turnover_mean": turnover_mean,
        "avg_holding_days": avg_holding_days,
        "score_dir": str(score_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep adaptive regime review intervals.")
    parser.add_argument("--forward-days", type=int, default=10)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--normal-days", type=parse_int_list, default=parse_int_list("10,14,20,30,40,60"))
    parser.add_argument("--grind-up-days", type=parse_int_list, default=parse_int_list("10,14,20,30,40,60"))
    parser.add_argument("--slow-bleed-days", type=int, default=None, help="If omitted, uses the normal-days value for each run.")
    parser.add_argument("--unknown-days", type=int, default=None, help="If omitted, uses the normal-days value for each run.")
    parser.add_argument("--squeeze-days", type=int, default=5)
    parser.add_argument("--crisis-days", type=int, default=5)
    parser.add_argument("--hysteresis-mode", type=str, default="combined")
    parser.add_argument("--variant-prefix", type=str, default="adaptive_regime_sweep")
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--scores-dir", type=str, default=None)
    parser.add_argument("--python-bin", type=str, default=default_python_bin())
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    combos = list(itertools.product(args.normal_days, args.grind_up_days))
    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    run_label = args.run_label or f"{args.variant_prefix}_{args.forward_days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = SWEEP_DIR / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(BASE_DIR / "backtest_results" / ".mplconfig"))

    summary_rows: list[dict[str, object]] = []
    total_runs = len(combos)
    print(f"\n{'=' * 90}")
    print("ADAPTIVE REGIME SWEEP")
    print(f"{'=' * 90}")
    print(f"Forward Horizon: {args.forward_days}d")
    print(f"Normal Days Grid: {args.normal_days}")
    print(f"Grind-Up Days Grid: {args.grind_up_days}")
    print(f"Slow-Bleed Days: {'match normal' if args.slow_bleed_days is None else args.slow_bleed_days}")
    print(f"Unknown Days: {'match normal' if args.unknown_days is None else args.unknown_days}")
    print(f"Squeeze / Crisis: {args.squeeze_days} / {args.crisis_days}")
    print(f"Hysteresis Mode: {args.hysteresis_mode}")
    print(f"Run Label: {run_label}")
    print(f"Python: {args.python_bin}")
    print(f"Total Combos: {total_runs}")
    print(f"{'=' * 90}\n")

    for idx, (normal, grind_up) in enumerate(combos, start=1):
        slow_bleed = args.slow_bleed_days if args.slow_bleed_days is not None else normal
        unknown = args.unknown_days if args.unknown_days is not None else normal
        variant_label = build_variant_label(args.variant_prefix, normal, grind_up, slow_bleed, unknown)
        result_root = BACKTEST_DIR / variant_label / f"{args.forward_days}d_horizon"

        print(f"[{idx}/{total_runs}] {variant_label}")
        start_time = time.time()

        if args.skip_existing and result_root.exists():
            print("   ↳ Skipping existing result directory")
        else:
            cmd = [
                args.python_bin,
                "factor_backtest_engine.py",
                str(args.forward_days),
            ]
            if args.start:
                cmd.append(args.start)
            if args.end:
                cmd.append(args.end)
            cmd += [
                "--review-mode", "adaptive",
                "--hysteresis-mode", args.hysteresis_mode,
                "--review-normal-days", str(normal),
                "--review-grind-up-days", str(grind_up),
                "--review-slow-bleed-days", str(slow_bleed),
                "--review-squeeze-days", str(args.squeeze_days),
                "--review-crisis-days", str(args.crisis_days),
                "--review-unknown-days", str(unknown),
                "--variant-label", variant_label,
                "--skip-plots",
            ]
            if args.scores_dir:
                cmd += ["--scores-dir", args.scores_dir]

            subprocess.run(cmd, cwd=BASE_DIR, env=env, check=True)

        for score_type in ("direction_score", "magnitude_score"):
            row = summarize_result(variant_label, args.forward_days, score_type)
            row.update({
                "normal_days": normal,
                "grind_up_days": grind_up,
                "slow_bleed_days": slow_bleed,
                "squeeze_days": args.squeeze_days,
                "crisis_days": args.crisis_days,
                "unknown_days": unknown,
                "runtime_sec": time.time() - start_time,
            })
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = run_dir / "regime_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    for score_type in summary_df["score_type"].unique():
        score_df = summary_df[summary_df["score_type"] == score_type].sort_values(
            ["total_ret", "sharpe"], ascending=[False, False]
        )
        top_path = run_dir / f"top_{score_type}_by_total_return.csv"
        score_df.head(10).to_csv(top_path, index=False)

    print(f"\n{'=' * 90}")
    print("SWEEP COMPLETE")
    print(f"Summary: {summary_path}")
    for score_type in summary_df["score_type"].unique():
        score_df = summary_df[summary_df["score_type"] == score_type].sort_values(
            ["total_ret", "sharpe"], ascending=[False, False]
        )
        if score_df.empty:
            continue
        best = score_df.iloc[0]
        print(
            f"Best {score_type}: normal={int(best['normal_days'])}, grind_up={int(best['grind_up_days'])}, "
            f"total_ret={best['total_ret']:+.2f}%, sharpe={best['sharpe']:+.3f}, "
            f"cost={best['total_cost_bps']:.1f} bps"
        )
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
