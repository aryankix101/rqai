#!/usr/bin/env python3
"""Compare weekly mean reversion against a small DET set across cost levels."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from data_source_config import (
    DEFAULT_COMPOSITE_SCORES_DIR,
    RUSSELL_MEMBERSHIP_PATH,
    RUSSELL_OHLCV_PATH,
    SPY_SIGNAL_DATA_PATH,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = BASE_DIR / "backtest_results" / "strategy_cost_comparison"
FACTOR_BACKTEST_DIR = BASE_DIR / "backtest_results" / "factor_backtest"
DEFAULT_COST_BPS_LEVELS = (0.0, 2.5, 5.0, 15.0)
TRADING_DAYS_PER_YEAR = 252.0
WEEKS_PER_YEAR = 52.0


@dataclass(frozen=True)
class DetConfig:
    name: str
    display_name: str
    weighting_mode: str
    top_n: int
    candidate_pool_mode: str
    top_n_magnitude: int | None = None
    direction_threshold: float | None = None


DEFAULT_DET_CONFIGS = (
    DetConfig(
        name="det_rank_100x100_mirror_m200_d10",
        display_name="DET rank 100x100 mirror m200 d10",
        weighting_mode="rank",
        top_n=100,
        candidate_pool_mode="llm_mirror",
        top_n_magnitude=200,
        direction_threshold=10.0,
    ),
    DetConfig(
        name="det_equal_100x100_mirror_m200_d10",
        display_name="DET equal 100x100 mirror m200 d10",
        weighting_mode="equal",
        top_n=100,
        candidate_pool_mode="llm_mirror",
        top_n_magnitude=200,
        direction_threshold=10.0,
    ),
    DetConfig(
        name="det_rank_100x100_raw",
        display_name="DET rank 100x100 raw",
        weighting_mode="rank",
        top_n=100,
        candidate_pool_mode="raw",
    ),
    DetConfig(
        name="det_rank_60x60_mirror_m100_d10",
        display_name="DET rank 60x60 mirror m100 d10",
        weighting_mode="rank",
        top_n=60,
        candidate_pool_mode="llm_mirror",
        top_n_magnitude=100,
        direction_threshold=10.0,
    ),
)


def default_python_bin() -> str:
    venv_python = BASE_DIR / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def parse_cost_bps_levels(value: str) -> tuple[float, ...]:
    levels: list[float] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        bps = float(chunk)
        if bps < 0:
            raise argparse.ArgumentTypeError("Cost bps levels must be non-negative")
        levels.append(bps)
    if not levels:
        raise argparse.ArgumentTypeError("Expected at least one cost level")
    if 0.0 not in levels:
        levels.insert(0, 0.0)
    return tuple(dict.fromkeys(levels))


def format_bps_label(bps: float) -> str:
    if float(bps).is_integer():
        return str(int(bps))
    return str(bps).replace(".", "p")


def strategy_cost_name(base_name: str, cost_bps: float) -> str:
    return f"{base_name} gross" if float(cost_bps) == 0.0 else f"{base_name} {cost_bps:g}bps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weekly MR vs DET fast cost comparison.")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE)
    parser.add_argument("--python-bin", default=default_python_bin())
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_COMPOSITE_SCORES_DIR)
    parser.add_argument("--cost-bps-levels", type=parse_cost_bps_levels, default=DEFAULT_COST_BPS_LEVELS)
    parser.add_argument("--det-config-limit", type=int, default=None, help="Limit DET configs for smoke tests.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing weekly/DET outputs when present.")
    parser.add_argument("--skip-preflight-tests", action="store_true")
    return parser.parse_args()


def git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BASE_DIR,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=BASE_DIR, env=env, check=True)


def preflight(args: argparse.Namespace, env: dict[str, str]) -> None:
    required_paths = [
        RUSSELL_MEMBERSHIP_PATH,
        RUSSELL_OHLCV_PATH,
        args.scores_dir,
        SPY_SIGNAL_DATA_PATH,
    ]
    missing = [path for path in required_paths if not Path(path).exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required comparison inputs:\n{missing_text}")

    if not args.skip_preflight_tests:
        run_command(
            [
                args.python_bin,
                "-m",
                "unittest",
                "test_portfolio_policies.py",
                "test_factor_backtest_engine.py",
            ],
            env,
        )


def run_weekly_benchmark(args: argparse.Namespace, run_dir: Path, env: dict[str, str]) -> Path:
    output_dir = run_dir / "weekly_mean_reversion"
    if args.skip_existing and (output_dir / "strategy_summary.csv").exists():
        print(f"Skipping existing weekly benchmark: {output_dir}")
        return output_dir

    cmd = [
        args.python_bin,
        "run_backtest_v3.py",
        "--start",
        args.start,
        "--end",
        args.end,
        "--output-dir",
        str(output_dir),
        "--membership-path",
        str(RUSSELL_MEMBERSHIP_PATH),
        "--cost-bps-levels",
        ",".join(f"{cost:g}" for cost in args.cost_bps_levels),
    ]
    run_command(cmd, env)
    return output_dir


def det_variant_label(run_label: str, config: DetConfig) -> str:
    return f"{run_label}__{config.name}"


def det_score_dir(variant_label: str) -> Path:
    return FACTOR_BACKTEST_DIR / variant_label / "10d_horizon" / "direction_score"


def build_det_command(args: argparse.Namespace, variant_label: str, config: DetConfig) -> list[str]:
    cmd = [
        args.python_bin,
        "factor_backtest_engine.py",
        "10",
        args.start,
        args.end,
        "--variant-label",
        variant_label,
        "--scores-dir",
        str(args.scores_dir),
        "--score-types",
        "direction_score",
        "--review-mode",
        "static",
        "--review-days",
        "10",
        "--periodic-static",
        "--hysteresis-mode",
        "combined",
        "--weighting-mode",
        config.weighting_mode,
        "--top-n",
        str(config.top_n),
        "--bottom-n",
        str(config.top_n),
        "--long-retain-buffer",
        "20",
        "--short-retain-buffer",
        "5",
        "--replacement-gap",
        "5",
        "--candidate-pool-mode",
        config.candidate_pool_mode,
        "--cost-bps",
        "0",
        "--skip-plots",
    ]
    if config.candidate_pool_mode == "llm_mirror":
        cmd += [
            "--candidate-top-n-magnitude",
            str(config.top_n_magnitude),
            "--candidate-direction-threshold",
            str(config.direction_threshold),
        ]
    return cmd


def run_det_configs(
    args: argparse.Namespace,
    run_label: str,
    configs: tuple[DetConfig, ...],
    env: dict[str, str],
) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for config in configs:
        variant_label = det_variant_label(run_label, config)
        score_dir = det_score_dir(variant_label)
        cmd = build_det_command(args, variant_label, config)
        if args.skip_existing and (score_dir / "factor_pnl.csv").exists():
            print(f"Skipping existing DET run: {score_dir}")
        else:
            run_command(cmd, env)
        runs.append(
            {
                "config": config,
                "variant_label": variant_label,
                "score_dir": score_dir,
                "command": cmd,
            }
        )
    return runs


def safe_cagr(returns: pd.Series, periods_per_year: float, total_periods: float | None = None) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    cumulative = float((1.0 + clean).prod())
    if cumulative <= 0:
        return np.nan
    years = (total_periods / periods_per_year) if total_periods and total_periods > 0 else (len(clean) / periods_per_year)
    return float((cumulative ** (1.0 / years) - 1.0) * 100.0) if years > 0 else np.nan


def compute_drawdown(returns: pd.Series) -> pd.Series:
    clean = returns.fillna(0.0)
    cumulative = (1.0 + clean).cumprod()
    return cumulative / cumulative.cummax() - 1.0


def annualized_sharpe_from_periods(returns: pd.Series, period_lengths: pd.Series) -> float:
    aligned = pd.DataFrame({"returns": returns, "period_lengths": period_lengths}).dropna()
    if aligned.empty:
        return np.nan
    lengths = aligned["period_lengths"].clip(lower=1).astype(float)
    period_returns = aligned["returns"].astype(float)
    dailyized = np.where(
        (1.0 + period_returns) > 0,
        np.power(1.0 + period_returns, 1.0 / lengths) - 1.0,
        period_returns / lengths,
    )
    std = float(np.std(dailyized, ddof=1)) if len(dailyized) > 1 else 0.0
    return float(np.mean(dailyized) / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0


def numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def build_det_cost_overlay_outputs(
    pnl_df: pd.DataFrame,
    *,
    config: DetConfig,
    variant_label: str,
    score_dir: Path | str,
    cost_bps_levels: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pnl_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    frame = pnl_df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    gross_bps = numeric_series(
        frame,
        "gross_daily_ret_bps" if "gross_daily_ret_bps" in frame.columns else "daily_ret_bps",
        0.0,
    )
    turnover = numeric_series(frame, "turnover_total", 0.0)
    period_lengths = numeric_series(frame, "period_length_days", 1.0).clip(lower=1)
    gross_returns = gross_bps / 10000.0
    gross_total_return_pct = float(((1.0 + gross_returns).prod() - 1.0) * 100.0)

    summary_rows: list[dict[str, object]] = []
    timeseries_rows: list[pd.DataFrame] = []
    for cost_bps in cost_bps_levels:
        net_bps = gross_bps - (float(cost_bps) * turnover)
        returns = net_bps / 10000.0
        cumulative = (1.0 + returns).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1.0
        total_return_pct = float((cumulative.iloc[-1] - 1.0) * 100.0)
        cagr_pct = safe_cagr(returns, TRADING_DAYS_PER_YEAR, float(period_lengths.sum()))
        sharpe = annualized_sharpe_from_periods(returns, period_lengths)
        total_cost_bps = float((float(cost_bps) * turnover).sum())

        summary_rows.append(
            {
                "strategy_family": "DET",
                "strategy_name": strategy_cost_name(config.display_name, cost_bps),
                "base_strategy_name": config.display_name,
                "config_label": config.name,
                "variant_name": variant_label,
                "execution_model": "periodic_static_close",
                "cost_bps_per_side": float(cost_bps),
                "total_return_pct": total_return_pct,
                "gross_total_return_pct": gross_total_return_pct,
                "cost_drag_pct": gross_total_return_pct - total_return_pct,
                "cagr_pct": cagr_pct,
                "sharpe": sharpe,
                "max_drawdown_pct": float(drawdown.min() * 100.0),
                "avg_turnover": float(turnover.mean()),
                "n_periods": int(len(returns)),
                "total_cost_bps": total_cost_bps,
                "invalid_periods_at_or_below_minus_100pct": int((returns <= -1.0).sum()),
                "source_dir": str(score_dir),
                "is_primary_comparison": True,
            }
        )

        timeseries_rows.append(
            pd.DataFrame(
                {
                    "date": frame["date"],
                    "strategy_family": "DET",
                    "base_strategy_name": config.display_name,
                    "config_label": config.name,
                    "execution_model": "periodic_static_close",
                    "cost_bps_per_side": float(cost_bps),
                    "period_return": returns,
                    "cumulative_return_pct": (cumulative - 1.0) * 100.0,
                    "drawdown_pct": drawdown * 100.0,
                    "period_length_days": period_lengths,
                    "turnover_total": turnover,
                }
            )
        )

    return pd.DataFrame(summary_rows), pd.concat(timeseries_rows, ignore_index=True)


def load_weekly_outputs(weekly_dir: Path, cost_bps_levels: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(weekly_dir / "strategy_summary.csv")
    weekly_returns = pd.read_csv(weekly_dir / "weekly_returns.csv", parse_dates=["date"])
    summary_rows: list[dict[str, object]] = []
    timeseries_rows: list[pd.DataFrame] = []

    for _, row in summary.iterrows():
        execution_model = str(row.get("execution_model", ""))
        if execution_model not in {"C2C", "O2O"}:
            continue
        cost_bps = float(row["cost_bps_per_side"])
        if not any(np.isclose(cost_bps, level) for level in cost_bps_levels):
            continue
        variant_name = str(row["variant_name"])
        summary_rows.append(
            {
                "strategy_family": "WeeklyMR",
                "strategy_name": strategy_cost_name(f"Weekly MR {execution_model}", cost_bps),
                "base_strategy_name": f"Weekly MR {execution_model}",
                "config_label": "weekly_mean_reversion_decile",
                "variant_name": variant_name,
                "execution_model": execution_model,
                "cost_bps_per_side": cost_bps,
                "total_return_pct": row.get("total_return_pct"),
                "gross_total_return_pct": row.get("gross_total_return_pct"),
                "cost_drag_pct": row.get("cost_drag_pct"),
                "cagr_pct": row.get("cagr_pct"),
                "sharpe": row.get("sharpe"),
                "max_drawdown_pct": row.get("max_drawdown_pct"),
                "avg_turnover": row.get("avg_turnover"),
                "n_periods": row.get("n_periods"),
                "total_cost_bps": row.get("total_cost_bps"),
                "invalid_periods_at_or_below_minus_100pct": 0,
                "source_dir": str(weekly_dir),
                "is_primary_comparison": execution_model == "C2C",
            }
        )

        prefix = "ls_c2c" if execution_model == "C2C" else "ls_o2o"
        column = f"{prefix}_{format_bps_label(cost_bps)}"
        if column not in weekly_returns.columns:
            continue
        returns = pd.to_numeric(weekly_returns[column], errors="coerce").fillna(0.0)
        cumulative = (1.0 + returns).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1.0
        turnover_total = (
            pd.to_numeric(weekly_returns.get("turnover_long", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(weekly_returns.get("turnover_short", 0.0), errors="coerce").fillna(0.0)
        )
        timeseries_rows.append(
            pd.DataFrame(
                {
                    "date": weekly_returns["date"],
                    "strategy_family": "WeeklyMR",
                    "base_strategy_name": f"Weekly MR {execution_model}",
                    "config_label": "weekly_mean_reversion_decile",
                    "execution_model": execution_model,
                    "cost_bps_per_side": cost_bps,
                    "period_return": returns,
                    "cumulative_return_pct": (cumulative - 1.0) * 100.0,
                    "drawdown_pct": drawdown * 100.0,
                    "period_length_days": 5.0,
                    "turnover_total": turnover_total,
                }
            )
        )

    return pd.DataFrame(summary_rows), pd.concat(timeseries_rows, ignore_index=True)


def make_short_label(label: str) -> str:
    replacements = {
        "Weekly MR C2C": "Weekly C2C",
        "DET rank ": "DET r ",
        "DET equal ": "DET e ",
        "mirror ": "",
        "periodic_static_close": "DET",
    }
    short = label
    for old, new in replacements.items():
        short = short.replace(old, new)
    return short


def plot_leaderboard_by_cost(summary: pd.DataFrame, output_dir: Path, cost_levels: tuple[float, ...]) -> None:
    primary = summary[summary["is_primary_comparison"] == True].copy()
    if primary.empty:
        return
    fig, axes = plt.subplots(len(cost_levels), 1, figsize=(13, max(4, 3.2 * len(cost_levels))), squeeze=False)
    for ax, cost_bps in zip(axes[:, 0], cost_levels):
        subset = primary[np.isclose(primary["cost_bps_per_side"].astype(float), cost_bps)].copy()
        subset = subset.sort_values("total_return_pct", ascending=True)
        colors = ["#2ca02c" if value >= 0 else "#d62728" for value in subset["total_return_pct"]]
        ax.barh(subset["base_strategy_name"].map(make_short_label), subset["total_return_pct"], color=colors, alpha=0.9)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(f"Total Return at {cost_bps:g} bps")
        ax.set_xlabel("Total Return (%)")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "leaderboard_total_return_by_cost.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_sensitivity(summary: pd.DataFrame, output_dir: Path, metric: str, filename: str, ylabel: str) -> None:
    primary = summary[summary["is_primary_comparison"] == True].copy()
    if primary.empty or metric not in primary.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    for base_name, group in primary.groupby("base_strategy_name"):
        group = group.sort_values("cost_bps_per_side")
        ax.plot(
            group["cost_bps_per_side"],
            group[metric],
            marker="o",
            linewidth=2,
            label=make_short_label(base_name),
        )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("One-Way Transaction Cost (bps)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by Cost")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries_5bps(timeseries: pd.DataFrame, output_dir: Path, metric: str, filename: str, ylabel: str) -> None:
    subset = timeseries[
        (timeseries["cost_bps_per_side"].astype(float).pipe(lambda series: np.isclose(series, 5.0)))
        & (timeseries["execution_model"].isin(["C2C", "periodic_static_close"]))
    ].copy()
    if subset.empty or metric not in subset.columns:
        return
    fig, ax = plt.subplots(figsize=(13, 7))
    for base_name, group in subset.groupby("base_strategy_name"):
        group = group.sort_values("date")
        ax.plot(group["date"], group[metric], linewidth=1.8, label=make_short_label(base_name))
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_title(ylabel + " at 5 bps")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_comparison_charts(
    summary: pd.DataFrame,
    timeseries: pd.DataFrame,
    output_dir: Path,
    cost_levels: tuple[float, ...],
) -> None:
    plot_leaderboard_by_cost(summary, output_dir, cost_levels)
    plot_timeseries_5bps(timeseries, output_dir, "cumulative_return_pct", "cumulative_returns_5bps.png", "Cumulative Return (%)")
    plot_metric_sensitivity(summary, output_dir, "total_return_pct", "cost_sensitivity_total_return.png", "Total Return (%)")
    plot_metric_sensitivity(summary, output_dir, "sharpe", "cost_sensitivity_sharpe.png", "Sharpe")
    plot_timeseries_5bps(timeseries, output_dir, "drawdown_pct", "drawdown_5bps.png", "Drawdown (%)")


def build_manifest(
    args: argparse.Namespace,
    run_label: str,
    run_dir: Path,
    weekly_dir: Path,
    det_runs: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "run_label": run_label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "start": args.start,
        "end": args.end,
        "cost_bps_levels": list(args.cost_bps_levels),
        "run_dir": str(run_dir),
        "paths": {
            "membership": str(RUSSELL_MEMBERSHIP_PATH),
            "russell_ohlcv": str(RUSSELL_OHLCV_PATH),
            "scores_dir": str(args.scores_dir),
            "spy": str(SPY_SIGNAL_DATA_PATH),
            "weekly_output": str(weekly_dir),
        },
        "det_configs": [
            {
                **asdict(run["config"]),
                "variant_label": run["variant_label"],
                "score_dir": str(run["score_dir"]),
                "command": run["command"],
            }
            for run in det_runs
        ],
    }


def main() -> None:
    args = parse_args()
    run_label = args.run_label or f"weekly_mr_vs_det_fast4_costs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_base) / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(BASE_DIR / "backtest_results" / ".mplconfig"))

    configs = DEFAULT_DET_CONFIGS
    if args.det_config_limit is not None:
        configs = configs[: args.det_config_limit]

    print("\n" + "=" * 96)
    print("STRATEGY COST COMPARISON")
    print("=" * 96)
    print(f"Run label : {run_label}")
    print(f"Date range: {args.start} -> {args.end}")
    print(f"Costs     : {', '.join(f'{cost:g}bps' for cost in args.cost_bps_levels)}")
    print(f"DET runs  : {len(configs)}")
    print("=" * 96)

    preflight(args, env)
    weekly_dir = run_weekly_benchmark(args, run_dir, env)
    det_runs = run_det_configs(args, run_label, configs, env)

    weekly_summary, weekly_timeseries = load_weekly_outputs(weekly_dir, args.cost_bps_levels)
    det_summaries: list[pd.DataFrame] = []
    det_timeseries: list[pd.DataFrame] = []
    for run in det_runs:
        score_dir = Path(run["score_dir"])
        pnl_path = score_dir / "factor_pnl.csv"
        if not pnl_path.exists():
            raise FileNotFoundError(f"DET run did not produce factor_pnl.csv: {pnl_path}")
        pnl_df = pd.read_csv(pnl_path)
        det_summary, det_ts = build_det_cost_overlay_outputs(
            pnl_df,
            config=run["config"],
            variant_label=str(run["variant_label"]),
            score_dir=score_dir,
            cost_bps_levels=args.cost_bps_levels,
        )
        det_summaries.append(det_summary)
        det_timeseries.append(det_ts)

    det_summary_df = pd.concat(det_summaries, ignore_index=True) if det_summaries else pd.DataFrame()
    det_timeseries_df = pd.concat(det_timeseries, ignore_index=True) if det_timeseries else pd.DataFrame()
    comparison_summary = pd.concat([weekly_summary, det_summary_df], ignore_index=True)
    comparison_timeseries = pd.concat([weekly_timeseries, det_timeseries_df], ignore_index=True)

    det_summary_df.to_csv(run_dir / "det_cost_overlay_summary.csv", index=False)
    comparison_summary.to_csv(run_dir / "comparison_summary.csv", index=False)
    comparison_timeseries.to_csv(run_dir / "comparison_timeseries.csv", index=False)

    leaderboard_5bps = comparison_summary[
        (comparison_summary["is_primary_comparison"] == True)
        & np.isclose(comparison_summary["cost_bps_per_side"].astype(float), 5.0)
    ].sort_values(["total_return_pct", "sharpe"], ascending=[False, False])
    leaderboard_5bps.to_csv(run_dir / "leaderboard_5bps.csv", index=False)

    manifest = build_manifest(args, run_label, run_dir, weekly_dir, det_runs)
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    save_comparison_charts(comparison_summary, comparison_timeseries, run_dir, args.cost_bps_levels)

    invalid = det_summary_df["invalid_periods_at_or_below_minus_100pct"].sum() if not det_summary_df.empty else 0
    if invalid:
        print(f"\n⚠️  DET overlay found {int(invalid)} invalid <= -100% periods. Inspect before trusting results.")

    print("\nSaved comparison outputs:")
    for name in [
        "run_manifest.json",
        "det_cost_overlay_summary.csv",
        "comparison_summary.csv",
        "comparison_timeseries.csv",
        "leaderboard_5bps.csv",
        "leaderboard_total_return_by_cost.png",
        "cumulative_returns_5bps.png",
        "cost_sensitivity_total_return.png",
        "cost_sensitivity_sharpe.png",
        "drawdown_5bps.png",
    ]:
        print(f"  {run_dir / name}")


if __name__ == "__main__":
    main()
