#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _compound_pct(ret_series: pd.Series) -> float:
    return float(((1.0 + ret_series.fillna(0.0)).prod() - 1.0) * 100.0)


def _max_drawdown_pct(cumulative_pct: pd.Series) -> float:
    equity = 1.0 + cumulative_pct.fillna(0.0) / 100.0
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min() * 100.0)


def compute_summary(pnl: pd.DataFrame, review_log: pd.DataFrame | None) -> dict[str, float]:
    net_ret = pnl["net_daily_ret_bps"] / 10000.0
    gross_ret = pnl["gross_daily_ret_bps"] / 10000.0
    long_ret = pnl["long_pnl_bps"] / 10000.0
    short_ret = pnl["short_pnl_bps"] / 10000.0

    mean_daily = float(net_ret.mean())
    std_daily = float(net_ret.std(ddof=1))
    sharpe = float(np.sqrt(252.0) * mean_daily / std_daily) if std_daily > 0 else float("nan")
    t_stat = float(np.sqrt(len(net_ret)) * mean_daily / std_daily) if std_daily > 0 else float("nan")
    win_rate = float((net_ret > 0).mean() * 100.0)

    summary = {
        "net_return_pct": _compound_pct(net_ret),
        "gross_return_pct": _compound_pct(gross_ret),
        "long_return_pct": _compound_pct(long_ret),
        "short_return_pct": _compound_pct(short_ret),
        "cost_drag_pct": float(pnl["cost_bps_applied"].fillna(0.0).sum() / 100.0),
        "sharpe": sharpe,
        "daily_mean_bps": mean_daily * 10000.0,
        "daily_t_stat": t_stat,
        "win_rate_pct": win_rate,
        "max_drawdown_pct": _max_drawdown_pct(pnl["cumulative_ret_pct"]),
        "trading_days": int(len(pnl)),
        "gross_nonzero_days": int((pnl["gross_daily_ret_bps"].fillna(0.0) != 0.0).sum()),
    }

    if review_log is not None and not review_log.empty:
        executed = (
            review_log["executed"].astype(str).str.lower().isin({"true", "1", "yes"})
            if "executed" in review_log.columns
            else pd.Series(False, index=review_log.index)
        )
        executed_reviews = review_log.loc[executed]
        summary["reviews"] = int(len(review_log))
        summary["executed_reviews"] = int(executed.sum())
        if not executed_reviews.empty:
            summary["avg_turnover_pct"] = float(_safe_numeric(executed_reviews["turnover_total"]).mean() * 100.0)
            summary["avg_turnover_long_pct"] = float(_safe_numeric(executed_reviews["turnover_long"]).mean() * 100.0)
            summary["avg_turnover_short_pct"] = float(_safe_numeric(executed_reviews["turnover_short"]).mean() * 100.0)

    return summary


def plot_dashboard(pnl: pd.DataFrame, decile: pd.DataFrame | None, review_log: pd.DataFrame | None, output_path: Path, title: str) -> None:
    pnl = pnl.sort_values("date").reset_index(drop=True)
    pnl["date"] = pd.to_datetime(pnl["date"])

    for col in [
        "daily_ret_bps",
        "gross_daily_ret_bps",
        "net_daily_ret_bps",
        "long_pnl_bps",
        "short_pnl_bps",
        "cost_bps_applied",
        "turnover_total",
        "cumulative_ret_pct",
    ]:
        if col in pnl.columns:
            pnl[col] = _safe_numeric(pnl[col]).fillna(0.0)

    pnl["gross_ret"] = pnl["gross_daily_ret_bps"] / 10000.0
    pnl["net_ret"] = pnl["net_daily_ret_bps"] / 10000.0
    pnl["long_ret"] = pnl["long_pnl_bps"] / 10000.0
    pnl["short_ret"] = pnl["short_pnl_bps"] / 10000.0
    pnl["cum_gross_pct"] = ((1.0 + pnl["gross_ret"]).cumprod() - 1.0) * 100.0
    pnl["cum_net_pct"] = ((1.0 + pnl["net_ret"]).cumprod() - 1.0) * 100.0
    pnl["cum_long_pct"] = ((1.0 + pnl["long_ret"]).cumprod() - 1.0) * 100.0
    pnl["cum_short_pct"] = ((1.0 + pnl["short_ret"]).cumprod() - 1.0) * 100.0
    pnl["cum_cost_pct"] = pnl["cost_bps_applied"].cumsum() / 100.0
    pnl["rolling_sharpe"] = (
        pnl["net_ret"].rolling(252).mean()
        / pnl["net_ret"].rolling(252).std().replace(0.0, np.nan)
        * np.sqrt(252.0)
    )

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(title, fontsize=16)

    ax = axes[0, 0]
    ax.plot(pnl["date"], pnl["cum_net_pct"], label="Net", linewidth=2.4, color="#2F6FDB")
    ax.plot(pnl["date"], pnl["cum_gross_pct"], label="Gross", linewidth=1.8, color="#1A9E55", alpha=0.9)
    ax.set_title("Net vs Gross Cumulative Return")
    ax.set_ylabel("Return (%)")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(pnl["date"], pnl["cum_long_pct"], label="Long Leg", linewidth=2.0, color="#1A9E55")
    ax.plot(pnl["date"], pnl["cum_short_pct"], label="Short Leg", linewidth=2.0, color="#D64545")
    ax.plot(pnl["date"], pnl["cum_net_pct"], label="Combined Net", linewidth=2.4, color="#2F6FDB")
    ax.set_title("Long vs Short Contribution")
    ax.set_ylabel("Return (%)")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(pnl["date"], pnl["rolling_sharpe"], color="#7A3E9D", linewidth=1.7)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_title("Rolling 252d Sharpe")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    equity = 1.0 + pnl["cum_net_pct"] / 100.0
    running_peak = equity.cummax()
    drawdown_pct = (equity / running_peak - 1.0) * 100.0
    ax.fill_between(pnl["date"], drawdown_pct, 0.0, color="#D64545", alpha=0.35)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.25)

    ax = axes[2, 0]
    ax.plot(pnl["date"], pnl["cum_cost_pct"], color="#C03A2B", linewidth=2.0, label="Cumulative cost drag")
    if review_log is not None and not review_log.empty:
        review_log = review_log.copy()
        if "review_date" in review_log.columns:
            review_log["review_date"] = pd.to_datetime(review_log["review_date"], errors="coerce")
        if "turnover_total" in review_log.columns:
            review_log["turnover_total"] = _safe_numeric(review_log["turnover_total"])
            ax2 = ax.twinx()
            ax2.plot(
                review_log["review_date"],
                review_log["turnover_total"] * 100.0,
                color="#F08C2E",
                linewidth=1.3,
                alpha=0.6,
                label="Turnover per review",
            )
            ax2.set_ylabel("Turnover (%)")
    ax.set_title("Costs and Turnover")
    ax.set_ylabel("Cost Drag (%)")
    ax.grid(alpha=0.25)

    ax = axes[2, 1]
    if decile is not None and not decile.empty:
        decile = decile.copy()
        decile["date"] = pd.to_datetime(decile["date"], errors="coerce")
        decile["decile"] = _safe_numeric(decile["decile"])
        decile["mean_ret"] = _safe_numeric(decile["mean_ret"])
        bar = decile.groupby("decile", as_index=False)["mean_ret"].mean()
        ax.bar(bar["decile"].astype(int).astype(str), bar["mean_ret"] * 100.0, color="#5B8FF9")
        ax.set_title("Average Decile Return")
        ax.set_ylabel("Return (%)")
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No decile_analysis.csv found", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a factor backtest folder from saved CSVs.")
    parser.add_argument("results_dir", help="Path to a score-type folder like .../direction_score")
    parser.add_argument("--title", default=None, help="Optional dashboard title")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    pnl_path = results_dir / "factor_pnl.csv"
    if not pnl_path.exists():
        raise SystemExit(f"Missing {pnl_path}")

    pnl = pd.read_csv(pnl_path, parse_dates=["date"])
    decile_path = results_dir / "decile_analysis.csv"
    review_log_path = results_dir / "review_log.csv"
    decile = pd.read_csv(decile_path, parse_dates=["date"]) if decile_path.exists() else None
    review_log = pd.read_csv(review_log_path, parse_dates=["review_date", "signal_date"]) if review_log_path.exists() else None

    summary = compute_summary(pnl, review_log)
    summary_path = results_dir / "csv_viz_summary.json"
    dashboard_path = results_dir / "csv_viz_dashboard.png"

    summary_path.write_text(json.dumps(summary, indent=2))
    plot_dashboard(pnl, decile, review_log, dashboard_path, args.title or results_dir.name)

    print("Saved:")
    print(summary_path)
    print(dashboard_path)
    print()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
