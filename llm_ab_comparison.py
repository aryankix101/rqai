"""
LLM A/B Comparison Dashboard

Reads deterministic and LLM factor_pnl.csv outputs and generates a
side-by-side comparison dashboard with overlaid equity curves, IC,
regime breakdown, and statistical significance tests.

Usage:
    python llm_ab_comparison.py \
        --det-dir backtest_results/factor_backtest/10d_horizon \
        --llm-dir backtest_results/factor_backtest/composite_scores_llm_10d_horizon \
        --output-dir backtest_results/factor_backtest/ab_comparison_10d \
        --start 2019-01-01 --end 2019-12-31
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

sns.set_style("darkgrid")


# =============================================================================
# Data Loading
# =============================================================================

def load_pnl(results_dir: Path, score_type: str = "direction_score") -> pd.DataFrame:
    """Load factor_pnl.csv from a results directory."""
    pnl_path = results_dir / score_type / "factor_pnl.csv"
    if not pnl_path.exists():
        print(f"⚠️  PnL file not found: {pnl_path}")
        return pd.DataFrame()
    df = pd.read_csv(pnl_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_optional_csv(results_dir: Path, score_type: str, filename: str) -> pd.DataFrame:
    """Load an optional per-score CSV if it exists."""
    path = results_dir / score_type / filename
    if not path.exists():
        return pd.DataFrame()
    parse_dates = ["date"]
    if filename == "review_log.csv":
        parse_dates = ["review_date", "signal_date"]
    elif filename == "trade_log.csv":
        parse_dates = ["trade_date", "signal_date"]
    elif filename == "portfolio_selections.csv":
        parse_dates = ["rebalance_date", "signal_date", "entry_date", "exit_date", "position_entry_date"]
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _max_series_value(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return 0.0
    return float(series.max())


def _count_truthy(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns or df.empty:
        return 0
    values = df[column]
    if values.dtype == bool:
        return int(values.sum())
    normalized = values.astype(str).str.lower().isin({"true", "1", "yes"})
    return int(normalized.sum())


def summarize_backtest_health(
    pnl_df: pd.DataFrame,
    review_log_df: pd.DataFrame | None = None,
    trade_log_df: pd.DataFrame | None = None,
    portfolio_selections_df: pd.DataFrame | None = None,
) -> dict:
    """Detect degenerate backtest outputs that would make the A/B dashboard misleading."""
    review_log_df = review_log_df if review_log_df is not None else pd.DataFrame()
    trade_log_df = trade_log_df if trade_log_df is not None else pd.DataFrame()
    portfolio_selections_df = portfolio_selections_df if portfolio_selections_df is not None else pd.DataFrame()

    numeric_cols = [
        col
        for col in ("daily_ret_bps", "gross_daily_ret_bps", "long_pnl_bps", "short_pnl_bps", "cost_bps_applied")
        if col in pnl_df.columns
    ]
    flatline = True
    if numeric_cols and not pnl_df.empty:
        flatline = bool((pd.to_numeric(pnl_df[numeric_cols].stack(), errors="coerce").fillna(0.0).abs() <= 1e-12).all())

    max_num_longs = _max_series_value(pnl_df, "num_longs")
    max_num_shorts = _max_series_value(pnl_df, "num_shorts")
    max_candidate_longs = _max_series_value(review_log_df, "candidate_longs")
    max_candidate_shorts = _max_series_value(review_log_df, "candidate_shorts")
    n_executed_reviews = _count_truthy(review_log_df, "executed")

    has_positions = max(max_num_longs, max_num_shorts) > 0
    has_candidates = max(max_candidate_longs, max_candidate_shorts) > 0
    has_executions = (
        n_executed_reviews > 0
        or not trade_log_df.empty
        or not portfolio_selections_df.empty
    )
    is_degenerate = (
        pnl_df.empty
        or (
            flatline
            and not has_positions
            and not has_candidates
            and not has_executions
        )
    )

    return {
        "n_pnl_rows": int(len(pnl_df)),
        "n_reviews": int(len(review_log_df)),
        "n_executed_reviews": int(n_executed_reviews),
        "n_trade_rows": int(len(trade_log_df)),
        "n_portfolio_rows": int(len(portfolio_selections_df)),
        "max_num_longs": int(max_num_longs),
        "max_num_shorts": int(max_num_shorts),
        "max_candidate_longs": int(max_candidate_longs),
        "max_candidate_shorts": int(max_candidate_shorts),
        "flatline_pnl": bool(flatline),
        "has_positions": bool(has_positions),
        "has_candidates": bool(has_candidates),
        "has_executions": bool(has_executions),
        "is_degenerate": bool(is_degenerate),
    }


def build_health_error(label: str, results_dir: Path, score_type: str, health: dict) -> str:
    """Create a compact, actionable error for empty upstream backtests."""
    score_dir = results_dir / score_type
    return (
        f"{label} backtest output is degenerate for {score_type} at {score_dir}: "
        f"pnl_rows={health['n_pnl_rows']}, reviews={health['n_reviews']}, "
        f"executed_reviews={health['n_executed_reviews']}, trade_rows={health['n_trade_rows']}, "
        f"portfolio_rows={health['n_portfolio_rows']}, max_num_longs={health['max_num_longs']}, "
        f"max_num_shorts={health['max_num_shorts']}, max_candidate_longs={health['max_candidate_longs']}, "
        f"max_candidate_shorts={health['max_candidate_shorts']}. "
        "The comparison dashboard would be misleading because the upstream backtest never formed a live book."
    )


def compute_cumulative(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative returns from daily PnL."""
    df = pnl_df.copy()
    df["daily_ret"] = df["daily_ret_bps"] / 10000.0
    df["cumulative"] = (1 + df["daily_ret"]).cumprod()
    df["cumulative_pct"] = (df["cumulative"] - 1) * 100
    df["long_daily_ret"] = df["long_pnl_bps"] / 10000.0
    df["short_daily_ret"] = df["short_pnl_bps"] / 10000.0
    df["long_cumulative"] = (1 + df["long_daily_ret"]).cumprod()
    df["short_cumulative"] = (1 + df["short_daily_ret"]).cumprod()
    return df


def compute_metrics(pnl_df: pd.DataFrame, label: str) -> dict:
    """Compute summary metrics from a PnL DataFrame."""
    daily_ret = pnl_df["daily_ret_bps"] / 10000.0
    gross_daily_ret = (
        pnl_df["gross_daily_ret_bps"] / 10000.0
        if "gross_daily_ret_bps" in pnl_df.columns
        else daily_ret
    )
    n_days = len(daily_ret)

    if n_days == 0:
        return {"label": label}

    net_total_ret = (1 + daily_ret).prod() - 1
    gross_total_ret = (1 + gross_daily_ret).prod() - 1
    ann_ret = (1 + net_total_ret) ** (252 / max(n_days, 1)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cumulative = (1 + daily_ret).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate
    win_rate = (daily_ret > 0).sum() / max(n_days, 1)

    # Calmar
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0

    # Total cost
    total_cost_bps = pnl_df["cost_bps_applied"].sum()

    # Long/short leg returns
    long_ret = (1 + pnl_df["long_pnl_bps"] / 10000.0).prod() - 1
    short_ret = (1 + pnl_df["short_pnl_bps"] / 10000.0).prod() - 1

    return {
        "label": label,
        "total_return_pct": net_total_ret * 100,
        "net_total_return_pct": net_total_ret * 100,
        "gross_total_return_pct": gross_total_ret * 100,
        "cost_drag_return_pct": (gross_total_ret - net_total_ret) * 100,
        "annualized_return_pct": ann_ret * 100,
        "annualized_vol_pct": ann_vol * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "calmar": calmar,
        "win_rate_pct": win_rate * 100,
        "n_trading_days": n_days,
        "total_cost_bps": total_cost_bps,
        "long_return_pct": long_ret * 100,
        "short_return_pct": short_ret * 100,
        "avg_daily_ret_bps": daily_ret.mean() * 10000,
        "daily_ret_std_bps": daily_ret.std() * 10000,
    }


def paired_ttest(det_daily, llm_daily):
    """Run paired t-test on daily return differences."""
    if len(det_daily) != len(llm_daily):
        # Align by date
        min_len = min(len(det_daily), len(llm_daily))
        det_daily = det_daily[:min_len]
        llm_daily = llm_daily[:min_len]

    diff = llm_daily - det_daily
    if diff.std() == 0:
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_rel(llm_daily, det_daily)
    return t_stat, p_val


def align_daily_series(det_df: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
    """Align daily return series by date for direct comparison."""
    det = det_df[["date", "daily_ret_bps"]].rename(columns={"daily_ret_bps": "det_daily_ret_bps"})
    llm = llm_df[["date", "daily_ret_bps"]].rename(columns={"daily_ret_bps": "llm_daily_ret_bps"})
    aligned = det.merge(llm, on="date", how="inner").sort_values("date").reset_index(drop=True)
    aligned["det_daily_ret"] = aligned["det_daily_ret_bps"] / 10000.0
    aligned["llm_daily_ret"] = aligned["llm_daily_ret_bps"] / 10000.0
    aligned["spread_daily_ret"] = aligned["llm_daily_ret"] - aligned["det_daily_ret"]
    aligned["spread_daily_ret_bps"] = aligned["spread_daily_ret"] * 10000.0
    return aligned


def _corr_t_stat(corr: float, n_obs: int) -> float:
    if n_obs <= 2 or abs(corr) >= 1.0:
        return np.nan
    return corr * np.sqrt((n_obs - 2) / max(1e-12, 1.0 - corr ** 2))


def compute_comparison_stats(det_df: pd.DataFrame, llm_df: pd.DataFrame) -> dict:
    """Compute paired-return comparison diagnostics."""
    aligned = align_daily_series(det_df, llm_df)
    n_obs = len(aligned)
    if n_obs == 0:
        return {"n_obs": 0, "aligned": aligned}
    if n_obs < 2:
        return {
            "n_obs": n_obs,
            "aligned": aligned,
            "spread_mean_bps": aligned["spread_daily_ret_bps"].mean(),
            "spread_std_bps": aligned["spread_daily_ret_bps"].std(),
            "paired_t_stat": np.nan,
            "paired_t_p_value": np.nan,
            "pearson_corr": np.nan,
            "pearson_p_value": np.nan,
            "pearson_t_stat": np.nan,
            "spearman_corr": np.nan,
            "spearman_p_value": np.nan,
            "spearman_t_stat": np.nan,
        }

    t_stat, p_val = paired_ttest(aligned["det_daily_ret"], aligned["llm_daily_ret"])
    pearson_r, pearson_p = stats.pearsonr(aligned["det_daily_ret"], aligned["llm_daily_ret"])
    spearman_r, spearman_p = stats.spearmanr(aligned["det_daily_ret"], aligned["llm_daily_ret"])

    return {
        "n_obs": n_obs,
        "aligned": aligned,
        "spread_mean_bps": aligned["spread_daily_ret_bps"].mean(),
        "spread_std_bps": aligned["spread_daily_ret_bps"].std(),
        "paired_t_stat": t_stat,
        "paired_t_p_value": p_val,
        "pearson_corr": pearson_r,
        "pearson_p_value": pearson_p,
        "pearson_t_stat": _corr_t_stat(pearson_r, n_obs),
        "spearman_corr": spearman_r,
        "spearman_p_value": spearman_p,
        "spearman_t_stat": _corr_t_stat(spearman_r, n_obs),
    }


def adaptive_rolling_windows(n_obs: int) -> tuple[int, int]:
    """Choose rolling windows that still render for short samples."""
    short_window = max(5, min(21, max(5, n_obs // 4)))
    medium_window = max(short_window + 2, min(63, max(short_window + 2, n_obs // 2)))
    if medium_window >= n_obs:
        medium_window = max(short_window + 1, n_obs - 1)
    return short_window, max(short_window + 1, medium_window)


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison_dashboard(
    det_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    det_metrics: dict,
    llm_metrics: dict,
    comparison_stats: dict,
    output_dir: Path,
    score_type: str = "direction_score",
    start_date: str = "",
    end_date: str = "",
):
    """Generate a multi-panel comparison dashboard."""
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(
        f"LLM vs Deterministic A/B Comparison — {score_type.replace('_', ' ').title()}\n"
        f"{start_date} to {end_date}",
        fontsize=18, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3, top=0.93, bottom=0.05)

    det_c = compute_cumulative(det_df)
    llm_c = compute_cumulative(llm_df)

    # Set x-axis limits to test window for all plots (force explicit date range)
    if len(det_df) > 0 and len(llm_df) > 0:
        x_min = max(det_df["date"].min(), llm_df["date"].min())
        x_max = min(det_df["date"].max(), llm_df["date"].max())
    else:
        x_min = None
        x_max = None

    # ── Panel 1: Overlaid Equity Curves ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(det_c["date"], det_c["cumulative_pct"], label="Deterministic", color="#2196F3", linewidth=2)
    ax1.plot(llm_c["date"], llm_c["cumulative_pct"], label="LLM", color="#FF5722", linewidth=2)
    ax1.fill_between(det_c["date"], det_c["cumulative_pct"], alpha=0.1, color="#2196F3")
    ax1.fill_between(llm_c["date"], llm_c["cumulative_pct"], alpha=0.1, color="#FF5722")
    ax1.set_title("Cumulative Returns (%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(alpha=0.3)
    if x_min is not None and x_max is not None:
        ax1.set_xlim(x_min, x_max)

    # ── Panel 2: Summary Metrics Table ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    metrics_data = [
        ["Metric", "Deterministic", "LLM", "Δ (LLM-Det)"],
        ["Net Return (%)",
         f"{det_metrics.get('net_total_return_pct', det_metrics.get('total_return_pct', 0)):.2f}",
         f"{llm_metrics.get('net_total_return_pct', llm_metrics.get('total_return_pct', 0)):.2f}",
         f"{llm_metrics.get('net_total_return_pct', llm_metrics.get('total_return_pct', 0)) - det_metrics.get('net_total_return_pct', det_metrics.get('total_return_pct', 0)):+.2f}"],
        ["Gross Return (%)",
         f"{det_metrics.get('gross_total_return_pct', det_metrics.get('total_return_pct', 0)):.2f}",
         f"{llm_metrics.get('gross_total_return_pct', llm_metrics.get('total_return_pct', 0)):.2f}",
         f"{llm_metrics.get('gross_total_return_pct', llm_metrics.get('total_return_pct', 0)) - det_metrics.get('gross_total_return_pct', det_metrics.get('total_return_pct', 0)):+.2f}"],
        ["Cost Drag (%)",
         f"{det_metrics.get('cost_drag_return_pct', 0):.2f}",
         f"{llm_metrics.get('cost_drag_return_pct', 0):.2f}",
         f"{llm_metrics.get('cost_drag_return_pct', 0) - det_metrics.get('cost_drag_return_pct', 0):+.2f}"],
        ["Sharpe Ratio",
         f"{det_metrics.get('sharpe', 0):.3f}",
         f"{llm_metrics.get('sharpe', 0):.3f}",
         f"{llm_metrics.get('sharpe', 0) - det_metrics.get('sharpe', 0):+.3f}"],
        ["Max Drawdown (%)",
         f"{det_metrics.get('max_drawdown_pct', 0):.2f}",
         f"{llm_metrics.get('max_drawdown_pct', 0):.2f}",
         f"{llm_metrics.get('max_drawdown_pct', 0) - det_metrics.get('max_drawdown_pct', 0):+.2f}"],
        ["Calmar Ratio",
         f"{det_metrics.get('calmar', 0):.3f}",
         f"{llm_metrics.get('calmar', 0):.3f}",
         f"{llm_metrics.get('calmar', 0) - det_metrics.get('calmar', 0):+.3f}"],
        ["Win Rate (%)",
         f"{det_metrics.get('win_rate_pct', 0):.1f}",
         f"{llm_metrics.get('win_rate_pct', 0):.1f}",
         f"{llm_metrics.get('win_rate_pct', 0) - det_metrics.get('win_rate_pct', 0):+.1f}"],
        ["Ann. Vol (%)",
         f"{det_metrics.get('annualized_vol_pct', 0):.2f}",
         f"{llm_metrics.get('annualized_vol_pct', 0):.2f}",
         f"{llm_metrics.get('annualized_vol_pct', 0) - det_metrics.get('annualized_vol_pct', 0):+.2f}"],
        ["Total Cost (bps)",
         f"{det_metrics.get('total_cost_bps', 0):.1f}",
         f"{llm_metrics.get('total_cost_bps', 0):.1f}",
         f"{llm_metrics.get('total_cost_bps', 0) - det_metrics.get('total_cost_bps', 0):+.1f}"],
        ["Long Return (%)",
         f"{det_metrics.get('long_return_pct', 0):.2f}",
         f"{llm_metrics.get('long_return_pct', 0):.2f}",
         f"{llm_metrics.get('long_return_pct', 0) - det_metrics.get('long_return_pct', 0):+.2f}"],
        ["Short Return (%)",
         f"{det_metrics.get('short_return_pct', 0):.2f}",
         f"{llm_metrics.get('short_return_pct', 0):.2f}",
         f"{llm_metrics.get('short_return_pct', 0) - det_metrics.get('short_return_pct', 0):+.2f}"],
    ]

    table = ax2.table(
        cellText=metrics_data,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Color header row
    for j in range(4):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color delta column
    for i in range(1, len(metrics_data)):
        delta_str = metrics_data[i][3]
        try:
            delta_val = float(delta_str.replace("+", ""))
            if delta_val > 0:
                table[i, 3].set_facecolor("#C8E6C9")
            elif delta_val < 0:
                table[i, 3].set_facecolor("#FFCDD2")
        except ValueError:
            pass

    ax2.set_title("Performance Comparison", fontsize=13, fontweight="bold", pad=20)

    # ── Panel 3: Rolling Correlation ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    aligned = comparison_stats.get("aligned", pd.DataFrame())
    if not aligned.empty:
        short_window, medium_window = adaptive_rolling_windows(len(aligned))
        rolling_short = aligned["det_daily_ret"].rolling(short_window).corr(aligned["llm_daily_ret"])
        rolling_medium = aligned["det_daily_ret"].rolling(medium_window).corr(aligned["llm_daily_ret"])
        ax3.plot(aligned["date"], rolling_short, label=f"{short_window}d", color="#00796B", linewidth=1.75)
        ax3.plot(aligned["date"], rolling_medium, label=f"{medium_window}d", color="#8E24AA", linewidth=1.75)
        ax3.axhline(y=0, color="gray", linewidth=0.5)
        ax3.set_title("Rolling Return Correlation", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Correlation")
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
        ax3.set_ylim(-1.05, 1.05)
        if x_min is not None and x_max is not None:
            ax3.set_xlim(x_min, x_max)
    else:
        ax3.text(0.5, 0.5, "No overlapping dates", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Rolling Return Correlation", fontsize=12, fontweight="bold")
        ax3.axis("off")

    # ── Panel 4: Daily Return Distribution ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    det_daily_bps = det_df["daily_ret_bps"].values
    llm_daily_bps = llm_df["daily_ret_bps"].values
    bins = np.linspace(
        min(det_daily_bps.min(), llm_daily_bps.min()),
        max(det_daily_bps.max(), llm_daily_bps.max()),
        50
    )
    ax4.hist(det_daily_bps, bins=bins, alpha=0.5, label="Deterministic", color="#2196F3", edgecolor="white")
    ax4.hist(llm_daily_bps, bins=bins, alpha=0.5, label="LLM", color="#FF5722", edgecolor="white")
    ax4.axvline(x=det_daily_bps.mean(), color="#2196F3", linestyle="--", linewidth=1.5, label=f"Det mean: {det_daily_bps.mean():.1f}")
    ax4.axvline(x=llm_daily_bps.mean(), color="#FF5722", linestyle="--", linewidth=1.5, label=f"LLM mean: {llm_daily_bps.mean():.1f}")
    ax4.set_title("Daily Return Distribution (bps)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Daily Return (bps)")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # ── Panel 5: Drawdown Comparison ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    det_cum = det_c["cumulative"]
    llm_cum = llm_c["cumulative"]
    det_dd = (det_cum - det_cum.cummax()) / det_cum.cummax() * 100
    llm_dd = (llm_cum - llm_cum.cummax()) / llm_cum.cummax() * 100
    ax5.fill_between(det_c["date"], det_dd, alpha=0.4, color="#2196F3", label="Deterministic")
    ax5.fill_between(llm_c["date"], llm_dd, alpha=0.4, color="#FF5722", label="LLM")
    ax5.set_title("Drawdown (%)", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Drawdown (%)")
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    if x_min is not None and x_max is not None:
        ax5.set_xlim(x_min, x_max)

    # ── Panel 6: Long vs Short Leg Breakdown ─────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    x_labels = ["Det Long", "Det Short", "LLM Long", "LLM Short"]
    returns = [
        det_metrics.get("long_return_pct", 0),
        det_metrics.get("short_return_pct", 0),
        llm_metrics.get("long_return_pct", 0),
        llm_metrics.get("short_return_pct", 0),
    ]
    colors = ["#64B5F6", "#90CAF9", "#FF8A65", "#FFAB91"]
    bars = ax6.bar(x_labels, returns, color=colors, edgecolor="white")
    for bar, ret in zip(bars, returns):
        ax6.text(bar.get_x() + bar.get_width() / 2, ret + 0.1,
                 f"{ret:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax6.set_title("Long vs Short Leg Returns (%)", fontsize=12, fontweight="bold")
    ax6.axhline(y=0, color="gray", linewidth=0.5)
    ax6.grid(alpha=0.3, axis="y")

    # ── Panel 7: Cost Comparison ─────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    det_cost_cum = det_df["cost_bps_applied"].cumsum()
    llm_cost_cum = llm_df["cost_bps_applied"].cumsum()
    ax7.plot(det_c["date"], det_cost_cum, label="Deterministic", color="#2196F3", linewidth=2)
    ax7.plot(llm_c["date"], llm_cost_cum, label="LLM", color="#FF5722", linewidth=2)
    ax7.set_title("Cumulative Transaction Costs (bps)", fontsize=12, fontweight="bold")
    ax7.set_ylabel("Cumulative Cost (bps)")
    ax7.legend(fontsize=9)
    ax7.grid(alpha=0.3)

    # ── Panel 8: Statistical Significance ────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")

    t_stat = comparison_stats.get("paired_t_stat", np.nan)
    p_val = comparison_stats.get("paired_t_p_value", np.nan)
    diff_mean = comparison_stats.get("spread_mean_bps", np.nan)
    diff_std = comparison_stats.get("spread_std_bps", np.nan)
    pearson_corr = comparison_stats.get("pearson_corr", np.nan)
    pearson_t = comparison_stats.get("pearson_t_stat", np.nan)
    pearson_p = comparison_stats.get("pearson_p_value", np.nan)
    spearman_corr = comparison_stats.get("spearman_corr", np.nan)
    spearman_t = comparison_stats.get("spearman_t_stat", np.nan)
    spearman_p = comparison_stats.get("spearman_p_value", np.nan)
    n_obs = comparison_stats.get("n_obs", 0)

    sig_text = []
    sig_text.append("STATISTICAL SIGNIFICANCE TEST")
    sig_text.append(f"{'─'*40}")
    sig_text.append("Paired t-test (LLM − Deterministic)")
    sig_text.append("")
    sig_text.append(f"Obs:                   {n_obs}")
    sig_text.append(f"Mean daily difference: {diff_mean:+.2f} bps")
    sig_text.append(f"Std of difference:     {diff_std:.2f} bps")
    sig_text.append(f"t-statistic:           {t_stat:.3f}")
    sig_text.append(f"p-value:               {p_val:.4f}")
    sig_text.append("")
    sig_text.append("Return correlation")
    sig_text.append(f"Pearson r / t / p:     {pearson_corr:+.3f} / {pearson_t:+.3f} / {pearson_p:.4f}")
    sig_text.append(f"Spearman r / t / p:    {spearman_corr:+.3f} / {spearman_t:+.3f} / {spearman_p:.4f}")
    sig_text.append("")

    if p_val < 0.01:
        sig_text.append(f"★★★ Highly Significant (p < 0.01)")
        sig_color = "#2E7D32" if diff_mean > 0 else "#C62828"
    elif p_val < 0.05:
        sig_text.append(f"★★ Significant (p < 0.05)")
        sig_color = "#388E3C" if diff_mean > 0 else "#D32F2F"
    elif p_val < 0.10:
        sig_text.append(f"★ Marginally Significant (p < 0.10)")
        sig_color = "#689F38" if diff_mean > 0 else "#E53935"
    else:
        sig_text.append(f"○ Not Significant (p ≥ 0.10)")
        sig_color = "#757575"

    winner = "LLM" if diff_mean > 0 else "Deterministic"
    sig_text.append(f"")
    sig_text.append(f"Winner: {winner}")

    ax8.text(0.5, 0.5, "\n".join(sig_text),
             transform=ax8.transAxes, fontsize=11,
             verticalalignment="center", horizontalalignment="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=sig_color, alpha=0.15))
    ax8.set_title("Statistical Test", fontsize=13, fontweight="bold")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"ab_comparison_{score_type}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Dashboard saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate A/B comparison dashboard: Deterministic vs LLM"
    )
    parser.add_argument("--det-dir", required=True, help="Deterministic backtest results directory")
    parser.add_argument("--llm-dir", required=True, help="LLM backtest results directory")
    parser.add_argument("--output-dir", default="backtest_results/factor_backtest/ab_comparison",
                        help="Output directory for comparison dashboard")
    parser.add_argument("--start", default="", help="Start date (for title)")
    parser.add_argument("--end", default="", help="End date (for title)")

    args = parser.parse_args()

    det_dir = Path(args.det_dir)
    llm_dir = Path(args.llm_dir)
    output_dir = Path(args.output_dir)

    print(f"\n{'='*80}")
    print(f"LLM A/B COMPARISON DASHBOARD")
    print(f"Deterministic: {det_dir}")
    print(f"LLM:           {llm_dir}")
    print(f"Output:        {output_dir}")
    print(f"{'='*80}\n")

    score_types = ["direction_score", "magnitude_score"]
    all_metrics = {}
    health_errors = []

    # Determine test window — use explicit args if provided, else infer from data
    if args.start and args.end:
        test_start = pd.Timestamp(args.start)
        test_end = pd.Timestamp(args.end)
        filter_to_window = True
        print(f"\n📋 Test window (explicit): {args.start} to {args.end}")
    else:
        filter_to_window = False
        print(f"\n📋 Test window (inferred from data)")

    for score_type in score_types:
        print(f"\n📊 Processing {score_type}...")

        det_pnl = load_pnl(det_dir, score_type)
        llm_pnl = load_pnl(llm_dir, score_type)
        det_review_log = load_optional_csv(det_dir, score_type, "review_log.csv")
        llm_review_log = load_optional_csv(llm_dir, score_type, "review_log.csv")
        det_trade_log = load_optional_csv(det_dir, score_type, "trade_log.csv")
        llm_trade_log = load_optional_csv(llm_dir, score_type, "trade_log.csv")
        det_portfolio = load_optional_csv(det_dir, score_type, "portfolio_selections.csv")
        llm_portfolio = load_optional_csv(llm_dir, score_type, "portfolio_selections.csv")

        if det_pnl.empty:
            print(f"   ⚠️  No deterministic PnL data for {score_type}")
            continue
        if llm_pnl.empty:
            print(f"   ⚠️  No LLM PnL data for {score_type}")
            continue

        det_health = summarize_backtest_health(det_pnl, det_review_log, det_trade_log, det_portfolio)
        llm_health = summarize_backtest_health(llm_pnl, llm_review_log, llm_trade_log, llm_portfolio)
        score_health_errors = []
        if det_health["is_degenerate"]:
            issue = build_health_error("Deterministic", det_dir, score_type, det_health)
            print(f"   ❌ {issue}")
            score_health_errors.append(issue)
        if llm_health["is_degenerate"]:
            issue = build_health_error("LLM", llm_dir, score_type, llm_health)
            print(f"   ❌ {issue}")
            score_health_errors.append(issue)
        if score_health_errors:
            health_errors.extend(score_health_errors)
            continue

        # Filter to explicit test window if provided; otherwise use intersection
        if filter_to_window:
            det_pnl = det_pnl[(det_pnl["date"] >= test_start) & (det_pnl["date"] <= test_end)].reset_index(drop=True)
            llm_pnl = llm_pnl[(llm_pnl["date"] >= test_start) & (llm_pnl["date"] <= test_end)].reset_index(drop=True)
            common_start = test_start
            common_end = test_end
        else:
            # Fallback: use intersection of whatever data is available
            common_start = max(det_pnl["date"].min(), llm_pnl["date"].min())
            common_end = min(det_pnl["date"].max(), llm_pnl["date"].max())
            det_pnl = det_pnl[(det_pnl["date"] >= common_start) & (det_pnl["date"] <= common_end)].reset_index(drop=True)
            llm_pnl = llm_pnl[(llm_pnl["date"] >= common_start) & (llm_pnl["date"] <= common_end)].reset_index(drop=True)

        print(f"   Date range: {common_start.date()} to {common_end.date()}")
        print(f"   Det trading days: {len(det_pnl)}")
        print(f"   LLM trading days: {len(llm_pnl)}")

        det_metrics = compute_metrics(det_pnl, f"Deterministic ({score_type})")
        llm_metrics = compute_metrics(llm_pnl, f"LLM ({score_type})")
        comparison_stats = compute_comparison_stats(det_pnl, llm_pnl)

        all_metrics[score_type] = {
            "deterministic": det_metrics,
            "llm": llm_metrics,
            "deterministic_health": det_health,
            "llm_health": llm_health,
            "comparison": {
                key: value
                for key, value in comparison_stats.items()
                if key != "aligned"
            },
        }

        # Print summary
        print(f"\n   {'Metric':<25} {'Deterministic':>15} {'LLM':>15} {'Δ':>12}")
        print(f"   {'─'*70}")
        summary_rows = [
            ("net_return_pct", det_metrics.get("net_total_return_pct", det_metrics.get("total_return_pct", 0)), llm_metrics.get("net_total_return_pct", llm_metrics.get("total_return_pct", 0))),
            ("gross_return_pct", det_metrics.get("gross_total_return_pct", det_metrics.get("total_return_pct", 0)), llm_metrics.get("gross_total_return_pct", llm_metrics.get("total_return_pct", 0))),
            ("cost_drag_pct", det_metrics.get("cost_drag_return_pct", 0), llm_metrics.get("cost_drag_return_pct", 0)),
            ("sharpe", det_metrics.get("sharpe", 0), llm_metrics.get("sharpe", 0)),
            ("max_drawdown_pct", det_metrics.get("max_drawdown_pct", 0), llm_metrics.get("max_drawdown_pct", 0)),
            ("win_rate_pct", det_metrics.get("win_rate_pct", 0), llm_metrics.get("win_rate_pct", 0)),
            ("total_cost_bps", det_metrics.get("total_cost_bps", 0), llm_metrics.get("total_cost_bps", 0)),
        ]
        for label, det_val, llm_val in summary_rows:
            delta = llm_val - det_val
            print(f"   {label:<25} {det_val:>15.2f} {llm_val:>15.2f} {delta:>+12.2f}")
        print(
            f"   {'pearson_corr':<25} {comparison_stats.get('pearson_corr', 0):>15.3f} "
            f"{comparison_stats.get('pearson_p_value', 0):>15.4f} "
            f"{comparison_stats.get('pearson_t_stat', 0):>+12.3f}"
        )
        print(
            f"   {'spearman_corr':<25} {comparison_stats.get('spearman_corr', 0):>15.3f} "
            f"{comparison_stats.get('spearman_p_value', 0):>15.4f} "
            f"{comparison_stats.get('spread_mean_bps', 0):>+12.2f}"
        )

        # Generate dashboard
        plot_comparison_dashboard(
            det_pnl, llm_pnl, det_metrics, llm_metrics, comparison_stats,
            output_dir, score_type, args.start, args.end
        )

    # Save metrics JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "ab_comparison_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n   📄 Metrics saved: {metrics_path}")

    if health_errors and not all_metrics:
        print("\n❌ A/B comparison aborted because both upstream runs were empty/degenerate.")
        for issue in health_errors:
            print(f"   - {issue}")
        raise SystemExit(1)

    print(f"\n{'='*80}")
    print(f"A/B COMPARISON COMPLETE")
    print(f"Dashboard PNGs: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
