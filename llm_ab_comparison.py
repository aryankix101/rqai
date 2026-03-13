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
    n_days = len(daily_ret)

    if n_days == 0:
        return {"label": label}

    total_ret = (1 + daily_ret).prod() - 1
    ann_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1
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
        "total_return_pct": total_ret * 100,
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


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison_dashboard(
    det_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    det_metrics: dict,
    llm_metrics: dict,
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
        ["Total Return (%)",
         f"{det_metrics.get('total_return_pct', 0):.2f}",
         f"{llm_metrics.get('total_return_pct', 0):.2f}",
         f"{llm_metrics.get('total_return_pct', 0) - det_metrics.get('total_return_pct', 0):+.2f}"],
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

    # ── Panel 3: Rolling 63d Sharpe ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    window = 63
    det_rolling = (det_c["daily_ret"].rolling(window).mean() / det_c["daily_ret"].rolling(window).std()) * np.sqrt(252)
    llm_rolling = (llm_c["daily_ret"].rolling(window).mean() / llm_c["daily_ret"].rolling(window).std()) * np.sqrt(252)
    ax3.plot(det_c["date"], det_rolling, label="Deterministic", color="#2196F3", linewidth=1.5)
    ax3.plot(llm_c["date"], llm_rolling, label="LLM", color="#FF5722", linewidth=1.5)
    ax3.axhline(y=0, color="gray", linewidth=0.5)
    ax3.set_title(f"Rolling {window}d Sharpe Ratio", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    if x_min is not None and x_max is not None:
        ax3.set_xlim(x_min, x_max)

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

    # Paired t-test
    det_daily_ret = det_df["daily_ret_bps"].values / 10000.0
    llm_daily_ret = llm_df["daily_ret_bps"].values / 10000.0
    min_len = min(len(det_daily_ret), len(llm_daily_ret))
    t_stat, p_val = paired_ttest(
        pd.Series(det_daily_ret[:min_len]),
        pd.Series(llm_daily_ret[:min_len])
    )

    diff_mean = (llm_daily_ret[:min_len] - det_daily_ret[:min_len]).mean() * 10000
    diff_std = (llm_daily_ret[:min_len] - det_daily_ret[:min_len]).std() * 10000

    sig_text = []
    sig_text.append(f"STATISTICAL SIGNIFICANCE TEST")
    sig_text.append(f"{'─'*40}")
    sig_text.append(f"Paired t-test (LLM − Deterministic)")
    sig_text.append(f"")
    sig_text.append(f"Mean daily difference: {diff_mean:+.2f} bps")
    sig_text.append(f"Std of difference:     {diff_std:.2f} bps")
    sig_text.append(f"t-statistic:           {t_stat:.3f}")
    sig_text.append(f"p-value:               {p_val:.4f}")
    sig_text.append(f"")

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

        if det_pnl.empty:
            print(f"   ⚠️  No deterministic PnL data for {score_type}")
            continue
        if llm_pnl.empty:
            print(f"   ⚠️  No LLM PnL data for {score_type}")
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

        all_metrics[score_type] = {
            "deterministic": det_metrics,
            "llm": llm_metrics,
        }

        # Print summary
        print(f"\n   {'Metric':<25} {'Deterministic':>15} {'LLM':>15} {'Δ':>12}")
        print(f"   {'─'*70}")
        for key in ["total_return_pct", "sharpe", "max_drawdown_pct", "win_rate_pct", "total_cost_bps"]:
            det_val = det_metrics.get(key, 0)
            llm_val = llm_metrics.get(key, 0)
            delta = llm_val - det_val
            print(f"   {key:<25} {det_val:>15.2f} {llm_val:>15.2f} {delta:>+12.2f}")

        # Generate dashboard
        plot_comparison_dashboard(
            det_pnl, llm_pnl, det_metrics, llm_metrics,
            output_dir, score_type, args.start, args.end
        )

    # Save metrics JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "ab_comparison_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n   📄 Metrics saved: {metrics_path}")

    print(f"\n{'='*80}")
    print(f"A/B COMPARISON COMPLETE")
    print(f"Dashboard PNGs: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
