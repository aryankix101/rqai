"""
Batch Results Aggregator — Compare composite scores across multiple stocks
Reads all composite score CSVs and creates comparison metrics/charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

COMPOSITE_SCORES_DIR = Path("backtest_results/composite_scores")
OUTPUT_DIR = Path("backtest_results/batch_analysis")


def load_all_composite_scores():
    """Load all composite score CSVs from directory"""
    results = {}
    if not COMPOSITE_SCORES_DIR.exists():
        print(f"❌ Directory not found: {COMPOSITE_SCORES_DIR}")
        return results
    
    csv_files = list(COMPOSITE_SCORES_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {COMPOSITE_SCORES_DIR}")
        return results
    
    print(f"📂 Loading {len(csv_files)} composite score files...\n")
    for csv_file in sorted(csv_files):
        try:
            ticker = csv_file.stem.split("_")[2]
            df = pd.read_csv(csv_file)
            results[ticker] = df
            print(f"   ✓ {ticker:6s} - {len(df)} records")
        except Exception as e:
            print(f"   ❌ {csv_file.name}: {e}")
    return results


def compute_metrics(df, ticker):
    """Compute key metrics for a single stock"""
    metrics = {"ticker": ticker, "records": len(df)}
    
    # Direction metrics
    if "direction_score" in df.columns and "fwd_10d_ret" in df.columns:
        valid = df[["direction_score", "fwd_10d_ret"]].dropna()
        if len(valid) > 0:
            ic_dir, _ = spearmanr(valid["direction_score"], valid["fwd_10d_ret"])
            metrics["dir_ic"] = ic_dir if np.isfinite(ic_dir) else np.nan
            
            bullish = (valid["direction_score"] > 0) & (valid["fwd_10d_ret"] > 0)
            bearish = (valid["direction_score"] < 0) & (valid["fwd_10d_ret"] < 0)
            hit_rate = (bullish.sum() + bearish.sum()) / len(valid) * 100
            metrics["dir_hitrate"] = hit_rate
    
    # Magnitude metrics
    if "magnitude_score" in df.columns and "fwd_abs_10d_ret" in df.columns:
        valid = df[["magnitude_score", "fwd_abs_10d_ret"]].dropna()
        if len(valid) > 0:
            ic_mag, _ = spearmanr(valid["magnitude_score"], valid["fwd_abs_10d_ret"])
            metrics["mag_ic"] = ic_mag if np.isfinite(ic_mag) else np.nan
    
    return metrics


def create_summary_table(all_results):
    """Create summary table of all stocks"""
    metrics_list = [compute_metrics(df, ticker) for ticker, df in all_results.items()]
    summary_df = pd.DataFrame(metrics_list)
    return summary_df.sort_values("dir_ic", ascending=False, na_position="last")


def plot_ic_comparison(summary_df):
    """Plot direction & magnitude IC across all stocks"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax, col, title in zip(axes, ["dir_ic", "mag_ic"], 
                               ["Direction Score IC (10d Forward Return)", 
                                "Magnitude Score IC (Absolute 10d Return)"]):
        sorted_df = summary_df.sort_values(col, ascending=True)
        colors = ["green" if x > 0 else "red" for x in sorted_df[col].fillna(0)]
        ax.barh(sorted_df["ticker"], sorted_df[col], color=colors, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Spearman IC", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_hitrate_comparison(summary_df):
    """Plot direction hit rate across stocks"""
    fig, ax = plt.subplots(figsize=(12, 8))
    sorted_df = summary_df.sort_values("dir_hitrate", ascending=True)
    colors = ["green" if x > 50 else "red" for x in sorted_df["dir_hitrate"].fillna(50)]
    ax.barh(sorted_df["ticker"], sorted_df["dir_hitrate"], color=colors, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Hit Rate (%)", fontsize=11)
    ax.set_title("Direction Score Hit Rate (% Sign Agreement)", fontsize=12, fontweight="bold")
    ax.axvline(x=50, color="black", linestyle="--", linewidth=1.5, label="50% (Random)")
    ax.set_xlim(40, 60)
    ax.legend(fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_score_distributions(all_results):
    """Plot score distributions across all stocks"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    all_dir_scores = [s for df in all_results.values() if "direction_score" in df.columns for s in df["direction_score"].dropna()]
    all_mag_scores = [s for df in all_results.values() if "magnitude_score" in df.columns for s in df["magnitude_score"].dropna()]
    
    for ax, scores, title, color in zip(axes, [all_dir_scores, all_mag_scores], 
                                         ["Direction Score Distribution", "Magnitude Score Distribution"],
                                         ["blue", "green"]):
        ax.hist(scores, bins=60, alpha=0.7, color=color, edgecolor="black")
        ax.axvline(x=np.mean(scores), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(scores):.2f}")
        ax.set_xlabel("Score", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"{title} (All {len(all_results)} Stocks)", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_quintile_analysis(all_results):
    """Plot direction and magnitude quintile analysis across all stocks"""
    # Combine all data
    all_data = []
    for df in all_results.values():
        if all(col in df.columns for col in ["direction_score", "magnitude_score", "fwd_10d_ret", "fwd_abs_10d_ret"]):
            all_data.append(df[["direction_score", "magnitude_score", "fwd_10d_ret", "fwd_abs_10d_ret"]].dropna())
    
    if not all_data:
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Direction quintiles
    combined_df["dir_quintile"] = pd.qcut(combined_df["direction_score"], q=5, 
                                          labels=["Q1 (Worst)", "Q2", "Q3", "Q4", "Q5 (Best)"], 
                                          duplicates='drop')
    dir_quintiles = combined_df.groupby("dir_quintile", observed=True)["fwd_10d_ret"].agg(['mean', 'std', 'count']) * 100
    
    # Magnitude quintiles
    combined_df["mag_quintile"] = pd.qcut(combined_df["magnitude_score"], q=5, 
                                          labels=["Q1 (Weakest)", "Q2", "Q3", "Q4", "Q5 (Strongest)"], 
                                          duplicates='drop')
    mag_quintiles = combined_df.groupby("mag_quintile", observed=True)["fwd_abs_10d_ret"].agg(['mean', 'std', 'count']) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Direction quintiles bar chart
    ax = axes[0]
    dir_quintiles["mean"].plot(kind="bar", ax=ax, color="steelblue", alpha=0.7, edgecolor="black", 
                               yerr=dir_quintiles["std"], capsize=5)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title("Direction Score Quintiles → Average 10d Returns (All Stocks)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Direction Score Quintile", fontsize=11)
    ax.set_ylabel("Average 10d Return (%)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Add count labels
    for i, (idx, row) in enumerate(dir_quintiles.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.05, f"n={int(row['count'])}", 
                ha='center', va='bottom', fontsize=9)
    
    # Magnitude quintiles bar chart
    ax = axes[1]
    mag_quintiles["mean"].plot(kind="bar", ax=ax, color="seagreen", alpha=0.7, edgecolor="black", 
                               yerr=mag_quintiles["std"], capsize=5)
    ax.set_title("Magnitude Score Quintiles → Average Absolute 10d Returns (All Stocks)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Magnitude Score Quintile", fontsize=11)
    ax.set_ylabel("Average Absolute 10d Return (%)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Add count labels
    for i, (idx, row) in enumerate(mag_quintiles.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.05, f"n={int(row['count'])}", 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def print_summary_table(summary_df):
    """Print formatted summary table"""
    print(f"\n{'='*120}")
    print(f"BATCH RESULTS SUMMARY")
    print(f"{'='*120}\n")
    
    print(f"📊 OVERALL METRICS:")
    print(f"   Stocks analyzed: {len(summary_df)}")
    print(f"   Mean Direction IC: {summary_df['dir_ic'].mean():.4f}")
    print(f"   Mean Magnitude IC: {summary_df['mag_ic'].mean():.4f}")
    print(f"   Mean Direction Hit Rate: {summary_df['dir_hitrate'].mean():.1f}%\n")
    
    print(f"🏆 TOP 5 BY DIRECTION IC:")
    for idx, row in summary_df.nlargest(5, "dir_ic").iterrows():
        print(f"   {row['ticker']:6s} | Dir IC: {row['dir_ic']:+.4f} | Hit Rate: {row['dir_hitrate']:5.1f}% | Mag IC: {row['mag_ic']:+.4f}")
    
    print(f"\n📋 FULL TABLE (sorted by Direction IC):")
    print(summary_df[["ticker", "records", "dir_ic", "dir_hitrate", "mag_ic"]].to_string(index=False))


def main():
    print(f"\n{'='*80}")
    print(f"BATCH RESULTS AGGREGATOR")
    print(f"{'='*80}\n")
    
    all_results = load_all_composite_scores()
    if not all_results:
        print("❌ No results to aggregate")
        return
    
    print(f"\n✓ Loaded {len(all_results)} stocks\n")
    summary_df = create_summary_table(all_results)
    print_summary_table(summary_df)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 Generating comparison charts...")
    
    fig1 = plot_ic_comparison(summary_df)
    fig1.savefig(f"{OUTPUT_DIR}/01_ic_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"   ✓ Saved: {OUTPUT_DIR}/01_ic_comparison.png")
    
    fig2 = plot_hitrate_comparison(summary_df)
    fig2.savefig(f"{OUTPUT_DIR}/02_hitrate_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"   ✓ Saved: {OUTPUT_DIR}/02_hitrate_comparison.png")
    
    fig3 = plot_score_distributions(all_results)
    fig3.savefig(f"{OUTPUT_DIR}/03_score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"   ✓ Saved: {OUTPUT_DIR}/03_score_distributions.png")
    
    fig4 = plot_quintile_analysis(all_results)
    if fig4:
        fig4.savefig(f"{OUTPUT_DIR}/04_quintile_analysis.png", dpi=150, bbox_inches="tight")
        plt.close(fig4)
        print(f"   ✓ Saved: {OUTPUT_DIR}/04_quintile_analysis.png")
    
    summary_df.to_csv(f"{OUTPUT_DIR}/batch_summary.csv", index=False)
    print(f"   ✓ Saved: {OUTPUT_DIR}/batch_summary.csv")
    
    print(f"\n{'='*80}")
    print("BATCH ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
