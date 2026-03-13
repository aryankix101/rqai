"""
Composite Score Calculator — Aggregate direction + magnitude scores
Using percentile normalization, sign correction, and weighted aggregation
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

# ----------------------------
# Constants
# ----------------------------

# Direction sign corrections (FLIP or KEEP)
DIRECTION_SIGNS = {
    "vix_percentile_252d": 1,                         # keep
    "vol_percentile_504d": 1,                         # keep
    "vix_level": 1,                                   # keep
    "vol_imbalance_20d": -1,                          # flip
    "m2_m1_ratio": -1,                                # flip
    "atr_14d_pct": 1,                                 # keep
    "realized_vol_10d": 1,                            # keep
    "vol_adjusted_trend_ma100": -1,                   # flip
    "range_pos_55d": -1,                              # flip
    "pct_from_52w_high": -1,                          # flip
    "range_pct": 1,                                   # keep
    "autocorr_5d": -1,                                # flip
    "mom_3m": -1,                                     # flip
    "rsi_5d": -1,                                     # flip
    "vel_10d_over_vol": -1,                           # flip
    "days_since_52w_high": 1,                         # keep
    "vol_adjusted_trend_ma100_slope": -1,             # flip
}

# Direction weights
DIRECTION_WEIGHTS = {
    "vix_percentile_252d": 0.089,
    "vol_percentile_504d": 0.071,
    "vix_level": 0.055,
    "vol_imbalance_20d": 0.071,
    "m2_m1_ratio": 0.106,
    "atr_14d_pct": 0.049,
    "realized_vol_10d": 0.057,
    "vol_adjusted_trend_ma100": 0.045,
    "range_pos_55d": 0.048,
    "pct_from_52w_high": 0.047,
    "range_pct": 0.054,
    "autocorr_5d": 0.095,
    "mom_3m": 0.041,
    "rsi_5d": 0.054,
    "vel_10d_over_vol": 0.043,
    "days_since_52w_high": 0.043,
    "vol_adjusted_trend_ma100_slope": 0.0,
}

# Magnitude sign corrections (FLIP or KEEP)
MAGNITUDE_SIGNS = {
    "vix_level": 1,                                   # keep
    "pct_from_52w_high": -1,                          # flip
    "atr_14d_pct": 1,                                 # keep
    "realized_vol_10d": 1,                            # keep
    "vol_adjusted_trend_ma100": -1,                   # flip
    "range_pos_55d": -1,                              # flip
    "range_pct": 1,                                   # keep
    "days_since_52w_high": 1,                         # keep
    "pct_from_ma100": -1,                             # flip
    "vix_percentile_252d": 1,                         # keep
    "vol_adjusted_trend_ma100_slope": -1,             # flip
    "rsi_5d": -1,                                     # flip
    "mom_3m": -1,                                     # flip
    "vol_percentile_504d": 1,                         # keep
    "vel_10d_over_vol": -1,                           # flip
    "vol_imbalance_20d": -1,                          # flip
    "autocorr_5d": -1,                                # flip
}

# Magnitude weights
MAGNITUDE_WEIGHTS = {
    "vix_level": 0.080,
    "pct_from_52w_high": 0.079,
    "atr_14d_pct": 0.077,
    "realized_vol_10d": 0.073,
    "range_pct": 0.069,
    "days_since_52w_high": 0.061,
    "vix_percentile_252d": 0.057,
    "vol_adjusted_trend_ma100": 0.071,
    "range_pos_55d": 0.070,
    "mom_3m": 0.052,
    "vol_adjusted_trend_ma100_slope": 0.054,
    "vol_percentile_504d": 0.050,
    "pct_from_ma100": 0.059,
    "rsi_5d": 0.053,
    "vol_imbalance_20d": 0.044,
    "vel_10d_over_vol": 0.050,
    "autocorr_5d": 0.002,
}


# ----------------------------
# Main class
# ----------------------------
class CompositeScoreCalculator:
    """Calculate direction and magnitude composite scores."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        # Extract ticker from filename for organizing charts
        self.ticker = self._extract_ticker_from_path(data_path)
    
    def _extract_ticker_from_path(self, path: str) -> str:
        """Extract ticker symbol from file path"""
        # Expected format: signal_data_dump_TICKER_YYYY-MM-DD_YYYY-MM-DD.csv
        import re
        match = re.search(r'_(TSLA|T|AAPL|MSFT|GOOGL|AMZN|META|NVDA|JPM|BAC|SPY|[A-Z]+)_\d{4}-\d{2}-\d{2}', self.data_path)
        if match:
            return match.group(1)
        # Fallback: try to extract from filename
        filename = Path(self.data_path).stem
        parts = filename.split('_')
        if len(parts) >= 2:
            return parts[1]
        return "UNKNOWN"

    def run(self):
        """Main pipeline"""
        print(f"\n{'='*80}")
        print(f"COMPOSITE SCORE CALCULATOR")
        print(f"Input: {self.data_path}")
        print(f"{'='*80}\n")

        # Load data
        print("📖 Loading signal data...")
        self.df = pd.read_csv(self.data_path, parse_dates=["date"])
        print(f"   Loaded {len(self.df)} records")

        # Check if percentiles are already calculated (from raw_signal_dump)
        percentile_cols = [f"{ind}_percentile" for ind in set(DIRECTION_SIGNS.keys()) | set(MAGNITUDE_SIGNS.keys())]
        has_percentiles = all(col in self.df.columns for col in percentile_cols)
        
        # Validate required columns
        # Support multiple forward return horizons: 5d, 10d, 20d, 30d, 60d
        forward_horizons = [5, 10, 20, 30, 60]
        required_fwd_cols = set(["date", "adj_close", "fwd_1d_ret"])
        for h in forward_horizons:
            required_fwd_cols.add(f"fwd_{h}d_ret")
            required_fwd_cols.add(f"fwd_abs_{h}d_ret")
        
        required_cols = required_fwd_cols.copy()
        if not has_percentiles:
            # Need raw indicators to calculate percentiles
            required_cols.update(set(DIRECTION_SIGNS.keys()) | set(MAGNITUDE_SIGNS.keys()))
        missing = required_cols - set(self.df.columns)
        if missing:
            print(f"   ❌ Missing columns: {missing}")
            print(f"   ℹ️  Available forward columns: {sorted([c for c in self.df.columns if c.startswith('fwd_')])}")
            return
        print(f"   ✓ All required columns present (including forward horizons: {forward_horizons})\n")

        # Calculate percentile ranks (504d rolling, expanding) only if needed
        if has_percentiles:
            print("📊 Percentile ranks already calculated (using existing _percentile columns)\n")
        else:
            print("📊 Computing percentile ranks (504d rolling)...")
            self._compute_percentiles()

        # Calculate direction score
        print("📈 Computing direction score...")
        self._compute_direction_score()

        # Calculate magnitude score
        print("📉 Computing magnitude score...")
        self._compute_magnitude_score()

        # Calculate rolling validation metrics
        print("📋 Computing rolling validation metrics...")
        self._compute_rolling_metrics()

        # Generate visualizations
        print("📊 Generating visualizations...")
        self._create_visualizations()

        # Save results
        self._save_results()

        print(f"\n{'='*80}")
        print("CALCULATION COMPLETE")
        print(f"{'='*80}\n")

    def _compute_percentiles(self):
        """Compute 504-day rolling percentile ranks (expanding for first 504 days)"""
        all_ind_cols = list(set(DIRECTION_SIGNS.keys()) | set(MAGNITUDE_SIGNS.keys()))

        for col in all_ind_cols:
            if col not in self.df.columns:
                continue

            # Use 504-day rolling window with expanding for early dates
            percentiles = []
            for idx in range(len(self.df)):
                start_idx = max(0, idx - 504 + 1)  # 504-day window
                window_data = self.df.loc[start_idx:idx, col].dropna()

                if len(window_data) == 0:
                    percentiles.append(np.nan)
                else:
                    val = self.df.loc[idx, col]
                    if pd.isna(val):
                        percentiles.append(np.nan)
                    else:
                        pct = (window_data < val).sum() / len(window_data)
                        percentiles.append(max(0.001, min(0.999, pct)))  # Bound [0.001, 0.999]

            self.df[f"{col}_percentile"] = percentiles

    def _compute_direction_score(self):
        """
        Direction Score:
        Step C: Percentile (expanding window, min 504 days)
        Step D: Rescale to [-1, +1]: (2 × percentile) - 1
        Step E: Sign correction (+1 = keep, -1 = flip)
        Step F: After Correction: D × E
        Step G: Weight (sum to 1.0)
        Step H: Contribution: F × G × 100
        Final: Sum all contributions
        Uses DIRECTION_SIGNS and DIRECTION_WEIGHTS
        """
        contributions = []
        for ind_name, sign_correction in DIRECTION_SIGNS.items():
            percentile_col = f"{ind_name}_percentile"
            if percentile_col not in self.df.columns:
                continue

            # Step D: Rescale percentile to [-1, +1]
            rescaled = 2 * self.df[percentile_col] - 1

            # Step E,F: Sign correct (multiply by sign correction factor)
            sign_corrected = rescaled * sign_correction

            # Step G: Get weight
            weight = DIRECTION_WEIGHTS[ind_name]
            
            # Step H: Contribution = (corrected value) × weight × 100
            contribution = sign_corrected * weight * 100

            contributions.append(contribution)

        # Sum all contributions across all indicators
        self.df["direction_score"] = pd.Series(np.nansum(contributions, axis=0), index=self.df.index)

        print(f"   ✓ Direction score computed")
        print(f"     Mean: {self.df['direction_score'].mean():.2f}")
        print(f"     Std:  {self.df['direction_score'].std():.2f}")
        print(f"     Min:  {self.df['direction_score'].min():.2f}")
        print(f"     Max:  {self.df['direction_score'].max():.2f}")

    def _compute_magnitude_score(self):
        """
        Magnitude Score:
        Step C: Percentile (expanding window, min 504 days)
        Step D: Sign - 'keep' means high percentile = big move. 'flip' means LOW percentile = big move
        Step E: After Correction:
          - If keep: use percentile as-is
          - If flip: use (1 - percentile)
          - After this, higher ALWAYS = bigger expected move
        Step F: Weight (IC-proportional)
        Step G: Contribution: E × F × 100
        
        Key difference from direction: NO rescaling to [-1,+1], NO demeaning.
        We want the absolute level — 80 should always mean 'big move ahead.'
        Uses MAGNITUDE_SIGNS (different from direction signs!)
        """
        contributions = []
        
        for ind_name, weight in MAGNITUDE_WEIGHTS.items():
            percentile_col = f"{ind_name}_percentile"
            if percentile_col not in self.df.columns:
                continue

            # Get the sign for this indicator from MAGNITUDE_SIGNS
            if ind_name in MAGNITUDE_SIGNS:
                sign = MAGNITUDE_SIGNS[ind_name]
            else:
                # Default to keep (high percentile = large magnitude)
                sign = 1
            
            # Step E: After Correction
            # If sign == 1 (keep): use percentile as-is (high percentile = large magnitude)
            # If sign == -1 (flip): use (1 - percentile) (low percentile = large magnitude)
            if sign == 1:
                # Keep: high percentile = big move
                corrected = self.df[percentile_col]
            else:
                # Flip: low percentile = big move
                corrected = 1 - self.df[percentile_col]
            
            # Step G: Contribution = corrected × weight × 100
            contribution = corrected * weight * 100
            
            contributions.append(contribution)

        # Sum all contributions across all indicators
        magnitude_score = pd.Series(np.nansum(contributions, axis=0), index=self.df.index)
        self.df["magnitude_score"] = magnitude_score

        print(f"   ✓ Magnitude score computed")
        print(f"     Mean: {self.df['magnitude_score'].mean():.2f}")
        print(f"     Std:  {self.df['magnitude_score'].std():.2f}")
        print(f"     Min:  {self.df['magnitude_score'].min():.2f}")
        print(f"     Max:  {self.df['magnitude_score'].max():.2f}")

    def _compute_rolling_metrics(self):
        """
        Compute rolling validation metrics (252-day window):
        - rolling_ic_direction: Spearman(direction_score, fwd_10d_ret)
        - rolling_hitrate_direction: % sign agreement
        - rolling_ic_magnitude: Spearman(magnitude_score, fwd_abs_10d_ret)
        """
        window = 252

        # Direction IC
        rolling_ic_dir = []
        rolling_hitrate_dir = []

        for idx in range(len(self.df)):
            start_idx = max(0, idx - window + 1)
            subset = self.df.loc[start_idx:idx].dropna(subset=["direction_score", "fwd_10d_ret"])

            if len(subset) < 30:
                rolling_ic_dir.append(np.nan)
                rolling_hitrate_dir.append(np.nan)
            else:
                # IC: Spearman correlation with signed return
                ic, _ = spearmanr(subset["direction_score"], subset["fwd_10d_ret"])
                rolling_ic_dir.append(ic if np.isfinite(ic) else np.nan)

                # Hit rate: % correct sign
                sign_agreement = (
                    (subset["direction_score"] > 0) & (subset["fwd_10d_ret"] > 0)
                ) | ((subset["direction_score"] < 0) & (subset["fwd_10d_ret"] < 0))
                hitrate = sign_agreement.sum() / len(subset) * 100
                rolling_hitrate_dir.append(hitrate)

        self.df["rolling_ic_direction"] = rolling_ic_dir
        self.df["rolling_hitrate_direction"] = rolling_hitrate_dir

        # Magnitude IC: Spearman with absolute 10d forward return
        rolling_ic_mag = []

        for idx in range(len(self.df)):
            start_idx = max(0, idx - window + 1)
            subset = self.df.loc[start_idx:idx].dropna(subset=["magnitude_score", "fwd_abs_10d_ret"])

            if len(subset) < 30:
                rolling_ic_mag.append(np.nan)
            else:
                ic, _ = spearmanr(subset["magnitude_score"], subset["fwd_abs_10d_ret"])
                rolling_ic_mag.append(ic if np.isfinite(ic) else np.nan)

        self.df["rolling_ic_magnitude"] = rolling_ic_mag

        # Store for visualization
        self.rolling_ic_dir = rolling_ic_dir
        self.rolling_hitrate_dir = rolling_hitrate_dir
        self.rolling_ic_mag = rolling_ic_mag

        print(f"   ✓ Rolling metrics computed")
        print(f"     Direction IC: {np.nanmean(rolling_ic_dir):.4f}")
        print(f"     Direction hit rate: {np.nanmean(rolling_hitrate_dir):.1f}%")
        print(f"     Magnitude IC (abs 10d return): {np.nanmean(rolling_ic_mag):.4f}")

    def _create_visualizations(self):
        """Generate 5 key visualization charts"""
        # Create ticker-specific chart directory
        chart_dir = f"backtest_results/charts/{self.ticker}"
        ensure_dir(chart_dir)

        df_plot = self.df.dropna(subset=["direction_score", "magnitude_score", "fwd_10d_ret"])

        # Chart 1: Rolling IC over time
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df_plot["date"], df_plot["rolling_ic_direction"], label="Direction IC", linewidth=1.5, alpha=0.8, color="blue")
        ax.plot(df_plot["date"], df_plot["rolling_ic_magnitude"], label="Magnitude IC", linewidth=1.5, alpha=0.8, color="orange")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.fill_between(df_plot["date"], 0, df_plot["rolling_ic_direction"], alpha=0.2, color="blue")
        ax.set_title("Rolling Information Coefficient (252d window)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Spearman IC")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{chart_dir}/01_rolling_ic.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Saved: {chart_dir}/01_rolling_ic.png")

        # Chart 2: Direction Score vs Forward 10d Return (scatter)
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            df_plot["direction_score"],
            df_plot["fwd_10d_ret"] * 100,
            c=df_plot["rolling_ic_direction"],
            cmap="RdYlGn",
            s=30,
            alpha=0.5,
            vmin=-0.1,
            vmax=0.1,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Direction Score", fontsize=11)
        ax.set_ylabel("Forward 10d Return (%)", fontsize=11)
        ax.set_title("Direction Score vs Forward Return (colored by rolling IC)", fontsize=12, fontweight="bold")
        cbar = plt.colorbar(scatter, ax=ax, label="Rolling IC")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{chart_dir}/02_direction_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Saved: {chart_dir}/02_direction_scatter.png")

        # Chart 3: Magnitude Score vs Absolute 10d Return (scatter)
        fig, ax = plt.subplots(figsize=(12, 8))
        df_mag_plot = df_plot.dropna(subset=["fwd_abs_10d_ret"])
        scatter = ax.scatter(
            df_mag_plot["magnitude_score"],
            df_mag_plot["fwd_abs_10d_ret"] * 100,
            c=df_mag_plot["rolling_ic_magnitude"],
            cmap="Greens",
            s=30,
            alpha=0.5,
            vmin=0,
            vmax=0.15,
        )
        ax.set_xlabel("Magnitude Score", fontsize=11)
        ax.set_ylabel("Absolute 10d Return (%)", fontsize=11)
        ax.set_title("Magnitude Score vs Absolute 10d Return (colored by rolling IC)", fontsize=12, fontweight="bold")
        cbar = plt.colorbar(scatter, ax=ax, label="Rolling IC")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{chart_dir}/03_magnitude_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Saved: {chart_dir}/03_magnitude_scatter.png")

        # Chart 4: Distribution of scores (histograms)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(df_plot["direction_score"], bins=50, alpha=0.7, color="blue", edgecolor="black")
        axes[0].axvline(df_plot["direction_score"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {df_plot['direction_score'].mean():.2f}")
        axes[0].set_xlabel("Direction Score", fontsize=11)
        axes[0].set_ylabel("Frequency", fontsize=11)
        axes[0].set_title("Distribution of Direction Scores", fontsize=12, fontweight="bold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].hist(df_plot["magnitude_score"], bins=50, alpha=0.7, color="green", edgecolor="black")
        axes[1].axvline(df_plot["magnitude_score"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {df_plot['magnitude_score'].mean():.2f}")
        axes[1].set_xlabel("Magnitude Score", fontsize=11)
        axes[1].set_ylabel("Frequency", fontsize=11)
        axes[1].set_title("Distribution of Magnitude Scores", fontsize=12, fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{chart_dir}/04_score_distributions.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Saved: {chart_dir}/04_score_distributions.png")

        # Chart 5: Hit rate over time
        df_rolling = self.df.dropna(subset=["rolling_hitrate_direction"])
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df_rolling["date"], df_rolling["rolling_hitrate_direction"], linewidth=1.5, alpha=0.8, color="purple", label="Direction Hit Rate")
        ax.axhline(50, color="black", linestyle="--", linewidth=1, alpha=0.7, label="Random (50%)")
        ax.fill_between(df_rolling["date"], 50, df_rolling["rolling_hitrate_direction"], alpha=0.2, color="purple")
        ax.set_ylim(40, 60)
        ax.set_title("Rolling Direction Hit Rate (252d window)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Hit Rate (%)")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{chart_dir}/05_hit_rate.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Saved: backtest_results/charts/05_hit_rate.png")
        
        # Chart 6: Direction Score Quintiles → Average Returns
        df_returns = self.df.dropna(subset=["direction_score", "fwd_10d_ret"])
        df_returns["dir_quintile"] = pd.qcut(df_returns["direction_score"], q=5, labels=["Q1 (Worst)", "Q2", "Q3", "Q4", "Q5 (Best)"], duplicates='drop')
        quintile_returns = df_returns.groupby("dir_quintile", observed=True)["fwd_10d_ret"].agg(['mean', 'std', 'count']) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        quintile_returns["mean"].plot(kind="bar", ax=ax, color="steelblue", alpha=0.7, edgecolor="black", yerr=quintile_returns["std"])
        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_title("Direction Score Quintiles → Average 10d Returns", fontsize=12, fontweight="bold")
        ax.set_xlabel("Direction Score Quintile", fontsize=11)
        ax.set_ylabel("Average 10d Return (%)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{chart_dir}/06_direction_quintiles.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Saved: {chart_dir}/06_direction_quintiles.png")
        for q in quintile_returns.index:
            print(f"     {q}: {quintile_returns.loc[q, 'mean']:.2f}% ± {quintile_returns.loc[q, 'std']:.2f}% (n={int(quintile_returns.loc[q, 'count'])})")
        
        # Chart 7: Magnitude Score Quintiles → Average Absolute Returns
        df_abs_returns = self.df.dropna(subset=["magnitude_score", "fwd_abs_10d_ret"])
        df_abs_returns["mag_quintile"] = pd.qcut(df_abs_returns["magnitude_score"], q=5, labels=["Q1 (Weakest)", "Q2", "Q3", "Q4", "Q5 (Strongest)"], duplicates='drop')
        quintile_abs_returns = df_abs_returns.groupby("mag_quintile", observed=True)["fwd_abs_10d_ret"].agg(['mean', 'std', 'count']) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        quintile_abs_returns["mean"].plot(kind="bar", ax=ax, color="seagreen", alpha=0.7, edgecolor="black", yerr=quintile_abs_returns["std"])
        ax.set_title("Magnitude Score Quintiles → Average Absolute 10d Returns", fontsize=12, fontweight="bold")
        ax.set_xlabel("Magnitude Score Quintile", fontsize=11)
        ax.set_ylabel("Average Absolute 10d Return (%)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{chart_dir}/07_magnitude_quintiles.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Saved: {chart_dir}/07_magnitude_quintiles.png")
        for q in quintile_abs_returns.index:
            print(f"     {q}: {quintile_abs_returns.loc[q, 'mean']:.2f}% ± {quintile_abs_returns.loc[q, 'std']:.2f}% (n={int(quintile_abs_returns.loc[q, 'count'])})")
        
        # Chart 8: 2D Heatmap - Direction Score vs Magnitude Score → Average Returns
        df_2d = self.df.dropna(subset=["direction_score", "magnitude_score", "fwd_10d_ret"])
        df_2d["dir_bin"] = pd.qcut(df_2d["direction_score"], q=5, labels=False, duplicates='drop')
        df_2d["mag_bin"] = pd.qcut(df_2d["magnitude_score"], q=5, labels=False, duplicates='drop')
        heatmap_data = df_2d.pivot_table(values="fwd_10d_ret", index="mag_bin", columns="dir_bin", aggfunc="mean") * 100
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax, cbar_kws={"label": "Avg 10d Return (%)"}, vmin=-2, vmax=2)
        ax.set_title("2D Heatmap: Direction vs Magnitude Scores → Average 10d Returns", fontsize=12, fontweight="bold")
        ax.set_xlabel("Direction Score (Q1=Worst, Q5=Best)", fontsize=11)
        ax.set_ylabel("Magnitude Score (Q1=Weakest, Q5=Strongest)", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{chart_dir}/08_direction_magnitude_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Saved: {chart_dir}/08_direction_magnitude_heatmap.png")
        
        print(f"\n     Mean IC (direction): {np.nanmean(self.rolling_ic_dir):.4f}")
        print(f"     Mean hit rate (direction): {np.nanmean(self.rolling_hitrate_dir):.1f}%")
        print(f"     Mean IC (magnitude): {np.nanmean(self.rolling_ic_mag):.4f}")

    def _save_results(self):
        """Save results to CSV"""
        ensure_dir("backtest_results/composite_scores")

        # Select columns to export
        # Include all forward horizons: 5d, 10d, 20d, 30d, 60d
        forward_horizons = [5, 10, 20, 30, 60]
        base_cols = ["date", "adj_close", "fwd_1d_ret"]
        for h in forward_horizons:
            base_cols.append(f"fwd_{h}d_ret")
            base_cols.append(f"fwd_abs_{h}d_ret")
        
        score_cols = ["direction_score", "magnitude_score", "rolling_ic_direction", "rolling_hitrate_direction", "rolling_ic_magnitude"]

        # Direction indicator columns
        dir_ind_cols = sorted([c for c in self.df.columns if c in [f"{ind}_percentile" for ind in DIRECTION_SIGNS.keys()]])

        # Magnitude indicator columns
        mag_ind_cols = sorted([c for c in self.df.columns if c in [f"{ind}_percentile" for ind in MAGNITUDE_SIGNS.keys()]])

        export_cols = base_cols + dir_ind_cols + mag_ind_cols + score_cols

        df_export = self.df[[c for c in export_cols if c in self.df.columns]]

        # Construct output path: backtest_results/composite_scores/composite_scores_{TICKER}_{START}_{END}.csv
        filename = Path(self.data_path).name  # Get filename only
        output_filename = filename.replace("signal_data_dump", "composite_scores")
        output_path = f"backtest_results/composite_scores/{output_filename}"
        df_export.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")
        print(f"  Included forward horizons: {forward_horizons} days")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python composite_score_calculator.py DATA_PATH")
        print("Example: python composite_score_calculator.py backtest_results/signal_data_dump_SPY_2000-01-01_2025-12-31.csv")
        sys.exit(1)

    data_path = sys.argv[1]

    if not Path(data_path).exists():
        print(f"❌ File not found: {data_path}")
        sys.exit(1)

    calculator = CompositeScoreCalculator(data_path)
    calculator.run()


if __name__ == "__main__":
    main()
