"""
Signal Orthogonality Module for Parameter Sweep

Computes correlation matrix between all indicators and adds orthogonality metrics
to parameter sweep results.

This integrates directly with parameter_sweep_tester.py - no separate files.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class IndicatorOrthogonalityComputer:
    """
    Computes pairwise correlations between all tested indicators during parameter sweep.
    
    Instead of creating separate correlation matrix, we:
    1. Track ALL indicator signals computed during sweep (each test result)
    2. Compute correlations between them
    3. Add metrics to sweep results table
    """

    def __init__(self):
        self.indicator_signals = {}  # {parameter: np.array of signal values}
        self.correlation_matrix = None
        self.indicator_names = None
        self.metrics_df = None

    def record_indicator_signal(self, parameter: str, signal_values: np.ndarray) -> None:
        """
        Record the signal values for an indicator during sweep.
        
        Args:
            parameter: Parameter identifier (e.g., "rsi_14d")
            signal_values: Array of indicator values across all test dates
        """
        self.indicator_signals[parameter] = signal_values

    def compute_correlations(self) -> np.ndarray:
        """
        Compute Spearman correlation matrix between all recorded indicators.
        
        Returns:
            Correlation matrix (NxN)
        """
        if len(self.indicator_signals) < 2:
            return None

        print(f"\n📊 Computing correlations between {len(self.indicator_signals)} indicators...")

        # Find max length
        max_len = max(len(v) for v in self.indicator_signals.values())
        
        # Pad all signals to same length with NaN
        signals_dict = {}
        for key, arr in self.indicator_signals.items():
            if len(arr) == max_len:
                signals_dict[key] = arr
            else:
                # Pad with NaN at the end
                padded = np.full(max_len, np.nan)
                padded[:len(arr)] = arr
                signals_dict[key] = padded
        
        # Create dataframe from aligned signals
        signals_df = pd.DataFrame(signals_dict)
        
        # Remove rows with any NaN
        signals_clean = signals_df.dropna()
        
        if len(signals_clean) < 3:
            print("⚠️  Insufficient data for correlation computation (need >= 3 observations)")
            return None

        print(f"   Using {len(signals_clean)} complete observations")

        # Compute Spearman correlation (better for non-linear relationships)
        corr_matrix = signals_clean.corr(method="spearman")
        
        # Ensure symmetric (fix floating point issues)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2

        self.correlation_matrix = corr_matrix.values
        self.indicator_names = corr_matrix.columns.tolist()

        print(f"   Correlation matrix computed: {self.correlation_matrix.shape}")

        return self.correlation_matrix

    def compute_orthogonality_metrics(self) -> pd.DataFrame:
        """
        For each indicator, compute:
        - avg_abs_corr: Average absolute correlation to all other indicators
        - max_abs_corr: Maximum absolute correlation to any other indicator
        - top_collinear_partner: Indicator most correlated with this one
        - top_corr_value: The actual correlation value
        """
        if self.correlation_matrix is None:
            return pd.DataFrame()

        metrics = []

        for i, ind_name in enumerate(self.indicator_names):
            # Get correlations with all other indicators (exclude self)
            corrs = np.abs(self.correlation_matrix[i, :])
            corrs[i] = 0  # Exclude self-correlation

            # Use nanmean/nanmax to handle NaN values in correlation matrix
            avg_abs_corr = np.nanmean(corrs)
            max_abs_corr = np.nanmax(corrs)
            max_idx = np.nanargmax(corrs)
            top_partner = self.indicator_names[max_idx]
            top_corr_val = self.correlation_matrix[i, max_idx]

            metrics.append({
                "parameter": ind_name,
                "avg_abs_corr": avg_abs_corr,
                "max_abs_corr": max_abs_corr,
                "top_collinear_partner": top_partner,
                "top_corr_value": top_corr_val,
            })

        self.metrics_df = pd.DataFrame(metrics)
        return self.metrics_df

    def add_metrics_to_sweep_results(self, sweep_results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Join orthogonality metrics with parameter sweep results.
        
        Args:
            sweep_results_df: DataFrame from parameter_sweep_tester.py (has 'parameter' column)
            
        Returns:
            Enriched DataFrame with orthogonality columns
        """
        if self.metrics_df is None or self.metrics_df.empty:
            # Return original if no correlations computed
            return sweep_results_df

        # Merge on 'parameter' column
        enriched = sweep_results_df.merge(
            self.metrics_df,
            on="parameter",
            how="left"
        )

        return enriched

    def plot_heatmap(self, output_path: str = "backtest_results/orthogonality_heatmap.png") -> None:
        """Generate correlation heatmap."""
        if self.correlation_matrix is None:
            print("⚠️  No correlation matrix to plot")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot
        im = ax.imshow(
            self.correlation_matrix,
            cmap="RdBu_r",
            aspect="auto",
            vmin=-1,
            vmax=1
        )

        # Ticks and labels
        ax.set_xticks(range(len(self.indicator_names)))
        ax.set_yticks(range(len(self.indicator_names)))
        ax.set_xticklabels(self.indicator_names, rotation=90, fontsize=9)
        ax.set_yticklabels(self.indicator_names, fontsize=9)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label="Spearman Correlation")

        ax.set_title("Indicator Orthogonality: Correlation Matrix", 
                     fontsize=13, fontweight="bold", pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Heatmap saved: {output_path}")
        plt.close()

    def plot_dendrogram(self, output_path: str = "backtest_results/orthogonality_dendrogram.png") -> None:
        """Generate hierarchical clustering dendrogram."""
        if self.correlation_matrix is None:
            print("⚠️  No correlation matrix to plot")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert correlation to distance
        corr_abs = np.abs(self.correlation_matrix)
        # Ensure symmetry
        corr_abs = (corr_abs + corr_abs.T) / 2
        distance_matrix = 1.0 - corr_abs
        
        # Replace NaN with 1.0 (maximum distance - no correlation info)
        distance_matrix = np.nan_to_num(distance_matrix, nan=1.0)

        # Compute linkage
        try:
            condensed = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed, method="average")
        except Exception as e:
            print(f"⚠️  Could not compute dendrogram: {e}")
            return

        fig, ax = plt.subplots(figsize=(16, 8))

        dendrogram(
            linkage_matrix,
            labels=self.indicator_names,
            ax=ax,
            leaf_font_size=9,
            leaf_rotation=90
        )

        ax.set_title("Indicator Clustering: Hierarchical Dendrogram", 
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Distance (1 - |Spearman correlation|)", fontsize=11)
        ax.axhline(y=0.3, color="red", linestyle="--", linewidth=2, alpha=0.7,
                   label="ρ = 0.70 threshold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Dendrogram saved: {output_path}")
        plt.close()

    def print_summary(self) -> None:
        """Print orthogonality summary."""
        if self.metrics_df is None or self.metrics_df.empty:
            print("⚠️  No orthogonality metrics computed")
            return

        print("\n" + "="*80)
        print("SIGNAL ORTHOGONALITY SUMMARY")
        print("="*80)

        # High redundancy
        print("\n⚠️  HIGH REDUNDANCY (avg_abs_corr > 0.60):")
        high = self.metrics_df[self.metrics_df["avg_abs_corr"] > 0.60].sort_values("avg_abs_corr", ascending=False)
        if len(high) > 0:
            for _, row in high.iterrows():
                print(f"  {row['parameter']:25s} avg_corr={row['avg_abs_corr']:.3f}, "
                      f"partner={row['top_collinear_partner']}, value={row['top_corr_value']:+.3f}")
        else:
            print("  None")

        # Good orthogonality
        print("\n✓ GOOD ORTHOGONALITY (avg_abs_corr < 0.40):")
        good = self.metrics_df[self.metrics_df["avg_abs_corr"] < 0.40].sort_values("avg_abs_corr")
        if len(good) > 0:
            for _, row in good.iterrows():
                print(f"  {row['parameter']:25s} avg_corr={row['avg_abs_corr']:.3f}")
        else:
            print("  None")

        print("\n" + "="*80)
