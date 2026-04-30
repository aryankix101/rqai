"""
Indicator Correlation Analyzer - Calculate inter-indicator correlations
to understand redundancy and relationships between technical indicators.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from technical_indicators_calculator import TechnicalIndicatorCalculator

# ----------------------------
# Config / Secrets
# ----------------------------
with open("secrets.json", "r") as f:
    secrets = json.load(f)
    TIINGO_API_KEY = secrets["TIINGO_API_KEY"]


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def make_retry_session(
    total_retries: int = 5,
    backoff_factor: float = 0.8,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ----------------------------
# Main class
# ----------------------------
class IndicatorCorrelationAnalyzer:
    """Analyze correlations between technical indicators."""

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        target_indicators: List[str],
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.target_indicators = target_indicators
        self.hist: Optional[pd.DataFrame] = None
        self._session = make_retry_session()
        self.indicator_data: Dict[str, List[float]] = {}

    def fetch_tiingo_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV data from Tiingo API"""
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {TIINGO_API_KEY}",
        }
        params = {"startDate": start_date, "endDate": end_date}

        try:
            response = self._session.get(url, headers=headers, params=params, timeout=30)
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
            return pd.DataFrame()

        if response.status_code != 200:
            print(f"Failed to fetch {ticker}: {response.status_code} | {response.text[:120]}")
            return pd.DataFrame()

        data = response.json()
        df = pd.DataFrame(data)
        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
        df = df.set_index("date").sort_index()
        
        # Tiingo returns lowercase columns; standardize to capitalized for TechnicalIndicatorCalculator
        column_mapping = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "adjOpen": "Adj_Open",
            "adjHigh": "Adj_High",
            "adjLow": "Adj_Low",
            "adjClose": "Adj_Close",
            "adjVolume": "Adj_Volume",
        }
        df = df.rename(columns=column_mapping)
        
        return df

    def run_analysis(self):
        """Main analysis pipeline"""
        print(f"\n{'='*80}")
        print(f"INDICATOR CORRELATION ANALYSIS")
        print(f"Symbol: {self.symbol}, Period: {self.start_date} to {self.end_date}")
        print(f"Target Indicators: {len(self.target_indicators)}")
        print(f"{'='*80}\n")

        # Fetch data
        print("📥 Fetching historical data...")
        self.hist = self.fetch_tiingo_data(self.symbol, self.start_date, self.end_date)
        if self.hist.empty:
            print("❌ Failed to fetch data")
            return
        print(f"   Downloaded {len(self.hist)} trading days")

        # Build indicator time series
        print("\n📊 Building indicator time series...")
        self._build_indicators()

        # Compute correlations
        print("\n🔗 Computing correlations...")
        self._compute_correlations()

    def _build_indicators(self):
        """Build time series for all target indicators (matching parameter_sweep logic)"""
        indicator_series: Dict[str, List[Tuple[pd.Timestamp, float]]] = {
            ind: [] for ind in self.target_indicators
        }

        # Warmup = 504 days (max rolling window used by any indicator)
        warmup = 504
        print(f"   Starting from index {warmup} (warmup: {warmup} days)")

        first_error_printed = False
        processed_count = 0

        for idx in range(warmup, len(self.hist)):
            if (idx - warmup) % 500 == 0 and (idx - warmup) > 0:
                print(f"   Processing day {idx - warmup}/{len(self.hist) - warmup}...", end="\r")

            date = self.hist.index[idx]
            hist_subset = self.hist.iloc[: idx + 1]

            try:
                calc = TechnicalIndicatorCalculator(hist_subset)

                # Collect all indicators (same as parameter_sweep)
                all_indicators = {}
                
                # Calculate all indicator categories
                mom = calc.calculate_momentum() or {}
                trend = calc.calculate_trend_indicators() or {}
                mean_rev = calc.calculate_mean_reversion_indicators() or {}
                trend_qual = calc.calculate_trend_quality_indicators() or {}
                range_struct = calc.calculate_range_structure_indicators() or {}
                vol = calc.calculate_volatility_indicators() or {}
                vol_liq = calc.calculate_volume_liquidity_indicators() or {}
                vol_part = calc.calculate_volume_participation_indicators() or {}

                # Merge all
                all_indicators.update(mom)
                all_indicators.update(trend)
                all_indicators.update(mean_rev)
                all_indicators.update(trend_qual)
                all_indicators.update(range_struct)
                all_indicators.update(vol)
                all_indicators.update(vol_liq)
                all_indicators.update(vol_part)

                # Add price events (contains range_pct!)
                price_events = calc.calculate_price_events() or {}
                all_indicators.update(price_events)

                # Add VIX and term structure
                vix_indicators = calc.calculate_vix_indicators(date) or {}
                term_structure = calc.calculate_term_structure_indicators(date) or {}
                all_indicators.update(vix_indicators)
                all_indicators.update(term_structure)

                # Extract target indicators only
                for ind_name in self.target_indicators:
                    if ind_name in all_indicators:
                        val = all_indicators[ind_name]
                        # Skip booleans and None values
                        if val is not None and not isinstance(val, (bool, np.bool_)):
                            try:
                                fval = float(val)
                                if np.isfinite(fval):
                                    indicator_series[ind_name].append((date, fval))
                            except (ValueError, TypeError):
                                pass

                processed_count += 1

            except Exception as e:
                if not first_error_printed:
                    print(f"\n    ⚠️  Warning: Error at idx={idx}: {str(e)[:100]}")
                    first_error_printed = True
                continue

        print(f"\n   ✓ Processed {processed_count} trading days")

        # Convert to DataFrame format
        print(f"\n   Converting to time series format...")
        for ind_name, values in indicator_series.items():
            if len(values) > 0:
                dates = [d for d, v in values]
                vals = [v for d, v in values]
                self.indicator_data[ind_name] = {
                    "dates": dates,
                    "values": vals,
                    "n_values": len(vals),
                }
                print(f"     {ind_name}: {len(vals)} values")
            else:
                print(f"     {ind_name}: ❌ NO DATA")

    def _compute_correlations(self):
        """Compute and visualize correlations between indicators"""
        print("\n   Building correlation matrix...")

        # Create aligned DataFrame: find common date range
        if not self.indicator_data:
            print("❌ No indicator data available")
            return

        valid_indicators = {k: v for k, v in self.indicator_data.items() if v["n_values"] > 0}
        if len(valid_indicators) == 0:
            print("❌ No valid indicators")
            return

        print(f"   Valid indicators: {len(valid_indicators)}")

        # Get common dates across all indicators
        all_dates = [set(v["dates"]) for v in valid_indicators.values()]
        common_dates = set.intersection(*all_dates) if all_dates else set()
        common_dates = sorted(list(common_dates))

        print(f"   Common date coverage: {len(common_dates)} days")

        if len(common_dates) < 50:
            print(f"   ⚠️  Warning: Only {len(common_dates)} common dates. Results may be unreliable.")

        # Build aligned DataFrame
        data_dict = {}
        for ind_name, ind_data in valid_indicators.items():
            date_to_val = {d: v for d, v in zip(ind_data["dates"], ind_data["values"])}
            data_dict[ind_name] = [date_to_val.get(d, np.nan) for d in common_dates]

        df_aligned = pd.DataFrame(data_dict, index=common_dates)

        # Compute correlation matrix
        print(f"   Computing Pearson correlations...")
        corr_matrix = df_aligned.corr(method="pearson")

        # Save correlation matrix
        csv_path = f"backtest_results/indicator_correlations_{self.symbol}_{self.start_date}_{self.end_date}.csv"
        ensure_dir("backtest_results")
        corr_matrix.to_csv(csv_path)
        print(f"   ✓ Saved: {csv_path}")

        # Save as JSON
        json_path = csv_path.replace(".csv", ".json")
        corr_dict = corr_matrix.to_dict()
        with open(json_path, "w") as f:
            json.dump(corr_dict, f, indent=2)
        print(f"   ✓ Saved: {json_path}")

        # Print summary statistics
        self._print_correlation_summary(corr_matrix)

        # Create heatmap
        self._create_heatmap(corr_matrix, df_aligned)

        # Create high-correlation pairs list
        self._print_high_correlations(corr_matrix)

    def _print_correlation_summary(self, corr_matrix: pd.DataFrame):
        """Print summary correlation statistics"""
        print(f"\n📈 CORRELATION SUMMARY:")
        print(f"   Matrix size: {corr_matrix.shape[0]} x {corr_matrix.shape[1]}")

        # Get upper triangle (avoid diagonal and duplicates)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_corrs = corr_matrix.values[mask]

        if len(upper_corrs) > 0:
            print(f"   Mean correlation: {upper_corrs.mean():.4f}")
            print(f"   Median correlation: {np.median(upper_corrs):.4f}")
            print(f"   Std dev: {upper_corrs.std():.4f}")
            print(f"   Min correlation: {upper_corrs.min():.4f}")
            print(f"   Max correlation: {upper_corrs.max():.4f}")

            # Count high correlations
            high_corr_count = (np.abs(upper_corrs) > 0.7).sum()
            print(f"   High correlation pairs (|ρ| > 0.7): {high_corr_count}")

    def _print_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7):
        """Print pairs with high correlations"""
        print(f"\n🔥 HIGH CORRELATION PAIRS (|ρ| > {threshold}):")

        pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    ind1 = corr_matrix.index[i]
                    ind2 = corr_matrix.columns[j]
                    pairs.append((ind1, ind2, corr_val))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        if len(pairs) == 0:
            print(f"   (None found)")
        else:
            for ind1, ind2, corr_val in pairs[:20]:  # Top 20
                print(f"   {ind1:35s} <-> {ind2:35s} : {corr_val:7.4f}")

    def _create_heatmap(self, corr_matrix: pd.DataFrame, df_aligned: pd.DataFrame):
        """Create and save correlation heatmap"""
        print(f"\n🎨 Creating heatmap...")

        fig, ax = plt.subplots(figsize=(16, 14))

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "Pearson Correlation"},
            ax=ax,
            square=True,
            linewidths=0.5,
            annot_kws={"size": 7},
        )

        ax.set_title(
            f"{self.symbol} - Inter-Indicator Correlations ({self.start_date} to {self.end_date})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        heatmap_path = (
            f"backtest_results/indicator_correlation_heatmap_{self.symbol}_{self.start_date}_{self.end_date}.png"
        )
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        print(f"   ✓ Saved: {heatmap_path}")
        plt.close()

        # Also create a smaller version without annotations for easier viewing
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "Pearson Correlation"},
            ax=ax,
            square=True,
            xticklabels=True,
            yticklabels=True,
        )
        ax.set_title(
            f"{self.symbol} - Indicator Correlations (Clean View)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        clean_path = heatmap_path.replace(".png", "_clean.png")
        plt.savefig(clean_path, dpi=150, bbox_inches="tight")
        print(f"   ✓ Saved: {clean_path}")
        plt.close()


def main():
    if len(sys.argv) < 4:
        print("Usage: python indicator_correlation_analyzer.py SYMBOL START_DATE END_DATE [IND1,IND2,...]")
        print()
        print("If indicators not provided, uses a default set.")
        print()
        print("Example:")
        print("  python indicator_correlation_analyzer.py SPY 2000-01-01 2025-12-31 'pct_from_52w_high,atr_14d_pct,mom_3m'")
        sys.exit(1)

    symbol = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]

    # Default indicators if not provided
    default_indicators = [
        "pct_from_52w_high",
        "atr_14d_pct",
        "days_since_52w_high",
        "vol_adjusted_trend_ma100",
        "mom_3m",
        "vol_adjusted_trend_ma100_slope",
        "realized_vol_10d",
        "range_pos_55d",
        "vol_imbalance_20d",
        "range_pct",
        "vel_10d_over_vol",
        "rsi_5d",
        "vix_percentile_252d",
        "pct_from_ma100",
        "vol_percentile_504d",
        "vix_level",
        "autocorr_5d",
    ]

    if len(sys.argv) > 4:
        indicators = [ind.strip() for ind in sys.argv[4].split(",")]
    else:
        indicators = default_indicators

    print(f"Analyzing {len(indicators)} indicators...")
    analyzer = IndicatorCorrelationAnalyzer(symbol, start_date, end_date, indicators)
    analyzer.run_analysis()

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
