"""
Signal Data Dump — Extract all raw indicator values with SPY prices and forward returns
Output: CSV with all ~90 indicators, labeled by type (direction/magnitude)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from technical_indicators_calculator import TechnicalIndicatorCalculator
from hit_rate_validator import calculate_forward_return_label

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


# Note: We dump ALL indicators from the calculator.
# The composite_score_calculator.py will filter by DIRECTION_SIGNS and MAGNITUDE_SIGNS keys.


# ----------------------------
# Main class
# ----------------------------
class SignalDataDumper:
    """Extract all indicators and forward returns for scoring pipeline."""

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.hist: pd.DataFrame = None
        self._session = make_retry_session()

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

        # Standardize column names
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

    def run(self):
        """Main pipeline"""
        print(f"\n{'='*80}")
        print(f"SIGNAL DATA DUMP")
        print(f"Symbol: {self.symbol}, Period: {self.start_date} to {self.end_date}")
        print(f"{'='*80}\n")

        # Fetch data
        print("📥 Fetching historical data...")
        self.hist = self.fetch_tiingo_data(self.symbol, self.start_date, self.end_date)
        if self.hist.empty:
            print("❌ Failed to fetch data")
            return
        print(f"   Downloaded {len(self.hist)} trading days\n")

        # Build indicators and forward returns
        print("📊 Building signal data...")
        data_records = self._build_signal_data()

        # Convert to DataFrame
        df_signals = pd.DataFrame(data_records)
        print(f"   ✓ Built {len(df_signals)} records\n")

        # Save to CSV
        self._save_signals(df_signals)

    def _build_signal_data(self) -> list:
        """Build indicator + return data for each trading day"""
        records = []

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

                # Collect all indicators
                all_indicators = {}

                # Calculate all categories
                mom = calc.calculate_momentum() or {}
                trend = calc.calculate_trend_indicators() or {}
                mean_rev = calc.calculate_mean_reversion_indicators() or {}
                trend_qual = calc.calculate_trend_quality_indicators() or {}
                range_struct = calc.calculate_range_structure_indicators() or {}
                vol = calc.calculate_volatility_indicators() or {}
                vol_liq = calc.calculate_volume_liquidity_indicators() or {}
                vol_part = calc.calculate_volume_participation_indicators() or {}
                price_events = calc.calculate_price_events() or {}

                all_indicators.update(mom)
                all_indicators.update(trend)
                all_indicators.update(mean_rev)
                all_indicators.update(trend_qual)
                all_indicators.update(range_struct)
                all_indicators.update(vol)
                all_indicators.update(vol_liq)
                all_indicators.update(vol_part)
                all_indicators.update(price_events)

                # Add VIX and term structure
                vix_indicators = calc.calculate_vix_indicators(date) or {}
                term_structure = calc.calculate_term_structure_indicators(date) or {}
                all_indicators.update(vix_indicators)
                all_indicators.update(term_structure)

                # Calculate forward returns for multiple horizons: 1d, 5d, 10d, 20d, 30d, 60d
                fwd_1d = calculate_forward_return_label(self.hist, date, 1, "next_open_to_close")
                fwd_5d = calculate_forward_return_label(self.hist, date, 5, "next_open_to_close_hold")
                fwd_10d = calculate_forward_return_label(self.hist, date, 10, "next_open_to_close_hold")
                fwd_20d = calculate_forward_return_label(self.hist, date, 20, "next_open_to_close_hold")
                fwd_30d = calculate_forward_return_label(self.hist, date, 30, "next_open_to_close_hold")
                fwd_60d = calculate_forward_return_label(self.hist, date, 60, "next_open_to_close_hold")
                
                # Absolute values for magnitude score IC
                fwd_abs_5d = abs(fwd_5d) if fwd_5d is not None and not np.isnan(fwd_5d) else np.nan
                fwd_abs_10d = abs(fwd_10d) if fwd_10d is not None and not np.isnan(fwd_10d) else np.nan
                fwd_abs_20d = abs(fwd_20d) if fwd_20d is not None and not np.isnan(fwd_20d) else np.nan
                fwd_abs_30d = abs(fwd_30d) if fwd_30d is not None and not np.isnan(fwd_30d) else np.nan
                fwd_abs_60d = abs(fwd_60d) if fwd_60d is not None and not np.isnan(fwd_60d) else np.nan

                # Get adjusted close and open (or unadjusted if not available)
                adj_close_col = "Adj_Close" if "Adj_Close" in self.hist.columns and self.hist.loc[date, "Adj_Close"] > 0 else "Close"
                adj_close = float(self.hist.loc[date, adj_close_col])
                
                adj_open_col = "Adj_Open" if "Adj_Open" in self.hist.columns and self.hist.loc[date, "Adj_Open"] > 0 else "Open"
                adj_open = float(self.hist.loc[date, adj_open_col]) if adj_open_col in self.hist.columns else adj_close

                # Build record
                record = {
                    "date": date,
                    "adj_open": adj_open,
                    "adj_close": adj_close,
                    "fwd_1d_ret": fwd_1d,
                    "fwd_5d_ret": fwd_5d,
                    "fwd_10d_ret": fwd_10d,
                    "fwd_20d_ret": fwd_20d,
                    "fwd_30d_ret": fwd_30d,
                    "fwd_60d_ret": fwd_60d,
                    "fwd_abs_5d_ret": fwd_abs_5d,
                    "fwd_abs_10d_ret": fwd_abs_10d,
                    "fwd_abs_20d_ret": fwd_abs_20d,
                    "fwd_abs_30d_ret": fwd_abs_30d,
                    "fwd_abs_60d_ret": fwd_abs_60d,
                }

                # Add all indicators with type labels
                for ind_name, ind_value in all_indicators.items():
                    if ind_value is None or isinstance(ind_value, (bool, np.bool_)):
                        record[ind_name] = np.nan
                        continue

                    try:
                        fval = float(ind_value)
                        if np.isfinite(fval):
                            record[ind_name] = fval
                        else:
                            record[ind_name] = np.nan
                    except (ValueError, TypeError):
                        record[ind_name] = np.nan

                records.append(record)
                processed_count += 1

            except Exception as e:
                if not first_error_printed:
                    print(f"\n    ⚠️  Warning: Error at idx={idx}: {str(e)[:100]}")
                    first_error_printed = True
                continue

        print(f"\n   ✓ Processed {processed_count} trading days\n")
        return records

    def _save_signals(self, df: pd.DataFrame):
        """Save signals to CSV with all indicators"""
        ensure_dir("backtest_results/signal_data_dump")

        # Metadata columns first: include all forward return horizons (5d, 10d, 20d, 30d, 60d)
        metadata_cols = ["date", "adj_close", "fwd_1d_ret", "fwd_5d_ret", "fwd_10d_ret", "fwd_20d_ret", "fwd_30d_ret", "fwd_60d_ret",
                         "fwd_abs_5d_ret", "fwd_abs_10d_ret", "fwd_abs_20d_ret", "fwd_abs_30d_ret", "fwd_abs_60d_ret"]
        indicator_cols = sorted([c for c in df.columns if c not in metadata_cols])

        column_order = metadata_cols + indicator_cols
        df_ordered = df[column_order]

        # Save main CSV
        csv_path = f"backtest_results/signal_data_dump/signal_data_dump_{self.symbol}_{self.start_date}_{self.end_date}.csv"
        df_ordered.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")

        # Print summary
        print(f"\nData Summary:")
        print(f"  Total records: {len(df)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Total indicators: {len(indicator_cols)}")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Forward horizons: 1d, 5d, 10d, 20d, 30d, 60d (with absolute versions for 5d-60d)")


def main():
    if len(sys.argv) < 4:
        print("Usage: python signal_data_dump.py SYMBOL START_DATE END_DATE")
        print("Example: python signal_data_dump.py SPY 2000-01-01 2025-12-31")
        sys.exit(1)

    symbol = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]

    dumper = SignalDataDumper(symbol, start_date, end_date)
    dumper.run()

    print(f"\n{'='*80}")
    print("DATA DUMP COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
