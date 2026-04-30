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

from deterministic.signal_contract import REQUIRED_INDICATORS
from deterministic.signal_generation import build_signal_frame_from_history

# ----------------------------
# Config / Secrets
# ----------------------------
with open(Path(__file__).resolve().parent.parent / "secrets.json", "r") as f:
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
        warmup = 504
        print(f"   Starting from index {warmup} (warmup: {warmup} days)")
        df = build_signal_frame_from_history(self.hist)
        print(f"   ✓ Processed {len(df)} trading days\n")
        return df.to_dict("records")

    def _save_signals(self, df: pd.DataFrame):
        """Save signals to CSV with only required indicators"""
        ensure_dir("backtest_results/signal_data_dump")

        # Metadata columns first: include all forward return horizons (5d, 10d, 20d, 30d, 60d)
        metadata_cols = ["date", "adj_open", "adj_close", "fwd_1d_ret", "fwd_5d_ret", "fwd_10d_ret", "fwd_20d_ret", "fwd_30d_ret", "fwd_60d_ret",
                         "fwd_abs_5d_ret", "fwd_abs_10d_ret", "fwd_abs_20d_ret", "fwd_abs_30d_ret", "fwd_abs_60d_ret"]
        
        # Only include indicators that are in REQUIRED_INDICATORS
        indicator_cols = sorted([c for c in df.columns if c in REQUIRED_INDICATORS])

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
        print("  Forward horizons: 1d, 5d, 10d, 20d, 30d, 60d")
        print("  Execution timing: signal close -> next open entry -> +N open exit")
        print(f"\nIndicators included ({len(indicator_cols)}):")
        for col in indicator_cols:
            print(f"    - {col}")
        print("  Absolute versions included for 5d-60d")


def main():
    if len(sys.argv) < 4:
        print("Usage: python -m deterministic.signal_data_dump SYMBOL START_DATE END_DATE")
        print("Example: python -m deterministic.signal_data_dump SPY 2000-01-01 2025-12-31")
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
