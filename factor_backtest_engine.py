"""
THIS BUILDS EQUAL WEIGHT PORTFOLIO
Factor Backtest Engine — Test composite scores as systematic return factors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from dataclasses import dataclass
from scipy.stats import spearmanr
import scipy.stats
import warnings
import sys
import os

from data_source_config import (
    DEFAULT_COMPOSITE_SCORES_DIR,
    RUSSELL_MEMBERSHIP_PATH,
    SPY_SIGNAL_DATA_PATH,
)
from deterministic.hit_rate_validator import CANONICAL_EXECUTION_MODEL, get_trade_window
from portfolio_constraints import (
    MIN_TRADE_PRICE,
    filter_symbol_hygiene_frame,
    has_min_trade_price,
)
from portfolio_policies import (
    DEFAULT_REGIME_REVIEW_INTERVALS,
    HysteresisPolicy,
    PortfolioConstructionConfig,
    PositionState,
    ReviewDecision,
    ReviewPolicy,
    build_target_weights,
    compute_trade_decision,
)
from russell_data_loader import load_russell_ohlcv_histories

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

# ----------------------------
# Config
# ----------------------------
COMPOSITE_SCORES_DIR = DEFAULT_COMPOSITE_SCORES_DIR
OUTPUT_DIR_BASE = Path("backtest_results/factor_backtest")

# Factor construction
FORWARD_DAYS = 10 
REBALANCE_DAYS = FORWARD_DAYS  
TOP_N = 30
BOTTOM_N = 30
COST_RATE = 0.0015  # 15 bps one-way cost (commission + spread + slippage per trade)
                     # Turnover = sum(|w_new - w_old|) = 2x full notional traded (long-short)
SCORE_TYPES = ["direction_score", "magnitude_score"]
LLM_REQUIRED_COLUMNS = {
    "llm_selected",
    "llm_side",
    "llm_weight_raw",
    "llm_weight_exec",
    "llm_source",
    "llm_execution_mode",
}
_RUSSELL_DATASET_HINTS = ("russell", "ohlcv")
_PRICE_HISTORY_CACHE = {}
_MEMBERSHIP_LOOKUP_CACHE = {}
_USE_RUSSELL_DENSE_PRICE_ENRICHMENT = os.environ.get("RUSSELL_DENSE_PRICES", "0").lower() in {"1", "true", "yes"}


def _ic_target(score_type: str) -> str:
    """Forward return column for IC computation.
    
    Args:
        score_type: either 'direction_score' or 'magnitude_score'
        
    Returns:
        Column name for forward returns based on FORWARD_DAYS config:
        direction_score  → signed fwd_{FORWARD_DAYS}d_ret          (predicts direction)
        magnitude_score → absolute fwd_abs_{FORWARD_DAYS}d_ret    (predicts move size)
    """
    if score_type == "magnitude_score":
        return f"fwd_abs_{FORWARD_DAYS}d_ret"
    return f"fwd_{FORWARD_DAYS}d_ret"


@dataclass(frozen=True)
class RussellMembershipLookup:
    dates_ns: np.ndarray
    universes: tuple[frozenset[str], ...]
    path: str


def _membership_flag_is_active(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) != 0.0
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _normalize_membership_symbol(symbol) -> str:
    if pd.isna(symbol):
        return ""
    return str(symbol).strip().upper().split("-", 1)[0]


def load_russell_membership_lookup(path=RUSSELL_MEMBERSHIP_PATH) -> RussellMembershipLookup:
    membership_path = Path(path).expanduser()
    cache_key = str(membership_path.resolve()) if membership_path.exists() else str(membership_path)
    cached = _MEMBERSHIP_LOOKUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not membership_path.exists():
        raise FileNotFoundError(f"Russell membership file not found: {membership_path}")

    if membership_path.suffix.lower() == ".parquet":
        membership_df = pd.read_parquet(membership_path)
    else:
        membership_df = pd.read_csv(membership_path, parse_dates=["date"])

    required_cols = {"date", "symbol"}
    missing_cols = required_cols - set(membership_df.columns)
    if missing_cols:
        raise ValueError(
            f"Russell membership file missing required columns: {sorted(missing_cols)}"
        )

    membership_df = membership_df.copy()
    membership_df["date"] = pd.to_datetime(membership_df["date"], errors="coerce")
    membership_df["symbol"] = membership_df["symbol"].map(_normalize_membership_symbol)
    membership_df = membership_df.dropna(subset=["date"])
    membership_df = membership_df[membership_df["symbol"] != ""]

    if "in_r1000" in membership_df.columns:
        membership_df = membership_df[membership_df["in_r1000"].map(_membership_flag_is_active)]

    if membership_df.empty:
        raise ValueError(f"Russell membership file has no active rows: {membership_path}")

    membership_df = (
        membership_df[["date", "symbol"]]
        .drop_duplicates()
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )

    grouped = membership_df.groupby("date", sort=True)["symbol"].agg(
        lambda values: frozenset(str(value) for value in values if value)
    )
    dates_ns = grouped.index.to_numpy(dtype="datetime64[ns]").view("i8")
    universes = tuple(grouped.tolist())

    lookup = RussellMembershipLookup(
        dates_ns=dates_ns,
        universes=universes,
        path=str(membership_path),
    )
    _MEMBERSHIP_LOOKUP_CACHE[cache_key] = lookup
    return lookup


def _is_membership_active(
    membership_lookup: RussellMembershipLookup | None,
    ticker,
    as_of_date,
) -> bool:
    if membership_lookup is None:
        return True
    if ticker is None or as_of_date is None or membership_lookup.dates_ns.size == 0:
        return False

    target_ns = pd.Timestamp(as_of_date).value
    idx = np.searchsorted(membership_lookup.dates_ns, target_ns, side="right") - 1
    if idx < 0:
        return False

    normalized_ticker = _normalize_membership_symbol(ticker)
    if not normalized_ticker:
        return False
    return normalized_ticker in membership_lookup.universes[idx]

# ----------------------------
# Data Loading
# ----------------------------

def load_all_composites(ticker_filter=None):
    """Load all composite score CSVs, optionally filtered to specific tickers
    
    Args:
        ticker_filter: List of tickers to load. If None, loads all CSVs.
    """
    results = {}
    
    if not COMPOSITE_SCORES_DIR.exists():
        print(f"❌ Directory not found: {COMPOSITE_SCORES_DIR}")
        return results
    
    csv_files = sorted(COMPOSITE_SCORES_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {COMPOSITE_SCORES_DIR}")
        return results
    
    # Filter CSV files if ticker list provided
    if ticker_filter:
        ticker_set = {t.upper() for t in ticker_filter}
        csv_files = [f for f in csv_files if f.stem.split("_")[2].upper() in ticker_set]
    
    
    print(f"📂 Loading {len(csv_files)} stocks...\n")
    filtered_rows = 0
    touched_tickers = 0
    
    for csv_file in csv_files:
        try:
            ticker = csv_file.stem.split("_")[2]
            df = pd.read_csv(csv_file, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            if "adj_close" not in df.columns:
                raise ValueError(
                    "missing required column ['adj_close']; "
                    "re-run composite_score_calculator.py with the updated signal dump"
                )
            cleaned = filter_symbol_hygiene_frame(df)
            removed = len(df) - len(cleaned)
            if removed > 0:
                filtered_rows += removed
                touched_tickers += 1
            df = cleaned
            if df.empty:
                continue
            results[ticker] = df
            print(f"   ✓ {ticker:6s} - {len(df)} records")
        except Exception as e:
            print(f"   ❌ {csv_file.name}: {e}")
    if filtered_rows > 0:
        print(
            f"\n   ✓ Symbol hygiene filter removed {filtered_rows} ambiguous/recycled rows "
            f"across {touched_tickers} tickers"
        )
    
    return results


def load_spy_data(spy_path=str(SPY_SIGNAL_DATA_PATH)):
    """Load SPY signal data for regime calculation"""
    if not Path(spy_path).exists():
        print(f"⚠️  SPY data not found at {spy_path}")
        return None
    
    try:
        spy_df = pd.read_csv(spy_path, parse_dates=["date"])
        spy_df = spy_df.sort_values("date").reset_index(drop=True)
        print(f"   ✓ SPY data loaded: {len(spy_df)} records\n")
        return spy_df
    except Exception as e:
        print(f"⚠️  Error loading SPY data: {e}")
        return None


def get_rebalance_dates(spy_dates, rebalance_days=10):
    """Get rebalance dates from SPY calendar (master calendar)
    
    Uses SPY trading dates as the official calendar.
    Selects every Nth date (N=rebalance_days) as a rebalance point.
    All dates are normalized to pd.Timestamp for consistency.
    """
    spy_dates = pd.to_datetime(spy_dates)
    spy_dates = sorted(spy_dates.unique())
    rebalance_dates = spy_dates[::rebalance_days]
    return spy_dates, rebalance_dates


def _build_execution_schedule(calendar_dates, signal_dates):
    """Map each signal close date to the next trading day's execution open."""
    calendar = [pd.Timestamp(d) for d in sorted(pd.to_datetime(calendar_dates).unique())]
    date_to_idx = {date: idx for idx, date in enumerate(calendar)}
    execution_schedule = {}

    for signal_date in sorted(pd.to_datetime(signal_dates)):
        signal_ts = pd.Timestamp(signal_date)
        idx = date_to_idx.get(signal_ts)
        if idx is None or idx + 1 >= len(calendar):
            continue
        execution_schedule[calendar[idx + 1]] = signal_ts

    return execution_schedule


def _cross_sectional_snapshot(
    all_data,
    signal_date,
    score_type="direction_score",
    require_exact_dates: bool | None = None,
    selected_only: bool = False,
    membership_lookup: RussellMembershipLookup | None = None,
):
    """Scores and realized forward returns from a single signal date."""
    ret_col = _ic_target(score_type)
    rows = []
    if require_exact_dates is None:
        require_exact_dates = _looks_like_russell_dataset(all_data) and _is_close_only_dataset(all_data)

    for ticker, df in all_data.items():
        if not _is_membership_active(membership_lookup, ticker, signal_date):
            continue
        row = _row_on_exact_date(df, signal_date) if require_exact_dates else _row_on_or_before(df, signal_date)
        if row is None or ret_col not in row.index:
            continue

        if _has_explicit_llm_columns(df):
            if selected_only and int(row.get("llm_selected", 0)) != 1:
                continue
            score = row.get(f"base_{score_type}", row.get(score_type, np.nan))
        else:
            score = row.get(score_type, np.nan)
        ret_pct = row.get(ret_col, np.nan)
        if pd.notna(score) and pd.notna(ret_pct):
            rows.append({
                "ticker": ticker,
                "score": float(score),
                "ret": float(ret_pct) / 100.0,
            })

    return pd.DataFrame(rows)


def _has_explicit_llm_columns(df):
    return LLM_REQUIRED_COLUMNS.issubset(set(df.columns))


def _is_explicit_llm_backtest(all_data):
    return any(_has_explicit_llm_columns(df) for df in all_data.values())


def _available_explicit_llm_signal_dates(all_data):
    """Return the cached LLM decision dates present in the score files."""
    available_dates = set()
    for df in all_data.values():
        if not _has_explicit_llm_columns(df):
            continue
        selected_rows = df[df["llm_selected"] == 1]
        if selected_rows.empty:
            continue
        available_dates.update(pd.to_datetime(selected_rows["date"]).tolist())
    return sorted(pd.Timestamp(date) for date in available_dates)


def _resolve_explicit_llm_signal_date(signal_date, available_signal_dates):
    """Use the latest cached LLM decision date on or before the requested signal date."""
    if signal_date is None or not available_signal_dates:
        return None

    signal_ts = pd.Timestamp(signal_date)
    resolved = None
    for candidate in available_signal_dates:
        if candidate > signal_ts:
            break
        resolved = candidate
    return resolved


def _build_explicit_llm_weights(
    all_data,
    signal_date,
    trade_date,
    prepared_ticker_data,
    available_signal_dates=None,
    membership_lookup: RussellMembershipLookup | None = None,
):
    """Build portfolio weights from explicit LLM selection metadata."""
    resolved_signal_date = _resolve_explicit_llm_signal_date(signal_date, available_signal_dates or [])
    if resolved_signal_date is None:
        resolved_signal_date = pd.Timestamp(signal_date) if signal_date is not None else None
    if resolved_signal_date is None:
        return {}

    longs = []
    shorts = []
    execution_mode = None

    for ticker, df in all_data.items():
        if not _has_explicit_llm_columns(df):
            continue
        if not _is_membership_active(membership_lookup, ticker, resolved_signal_date):
            continue

        row = df[df["date"] == resolved_signal_date]
        if row.empty:
            continue
        row = row.iloc[0]

        if int(row.get("llm_selected", 0)) != 1:
            continue

        prepared = prepared_ticker_data.get(ticker)
        price = _lookup_ffill_price(prepared, trade_date) if prepared is not None else np.nan
        if not has_min_trade_price(price):
            continue

        side = str(row.get("llm_side", "")).upper()
        weight_exec = abs(float(row.get("llm_weight_exec", 0.0)))
        execution_mode = execution_mode or str(row.get("llm_execution_mode", "equal_weight"))

        if side == "LONG":
            longs.append((ticker, weight_exec))
        elif side == "SHORT":
            shorts.append((ticker, weight_exec))

    if not longs or not shorts:
        return {}

    new_weights = {}
    if execution_mode == "llm_weighted":
        long_total = sum(weight for _, weight in longs)
        short_total = sum(weight for _, weight in shorts)
        if long_total <= 0 or short_total <= 0:
            return {}
        for ticker, weight in longs:
            new_weights[ticker] = weight / long_total
        for ticker, weight in shorts:
            new_weights[ticker] = -(weight / short_total)
    else:
        for ticker, _ in longs:
            new_weights[ticker] = 1.0 / len(longs)
        for ticker, _ in shorts:
            new_weights[ticker] = -(1.0 / len(shorts))

    return new_weights


def _explicit_llm_snapshot(
    all_data,
    score_type,
    signal_date,
    trade_date,
    prepared_ticker_data,
    available_signal_dates=None,
    membership_lookup: RussellMembershipLookup | None = None,
):
    """Collect explicit LLM selections with prices/scores for logging and execution."""
    resolved_signal_date = _resolve_explicit_llm_signal_date(signal_date, available_signal_dates or [])
    if resolved_signal_date is None:
        resolved_signal_date = pd.Timestamp(signal_date) if signal_date is not None else None
    if resolved_signal_date is None:
        return {
            "resolved_signal_date": None,
            "date_scores": {},
            "long_candidates": [],
            "short_candidates": [],
            "long_rank_map": {},
            "short_rank_map": {},
            "target_weights": {},
        }

    raw_longs = []
    raw_shorts = []
    execution_mode = None

    for ticker, df in all_data.items():
        if not _has_explicit_llm_columns(df):
            continue
        if not _is_membership_active(membership_lookup, ticker, resolved_signal_date):
            continue

        row = df[df["date"] == resolved_signal_date]
        if row.empty:
            continue
        row = row.iloc[0]

        if int(row.get("llm_selected", 0)) != 1:
            continue

        prepared = prepared_ticker_data.get(ticker)
        price = _lookup_ffill_price(prepared, trade_date) if prepared is not None else np.nan
        if not has_min_trade_price(price):
            continue

        side = str(row.get("llm_side", "")).upper()
        weight_exec = abs(float(row.get("llm_weight_exec", 0.0)))
        score = row.get(f"base_{score_type}", row.get(score_type, np.nan))
        execution_mode = execution_mode or str(row.get("llm_execution_mode", "equal_weight"))

        payload = {
            "ticker": ticker,
            "score": float(score) if pd.notna(score) else np.nan,
            "price": float(price),
            "weight_exec": weight_exec,
        }
        if side == "LONG":
            raw_longs.append(payload)
        elif side == "SHORT":
            raw_shorts.append(payload)

    def _sort_key(item, reverse=False):
        score = item["score"]
        if pd.isna(score):
            return (-np.inf if reverse else np.inf, item["ticker"])
        return (score, item["ticker"])

    sorted_longs = sorted(raw_longs, key=lambda item: _sort_key(item, reverse=True), reverse=True)
    sorted_shorts = sorted(raw_shorts, key=lambda item: _sort_key(item, reverse=False))

    date_scores = {
        item["ticker"]: {"score": item["score"], "price": item["price"]}
        for item in sorted_longs + sorted_shorts
    }
    long_candidates = [item["ticker"] for item in sorted_longs]
    short_candidates = [item["ticker"] for item in sorted_shorts]
    long_rank_map = {ticker: idx + 1 for idx, ticker in enumerate(long_candidates)}
    short_rank_map = {ticker: idx + 1 for idx, ticker in enumerate(short_candidates)}

    target_weights = {}
    if long_candidates and short_candidates:
        if execution_mode == "llm_weighted":
            long_total = sum(item["weight_exec"] for item in sorted_longs)
            short_total = sum(item["weight_exec"] for item in sorted_shorts)
            if long_total > 0 and short_total > 0:
                for item in sorted_longs:
                    target_weights[item["ticker"]] = item["weight_exec"] / long_total
                for item in sorted_shorts:
                    target_weights[item["ticker"]] = -(item["weight_exec"] / short_total)
        else:
            for item in sorted_longs:
                target_weights[item["ticker"]] = 1.0 / len(sorted_longs)
            for item in sorted_shorts:
                target_weights[item["ticker"]] = -(1.0 / len(sorted_shorts))

    return {
        "resolved_signal_date": resolved_signal_date,
        "date_scores": date_scores,
        "long_candidates": long_candidates,
        "short_candidates": short_candidates,
        "long_rank_map": long_rank_map,
        "short_rank_map": short_rank_map,
        "target_weights": target_weights,
    }


def identify_regimes(spy_df):
    """Identify market regimes based on SPY data
    
    Volatility regime (VIX percentile from SPY):
    - high_vol: VIX percentile > 80th percentile
    - low_vol: VIX percentile < 30th percentile
    - mid_vol: 30-80
    
    Trend regime (SPY 200d MA):
    - uptrend: Close > 200d MA
    - downtrend: Close <= 200d MA
    
    Combined regimes:
    - crisis: high_vol + downtrend
    - squeeze: high_vol + uptrend
    - grind_up: low_vol + uptrend
    - slow_bleed: low_vol + downtrend
    - normal: mid_vol + any trend
    """
    regime_map = {}
    
    if spy_df is None or len(spy_df) == 0:
        print("❌ SPY data required for regime identification")
        return regime_map
    
    # Calculate SPY 200d MA — require a full 200-day window so early dates
    # produce NaN and are correctly labeled 'unknown' (not a noisy partial MA).
    spy_df = spy_df.sort_values("date").reset_index(drop=True)
    spy_df["spy_200d_ma"] = spy_df["adj_close"].rolling(window=200, min_periods=200).mean()
    
    # Compute VIX percentile thresholds
    if "vix_percentile_252d" in spy_df.columns:
        vix_pct_col = "vix_percentile_252d"
    elif "vix_percentile" in spy_df.columns:
        vix_pct_col = "vix_percentile"
    else:
        print("⚠️  VIX percentile column not found, using constant thresholds")
        vix_pct_high = 0.80
        vix_pct_low = 0.30
        use_percentiles = False
    
    if "vix_percentile_252d" in spy_df.columns or "vix_percentile" in spy_df.columns:
        # Expanding-window thresholds: on any given date, the threshold is
        # computed only from VIX history up to that date — no future data.
        spy_df["_vix_thresh_high"] = spy_df[vix_pct_col].expanding().quantile(0.80)
        spy_df["_vix_thresh_low"]  = spy_df[vix_pct_col].expanding().quantile(0.30)
        use_percentiles = True
    else:
        use_percentiles = False
    
    # Build regime map
    for idx, row in spy_df.iterrows():
        date = row["date"]
        close = row["adj_close"]
        ma200 = row["spy_200d_ma"]
        
        # Get volatility regime
        if use_percentiles:
            vix_val        = row.get(vix_pct_col, np.nan)
            thresh_high    = row.get("_vix_thresh_high", np.nan)
            thresh_low     = row.get("_vix_thresh_low", np.nan)
            if pd.isna(vix_val) or pd.isna(thresh_high) or pd.isna(thresh_low):
                regime_map[date] = "unknown"
                continue

            if vix_val > thresh_high:
                vol_regime = "high_vol"
            elif vix_val < thresh_low:
                vol_regime = "low_vol"
            else:
                vol_regime = "mid_vol"
        else:
            vol_regime = "mid_vol"
        
        # Get trend regime
        if pd.isna(close) or pd.isna(ma200):
            regime_map[date] = "unknown"
            continue
        
        trend_regime = "uptrend" if close > ma200 else "downtrend"
        
        # Combined regime
        if vol_regime == "high_vol" and trend_regime == "downtrend":
            regime_map[date] = "crisis"
        elif vol_regime == "high_vol" and trend_regime == "uptrend":
            regime_map[date] = "squeeze"
        elif vol_regime == "low_vol" and trend_regime == "uptrend":
            regime_map[date] = "grind_up"
        elif vol_regime == "low_vol" and trend_regime == "downtrend":
            regime_map[date] = "slow_bleed"
        else:  # mid_vol
            regime_map[date] = "normal"
    
    return regime_map


# ----------------------------
# Portfolio Construction
# ----------------------------

def _execution_price_column(df: pd.DataFrame) -> str:
    if "adj_open" in df.columns:
        return "adj_open"
    return "adj_close"


@dataclass
class PreparedTickerData:
    price_dates_ns: np.ndarray
    price_values: np.ndarray
    score_dates_ns: np.ndarray
    score_values: np.ndarray
    direction_dates_ns: np.ndarray
    direction_values: np.ndarray
    magnitude_dates_ns: np.ndarray
    magnitude_values: np.ndarray


@dataclass(frozen=True)
class CandidatePoolConfig:
    mode: str = "raw"
    direction_threshold: float = 5.0
    top_n_magnitude_per_side: int = 60

    def is_default(self) -> bool:
        return self.mode == "raw"

    def describe(self) -> str:
        if self.mode == "raw":
            return "raw score ranking across the full tradable universe"
        if self.mode == "llm_mirror_fill_raw":
            return (
                "LLM-mirror pool with raw backfill: "
                f"dir>|{self.direction_threshold:g}| split, "
                f"top {self.top_n_magnitude_per_side} by magnitude per side, "
                "rank within side by the selected score, then raw-fill shortages"
            )
        return (
            "LLM-mirror pool: "
            f"dir>|{self.direction_threshold:g}| split, "
            f"top {self.top_n_magnitude_per_side} by magnitude per side, "
            "then rank within side by the selected score"
        )

    def label_suffix(self) -> str:
        if self.mode == "raw":
            return "pool_raw"
        threshold = str(self.direction_threshold).replace(".", "p")
        return f"pool_{self.mode}_m{self.top_n_magnitude_per_side}_d{threshold}"


def _resolve_price_column(price_df: pd.DataFrame, preferred_price_column: str) -> str:
    if preferred_price_column in price_df.columns:
        return preferred_price_column
    if "adj_close" in price_df.columns:
        return "adj_close"
    if "adj_open" in price_df.columns:
        return "adj_open"
    raise ValueError(f"No compatible price column found in {list(price_df.columns)}")


def _prepare_ticker_data(
    df: pd.DataFrame,
    score_type: str,
    price_df: pd.DataFrame | None = None,
    preferred_price_column: str | None = None,
) -> PreparedTickerData:
    """Store compact per-ticker arrays for score lookups and ffilled price lookups."""
    clean_df = filter_symbol_hygiene_frame(df)

    def _numeric_column_or_nan(column_name: str) -> pd.Series:
        if column_name in clean_df.columns:
            return pd.to_numeric(clean_df[column_name], errors="coerce")
        return pd.Series(np.nan, index=clean_df.index, dtype=float)

    price_source = price_df if price_df is not None else clean_df
    if price_df is not None and "date" in price_df.columns and "date" in clean_df.columns:
        allowed_dates = pd.to_datetime(clean_df["date"], errors="coerce").dropna().unique()
        price_source = price_df[pd.to_datetime(price_df["date"], errors="coerce").isin(allowed_dates)].copy()
    price_col = _resolve_price_column(price_source, preferred_price_column or _execution_price_column(clean_df))
    price_table = pd.DataFrame(
        {
            "date": pd.to_datetime(price_source["date"], errors="coerce"),
            "price": pd.to_numeric(price_source[price_col], errors="coerce"),
        }
    )
    price_table = (
        price_table.dropna(subset=["date", "price"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    score_table = pd.DataFrame(
        {
            "date": pd.to_datetime(clean_df["date"], errors="coerce"),
            "score": pd.to_numeric(clean_df[score_type], errors="coerce"),
        }
    )
    score_table = (
        score_table.dropna(subset=["date", "score"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    direction_table = pd.DataFrame(
        {
            "date": pd.to_datetime(clean_df["date"], errors="coerce"),
            "value": _numeric_column_or_nan("direction_score"),
        }
    )
    direction_table = (
        direction_table.dropna(subset=["date", "value"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    magnitude_table = pd.DataFrame(
        {
            "date": pd.to_datetime(clean_df["date"], errors="coerce"),
            "value": _numeric_column_or_nan("magnitude_score"),
        }
    )
    magnitude_table = (
        magnitude_table.dropna(subset=["date", "value"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    # Force nanosecond resolution across pandas versions before converting to int64.
    # Pandas 2 can preserve lower-resolution datetime64 units (for example us/s),
    # while Timestamp.value is always nanoseconds. Mixing those units breaks searchsorted.
    price_dates_ns = price_table["date"].to_numpy(dtype="datetime64[ns]").view("i8")
    price_values = price_table["price"].to_numpy(dtype=np.float32, copy=True)
    score_dates_ns = score_table["date"].to_numpy(dtype="datetime64[ns]").view("i8")
    score_values = score_table["score"].to_numpy(dtype=np.float32, copy=True)
    direction_dates_ns = direction_table["date"].to_numpy(dtype="datetime64[ns]").view("i8")
    direction_values = direction_table["value"].to_numpy(dtype=np.float32, copy=True)
    magnitude_dates_ns = magnitude_table["date"].to_numpy(dtype="datetime64[ns]").view("i8")
    magnitude_values = magnitude_table["value"].to_numpy(dtype=np.float32, copy=True)

    return PreparedTickerData(
        price_dates_ns=price_dates_ns,
        price_values=price_values,
        score_dates_ns=score_dates_ns,
        score_values=score_values,
        direction_dates_ns=direction_dates_ns,
        direction_values=direction_values,
        magnitude_dates_ns=magnitude_dates_ns,
        magnitude_values=magnitude_values,
    )


def _lookup_ffill_price(prepared: PreparedTickerData, trade_date) -> float:
    if prepared.price_dates_ns.size == 0 or trade_date is None:
        return np.nan
    target_ns = pd.Timestamp(trade_date).value
    idx = np.searchsorted(prepared.price_dates_ns, target_ns, side="right") - 1
    if idx < 0:
        return np.nan
    return float(prepared.price_values[idx])


def _lookup_exact_price(prepared: PreparedTickerData, trade_date) -> float:
    if prepared.price_dates_ns.size == 0 or trade_date is None:
        return np.nan
    target_ns = pd.Timestamp(trade_date).value
    idx = np.searchsorted(prepared.price_dates_ns, target_ns, side="left")
    if idx >= prepared.price_dates_ns.size or prepared.price_dates_ns[idx] != target_ns:
        return np.nan
    return float(prepared.price_values[idx])


def _lookup_value_on_or_before(dates_ns: np.ndarray, values: np.ndarray, signal_date) -> float:
    if dates_ns.size == 0 or signal_date is None:
        return np.nan
    target_ns = pd.Timestamp(signal_date).value
    idx = np.searchsorted(dates_ns, target_ns, side="right") - 1
    if idx < 0:
        return np.nan
    return float(values[idx])


def _lookup_exact_value(dates_ns: np.ndarray, values: np.ndarray, signal_date) -> float:
    if dates_ns.size == 0 or signal_date is None:
        return np.nan
    target_ns = pd.Timestamp(signal_date).value
    idx = np.searchsorted(dates_ns, target_ns, side="left")
    if idx >= dates_ns.size or dates_ns[idx] != target_ns:
        return np.nan
    return float(values[idx])


def _lookup_score_on_or_before(prepared: PreparedTickerData, signal_date) -> float:
    return _lookup_value_on_or_before(prepared.score_dates_ns, prepared.score_values, signal_date)


def _lookup_exact_score(prepared: PreparedTickerData, signal_date) -> float:
    return _lookup_exact_value(prepared.score_dates_ns, prepared.score_values, signal_date)


def _is_close_only_dataset(all_data) -> bool:
    if not all_data:
        return False
    return all("adj_open" not in df.columns for df in all_data.values())


def _build_trade_signal_map(calendar_dates, close_only: bool = False):
    """Map execution dates to the signal date used for portfolio construction."""
    calendar = [pd.Timestamp(d) for d in sorted(pd.to_datetime(calendar_dates).unique())]
    if close_only:
        return {date: date for date in calendar}
    return {calendar[idx]: calendar[idx - 1] for idx in range(1, len(calendar))}


def _filter_trading_dates(calendar_dates, start_date=None, end_date=None):
    filtered = []
    for date in sorted(pd.to_datetime(calendar_dates).unique()):
        ts = pd.Timestamp(date)
        if start_date is not None and ts < start_date:
            continue
        if end_date is not None and ts > end_date:
            continue
        filtered.append(ts)
    return filtered


def _row_on_or_before(df: pd.DataFrame, target_date):
    if df is None or df.empty or target_date is None or "date" not in df.columns:
        return None

    dates = pd.to_datetime(df["date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    if dates.size == 0:
        return None

    idx = np.searchsorted(dates, np.datetime64(pd.Timestamp(target_date)), side="right") - 1
    if idx < 0:
        return None
    return df.iloc[int(idx)]


def _row_on_exact_date(df: pd.DataFrame, target_date):
    if df is None or df.empty or target_date is None or "date" not in df.columns:
        return None

    dates = pd.to_datetime(df["date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    if dates.size == 0:
        return None

    target = np.datetime64(pd.Timestamp(target_date))
    idx = np.searchsorted(dates, target, side="left")
    if idx >= len(df) or dates[idx] != target:
        return None
    return df.iloc[int(idx)]


def _looks_like_russell_dataset(all_data) -> bool:
    dataset_tokens = set()
    for df in all_data.values():
        for col in ("dataset_label", "dataset_version"):
            if col not in df.columns:
                continue
            values = df[col].dropna().astype(str).str.lower().unique().tolist()
            dataset_tokens.update(values[:5])
        if dataset_tokens:
            break

    if not dataset_tokens:
        return False

    dataset_text = " ".join(sorted(dataset_tokens))
    return all(token in dataset_text for token in _RUSSELL_DATASET_HINTS)


def _load_dense_price_histories(tickers) -> dict[str, pd.DataFrame]:
    cache_key = tuple(sorted(str(ticker).upper() for ticker in tickers))
    if not cache_key:
        return {}
    if cache_key in _PRICE_HISTORY_CACHE:
        return _PRICE_HISTORY_CACHE[cache_key]

    histories = load_russell_ohlcv_histories(tickers=cache_key)
    cleaned = {}
    for ticker, df in histories.items():
        if df.empty or "date" not in df.columns:
            continue
        cols = [col for col in ("date", "adj_open", "adj_close") if col in df.columns]
        if len(cols) < 2:
            continue
        cleaned[str(ticker).upper()] = (
            df[cols]
            .copy()
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

    _PRICE_HISTORY_CACHE[cache_key] = cleaned
    return cleaned


def _candidate_snapshot(
    score_type,
    prepared_ticker_data,
    signal_date,
    trade_date,
    require_exact_dates=False,
    candidate_pool_config: CandidatePoolConfig | None = None,
    membership_lookup: RussellMembershipLookup | None = None,
):
    candidate_pool_config = candidate_pool_config or CandidatePoolConfig()
    date_scores = {}
    for ticker, prepared in prepared_ticker_data.items():
        if not _is_membership_active(membership_lookup, ticker, signal_date):
            continue
        if require_exact_dates:
            score = _lookup_exact_score(prepared, signal_date)
            price = _lookup_exact_price(prepared, trade_date)
            direction_score = _lookup_exact_value(
                prepared.direction_dates_ns,
                prepared.direction_values,
                signal_date,
            )
            magnitude_score = _lookup_exact_value(
                prepared.magnitude_dates_ns,
                prepared.magnitude_values,
                signal_date,
            )
        else:
            score = _lookup_score_on_or_before(prepared, signal_date)
            price = _lookup_ffill_price(prepared, trade_date)
            direction_score = _lookup_value_on_or_before(
                prepared.direction_dates_ns,
                prepared.direction_values,
                signal_date,
            )
            magnitude_score = _lookup_value_on_or_before(
                prepared.magnitude_dates_ns,
                prepared.magnitude_values,
                signal_date,
            )
        if pd.notna(score) and has_min_trade_price(price):
            date_scores[ticker] = {
                "score": float(score),
                "price": float(price),
                "direction_score": float(direction_score) if pd.notna(direction_score) else np.nan,
                "magnitude_score": float(magnitude_score) if pd.notna(magnitude_score) else np.nan,
            }

    if not date_scores:
        return {
            "date_scores": {},
            "long_candidates": [],
            "short_candidates": [],
            "long_rank_map": {},
            "short_rank_map": {},
        }

    raw_long_candidates = sorted(
        date_scores.keys(),
        key=lambda ticker: (date_scores[ticker]["score"], ticker),
        reverse=True,
    )
    raw_short_candidates = sorted(
        date_scores.keys(),
        key=lambda ticker: (date_scores[ticker]["score"], ticker),
    )

    if candidate_pool_config.mode in {"llm_mirror", "llm_mirror_fill_raw"}:
        threshold = float(candidate_pool_config.direction_threshold)
        top_n_magnitude = int(candidate_pool_config.top_n_magnitude_per_side)

        long_pool = [
            ticker
            for ticker, payload in date_scores.items()
            if pd.notna(payload["direction_score"])
            and payload["direction_score"] > threshold
            and pd.notna(payload["magnitude_score"])
        ]
        short_pool = [
            ticker
            for ticker, payload in date_scores.items()
            if pd.notna(payload["direction_score"])
            and payload["direction_score"] < -threshold
            and pd.notna(payload["magnitude_score"])
        ]

        long_pool = sorted(
            long_pool,
            key=lambda ticker: (date_scores[ticker]["magnitude_score"], ticker),
            reverse=True,
        )[:top_n_magnitude]
        short_pool = sorted(
            short_pool,
            key=lambda ticker: (date_scores[ticker]["magnitude_score"], ticker),
            reverse=True,
        )[:top_n_magnitude]

        long_candidates = sorted(
            long_pool,
            key=lambda ticker: (date_scores[ticker]["score"], ticker),
            reverse=True,
        )
        short_candidates = sorted(
            short_pool,
            key=lambda ticker: (date_scores[ticker]["score"], ticker),
        )
        if candidate_pool_config.mode == "llm_mirror_fill_raw":
            long_seen = set(long_candidates)
            short_seen = set(short_candidates)
            long_candidates.extend([ticker for ticker in raw_long_candidates if ticker not in long_seen])
            short_candidates.extend([ticker for ticker in raw_short_candidates if ticker not in short_seen])
        else:
            filtered_tickers = set(long_pool) | set(short_pool)
            date_scores = {ticker: date_scores[ticker] for ticker in filtered_tickers}
    else:
        long_candidates = raw_long_candidates
        short_candidates = raw_short_candidates

    long_rank_map = {ticker: idx + 1 for idx, ticker in enumerate(long_candidates)}
    short_rank_map = {ticker: idx + 1 for idx, ticker in enumerate(short_candidates)}
    return {
        "date_scores": date_scores,
        "long_candidates": long_candidates,
        "short_candidates": short_candidates,
        "long_rank_map": long_rank_map,
        "short_rank_map": short_rank_map,
    }


def _review_signal_dates(review_log_df):
    if review_log_df is None or review_log_df.empty:
        return []
    return sorted(
        pd.to_datetime(review_log_df["signal_date"], errors="coerce").dropna().unique()
    )


def _summarize_policy_label(config: PortfolioConstructionConfig) -> str:
    return (
        f"review={config.review_policy.mode}"
        f", static_days={config.review_policy.static_review_days}"
        f", hysteresis={config.hysteresis_policy.mode}"
        f", book={config.book_sizing_mode}"
    )


def _close_position_segment(
    holdings_records,
    position: PositionState,
    exit_date,
    exit_signal_date,
    exit_price,
    exit_reason,
    calendar_index,
):
    if position.entry_price is None or exit_price is None:
        realized_return_pct = np.nan
    else:
        realized_return_pct = (float(exit_price) / float(position.entry_price) - 1.0) * 100.0
        if position.side == "SHORT":
            realized_return_pct = -realized_return_pct

    entry_idx = calendar_index.get(pd.Timestamp(position.entry_date))
    exit_idx = calendar_index.get(pd.Timestamp(exit_date))
    holding_days = exit_idx - entry_idx if entry_idx is not None and exit_idx is not None else np.nan

    holdings_records.append({
        "ticker": position.ticker,
        "position_type": position.side,
        "entry_date": pd.Timestamp(position.entry_date),
        "exit_date": pd.Timestamp(exit_date),
        "entry_signal_date": pd.Timestamp(position.entry_signal_date) if position.entry_signal_date is not None else pd.NaT,
        "exit_signal_date": pd.Timestamp(exit_signal_date) if exit_signal_date is not None else pd.NaT,
        "entry_weight": position.weight,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "entry_score": position.entry_score,
        "holding_days": holding_days,
        "exit_reason": exit_reason,
        "realized_return_pct": realized_return_pct,
    })


def _build_portfolio_selections_from_reviews(review_snapshots, review_log_df, prepared_ticker_data):
    if review_log_df.empty or not review_snapshots:
        return pd.DataFrame(columns=[
            "rebalance_date", "signal_date", "entry_date", "exit_date", "ticker",
            "position_type", "rank", "score", "entry_price", "exit_price",
            "realized_return_pct", "realized_10d_return_pct", "weight",
            "position_entry_date",
        ])

    review_df = review_log_df.sort_values("review_date").reset_index(drop=True)
    snapshot_df = pd.DataFrame(review_snapshots)
    selections = []

    for idx in range(len(review_df) - 1):
        review_row = review_df.iloc[idx]
        next_review_row = review_df.iloc[idx + 1]
        review_id = int(review_row["review_id"])
        review_date = pd.Timestamp(review_row["review_date"])
        next_review_date = pd.Timestamp(next_review_row["review_date"])
        signal_date = pd.Timestamp(review_row["signal_date"])
        review_positions = snapshot_df[snapshot_df["review_id"] == review_id]
        if review_positions.empty:
            continue

        for _, row in review_positions.iterrows():
            ticker = row["ticker"]
            prepared = prepared_ticker_data.get(ticker)
            entry_price = _lookup_ffill_price(prepared, review_date) if prepared is not None else np.nan
            exit_price = _lookup_ffill_price(prepared, next_review_date) if prepared is not None else np.nan
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0 or exit_price <= 0:
                continue

            realized = (float(exit_price) / float(entry_price) - 1.0) * 100.0
            if row["position_type"] == "SHORT":
                realized = -realized

            selections.append({
                "rebalance_date": signal_date,
                "signal_date": signal_date,
                "entry_date": review_date,
                "exit_date": next_review_date,
                "ticker": ticker,
                "position_type": row["position_type"],
                "rank": row.get("rank", np.nan),
                "score": row.get("score", np.nan),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "realized_return_pct": realized,
                "realized_10d_return_pct": realized,
                "weight": row.get("weight", np.nan),
                "position_entry_date": row.get("position_entry_date", pd.NaT),
            })

    return pd.DataFrame(selections)


def _empty_portfolio_result():
    empty_df = pd.DataFrame(
        columns=[
            "date",
            "daily_ret_bps",
            "gross_daily_ret_bps",
            "net_daily_ret_bps",
            "long_pnl_bps",
            "short_pnl_bps",
            "num_longs",
            "num_shorts",
            "cost_bps_applied",
            "rebalance_signal_date",
            "reviewed_today",
            "executed_today",
            "review_reason",
            "turnover_total",
            "turnover_long",
            "turnover_short",
            "period_length_days",
            "return_granularity",
        ]
    )
    return {
        "pnl_df": empty_df,
        "review_log_df": pd.DataFrame(),
        "trade_log_df": pd.DataFrame(),
        "holdings_log_df": pd.DataFrame(),
        "portfolio_selections_df": pd.DataFrame(),
    }


def _mark_to_market_period_bps(positions, prepared_ticker_data, period_end_date):
    gross_long_pnl_bps = 0.0
    gross_short_pnl_bps = 0.0
    updated_positions = {}

    for ticker, position in positions.items():
        prepared = prepared_ticker_data.get(ticker)
        current_price = _lookup_ffill_price(prepared, period_end_date) if prepared is not None else np.nan
        last_price = position.last_price
        if pd.notna(current_price) and current_price > 0 and last_price and last_price > 0:
            period_ret = current_price / last_price - 1.0
            pnl_component = position.weight * period_ret * 10000
            if position.weight > 0:
                gross_long_pnl_bps += pnl_component
            else:
                gross_short_pnl_bps += pnl_component
            updated_positions[ticker] = PositionState(
                ticker=position.ticker,
                side=position.side,
                weight=position.weight,
                entry_date=position.entry_date,
                entry_signal_date=position.entry_signal_date,
                entry_price=position.entry_price,
                entry_score=position.entry_score,
                last_price=float(current_price),
            )
        else:
            updated_positions[ticker] = position

    return gross_long_pnl_bps, gross_short_pnl_bps, updated_positions


def _filter_target_weights_for_review_eligibility(
    target_weights,
    prepared_ticker_data,
    membership_lookup,
    signal_date,
    trade_date,
):
    """Drop weights that are no longer eligible at this review.

    Candidate construction already applies membership and price filters for new
    entries. This guard also applies them to retained positions and fallback
    weights, so stale/delisted names are forced out instead of being carried.
    """
    eligible_weights = {}
    for ticker, weight in (target_weights or {}).items():
        weight = float(weight)
        if abs(weight) <= 1e-12:
            continue
        if not _is_membership_active(membership_lookup, ticker, signal_date):
            continue
        prepared = prepared_ticker_data.get(ticker)
        price = _lookup_ffill_price(prepared, trade_date) if prepared is not None else np.nan
        if not has_min_trade_price(price):
            continue
        eligible_weights[ticker] = weight
    return eligible_weights


def _build_factor_portfolio_periodic_static(
    all_data,
    spy_df,
    rebalance_dates,
    score_type,
    regime_map,
    config,
    trading_dates,
    candidate_pool_config,
    membership_lookup,
):
    explicit_llm_mode = _is_explicit_llm_backtest(all_data)
    close_only_mode = _is_close_only_dataset(all_data)
    require_exact_candidate_dates = close_only_mode and _looks_like_russell_dataset(all_data)
    explicit_llm_signal_dates = _available_explicit_llm_signal_dates(all_data) if explicit_llm_mode else []

    pnl_records = []
    review_records = []
    trade_records = []
    holdings_records = []
    review_snapshots = []

    print(f"\n📊 Building factor portfolio...")
    print(f"   Book sizing: {config.describe_book_sizing()}")
    print(f"   Gross exposure: 2.0x (long-short market-neutral)")
    print(f"   Cost model: {COST_RATE*10000:.0f} bps one-way")
    print(f"   Min entry price: > ${MIN_TRADE_PRICE:.2f}")
    print(f"   Score type: {score_type}")
    print(f"   Policy: {_summarize_policy_label(config)}")
    print(f"   Candidate pool: {candidate_pool_config.describe()}")
    print(f"   Weighting mode: {config.weighting_mode}")
    print("   Return path: periodic static (review-to-review)")
    print(f"   Price source: {'adj_close (close-only Russell path)' if close_only_mode else 'adj_open (legacy open-execution path)'}")
    print(
        "   Russell membership gating: "
        f"{'enabled' if membership_lookup is not None else 'disabled'}"
    )
    if explicit_llm_mode:
        print("   Execution path: explicit LLM baskets")
        print(f"   Cached LLM decision dates: {len(explicit_llm_signal_dates)}")

    for ticker in all_data:
        all_data[ticker]["date"] = pd.to_datetime(all_data[ticker]["date"])
    spy_df["date"] = pd.to_datetime(spy_df["date"])

    calendar_dates = [pd.Timestamp(d) for d in sorted(pd.to_datetime(spy_df["date"].values).unique())]
    trade_signal_map = _build_trade_signal_map(calendar_dates, close_only=close_only_mode)
    calendar_index = {date: idx for idx, date in enumerate(calendar_dates)}

    if trading_dates is None:
        all_trade_dates = [date for date in calendar_dates if date in trade_signal_map]
    else:
        all_trade_dates = [pd.Timestamp(date) for date in trading_dates if pd.Timestamp(date) in trade_signal_map]
    if not all_trade_dates:
        return _empty_portfolio_result()

    print(f"   Trading dates: {len(all_trade_dates)}")

    dense_price_histories = {}
    preferred_price_column = "adj_close" if close_only_mode else "adj_open"
    if _looks_like_russell_dataset(all_data) and _USE_RUSSELL_DENSE_PRICE_ENRICHMENT:
        try:
            dense_price_histories = _load_dense_price_histories(all_data.keys())
            if dense_price_histories:
                print(f"   Dense Russell price histories loaded: {len(dense_price_histories)} tickers")
        except Exception as exc:
            print(f"   ⚠️  Dense Russell price enrichment unavailable: {exc}")
    elif _looks_like_russell_dataset(all_data):
        print("   Dense Russell price enrichment: disabled (using score-file prices to avoid OOM)")

    prepared_ticker_data = {}
    for ticker, df in all_data.items():
        prepared_ticker_data[ticker] = _prepare_ticker_data(
            df,
            score_type,
            price_df=dense_price_histories.get(ticker),
            preferred_price_column=preferred_price_column,
        )

    review_trade_dates = [
        trade_date
        for trade_date in all_trade_dates
        if pd.Timestamp(trade_signal_map.get(trade_date)) in config.review_policy.scheduled_signal_dates
    ]
    if not review_trade_dates:
        return _empty_portfolio_result()

    positions = {}
    old_weights = {}
    previous_regime = None
    review_id = 0
    previous_anchor_trade_date = None

    for trade_date in review_trade_dates:
        signal_date = trade_signal_map.get(trade_date)
        current_regime = regime_map.get(pd.Timestamp(signal_date), "unknown") if regime_map else "unknown"

        if previous_anchor_trade_date is None:
            gross_long_pnl_bps = 0.0
            gross_short_pnl_bps = 0.0
            period_length_days = 1
        else:
            gross_long_pnl_bps, gross_short_pnl_bps, positions = _mark_to_market_period_bps(
                positions,
                prepared_ticker_data,
                trade_date,
            )
            period_length_days = max(1, calendar_index.get(pd.Timestamp(trade_date), 0) - calendar_index.get(pd.Timestamp(previous_anchor_trade_date), 0))

        review_id += 1
        review_reason = "scheduled_interval"
        num_longs_before_review = sum(1 for position in positions.values() if position.weight > 0)
        num_shorts_before_review = sum(1 for position in positions.values() if position.weight < 0)

        target_weights_raw = None
        target_longs = []
        target_shorts = []
        target_long_n = 0
        target_short_n = 0
        candidate_info = {
            "date_scores": {},
            "long_candidates": [],
            "short_candidates": [],
            "long_rank_map": {},
            "short_rank_map": {},
        }

        if explicit_llm_mode:
            candidate_info = _explicit_llm_snapshot(
                all_data,
                score_type,
                signal_date,
                trade_date,
                prepared_ticker_data,
                available_signal_dates=explicit_llm_signal_dates,
                membership_lookup=membership_lookup,
            )
            target_weights_raw = candidate_info["target_weights"]
            target_longs = [ticker for ticker, weight in target_weights_raw.items() if weight > 0]
            target_shorts = [ticker for ticker, weight in target_weights_raw.items() if weight < 0]
            target_long_n = len(target_longs)
            target_short_n = len(target_shorts)
        else:
            candidate_info = _candidate_snapshot(
                score_type,
                prepared_ticker_data,
                signal_date,
                trade_date,
                require_exact_dates=require_exact_candidate_dates,
                candidate_pool_config=candidate_pool_config,
                membership_lookup=membership_lookup,
            )
            target_long_n, target_short_n = config.target_book_sizes(len(candidate_info["date_scores"]))
            if (
                target_long_n > 0
                and target_short_n > 0
                and len(candidate_info["date_scores"]) >= target_long_n + target_short_n
            ):
                current_longs = [ticker for ticker, position in positions.items() if position.weight > 0]
                current_shorts = [ticker for ticker, position in positions.items() if position.weight < 0]
                target_longs, target_shorts = config.hysteresis_policy.target_book(
                    current_longs=current_longs,
                    current_shorts=current_shorts,
                    candidate_longs=candidate_info["long_candidates"],
                    candidate_shorts=candidate_info["short_candidates"],
                    long_rank_map=candidate_info["long_rank_map"],
                    short_rank_map=candidate_info["short_rank_map"],
                    target_long_n=target_long_n,
                    target_short_n=target_short_n,
                )
                target_weights_raw = build_target_weights(
                    target_longs,
                    target_shorts,
                    weighting_mode=config.weighting_mode,
                )
            else:
                review_reason = f"{review_reason}_insufficient_candidates"

        review_decision = ReviewDecision(
            due=True,
            reason=review_reason,
            review_date=pd.Timestamp(trade_date),
            signal_date=pd.Timestamp(signal_date) if signal_date is not None else None,
            regime=current_regime,
            candidate_longs=tuple(candidate_info["long_candidates"][:target_long_n]),
            candidate_shorts=tuple(candidate_info["short_candidates"][:target_short_n]),
            target_longs=tuple(target_longs),
            target_shorts=tuple(target_shorts),
        )

        if not target_weights_raw:
            target_weights_raw = dict(old_weights)
        target_weights_raw = _filter_target_weights_for_review_eligibility(
            target_weights_raw,
            prepared_ticker_data,
            membership_lookup,
            signal_date,
            trade_date,
        )

        trade_decision = compute_trade_decision(
            old_weights=old_weights,
            target_weights=target_weights_raw,
            cost_rate=COST_RATE,
            min_weight_delta=config.hysteresis_policy.min_weight_delta,
        )
        executed_today = trade_decision.executed
        turnover_total = trade_decision.turnover_total
        turnover_long = trade_decision.turnover_long
        turnover_short = trade_decision.turnover_short
        cost_today = trade_decision.cost_bps
        long_entries = trade_decision.long_entries
        long_exits = trade_decision.long_exits
        short_entries = trade_decision.short_entries
        short_exits = trade_decision.short_exits
        replacements = trade_decision.replacements

        if executed_today:
            new_positions = dict(positions)
            tickers_in_trade = set(old_weights.keys()) | set(trade_decision.target_weights.keys())
            for ticker in tickers_in_trade:
                old_weight = float(old_weights.get(ticker, 0.0))
                new_weight = float(trade_decision.target_weights.get(ticker, 0.0))
                if abs(new_weight - old_weight) <= 1e-12:
                    continue

                old_side = "LONG" if old_weight > 0 else "SHORT" if old_weight < 0 else ""
                new_side = "LONG" if new_weight > 0 else "SHORT" if new_weight < 0 else ""
                old_position = positions.get(ticker)
                prepared = prepared_ticker_data.get(ticker)
                current_price = _lookup_ffill_price(prepared, trade_date) if prepared is not None else np.nan

                if old_position is not None and old_side and old_side != new_side:
                    _close_position_segment(
                        holdings_records=holdings_records,
                        position=old_position,
                        exit_date=trade_date,
                        exit_signal_date=signal_date,
                        exit_price=current_price,
                        exit_reason="flip" if new_side else "review_exit",
                        calendar_index=calendar_index,
                    )
                    new_positions.pop(ticker, None)
                elif old_position is not None and not new_side:
                    _close_position_segment(
                        holdings_records=holdings_records,
                        position=old_position,
                        exit_date=trade_date,
                        exit_signal_date=signal_date,
                        exit_price=current_price,
                        exit_reason="review_exit",
                        calendar_index=calendar_index,
                    )
                    new_positions.pop(ticker, None)

                current_score = candidate_info["date_scores"].get(ticker, {}).get("score", np.nan)
                if pd.isna(current_score) and old_position is not None:
                    current_score = old_position.entry_score

                if new_side:
                    if old_side == new_side and old_position is not None:
                        new_positions[ticker] = PositionState(
                            ticker=ticker,
                            side=new_side,
                            weight=new_weight,
                            entry_date=old_position.entry_date,
                            entry_signal_date=old_position.entry_signal_date,
                            entry_price=old_position.entry_price,
                            entry_score=old_position.entry_score,
                            last_price=float(current_price) if pd.notna(current_price) else old_position.last_price,
                        )
                    else:
                        new_positions[ticker] = PositionState(
                            ticker=ticker,
                            side=new_side,
                            weight=new_weight,
                            entry_date=pd.Timestamp(trade_date),
                            entry_signal_date=pd.Timestamp(signal_date) if signal_date is not None else None,
                            entry_price=float(current_price) if pd.notna(current_price) else np.nan,
                            entry_score=float(current_score) if pd.notna(current_score) else np.nan,
                            last_price=float(current_price) if pd.notna(current_price) else np.nan,
                        )

                if old_side and new_side and old_side != new_side:
                    action = f"FLIP_TO_{new_side}"
                elif not old_side and new_side:
                    action = "ENTER"
                elif old_side and not new_side:
                    action = "EXIT"
                elif abs(new_weight) > abs(old_weight):
                    action = "INCREASE"
                else:
                    action = "DECREASE"

                trade_records.append({
                    "trade_date": trade_date,
                    "signal_date": signal_date,
                    "regime": current_regime,
                    "review_reason": review_reason,
                    "ticker": ticker,
                    "action": action,
                    "from_weight": old_weight,
                    "to_weight": new_weight,
                    "delta_weight": new_weight - old_weight,
                    "from_side": old_side,
                    "to_side": new_side,
                    "old_rank": candidate_info["long_rank_map"].get(ticker) if old_side == "LONG" else candidate_info["short_rank_map"].get(ticker),
                    "new_rank": candidate_info["long_rank_map"].get(ticker) if new_side == "LONG" else candidate_info["short_rank_map"].get(ticker),
                    "score": current_score,
                })

            positions = new_positions
            old_weights = {ticker: position.weight for ticker, position in positions.items()}

        review_records.append({
            "review_id": review_id,
            "review_date": trade_date,
            "signal_date": signal_date,
            "regime": current_regime,
            "review_reason": review_reason,
            "executed": executed_today,
            "num_longs_before": num_longs_before_review,
            "num_shorts_before": num_shorts_before_review,
            "num_longs_after": sum(1 for position in positions.values() if position.weight > 0),
            "num_shorts_after": sum(1 for position in positions.values() if position.weight < 0),
            "candidate_longs": len(review_decision.candidate_longs),
            "candidate_shorts": len(review_decision.candidate_shorts),
            "target_longs": len(review_decision.target_longs),
            "target_shorts": len(review_decision.target_shorts),
            "turnover_total": turnover_total,
            "turnover_long": turnover_long,
            "turnover_short": turnover_short,
            "cost_bps_applied": cost_today,
            "long_entries": long_entries,
            "long_exits": long_exits,
            "short_entries": short_entries,
            "short_exits": short_exits,
            "replacements": replacements,
        })

        for ticker, position in sorted(positions.items(), key=lambda item: (item[1].side, item[0])):
            if position.weight > 0:
                rank = candidate_info["long_rank_map"].get(ticker, np.nan)
                side = "LONG"
            else:
                rank = candidate_info["short_rank_map"].get(ticker, np.nan)
                side = "SHORT"
            score_value = candidate_info["date_scores"].get(ticker, {}).get("score", position.entry_score)
            review_snapshots.append({
                "review_id": review_id,
                "review_date": trade_date,
                "signal_date": signal_date,
                "ticker": ticker,
                "position_type": side,
                "rank": rank,
                "score": score_value,
                "weight": position.weight,
                "position_entry_date": position.entry_date,
            })

        long_cost_bps = COST_RATE * turnover_long * 10000
        short_cost_bps = COST_RATE * turnover_short * 10000
        long_pnl_bps = gross_long_pnl_bps - long_cost_bps
        short_pnl_bps = gross_short_pnl_bps - short_cost_bps
        gross_period_ret_bps = gross_long_pnl_bps + gross_short_pnl_bps
        net_period_ret_bps = long_pnl_bps + short_pnl_bps

        pnl_records.append({
            "date": trade_date,
            "daily_ret_bps": net_period_ret_bps,
            "gross_daily_ret_bps": gross_period_ret_bps,
            "net_daily_ret_bps": net_period_ret_bps,
            "long_pnl_bps": long_pnl_bps,
            "short_pnl_bps": short_pnl_bps,
            "num_longs": sum(1 for position in positions.values() if position.weight > 0),
            "num_shorts": sum(1 for position in positions.values() if position.weight < 0),
            "cost_bps_applied": cost_today,
            "rebalance_signal_date": signal_date,
            "reviewed_today": True,
            "executed_today": executed_today,
            "review_reason": review_reason,
            "turnover_total": turnover_total,
            "turnover_long": turnover_long,
            "turnover_short": turnover_short,
            "period_length_days": period_length_days,
            "return_granularity": "periodic_static",
        })

        previous_regime = current_regime
        previous_anchor_trade_date = trade_date

    final_trade_date = all_trade_dates[-1]
    final_signal_date = trade_signal_map.get(final_trade_date)
    if previous_anchor_trade_date is not None and final_trade_date > previous_anchor_trade_date and positions:
        gross_long_pnl_bps, gross_short_pnl_bps, positions = _mark_to_market_period_bps(
            positions,
            prepared_ticker_data,
            final_trade_date,
        )
        period_length_days = max(1, calendar_index.get(pd.Timestamp(final_trade_date), 0) - calendar_index.get(pd.Timestamp(previous_anchor_trade_date), 0))
        pnl_records.append({
            "date": final_trade_date,
            "daily_ret_bps": gross_long_pnl_bps + gross_short_pnl_bps,
            "gross_daily_ret_bps": gross_long_pnl_bps + gross_short_pnl_bps,
            "net_daily_ret_bps": gross_long_pnl_bps + gross_short_pnl_bps,
            "long_pnl_bps": gross_long_pnl_bps,
            "short_pnl_bps": gross_short_pnl_bps,
            "num_longs": sum(1 for position in positions.values() if position.weight > 0),
            "num_shorts": sum(1 for position in positions.values() if position.weight < 0),
            "cost_bps_applied": 0.0,
            "rebalance_signal_date": pd.NaT,
            "reviewed_today": False,
            "executed_today": False,
            "review_reason": "",
            "turnover_total": 0.0,
            "turnover_long": 0.0,
            "turnover_short": 0.0,
            "period_length_days": period_length_days,
            "return_granularity": "periodic_static",
        })

    for ticker, position in list(positions.items()):
        prepared = prepared_ticker_data.get(ticker)
        exit_price = _lookup_ffill_price(prepared, final_trade_date) if prepared is not None else np.nan
        _close_position_segment(
            holdings_records=holdings_records,
            position=position,
            exit_date=final_trade_date,
            exit_signal_date=final_signal_date,
            exit_price=exit_price,
            exit_reason="end_of_backtest",
            calendar_index=calendar_index,
        )

    review_log_df = pd.DataFrame(review_records)
    portfolio_selections_df = _build_portfolio_selections_from_reviews(
        review_snapshots,
        review_log_df,
        prepared_ticker_data,
    )
    return {
        "pnl_df": pd.DataFrame(pnl_records),
        "review_log_df": review_log_df,
        "trade_log_df": pd.DataFrame(trade_records),
        "holdings_log_df": pd.DataFrame(holdings_records),
        "portfolio_selections_df": portfolio_selections_df,
    }

def build_factor_portfolio(
    all_data,
    spy_df,
    rebalance_dates,
    score_type="direction_score",
    regime_map=None,
    config=None,
    trading_dates=None,
    candidate_pool_config: CandidatePoolConfig | None = None,
    membership_lookup: RussellMembershipLookup | None = None,
    periodic_static: bool = False,
):
    """Build portfolio returns with review timing separated from trade execution."""

    explicit_llm_mode = _is_explicit_llm_backtest(all_data)
    close_only_mode = _is_close_only_dataset(all_data)
    candidate_pool_config = candidate_pool_config or CandidatePoolConfig()
    require_exact_candidate_dates = close_only_mode and _looks_like_russell_dataset(all_data)
    explicit_llm_signal_dates = _available_explicit_llm_signal_dates(all_data) if explicit_llm_mode else []
    if config is None:
        config = PortfolioConstructionConfig(
            review_policy=ReviewPolicy(
                mode="static",
                static_review_days=REBALANCE_DAYS,
                regime_intervals=DEFAULT_REGIME_REVIEW_INTERVALS.copy(),
                scheduled_signal_dates=frozenset(pd.to_datetime(rebalance_dates)),
            ),
            hysteresis_policy=HysteresisPolicy(
                mode="none",
                top_n=TOP_N,
                bottom_n=BOTTOM_N,
            ),
            top_n=TOP_N,
            bottom_n=BOTTOM_N,
            weighting_mode="equal",
            book_sizing_mode="fixed",
        )
    elif explicit_llm_mode:
        # Explicit LLM backtests should review on the cached LLM decision dates
        # rather than on the deterministic review cadence.
        config = PortfolioConstructionConfig(
            review_policy=ReviewPolicy(
                mode="static",
                static_review_days=config.review_policy.static_review_days,
                regime_intervals=config.review_policy.regime_intervals,
                early_review_regimes=config.review_policy.early_review_regimes,
                scheduled_signal_dates=frozenset(explicit_llm_signal_dates),
            ),
            hysteresis_policy=config.hysteresis_policy,
            top_n=config.top_n,
            bottom_n=config.bottom_n,
            weighting_mode=config.weighting_mode,
            book_sizing_mode=config.book_sizing_mode,
        )

    if periodic_static:
        if config.review_policy.mode != "static":
            raise ValueError("--periodic-static currently supports review_mode=static only")
        return _build_factor_portfolio_periodic_static(
            all_data,
            spy_df,
            rebalance_dates,
            score_type,
            regime_map,
            config,
            trading_dates,
            candidate_pool_config,
            membership_lookup,
        )

    pnl_records = []
    review_records = []
    trade_records = []
    holdings_records = []
    review_snapshots = []

    print(f"\n📊 Building factor portfolio...")
    print(f"   Book sizing: {config.describe_book_sizing()}")
    print(f"   Gross exposure: 2.0x (long-short market-neutral)")
    print(f"   Cost model: {COST_RATE*10000:.0f} bps one-way")
    print(f"   Min entry price: > ${MIN_TRADE_PRICE:.2f}")
    print(f"   Score type: {score_type}")
    print(f"   Policy: {_summarize_policy_label(config)}")
    print(f"   Candidate pool: {candidate_pool_config.describe()}")
    print(f"   Weighting mode: {config.weighting_mode}")
    print(f"   Price source: {'adj_close (close-only Russell path)' if close_only_mode else 'adj_open (legacy open-execution path)'}")
    print(
        "   Russell membership gating: "
        f"{'enabled' if membership_lookup is not None else 'disabled'}"
    )
    if explicit_llm_mode:
        print("   Execution path: explicit LLM baskets")
        print(f"   Cached LLM decision dates: {len(explicit_llm_signal_dates)}")

    for ticker in all_data:
        all_data[ticker]["date"] = pd.to_datetime(all_data[ticker]["date"])
    spy_df["date"] = pd.to_datetime(spy_df["date"])

    calendar_dates = [pd.Timestamp(d) for d in sorted(pd.to_datetime(spy_df["date"].values).unique())]
    trade_signal_map = _build_trade_signal_map(calendar_dates, close_only=close_only_mode)
    calendar_index = {date: idx for idx, date in enumerate(calendar_dates)}

    if trading_dates is None:
        all_trade_dates = [date for date in calendar_dates if date in trade_signal_map]
    else:
        all_trade_dates = [pd.Timestamp(date) for date in trading_dates if pd.Timestamp(date) in trade_signal_map]
    if not all_trade_dates:
        return _empty_portfolio_result()

    print(f"   Trading dates: {len(all_trade_dates)}")
    if close_only_mode:
        print("   Execution model: close_to_close_hold (signal close -> same close reference)")
        print("   Preparing compact close lookup arrays...")
    else:
        print(f"   Execution model: {CANONICAL_EXECUTION_MODEL} (signal close -> next open)")
        print("   Preparing compact open lookup arrays...")

    dense_price_histories = {}
    preferred_price_column = "adj_close" if close_only_mode else "adj_open"
    if _looks_like_russell_dataset(all_data) and _USE_RUSSELL_DENSE_PRICE_ENRICHMENT:
        try:
            dense_price_histories = _load_dense_price_histories(all_data.keys())
            if dense_price_histories:
                print(f"   Dense Russell price histories loaded: {len(dense_price_histories)} tickers")
        except Exception as exc:
            print(f"   ⚠️  Dense Russell price enrichment unavailable: {exc}")
    elif _looks_like_russell_dataset(all_data):
        print("   Dense Russell price enrichment: disabled (using score-file prices to avoid OOM)")

    prepared_ticker_data = {}
    for ticker, df in all_data.items():
        prepared_ticker_data[ticker] = _prepare_ticker_data(
            df,
            score_type,
            price_df=dense_price_histories.get(ticker),
            preferred_price_column=preferred_price_column,
        )

    positions = {}
    old_weights = {}
    last_review_loop_index = None
    previous_regime = None
    review_id = 0

    for loop_idx, trade_date in enumerate(all_trade_dates):
        signal_date = trade_signal_map.get(trade_date)
        current_regime = regime_map.get(pd.Timestamp(signal_date), "unknown") if regime_map else "unknown"

        gross_long_pnl_bps = 0.0
        gross_short_pnl_bps = 0.0

        for ticker, position in list(positions.items()):
            prepared = prepared_ticker_data.get(ticker)
            current_price = _lookup_ffill_price(prepared, trade_date) if prepared is not None else np.nan
            last_price = position.last_price
            if pd.notna(current_price) and current_price > 0 and last_price and last_price > 0:
                daily_ret = current_price / last_price - 1.0
                pnl_component = position.weight * daily_ret * 10000
                if position.weight > 0:
                    gross_long_pnl_bps += pnl_component
                else:
                    gross_short_pnl_bps += pnl_component
                positions[ticker] = PositionState(
                    ticker=position.ticker,
                    side=position.side,
                    weight=position.weight,
                    entry_date=position.entry_date,
                    entry_signal_date=position.entry_signal_date,
                    entry_price=position.entry_price,
                    entry_score=position.entry_score,
                    last_price=float(current_price),
                )

        reviewed_today = False
        executed_today = False
        review_reason = ""
        turnover_total = 0.0
        turnover_long = 0.0
        turnover_short = 0.0
        cost_today = 0.0
        long_entries = long_exits = short_entries = short_exits = replacements = 0
        review_decision = ReviewDecision(
            due=False,
            reason="",
            review_date=None,
            signal_date=signal_date,
            regime=current_regime,
        )
        candidate_info = {
            "date_scores": {},
            "long_candidates": [],
            "short_candidates": [],
            "long_rank_map": {},
            "short_rank_map": {},
        }

        days_since_last_review = None if last_review_loop_index is None else loop_idx - last_review_loop_index
        review_due, review_reason = config.review_policy.should_review(
            signal_date=signal_date,
            current_regime=current_regime,
            previous_regime=previous_regime,
            days_since_last_review=days_since_last_review,
        )

        if review_due:
            reviewed_today = True
            review_id += 1
            review_reason = review_reason or "scheduled_interval"
            num_longs_before_review = sum(1 for position in positions.values() if position.weight > 0)
            num_shorts_before_review = sum(1 for position in positions.values() if position.weight < 0)

            target_weights_raw = None
            target_longs = []
            target_shorts = []
            target_long_n = 0
            target_short_n = 0

            if explicit_llm_mode:
                candidate_info = _explicit_llm_snapshot(
                    all_data,
                    score_type,
                    signal_date,
                    trade_date,
                    prepared_ticker_data,
                    available_signal_dates=explicit_llm_signal_dates,
                    membership_lookup=membership_lookup,
                )
                target_weights_raw = candidate_info["target_weights"]
                target_longs = [ticker for ticker, weight in target_weights_raw.items() if weight > 0]
                target_shorts = [ticker for ticker, weight in target_weights_raw.items() if weight < 0]
                target_long_n = len(target_longs)
                target_short_n = len(target_shorts)
            else:
                candidate_info = _candidate_snapshot(
                    score_type,
                    prepared_ticker_data,
                    signal_date,
                    trade_date,
                    require_exact_dates=require_exact_candidate_dates,
                    candidate_pool_config=candidate_pool_config,
                    membership_lookup=membership_lookup,
                )
                target_long_n, target_short_n = config.target_book_sizes(len(candidate_info["date_scores"]))
                if (
                    target_long_n > 0
                    and target_short_n > 0
                    and len(candidate_info["date_scores"]) >= target_long_n + target_short_n
                ):
                    current_longs = [ticker for ticker, position in positions.items() if position.weight > 0]
                    current_shorts = [ticker for ticker, position in positions.items() if position.weight < 0]
                    target_longs, target_shorts = config.hysteresis_policy.target_book(
                        current_longs=current_longs,
                        current_shorts=current_shorts,
                        candidate_longs=candidate_info["long_candidates"],
                        candidate_shorts=candidate_info["short_candidates"],
                        long_rank_map=candidate_info["long_rank_map"],
                        short_rank_map=candidate_info["short_rank_map"],
                        target_long_n=target_long_n,
                        target_short_n=target_short_n,
                    )
                    target_weights_raw = build_target_weights(
                        target_longs,
                        target_shorts,
                        weighting_mode=config.weighting_mode,
                    )
                else:
                    review_reason = f"{review_reason}_insufficient_candidates"

            review_decision = ReviewDecision(
                due=True,
                reason=review_reason,
                review_date=pd.Timestamp(trade_date),
                signal_date=pd.Timestamp(signal_date) if signal_date is not None else None,
                regime=current_regime,
                candidate_longs=tuple(candidate_info["long_candidates"][:target_long_n]),
                candidate_shorts=tuple(candidate_info["short_candidates"][:target_short_n]),
                target_longs=tuple(target_longs),
                target_shorts=tuple(target_shorts),
            )

            if not target_weights_raw:
                target_weights_raw = dict(old_weights)
            target_weights_raw = _filter_target_weights_for_review_eligibility(
                target_weights_raw,
                prepared_ticker_data,
                membership_lookup,
                signal_date,
                trade_date,
            )

            trade_decision = compute_trade_decision(
                old_weights=old_weights,
                target_weights=target_weights_raw,
                cost_rate=COST_RATE,
                min_weight_delta=config.hysteresis_policy.min_weight_delta,
            )
            executed_today = trade_decision.executed
            turnover_total = trade_decision.turnover_total
            turnover_long = trade_decision.turnover_long
            turnover_short = trade_decision.turnover_short
            cost_today = trade_decision.cost_bps
            long_entries = trade_decision.long_entries
            long_exits = trade_decision.long_exits
            short_entries = trade_decision.short_entries
            short_exits = trade_decision.short_exits
            replacements = trade_decision.replacements

            if executed_today:
                new_positions = dict(positions)
                tickers_in_trade = set(old_weights.keys()) | set(trade_decision.target_weights.keys())
                for ticker in tickers_in_trade:
                    old_weight = float(old_weights.get(ticker, 0.0))
                    new_weight = float(trade_decision.target_weights.get(ticker, 0.0))
                    if abs(new_weight - old_weight) <= 1e-12:
                        continue

                    old_side = "LONG" if old_weight > 0 else "SHORT" if old_weight < 0 else ""
                    new_side = "LONG" if new_weight > 0 else "SHORT" if new_weight < 0 else ""
                    old_position = positions.get(ticker)
                    prepared = prepared_ticker_data.get(ticker)
                    current_price = _lookup_ffill_price(prepared, trade_date) if prepared is not None else np.nan

                    if old_position is not None and old_side and old_side != new_side:
                        _close_position_segment(
                            holdings_records=holdings_records,
                            position=old_position,
                            exit_date=trade_date,
                            exit_signal_date=signal_date,
                            exit_price=current_price,
                            exit_reason="flip" if new_side else "review_exit",
                            calendar_index=calendar_index,
                        )
                        new_positions.pop(ticker, None)
                    elif old_position is not None and not new_side:
                        _close_position_segment(
                            holdings_records=holdings_records,
                            position=old_position,
                            exit_date=trade_date,
                            exit_signal_date=signal_date,
                            exit_price=current_price,
                            exit_reason="review_exit",
                            calendar_index=calendar_index,
                        )
                        new_positions.pop(ticker, None)

                    current_score = candidate_info["date_scores"].get(ticker, {}).get("score", np.nan)
                    if pd.isna(current_score) and old_position is not None:
                        current_score = old_position.entry_score

                    if new_side:
                        if old_side == new_side and old_position is not None:
                            new_positions[ticker] = PositionState(
                                ticker=ticker,
                                side=new_side,
                                weight=new_weight,
                                entry_date=old_position.entry_date,
                                entry_signal_date=old_position.entry_signal_date,
                                entry_price=old_position.entry_price,
                                entry_score=old_position.entry_score,
                                last_price=float(current_price) if pd.notna(current_price) else old_position.last_price,
                            )
                        else:
                            new_positions[ticker] = PositionState(
                                ticker=ticker,
                                side=new_side,
                                weight=new_weight,
                                entry_date=pd.Timestamp(trade_date),
                                entry_signal_date=pd.Timestamp(signal_date) if signal_date is not None else None,
                                entry_price=float(current_price) if pd.notna(current_price) else np.nan,
                                entry_score=float(current_score) if pd.notna(current_score) else np.nan,
                                last_price=float(current_price) if pd.notna(current_price) else np.nan,
                            )

                    if old_side and new_side and old_side != new_side:
                        action = f"FLIP_TO_{new_side}"
                    elif not old_side and new_side:
                        action = "ENTER"
                    elif old_side and not new_side:
                        action = "EXIT"
                    elif abs(new_weight) > abs(old_weight):
                        action = "INCREASE"
                    else:
                        action = "DECREASE"

                    trade_records.append({
                        "trade_date": trade_date,
                        "signal_date": signal_date,
                        "regime": current_regime,
                        "review_reason": review_reason,
                        "ticker": ticker,
                        "action": action,
                        "from_weight": old_weight,
                        "to_weight": new_weight,
                        "delta_weight": new_weight - old_weight,
                        "from_side": old_side,
                        "to_side": new_side,
                        "old_rank": candidate_info["long_rank_map"].get(ticker) if old_side == "LONG" else candidate_info["short_rank_map"].get(ticker),
                        "new_rank": candidate_info["long_rank_map"].get(ticker) if new_side == "LONG" else candidate_info["short_rank_map"].get(ticker),
                        "score": current_score,
                    })

                positions = new_positions
                old_weights = {ticker: position.weight for ticker, position in positions.items()}

            last_review_loop_index = loop_idx
            review_records.append({
                "review_id": review_id,
                "review_date": trade_date,
                "signal_date": signal_date,
                "regime": current_regime,
                "review_reason": review_reason,
                "executed": executed_today,
                "num_longs_before": num_longs_before_review,
                "num_shorts_before": num_shorts_before_review,
                "num_longs_after": sum(1 for position in positions.values() if position.weight > 0),
                "num_shorts_after": sum(1 for position in positions.values() if position.weight < 0),
                "candidate_longs": len(review_decision.candidate_longs),
                "candidate_shorts": len(review_decision.candidate_shorts),
                "target_longs": len(review_decision.target_longs),
                "target_shorts": len(review_decision.target_shorts),
                "turnover_total": turnover_total,
                "turnover_long": turnover_long,
                "turnover_short": turnover_short,
                "cost_bps_applied": cost_today,
                "long_entries": long_entries,
                "long_exits": long_exits,
                "short_entries": short_entries,
                "short_exits": short_exits,
                "replacements": replacements,
            })

            for ticker, position in sorted(positions.items(), key=lambda item: (item[1].side, item[0])):
                if position.weight > 0:
                    rank = candidate_info["long_rank_map"].get(ticker, np.nan)
                    side = "LONG"
                else:
                    rank = candidate_info["short_rank_map"].get(ticker, np.nan)
                    side = "SHORT"
                score_value = candidate_info["date_scores"].get(ticker, {}).get("score", position.entry_score)
                review_snapshots.append({
                    "review_id": review_id,
                    "review_date": trade_date,
                    "signal_date": signal_date,
                    "ticker": ticker,
                    "position_type": side,
                    "rank": rank,
                    "score": score_value,
                    "weight": position.weight,
                    "position_entry_date": position.entry_date,
                })

        long_cost_bps = COST_RATE * turnover_long * 10000
        short_cost_bps = COST_RATE * turnover_short * 10000
        long_pnl_bps = gross_long_pnl_bps - long_cost_bps
        short_pnl_bps = gross_short_pnl_bps - short_cost_bps
        gross_daily_ret_bps = gross_long_pnl_bps + gross_short_pnl_bps
        net_daily_ret_bps = long_pnl_bps + short_pnl_bps

        num_longs = sum(1 for position in positions.values() if position.weight > 0)
        num_shorts = sum(1 for position in positions.values() if position.weight < 0)
        pnl_records.append({
            "date": trade_date,
            "daily_ret_bps": net_daily_ret_bps,
            "gross_daily_ret_bps": gross_daily_ret_bps,
            "net_daily_ret_bps": net_daily_ret_bps,
            "long_pnl_bps": long_pnl_bps,
            "short_pnl_bps": short_pnl_bps,
            "num_longs": num_longs,
            "num_shorts": num_shorts,
            "cost_bps_applied": cost_today,
            "rebalance_signal_date": signal_date if reviewed_today else pd.NaT,
            "reviewed_today": reviewed_today,
            "executed_today": executed_today,
            "review_reason": review_reason if reviewed_today else "",
            "turnover_total": turnover_total,
            "turnover_long": turnover_long,
            "turnover_short": turnover_short,
            "period_length_days": 1,
            "return_granularity": "daily",
        })
        previous_regime = current_regime

    final_trade_date = all_trade_dates[-1]
    final_signal_date = trade_signal_map.get(final_trade_date)
    for ticker, position in list(positions.items()):
        prepared = prepared_ticker_data.get(ticker)
        exit_price = _lookup_ffill_price(prepared, final_trade_date) if prepared is not None else np.nan
        _close_position_segment(
            holdings_records=holdings_records,
            position=position,
            exit_date=final_trade_date,
            exit_signal_date=final_signal_date,
            exit_price=exit_price,
            exit_reason="end_of_backtest",
            calendar_index=calendar_index,
        )

    pnl_df = pd.DataFrame(pnl_records)
    review_log_df = pd.DataFrame(review_records)
    trade_log_df = pd.DataFrame(trade_records)
    holdings_log_df = pd.DataFrame(holdings_records)
    portfolio_selections_df = _build_portfolio_selections_from_reviews(
        review_snapshots,
        review_log_df,
        prepared_ticker_data,
    )
    if not pnl_df.empty and "gross_daily_ret_bps" in pnl_df.columns:
        gross_total = pd.to_numeric(pnl_df["gross_daily_ret_bps"], errors="coerce").fillna(0.0).abs().sum()
        live_book_days = (
            pd.to_numeric(pnl_df.get("num_longs", 0), errors="coerce").fillna(0.0)
            + pd.to_numeric(pnl_df.get("num_shorts", 0), errors="coerce").fillna(0.0)
        ) > 0
        if live_book_days.any() and np.isclose(gross_total, 0.0):
            raise RuntimeError(
                "Degenerate backtest output: positions were held but gross_daily_ret_bps stayed identically zero. "
                "This usually indicates stale score files or a broken price lookup path."
            )
    return {
        "pnl_df": pnl_df,
        "review_log_df": review_log_df,
        "trade_log_df": trade_log_df,
        "holdings_log_df": holdings_log_df,
        "portfolio_selections_df": portfolio_selections_df,
    }


# ----------------------------
# Metrics Calculation
# ----------------------------

def _compute_activity_metrics(pnl_df, review_log_df, trade_log_df, holdings_log_df):
    overall = {}

    gross_daily = pnl_df["gross_daily_ret_bps"] / 10000 if "gross_daily_ret_bps" in pnl_df.columns else pnl_df["daily_ret_bps"] / 10000
    net_daily = pnl_df["daily_ret_bps"] / 10000
    gross_total_ret = ((1 + gross_daily).prod() - 1) * 100 if len(pnl_df) > 0 else 0.0
    net_total_ret = ((1 + net_daily).prod() - 1) * 100 if len(pnl_df) > 0 else 0.0
    total_cost_bps = float(pnl_df["cost_bps_applied"].sum()) if "cost_bps_applied" in pnl_df.columns else 0.0

    overall.update({
        "gross_total_ret": gross_total_ret,
        "net_total_ret": net_total_ret,
        "total_cost_bps": total_cost_bps,
        "cost_drag_ret_pct": gross_total_ret - net_total_ret,
        "cost_drag_pct_of_gross": ((gross_total_ret - net_total_ret) / abs(gross_total_ret) * 100) if abs(gross_total_ret) > 1e-12 else np.nan,
    })

    if review_log_df is not None and not review_log_df.empty:
        executed_reviews = review_log_df[review_log_df["executed"] == True]
        overall.update({
            "n_reviews": int(len(review_log_df)),
            "n_trade_reviews": int(len(executed_reviews)),
            "review_to_trade_rate": float(len(executed_reviews) / len(review_log_df) * 100),
            "turnover_mean": float(review_log_df["turnover_total"].mean()),
            "turnover_long_mean": float(review_log_df["turnover_long"].mean()),
            "turnover_short_mean": float(review_log_df["turnover_short"].mean()),
            "total_replacements": int(review_log_df["replacements"].sum()),
            "replacement_rate": float(review_log_df["replacements"].sum() / max(len(executed_reviews), 1)),
            "total_entries": int(review_log_df["long_entries"].sum() + review_log_df["short_entries"].sum()),
            "total_exits": int(review_log_df["long_exits"].sum() + review_log_df["short_exits"].sum()),
        })
    else:
        overall.update({
            "n_reviews": 0,
            "n_trade_reviews": 0,
            "review_to_trade_rate": 0.0,
            "turnover_mean": np.nan,
            "turnover_long_mean": np.nan,
            "turnover_short_mean": np.nan,
            "total_replacements": 0,
            "replacement_rate": np.nan,
            "total_entries": 0,
            "total_exits": 0,
        })

    if trade_log_df is not None and not trade_log_df.empty:
        overall["n_trade_lines"] = int(len(trade_log_df))

    if holdings_log_df is not None and not holdings_log_df.empty:
        overall["avg_holding_days"] = float(holdings_log_df["holding_days"].mean())
        long_holds = holdings_log_df[holdings_log_df["position_type"] == "LONG"]["holding_days"]
        short_holds = holdings_log_df[holdings_log_df["position_type"] == "SHORT"]["holding_days"]
        overall["avg_holding_days_long"] = float(long_holds.mean()) if len(long_holds) > 0 else np.nan
        overall["avg_holding_days_short"] = float(short_holds.mean()) if len(short_holds) > 0 else np.nan
    else:
        overall["avg_holding_days"] = np.nan
        overall["avg_holding_days_long"] = np.nan
        overall["avg_holding_days_short"] = np.nan

    regime_activity = {}
    if review_log_df is not None and not review_log_df.empty:
        grouped = review_log_df.groupby("regime")
        for regime, group in grouped:
            executed = group[group["executed"] == True]
            regime_activity[regime] = {
                "reviews": int(len(group)),
                "trade_reviews": int(len(executed)),
                "review_to_trade_rate": float(len(executed) / len(group) * 100),
                "turnover_mean": float(group["turnover_total"].mean()),
                "cost_drag_bps": float(group["cost_bps_applied"].sum()),
            }

    return overall, regime_activity


def compute_metrics(
    pnl_df,
    regime_map,
    all_data,
    review_signal_dates,
    score_type="direction_score",
    review_log_df=None,
    trade_log_df=None,
    holdings_log_df=None,
    membership_lookup: RussellMembershipLookup | None = None,
):
    """Compute performance metrics by regime + cross-sectional IC
    
    Fixes:
    - Normalize date types (pd.Timestamp)
    - Handle missing regimes gracefully
    - Correct cumulative return calculation
    - Compute cross-sectional IC using the same forward labels as the signal dump
    """

    def _coerce_period_length_series(frame: pd.DataFrame) -> pd.Series:
        raw = frame.get("period_length_days")
        if raw is None:
            return pd.Series(1.0, index=frame.index, dtype=float)
        return pd.to_numeric(raw, errors="coerce").fillna(1.0).clip(lower=1.0)
    
    # Normalize dates to pd.Timestamp
    pnl_df = pnl_df.copy()
    pnl_df["date"] = pd.to_datetime(pnl_df["date"])
    period_lengths = _coerce_period_length_series(pnl_df)
    period_returns = pnl_df["daily_ret_bps"] / 10000
    dailyized_returns = pd.Series(
        np.where(
            (1.0 + period_returns.to_numpy(dtype=float)) > 0,
            np.power(1.0 + period_returns.to_numpy(dtype=float), 1.0 / period_lengths.to_numpy(dtype=float)) - 1.0,
            period_returns.to_numpy(dtype=float) / period_lengths.to_numpy(dtype=float),
        ),
        index=pnl_df.index,
        dtype=float,
    )
    observation_factor = 252.0 / float(period_lengths.mean()) if len(period_lengths) > 0 else 252.0
    
    # Map regimes with type safety
    pnl_df["regime"] = pnl_df["date"].apply(
        lambda d: regime_map.get(pd.Timestamp(d), None) if isinstance(d, (pd.Timestamp, np.datetime64, str)) else None
    )
    
    # Compute cumulative return properly
    pnl_df["daily_ret"] = period_returns
    pnl_df["dailyized_ret"] = dailyized_returns
    pnl_df["cumulative_ret"] = (1 + pnl_df["daily_ret"]).cumprod()
    pnl_df["cumulative_ret_pct"] = (pnl_df["cumulative_ret"] - 1) * 100
    
    # Overall metrics
    total_ret = (pnl_df["cumulative_ret"].iloc[-1] - 1) * 100
    avg_daily_ret = float(dailyized_returns.mean() * 10000)
    period_vol = pnl_df["daily_ret_bps"].std()
    sharpe = (pnl_df["daily_ret"].mean() / pnl_df["daily_ret"].std()) * np.sqrt(observation_factor) if pnl_df["daily_ret"].std() > 0 else 0
    
    cumret = pnl_df["cumulative_ret"].values
    running_max = np.maximum.accumulate(cumret)
    dd = (cumret - running_max) / running_max
    maxdd = np.min(dd) * 100
    
    # Compute cross-sectional IC on signal dates using the canonical forward labels.
    ics = []
    for review_date in review_signal_dates:
        snapshot = _cross_sectional_snapshot(
            all_data,
            review_date,
            score_type,
            require_exact_dates=True,
            selected_only=False,
            membership_lookup=membership_lookup,
        )
        if len(snapshot) >= 10:
            ic, pval = spearmanr(snapshot["score"], snapshot["ret"])
            ics.append(ic)
    
    ic_mean = np.mean(ics) if len(ics) > 0 else 0.0
    ic_std = np.std(ics) if len(ics) > 1 else 0.0
    ic_tstat = (ic_mean / (ic_std + 1e-10)) * np.sqrt(len(ics)) if ic_std > 0 else 0.0
    ic_pval = 2 * (1 - scipy.stats.t.cdf(abs(ic_tstat), len(ics)-1)) if len(ics) > 1 else 1.0
    
    activity_overall, regime_activity = _compute_activity_metrics(
        pnl_df=pnl_df,
        review_log_df=review_log_df,
        trade_log_df=trade_log_df,
        holdings_log_df=holdings_log_df,
    )

    metrics = {
        "overall": {
            "total_ret": total_ret,
            "avg_daily_ret": avg_daily_ret,
            "sharpe": sharpe,
            "maxdd": maxdd,
            "win_rate": (pnl_df["daily_ret_bps"] > 0).sum() / len(pnl_df) * 100,
            "n_trades": len(pnl_df),
            "observation_factor": observation_factor,
            "total_period_days": int(period_lengths.sum()),
            "ic_mean": ic_mean,
            "ic_tstat": ic_tstat,
            "ic_pval": ic_pval,
            "n_ic_periods": len(ics),
            **activity_overall,
        }
    }
    
    # Regime-specific metrics
    for regime in pnl_df["regime"].dropna().unique():
        regime_data = pnl_df[pnl_df["regime"] == regime].copy()
        if len(regime_data) > 10:
            # Compound only the regime's own daily returns (non-contiguous is fine)
            regime_ret = ((1 + regime_data["daily_ret"]).prod() - 1) * 100
            regime_lengths = _coerce_period_length_series(regime_data)
            regime_factor = 252.0 / float(regime_lengths.mean()) if len(regime_lengths) > 0 else 252.0
            regime_vol = regime_data["daily_ret"].std()
            regime_sharpe = (regime_data["daily_ret"].mean() / regime_vol) * np.sqrt(regime_factor) if regime_vol > 0 else 0
            
            metrics[regime] = {
                "total_ret": regime_ret,
                "sharpe": regime_sharpe,
                "n_days": int(regime_lengths.sum()),
            }
            if regime in regime_activity:
                metrics[regime].update(regime_activity[regime])
    
    return pnl_df, metrics


def compute_decile_returns(
    all_data,
    rebalance_dates,
    score_type="direction_score",
    membership_lookup: RussellMembershipLookup | None = None,
):
    """Analyze forward {FORWARD_DAYS}d returns by score decile.

    Handles:
    - Missing deciles from qcut(duplicates='drop')
    - Gracefully skips periods with insufficient data
    - Robust decile spread calculation
    - Uses absolute returns for magnitude_score decile analysis
    """
    
    decile_results = []
    
    for rebalance_date in rebalance_dates:
        snapshot = _cross_sectional_snapshot(
            all_data,
            rebalance_date,
            score_type,
            require_exact_dates=True,
            selected_only=False,
            membership_lookup=membership_lookup,
        )
        if len(snapshot) < 10:
            continue

        decile_df = snapshot.copy()
        decile_df["decile"] = pd.qcut(decile_df["score"], q=10, labels=False, duplicates='drop')
        
        # Calculate decile statistics
        for decile in sorted(decile_df["decile"].unique()):
            decile_data = decile_df[decile_df["decile"] == decile]
            if len(decile_data) > 0:
                decile_results.append({
                    "date": rebalance_date,
                    "decile": int(decile) + 1,
                    "mean_ret": decile_data["ret"].mean() * 100,
                    "std_ret": decile_data["ret"].std() * 100,
                    "n_stocks": len(decile_data)
                })
    
    return pd.DataFrame(decile_results)


# ----------------------------
# Visualization
# ----------------------------

def plot_performance(pnl_df, spy_df=None):
    """Plot factor performance: total vs SPY, long vs short breakdown, daily dist"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # [0] Factor vs SPY (single clean lines, no regime colors)
    ax = axes[0]
    ax.plot(pnl_df["date"], pnl_df["cumulative_ret_pct"], label="Factor", linewidth=2.5, color="steelblue")
    
    if spy_df is not None:
        spy_df_copy = spy_df.copy()
        spy_df_copy["date"] = pd.to_datetime(spy_df_copy["date"])
        spy_df_copy = spy_df_copy.sort_values("date")
        spy_df_copy["spy_ret"] = (spy_df_copy["adj_close"] / spy_df_copy["adj_close"].iloc[0] - 1) * 100
        factor_min_date = pnl_df["date"].min()
        factor_max_date = pnl_df["date"].max()
        spy_df_copy = spy_df_copy[(spy_df_copy["date"] >= factor_min_date) & (spy_df_copy["date"] <= factor_max_date)]
        ax.plot(spy_df_copy["date"], spy_df_copy["spy_ret"], label="S&P 500", linewidth=2.5, color="darkred", linestyle="--")
    
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)
    ax.fill_between(pnl_df["date"], 0, pnl_df["cumulative_ret_pct"], alpha=0.15, color="steelblue")
    ax.set_title("Factor vs S&P 500 Cumulative Return", fontsize=12, fontweight="bold")
    ax.set_ylabel("Return (%)", fontsize=11)
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)
    
    # [1] Rolling 6-month return (momentum of factor)
    ax = axes[1]
    
    rolling_window = 126  # ~6 months
    rolling_ret = pnl_df["daily_ret_bps"].rolling(rolling_window).mean()
    ax.plot(pnl_df["date"], rolling_ret, linewidth=2, color="steelblue", alpha=0.9)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.fill_between(pnl_df["date"], 0, rolling_ret, where=(rolling_ret >= 0), alpha=0.3, color="green", label="Positive")
    ax.fill_between(pnl_df["date"], 0, rolling_ret, where=(rolling_ret < 0), alpha=0.3, color="red", label="Negative")
    ax.set_title("Rolling 6-Month Avg Daily Return (bps)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg Daily Return (bps)", fontsize=11)
    ax.legend(fontsize=10, loc="best")
    ax.grid(alpha=0.3)
    
    # [2] Daily returns distribution
    ax = axes[2]
    ax.hist(pnl_df["daily_ret_bps"], bins=60, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(x=pnl_df["daily_ret_bps"].mean(), color="green", linestyle="--", linewidth=1.5, label=f"Mean: {pnl_df['daily_ret_bps'].mean():.1f} bps")
    ax.set_title("Daily Returns Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Return (bps)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_decile_monotonicity(decile_df):
    """Plot decile spread and monotonicity
    
    Robustly handles missing deciles from qcut(duplicates='drop')
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Decile returns
    ax = axes[0]
    decile_avg = decile_df.groupby("decile")["mean_ret"].mean()
    decile_std = decile_df.groupby("decile")["mean_ret"].std()
    
    colors = ["red" if x < 0 else "green" for x in decile_avg]
    ax.bar(decile_avg.index, decile_avg.values, color=colors, alpha=0.7, edgecolor="black", 
           yerr=decile_std.values, capsize=5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title(f"Forward {FORWARD_DAYS}d Return by Direction Score Decile (Monotonicity Test)", 
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Decile (1=Worst, 10=Best)", fontsize=11)
    ax.set_ylabel(f"Average {FORWARD_DAYS}d Return (%)", fontsize=11)
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y", alpha=0.3)
    
    # Decile spread over time (robust to missing deciles)
    ax = axes[1]
    decile_ts = decile_df.pivot_table(values="mean_ret", index="date", columns="decile", aggfunc="mean")
    
    # Compute spread using min/max available deciles, or between available extremes
    if 10 in decile_ts.columns and 1 in decile_ts.columns:
        decile_spread = decile_ts[10] - decile_ts[1]
        label = "D10-D1 Spread"
    elif len(decile_ts.columns) >= 2:
        max_decile = decile_ts.columns.max()
        min_decile = decile_ts.columns.min()
        decile_spread = decile_ts[max_decile] - decile_ts[min_decile]
        label = f"D{max_decile}-D{min_decile} Spread"
    else:
        label = "Spread"
        decile_spread = pd.Series(0, index=decile_ts.index)
    
    ax.plot(decile_spread.index, decile_spread.values, linewidth=2, color="purple", label=label)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.fill_between(decile_spread.index, decile_spread.values, 0, alpha=0.3, color="purple")
    ax.set_title("Decile Spread Over Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Spread (%)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig


# ----------------------------
# Output
# ----------------------------

def save_portfolio_selections(selections_df, output_dir):
    """Persist executed review-period holdings for dashboard analysis."""

    if selections_df is None:
        selections_df = pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "portfolio_selections.csv"
    selections_df.to_csv(csv_path, index=False)
    print(f"   ✓ Saved: {csv_path}")
    return selections_df


def print_summary(metrics, decile_df):
    """Print factor backtest summary
    
    Robustly handles missing deciles from qcut(duplicates='drop')
    """
    
    print(f"\n{'='*100}")
    print(f"FACTOR BACKTEST RESULTS")
    print(f"{'='*100}\n")
    
    print(f"📊 OVERALL PERFORMANCE:")
    m = metrics["overall"]
    net_return = m.get("net_total_ret", m["total_ret"])
    print(f"   Net Return: {net_return:+.2f}%")
    if "gross_total_ret" in m:
        print(f"   Gross Return: {m['gross_total_ret']:+.2f}%")
    if "cost_drag_ret_pct" in m:
        print(f"   Cost Drag: {m['cost_drag_ret_pct']:+.2f}%")
    print(f"   Avg Daily Return: {m['avg_daily_ret']:+.2f} bps")
    print(f"   Sharpe Ratio: {m['sharpe']:.3f}")
    print(f"   Max Drawdown: {m['maxdd']:.2f}%")
    print(f"   Win Rate: {m['win_rate']:.1f}%")
    if m.get("total_period_days", m["n_trades"]) != m["n_trades"]:
        print(f"   Period Observations: {m['n_trades']}")
        print(f"   Approx Trading Days: {m['total_period_days']}")
    else:
        print(f"   Trading Days: {m['n_trades']}")
    if "n_reviews" in m:
        print(f"   Reviews / Trade Reviews: {m['n_reviews']} / {m['n_trade_reviews']}")
        print(f"   Review→Trade Rate: {m['review_to_trade_rate']:.1f}%")
        print(f"   Mean Turnover: {m['turnover_mean']:.3f}")
        print(f"   Avg Holding Days: {m['avg_holding_days']:.1f}")
        print(f"   Total Cost: {m['total_cost_bps']:.1f} bps")
    
    # Cross-sectional IC
    if "ic_mean" in metrics["overall"]:
        ic_data = metrics["overall"]
        print(f"   Cross-Sectional IC: {ic_data['ic_mean']:+.4f}")
        print(f"   IC t-stat: {ic_data['ic_tstat']:+.2f}")
        print(f"   IC p-value: {ic_data['ic_pval']:.4f}\n")
    else:
        print()
    
    print(f"📈 REGIME-SPECIFIC PERFORMANCE:")
    for regime, m in metrics.items():
        if regime != "overall":
            print(f"   {regime:12s} | Return: {m['total_ret']:+.2f}% | Sharpe: {m['sharpe']:+.3f} | Days: {m['n_days']}")
    
    print(f"\n🎯 DECILE MONOTONICITY:")
    if len(decile_df) == 0:
        print("   No decile data available")
    else:
        decile_avg = decile_df.groupby("decile")["mean_ret"].mean()
        for d in sorted(decile_avg.index):
            print(f"   D{int(d):2d}: {decile_avg[d]:+.3f}%")
        
        # Compute spread robustly
        if 10 in decile_avg.index and 1 in decile_avg.index:
            spread = decile_avg[10] - decile_avg[1]
            print(f"   Spread (D10-D1): {spread:+.3f}%")
        elif len(decile_avg) >= 2:
            max_decile = decile_avg.idxmax()
            min_decile = decile_avg.idxmin()
            spread = decile_avg[max_decile] - decile_avg[min_decile]
            print(f"   Spread (D{max_decile}-D{min_decile}): {spread:+.3f}%")


# ----------------------------
# Output Deck — 5 Slides
# ----------------------------

# ── Colour palette (dark GitHub theme) ───────────────────────────────────────
_DARK_BG   = "#0d1117"
_CARD_BG   = "#161b22"
_BORDER    = "#30363d"
_TEXT      = "#f0f6fc"
_MUTED     = "#8b949e"
_BLUE      = "#58a6ff"
_GREEN     = "#3fb950"
_RED       = "#f85149"
_YELLOW    = "#d29922"
_TITLE_CLR = "#58a6ff"


def _dark_ax(ax):
    ax.set_facecolor(_CARD_BG)
    ax.tick_params(colors=_MUTED)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER)


def _cagr(cum_series):
    """Annualised return from a cumulative-return Series (base = 1.0)."""
    n = len(cum_series)
    if n < 2:
        return 0.0
    terminal = float(cum_series.iloc[-1])
    if not np.isfinite(terminal) or terminal <= 0:
        return np.nan
    return (terminal ** (252.0 / n) - 1) * 100


def _leg_metrics(daily_bps, spy_aligned=None):
    """CAGR, Sharpe, Beta and cumulative-return series for one leg (bps)."""
    r   = daily_bps / 10_000
    cum = (1 + r).cumprod()
    vol = r.std()
    sharpe = (r.mean() / vol) * np.sqrt(252) if vol > 0 else 0.0
    cagr   = _cagr(cum)
    beta   = np.nan
    if spy_aligned is not None and len(spy_aligned) == len(r):
        r_arr   = r.values if hasattr(r, "values") else np.asarray(r)
        spy_arr = np.asarray(spy_aligned, dtype=float)
        v = np.var(spy_arr)
        if v > 0:
            beta = float(np.cov(r_arr, spy_arr)[0, 1] / v)
    return {"cagr": cagr, "sharpe": sharpe, "beta": beta, "cum": cum}


def compute_ic_series(
    all_data,
    rebalance_dates,
    score_type="direction_score",
    membership_lookup: RussellMembershipLookup | None = None,
):
    """Return a pd.Series of per-period cross-sectional Spearman ICs, indexed by date.

    For magnitude_score, ICs are computed against absolute forward returns.
    Uses the same forward labels created by the signal dump.
    """
    records = []
    for rd in rebalance_dates:
        snapshot = _cross_sectional_snapshot(
            all_data,
            rd,
            score_type,
            require_exact_dates=True,
            selected_only=False,
            membership_lookup=membership_lookup,
        )
        if len(snapshot) >= 10:
            ic, _ = spearmanr(snapshot["score"], snapshot["ret"])
            records.append({"date": pd.Timestamp(rd), "ic": float(ic)})
    if not records:
        return pd.Series(dtype=float)
    return pd.DataFrame(records).set_index("date")["ic"]


def compute_regime_signal_decomposition(
    all_data,
    pnl_df,
    rebalance_dates,
    score_type="direction_score",
    membership_lookup: RussellMembershipLookup | None = None,
):
    """Decompose signal quality by market regime.
    
    For each rebalance period, compute:
    - Cross-sectional IC (Spearman)
    - IC t-stat (per-regime mean / std)
    - Decile spread (D10-D1 realized return)
    - Portfolio turnover (from pnl_df if available)
    - Cross-sectional dispersion (std of scores)
    - Period return (portfolio PnL)
    
    Helps diagnose: Is IC collapsing? Are returns negative despite good IC?
    """
    regime_decomp = {}  # regime -> [list of period metrics]
    execution_schedule = {
        pd.Timestamp(signal_date): pd.Timestamp(trade_date)
        for trade_date, signal_date in zip(
            pd.to_datetime(pnl_df["date"]),
            pd.to_datetime(pnl_df["rebalance_signal_date"], errors="coerce"),
        )
        if pd.notna(signal_date)
    }
    
    for rd in rebalance_dates:
        snapshot = _cross_sectional_snapshot(
            all_data,
            rd,
            score_type,
            require_exact_dates=True,
            selected_only=False,
            membership_lookup=membership_lookup,
        )
        if len(snapshot) < 10:
            continue

        sc = snapshot["score"].tolist()
        rt = snapshot["ret"].tolist()
        
        # Compute IC
        ic, pval = spearmanr(sc, rt)
        
        # Compute decile spread
        df_tmp = pd.DataFrame({"score": sc, "ret": rt})
        df_tmp["decile"] = pd.qcut(df_tmp["score"], q=10, labels=False, duplicates='drop')
        if 0 in df_tmp["decile"].values and 9 in df_tmp["decile"].values:
            d10_ret = df_tmp[df_tmp["decile"] == 9]["ret"].mean()
            d1_ret = df_tmp[df_tmp["decile"] == 0]["ret"].mean()
            decile_spread = (d10_ret - d1_ret) * 100
        else:
            decile_spread = np.nan
        
        # Cross-sectional dispersion (std of scores)
        cs_disp = np.std(sc)
        
        # Attribute the period to the execution date (the next open after the signal close).
        execution_date = execution_schedule.get(pd.Timestamp(rd))
        row_in_pnl = pnl_df[pnl_df["date"] == execution_date] if execution_date is not None else pd.DataFrame()
        if not row_in_pnl.empty:
            regime = row_in_pnl["regime"].values[0]
            turnover = row_in_pnl["cost_bps_applied"].values[0] / COST_RATE / 10000 if COST_RATE > 0 else np.nan
            period_return = row_in_pnl["daily_ret_bps"].values[0]
        else:
            regime = "unknown"
            turnover = np.nan
            period_return = np.nan
        
        if regime not in regime_decomp:
            regime_decomp[regime] = []
        
        regime_decomp[regime].append({
            "date": rd,
            "ic": float(ic) if np.isfinite(ic) else np.nan,
            "pval": float(pval),
            "decile_spread": decile_spread,
            "cs_disp": cs_disp,
            "turnover": turnover,
            "period_return_bps": period_return,
            "n_obs": len(sc),
        })
    
    # Aggregate per regime
    regime_stats = {}
    for regime, periods in regime_decomp.items():
        ics = [p["ic"] for p in periods if pd.notna(p["ic"])]
        spreads = [p["decile_spread"] for p in periods if pd.notna(p["decile_spread"])]
        turnovers = [p["turnover"] for p in periods if pd.notna(p["turnover"])]
        
        if len(ics) > 0:
            ic_mean = np.mean(ics)
            ic_std = np.std(ics)
            ic_tstat = (ic_mean / ic_std) * np.sqrt(len(ics)) if ic_std > 0 else 0.0
        else:
            ic_mean = ic_std = ic_tstat = np.nan
        
        # Compute total regime return from pnl_df (matches the regime performance summary)
        regime_pnl = pnl_df[pnl_df["regime"] == regime].copy()
        if len(regime_pnl) > 0:
            # daily_ret_bps is in basis points; convert to decimal: bps / 10000 = decimal return
            regime_cum = (1 + regime_pnl["daily_ret_bps"] / 10_000).prod()
            total_regime_ret = (regime_cum - 1) * 100
            
            # Decompose into long and short leg returns (only for ACTIVE leg)
            if "long_pnl_bps" in regime_pnl.columns and "short_pnl_bps" in regime_pnl.columns:
                # Only compound days where each leg was actually active
                long_active_days = regime_pnl[regime_pnl["long_pnl_bps"].abs() > 0.001]
                short_active_days = regime_pnl[regime_pnl["short_pnl_bps"].abs() > 0.001]

                if len(long_active_days) > 0:
                    long_cum = (1 + long_active_days["long_pnl_bps"] / 10_000).prod()
                    long_ret = (long_cum - 1) * 100
                else:
                    long_ret = 0.0

                if len(short_active_days) > 0:
                    short_cum = (1 + short_active_days["short_pnl_bps"] / 10_000).prod()
                    short_ret = (short_cum - 1) * 100
                else:
                    short_ret = 0.0
            else:
                long_ret = short_ret = np.nan
        else:
            total_regime_ret = np.nan
            long_ret = short_ret = np.nan
        
        regime_stats[regime] = {
            "n_periods": len(periods),
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_tstat": ic_tstat,
            "decile_spread_mean": np.mean(spreads) if spreads else np.nan,
            "decile_spread_std": np.std(spreads) if spreads else np.nan,
            "turnover_mean": np.mean(turnovers) if turnovers else np.nan,
            "cs_disp_mean": np.mean([p["cs_disp"] for p in periods]),
            "total_regime_return_pct": total_regime_ret,
            "long_regime_return_pct": long_ret,
            "short_regime_return_pct": short_ret,
        }
    
    return regime_stats


def print_regime_occurrences(pnl_df):
    """Print date ranges and clustering for crisis/squeeze regimes."""
    print(f"\n📅 REGIME OCCURRENCE DATES:\n")
    
    for regime in ["crisis", "squeeze"]:
        regime_dates = pnl_df[pnl_df["regime"] == regime]["date"].sort_values()
        
        if len(regime_dates) == 0:
            print(f"   {regime.upper()}: No days")
            continue
        
        # Find clusters (consecutive or near-consecutive dates)
        regime_dates_list = regime_dates.values
        date_diffs = np.diff(regime_dates_list)
        
        # If more than 1 day gap, it's a new cluster
        gap_threshold = np.timedelta64(10, 'D') 
        cluster_breaks = np.where(date_diffs > gap_threshold)[0]
        
        clusters = []
        start_idx = 0
        for break_idx in cluster_breaks:
            clusters.append((regime_dates_list[start_idx], regime_dates_list[break_idx]))
            start_idx = break_idx + 1
        clusters.append((regime_dates_list[start_idx], regime_dates_list[-1]))
        
        print(f"   {regime.upper()}: {len(regime_dates)} days in {len(clusters)} cluster(s)")
        for i, (start, end) in enumerate(clusters):
            start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
            end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
            duration = (pd.Timestamp(end) - pd.Timestamp(start)).days
            print(f"      Cluster {i+1}: {start_str} → {end_str} ({duration} days)")
        print()


# ── Slide 1 — Factor Summary ──────────────────────────────────────────────────
def _slide1_factor_summary(metrics, all_data, ic_series, output_dir, score_type="direction_score"):
    m      = metrics["overall"]
    ic_std = float(ic_series.std()) if len(ic_series) > 1 else 0.0
    icir   = m["ic_mean"] / ic_std   if ic_std > 0 else 0.0

    predicts_label = {
        "direction_score": "10-Day Forward Return Direction",
        "magnitude_score": "10-Day Absolute Forward Return",
    }.get(score_type, "10-Day Forward Return")

    def _color(val_str, signed):
        if not signed:
            return _TEXT
        try:
            return _GREEN if float(str(val_str).replace("%", "").replace("+", "")) > 0 else _RED
        except Exception:
            return _TEXT

    rows = [
        ("Factor",       score_type.replace("_", " ").title(),          False),
        ("Predicts",     predicts_label,                                 False),
        ("Horizon",      f"{REBALANCE_DAYS} Trading Days",              False),
        ("Universe",     f"{len(all_data)} Stocks",                     False),
        ("IC Periods",   str(m.get("n_ic_periods", len(ic_series))),    False),
        (None, None, None),
        ("IC Mean",      f"{m['ic_mean']:+.4f}",                        True),
        ("IC Std",       f"{ic_std:.4f}",                               False),
        ("ICIR",         f"{icir:+.3f}",                                True),
        ("IC t-stat",    f"{m['ic_tstat']:+.2f}",                       True),
        ("IC p-value",   f"{m['ic_pval']:.4f}",                         False),
        (None, None, None),
        ("Total Return", f"{m['total_ret']:+.2f}%",                     True),
        ("Sharpe Ratio", f"{m['sharpe']:.3f}",                          True),
        ("Max Drawdown", f"{m['maxdd']:.2f}%",                          False),
        ("Win Rate",     f"{m['win_rate']:.1f}%",                       False),
    ]

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor(_DARK_BG)

    ax_l = fig.add_axes([0.03, 0.05, 0.42, 0.90])
    ax_l.set_facecolor(_DARK_BG)
    ax_l.axis("off")
    ax_l.text(0.04, 0.98, "SLIDE 1 — FACTOR SUMMARY", color=_TITLE_CLR,
              fontsize=10, fontweight="bold", transform=ax_l.transAxes, va="top")

    y, dy = 0.90, 0.056
    for label, val, signed in rows:
        if label is None:
            y -= dy * 0.35
            continue
        ax_l.text(0.04, y, f"{label}:", color=_MUTED, fontsize=10.5,
                  transform=ax_l.transAxes, va="top")
        ax_l.text(0.48, y, str(val), color=_color(val, signed), fontsize=10.5,
                  fontweight="bold", transform=ax_l.transAxes, va="top")
        y -= dy

    ax_r = fig.add_axes([0.52, 0.12, 0.44, 0.74])
    _dark_ax(ax_r)
    if len(ic_series) > 5:
        iv = ic_series.dropna().values
        ax_r.hist(iv, bins=25, color=_BLUE, alpha=0.75, edgecolor=_DARK_BG, linewidth=0.6)
        ax_r.axvline(iv.mean(), color=_GREEN, linewidth=2, linestyle="--",
                     label=f"Mean = {iv.mean():+.4f}")
        ax_r.axvline(0, color=_RED, linewidth=1.2, linestyle="-", alpha=0.6, label="Zero")
        ax_r.legend(fontsize=10, facecolor="#21262d", labelcolor=_TEXT, framealpha=0.9)
    ax_r.set_title("IC Distribution (Cross-Sectional Spearman)", color=_TEXT, fontsize=12, pad=10)
    ax_r.set_xlabel("Information Coefficient", color=_MUTED, fontsize=10)
    ax_r.set_ylabel("Frequency",               color=_MUTED, fontsize=10)

    fig.suptitle("Factor Analytics — Summary Card", color=_TEXT, fontsize=15, fontweight="bold", y=1.01)
    path = output_dir / "slide_01_factor_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    print(f"   ✓ Slide 1: {path}")


# ── Slide 2 — Cumulative Performance ─────────────────────────────────────────
def _slide2_cumulative_performance(pnl_df, output_dir):
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor(_DARK_BG)

    ax = fig.add_axes([0.06, 0.15, 0.88, 0.75])
    _dark_ax(ax)

    ax.plot(pnl_df["date"], pnl_df["cumulative_ret_pct"],
            color=_BLUE, linewidth=2.8)
    ax.axhline(0, color=_MUTED, linewidth=0.9, linestyle="--")
    ax.fill_between(pnl_df["date"], pnl_df["cumulative_ret_pct"], 0,
                    where=(pnl_df["cumulative_ret_pct"] >= 0),
                    alpha=0.15, color=_GREEN)
    ax.fill_between(pnl_df["date"], pnl_df["cumulative_ret_pct"], 0,
                    where=(pnl_df["cumulative_ret_pct"] < 0),
                    alpha=0.15, color=_RED)

    ax.set_title("Cumulative Factor Performance", color=_TEXT,
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date", color=_MUTED, fontsize=11)
    ax.set_ylabel("Cumulative Return (%)", color=_MUTED, fontsize=11)

    fig.suptitle("Factor Analytics — Performance Over Time",
                 color=_TEXT, fontsize=15, fontweight="bold", y=1.01)
    path = output_dir / "slide_02_cumulative_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    print(f"   ✓ Slide 2: {path}")


# ── Slide 3 — Decile Spread ───────────────────────────────────────────────────
def _slide3_decile_spread(decile_df, output_dir):
    if decile_df.empty:
        print("   ⚠️  Slide 3 skipped — no decile data")
        return

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor(_DARK_BG)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    ax_bar = fig.add_axes([0.07, 0.12, 0.54, 0.76])
    _dark_ax(ax_bar)

    decile_avg = decile_df.groupby("decile")["mean_ret"].mean()
    decile_std = decile_df.groupby("decile")["std_ret"].mean()
    n_avg      = decile_df.groupby("decile")["n_stocks"].mean()

    cmap       = plt.cm.RdYlGn
    n          = len(decile_avg)
    bar_colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    ax_bar.bar(decile_avg.index, decile_avg.values,
               color=bar_colors, alpha=0.85, edgecolor=_DARK_BG, linewidth=0.6,
               yerr=decile_std.values, capsize=4,
               error_kw={"ecolor": _MUTED, "linewidth": 1})
    ax_bar.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")

    y_pad = decile_avg.abs().max() * 0.12 + 0.005
    for d, val in decile_avg.items():
        ns = n_avg.get(d, 0)
        yp = val + y_pad if val >= 0 else val - y_pad
        va = "bottom" if val >= 0 else "top"
        ax_bar.text(d, yp, f"n≈{ns:.0f}", ha="center", va=va, color=_MUTED, fontsize=8)

    ax_bar.set_title(f"Forward {FORWARD_DAYS}d Return by Score Decile", color=_TEXT,
                     fontsize=13, fontweight="bold", pad=10)
    ax_bar.set_xlabel("Score Decile  (1 = Worst  →  10 = Best)", color=_MUTED, fontsize=11)
    ax_bar.set_ylabel(f"Avg {FORWARD_DAYS}d Return (%)", color=_MUTED, fontsize=11)
    ax_bar.set_xticks(range(1, 11))

    # ── Stats card ────────────────────────────────────────────────────────────
    ax_s = fig.add_axes([0.66, 0.12, 0.31, 0.76])
    ax_s.set_facecolor(_CARD_BG)
    ax_s.axis("off")
    for sp in ax_s.spines.values():
        sp.set_edgecolor(_BORDER)

    decile_ts = decile_df.pivot_table(values="mean_ret", index="date",
                                      columns="decile", aggfunc="mean")
    if 10 in decile_ts.columns and 1 in decile_ts.columns:
        spread_ts    = (decile_ts[10] - decile_ts[1]).dropna()
        spread_label = "D10 − D1"
    elif len(decile_ts.columns) >= 2:
        hi, lo       = decile_ts.columns.max(), decile_ts.columns.min()
        spread_ts    = (decile_ts[hi] - decile_ts[lo]).dropna()
        spread_label = f"D{hi} − D{lo}"
    else:
        spread_ts    = pd.Series(dtype=float)
        spread_label = "N/A"

    ppy          = 252.0 / REBALANCE_DAYS
    spread_avg   = spread_ts.mean()   if len(spread_ts) > 0 else 0.0
    spread_sharpe = (spread_ts.mean() / spread_ts.std()) * np.sqrt(ppy) \
                    if len(spread_ts) > 1 and spread_ts.std() > 0 else 0.0
    spread_hit   = (spread_ts > 0).mean() * 100 if len(spread_ts) > 0 else 0.0

    vals_arr = decile_avg.values
    n_up     = sum(vals_arr[i + 1] > vals_arr[i] for i in range(len(vals_arr) - 1))
    mono     = n_up / (len(vals_arr) - 1) * 100 if len(vals_arr) > 1 else 0.0

    d1_ret  = decile_avg.get(1,  np.nan)
    d10_ret = decile_avg.get(10, np.nan)

    stat_rows = [
        ("SLIDE 3 — DECILE SPREAD",  None,                                                None),
        (None, None, None),
        ("Spread",        f"{spread_label}: {spread_avg:+.3f}%",  _GREEN if spread_avg > 0 else _RED),
        ("Spread Sharpe", f"{spread_sharpe:.3f}",                  _GREEN if spread_sharpe > 0 else _RED),
        ("Hit Rate",      f"{spread_hit:.1f}%",                    _GREEN if spread_hit > 50 else _RED),
        ("Monotonicity",  f"{mono:.0f}% of steps ↑",              _GREEN if mono >= 70 else _YELLOW),
        (None, None, None),
        ("D10 Avg Ret",   f"{d10_ret:+.3f}%" if pd.notna(d10_ret) else "N/A",  _TEXT),
        ("D1  Avg Ret",   f"{d1_ret:+.3f}%"  if pd.notna(d1_ret)  else "N/A",  _TEXT),
        ("Periods",       str(len(spread_ts)),                                   _TEXT),
    ]

    y, dy = 0.96, 0.09
    for label, val, vc in stat_rows:
        if label is None:
            y -= dy * 0.3
            continue
        if vc is None:
            ax_s.text(0.06, y, label, color=_TITLE_CLR, fontsize=9.5,
                      fontweight="bold", transform=ax_s.transAxes, va="top")
        else:
            ax_s.text(0.06, y,          f"{label}:", color=_MUTED, fontsize=10.5,
                      transform=ax_s.transAxes, va="top")
            ax_s.text(0.06, y - 0.045, str(val),     color=vc,    fontsize=12,
                      fontweight="bold", transform=ax_s.transAxes, va="top")
        y -= dy

    fig.suptitle("Factor Analytics — Decile Spread Analysis",
                 color=_TEXT, fontsize=15, fontweight="bold", y=1.01)
    path = output_dir / "slide_03_decile_spread.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    print(f"   ✓ Slide 3: {path}")


# ── Slide 4 — Long / Short Breakdown ─────────────────────────────────────────
def _slide4_long_short_breakdown(pnl_df, spy_df, output_dir):
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor(_DARK_BG)

    spy_aligned = None
    if spy_df is not None:
        spy_cp = spy_df[["date", "adj_close"]].copy()
        spy_cp["date"] = pd.to_datetime(spy_cp["date"])
        spy_cp = spy_cp.sort_values("date").set_index("date")
        spy_cp["spy_r"] = spy_cp["adj_close"].pct_change()
        spy_aligned = spy_cp["spy_r"].reindex(pnl_df["date"].values).fillna(0).values

    long_m = _leg_metrics(pnl_df["long_pnl_bps"],  spy_aligned)
    sht_m  = _leg_metrics(pnl_df["short_pnl_bps"], spy_aligned)
    comb_m = _leg_metrics(pnl_df["daily_ret_bps"],  spy_aligned)

    # ── Cumulative return chart ───────────────────────────────────────────────
    ax = fig.add_axes([0.06, 0.20, 0.58, 0.70])
    _dark_ax(ax)
    ax.plot(pnl_df["date"], (long_m["cum"] - 1) * 100, color=_GREEN, linewidth=2,   label="Long Leg")
    ax.plot(pnl_df["date"], (sht_m["cum"]  - 1) * 100, color=_RED,   linewidth=2,   label="Short Leg")
    ax.plot(pnl_df["date"], (comb_m["cum"] - 1) * 100, color=_BLUE,  linewidth=2.5, label="Combined", zorder=5)
    ax.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
    ax.set_title("Long vs Short Leg Cumulative Return", color=_TEXT,
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Cumulative Return (%)", color=_MUTED, fontsize=11)
    ax.legend(fontsize=11, facecolor="#21262d", labelcolor=_TEXT, framealpha=0.9)

    # ── Stats table ───────────────────────────────────────────────────────────
    ax_t = fig.add_axes([0.68, 0.10, 0.29, 0.82])
    ax_t.set_facecolor(_CARD_BG)
    ax_t.axis("off")
    for sp in ax_t.spines.values():
        sp.set_edgecolor(_BORDER)

    def fmt_b(b):
        return f"{b:+.3f}" if pd.notna(b) else "N/A"

    ax_t.text(0.04, 0.97, "SLIDE 4 — L/S BREAKDOWN", color=_TITLE_CLR,
              fontsize=9.5, fontweight="bold", transform=ax_t.transAxes, va="top")

    col_x = [0.04, 0.30, 0.55, 0.78]
    headers = ("Metric", "Long", "Short", "Comb")
    metric_rows = [
        ("CAGR",   f"{long_m['cagr']:+.2f}%",   f"{sht_m['cagr']:+.2f}%",   f"{comb_m['cagr']:+.2f}%"),
        ("Sharpe", f"{long_m['sharpe']:.3f}",    f"{sht_m['sharpe']:.3f}",   f"{comb_m['sharpe']:.3f}"),
        ("Beta",   fmt_b(long_m["beta"]),         fmt_b(sht_m["beta"]),       fmt_b(comb_m["beta"])),
    ]

    y = 0.84
    for j, h in enumerate(headers):
        ax_t.text(col_x[j], y, h, color=_BLUE, fontsize=10, fontweight="bold",
                  transform=ax_t.transAxes, va="top")
    y -= 0.10

    for label, v_l, v_s, v_c in metric_rows:
        ax_t.text(col_x[0], y, f"{label}:", color=_MUTED, fontsize=11,
                  transform=ax_t.transAxes, va="top")
        for cx, val in zip(col_x[1:], [v_l, v_s, v_c]):
            try:
                num = float(str(val).replace("%", "").replace("+", ""))
                vc  = _GREEN if num > 0 else (_RED if num < 0 else _TEXT)
            except Exception:
                vc = _TEXT
            ax_t.text(cx, y, val, color=vc, fontsize=11, fontweight="bold",
                      transform=ax_t.transAxes, va="top")
        y -= 0.13

    fig.suptitle("Factor Analytics — Long / Short Breakdown",
                 color=_TEXT, fontsize=15, fontweight="bold", y=1.01)
    path = output_dir / "slide_04_long_short_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    print(f"   ✓ Slide 4: {path}")


# ── Slide 5 — Rolling Stability ───────────────────────────────────────────────
def _slide5_rolling_stability(pnl_df, ic_series, output_dir):
    WIN_D  = 63   # ~3-month rolling Sharpe window (trading days)
    WIN_IC = 12   # 12-period rolling IC window

    r       = pnl_df["daily_ret_bps"] / 10_000
    rm, rs  = r.rolling(WIN_D).mean(), r.rolling(WIN_D).std()
    roll_sh = ((rm / rs) * np.sqrt(252)).replace([np.inf, -np.inf], np.nan)

    fig, (ax_sh, ax_ic) = plt.subplots(2, 1, figsize=(16, 9))
    fig.patch.set_facecolor(_DARK_BG)

    _dark_ax(ax_sh)
    ax_sh.plot(pnl_df["date"], roll_sh, color=_BLUE, linewidth=1.5,
               label=f"Rolling {WIN_D}d Sharpe")
    ax_sh.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
    ax_sh.fill_between(pnl_df["date"], roll_sh.fillna(0), 0,
                       where=(roll_sh.fillna(0) >= 0), alpha=0.20, color=_GREEN)
    ax_sh.fill_between(pnl_df["date"], roll_sh.fillna(0), 0,
                       where=(roll_sh.fillna(0) <  0), alpha=0.20, color=_RED)
    ax_sh.set_title(f"Rolling {WIN_D}-Day Sharpe Ratio", color=_TEXT,
                    fontsize=12, fontweight="bold", pad=8)
    ax_sh.set_ylabel("Sharpe", color=_MUTED, fontsize=11)
    ax_sh.legend(fontsize=10, facecolor="#21262d", labelcolor=_TEXT)

    _dark_ax(ax_ic)
    if len(ic_series) > WIN_IC:
        roll_ic = ic_series.rolling(WIN_IC).mean()
        ax_ic.plot(ic_series.index, ic_series.values,
                   color=_MUTED, linewidth=0.8, alpha=0.55, label="Raw IC per period")
        ax_ic.plot(roll_ic.index, roll_ic.values,
                   color=_YELLOW, linewidth=2, label=f"Rolling {WIN_IC}-period IC")
        ax_ic.fill_between(roll_ic.index, roll_ic.fillna(0).values, 0,
                           where=(roll_ic.fillna(0) >= 0), alpha=0.20, color=_GREEN)
        ax_ic.fill_between(roll_ic.index, roll_ic.fillna(0).values, 0,
                           where=(roll_ic.fillna(0) <  0), alpha=0.20, color=_RED)
        ax_ic.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
        ax_ic.axhline(ic_series.mean(), color=_YELLOW, linewidth=1.2, linestyle=":",
                      alpha=0.8, label=f"Mean IC = {ic_series.mean():+.4f}")
        ax_ic.legend(fontsize=10, facecolor="#21262d", labelcolor=_TEXT)
    else:
        ax_ic.text(0.5, 0.5, "Insufficient IC data for rolling window",
                   transform=ax_ic.transAxes, ha="center", va="center",
                   color=_MUTED, fontsize=12)
    ax_ic.set_title(f"Rolling {WIN_IC}-Period Information Coefficient (Spearman)",
                    color=_TEXT, fontsize=12, fontweight="bold", pad=8)
    ax_ic.set_ylabel("IC", color=_MUTED, fontsize=11)

    fig.suptitle("Factor Analytics — Rolling Stability",
                 color=_TEXT, fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = output_dir / "slide_05_rolling_stability.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    print(f"   ✓ Slide 5: {path}")


# ── Slide 6 — Cost Sensitivity ────────────────────────────────────────────────
def _slide6_cost_sensitivity(pnl_df, output_dir):
    if "cost_bps_applied" not in pnl_df.columns:
        print("   ⚠️  Slide 5 skipped — cost_bps_applied column missing")
        return

    gross_pnl  = pnl_df["daily_ret_bps"] + pnl_df["cost_bps_applied"]
    cost_sweep = np.arange(0, 55, 5)
    sharpes, cagrs = [], []

    for c_bps in cost_sweep:
        c_rate = c_bps / 10_000
        scale  = c_rate / COST_RATE if COST_RATE > 0 else 0.0
        adj    = gross_pnl - pnl_df["cost_bps_applied"] * scale
        r      = adj / 10_000
        cum    = (1 + r).cumprod()
        vol    = r.std()
        sharpes.append((r.mean() / vol) * np.sqrt(252) if vol > 0 else 0.0)
        cagrs.append(_cagr(cum))

    cur_bps = COST_RATE * 10_000

    fig, (ax_sh, ax_cagr) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(_DARK_BG)

    for ax, y_vals, y_lbl, title in [
        (ax_sh,   sharpes, "Sharpe Ratio", "Sharpe vs Transaction Cost"),
        (ax_cagr, cagrs,   "CAGR (%)",     "CAGR vs Transaction Cost"),
    ]:
        _dark_ax(ax)
        y_arr = np.asarray(np.real_if_close(y_vals), dtype=float)
        for i in range(len(cost_sweep) - 1):
            segment = y_arr[i:i + 2]
            if not np.all(np.isfinite(segment)):
                continue
            seg_col = _GREEN if segment[0] > 0 else _RED
            ax.plot(cost_sweep[i:i + 2], segment, color=seg_col, linewidth=2.5)
        finite_mask = np.isfinite(y_arr)
        ax.scatter(cost_sweep[finite_mask], y_arr[finite_mask], color=_BLUE, zorder=5, s=55)
        ax.axvline(cur_bps, color=_YELLOW, linewidth=2, linestyle="--",
                   label=f"Current = {cur_bps:.0f} bps")
        ax.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
        ax.fill_between(cost_sweep, y_arr, 0,
                        where=finite_mask & (y_arr >= 0), alpha=0.10, color=_GREEN)
        ax.fill_between(cost_sweep, y_arr, 0,
                        where=finite_mask & (y_arr <  0), alpha=0.10, color=_RED)
        # Breakeven marker
        for i in range(len(cost_sweep) - 1):
            if not np.all(np.isfinite(y_arr[i:i + 2])):
                continue
            if (y_arr[i] >= 0 > y_arr[i + 1]) or (y_arr[i] < 0 <= y_arr[i + 1]):
                be = float(np.interp(0, [y_arr[i], y_arr[i + 1]],
                                        [cost_sweep[i], cost_sweep[i + 1]]))
                ax.axvline(be, color=_RED, linewidth=1.5, linestyle=":",
                           label=f"Breakeven ≈ {be:.1f} bps")
                break
        ax.set_title(title, color=_TEXT, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("One-Way Transaction Cost (bps)", color=_MUTED, fontsize=11)
        ax.set_ylabel(y_lbl, color=_MUTED, fontsize=11)
        ax.set_xticks(cost_sweep)
        ax.legend(fontsize=11, facecolor="#21262d", labelcolor=_TEXT)

    fig.suptitle("Factor Analytics — Cost Sensitivity",
                 color=_TEXT, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = output_dir / "slide_06_cost_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    print(f"   ✓ Slide 6: {path}")


# ── Slide 7 — Long-Only Summary ──────────────────────────────────────────────
def _slide7_long_only_summary(pnl_df, metrics, all_data, ic_series, output_dir, score_type="direction_score"):
    """Same stat card as Slide 1 but metrics computed from the long leg only."""
    m_full = metrics["overall"]
    ic_std = float(ic_series.std()) if len(ic_series) > 1 else 0.0
    icir   = m_full["ic_mean"] / ic_std if ic_std > 0 else 0.0

    # ── Long-only performance ──────────────────────────────────────────────────
    lo_r     = pnl_df["long_pnl_bps"] / 10_000
    lo_cum   = (1 + lo_r).cumprod()
    lo_vol   = lo_r.std()
    lo_sharpe = (lo_r.mean() / lo_vol) * np.sqrt(252) if lo_vol > 0 else 0.0
    lo_total  = (float(lo_cum.iloc[-1]) - 1) * 100
    lo_maxdd  = float(
        np.min((lo_cum.values - np.maximum.accumulate(lo_cum.values))
               / np.maximum.accumulate(lo_cum.values)) * 100
    )
    lo_winrate = (pnl_df["long_pnl_bps"] > 0).sum() / len(pnl_df) * 100

    def _color(val_str, signed):
        if not signed:
            return _TEXT
        try:
            return _GREEN if float(str(val_str).replace("%", "").replace("+", "")) > 0 else _RED
        except Exception:
            return _TEXT

    rows = [
        ("Factor",        score_type.replace("_", " ").title(),          False),
        ("Strategy",      "Long-Only  (Top N)",                           False),
        ("Horizon",       f"{REBALANCE_DAYS} Trading Days",              False),
        ("Universe",      f"{len(all_data)} Stocks",                     False),
        ("IC Periods",    str(m_full.get("n_ic_periods", len(ic_series))), False),
        (None, None, None),
        ("IC Mean",       f"{m_full['ic_mean']:+.4f}",                   True),
        ("IC Std",        f"{ic_std:.4f}",                               False),
        ("ICIR",          f"{icir:+.3f}",                                True),
        ("IC t-stat",     f"{m_full['ic_tstat']:+.2f}",                  True),
        ("IC p-value",    f"{m_full['ic_pval']:.4f}",                    False),
        (None, None, None),
        ("Total Return",  f"{lo_total:+.2f}%",                           True),
        ("Sharpe Ratio",  f"{lo_sharpe:.3f}",                            True),
        ("Max Drawdown",  f"{lo_maxdd:.2f}%",                            False),
        ("Win Rate",      f"{lo_winrate:.1f}%",                          False),
    ]

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor(_DARK_BG)

    ax_l = fig.add_axes([0.03, 0.05, 0.42, 0.90])
    ax_l.set_facecolor(_DARK_BG)
    ax_l.axis("off")
    ax_l.text(0.04, 0.98, "SLIDE 7 — LONG-ONLY SUMMARY", color=_TITLE_CLR,
              fontsize=10, fontweight="bold", transform=ax_l.transAxes, va="top")

    y, dy = 0.90, 0.056
    for label, val, signed in rows:
        if label is None:
            y -= dy * 0.35
            continue
        ax_l.text(0.04, y, f"{label}:", color=_MUTED, fontsize=10.5,
                  transform=ax_l.transAxes, va="top")
        ax_l.text(0.48, y, str(val), color=_color(val, signed), fontsize=10.5,
                  fontweight="bold", transform=ax_l.transAxes, va="top")
        y -= dy

    # ── Right panel: long-only cumulative return ──────────────────────────────
    ax_r = fig.add_axes([0.52, 0.12, 0.44, 0.74])
    _dark_ax(ax_r)
    lo_cum_pct = (lo_cum - 1) * 100
    ax_r.plot(pnl_df["date"], lo_cum_pct, color=_GREEN, linewidth=2)
    ax_r.fill_between(pnl_df["date"], lo_cum_pct, 0,
                      where=(lo_cum_pct >= 0), alpha=0.15, color=_GREEN)
    ax_r.fill_between(pnl_df["date"], lo_cum_pct, 0,
                      where=(lo_cum_pct <  0), alpha=0.15, color=_RED)
    ax_r.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
    ax_r.set_title("Long-Only Cumulative Return", color=_TEXT, fontsize=12, pad=10)
    ax_r.set_xlabel("Date",          color=_MUTED, fontsize=10)
    ax_r.set_ylabel("Return (%)",    color=_MUTED, fontsize=10)

    # Annotate final return
    ax_r.text(0.98, 0.04, f"Final: {lo_total:+.1f}%",
              transform=ax_r.transAxes, ha="right", va="bottom",
              color=_GREEN if lo_total >= 0 else _RED,
              fontsize=11, fontweight="bold")

    fig.suptitle("Factor Analytics — Long-Only Summary Card",
                 color=_TEXT, fontsize=15, fontweight="bold", y=1.01)
    path = output_dir / "slide_07_long_only_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    print(f"   ✓ Slide 7: {path}")


# ── Slide 8 — Short-Only Summary ─────────────────────────────────────────────
def _slide8_short_only_summary(pnl_df, metrics, all_data, ic_series, output_dir, score_type="direction_score"):
    """Same stat card as Slide 7 but metrics computed from the short leg only."""
    m_full = metrics["overall"]
    ic_std = float(ic_series.std()) if len(ic_series) > 1 else 0.0
    icir   = m_full["ic_mean"] / ic_std if ic_std > 0 else 0.0

    # ── Short-only performance ──────────────────────────────────────────────────
    so_r     = pnl_df["short_pnl_bps"] / 10_000
    so_cum   = (1 + so_r).cumprod()
    so_vol   = so_r.std()
    so_sharpe = (so_r.mean() / so_vol) * np.sqrt(252) if so_vol > 0 else 0.0
    so_total  = (float(so_cum.iloc[-1]) - 1) * 100
    so_maxdd  = float(
        np.min((so_cum.values - np.maximum.accumulate(so_cum.values))
               / np.maximum.accumulate(so_cum.values)) * 100
    )
    so_winrate = (pnl_df["short_pnl_bps"] > 0).sum() / len(pnl_df) * 100

    def _color(val_str, signed):
        if not signed:
            return _TEXT
        try:
            return _GREEN if float(str(val_str).replace("%", "").replace("+", "")) > 0 else _RED
        except Exception:
            return _TEXT

    rows = [
        ("Factor",        score_type.replace("_", " ").title(),          False),
        ("Strategy",      "Short-Only (Bottom N)",                        False),
        ("Horizon",       f"{REBALANCE_DAYS} Trading Days",              False),
        ("Universe",      f"{len(all_data)} Stocks",                     False),
        ("IC Periods",    str(m_full.get("n_ic_periods", len(ic_series))), False),
        (None, None, None),
        ("IC Mean",       f"{m_full['ic_mean']:+.4f}",                   True),
        ("IC Std",        f"{ic_std:.4f}",                               False),
        ("ICIR",          f"{icir:+.3f}",                                True),
        ("IC t-stat",     f"{m_full['ic_tstat']:+.2f}",                  True),
        ("IC p-value",    f"{m_full['ic_pval']:.4f}",                    False),
        (None, None, None),
        ("Total Return",  f"{so_total:+.2f}%",                           True),
        ("Sharpe Ratio",  f"{so_sharpe:.3f}",                            True),
        ("Max Drawdown",  f"{so_maxdd:.2f}%",                            False),
        ("Win Rate",      f"{so_winrate:.1f}%",                          False),
    ]

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor(_DARK_BG)

    ax_l = fig.add_axes([0.03, 0.05, 0.42, 0.90])
    ax_l.set_facecolor(_DARK_BG)
    ax_l.axis("off")
    ax_l.text(0.04, 0.98, "SLIDE 8 — SHORT-ONLY SUMMARY", color=_TITLE_CLR,
              fontsize=10, fontweight="bold", transform=ax_l.transAxes, va="top")

    y, dy = 0.90, 0.056
    for label, val, signed in rows:
        if label is None:
            y -= dy * 0.35
            continue
        ax_l.text(0.04, y, f"{label}:", color=_MUTED, fontsize=10.5,
                  transform=ax_l.transAxes, va="top")
        ax_l.text(0.48, y, str(val), color=_color(val, signed), fontsize=10.5,
                  fontweight="bold", transform=ax_l.transAxes, va="top")
        y -= dy

    # ── Right panel: short-only cumulative return ──────────────────────────────
    ax_r = fig.add_axes([0.52, 0.12, 0.44, 0.74])
    _dark_ax(ax_r)
    so_cum_pct = (so_cum - 1) * 100
    ax_r.plot(pnl_df["date"], so_cum_pct, color=_RED, linewidth=2)
    ax_r.fill_between(pnl_df["date"], so_cum_pct, 0,
                      where=(so_cum_pct >= 0), alpha=0.15, color=_GREEN)
    ax_r.fill_between(pnl_df["date"], so_cum_pct, 0,
                      where=(so_cum_pct <  0), alpha=0.15, color=_RED)
    ax_r.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
    ax_r.set_title("Short-Only Cumulative Return", color=_TEXT, fontsize=12, pad=10)
    ax_r.set_xlabel("Date",          color=_MUTED, fontsize=10)
    ax_r.set_ylabel("Return (%)",    color=_MUTED, fontsize=10)

    # Annotate final return
    ax_r.text(0.98, 0.04, f"Final: {so_total:+.1f}%",
              transform=ax_r.transAxes, ha="right", va="bottom",
              color=_GREEN if so_total >= 0 else _RED,
              fontsize=11, fontweight="bold")

    fig.suptitle("Factor Analytics — Short-Only Summary Card",
                 color=_TEXT, fontsize=15, fontweight="bold", y=1.01)
    path = output_dir / "slide_08_short_only_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    print(f"   ✓ Slide 8: {path}")


# ── Per-factor colour coding (used in slide 7) ───────────────────────────────
_FACTOR_COLORS = {
    "direction_score": _BLUE,
    "magnitude_score": _GREEN,
}


# ── Slide 7 — Multi-Factor Comparison ────────────────────────────────────────
def _slide7_comparison(results, spy_df, output_dir):
    """Overlay cumulative returns + side-by-side stats table for all score types.

    results: dict  {score_type: (pnl_df, metrics)}
    """
    if not results:
        return

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor(_DARK_BG)

    # ── Cumulative return chart ───────────────────────────────────────────────
    ax = fig.add_axes([0.05, 0.12, 0.57, 0.76])
    _dark_ax(ax)

    for st, (pnl_df, _) in results.items():
        color = _FACTOR_COLORS.get(st, _TEXT)
        label = st.replace("_", " ").title()
        ax.plot(pnl_df["date"], pnl_df["cumulative_ret_pct"],
                color=color, linewidth=2, label=label)

    if spy_df is not None:
        spy_cp = spy_df[["date", "adj_close"]].copy()
        spy_cp["date"] = pd.to_datetime(spy_cp["date"])
        spy_cp = spy_cp.sort_values("date")
        first_pnl = next(iter(results.values()))[0]
        spy_cp = spy_cp[
            (spy_cp["date"] >= first_pnl["date"].min()) &
            (spy_cp["date"] <= first_pnl["date"].max())
        ]
        if len(spy_cp) > 0:
            spy_ret = (spy_cp["adj_close"] / spy_cp["adj_close"].iloc[0] - 1) * 100
            ax.plot(spy_cp["date"], spy_ret,
                    color=_MUTED, linewidth=1.5, linestyle="--",
                    label="S&P 500", alpha=0.7)

    ax.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
    ax.set_title("Multi-Factor Cumulative Return Comparison", color=_TEXT,
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Cumulative Return (%)", color=_MUTED, fontsize=11)
    ax.legend(fontsize=11, facecolor="#21262d", labelcolor=_TEXT, framealpha=0.9)

    # ── Stats table ───────────────────────────────────────────────────────────
    ax_t = fig.add_axes([0.66, 0.08, 0.32, 0.84])
    ax_t.set_facecolor(_CARD_BG)
    ax_t.axis("off")
    for sp in ax_t.spines.values():
        sp.set_edgecolor(_BORDER)

    score_types  = list(results.keys())
    short_labels = [st.replace("_score", "").replace("_", " ").title() for st in score_types]
    n_cols = len(score_types)
    col_x  = [0.03] + [0.03 + (i + 1) * (0.94 / (n_cols + 1)) for i in range(n_cols)]

    ax_t.text(0.06, 0.97, "SLIDE 7 — FACTOR COMPARISON", color=_TITLE_CLR,
              fontsize=9.5, fontweight="bold", transform=ax_t.transAxes, va="top")

    y = 0.86
    # Header row
    ax_t.text(col_x[0], y, "Metric", color=_BLUE, fontsize=9, fontweight="bold",
              transform=ax_t.transAxes, va="top")
    for j, (lbl, st) in enumerate(zip(short_labels, score_types)):
        ax_t.text(col_x[j + 1], y, lbl,
                  color=_FACTOR_COLORS.get(st, _TEXT), fontsize=9, fontweight="bold",
                  transform=ax_t.transAxes, va="top")
    y -= 0.08

    # Build per-score-type stat dicts
    stat_rows_data = []
    for st in score_types:
        pnl_df, met = results[st]
        m = met["overall"]
        stat_rows_data.append({
            "Total Ret": m["total_ret"],
            "CAGR":      _cagr(pnl_df["cumulative_ret"]),
            "Sharpe":    m["sharpe"],
            "Max DD":    m["maxdd"],
            "Win Rate":  m["win_rate"],
            "IC Mean":   m["ic_mean"],
            "IC t-stat": m["ic_tstat"],
            "IC p-val":  m["ic_pval"],
        })

    stat_defs = [
        # (display_label, dict_key, fmt_str, is_pct, is_signed)
        ("Total Ret",  "Total Ret",  ".1f",  True,  True),
        ("CAGR",       "CAGR",       ".2f",  True,  True),
        ("Sharpe",     "Sharpe",     ".3f",  False, True),
        ("Max DD",     "Max DD",     ".1f",  True,  False),
        ("Win Rate",   "Win Rate",   ".1f",  True,  False),
        ("IC Mean",    "IC Mean",    ".4f",  False, True),
        ("IC t-stat",  "IC t-stat",  ".2f",  False, True),
        ("IC p-val",   "IC p-val",   ".4f",  False, False),
    ]

    dy = 0.088
    for stat_label, key, fmt_str, is_pct, is_signed in stat_defs:
        ax_t.text(col_x[0], y, f"{stat_label}:", color=_MUTED, fontsize=9,
                  transform=ax_t.transAxes, va="top")
        for j, srd in enumerate(stat_rows_data):
            val = srd[key]
            if pd.isna(val):
                num_str, vc = "N/A", _MUTED
            else:
                num_str = format(abs(val), fmt_str)
                if is_pct:
                    num_str += "%"
                if is_signed and val > 0:
                    num_str = "+" + num_str
                elif val < 0:
                    num_str = "\u2212" + num_str
                vc = (_GREEN if val > 0 else _RED) if is_signed else _TEXT
            ax_t.text(col_x[j + 1], y, num_str, color=vc, fontsize=9,
                      fontweight="bold", transform=ax_t.transAxes, va="top")
        y -= dy

    fig.suptitle("Factor Analytics \u2014 Multi-Factor Comparison",
                 color=_TEXT, fontsize=15, fontweight="bold", y=1.01)
    path = output_dir / "slide_07_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    print(f"   ✓ Slide 7 (comparison): {path}")


# ── Slide 8 — High Volatility Signal Decomposition ──────────────────────────
def _slide9_crisis_decomposition(regime_stats, pnl_df, output_dir):
    """Decompose signal quality failure during crisis/squeeze periods.
    
    Shows: IC mean, IC t-stat, decile spread, turnover, cross-sectional dispersion,
    and period returns for high-vol regimes vs normal regimes.
    
    Diagnoses: Is IC collapsing? Are returns negative despite positive IC?
    """
    if not regime_stats or len(regime_stats) == 0:
        print("   ⚠️  Slide 9 skipped — no regime data")
        return
    
    # Identify high-vol regimes
    high_vol_regimes = {k: v for k, v in regime_stats.items() if k in ["crisis", "squeeze"]}
    normal_regimes = {k: v for k, v in regime_stats.items() if k in ["normal", "grind_up", "slow_bleed"]}
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(_DARK_BG)
    
    # ── Left: Metrics table ───────────────────────────────────────────────────
    ax_t = fig.add_axes([0.05, 0.25, 0.40, 0.70])
    ax_t.set_facecolor(_CARD_BG)
    ax_t.axis("off")
    for sp in ax_t.spines.values():
        sp.set_edgecolor(_BORDER)
    
    ax_t.text(0.04, 0.98, "SLIDE 9 — CRISIS/SQUEEZE DECOMPOSITION", color=_TITLE_CLR,
              fontsize=10, fontweight="bold", transform=ax_t.transAxes, va="top")
    
    # Calculate regime day counts and frequencies
    total_days = len(pnl_df)
    regime_day_counts = {}
    for regime in regime_stats.keys():
        regime_day_counts[regime] = len(pnl_df[pnl_df["regime"] == regime])
    
    # Build table: regime name | Days % | IC mean | IC t-stat | D10-D1 | Turnover | CS Disp | Total Return %
    col_x = [0.02, 0.14, 0.25, 0.36, 0.47, 0.58, 0.69, 0.84]
    headers = ("Regime", "Days%", "IC Mean", "IC t-stat", "D10-D1%", "TO x", "CS Disp", "Ret%")
    
    y = 0.90
    for j, h in enumerate(headers):
        ax_t.text(col_x[j], y, h, color=_BLUE, fontsize=8, fontweight="bold",
                  transform=ax_t.transAxes, va="top")
    y -= 0.12
    
    # Sort regimes: high-vol first, then normal
    sorted_regimes = sorted(high_vol_regimes.keys()) + sorted(normal_regimes.keys())
    
    for regime in sorted_regimes:
        stats = regime_stats[regime]
        
        # Highlight high-vol rows
        is_highvol = regime in high_vol_regimes
        row_bg_alpha = 0.15 if is_highvol else 0.0
        
        if is_highvol:
            ax_t.add_patch(plt.Rectangle((0.01, y - 0.045), 0.98, 0.048,
                                        transform=ax_t.transAxes, facecolor=_RED,
                                        alpha=row_bg_alpha, zorder=-1))
        
        # Regime name
        ax_t.text(col_x[0], y, regime[:12], color=_TEXT, fontsize=8,
                  transform=ax_t.transAxes, va="top")
        
        # Days percentage
        days_in_regime = regime_day_counts[regime]
        pct_days = (days_in_regime / total_days * 100) if total_days > 0 else 0.0
        ax_t.text(col_x[1], y, f"{pct_days:.1f}%", color=_TEXT, fontsize=7,
                  transform=ax_t.transAxes, va="top")
        
        # IC Mean
        ic_m = stats["ic_mean"]
        ic_c = _GREEN if ic_m > 0 else _RED
        ax_t.text(col_x[2], y, f"{ic_m:+.4f}", color=ic_c, fontsize=8, fontweight="bold",
                  transform=ax_t.transAxes, va="top")
        
        # IC t-stat
        ic_t = stats["ic_tstat"]
        ic_t_c = _GREEN if ic_t > 1.96 else (_RED if ic_t < -1.96 else _MUTED)
        ax_t.text(col_x[3], y, f"{ic_t:+.2f}", color=ic_t_c, fontsize=8, fontweight="bold",
                  transform=ax_t.transAxes, va="top")
        
        # Decile spread (D10-D1)
        ds = stats["decile_spread_mean"]
        ds_c = _GREEN if ds > 0 else _RED
        ax_t.text(col_x[4], y, f"{ds:+.2f}", color=ds_c, fontsize=8, fontweight="bold",
                  transform=ax_t.transAxes, va="top")
        
        # Turnover
        to = stats["turnover_mean"]
        ax_t.text(col_x[5], y, f"{to:.2f}" if pd.notna(to) else "N/A", color=_TEXT, fontsize=8,
                  transform=ax_t.transAxes, va="top")
        
        # Cross-sectional dispersion
        cs = stats["cs_disp_mean"]
        ax_t.text(col_x[6], y, f"{cs:.2f}", color=_TEXT, fontsize=8,
                  transform=ax_t.transAxes, va="top")
        
        # Total regime return (%)
        pr = stats["total_regime_return_pct"]
        pr_c = _GREEN if pr > 0 else _RED
        ax_t.text(col_x[7], y, f"{pr:+.1f}%" if pd.notna(pr) else "N/A", color=pr_c, fontsize=8, fontweight="bold",
                  transform=ax_t.transAxes, va="top")
        
        y -= 0.10
    
    # Add legend: signal vs execution failure
    y -= 0.05
    ax_t.text(0.04, y, "Diagnosis Guide:", color=_MUTED, fontsize=8, fontweight="bold",
              transform=ax_t.transAxes, va="top", style="italic")
    y -= 0.08
    ax_t.text(0.04, y, "IC → 0, Return < 0: SIGNAL FAILURE", color=_RED, fontsize=7,
              transform=ax_t.transAxes, va="top")
    y -= 0.07
    ax_t.text(0.04, y, "IC > 0, Return < 0: EXECUTION/EXPOSURE FAILURE", color=_YELLOW, fontsize=7,
              transform=ax_t.transAxes, va="top")
    y -= 0.07
    ax_t.text(0.04, y, "IC > 0, Return > 0: SIGNAL WORKING", color=_GREEN, fontsize=7,
              transform=ax_t.transAxes, va="top")
    
    # ── Long/Short decomposition (below main table) ────────────────────────────
    y -= 0.15
    ax_t.text(0.04, y, "Long vs Short:", color=_MUTED, fontsize=8, fontweight="bold",
              transform=ax_t.transAxes, va="top", style="italic")
    y -= 0.08
    
    col_x_legs = [0.04, 0.20, 0.40]
    headers_legs = ("Regime", "Long %", "Short %")
    for j, h in enumerate(headers_legs):
        ax_t.text(col_x_legs[j], y, h, color=_BLUE, fontsize=7, fontweight="bold",
                  transform=ax_t.transAxes, va="top")
    y -= 0.07
    
    for regime in sorted_regimes:
        stats = regime_stats[regime]
        
        # Regime name
        ax_t.text(col_x_legs[0], y, regime[:8], color=_TEXT, fontsize=7,
                  transform=ax_t.transAxes, va="top")
        
        short_r = stats.get("short_regime_return_pct", np.nan)
        short_c = _GREEN if short_r > 0 else _RED
        ax_t.text(col_x_legs[2], y, f"{short_r:+.1f}" if pd.notna(short_r) else "N/A", color=short_c, fontsize=7, fontweight="bold",
                    transform=ax_t.transAxes, va="top")

        long_r = stats.get("long_regime_return_pct", np.nan)
        long_c = _GREEN if long_r > 0 else _RED
        ax_t.text(col_x_legs[1], y, f"{long_r:+.1f}" if pd.notna(long_r) else "N/A", color=long_c, fontsize=7, fontweight="bold",
                    transform=ax_t.transAxes, va="top")
        
        y -= 0.07
    
    ax_s = fig.add_axes([0.55, 0.25, 0.40, 0.70])
    _dark_ax(ax_s)
    
    for regime in sorted_regimes:
        stats = regime_stats[regime]
        is_highvol = regime in high_vol_regimes
        
        color = _RED if is_highvol else _BLUE
        marker = "X" if is_highvol else "o"
        size = 200 if is_highvol else 100
        
        ax_s.scatter(stats["ic_mean"], stats["total_regime_return_pct"],
                    color=color, s=size, alpha=0.7, marker=marker, edgecolors=_TEXT, linewidth=1.5,
                    label=regime)
    
    ax_s.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--", alpha=0.5)
    ax_s.axvline(0, color=_MUTED, linewidth=0.8, linestyle="--", alpha=0.5)
    ax_s.set_xlabel("IC Mean (Signal Quality)", color=_MUTED, fontsize=11)
    ax_s.set_ylabel("Total Regime Return (%)", color=_MUTED, fontsize=11)
    ax_s.set_title("IC vs Total Return by Regime\n(X=Crisis/Squeeze, o=Normal)", color=_TEXT,
                   fontsize=11, fontweight="bold", pad=10)
    ax_s.legend(fontsize=8, facecolor="#21262d", labelcolor=_TEXT, loc="best")
    
    # Quadrant labels
    ax_s.text(0.95, 0.95, "Good Signal\nGood Returns", transform=ax_s.transAxes,
             fontsize=9, color=_GREEN, ha="right", va="top", style="italic", alpha=0.6)
    ax_s.text(0.05, 0.05, "Bad Signal\nBad Returns", transform=ax_s.transAxes,
             fontsize=9, color=_RED, ha="left", va="bottom", style="italic", alpha=0.6)
    ax_s.text(0.95, 0.05, "Good Signal\nBad Returns\n(Execution)", transform=ax_s.transAxes,
             fontsize=9, color=_YELLOW, ha="right", va="bottom", style="italic", alpha=0.6)
    
    fig.suptitle("Factor Analytics — Crisis/Squeeze Signal Decomposition",
                 color=_TEXT, fontsize=15, fontweight="bold", y=0.98)
    path = output_dir / "slide_08_crisis_decomposition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    print(f"   ✓ Slide 8: {path}")


def plot_output_deck(
    pnl_df,
    decile_df,
    metrics,
    all_data,
    rebalance_dates,
    spy_df,
    output_dir,
    score_type="direction_score",
    membership_lookup: RussellMembershipLookup | None = None,
):
    """Generate the 9-slide clean output deck and save PNGs to output_dir."""
    print(f"\n🎨 Generating Output Deck (9 slides) [{score_type}]...")
    ic_series = compute_ic_series(
        all_data,
        rebalance_dates,
        score_type,
        membership_lookup=membership_lookup,
    )
    regime_stats = compute_regime_signal_decomposition(
        all_data,
        pnl_df,
        rebalance_dates,
        score_type,
        membership_lookup=membership_lookup,
    )
    
    _slide1_factor_summary(metrics, all_data, ic_series, output_dir, score_type)
    _slide2_cumulative_performance(pnl_df, output_dir)
    _slide3_decile_spread(decile_df, output_dir)
    _slide4_long_short_breakdown(pnl_df, spy_df, output_dir)
    _slide5_rolling_stability(pnl_df, ic_series, output_dir)
    _slide6_cost_sensitivity(pnl_df, output_dir)
    _slide7_long_only_summary(pnl_df, metrics, all_data, ic_series, output_dir, score_type)
    _slide8_short_only_summary(pnl_df, metrics, all_data, ic_series, output_dir, score_type)
    _slide9_crisis_decomposition(regime_stats, pnl_df, output_dir)
    print(f"   ✓ All 9 slides saved → {output_dir}\n")


# ----------------------------
# Per-Score Backtest Runner
# ----------------------------

def run_backtest(
    score_type,
    all_data,
    spy_df,
    regime_map,
    rebalance_dates,
    output_dir,
    config=None,
    trading_dates=None,
    skip_plots=False,
    candidate_pool_config: CandidatePoolConfig | None = None,
    membership_lookup: RussellMembershipLookup | None = None,
    periodic_static: bool = False,
):
    """Run the full factor backtest pipeline for one score type.

    Saves an independent 9-slide deck + CSVs to output_dir/<score_type>/.
    Returns (pnl_df, metrics, decile_df) for cross-factor comparison.
    """
    subdir = output_dir / score_type
    subdir.mkdir(parents=True, exist_ok=True)
    divider = "─" * 70

    print(f"\n{divider}")
    print(f"  Backtest: {score_type.upper().replace('_', ' ')}")
    print(divider)

    portfolio_result = build_factor_portfolio(
        all_data,
        spy_df,
        rebalance_dates,
        score_type,
        regime_map,
        config=config,
        trading_dates=trading_dates,
        candidate_pool_config=candidate_pool_config,
        membership_lookup=membership_lookup,
        periodic_static=periodic_static,
    )
    pnl_df = portfolio_result["pnl_df"]
    review_log_df = portfolio_result["review_log_df"]
    trade_log_df = portfolio_result["trade_log_df"]
    holdings_log_df = portfolio_result["holdings_log_df"]
    portfolio_selections_df = portfolio_result["portfolio_selections_df"]
    review_signal_dates = _review_signal_dates(review_log_df)

    pnl_df, metrics = compute_metrics(
        pnl_df,
        regime_map,
        all_data,
        review_signal_dates,
        score_type,
        review_log_df=review_log_df,
        trade_log_df=trade_log_df,
        holdings_log_df=holdings_log_df,
        membership_lookup=membership_lookup,
    )
    decile_df = compute_decile_returns(
        all_data,
        review_signal_dates,
        score_type,
        membership_lookup=membership_lookup,
    )

    print_summary(metrics, decile_df)

    save_portfolio_selections(portfolio_selections_df, subdir)

    # CSVs
    pnl_df.to_csv(subdir / "factor_pnl.csv", index=False)
    decile_df.to_csv(subdir / "decile_analysis.csv", index=False)
    review_log_df.to_csv(subdir / "review_log.csv", index=False)
    trade_log_df.to_csv(subdir / "trade_log.csv", index=False)
    holdings_log_df.to_csv(subdir / "holdings_log.csv", index=False)
    print(f"   \u2713 CSVs saved \u2192 {subdir}")
    
    if skip_plots:
        print("   ✓ Skipped plots (--skip-plots)")
    else:
        # 8-slide output deck
        plot_output_deck(
            pnl_df,
            decile_df,
            metrics,
            all_data,
            review_signal_dates,
            spy_df,
            subdir,
            score_type,
            membership_lookup=membership_lookup,
        )

    return pnl_df, metrics, decile_df


def _pop_flag(argv, flag, cast=str, default=None):
    if flag not in argv:
        return default, argv
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        raise ValueError(f"{flag} requires a value")
    value = cast(argv[idx + 1])
    argv = argv[:idx] + argv[idx + 2:]
    return value, argv


def _parse_score_types(value: str) -> list[str]:
    score_types = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not score_types:
        raise ValueError("--score-types requires at least one score type")
    return score_types


def main():
    """Parse CLI arguments and run backtest.
    
    Usage:
        python factor_backtest_engine.py [forward_days] [start_date] [end_date] [ticker1] [ticker2] ...
    
    Examples:
        # Run 5-day forward horizon
        python factor_backtest_engine.py 5
        
        # Run 20-day forward horizon for specific date range
        python factor_backtest_engine.py 20 2023-01-01 2024-12-31
        
        # Run 10-day horizon for specific tickers
        python factor_backtest_engine.py 10 AAPL MSFT GOOGL
        
        # Run 15-day horizon for date range + specific tickers
        python factor_backtest_engine.py 15 2023-01-01 2024-12-31 AAPL MSFT
    """
    global FORWARD_DAYS, REBALANCE_DAYS, COMPOSITE_SCORES_DIR, TOP_N, BOTTOM_N, SCORE_TYPES, COST_RATE
    
    # Parse command-line arguments
    forward_days = None
    start_date = None
    end_date = None
    ticker_filter = None
    scores_dir_override = None
    review_mode = "static"
    review_days_override = None
    hysteresis_mode = "none"
    variant_label = None
    skip_plots = False
    long_retain_buffer = 15
    short_retain_buffer = 10
    replacement_gap = 5
    weighting_mode = "equal"
    book_sizing_mode = "fixed"
    cost_bps = COST_RATE * 10000.0
    periodic_static = False
    min_weight_delta = None
    candidate_pool_mode = "raw"
    candidate_top_n_magnitude_per_side = 60
    candidate_direction_threshold = 5.0
    top_n = TOP_N
    bottom_n = BOTTOM_N
    score_types = list(SCORE_TYPES)
    regime_intervals = DEFAULT_REGIME_REVIEW_INTERVALS.copy()
    
    # Extract --scores-dir flag before positional parsing
    argv = list(sys.argv[1:])
    if "--skip-plots" in argv:
        skip_plots = True
        argv = [arg for arg in argv if arg != "--skip-plots"]
    if "--periodic-static" in argv:
        periodic_static = True
        argv = [arg for arg in argv if arg != "--periodic-static"]
    try:
        scores_dir_override, argv = _pop_flag(argv, "--scores-dir", str, None)
        review_mode, argv = _pop_flag(argv, "--review-mode", str, review_mode)
        review_days_override, argv = _pop_flag(argv, "--review-days", int, review_days_override)
        hysteresis_mode, argv = _pop_flag(argv, "--hysteresis-mode", str, hysteresis_mode)
        variant_label, argv = _pop_flag(argv, "--variant-label", str, variant_label)
        long_retain_buffer, argv = _pop_flag(argv, "--long-retain-buffer", int, long_retain_buffer)
        short_retain_buffer, argv = _pop_flag(argv, "--short-retain-buffer", int, short_retain_buffer)
        replacement_gap, argv = _pop_flag(argv, "--replacement-gap", int, replacement_gap)
        weighting_mode, argv = _pop_flag(argv, "--weighting-mode", str, weighting_mode)
        book_sizing_mode, argv = _pop_flag(argv, "--book-sizing-mode", str, book_sizing_mode)
        cost_bps, argv = _pop_flag(argv, "--cost-bps", float, cost_bps)
        top_n, argv = _pop_flag(argv, "--top-n", int, top_n)
        bottom_n, argv = _pop_flag(argv, "--bottom-n", int, bottom_n)
        score_types, argv = _pop_flag(argv, "--score-types", _parse_score_types, score_types)
        min_weight_delta, argv = _pop_flag(argv, "--min-weight-delta", float, min_weight_delta)
        candidate_pool_mode, argv = _pop_flag(argv, "--candidate-pool-mode", str, candidate_pool_mode)
        candidate_top_n_magnitude_per_side, argv = _pop_flag(
            argv,
            "--candidate-top-n-magnitude",
            int,
            candidate_top_n_magnitude_per_side,
        )
        candidate_direction_threshold, argv = _pop_flag(
            argv,
            "--candidate-direction-threshold",
            float,
            candidate_direction_threshold,
        )
        regime_intervals["normal"], argv = _pop_flag(argv, "--review-normal-days", int, regime_intervals["normal"])
        regime_intervals["grind_up"], argv = _pop_flag(argv, "--review-grind-up-days", int, regime_intervals["grind_up"])
        regime_intervals["slow_bleed"], argv = _pop_flag(argv, "--review-slow-bleed-days", int, regime_intervals["slow_bleed"])
        regime_intervals["squeeze"], argv = _pop_flag(argv, "--review-squeeze-days", int, regime_intervals["squeeze"])
        regime_intervals["crisis"], argv = _pop_flag(argv, "--review-crisis-days", int, regime_intervals["crisis"])
        regime_intervals["unknown"], argv = _pop_flag(argv, "--review-unknown-days", int, regime_intervals["unknown"])
    except ValueError as exc:
        print(f"❌ {exc}")
        return
    
    if scores_dir_override:
        COMPOSITE_SCORES_DIR = Path(scores_dir_override)
        print(f"📂 Using custom scores directory: {COMPOSITE_SCORES_DIR}")

    unsupported_args = [arg for arg in argv if str(arg).startswith("-")]
    if unsupported_args:
        print(f"❌ Unsupported argument(s): {', '.join(unsupported_args)}")
        return
    
    argc = len(argv) + 1  # +1 to match original sys.argv indexing
    sys_argv_compat = ['prog'] + argv  # fake argv for index math below
    
    if len(argv) > 0:
        # First arg: try to parse as forward_days (integer)
        try:
            forward_days = int(sys_argv_compat[1])
            FORWARD_DAYS = forward_days
            REBALANCE_DAYS = forward_days  # Rebalance frequency matches forward horizon
            idx = 2
        except ValueError:
            # Not an integer, treat as start_date
            start_date = sys_argv_compat[1]
            idx = 2
        
        # Next args: check for up to two dates (start_date, end_date)
        if argc > idx:
            arg = sys_argv_compat[idx]
            if '-' in arg or '/' in arg:
                start_date = arg
                idx += 1
        
        if argc > idx:
            arg = sys_argv_compat[idx]
            if '-' in arg or '/' in arg:
                end_date = arg
                idx += 1
        
        # Remaining args are tickers
        if argc > idx:
            ticker_filter = sys_argv_compat[idx:]
    
    start_date_ts = pd.Timestamp(start_date) if start_date else None
    end_date_ts = pd.Timestamp(end_date) if end_date else None
    if review_mode not in {"static", "adaptive"}:
        print(f"❌ Unsupported review mode: {review_mode}")
        return
    if hysteresis_mode not in {"none", "rank_band_symmetric", "rank_band_asymmetric", "replacement_gap", "combined"}:
        print(f"❌ Unsupported hysteresis mode: {hysteresis_mode}")
        return
    if weighting_mode not in {"equal", "rank"}:
        print(f"❌ Unsupported weighting mode: {weighting_mode}")
        return
    if book_sizing_mode not in {"fixed", "decile"}:
        print(f"❌ Unsupported book sizing mode: {book_sizing_mode}")
        return
    if periodic_static and review_mode != "static":
        print("❌ --periodic-static currently supports --review-mode static only")
        return
    if cost_bps < 0:
        print("❌ --cost-bps must be non-negative")
        return
    if top_n <= 0 or bottom_n <= 0:
        print("❌ --top-n and --bottom-n must be positive")
        return
    allowed_score_types = {"direction_score", "magnitude_score"}
    invalid_score_types = [score_type for score_type in score_types if score_type not in allowed_score_types]
    if invalid_score_types:
        print(f"❌ Unsupported score type(s): {', '.join(invalid_score_types)}")
        return
    if candidate_pool_mode not in {"raw", "llm_mirror", "llm_mirror_fill_raw"}:
        print(f"❌ Unsupported candidate pool mode: {candidate_pool_mode}")
        return
    TOP_N = top_n
    BOTTOM_N = bottom_n
    SCORE_TYPES = list(score_types)
    COST_RATE = cost_bps / 10000.0
    static_review_days = review_days_override if review_days_override is not None else REBALANCE_DAYS
    if min_weight_delta is None and hysteresis_mode == "combined":
        min_weight_delta = 0.5 / top_n
    candidate_pool_config = CandidatePoolConfig(
        mode=candidate_pool_mode,
        direction_threshold=candidate_direction_threshold,
        top_n_magnitude_per_side=candidate_top_n_magnitude_per_side,
    )

    config = PortfolioConstructionConfig(
        review_policy=ReviewPolicy(
            mode=review_mode,
            static_review_days=static_review_days,
            regime_intervals=regime_intervals,
            scheduled_signal_dates=frozenset(),
        ),
        hysteresis_policy=HysteresisPolicy(
            mode=hysteresis_mode,
            top_n=top_n,
            bottom_n=bottom_n,
            long_retain_buffer=long_retain_buffer,
            short_retain_buffer=short_retain_buffer,
            replacement_gap=replacement_gap,
            min_weight_delta=min_weight_delta,
        ),
        top_n=top_n,
        bottom_n=bottom_n,
        weighting_mode=weighting_mode,
        book_sizing_mode=book_sizing_mode,
    )
    
    print(f"\n{'='*80}")
    print(f"FACTOR BACKTEST ENGINE")
    if book_sizing_mode == "fixed":
        strategy_label = f"Top {top_n} Long / Bottom {bottom_n} Short"
    else:
        strategy_label = "Top Decile Long / Bottom Decile Short"
    print(f"Strategy: {strategy_label}, Review Policy {review_mode} / Hysteresis {hysteresis_mode}")
    print(f"Forward Horizon: {FORWARD_DAYS} days")
    print(f"Review Days: {static_review_days}")
    if review_mode == "adaptive":
        print(f"Adaptive Regime Days: {regime_intervals}")
    if hysteresis_mode != "none":
        print(
            "Hysteresis Params: "
            f"long_buffer={long_retain_buffer}, short_buffer={short_retain_buffer}, "
            f"replacement_gap={replacement_gap}"
        )
    print(f"Candidate Pool: {candidate_pool_config.describe()}")
    print(f"Weighting Mode: {weighting_mode}")
    print(f"Book Sizing Mode: {book_sizing_mode}")
    print(f"Transaction Cost: {cost_bps:g} bps one-way")
    print(f"Periodic Static: {'enabled' if periodic_static else 'disabled'}")
    print(f"Min entry price: > ${MIN_TRADE_PRICE:.2f}")
    print(f"Score Types: {', '.join(score_types)}")
    if start_date_ts or end_date_ts:
        date_range = f"{start_date_ts or 'start'} to {end_date_ts or 'end'}"
        print(f"Trading Period: {date_range}")
    if ticker_filter:
        print(f"Universe: {', '.join(ticker_filter)}")
    if scores_dir_override:
        print(f"Scores Directory: {COMPOSITE_SCORES_DIR}")
    print(f"{'='*80}\n")

    # ── Load composite score data (FULL HISTORY, optionally filtered by ticker) ──
    all_data = load_all_composites(ticker_filter=ticker_filter)
    if not all_data:
        print("❌ No data loaded")
        return
    
    print(f"\n✓ Loaded {len(all_data)} stocks with full signal history\n")
    membership_lookup = None
    if _looks_like_russell_dataset(all_data):
        try:
            membership_lookup = load_russell_membership_lookup()
            print(
                "✓ Russell membership gating enabled "
                f"({Path(membership_lookup.path).name})\n"
            )
        except Exception as exc:
            print(f"⚠️  Russell membership gating unavailable: {exc}\n")

    # ── Load SPY for regime identification (FULL HISTORY) ────────────────────
    print("\n� Loading SPY data for regime identification...")
    spy_df = load_spy_data()
    if spy_df is None:
        print("❌ SPY data unavailable")
        return

    # Shared calendar and regime map (identical for all score types)
    all_dates = sorted(pd.to_datetime(spy_df["date"]).unique())
    trading_dates = _filter_trading_dates(all_dates, start_date_ts, end_date_ts)
    _, rebalance_dates = get_rebalance_dates(spy_df["date"], static_review_days)
    regime_map = identify_regimes(spy_df)
    
    if start_date_ts or end_date_ts:
        print("Filtering trading dates by range...")
        rebalance_dates = [
            pd.Timestamp(rd)
            for rd in rebalance_dates
            if (start_date_ts is None or pd.Timestamp(rd) >= start_date_ts)
            and (end_date_ts is None or pd.Timestamp(rd) <= end_date_ts)
        ]
        print(f"   {len(trading_dates)} trading dates in trading period")
        print(f"   {len(rebalance_dates)} scheduled static review dates in trading period\n")

    config = PortfolioConstructionConfig(
        review_policy=ReviewPolicy(
            mode=review_mode,
            static_review_days=static_review_days,
            regime_intervals=regime_intervals,
            scheduled_signal_dates=frozenset(pd.to_datetime(rebalance_dates)),
        ),
        hysteresis_policy=config.hysteresis_policy,
        top_n=top_n,
        bottom_n=bottom_n,
        weighting_mode=config.weighting_mode,
        book_sizing_mode=config.book_sizing_mode,
    )

    # ── Run independent backtest per score type ────────────────────────────────
    # Create output directory with forward days in the path
    # If using custom scores dir (e.g., LLM), add a label to avoid overwriting deterministic results
    nonbaseline_policy = (
        review_mode != "static"
        or static_review_days != REBALANCE_DAYS
        or hysteresis_mode != "none"
        or book_sizing_mode != "fixed"
    )
    nondefault_candidate_pool = not candidate_pool_config.is_default()

    if scores_dir_override and not variant_label and not nonbaseline_policy and not nondefault_candidate_pool:
        scores_label = Path(scores_dir_override).name  # e.g., "composite_scores_llm"
        OUTPUT_DIR = OUTPUT_DIR_BASE / f"{scores_label}_{FORWARD_DAYS}d_horizon"
    elif variant_label:
        OUTPUT_DIR = OUTPUT_DIR_BASE / variant_label / f"{FORWARD_DAYS}d_horizon"
    elif nonbaseline_policy or nondefault_candidate_pool:
        label_parts = []
        if nonbaseline_policy:
            label_parts.append(config.variant_label())
        if nondefault_candidate_pool:
            label_parts.append(candidate_pool_config.label_suffix())
        OUTPUT_DIR = OUTPUT_DIR_BASE / "__".join(label_parts) / f"{FORWARD_DAYS}d_horizon"
    else:
        OUTPUT_DIR = OUTPUT_DIR_BASE / f"{FORWARD_DAYS}d_horizon"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "forward_days": FORWARD_DAYS,
        "rebalance_days": REBALANCE_DAYS,
        "start_date": start_date_ts.strftime("%Y-%m-%d") if start_date_ts is not None else None,
        "end_date": end_date_ts.strftime("%Y-%m-%d") if end_date_ts is not None else None,
        "score_types": list(score_types),
        "top_n": top_n,
        "bottom_n": bottom_n,
        "book_sizing_mode": book_sizing_mode,
        "min_trade_price": MIN_TRADE_PRICE,
        "cost_rate": COST_RATE,
        "cost_bps": cost_bps,
        "scores_dir": str(COMPOSITE_SCORES_DIR),
        "scores_dir_override": str(scores_dir_override) if scores_dir_override else None,
        "variant_label": variant_label,
        "output_dir": str(OUTPUT_DIR),
        "skip_plots": bool(skip_plots),
        "periodic_static": bool(periodic_static),
        "ticker_filter": list(ticker_filter) if ticker_filter else [],
        "review_policy": {
            "mode": review_mode,
            "static_review_days": static_review_days,
            "regime_intervals": {k: int(v) for k, v in regime_intervals.items()},
        },
        "hysteresis_policy": {
            "mode": hysteresis_mode,
            "long_retain_buffer": long_retain_buffer,
            "short_retain_buffer": short_retain_buffer,
            "replacement_gap": replacement_gap,
            "min_weight_delta": min_weight_delta,
        },
        "candidate_pool": {
            "mode": candidate_pool_config.mode,
            "direction_threshold": candidate_pool_config.direction_threshold,
            "top_n_magnitude_per_side": candidate_pool_config.top_n_magnitude_per_side,
        },
        "weighting_mode": weighting_mode,
        "membership_gating_enabled": bool(membership_lookup is not None),
        "membership_path": membership_lookup.path if membership_lookup is not None else None,
    }
    with (OUTPUT_DIR / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(run_manifest, fh, indent=2, sort_keys=True)

    results = {}  # {score_type: (pnl_df, metrics)}

    for score_type in score_types:
        pnl_df, metrics, _ = run_backtest(
            score_type, all_data, spy_df, regime_map, rebalance_dates, OUTPUT_DIR,
            config=config, trading_dates=trading_dates, skip_plots=skip_plots,
            candidate_pool_config=candidate_pool_config,
            membership_lookup=membership_lookup,
            periodic_static=periodic_static,
        )
        results[score_type] = (pnl_df, metrics)

    # ── Cross-factor comparison slide (saved at root output dir) ──────────────
    if skip_plots:
        print("\n🎨 Skipping comparison slide (--skip-plots)")
    else:
        print(f"\n🎨 Generating cross-factor comparison slide...")
        _slide7_comparison(results, spy_df, OUTPUT_DIR)

    print(f"\n{'='*80}")
    print("FACTOR BACKTEST COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}/")
    for st in score_types:
        if skip_plots:
            print(f"  └── {st}/  (CSVs only)")
        else:
            print(f"  └── {st}/  (9-slide deck + CSVs)")
    if not skip_plots:
        print(f"  └── slide_07_comparison.png")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
