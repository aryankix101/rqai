"""
THIS BUILDS EQUAL WEIGHT PORTFOLIO
Factor Backtest Engine — Test composite scores as systematic return factors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
import scipy.stats
import warnings
import sys

from hit_rate_validator import CANONICAL_EXECUTION_MODEL, get_trade_window
from portfolio_constraints import MIN_TRADE_PRICE, has_min_trade_price

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

# ----------------------------
# Config
# ----------------------------
COMPOSITE_SCORES_DIR = Path("backtest_results/composite_scores")
OUTPUT_DIR_BASE = Path("backtest_results/factor_backtest")

# Factor construction
FORWARD_DAYS = 10 
REBALANCE_DAYS = FORWARD_DAYS  
TOP_N = 30
BOTTOM_N = 30
COST_RATE = 0.0015  # 15 bps one-way cost (commission + spread + slippage per trade)
                     # Turnover = sum(|w_new - w_old|) = 2x full notional traded (long-short)
SCORE_TYPES = ["direction_score", "magnitude_score"]


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
    
    for csv_file in csv_files:
        try:
            ticker = csv_file.stem.split("_")[2]
            df = pd.read_csv(csv_file, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            missing_price_cols = {"adj_open", "adj_close"} - set(df.columns)
            if missing_price_cols:
                raise ValueError(
                    f"missing required columns {sorted(missing_price_cols)}; "
                    "re-run composite_score_calculator.py with the updated signal dump"
                )
            results[ticker] = df
            print(f"   ✓ {ticker:6s} - {len(df)} records")
        except Exception as e:
            print(f"   ❌ {csv_file.name}: {e}")
    
    return results


def load_spy_data(spy_path="backtest_results/signal_data_dump/signal_data_dump_SPY_1993-01-01_2025-12-31.csv"):
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


def _cross_sectional_snapshot(all_data, signal_date, score_type="direction_score"):
    """Scores and realized forward returns from a single signal date."""
    ret_col = _ic_target(score_type)
    rows = []

    for ticker, df in all_data.items():
        row = df[df["date"] == signal_date]
        if row.empty or ret_col not in row.columns:
            continue

        row = row.iloc[0]
        score = row.get(score_type, np.nan)
        ret_pct = row.get(ret_col, np.nan)
        if pd.notna(score) and pd.notna(ret_pct):
            rows.append({
                "ticker": ticker,
                "score": float(score),
                "ret": float(ret_pct) / 100.0,
            })

    return pd.DataFrame(rows)


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

def build_factor_portfolio(all_data, spy_df, rebalance_dates, score_type="direction_score", regime_map=None):
    """Build daily returns from top n / bottom n equal weight market-neutral portfolio
    
    Canonical execution mechanics:
    - Signal is observed at the rebalance date close
    - Portfolio is entered/exited at the NEXT trading day's open
    - PnL accrues open-to-open
    - Transaction cost = COST_RATE (one-way) * turnover on each rebalance open
    - Only stocks with execution prices above the minimum trade price are eligible
    """
    
    pnl_records = []
    
    print(f"\n📊 Building factor portfolio...")
    print(f"   Long leg : Top    {TOP_N}")
    print(f"   Short leg: Bottom {BOTTOM_N}")
    print(f"   Gross exposure: 2.0x (long-short market-neutral)")
    print(f"   Rebalance period: {REBALANCE_DAYS} days")
    print(f"   Cost model: {COST_RATE*10000:.0f} bps one-way")
    print(f"   Min entry price: > ${MIN_TRADE_PRICE:.2f}")
    print(f"   Score type: {score_type}")
    print(f"   Rebalance dates: {len(rebalance_dates)}\n")
    
    # Normalize all dates to pd.Timestamp
    for ticker in all_data:
        all_data[ticker]["date"] = pd.to_datetime(all_data[ticker]["date"])
    spy_df["date"] = pd.to_datetime(spy_df["date"])
    
    # Master calendar from SPY
    calendar_dates = [pd.Timestamp(d) for d in sorted(pd.to_datetime(spy_df["date"].values).unique())]
    execution_schedule = _build_execution_schedule(calendar_dates, rebalance_dates)
    if not execution_schedule:
        return pd.DataFrame(columns=[
            "date", "daily_ret_bps", "long_pnl_bps", "short_pnl_bps",
            "num_longs", "num_shorts", "cost_bps_applied", "rebalance_signal_date",
        ])

    first_execution = min(execution_schedule.keys())
    all_dates = [d for d in calendar_dates if d >= first_execution]

    # Pre-align opens to the SPY calendar.
    # Forward-fill handles non-trading stock days as stale-open / zero-return days.
    print(f"   Execution model: {CANONICAL_EXECUTION_MODEL} (signal close -> next open)")
    print("   Pre-aligning opens to SPY calendar (forward-fill)...")
    price_matrix = {}  # ticker -> pd.Series indexed by SPY dates
    score_matrix = {}  # ticker -> raw scores indexed by signal date
    for ticker, df in all_data.items():
        df_indexed = df.set_index("date")
        price_matrix[ticker] = df_indexed["adj_open"].reindex(all_dates).ffill()
        score_matrix[ticker] = df_indexed[score_type]
    
    positions = {}   # ticker -> {'weight': float, 'last_price': float}
    old_weights = {} # previous period weights for turnover calc
    
    for trade_date in all_dates:
        long_pnl_bps = 0.0
        short_pnl_bps = 0.0
        cost_today    = 0.0
        num_longs = 0
        num_shorts = 0
        
        # ── Step 1: Mark-to-market existing positions open-to-open ───────────
        for ticker, pos_info in list(positions.items()):
            current_price = price_matrix[ticker].get(trade_date, np.nan) if ticker in price_matrix else np.nan
            last_price = pos_info["last_price"]
            weight = pos_info["weight"]
            
            if pd.notna(current_price) and current_price > 0 and last_price > 0:
                daily_ret = (current_price / last_price - 1.0)
                daily_pnl_component = weight * daily_ret * 10000
                
                if weight > 0:
                    long_pnl_bps += daily_pnl_component
                else:
                    short_pnl_bps += daily_pnl_component
                
                positions[ticker]["last_price"] = current_price
            # NaN price (pre-IPO): forward-fill handles this; 0 contribution
            
        # ── Step 2: Rebalance at today's open using prior-close signals ──────
        signal_date = execution_schedule.get(trade_date)
        if signal_date is not None:
            date_scores = {}
            for ticker in all_data:
                score = score_matrix[ticker].get(signal_date, np.nan) if ticker in score_matrix else np.nan
                price = price_matrix[ticker].get(trade_date, np.nan) if ticker in price_matrix else np.nan
                if pd.notna(score) and has_min_trade_price(price):
                    date_scores[ticker] = (score, price)
            
            if len(date_scores) >= TOP_N + BOTTOM_N:
                sorted_tickers = sorted(date_scores.keys(), key=lambda t: date_scores[t][0], reverse=True)
                
                # Simple long-short construction: top N long, bottom N short
                new_weights = {}
                # Long positions (top N)
                for i in range(TOP_N):
                    new_weights[sorted_tickers[i]] = 1.0 / TOP_N
                # Short positions (bottom N)
                for i in range(BOTTOM_N):
                    new_weights[sorted_tickers[-(i + 1)]] = -1.0 / BOTTOM_N
                
                # Turnover and transaction cost — per-leg attribution
                all_tickers_in_turnover = set(old_weights.keys()) | set(new_weights.keys())
                turnover = 0.0
                long_turnover = 0.0
                short_turnover = 0.0
                for t in all_tickers_in_turnover:
                    nw = new_weights.get(t, 0.0)
                    ow = old_weights.get(t, 0.0)
                    delta = abs(nw - ow)
                    turnover += delta
                    if nw >= 0 and ow >= 0:
                        # Stays in long leg (or enters/exits long)
                        long_turnover += delta
                    elif nw <= 0 and ow <= 0:
                        # Stays in short leg (or enters/exits short)
                        short_turnover += delta
                    elif ow > 0 and nw < 0:
                        # Flips long → short: split at zero boundary
                        long_turnover += ow        # closing the long
                        short_turnover += abs(nw)   # opening the short
                    else:
                        # Flips short → long
                        short_turnover += abs(ow)   # closing the short
                        long_turnover += nw          # opening the long

                long_cost_bps  = COST_RATE * long_turnover * 10000
                short_cost_bps = COST_RATE * short_turnover * 10000
                long_pnl_bps  -= long_cost_bps
                short_pnl_bps -= short_cost_bps
                cost_today = long_cost_bps + short_cost_bps
                
                # Reset positions at today's open; accrual resumes from this open.
                positions = {}
                for ticker, weight in new_weights.items():
                    price = price_matrix[ticker].get(trade_date, np.nan) if ticker in price_matrix else np.nan
                    if has_min_trade_price(price):
                        positions[ticker] = {"weight": weight, "last_price": price}
                
                # Track old_weights from ACTUAL positions, not intended (avoids phantom turnover)
                old_weights = {t: p["weight"] for t, p in positions.items()}
        
        num_longs = sum(1 for pos in positions.values() if pos["weight"] > 0)
        num_shorts = sum(1 for pos in positions.values() if pos["weight"] < 0)

        # Compute total daily return as sum of long and short
        daily_pnl_bps = long_pnl_bps + short_pnl_bps
        
        pnl_records.append({
            "date": trade_date,
            "daily_ret_bps": daily_pnl_bps,
            "long_pnl_bps": long_pnl_bps,
            "short_pnl_bps": short_pnl_bps,
            "num_longs": num_longs,
            "num_shorts": num_shorts,
            "cost_bps_applied": cost_today,
            "rebalance_signal_date": signal_date,
        })
    
    pnl_df = pd.DataFrame(pnl_records)
    return pnl_df


# ----------------------------
# Metrics Calculation
# ----------------------------

def compute_metrics(pnl_df, regime_map, all_data, rebalance_dates, score_type="direction_score"):
    """Compute performance metrics by regime + cross-sectional IC
    
    Fixes:
    - Normalize date types (pd.Timestamp)
    - Handle missing regimes gracefully
    - Correct cumulative return calculation
    - Compute cross-sectional IC using the same forward labels as the signal dump
    """
    
    # Normalize dates to pd.Timestamp
    pnl_df = pnl_df.copy()
    pnl_df["date"] = pd.to_datetime(pnl_df["date"])
    
    # Map regimes with type safety
    pnl_df["regime"] = pnl_df["date"].apply(
        lambda d: regime_map.get(pd.Timestamp(d), None) if isinstance(d, (pd.Timestamp, np.datetime64, str)) else None
    )
    
    # Compute cumulative return properly
    pnl_df["daily_ret"] = pnl_df["daily_ret_bps"] / 10000
    pnl_df["cumulative_ret"] = (1 + pnl_df["daily_ret"]).cumprod()
    pnl_df["cumulative_ret_pct"] = (pnl_df["cumulative_ret"] - 1) * 100
    
    # Overall metrics
    total_ret = (pnl_df["cumulative_ret"].iloc[-1] - 1) * 100
    avg_daily_ret = pnl_df["daily_ret_bps"].mean()
    daily_vol = pnl_df["daily_ret_bps"].std()
    sharpe = (avg_daily_ret / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0
    
    cumret = pnl_df["cumulative_ret"].values
    running_max = np.maximum.accumulate(cumret)
    dd = (cumret - running_max) / running_max
    maxdd = np.min(dd) * 100
    
    # Compute cross-sectional IC on signal dates using the canonical forward labels.
    ics = []
    for rebalance_date in rebalance_dates:
        snapshot = _cross_sectional_snapshot(all_data, rebalance_date, score_type)
        if len(snapshot) >= 10:
            ic, pval = spearmanr(snapshot["score"], snapshot["ret"])
            ics.append(ic)
    
    ic_mean = np.mean(ics) if len(ics) > 0 else 0.0
    ic_std = np.std(ics) if len(ics) > 1 else 0.0
    ic_tstat = (ic_mean / (ic_std + 1e-10)) * np.sqrt(len(ics)) if ic_std > 0 else 0.0
    ic_pval = 2 * (1 - scipy.stats.t.cdf(abs(ic_tstat), len(ics)-1)) if len(ics) > 1 else 1.0
    
    metrics = {
        "overall": {
            "total_ret": total_ret,
            "avg_daily_ret": avg_daily_ret,
            "sharpe": sharpe,
            "maxdd": maxdd,
            "win_rate": (pnl_df["daily_ret_bps"] > 0).sum() / len(pnl_df) * 100,
            "n_trades": len(pnl_df),
            "ic_mean": ic_mean,
            "ic_tstat": ic_tstat,
            "ic_pval": ic_pval,
            "n_ic_periods": len(ics),
        }
    }
    
    # Regime-specific metrics
    for regime in pnl_df["regime"].dropna().unique():
        regime_data = pnl_df[pnl_df["regime"] == regime].copy()
        if len(regime_data) > 10:
            # Compound only the regime's own daily returns (non-contiguous is fine)
            regime_ret = ((1 + regime_data["daily_ret"]).prod() - 1) * 100
            regime_vol = regime_data["daily_ret_bps"].std()
            regime_sharpe = (regime_data["daily_ret_bps"].mean() / regime_vol) * np.sqrt(252) if regime_vol > 0 else 0
            
            metrics[regime] = {
                "total_ret": regime_ret,
                "sharpe": regime_sharpe,
                "n_days": len(regime_data)
            }
    
    return pnl_df, metrics


def compute_decile_returns(all_data, rebalance_dates, score_type="direction_score"):
    """Analyze forward {FORWARD_DAYS}d returns by score decile.

    Handles:
    - Missing deciles from qcut(duplicates='drop')
    - Gracefully skips periods with insufficient data
    - Robust decile spread calculation
    - Uses absolute returns for magnitude_score decile analysis
    """
    
    decile_results = []
    
    for rebalance_date in rebalance_dates:
        snapshot = _cross_sectional_snapshot(all_data, rebalance_date, score_type)
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

def save_portfolio_selections(all_data, rebalance_dates, output_dir, score_type="direction_score"):
    """Save detailed portfolio selections for each rebalance date
    
    Outputs CSV showing:
    - rebalance_date (signal date at close)
    - entry_date / exit_date (actual trade opens)
    - ticker
    - position_type (long/short)
    - score (direction_score or magnitude_score)
    - entry_price / exit_price (adjusted opens)
    - realized return (%)

    Only stocks with entry prices above the minimum trade price are included.
    """
    
    selections = []
    indexed_data = {
        ticker: df.sort_values("date").set_index("date")
        for ticker, df in all_data.items()
    }

    for rebalance_date in rebalance_dates:
        # Get all scores for this date
        date_scores = {}
        trade_windows = {}
        for ticker, df_indexed in indexed_data.items():
            row = df_indexed[df_indexed.index == rebalance_date]
            if not row.empty:
                score = row[score_type].values[0]
                trade_window = get_trade_window(
                    hist=df_indexed,
                    signal_date=rebalance_date,
                    holding_days=FORWARD_DAYS,
                    execution_model=CANONICAL_EXECUTION_MODEL,
                )
                if (
                    pd.notna(score)
                    and trade_window is not None
                    and has_min_trade_price(trade_window.entry_price)
                ):
                    date_scores[ticker] = score
                    trade_windows[ticker] = trade_window
        
        if len(date_scores) < TOP_N + BOTTOM_N:
            continue
        
        # Rank by score
        sorted_tickers = sorted(date_scores.keys(), key=lambda t: date_scores[t], reverse=True)
        
        # Long positions (top N)
        for rank_idx in range(TOP_N):
            ticker = sorted_tickers[rank_idx]
            trade_window = trade_windows.get(ticker)
            if trade_window is None:
                continue
            entry = trade_window.entry_price
            exit_p = trade_window.exit_price
            ret_10d = (exit_p / entry - 1) * 100 if (entry > 0 and exit_p > 0) else np.nan
            
            selections.append({
                "rebalance_date": rebalance_date,
                "signal_date": trade_window.signal_date,
                "entry_date": trade_window.entry_date,
                "exit_date": trade_window.exit_date,
                "ticker": ticker,
                "position_type": "LONG",
                "rank": rank_idx + 1,
                "score": date_scores[ticker],
                "entry_price": entry,
                "exit_price": exit_p,
                "realized_return_pct": ret_10d,
                "realized_10d_return_pct": ret_10d,
            })
        
        # Short positions (bottom N)
        for rank_idx in range(BOTTOM_N):
            ticker = sorted_tickers[-(rank_idx + 1)]
            trade_window = trade_windows.get(ticker)
            if trade_window is None:
                continue
            entry = trade_window.entry_price
            exit_p = trade_window.exit_price
            # Short P&L: profit when price falls
            ret_10d = -(exit_p / entry - 1) * 100 if (entry > 0 and exit_p > 0) else np.nan
            
            selections.append({
                "rebalance_date": rebalance_date,
                "signal_date": trade_window.signal_date,
                "entry_date": trade_window.entry_date,
                "exit_date": trade_window.exit_date,
                "ticker": ticker,
                "position_type": "SHORT",
                "rank": rank_idx + 1,
                "score": date_scores[ticker],
                "entry_price": entry,
                "exit_price": exit_p,
                "realized_return_pct": ret_10d,
                "realized_10d_return_pct": ret_10d,
            })
    
    df_selections = pd.DataFrame(selections)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "portfolio_selections.csv"
    df_selections.to_csv(csv_path, index=False)
    print(f"   ✓ Saved: {csv_path}")
    
    return df_selections


def print_summary(metrics, decile_df):
    """Print factor backtest summary
    
    Robustly handles missing deciles from qcut(duplicates='drop')
    """
    
    print(f"\n{'='*100}")
    print(f"FACTOR BACKTEST RESULTS")
    print(f"{'='*100}\n")
    
    print(f"📊 OVERALL PERFORMANCE:")
    m = metrics["overall"]
    print(f"   Total Return: {m['total_ret']:+.2f}%")
    print(f"   Avg Daily Return: {m['avg_daily_ret']:+.2f} bps")
    print(f"   Sharpe Ratio: {m['sharpe']:.3f}")
    print(f"   Max Drawdown: {m['maxdd']:.2f}%")
    print(f"   Win Rate: {m['win_rate']:.1f}%")
    print(f"   Trading Days: {m['n_trades']}")
    
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
    return (float(cum_series.iloc[-1]) ** (252.0 / n) - 1) * 100


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


def compute_ic_series(all_data, rebalance_dates, score_type="direction_score"):
    """Return a pd.Series of per-period cross-sectional Spearman ICs, indexed by date.

    For magnitude_score, ICs are computed against absolute forward returns.
    Uses the same forward labels created by the signal dump.
    """
    records = []
    for rd in rebalance_dates:
        snapshot = _cross_sectional_snapshot(all_data, rd, score_type)
        if len(snapshot) >= 10:
            ic, _ = spearmanr(snapshot["score"], snapshot["ret"])
            records.append({"date": pd.Timestamp(rd), "ic": float(ic)})
    if not records:
        return pd.Series(dtype=float)
    return pd.DataFrame(records).set_index("date")["ic"]


def compute_regime_signal_decomposition(all_data, pnl_df, rebalance_dates, score_type="direction_score"):
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
        snapshot = _cross_sectional_snapshot(all_data, rd, score_type)
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
        for i in range(len(cost_sweep) - 1):
            seg_col = _GREEN if y_vals[i] > 0 else _RED
            ax.plot(cost_sweep[i:i + 2], y_vals[i:i + 2], color=seg_col, linewidth=2.5)
        ax.scatter(cost_sweep, y_vals, color=_BLUE, zorder=5, s=55)
        ax.axvline(cur_bps, color=_YELLOW, linewidth=2, linestyle="--",
                   label=f"Current = {cur_bps:.0f} bps")
        ax.axhline(0, color=_MUTED, linewidth=0.8, linestyle="--")
        ax.fill_between(cost_sweep, y_vals, 0,
                        where=np.array(y_vals) >= 0, alpha=0.10, color=_GREEN)
        ax.fill_between(cost_sweep, y_vals, 0,
                        where=np.array(y_vals) <  0, alpha=0.10, color=_RED)
        # Breakeven marker
        for i in range(len(cost_sweep) - 1):
            if (y_vals[i] >= 0 > y_vals[i + 1]) or (y_vals[i] < 0 <= y_vals[i + 1]):
                be = float(np.interp(0, [y_vals[i], y_vals[i + 1]],
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
        ax_t.text(col_x_legs[1], y, "—", color=_MUTED, fontsize=7,
                    transform=ax_t.transAxes, va="top")
        ax_t.text(col_x_legs[2], y, f"{short_r:+.1f}" if pd.notna(short_r) else "N/A", color=short_c, fontsize=7, fontweight="bold",
                    transform=ax_t.transAxes, va="top")

        long_r = stats.get("long_regime_return_pct", np.nan)
        long_c = _GREEN if long_r > 0 else _RED
        ax_t.text(col_x_legs[1], y, f"{long_r:+.1f}" if pd.notna(long_r) else "N/A", color=long_c, fontsize=7, fontweight="bold",
                    transform=ax_t.transAxes, va="top")
        ax_t.text(col_x_legs[2], y, "—", color=_MUTED, fontsize=7,
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


def plot_output_deck(pnl_df, decile_df, metrics, all_data, rebalance_dates, spy_df, output_dir, score_type="direction_score"):
    """Generate the 9-slide clean output deck and save PNGs to output_dir."""
    print(f"\n🎨 Generating Output Deck (9 slides) [{score_type}]...")
    ic_series = compute_ic_series(all_data, rebalance_dates, score_type)
    regime_stats = compute_regime_signal_decomposition(all_data, pnl_df, rebalance_dates, score_type)
    
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

def run_backtest(score_type, all_data, spy_df, regime_map, rebalance_dates, output_dir):
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

    pnl_df  = build_factor_portfolio(all_data, spy_df, rebalance_dates, score_type, regime_map)
    pnl_df, metrics = compute_metrics(pnl_df, regime_map, all_data, rebalance_dates, score_type)
    decile_df = compute_decile_returns(all_data, rebalance_dates, score_type)

    print_summary(metrics, decile_df)

    save_portfolio_selections(all_data, rebalance_dates, subdir, score_type)

    # CSVs
    pnl_df.to_csv(subdir / "factor_pnl.csv", index=False)
    decile_df.to_csv(subdir / "decile_analysis.csv", index=False)
    print(f"   \u2713 CSVs saved \u2192 {subdir}")
    
    # 8-slide output deck
    plot_output_deck(pnl_df, decile_df, metrics, all_data, rebalance_dates, spy_df, subdir, score_type)

    return pnl_df, metrics, decile_df


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
    global FORWARD_DAYS, REBALANCE_DAYS, COMPOSITE_SCORES_DIR
    
    # Parse command-line arguments
    forward_days = None
    start_date = None
    end_date = None
    ticker_filter = None
    scores_dir_override = None
    
    # Extract --scores-dir flag before positional parsing
    argv = list(sys.argv[1:])
    if '--scores-dir' in argv:
        sd_idx = argv.index('--scores-dir')
        if sd_idx + 1 < len(argv):
            scores_dir_override = argv[sd_idx + 1]
            argv = argv[:sd_idx] + argv[sd_idx+2:]
        else:
            print("❌ --scores-dir requires a path argument")
            return
    
    if scores_dir_override:
        COMPOSITE_SCORES_DIR = Path(scores_dir_override)
        print(f"📂 Using custom scores directory: {COMPOSITE_SCORES_DIR}")
    
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
    
    print(f"\n{'='*80}")
    print(f"FACTOR BACKTEST ENGINE")
    print(f"Strategy: Top {TOP_N} Long / Bottom {BOTTOM_N} Short, "
          f"Rebalance Every {REBALANCE_DAYS} Days, Equal Weight")
    print(f"Forward Horizon: {FORWARD_DAYS} days")
    print(f"Min entry price: > ${MIN_TRADE_PRICE:.2f}")
    print(f"Score Types: {', '.join(SCORE_TYPES)}")
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

    # ── Load SPY for regime identification (FULL HISTORY) ────────────────────
    print("\n� Loading SPY data for regime identification...")
    spy_df = load_spy_data()

    # Shared calendar and regime map (identical for all score types)
    all_dates, rebalance_dates = get_rebalance_dates(spy_df["date"], REBALANCE_DAYS)
    regime_map = identify_regimes(spy_df)
    
    # FILTER REBALANCE DATES (trading period only, signals use full history)
    if start_date_ts or end_date_ts:
        print(f"Filtering trading dates by range...")
        rebalance_dates_filtered = []
        for rd in rebalance_dates:
            rd_ts = pd.Timestamp(rd)
            if start_date_ts and rd_ts < start_date_ts:
                continue
            if end_date_ts and rd_ts > end_date_ts:
                continue
            rebalance_dates_filtered.append(rd)
        rebalance_dates = rebalance_dates_filtered
        print(f"   {len(rebalance_dates)} rebalance dates in trading period\n")

    # ── Run independent backtest per score type ────────────────────────────────
    # Create output directory with forward days in the path
    # If using custom scores dir (e.g., LLM), add a label to avoid overwriting deterministic results
    if scores_dir_override:
        scores_label = Path(scores_dir_override).name  # e.g., "composite_scores_llm"
        OUTPUT_DIR = OUTPUT_DIR_BASE / f"{scores_label}_{FORWARD_DAYS}d_horizon"
    else:
        OUTPUT_DIR = OUTPUT_DIR_BASE / f"{FORWARD_DAYS}d_horizon"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}  # {score_type: (pnl_df, metrics)}

    for score_type in SCORE_TYPES:
        pnl_df, metrics, _ = run_backtest(
            score_type, all_data, spy_df, regime_map, rebalance_dates, OUTPUT_DIR
        )
        results[score_type] = (pnl_df, metrics)

    # ── Cross-factor comparison slide (saved at root output dir) ──────────────
    print(f"\n🎨 Generating cross-factor comparison slide...")
    _slide7_comparison(results, spy_df, OUTPUT_DIR)

    print(f"\n{'='*80}")
    print("FACTOR BACKTEST COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}/")
    for st in SCORE_TYPES:
        print(f"  └── {st}/  (9-slide deck + CSVs)")
    print(f"  └── slide_07_comparison.png")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
