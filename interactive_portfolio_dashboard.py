"""
Multi-Horizon Portfolio Dashboard
Reads portfolio_selections.csv files from factor_backtest output directories
and generates a unified interactive HTML dashboard.

Data conventions (from factor_backtest_engine.py):
  Signal is generated on rebalance_date close.
  Entry occurs on the next trading day's open.
  Exit occurs on the later rebalance's execution open.

  LONG  return_col = (exit/entry - 1) * 100       positive if stock went up
  SHORT return_col = -(exit/entry - 1) * 100      positive if stock went DOWN (profit)

  Portfolio period return = mean(LONG_pnl) + mean(SHORT_pnl)
  Equity curve: compound once per rebalance date, forward-fill to daily for display.

  Cross-sectional IC = Spearman(score, raw_stock_return)
  where raw_stock_return = (exit/entry - 1)*100 for ALL positions (not position PnL).
"""

import sys
import argparse
import json
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
BACKTEST_DIR  = BASE_DIR / "backtest_results" / "factor_backtest"
OUTPUT_DIR    = BASE_DIR / "backtest_results" / "analysis_output"

# ─────────────────────────────────────────────────────────────────────────────
# DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
def discover_data() -> dict:
    """
    Scan backtest_results/factor_backtest/ for portfolio_selections.csv files.
    New structure: variants at top level, then horizons within each.
    Returns: {variant_str: {horizon_int: Path}}
    e.g. {'top10_bottom10': {10: Path('...top10_bottom10/10d_horizon/portfolio_selections.csv')}}
    """
    found = {}
    if not BACKTEST_DIR.exists():
        print(f"  ERROR: {BACKTEST_DIR} not found")
        return found

    print(f"  Scanning variants in {BACKTEST_DIR}...")
    # Iterate over variants (top-level folders)
    for variant_dir in sorted(BACKTEST_DIR.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant_name = variant_dir.name
        print(f"    Found variant: {variant_name}")

        # Iterate over horizons within each variant
        for horizon_dir in sorted(variant_dir.iterdir()):
            if not horizon_dir.is_dir() or not horizon_dir.name.endswith("d_horizon"):
                continue
            try:
                h = int(horizon_dir.name.replace("d_horizon", ""))
            except ValueError:
                continue

            csv_path = horizon_dir / "direction_score" / "portfolio_selections.csv"
            if csv_path.exists():
                found.setdefault(variant_name, {})[h] = csv_path
                print(f"      ✓ {h}d horizon: {csv_path.name}")
            else:
                print(f"      ✗ {h}d horizon: CSV not found at {csv_path}")

    return found


def _parse_horizon_from_dir_name(name: str) -> int | None:
    if not name.endswith("d_horizon"):
        return None
    try:
        return int(name.replace("d_horizon", ""))
    except ValueError:
        return None


def _register_score_dir(found: dict, score_dir: Path) -> None:
    csv_path = score_dir / "portfolio_selections.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing portfolio_selections.csv in {score_dir}")

    if score_dir.name not in {"direction_score", "magnitude_score"}:
        raise ValueError(f"Expected a score-type dir, got {score_dir}")

    horizon_dir = score_dir.parent
    variant_dir = horizon_dir.parent
    horizon = _parse_horizon_from_dir_name(horizon_dir.name)
    if horizon is None:
        raise ValueError(f"Could not parse horizon from {horizon_dir}")

    found.setdefault(variant_dir.name, {})[horizon] = csv_path


def discover_data_from_paths(input_paths: list[str]) -> dict:
    """
    Accept explicit paths to a result variant / horizon / score dir / CSV and
    normalize them into the same structure as discover_data().
    """
    found = {}
    for raw in input_paths:
        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

        if path.is_file():
            if path.name != "portfolio_selections.csv":
                raise ValueError(f"Expected portfolio_selections.csv, got {path}")
            _register_score_dir(found, path.parent)
            continue

        if path.name in {"direction_score", "magnitude_score"}:
            _register_score_dir(found, path)
            continue

        horizon = _parse_horizon_from_dir_name(path.name)
        if horizon is not None:
            score_dir = path / "direction_score"
            _register_score_dir(found, score_dir)
            continue

        # Treat as a variant dir and scan horizon children.
        found_variant = False
        for horizon_dir in sorted(path.iterdir()):
            if not horizon_dir.is_dir():
                continue
            horizon = _parse_horizon_from_dir_name(horizon_dir.name)
            if horizon is None:
                continue
            score_dir = horizon_dir / "direction_score"
            if score_dir.exists():
                _register_score_dir(found, score_dir)
                found_variant = True
        if not found_variant:
            raise ValueError(
                f"Could not find a valid direction_score result under {path}. "
                "Pass a variant dir, a 10d_horizon dir, a direction_score dir, or portfolio_selections.csv."
            )

    return found


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(csv_path: Path, horizon: int):
    """Load portfolio_selections.csv. Returns (df, return_col).
    
    NOTE: Older files use realized_10d_return_pct while newer files use
    realized_return_pct. Both contain returns for the actual horizon.
    """
    df = pd.read_csv(csv_path)
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])

    return_col = "realized_return_pct"
    if return_col not in df.columns:
        return_col = "realized_10d_return_pct"
    if return_col not in df.columns:
        ret_cols = [c for c in df.columns if "return_pct" in c]
        if ret_cols:
            return_col = ret_cols[0]
        else:
            raise ValueError(f"No return column found in {csv_path.name}")

    # Drop rows with missing returns or prices
    df = df.dropna(subset=[return_col, "entry_price", "exit_price", "score"])
    df = df[df["entry_price"] > 0]
    df = df[df["exit_price"] > 0]

    # Compute raw stock return for IC calculations (always positive = up)
    # LONG: return_col already is (exit/entry-1)*100
    # SHORT: return_col is -(exit/entry-1)*100, so raw = -return_col
    df["raw_stock_return"] = np.where(
        df["position_type"] == "SHORT",
        -df[return_col],
        df[return_col],
    )

    return df, return_col


def load_engine_outputs(score_dir: Path):
    """
    Load saved engine outputs when available so charts match factor_pnl.csv
    instead of reconstructing from portfolio_selections.csv.
    """
    pnl_path = score_dir / "factor_pnl.csv"
    review_path = score_dir / "review_log.csv"

    pnl_df = None
    review_log_df = None

    if pnl_path.exists():
        pnl_df = pd.read_csv(pnl_path, parse_dates=["date"])
        for col in [
            "daily_ret_bps",
            "gross_daily_ret_bps",
            "net_daily_ret_bps",
            "long_pnl_bps",
            "short_pnl_bps",
            "cost_bps_applied",
            "turnover_total",
            "turnover_long",
            "turnover_short",
            "cumulative_ret_pct",
            "daily_ret",
            "cumulative_ret",
        ]:
            if col in pnl_df.columns:
                pnl_df[col] = pd.to_numeric(pnl_df[col], errors="coerce")

    if review_path.exists():
        parse_cols = [c for c in ["review_date", "signal_date"] if c in pd.read_csv(review_path, nrows=0).columns]
        review_log_df = pd.read_csv(review_path, parse_dates=parse_cols)
        for col in [
            "turnover_total",
            "turnover_long",
            "turnover_short",
            "cost_bps_applied",
            "candidate_longs",
            "candidate_shorts",
            "long_entries",
            "long_exits",
            "short_entries",
            "short_exits",
        ]:
            if col in review_log_df.columns:
                review_log_df[col] = pd.to_numeric(review_log_df[col], errors="coerce")

    return pnl_df, review_log_df


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def build_equity_curve(df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    """
    Compound portfolio returns at each rebalance date, applying transaction costs.
    
    Portfolio period return = mean(LONG pnl) + mean(SHORT pnl) - transaction costs
    Turnover calculated as % of positions that change between rebalances.
    Cost model: 15 bps one-way per unit turnover (matches factor_backtest_engine.py).
    """
    COST_RATE = 0.0015  # 15 bps one-way cost
    
    rebalance_dates = sorted(df["rebalance_date"].unique())
    rows = []
    
    prev_long_tickers = set()
    prev_short_tickers = set()

    for i, date in enumerate(rebalance_dates):
        period = df[df["rebalance_date"] == date]
        longs  = period[period["position_type"] == "LONG"][return_col]
        shorts = period[period["position_type"] == "SHORT"][return_col]

        long_ret  = longs.mean()  if len(longs)  > 0 else 0.0
        short_ret = shorts.mean() if len(shorts) > 0 else 0.0
        period_ret = long_ret + short_ret   # CORRECT: both are already PnL
        
        # Calculate turnover
        curr_long_tickers = set(period[period["position_type"] == "LONG"]["ticker"].unique())
        curr_short_tickers = set(period[period["position_type"] == "SHORT"]["ticker"].unique())
        
        # Turnover = sum of absolute weight changes
        if i == 0:
            long_turnover = 1.0
            short_turnover = 1.0
        else:
            # Match the backtest engine's equal-weight turnover:
            # each exit or entry trades 1/N of capital within that leg.
            long_exits = len(prev_long_tickers - curr_long_tickers)
            long_entries = len(curr_long_tickers - prev_long_tickers)
            long_n = max(len(curr_long_tickers), 1)
            long_turnover = (long_exits + long_entries) / long_n
            
            short_exits = len(prev_short_tickers - curr_short_tickers)
            short_entries = len(curr_short_tickers - prev_short_tickers)
            short_n = max(len(curr_short_tickers), 1)
            short_turnover = (short_exits + short_entries) / short_n
        
        total_turnover = long_turnover + short_turnover
        
        # Apply transaction costs (convert from bps to %)
        cost_bps = COST_RATE * total_turnover * 10000
        cost_pct = cost_bps / 100.0
        
        # Reduce period return by costs
        period_ret_after_costs = period_ret - cost_pct

        rows.append({
            "date":                date,
            "long_ret":            long_ret,
            "short_ret":           short_ret,
            "period_ret":          period_ret,
            "period_ret_gross":    period_ret,
            "turnover":            total_turnover,
            "cost_bps":            cost_bps,
            "period_ret_net":      period_ret_after_costs,
        })
        
        prev_long_tickers = curr_long_tickers
        prev_short_tickers = curr_short_tickers

    eq = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    
    # Compound NET returns (after costs)
    eq["equity"]               = (1 + eq["period_ret_net"] / 100.0).cumprod()
    eq["cumulative_return_pct"] = (eq["equity"] - 1) * 100

    # Forward-fill to daily business days (flat between rebalances)
    all_bdays = pd.bdate_range(
        start=eq["date"].min(),
        end=eq["date"].max() + pd.Timedelta(days=5),
        freq="B",
    )
    daily = pd.DataFrame({"date": all_bdays})
    daily = daily.merge(
        eq[["date", "cumulative_return_pct", "equity", "period_ret_gross", "period_ret_net", 
            "long_ret", "short_ret", "turnover", "cost_bps"]],
        on="date", how="left",
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["cumulative_return_pct"] = daily["cumulative_return_pct"].ffill().fillna(0)
    daily["equity"]               = daily["equity"].ffill().fillna(1)

    return daily


def build_equity_curve_from_pnl(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize factor_pnl.csv into the dashboard's equity_df schema.
    Values are kept at daily frequency, matching the engine output exactly.
    """
    eq = pnl_df.copy().sort_values("date").reset_index(drop=True)
    eq["period_ret_gross"] = pd.to_numeric(eq.get("gross_daily_ret_bps"), errors="coerce").fillna(0.0) / 100.0
    eq["period_ret_net"] = pd.to_numeric(eq.get("net_daily_ret_bps"), errors="coerce").fillna(0.0) / 100.0
    eq["long_ret"] = pd.to_numeric(eq.get("long_pnl_bps"), errors="coerce").fillna(0.0) / 100.0
    eq["short_ret"] = pd.to_numeric(eq.get("short_pnl_bps"), errors="coerce").fillna(0.0) / 100.0
    eq["turnover"] = pd.to_numeric(eq.get("turnover_total"), errors="coerce").fillna(0.0)
    eq["cost_bps"] = pd.to_numeric(eq.get("cost_bps_applied"), errors="coerce").fillna(0.0)
    if "cumulative_return_pct" not in eq.columns:
        eq["cumulative_return_pct"] = pd.to_numeric(eq.get("cumulative_ret_pct"), errors="coerce").fillna(0.0)
    else:
        eq["cumulative_return_pct"] = pd.to_numeric(eq["cumulative_return_pct"], errors="coerce").fillna(0.0)
    if "equity" not in eq.columns:
        eq["equity"] = 1.0 + eq["cumulative_return_pct"] / 100.0
    else:
        eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce").fillna(1.0 + eq["cumulative_return_pct"] / 100.0)
    eq.attrs["periods_per_year"] = 252.0
    eq.attrs["source"] = "factor_pnl"
    return eq


def build_ic_filtered_equity_curve(df: pd.DataFrame, ic_df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    """
    Build equity curve trading ONLY when recent IC is positive.
    Uses LAGGED rolling IC (previous period's 20-period rolling mean) to
    decide whether to trade — avoids look-ahead bias.
    On non-trading periods, hold cash (no return, no cost).
    """
    COST_RATE = 0.0015  # 15 bps one-way cost
    
    # Build lagged IC lookup: for each date, use the PREVIOUS period's rolling IC.
    # This avoids look-ahead bias — we can't know current-period IC until the
    # period is over, so we use the most recent completed rolling IC.
    ic_sorted = ic_df.sort_values("date").reset_index(drop=True)
    ic_lagged = {}
    for i in range(len(ic_sorted)):
        date = ic_sorted.loc[i, "date"]
        if i > 0:
            prev_val = ic_sorted.loc[i - 1, "rolling_ic"]
            ic_lagged[date] = float(prev_val) if not pd.isna(prev_val) else np.nan
        else:
            ic_lagged[date] = np.nan  # No prior info for first period
    
    rebalance_dates = sorted(df["rebalance_date"].unique())
    rows = []
    
    prev_long_tickers = set()
    prev_short_tickers = set()
    in_cash = True  # Start in cash (no positions)

    for date in rebalance_dates:
        # Use LAGGED rolling IC (previous period) to decide
        ic_value = ic_lagged.get(date, np.nan)
        is_trading_period = ic_value > 0 if not np.isnan(ic_value) else False
        
        if is_trading_period:
            # Enter positions
            period = df[df["rebalance_date"] == date]
            longs  = period[period["position_type"] == "LONG"][return_col]
            shorts = period[period["position_type"] == "SHORT"][return_col]
            
            long_ret  = longs.mean()  if len(longs)  > 0 else 0.0
            short_ret = shorts.mean() if len(shorts) > 0 else 0.0
            period_ret = long_ret + short_ret
            
            curr_long_tickers = set(period[period["position_type"] == "LONG"]["ticker"].unique())
            curr_short_tickers = set(period[period["position_type"] == "SHORT"]["ticker"].unique())
            
            # Turnover: from previous state (cash or positions) to current positions
            if in_cash:
                # Coming from cash: opening both long and short books
                total_turnover = 2.0
            else:
                # Match the backtest engine's equal-weight turnover:
                # each exit or entry trades 1/N of capital within that leg.
                long_exits = len(prev_long_tickers - curr_long_tickers)
                long_entries = len(curr_long_tickers - prev_long_tickers)
                long_n = max(len(curr_long_tickers), 1)
                long_turnover = (long_exits + long_entries) / long_n
                
                short_exits = len(prev_short_tickers - curr_short_tickers)
                short_entries = len(curr_short_tickers - prev_short_tickers)
                short_n = max(len(curr_short_tickers), 1)
                short_turnover = (short_exits + short_entries) / short_n
                
                total_turnover = long_turnover + short_turnover
            
            cost_bps = COST_RATE * total_turnover * 10000
            cost_pct = cost_bps / 100.0
            
            period_ret_after_costs = period_ret - cost_pct
            
            prev_long_tickers = curr_long_tickers
            prev_short_tickers = curr_short_tickers
            in_cash = False
        else:
            # Stay in cash. If we were invested previously, exiting both books costs 2 turns.
            period_ret = 0.0
            total_turnover = 2.0 if not in_cash else 0.0
            cost_bps = COST_RATE * total_turnover * 10000
            cost_pct = cost_bps / 100.0
            period_ret_after_costs = -cost_pct
            long_ret = 0.0
            short_ret = 0.0
            in_cash = True  # Flag that we're in cash for next period's turnover calc

        rows.append({
            "date":                date,
            "long_ret":            long_ret if is_trading_period else 0.0,
            "short_ret":           short_ret if is_trading_period else 0.0,
            "period_ret":          period_ret,
            "period_ret_gross":    period_ret,
            "turnover":            total_turnover,
            "cost_bps":            cost_bps,
            "period_ret_net":      period_ret_after_costs,
            "ic_positive":         is_trading_period,
        })

    eq = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    
    # Compound NET returns (after costs)
    eq["equity"]               = (1 + eq["period_ret_net"] / 100.0).cumprod()
    eq["cumulative_return_pct"] = (eq["equity"] - 1) * 100

    # Forward-fill to daily business days (flat between rebalances)
    all_bdays = pd.bdate_range(
        start=eq["date"].min(),
        end=eq["date"].max() + pd.Timedelta(days=5),
        freq="B",
    )
    daily = pd.DataFrame({"date": all_bdays})
    daily = daily.merge(
        eq[["date", "cumulative_return_pct", "equity", "period_ret_gross", "period_ret_net", 
            "long_ret", "short_ret", "turnover", "cost_bps", "ic_positive"]],
        on="date", how="left",
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["cumulative_return_pct"] = daily["cumulative_return_pct"].ffill().fillna(0)
    daily["equity"]               = daily["equity"].ffill().fillna(1)

    return daily


def compute_rolling_ic(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Per-period cross-sectional IC = Spearman(score, raw_stock_return).
    Rolling IC = mean of last `window` period ICs.
    """
    dates = sorted(df["rebalance_date"].unique())
    per_date_ic = {}

    for date in dates:
        day = df[df["rebalance_date"] == date].dropna(subset=["score", "raw_stock_return"])
        if len(day) >= 5:
            ic, _ = spearmanr(day["score"], day["raw_stock_return"])
            per_date_ic[date] = float(ic) if not np.isnan(ic) else np.nan
        else:
            per_date_ic[date] = np.nan

    rows = []
    for i, date in enumerate(dates):
        window_slice = [
            per_date_ic[dates[j]]
            for j in range(max(0, i - window + 1), i + 1)
            if not np.isnan(per_date_ic.get(dates[j], np.nan))
        ]
        rows.append({
            "date":              date,
            "cross_sectional_ic": per_date_ic[date],
            "rolling_ic":        float(np.mean(window_slice)) if window_slice else np.nan,
        })

    return pd.DataFrame(rows)


def compute_decile_analysis(df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    """Bin positions into score deciles; compute mean return per decile.
    
    Deciles are computed WITHIN each subset (ALL/LONG/SHORT) independently.
    This way, LONG-only decile 1 = lowest-scored LONGs, decile 10 = highest-scored LONGs.
    Without per-subset binning, LONGs cluster in global deciles 7-10 and 1-3 appear empty.
    """
    dc = df.dropna(subset=["score", return_col]).copy()
    if len(dc) < 10:
        return pd.DataFrame(columns=["decile", "mean_ret", "std_ret", "count", "subset"])

    frames = []
    for name, sub_df in [("ALL", dc),
                          ("LONG",  dc[dc["position_type"] == "LONG"]),
                          ("SHORT", dc[dc["position_type"] == "SHORT"])]:
        sub = sub_df.copy()
        if len(sub) < 10:
            continue
        try:
            sub["decile"] = pd.qcut(sub["score"], q=10, labels=False, duplicates="drop") + 1
        except Exception:
            continue

        n_bins = int(sub["decile"].max())
        all_deciles = list(range(1, n_bins + 1))

        stats = sub.groupby("decile")[return_col].agg(
            mean_ret="mean", std_ret="std", count="count"
        ).reset_index()
        stats = stats.set_index("decile").reindex(all_deciles).reset_index()
        stats["subset"] = name
        stats["count"] = stats["count"].fillna(0).astype(int)
        stats["mean_ret"] = stats["mean_ret"].fillna(0)
        stats["std_ret"] = stats["std_ret"].fillna(0)
        frames.append(stats)

    if not frames:
        return pd.DataFrame(columns=["decile", "mean_ret", "std_ret", "count", "subset"])
    return pd.concat(frames, ignore_index=True)


def compute_ic_regime_stats(df: pd.DataFrame, ic_df: pd.DataFrame, return_col: str) -> dict:
    """
    Split dates by median cross-sectional IC.
    Compute mean return and win rate per regime × position type.
    """
    merged = df.merge(
        ic_df[["date", "cross_sectional_ic"]],
        left_on="rebalance_date", right_on="date", how="left",
    ).dropna(subset=["cross_sectional_ic"])

    median_ic = float(merged["cross_sectional_ic"].median())
    result = {"median_ic": median_ic, "high": {}, "low": {}}

    for regime, mask in [
        ("high", merged["cross_sectional_ic"] >  median_ic),
        ("low",  merged["cross_sectional_ic"] <= median_ic),
    ]:
        rdf = merged[mask]
        for pos in ["LONG", "SHORT", "ALL"]:
            sub = rdf if pos == "ALL" else rdf[rdf["position_type"] == pos]
            rets = sub[return_col]
            result[regime][pos] = {
                "mean_ret": float(rets.mean()) if len(rets) > 0 else 0.0,
                "win_rate": float((rets > 0).mean()) if len(rets) > 0 else 0.0,
                "n":        int(len(rets)),
            }

    return result


def compute_rank_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Signal persistence: Spearman rank correlation of scores between consecutive
    rebalance dates.  High ρ → signal is stable (genuine factor).  Low ρ → noise.
    Also tracks what fraction of tickers persist between rebalances.
    """
    dates = sorted(df["rebalance_date"].unique())
    rows = []

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        prev = df[df["rebalance_date"] == prev_date][["ticker", "score"]].set_index("ticker")
        curr = df[df["rebalance_date"] == curr_date][["ticker", "score"]].set_index("ticker")

        common = prev.index.intersection(curr.index)
        if len(common) >= 5:
            rho, _ = spearmanr(prev.loc[common, "score"], curr.loc[common, "score"])
        else:
            rho = np.nan

        rows.append({
            "date":           curr_date,
            "rank_autocorr":  float(rho) if not np.isnan(rho) else np.nan,
            "n_common":       len(common),
            "n_prev":         len(prev),
            "n_curr":         len(curr),
            "pct_persisted":  len(common) / max(len(curr), 1) * 100,
        })

    return pd.DataFrame(rows)


def compute_turnover_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose turnover into: entries (new names), exits (dropped names),
    persists (held names) — separately for long and short legs.
    """
    dates = sorted(df["rebalance_date"].unique())
    rows = []

    prev_long = set()
    prev_short = set()

    for i, date in enumerate(dates):
        period = df[df["rebalance_date"] == date]
        curr_long  = set(period[period["position_type"] == "LONG"]["ticker"].unique())
        curr_short = set(period[period["position_type"] == "SHORT"]["ticker"].unique())

        if i == 0:
            rows.append({
                "date": date,
                "long_entries": len(curr_long), "long_exits": 0,
                "long_persists": 0,
                "short_entries": len(curr_short), "short_exits": 0,
                "short_persists": 0,
            })
        else:
            rows.append({
                "date": date,
                "long_entries":  len(curr_long  - prev_long),
                "long_exits":    len(prev_long  - curr_long),
                "long_persists": len(curr_long  & prev_long),
                "short_entries": len(curr_short - prev_short),
                "short_exits":   len(prev_short - curr_short),
                "short_persists":len(curr_short & prev_short),
            })

        prev_long  = curr_long
        prev_short = curr_short

    return pd.DataFrame(rows)


def compute_turnover_attribution_from_review_log(review_log_df: pd.DataFrame) -> pd.DataFrame:
    if review_log_df is None or review_log_df.empty:
        return pd.DataFrame()
    if "review_date" not in review_log_df.columns:
        return pd.DataFrame()

    rows = []
    for _, row in review_log_df.sort_values("review_date").iterrows():
        rows.append({
            "date": row["review_date"],
            "long_entries": int(row.get("long_entries", 0) or 0),
            "long_exits": int(row.get("long_exits", 0) or 0),
            "long_persists": int(max((row.get("num_longs_after", 0) or 0) - (row.get("long_entries", 0) or 0), 0)),
            "short_entries": int(row.get("short_entries", 0) or 0),
            "short_exits": int(row.get("short_exits", 0) or 0),
            "short_persists": int(max((row.get("num_shorts_after", 0) or 0) - (row.get("short_entries", 0) or 0), 0)),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHARTS
# ─────────────────────────────────────────────────────────────────────────────
_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(family="Segoe UI, Arial", color="#e0e0e0", size=12),
    margin=dict(l=50, r=20, t=50, b=50),
    autosize=True,
)


def _fig_dict(fig: go.Figure) -> dict:
    """Serialize a Plotly figure to a plain dict (safe for json.dumps nesting).
    Extracts data from figure BEFORE Plotly applies binary encoding of arrays."""
    # Get the basic figure structure
    fig_dict = fig.to_dict()
    
    # Replace binary-encoded data arrays with plain Python lists
    if 'data' in fig_dict:
        for trace_idx, trace in enumerate(fig_dict['data']):
            # Get the actual trace object to extract data
            trace_obj = fig.data[trace_idx]
            
            # Convert x array
            if hasattr(trace_obj, 'x') and trace_obj.x is not None:
                x_data = trace_obj.x
                if isinstance(x_data, np.ndarray):
                    x_data = x_data.tolist()
                elif isinstance(x_data, pd.Index):
                    x_data = x_data.tolist()
                elif isinstance(x_data, list):
                    pass  # already a list
                else:
                    x_data = list(x_data)
                # Convert dates to ISO strings, handle numpy datetime64
                converted_x = []
                for v in x_data:
                    if isinstance(v, (pd.Timestamp, np.datetime64)):
                        try:
                            converted_x.append(pd.Timestamp(v).isoformat())
                        except:
                            converted_x.append(str(v))
                    elif hasattr(v, 'isoformat'):
                        converted_x.append(v.isoformat())
                    else:
                        converted_x.append(v)
                trace['x'] = converted_x
            
            # Convert y array
            if hasattr(trace_obj, 'y') and trace_obj.y is not None:
                y_data = trace_obj.y
                if isinstance(y_data, np.ndarray):
                    y_data = y_data.tolist()
                elif isinstance(y_data, pd.Series):
                    y_data = y_data.tolist()
                # Convert NaN/inf to None
                y_data = [None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v for v in y_data]
                trace['y'] = y_data
            
            # Convert text array if present
            if hasattr(trace_obj, 'text') and trace_obj.text is not None:
                text_data = trace_obj.text
                if isinstance(text_data, (np.ndarray, pd.Series)):
                    text_data = text_data.tolist()
                trace['text'] = text_data
    
    return fig_dict


def chart_equity(equity_df: pd.DataFrame, horizon: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["date"].dt.strftime("%Y-%m-%d").tolist(), y=equity_df["cumulative_return_pct"].tolist(),
        mode="lines", name="Portfolio",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        hovertemplate="<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(**_THEME, height=360,
                      title=f"Cumulative Return — {horizon}d Horizon",
                      xaxis_title="Date", yaxis_title="Return (%)")
    return fig


def chart_rolling_ic(ic_df: pd.DataFrame, horizon: int) -> go.Figure:
    # Skip first 12 periods to allow rolling window to stabilize
    ic_plot = ic_df.iloc[12:].reset_index(drop=True)
    
    dates_str = ic_plot["date"].dt.strftime("%Y-%m-%d").tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_str, y=ic_plot["cross_sectional_ic"].tolist(),
        mode="markers", name="Period IC",
        marker=dict(size=4, color="rgba(120,200,120,0.35)"),
        hovertemplate="<b>%{x}</b><br>IC: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates_str, y=ic_plot["rolling_ic"].tolist(),
        mode="lines", name="Rolling IC (12-period)",
        line=dict(color="#f97316", width=2),
        hovertemplate="<b>%{x}</b><br>Rolling IC: %{y:.4f}<extra></extra>",
    ))
    mean_ic = ic_df["rolling_ic"].mean()
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,80,80,0.4)")
    if not np.isnan(mean_ic):
        fig.add_hline(y=mean_ic, line_dash="dot", line_color="rgba(100,200,100,0.6)",
                      annotation_text=f"Mean {mean_ic:.4f}", annotation_font_color="#64c864",
                      annotation_position="bottom right")
    fig.update_layout(**_THEME, height=360,
                      title=f"Rolling IC — {horizon}d Horizon",
                      xaxis_title="Date", yaxis_title="IC")
    return fig


def chart_period_return_vs_ic(df: pd.DataFrame, ic_df: pd.DataFrame,
                               return_col: str, horizon: int) -> go.Figure:
    # Compute proper portfolio return: mean(LONG) + mean(SHORT)
    period_stats = df.groupby("rebalance_date").apply(
        lambda g: (
            g[g["position_type"] == "LONG"][return_col].mean()
            + g[g["position_type"] == "SHORT"][return_col].mean()
        )
    ).reset_index()
    period_stats.columns = ["date", "portfolio_ret"]
    period_stats = period_stats.merge(ic_df[["date", "cross_sectional_ic"]], on="date", how="left")

    fig = go.Figure(go.Scatter(
        x=period_stats["date"].dt.strftime("%Y-%m-%d").tolist(),
        y=period_stats["portfolio_ret"].tolist(),
        mode="markers", name="Period Return",
        marker=dict(
            size=6,
            color=period_stats["cross_sectional_ic"].fillna(0).tolist(),
            colorscale="RdYlGn", showscale=True,
            colorbar=dict(title="IC", thickness=10, len=0.7),
            line=dict(width=0.3, color="white"),
        ),
        hovertemplate="<b>%{x}</b><br>Portfolio Ret: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(**_THEME, height=360,
                      title=f"Per-Period Portfolio Return (colored by IC) — {horizon}d",
                      xaxis_title="Date", yaxis_title="Portfolio Return (%)")
    return fig


def chart_long_vs_short(df: pd.DataFrame, return_col: str, horizon: int, equity_df: pd.DataFrame = None) -> go.Figure:
    """Compare long vs short legs using compounded equity curves (gross, before costs).
    
    Note: Costs apply to the combined portfolio turnover and are reflected in the
    main equity curve. Individual legs show signal quality (before transaction costs).
    """
    # Get per-period mean returns by leg
    ls = df.groupby(["rebalance_date", "position_type"])[return_col].mean().reset_index()
    fig = go.Figure()
    
    for pos, color, fill in [("LONG", "#3fb950", "rgba(63,185,80,0.08)"),
                              ("SHORT", "#f85149", "rgba(248,81,73,0.08)")]:
        sub = ls[ls["position_type"] == pos].sort_values("rebalance_date").reset_index(drop=True)
        # Compound returns: (1 + r/100).cumprod() - 1) * 100
        compounded_equity = (1 + sub[return_col] / 100.0).cumprod()
        cum_pnl = (compounded_equity - 1) * 100  # Back to percentage
        
        fig.add_trace(go.Scatter(
            x=sub["rebalance_date"].dt.strftime("%Y-%m-%d").tolist(), y=cum_pnl.tolist(),
            mode="lines", name=pos, line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=fill,
            hovertemplate=f"<b>%{{x}}</b><br>{pos} cumulative return (gross): %{{y:.2f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(**_THEME, height=360,
                      title=f"Compounded Long vs Short Returns (Gross) — {horizon}d",
                      xaxis_title="Date", yaxis_title="Cumulative Return (%)")
    return fig


def chart_score_vs_return(df: pd.DataFrame, horizon: int) -> go.Figure:
    """Score vs raw stock return (to show IC relationship regardless of L/S)."""
    dc = df.dropna(subset=["score", "raw_stock_return"])
    overall_ic, _ = spearmanr(dc["score"], dc["raw_stock_return"])

    z    = np.polyfit(dc["score"], dc["raw_stock_return"], 1)
    pfn  = np.poly1d(z)
    xr   = np.linspace(dc["score"].quantile(0.01), dc["score"].quantile(0.99), 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dc["score"], y=dc["raw_stock_return"],
        mode="markers", name="Positions",
        marker=dict(size=3, opacity=0.2, color="#58a6ff"),
        hovertemplate="Score: %{x:.3f}<br>Ret: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xr, y=pfn(xr), mode="lines",
        name=f"Trend (IC={overall_ic:.4f})",
        line=dict(color="#f97316", width=2, dash="dash"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,80,80,0.3)")
    fig.update_layout(**_THEME, height=360,
                      title=f"Score vs Raw Stock Return — {horizon}d (IC={overall_ic:.4f})",
                      xaxis_title="Composite Score", yaxis_title="Raw Stock Return (%)")
    return fig


def chart_high_low_ic_dist(df: pd.DataFrame, ic_df: pd.DataFrame,
                            return_col: str, horizon: int) -> go.Figure:
    merged   = df.merge(ic_df[["date", "cross_sectional_ic"]],
                        left_on="rebalance_date", right_on="date", how="left")
    med      = merged["cross_sectional_ic"].median()
    hi_rets  = merged[merged["cross_sectional_ic"] >  med][return_col].dropna()
    lo_rets  = merged[merged["cross_sectional_ic"] <= med][return_col].dropna()

    fig = go.Figure()
    fig.add_trace(go.Box(y=hi_rets, name=f"High IC (>{med:.3f})", boxmean="sd",
                         marker_color="#3fb950", fillcolor="rgba(63,185,80,0.15)"))
    fig.add_trace(go.Box(y=lo_rets, name=f"Low IC (≤{med:.3f})", boxmean="sd",
                         marker_color="#f85149", fillcolor="rgba(248,81,73,0.15)"))
    fig.update_layout(**_THEME, height=360,
                      title=f"Return Distribution: High vs Low IC — {horizon}d",
                      yaxis_title="Return / P&L (%)")
    return fig


def chart_deciles(decile_df: pd.DataFrame, horizon: int) -> go.Figure:
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("All Positions", "Long Only", "Short Only"),
                        shared_yaxes=True,
                        horizontal_spacing=0.12)

    for col_i, subset in enumerate(["ALL", "LONG", "SHORT"], start=1):
        sub = decile_df[decile_df["subset"] == subset].sort_values("decile")
        if sub.empty:
            continue
        colors = ["#3fb950" if v >= 0 else "#f85149" for v in sub["mean_ret"]]
        fig.add_trace(go.Bar(
            x=sub["decile"], y=sub["mean_ret"],
            name=subset, marker_color=colors,
            hovertemplate="D%{x}: %{y:.2f}%<extra></extra>",
            showlegend=False,
        ), row=1, col=col_i)
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)",
                      row=1, col=col_i)

    fig.update_xaxes(title_text="Score Decile")
    fig.update_yaxes(title_text="Avg Return (%)", col=1)
    fig.update_layout(**_THEME, height=420,
                      title=f"Decile Forward Returns — {horizon}d Horizon")
    fig.update_layout(margin=dict(l=60, r=60, t=80, b=60))
    return fig


def chart_cost_impact(equity_df: pd.DataFrame, horizon: int) -> go.Figure:
    """Show gross vs net cumulative return paths."""
    # Build gross equity curve (no costs)
    eq_rebal = equity_df.dropna(subset=["period_ret_gross"])
    gross_equity = (1 + eq_rebal["period_ret_gross"] / 100.0).cumprod()
    gross_cumret = (gross_equity - 1) * 100

    # Net equity curve (with costs) already in equity_df
    net_cumret = eq_rebal["cumulative_return_pct"]

    dates_str = eq_rebal["date"].dt.strftime("%Y-%m-%d").tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_str, y=gross_cumret.values,
        mode="lines", name="Gross Cumulative Return",
        line=dict(color="#58a6ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        hovertemplate="<b>%{x}</b><br>Gross: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates_str, y=net_cumret.values,
        mode="lines", name="Net Cumulative Return",
        line=dict(color="#3fb950", width=2.5),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.08)",
        hovertemplate="<b>%{x}</b><br>Net: %{y:.2f}%<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(**_THEME, height=420,
                      title=f"Gross vs Net Cumulative Return — {horizon}d Horizon",
                      xaxis_title="Date", yaxis_title="Cumulative Return (%)",
                      hovermode="x unified")
    return fig


def chart_turnover_costs(equity_df: pd.DataFrame, horizon: int) -> go.Figure:
    """Show per-period turnover (%) and costs (bps) relationship."""
    eq_rebal = equity_df.dropna(subset=["turnover", "cost_bps"])
    dates_str = eq_rebal["date"].dt.strftime("%Y-%m-%d").tolist()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Turnover (%)", "Transaction Cost (bps)"),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )
    
    # Turnover
    fig.add_trace(go.Bar(
        x=dates_str, y=eq_rebal["turnover"].values * 100,
        name="Turnover",
        marker_color="#58a6ff",
        hovertemplate="<b>%{x}</b><br>Turnover: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    
    # Cost in bps
    fig.add_trace(go.Bar(
        x=dates_str, y=eq_rebal["cost_bps"].values,
        name="Cost",
        marker_color="#f85149",
        hovertemplate="<b>%{x}</b><br>Cost: %{y:.1f} bps<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    
    fig.update_yaxes(title_text="Turnover (%)", row=1, col=1)
    fig.update_yaxes(title_text="Cost (bps)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    fig.update_layout(**_THEME, height=480,
                      title=f"Turnover & Transaction Costs — {horizon}d Horizon")
    return fig


def chart_cost_pct_of_return(equity_df: pd.DataFrame, horizon: int) -> go.Figure:
    """Show what % of gross return is consumed by costs."""
    eq_rebal = equity_df.dropna(subset=["period_ret_gross", "cost_bps"])
    eq_rebal = eq_rebal[eq_rebal["period_ret_gross"] != 0].copy()
    
    # Convert cost bps to %
    cost_pct = eq_rebal["cost_bps"] / 10000.0
    gross_ret = eq_rebal["period_ret_gross"]
    
    # Avoid division by zero
    cost_as_pct_of_return = np.where(
        gross_ret != 0,
        (cost_pct / np.abs(gross_ret)) * 100,
        0
    )
    
    dates_str = eq_rebal["date"].dt.strftime("%Y-%m-%d").tolist()
    
    fig = go.Figure()
    colors = ["#f85149" if c > 50 else "#ffa500" if c > 25 else "#3fb950" for c in cost_as_pct_of_return]
    fig.add_trace(go.Bar(
        x=dates_str, y=cost_as_pct_of_return,
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Cost: %{y:.1f}% of gross return<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=25, line_dash="dash", line_color="rgba(255,165,0,0.5)",
                  annotation_text="25% threshold", annotation_position="right")
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(248,81,73,0.5)",
                  annotation_text="50% threshold", annotation_position="right")
    
    fig.update_layout(**_THEME, height=360,
                      title=f"Cost as % of Gross Return — {horizon}d Horizon",
                      xaxis_title="Date", yaxis_title="Cost (% of Gross Return)")
    return fig


def chart_equity_comparison(equity_full: pd.DataFrame, equity_filtered: pd.DataFrame, horizon: int) -> go.Figure:
    """Compare full backtest vs IC-filtered (only trade when IC > 0)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_full["date"].dt.strftime("%Y-%m-%d").tolist(),
        y=equity_full["cumulative_return_pct"].tolist(),
        mode="lines", name="Full Backtest (Always Trade)",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        hovertemplate="<b>%{x}</b><br>Full: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=equity_filtered["date"].dt.strftime("%Y-%m-%d").tolist(),
        y=equity_filtered["cumulative_return_pct"].tolist(),
        mode="lines", name="IC-Filtered (lagged rolling IC > 0)",
        line=dict(color="#3fb950", width=2),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.08)",
        hovertemplate="<b>%{x}</b><br>Filtered: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(**_THEME, height=360,
                      title=f"Full vs IC-Filtered (lagged) — {horizon}d Horizon",
                      xaxis_title="Date", yaxis_title="Return (%)",
                      hovermode="x unified")
    return fig


def chart_ic_regimes(ic_regime_stats: dict, horizon: int) -> go.Figure:
    cats  = ["LONG", "SHORT", "ALL"]
    high  = [ic_regime_stats["high"][c]["mean_ret"] for c in cats]
    low   = [ic_regime_stats["low"][c]["mean_ret"] for c in cats]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=cats, y=high, name="High IC",
                         marker_color="#3fb950",
                         hovertemplate="%{x}: %{y:.2f}%<extra>High IC</extra>"))
    fig.add_trace(go.Bar(x=cats, y=low, name="Low IC",
                         marker_color="#f85149",
                         hovertemplate="%{x}: %{y:.2f}%<extra>Low IC</extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(**_THEME, height=380, barmode="group",
                      title=f"Mean Return by IC Regime — {horizon}d Horizon",
                      xaxis_title="Position Type", yaxis_title="Avg Return / P&L (%)")
    return fig


def chart_rank_stability(rank_df: pd.DataFrame, horizon: int) -> go.Figure:
    """Signal persistence: rank autocorrelation between consecutive rebalances."""
    if rank_df.empty:
        fig = go.Figure()
        fig.update_layout(**_THEME, height=500,
                          title=f"Rank Stability — {horizon}d (insufficient data)")
        return fig

    dates_str = rank_df["date"].dt.strftime("%Y-%m-%d").tolist()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Score Rank Autocorrelation (ρ between consecutive rebalances)",
            "% of Universe Persisting Across Rebalances",
        ),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    # ── Row 1: rank autocorrelation ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates_str, y=rank_df["rank_autocorr"].tolist(),
        mode="lines+markers", name="Rank ρ",
        line=dict(color="#58a6ff", width=2),
        marker=dict(size=4),
        hovertemplate="<b>%{x}</b><br>ρ = %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    mean_rho = rank_df["rank_autocorr"].mean()
    if not np.isnan(mean_rho):
        fig.add_hline(y=mean_rho, line_dash="dot",
                      line_color="rgba(249,115,22,0.6)",
                      annotation_text=f"Mean ρ = {mean_rho:.3f}",
                      annotation_font_color="#f97316",
                      annotation_position="bottom right",
                      row=1, col=1)
    fig.add_hline(y=0, line_dash="dash",
                  line_color="rgba(255,255,255,0.2)", row=1, col=1)

    # ── Row 2: persistence % ────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=dates_str, y=rank_df["pct_persisted"].tolist(),
        name="% Persisted",
        marker_color="#3fb950",
        hovertemplate="<b>%{x}</b><br>%{y:.1f}% persisted<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    mean_pct = rank_df["pct_persisted"].mean()
    fig.add_hline(y=mean_pct, line_dash="dot",
                  line_color="rgba(249,115,22,0.6)",
                  annotation_text=f"Mean = {mean_pct:.1f}%",
                  annotation_font_color="#f97316",
                  annotation_position="bottom right",
                  row=2, col=1)

    fig.update_yaxes(title_text="Spearman ρ", row=1, col=1)
    fig.update_yaxes(title_text="% Persisted", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(**_THEME, height=520, showlegend=False,
                      title=f"Signal Persistence / Rank Stability — {horizon}d Horizon")
    return fig


def chart_turnover_attribution(turnover_attr_df: pd.DataFrame, horizon: int) -> go.Figure:
    """Stacked bar: entries / exits / persists for each leg per rebalance."""
    if turnover_attr_df.empty:
        fig = go.Figure()
        fig.update_layout(**_THEME, height=480,
                          title=f"Turnover Attribution — {horizon}d (no data)")
        return fig

    ta = turnover_attr_df.iloc[1:].reset_index(drop=True)  # skip first period (all entries)
    dates_str = ta["date"].dt.strftime("%Y-%m-%d").tolist()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Long Leg Composition", "Short Leg Composition"),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    # ── Long leg ────────────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=dates_str, y=ta["long_persists"].tolist(),
        name="Held", marker_color="#3fb950",
        hovertemplate="<b>%{x}</b><br>Held: %{y}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=dates_str, y=ta["long_entries"].tolist(),
        name="New Entry", marker_color="#58a6ff",
        hovertemplate="<b>%{x}</b><br>New: %{y}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=dates_str, y=ta["long_exits"].tolist(),
        name="Exited", marker_color="#f85149",
        hovertemplate="<b>%{x}</b><br>Exited: %{y}<extra></extra>",
    ), row=1, col=1)

    # ── Short leg ───────────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=dates_str, y=ta["short_persists"].tolist(),
        name="Held", marker_color="#3fb950",
        hovertemplate="<b>%{x}</b><br>Held: %{y}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=dates_str, y=ta["short_entries"].tolist(),
        name="New Entry", marker_color="#58a6ff",
        hovertemplate="<b>%{x}</b><br>New: %{y}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=dates_str, y=ta["short_exits"].tolist(),
        name="Exited", marker_color="#f85149",
        hovertemplate="<b>%{x}</b><br>Exited: %{y}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    fig.update_yaxes(title_text="# Positions", row=1, col=1)
    fig.update_yaxes(title_text="# Positions", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(**_THEME, height=520, barmode="stack",
                      title=f"Turnover Attribution — {horizon}d Horizon",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1))
    return fig


def chart_implementation_shortfall(equity_df: pd.DataFrame, horizon: int) -> go.Figure:
    """Arithmetic cumulative fees paid through time."""
    eq_rebal = equity_df.dropna(subset=["cost_bps"]).copy()
    if eq_rebal.empty:
        fig = go.Figure()
        fig.update_layout(**_THEME, height=400,
                          title=f"Cumulative Fees Paid — {horizon}d (no data)")
        return fig

    per_period_cost_pct = eq_rebal["cost_bps"] / 100.0
    cum_fees_paid = per_period_cost_pct.cumsum()
    dates_str = eq_rebal["date"].dt.strftime("%Y-%m-%d").tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_str, y=cum_fees_paid.values,
        mode="lines", name="Cumulative Fees Paid",
        line=dict(color="#f85149", width=2.5),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.10)",
        hovertemplate="<b>%{x}</b><br>Cum. Fees: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=dates_str, y=per_period_cost_pct.values,
        name="Per-Period Fee",
        marker_color="rgba(88,166,255,0.45)",
        hovertemplate="<b>%{x}</b><br>Period Fee: %{y:.2f}%<extra></extra>",
    ))
    total = cum_fees_paid.iloc[-1] if len(cum_fees_paid) > 0 else 0
    fig.update_layout(**_THEME, height=400,
                      title=f"Cumulative Fees Paid — {horizon}d &nbsp; "
                            f"(arithmetic total: {total:.2f}%)",
                      xaxis_title="Date",
                      yaxis_title="Fees Paid (%)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1))
    return fig


def chart_horizon_efficiency(horizon_stats: dict) -> go.Figure:
    """Cross-horizon: Net Sharpe, Total Return, Mean IC vs holding period."""
    horizons = sorted(horizon_stats.keys())
    sharpes  = [horizon_stats[h]["sharpe"]    for h in horizons]
    returns  = [horizon_stats[h]["total_ret"] for h in horizons]
    ics      = [horizon_stats[h]["mean_ic"]   for h in horizons]
    labels   = [f"{h}d" for h in horizons]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Net Sharpe", "Total Return (%)", "Mean IC"),
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Bar(
        x=labels, y=sharpes, name="Net Sharpe",
        marker_color=["#3fb950" if s > 0 else "#f85149" for s in sharpes],
        text=[f"{s:.2f}" for s in sharpes], textposition="outside",
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=labels, y=returns, name="Total Return",
        marker_color=["#3fb950" if r > 0 else "#f85149" for r in returns],
        text=[f"{r:.1f}%" for r in returns], textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=labels, y=ics, name="Mean IC",
        marker_color=["#3fb950" if ic > 0 else "#f85149" for ic in ics],
        text=[f"{ic:.3f}" for ic in ics], textposition="outside",
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=3)

    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="IC", row=1, col=3)
    fig.update_layout(**_THEME, height=420,
                      title="Horizon Efficiency — Where Does Alpha Live?")
    fig.update_layout(margin=dict(l=50, r=50, t=80, b=50))
    return fig


def chart_alpha_decay(horizon_stats: dict) -> go.Figure:
    """Cross-horizon alpha decay: IC and Sharpe as a function of holding period."""
    horizons = sorted(horizon_stats.keys())
    ics      = [horizon_stats[h]["mean_ic"] for h in horizons]
    sharpes  = [horizon_stats[h]["sharpe"]  for h in horizons]
    labels   = [f"{h}d" for h in horizons]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=labels, y=ics,
        mode="lines+markers+text", name="Mean IC",
        line=dict(color="#58a6ff", width=3),
        marker=dict(size=10),
        text=[f"{ic:.4f}" for ic in ics],
        textposition="top center",
        textfont=dict(color="#58a6ff", size=11),
        hovertemplate="%{x}: IC = %{y:.4f}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=labels, y=sharpes,
        mode="lines+markers+text", name="Net Sharpe",
        line=dict(color="#f97316", width=3, dash="dot"),
        marker=dict(size=10, symbol="diamond"),
        text=[f"{s:.2f}" for s in sharpes],
        textposition="bottom center",
        textfont=dict(color="#f97316", size=11),
        hovertemplate="%{x}: Sharpe = %{y:.3f}<extra></extra>",
    ), secondary_y=True)

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.update_yaxes(title_text="Mean IC", secondary_y=False)
    fig.update_yaxes(title_text="Net Sharpe", secondary_y=True)
    fig.update_layout(**_THEME, height=420,
                      title="Alpha Decay — Signal Strength vs Holding Period",
                      xaxis_title="Holding Horizon",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HOLDINGS EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def build_holdings_data(df: pd.DataFrame, ic_df: pd.DataFrame, return_col: str) -> dict:
    """Build per-date holdings dict for the interactive explorer."""
    ic_lookup = {}
    for _, row in ic_df.iterrows():
        ds = row["date"].strftime("%Y-%m-%d")
        v  = row["cross_sectional_ic"]
        ic_lookup[ds] = None if pd.isna(v) else round(float(v), 4)

    dates_data = {}
    for date in sorted(df["rebalance_date"].unique()):
        ds      = date.strftime("%Y-%m-%d")
        day_df  = df[df["rebalance_date"] == date]
        longs   = day_df[day_df["position_type"] == "LONG"][return_col]
        shorts  = day_df[day_df["position_type"] == "SHORT"][return_col]
        all_ret = day_df[return_col]

        long_r  = float(longs.mean())  if len(longs)  > 0 else 0.0
        short_r = float(shorts.mean()) if len(shorts) > 0 else 0.0

        stats = {
            "portfolio_ret": round(long_r + short_r, 4),
            "long_ret":      round(long_r, 4),
            "short_ret":     round(short_r, 4),
            "win_rate":      round(float((all_ret > 0).mean()), 4),
            "avg_score":     round(float(day_df["score"].mean()), 4),
            "n":             int(len(day_df)),
            "ic":            ic_lookup.get(ds),
        }

        holdings = []
        for _, row in day_df.sort_values("rank").iterrows():
            rank_val = row.get("rank")
            rank_out = int(rank_val) if pd.notna(rank_val) else None
            holdings.append({
                "ticker":        str(row["ticker"]),
                "position_type": str(row["position_type"]),
                "rank":          rank_out,
                "score":         round(float(row["score"]), 3),
                "entry_price":   round(float(row["entry_price"]), 2),
                "exit_price":    round(float(row["exit_price"]), 2),
                "pnl":           round(float(row[return_col]), 3),
                "raw_ret":       round(float(row["raw_stock_return"]), 3),
            })

        # Per-date scatter chart: score vs raw stock return
        dc = day_df.dropna(subset=["score", "raw_stock_return"])
        sc_fig = go.Figure()
        for pos, color in [("LONG", "#3fb950"), ("SHORT", "#f85149")]:
            sub = dc[dc["position_type"] == pos]
            if len(sub) > 0:
                sc_fig.add_trace(go.Scatter(
                    x=sub["score"].tolist(), y=sub["raw_stock_return"].tolist(),
                    mode="markers+text", name=pos,
                    text=sub["ticker"].tolist(),
                    textposition="top center",
                    textfont=dict(size=8),
                    marker=dict(size=8, color=color, opacity=0.85),
                    hovertemplate="%{text}: Score=%{x:.3f}, Raw Ret=%{y:.2f}%<extra></extra>",
                ))
        sc_fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        sc_fig.update_layout(**_THEME, height=360,
                             title=f"Score vs Raw Return — {ds}",
                             xaxis_title="Score", yaxis_title="Raw Stock Return (%)",
                             showlegend=True)

        dates_data[ds] = {
            "stats":      stats,
            "holdings":   holdings,
            "chart_json": _fig_dict(sc_fig),
        }

    ticker_index = {}
    for _, row in df.sort_values(["ticker", "rebalance_date"]).iterrows():
        ticker = str(row["ticker"]).upper()
        rank_val = row.get("rank")
        rank_out = int(rank_val) if pd.notna(rank_val) else None
        ticker_index.setdefault(ticker, []).append({
            "rebalance_date": pd.Timestamp(row["rebalance_date"]).strftime("%Y-%m-%d"),
            "signal_date": pd.Timestamp(row["signal_date"]).strftime("%Y-%m-%d") if pd.notna(row.get("signal_date")) else "",
            "entry_date": pd.Timestamp(row["entry_date"]).strftime("%Y-%m-%d") if pd.notna(row.get("entry_date")) else "",
            "exit_date": pd.Timestamp(row["exit_date"]).strftime("%Y-%m-%d") if pd.notna(row.get("exit_date")) else "",
            "position_type": str(row["position_type"]),
            "rank": rank_out,
            "score": round(float(row["score"]), 3),
            "entry_price": round(float(row["entry_price"]), 2),
            "exit_price": round(float(row["exit_price"]), 2),
            "pnl": round(float(row[return_col]), 3),
            "raw_ret": round(float(row["raw_stock_return"]), 3),
        })

    dates_data["__ticker_index__"] = ticker_index
    return dates_data


# ─────────────────────────────────────────────────────────────────────────────
# HTML ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(v, decimals=2, suffix="%"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{decimals}f}{suffix}"


def build_dashboard(all_variants: dict) -> str:
    """
    Build dashboard for multiple variants.
    all_variants: {variant_name: {horizon: {...data...}}}
    """
    variants = sorted(all_variants.keys())
    first_v  = variants[0]
    horizons = sorted(all_variants[first_v].keys())
    first_h  = horizons[0]

    # ── Generate charts for all variants + horizons ──────────────────────────
    all_charts = {}  # {variant: {horizon: {chart_name: chart_json}}}
    all_stats = {}   # {variant: {horizon: stats_dict}}
    all_holdings = {} # {variant: {horizon: holdings_data}}
    all_cross = {}    # {variant: cross_charts}

    for variant in variants:
        variant_data = all_variants[variant]
        charts = {}
        stats = {}
        holdings = {}

        for h in horizons:
            d  = variant_data[h]
            df = d["df"]
            rc = d["return_col"]

            # Generate charts
            charts[h] = {
                "equity":     _fig_dict(chart_equity(d["equity_df"], h)),
                "rolling_ic": _fig_dict(chart_rolling_ic(d["ic_df"], h)),
                "period_ret": _fig_dict(chart_period_return_vs_ic(df, d["ic_df"], rc, h)),
                "ls":         _fig_dict(chart_long_vs_short(df, rc, h)),
                "scatter":    _fig_dict(chart_score_vs_return(df, h)),
                "hi_lo":      _fig_dict(chart_high_low_ic_dist(df, d["ic_df"], rc, h)),
                "deciles":    _fig_dict(chart_deciles(d["decile_df"], h)),
                "ic_regimes": _fig_dict(chart_ic_regimes(d["ic_regime_stats"], h)),
                "cost_impact": _fig_dict(chart_cost_impact(d["equity_df"], h)),
                "turnover_costs": _fig_dict(chart_turnover_costs(d["equity_df"], h)),
                "cost_pct": _fig_dict(chart_cost_pct_of_return(d["equity_df"], h)),
                "rank_stability": _fig_dict(chart_rank_stability(d["rank_stability_df"], h)),
                "turnover_attr": _fig_dict(chart_turnover_attribution(d["turnover_attr_df"], h)),
                "impl_shortfall": _fig_dict(chart_implementation_shortfall(d["equity_df"], h)),
            }

            # Generate stats
            eq = d["equity_df"]
            ic = d["ic_df"]
            total_ret = float(eq["cumulative_return_pct"].iloc[-1])
            first_d = eq["date"].min()
            last_d = eq["date"].max()
            years = max((last_d - first_d).days / 365.25, 0.01)
            cagr = (float(eq["equity"].iloc[-1]) ** (1 / years) - 1) * 100
            net_rets = eq["period_ret_net"].dropna()
            periods_per_year = float(eq.attrs.get("periods_per_year", 252 / h))
            sharpe = float((net_rets.mean() / net_rets.std() * np.sqrt(periods_per_year))
                           if len(net_rets) > 1 and net_rets.std() > 0 else 0)
            eq_vals = eq["equity"].values
            running_max = np.maximum.accumulate(eq_vals)
            max_dd = float(np.min((eq_vals - running_max) / running_max) * 100)
            win_rate = float((df[rc] > 0).mean() * 100)
            mean_ic = float(ic["rolling_ic"].mean())
            n_periods = df["rebalance_date"].nunique()
            n_tickers = df["ticker"].nunique()
            n_long = int((df["position_type"] == "LONG").sum() // n_periods)
            n_short = int((df["position_type"] == "SHORT").sum() // n_periods)

            stats[h] = {
                "total_ret": total_ret, "cagr": cagr, "sharpe": sharpe,
                "max_dd": max_dd, "win_rate": win_rate, "mean_ic": mean_ic,
                "n_periods": n_periods, "n_tickers": n_tickers,
                "n_long": n_long, "n_short": n_short,
            }

            # Holdings data
            holdings[h] = d["holdings_data"]

        all_charts[variant] = charts
        all_stats[variant] = stats
        all_holdings[variant] = holdings

        # Cross-horizon charts
        cross_charts_v = {}
        if len(horizons) >= 2:
            cross_charts_v = {
                "horizon_eff": _fig_dict(chart_horizon_efficiency(stats)),
                "alpha_decay": _fig_dict(chart_alpha_decay(stats)),
            }
        all_cross[variant] = cross_charts_v

    # ── Generate HTML snippets for all variants + horizons ──────────────────
    def make_stats_html(h, stats_dict):
        s = stats_dict[h]
        rows = [
            ("Total Return",   _fmt(s["total_ret"]),      "pos" if s["total_ret"] >= 0 else "neg"),
            ("CAGR",           _fmt(s["cagr"]),            "pos" if s["cagr"] >= 0 else "neg"),
            ("Sharpe Ratio",   f"{s['sharpe']:.3f}",       "pos" if s["sharpe"] >= 0 else "neg"),
            ("Max Drawdown",   _fmt(s["max_dd"]),          "neg"),
            ("Mean IC",        f"{s['mean_ic']:.4f}",      "pos" if s["mean_ic"] >= 0 else "neg"),
            ("Win Rate",       f"{s['win_rate']:.1f}%",    ""),
            ("Rebal. Periods", str(s["n_periods"]),        ""),
        ]
        cards = "".join(
            f'<div class="scard"><div class="slabel">{label}</div>'
            f'<div class="sval {cls}">{val}</div></div>'
            for label, val, cls in rows
        )
        return f'<div class="stats-grid">{cards}</div>'

    def make_regime_html(h, variant):
        ir  = all_variants[variant][h]["ic_regime_stats"]
        med = ir["median_ic"]
        def row(label, regime, pos):
            v   = ir[regime][pos]["mean_ret"]
            wr  = ir[regime][pos]["win_rate"] * 100
            cls = "pos" if v >= 0 else "neg"
            return (f'<div class="rrow"><span>{label}</span>'
                    f'<span class="{cls}">{_fmt(v)} &nbsp; WR {wr:.1f}%</span></div>')
        return f"""
<div class="regime-meta">Median cross-sectional IC = <strong>{med:.4f}</strong></div>
<div class="regime-grid">
  <div class="rcard high-ic">
    <div class="rc-title">✅ High IC Periods &nbsp; (IC > {med:.4f})</div>
    <div class="rc-sub">Score was predictive — N = {ir['high']['ALL']['n']:,}</div>
    {row('LONG avg P&amp;L',  'high', 'LONG')}
    {row('SHORT avg P&amp;L', 'high', 'SHORT')}
    {row('Portfolio return',  'high', 'ALL')}
  </div>
  <div class="rcard low-ic">
    <div class="rc-title">❌ Low IC Periods &nbsp; (IC ≤ {med:.4f})</div>
    <div class="rc-sub">Score failed or reversed — N = {ir['low']['ALL']['n']:,}</div>
    {row('LONG avg P&amp;L',  'low', 'LONG')}
    {row('SHORT avg P&amp;L', 'low', 'SHORT')}
    {row('Portfolio return',  'low', 'ALL')}
  </div>
</div>"""

    def make_dates_html(h, variant):
        dates = sorted(all_variants[variant][h]["df"]["rebalance_date"].unique())
        return "\n".join(
            f'<option value="{d.strftime("%Y-%m-%d")}">{d.strftime("%Y-%m-%d")}</option>'
            for d in dates
        )

    def make_extremes_html(h, variant):
        d = all_variants[variant][h]
        eq = d["equity_df"].copy()
        df = d["df"].copy()
        rc = d["return_col"]

        eq = eq.dropna(subset=["date", "period_ret_net"]).copy()
        if not eq.empty:
            eq["date_str"] = pd.to_datetime(eq["date"]).dt.strftime("%Y-%m-%d")
            eq["net_pct"] = pd.to_numeric(eq["period_ret_net"], errors="coerce").fillna(0.0)
            eq["gross_pct"] = pd.to_numeric(eq["period_ret_gross"], errors="coerce").fillna(0.0)
            eq["cost_pct"] = eq["gross_pct"] - eq["net_pct"]
            best_days = eq.sort_values("net_pct", ascending=False).head(10)
            worst_days = eq.sort_values("net_pct", ascending=True).head(10)
        else:
            best_days = pd.DataFrame()
            worst_days = pd.DataFrame()

        df = df.dropna(subset=[rc, "ticker"]).copy()
        df["ret_pct"] = pd.to_numeric(df[rc], errors="coerce").fillna(0.0)
        df["rebalance_date_str"] = pd.to_datetime(df["rebalance_date"]).dt.strftime("%Y-%m-%d")
        best_stocks = df.sort_values("ret_pct", ascending=False).head(10)
        worst_stocks = df.sort_values("ret_pct", ascending=True).head(10)

        def render_day_rows(sub):
            if sub.empty:
                return '<tr><td colspan="4">No data</td></tr>'
            rows = []
            for _, row in sub.iterrows():
                cls = "pos" if row["net_pct"] >= 0 else "neg"
                rows.append(
                    f"<tr><td><strong>{row['date_str']}</strong></td>"
                    f"<td class=\"{cls}\">{row['net_pct']:+.2f}%</td>"
                    f"<td>{row['gross_pct']:+.2f}%</td>"
                    f"<td>{row['cost_pct']:+.2f}%</td></tr>"
                )
            return "".join(rows)

        def render_stock_rows(sub):
            if sub.empty:
                return '<tr><td colspan="6">No data</td></tr>'
            rows = []
            for _, row in sub.iterrows():
                cls = "pos" if row["ret_pct"] >= 0 else "neg"
                rank_val = row.get("rank")
                rank_txt = "—" if pd.isna(rank_val) else str(int(rank_val))
                rows.append(
                    f"<tr><td><strong>{row['ticker']}</strong></td>"
                    f"<td>{row['rebalance_date_str']}</td>"
                    f"<td>{row['position_type']}</td>"
                    f"<td>{rank_txt}</td>"
                    f"<td>{float(row['score']):.3f}</td>"
                    f"<td class=\"{cls}\">{row['ret_pct']:+.2f}%</td></tr>"
                )
            return "".join(rows)

        return f"""
<div class="g2">
  <div class="search-card">
    <div class="search-title">Best Net Days</div>
    <div class="search-sub">Top daily portfolio results after costs</div>
    <table><thead><tr><th>Date</th><th>Net</th><th>Gross</th><th>Cost</th></tr></thead><tbody>{render_day_rows(best_days)}</tbody></table>
  </div>
  <div class="search-card">
    <div class="search-title">Worst Net Days</div>
    <div class="search-sub">Bottom daily portfolio results after costs</div>
    <table><thead><tr><th>Date</th><th>Net</th><th>Gross</th><th>Cost</th></tr></thead><tbody>{render_day_rows(worst_days)}</tbody></table>
  </div>
</div>
<div class="g2">
  <div class="search-card">
    <div class="search-title">Best Stock Episodes</div>
    <div class="search-sub">Top individual position outcomes</div>
    <table><thead><tr><th>Ticker</th><th>Rebalance</th><th>Side</th><th>Rank</th><th>Score</th><th>P&amp;L</th></tr></thead><tbody>{render_stock_rows(best_stocks)}</tbody></table>
  </div>
  <div class="search-card">
    <div class="search-title">Worst Stock Episodes</div>
    <div class="search-sub">Bottom individual position outcomes</div>
    <table><thead><tr><th>Ticker</th><th>Rebalance</th><th>Side</th><th>Rank</th><th>Score</th><th>P&amp;L</th></tr></thead><tbody>{render_stock_rows(worst_stocks)}</tbody></table>
  </div>
</div>"""

    # Pre-generate all stats/regime/dates HTML
    stats_html_all = {}
    regime_html_all = {}
    dates_html_all = {}
    extremes_html_all = {}
    for variant in variants:
        stats_html_all[variant] = {h: make_stats_html(h, all_stats[variant]) for h in horizons}
        regime_html_all[variant] = {h: make_regime_html(h, variant) for h in horizons}
        dates_html_all[variant] = {h: make_dates_html(h, variant) for h in horizons}
        extremes_html_all[variant] = {h: make_extremes_html(h, variant) for h in horizons}

    # ── JavaScript blobs ─────────────────────────────────────────────────────
    charts_js   = f"const CHARTS = {json.dumps(all_charts)};"
    cross_js    = f"const CROSS_CHARTS = {json.dumps(all_cross)};"
    holdings_js = f"const HOLDINGS = {json.dumps(all_holdings)};"
    stats_js    = f"const STATS_HTML = {json.dumps(stats_html_all)};"
    regime_js   = f"const REGIME_HTML = {json.dumps(regime_html_all)};"
    dates_js    = f"const DATES_HTML = {json.dumps(dates_html_all)};"
    extremes_js = f"const EXTREMES_HTML = {json.dumps(extremes_html_all)};"

    # Variant and horizon buttons
    vbtn_html = "\n".join(
        f'<button class="hbtn {"active" if i == 0 else ""}" id="vbtn-{v}" '
        f'onclick="switchV(\'{v}\')">{v}</button>'
        for i, v in enumerate(variants)
    )
    hbtn_html = "\n".join(
        f'<button class="hbtn {"active" if i == 0 else ""}" id="hbtn-{h}" '
        f'onclick="switchH({h})">{h}d</button>'
        for i, h in enumerate(horizons)
    )
    # ── Full HTML ────────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-Horizon Portfolio Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh;}}
.hdr{{background:#161b22;padding:20px 32px;border-bottom:1px solid #30363d;display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;}}
.hdr h1{{font-size:1.5em;color:#58a6ff;}}
.hdr p{{color:#8b949e;font-size:.88em;}}
.hbar{{display:flex;gap:8px;padding:12px 32px;background:#0d1117;border-bottom:1px solid #21262d;flex-wrap:wrap;}}
.hbtn{{padding:7px 18px;border:2px solid #30363d;border-radius:6px;background:transparent;color:#8b949e;cursor:pointer;font-size:.9em;font-weight:600;transition:all .2s;}}
.hbtn:hover{{border-color:#58a6ff;color:#58a6ff;}}
.hbtn.active{{border-color:#58a6ff;background:#131d33;color:#58a6ff;}}
.wrap{{max-width:1600px;margin:0 auto;padding:20px 28px;}}
/* stats */
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:20px;}}
.scard{{background:#161b22;border:1px solid #30363d;border-radius:7px;padding:14px;}}
.slabel{{color:#8b949e;font-size:.74em;text-transform:uppercase;letter-spacing:.7px;margin-bottom:5px;}}
.sval{{font-size:1.35em;font-weight:700;color:#e6edf3;}}
.pos{{color:#3fb950;}}.neg{{color:#f85149;}}
/* sub tabs */
.stabbar{{display:flex;gap:0;border-bottom:2px solid #21262d;margin-bottom:18px;overflow-x:auto;}}
.stab{{padding:10px 20px;background:none;border:none;color:#8b949e;font-size:.9em;cursor:pointer;border-bottom:3px solid transparent;margin-bottom:-2px;transition:all .2s;white-space:nowrap;}}
.stab:hover{{color:#58a6ff;}}.stab.active{{color:#58a6ff;border-bottom-color:#58a6ff;font-weight:600;}}
.sc{{display:none;}}.sc.active{{display:block;}}
/* chart grid */
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;}}
.g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px;}}
.cbox{{background:#161b22;border:1px solid #30363d;border-radius:7px;overflow:visible;min-height:380px;}}
.cfull{{background:#161b22;border:1px solid #30363d;border-radius:7px;overflow:visible;margin-bottom:14px;min-height:420px;}}
/* regime */
.regime-meta{{color:#8b949e;font-size:.9em;margin-bottom:14px;}}
.regime-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px;}}
.rcard{{padding:22px;border-radius:8px;border:1px solid #30363d;}}
.high-ic{{background:linear-gradient(135deg,#0c2218,#163824);border-color:#238636;}}
.low-ic{{background:linear-gradient(135deg,#2d0d10,#451618);border-color:#da3633;}}
.rc-title{{font-size:1.05em;font-weight:700;margin-bottom:3px;}}
.rc-sub{{font-size:.8em;color:#8b949e;margin-bottom:14px;}}
.rrow{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.05);font-size:.9em;}}
.rrow:last-child{{border-bottom:none;font-weight:700;margin-top:2px;}}
/* holdings */
.controls{{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:18px;}}
.controls label{{font-weight:600;color:#8b949e;font-size:.85em;}}
.controls select,.controls input{{background:#161b22;border:1px solid #30363d;color:#e6edf3;padding:7px 11px;border-radius:6px;font-size:.88em;min-width:170px;}}
.pstats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:16px 0;}}
.pscard{{background:#161b22;border:1px solid #30363d;border-radius:7px;padding:12px;}}
.pslabel{{color:#8b949e;font-size:.72em;text-transform:uppercase;letter-spacing:.7px;margin-bottom:4px;}}
.psval{{font-size:1.25em;font-weight:700;}}
.search-results{{margin:12px 0 18px 0;}}
.search-card{{background:#161b22;border:1px solid #30363d;border-radius:7px;padding:14px;margin-bottom:12px;}}
.search-title{{font-size:1.02em;font-weight:700;margin-bottom:8px;}}
.search-sub{{font-size:.82em;color:#8b949e;margin-bottom:10px;}}
.search-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;}}
.search-grid table{{width:100%;border-collapse:collapse;font-size:.84em;}}
.search-grid th{{text-align:left;color:#8b949e;padding:6px 8px;border-bottom:1px solid #30363d;font-size:.75em;text-transform:uppercase;}}
.search-grid td{{padding:6px 8px;border-bottom:1px solid #21262d;}}
#htable-wrap table{{width:100%;border-collapse:collapse;font-size:.88em;margin-top:12px;}}
#htable-wrap th{{background:#161b22;padding:9px 11px;text-align:left;color:#8b949e;font-weight:600;border-bottom:2px solid #30363d;font-size:.78em;text-transform:uppercase;letter-spacing:.5px;}}
#htable-wrap td{{padding:8px 11px;border-bottom:1px solid #21262d;}}
#htable-wrap tr:hover td{{background:#161b22;}}
@media(max-width:900px){{.g2,.g3,.regime-grid{{grid-template-columns:1fr;}}.cbox,.cfull{{min-height:340px;}}}}
@media(max-width:900px){{.search-grid{{grid-template-columns:1fr;}}}}
</style>
</head>
<body>
<div class="hdr">
  <h1>📊 Multi-Horizon Portfolio Dashboard</h1>
  <p>Variants: {', '.join(variants)} &nbsp;·&nbsp; Horizons: {', '.join(f'{h}d' for h in horizons)} &nbsp;·&nbsp; {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
<div class="hbar" style="display:flex; gap:24px; padding:12px 32px; background:#0d1117; border-bottom:1px solid #21262d;">
  <div style="flex-shrink:0;">
    <span style="color:#8b949e; font-size:.85em; font-weight:600;">Portfolio Size:</span>
    <div style="display:flex; gap:8px; margin-top:6px;">{vbtn_html}</div>
  </div>
  <div style="flex-shrink:0;">
    <span style="color:#8b949e; font-size:.85em; font-weight:600;">Horizon:</span>
    <div style="display:flex; gap:8px; margin-top:6px; flex-wrap:wrap;">{hbtn_html}</div>
  </div>
</div>
<div class="wrap">
  <div id="stats-wrap"></div>
  <div class="stabbar">
    <button class="stab active" onclick="switchTab(event,'overview')">📈 Overview</button>
    <button class="stab" onclick="switchTab(event,'signal')">🔬 Signal Quality</button>
    <button class="stab" onclick="switchTab(event,'costs')">💸 Transaction Costs</button>
    <button class="stab" onclick="switchTab(event,'regimes')">🎯 IC Regimes</button>
    <button class="stab" onclick="switchTab(event,'deciles')">📊 Deciles</button>
    <button class="stab" onclick="switchTab(event,'extremes')">🔥 Best & Worst</button>
    <button class="stab" onclick="switchTab(event,'crosshorizon')">📐 Cross-Horizon</button>
    <button class="stab" onclick="switchTab(event,'holdings')">📋 Holdings Explorer</button>
  </div>
  <!-- Overview -->
  <div id="sc-overview" class="sc active">
    <div class="g2">
      <div class="cbox" id="c-equity"></div>
    </div>
    <div class="g2">
      <div class="cbox" id="c-pret"></div>
    </div>
    <div class="g2">
      <div class="cbox" id="c-ls"></div>
      <div class="cbox" id="c-scatter"></div>
    </div>
    <div class="g2">
      <div class="cbox" id="c-hilo"></div>
    </div>
    <div class="cfull" id="c-ric"></div>
  </div>
  <!-- Signal Quality -->
  <div id="sc-signal" class="sc">
    <div class="cfull" id="c-rank-stability"></div>
  </div>
  <!-- Transaction Costs -->
  <div id="sc-costs" class="sc">
    <div class="cfull" id="c-cost-impact"></div>
    <div class="cfull" id="c-impl-shortfall"></div>
    <div class="cfull" id="c-turnover-attr"></div>
    <div class="g2">
      <div class="cbox" id="c-turnover-costs"></div>
      <div class="cbox" id="c-cost-pct"></div>
    </div>
  </div>
  <!-- IC Regimes -->
  <div id="sc-regimes" class="sc">
    <div id="regime-wrap"></div>
    <div class="cfull" id="c-icreg"></div>
  </div>
  <!-- Deciles -->
  <div id="sc-deciles" class="sc">
    <div class="cfull" id="c-deciles"></div>
  </div>
  <!-- Best / Worst -->
  <div id="sc-extremes" class="sc">
    <div id="extremes-wrap"></div>
  </div>
  <!-- Cross-Horizon -->
  <div id="sc-crosshorizon" class="sc">
    <div class="cfull" id="c-alpha-decay"></div>
    <div class="cfull" id="c-horizon-eff"></div>
  </div>
  <!-- Holdings Explorer -->
  <div id="sc-holdings" class="sc">
    <div class="controls">
      <label>Rebalance Date:</label>
      <select id="dpicker" onchange="onDate()">
        <option value="">— select date —</option>
      </select>
      <label>Sort by:</label>
      <select id="spicker" onchange="renderTable()">
        <option value="rank">Rank</option>
        <option value="ret_d">P&L ↓</option>
        <option value="ret_a">P&L ↑</option>
        <option value="score_d">Score ↓</option>
      </select>
      <label>Ticker Search:</label>
      <input id="tpicker" type="text" placeholder="e.g. MAXX, AAPL" oninput="onTickerSearch()" />
    </div>
    <div class="pstats" id="pstats" style="display:none;"></div>
    <div class="search-results" id="search-results"></div>
    <div class="cfull" id="c-hscatter"></div>
    <div id="htable-wrap"></div>
  </div>
</div>

<script>
{charts_js}
{cross_js}
{holdings_js}
{stats_js}
{regime_js}
{dates_js}
{extremes_js}

var V = '{first_v}';
var H = {first_h};
var activeTab = 'overview';

function switchV(v) {{
  V = v;
  // Update active button highlighting
  document.querySelectorAll('[id^="vbtn-"]').forEach(b => b.classList.remove('active'));
  document.getElementById('vbtn-' + v).classList.add('active');
  
  document.getElementById('stats-wrap').innerHTML = STATS_HTML[V][H];
  document.getElementById('regime-wrap').innerHTML = REGIME_HTML[V][H];
  document.getElementById('extremes-wrap').innerHTML = EXTREMES_HTML[V][H];
  var dp = document.getElementById('dpicker');
  dp.innerHTML = '<option value="">— select date —</option>' + DATES_HTML[V][H];
  document.getElementById('pstats').style.display = 'none';
  document.getElementById('search-results').innerHTML = '';
  document.getElementById('tpicker').value = '';
  document.getElementById('htable-wrap').innerHTML = '';
  document.getElementById('c-hscatter').innerHTML = '';
  renderCharts();
  window.dispatchEvent(new Event('resize'));
}}

function switchH(h) {{
  H = h;
  document.querySelectorAll('.hbtn').forEach(b => b.classList.remove('active'));
  document.getElementById('hbtn-' + h).classList.add('active');
  document.getElementById('stats-wrap').innerHTML = STATS_HTML[V][H];
  document.getElementById('regime-wrap').innerHTML = REGIME_HTML[V][H];
  document.getElementById('extremes-wrap').innerHTML = EXTREMES_HTML[V][H];
  var dp = document.getElementById('dpicker');
  dp.innerHTML = '<option value="">— select date —</option>' + DATES_HTML[V][H];
  document.getElementById('pstats').style.display = 'none';
  document.getElementById('search-results').innerHTML = '';
  document.getElementById('tpicker').value = '';
  document.getElementById('htable-wrap').innerHTML = '';
  document.getElementById('c-hscatter').innerHTML = '';
  renderCharts();
  window.dispatchEvent(new Event('resize'));
}}

function switchTab(evt, tab) {{
  document.querySelectorAll('.stab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.sc').forEach(c => c.classList.remove('active'));
  evt.currentTarget.classList.add('active');
  document.getElementById('sc-' + tab).classList.add('active');
  activeTab = tab;
  renderCharts();
  window.dispatchEvent(new Event('resize'));
}}

function renderCharts() {{
  var c = CHARTS[V][H];
  var cfg = {{responsive: true}};
  var ids = [];
  if (activeTab === 'overview') {{
    Plotly.react('c-equity',   c.equity.data,                 c.equity.layout,                 cfg);
    Plotly.react('c-ric',      c.rolling_ic.data,             c.rolling_ic.layout,             cfg);
    Plotly.react('c-pret',     c.period_ret.data,             c.period_ret.layout,             cfg);
    Plotly.react('c-ls',       c.ls.data,                     c.ls.layout,                     cfg);
    Plotly.react('c-scatter',  c.scatter.data,                c.scatter.layout,                cfg);
    Plotly.react('c-hilo',     c.hi_lo.data,                  c.hi_lo.layout,                  cfg);
    ids = ['c-equity','c-ric','c-pret','c-ls','c-scatter','c-hilo'];
  }} else if (activeTab === 'signal') {{
    Plotly.react('c-rank-stability', c.rank_stability.data, c.rank_stability.layout, cfg);
    ids = ['c-rank-stability'];
  }} else if (activeTab === 'costs') {{
    Plotly.react('c-cost-impact',    c.cost_impact.data,       c.cost_impact.layout,       cfg);
    Plotly.react('c-impl-shortfall', c.impl_shortfall.data,    c.impl_shortfall.layout,    cfg);
    Plotly.react('c-turnover-attr',  c.turnover_attr.data,     c.turnover_attr.layout,     cfg);
    Plotly.react('c-turnover-costs', c.turnover_costs.data,    c.turnover_costs.layout,    cfg);
    Plotly.react('c-cost-pct',       c.cost_pct.data,          c.cost_pct.layout,          cfg);
    ids = ['c-cost-impact', 'c-impl-shortfall', 'c-turnover-attr', 'c-turnover-costs', 'c-cost-pct'];
  }} else if (activeTab === 'regimes') {{
    document.getElementById('regime-wrap').innerHTML = REGIME_HTML[V][H];
    Plotly.react('c-icreg',   c.ic_regimes.data, c.ic_regimes.layout, cfg);
    ids = ['c-icreg'];
  }} else if (activeTab === 'deciles') {{
    Plotly.react('c-deciles', c.deciles.data,    c.deciles.layout,    cfg);
    ids = ['c-deciles'];
  }} else if (activeTab === 'extremes') {{
    document.getElementById('extremes-wrap').innerHTML = EXTREMES_HTML[V][H];
  }} else if (activeTab === 'crosshorizon') {{
    var vc = CROSS_CHARTS[V];
    if (vc && vc.alpha_decay) {{
      Plotly.react('c-alpha-decay',  vc.alpha_decay.data,  vc.alpha_decay.layout,  cfg);
      Plotly.react('c-horizon-eff',  vc.horizon_eff.data,  vc.horizon_eff.layout,  cfg);
      ids = ['c-alpha-decay', 'c-horizon-eff'];
    }}
  }}
  // Force relayout after a tick so Plotly picks up actual container sizes
  setTimeout(function() {{
    ids.forEach(function(id) {{
      var el = document.getElementById(id);
      if (el && el.data) Plotly.Plots.resize(el);
    }});
  }}, 50);
}}

function onDate() {{
  renderTable();
  renderHoldingsChart();
}}

function onTickerSearch() {{
  renderTable();
  renderSearchResults();
}}

function renderHoldingsChart() {{
  var date = document.getElementById('dpicker').value;
  if (!date) return;
  var dd = HOLDINGS[V][H][date];
  if (!dd) return;
  var ch = dd.chart_json;
  Plotly.react('c-hscatter', ch.data, ch.layout, {{responsive: true}});
}}

function renderTable() {{
  var date = document.getElementById('dpicker').value;
  if (!date) return;
  var dd = HOLDINGS[V][H][date];
  if (!dd) {{ document.getElementById('htable-wrap').innerHTML = '<p style="color:#8b949e;padding:16px;">No data.</p>'; return; }}

  var s = dd.stats;
  var pnlCls = function(v) {{ return v >= 0 ? 'pos' : 'neg'; }};
  var fmtP   = function(v) {{ return (v >= 0 ? '+' : '') + v.toFixed(2) + '%'; }};
  var icStr  = s.ic !== null && s.ic !== undefined ? s.ic.toFixed(4) : 'N/A';

  document.getElementById('pstats').innerHTML =
    '<div class="pscard"><div class="pslabel">Portfolio Return</div><div class="psval ' + pnlCls(s.portfolio_ret) + '">' + fmtP(s.portfolio_ret) + '</div></div>' +
    '<div class="pscard"><div class="pslabel">Long P&L</div><div class="psval ' + pnlCls(s.long_ret) + '">' + fmtP(s.long_ret) + '</div></div>' +
    '<div class="pscard"><div class="pslabel">Short P&L</div><div class="psval ' + pnlCls(s.short_ret) + '">' + fmtP(s.short_ret) + '</div></div>' +
    '<div class="pscard"><div class="pslabel">Win Rate</div><div class="psval">' + (s.win_rate * 100).toFixed(1) + '%</div></div>' +
    '<div class="pscard"><div class="pslabel">Avg Score</div><div class="psval">' + s.avg_score.toFixed(3) + '</div></div>' +
    '<div class="pscard"><div class="pslabel">Cross-Sect. IC</div><div class="psval">' + icStr + '</div></div>' +
    '<div class="pscard"><div class="pslabel">N Positions</div><div class="psval">' + s.n + '</div></div>';
  document.getElementById('pstats').style.display = 'grid';

  var holdings = dd.holdings.slice();
  var query = document.getElementById('tpicker').value.trim().toUpperCase();
  if (query) {{
    holdings = holdings.filter(function(h) {{ return (h.ticker || '').toUpperCase().includes(query); }});
  }}
  var sort = document.getElementById('spicker').value;
  if (sort === 'rank')    holdings.sort((a,b) => (a.rank ?? 999999) - (b.rank ?? 999999));
  if (sort === 'ret_d')   holdings.sort((a,b) => b.pnl - a.pnl);
  if (sort === 'ret_a')   holdings.sort((a,b) => a.pnl - b.pnl);
  if (sort === 'score_d') holdings.sort((a,b) => b.score - a.score);

  if (holdings.length === 0) {{
    document.getElementById('htable-wrap').innerHTML = '<p style="color:#8b949e;padding:16px;">No holdings match this ticker/date filter.</p>';
    return;
  }}

  var html = '<table><thead><tr><th>Rank</th><th>Ticker</th><th>Side</th><th>Score</th><th>Entry</th><th>Exit</th><th>Raw Ret</th><th>P&L</th></tr></thead><tbody>';
  holdings.forEach(function(h) {{
    var pc = h.pnl >= 0 ? 'pos' : 'neg';
    var rc = h.raw_ret >= 0 ? 'pos' : 'neg';
    var rankText = h.rank === null || h.rank === undefined ? '—' : h.rank;
    html += '<tr><td>' + rankText + '</td><td><strong>' + h.ticker + '</strong></td>' +
            '<td>' + h.position_type + '</td>' +
            '<td>' + h.score.toFixed(3) + '</td>' +
            '<td>$' + h.entry_price.toFixed(2) + '</td>' +
            '<td>$' + h.exit_price.toFixed(2) + '</td>' +
            '<td class="' + rc + '">' + (h.raw_ret >= 0 ? '+' : '') + h.raw_ret.toFixed(2) + '%</td>' +
            '<td class="' + pc + '">' + (h.pnl >= 0 ? '+' : '') + h.pnl.toFixed(2) + '%</td></tr>';
  }});
  html += '</tbody></table>';
  document.getElementById('htable-wrap').innerHTML = html;
}}

function renderSearchResults() {{
  var query = document.getElementById('tpicker').value.trim().toUpperCase();
  var wrap = document.getElementById('search-results');
  if (!query) {{
    wrap.innerHTML = '';
    return;
  }}
  var idx = HOLDINGS[V][H]['__ticker_index__'] || {{}};
  var matches = Object.keys(idx).filter(function(t) {{ return t.includes(query); }}).sort();
  if (matches.length === 0) {{
    wrap.innerHTML = '<div class="search-card"><div class="search-title">Ticker Search</div><div class="search-sub">No ticker matches \"' + query + '\" in this variant/horizon.</div></div>';
    return;
  }}

  function renderRows(rows) {{
    return rows.map(function(r) {{
      var pnlCls = r.pnl >= 0 ? 'pos' : 'neg';
      var rankText = r.rank === null || r.rank === undefined ? '—' : r.rank;
      return '<tr>' +
        '<td><strong>' + r.rebalance_date + '</strong></td>' +
        '<td>' + r.position_type + '</td>' +
        '<td>' + rankText + '</td>' +
        '<td>' + r.score.toFixed(3) + '</td>' +
        '<td class="' + pnlCls + '">' + (r.pnl >= 0 ? '+' : '') + r.pnl.toFixed(2) + '%</td>' +
        '<td>$' + r.entry_price.toFixed(2) + ' → $' + r.exit_price.toFixed(2) + '</td>' +
      '</tr>';
    }}).join('');
  }}

  var html = '';
  matches.slice(0, 8).forEach(function(ticker) {{
    var rows = idx[ticker].slice();
    rows.sort(function(a,b) {{ return b.pnl - a.pnl; }});
    var best = rows.slice(0, 5);
    var worst = rows.slice().sort(function(a,b) {{ return a.pnl - b.pnl; }}).slice(0, 5);
    var avg = rows.reduce(function(acc, r) {{ return acc + r.pnl; }}, 0) / Math.max(rows.length, 1);
    html += '<div class="search-card">' +
      '<div class="search-title">' + ticker + '</div>' +
      '<div class="search-sub">Occurrences: ' + rows.length + ' · Average P&L: ' + (avg >= 0 ? '+' : '') + avg.toFixed(2) + '%</div>' +
      '<div class="search-grid">' +
      '<div><div class="search-sub">Best Winners</div><table><thead><tr><th>Date</th><th>Side</th><th>Rank</th><th>Score</th><th>P&L</th><th>Prices</th></tr></thead><tbody>' + renderRows(best) + '</tbody></table></div>' +
      '<div><div class="search-sub">Worst Losers</div><table><thead><tr><th>Date</th><th>Side</th><th>Rank</th><th>Score</th><th>P&L</th><th>Prices</th></tr></thead><tbody>' + renderRows(worst) + '</tbody></table></div>' +
      '</div></div>';
  }});
  if (matches.length > 8) {{
    html += '<div class="search-card"><div class="search-sub">Showing first 8 ticker matches out of ' + matches.length + '.</div></div>';
  }}
  wrap.innerHTML = html;
}}

// Boot
(function() {{
  document.getElementById('stats-wrap').innerHTML = STATS_HTML[V][H];
  document.getElementById('regime-wrap').innerHTML = REGIME_HTML[V][H];
  document.getElementById('extremes-wrap').innerHTML = EXTREMES_HTML[V][H];
  document.getElementById('dpicker').innerHTML = '<option value="">— select date —</option>' + DATES_HTML[V][H];
  document.getElementById('search-results').innerHTML = '';
  renderCharts();
}})();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML dashboard from factor backtest output folders."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "Optional explicit paths to result folders or portfolio_selections.csv files. "
            "You can pass a variant dir, a 10d_horizon dir, a direction_score dir, or portfolio_selections.csv."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "dashboard.html"),
        help="Output HTML path (default: backtest_results/analysis_output/dashboard.html)",
    )
    args = parser.parse_args()

    print("\n📊 Multi-Horizon Portfolio Dashboard Generator")
    print("=" * 60)
    if args.inputs:
        print("\n🔍 Using explicit input paths...")
        discovered = discover_data_from_paths(args.inputs)
    else:
        print(f"\n🔍 Scanning {BACKTEST_DIR}...")
        discovered = discover_data()
    if not discovered:
        print("❌ No data found. Run factor_backtest_engine.py first.")
        sys.exit(1)

    all_variants = {}  # {variant_name: {horizon: {...}}}

    for variant_name in sorted(discovered.keys()):
        print(f"\n🔄 Processing variant: {variant_name}")
        print("-" * 60)
        
        variant_data = {}
        
        for h in sorted(discovered[variant_name].keys()):
            csv_path = discovered[variant_name][h]
            print(f"\n⏳ [{variant_name} / {h}d] Loading {csv_path.name}...")

            try:
                df, return_col = load_csv(csv_path, h)
                print(f"   {len(df):,} positions · {df['rebalance_date'].nunique()} rebalance dates")

                score_dir = csv_path.parent
                pnl_df, review_log_df = load_engine_outputs(score_dir)

                if pnl_df is not None:
                    print(f"   Building equity curve from factor_pnl.csv...")
                    equity_df = build_equity_curve_from_pnl(pnl_df)
                else:
                    print(f"   Building equity curve from portfolio_selections.csv...")
                    equity_df = build_equity_curve(df, return_col)
                total_ret = float(equity_df["cumulative_return_pct"].iloc[-1])
                print(f"   Total return: {total_ret:+.2f}%")

                print(f"   Computing rolling IC...")
                ic_df = compute_rolling_ic(df)

                print(f"   Computing decile analysis...")
                decile_df = compute_decile_analysis(df, return_col)

                print(f"   Computing IC regime stats...")
                ic_regime_stats = compute_ic_regime_stats(df, ic_df, return_col)

                print(f"   Computing rank stability...")
                rank_stability_df = compute_rank_stability(df)

                print(f"   Computing turnover attribution...")
                if review_log_df is not None:
                    turnover_attr_df = compute_turnover_attribution_from_review_log(review_log_df)
                else:
                    turnover_attr_df = compute_turnover_attribution(df)

                print(f"   Building holdings data...")
                holdings_data = build_holdings_data(df, ic_df, return_col)

                variant_data[h] = {
                    "df":                df,
                    "return_col":        return_col,
                    "equity_df":         equity_df,
                    "pnl_df":            pnl_df,
                    "review_log_df":     review_log_df,
                    "ic_df":             ic_df,
                    "decile_df":         decile_df,
                    "ic_regime_stats":   ic_regime_stats,
                    "rank_stability_df": rank_stability_df,
                    "turnover_attr_df":  turnover_attr_df,
                    "holdings_data":     holdings_data,
                }

            except Exception as e:
                print(f"   ❌ Error: {e}")
                traceback.print_exc()
                continue
        
        if variant_data:
            all_variants[variant_name] = variant_data

    if not all_variants:
        print("❌ No variants loaded.")
        sys.exit(1)

    print(f"\n🎨 Generating dashboard HTML with {len(all_variants)} variants...")
    html = build_dashboard(all_variants)

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"✅ Saved → {out}")
    print(f"   Size: {out.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
