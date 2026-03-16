"""
Parameter Sweep Tester - Test all technical indicators across multiple horizons
to find which indicators have the best predictive power.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.stats import spearmanr
import warnings

# Suppress pandas FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from hit_rate_validator import calculate_forward_return_label
from technical_indicators_calculator import TechnicalIndicatorCalculator
from indicator_orthogonality import IndicatorOrthogonalityComputer

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
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
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


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan, np.nan
    corr, p = spearmanr(x, y, nan_policy="omit")
    return float(corr), float(p)


def hit_rate_directional(
    indicator: pd.Series,
    target: pd.Series,
    center: Optional[float] = None,
    drop_zeros: bool = True,
) -> float:
    """
    Directional hit-rate:
      sign(indicator - center) == sign(target)

    - If center is None, uses median(indicator).
    - If drop_zeros True, rows with sign==0 on either side are removed.
    """
    if center is None:
        center = float(indicator.median())

    ind_sign = np.sign(indicator - center)
    tgt_sign = np.sign(target)

    if drop_zeros:
        mask = (ind_sign != 0) & (tgt_sign != 0)
        if mask.sum() == 0:
            return np.nan
        ind_sign = ind_sign[mask]
        tgt_sign = tgt_sign[mask]

    return float((ind_sign == tgt_sign).mean() * 100.0)


def compute_decile_spreads(
    indicator: pd.Series,
    target: pd.Series,
    n_deciles: int = 10,
) -> Dict[str, float]:
    """
    Compute return spread between top and bottom deciles.
    
    Returns:
      - 'top_decile_return': mean return of top decile
      - 'bottom_decile_return': mean return of bottom decile
      - 'spread': difference
      - 'spread_t_stat': t-statistic of spread
    """
    df = pd.DataFrame({"ind": indicator, "ret": target}).dropna()
    if len(df) < 20:
        return {"spread": np.nan, "top_decile_return": np.nan, "bottom_decile_return": np.nan, "spread_t_stat": np.nan}
    
    df["decile"] = pd.qcut(df["ind"], q=n_deciles, labels=False, duplicates="drop")
    
    top_ret = df[df["decile"] == df["decile"].max()]["ret"].mean()
    bot_ret = df[df["decile"] == df["decile"].min()]["ret"].mean()
    spread = top_ret - bot_ret
    
    # t-stat of spread
    top_vals = df[df["decile"] == df["decile"].max()]["ret"].values
    bot_vals = df[df["decile"] == df["decile"].min()]["ret"].values
    
    if len(top_vals) > 1 and len(bot_vals) > 1:
        pooled_std = np.sqrt(
            ((len(top_vals) - 1) * np.var(top_vals, ddof=1) + (len(bot_vals) - 1) * np.var(bot_vals, ddof=1))
            / (len(top_vals) + len(bot_vals) - 2)
        )
        if pooled_std > 0:
            t_stat = spread / (pooled_std * np.sqrt(1.0 / len(top_vals) + 1.0 / len(bot_vals)))
        else:
            t_stat = np.nan
    else:
        t_stat = np.nan
    
    return {
        "spread": float(spread),
        "top_decile_return": float(top_ret),
        "bottom_decile_return": float(bot_ret),
        "spread_t_stat": float(t_stat),
    }


def classify_vol_regime(hist: pd.DataFrame, date: pd.Timestamp, window: int = 21) -> str:
    """
    Classify vol regime at date into percentile buckets.
    - 'low': bottom 33%
    - 'medium': middle 33%
    - 'high': top 33%
    """
    idx_date = hist.index.get_loc(date)
    if idx_date < window + 252:
        return "medium"  # Default if not enough history
    
    # Compute realized vol over window (use adjusted close if available)
    price_col = "Adj_Close" if "Adj_Close" in hist.columns and hist["Adj_Close"].notna().any() else "Close"
    returns = hist[price_col].pct_change()
    recent_vol = returns.iloc[idx_date - window + 1 : idx_date + 1].std() * np.sqrt(252)
    
    # Compute historical vol percentiles
    hist_vols = []
    for i in range(252):
        end_idx = idx_date - i
        start_idx = end_idx - window + 1
        if start_idx >= 0:
            h_vol = returns.iloc[start_idx : end_idx + 1].std() * np.sqrt(252)
            if not np.isnan(h_vol):
                hist_vols.append(h_vol)
    
    if len(hist_vols) < 50:
        return "medium"
    
    p33 = np.percentile(hist_vols, 33)
    p67 = np.percentile(hist_vols, 67)
    
    if recent_vol <= p33:
        return "low"
    elif recent_vol >= p67:
        return "high"
    else:
        return "medium"


def classify_trend_regime(hist: pd.DataFrame, date: pd.Timestamp, ma_period: int = 200) -> str:
    """
    Classify trend regime: above or below MA200.
    - 'uptrend': price > MA200
    - 'downtrend': price < MA200
    """
    idx_date = hist.index.get_loc(date)
    if idx_date < ma_period:
        return "neutral"
    
    # Use adjusted close if available
    price_col = "Adj_Close" if "Adj_Close" in hist.columns and hist["Adj_Close"].notna().any() else "Close"
    ma = hist[price_col].iloc[: idx_date + 1].rolling(window=ma_period).mean()
    price = hist[price_col].iloc[idx_date]
    ma_val = ma.iloc[idx_date]
    
    if pd.isna(ma_val):
        return "neutral"
    
    if price > ma_val * 1.01:  # Buffer
        return "uptrend"
    elif price < ma_val * 0.99:
        return "downtrend"
    else:
        return "neutral"


def bh_fdr_qvalues(pvals: pd.Series) -> pd.Series:
    """
    Benjamini–Hochberg FDR correction (q-values).
    Returns q-values aligned to pvals index. NaNs preserved.
    """
    p = pvals.copy()
    nan_mask = p.isna()
    p_nonan = p[~nan_mask].astype(float)

    if len(p_nonan) == 0:
        return pd.Series(np.nan, index=p.index)

    order = np.argsort(p_nonan.values)
    ranked = p_nonan.values[order]
    m = len(ranked)

    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = (ranked[i] * m) / rank
        prev = min(prev, val)
        q[i] = prev

    qvals = np.empty(m, dtype=float)
    qvals[order] = q

    out = pd.Series(np.nan, index=p.index, dtype=float)
    out.loc[p_nonan.index] = qvals
    return out


# ----------------------------
# Main class
# ----------------------------
class ComprehensiveIndicatorSweepTester:
    """Test many different parameter configurations for technical indicators."""

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        forward_horizon: int = 1,
        execution_model: str = "next_open_to_open_hold",
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.forward_horizon = forward_horizon
        self.execution_model = execution_model
        self.hist: Optional[pd.DataFrame] = None
        self.results: List[Dict[str, Any]] = []
        self.orthogonality_computer = IndicatorOrthogonalityComputer()

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

        # Rename all required columns for hit_rate_validator + keep Adj_* if present
        df = df.rename(
            columns={
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
        )

        return df

    def _price_series(self) -> pd.Series:
        assert self.hist is not None
        if "Adj_Close" in self.hist.columns and self.hist["Adj_Close"].notna().any():
            return self.hist["Adj_Close"]
        return self.hist["Close"]

    def calculate_momentum(self, prices: pd.Series, lookback_days: int) -> pd.Series:
        """Calculate momentum over lookback period."""
        return (prices / prices.shift(lookback_days) - 1.0) * 100.0

    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI with given period."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        loss = loss.replace(0.0, np.nan)  # avoid div-by-zero spikes
        rs = gain / loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def calculate_ema(self, prices: pd.Series, span: int) -> pd.Series:
        """Calculate EMA."""
        return prices.ewm(span=span, adjust=False).mean()

    def calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate SMA."""
        return prices.rolling(window=window).mean()

    def calculate_macd(
        self, prices: pd.Series, fast: int, slow: int, signal: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD components."""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling volatility."""
        return returns.rolling(window=window).std() * np.sqrt(252.0) * 100.0

    def _build_test_df(
        self,
        indicator_series: pd.Series,
        start_idx: int,
        transform_target: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Align indicator values with forward returns.

        transform_target:
          - None: forward_return
          - "abs": abs(forward_return)
        """
        assert self.hist is not None
        test_data = []
        for date in self.hist.index[start_idx:]:
            if date not in indicator_series.index:
                continue
            v = indicator_series.loc[date]
            if pd.isna(v):
                continue

            fwd_ret = calculate_forward_return_label(
                self.hist, date, self.forward_horizon, self.execution_model
            )
            if fwd_ret is None or (isinstance(fwd_ret, float) and np.isnan(fwd_ret)):
                continue

            if transform_target == "abs":
                fwd_ret = abs(float(fwd_ret))

            test_data.append({"indicator_value": float(v), "forward_return": float(fwd_ret)})

        return pd.DataFrame(test_data)


    def test_all_technical_indicators(self):
        """Test all indicators from TechnicalIndicatorCalculator by building time series."""
        print("Testing all technical indicators (building time series)...")

        assert self.hist is not None

        # Comprehensive whitelist of indicators to test
        metric_whitelist = {
            'adx_14d', 'adx_14d_zscore', 'atr_14d', 'atr_14d_pct', 'autocorr_10d', 'autocorr_21d',
            'autocorr_5d', 'avg_dollar_vol_21d', 'avg_vol_3m', 'close_loc_10d', 'days_since_52w_high',
            'days_since_52w_low', 'directional_efficiency_10d', 'directional_efficiency_20d',
            'gap_size_pct', 'gap_size_zscore', 'intraday_ret', 'intraday_ret_zscore', 'liquidity_score_0_100',
            'ma10_distance_vol_normalized', 'ma10_distance_zscore', 'ma5_distance_vol_normalized',
            'ma5_distance_zscore', 'mom_1m', 'mom_1m_zscore', 'mom_3m', 'mom_3m_zscore',
            'pct_from_52w_high', 'pct_from_ma100', 'pct_from_ma20', 'pct_from_ma75',
            'pct_up_days_10d', 'pct_up_days_20d', 'pct_up_days_5d', 'range_pct', 'range_pct_zscore',
            'range_pos_20d', 'range_pos_55d', 'range_vol_impulse', 'realized_vol_10d', 'realized_vol_14d',
            'realized_vol_21d', 'ret_10d', 'ret_10d_zscore', 'ret_1d', 'ret_1d_zscore', 'ret_2d',
            'ret_2d_zscore', 'ret_3d', 'ret_3d_zscore', 'ret_5d', 'ret_5d_zscore', 'roll_spread_21',
            'rsi_2d', 'rsi_3d', 'rsi_4d', 'rsi_5d', 'stretch_pct_from_ma20', 'stretch_z_20', 'vel_10d',
            'vel_10d_over_vol', 'vel_5d', 'vel_5d_over_vol', 'vol_adjusted_trend_ma100',
            'vol_adjusted_trend_ma100_slope', 'vol_adjusted_trend_ma20', 'vol_adjusted_trend_ma20_slope',
            'vol_imbalance_10d', 'vol_imbalance_20d', 'vol_last', 'vol_percentile', 'vol_percentile_252d',
            'vol_percentile_504d', 'vol_roc_pct_20v10', 'vol_roc_ratio_20v10', 'volume_spike_z',
            # Bucket 7: Market Plumbing (VIX + Term Structure + Calendar)
            'vix_level', 'vix_percentile_252d', 'vix_roc_1d', 'vix_roc_5d', 'vix_roc_20d',
            'is_vix_low', 'is_vix_medium', 'is_vix_high',
            'm2_m1_ratio', 'm2_m1_ratio_percentile_52w', 'm2_roc_4w', 'm1_roc_4w',
            'is_expansion', 'is_neutral', 'is_contraction',
            'is_opex_normal', 'is_pre_opex', 'is_opex_day', 'is_post_opex',
        }

        indicator_series: Dict[str, Dict[str, Any]] = {}
        print("  Building indicator time series...")
        first_error_printed = False
        
        # Warmup = 600 days (504d percentile window + ~100d buffer for slower indicators)
        warmup = 600
        print(f"  Starting from index {warmup} (warmup: {warmup} days)")

        for idx in range(warmup, len(self.hist)):
            date = self.hist.index[idx]
            hist_subset = self.hist.iloc[: idx + 1]

            try:
                calc = TechnicalIndicatorCalculator(hist_subset)

                def collect(d: Dict[str, Any], typ: str, numeric_only: bool = True):
                    for ind_name, ind_value in d.items():
                        if ind_value is None:
                            continue
                        # EXCLUDE booleans (bool is a subclass of int in Python)
                        if isinstance(ind_value, (bool, np.bool_)):
                            continue
                        if numeric_only and not isinstance(ind_value, (int, float, np.floating, np.integer)):
                            continue
                        if ind_name not in indicator_series:
                            indicator_series[ind_name] = {"type": typ, "values": []}
                        indicator_series[ind_name]["values"].append((date, float(ind_value)))

                mom = calc.calculate_momentum()
                trend = calc.calculate_trend_indicators()
                mean_rev = calc.calculate_mean_reversion_indicators()
                trend_qual = calc.calculate_trend_quality_indicators()
                range_struct = calc.calculate_range_structure_indicators()
                vol = calc.calculate_volatility_indicators()
                vol_liq = calc.calculate_volume_liquidity_indicators()
                vol_part = calc.calculate_volume_participation_indicators()
                
                collect(mom, "momentum", numeric_only=True)
                collect(trend, "trend", numeric_only=True)
                collect(mean_rev, "mean_reversion", numeric_only=True)
                collect(trend_qual, "trend_quality", numeric_only=True)
                collect(range_struct, "range_structure", numeric_only=True)
                collect(vol, "volatility", numeric_only=True)
                collect(vol_liq, "volume_liquidity", numeric_only=True)
                collect(vol_part, "volume_participation", numeric_only=True)
                
                vix_indicators = calc.calculate_vix_indicators(date)
                term_structure_indicators = calc.calculate_term_structure_indicators(date)
                market_plumbing = {**vix_indicators, **term_structure_indicators}
                collect(market_plumbing, "market_plumbing", numeric_only=True)
                collect(calc.calculate_price_events(), "price_events", numeric_only=True)

            except Exception as e:
                if not first_error_printed:
                    print(f"    Warning: Error calculating indicators at idx={idx}: {str(e)[:140]}")
                    first_error_printed = True
                continue

        # Filter to whitelist metrics only
        print(f"  Total calculated: {len(indicator_series)} indicators")
        print(f"  Filtering to whitelist: {len(metric_whitelist)} metrics")
        indicator_series_filtered = {k: v for k, v in indicator_series.items() if k in metric_whitelist}
        print(f"  After filtering: {len(indicator_series_filtered)} metrics")
        indicator_series = indicator_series_filtered

        print("  Testing indicator correlations (directional + magnitude)...")
        tested_count = 0

        for ind_name, ind_data in indicator_series.items():
            if len(ind_data["values"]) < 100:
                continue

            dates = []
            vals = []
            for d, v in ind_data["values"]:
                if pd.isna(v):
                    continue
                dates.append(d)
                vals.append(v)

            if len(vals) < 100:
                continue

            # Build aligned test set
            test_data = []
            for date, ind_value in zip(dates, vals):
                fwd_ret = calculate_forward_return_label(
                    self.hist, date, self.forward_horizon, self.execution_model
                )
                if fwd_ret is None or (isinstance(fwd_ret, float) and np.isnan(fwd_ret)):
                    continue
                test_data.append({
                    "indicator_value": float(ind_value),
                    "forward_return": float(fwd_ret),
                    "abs_return": float(abs(fwd_ret)),
                    "date": date,
                })

            if len(test_data) < 100:
                continue

            test_df = pd.DataFrame(test_data)

            if test_df["indicator_value"].std() < 1e-10:
                continue

            # ===== DIRECTIONAL TEST =====
            corr_dir, p_val_dir = safe_spearman(
                test_df["indicator_value"].values, test_df["forward_return"].values
            )
            hit_dir = hit_rate_directional(
                test_df["indicator_value"],
                test_df["forward_return"],
                center=float(test_df["indicator_value"].median()),
            )
            decile_dir = compute_decile_spreads(test_df["indicator_value"], test_df["forward_return"])

            self.results.append(
                {
                    "indicator_type": ind_data["type"],
                    "parameter": ind_name,
                    "target": "direction",
                    "spearman_corr": corr_dir,
                    "p_value": p_val_dir,
                    "hit_rate": hit_dir,
                    "n_samples": len(test_df),
                    "abs_corr": abs(corr_dir) if np.isfinite(corr_dir) else np.nan,
                    "decile_spread": decile_dir.get("spread", np.nan),
                    "decile_spread_t": decile_dir.get("spread_t_stat", np.nan),
                }
            )
            tested_count += 1

            # ===== MAGNITUDE TEST =====
            corr_mag, p_val_mag = safe_spearman(
                test_df["indicator_value"].values, test_df["abs_return"].values
            )
            
            # Hit rate for magnitude: high indicator == large move
            mag_med = float(test_df["abs_return"].median())
            ind_med = float(test_df["indicator_value"].median())
            ind_sign = np.sign(test_df["indicator_value"] - ind_med)
            tgt_sign = np.sign(test_df["abs_return"] - mag_med)
            mask = (ind_sign != 0) & (tgt_sign != 0)
            hit_mag = float((ind_sign[mask] == tgt_sign[mask]).mean() * 100.0) if mask.sum() > 0 else np.nan
            
            decile_mag = compute_decile_spreads(test_df["indicator_value"], test_df["abs_return"])

            self.results.append(
                {
                    "indicator_type": ind_data["type"],
                    "parameter": ind_name,
                    "target": "magnitude",
                    "spearman_corr": corr_mag,
                    "p_value": p_val_mag,
                    "hit_rate": hit_mag,
                    "n_samples": len(test_df),
                    "abs_corr": abs(corr_mag) if np.isfinite(corr_mag) else np.nan,
                    "decile_spread": decile_mag.get("spread", np.nan),
                    "decile_spread_t": decile_mag.get("spread_t_stat", np.nan),
                }
            )
            tested_count += 1

        print(f"  ✓ Tested {tested_count} comprehensive technical indicators (direction + magnitude)")
        
        # Record all signals to orthogonality computer for correlation analysis
        print("  Recording signals for orthogonality analysis...")
        for ind_name, ind_data in indicator_series.items():
            if len(ind_data["values"]) >= 100:
                dates = [d for d, v in ind_data["values"] if not pd.isna(v)]
                vals = np.array([v for d, v in ind_data["values"] if not pd.isna(v)], dtype=float)
                if len(vals) >= 100:
                    self.orthogonality_computer.record_indicator_signal(ind_name, vals)

    def test_conditional_regimes(self):
        """Test indicators within vol & trend regimes."""
        print("Testing conditional IC (within vol + trend regimes)...")
        
        assert self.hist is not None
        if len(self.results) == 0:
            print("  Skipping: no results yet")
            return
        
        # Collect all indicator series again (reuse from test_all_technical_indicators if possible)
        # For now, we'll build a limited set for speed
        indicator_series: Dict[str, Dict[str, Any]] = {}
        
        print("  Building regime classifications...")
        regime_data = []
        
        for idx in range(252, len(self.hist)):
            date = self.hist.index[idx]
            vol_regime = classify_vol_regime(self.hist, date)
            trend_regime = classify_trend_regime(self.hist, date)
            regime_data.append({"date": date, "vol_regime": vol_regime, "trend_regime": trend_regime})
        
        regime_df = pd.DataFrame(regime_data)
        
        print("  Computing conditional ICs within regimes...")
        conditional_results = []
        
        # For top 30 indicators by abs_corr, compute conditional IC
        top_indicators = pd.DataFrame(self.results).nlargest(30, "abs_corr")
        
        for _, row in top_indicators.iterrows():
            ind_name = row["parameter"]
            target_type = row["target"]
            
            # Rebuild this specific indicator series (slow but thorough)
            try:
                indicator_vals = []
                for idx in range(252, len(self.hist)):
                    date = self.hist.index[idx]
                    hist_subset = self.hist.iloc[: idx + 1]
                    calc = TechnicalIndicatorCalculator(hist_subset)
                    
                    # Try to extract from all categories
                    for category_method in [
                        calc.calculate_momentum,
                        calc.calculate_trend_indicators,
                        calc.calculate_volatility_indicators,
                        calc.calculate_volume_liquidity_indicators,
                        calc.calculate_price_events,
                    ]:
                        try:
                            d = category_method()
                            if ind_name in d and d[ind_name] is not None:
                                indicator_vals.append((date, float(d[ind_name])))
                                break
                        except:
                            pass
            except:
                continue
            
            if len(indicator_vals) < 100:
                continue
            
            # Align with returns and regimes
            dates = [d for d, v in indicator_vals]
            vals = [v for d, v in indicator_vals]
            
            test_list = []
            for date, ind_val in zip(dates, vals):
                fwd_ret = calculate_forward_return_label(
                    self.hist, date, self.forward_horizon, self.execution_model
                )
                if fwd_ret is None or pd.isna(fwd_ret):
                    continue
                
                regime_row = regime_df[regime_df["date"] == date]
                if regime_row.empty:
                    continue
                
                test_list.append({
                    "ind_val": float(ind_val),
                    "ret": float(fwd_ret),
                    "abs_ret": float(abs(fwd_ret)),
                    "vol_regime": regime_row.iloc[0]["vol_regime"],
                    "trend_regime": regime_row.iloc[0]["trend_regime"],
                })
            
            if len(test_list) < 50:
                continue
            
            test_df = pd.DataFrame(test_list)
            
            # Compute IC within each vol regime
            for vol_r in ["low", "medium", "high"]:
                sub = test_df[test_df["vol_regime"] == vol_r]
                if len(sub) < 30:
                    continue
                
                target_col = "ret" if target_type == "direction" else "abs_ret"
                corr, pval = safe_spearman(sub["ind_val"].values, sub[target_col].values)
                
                conditional_results.append({
                    "indicator": ind_name,
                    "target": target_type,
                    "regime_type": "vol_regime",
                    "regime": vol_r,
                    "spearman_corr": corr,
                    "p_value": pval,
                    "n_samples": len(sub),
                })
            
            # Compute IC within each trend regime
            for tr in ["uptrend", "neutral", "downtrend"]:
                sub = test_df[test_df["trend_regime"] == tr]
                if len(sub) < 30:
                    continue
                
                target_col = "ret" if target_type == "direction" else "abs_ret"
                corr, pval = safe_spearman(sub["ind_val"].values, sub[target_col].values)
                
                conditional_results.append({
                    "indicator": ind_name,
                    "target": target_type,
                    "regime_type": "trend_regime",
                    "regime": tr,
                    "spearman_corr": corr,
                    "p_value": pval,
                    "n_samples": len(sub),
                })
        
        # Save conditional results
        if conditional_results:
            cond_df = pd.DataFrame(conditional_results)
            cond_path = f"backtest_results/conditional_ic_{self.symbol}_{self.forward_horizon}d.csv"
            cond_df.to_csv(cond_path, index=False)
            print(f"  ✓ Saved conditional IC: {cond_path}")
            
            # Summary print
            print(f"\n  CONDITIONAL IC SUMMARY (top 5 by absolute IC within each regime):")
            for regime_type in ["vol_regime", "trend_regime"]:
                sub = cond_df[cond_df["regime_type"] == regime_type]
                for regime in sub["regime"].unique():
                    r_sub = sub[sub["regime"] == regime].copy()
                    r_sub["abs_corr"] = r_sub["spearman_corr"].abs()
                    top = r_sub.nlargest(3, "abs_corr")
                    print(f"\n    {regime_type}={regime}:")
                    for _, row in top.iterrows():
                        print(f"      {row['indicator']:30s} | corr={row['spearman_corr']:+.4f} | n={row['n_samples']}")

    def run_all_tests(self):
        """Run all parameter sweep tests."""
        ensure_dir("backtest_results")

        print(f"\n{'='*80}")
        print(f"PARAMETER SWEEP - {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Forward horizon: {self.forward_horizon} day(s)")
        print(f"Execution model: {self.execution_model}")
        print(f"{'='*80}\n")

        print("Fetching data...")
        self.hist = self.fetch_tiingo_data(self.symbol, self.start_date, self.end_date)

        if self.hist is None or self.hist.empty:
            print(f"No data for {self.symbol}")
            return

        print(f"Loaded {len(self.hist)} days of data\n")

        # Comprehensive test: all indicators from all buckets
        self.test_all_technical_indicators()
        self.test_conditional_regimes()

        self.generate_report()
        self.plot_results()
        self.generate_orthogonality_analysis()

    def generate_report(self):
        """Generate summary report."""
        if not self.results:
            print("\n❌ NO RESULTS GENERATED - No parameters produced valid tests")
            print("This likely means insufficient data or all tests failed.")
            return

        results_df = pd.DataFrame(self.results)
        results_df = results_df.replace([np.inf, -np.inf], np.nan)

        # FDR (BH) within each (indicator_type, target) family
        results_df["q_value"] = np.nan
        for ind_type in results_df["indicator_type"].dropna().unique():
            for target in results_df[results_df["indicator_type"] == ind_type]["target"].dropna().unique():
                mask = (results_df["indicator_type"] == ind_type) & (results_df["target"] == target)
                results_df.loc[mask, "q_value"] = bh_fdr_qvalues(results_df.loc[mask, "p_value"])

        results_df = results_df.sort_values("abs_corr", ascending=False, na_position="last")

        print(f"\n{'='*80}")
        print("TOP 25 PARAMETER CONFIGURATIONS (by |ρ|)")
        print("=" * 80)

        display_cols = [
            "indicator_type",
            "parameter",
            "target",
            "spearman_corr",
            "p_value",
            "q_value",
            "hit_rate",
            "decile_spread",
            "decile_spread_t",
            "n_samples",
        ]
        print(results_df[display_cols].head(25).to_string(index=False))

        # Best by (type, target)
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION BY (INDICATOR TYPE, TARGET) PAIR")
        print("=" * 80)

        for ind_type in sorted(results_df["indicator_type"].dropna().unique()):
            for target in ["direction", "magnitude"]:
                sub = results_df[
                    (results_df["indicator_type"] == ind_type) & (results_df["target"] == target)
                ].copy()
                if sub.empty:
                    continue
                sub = sub.sort_values("abs_corr", ascending=False, na_position="last")
                best = sub.iloc[0]
                print(f"\n{ind_type.upper()} -> {target.upper()}:")
                print(f"  Best: {best['parameter']}")
                print(f"  Correlation: {best['spearman_corr']:+.4f} (p={best['p_value']:.4g}, q={best['q_value']:.4g})")
                print(f"  Hit rate: {best['hit_rate']:.1f}%")
                print(f"  Decile spread: {best['decile_spread']:+.6f} (t={best['decile_spread_t']:.2f})")
                print(f"  Samples: {int(best['n_samples'])}")

        # Significance
        print(f"\n{'='*80}")
        print("SIGNIFICANT CONFIGURATIONS")
        print("=" * 80)

        sig_p = results_df[results_df["p_value"] < 0.05].sort_values(
            "abs_corr", ascending=False, na_position="last"
        )
        sig_q = results_df[results_df["q_value"] < 0.05].sort_values(
            "abs_corr", ascending=False, na_position="last"
        )

        print(f"\nBy raw p < 0.05: {len(sig_p)} configs")
        if len(sig_p) > 0:
            print(sig_p[display_cols].head(15).to_string(index=False))
        else:
            print("  ❌ None")

        print(f"\nBy FDR q < 0.05 (BH within family): {len(sig_q)} configs")
        if len(sig_q) > 0:
            print(sig_q[display_cols].head(15).to_string(index=False))
        else:
            print("  ❌ None")

        # Analysis: which are vol vs direction predictors?
        print(f"\n{'='*80}")
        print("MAGNITUDE vs DIRECTION INSIGHTS")
        print("=" * 80)

        direction_df = results_df[results_df["target"] == "direction"].copy()
        magnitude_df = results_df[results_df["target"] == "magnitude"].copy()

        # Group by indicator, compare direction vs magnitude IC
        indicator_comparison = []
        for param in direction_df["parameter"].unique():
            dir_row = direction_df[direction_df["parameter"] == param]
            mag_row = magnitude_df[magnitude_df["parameter"] == param]

            if not dir_row.empty and not mag_row.empty:
                dir_ic = dir_row.iloc[0]["spearman_corr"]
                mag_ic = mag_row.iloc[0]["spearman_corr"]
                dir_spread = dir_row.iloc[0]["decile_spread"]
                mag_spread = mag_row.iloc[0]["decile_spread"]

                indicator_comparison.append({
                    "parameter": param,
                    "dir_ic": dir_ic,
                    "mag_ic": mag_ic,
                    "dir_spread": dir_spread,
                    "mag_spread": mag_spread,
                    "ic_ratio": abs(mag_ic) / abs(dir_ic) if abs(dir_ic) > 1e-6 else np.nan,
                })

        comp_df = pd.DataFrame(indicator_comparison)
        if not comp_df.empty:
            print("\nIndicators that predict MAGNITUDE better than DIRECTION:")
            vol_pred = comp_df[comp_df["ic_ratio"] > 1.5].sort_values("ic_ratio", ascending=False)
            if not vol_pred.empty:
                for _, row in vol_pred.head(10).iterrows():
                    print(
                        f"  {row['parameter']:30s} | "
                        f"dir_IC={row['dir_ic']:+.4f} | mag_IC={row['mag_ic']:+.4f} | "
                        f"ratio={row['ic_ratio']:.2f}"
                    )
            else:
                print("  None found")

        # Don't save regular CSV here - it will be saved with orthogonality metrics instead
        # (avoid creating duplicate files)

    def generate_orthogonality_analysis(self):
        """
        Compute signal orthogonality and add metrics to sweep results.
        Creates:
        - Correlation heatmap
        - Hierarchical clustering dendrogram
        - Single CSV with orthogonality columns (replaces raw results)
        """
        print("\n" + "="*80)
        print("SIGNAL ORTHOGONALITY ANALYSIS")
        print("="*80)

        # Compute correlations from recorded signals
        if self.orthogonality_computer.correlation_matrix is None:
            corr_matrix = self.orthogonality_computer.compute_correlations()
            if corr_matrix is None:
                print("⚠️  Insufficient data for orthogonality analysis")
                return

        # Compute per-indicator metrics
        metrics_df = self.orthogonality_computer.compute_orthogonality_metrics()

        if metrics_df is None or metrics_df.empty:
            print("⚠️  Could not compute orthogonality metrics")
            return

        # Create enriched results from scratch using self.results
        results_df = pd.DataFrame(self.results).replace([np.inf, -np.inf], np.nan)

        # Merge orthogonality metrics with sweep results
        enriched_df = results_df.merge(
            metrics_df,
            left_on="parameter",
            right_on="parameter",
            how="left"
        )

        # Save enriched results with standard name (not _WITH_ORTHOGONALITY)
        enriched_csv = f"backtest_results/parameter_sweep_{self.symbol}_{self.forward_horizon}d.csv"
        enriched_df.to_csv(enriched_csv, index=False)
        print(f"✅ Results saved: {enriched_csv}")

        # Generate visualizations
        self.orthogonality_computer.plot_heatmap(
            f"backtest_results/orthogonality_heatmap_{self.symbol}_{self.forward_horizon}d.png"
        )
        self.orthogonality_computer.plot_dendrogram(
            f"backtest_results/orthogonality_dendrogram_{self.symbol}_{self.forward_horizon}d.png"
        )

        # Print summary
        self.orthogonality_computer.print_summary()

        # Summary of orthogonality-informed selection
        print("\nTOP 10 INDICATORS BY PERFORMANCE + GOOD ORTHOGONALITY:")
        print("-"*80)
        
        # Create balance score: high IC but low redundancy
        enriched_df["balance_score"] = (
            enriched_df["abs_corr"].fillna(0) * 
            (1.0 - 0.3 * enriched_df["avg_abs_corr"].fillna(0.5))
        )
        
        top_balanced = enriched_df.nlargest(10, "balance_score")[
            ["indicator_type", "parameter", "target", "spearman_corr", 
             "hit_rate", "avg_abs_corr", "max_abs_corr", "top_collinear_partner"]
        ]
        
        print(top_balanced.to_string(index=False))
        print("\n" + "="*80)

    def plot_results(self):
        """Plot parameter sweep results: separate charts for MAGNITUDE vs DIRECTION."""
        if not self.results:
            print("⚠️  No results to plot")
            return

        ensure_dir("backtest_results")

        results_df = pd.DataFrame(self.results).replace([np.inf, -np.inf], np.nan)

        # Recompute q-values for plotting
        results_df["q_value"] = np.nan
        for target in ["direction", "magnitude"]:
            for ind_type in results_df[results_df["target"] == target]["indicator_type"].dropna().unique():
                mask = (results_df["indicator_type"] == ind_type) & (results_df["target"] == target)
                results_df.loc[mask, "q_value"] = bh_fdr_qvalues(results_df.loc[mask, "p_value"])

        # ===== DIRECTION CHART =====
        print("\nGenerating DIRECTION chart...")
        direction_df = results_df[results_df["target"] == "direction"].copy()
        self._plot_target_chart(direction_df, target="direction")

        # ===== MAGNITUDE CHART =====
        print("Generating MAGNITUDE chart...")
        magnitude_df = results_df[results_df["target"] == "magnitude"].copy()
        self._plot_target_chart(magnitude_df, target="magnitude")

    def _plot_target_chart(self, target_df: pd.DataFrame, target: str) -> None:
        """Plot results for a specific target (direction or magnitude)."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        indicator_types = list(target_df["indicator_type"].dropna().unique())

        for idx, ind_type in enumerate(indicator_types[: len(axes)]):
            ax = axes[idx]
            subset = target_df[target_df["indicator_type"] == ind_type].copy()

            # Choose x axis
            if ind_type == "momentum" and "lookback_days" in subset.columns:
                x_col, x_label = "lookback_days", "Lookback Days"
            elif ind_type == "rsi" and "period" in subset.columns:
                x_col, x_label = "period", "Period"
            elif ind_type == "sma_deviation" and "period" in subset.columns:
                x_col, x_label = "period", "Period"
            elif ind_type == "volatility" and "window" in subset.columns:
                x_col, x_label = "window", "Window"
            else:
                subset["config_idx"] = range(len(subset))
                x_col, x_label = "config_idx", "Configuration"

            subset = subset.dropna(subset=["spearman_corr"])
            if subset.empty:
                ax.set_title(f"{ind_type.upper()} (no valid points)")
                ax.axis("off")
                continue

            # Add parameter names as labels
            subset["label"] = subset["parameter"].astype(str)

            scatter = ax.scatter(
                subset[x_col],
                subset["spearman_corr"],
                s=150,
                alpha=0.65,
                c=subset["abs_corr"].fillna(0),
                cmap="RdYlGn" if target == "direction" else "Greens",
                vmin=0,
                vmax=max(0.15, float(subset["abs_corr"].max() if np.isfinite(subset["abs_corr"].max()) else 0.15)),
            )

            # Annotate each point with parameter name
            for _, row in subset.iterrows():
                ax.annotate(
                    row["label"],
                    xy=(row[x_col], row["spearman_corr"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                )

            # Highlight q < 0.05 (FDR) if present
            sig_subset = subset[(subset["q_value"] < 0.05)]
            if len(sig_subset) > 0:
                ax.scatter(
                    sig_subset[x_col],
                    sig_subset["spearman_corr"],
                    s=250,
                    alpha=0.9,
                    edgecolors="black",
                    linewidths=2.5,
                    facecolors="none",
                    label="q < 0.05 (FDR significant)",
                )
                ax.legend(fontsize=10)

            ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel("Spearman Correlation", fontsize=11)
            ax.set_title(f"{ind_type.upper()}", fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3)

            cbar = plt.colorbar(scatter, ax=ax, label="|ρ|")

        # Hide unused subplots
        for j in range(len(indicator_types), len(axes)):
            axes[j].axis("off")

        target_display = "MAGNITUDE PREDICTION" if target == "magnitude" else "DIRECTION PREDICTION"
        plt.suptitle(
            f"{self.symbol} - {target_display} ({self.forward_horizon}d, {self.execution_model})",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        plot_path = f"backtest_results/parameter_sweep_{self.symbol}_{self.forward_horizon}d_{target.upper()}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Saved: {plot_path}")
        plt.close()


def main():
    if len(sys.argv) < 4:
        print("Usage: python parameter_sweep_tester.py SYMBOL START_DATE END_DATE [FORWARD_HORIZONS] [EXECUTION_MODEL]")
        print()
        print("EXECUTION MODELS:")
        print("  'next_open_to_open_hold': Canonical no-lookahead hold [RECOMMENDED]")
        print("  'next_open_to_close': Intraday next-open to same-day close [legacy]")
        print("  'next_open_to_close_hold': Next-open to target-day close [legacy]")
        print()
        print("Example (canonical multi-day):")
        print("  python parameter_sweep_tester.py SPY 2015-01-01 2024-12-31 '1,3,5,10' next_open_to_open_hold")
        print()
        print("Example (intraday only):")
        print("  python parameter_sweep_tester.py SPY 2015-01-01 2024-12-31 '1' next_open_to_close")
        sys.exit(1)

    symbol = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]

    horizons_str = sys.argv[4] if len(sys.argv) > 4 else "1"
    try:
        forward_horizons = [int(h.strip()) for h in horizons_str.split(",")]
    except ValueError:
        print(f"Error: Invalid forward horizons '{horizons_str}'. Use comma-separated integers like '1,5,10'")
        sys.exit(1)

    execution_model = sys.argv[5] if len(sys.argv) > 5 else "next_open_to_open_hold"
    
    # Warn when the requested model differs from the canonical open-to-open hold.
    if any(h > 1 for h in forward_horizons) and execution_model in {"next_open_to_close", "next_open_to_close_hold"}:
        print(f"\n⚠️  WARNING: Using '{execution_model}' with multi-day horizons {forward_horizons}")
        print(f"   This will not match the canonical next-open to next-open holding convention.")
        print(f"   Use 'next_open_to_open_hold' for pipeline-consistent multi-day returns.\n")

    for horizon in forward_horizons:
        print(f"\n\n{'#'*80}")
        print(f"# Running sweep for {horizon}-day forward horizon")
        print(f"# Execution model: {execution_model}")
        print(f"{'#'*80}\n")

        tester = ComprehensiveIndicatorSweepTester(symbol, start_date, end_date, horizon, execution_model)
        tester.run_all_tests()

    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
