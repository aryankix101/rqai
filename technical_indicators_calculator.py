# technical_indicators.py

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import talib
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Suppress INFO logs during sweeps (called thousands of times)

# Moving average buffer to reduce regime oscillations
MA_BUFFER = 0.02

class TechnicalIndicatorCalculator:
    def __init__(self, hist: pd.DataFrame):
        if not isinstance(hist.index, pd.DatetimeIndex):
            hist.index = pd.to_datetime(hist.index)

        self.hist = hist.sort_index()

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing = required_cols - set(self.hist.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        # Helper to prefer adjusted columns when present
        def _col(name: str) -> str:
            adj = f"Adj_{name}"
            if adj in self.hist.columns and self.hist[adj].notna().any():
                return adj
            return name

        # Use adjusted prices if available (splits/dividends), else fallback to unadjusted
        self.close = self.hist[_col("Close")].values.astype(np.float64)
        self.high = self.hist[_col("High")].values.astype(np.float64)
        self.low = self.hist[_col("Low")].values.astype(np.float64)
        self.open = self.hist[_col("Open")].values.astype(np.float64)
        # NOTE: Volume is never adjusted (vendors don't provide Adj_Volume, and we want raw volume)
        self.volume = self.hist["Volume"].values.astype(np.float64)

        # Load VIX data from FRED VIXCLS.csv for market regime indicators
        self.vix_data = None
        vix_path = Path(__file__).parent / "VIXCLS.csv"
        if vix_path.exists():
            try:
                vix_df = pd.read_csv(vix_path, parse_dates=['observation_date'], index_col='observation_date')
                self.vix_data = vix_df['VIXCLS'].sort_index()
                # Drop NaN values (FRED sometimes has blanks)
                self.vix_data = self.vix_data.dropna()
                logger.info(f"Loaded VIX data: {len(self.vix_data)} observations from {self.vix_data.index[0]} to {self.vix_data.index[-1]}")
            except Exception as e:
                logger.warning(f"Failed to load VIX data from {vix_path}: {e}")
                self.vix_data = None
        else:
            logger.warning(f"VIX data file not found at {vix_path}")

        # Load M1/M2 money supply data from FRED for term structure indicators
        self.m1_data = None
        self.m2_data = None
        m1_path = Path(__file__).parent / "WM1NS.csv"
        m2_path = Path(__file__).parent / "WM2NS.csv"
        
        if m1_path.exists():
            try:
                m1_df = pd.read_csv(m1_path, parse_dates=['observation_date'], index_col='observation_date')
                self.m1_data = m1_df['WM1NS'].sort_index().dropna()
                logger.info(f"Loaded M1 data: {len(self.m1_data)} observations from {self.m1_data.index[0]} to {self.m1_data.index[-1]}")
            except Exception as e:
                logger.warning(f"Failed to load M1 data from {m1_path}: {e}")
                self.m1_data = None
        else:
            logger.warning(f"M1 data file not found at {m1_path}")
        
        if m2_path.exists():
            try:
                m2_df = pd.read_csv(m2_path, parse_dates=['observation_date'], index_col='observation_date')
                self.m2_data = m2_df['WM2NS'].sort_index().dropna()
                logger.info(f"Loaded M2 data: {len(self.m2_data)} observations from {self.m2_data.index[0]} to {self.m2_data.index[-1]}")
            except Exception as e:
                logger.warning(f"Failed to load M2 data from {m2_path}: {e}")
                self.m2_data = None
        else:
            logger.warning(f"M2 data file not found at {m2_path}")

        logger.info(f"Initialized calculator with {len(hist)} bars (using adjusted={_col('Close').startswith('Adj')})")

    def safe_float(self, val: Any, default: Optional[float] = None) -> Optional[float]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            out = float(val)
            return out if not np.isnan(out) else default
        except (TypeError, ValueError):
            return default

    def safe_bool(self, val: Any) -> bool:
        if val is None:
            return False
        if isinstance(val, (np.bool_, np.generic)):
            return bool(val)
        return bool(val)

    def robust_zscore(self, x: float, history: np.ndarray) -> Optional[float]:
        history = history[~np.isnan(history)]
        if len(history) < 30:
            return None
        median = np.median(history)
        mad = np.median(np.abs(history - median))
        if mad < 1e-10:
            return 0.0
        return (x - median) / (1.4826 * mad)

    def robust_zscore_sigmoid(self, x: float, history: np.ndarray, k: float = 2.0) -> Optional[float]:
        z = self.robust_zscore(x, history)
        if z is None:
            return None
        z = np.clip(z, -3.0, 3.0)
        return float(np.tanh(z / k))

    # =========================================================================
    # MOMENTUM
    # =========================================================================

    def calculate_momentum(self) -> Dict[str, Optional[float]]:
        """
        Calculate short-term returns (mean-reversion inputs) and medium-term momentum.
        
        SHORT-TERM RETURNS (ret_1d through ret_10d):
        - Strong directional predictors at 1-5d horizons (mean-reversion fade)
        - Included with z-scores for volatility normalization
        
        MEDIUM-TERM MOMENTUM (mom_3m):
        - Used for regime context and internal classification
        - Kept for drift + regime gating (not output for 1-3d sweeps, internal only)
        """
        results: Dict[str, Optional[float]] = {}

        # ===== SHORT-TERM RETURNS (1d-10d) — Mean-reversion inputs =====
        short_periods = {
            "ret_1d": 1,
            "ret_2d": 2,
            "ret_3d": 3,
            "ret_5d": 5,
            "ret_10d": 10,
        }

        for key, period in short_periods.items():
            if len(self.close) >= period + 1:
                ret = (self.close[-1] / self.close[-period - 1]) - 1.0
                results[key] = self.safe_float(ret)

                lookback = min(252, len(self.close) - period - 1)
                if lookback >= 50:
                    historical_returns = []
                    for i in range(lookback):
                        idx = -(i + 1)
                        start_idx = idx - period
                        if abs(start_idx) < len(self.close):
                            hist_ret = (self.close[idx] / self.close[start_idx]) - 1.0
                            if not np.isnan(hist_ret):
                                historical_returns.append(hist_ret)
                    hist_arr = np.array(historical_returns, dtype=np.float64)
                    z = self.robust_zscore(ret, hist_arr)
                    results[f"{key}_zscore"] = self.safe_float(z) if z is not None else None
                else:
                    results[f"{key}_zscore"] = None
            else:
                results[key] = None
                results[f"{key}_zscore"] = None

        # ===== MEDIUM-TERM MOMENTUM (1m and 3m) — Drift and mean-reversion signals =====
        # Both are output for testing: mom_1m is negative factor (mean-revert), mom_3m is positive (drift)
        
        # 1-month (21d) momentum — negative momentum factor
        if len(self.close) >= 22:
            mom_1m_ret = (self.close[-1] / self.close[-22]) - 1.0
            results["mom_1m"] = self.safe_float(mom_1m_ret)
            
            lookback = min(252, len(self.close) - 22)
            if lookback >= 50:
                historical_returns = []
                for i in range(lookback):
                    idx = -(i + 1)
                    start_idx = idx - 21
                    if abs(start_idx) < len(self.close):
                        hist_ret = (self.close[idx] / self.close[start_idx]) - 1.0
                        if not np.isnan(hist_ret):
                            historical_returns.append(hist_ret)
                hist_arr = np.array(historical_returns, dtype=np.float64)
                z = self.robust_zscore(mom_1m_ret, hist_arr)
                results["mom_1m_zscore"] = self.safe_float(z) if z is not None else None
            else:
                results["mom_1m_zscore"] = None
        else:
            results["mom_1m"] = None
            results["mom_1m_zscore"] = None

        # 3-month (63d) momentum — positive momentum factor
        if len(self.close) >= 64:
            mom_3m_ret = (self.close[-1] / self.close[-64]) - 1.0
            results["mom_3m"] = self.safe_float(mom_3m_ret)
            
            lookback = min(252, len(self.close) - 64)
            if lookback >= 50:
                historical_returns = []
                for i in range(lookback):
                    idx = -(i + 1)
                    start_idx = idx - 63
                    if abs(start_idx) < len(self.close):
                        hist_ret = (self.close[idx] / self.close[start_idx]) - 1.0
                        if not np.isnan(hist_ret):
                            historical_returns.append(hist_ret)
                hist_arr = np.array(historical_returns, dtype=np.float64)
                z = self.robust_zscore(mom_3m_ret, hist_arr)
                results["mom_3m_zscore"] = self.safe_float(z) if z is not None else None
            else:
                results["mom_3m_zscore"] = None
        else:
            results["mom_3m"] = None
            results["mom_3m_zscore"] = None

        # ===== AUTOCORRELATION — Mean-reversion persistence =====
        autocorr_periods = {"autocorr_5d": 5, "autocorr_10d": 10, "autocorr_21d": 21}

        for key, lag in autocorr_periods.items():
            min_obs = lag * 4
            if len(self.close) >= min_obs:
                returns = np.diff(np.log(self.close[-min_obs:]))
                if len(returns) > lag:
                    ret_current = returns[lag:]
                    ret_lagged = returns[:-lag]
                    if (
                        len(ret_current) >= lag
                        and np.std(ret_current) > 0.0001
                        and np.std(ret_lagged) > 0.0001
                    ):
                        autocorr = np.corrcoef(ret_current, ret_lagged)[0, 1]
                        results[key] = self.safe_float(autocorr) if not np.isnan(autocorr) else 0.0
                    else:
                        results[key] = 0.0
                else:
                    results[key] = None
            else:
                results[key] = None

        return results

    # =========================================================================
    # TREND
    # =========================================================================

    def calculate_trend_indicators(self) -> Dict[str, Any]:
        """
        Calculate trend and momentum indicators for directional prediction.
        Outputs: price vs MA signals (boolean + continuous), momentum, vol-adjusted trend.
        
        NOTE: MA buffer is now vol-scaled (ATR-based) instead of fixed 2%.
        - Tight buffer in calm markets (clear signals)
        - Wider buffer in volatile markets (avoid whipsaws)
        - Minimum 0.2%, typically 0.5-1.5% depending on ATR
        """
        results: Dict[str, Any] = {}

        # Compute ATR-scaled buffer for MA signals
        if len(self.close) >= 14:
            atr = talib.ATR(self.high, self.low, self.close, timeperiod=14)
            atr_val = atr[-1]
            if not np.isnan(atr_val) and self.close[-1] > 0:
                atr_pct = atr_val / self.close[-1]
                # Buffer: at least 0.2%, scale with ATR (0.5x)
                ma_buffer = max(0.002, 0.5 * atr_pct)
            else:
                ma_buffer = 0.002  # fallback to 0.2%
        else:
            ma_buffer = 0.002

        # ===== MA(20) =====
        if len(self.close) >= 20:
            ma20 = talib.SMA(self.close, timeperiod=20)
            ma20_val = ma20[-1]
            if not np.isnan(ma20_val):
                price_ma_ratio = self.close[-1] / ma20_val - 1
                # Boolean signal with vol-scaled buffer
                if price_ma_ratio > ma_buffer:
                    results["price_above_ma20"] = True
                elif price_ma_ratio < -ma_buffer:
                    results["price_above_ma20"] = False
                else:
                    results["price_above_ma20"] = None  # Neutral zone
                # Continuous % distance
                results["pct_from_ma20"] = self.safe_float(price_ma_ratio * 100.0)
            else:
                results["price_above_ma20"] = None
                results["pct_from_ma20"] = None
        else:
            results["price_above_ma20"] = None
            results["pct_from_ma20"] = None

        # ===== MA(75) =====
        if len(self.close) >= 75:
            ma75 = talib.SMA(self.close, timeperiod=75)
            ma75_val = ma75[-1]
            if not np.isnan(ma75_val):
                price_ma_ratio = self.close[-1] / ma75_val - 1
                if price_ma_ratio > ma_buffer:
                    results["price_above_ma75"] = True
                elif price_ma_ratio < -ma_buffer:
                    results["price_above_ma75"] = False
                else:
                    results["price_above_ma75"] = None
                results["pct_from_ma75"] = self.safe_float(price_ma_ratio * 100.0)
            else:
                results["price_above_ma75"] = None
                results["pct_from_ma75"] = None
        else:
            results["price_above_ma75"] = None
            results["pct_from_ma75"] = None

        # ===== MA(100) =====
        if len(self.close) >= 100:
            ma100 = talib.SMA(self.close, timeperiod=100)
            ma100_val = ma100[-1]
            if not np.isnan(ma100_val):
                price_ma_ratio = self.close[-1] / ma100_val - 1
                if price_ma_ratio > ma_buffer:
                    results["price_above_ma100"] = True
                elif price_ma_ratio < -ma_buffer:
                    results["price_above_ma100"] = False
                else:
                    results["price_above_ma100"] = None
                results["pct_from_ma100"] = self.safe_float(price_ma_ratio * 100.0)
            else:
                results["price_above_ma100"] = None
                results["pct_from_ma100"] = None
        else:
            results["price_above_ma100"] = None
            results["pct_from_ma100"] = None

        # ===== Momentum (3m = ~60d) for directional bias =====
        if len(self.close) >= 64:
            mom_3m_ret = (self.close[-1] / self.close[-64]) - 1.0
            results["mom_3m"] = self.safe_float(mom_3m_ret)
        else:
            results["mom_3m"] = None

        # ===== Vol-adjusted trend (ATR%-scaled distance from MA) =====
        # Normalize MA distance by ATR% (daily volatility proxy) for comparable magnitude signals
        atr_pct = None
        if len(self.close) >= 14:
            atr = talib.ATR(self.high, self.low, self.close, timeperiod=14)
            atr_val = atr[-1]
            if not np.isnan(atr_val) and self.close[-1] > 0:
                atr_pct = atr_val / self.close[-1]

        current_price = self.close[-1]

        # Vol-adjusted trend for MA(20)
        if len(self.close) >= 20 and atr_pct is not None and atr_pct > 0:
            ma20 = talib.SMA(self.close, timeperiod=20)
            ma20_val = ma20[-1]
            if not np.isnan(ma20_val):
                # Distance from MA as ratio, normalized by ATR%
                ma_distance_ratio = (current_price / ma20_val) - 1.0
                vol_adjusted = ma_distance_ratio / atr_pct
                results["vol_adjusted_trend_ma20"] = self.safe_float(vol_adjusted)
                
                # MA20 slope (20-day change), normalized by ATR%
                if len(ma20) >= 20 and not np.isnan(ma20[-20]):
                    ma20_slope = (ma20_val - ma20[-20]) / ma20[-20]
                    vol_adjusted_slope = ma20_slope / atr_pct
                    results["vol_adjusted_trend_ma20_slope"] = self.safe_float(vol_adjusted_slope)
                else:
                    results["vol_adjusted_trend_ma20_slope"] = None
            else:
                results["vol_adjusted_trend_ma20"] = None
                results["vol_adjusted_trend_ma20_slope"] = None
        else:
            results["vol_adjusted_trend_ma20"] = None
            results["vol_adjusted_trend_ma20_slope"] = None

        # Vol-adjusted trend for MA(100)
        if len(self.close) >= 100 and atr_pct is not None and atr_pct > 0:
            ma100 = talib.SMA(self.close, timeperiod=100)
            ma100_val = ma100[-1]
            if not np.isnan(ma100_val):
                # Distance from MA as ratio, normalized by ATR%
                ma_distance_ratio = (current_price / ma100_val) - 1.0
                vol_adjusted = ma_distance_ratio / atr_pct
                results["vol_adjusted_trend_ma100"] = self.safe_float(vol_adjusted)
                
                # MA100 slope (20-day change), normalized by ATR%
                if len(ma100) >= 20 and not np.isnan(ma100[-20]):
                    ma100_slope = (ma100_val - ma100[-20]) / ma100[-20]
                    vol_adjusted_slope = ma100_slope / atr_pct
                    results["vol_adjusted_trend_ma100_slope"] = self.safe_float(vol_adjusted_slope)
                else:
                    results["vol_adjusted_trend_ma100_slope"] = None
            else:
                results["vol_adjusted_trend_ma100"] = None
                results["vol_adjusted_trend_ma100_slope"] = None
        else:
            results["vol_adjusted_trend_ma100"] = None
            results["vol_adjusted_trend_ma100_slope"] = None

        # ===== Price structure (kept from original) =====
        results["higher_highs_lows"] = self._analyze_price_structure()

        return results

    def _compute_long_momentum(self) -> Dict[str, Optional[float]]:
        """
        Compute 6m and 12m momentum for regime context (not output to JSON for 1–3d sweeps).
        Used internally for regime classification / gating, not for Tier-1 alpha.
        """
        long_mom: Dict[str, Optional[float]] = {}

        if len(self.close) >= 127:
            mom_6m_ret = (self.close[-1] / self.close[-127]) - 1.0
            long_mom["mom_6m"] = self.safe_float(mom_6m_ret)
        else:
            long_mom["mom_6m"] = None

        if len(self.close) >= 253:
            mom_12m_ret = (self.close[-1] / self.close[-253]) - 1.0
            long_mom["mom_12m"] = self.safe_float(mom_12m_ret)
        else:
            long_mom["mom_12m"] = None

        return long_mom


    def _analyze_price_structure(self) -> str:
        if len(self.close) < 40:
            return "mixed"

        recent_high = np.max(self.high[-20:])
        recent_low = np.min(self.low[-20:])
        prev_high = np.max(self.high[-40:-20])
        prev_low = np.min(self.low[-40:-20])

        higher_high = recent_high > prev_high
        higher_low = recent_low > prev_low
        lower_high = recent_high < prev_high
        lower_low = recent_low < prev_low

        if higher_high and higher_low:
            return "uptrend"
        if lower_high and lower_low:
            return "downtrend"
        return "mixed"

    # =========================================================================
    # RANGE & PRICE STRUCTURE
    # =========================================================================

    def calculate_range_structure_indicators(self) -> Dict[str, Any]:
        """
        Calculate range and price structure indicators.
        Outputs: 52w high/low metrics, range position, breakout/failed breakout flags.
        Role: Structural context for direction.
        """
        results: Dict[str, Any] = {}

        current_price = self.close[-1]

        # ===== 52-week (252d) high/low metrics =====
        if len(self.close) >= 253:
            high_252 = np.max(self.high[-253:])
            low_252 = np.min(self.low[-253:])

            # % from 52w high (signed, ≤ 0)
            if high_252 > 0:
                pct_from_high_252 = (current_price / high_252) - 1.0
                results["pct_from_52w_high"] = self.safe_float(pct_from_high_252)
            else:
                results["pct_from_52w_high"] = None

            # New high / new low flags
            # Check if current price is within 0.1% of 52w high/low (avoids rounding issues)
            tolerance = 0.001
            is_new_high = (current_price >= high_252 * (1 - tolerance))
            is_new_low = (current_price <= low_252 * (1 + tolerance))
            results["is_new_52w_high"] = self.safe_bool(is_new_high)
            results["is_new_52w_low"] = self.safe_bool(is_new_low)

            # Days since last new high/low
            results["days_since_52w_high"] = self._days_since_level(self.high, high_252)
            results["days_since_52w_low"] = self._days_since_level(self.low, low_252, is_high=False)
        else:
            results["pct_from_52w_high"] = None
            results["is_new_52w_high"] = False
            results["is_new_52w_low"] = False
            results["days_since_52w_high"] = None
            results["days_since_52w_low"] = None

        # ===== Range position (20d and 55d) =====
        for window in [20, 55]:
            if len(self.close) >= window + 1:
                high_N = np.max(self.high[-window:])
                low_N = np.min(self.low[-window:])
                range_width = high_N - low_N

                if range_width > 0:
                    range_position = (current_price - low_N) / range_width
                    range_position = np.clip(range_position, 0.0, 1.0)
                    results[f"range_pos_{window}d"] = self.safe_float(range_position)
                else:
                    results[f"range_pos_{window}d"] = None
            else:
                results[f"range_pos_{window}d"] = None

        # ===== Breakout and failed breakout detection (20d and 55d, 10-day recent window) =====
        for window in [20, 55]:
            if len(self.close) >= window + 1:
                # Prior high/low (excluding today)
                prior_high = np.max(self.high[-window:-1])
                prior_low = np.min(self.low[-window:-1])

                # Check for breakout in last 10 days
                breakout_occurred = False
                failed_breakout_occurred = False

                lookback = min(10, len(self.close) - window)
                for i in range(lookback):
                    idx = -(i + 1)
                    if abs(idx) <= len(self.close):
                        if self.close[idx] > prior_high:
                            breakout_occurred = True
                            # Check if it failed in next 5 days
                            for j in range(1, min(6, i + 1)):
                                check_idx = -(j + 1)
                                if abs(check_idx) <= len(self.close):
                                    if self.close[check_idx] < prior_high:
                                        failed_breakout_occurred = True
                                        break
                            break

                results[f"breakout_{window}d_recent"] = self.safe_bool(breakout_occurred)
                results[f"failed_breakout_{window}d_recent"] = self.safe_bool(failed_breakout_occurred)
            else:
                results[f"breakout_{window}d_recent"] = False
                results[f"failed_breakout_{window}d_recent"] = False

        return results

    def _days_since_level(
        self, price_series: np.ndarray, level: float, is_high: bool = True
    ) -> Optional[int]:
        """
        Count trading days since price touched a specific level.
        is_high=True: check if high >= level
        is_high=False: check if low <= level
        
        Tolerance: 0.1% to handle rounding/precision issues.
        """
        tol = 0.001  # 0.1% tolerance
        for i in range(len(price_series) - 1):
            idx = -(i + 1)
            if is_high:
                if price_series[idx] >= level * (1 - tol):
                    return i
            else:
                if price_series[idx] <= level * (1 + tol):
                    return i
        return None

    # =========================================================================
    # TREND QUALITY / PERSISTENCE
    # =========================================================================

    def calculate_trend_quality_indicators(self) -> Dict[str, Any]:
        """
        Calculate trend quality and persistence indicators.
        Outputs: ADX, % of up days, directional efficiency.
        Role: Modulates confidence in trend signals.
        """
        results: Dict[str, Any] = {}

        # ===== ADX(14) — raw and z-score =====
        if len(self.close) >= 14:
            try:
                adx = talib.ADX(self.high, self.low, self.close, timeperiod=14)
                adx_val = self.safe_float(adx[-1])
                results["adx_14d"] = adx_val

                # Z-score ADX within historical context
                if len(adx) >= 50 and adx_val is not None:
                    hist_adx = adx[-50:]
                    hist_adx = hist_adx[~np.isnan(hist_adx)]
                    if len(hist_adx) >= 20:
                        z = self.robust_zscore(adx_val, hist_adx)
                        results["adx_14d_zscore"] = self.safe_float(z) if z is not None else None
                    else:
                        results["adx_14d_zscore"] = None
                else:
                    results["adx_14d_zscore"] = None
            except Exception as e:
                logger.warning(f"ADX calculation error: {e}")
                results["adx_14d"] = None
                results["adx_14d_zscore"] = None
        else:
            results["adx_14d"] = None
            results["adx_14d_zscore"] = None

        # ===== % of up days (5d, 10d, 20d) =====
        for window in [5, 10, 20]:
            if len(self.close) >= window + 1:
                up_days = 0
                total_days = window
                for i in range(window):
                    idx = -(i + 1)
                    prev_idx = -(i + 2)
                    if abs(prev_idx) <= len(self.close):
                        if self.close[idx] > self.close[prev_idx]:
                            up_days += 1
                pct_up = up_days / total_days
                results[f"pct_up_days_{window}d"] = self.safe_float(pct_up)
            else:
                results[f"pct_up_days_{window}d"] = None

        # ===== Directional efficiency (10d, 20d) =====
        for window in [10, 20]:
            if len(self.close) >= window + 1:
                # Net move
                net_move = abs(self.close[-1] - self.close[-window - 1])
                
                # Total movement (sum of absolute daily moves)
                total_movement = 0.0
                for i in range(window):
                    idx = -(i + 1)
                    prev_idx = -(i + 2)
                    if abs(prev_idx) <= len(self.close):
                        daily_move = abs(self.close[idx] - self.close[prev_idx])
                        total_movement += daily_move

                # Efficiency = net move / total movement, capped at [0, 1]
                if total_movement > 0:
                    efficiency = net_move / total_movement
                    efficiency = np.clip(efficiency, 0.0, 1.0)
                    results[f"directional_efficiency_{window}d"] = self.safe_float(efficiency)
                else:
                    results[f"directional_efficiency_{window}d"] = None
            else:
                results[f"directional_efficiency_{window}d"] = None

        return results

    # =========================================================================
    # MEAN REVERSION / EXHAUSTION
    # =========================================================================

    def calculate_mean_reversion_indicators(self) -> Dict[str, Any]:
        """
        Calculate mean reversion and exhaustion indicators for counter-trend fades.
        Outputs: RSI(2-5), short-term MA z-scores, stretch metrics, price velocity vs vol.
        """
        results: Dict[str, Any] = {}

        # ===== RSI(2–5) for micro mean-reversion =====
        rsi_2d = talib.RSI(self.close, timeperiod=2) if len(self.close) >= 3 else None
        results["rsi_2d"] = self.safe_float(rsi_2d[-1]) if rsi_2d is not None and len(rsi_2d) > 0 else None

        rsi_3d = talib.RSI(self.close, timeperiod=3) if len(self.close) >= 4 else None
        results["rsi_3d"] = self.safe_float(rsi_3d[-1]) if rsi_3d is not None and len(rsi_3d) > 0 else None

        rsi_4d = talib.RSI(self.close, timeperiod=4) if len(self.close) >= 5 else None
        results["rsi_4d"] = self.safe_float(rsi_4d[-1]) if rsi_4d is not None and len(rsi_4d) > 0 else None

        rsi_5d = talib.RSI(self.close, timeperiod=5) if len(self.close) >= 6 else None
        results["rsi_5d"] = self.safe_float(rsi_5d[-1]) if rsi_5d is not None and len(rsi_5d) > 0 else None

        # ===== Short-term MA z-scores (distance to MA normalized by vol) =====
        vols = self._compute_internal_vols()
        vol_21d = vols.get("realized_vol_21d")
        current_price = self.close[-1]

        # MA(5) z-score of distance
        if len(self.close) >= 6 and vol_21d is not None and vol_21d > 0:
            ma5 = talib.SMA(self.close, timeperiod=5)
            ma5_val = ma5[-1]
            if not np.isnan(ma5_val):
                distance_metric = (current_price - ma5_val) / (current_price * vol_21d)
                results["ma5_distance_vol_normalized"] = self.safe_float(distance_metric)
                # Z-score this metric over historical context
                results["ma5_distance_zscore"] = self._compute_metric_zscore(
                    distance_metric, window=100, metric_name="ma5_distance"
                )
            else:
                results["ma5_distance_vol_normalized"] = None
                results["ma5_distance_zscore"] = None
        else:
            results["ma5_distance_vol_normalized"] = None
            results["ma5_distance_zscore"] = None

        # MA(10) z-score of distance
        if len(self.close) >= 11 and vol_21d is not None and vol_21d > 0:
            ma10 = talib.SMA(self.close, timeperiod=10)
            ma10_val = ma10[-1]
            if not np.isnan(ma10_val):
                distance_metric = (current_price - ma10_val) / (current_price * vol_21d)
                results["ma10_distance_vol_normalized"] = self.safe_float(distance_metric)
                results["ma10_distance_zscore"] = self._compute_metric_zscore(
                    distance_metric, window=100, metric_name="ma10_distance"
                )
            else:
                results["ma10_distance_vol_normalized"] = None
                results["ma10_distance_zscore"] = None
        else:
            results["ma10_distance_vol_normalized"] = None
            results["ma10_distance_zscore"] = None

        # ===== Stretch metrics (MA20-based) =====
        if len(self.close) >= 20 and vol_21d is not None and vol_21d > 0:
            ma20 = talib.SMA(self.close, timeperiod=20)
            ma20_val = ma20[-1]
            if not np.isnan(ma20_val):
                # Simple % distance
                stretch_pct = (current_price / ma20_val - 1.0) * 100.0
                results["stretch_pct_from_ma20"] = self.safe_float(stretch_pct)

                # Vol-normalized stretch, z-scored
                stretch_vol_normalized = (current_price - ma20_val) / (current_price * vol_21d)
                results["stretch_z_20"] = self._compute_metric_zscore(
                    stretch_vol_normalized, window=100, metric_name="stretch_ma20"
                )
            else:
                results["stretch_pct_from_ma20"] = None
                results["stretch_z_20"] = None
        else:
            results["stretch_pct_from_ma20"] = None
            results["stretch_z_20"] = None

        # ===== Price velocity vs recent volatility =====
        if len(self.close) >= 6 and vol_21d is not None and vol_21d > 0:
            vel_5d = (current_price / self.close[-6]) - 1.0
            results["vel_5d"] = self.safe_float(vel_5d)
            results["vel_5d_over_vol"] = self.safe_float(vel_5d / vol_21d)
        else:
            results["vel_5d"] = None
            results["vel_5d_over_vol"] = None

        if len(self.close) >= 11 and vol_21d is not None and vol_21d > 0:
            vel_10d = (current_price / self.close[-11]) - 1.0
            results["vel_10d"] = self.safe_float(vel_10d)
            results["vel_10d_over_vol"] = self.safe_float(vel_10d / vol_21d)
        else:
            results["vel_10d"] = None
            results["vel_10d_over_vol"] = None

        return results

    def _compute_metric_zscore(
        self, current_value: float, window: int = 100, metric_name: str = "metric"
    ) -> Optional[float]:
        """
        Z-score a metric over a rolling historical window.
        For distance-to-MA and stretch metrics.
        """
        if len(self.close) < window + 20:
            return None

        metric_values = []
        for i in range(window):
            end_idx = -(i + 1)
            start_idx = end_idx - 20
            if abs(start_idx) <= len(self.close) and abs(end_idx) <= len(self.close):
                hist_price = self.close[end_idx]
                hist_returns = np.diff(np.log(self.close[start_idx:end_idx]))
                if len(hist_returns) >= 19:
                    hist_vol = np.std(hist_returns, ddof=1) * np.sqrt(252)
                    if hist_vol > 0:
                        # Recompute for this period's MA(20) or MA(5/10) as appropriate
                        if metric_name in ["ma5_distance", "ma10_distance"]:
                            period = 5 if metric_name == "ma5_distance" else 10
                            hist_ma = self.close[start_idx:end_idx]
                            if len(hist_ma) >= period:
                                ma_val = np.mean(hist_ma[-period:])
                                metric = (hist_price - ma_val) / (hist_price * hist_vol)
                            else:
                                continue
                        elif metric_name == "stretch_ma20":
                            hist_ma = self.close[start_idx:end_idx]
                            if len(hist_ma) >= 20:
                                ma_val = np.mean(hist_ma[-20:])
                                metric = (hist_price - ma_val) / (hist_price * hist_vol)
                            else:
                                continue
                        else:
                            continue

                        if not np.isnan(metric):
                            metric_values.append(metric)

        if len(metric_values) < 30:
            return None

        metric_arr = np.array(metric_values, dtype=np.float64)
        z = self.robust_zscore(current_value, metric_arr)
        return self.safe_float(z) if z is not None else None

    # =========================================================================
    # VOLATILITY
    # =========================================================================


    def _compute_internal_vols(self) -> Dict[str, Optional[float]]:
        """
        Compute realized volatilities for multiple windows (10d, 14d, 20d, 21d, 63d, 252d).
        These are used for indicators, regimes, and percentile calculations.
        Returns all computed vols; caller decides what to expose in JSON.
        """
        vols: Dict[str, Optional[float]] = {}

        # 10-day realized vol
        if len(self.close) >= 11:
            returns_10d = np.diff(np.log(self.close[-11:]))
            vols["realized_vol_10d"] = np.std(returns_10d, ddof=1) * np.sqrt(252)
        else:
            vols["realized_vol_10d"] = None

        # 14-day realized vol
        if len(self.close) >= 15:
            returns_14d = np.diff(np.log(self.close[-15:]))
            vols["realized_vol_14d"] = np.std(returns_14d, ddof=1) * np.sqrt(252)
        else:
            vols["realized_vol_14d"] = None

        # 20-day realized vol
        if len(self.close) >= 21:
            returns_20d = np.diff(np.log(self.close[-21:]))
            vols["realized_vol_20d"] = np.std(returns_20d, ddof=1) * np.sqrt(252)
        else:
            vols["realized_vol_20d"] = None

        # 21-day realized vol
        if len(self.close) >= 22:
            returns_21d = np.diff(np.log(self.close[-22:]))
            vols["realized_vol_21d"] = np.std(returns_21d, ddof=1) * np.sqrt(252)
        else:
            vols["realized_vol_21d"] = None

        # 63-day realized vol (for regime classification only)
        if len(self.close) >= 64:
            returns_63d = np.diff(np.log(self.close[-64:]))
            vols["realized_vol_63d"] = np.std(returns_63d, ddof=1) * np.sqrt(252)
        else:
            vols["realized_vol_63d"] = None

        # 252-day realized vol (for regime classification only)
        if len(self.close) >= 253:
            returns_252d = np.diff(np.log(self.close[-253:]))
            vols["realized_vol_252d"] = np.std(returns_252d, ddof=1) * np.sqrt(252)
        else:
            vols["realized_vol_252d"] = None

        return vols

    def calculate_volatility_indicators(self) -> Dict[str, Any]:
        """
        Calculate volatility indicators for magnitude prediction and regime gating.
        Outputs: short-term realized vols (10d, 14d, 21d), ATR, vol ROC, vol percentiles.
        """
        results: Dict[str, Any] = {}

        # Compute all internal volatilities
        vols = self._compute_internal_vols()

        # ===== OUTPUT: Short-term realized volatilities =====
        results["realized_vol_10d"] = self.safe_float(vols["realized_vol_10d"])
        results["realized_vol_14d"] = self.safe_float(vols["realized_vol_14d"])
        results["realized_vol_21d"] = self.safe_float(vols["realized_vol_21d"])

        # ===== OUTPUT: ATR and ATR % =====
        if len(self.close) >= 14:
            atr = talib.ATR(self.high, self.low, self.close, timeperiod=14)
            atr_val = self.safe_float(atr[-1])
            results["atr_14d"] = atr_val
            if atr_val and self.close[-1] > 0:
                results["atr_14d_pct"] = atr_val / self.close[-1]
            else:
                results["atr_14d_pct"] = None
        else:
            results["atr_14d"] = None
            results["atr_14d_pct"] = None

        # ===== OUTPUT: Volatility ROC (10d vs 20d) =====
        if vols["realized_vol_10d"] is not None and vols["realized_vol_20d"] is not None:
            # Ratio: vol_20d / vol_10d (>1 = expanding, <1 = contracting)
            if vols["realized_vol_10d"] > 0:
                results["vol_roc_ratio_20v10"] = vols["realized_vol_20d"] / vols["realized_vol_10d"]
                results["vol_roc_pct_20v10"] = (vols["realized_vol_20d"] - vols["realized_vol_10d"]) / vols["realized_vol_10d"]
            else:
                results["vol_roc_ratio_20v10"] = None
                results["vol_roc_pct_20v10"] = None
        else:
            results["vol_roc_ratio_20v10"] = None
            results["vol_roc_pct_20v10"] = None

        # ===== OUTPUT: Volatility percentile ranks =====
        results["vol_percentile_252d"] = self._calculate_vol_percentile(lookback_days=252)
        results["vol_percentile_504d"] = self._calculate_vol_percentile(lookback_days=504)

        # ===== OUTPUT: Regime classification (uses internal long vols) =====
        vol_regime_result = self._classify_vol_regime_percentile()
        results["vol_regime"] = vol_regime_result["regime"]
        results["vol_percentile"] = vol_regime_result["percentile"]

        return results

    def _calculate_vol_percentile(self, lookback_days: int = 252) -> Optional[float]:
        """
        Calculate the percentile rank of current realized_vol_21d within
        the rolling distribution of realized vols over the lookback period.

        Args:
            lookback_days: Window size for percentile calculation (252 or 504)

        Returns:
            Percentile rank (0–100) or None if insufficient data
        """
        current_vol = self._compute_internal_vols().get("realized_vol_21d")
        if current_vol is None:
            return None

        if len(self.close) < lookback_days + 21:
            return None

        # Build rolling realized_vol series over the lookback period
        rolling_vols = []
        for i in range(lookback_days):
            end_idx = -(i + 1)
            start_idx = end_idx - 21
            if abs(start_idx) <= len(self.close) and abs(end_idx) <= len(self.close):
                hist_returns = np.diff(np.log(self.close[start_idx:end_idx]))
                if len(hist_returns) >= 20:
                    rolling_vol = np.std(hist_returns, ddof=1) * np.sqrt(252)
                    if not np.isnan(rolling_vol):
                        rolling_vols.append(rolling_vol)

        if len(rolling_vols) < 50:
            return None

        rolling_vols_arr = np.array(rolling_vols, dtype=np.float64)
        percentile = (rolling_vols_arr < current_vol).sum() / len(rolling_vols_arr) * 100.0
        return self.safe_float(percentile)

    def _classify_vol_regime_percentile(self) -> Dict[str, Any]:
        min_required = 273
        if len(self.close) < min_required:
            return {"regime": "neutral", "percentile": 50.0}

        current_returns = np.diff(np.log(self.close[-22:]))
        current_vol = np.std(current_returns, ddof=1) * np.sqrt(252)

        historical_vols = []
        for i in range(252):
            end_idx = -(i + 1)
            start_idx = end_idx - 21
            if abs(start_idx) <= len(self.close):
                hist_returns = np.diff(np.log(self.close[start_idx:end_idx]))
                if len(hist_returns) >= 20:
                    hist_vol = np.std(hist_returns, ddof=1) * np.sqrt(252)
                    if not np.isnan(hist_vol):
                        historical_vols.append(hist_vol)

        if len(historical_vols) < 50:
            return {"regime": "neutral", "percentile": 50.0}

        historical_vols_arr = np.array(historical_vols, dtype=np.float64)
        percentile = (np.sum(historical_vols_arr < current_vol) / len(historical_vols_arr)) * 100

        if percentile >= 80:
            regime = "high"
        elif percentile <= 20:
            regime = "low"
        else:
            regime = "neutral"

        return {"regime": regime, "percentile": self.safe_float(percentile)}

    # =========================================================================
    # RISK
    # =========================================================================

    def calculate_risk_indicators(self, benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if len(self.close) >= 253:
            returns = np.diff(np.log(self.close[-253:]))

            if benchmark_returns is not None and len(benchmark_returns) >= 252:
                bench_ret = benchmark_returns[-252:]
                asset_ret = returns[-252:]
                if len(asset_ret) == len(bench_ret):
                    covariance = np.cov(asset_ret, bench_ret)[0, 1]
                    benchmark_var = np.var(bench_ret, ddof=1)
                    results["beta_1y"] = self.safe_float(covariance / benchmark_var) if benchmark_var > 0 else None

                    corr = np.corrcoef(asset_ret, bench_ret)[0, 1]
                    results["corr_sp500"] = self.safe_float(corr)
                    results["r2_sp500"] = self.safe_float(corr**2) if not np.isnan(corr) else None
                else:
                    results["beta_1y"] = None
                    results["corr_sp500"] = None
                    results["r2_sp500"] = None
            else:
                results["beta_1y"] = None
                results["corr_sp500"] = None
                results["r2_sp500"] = None

            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            if std_return > 0:
                sharpe = (mean_return * 252) / (std_return * np.sqrt(252))
                results["sharpe_1y"] = self.safe_float(sharpe)
            else:
                results["sharpe_1y"] = None

            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns, ddof=1)
                if downside_std > 0:
                    sortino = (mean_return * 252) / (downside_std * np.sqrt(252))
                    results["sortino_1y"] = self.safe_float(sortino)
                else:
                    results["sortino_1y"] = None
            else:
                results["sortino_1y"] = None

            cumulative = np.exp(np.cumsum(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_dd = np.min(drawdown)
            results["max_drawdown_1y"] = self.safe_float(abs(max_dd))

            if benchmark_returns is not None and len(benchmark_returns) >= 252:
                bench_ret = benchmark_returns[-252:]
                asset_ret = returns[-252:]

                up_periods = bench_ret > 0
                if np.sum(up_periods) > 0:
                    up_asset = np.mean(asset_ret[up_periods])
                    up_bench = np.mean(bench_ret[up_periods])
                    results["up_capture_1y"] = self.safe_float(up_asset / up_bench) if up_bench != 0 else None
                else:
                    results["up_capture_1y"] = None

                down_periods = bench_ret < 0
                if np.sum(down_periods) > 0:
                    down_asset = np.mean(asset_ret[down_periods])
                    down_bench = np.mean(bench_ret[down_periods])
                    results["down_capture_1y"] = self.safe_float(down_asset / down_bench) if down_bench != 0 else None
                else:
                    results["down_capture_1y"] = None
            else:
                results["up_capture_1y"] = None
                results["down_capture_1y"] = None
        else:
            results["beta_1y"] = None
            results["corr_sp500"] = None
            results["r2_sp500"] = None
            results["sharpe_1y"] = None
            results["sortino_1y"] = None
            results["max_drawdown_1y"] = None
            results["up_capture_1y"] = None
            results["down_capture_1y"] = None

        return results

    # =========================================================================
    # VOLUME / LIQUIDITY
    # =========================================================================

    def calculate_volume_liquidity_indicators(self) -> Dict[str, Any]:
        """
        Calculate volume and liquidity metrics.
        Removed: OBV/A-D trends (redundant with price structure).
        Kept: Volume z-score, dollar volume, spreads, liquidity score.
        """
        results: Dict[str, Any] = {}

        if len(self.volume) >= 63:
            results["avg_vol_3m"] = self.safe_float(np.mean(self.volume[-63:]))
        else:
            results["avg_vol_3m"] = None

        if len(self.volume) >= 21:
            dollar_vol = self.close[-21:] * self.volume[-21:]
            results["avg_dollar_vol_21d"] = self.safe_float(np.mean(dollar_vol))
        else:
            results["avg_dollar_vol_21d"] = None

        results["vol_last"] = self.safe_float(self.volume[-1])

        if len(self.volume) >= 21:
            vol_mean = np.mean(self.volume[-21:])
            vol_std = np.std(self.volume[-21:], ddof=1)
            results["volume_spike_z"] = self.safe_float((self.volume[-1] - vol_mean) / vol_std) if vol_std > 0 else None
        else:
            results["volume_spike_z"] = None

        # Removed OBV and A-D trends (redundant with price structure analysis)

        results["roll_spread_21"] = self._calculate_roll_spread()
        results["cs_spread_21"] = self._calculate_corwin_schultz_spread()
        results["liquidity_score_0_100"] = self._calculate_liquidity_score(results)

        return results

    # =========================================================================
    # VOLUME PARTICIPATION / CONFIRMATION
    # =========================================================================

    def calculate_volume_participation_indicators(self) -> Dict[str, Any]:
        """
        Calculate volume participation and confirmation indicators.
        Outputs: range-volume impulse, up/down volume imbalance, volume confirmation flags.
        Role: Confirmation / veto layer.
        """
        results: Dict[str, Any] = {}

        current_price = self.close[-1]

        # ===== Range × Volume impulse =====
        # Combine range (high-low) and volume z-score
        if len(self.close) >= 21 and len(self.volume) >= 21:
            vol_mean = np.mean(self.volume[-21:])
            vol_std = np.std(self.volume[-21:], ddof=1)
            volume_z = (self.volume[-1] - vol_mean) / vol_std if vol_std > 0 else 0.0

            range_pct = (self.high[-1] - self.low[-1]) / current_price if current_price > 0 else 0.0
            range_vol_impulse = range_pct * volume_z
            results["range_vol_impulse"] = self.safe_float(range_vol_impulse)
        else:
            results["range_vol_impulse"] = None

        # ===== Up-volume vs down-volume imbalance (10d and 20d) =====
        for window in [10, 20]:
            if len(self.close) >= window + 1 and len(self.volume) >= window + 1:
                vol_up = 0.0
                vol_down = 0.0

                for i in range(window):
                    idx = -(i + 1)
                    prev_idx = -(i + 2)
                    if abs(prev_idx) <= len(self.close):
                        if self.close[idx] > self.close[prev_idx]:
                            vol_up += self.volume[idx]
                        elif self.close[idx] < self.close[prev_idx]:
                            vol_down += self.volume[idx]

                # Signed imbalance: (vol_up - vol_down) / (vol_up + vol_down)
                total_vol = vol_up + vol_down
                if total_vol > 0:
                    imbalance = (vol_up - vol_down) / total_vol
                    imbalance = np.clip(imbalance, -1.0, 1.0)
                    results[f"vol_imbalance_{window}d"] = self.safe_float(imbalance)
                else:
                    results[f"vol_imbalance_{window}d"] = None
            else:
                results[f"vol_imbalance_{window}d"] = None

        # ===== Volume confirmation flags =====
        # Baseline: 20-day average volume
        if len(self.volume) >= 21:
            avg_vol_20d = np.mean(self.volume[-21:])
        else:
            avg_vol_20d = None

        # Confirmation threshold: 1.2x baseline (120%)
        vol_threshold = 1.2

        # Breakout confirmation (55d): breakout + volume spike
        if len(self.close) >= 56 and avg_vol_20d is not None:
            prior_high_55d = np.max(self.high[-56:-1])
            is_breakout = self.close[-1] > prior_high_55d
            is_vol_high = self.volume[-1] > avg_vol_20d * vol_threshold
            results["is_vol_confirmed_breakout_55d"] = self.safe_bool(is_breakout and is_vol_high)
        else:
            results["is_vol_confirmed_breakout_55d"] = False

        # Also add 20d breakout confirmation for completeness
        if len(self.close) >= 21 and avg_vol_20d is not None:
            prior_high_20d = np.max(self.high[-21:-1])
            is_breakout_20 = self.close[-1] > prior_high_20d
            is_vol_high = self.volume[-1] > avg_vol_20d * vol_threshold
            results["is_vol_confirmed_breakout_20d"] = self.safe_bool(is_breakout_20 and is_vol_high)
        else:
            results["is_vol_confirmed_breakout_20d"] = False

        # Strength confirmation: up day + volume spike
        if len(self.close) >= 2 and avg_vol_20d is not None:
            is_up_day = self.close[-1] > self.close[-2]
            is_vol_high = self.volume[-1] > avg_vol_20d * vol_threshold
            results["is_vol_confirmed_strength"] = self.safe_bool(is_up_day and is_vol_high)
        else:
            results["is_vol_confirmed_strength"] = False

        return results

    # =========================================================================
    # MARKET PLUMBING / REGIME (Bucket 7)
    # =========================================================================

    def calculate_term_structure_indicators(self, date: datetime) -> Dict[str, Any]:
        """
        Calculate M1/M2 term structure indicators (Bucket 7 - Money Supply).
        
        Metrics:
        - m2_m1_ratio: M2/M1 ratio (money supply structure, shows excess reserves)
        - m2_m1_ratio_percentile_52w: M2/M1 percentile rank within last 52 weeks
        - m2_roc_4w: M2 rate of change over 4 weeks (monetary expansion/contraction)
        - m1_roc_4w: M1 rate of change over 4 weeks (transaction money velocity)
        - money_supply_regime: Categorical (expansion/neutral/contraction) based on M2 ROC percentile
        
        Role: Monetary cycle awareness, liquidity regime classification, term structure context
        """
        results: Dict[str, Any] = {}

        if self.m1_data is None or self.m1_data.empty or self.m2_data is None or self.m2_data.empty:
            return {
                'm2_m1_ratio': None,
                'm2_m1_ratio_percentile_52w': None,
                'm2_roc_4w': None,
                'm1_roc_4w': None,
                'money_supply_regime': None,
                'opex_window': None,
            }

        # Strip timezone if present (M1/M2 data is timezone-naive)
        if hasattr(date, 'tz') and date.tz is not None:
            date = date.replace(tzinfo=None)

        try:
            # Find M2 value for this date (or nearest prior)
            if date in self.m2_data.index:
                m2_idx = self.m2_data.index.get_loc(date)
            else:
                mask = self.m2_data.index <= date
                if mask.any():
                    m2_idx = mask.nonzero()[0][-1]
                else:
                    return {
                        'm2_m1_ratio': None,
                        'm2_m1_ratio_percentile_52w': None,
                        'm2_roc_4w': None,
                        'm1_roc_4w': None,
                        'money_supply_regime': None,
                        'opex_window': None,
                    }
            
            # Find M1 value for same date
            if date in self.m1_data.index:
                m1_idx = self.m1_data.index.get_loc(date)
            else:
                mask = self.m1_data.index <= date
                if mask.any():
                    m1_idx = mask.nonzero()[0][-1]
                else:
                    return {
                        'm2_m1_ratio': None,
                        'm2_m1_ratio_percentile_52w': None,
                        'm2_roc_4w': None,
                        'm1_roc_4w': None,
                        'money_supply_regime': None,
                        'opex_window': None,
                    }
        except Exception as e:
            logger.warning(f"Error finding M1/M2 for date {date}: {e}")
            return {
                'm2_m1_ratio': None,
                'm2_m1_ratio_percentile_52w': None,
                'm2_roc_4w': None,
                'm1_roc_4w': None,
                'money_supply_regime': None,
                'opex_window': None,
            }

        current_m2 = self.m2_data.iloc[m2_idx]
        current_m1 = self.m1_data.iloc[m1_idx]

        if pd.isna(current_m2) or pd.isna(current_m1):
            return {
                'm2_m1_ratio': None,
                'm2_m1_ratio_percentile_52w': None,
                'm2_roc_4w': None,
                'm1_roc_4w': None,
                'money_supply_regime': None,
                'opex_window': None,
            }

        # M2/M1 ratio
        if current_m1 > 0:
            m2_m1_ratio = current_m2 / current_m1
            results['m2_m1_ratio'] = self.safe_float(m2_m1_ratio)
        else:
            results['m2_m1_ratio'] = None

        # M2/M1 ratio percentile (52 weeks = ~1 year)
        if results['m2_m1_ratio'] is not None:
            start_idx = max(0, m2_idx - 52)  # Approximate 52 weeks (weekly data)
            m2_window = self.m2_data.iloc[start_idx:m2_idx+1]
            m1_window = self.m1_data.iloc[start_idx:m2_idx+1]
            
            valid_indices = [i for i in range(len(m2_window)) if not pd.isna(m2_window.iloc[i]) and not pd.isna(m1_window.iloc[i]) and m1_window.iloc[i] > 0]
            if valid_indices:
                ratios = [m2_window.iloc[i] / m1_window.iloc[i] for i in valid_indices]
                percentile = stats.percentileofscore(ratios, m2_m1_ratio, nan_policy='omit')
                results['m2_m1_ratio_percentile_52w'] = self.safe_float(percentile)
            else:
                results['m2_m1_ratio_percentile_52w'] = None
        else:
            results['m2_m1_ratio_percentile_52w'] = None

        # M2 and M1 ROC (4 weeks)
        for lag in [4]:
            if m2_idx >= lag:
                prior_m2 = self.m2_data.iloc[m2_idx - lag]
                if not pd.isna(prior_m2) and prior_m2 > 0:
                    m2_roc = (current_m2 - prior_m2) / prior_m2
                    results['m2_roc_4w'] = self.safe_float(m2_roc)
                else:
                    results['m2_roc_4w'] = None
            else:
                results['m2_roc_4w'] = None

            if m1_idx >= lag:
                prior_m1 = self.m1_data.iloc[m1_idx - lag]
                if not pd.isna(prior_m1) and prior_m1 > 0:
                    m1_roc = (current_m1 - prior_m1) / prior_m1
                    results['m1_roc_4w'] = self.safe_float(m1_roc)
                else:
                    results['m1_roc_4w'] = None
            else:
                results['m1_roc_4w'] = None

        # Money supply regime (based on M2 ROC percentile)
        if results['m2_roc_4w'] is not None:
            start_idx = max(0, m2_idx - 52)  # Last 52 weeks
            m2_roc_window = []
            for i in range(start_idx, m2_idx + 1):
                if i >= 4 and not pd.isna(self.m2_data.iloc[i]) and not pd.isna(self.m2_data.iloc[i-4]):
                    prior = self.m2_data.iloc[i - 4]
                    if prior > 0:
                        roc = (self.m2_data.iloc[i] - prior) / prior
                        m2_roc_window.append(roc)
            
            if m2_roc_window:
                percentile = stats.percentileofscore(m2_roc_window, results['m2_roc_4w'], nan_policy='omit')
                if percentile < 33:
                    results['money_supply_regime'] = 'contraction'
                elif percentile < 67:
                    results['money_supply_regime'] = 'neutral'
                else:
                    results['money_supply_regime'] = 'expansion'
            else:
                results['money_supply_regime'] = None
        else:
            results['money_supply_regime'] = None

        # OPEX window classification
        results['opex_window'] = self._get_opex_window(date)

        # Binary flags for testing (easier for parameter sweep)
        # Money supply regime flags
        if results.get('money_supply_regime'):
            results['is_expansion'] = 1 if results['money_supply_regime'] == 'expansion' else 0
            results['is_neutral'] = 1 if results['money_supply_regime'] == 'neutral' else 0
            results['is_contraction'] = 1 if results['money_supply_regime'] == 'contraction' else 0
        else:
            results['is_expansion'] = None
            results['is_neutral'] = None
            results['is_contraction'] = None

        # OPEX window flags
        if results.get('opex_window'):
            results['is_opex_normal'] = 1 if results['opex_window'] == 'normal' else 0
            results['is_pre_opex'] = 1 if results['opex_window'] == 'pre_opex' else 0
            results['is_opex_day'] = 1 if results['opex_window'] == 'opex_day' else 0
            results['is_post_opex'] = 1 if results['opex_window'] == 'post_opex' else 0
        else:
            results['is_opex_normal'] = None
            results['is_pre_opex'] = None
            results['is_opex_day'] = None
            results['is_post_opex'] = None

        return results

    def _get_opex_dates(self, start_year: int, end_year: int) -> set:
        """
        Calculate all OPEX dates (third Friday of each month) for a given range.
        Returns set of datetime.date objects for fast lookup.
        """
        from datetime import date, timedelta
        opex_dates = set()
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Start with first day of month
                first_day = date(year, month, 1)
                # Find first Friday (weekday 4 = Friday)
                days_until_friday = (4 - first_day.weekday()) % 7
                if days_until_friday == 0:
                    days_until_friday = 7
                first_friday = first_day + timedelta(days=days_until_friday)
                # Third Friday is 2 weeks later
                third_friday = first_friday + timedelta(days=14)
                opex_dates.add(third_friday)
        
        return opex_dates

    def _get_opex_window(self, date: datetime) -> str:
        """
        Get OPEX window classification for a given date.
        Returns: "normal", "pre_opex", "opex_day", or "post_opex"
        """
        from datetime import date as date_type, timedelta
        
        # Get date part only
        if isinstance(date, datetime):
            test_date = date.date()
        else:
            test_date = date
        
        # Build set of OPEX dates (cache could be added if called frequently)
        year_start = test_date.year - 1
        year_end = test_date.year + 1
        opex_dates = self._get_opex_dates(year_start, year_end)
        
        # Check current date and neighbors
        for offset in range(-2, 3):  # Check 2 days before to 2 days after
            check_date = test_date + timedelta(days=offset)
            if check_date in opex_dates:
                if offset == 0:
                    return 'opex_day'
                elif offset in [-2, -1]:
                    return 'pre_opex'
                elif offset in [1, 2]:
                    return 'post_opex'
        
        return 'normal'

    def calculate_vix_indicators(self, date: datetime) -> Dict[str, Any]:
        """
        Calculate VIX-based market regime indicators (Bucket 7).
        
        Metrics:
        - vix_level: Current VIX closing value
        - vix_percentile_252d: VIX percentile rank within last 252 days (0-100)
        - vix_roc_1d, vix_roc_5d, vix_roc_20d: VIX rate of change
        - vix_regime: Categorical (low/medium/high) based on percentile
        
        Role: Global risk gating, sizing, and LLM awareness
        """
        results: Dict[str, Any] = {}

        if self.vix_data is None or self.vix_data.empty:
            return {
                'vix_level': None,
                'vix_percentile_252d': None,
                'vix_roc_1d': None,
                'vix_roc_5d': None,
                'vix_roc_20d': None,
                'vix_regime': None,
            }

        # Strip timezone if present (VIX data is timezone-naive)
        if hasattr(date, 'tz') and date.tz is not None:
            date = date.replace(tzinfo=None)

        # Find VIX value for this date (or nearest prior trading date)
        try:
            # Try exact match first
            if date in self.vix_data.index:
                vix_idx = self.vix_data.index.get_loc(date)
            else:
                # Get nearest prior date
                mask = self.vix_data.index <= date
                if mask.any():
                    vix_idx = mask.nonzero()[0][-1]
                else:
                    return {
                        'vix_level': None,
                        'vix_percentile_252d': None,
                        'vix_roc_1d': None,
                        'vix_roc_5d': None,
                        'vix_roc_20d': None,
                        'vix_regime': None,
                    }
        except Exception as e:
            logger.warning(f"Error finding VIX for date {date}: {e}")
            return {
                'vix_level': None,
                'vix_percentile_252d': None,
                'vix_roc_1d': None,
                'vix_roc_5d': None,
                'vix_roc_20d': None,
                'vix_regime': None,
            }

        current_vix = self.vix_data.iloc[vix_idx]

        if pd.isna(current_vix):
            return {
                'vix_level': None,
                'vix_percentile_252d': None,
                'vix_roc_1d': None,
                'vix_roc_5d': None,
                'vix_roc_20d': None,
                'vix_regime': None,
            }

        results['vix_level'] = self.safe_float(current_vix)

        # VIX percentile rank (252d window)
        start_idx = max(0, vix_idx - 252)
        vix_window = self.vix_data.iloc[start_idx:vix_idx+1]
        vix_window_valid = vix_window.dropna()

        if len(vix_window_valid) > 0:
            percentile = stats.percentileofscore(vix_window_valid.values, current_vix, nan_policy='omit')
            results['vix_percentile_252d'] = self.safe_float(percentile)
        else:
            results['vix_percentile_252d'] = None

        # VIX ROC (1d, 5d, 20d)
        for lag in [1, 5, 20]:
            if vix_idx >= lag:
                prior_vix = self.vix_data.iloc[vix_idx - lag]
                if not pd.isna(prior_vix) and prior_vix != 0:
                    roc = (current_vix - prior_vix) / prior_vix
                    results[f'vix_roc_{lag}d'] = self.safe_float(roc)
                else:
                    results[f'vix_roc_{lag}d'] = None
            else:
                results[f'vix_roc_{lag}d'] = None

        # VIX regime (low/medium/high based on percentile)
        if results['vix_percentile_252d'] is not None:
            percentile = results['vix_percentile_252d']
            if percentile < 33:
                results['vix_regime'] = 'low'
            elif percentile < 67:
                results['vix_regime'] = 'medium'
            else:
                results['vix_regime'] = 'high'
        else:
            results['vix_regime'] = None

        # Binary flags for VIX regime (for parameter sweep testing)
        if results.get('vix_regime'):
            results['is_vix_low'] = 1 if results['vix_regime'] == 'low' else 0
            results['is_vix_medium'] = 1 if results['vix_regime'] == 'medium' else 0
            results['is_vix_high'] = 1 if results['vix_regime'] == 'high' else 0
        else:
            results['is_vix_low'] = None
            results['is_vix_medium'] = None
            results['is_vix_high'] = None

        return results

    def _classify_trend(self, series: np.ndarray) -> str:
        if len(series) < 5:
            return "flat"
        x = np.arange(len(series))
        valid = ~np.isnan(series)
        if np.sum(valid) < 5:
            return "flat"
        slope, _, _, p_value, _ = stats.linregress(x[valid], series[valid])
        if p_value < 0.05:
            if slope > 0:
                return "up"
            if slope < 0:
                return "down"
        return "flat"

    def _calculate_roll_spread(self) -> Optional[float]:
        if len(self.close) < 22:
            return None
        returns = np.diff(np.log(self.close[-22:]))
        if len(returns) < 2:
            return None
        cov = np.cov(returns[:-1], returns[1:])[0, 1]
        if cov < 0:
            return self.safe_float(2 * np.sqrt(-cov))
        return 0.0

    def _calculate_corwin_schultz_spread(self) -> Optional[float]:
        if len(self.high) < 21:
            return None

        high = self.high[-21:]
        low = self.low[-21:]

        hl_ratio = np.log(high / low) ** 2
        beta = np.sum(hl_ratio)

        high_2d = np.maximum(high[:-1], high[1:])
        low_2d = np.minimum(low[:-1], low[1:])
        gamma = np.sum(np.log(high_2d / low_2d) ** 2)

        if beta > 0 and gamma > 0:
            alpha = (
                (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
                - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
            )
            spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
            return self.safe_float(max(0.0, spread))
        return None

    def _calculate_liquidity_score(self, vol_metrics: Dict[str, Any]) -> int:
        score = 50

        dollar_vol = vol_metrics.get("avg_dollar_vol_21d")
        if dollar_vol:
            if dollar_vol > 50_000_000:
                score += 20
            elif dollar_vol > 10_000_000:
                score += 10
            elif dollar_vol < 1_000_000:
                score -= 20

        spread = vol_metrics.get("cs_spread_21")
        if spread is not None:
            if spread < 0.001:
                score += 15
            elif spread < 0.005:
                score += 5
            elif spread > 0.02:
                score -= 15

        vol_spike_z = vol_metrics.get("volume_spike_z")
        if vol_spike_z is not None:
            if abs(vol_spike_z) > 3:
                score -= 10
            elif abs(vol_spike_z) < 1:
                score += 10

        return int(max(0, min(100, score)))

    # =========================================================================
    # EVENTS
    # =========================================================================

    def calculate_price_events(self) -> Dict[str, Any]:
        """
        Calculate intraday price action and event flags.
        NOTE: Canonical pct_from_52w_high/low and is_new_52w_high/low are defined in
        calculate_range_structure_indicators() with 0.1% tolerance (strict "new high/low").
        This section defines near_52w_high/low_2pct for 2% "near high/low" context.
        """
        results: Dict[str, Any] = {}

        if len(self.close) >= 1:
            intraday_ret = (self.close[-1] / self.open[-1]) - 1.0
            results["intraday_ret"] = self.safe_float(intraday_ret)

            range_pct = (self.high[-1] - self.low[-1]) / self.close[-1]
            results["range_pct"] = self.safe_float(range_pct)

            lookback = min(252, len(self.close) - 1)
            if lookback >= 50:
                hist_intraday = (self.close[-lookback - 1 : -1] / self.open[-lookback - 1 : -1]) - 1.0
                z_intraday = self.robust_zscore(intraday_ret, hist_intraday)
                results["intraday_ret_zscore"] = self.safe_float(z_intraday) if z_intraday is not None else None

                hist_range = (self.high[-lookback - 1 : -1] - self.low[-lookback - 1 : -1]) / self.close[
                    -lookback - 1 : -1
                ]
                z_range = self.robust_zscore(range_pct, hist_range)
                results["range_pct_zscore"] = self.safe_float(z_range) if z_range is not None else None
            else:
                results["intraday_ret_zscore"] = None
                results["range_pct_zscore"] = None
        else:
            results["intraday_ret"] = None
            results["range_pct"] = None
            results["intraday_ret_zscore"] = None
            results["range_pct_zscore"] = None

        if len(self.close) >= 10:
            high_10d = np.max(self.high[-10:])
            low_10d = np.min(self.low[-10:])
            if high_10d > low_10d:
                close_loc = (self.close[-1] - low_10d) / (high_10d - low_10d)
                results["close_loc_10d"] = self.safe_float(float(np.clip(close_loc, 0.0, 1.0)))
            else:
                results["close_loc_10d"] = 0.5
        else:
            results["close_loc_10d"] = None

        if len(self.close) >= 2:
            prev_close = self.close[-2]
            today_open = self.open[-1]
            gap_pct = (today_open - prev_close) / prev_close

            results["gap_up"] = self.safe_bool(gap_pct > 0.02)
            results["gap_down"] = self.safe_bool(gap_pct < -0.02)
            results["gap_size_pct"] = self.safe_float(gap_pct)

            lookback = min(252, len(self.close) - 2)
            if lookback >= 50:
                hist_gaps = (self.open[-lookback - 1 : -1] - self.close[-lookback - 2 : -2]) / self.close[
                    -lookback - 2 : -2
                ]
                hist_gaps = hist_gaps[~np.isnan(hist_gaps)]
                z = self.robust_zscore(gap_pct, hist_gaps)
                results["gap_size_zscore"] = self.safe_float(z) if z is not None else None
            else:
                results["gap_size_zscore"] = None
        else:
            results["gap_up"] = False
            results["gap_down"] = False
            results["gap_size_pct"] = 0.0
            results["gap_size_zscore"] = None

        if len(self.close) >= 252:
            high_52w = np.max(self.high[-252:])
            low_52w = np.min(self.low[-252:])
            current_price = self.close[-1]

            # NOTE: Canonical pct_from_52w_high/low are defined in calculate_range_structure_indicators()
            # These flags are "near" definitions (2% tolerance), not "new" (which use 0.1% tolerance)
            results["near_52w_high_2pct"] = self.safe_bool(current_price >= high_52w * 0.98)
            results["near_52w_low_2pct"] = self.safe_bool(current_price <= low_52w * 1.02)
        else:
            results["near_52w_high_2pct"] = False
            results["near_52w_low_2pct"] = False

        if len(self.close) >= 20:
            recent_high = np.max(self.high[-10:])
            prev_high = np.max(self.high[-20:-10])
            recent_low = np.min(self.low[-10:])
            prev_low = np.min(self.low[-20:-10])

            results["higher_highs"] = self.safe_bool(recent_high > prev_high)
            results["higher_lows"] = self.safe_bool(recent_low > prev_low)
            results["prior_swing_break_up"] = self.safe_bool(self.close[-1] > prev_high)
            results["prior_swing_break_down"] = self.safe_bool(self.close[-1] < prev_low)
        else:
            results["higher_highs"] = False
            results["higher_lows"] = False
            results["prior_swing_break_up"] = False
            results["prior_swing_break_down"] = False

        return results

    # =========================================================================
    # RELATIVE PERFORMANCE
    # =========================================================================

    def calculate_relative_performance(
        self,
        benchmark_prices: Optional[np.ndarray] = None,
        sector_prices: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if benchmark_prices is not None and len(benchmark_prices) >= 253:
            for period, days in [("3m", 63), ("6m", 126), ("12m", 252)]:
                if len(self.close) >= days + 1 and len(benchmark_prices) >= days + 1:
                    asset_ret = (self.close[-1] / self.close[-days - 1]) - 1.0
                    bench_ret = (benchmark_prices[-1] / benchmark_prices[-days - 1]) - 1.0
                    results[f"vs_sp500_{period}"] = self.safe_float(asset_ret - bench_ret)
                else:
                    results[f"vs_sp500_{period}"] = None

            if all(results.get(f"vs_sp500_{p}") is not None for p in ["3m", "6m", "12m"]):
                rs_3m = results["vs_sp500_3m"]
                rs_6m = results["vs_sp500_6m"]
                rs_12m = results["vs_sp500_12m"]
                if rs_3m > rs_6m > rs_12m:
                    results["rel_strength_trend"] = "rising"
                elif rs_3m < rs_6m < rs_12m:
                    results["rel_strength_trend"] = "falling"
                else:
                    results["rel_strength_trend"] = "flat"
            else:
                results["rel_strength_trend"] = "flat"
        else:
            results["vs_sp500_3m"] = None
            results["vs_sp500_6m"] = None
            results["vs_sp500_12m"] = None
            results["rel_strength_trend"] = "flat"

        if sector_prices is not None and len(sector_prices) >= 64 and len(self.close) >= 64:
            asset_ret = (self.close[-1] / self.close[-64]) - 1.0
            sector_ret = (sector_prices[-1] / sector_prices[-64]) - 1.0
            results["rel_strength_sector_3m"] = self.safe_float(asset_ret - sector_ret)
        else:
            results["rel_strength_sector_3m"] = None

        return results

    # =========================================================================
    # MASTER
    # =========================================================================

    def calculate_all_indicators(
        self,
        symbol: str,
        sector: str = "Unknown",
        industry: str = "Unknown",
        benchmark_prices: Optional[np.ndarray] = None,
        benchmark_returns: Optional[np.ndarray] = None,
        sector_prices: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        logger.info(f"Calculating all indicators for {symbol}")

        asof = self.hist.index[-1].strftime("%Y-%m-%d")
        asof_date = self.hist.index[-1]
        last_price = self.safe_float(self.close[-1])

        payload: Dict[str, Any] = {
            "asof": asof,
            "id": {
                "symbol": symbol,
                "country": "US",
                "currency": "USD",
                "sector": sector,
                "industry": industry,
                "price": last_price,
            },
            "momentum": self.calculate_momentum(),
            "trend": self.calculate_trend_indicators(),
            "trend_quality": self.calculate_trend_quality_indicators(),
            "mean_reversion": self.calculate_mean_reversion_indicators(),
            "range_structure": self.calculate_range_structure_indicators(),
            "volatility": self.calculate_volatility_indicators(),
            "volume_liquidity": self.calculate_volume_liquidity_indicators(),
            "volume_participation": self.calculate_volume_participation_indicators(),
            "market_plumbing": {**self.calculate_vix_indicators(asof_date), **self.calculate_term_structure_indicators(asof_date)},
            "risk": self.calculate_risk_indicators(benchmark_returns),
            "events": self.calculate_price_events(),
            "relative_perf": self.calculate_relative_performance(benchmark_prices, sector_prices),
        }

        logger.info(f"Completed indicator calculation for {symbol}")
        return payload


def example_usage_with_tiingo():
    try:
        from technical_json_pipeline import fetch_tiingo_ohlcv
    except ImportError:
        logger.error("Cannot import fetch_tiingo_ohlcv. Make sure technical_json_pipeline.py is available.")
        return

    symbol = "AAPL"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=800)).strftime("%Y-%m-%d")

    logger.info(f"Fetching {symbol} data from Tiingo: {start_date} to {end_date}")
    hist = fetch_tiingo_ohlcv(symbol=symbol, start=start_date, end=end_date, cache_dir="./cache")
    if hist.empty:
        logger.error(f"No data returned for {symbol}")
        return

    logger.info(f"Fetched {len(hist)} days of adjusted OHLCV data for {symbol}")

    logger.info("Fetching SPY benchmark data...")
    spy_hist = fetch_tiingo_ohlcv(symbol="SPY", start=start_date, end=end_date, cache_dir="./cache")

    if not spy_hist.empty:
        common_dates = hist.index.intersection(spy_hist.index)
        hist_aligned = hist.loc[common_dates]
        spy_aligned = spy_hist.loc[common_dates]

        benchmark_prices = spy_aligned["Close"].values
        benchmark_returns = np.diff(np.log(benchmark_prices))
        logger.info(f"Aligned {len(common_dates)} dates between {symbol} and SPY")
    else:
        hist_aligned = hist
        benchmark_prices = None
        benchmark_returns = None
        logger.warning("Could not fetch SPY data, relative metrics will be None")

    calculator = TechnicalIndicatorCalculator(hist_aligned)

    indicators = calculator.calculate_all_indicators(
        symbol=symbol,
        sector="Technology",
        industry="Consumer Electronics",
        benchmark_prices=benchmark_prices,
        benchmark_returns=benchmark_returns,
    )

    print(json.dumps(indicators, indent=2))

    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{symbol}_technical_indicators.json"
    with open(output_path, "w") as f:
        json.dump(indicators, f, indent=2)

    logger.info(f"Saved indicators to {output_path}")
    return indicators


if __name__ == "__main__":
    example_usage_with_tiingo()
