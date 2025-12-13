import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import talib
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TechnicalIndicatorCalculator:
    def __init__(self, hist: pd.DataFrame):
        if not isinstance(hist.index, pd.DatetimeIndex):
            hist.index = pd.to_datetime(hist.index)
        
        self.hist = hist.sort_index()
        self.close = self.hist['Close'].values.astype(np.float64)
        self.high = self.hist['High'].values.astype(np.float64)
        self.low = self.hist['Low'].values.astype(np.float64)
        self.open = self.hist['Open'].values.astype(np.float64)
        self.volume = self.hist['Volume'].values.astype(np.float64)
        
        logger.info(f"Initialized calculator with {len(hist)} bars")
    
    def safe_float(self, val: Any, default: Optional[float] = None) -> Optional[float]:
        """Safely convert value to float, handling NaN."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            result = float(val)
            return result if not np.isnan(result) else default
        except (TypeError, ValueError):
            return default
    
    def safe_bool(self, val: Any) -> bool:
        """Safely convert to boolean, handling numpy booleans."""
        if val is None:
            return False
        # Handle numpy booleans
        if isinstance(val, (np.bool_, np.generic)):
            return bool(val)
        return bool(val)
    
    # =========================================================================
    # MOMENTUM INDICATORS
    # =========================================================================
    
    def calculate_momentum(self) -> Dict[str, Optional[float]]:
        """
        Calculate price momentum (returns) over multiple horizons.
        Returns normalized z-scores.
        """
        results = {}
        
        # Calculate raw returns
        periods = {
            'mom_1m': 21,   # ~1 month
            'mom_3m': 63,   # ~3 months
            'mom_6m': 126,  # ~6 months
            'mom_12m': 252  # ~12 months
        }
        
        for key, period in periods.items():
            if len(self.close) >= period + 1:
                ret = (self.close[-1] / self.close[-period-1]) - 1.0
                results[key] = self.safe_float(ret)
            else:
                results[key] = None
        
        return results
    
    # =========================================================================
    # TREND INDICATORS
    # =========================================================================
    
    def calculate_trend_indicators(self) -> Dict[str, Any]:
        """
        Calculate moving averages, MACD, RSI, ADX and trend structure.
        """
        results = {}
        
        # Moving Averages
        if len(self.close) >= 50:
            ma50 = talib.SMA(self.close, timeperiod=50)
            results['ma50'] = self.safe_float(ma50[-1])
            results['price_above_ma50'] = self.safe_bool(self.close[-1] > ma50[-1]) if not np.isnan(ma50[-1]) else None
        else:
            results['ma50'] = None
            results['price_above_ma50'] = None
        
        if len(self.close) >= 200:
            ma200 = talib.SMA(self.close, timeperiod=200)
            results['ma200'] = self.safe_float(ma200[-1])
            results['price_above_ma200'] = self.safe_bool(self.close[-1] > ma200[-1]) if not np.isnan(ma200[-1]) else None
            
            # Golden/Death Cross detection (check last 5 days)
            if len(self.close) >= 205:
                ma50_full = talib.SMA(self.close, timeperiod=50)
                ma200_full = talib.SMA(self.close, timeperiod=200)
                
                # Golden cross: MA50 crosses above MA200
                golden_cross = False
                death_cross = False
                
                for i in range(-5, 0):
                    if (not np.isnan(ma50_full[i]) and not np.isnan(ma200_full[i]) and
                        not np.isnan(ma50_full[i-1]) and not np.isnan(ma200_full[i-1])):
                        if ma50_full[i-1] <= ma200_full[i-1] and ma50_full[i] > ma200_full[i]:
                            golden_cross = True
                        if ma50_full[i-1] >= ma200_full[i-1] and ma50_full[i] < ma200_full[i]:
                            death_cross = True
                
                results['golden_cross_recent'] = golden_cross
                results['death_cross_recent'] = death_cross
            else:
                results['golden_cross_recent'] = False
                results['death_cross_recent'] = False
        else:
            results['ma200'] = None
            results['price_above_ma200'] = None
            results['golden_cross_recent'] = False
            results['death_cross_recent'] = False
        
        # Higher Highs/Lower Lows Analysis
        results['higher_highs_lows'] = self._analyze_price_structure()
        
        # MACD
        if len(self.close) >= 33:  # Need at least 26 + 9 - 1 bars
            macd_line, signal_line, histogram = talib.MACD(
                self.close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            results['macd_line'] = self.safe_float(macd_line[-1])
            results['macd_signal'] = self.safe_float(signal_line[-1])
            results['macd_histogram'] = self.safe_float(histogram[-1])
            
            # Detect recent crossovers (last 5 bars)
            macd_bull_cross = False
            macd_bear_cross = False
            if len(macd_line) >= 5:
                for i in range(-5, 0):
                    if (not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]) and
                        not np.isnan(macd_line[i-1]) and not np.isnan(signal_line[i-1])):
                        if macd_line[i-1] <= signal_line[i-1] and macd_line[i] > signal_line[i]:
                            macd_bull_cross = True
                        if macd_line[i-1] >= signal_line[i-1] and macd_line[i] < signal_line[i]:
                            macd_bear_cross = True
            
            results['macd_bull_cross'] = macd_bull_cross
            results['macd_bear_cross'] = macd_bear_cross
        else:
            results['macd_line'] = None
            results['macd_signal'] = None
            results['macd_histogram'] = None
            results['macd_bull_cross'] = False
            results['macd_bear_cross'] = False
        
        # RSI
        if len(self.close) >= 14:
            rsi = talib.RSI(self.close, timeperiod=14)
            results['rsi_14d'] = self.safe_float(rsi[-1])
            
            rsi_val = rsi[-1]
            results['rsi_overbought'] = self.safe_bool(rsi_val > 70 if not np.isnan(rsi_val) else False)
            results['rsi_oversold'] = self.safe_bool(rsi_val < 30 if not np.isnan(rsi_val) else False)
        else:
            results['rsi_14d'] = None
            results['rsi_overbought'] = False
            results['rsi_oversold'] = False
        
        # ADX (Trend Strength)
        if len(self.close) >= 14:
            adx = talib.ADX(self.high, self.low, self.close, timeperiod=14)
            results['adx_14d'] = self.safe_float(adx[-1])
        else:
            results['adx_14d'] = None
        
        return results
    
    def _analyze_price_structure(self) -> str:
        """
        Analyze if price is making higher highs/higher lows (uptrend),
        lower highs/lower lows (downtrend), or mixed.
        """
        if len(self.close) < 40:
            return "mixed"
        
        # Look at last 20 bars vs previous 20 bars
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
        elif lower_high and lower_low:
            return "downtrend"
        else:
            return "mixed"
    
    # =========================================================================
    # VOLATILITY INDICATORS
    # =========================================================================
    
    def calculate_volatility_indicators(self) -> Dict[str, Any]:
        """
        Calculate realized volatility, ATR, and volatility regime.
        """
        results = {}
        
        # Realized Volatility (annualized)
        if len(self.close) >= 22:
            returns_21d = np.diff(np.log(self.close[-22:]))
            realized_vol_21d = np.std(returns_21d, ddof=1) * np.sqrt(252)
            results['realized_vol_21d'] = self.safe_float(realized_vol_21d)
        else:
            results['realized_vol_21d'] = None
        
        if len(self.close) >= 63:
            returns_63d = np.diff(np.log(self.close[-64:]))
            realized_vol_63d = np.std(returns_63d, ddof=1) * np.sqrt(252)
            results['realized_vol_63d'] = self.safe_float(realized_vol_63d)
        else:
            results['realized_vol_63d'] = None
        
        if len(self.close) >= 253:
            returns_252d = np.diff(np.log(self.close[-253:]))
            realized_vol_252d = np.std(returns_252d, ddof=1) * np.sqrt(252)
            results['realized_vol_252d'] = self.safe_float(realized_vol_252d)
            
            # Vol ratio
            if results['realized_vol_21d'] is not None and realized_vol_252d > 0:
                results['vol_ratio_21v252'] = results['realized_vol_21d'] / realized_vol_252d
            else:
                results['vol_ratio_21v252'] = None
        else:
            results['realized_vol_252d'] = None
            results['vol_ratio_21v252'] = None
        
        # ATR (Average True Range)
        if len(self.close) >= 14:
            atr = talib.ATR(self.high, self.low, self.close, timeperiod=14)
            atr_val = self.safe_float(atr[-1])
            results['atr_14d'] = atr_val
            
            # ATR as percentage of price
            if atr_val and self.close[-1] > 0:
                results['atr_14d_pct'] = atr_val / self.close[-1]
            else:
                results['atr_14d_pct'] = None
        else:
            results['atr_14d'] = None
            results['atr_14d_pct'] = None
        
        # Volatility Regime Classification
        results['vol_regime'] = self._classify_vol_regime(
            results.get('realized_vol_252d'),
            results.get('realized_vol_21d')
        )
        
        # results['IV_rank'] = None
        
        return results
    
    def _classify_vol_regime(self, vol_252d: Optional[float], vol_21d: Optional[float]) -> str:
        """
        Classify volatility regime as high, neutral, or low.
        Uses 252-day vol and recent vol spike.
        """
        if vol_252d is None:
            return "neutral"
        
        # Simple classification based on annualized vol levels
        if vol_252d > 0.35:  # > 35% annualized
            return "high"
        elif vol_252d < 0.15:  # < 15% annualized
            return "low"
        else:
            return "neutral"
    
    # =========================================================================
    # RISK INDICATORS
    # =========================================================================
    
    def calculate_risk_indicators(self, benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate beta, correlation, Sharpe, Sortino, drawdowns.
        
        Args:
            benchmark_returns: Optional array of benchmark returns (e.g., S&P 500)
                             Should be aligned with self.close dates
        """
        results = {}
        
        if len(self.close) >= 253:
            # Calculate 1-year returns
            returns = np.diff(np.log(self.close[-253:]))
            
            # Beta and Correlation (if benchmark provided)
            if benchmark_returns is not None and len(benchmark_returns) >= 252:
                bench_ret = benchmark_returns[-252:]
                asset_ret = returns[-252:]
                
                if len(asset_ret) == len(bench_ret):
                    # Beta
                    covariance = np.cov(asset_ret, bench_ret)[0, 1]
                    benchmark_var = np.var(bench_ret, ddof=1)
                    if benchmark_var > 0:
                        results['beta_1y'] = self.safe_float(covariance / benchmark_var)
                    else:
                        results['beta_1y'] = None
                    
                    # Correlation
                    corr = np.corrcoef(asset_ret, bench_ret)[0, 1]
                    results['corr_sp500'] = self.safe_float(corr)
                    
                    # R-squared
                    results['r2_sp500'] = self.safe_float(corr ** 2) if not np.isnan(corr) else None
                else:
                    results['beta_1y'] = None
                    results['corr_sp500'] = None
                    results['r2_sp500'] = None
            else:
                results['beta_1y'] = None
                results['corr_sp500'] = None
                results['r2_sp500'] = None
            
            # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            if std_return > 0:
                sharpe = (mean_return * 252) / (std_return * np.sqrt(252))
                results['sharpe_1y'] = self.safe_float(sharpe)
            else:
                results['sharpe_1y'] = None
            
            # Sortino Ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns, ddof=1)
                if downside_std > 0:
                    sortino = (mean_return * 252) / (downside_std * np.sqrt(252))
                    results['sortino_1y'] = self.safe_float(sortino)
                else:
                    results['sortino_1y'] = None
            else:
                results['sortino_1y'] = None
            
            # Maximum Drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_dd = np.min(drawdown)
            results['max_drawdown_1y'] = self.safe_float(abs(max_dd))
            
            # Capture Ratios (if benchmark provided)
            if benchmark_returns is not None and len(benchmark_returns) >= 252:
                bench_ret = benchmark_returns[-252:]
                asset_ret = returns[-252:]
                
                # Up capture
                up_periods = bench_ret > 0
                if np.sum(up_periods) > 0:
                    up_asset = np.mean(asset_ret[up_periods])
                    up_bench = np.mean(bench_ret[up_periods])
                    if up_bench != 0:
                        results['up_capture_1y'] = self.safe_float(up_asset / up_bench)
                    else:
                        results['up_capture_1y'] = None
                else:
                    results['up_capture_1y'] = None
                
                # Down capture
                down_periods = bench_ret < 0
                if np.sum(down_periods) > 0:
                    down_asset = np.mean(asset_ret[down_periods])
                    down_bench = np.mean(bench_ret[down_periods])
                    if down_bench != 0:
                        results['down_capture_1y'] = self.safe_float(down_asset / down_bench)
                    else:
                        results['down_capture_1y'] = None
                else:
                    results['down_capture_1y'] = None
            else:
                results['up_capture_1y'] = None
                results['down_capture_1y'] = None
        else:
            results['beta_1y'] = None
            results['corr_sp500'] = None
            results['r2_sp500'] = None
            results['sharpe_1y'] = None
            results['sortino_1y'] = None
            results['max_drawdown_1y'] = None
            results['up_capture_1y'] = None
            results['down_capture_1y'] = None
        
        return results
    
    # =========================================================================
    # VOLUME & LIQUIDITY INDICATORS
    # =========================================================================
    
    def calculate_volume_liquidity_indicators(self) -> Dict[str, Any]:
        """
        Calculate volume-based indicators: OBV, A/D, volume spikes, liquidity metrics.
        """
        results = {}
        
        # Average Volume (3 months)
        if len(self.volume) >= 63:
            results['avg_vol_3m'] = self.safe_float(np.mean(self.volume[-63:]))
        else:
            results['avg_vol_3m'] = None
        
        # Average Dollar Volume (21 days)
        if len(self.volume) >= 21:
            dollar_vol = self.close[-21:] * self.volume[-21:]
            results['avg_dollar_vol_21d'] = self.safe_float(np.mean(dollar_vol))
        else:
            results['avg_dollar_vol_21d'] = None
        
        # Last Volume
        results['vol_last'] = self.safe_float(self.volume[-1])
        
        # Volume Spike (Z-score)
        if len(self.volume) >= 21:
            vol_mean = np.mean(self.volume[-21:])
            vol_std = np.std(self.volume[-21:], ddof=1)
            if vol_std > 0:
                z_score = (self.volume[-1] - vol_mean) / vol_std
                results['volume_spike_z'] = self.safe_float(z_score)
            else:
                results['volume_spike_z'] = None
        else:
            results['volume_spike_z'] = None
        
        # OBV Trend
        if len(self.close) >= 21:
            obv = talib.OBV(self.close, self.volume)
            obv_trend = self._classify_trend(obv[-21:])
            results['obv_trend_21d'] = obv_trend
        else:
            results['obv_trend_21d'] = "flat"
        
        # Accumulation/Distribution Trend
        if len(self.close) >= 21:
            ad = talib.AD(self.high, self.low, self.close, self.volume)
            ad_trend = self._classify_trend(ad[-21:])
            results['acc_dist_trend_21d'] = ad_trend
        else:
            results['acc_dist_trend_21d'] = "flat"
        
        # Roll Spread (simplified bid-ask spread estimator)
        results['roll_spread_21'] = self._calculate_roll_spread()
        
        # Corwin-Schultz Spread (high-low spread estimator)
        results['cs_spread_21'] = self._calculate_corwin_schultz_spread()
        
        # results['turnover_pct_21'] = None
        
        # Composite Liquidity Score (0-100)
        results['liquidity_score_0_100'] = self._calculate_liquidity_score(results)
        
        return results
    
    def _classify_trend(self, series: np.ndarray) -> str:
        """Classify if series is trending up, down, or flat using linear regression."""
        if len(series) < 5:
            return "flat"
        
        x = np.arange(len(series))
        valid_idx = ~np.isnan(series)
        
        if np.sum(valid_idx) < 5:
            return "flat"
        
        slope, _, _, p_value, _ = stats.linregress(x[valid_idx], series[valid_idx])
        
        # Check if slope is statistically significant (p < 0.05)
        if p_value < 0.05:
            if slope > 0:
                return "up"
            elif slope < 0:
                return "down"
        
        return "flat"
    
    def _calculate_roll_spread(self) -> Optional[float]:
        """
        Roll (1984) spread estimator using serial covariance of price changes.
        """
        if len(self.close) < 22:
            return None
        
        returns = np.diff(np.log(self.close[-22:]))
        if len(returns) < 2:
            return None
        
        # Serial covariance
        cov = np.cov(returns[:-1], returns[1:])[0, 1]
        
        if cov < 0:
            spread = 2 * np.sqrt(-cov)
            return self.safe_float(spread)
        else:
            return 0.0
    
    def _calculate_corwin_schultz_spread(self) -> Optional[float]:
        """
        Corwin-Schultz (2012) high-low spread estimator.
        """
        if len(self.high) < 21:
            return None
        
        high = self.high[-21:]
        low = self.low[-21:]
        
        # Calculate beta (high-low ratio)
        hl_ratio = np.log(high / low) ** 2
        beta = np.sum(hl_ratio)
        
        # Calculate gamma (two-day high-low)
        high_2d = np.maximum(high[:-1], high[1:])
        low_2d = np.minimum(low[:-1], low[1:])
        gamma = np.sum(np.log(high_2d / low_2d) ** 2)
        
        # Spread estimate
        if beta > 0 and gamma > 0:
            alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
            spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
            return self.safe_float(max(0, spread))
        else:
            return None
    
    def _calculate_liquidity_score(self, vol_metrics: Dict[str, Any]) -> Optional[int]:
        """
        Calculate composite liquidity score (0-100) based on volume and spread metrics.
        Higher score = better liquidity.
        """
        score = 50  # Start at neutral
        
        # Dollar volume component (±20 points)
        dollar_vol = vol_metrics.get('avg_dollar_vol_21d')
        if dollar_vol:
            if dollar_vol > 50_000_000:
                score += 20
            elif dollar_vol > 10_000_000:
                score += 10
            elif dollar_vol < 1_000_000:
                score -= 20
        
        # Spread component (±15 points)
        spread = vol_metrics.get('cs_spread_21')
        if spread is not None:
            if spread < 0.001:
                score += 15
            elif spread < 0.005:
                score += 5
            elif spread > 0.02:
                score -= 15
        
        # Volume consistency (±15 points)
        vol_spike_z = vol_metrics.get('volume_spike_z')
        if vol_spike_z is not None:
            # Penalize extreme volume spikes (can indicate liquidity stress)
            if abs(vol_spike_z) > 3:
                score -= 10
            elif abs(vol_spike_z) < 1:
                score += 10
        
        return max(0, min(100, score))
    
    # =========================================================================
    # PRICE PATTERNS & EVENTS
    # =========================================================================
    
    def calculate_price_events(self) -> Dict[str, Any]:
        """
        Detect gaps, 52-week highs/lows, and price structure breaks.
        """
        results = {}
        
        # Gap detection (compare today's open to yesterday's close)
        if len(self.close) >= 2:
            prev_close = self.close[-2]
            today_open = self.open[-1]
            gap_pct = (today_open - prev_close) / prev_close
            
            results['gap_up'] = self.safe_bool(gap_pct > 0.02)  # > 2% gap
            results['gap_down'] = self.safe_bool(gap_pct < -0.02)  # < -2% gap
            results['gap_size_pct'] = self.safe_float(gap_pct)
        else:
            results['gap_up'] = False
            results['gap_down'] = False
            results['gap_size_pct'] = 0.0
        
        # 52-week high/low
        if len(self.close) >= 252:
            high_52w = np.max(self.high[-252:])
            low_52w = np.min(self.low[-252:])
            current_price = self.close[-1]
            
            # Within 2% of 52w high/low
            results['new_52w_high'] = self.safe_bool(current_price >= high_52w * 0.98)
            results['new_52w_low'] = self.safe_bool(current_price <= low_52w * 1.02)
            
            results['pct_from_52w_high'] = self.safe_float((current_price - high_52w) / high_52w)
            results['pct_from_52w_low'] = self.safe_float((current_price - low_52w) / low_52w)
        else:
            results['new_52w_high'] = False
            results['new_52w_low'] = False
            results['pct_from_52w_high'] = None
            results['pct_from_52w_low'] = None
        
        # Higher highs / Higher lows (recent structure)
        if len(self.close) >= 20:
            recent_high = np.max(self.high[-10:])
            prev_high = np.max(self.high[-20:-10])
            recent_low = np.min(self.low[-10:])
            prev_low = np.min(self.low[-20:-10])
            
            results['higher_highs'] = self.safe_bool(recent_high > prev_high)
            results['higher_lows'] = self.safe_bool(recent_low > prev_low)
            
            # Swing breaks (price breaking above/below recent range)
            results['prior_swing_break_up'] = self.safe_bool(self.close[-1] > prev_high)
            results['prior_swing_break_down'] = self.safe_bool(self.close[-1] < prev_low)
        else:
            results['higher_highs'] = False
            results['higher_lows'] = False
            results['prior_swing_break_up'] = False
            results['prior_swing_break_down'] = False
        
        return results
    
    # =========================================================================
    # RELATIVE PERFORMANCE
    # =========================================================================
    
    def calculate_relative_performance(
        self, 
        benchmark_prices: Optional[np.ndarray] = None,
        sector_prices: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate relative strength vs benchmark and sector.
        
        Args:
            benchmark_prices: Array of benchmark prices (e.g., S&P 500) aligned with dates
            sector_prices: Array of sector index prices aligned with dates
        """
        results = {}
        
        # Relative strength vs benchmark (S&P 500)
        if benchmark_prices is not None and len(benchmark_prices) >= 253:
            for period, days in [('3m', 63), ('6m', 126), ('12m', 252)]:
                if len(self.close) >= days + 1 and len(benchmark_prices) >= days + 1:
                    asset_ret = (self.close[-1] / self.close[-days-1]) - 1.0
                    bench_ret = (benchmark_prices[-1] / benchmark_prices[-days-1]) - 1.0
                    rel_perf = asset_ret - bench_ret
                    results[f'vs_sp500_{period}'] = self.safe_float(rel_perf)
                else:
                    results[f'vs_sp500_{period}'] = None
            
            # Relative strength trend
            if all(results.get(f'vs_sp500_{p}') is not None for p in ['3m', '6m', '12m']):
                rs_3m = results['vs_sp500_3m']
                rs_6m = results['vs_sp500_6m']
                rs_12m = results['vs_sp500_12m']
                
                if rs_3m > rs_6m > rs_12m:
                    results['rel_strength_trend'] = "rising"
                elif rs_3m < rs_6m < rs_12m:
                    results['rel_strength_trend'] = "falling"
                else:
                    results['rel_strength_trend'] = "flat"
            else:
                results['rel_strength_trend'] = "flat"
        else:
            results['vs_sp500_3m'] = None
            results['vs_sp500_6m'] = None
            results['vs_sp500_12m'] = None
            results['rel_strength_trend'] = "flat"
        
        # Relative strength vs sector
        if sector_prices is not None and len(sector_prices) >= 64:
            if len(self.close) >= 64:
                asset_ret = (self.close[-1] / self.close[-64]) - 1.0
                sector_ret = (sector_prices[-1] / sector_prices[-64]) - 1.0
                rel_perf = asset_ret - sector_ret
                results['rel_strength_sector_3m'] = self.safe_float(rel_perf)
            else:
                results['rel_strength_sector_3m'] = None
        else:
            results['rel_strength_sector_3m'] = None
        
        # results['sector_rank_percentile'] = None
        
        return results
    
    # =========================================================================
    # MASTER FUNCTION: Calculate All Indicators
    # =========================================================================
    
    def calculate_all_indicators(
        self,
        symbol: str,
        sector: str = "Unknown",
        industry: str = "Unknown",
        benchmark_prices: Optional[np.ndarray] = None,
        benchmark_returns: Optional[np.ndarray] = None,
        sector_prices: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate all technical indicators and return complete JSON payload.
        
        Args:
            symbol: Ticker symbol
            sector: Company sector
            industry: Company industry
            benchmark_prices: S&P 500 or other benchmark price series
            benchmark_returns: S&P 500 returns (for beta/correlation)
            sector_prices: Sector index prices
        
        Returns:
            Complete technical metrics dictionary matching schema
        """
        logger.info(f"Calculating all indicators for {symbol}")
        
        asof = self.hist.index[-1].strftime('%Y-%m-%d')
        
        payload = {
            "asof": asof,
            "id": {
                "symbol": symbol,
                "country": "US",
                "currency": "USD",
                "sector": sector,
                "industry": industry
            },
            "momentum": self.calculate_momentum(),
            "trend": self.calculate_trend_indicators(),
            "volatility": self.calculate_volatility_indicators(),
            "risk": self.calculate_risk_indicators(benchmark_returns),
            "volume_liquidity": self.calculate_volume_liquidity_indicators(),
            "events": self.calculate_price_events(),
            "relative_perf": self.calculate_relative_performance(benchmark_prices, sector_prices)
        }
        
        logger.info(f"Completed indicator calculation for {symbol}")
        return payload


# =============================================================================
# Example Usage with Tiingo Integration
# =============================================================================

def example_usage_with_tiingo():
    """
    Example of how to use the TechnicalIndicatorCalculator with Tiingo data.
    Demonstrates integration with existing technical_json_pipeline.py
    """
    # Import Tiingo functions from technical_json_pipeline
    # Assumes technical_json_pipeline.py is in the same directory
    try:
        from technical_json_pipeline import fetch_tiingo_ohlcv
    except ImportError:
        logger.error("Cannot import fetch_tiingo_ohlcv. Make sure technical_json_pipeline.py is available.")
        return
    
    symbol = "AAPL"
    
    # Calculate date range (need 2+ years for all indicators)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=800)).strftime('%Y-%m-%d')
    
    logger.info(f"Fetching {symbol} data from Tiingo: {start_date} to {end_date}")
    
    # Fetch adjusted OHLCV data from Tiingo
    # Tiingo provides adjOpen, adjHigh, adjLow, adjClose - all adjusted for splits/dividends
    hist = fetch_tiingo_ohlcv(
        symbol=symbol,
        start=start_date,
        end=end_date,
        cache_dir="./cache"  # Optional: cache data locally
    )
    
    if hist.empty:
        logger.error(f"No data returned for {symbol}")
        return
    
    logger.info(f"Fetched {len(hist)} days of adjusted OHLCV data for {symbol}")
    
    # Optional: Get benchmark data (S&P 500) for relative metrics
    logger.info("Fetching SPY benchmark data...")
    spy_hist = fetch_tiingo_ohlcv(
        symbol="SPY",
        start=start_date,
        end=end_date,
        cache_dir="./cache"
    )
    
    # Align benchmark data with stock data
    if not spy_hist.empty:
        # Ensure same date range
        common_dates = hist.index.intersection(spy_hist.index)
        hist_aligned = hist.loc[common_dates]
        spy_aligned = spy_hist.loc[common_dates]
        
        benchmark_prices = spy_aligned['Close'].values
        benchmark_returns = np.diff(np.log(benchmark_prices))
        
        logger.info(f"Aligned {len(common_dates)} dates between {symbol} and SPY")
    else:
        hist_aligned = hist
        benchmark_prices = None
        benchmark_returns = None
        logger.warning("Could not fetch SPY data, relative metrics will be None")
    
    # Initialize calculator with adjusted data
    calculator = TechnicalIndicatorCalculator(hist_aligned)
    
    # Calculate all indicators
    indicators = calculator.calculate_all_indicators(
        symbol=symbol,
        sector="Technology",
        industry="Consumer Electronics",
        benchmark_prices=benchmark_prices,
        benchmark_returns=benchmark_returns
    )
    
    # Output to JSON
    print(json.dumps(indicators, indent=2))
    
    # Save to file
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{symbol}_technical_indicators.json"
    
    with open(output_path, 'w') as f:
        json.dump(indicators, f, indent=2)
    
    logger.info(f"Saved indicators to {output_path}")
    
    return indicators


if __name__ == "__main__":
    example_usage_with_tiingo()
