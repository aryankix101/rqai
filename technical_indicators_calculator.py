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

        self.close = self.hist["Close"].values.astype(np.float64)
        self.high = self.hist["High"].values.astype(np.float64)
        self.low = self.hist["Low"].values.astype(np.float64)
        self.open = self.hist["Open"].values.astype(np.float64)
        self.volume = self.hist["Volume"].values.astype(np.float64)

        logger.info(f"Initialized calculator with {len(hist)} bars")

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
        results: Dict[str, Optional[float]] = {}

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

        periods = {"mom_1m": 21, "mom_3m": 63, "mom_6m": 126, "mom_12m": 252}

        for key, period in periods.items():
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
        results: Dict[str, Any] = {}

        if len(self.close) >= 20:
            ma20 = talib.SMA(self.close, timeperiod=20)
            results["ma20"] = self.safe_float(ma20[-1])
            if not np.isnan(ma20[-1]):
                price_ma_ratio = self.close[-1] / ma20[-1] - 1
                if price_ma_ratio > MA_BUFFER:
                    results["price_above_ma20"] = True
                elif price_ma_ratio < -MA_BUFFER:
                    results["price_above_ma20"] = False
                else:
                    results["price_above_ma20"] = None  # Neutral zone
            else:
                results["price_above_ma20"] = None
        else:
            results["ma20"] = None
            results["price_above_ma20"] = None

        if len(self.close) >= 75:
            ma75 = talib.SMA(self.close, timeperiod=75)
            results["ma75"] = self.safe_float(ma75[-1])
            if not np.isnan(ma75[-1]):
                price_ma_ratio = self.close[-1] / ma75[-1] - 1
                if price_ma_ratio > MA_BUFFER:
                    results["price_above_ma75"] = True
                elif price_ma_ratio < -MA_BUFFER:
                    results["price_above_ma75"] = False
                else:
                    results["price_above_ma75"] = None  # Neutral zone
            else:
                results["price_above_ma75"] = None
        else:
            results["ma75"] = None
            results["price_above_ma75"] = None

        if len(self.close) >= 200:
            ma200 = talib.SMA(self.close, timeperiod=200)
            results["ma200"] = self.safe_float(ma200[-1])
            if not np.isnan(ma200[-1]):
                price_ma_ratio = self.close[-1] / ma200[-1] - 1
                if price_ma_ratio > MA_BUFFER:
                    results["price_above_ma200"] = True
                elif price_ma_ratio < -MA_BUFFER:
                    results["price_above_ma200"] = False
                else:
                    results["price_above_ma200"] = None  # Neutral zone
            else:
                results["price_above_ma200"] = None

            if len(ma200) >= 20 and not np.isnan(ma200[-1]) and not np.isnan(ma200[-20]):
                ma200_slope = (ma200[-1] - ma200[-20]) / ma200[-20]
                results["ma200_slope_20d"] = self.safe_float(ma200_slope)
                results["ma200_slope_up"] = self.safe_bool(ma200_slope > 0)
            else:
                results["ma200_slope_20d"] = None
                results["ma200_slope_up"] = None

            if len(self.close) >= 205:
                ma75_full = talib.SMA(self.close, timeperiod=75)
                ma200_full = talib.SMA(self.close, timeperiod=200)

                golden_cross = False
                death_cross = False
                for i in range(-5, 0):
                    if (
                        not np.isnan(ma75_full[i])
                        and not np.isnan(ma200_full[i])
                        and not np.isnan(ma75_full[i - 1])
                        and not np.isnan(ma200_full[i - 1])
                    ):
                        if ma75_full[i - 1] <= ma200_full[i - 1] and ma75_full[i] > ma200_full[i]:
                            golden_cross = True
                        if ma75_full[i - 1] >= ma200_full[i - 1] and ma75_full[i] < ma200_full[i]:
                            death_cross = True

                results["golden_cross_recent"] = golden_cross
                results["death_cross_recent"] = death_cross
            else:
                results["golden_cross_recent"] = False
                results["death_cross_recent"] = False
        else:
            results["ma200"] = None
            results["price_above_ma200"] = None
            results["ma200_slope_20d"] = None
            results["ma200_slope_up"] = None
            results["golden_cross_recent"] = False
            results["death_cross_recent"] = False

        results["higher_highs_lows"] = self._analyze_price_structure()

        if len(self.close) >= 33:
            macd_line, signal_line, histogram = talib.MACD(
                self.close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            results["macd_line"] = self.safe_float(macd_line[-1])
            results["macd_signal"] = self.safe_float(signal_line[-1])
            results["macd_histogram"] = self.safe_float(histogram[-1])

            macd_bull_cross = False
            macd_bear_cross = False
            if len(macd_line) >= 5:
                for i in range(-5, 0):
                    if (
                        not np.isnan(macd_line[i])
                        and not np.isnan(signal_line[i])
                        and not np.isnan(macd_line[i - 1])
                        and not np.isnan(signal_line[i - 1])
                    ):
                        if macd_line[i - 1] <= signal_line[i - 1] and macd_line[i] > signal_line[i]:
                            macd_bull_cross = True
                        if macd_line[i - 1] >= signal_line[i - 1] and macd_line[i] < signal_line[i]:
                            macd_bear_cross = True

            results["macd_bull_cross"] = macd_bull_cross
            results["macd_bear_cross"] = macd_bear_cross
        else:
            results["macd_line"] = None
            results["macd_signal"] = None
            results["macd_histogram"] = None
            results["macd_bull_cross"] = False
            results["macd_bear_cross"] = False

        if len(self.close) >= 3:
            rsi_2d = talib.RSI(self.close, timeperiod=2)
            results["rsi_2d"] = self.safe_float(rsi_2d[-1])
        else:
            results["rsi_2d"] = None

        if len(self.close) >= 4:
            rsi_3d = talib.RSI(self.close, timeperiod=3)
            results["rsi_3d"] = self.safe_float(rsi_3d[-1])
        else:
            results["rsi_3d"] = None

        if len(self.close) >= 14:
            rsi = talib.RSI(self.close, timeperiod=14)
            results["rsi_14d"] = self.safe_float(rsi[-1])
            rsi_val = rsi[-1]
            results["rsi_overbought"] = self.safe_bool(rsi_val > 70 if not np.isnan(rsi_val) else False)
            results["rsi_oversold"] = self.safe_bool(rsi_val < 30 if not np.isnan(rsi_val) else False)
        else:
            results["rsi_14d"] = None
            results["rsi_overbought"] = False
            results["rsi_oversold"] = False

        if len(self.close) >= 14:
            try:
                adx = talib.ADX(self.high, self.low, self.close, timeperiod=14)
                adx_val = self.safe_float(adx[-1])
                results["adx_14d"] = adx_val

                lookback = min(252, len(adx) - 1)
                if lookback >= 50 and adx_val is not None:
                    hist_adx = adx[-lookback - 1 : -1]
                    hist_adx = hist_adx[~np.isnan(hist_adx)]
                    z = self.robust_zscore(adx_val, hist_adx)
                    results["adx_14d_zscore"] = self.safe_float(z) if z is not None else None
                else:
                    results["adx_14d_zscore"] = None
            except Exception as e:
                print(f"ADX calculation error: {e}")
                results["adx_14d"] = None
                results["adx_14d_zscore"] = None
        else:
            results["adx_14d"] = None
            results["adx_14d_zscore"] = None

        return results

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
    # VOLATILITY
    # =========================================================================

    def calculate_volatility_indicators(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if len(self.close) >= 22:
            returns_21d = np.diff(np.log(self.close[-22:]))
            realized_vol_21d = np.std(returns_21d, ddof=1) * np.sqrt(252)
            results["realized_vol_21d"] = self.safe_float(realized_vol_21d)
        else:
            results["realized_vol_21d"] = None

        if len(self.close) >= 63:
            returns_63d = np.diff(np.log(self.close[-64:]))
            realized_vol_63d = np.std(returns_63d, ddof=1) * np.sqrt(252)
            results["realized_vol_63d"] = self.safe_float(realized_vol_63d)
        else:
            results["realized_vol_63d"] = None

        if len(self.close) >= 253:
            returns_252d = np.diff(np.log(self.close[-253:]))
            realized_vol_252d = np.std(returns_252d, ddof=1) * np.sqrt(252)
            results["realized_vol_252d"] = self.safe_float(realized_vol_252d)

            if len(self.close) >= 505:
                hist_vols = []
                for i in range(252):
                    end_idx = -(i + 1)
                    start_idx = end_idx - 252
                    if abs(start_idx) <= len(self.close):
                        hist_returns = np.diff(np.log(self.close[start_idx:end_idx]))
                        if len(hist_returns) >= 252:
                            hist_vol = np.std(hist_returns, ddof=1) * np.sqrt(252)
                            if not np.isnan(hist_vol):
                                hist_vols.append(hist_vol)
                hist_vols_arr = np.array(hist_vols, dtype=np.float64)
                z = self.robust_zscore(realized_vol_252d, hist_vols_arr)
                results["realized_vol_252d_zscore"] = self.safe_float(z) if z is not None else None
            else:
                results["realized_vol_252d_zscore"] = None

            if results["realized_vol_21d"] is not None and realized_vol_252d > 0:
                results["vol_ratio_21v252"] = results["realized_vol_21d"] / realized_vol_252d
            else:
                results["vol_ratio_21v252"] = None
        else:
            results["realized_vol_252d"] = None
            results["realized_vol_252d_zscore"] = None
            results["vol_ratio_21v252"] = None

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

        vol_regime_result = self._classify_vol_regime_percentile()
        results["vol_regime"] = vol_regime_result["regime"]
        results["vol_percentile"] = vol_regime_result["percentile"]

        return results

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

        if len(self.close) >= 21:
            obv = talib.OBV(self.close, self.volume)
            results["obv_trend_21d"] = self._classify_trend(obv[-21:])
        else:
            results["obv_trend_21d"] = "flat"

        if len(self.close) >= 21:
            ad = talib.AD(self.high, self.low, self.close, self.volume)
            results["acc_dist_trend_21d"] = self._classify_trend(ad[-21:])
        else:
            results["acc_dist_trend_21d"] = "flat"

        results["roll_spread_21"] = self._calculate_roll_spread()
        results["cs_spread_21"] = self._calculate_corwin_schultz_spread()
        results["liquidity_score_0_100"] = self._calculate_liquidity_score(results)

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

            results["new_52w_high"] = self.safe_bool(current_price >= high_52w * 0.98)
            results["new_52w_low"] = self.safe_bool(current_price <= low_52w * 1.02)
            results["pct_from_52w_high"] = self.safe_float((current_price - high_52w) / high_52w)
            results["pct_from_52w_low"] = self.safe_float((current_price - low_52w) / low_52w)
        else:
            results["new_52w_high"] = False
            results["new_52w_low"] = False
            results["pct_from_52w_high"] = None
            results["pct_from_52w_low"] = None

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
            "volatility": self.calculate_volatility_indicators(),
            "risk": self.calculate_risk_indicators(benchmark_returns),
            "volume_liquidity": self.calculate_volume_liquidity_indicators(),
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
