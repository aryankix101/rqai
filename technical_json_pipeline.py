import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import talib

# =============================================================================
# Configuration & Constants
# =============================================================================

TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"
LOOKBACK_DAYS = 252  # 1 year lookback for complete metrics
WEEKLY_STRIDE = 5
BB_WINDOW = 20
BB_K = 2.0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Layer: Tiingo API Integration
# =============================================================================

def get_api_key() -> str:
    """
    Retrieve Tiingo API key from secrets file or environment.
    """
    secrets_path = Path(__file__).parent / "secrets.json"
    if secrets_path.exists():
        try:
            with open(secrets_path, "r") as f:
                secrets = json.load(f)
                api_key = secrets.get("TIINGO_API_KEY")
                if api_key:
                    logger.debug("Loaded API key from secrets.json")
                    return api_key
        except Exception as e:
            logger.warning(f"Failed to read secrets.json: {e}")


def create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_tiingo_ohlcv(
    symbol: str,
    start: str,
    end: str,
    *,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch daily adjusted OHLCV from Tiingo with caching support.
    
    Args:
        symbol: Ticker symbol (e.g., 'AAPL')
        start: Start date 'YYYY-MM-DD'
        end: End date 'YYYY-MM-DD'
        cache_dir: Optional directory for CSV caching
    
    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] and DatetimeIndex
    """
    api_key = get_api_key()
    
    # Setup cache
    cache_file = None
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / f"{symbol}_tiingo_eod.csv"
    
    # Try loading from cache
    cached_df = None
    if cache_file and cache_file.exists():
        try:
            cached_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            cached_df.index = pd.to_datetime(cached_df.index).date
            logger.info(f"Loaded {len(cached_df)} cached rows for {symbol}")
        except Exception as e:
            logger.warning(f"Cache read error for {symbol}: {e}")
    
    # Determine date range to fetch
    start_dt = pd.to_datetime(start).date()
    end_dt = pd.to_datetime(end).date()
    
    need_fetch = True
    if cached_df is not None and len(cached_df) > 0:
        cache_start = min(cached_df.index)
        cache_end = max(cached_df.index)
        if cache_start <= start_dt and cache_end >= end_dt:
            # Cache fully covers requested range
            need_fetch = False
            result_df = cached_df.loc[start_dt:end_dt].copy()
        else:
            # Partial cache - fetch missing data
            fetch_start = min(start_dt, cache_start)
            fetch_end = max(end_dt, cache_end)
            start = fetch_start.isoformat()
            end = fetch_end.isoformat()
    
    if need_fetch:
        # Fetch from Tiingo
        url = f"{TIINGO_BASE_URL}/{symbol}/prices"
        params = {
            "startDate": start,
            "endDate": end,
            "format": "json",
            "resampleFreq": "daily",
            "token": api_key
        }
        
        session = create_session()
        try:
            logger.info(f"Fetching {symbol} from Tiingo: {start} to {end}")
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning(f"Empty response for {symbol}")
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            
            # Validate schema
            required_fields = ["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "volume"]
            if not all(field in data[0] for field in required_fields):
                raise ValueError(f"Missing required fields in Tiingo response for {symbol}")
            
            # Normalize to standard OHLCV
            records = []
            for row in data:
                dt = pd.to_datetime(row["date"]).date()
                records.append({
                    "date": dt,
                    "Open": row["adjOpen"],
                    "High": row["adjHigh"],
                    "Low": row["adjLow"],
                    "Close": row["adjClose"],
                    "Volume": row["volume"]
                })
            
            fetched_df = pd.DataFrame(records)
            fetched_df.set_index("date", inplace=True)
            fetched_df.sort_index(inplace=True)
            fetched_df = fetched_df[~fetched_df.index.duplicated(keep='last')]
            
            # Merge with cache
            if cached_df is not None:
                combined_df = pd.concat([cached_df, fetched_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df.sort_index(inplace=True)
            else:
                combined_df = fetched_df
            
            # Save to cache
            if cache_file:
                combined_df.to_csv(cache_file)
                logger.info(f"Saved {len(combined_df)} rows to cache for {symbol}")
            
            # Extract requested range
            result_df = combined_df.loc[start_dt:end_dt].copy()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching {symbol}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status: {e.response.status_code}, Body: {e.response.text[:200]}")
            raise
    
    # Convert index to DatetimeIndex for compatibility
    result_df.index = pd.to_datetime(result_df.index)
    
    logger.info(f"Returning {len(result_df)} rows for {symbol}")
    return result_df


def fetch_many(
    symbols: List[str],
    start: str,
    end: str,
    cache_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple symbols.
    
    Args:
        symbols: List of ticker symbols
        start: Start date 'YYYY-MM-DD'
        end: End date 'YYYY-MM-DD'
        cache_dir: Optional cache directory
    
    Returns:
        Dictionary mapping symbol to DataFrame
    """
    results = {}
    for symbol in symbols:
        try:
            df = fetch_tiingo_ohlcv(symbol, start, end, cache_dir=cache_dir)
            results[symbol] = df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            results[symbol] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    return results


# =============================================================================
# Analysis Layer: Feature Computation
# =============================================================================

def safe_float(val: Any) -> Optional[float]:
    """Convert numpy/pandas types to Python float, handle NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        result = float(val)
        return None if np.isnan(result) or np.isinf(result) else result
    except (TypeError, ValueError):
        return None


def safe_int(val: Any) -> Optional[int]:
    """Convert to Python int, handle NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def safe_bool(val: Any) -> Optional[bool]:
    """Convert to Python bool."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return bool(val)


def compute_momentum(hist: pd.DataFrame) -> Dict[str, Any]:
    """Compute momentum metrics."""
    close = hist["Close"].values.astype(np.float64)
    
    # Simple returns
    mom_1m = safe_float((close[-1] / close[-21] - 1) if len(close) >= 21 else None)
    mom_3m = safe_float((close[-1] / close[-63] - 1) if len(close) >= 63 else None)
    mom_6m = safe_float((close[-1] / close[-126] - 1) if len(close) >= 126 else None)
    mom_12m = safe_float((close[-1] / close[-252] - 1) if len(close) >= 252 else None)
    
    # Z-score combo
    moms = [mom_1m, mom_3m, mom_6m]
    valid_moms = [m for m in moms if m is not None]
    if len(valid_moms) >= 2:
        mean_mom = np.mean(valid_moms)
        std_mom = np.std(valid_moms)
        z_mom_combo = safe_float(mean_mom / std_mom if std_mom > 0 else 0)
    else:
        z_mom_combo = None
    
    return {
        "mom_1m": mom_1m,
        "mom_3m": mom_3m,
        "mom_6m": mom_6m,
        "mom_12m": mom_12m,
        "z_mom_combo": z_mom_combo
    }


def compute_trend(hist: pd.DataFrame) -> Dict[str, Any]:
    """Compute trend metrics."""
    close = hist["Close"].values.astype(np.float64)
    
    # Moving averages
    sma_50 = safe_float(talib.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else None)
    sma_200 = safe_float(talib.SMA(close, timeperiod=200)[-1] if len(close) >= 200 else None)
    
    price_above_sma50 = safe_bool(close[-1] > sma_50 if sma_50 else None)
    price_above_sma200 = safe_bool(close[-1] > sma_200 if sma_200 else None)
    
    # Golden/Death cross
    golden_cross_recent = False
    death_cross_recent = False
    if sma_50 and sma_200 and len(close) >= 220:
        sma50_series = talib.SMA(close, timeperiod=50)
        sma200_series = talib.SMA(close, timeperiod=200)
        last_20 = min(20, len(sma50_series))
        for i in range(1, last_20):
            if not np.isnan(sma50_series[-i-1]) and not np.isnan(sma200_series[-i-1]) and not np.isnan(sma50_series[-i]) and not np.isnan(sma200_series[-i]):
                if sma50_series[-i-1] <= sma200_series[-i-1] and sma50_series[-i] > sma200_series[-i]:
                    golden_cross_recent = True
                    break
        if not golden_cross_recent:
            for i in range(1, last_20):
                if not np.isnan(sma50_series[-i-1]) and not np.isnan(sma200_series[-i-1]) and not np.isnan(sma50_series[-i]) and not np.isnan(sma200_series[-i]):
                    if sma50_series[-i-1] >= sma200_series[-i-1] and sma50_series[-i] < sma200_series[-i]:
                        death_cross_recent = True
                        break
    
    # MACD
    macd_line = macd_signal = macd_hist = None
    macd_bullish_cross = False
    macd_bearish_cross = False
    if len(close) >= 34:
        macd_vals, signal_vals, hist_vals = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_line = safe_float(macd_vals[-1])
        macd_signal = safe_float(signal_vals[-1])
        macd_hist = safe_float(hist_vals[-1])
        
        # Check for recent crosses (last 10 bars)
        last_10 = min(10, len(hist_vals))
        for i in range(1, last_10):
            if not np.isnan(hist_vals[-i-1]) and not np.isnan(hist_vals[-i]):
                if hist_vals[-i-1] <= 0 and hist_vals[-i] > 0:
                    macd_bullish_cross = True
                    break
        if not macd_bullish_cross:
            for i in range(1, last_10):
                if not np.isnan(hist_vals[-i-1]) and not np.isnan(hist_vals[-i]):
                    if hist_vals[-i-1] >= 0 and hist_vals[-i] < 0:
                        macd_bearish_cross = True
                        break
    
    # RSI
    rsi = safe_float(talib.RSI(close, timeperiod=14)[-1] if len(close) >= 14 else None)
    
    # ADX
    if len(hist) >= 14:
        adx = safe_float(talib.ADX(hist["High"].values.astype(np.float64), hist["Low"].values.astype(np.float64), close, timeperiod=14)[-1])
    else:
        adx = None
    
    return {
        "sma_50": sma_50,
        "sma_200": sma_200,
        "price_above_sma50": safe_bool(price_above_sma50),
        "price_above_sma200": safe_bool(price_above_sma200),
        "golden_cross_recent": safe_bool(golden_cross_recent),
        "death_cross_recent": safe_bool(death_cross_recent),
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_hist,
        "macd_bullish_cross_recent": safe_bool(macd_bullish_cross),
        "macd_bearish_cross_recent": safe_bool(macd_bearish_cross),
        "rsi_14": rsi,
        "adx_14": adx
    }


def compute_volatility_risk(hist: pd.DataFrame) -> Dict[str, Any]:
    """Compute volatility and risk metrics."""
    close = hist["Close"].values.astype(np.float64)
    returns = np.diff(np.log(close))
    
    # Realized volatility (annualized)
    vol_21 = safe_float(np.std(returns[-21:]) * np.sqrt(252) if len(returns) >= 21 else None)
    vol_63 = safe_float(np.std(returns[-63:]) * np.sqrt(252) if len(returns) >= 63 else None)
    vol_252 = safe_float(np.std(returns) * np.sqrt(252) if len(returns) >= 251 else None)
    
    vol_ratio = safe_float(vol_21 / vol_252 if vol_21 and vol_252 and vol_252 > 0 else None)
    
    # ATR
    if len(hist) >= 14:
        atr = safe_float(talib.ATR(hist["High"].values.astype(np.float64), hist["Low"].values.astype(np.float64), close, timeperiod=14)[-1])
        atr_pct = safe_float(atr / close[-1] * 100 if atr else None)
    else:
        atr = atr_pct = None
    
    # Sharpe/Sortino (1y, rf=0)
    sharpe_1y = sortino_1y = None
    if len(returns) >= 251:
        ret_1y = returns  # Use all available returns in the window
        mean_ret = np.mean(ret_1y) * 252
        std_ret = np.std(ret_1y) * np.sqrt(252)
        sharpe_1y = safe_float(mean_ret / std_ret if std_ret > 0 else None)
        
        downside = ret_1y[ret_1y < 0]
        if len(downside) > 0:
            downside_std = np.std(downside) * np.sqrt(252)
            sortino_1y = safe_float(mean_ret / downside_std if downside_std > 0 else None)
    
    # Max drawdown (1y)
    max_dd = None
    if len(close) >= 252:
        prices_1y = close  # Use all prices in the window
        cummax = np.maximum.accumulate(prices_1y)
        drawdowns = (prices_1y - cummax) / cummax
        max_dd = safe_float(np.min(drawdowns))
    
    # Distance from 52w high/low
    pct_from_52w_high = pct_from_52w_low = None
    if len(close) >= 252:
        high_52w = np.max(close)
        low_52w = np.min(close)
        pct_from_52w_high = safe_float((close[-1] - high_52w) / high_52w)
        pct_from_52w_low = safe_float((close[-1] - low_52w) / low_52w)
    
    return {
        "realized_vol_21d": vol_21,
        "realized_vol_63d": vol_63,
        "realized_vol_252d": vol_252,
        "vol_ratio_21vs252": vol_ratio,
        "atr_14": atr,
        "atr_pct": atr_pct,
        "sharpe_1y": sharpe_1y,
        "sortino_1y": sortino_1y,
        "max_drawdown_1y": max_dd,
        "pct_from_52w_high": pct_from_52w_high,
        "pct_from_52w_low": pct_from_52w_low
    }


def compute_volume_liquidity(hist: pd.DataFrame) -> Dict[str, Any]:
    """Compute volume and liquidity metrics."""
    close = hist["Close"].values.astype(np.float64)
    volume = hist["Volume"].values.astype(np.float64)
    
    # Average daily volume
    adv_21 = safe_float(np.mean(volume[-21:]) if len(volume) >= 21 else None)
    dollar_adv_21 = safe_float(adv_21 * close[-1] if adv_21 else None)
    
    # Volume spike (z-score)
    z_volume_spike = None
    if len(volume) >= 21:
        vol_mean = np.mean(volume[-21:])
        vol_std = np.std(volume[-21:])
        if vol_std > 0:
            z_volume_spike = safe_float((volume[-1] - vol_mean) / vol_std)
    
    # OBV trend (21d slope z-score)
    obv_trend_21d = None
    if len(hist) >= 21:
        obv = talib.OBV(close, volume)
        obv_window = obv[-21:]
        x = np.arange(len(obv_window))
        slope = np.polyfit(x, obv_window, 1)[0]
        slope_std = np.std(np.diff(obv_window))
        obv_trend_21d = safe_float(slope / slope_std if slope_std > 0 else 0)
    
    # A/D Line trend (21d)
    ad_trend_21d = None
    if len(hist) >= 21:
        ad = talib.AD(hist["High"].values.astype(np.float64), hist["Low"].values.astype(np.float64), close, volume)
        ad_window = ad[-21:]
        x = np.arange(len(ad_window))
        slope = np.polyfit(x, ad_window, 1)[0]
        slope_std = np.std(np.diff(ad_window))
        ad_trend_21d = safe_float(slope / slope_std if slope_std > 0 else 0)
    
    # Amihud illiquidity
    amihud_21 = amihud_63 = None
    returns = np.diff(np.log(close))
    if len(returns) >= 21 and len(volume) >= 21:
        dollar_vol = close[1:] * volume[1:]
        illiq = np.abs(returns) / (dollar_vol + 1e-10)
        amihud_21 = safe_float(np.mean(illiq[-21:]) * 1e6)
    if len(returns) >= 63 and len(volume) >= 63:
        dollar_vol = close[1:] * volume[1:]
        illiq = np.abs(returns) / (dollar_vol + 1e-10)
        amihud_63 = safe_float(np.mean(illiq[-63:]) * 1e6)
    
    # Roll spread estimator
    roll_spread_21 = None
    if len(returns) >= 22:
        ret_window = returns[-22:]
        if len(ret_window) >= 2:
            cov = np.cov(ret_window[:-1], ret_window[1:])[0, 1]
            if cov < 0:
                roll_spread_21 = safe_float(2 * np.sqrt(-cov))
            else:
                roll_spread_21 = 0.0  # No negative covariance means no roll spread
    
    # Corwin-Schultz spread estimator
    corwin_schultz_21 = None
    if len(hist) >= 21:
        high = hist["High"].values[-21:]
        low = hist["Low"].values[-21:]
        hl_ratio = np.log(high / low)
        beta = np.sum(hl_ratio ** 2)
        gamma = np.sum((np.log(high[1:] / low[:-1])) ** 2)
        if beta > 0:
            alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
            corwin_schultz_21 = safe_float(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)))
    
    # Volume volatility
    vol_of_volume_63 = safe_float(np.std(volume[-63:]) / np.mean(volume[-63:]) if len(volume) >= 63 else None)
    
    # Zero return days
    zero_ret_days_63 = None
    if len(returns) >= 63:
        zero_days = np.sum(np.abs(returns[-63:]) < 1e-6)
        zero_ret_days_63 = safe_int(zero_days)
    
    # Gap frequency
    gap_freq_63 = None
    if len(hist) >= 64:
        gaps = np.abs((hist["Open"].values[1:] - hist["Close"].values[:-1]) / hist["Close"].values[:-1])
        large_gaps = np.sum(gaps[-63:] > 0.01)
        gap_freq_63 = safe_int(large_gaps)
    
    # Capacity (simple estimate: 21d ADV * close * 0.1)
    capacity_usd_21 = safe_float(adv_21 * close[-1] * 0.1 if adv_21 else None)
    
    # Liquidity score (0-100)
    liquidity_score = None
    components = []
    if dollar_adv_21 and dollar_adv_21 > 0:
        components.append(min(np.log10(dollar_adv_21) / 8 * 25, 25))  # 0-25 pts
    if amihud_21 is not None:
        components.append(max(0, 25 - amihud_21 * 5))  # 0-25 pts
    if roll_spread_21 is not None:
        components.append(max(0, 25 - roll_spread_21 * 100))  # 0-25 pts
    if vol_of_volume_63 is not None:
        components.append(max(0, 25 - vol_of_volume_63 * 10))  # 0-25 pts
    
    if len(components) >= 2:
        liquidity_score = safe_float(min(100, sum(components)))
    
    # Filters
    min_dollar_volume_filter = safe_bool(dollar_adv_21 >= 1e6 if dollar_adv_21 else False)
    min_liquidity_score_filter = safe_bool(liquidity_score >= 30 if liquidity_score else False)
    
    return {
        "adv_21d": adv_21,
        "dollar_adv_21d": dollar_adv_21,
        "z_volume_spike": z_volume_spike,
        "obv_trend_21d": safe_float(obv_trend_21d),
        "ad_trend_21d": safe_float(ad_trend_21d),
        "amihud_illiq_21d": amihud_21,
        "amihud_illiq_63d": amihud_63,
        "roll_spread_21d": roll_spread_21,
        "corwin_schultz_spread_21d": corwin_schultz_21,
        "vol_of_volume_63d": vol_of_volume_63,
        "zero_ret_days_63d": zero_ret_days_63,
        "gap_freq_63d": gap_freq_63,
        "capacity_usd_21d": capacity_usd_21,
        "liquidity_score_0_100": liquidity_score,
        "min_dollar_volume_filter": min_dollar_volume_filter,
        "min_liquidity_score_filter": safe_bool(min_liquidity_score_filter)
    }


def compute_price_events(hist: pd.DataFrame) -> Dict[str, Any]:
    """Compute price event flags."""
    close = hist["Close"].values.astype(np.float64)
    high = hist["High"].values.astype(np.float64)
    low = hist["Low"].values.astype(np.float64)
    
    # Recent gaps (last 5 days)
    large_gap_up_recent = False
    large_gap_down_recent = False
    if len(hist) >= 6:
        for i in range(1, 6):
            gap = (hist["Open"].iloc[-i] - hist["Close"].iloc[-i-1]) / hist["Close"].iloc[-i-1]
            if gap > 0.01:
                large_gap_up_recent = True
            if gap < -0.01:
                large_gap_down_recent = True
    
    # 52w high/low recent
    near_52w_high = False
    near_52w_low = False
    if len(close) >= 252:
        high_52w = np.max(high)
        low_52w = np.min(low)
        if close[-1] >= high_52w * 0.98:
            near_52w_high = True
        if close[-1] <= low_52w * 1.02:
            near_52w_low = True
    
    # Higher highs / lower lows (pivot analysis)
    higher_highs_lows = False
    lower_highs_lows = False
    if len(high) >= 20:
        recent_highs = [high[-i] for i in range(1, 11) if i < len(high)]
        recent_lows = [low[-i] for i in range(1, 11) if i < len(low)]
        older_highs = [high[-i] for i in range(11, 21) if i < len(high)]
        older_lows = [low[-i] for i in range(11, 21) if i < len(low)]
        
        if len(recent_highs) >= 5 and len(older_highs) >= 5 and len(recent_lows) >= 5 and len(older_lows) >= 5:
            if max(recent_highs) > max(older_highs) and min(recent_lows) > min(older_lows):
                higher_highs_lows = True
            elif max(recent_highs) < max(older_highs) and min(recent_lows) < min(older_lows):
                lower_highs_lows = True
    
    # Bollinger breakout
    bb_breakout_recent = False
    if len(close) >= BB_WINDOW + 5:
        upper, middle, lower = talib.BBANDS(close, timeperiod=BB_WINDOW, nbdevup=BB_K, nbdevdn=BB_K)
        for i in range(1, 6):
            if not np.isnan(upper[-i]) and close[-i] > upper[-i]:
                bb_breakout_recent = True
                break
    
    # Volatility regime
    vol_regime = None
    if len(close) >= 252:
        returns = np.diff(np.log(close))
        rolling_vol = pd.Series(returns).rolling(21).std() * np.sqrt(252)
        vol_all = rolling_vol.dropna()
        if len(vol_all) > 0:
            terciles = np.percentile(vol_all, [33.33, 66.67])
            current_vol = rolling_vol.iloc[-1]
            if pd.notna(current_vol):
                if current_vol <= terciles[0]:
                    vol_regime = "low"
                elif current_vol <= terciles[1]:
                    vol_regime = "medium"
                else:
                    vol_regime = "high"
    
    return {
        "large_gap_up_recent": large_gap_up_recent,
        "large_gap_down_recent": large_gap_down_recent,
        "near_52w_high": safe_bool(near_52w_high),
        "near_52w_low": safe_bool(near_52w_low),
        "higher_highs_lows": safe_bool(higher_highs_lows),
        "lower_highs_lows": safe_bool(lower_highs_lows),
        "bb_breakout_recent": safe_bool(bb_breakout_recent),
        "vol_regime": vol_regime
    }


def compute_relative_performance(
    hist: pd.DataFrame,
    benchmark_hist: Optional[pd.Series] = None,
    sector_hist: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """Compute relative performance metrics."""
    if benchmark_hist is None:
        return {
            "excess_ret_3m": None,
            "excess_ret_6m": None,
            "excess_ret_12m": None,
            "rel_strength_trend": None,
            "up_capture_1y": None,
            "down_capture_1y": None
        }
    
    close = hist["Close"].values
    bench = benchmark_hist.values
    
    # Align lengths
    min_len = min(len(close), len(bench))
    close = close[-min_len:]
    bench = bench[-min_len:]
    
    # Excess returns
    excess_3m = safe_float((close[-1]/close[-63] - bench[-1]/bench[-63]) if min_len >= 63 else None)
    excess_6m = safe_float((close[-1]/close[-126] - bench[-1]/bench[-126]) if min_len >= 126 else None)
    excess_12m = safe_float((close[-1]/close[-252] - bench[-1]/bench[-252]) if min_len >= 252 else None)
    
    # Relative strength trend (slope z-score of ratio)
    rs_trend = None
    if min_len >= 63:
        ratio = close[-63:] / bench[-63:]
        x = np.arange(len(ratio))
        slope = np.polyfit(x, ratio, 1)[0]
        slope_std = np.std(np.diff(ratio))
        rs_trend = safe_float(slope / slope_std if slope_std > 0 else 0)
    
    # Up/Down capture
    up_capture = down_capture = None
    if min_len >= 252:
        ret_asset = np.diff(close) / close[:-1]
        ret_bench = np.diff(bench) / bench[:-1]
        
        up_days = ret_bench > 0
        down_days = ret_bench < 0
        
        if np.sum(up_days) > 0:
            avg_up_asset = np.mean(ret_asset[up_days])
            avg_up_bench = np.mean(ret_bench[up_days])
            up_capture = safe_float(avg_up_asset / avg_up_bench if avg_up_bench != 0 else None)
        
        if np.sum(down_days) > 0:
            avg_down_asset = np.mean(ret_asset[down_days])
            avg_down_bench = np.mean(ret_bench[down_days])
            down_capture = safe_float(avg_down_asset / avg_down_bench if avg_down_bench != 0 else None)
    
    return {
        "excess_ret_3m": excess_3m,
        "excess_ret_6m": excess_6m,
        "excess_ret_12m": excess_12m,
        "rel_strength_trend": rs_trend,
        "up_capture_1y": up_capture,
        "down_capture_1y": down_capture
    }


def compute_composites(
    momentum: Dict,
    trend: Dict,
    vol_risk: Dict,
    volume: Dict,
    relative: Dict
) -> Dict[str, Any]:
    """Compute composite scores."""
    
    # Momentum score (-100 to +100)
    momentum_score = 0
    if momentum["mom_1m"]: momentum_score += momentum["mom_1m"] * 25
    if momentum["mom_3m"]: momentum_score += momentum["mom_3m"] * 25
    if momentum["mom_6m"]: momentum_score += momentum["mom_6m"] * 25
    if momentum["z_mom_combo"]: momentum_score += momentum["z_mom_combo"] * 25
    momentum_score = safe_float(np.clip(momentum_score, -100, 100))
    
    # Trend score (-100 to +100)
    trend_score = 0
    if trend["price_above_sma50"]: trend_score += 25
    if trend["price_above_sma200"]: trend_score += 25
    if trend["rsi_14"]: trend_score += (trend["rsi_14"] - 50) * 0.5  # -25 to +25
    if trend["adx_14"]: trend_score += min(trend["adx_14"], 50) * 0.5  # 0 to +25
    trend_score = safe_float(np.clip(trend_score, -100, 100))
    
    # Risk score (0 to 100, higher = lower risk)
    risk_score = 50
    if vol_risk["vol_ratio_21vs252"] and vol_risk["vol_ratio_21vs252"] < 1:
        risk_score += 20
    elif vol_risk["vol_ratio_21vs252"] and vol_risk["vol_ratio_21vs252"] > 1.5:
        risk_score -= 20
    if vol_risk["sharpe_1y"]: risk_score += min(max(vol_risk["sharpe_1y"] * 10, -25), 25)
    risk_score = safe_float(np.clip(risk_score, 0, 100))
    
    # Volume confirmation score (0 to 100)
    vol_confirm = 50
    if volume["z_volume_spike"] and volume["z_volume_spike"] > 1: vol_confirm += 15
    if volume["obv_trend_21d"] and volume["obv_trend_21d"] > 0: vol_confirm += 15
    if volume["liquidity_score_0_100"]: vol_confirm += (volume["liquidity_score_0_100"] - 50) * 0.4
    vol_confirm = safe_float(np.clip(vol_confirm, 0, 100))
    
    # Relative strength score (-100 to +100)
    rs_score = 0
    if relative["excess_ret_3m"]: rs_score += relative["excess_ret_3m"] * 50
    if relative["excess_ret_6m"]: rs_score += relative["excess_ret_6m"] * 30
    if relative["rel_strength_trend"]: rs_score += relative["rel_strength_trend"] * 20
    rs_score = safe_float(np.clip(rs_score, -100, 100))
    
    # Master score (weighted blend, gated by liquidity)
    master_score = 0
    weights = [0.25, 0.25, 0.15, 0.15, 0.20]
    scores = [momentum_score or 0, trend_score or 0, risk_score or 0, vol_confirm or 0, rs_score or 0]
    master_score = sum(w * s for w, s in zip(weights, scores))
    
    # Gate by liquidity
    if volume["min_liquidity_score_filter"] is False:
        master_score *= 0.5
    
    master_score = safe_float(np.clip(master_score, -100, 100))
    
    return {
        "momentum_score": momentum_score,
        "trend_score": trend_score,
        "risk_score": risk_score,
        "volume_confirm_score": vol_confirm,
        "relative_strength_score": rs_score,
        "tier1_master_score": master_score
    }


def compute_features(
    hist: pd.DataFrame,
    benchmark_hist: Optional[pd.Series] = None,
    sector_hist: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Compute all technical features for a historical window.
    
    Args:
        hist: DataFrame with OHLCV, excluding current eval bar
        benchmark_hist: Optional benchmark close prices
        sector_hist: Optional sector close prices
    
    Returns:
        Dictionary of feature groups
    """
    momentum = compute_momentum(hist)
    trend = compute_trend(hist)
    vol_risk = compute_volatility_risk(hist)
    volume = compute_volume_liquidity(hist)
    price_events = compute_price_events(hist)
    relative = compute_relative_performance(hist, benchmark_hist, sector_hist)
    composites = compute_composites(momentum, trend, vol_risk, volume, relative)
    
    return {
        "Momentum": momentum,
        "Trend": trend,
        "Volatility_Risk": vol_risk,
        "Volume_Liquidity": volume,
        "Price_Events": price_events,
        "Relative_Performance": relative,
        "Composites": composites
    }


def build_weekly_payloads(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    benchmark: Optional[pd.Series] = None,
    sector_bench: Optional[pd.Series] = None
) -> List[Dict[str, Any]]:
    """
    Generate weekly rolling technical analysis payloads.
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        meta: Asset metadata (symbol, country, currency, sector, industry)
        benchmark: Optional benchmark close series
        sector_bench: Optional sector benchmark close series
    
    Returns:
        List of JSON-serializable payload dictionaries
    """
    if len(df) < LOOKBACK_DAYS:
        logger.warning(f"Insufficient data: {len(df)} < {LOOKBACK_DAYS}")
        return []
    
    payloads = []
    
    # Iterate every 5th bar starting from LOOKBACK_DAYS
    for t in range(LOOKBACK_DAYS, len(df), WEEKLY_STRIDE):
        # Extract history excluding current bar t
        hist = df.iloc[t - LOOKBACK_DAYS:t].copy()
        
        # Align benchmark if provided
        bench_hist = None
        if benchmark is not None:
            bench_aligned = benchmark.loc[hist.index]
            if len(bench_aligned) == len(hist):
                bench_hist = bench_aligned
        
        sector_hist = None
        if sector_bench is not None:
            sector_aligned = sector_bench.loc[hist.index]
            if len(sector_aligned) == len(hist):
                sector_hist = sector_aligned
        
        # Compute features
        features = compute_features(hist, bench_hist, sector_hist)
        
        # Build payload
        asof_date = df.index[t - 1].date().isoformat()
        
        payload = {
            "metadata": {
                "symbol": meta.get("symbol", "UNKNOWN"),
                "asof": asof_date,
                "country": meta.get("country"),
                "currency": meta.get("currency")
            },
            "Momentum": features["Momentum"],
            "Trend": features["Trend"],
            "Volatility_Risk": features["Volatility_Risk"],
            "Volume_Liquidity": features["Volume_Liquidity"],
            "Price_Events": features["Price_Events"],
            "Relative_Performance": features["Relative_Performance"],
            "Composites": features["Composites"]
        }
        
        payloads.append(payload)
    
    return payloads


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    # Verify TA-Lib is working
    try:
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        _ = talib.SMA(test_data, timeperiod=5)
        logger.info("TA-Lib verification: OK")
    except Exception as e:
        logger.error(f"TA-Lib verification FAILED: {e}")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Technical JSON Pipeline - Generate rolling technical analysis from Tiingo data"
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated list of symbols (e.g., AAPL,MSFT)"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date YYYY-MM-DD"
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Cache directory for CSV storage"
    )
    parser.add_argument(
        "--out",
        default="./out",
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--benchmark",
        default="SPY",
        help="Benchmark symbol (default: SPY); use 'none' to disable"
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Disable benchmark fetching"
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # Setup output directory
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Fetch benchmark
    benchmark_df = None
    if not args.no_benchmark and args.benchmark.lower() != "none":
        try:
            logger.info(f"Fetching benchmark: {args.benchmark}")
            benchmark_df = fetch_tiingo_ohlcv(
                args.benchmark,
                args.start,
                args.end,
                cache_dir=args.cache
            )
        except Exception as e:
            logger.error(f"Failed to fetch benchmark {args.benchmark}: {e}")
    
    # Process each symbol
    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*60}")
        
        try:
            # Fetch data
            df = fetch_tiingo_ohlcv(symbol, args.start, args.end, cache_dir=args.cache)
            logger.info(f"Fetched {len(df)} rows for {symbol}")
            
            if len(df) == 0:
                logger.error(f"ERROR: No data returned for {symbol}")
                continue
            
            if len(df) < LOOKBACK_DAYS:
                logger.error(f"ERROR: Skipping {symbol}: insufficient data ({len(df)} rows < {LOOKBACK_DAYS} required)")
                logger.error(f"Date range: {df.index[0]} to {df.index[-1]}")
                continue
            
            # Prepare metadata (removed sector/industry as they're not populated)
            meta = {
                "symbol": symbol,
                "country": "US",  # Default, can be enriched later
                "currency": "USD"
            }
            
            # Align benchmark
            benchmark_series = None
            if benchmark_df is not None:
                # Align benchmark to symbol dates
                benchmark_series = benchmark_df["Close"].reindex(df.index, method='ffill')
            
            # Build payloads
            logger.info(f"Building weekly payloads for {symbol}...")
            payloads = build_weekly_payloads(df, meta, benchmark_series)
            
            logger.info(f"Payload generation complete. Generated {len(payloads)} payloads")
            
            if not payloads:
                logger.error(f"ERROR: No payloads generated for {symbol} - this should not happen with {len(df)} rows")
                continue
            
            logger.info(f"Generated {len(payloads)} weekly snapshots for {symbol}")
            
            # Write latest snapshot
            latest_file = out_path / f"{symbol}_latest.json"
            with open(latest_file, "w") as f:
                json.dump(payloads[-1], f, indent=2)
            logger.info(f"Wrote latest snapshot: {latest_file}")
            
            # Write all snapshots (JSONL)
            all_file = out_path / f"{symbol}_all.jsonl"
            with open(all_file, "w") as f:
                for payload in payloads:
                    f.write(json.dumps(payload) + "\n")
            logger.info(f"Wrote all {len(payloads)} snapshots: {all_file}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            continue
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
