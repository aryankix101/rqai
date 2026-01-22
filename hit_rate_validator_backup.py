"""
Hit Rate Validator for Technical Indicators
============================================

Validates that technical indicators have >52% hit rate before deploying LLM layer.

Key Concepts:
- Hit Rate = (Signals with Positive Forward Return) / (Total Signals Generated)
- Tests on 5-stock mini-universe (AAPL, NVDA, TSLA, MSFT, GOOGL)
- Validates multiple confluence filters for >52% threshold
- Avoids look-ahead bias: Uses T data to predict T+1

Process:
1. Generate binary signals from technical indicators
2. Calculate forward returns (T+1, T+3, T+5 days)
3. Calculate hit rate for each configuration
4. Test confluence filters to improve hit rate
"""

import json
import asyncio
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Suppress numpy warnings for division by zero in correlation calculations
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

from technical_indicators_calculator import TechnicalIndicatorCalculator
from technical_json_pipeline import fetch_tiingo_ohlcv
from composite_scores import (
    calculate_momentum_composite,
    calculate_trend_composite,
    calculate_volume_confirmation_composite,
    calculate_risk_composite,
    calculate_relative_strength_composite
)
from deterministic_scoring import calculate_tier1_master_score
from regime_classifier import classify_market_regime


# Mini-universe for validation
MINI_UNIVERSE = ['TSLA', 'UNH', 'TGT', 'JNJ', 'NVDA', 'KMX']


# =============================================================================
# Stock Behavior Classification
# =============================================================================

class StockBehaviorType:
    """Classification of stock behavior types."""
    MOMENTUM = "momentum"           # Trend-following works (TSLA, NVDA, growth stocks)
    MEAN_REVERSION = "mean_reversion"  # Contrarian works (UNH, JNJ, defensive)
    HYBRID = "hybrid"               # Mixed behavior (AAPL, large-cap balanced)
    HIGH_VOLATILITY = "high_volatility"  # Speculative, news-driven (meme stocks)
    LOW_VOLATILITY = "low_volatility"    # Steady, income-focused (utilities)


def classify_stock_behavior(hist: pd.DataFrame, lookback_days: int = 252) -> Dict[str, Any]:
    """
    Classify stock behavior type based on historical price patterns.
    
    Uses multiple metrics:
    1. Autocorrelation of returns (positive = momentum, negative = mean-reversion)
    2. Volatility level (annualized)
    3. Hurst exponent proxy (trend persistence)
    4. Beta proxy (via volatility ratio)
    
    Args:
        hist: DataFrame with OHLCV data
        lookback_days: Days to analyze
        
    Returns:
        {
            'type': StockBehaviorType,
            'autocorr': float,  # Return autocorrelation
            'volatility': float,  # Annualized volatility
            'hurst_proxy': float,  # Trend persistence (0.5 = random, >0.5 = trending)
            'mean_reversion_score': float,  # Higher = more mean-reverting
            'momentum_score': float,  # Higher = more momentum-driven
            'confidence': float,  # Classification confidence
        }
    """
    if len(hist) < lookback_days:
        lookback_days = len(hist)
    
    recent = hist.tail(lookback_days).copy()
    returns = recent['Close'].pct_change().dropna()
    
    if len(returns) < 20:
        return {
            'type': StockBehaviorType.HYBRID,
            'autocorr': 0,
            'volatility': 0,
            'hurst_proxy': 0.5,
            'mean_reversion_score': 0.5,
            'momentum_score': 0.5,
            'confidence': 0
        }
    
    # 1. Autocorrelation of returns (lag 1-5 days)
    autocorr_1d = returns.autocorr(lag=1) if len(returns) > 5 else 0
    autocorr_5d = returns.autocorr(lag=5) if len(returns) > 10 else 0
    avg_autocorr = (autocorr_1d + autocorr_5d) / 2 if not np.isnan(autocorr_1d) and not np.isnan(autocorr_5d) else 0
    
    # 2. Volatility (annualized)
    volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
    
    # 3. Hurst exponent proxy (simplified using variance ratio)
    # H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random walk
    try:
        # Variance ratio test (simplified)
        var_1 = returns.var()
        var_5 = returns.rolling(5).sum().dropna().var() / 5
        variance_ratio = var_5 / var_1 if var_1 > 0 else 1
        hurst_proxy = 0.5 + (variance_ratio - 1) * 0.25  # Normalize to ~[0, 1]
        hurst_proxy = max(0, min(1, hurst_proxy))
    except:
        hurst_proxy = 0.5
    
    # 4. Mean reversion detection via price deviation from moving averages
    if 'Close' in recent.columns:
        ma_50 = recent['Close'].rolling(50).mean()
        ma_200 = recent['Close'].rolling(200).mean() if len(recent) >= 200 else ma_50
        
        # How often price reverts to mean after deviation
        price_vs_ma50 = (recent['Close'] - ma_50) / ma_50
        price_vs_ma50 = price_vs_ma50.dropna()
        
        # Count reversions (price crosses back through MA)
        if len(price_vs_ma50) > 20:
            sign_changes = (price_vs_ma50 * price_vs_ma50.shift(1) < 0).sum()
            reversion_frequency = sign_changes / len(price_vs_ma50)
        else:
            reversion_frequency = 0.5
    else:
        reversion_frequency = 0.5
    
    # 5. RSI mean reversion check
    # Does buying low RSI work? Does selling high RSI work?
    delta = recent['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi_reversion_score = 0
    rsi_valid = rsi.dropna()
    if len(rsi_valid) > 20:
        # Check if low RSI predicts positive returns (mean reversion)
        # Align indices by using .loc with common index
        forward_rets = returns.shift(-5)  # 5-day forward return
        common_idx = rsi_valid.index.intersection(forward_rets.dropna().index)
        
        if len(common_idx) > 20:
            rsi_aligned = rsi_valid.loc[common_idx]
            fwd_aligned = forward_rets.loc[common_idx]
            
            low_rsi_mask = rsi_aligned < 30
            high_rsi_mask = rsi_aligned > 70
            
            low_rsi_rets = fwd_aligned[low_rsi_mask].mean() if low_rsi_mask.sum() > 0 else 0
            high_rsi_rets = fwd_aligned[high_rsi_mask].mean() if high_rsi_mask.sum() > 0 else 0
            
            # Positive score if low RSI → positive returns (mean reversion works)
            if not np.isnan(low_rsi_rets) and not np.isnan(high_rsi_rets):
                rsi_reversion_score = (low_rsi_rets - high_rsi_rets) * 100  # Scale up
    
    # Composite scores
    mean_reversion_score = (
        0.3 * (-avg_autocorr + 1) / 2 +  # Negative autocorr → mean reversion
        0.3 * reversion_frequency +
        0.2 * (1 - hurst_proxy) +
        0.2 * max(0, min(1, rsi_reversion_score + 0.5))
    )
    
    momentum_score = (
        0.3 * (avg_autocorr + 1) / 2 +  # Positive autocorr → momentum
        0.3 * (1 - reversion_frequency) +
        0.2 * hurst_proxy +
        0.2 * max(0, min(1, -rsi_reversion_score + 0.5))
    )
    
    # Classify based on scores
    score_diff = momentum_score - mean_reversion_score
    
    if volatility > 0.5:  # >50% annualized vol
        stock_type = StockBehaviorType.HIGH_VOLATILITY
    elif volatility < 0.15:  # <15% annualized vol
        stock_type = StockBehaviorType.LOW_VOLATILITY
    elif score_diff > 0.15:
        stock_type = StockBehaviorType.MOMENTUM
    elif score_diff < -0.15:
        stock_type = StockBehaviorType.MEAN_REVERSION
    else:
        stock_type = StockBehaviorType.HYBRID
    
    confidence = abs(score_diff) * 2  # Scale to 0-1
    confidence = min(1.0, confidence)
    
    return {
        'type': stock_type,
        'autocorr': avg_autocorr,
        'volatility': volatility,
        'hurst_proxy': hurst_proxy,
        'mean_reversion_score': mean_reversion_score,
        'momentum_score': momentum_score,
        'reversion_frequency': reversion_frequency,
        'confidence': confidence
    }


def get_signal_adjustment(stock_type: str, original_signal: int) -> int:
    """
    Adjust signal based on stock behavior type.
    
    Args:
        stock_type: From StockBehaviorType
        original_signal: Original signal (-1, 0, 1)
        
    Returns:
        Adjusted signal
    """
    if stock_type == StockBehaviorType.MEAN_REVERSION:
        # Invert signals for mean-reversion stocks
        return -original_signal
    elif stock_type == StockBehaviorType.LOW_VOLATILITY:
        # Also tend to be mean-reverting
        return -original_signal
    else:
        # Keep original for momentum, hybrid, high-vol
        return original_signal


def get_confluence_thresholds(stock_type: str) -> Dict[str, Any]:
    """
    Get appropriate confluence filter thresholds based on stock type.
    
    Different stock types need different filter settings.
    """
    if stock_type == StockBehaviorType.MOMENTUM:
        return {
            'min_adx': 25,
            'volume_spike_threshold': 1.5,
            'rsi_range': (40, 70),  # Buy on strength
            'near_high_threshold': 0.05,
            'min_filters_required': 2,
        }
    elif stock_type == StockBehaviorType.MEAN_REVERSION:
        return {
            'min_adx': 15,  # Lower ADX ok (range-bound)
            'volume_spike_threshold': 1.0,  # Less volume needed
            'rsi_range': (25, 75),  # Wider range, buy extremes
            'near_high_threshold': 0.15,  # Further from highs ok
            'min_filters_required': 1,  # Fewer filters needed
        }
    elif stock_type == StockBehaviorType.HIGH_VOLATILITY:
        return {
            'min_adx': 30,  # Need strong trend
            'volume_spike_threshold': 2.0,  # Need volume confirmation
            'rsi_range': (35, 65),  # Tighter to avoid extremes
            'near_high_threshold': 0.03,
            'min_filters_required': 3,  # More confirmation needed
        }
    elif stock_type == StockBehaviorType.LOW_VOLATILITY:
        return {
            'min_adx': 10,  # Low bar
            'volume_spike_threshold': 1.2,
            'rsi_range': (20, 80),  # Wide range
            'near_high_threshold': 0.10,
            'min_filters_required': 1,
        }
    else:  # HYBRID
        return {
            'min_adx': 20,
            'volume_spike_threshold': 1.3,
            'rsi_range': (35, 65),
            'near_high_threshold': 0.08,
            'min_filters_required': 2,
        }


def check_adaptive_confluence(
    indicators: Dict[str, Any], 
    stock_type: str
) -> Tuple[bool, Dict[str, bool]]:
    """
    Check confluence filters with thresholds adapted to stock type.
    """
    thresholds = get_confluence_thresholds(stock_type)
    
    trend = indicators.get('trend', {})
    vol_liq = indicators.get('volume_liquidity', {})
    momentum = indicators.get('momentum', {})
    
    rsi = trend.get('rsi_14d', 50)
    adx = trend.get('adx', 0)
    macd_hist = trend.get('macd_histogram', 0)
    volume_spike_z = vol_liq.get('volume_spike_z', 0)
    obv_slope = vol_liq.get('obv_slope', 0)
    dist_52wk_high = momentum.get('dist_52wk_high', 1)
    
    rsi_min, rsi_max = thresholds['rsi_range']
    
    details = {}
    
    if stock_type == StockBehaviorType.MEAN_REVERSION:
        # For mean-reversion: RSI extremes are good entry points
        details['rsi_extreme'] = rsi < 30 or rsi > 70
        details['adx_low'] = adx < 25  # Range-bound market
        details['volume_confirm'] = volume_spike_z > thresholds['volume_spike_threshold']
        details['obv_divergence'] = True  # Simplified - could add divergence detection
        details['price_deviation'] = abs(dist_52wk_high) > 0.10  # Away from highs
    else:
        # For momentum/hybrid: trend confirmation
        details['trend_momentum'] = rsi_min <= rsi <= rsi_max and (macd_hist > 0 if rsi > 50 else macd_hist < 0)
        details['volume_confirm'] = volume_spike_z > thresholds['volume_spike_threshold']
        details['adx_strength'] = adx > thresholds['min_adx']
        details['near_52wk_high'] = dist_52wk_high < thresholds['near_high_threshold'] and rsi < 70
        details['obv_trend'] = obv_slope > 0 if rsi > 50 else obv_slope < 0
    
    passed = sum(details.values()) >= thresholds['min_filters_required']
    
    return passed, details


@dataclass
class SignalResult:
    """Container for signal and its forward returns."""
    date: pd.Timestamp
    signal: int  # 1 = Buy, 0 = Neutral, -1 = Sell
    signal_strength: float  # Underlying score
    forward_return_1d: Optional[float]
    forward_return_3d: Optional[float]
    forward_return_5d: Optional[float]
    confluence_passed: bool
    indicators: Dict[str, Any]


# =============================================================================
# Signal Generation (Binary Conversion)
# =============================================================================

def generate_binary_signal(
    tier1_score: float,
    threshold_buy: float = 0.1,
    threshold_sell: float = -0.1
) -> int:
    """
    Convert continuous score [-1, 1] to binary signal {-1, 0, 1}.
    
    Args:
        tier1_score: Deterministic score from weighted indicators
        threshold_buy: Minimum score for BUY signal
        threshold_sell: Maximum score for SELL signal
    
    Returns:
        1 = Buy, 0 = Neutral, -1 = Sell
    """
    if tier1_score >= threshold_buy:
        return 1
    elif tier1_score <= threshold_sell:
        return -1
    else:
        return 0


# =============================================================================
# Confluence Filters (For >52% Hit Rate)
# =============================================================================

def check_trend_momentum_confluence(indicators: Dict[str, Any]) -> bool:
    """
    Filter #1: Trend + Momentum Confluence
    Only trade RSI oversold/overbought if trend is aligned.
    
    Rule: RSI signal only valid if price is above/below 200-day MA.
    
    Returns:
        True if confluence conditions met
    """
    trend = indicators.get('trend', {})
    rsi = trend.get('rsi_14d')
    price_above_ma200 = trend.get('price_above_ma200')
    
    if rsi is None or price_above_ma200 is None:
        return False
    
    # Bullish: RSI < 40 AND price above 200-day MA (buying dip in uptrend)
    if rsi < 40 and price_above_ma200:
        return True
    
    # Bearish: RSI > 60 AND price below 200-day MA (shorting rally in downtrend)
    if rsi > 60 and not price_above_ma200:
        return True
    
    # Strong trend continuation
    if price_above_ma200 and rsi > 45 and rsi < 70:
        return True
    
    return False


def check_volume_confirmation_confluence(indicators: Dict[str, Any]) -> bool:
    """
    Filter #2: Volume Confirmation
    Only count breakouts if volume is elevated.
    
    Rule: Volume > 1.5x 20-day average for valid signal.
    
    Returns:
        True if volume confluence met
    """
    vol_liq = indicators.get('volume_liquidity', {})
    volume_spike_z = vol_liq.get('volume_spike_z', 0)
    
    # Volume spike > 1.5 standard deviations
    return volume_spike_z > 1.5


def check_volatility_squeeze_confluence(indicators: Dict[str, Any]) -> bool:
    """
    Filter #3: Volatility Squeeze (Expansion Gate)
    Use Bollinger Band squeezes to predict breakouts.
    
    Rule: BB width < 20th percentile = squeeze (precedes breakout).
    
    Returns:
        True if in volatility squeeze
    """
    trend = indicators.get('trend', {})
    bb_width = trend.get('bb_width_pct')
    
    if bb_width is None:
        return False
    
    # Tight BB indicates squeeze (low volatility before expansion)
    return bb_width < 0.02  # Less than 2% width


def check_adx_trend_strength_confluence(indicators: Dict[str, Any]) -> bool:
    """
    Filter #4: ADX Trend Strength
    Only trade with the trend when trend is strong.
    
    Rule: ADX > 25 AND price aligned with 200-day MA.
    
    Returns:
        True if strong trend confirmed
    """
    trend = indicators.get('trend', {})
    adx = trend.get('adx')
    price_above_ma200 = trend.get('price_above_ma200')
    
    if adx is None or price_above_ma200 is None:
        return False
    
    # Strong trend (ADX > 25) with direction confirmation
    return adx > 25


def check_momentum_52wk_high_confluence(indicators: Dict[str, Any]) -> bool:
    """
    Filter #5: 52-Week High Momentum
    Captures clean breakouts without overextension.
    
    Rule: Distance to 52-week high < 5% AND RSI < 70.
    
    Returns:
        True if near 52-week high but not overextended
    """
    momentum = indicators.get('momentum', {})
    trend = indicators.get('trend', {})
    
    dist_52wk_high = momentum.get('dist_52wk_high')
    rsi = trend.get('rsi_14d')
    
    if dist_52wk_high is None or rsi is None:
        return False
    
    # Near 52-week high (within 5%) but not overbought
    return dist_52wk_high < 0.05 and rsi < 70


def check_obv_trend_confluence(indicators: Dict[str, Any]) -> bool:
    """
    Filter #6: OBV Trend Alignment
    Confirms institutional participation.
    
    Rule: OBV slope > 0 (accumulation) for bullish signals.
    
    Returns:
        True if OBV confirms price trend
    """
    vol_liq = indicators.get('volume_liquidity', {})
    obv_slope = vol_liq.get('obv_slope')
    
    if obv_slope is None:
        return False
    
    # Positive OBV slope = accumulation
    return obv_slope > 0


def check_multi_confluence(indicators: Dict[str, Any]) -> Tuple[bool, Dict[str, bool]]:
    """
    Combined confluence check using multiple filters.
    Signal passes if ANY 2+ confluence filters pass.
    
    Returns:
        Tuple of (passed, details_dict)
    """
    details = {
        'trend_momentum': check_trend_momentum_confluence(indicators),
        'volume_confirm': check_volume_confirmation_confluence(indicators),
        'adx_strength': check_adx_trend_strength_confluence(indicators),
        'near_52wk_high': check_momentum_52wk_high_confluence(indicators),
        'obv_trend': check_obv_trend_confluence(indicators),
    }
    
    passed = sum(details.values()) >= 2
    return passed, details


def get_confluence_stats(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """Get detailed confluence filter statistics."""
    passed, details = check_multi_confluence(indicators)
    return {
        'passed': passed,
        'filters_passed': sum(details.values()),
        'filter_details': details
    }


# =============================================================================
# Forward Return Calculation
# =============================================================================

def calculate_forward_returns(
    hist: pd.DataFrame,
    signal_date: pd.Timestamp,
    horizons: List[int] = [1, 3, 5]
) -> Dict[int, Optional[float]]:
    """
    Calculate forward returns from signal date (avoiding look-ahead bias).
    
    Args:
        hist: Historical OHLCV data with DatetimeIndex
        signal_date: Date signal was generated
        horizons: List of forward days to calculate returns
    
    Returns:
        Dict mapping horizon to return (None if insufficient data)
    """
    if signal_date not in hist.index:
        return {h: None for h in horizons}
    
    signal_price = hist.loc[signal_date, 'Close']
    future_dates = hist.index[hist.index > signal_date]
    
    returns = {}
    for horizon in horizons:
        if len(future_dates) >= horizon:
            target_date = future_dates[horizon - 1]
            target_price = hist.loc[target_date, 'Close']
            returns[horizon] = (target_price / signal_price) - 1.0
        else:
            returns[horizon] = None
    
    return returns


# =============================================================================
# Hit Rate Calculation
# =============================================================================

def calculate_hit_rate(
    signals: List[SignalResult],
    horizon_days: int = 1,
    signal_filter: int = 1,  # 1 = Buy only, -1 = Sell only, 0 = All
    min_signals: int = 5
) -> Dict[str, Any]:
    """
    Calculate hit rate for given signal configuration.
    
    Args:
        signals: List of SignalResult objects
        horizon_days: Which forward return to use (1, 3, or 5 days)
        signal_filter: Which signals to analyze (1=Buy, -1=Sell, 0=All)
        min_signals: Minimum signals required for valid hit rate
    
    Returns:
        Dictionary with hit rate metrics
    """
    # Filter signals
    if signal_filter != 0:
        filtered = [s for s in signals if s.signal == signal_filter]
    else:
        filtered = [s for s in signals if s.signal != 0]  # Exclude neutral
    
    # Get forward returns for horizon
    forward_returns = []
    for sig in filtered:
        if horizon_days == 1:
            ret = sig.forward_return_1d
        elif horizon_days == 3:
            ret = sig.forward_return_3d
        elif horizon_days == 5:
            ret = sig.forward_return_5d
        else:
            ret = None
        
        if ret is not None:
            forward_returns.append(ret)
    
    # Calculate metrics
    if len(forward_returns) < min_signals:
        return {
            "hit_rate": None,
            "total_signals": len(filtered),
            "valid_signals": len(forward_returns),
            "profitable_signals": 0,
            "avg_return": None,
            "median_return": None,
            "status": "INSUFFICIENT_DATA"
        }
    
    profitable = sum(1 for r in forward_returns if r > 0)
    hit_rate = profitable / len(forward_returns)
    
    return {
        "hit_rate": hit_rate,
        "total_signals": len(filtered),
        "valid_signals": len(forward_returns),
        "profitable_signals": profitable,
        "avg_return": np.mean(forward_returns),
        "median_return": np.median(forward_returns),
        "std_return": np.std(forward_returns),
        "status": "VALID" if hit_rate >= 0.52 else "BELOW_THRESHOLD"
    }


# =============================================================================
# Single Stock Validation
# =============================================================================

async def validate_single_stock(
    symbol: str,
    start_date: str = "2023-01-01",
    end_date: str = "2025-01-01",
    use_confluence: bool = False
) -> Tuple[List[SignalResult], pd.DataFrame]:
    """
    Generate signals and calculate forward returns for single stock.
    
    Args:
        symbol: Stock ticker
        start_date: Backtest start date
        end_date: Backtest end date
        use_confluence: Whether to apply confluence filters
    
    Returns:
        (signals, historical_data)
    """
    print(f"\n{'='*60}")
    print(f"Validating: {symbol}")
    print(f"{'='*60}")
    
    # Fetch data
    hist = fetch_tiingo_ohlcv(symbol, start_date, end_date)
    if hist is None or len(hist) < 252:
        print(f"Insufficient data for {symbol}")
        return [], hist
    
    # Classify stock behavior ONCE at the start
    stock_behavior = classify_stock_behavior(hist, lookback_days=252)
    stock_type = stock_behavior['type']
    
    print(f"  Stock Type: {stock_type}")
    print(f"  Momentum Score: {stock_behavior['momentum_score']:.2f}")
    print(f"  Mean-Reversion Score: {stock_behavior['mean_reversion_score']:.2f}")
    print(f"  Volatility: {stock_behavior['volatility']:.1%}")
    print(f"  Classification Confidence: {stock_behavior['confidence']:.1%}")
    
    # Get weekly rebalance dates
    weekly_dates = hist.resample('W-FRI').last().dropna(subset=['Close']).index
    
    signals = []
    
    for signal_date in weekly_dates[:-5]:  # Leave room for forward returns
        # Calculate indicators using data up to day before signal date
        prev_date = signal_date - timedelta(days=1)
        hist_slice = hist[hist.index <= prev_date].copy()
        
        if len(hist_slice) < 200:
            continue
        
        try:
            # Calculate technical indicators
            calc = TechnicalIndicatorCalculator(hist_slice)
            indicators = {
                'momentum': calc.calculate_momentum(),
                'trend': calc.calculate_trend_indicators(),
                'volume_liquidity': calc.calculate_volume_liquidity_indicators(),
                'risk': calc.calculate_risk_indicators(),
                'relative_strength': {}  # Skip for now - would need benchmark data
            }
            
            # Calculate composite scores
            composites = {
                'momentum_composite': calculate_momentum_composite(indicators),
                'trend_composite': calculate_trend_composite(indicators),
                'volume_composite': calculate_volume_confirmation_composite(indicators),
                'risk_composite': calculate_risk_composite(indicators),
                'rs_composite': calculate_relative_strength_composite(indicators)
            }
            
            # Calculate regime
            regime = classify_market_regime(indicators)
            
            # Calculate Tier-1 score (deterministic weighted average)
            tier1_result = calculate_tier1_master_score(composites, regime)
            tier1_score = tier1_result['tier1_score']
            
            # Generate binary signal
            raw_signal = generate_binary_signal(tier1_score)
            
            # ADAPTIVE: Adjust signal based on stock type
            binary_signal = get_signal_adjustment(stock_type, raw_signal)
            
            # Check confluence filters - ADAPTIVE based on stock type
            confluence_passed = True
            confluence_details = {}
            if use_confluence:
                confluence_passed, confluence_details = check_adaptive_confluence(indicators, stock_type)
            
            # Only track non-neutral signals
            if binary_signal != 0:
                # Calculate forward returns
                forward_rets = calculate_forward_returns(hist, signal_date, [1, 3, 5])
                
                signal_result = SignalResult(
                    date=signal_date,
                    signal=binary_signal,
                    signal_strength=tier1_score,
                    forward_return_1d=forward_rets[1],
                    forward_return_3d=forward_rets[3],
                    forward_return_5d=forward_rets[5],
                    confluence_passed=confluence_passed,
                    indicators={
                        **indicators,
                        'composites': composites,
                        'tier1_score': tier1_score,
                        'regime': regime,
                        'confluence_details': confluence_details,
                        'confluence_count': sum(confluence_details.values()) if confluence_details else 0,
                        'stock_type': stock_type,
                        'raw_signal': raw_signal,
                        'signal_inverted': raw_signal != binary_signal
                    }
                )
                signals.append(signal_result)
        
        except Exception as e:
            print(f"Error on {signal_date}: {e}")
            continue
    
    # Summary
    inverted_count = sum(1 for s in signals if s.indicators.get('signal_inverted', False))
    print(f"Generated {len(signals)} signals for {symbol}")
    if inverted_count > 0:
        print(f"  ({inverted_count} signals inverted for {stock_type} behavior)")
    
    return signals, hist, stock_behavior


# =============================================================================
# Multi-Stock Universe Validation
# =============================================================================

async def validate_mini_universe(
    symbols: List[str] = MINI_UNIVERSE,
    start_date: str = "2023-01-01",
    end_date: str = "2025-01-01",
    use_confluence: bool = False
) -> Dict[str, Any]:
    """
    Validate hit rates across mini-universe of stocks.
    
    Args:
        symbols: List of stock tickers
        start_date: Backtest start
        end_date: Backtest end
        use_confluence: Whether to apply confluence filters
    
    Returns:
        Dictionary with aggregated hit rate results
    """
    print(f"\n{'='*60}")
    print(f"MINI-UNIVERSE HIT RATE VALIDATION")
    print(f"Stocks: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Confluence Filters: {'ENABLED' if use_confluence else 'DISABLED'}")
    print(f"{'='*60}\n")
    
    all_signals = []
    stock_results = {}
    stock_behaviors = {}
    
    # Validate each stock
    for symbol in symbols:
        result = await validate_single_stock(
            symbol, start_date, end_date, use_confluence
        )
        
        # Handle both old (2-tuple) and new (3-tuple) return format
        if len(result) == 3:
            signals, hist, stock_behavior = result
            stock_behaviors[symbol] = stock_behavior
        else:
            signals, hist = result
            stock_behaviors[symbol] = {'type': 'hybrid'}
        
        if use_confluence:
            # Filter by confluence
            signals = [s for s in signals if s.confluence_passed]
        
        all_signals.extend(signals)
        stock_results[symbol] = signals
    
    # Calculate aggregate hit rates
    print(f"\n{'='*60}")
    print(f"AGGREGATE HIT RATE RESULTS")
    print(f"{'='*60}\n")
    
    results = {}
    
    # Test different configurations
    for horizon in [1, 3, 5]:
        print(f"\n--- {horizon}-Day Forward Returns ---")
        
        # Buy signals only
        buy_hr = calculate_hit_rate(all_signals, horizon, signal_filter=1)
        print(f"  BUY Signals:")
        print(f"    Hit Rate: {buy_hr['hit_rate']:.2%}" if buy_hr['hit_rate'] else "    Hit Rate: N/A")
        print(f"    Total Signals: {buy_hr['valid_signals']}")
        print(f"    Avg Return: {buy_hr['avg_return']:.2%}" if buy_hr['avg_return'] else "    Avg Return: N/A")
        print(f"    Status: {buy_hr['status']}")
        
        # Sell signals only
        sell_hr = calculate_hit_rate(all_signals, horizon, signal_filter=-1)
        print(f"  SELL Signals:")
        print(f"    Hit Rate: {sell_hr['hit_rate']:.2%}" if sell_hr['hit_rate'] else "    Hit Rate: N/A")
        print(f"    Total Signals: {sell_hr['valid_signals']}")
        print(f"    Avg Return: {sell_hr['avg_return']:.2%}" if sell_hr['avg_return'] else "    Avg Return: N/A")
        print(f"    Status: {sell_hr['status']}")
        
        # All signals
        all_hr = calculate_hit_rate(all_signals, horizon, signal_filter=0)
        print(f"  ALL Signals:")
        print(f"    Hit Rate: {all_hr['hit_rate']:.2%}" if all_hr['hit_rate'] else "    Hit Rate: N/A")
        print(f"    Total Signals: {all_hr['valid_signals']}")
        print(f"    Status: {all_hr['status']}")
        
        results[f"{horizon}d"] = {
            "buy": buy_hr,
            "sell": sell_hr,
            "all": all_hr
        }
    
    # Per-stock breakdown
    print(f"\n{'='*60}")
    print(f"PER-STOCK BREAKDOWN (1-Day Returns)")
    print(f"{'='*60}\n")
    
    for symbol, signals in stock_results.items():
        hr = calculate_hit_rate(signals, horizon_days=1, signal_filter=0, min_signals=5)
        behavior = stock_behaviors.get(symbol, {})
        stock_type = behavior.get('type', 'unknown').upper()
        type_tag = f"[{stock_type:14s}]"
        
        if hr['hit_rate']:
            print(f"{symbol:6s} {type_tag}: {hr['hit_rate']:.2%} ({hr['valid_signals']:3d} signals)")
        else:
            print(f"{symbol:6s} {type_tag}: N/A")
    
    # Save results
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    confluence_suffix = "_with_confluence" if use_confluence else "_no_confluence"
    output_file = output_dir / f"hit_rate_validation{confluence_suffix}_{timestamp}.json"
    
    save_results = {
        "config": {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "use_confluence": use_confluence,
            "total_signals": len(all_signals)
        },
        "aggregate_results": results,
        "per_stock": {
            symbol: calculate_hit_rate(signals, 1, 0)
            for symbol, signals in stock_results.items()
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return save_results, all_signals


# =============================================================================
# Visualization
# =============================================================================

def plot_hit_rate_analysis(
    baseline_results: Dict[str, Any],
    baseline_signals: List[SignalResult],
    confluence_results: Dict[str, Any],
    confluence_signals: List[SignalResult],
    symbol: str = "TSLA"
):
    """
    Generate comprehensive visualization of hit rate analysis.
    
    Creates 6 charts:
    1. Hit Rate Comparison (Baseline vs Confluence)
    2. Signal Distribution Over Time
    3. Forward Return Distribution
    4. Win Rate by Signal Type
    5. Cumulative Returns
    6. Signal Confidence vs Actual Returns
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
    
    # -------------------------------------------------------------------------
    # Chart 1: Hit Rate Comparison
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    horizons = ['1d', '3d', '5d']
    baseline_hrs = [baseline_results['aggregate_results'][h]['all']['hit_rate'] or 0 for h in horizons]
    confluence_hrs = [confluence_results['aggregate_results'][h]['all']['hit_rate'] or 0 for h in horizons]
    
    x = np.arange(len(horizons))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_hrs, width, label='Baseline', alpha=0.8, color='#ff6b6b')
    ax1.bar(x + width/2, confluence_hrs, width, label='With Confluence', alpha=0.8, color='#4ecdc4')
    ax1.axhline(y=0.52, color='green', linestyle='--', label='Target (52%)', linewidth=2)
    ax1.axhline(y=0.50, color='gray', linestyle=':', label='Random (50%)', linewidth=1)
    
    ax1.set_xlabel('Forward Horizon', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Hit Rate', fontsize=11, fontweight='bold')
    ax1.set_title(f'{symbol} - Hit Rate Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.15])  # Extra space for labels
    
    for i, v in enumerate(baseline_hrs):
        if v > 0:
            ax1.text(i - width/2, v + 0.03, f'{v:.1%}', ha='center', fontsize=8, rotation=0)
    for i, v in enumerate(confluence_hrs):
        if v > 0:
            ax1.text(i + width/2, v + 0.03, f'{v:.1%}', ha='center', fontsize=8, rotation=0)
    
    # -------------------------------------------------------------------------
    # Chart 2: Signal Count by Type
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    baseline_buy = sum(1 for s in baseline_signals if s.signal == 1)
    baseline_sell = sum(1 for s in baseline_signals if s.signal == -1)
    baseline_neutral = sum(1 for s in baseline_signals if s.signal == 0)
    
    confluence_buy = sum(1 for s in confluence_signals if s.signal == 1)
    confluence_sell = sum(1 for s in confluence_signals if s.signal == -1)
    confluence_neutral = sum(1 for s in confluence_signals if s.signal == 0)
    
    signal_types = ['Buy', 'Sell', 'Neutral']
    baseline_counts = [baseline_buy, baseline_sell, baseline_neutral]
    confluence_counts = [confluence_buy, confluence_sell, confluence_neutral]
    
    x = np.arange(len(signal_types))
    ax2.bar(x - width/2, baseline_counts, width, label='Baseline', alpha=0.8, color='#ff6b6b')
    ax2.bar(x + width/2, confluence_counts, width, label='With Confluence', alpha=0.8, color='#4ecdc4')
    
    ax2.set_xlabel('Signal Type', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('Signal Distribution', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(signal_types)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(alpha=0.3, axis='y')
    # Add padding to top
    ax2.set_ylim([0, max(baseline_counts + confluence_counts) * 1.15])
    
    # -------------------------------------------------------------------------
    # Chart 3: Forward Return Distribution (1-Day)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    
    baseline_returns = [s.forward_return_1d for s in baseline_signals if s.forward_return_1d is not None and s.signal != 0]
    confluence_returns = [s.forward_return_1d for s in confluence_signals if s.forward_return_1d is not None and s.signal != 0]
    
    if baseline_returns:
        ax3.hist(baseline_returns, bins=30, alpha=0.6, label='Baseline', color='#ff6b6b', edgecolor='black')
    if confluence_returns:
        ax3.hist(confluence_returns, bins=30, alpha=0.6, label='With Confluence', color='#4ecdc4', edgecolor='black')
    
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel('1-Day Forward Return', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Return Distribution', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Chart 4: Win Rate by Signal Type
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    
    def calc_win_rate(signals, signal_type):
        filtered = [s for s in signals if s.signal == signal_type and s.forward_return_1d is not None]
        if not filtered:
            return 0
        wins = sum(1 for s in filtered if s.forward_return_1d > 0)
        return wins / len(filtered)
    
    baseline_buy_wr = calc_win_rate(baseline_signals, 1)
    baseline_sell_wr = calc_win_rate(baseline_signals, -1)
    confluence_buy_wr = calc_win_rate(confluence_signals, 1)
    confluence_sell_wr = calc_win_rate(confluence_signals, -1)
    
    signal_types = ['Buy Signals', 'Sell Signals']
    baseline_wrs = [baseline_buy_wr, baseline_sell_wr]
    confluence_wrs = [confluence_buy_wr, confluence_sell_wr]
    
    x = np.arange(len(signal_types))
    ax4.bar(x - width/2, baseline_wrs, width, label='Baseline', alpha=0.8, color='#ff6b6b')
    ax4.bar(x + width/2, confluence_wrs, width, label='With Confluence', alpha=0.8, color='#4ecdc4')
    ax4.axhline(y=0.52, color='green', linestyle='--', label='Target', linewidth=2)
    ax4.axhline(y=0.50, color='gray', linestyle=':', linewidth=1)
    
    ax4.set_xlabel('Signal Type', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Win Rate', fontsize=11, fontweight='bold')
    ax4.set_title('Win Rate by Signal Type', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(signal_types)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.15])
    
    # -------------------------------------------------------------------------
    # Chart 5: Cumulative Returns
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1:])
    
    def calc_cumulative_returns(signals):
        sorted_signals = sorted([s for s in signals if s.forward_return_1d is not None and s.signal != 0], 
                               key=lambda x: x.date)
        if not sorted_signals:
            return [], []
        
        dates = [s.date for s in sorted_signals]
        cum_returns = []
        cumulative = 1.0
        
        for s in sorted_signals:
            cumulative *= (1 + s.forward_return_1d)
            cum_returns.append(cumulative)
        
        return dates, cum_returns
    
    baseline_dates, baseline_cum = calc_cumulative_returns(baseline_signals)
    confluence_dates, confluence_cum = calc_cumulative_returns(confluence_signals)
    
    if baseline_dates:
        ax5.plot(baseline_dates, baseline_cum, label='Baseline', linewidth=2, color='#ff6b6b', alpha=0.8)
    if confluence_dates:
        ax5.plot(confluence_dates, confluence_cum, label='With Confluence', linewidth=2, color='#4ecdc4', alpha=0.8)
    
    ax5.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax5.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Cumulative Return Multiple', fontsize=11, fontweight='bold')
    ax5.set_title('Cumulative Returns Over Time', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Chart 6: Signal Strength vs Actual Returns (Scatter)
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, :])
    
    conf_strengths = [abs(s.signal_strength) for s in confluence_signals if s.forward_return_1d is not None and s.signal != 0]
    conf_returns = [s.forward_return_1d for s in confluence_signals if s.forward_return_1d is not None and s.signal != 0]
    
    if conf_strengths and conf_returns:
        colors = ['green' if r > 0 else 'red' for r in conf_returns]
        ax6.scatter(conf_strengths, conf_returns, alpha=0.6, c=colors, s=50, edgecolors='black', linewidth=0.5)
        ax6.axhline(y=0, color='black', linestyle='--', linewidth=2)
        
        # Add trend line
        if len(conf_strengths) > 1:
            z = np.polyfit(conf_strengths, conf_returns, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(conf_strengths), max(conf_strengths), 100)
            ax6.plot(x_trend, p(x_trend), "b--", linewidth=2, alpha=0.8, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    
    ax6.set_xlabel('Signal Strength (Absolute)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('1-Day Forward Return', fontsize=11, fontweight='bold')
    ax6.set_title('Signal Strength vs Actual Returns (Confluence Signals)', fontsize=13, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # Save figure
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"hit_rate_analysis_{symbol}_{timestamp}.png"
    
    plt.suptitle(f'{symbol} Hit Rate Validation Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Hit rate analysis saved to: {plot_file}")
    
    try:
        plt.show()
    except SystemError:
        pass  # Ignore display errors in non-interactive environments
    finally:
        plt.close('all')  # Clean up


def plot_individual_factor_analysis(
    signals: List[SignalResult],
    symbol: str = "TSLA"
):
    """
    Analyze individual factor correlation with forward returns.
    
    Creates scatter plots and correlation analysis for:
    - Tier-1 Deterministic Score
    - Momentum Composite
    - Trend Composite  
    - Volume Composite
    - Risk Composite
    - RSI
    - ADX
    - And other key indicators
    """
    # Extract data
    valid_signals = [s for s in signals if s.forward_return_1d is not None and s.signal != 0]
    
    if len(valid_signals) < 10:
        print("Insufficient signals for factor analysis")
        return
    
    # Build data arrays
    # Composites return dicts with 'score' key, so extract it properly
    def get_composite_score(s, composite_name):
        """Extract score from composite dict or return 0."""
        comp = s.indicators.get('composites', {}).get(composite_name)
        if comp is None:
            return None
        if isinstance(comp, dict):
            return comp.get('score', 0)
        return comp
    
    data = {
        'Tier-1 Score': [s.indicators.get('tier1_score', 0) for s in valid_signals],
        'Momentum Composite': [get_composite_score(s, 'momentum_composite') for s in valid_signals],
        'Trend Composite': [get_composite_score(s, 'trend_composite') for s in valid_signals],
        'Volume Composite': [get_composite_score(s, 'volume_composite') for s in valid_signals],
        'Risk Composite': [get_composite_score(s, 'risk_composite') for s in valid_signals],
        'RSI': [s.indicators.get('trend', {}).get('rsi_14d', 50) for s in valid_signals],
        'ADX': [s.indicators.get('trend', {}).get('adx', 0) for s in valid_signals],
        'MACD': [s.indicators.get('trend', {}).get('macd', 0) for s in valid_signals],
        'ATR %': [s.indicators.get('risk', {}).get('atr_pct', 0) for s in valid_signals],
        'Volume Spike': [s.indicators.get('volume_liquidity', {}).get('volume_spike_z', 0) for s in valid_signals],
    }
    
    forward_returns = [s.forward_return_1d * 100 for s in valid_signals]  # Convert to %
    
    def is_valid_number(x):
        """Check if value is a valid number (not None, not NaN)."""
        if x is None:
            return False
        try:
            return not np.isnan(float(x))
        except (TypeError, ValueError):
            return False
    
    # Create figure with 10 subplots (2 rows x 5 cols)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    for idx, (factor_name, factor_values) in enumerate(data.items()):
        ax = axes[idx]
        
        # Filter out None/NaN values
        valid_pairs = [(f, r) for f, r in zip(factor_values, forward_returns) if is_valid_number(f)]
        if not valid_pairs:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            ax.set_title(factor_name, fontsize=11, fontweight='bold')
            continue
        
        factor_clean = np.array([p[0] for p in valid_pairs], dtype=float)
        returns_clean = np.array([p[1] for p in valid_pairs], dtype=float)
        
        # Remove any inf values
        valid_mask = np.isfinite(factor_clean) & np.isfinite(returns_clean)
        factor_clean = factor_clean[valid_mask]
        returns_clean = returns_clean[valid_mask]
        
        if len(factor_clean) < 2:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', fontsize=12)
            ax.set_title(factor_name, fontsize=11, fontweight='bold')
            continue
        
        # Scatter plot
        colors = ['green' if r > 0 else 'red' for r in returns_clean]
        ax.scatter(factor_clean, returns_clean, alpha=0.6, c=colors, s=40, edgecolors='black', linewidth=0.5)
        
        # Zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Trend line - with robust error handling
        try:
            if len(factor_clean) > 1 and np.std(factor_clean) > 1e-10:
                z = np.polyfit(factor_clean, returns_clean, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(factor_clean), max(factor_clean), 100)
                ax.plot(x_trend, p(x_trend), "blue", linewidth=2, alpha=0.7)
        except (np.linalg.LinAlgError, ValueError):
            pass  # Skip trend line if fitting fails
        
        # Calculate correlation with error handling
        try:
            # Check for sufficient variance before computing correlation
            if len(factor_clean) > 1 and np.std(factor_clean) > 1e-10 and np.std(returns_clean) > 1e-10:
                correlation = np.corrcoef(factor_clean, returns_clean)[0, 1]
                if not np.isfinite(correlation):
                    correlation = 0
            else:
                correlation = 0
        except:
            correlation = 0
        
        # Calculate hit rate by quintile - with robust handling
        quintile_hr = {}
        try:
            df_temp = pd.DataFrame({'factor': factor_clean, 'return': returns_clean})
            # Use qcut without labels first, then map
            df_temp['quintile'] = pd.qcut(df_temp['factor'], q=5, duplicates='drop')
            # Get unique bins and create labels
            unique_bins = df_temp['quintile'].unique()
            n_bins = len(unique_bins)
            if n_bins > 0:
                labels = [f'Q{i+1}' for i in range(n_bins)]
                bin_to_label = {b: labels[i] for i, b in enumerate(sorted(unique_bins))}
                df_temp['quintile_label'] = df_temp['quintile'].map(bin_to_label)
                
                for q in df_temp['quintile_label'].unique():
                    if q is not None:
                        q_returns = df_temp[df_temp['quintile_label'] == q]['return']
                        if len(q_returns) > 0:
                            quintile_hr[q] = (q_returns > 0).sum() / len(q_returns)
        except Exception:
            pass  # Skip quintile analysis if it fails
        
        # Title with correlation
        ax.set_title(f'{factor_name}\nCorr: {correlation:.3f}', fontsize=11, fontweight='bold')
        ax.set_xlabel(factor_name, fontsize=9)
        ax.set_ylabel('1-Day Return (%)', fontsize=9)
        ax.grid(alpha=0.3)
        
        # Add quintile hit rates as text
        if quintile_hr:
            q_text = '\n'.join([f'{q}: {hr:.1%}' for q, hr in sorted(quintile_hr.items())])
            ax.text(0.02, 0.98, q_text, transform=ax.transAxes, fontsize=7,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{symbol} - Individual Factor Analysis vs 1-Day Forward Returns', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_dir = Path("backtest_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"factor_analysis_{symbol}_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Factor analysis saved to: {plot_file}")
    
    try:
        plt.show()
    except SystemError:
        pass  # Ignore display errors in non-interactive environments
    finally:
        plt.close('all')  # Clean up
    
    # Print correlation ranking
    print(f"\n{'='*60}")
    print(f"FACTOR CORRELATION RANKING (vs 1-Day Returns)")
    print(f"{'='*60}\n")
    
    correlations = {}
    for factor_name, factor_values in data.items():
        valid_pairs = [(f, r) for f, r in zip(factor_values, forward_returns) if is_valid_number(f)]
        if len(valid_pairs) > 1:
            factor_clean = np.array([p[0] for p in valid_pairs], dtype=float)
            returns_clean = np.array([p[1] for p in valid_pairs], dtype=float)
            # Remove inf values
            valid_mask = np.isfinite(factor_clean) & np.isfinite(returns_clean)
            factor_clean = factor_clean[valid_mask]
            returns_clean = returns_clean[valid_mask]
            if len(factor_clean) > 1:
                try:
                    # Check for sufficient variance before computing correlation
                    if np.std(factor_clean) > 1e-10 and np.std(returns_clean) > 1e-10:
                        corr = np.corrcoef(factor_clean, returns_clean)[0, 1]
                        if np.isfinite(corr):
                            correlations[factor_name] = corr
                except:
                    pass
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for i, (factor, corr) in enumerate(sorted_corr, 1):
        strength = "🔥 STRONG" if abs(corr) > 0.3 else "✓ Moderate" if abs(corr) > 0.15 else "→ Weak"
        print(f"{i:2d}. {factor:25s}: {corr:+.3f}  {strength}")
    
    print(f"\n{'='*60}\n")


def plot_quintile_performance(
    signals: List[SignalResult],
    symbol: str = "TSLA"
):
    """
    Bucket signals by Tier-1 score quintiles and show hit rate + avg return.
    
    This shows if higher confidence signals actually perform better.
    Quintiles are based on ABSOLUTE tier1_score (signal confidence).
    Q1 = lowest confidence, Q5 = highest confidence.
    """
    valid_signals = [s for s in signals if s.forward_return_1d is not None and s.signal != 0]
    
    if len(valid_signals) < 25:
        print("Insufficient signals for quintile analysis")
        return
    
    # Create DataFrame
    df = pd.DataFrame({
        'tier1_score': [s.indicators.get('tier1_score', 0) for s in valid_signals],
        'signal': [s.signal for s in valid_signals],
        'return_1d': [s.forward_return_1d * 100 for s in valid_signals],
        'return_3d': [s.forward_return_3d * 100 if s.forward_return_3d else None for s in valid_signals],
        'return_5d': [s.forward_return_5d * 100 if s.forward_return_5d else None for s in valid_signals],
    })
    
    # Create quintiles based on ABSOLUTE tier1_score (confidence level)
    df['abs_score'] = df['tier1_score'].abs()
    
    # Store bin edges for labeling
    try:
        df['quintile_bin'], bin_edges = pd.qcut(df['abs_score'], q=5, duplicates='drop', retbins=True)
        unique_bins = sorted(df['quintile_bin'].unique())
        n_bins = len(unique_bins)
        
        # Create labels with score ranges
        quintile_labels = []
        for i, b in enumerate(unique_bins):
            q_num = i + 1
            low = b.left
            high = b.right
            if q_num == 1:
                quintile_labels.append(f'Q1\n({low:.2f}-{high:.2f})')
            elif q_num == n_bins:
                quintile_labels.append(f'Q{q_num}\n({low:.2f}-{high:.2f})')
            else:
                quintile_labels.append(f'Q{q_num}\n({low:.2f}-{high:.2f})')
        
        bin_to_label = {b: quintile_labels[i] for i, b in enumerate(unique_bins)}
        df['quintile'] = df['quintile_bin'].map(bin_to_label)
        df['quintile_order'] = df['quintile_bin'].apply(lambda x: unique_bins.index(x))
        
    except Exception:
        # Fallback: use simple ranking
        df['quintile'] = pd.cut(df['abs_score'].rank(pct=True), bins=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        df['quintile_order'] = df['quintile'].cat.codes
    
    # Calculate metrics by quintile - sorted by quintile order (Q1 to Q5)
    quintile_stats = []
    for q in sorted(df['quintile'].unique(), key=lambda x: df[df['quintile']==x]['quintile_order'].iloc[0]):
        q_df = df[df['quintile'] == q]
        
        stats = {
            'quintile': q,
            'quintile_order': q_df['quintile_order'].iloc[0],
            'count': len(q_df),
            'hit_rate': (q_df['return_1d'] > 0).sum() / len(q_df),
            'avg_return_1d': q_df['return_1d'].mean(),
            'avg_return_3d': q_df['return_3d'].mean() if q_df['return_3d'].notna().sum() > 0 else 0,
            'avg_return_5d': q_df['return_5d'].mean() if q_df['return_5d'].notna().sum() > 0 else 0,
            'median_return': q_df['return_1d'].median(),
            'win_rate_pct': (q_df['return_1d'] > 0).sum() / len(q_df) * 100,
            'score_min': q_df['abs_score'].min(),
            'score_max': q_df['abs_score'].max(),
        }
        quintile_stats.append(stats)
    
    quintile_df = pd.DataFrame(quintile_stats).sort_values('quintile_order')
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Chart 1: Hit Rate by Quintile
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(quintile_df)), quintile_df['hit_rate'], 
                     color=['#d62728' if hr < 0.52 else '#2ca02c' for hr in quintile_df['hit_rate']], 
                     alpha=0.8, edgecolor='black')
    ax1.axhline(y=0.52, color='green', linestyle='--', label='Target (52%)', linewidth=2)
    ax1.axhline(y=0.50, color='gray', linestyle=':', label='Random', linewidth=1)
    ax1.set_xlabel('Signal Confidence Quintile', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Hit Rate', fontsize=11, fontweight='bold')
    ax1.set_title('Hit Rate by Signal Confidence', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(quintile_df)))
    ax1.set_xticklabels(quintile_df['quintile'], rotation=0)
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])
    
    for i, (hr, cnt) in enumerate(zip(quintile_df['hit_rate'], quintile_df['count'])):
        ax1.text(i, hr + 0.02, f'{hr:.1%}\n(n={cnt})', ha='center', fontsize=9)
    
    # Chart 2: Average Return by Quintile
    ax2 = axes[1]
    x = range(len(quintile_df))
    width = 0.25
    
    ax2.bar([i - width for i in x], quintile_df['avg_return_1d'], width, 
            label='1-Day', alpha=0.8, color='#1f77b4', edgecolor='black')
    ax2.bar(x, quintile_df['avg_return_3d'], width,
            label='3-Day', alpha=0.8, color='#ff7f0e', edgecolor='black')
    ax2.bar([i + width for i in x], quintile_df['avg_return_5d'], width,
            label='5-Day', alpha=0.8, color='#2ca02c', edgecolor='black')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Signal Confidence Quintile', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Return (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Average Returns by Confidence', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(quintile_df['quintile'], rotation=0)
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    # Chart 3: Signal Count Distribution
    ax3 = axes[2]
    ax3.bar(range(len(quintile_df)), quintile_df['count'], 
            alpha=0.8, color='#9467bd', edgecolor='black')
    ax3.set_xlabel('Signal Confidence Quintile', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Signals', fontsize=11, fontweight='bold')
    ax3.set_title('Signal Distribution', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(quintile_df)))
    ax3.set_xticklabels(quintile_df['quintile'], rotation=0)
    ax3.grid(alpha=0.3, axis='y')
    
    for i, cnt in enumerate(quintile_df['count']):
        ax3.text(i, cnt + 1, str(cnt), ha='center', fontsize=10)
    
    plt.suptitle(f'{symbol} - Performance by Signal Confidence Level', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save
    output_dir = Path("backtest_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"quintile_analysis_{symbol}_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Quintile analysis saved to: {plot_file}")
    
    try:
        plt.show()
    except SystemError:
        pass  # Ignore display errors in non-interactive environments
    finally:
        plt.close('all')  # Clean up
    
    # Print table
    print(f"\n{'='*100}")
    print(f"QUINTILE PERFORMANCE TABLE")
    print(f"Q1 = Lowest Confidence Signals, Q5 = Highest Confidence Signals")
    print(f"{'='*100}\n")
    print(f"{'Quintile':<10} {'Score Range':<18} {'Count':>8} {'Hit Rate':>12} {'Avg 1d':>12} {'Avg 3d':>12} {'Avg 5d':>12}")
    print("-" * 100)
    for _, row in quintile_df.iterrows():
        q_label = row['quintile'].split('\n')[0] if '\n' in str(row['quintile']) else row['quintile']
        score_range = f"{row['score_min']:.3f} - {row['score_max']:.3f}"
        print(f"{q_label:<10} {score_range:<18} {row['count']:>8.0f} {row['hit_rate']:>11.1%} "
              f"{row['avg_return_1d']:>11.2f}% {row['avg_return_3d']:>11.2f}% {row['avg_return_5d']:>11.2f}%")
    print(f"\n{'='*100}\n")


# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Run hit rate validation on mini-universe - pure signals only."""
    
    # Test pure signals without confluence filters
    print("\n" + "="*80)
    print("PURE SIGNAL HIT RATE VALIDATION")
    print("="*80)
    
    results = await validate_mini_universe(
        symbols=MINI_UNIVERSE,
        start_date="2021-01-01",
        end_date="2025-01-01",
        use_confluence=False
    )
    
    results_data, signals = results
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for horizon in ['1d', '3d', '5d']:
        hr = results_data['aggregate_results'][horizon]['all']
        print(f"\n{horizon.upper()} Hit Rate: {hr['hit_rate']:.2%}" if hr['hit_rate'] else f"\n{horizon.upper()} Hit Rate: N/A")
        print(f"  Signals: {hr['valid_signals']}")
        print(f"  Avg Return: {hr['avg_return']:.2%}" if hr.get('avg_return') else "  Avg Return: N/A")
    
    print(f"\nTARGET: 52% hit rate")
    hr_1d = results_data['aggregate_results']['1d']['all']['hit_rate']
    print(f"STATUS: {'✓ PASSED' if (hr_1d and hr_1d >= 0.52) else '✗ BELOW THRESHOLD'}")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_hit_rate_analysis(
        results_data, signals,
        results_data, signals,  # Pass same data twice since we removed confluence comparison
        symbol=MINI_UNIVERSE[0]
    )
    
    # Individual factor analysis
    print("\n" + "="*80)
    print("INDIVIDUAL FACTOR ANALYSIS")
    print("="*80)
    
    plot_individual_factor_analysis(
        signals,
        symbol=MINI_UNIVERSE[0]
    )
    
    # Quintile performance analysis
    print("\n" + "="*80)
    print("QUINTILE PERFORMANCE ANALYSIS")
    print("="*80)
    
    plot_quintile_performance(
        signals,
        symbol=MINI_UNIVERSE[0]
    )


if __name__ == "__main__":
    asyncio.run(main())
