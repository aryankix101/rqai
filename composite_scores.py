"""
Composite Score Calculation Module

This module calculates two types of composite scores:

1. TIER-1 COMPOSITES (Context/Slow-Moving):
   - Momentum, Trend, Relative Strength, Volume/Liquidity
   - Can use LEVEL-BASED features (normalized to [-1,1] via absolute thresholds)
   - Variables ending in "_level" are level-based: adx_level_score, capture_level_score, etc.
   - Used for: Regime classification, confirmation gating, long-term context
   - NOT directly used for next-day alpha prediction

2. NEXT-DAY ALPHA COMPOSITE (Fast/Predictive):
   - Uses ONLY Z-SCORED features (statistically normalized via median/MAD + tanh)
   - Variables ending in "_z" or "_zscore" are z-scored: ret_1d_z, gap_size_z, etc.
   - Adaptive to each stock's own volatility characteristics
   - Used for: T+1 trading decisions, short-horizon prediction

CRITICAL SEPARATION:
- Level-based features NEVER contaminate next-day alpha core
- Z-scored features provide statistical robustness across varying market conditions
- This separation prevents absolute-threshold biases in predictive scoring
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

 

def clip(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    if value is None:
        return 0.0
    return max(min_val, min(max_val, value))


def normalize(value: Optional[float], min_val: float, max_val: float) -> float:
    """
    Normalize value to [-1, 1] range based on ABSOLUTE min/max bounds.
    
    **LEVEL-BASED NORMALIZATION** - uses fixed thresholds, not adaptive statistics.
    Variables using this function should end in '_level' suffix.
    
    Use ONLY for Tier-1 context composites, NOT for next-day alpha prediction.
    """
    if value is None:
        return 0.0
    if max_val == min_val:
        return 0.0
    normalized = (value - min_val) / (max_val - min_val)
    return clip(normalized * 2 - 1, -1.0, 1.0)


def safe_get(data: Dict, key: str, default: float = 0.0) -> float:
    """Safely get value from dict with default."""
    val = data.get(key, default)
    return val if val is not None else default


# =============================================================================
# Momentum Composite Score (SLOW CONTEXT - Not for T+1 Trading)
# =============================================================================

def calculate_momentum_composite(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate momentum composite score [-1, 1].
    
    NOTE: This is a SLOW CONTEXT indicator, not suitable for T+1 alpha.
    Use for:
    - Regime/context detection
    - Position sizing modifier
    - Longer-term trend confirmation
    
    For T+1 trading decisions, use calculate_next_day_alpha_composite() instead.
    
    Inputs:
    - mom_1m, mom_3m, mom_6m, mom_12m (price momentum)
    - macd_histogram (momentum direction, normalized by ATR)
    - rsi_14d (momentum exhaustion check)
    
    Output:
    - Positive score = bullish momentum context
    - Negative score = bearish momentum context
    
    Args:
        indicators: Complete technical indicators dictionary
    
    Returns:
        {
            "score": float,              # [-1, 1]
            "components": dict,          # Individual component scores
            "interpretation": str        # Human-readable
        }
    """
    mom = indicators.get('momentum', {})
    trend = indicators.get('trend', {})
    
    # Get momentum z-scores if available, fall back to raw with normalization
    # Use .get() instead of safe_get to distinguish missing (None) from valid 0.0
    mom_1m_z = mom.get('mom_1m_zscore', None)
    mom_3m_z = mom.get('mom_3m_zscore', None)
    mom_6m_z = mom.get('mom_6m_zscore', None)
    mom_12m_z = mom.get('mom_12m_zscore', None)
    
    # Use z-scores if available (preferred), otherwise fall back to hardcoded normalization
    # Check for None explicitly - 0.0 is a valid z-score!
    if mom_3m_z is not None:
        # Z-scores: clip to [-3, 3] then scale to [-1, 1]
        # Note: These are statistically normalized (z-scored), but we use a common variable
        mom_1m_norm = clip((mom_1m_z if mom_1m_z is not None else 0.0) / 3.0, -1, 1)
        mom_3m_norm = clip((mom_3m_z if mom_3m_z is not None else 0.0) / 3.0, -1, 1)
        mom_6m_norm = clip((mom_6m_z if mom_6m_z is not None else 0.0) / 3.0, -1, 1)
        mom_12m_norm = clip((mom_12m_z if mom_12m_z is not None else 0.0) / 3.0, -1, 1)
    else:
        # Fallback: hardcoded ranges (legacy LEVEL-BASED behavior)
        mom_1m = safe_get(mom, 'mom_1m')
        mom_3m = safe_get(mom, 'mom_3m')
        mom_6m = safe_get(mom, 'mom_6m')
        mom_12m = safe_get(mom, 'mom_12m')
        mom_1m_norm = normalize(mom_1m, -0.10, 0.10)  # LEVEL-BASED fallback
        mom_3m_norm = normalize(mom_3m, -0.20, 0.20)  # LEVEL-BASED fallback
        mom_6m_norm = normalize(mom_6m, -0.30, 0.30)  # LEVEL-BASED fallback
        mom_12m_norm = normalize(mom_12m, -0.50, 0.50)  # LEVEL-BASED fallback
    
    # Weighted average (emphasize 3m most)
    mom_score = (
        0.40 * mom_3m_norm +   # 3-month most important
        0.30 * mom_1m_norm +   # Recent momentum
        0.20 * mom_6m_norm +   # Intermediate
        0.10 * mom_12m_norm    # Long-term context
    )
    
    # MACD confirmation - normalize by ATR% or price to handle scale differences
    macd_hist = safe_get(trend, 'macd_histogram')
    volatility = indicators.get('volatility', {})
    atr_pct = safe_get(volatility, 'atr_pct_14d', 0.02)  # Default 2%
    current_price = indicators.get('id', {}).get('price', 100.0)
    
    # Normalize MACD by ATR-based scale instead of hardcoded ±5
    if atr_pct > 0.001 and current_price > 0:
        # MACD in price terms, normalize by ATR in price terms
        atr_price = atr_pct * current_price
        macd_normalized = macd_hist / (atr_price * 2) if atr_price > 0 else 0
        macd_signal = clip(macd_normalized, -1, 1)
    else:
        # Fallback to price-based normalization
        macd_signal = clip(macd_hist / (current_price * 0.02), -1, 1) if current_price > 0 else 0
    
    # RSI moderation - only penalize extremes in weak momentum regimes
    # In strong trends, RSI overbought/oversold is normal and should not be penalized
    rsi = safe_get(trend, 'rsi_14d', 50.0)
    rsi_mod = 0.0
    
    # Only apply RSI adjustment if momentum is weak or mixed
    if abs(mom_score) < 0.4:  # Weak momentum regime
        if rsi > 75:  # Extreme overbought
            rsi_mod = -0.15
        elif rsi < 25:  # Extreme oversold
            rsi_mod = 0.15
    # In strong trends (|mom_score| >= 0.4), ignore RSI extremes
    
    rsi_mod = clip(rsi_mod, -0.2, 0.2)
    
    volume_liq = indicators.get('volume_liquidity', {})
    vol_spike_z = safe_get(volume_liq, 'volume_spike_z', 0.0)
    vol_mod = 0.0
    if mom_score > 0.3 and vol_spike_z < -1.5:
        # Positive momentum without volume = weak signal
        vol_mod = -0.1
    elif vol_spike_z > 2.0:
        # High volume confirms momentum
        vol_mod = 0.1
    vol_mod = clip(vol_mod, -0.15, 0.15)
    
    # Combine components
    final_score = clip(
        0.65 * mom_score +      # 65% raw momentum
        0.20 * macd_signal +    # 20% MACD confirmation
        0.10 * rsi_mod +        # 10% RSI moderation
        0.05 * vol_mod          # 5% volume confirmation
    )
    
    # Interpretation
    if final_score > 0.5:
        interpretation = "Strong bullish momentum"
    elif final_score > 0.2:
        interpretation = "Moderate bullish momentum"
    elif final_score > -0.2:
        interpretation = "Neutral momentum"
    elif final_score > -0.5:
        interpretation = "Moderate bearish momentum"
    else:
        interpretation = "Strong bearish momentum"
    
    return {
        "score": final_score,
        "components": {
            "momentum_returns": mom_score,
            "macd_signal": macd_signal,
            "rsi_moderation": rsi_mod
        },
        "interpretation": interpretation
    }


# =============================================================================
# Trend Composite Score
# =============================================================================

def calculate_trend_composite(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate trend composite score [-1, 1].
    
    Inputs:
    - Price vs MA20/MA75/MA200 (moving average structure)
    - ADX (trend strength)
    - higher_highs_lows (price structure)
    - Golden/death cross events (MA75 vs MA200)
    
    Output:
    - Positive = uptrend
    - Negative = downtrend
    
    Args:
        indicators: Complete technical indicators dictionary
    
    Returns:
        {
            "score": float,
            "components": dict,
            "interpretation": str
        }
    """
    trend = indicators.get('trend', {})
    
    # MA structure score - using MA20, MA75, MA200
    price_above_ma20 = trend.get('price_above_ma20')
    price_above_ma75 = trend.get('price_above_ma75')
    price_above_ma200 = trend.get('price_above_ma200')
    
    # Score based on MA alignment
    ma_score = 0.0
    if price_above_ma20 and price_above_ma75 and price_above_ma200:
        ma_score = 1.0  # Fully bullish structure (price > all MAs)
    elif not price_above_ma20 and not price_above_ma75 and not price_above_ma200:
        ma_score = -1.0  # Fully bearish structure (price < all MAs)
    elif price_above_ma75 and price_above_ma200:
        ma_score = 0.6  # Bullish but short-term weakness (below MA20)
    elif price_above_ma200:
        ma_score = 0.3  # Long-term bullish, medium-term weakness
    elif not price_above_ma200 and price_above_ma75:
        ma_score = -0.3  # Long-term bearish, medium-term strength
    elif not price_above_ma200 and not price_above_ma75 and price_above_ma20:
        ma_score = -0.6  # Bearish but short-term bounce
    else:
        ma_score = 0.0  # Mixed/unclear
    
    # Golden/Death cross boost
    golden_cross = trend.get('golden_cross_recent', False)
    death_cross = trend.get('death_cross_recent', False)
    
    cross_boost = 0.0
    if golden_cross:
        cross_boost = 0.3
    elif death_cross:
        cross_boost = -0.3
    
    # ADX trend strength multiplier (LEVEL-BASED: absolute thresholds)
    adx = safe_get(trend, 'adx_14d', 20.0)
    # ADX: 0-20 (weak), 20-25 (moderate), 25+ (strong)
    adx_level_score = normalize(adx, 0, 40)  # LEVEL-BASED
    adx_level_score = (adx_level_score + 1) / 2  # Convert to [0, 1] range
    adx_level_score = max(0.3, adx_level_score)   # Min 30% weight
    
    # Price structure
    structure = trend.get('higher_highs_lows', 'mixed')
    structure_score = 0.0
    if structure == 'uptrend':
        structure_score = 0.8
    elif structure == 'downtrend':
        structure_score = -0.8
    else:
        structure_score = 0.0
    
    # Combine: base trend * strength + boosts
    base_trend = (0.6 * ma_score + 0.4 * structure_score)
    
    # Check for trend consistency (all signals aligned)
    macd_line = safe_get(trend, 'macd_line', 0.0)
    consistency_multiplier = 1.0
    if (ma_score > 0 and structure_score > 0 and macd_line > 0):
        # All bullish indicators aligned - boost by 20%
        consistency_multiplier = 1.2
    elif (ma_score < 0 and structure_score < 0 and macd_line < 0):
        # All bearish indicators aligned - amplify bearish signal by 20%
        consistency_multiplier = 1.2
    
    # Apply ADX dampening for choppy markets, but not too aggressively
    if adx < 20 and abs(ma_score) < 0.5:
        # Weak trend AND unclear MA signal
        adx_dampener = 0.5
    elif adx < 20:
        # Weak trend but clear MA signal
        adx_dampener = 0.7
    else:
        adx_dampener = 1.0
    
    final_score = clip(
        base_trend * consistency_multiplier * adx_level_score * adx_dampener + cross_boost
    )
    
    # Interpretation
    adx_strength = "strong" if adx > 25 else ("moderate" if adx > 20 else "weak")
    if final_score > 0.5:
        interpretation = f"Strong uptrend ({adx_strength} ADX={adx:.1f})"
    elif final_score > 0.2:
        interpretation = f"Moderate uptrend ({adx_strength} ADX={adx:.1f})"
    elif final_score > -0.2:
        interpretation = f"Neutral/choppy ({adx_strength} ADX={adx:.1f})"
    elif final_score > -0.5:
        interpretation = f"Moderate downtrend ({adx_strength} ADX={adx:.1f})"
    else:
        interpretation = f"Strong downtrend ({adx_strength} ADX={adx:.1f})"
    
    return {
        "score": final_score,
        "components": {
            "ma_structure": ma_score,
            "price_structure": structure_score,
            "adx_level_score": adx_level_score,
            "cross_boost": cross_boost
        },
        "interpretation": interpretation
    }


# =============================================================================
# Relative Strength Composite Score
# =============================================================================

def calculate_relative_strength_composite(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate relative strength composite score [-1, 1].
    
    Inputs:
    - vs_sp500_3m, vs_sp500_6m, vs_sp500_12m
    - rel_strength_trend (rising/falling)
    - up_capture / down_capture ratios
    
    Output:
    - Positive = outperforming market
    - Negative = underperforming market
    
    Args:
        indicators: Complete technical indicators dictionary
    
    Returns:
        {
            "score": float,
            "components": dict,
            "interpretation": str
        }
    """
    rel_perf = indicators.get('relative_perf', {})
    risk = indicators.get('risk', {})
    
    # Relative performance vs S&P 500 (LEVEL-BASED: fixed thresholds)
    vs_sp500_3m = safe_get(rel_perf, 'vs_sp500_3m')
    vs_sp500_6m = safe_get(rel_perf, 'vs_sp500_6m')
    vs_sp500_12m = safe_get(rel_perf, 'vs_sp500_12m')
    
    # Normalize using LEVEL-BASED thresholds (±20% relative outperformance range)
    rs_3m_level = normalize(vs_sp500_3m, -0.20, 0.20)  # LEVEL-BASED
    rs_6m_level = normalize(vs_sp500_6m, -0.30, 0.30)  # LEVEL-BASED
    rs_12m_level = normalize(vs_sp500_12m, -0.50, 0.50)  # LEVEL-BASED
    
    # Weighted average (emphasize recent)
    rs_score = (
        0.50 * rs_3m_level +
        0.30 * rs_6m_level +
        0.20 * rs_12m_level
    )
    
    # Relative strength trend bonus
    rs_trend = rel_perf.get('rel_strength_trend', 'flat')
    trend_bonus = 0.0
    if rs_trend == 'rising':
        trend_bonus = 0.2
    elif rs_trend == 'falling':
        trend_bonus = -0.2
    
    # Capture ratios (if available) - LEVEL-BASED
    up_capture = safe_get(risk, 'up_capture_1y', 1.0)
    down_capture = safe_get(risk, 'down_capture_1y', 1.0)
    
    # Ideal: up_capture > 1.0, down_capture < 1.0
    capture_level_score = 0.0
    if up_capture > 0 and down_capture > 0:
        # Ratio of up/down capture (>1 is good)
        capture_ratio = up_capture / down_capture
        capture_level_score = normalize(np.log(capture_ratio), -0.5, 0.5)  # LEVEL-BASED
        capture_level_score *= 0.15  # Weight
    
    # Combine
    final_score = clip(rs_score + trend_bonus + capture_level_score)
    
    # Interpretation
    if final_score > 0.5:
        interpretation = "Strong outperformance vs S&P 500"
    elif final_score > 0.2:
        interpretation = "Moderate outperformance vs S&P 500"
    elif final_score > -0.2:
        interpretation = "In-line with S&P 500"
    elif final_score > -0.5:
        interpretation = "Moderate underperformance vs S&P 500"
    else:
        interpretation = "Strong underperformance vs S&P 500"
    
    return {
        "score": final_score,
        "components": {
            "relative_performance": rs_score,
            "trend_bonus": trend_bonus,
            "capture_ratio_score": capture_level_score
        },
        "interpretation": interpretation
    }


# =============================================================================
# Volume Confirmation Composite Score
# =============================================================================

def calculate_volume_confirmation_composite(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate volume confirmation composite score [0, 1].
    
    Inputs:
    - OBV trend
    - A/D trend
    - Volume spike
    - Liquidity score
    
    Output:
    - High score = volume confirms price action
    - Low score = volume divergence or weak
    
    Args:
        indicators: Complete technical indicators dictionary
    
    Returns:
        {
            "score": float,          # [0, 1]
            "components": dict,
            "interpretation": str
        }
    """
    vol_liq = indicators.get('volume_liquidity', {})
    momentum = indicators.get('momentum', {})
    
    # OBV trend
    obv_trend = vol_liq.get('obv_trend_21d', 'flat')
    obv_score = 0.5  # Neutral
    if obv_trend == 'up':
        obv_score = 1.0
    elif obv_trend == 'down':
        obv_score = 0.0
    
    # A/D trend
    ad_trend = vol_liq.get('acc_dist_trend_21d', 'flat')
    ad_score = 0.5  # Neutral
    if ad_trend == 'up':
        ad_score = 1.0
    elif ad_trend == 'down':
        ad_score = 0.0
    
    # Volume spike (moderate spike is good, extreme is suspicious, low is weak)
    vol_spike_z = safe_get(vol_liq, 'volume_spike_z')
    spike_score = 0.5
    if vol_spike_z >= 3:
        spike_score = 0.6  # Extreme - could be exhaustion but still strong
    elif vol_spike_z >= 2:
        spike_score = 1.0  # Strong volume increase
    elif vol_spike_z > 0:
        spike_score = 0.7  # Above average volume
    elif vol_spike_z > -1:
        spike_score = 0.3  # Slightly below average - weak
    else:
        spike_score = 0.1  # Very low volume - very weak signal
    
    # Check for volume-price confirmation
    # Use short-horizon returns for T+1 relevance (not mom_3m which is too slow)
    price_events = indicators.get('price_events', {})
    ret_1d = safe_get(momentum, 'ret_1d', 0.0)
    ret_3d = safe_get(momentum, 'ret_3d', 0.0)
    intraday_ret = safe_get(price_events, 'intraday_ret', 0.0)
    
    # Short-term price movement for confirmation
    short_move = 0.5 * ret_1d + 0.3 * ret_3d + 0.2 * intraday_ret
    
    confirmation_bonus = 0.0
    if short_move > 0.01 and obv_trend == 'up' and ad_trend == 'up':
        confirmation_bonus = 0.2  # Strong confirmation
    elif short_move < -0.01 and obv_trend == 'down' and ad_trend == 'down':
        confirmation_bonus = 0.1  # Bearish confirmation (less weight)
    elif short_move > 0.01 and (obv_trend == 'down' or ad_trend == 'down'):
        confirmation_bonus = -0.2  # Bearish divergence
    elif short_move < -0.01 and (obv_trend == 'up' or ad_trend == 'up'):
        confirmation_bonus = 0.2  # Bullish divergence (potential reversal)
    
    # Combine
    base_score = 0.4 * obv_score + 0.4 * ad_score + 0.2 * spike_score
    final_score = clip(base_score + confirmation_bonus, 0.0, 1.0)
    
    # Interpretation
    if final_score > 0.75:
        interpretation = "Strong volume confirmation"
    elif final_score > 0.6:
        interpretation = "Good volume confirmation"
    elif final_score > 0.4:
        interpretation = "Neutral volume"
    else:
        interpretation = "Weak volume or divergence"
    
    return {
        "score": final_score,
        "components": {
            "obv": obv_score,
            "accumulation_distribution": ad_score,
            "volume_spike": spike_score,
            "confirmation_bonus": confirmation_bonus
        },
        "interpretation": interpretation
    }


# =============================================================================
# Risk Composite Score
# =============================================================================

def calculate_risk_composite(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate risk composite score [0, 1].
    
    Inputs:
    - Sharpe ratio
    - Sortino ratio
    - Max drawdown
    - Beta
    - Volatility
    
    Output:
    - High score = favorable risk profile
    - Low score = unfavorable risk
    
    Args:
        indicators: Complete technical indicators dictionary
    
    Returns:
        {
            "score": float,          # [0, 1]
            "components": dict,
            "interpretation": str
        }
    """
    risk = indicators.get('risk', {})
    vol = indicators.get('volatility', {})
    
    # Sharpe ratio - use robust z-score + sigmoid for smooth scoring
    sharpe = safe_get(risk, 'sharpe_1y')
    if sharpe is not None:
        # Historical Sharpe values would be ideal, but we use heuristic scaling
        # Sharpe ~0.5 = neutral, ~1.5 = good, ~2.5 = excellent
        # Map via tanh: score = 0.5 * (tanh((sharpe - 0.5) / 0.8) + 1)
        sharpe_z = (sharpe - 0.5) / 0.8  # Center at 0.5, scale by 0.8
        sharpe_score = 0.5 * (np.tanh(np.clip(sharpe_z, -3, 3)) + 1)
    else:
        sharpe_score = 0.5
    
    # Sortino ratio - similar smooth scoring
    sortino = safe_get(risk, 'sortino_1y')
    if sortino is not None:
        # Sortino ~0.7 = neutral, ~2.0 = good, ~3.5 = excellent
        sortino_z = (sortino - 0.7) / 1.0
        sortino_score = 0.5 * (np.tanh(np.clip(sortino_z, -3, 3)) + 1)
    else:
        sortino_score = 0.5
    
    # Max drawdown - smooth exponential penalty (not harsh cutoff at 50%)
    max_dd = safe_get(risk, 'max_drawdown_1y', 0.3)
    # Smooth penalty curve:
    # DD < 10%: ~1.0 (excellent)
    # DD = 20%: ~0.80 (good)
    # DD = 30%: ~0.60 (moderate)
    # DD = 50%: ~0.30 (risky)
    # DD = 70%: ~0.10 (broken)
    # Formula: 1 - tanh(DD / 0.25)^1.5
    if max_dd < 0.05:
        dd_score = 1.0
    else:
        dd_normalized = max_dd / 0.25  # Scale: 25% DD → 1.0
        dd_score = 1.0 - np.tanh(dd_normalized) ** 1.5
        dd_score = max(0.0, min(1.0, dd_score))
    
    # Volatility scoring - lower volatility = better risk score
    realized_vol = safe_get(vol, 'realized_vol_252d', 0.3)
    if realized_vol < 0.12:
        vol_score = 1.0  # Very low volatility - excellent
    elif realized_vol < 0.20:
        vol_score = 0.85  # Low-moderate volatility - good
    elif realized_vol < 0.30:
        vol_score = 0.60  # Moderate volatility - acceptable
    elif realized_vol < 0.45:
        vol_score = 0.30  # High volatility - risky
    else:
        vol_score = 0.10  # Extreme volatility - very risky
    
    # Beta consideration (prefer 0.8-1.5 range)
    beta = safe_get(risk, 'beta_1y', 1.0)
    if 0.8 <= beta <= 1.5:
        beta_score = 1.0
    elif beta < 0.8:
        beta_score = 0.7  # Low beta (defensive)
    elif beta < 2.0:
        beta_score = 0.6  # High beta
    else:
        beta_score = 0.3  # Very high beta
    
    # Combine (emphasize drawdown and Sharpe most)
    final_score = clip(
        0.30 * sharpe_score +
        0.25 * dd_score +
        0.20 * sortino_score +
        0.15 * vol_score +
        0.10 * beta_score,
        0.0, 1.0
    )
    
    # Interpretation
    if final_score > 0.75:
        interpretation = "Excellent risk profile"
    elif final_score > 0.6:
        interpretation = "Good risk profile"
    elif final_score > 0.4:
        interpretation = "Moderate risk profile"
    else:
        interpretation = "Unfavorable risk profile"
    
    return {
        "score": final_score,
        "components": {
            "sharpe": sharpe_score,
            "sortino": sortino_score,
            "max_drawdown": dd_score,
            "volatility": vol_score,
            "beta": beta_score
        },
        "interpretation": interpretation
    }


# =============================================================================
# Next-Day Alpha Composite Score (T+1 Prediction)
# =============================================================================

def calculate_tier1_alpha_confirmation(
    alpha_score: float,
    tier1_score: float,
    alpha_threshold: float = 0.20,
    tier1_threshold: float = 0.15
) -> Dict[str, Any]:
    """
    Calculate magnitude-aware confirmation between alpha and tier1 signals.
    
    Only applies confirmation boost/penalty when BOTH signals are meaningful.
    Avoids penalizing good alpha trades when tier1 is noisy/flat.
    
    Args:
        alpha_score: Next-day alpha score [-1, +1]
        tier1_score: Tier-1 deterministic score [-1, +1]
        alpha_threshold: Minimum |alpha| to consider "real" signal
        tier1_threshold: Minimum |tier1| to consider "meaningful"
    
    Returns:
        {
            "confidence": str,  # "high", "low", "neutral"
            "size_multiplier": float,  # 0.7 to 1.2
            "agreement": bool,
            "alpha_strong": bool,
            "tier1_meaningful": bool
        }
    """
    alpha_strong = abs(alpha_score) >= alpha_threshold
    tier1_meaningful = abs(tier1_score) >= tier1_threshold
    signs_match = np.sign(alpha_score) == np.sign(tier1_score)
    
    # Only apply confirmation when BOTH are real signals
    if alpha_strong and tier1_meaningful:
        if signs_match:
            confidence = "high"
            size_multiplier = 1.2
            agreement = True
        else:
            confidence = "low"
            size_multiplier = 0.7
            agreement = False
    else:
        # Don't let noise shrink size
        confidence = "neutral"
        size_multiplier = 1.0
        agreement = None
    
    return {
        "confidence": confidence,
        "size_multiplier": size_multiplier,
        "agreement": agreement,
        "alpha_strong": alpha_strong,
        "tier1_meaningful": tier1_meaningful
    }


def calculate_next_day_alpha_composite(
    indicators: Dict[str, Any],
    hold_threshold: float = 0.20,
    liquidity_min_score: float = 30.0,
    liquidity_min_dollar_vol: float = 5_000_000.0,
    tier1_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate next-day alpha composite score [-1, 1] for T+1 prediction.
    
    This composite is specifically designed for short-horizon (next-day) trading
    decisions, using fast-reacting indicators and regime-aware weighting.
    
    Inputs:
    - ret_1d, ret_3d, ret_5d, ret_10d (short-horizon returns)
    - gap_size_pct, intraday_ret, range_pct (intraday features)
    - close_loc_10d (close location in recent range [0,1])
    - rsi_2d, rsi_3d (short RSI for mean reversion)
    - autocorr_5d, autocorr_10d, autocorr_21d (return autocorrelation for regime)
    - vol_ratio_21v252 (volatility regime - secondary)
    - adx_14d (trend strength)
    - ma200_slope_up (trend regime)
    
    Regime Detection:
    - autocorr < -0.08 → mean_reversion (fade moves)
    - autocorr > +0.08 → continuation (follow moves)
    - else use vol_ratio as tiebreaker
    
    Output:
    - score in [-1, 1]: positive = bullish T+1, negative = bearish T+1
    - decision: 'LONG', 'SHORT', or 'HOLD'
    - components: breakdown of score
    - interpretation: human-readable
    
    Args:
        indicators: Complete technical indicators dictionary
        hold_threshold: Minimum |score| to trade (default 0.20)
        liquidity_min_score: Minimum liquidity score to trade
        liquidity_min_dollar_vol: Minimum avg dollar volume to trade
    
    Returns:
        {
            "score": float,          # [-1, 1]
            "decision": str,         # 'LONG', 'SHORT', 'HOLD'
            "components": dict,
            "regime": str,           # 'mean_reversion', 'continuation', 'mixed'
            "gates": dict,           # Permission gate status
            "interpretation": str
        }
    """
    momentum = indicators.get('momentum', {})
    trend = indicators.get('trend', {})
    price_events = indicators.get('price_events', {})
    volatility = indicators.get('volatility', {})
    vol_liq = indicators.get('volume_liquidity', {})
    
    # ==========================================================================
    # 1. Short-horizon return features (use z-scores when available)
    # ==========================================================================
    ret_1d = safe_get(momentum, 'ret_1d', 0.0)
    ret_3d = safe_get(momentum, 'ret_3d', 0.0)
    ret_5d = safe_get(momentum, 'ret_5d', 0.0)
    ret_10d = safe_get(momentum, 'ret_10d', 0.0)
    
    # Z-scores (adaptive normalization based on stock's own history)
    ret_1d_z = safe_get(momentum, 'ret_1d_zscore', None)
    ret_3d_z = safe_get(momentum, 'ret_3d_zscore', None)
    ret_5d_z = safe_get(momentum, 'ret_5d_zscore', None)
    ret_10d_z = safe_get(momentum, 'ret_10d_zscore', None)
    
    # ==========================================================================
    # 2. Intraday/range features (use z-scores when available)
    # ==========================================================================
    gap_size = safe_get(price_events, 'gap_size_pct', 0.0)
    intraday_ret = safe_get(price_events, 'intraday_ret', 0.0)
    range_pct = safe_get(price_events, 'range_pct', 0.02)
    close_loc = safe_get(price_events, 'close_loc_10d', 0.5)
    
    # Z-scores for intraday features
    gap_size_z = safe_get(price_events, 'gap_size_zscore', None)
    intraday_ret_z = safe_get(price_events, 'intraday_ret_zscore', None)
    range_pct_z = safe_get(price_events, 'range_pct_zscore', None)
    
    # ==========================================================================
    # 3. Short RSI (mean reversion signals)
    # ==========================================================================
    rsi_2d = safe_get(trend, 'rsi_2d', 50.0)
    rsi_3d = safe_get(trend, 'rsi_3d', 50.0)
    
    # RSI mean reversion signal: extreme RSI suggests reversal
    # RSI 2d < 10 = very oversold (bullish), RSI 2d > 90 = very overbought (bearish)
    rsi_reversion = 0.0
    if rsi_2d < 10:
        rsi_reversion = 0.8  # Strong bullish reversion
    elif rsi_2d < 20:
        rsi_reversion = 0.5
    elif rsi_2d < 30:
        rsi_reversion = 0.2
    elif rsi_2d > 90:
        rsi_reversion = -0.8  # Strong bearish reversion
    elif rsi_2d > 80:
        rsi_reversion = -0.5
    elif rsi_2d > 70:
        rsi_reversion = -0.2
    
    # ==========================================================================
    # 4. Close location signal
    # ==========================================================================
    # close_loc_10d: 0 = at 10-day low, 1 = at 10-day high
    # For mean reversion: low close_loc = bullish, high = bearish
    # For continuation: high close_loc = bullish (breakout), low = bearish
    close_loc_reversion = (0.5 - close_loc) * 2  # Maps [0,1] to [1,-1]
    close_loc_continuation = (close_loc - 0.5) * 2  # Maps [0,1] to [-1,1]
    
    # ==========================================================================
    # 5. Regime detection: Mean Reversion vs Continuation
    # Uses autocorrelation (primary) + vol_ratio (secondary)
    # ==========================================================================
    vol_ratio = safe_get(volatility, 'vol_ratio_21v252', 1.0)
    adx = safe_get(trend, 'adx_14d', 20.0)
    ma200_slope_up = trend.get('ma200_slope_up', None)
    
    # Autocorrelation: positive = trending, negative = mean-reverting
    autocorr_5d = safe_get(momentum, 'autocorr_5d', 0.0)
    autocorr_10d = safe_get(momentum, 'autocorr_10d', 0.0)
    autocorr_21d = safe_get(momentum, 'autocorr_21d', 0.0)
    
    # Weighted average of autocorrelations (favor shorter-term for T+1)
    autocorr_avg = 0.5 * autocorr_5d + 0.3 * autocorr_10d + 0.2 * autocorr_21d
    
    # Regime scoring based on autocorrelation + vol_ratio
    # Autocorrelation is primary signal, vol_ratio is confirmation
    reversion_weight = 0.5  # Base weight
    continuation_weight = 0.5
    
    # Primary: Autocorrelation-based regime
    if autocorr_avg < -0.08:
        # Negative autocorr = mean-reverting stock
        reversion_weight = 0.70
        continuation_weight = 0.30
        regime = 'mean_reversion'
    elif autocorr_avg > 0.08:
        # Positive autocorr = trending stock
        reversion_weight = 0.30
        continuation_weight = 0.70
        regime = 'continuation'
    # Secondary: Vol ratio as tiebreaker when autocorr is unclear
    elif vol_ratio > 1.5:
        # High volatility regime - favor mean reversion
        reversion_weight = 0.65
        continuation_weight = 0.35
        regime = 'mean_reversion'
    elif vol_ratio < 0.8 and adx > 20 and ma200_slope_up:
        # Low vol, trending, upward slope - favor continuation
        reversion_weight = 0.35
        continuation_weight = 0.65
        regime = 'continuation'
    elif vol_ratio < 0.8 and adx > 20 and ma200_slope_up == False:
        # Trending down - bearish continuation
        reversion_weight = 0.35
        continuation_weight = 0.65
        regime = 'continuation'
    else:
        regime = 'mixed'
    
    # ==========================================================================
    # 6. Calculate regime-weighted signals (using z-scores for adaptive scaling)
    # ==========================================================================
    
    # Mean reversion signal
    # Negative short-term returns + oversold RSI = bullish reversion
    
    # Use z-score for short-term return signal (adaptive to stock's own vol)
    # If z-score not available, fall back to range-normalized
    if ret_1d_z is not None:
        short_ret_z_signal = clip(ret_1d_z / 2.0, -1, 1)  # ±2σ maps to ±1
    elif range_pct > 0.001:
        short_ret_z_signal = clip(ret_1d / range_pct / 3.0, -1, 1)
    else:
        short_ret_z_signal = 0.0
    
    # Use z-score for gap signal if available
    if gap_size_z is not None:
        gap_signal = clip(gap_size_z / 2.0, -1, 1)
    else:
        gap_signal = clip(gap_size * 10, -1, 1)
    
    # Use z-score for 3d return if available
    if ret_3d_z is not None:
        ret_3d_signal = clip(ret_3d_z / 2.0, -1, 1)
    else:
        ret_3d_signal = clip(ret_3d * 10, -1, 1)
    
    reversion_signal = (
        0.30 * rsi_reversion +           # RSI mean reversion
        0.30 * close_loc_reversion +     # Close location reversion
        0.25 * (-short_ret_z_signal) +   # Fade short-term move (z-scored)
        0.15 * (-gap_signal)             # Fade gaps (z-scored)
    )
    
    # Continuation signal
    # Positive momentum + high close_loc + gap in direction = bullish continuation
    continuation_signal = (
        0.35 * short_ret_z_signal +      # Follow short-term momentum (z-scored)
        0.30 * close_loc_continuation +  # High in range = bullish
        0.20 * ret_3d_signal +           # 3-day momentum (z-scored)
        0.15 * gap_signal                # Follow gaps (z-scored)
    )
    
    # Blend signals based on regime
    raw_alpha = (
        reversion_weight * reversion_signal +
        continuation_weight * continuation_signal
    )
    
    # ==========================================================================
    # 7. Permission gates
    # ==========================================================================
    liquidity_score = safe_get(vol_liq, 'liquidity_score_0_100', 50.0)
    avg_dollar_vol = safe_get(vol_liq, 'avg_dollar_vol_21d', 10_000_000.0)
    
    gates = {
        'liquidity_ok': liquidity_score >= liquidity_min_score,
        'dollar_vol_ok': avg_dollar_vol >= liquidity_min_dollar_vol,
        'long_permitted': True,
        'short_permitted': True
    }
    
    # If MA200 slope is down, cap long signals or raise threshold
    if ma200_slope_up == False:
        gates['long_permitted'] = raw_alpha > 0.4  # Higher bar for longs in downtrend
        if raw_alpha > 0 and raw_alpha <= 0.4:
            raw_alpha *= 0.5  # Dampen long signals in downtrend
    
    # If MA200 slope is up, cap short signals
    if ma200_slope_up == True:
        gates['short_permitted'] = raw_alpha < -0.4  # Higher bar for shorts in uptrend
        if raw_alpha < 0 and raw_alpha >= -0.4:
            raw_alpha *= 0.5  # Dampen short signals in uptrend
    
    # Final score
    final_score = clip(raw_alpha, -1, 1)
    
    # ==========================================================================
    # 8. Decision with HOLD zone
    # ==========================================================================
    if not gates['liquidity_ok'] or not gates['dollar_vol_ok']:
        decision = 'HOLD'
        decision_reason = 'Insufficient liquidity'
    elif final_score > hold_threshold and gates['long_permitted']:
        decision = 'LONG'
        decision_reason = f'Alpha score {final_score:.3f} > threshold {hold_threshold}'
    elif final_score < -hold_threshold and gates['short_permitted']:
        decision = 'SHORT'
        decision_reason = f'Alpha score {final_score:.3f} < -{hold_threshold}'
    else:
        decision = 'HOLD'
        decision_reason = f'Alpha score {abs(final_score):.3f} within HOLD zone ±{hold_threshold}'
    
    # ==========================================================================
    # 9. Tier-1 Confirmation (magnitude-aware)
    # ==========================================================================
    tier1_confirmation = None
    if tier1_score is not None:
        tier1_confirmation = calculate_tier1_alpha_confirmation(
            alpha_score=final_score,
            tier1_score=tier1_score,
            alpha_threshold=hold_threshold,
            tier1_threshold=0.15
        )
    
    # ==========================================================================
    # 10. Interpretation
    # ==========================================================================
    if final_score > 0.5:
        interpretation = f"Strong bullish T+1 signal ({regime} regime)"
    elif final_score > 0.2:
        interpretation = f"Moderate bullish T+1 signal ({regime} regime)"
    elif final_score > -0.2:
        interpretation = f"Neutral T+1 ({regime} regime)"
    elif final_score > -0.5:
        interpretation = f"Moderate bearish T+1 signal ({regime} regime)"
    else:
        interpretation = f"Strong bearish T+1 signal ({regime} regime)"
    
    # Append tier1 confirmation if available
    if tier1_confirmation and tier1_confirmation['confidence'] != 'neutral':
        conf_label = tier1_confirmation['confidence']
        interpretation += f" [Tier-1 {conf_label} confidence]"
    
    return {
        "score": final_score,
        "decision": decision,
        "decision_reason": decision_reason,
        "regime": regime,
        "components": {
            "reversion_signal": reversion_signal,
            "continuation_signal": continuation_signal,
            "reversion_weight": reversion_weight,
            "continuation_weight": continuation_weight,
            "autocorr_avg": autocorr_avg,
            "autocorr_5d": autocorr_5d,
            "autocorr_10d": autocorr_10d,
            "autocorr_21d": autocorr_21d,
            "vol_ratio": vol_ratio,
            "rsi_reversion": rsi_reversion,
            "close_loc_reversion": close_loc_reversion,
            "short_ret_z_signal": short_ret_z_signal,
            "gap_signal_z": gap_signal,
            "ret_3d_signal_z": ret_3d_signal,
            "using_zscore_ret_1d": ret_1d_z is not None,
            "using_zscore_gap": gap_size_z is not None,
            "using_zscore_ret_3d": ret_3d_z is not None
        },
        "gates": gates,
        "tier1_confirmation": tier1_confirmation,
        "interpretation": interpretation
    }


# =============================================================================
# Master Function: Calculate All Composites
# =============================================================================

def calculate_all_composites(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all composite scores for a stock.
    
    Args:
        indicators: Complete technical indicators dictionary
    
    Returns:
        {
            "symbol": str,
            "momentum_composite": dict,
            "trend_composite": dict,
            "rs_composite": dict,
            "volume_composite": dict,
            "risk_composite": dict
        }
    """
    symbol = indicators.get('id', {}).get('symbol', 'UNKNOWN')
    
    return {
        "symbol": symbol,
        "momentum_composite": calculate_momentum_composite(indicators),
        "trend_composite": calculate_trend_composite(indicators),
        "rs_composite": calculate_relative_strength_composite(indicators),
        "volume_composite": calculate_volume_confirmation_composite(indicators),
        "risk_composite": calculate_risk_composite(indicators),
        "next_day_alpha": calculate_next_day_alpha_composite(indicators)
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import json
    from pathlib import Path
    
    output_dir = Path("./output")
    
    if not output_dir.exists():
        logger.error("Output directory not found.")
        exit(1)
    
    # Load and process all stocks
    for json_file in output_dir.glob("*_technical_indicators.json"):
        with open(json_file) as f:
            indicators = json.load(f)
        
        symbol = indicators['id']['symbol']
        composites = calculate_all_composites(indicators)
        
        print(f"\n{'='*80}")
        print(f"{symbol} - COMPOSITE SCORES")
        print(f"{'='*80}")
        print(f"Momentum:  {composites['momentum_composite']['score']:+.3f} - {composites['momentum_composite']['interpretation']}")
        print(f"Trend:     {composites['trend_composite']['score']:+.3f} - {composites['trend_composite']['interpretation']}")
        print(f"Rel Str:   {composites['rs_composite']['score']:+.3f} - {composites['rs_composite']['interpretation']}")
        print(f"Volume:    {composites['volume_composite']['score']: .3f} - {composites['volume_composite']['interpretation']}")
        print(f"Risk:      {composites['risk_composite']['score']: .3f} - {composites['risk_composite']['interpretation']}")