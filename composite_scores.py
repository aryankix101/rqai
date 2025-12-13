import logging
from typing import Dict, Any, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize(value: float, min_val: float, max_val: float) -> float:
    if value is None:
        return 0.0
    if max_val == min_val:
        return 0.0
    normalized = (value - min_val) / (max_val - min_val) * 2 - 1
    return np.clip(normalized, -1.0, 1.0)

def clip(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    if value is None:
        return 0.0
    return max(min_val, min(max_val, value))


def safe_get(data: Dict, key: str, default: float = 0.0) -> float:
    """Safely get value from dict with default."""
    val = data.get(key, default)
    return val if val is not None else default


# =============================================================================
# Momentum Composite Score
# =============================================================================

def calculate_momentum_composite(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate momentum composite score [-1, 1].
    
    Inputs:
    - mom_1m, mom_3m, mom_6m, mom_12m (price momentum)
    - macd_histogram (momentum direction)
    - rsi_14d (momentum exhaustion check)
    
    Output:
    - Positive score = bullish momentum
    - Negative score = bearish momentum
    
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
    
    # Get momentum values
    mom_1m = safe_get(mom, 'mom_1m')
    mom_3m = safe_get(mom, 'mom_3m')
    mom_6m = safe_get(mom, 'mom_6m')
    mom_12m = safe_get(mom, 'mom_12m')
    
    # Normalize momentum returns to [-1, 1]
    # Expected ranges: 1m: ±10%, 3m: ±20%, 6m: ±30%, 12m: ±50%
    mom_1m_norm = normalize(mom_1m, -0.10, 0.10)
    mom_3m_norm = normalize(mom_3m, -0.20, 0.20)
    mom_6m_norm = normalize(mom_6m, -0.30, 0.30)
    mom_12m_norm = normalize(mom_12m, -0.50, 0.50)
    
    # Weighted average (emphasize 3m most)
    mom_score = (
        0.40 * mom_3m_norm +   # 3-month most important
        0.30 * mom_1m_norm +   # Recent momentum
        0.20 * mom_6m_norm +   # Intermediate
        0.10 * mom_12m_norm    # Long-term context
    )
    
    # MACD confirmation
    macd_hist = safe_get(trend, 'macd_histogram')
    # Normalize MACD histogram (typical range ±5)
    macd_signal = normalize(macd_hist, -5.0, 5.0)
    
    # RSI moderation (penalize extremes)
    rsi = safe_get(trend, 'rsi_14d', 50.0)
    rsi_mod = 0.0
    if rsi > 70:
        # Overbought - slight penalty
        rsi_mod = -0.15 * ((rsi - 70) / 30)
    elif rsi < 30:
        # Oversold - slight boost
        rsi_mod = 0.15 * ((30 - rsi) / 30)
    rsi_mod = clip(rsi_mod, -0.2, 0.2)
    
    # Combine components
    final_score = clip(
        0.70 * mom_score +      # 70% raw momentum
        0.20 * macd_signal +    # 20% MACD confirmation
        0.10 * rsi_mod          # 10% RSI moderation
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
    - Price vs MA50/MA200 (moving average structure)
    - ADX (trend strength)
    - higher_highs_lows (price structure)
    - Golden/death cross events
    
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
    
    # MA structure score
    price_above_ma50 = trend.get('price_above_ma50')
    price_above_ma200 = trend.get('price_above_ma200')
    
    ma_score = 0.0
    if price_above_ma50 and price_above_ma200:
        ma_score = 1.0  # Bullish structure
    elif not price_above_ma50 and not price_above_ma200:
        ma_score = -1.0  # Bearish structure
    elif price_above_ma200:
        ma_score = 0.5  # Mixed but leaning bullish
    else:
        ma_score = -0.5  # Mixed but leaning bearish
    
    # Golden/Death cross boost
    golden_cross = trend.get('golden_cross_recent', False)
    death_cross = trend.get('death_cross_recent', False)
    
    cross_boost = 0.0
    if golden_cross:
        cross_boost = 0.3
    elif death_cross:
        cross_boost = -0.3
    
    # ADX trend strength multiplier
    adx = safe_get(trend, 'adx_14d', 20.0)
    # ADX: 0-20 (weak), 20-25 (moderate), 25+ (strong)
    adx_multiplier = normalize(adx, 0, 40)
    adx_multiplier = (adx_multiplier + 1) / 2  # Convert to [0, 1] range
    adx_multiplier = max(0.3, adx_multiplier)   # Min 30% weight
    
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
    final_score = clip(
        base_trend * adx_multiplier + cross_boost
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
            "adx_multiplier": adx_multiplier,
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
    
    # Relative performance vs S&P 500
    vs_sp500_3m = safe_get(rel_perf, 'vs_sp500_3m')
    vs_sp500_6m = safe_get(rel_perf, 'vs_sp500_6m')
    vs_sp500_12m = safe_get(rel_perf, 'vs_sp500_12m')
    
    # Normalize (±20% relative outperformance range)
    rs_3m_norm = normalize(vs_sp500_3m, -0.20, 0.20)
    rs_6m_norm = normalize(vs_sp500_6m, -0.30, 0.30)
    rs_12m_norm = normalize(vs_sp500_12m, -0.50, 0.50)
    
    # Weighted average (emphasize recent)
    rs_score = (
        0.50 * rs_3m_norm +
        0.30 * rs_6m_norm +
        0.20 * rs_12m_norm
    )
    
    # Relative strength trend bonus
    rs_trend = rel_perf.get('rel_strength_trend', 'flat')
    trend_bonus = 0.0
    if rs_trend == 'rising':
        trend_bonus = 0.2
    elif rs_trend == 'falling':
        trend_bonus = -0.2
    
    # Capture ratios (if available)
    up_capture = safe_get(risk, 'up_capture_1y', 1.0)
    down_capture = safe_get(risk, 'down_capture_1y', 1.0)
    
    # Ideal: up_capture > 1.0, down_capture < 1.0
    capture_score = 0.0
    if up_capture > 0 and down_capture > 0:
        # Ratio of up/down capture (>1 is good)
        capture_ratio = up_capture / down_capture
        capture_score = normalize(np.log(capture_ratio), -0.5, 0.5)
        capture_score *= 0.15  # Weight
    
    # Combine
    final_score = clip(rs_score + trend_bonus + capture_score)
    
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
            "capture_ratio_score": capture_score
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
    
    # Volume spike (moderate spike is good, extreme is suspicious)
    vol_spike_z = safe_get(vol_liq, 'volume_spike_z')
    spike_score = 0.5
    if 0 < vol_spike_z < 2:
        spike_score = 0.8  # Healthy increase
    elif 2 <= vol_spike_z < 3:
        spike_score = 0.6  # High but acceptable
    elif vol_spike_z >= 3:
        spike_score = 0.3  # Suspiciously high
    elif -1 < vol_spike_z < 0:
        spike_score = 0.6  # Slightly below average
    elif vol_spike_z <= -1:
        spike_score = 0.2  # Weak volume
    
    # Check for volume-price confirmation
    # If price momentum is positive, volume should confirm (OBV/AD up)
    mom_3m = safe_get(momentum, 'mom_3m')
    confirmation_bonus = 0.0
    if mom_3m > 0.05 and obv_trend == 'up' and ad_trend == 'up':
        confirmation_bonus = 0.2  # Strong confirmation
    elif mom_3m < -0.05 and obv_trend == 'down' and ad_trend == 'down':
        confirmation_bonus = 0.1  # Bearish confirmation (less weight)
    elif mom_3m > 0.05 and (obv_trend == 'down' or ad_trend == 'down'):
        confirmation_bonus = -0.2  # Bearish divergence
    elif mom_3m < -0.05 and (obv_trend == 'up' or ad_trend == 'up'):
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
    
    # Sharpe ratio (normalize 0-2 range to 0-1)
    sharpe = safe_get(risk, 'sharpe_1y')
    sharpe_score = normalize(sharpe, -1.0, 2.0)
    sharpe_score = (sharpe_score + 1) / 2  # Convert to [0, 1]
    
    # Sortino ratio (normalize 0-3 range)
    sortino = safe_get(risk, 'sortino_1y')
    sortino_score = normalize(sortino, -1.0, 3.0)
    sortino_score = (sortino_score + 1) / 2  # Convert to [0, 1]
    
    # Max drawdown (lower is better)
    max_dd = safe_get(risk, 'max_drawdown_1y', 0.3)
    # Normalize 0-50% range (inverted)
    dd_score = 1.0 - normalize(max_dd, 0.0, 0.5)
    dd_score = max(0.0, dd_score)
    
    # Volatility (moderate is best)
    realized_vol = safe_get(vol, 'realized_vol_252d', 0.3)
    # Penalize both low (<15%) and high (>50%) vol
    if 0.15 <= realized_vol <= 0.35:
        vol_score = 1.0  # Ideal range
    elif realized_vol < 0.15:
        vol_score = 0.6  # Too low (limited opportunity)
    elif realized_vol < 0.50:
        vol_score = 0.7  # Moderate high
    else:
        vol_score = 0.3  # Too high (risky)
    
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
        "risk_composite": calculate_risk_composite(indicators)
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