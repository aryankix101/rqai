import logging
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HIGH_VOL_THRESHOLD = 0.35
LOW_VOL_THRESHOLD = 0.15
STRONG_TREND_ADX = 25
WEAK_TREND_ADX = 20
MA_BUFFER = 0.02

def classify_volatility_regime(indicators: Dict[str, Any]) -> Dict[str, Any]:
    vol = indicators.get('volatility', {})
    
    realized_vol_252d = vol.get('realized_vol_252d')
    vol_ratio = vol.get('vol_ratio_21v252')
    
    # Check for unit mismatch: if vol > 2.0, it's probably in percent units
    if realized_vol_252d is not None and realized_vol_252d > 2.0:
        logger.warning(f"realized_vol_252d = {realized_vol_252d:.2f} > 2.0, assuming percent units, dividing by 100")
        realized_vol_252d = realized_vol_252d / 100.0
    
    if realized_vol_252d is None:
        regime = "neutral"
        description = "Unknown (missing volatility data)"
    elif realized_vol_252d > HIGH_VOL_THRESHOLD:
        regime = "high"
        description = f"High volatility ({realized_vol_252d*100:.1f}% annualized)"
    elif realized_vol_252d < LOW_VOL_THRESHOLD:
        regime = "low"
        description = f"Low volatility ({realized_vol_252d*100:.1f}% annualized)"
    else:
        regime = "neutral"
        description = f"Normal volatility ({realized_vol_252d*100:.1f}% annualized)"
    
    vol_spiking = False
    if vol_ratio is not None and vol_ratio > 1.2:
        vol_spiking = True
        description += " [SPIKE: Recent vol elevated]"
    
    return {
        "regime": regime,
        "realized_vol_252d": realized_vol_252d,
        "vol_ratio_21v252": vol_ratio,
        "vol_spiking": vol_spiking,
        "description": description
    }


def classify_trend_regime(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify trend regime: uptrend, downtrend, or choppy.
    
    Uses multiple signals:
    - Price vs MA200
    - ADX (trend strength)
    - Higher highs/lower lows structure
    - MACD alignment
    
    Args:
        indicators: Complete technical indicators dictionary
    
    Returns:
        {
            "regime": str,                    # "uptrend", "downtrend", "choppy"
            "trend_strength": str,            # "strong", "weak", "none"
            "adx": float,                     # ADX value
            "price_structure": str,           # From indicators
            "ma_alignment": str,              # "bullish", "bearish", "neutral"
            "description": str                # Human-readable description
        }
    """
    trend = indicators.get('trend', {})
    
    price_above_ma75 = trend.get('price_above_ma75')
    price_above_ma200 = trend.get('price_above_ma200')
    adx = trend.get('adx_14d')
    higher_highs_lows = trend.get('higher_highs_lows', 'mixed')
    macd_histogram = trend.get('macd_histogram', 0.0)
    
    if adx is None:
        trend_strength = "unknown"
    elif adx > STRONG_TREND_ADX:
        trend_strength = "strong"
    elif adx > WEAK_TREND_ADX:
        trend_strength = "moderate"
    else:
        trend_strength = "weak"
    
    # Determine MA alignment with buffer logic
    if price_above_ma75 is None or price_above_ma200 is None:
        # If either MA is in neutral zone, consider mixed
        ma_alignment = "mixed"
    elif price_above_ma75 and price_above_ma200:
        ma_alignment = "bullish"
    elif not price_above_ma75 and not price_above_ma200:
        ma_alignment = "bearish"
    else:
        ma_alignment = "mixed"
    
    regime = _determine_trend_regime(
        adx=adx,
        price_above_ma200=price_above_ma200,
        higher_highs_lows=higher_highs_lows,
        ma_alignment=ma_alignment,
        macd_histogram=macd_histogram,
        trend_strength=trend_strength
    )
    
    description = _generate_trend_description(
        regime=regime,
        trend_strength=trend_strength,
        adx=adx,
        ma_alignment=ma_alignment,
        higher_highs_lows=higher_highs_lows
    )
    
    return {
        "regime": regime,
        "trend_strength": trend_strength,
        "adx": adx,
        "price_structure": higher_highs_lows,
        "ma_alignment": ma_alignment,
        "description": description
    }


def _determine_trend_regime(
    adx: float,
    price_above_ma200: bool,
    higher_highs_lows: str,
    ma_alignment: str,
    macd_histogram: float,
    trend_strength: str
) -> str:
    """
    Internal logic to determine trend regime from multiple signals.
    """
    # Strong trend scenario
    if trend_strength == "strong":
        # Bullish trend
        if ma_alignment == "bullish" and higher_highs_lows == "uptrend":
            return "uptrend"
        # Bearish trend
        elif ma_alignment == "bearish" and higher_highs_lows == "downtrend":
            return "downtrend"
    
    # Moderate trend scenario
    if trend_strength == "moderate":
        # Need at least 2 signals aligned
        bullish_signals = 0
        bearish_signals = 0
        
        if price_above_ma200:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if higher_highs_lows == "uptrend":
            bullish_signals += 1
        elif higher_highs_lows == "downtrend":
            bearish_signals += 1
        
        if macd_histogram is not None:
            if macd_histogram > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        if bullish_signals >= 2:
            return "uptrend"
        elif bearish_signals >= 2:
            return "downtrend"
    
    # Default to choppy if no clear trend
    return "choppy"


def _generate_trend_description(
    regime: str,
    trend_strength: str,
    adx: float,
    ma_alignment: str,
    higher_highs_lows: str
) -> str:
    """
    Generate human-readable trend description.
    """
    strength_desc = f"{trend_strength} trend" if adx else "unknown trend strength"
    
    if regime == "uptrend":
        desc = f"Uptrend ({strength_desc}, ADX={adx:.1f})" if adx else "Uptrend"
        if ma_alignment == "bullish":
            desc += " - Price above key MAs"
    elif regime == "downtrend":
        desc = f"Downtrend ({strength_desc}, ADX={adx:.1f})" if adx else "Downtrend"
        if ma_alignment == "bearish":
            desc += " - Price below key MAs"
    else:
        desc = f"Choppy/Sideways (ADX={adx:.1f})" if adx else "Choppy/Sideways"
        if higher_highs_lows == "mixed":
            desc += " - No clear direction"
    
    return desc


def classify_market_regime(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify both volatility and trend regimes.
    
    This is the main entry point for regime classification.
    
    Args:
        indicators: Complete technical indicators dictionary
    
    Returns:
        {
            "symbol": str,
            "volatility_regime": dict,        # Full volatility regime info
            "trend_regime": dict,             # Full trend regime info
            "combined_regime": str,           # e.g., "uptrend_high_vol"
            "regime_score": float,            # Quality score [0, 1]
            "description": str                # Full description
        }
    """
    symbol = indicators.get('id', {}).get('symbol', 'UNKNOWN')
    
    vol_regime = classify_volatility_regime(indicators)
    
    trend_regime = classify_trend_regime(indicators)
    
    combined_regime = f"{trend_regime['regime']}_{vol_regime['regime']}_vol"
    
    regime_score = _calculate_regime_score(vol_regime, trend_regime)
    
    description = f"{trend_regime['description']} | {vol_regime['description']}"
    
    return {
        "symbol": symbol,
        "volatility_regime": vol_regime,
        "trend_regime": trend_regime,
        "combined_regime": combined_regime,
        "regime_score": regime_score,
        "description": description
    }


def _calculate_regime_score(vol_regime: Dict, trend_regime: Dict) -> float:
    """
    Calculate regime quality score [0, 1].
    
    Higher score = better trading conditions:
    - Strong trends are better than choppy
    - Moderate volatility is better than extreme
    - No volatility spikes is better
    """
    score = 0.5
    

    if trend_regime['trend_strength'] == "strong":
        score += 0.3
    elif trend_regime['trend_strength'] == "moderate":
        score += 0.15
    elif trend_regime['trend_strength'] == "weak":
        score -= 0.15
    
    if trend_regime['regime'] != "choppy":
        score += 0.1
    else:
        score -= 0.1
    
    vol_reg = vol_regime['regime']
    if vol_reg == "neutral":
        score += 0.1
    elif vol_reg == "low":
        score -= 0.05
    elif vol_reg == "high":
        score -= 0.15
    
    if vol_regime.get('vol_spiking'):
        score -= 0.1
    
    return max(0.0, min(1.0, score))


def classify_universe_regimes(
    indicators_dict: Dict[str, Dict[str, Any]],
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Classify regimes for multiple stocks.
    
    Args:
        indicators_dict: Dictionary mapping symbol -> indicators
        verbose: If True, log results
    
    Returns:
        Dictionary mapping symbol -> regime classification
    """
    regime_results = {}
    
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info(f"REGIME CLASSIFICATION: Processing {len(indicators_dict)} symbols")
        logger.info(f"{'='*80}\n")
    
    for symbol, indicators in indicators_dict.items():
        regime = classify_market_regime(indicators)
        regime_results[symbol] = regime
        
        if verbose:
            logger.info(f"{symbol}:")
            logger.info(f"  Trend: {regime['trend_regime']['regime']} ({regime['trend_regime']['trend_strength']})")
            logger.info(f"  Vol:   {regime['volatility_regime']['regime']}")
            logger.info(f"  Score: {regime['regime_score']:.2f}/1.0")
            logger.info(f"  {regime['description']}\n")
    
    if verbose:
        trend_counts = {}
        vol_counts = {}
        for result in regime_results.values():
            trend = result['trend_regime']['regime']
            vol = result['volatility_regime']['regime']
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
            vol_counts[vol] = vol_counts.get(vol, 0) + 1
        
        logger.info(f"{'='*80}")
        logger.info(f"REGIME SUMMARY:")
        logger.info(f"  Trend: {dict(trend_counts)}")
        logger.info(f"  Vol:   {dict(vol_counts)}")
        logger.info(f"{'='*80}")
    
    return regime_results


if __name__ == "__main__":
    import json
    from pathlib import Path
    
    output_dir = Path("./output")
    
    if not output_dir.exists():
        logger.error("Output directory not found. Run calculate_all_indicators.py first.")
        exit(1)
    
    indicators_dict = {}
    for json_file in output_dir.glob("*_technical_indicators.json"):
        with open(json_file) as f:
            data = json.load(f)
            symbol = data['id']['symbol']
            indicators_dict[symbol] = data
    
    if not indicators_dict:
        logger.error("No indicator JSON files found in output directory.")
        exit(1)
    
    regime_results = classify_universe_regimes(indicators_dict, verbose=True)
    
    sorted_by_score = sorted(
        regime_results.items(),
        key=lambda x: x[1]['regime_score'],
        reverse=True
    )
    
    print("\n" + "="*80)
    print("RANKED BY REGIME QUALITY")
    print("="*80)
    for symbol, result in sorted_by_score:
        print(f"{symbol}: {result['regime_score']:.2f} - {result['combined_regime']}")
    print("="*80)
