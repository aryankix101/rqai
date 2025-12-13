import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REGIME_WEIGHTS = {
    "uptrend_low_vol": {
        "momentum": 0.35,
        "trend": 0.30,
        "relative_strength": 0.20,
        "volume": 0.10,
        "risk": 0.05
    },
    "uptrend_neutral_vol": {
        "momentum": 0.35,
        "trend": 0.25,
        "relative_strength": 0.20,
        "volume": 0.15,
        "risk": 0.05
    },
    "uptrend_high_vol": {
        "momentum": 0.30,
        "trend": 0.25,
        "relative_strength": 0.20,
        "volume": 0.15,
        "risk": 0.10  # Increase risk weight in high vol
    },
    
    # Downtrend regimes: Emphasize risk management
    "downtrend_low_vol": {
        "momentum": 0.20,
        "trend": 0.20,
        "relative_strength": 0.15,
        "volume": 0.10,
        "risk": 0.35  # Risk is most important
    },
    "downtrend_neutral_vol": {
        "momentum": 0.20,
        "trend": 0.15,
        "relative_strength": 0.15,
        "volume": 0.10,
        "risk": 0.40
    },
    "downtrend_high_vol": {
        "momentum": 0.15,
        "trend": 0.15,
        "relative_strength": 0.10,
        "volume": 0.10,
        "risk": 0.50  # Maximum risk focus
    },
    
    # Choppy/sideways regimes: Balanced approach
    "choppy_low_vol": {
        "momentum": 0.25,
        "trend": 0.20,
        "relative_strength": 0.25,
        "volume": 0.15,
        "risk": 0.15
    },
    "choppy_neutral_vol": {
        "momentum": 0.25,
        "trend": 0.25,
        "relative_strength": 0.25,
        "volume": 0.15,
        "risk": 0.10
    },
    "choppy_high_vol": {
        "momentum": 0.20,
        "trend": 0.20,
        "relative_strength": 0.20,
        "volume": 0.15,
        "risk": 0.25
    }
}

# Default weights if regime not recognized
DEFAULT_WEIGHTS = {
    "momentum": 0.30,
    "trend": 0.25,
    "relative_strength": 0.20,
    "volume": 0.15,
    "risk": 0.10
}


# =============================================================================
# Core Scoring Functions
# =============================================================================

def calculate_tier1_master_score(
    composite_scores: Dict[str, Dict[str, Any]],
    regime: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Calculate Tier-1 master deterministic score using regime-adjusted weights.
    
    Args:
        composite_scores: Dictionary with all composite score results
        regime: Market regime classification from regime_classifier
        verbose: If True, log detailed breakdown
    
    Returns:
        {
            "tier1_score": float,           # [-1, 1] master score
            "weights_used": dict,           # Weights applied
            "weighted_contributions": dict, # Each component's contribution
            "regime": str,                  # Regime used for weighting
            "interpretation": str,          # Human-readable
            "confidence": float             # [0, 1] confidence in signal
        }
    """
    # Extract composite scores
    momentum_score = composite_scores['momentum_composite']['score']
    trend_score = composite_scores['trend_composite']['score']
    rs_score = composite_scores['rs_composite']['score']
    volume_score = composite_scores['volume_composite']['score']
    risk_score = composite_scores['risk_composite']['score']
    
    # Get regime and select appropriate weights
    combined_regime = regime.get('combined_regime', 'unknown')
    weights = REGIME_WEIGHTS.get(combined_regime, DEFAULT_WEIGHTS)
    
    if verbose:
        logger.info(f"Using weights for regime: {combined_regime}")
        logger.info(f"Weights: {weights}")
    
    # Calculate weighted contributions
    # Note: risk and volume are [0,1], need to convert to contribution
    # For risk: high risk score = good, contributes positively
    # For volume: high volume score = good confirmation, contributes positively
    
    # Risk: Convert [0,1] to influence on signal
    # High risk score (good risk profile) allows full signal
    # Low risk score (bad risk profile) dampens signal
    risk_multiplier = 0.5 + (risk_score * 0.5)  # Range [0.5, 1.0]
    
    # Volume: Convert [0,1] to confirmation boost/penalty
    # High volume = boost, low volume = penalty
    volume_contribution = (volume_score - 0.5) * 2  # Convert to [-1, 1]
    
    # Calculate weighted contributions
    contributions = {
        "momentum": weights["momentum"] * momentum_score,
        "trend": weights["trend"] * trend_score,
        "relative_strength": weights["relative_strength"] * rs_score,
        "volume": weights["volume"] * volume_contribution,
        "risk_adjustment": 0.0  # Calculated below
    }
    
    # Sum base signal (before risk adjustment)
    base_signal = (
        contributions["momentum"] +
        contributions["trend"] +
        contributions["relative_strength"] +
        contributions["volume"]
    )
    
    # Apply risk multiplier
    tier1_score = base_signal * risk_multiplier
    contributions["risk_adjustment"] = tier1_score - base_signal
    
    # Clamp to [-1, 1]
    tier1_score = max(-1.0, min(1.0, tier1_score))
    
    # Calculate confidence based on:
    # 1. Signal strength (absolute value)
    # 2. Regime quality
    # 3. Volume confirmation
    # 4. Component agreement
    confidence = _calculate_confidence(
        tier1_score=tier1_score,
        regime_score=regime.get('regime_score', 0.5),
        volume_score=volume_score,
        composite_scores=composite_scores
    )
    
    # Interpretation
    interpretation = _generate_interpretation(tier1_score, confidence, combined_regime)
    
    result = {
        "tier1_score": tier1_score,
        "weights_used": weights,
        "weighted_contributions": contributions,
        "regime": combined_regime,
        "interpretation": interpretation,
        "confidence": confidence
    }
    
    if verbose:
        logger.info(f"Tier-1 Score: {tier1_score:+.3f}")
        logger.info(f"Confidence: {confidence:.3f}")
        logger.info(f"Interpretation: {interpretation}")
    
    return result


def _calculate_confidence(
    tier1_score: float,
    regime_score: float,
    volume_score: float,
    composite_scores: Dict[str, Dict[str, Any]]
) -> float:
    """
    Calculate confidence in the signal [0, 1].
    
    Factors:
    - Signal strength (higher absolute value = more confident)
    - Regime quality (good regime = more confident)
    - Volume confirmation (high volume = more confident)
    - Component agreement (aligned composites = more confident)
    """
    # Signal strength component [0, 1]
    signal_strength = abs(tier1_score)
    
    # Regime quality component [0, 1]
    regime_component = regime_score
    
    # Volume component [0, 1]
    volume_component = volume_score
    
    # Component agreement: Check if momentum and trend align
    momentum = composite_scores['momentum_composite']['score']
    trend = composite_scores['trend_composite']['score']
    rs = composite_scores['rs_composite']['score']
    
    # Agreement score: higher if components have same sign
    agreement = 0.5  # Start neutral
    
    if momentum > 0 and trend > 0 and rs > 0:
        agreement = 1.0  # All bullish
    elif momentum < 0 and trend < 0 and rs < 0:
        agreement = 0.9  # All bearish
    elif momentum > 0.3 and trend > 0.3:
        agreement = 0.8  # Momentum + trend bullish
    elif momentum < -0.3 and trend < -0.3:
        agreement = 0.7  # Momentum + trend bearish
    elif (momentum > 0) != (trend > 0):
        agreement = 0.3  # Divergence - low confidence
    
    # Weighted confidence
    confidence = (
        0.35 * signal_strength +
        0.25 * regime_component +
        0.20 * volume_component +
        0.20 * agreement
    )
    
    return max(0.0, min(1.0, confidence))


def _generate_interpretation(
    score: float,
    confidence: float,
    regime: str
) -> str:
    """Generate human-readable interpretation of the signal."""
    
    # Signal direction and strength
    if score > 0.5:
        signal = "STRONG BULLISH"
    elif score > 0.25:
        signal = "MODERATE BULLISH"
    elif score > -0.25:
        signal = "NEUTRAL"
    elif score > -0.5:
        signal = "MODERATE BEARISH"
    else:
        signal = "STRONG BEARISH"
    
    # Confidence qualifier
    if confidence > 0.75:
        conf_qual = "high confidence"
    elif confidence > 0.6:
        conf_qual = "good confidence"
    elif confidence > 0.4:
        conf_qual = "moderate confidence"
    else:
        conf_qual = "low confidence"
    
    return f"{signal} ({conf_qual}, {regime})"


# =============================================================================
# Batch Processing
# =============================================================================

def score_universe(
    indicators_dict: Dict[str, Dict[str, Any]],
    composites_dict: Dict[str, Dict[str, Any]],
    regimes_dict: Dict[str, Dict[str, Any]],
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate Tier-1 scores for multiple stocks.
    
    Args:
        indicators_dict: Raw indicators for each symbol
        composites_dict: Composite scores for each symbol
        regimes_dict: Regime classifications for each symbol
        verbose: If True, log results
    
    Returns:
        Dictionary mapping symbol -> tier1 score results
    """
    scores = {}
    
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info(f"TIER-1 DETERMINISTIC SCORING: Processing {len(composites_dict)} symbols")
        logger.info(f"{'='*80}\n")
    
    for symbol in composites_dict.keys():
        if symbol not in regimes_dict:
            logger.warning(f"Skipping {symbol}: missing regime data")
            continue
        
        tier1_result = calculate_tier1_master_score(
            composite_scores=composites_dict[symbol],
            regime=regimes_dict[symbol],
            verbose=False
        )
        
        scores[symbol] = tier1_result
        
        if verbose:
            score = tier1_result['tier1_score']
            conf = tier1_result['confidence']
            logger.info(f"{symbol}: {score:+.3f} (conf={conf:.2f}) - {tier1_result['interpretation']}")
    
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info(f"SCORING COMPLETE")
        logger.info(f"{'='*80}")
    
    return scores


def rank_by_score(
    scores_dict: Dict[str, Dict[str, Any]],
    min_confidence: float = 0.0
) -> list:
    """
    Rank stocks by Tier-1 score, filtered by minimum confidence.
    
    Args:
        scores_dict: Dictionary of tier1 score results
        min_confidence: Minimum confidence threshold
    
    Returns:
        List of (symbol, score, confidence) tuples, sorted by score descending
    """
    # Filter by confidence
    filtered = {
        symbol: result
        for symbol, result in scores_dict.items()
        if result['confidence'] >= min_confidence
    }
    
    # Sort by score (descending for longs, ascending for shorts)
    ranked = sorted(
        filtered.items(),
        key=lambda x: x[1]['tier1_score'],
        reverse=True
    )
    
    return [
        (symbol, result['tier1_score'], result['confidence'])
        for symbol, result in ranked
    ]


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import json
    from pathlib import Path
    from composite_scores import calculate_all_composites
    from regime_classifier import classify_market_regime
    
    output_dir = Path("./output")
    
    if not output_dir.exists():
        logger.error("Output directory not found.")
        exit(1)
    
    # Load all indicators
    indicators_dict = {}
    for json_file in output_dir.glob("*_technical_indicators.json"):
        with open(json_file) as f:
            data = json.load(f)
            symbol = data['id']['symbol']
            indicators_dict[symbol] = data
    
    if not indicators_dict:
        logger.error("No indicator files found.")
        exit(1)
    
    # Calculate composites for all
    logger.info("Calculating composite scores...")
    composites_dict = {}
    for symbol, indicators in indicators_dict.items():
        composites_dict[symbol] = calculate_all_composites(indicators)
    
    # Classify regimes for all
    logger.info("Classifying regimes...")
    regimes_dict = {}
    for symbol, indicators in indicators_dict.items():
        regimes_dict[symbol] = classify_market_regime(indicators)
    
    # Calculate Tier-1 scores
    scores_dict = score_universe(
        indicators_dict=indicators_dict,
        composites_dict=composites_dict,
        regimes_dict=regimes_dict,
        verbose=True
    )
    
    # Rank stocks
    print("\n" + "="*80)
    print("RANKED BY TIER-1 DETERMINISTIC SCORE")
    print("="*80)
    
    ranked = rank_by_score(scores_dict, min_confidence=0.0)
    
    print(f"\n{'Rank':<6} {'Symbol':<8} {'Score':<10} {'Confidence':<12} {'Signal'}")
    print("-" * 80)
    
    for i, (symbol, score, confidence) in enumerate(ranked, 1):
        signal_type = "LONG" if score > 0.25 else ("SHORT" if score < -0.25 else "NEUTRAL")
        print(f"{i:<6} {symbol:<8} {score:+.3f}     {confidence:.3f}        {signal_type}")
    
    print("="*80)
    
    # Summary statistics
    long_candidates = [s for s, sc, c in ranked if sc > 0.25 and c > 0.5]
    short_candidates = [s for s, sc, c in ranked if sc < -0.25 and c > 0.5]
    
    print(f"\nLONG candidates (score > 0.25, conf > 0.5): {', '.join(long_candidates) if long_candidates else 'None'}")
    print(f"SHORT candidates (score < -0.25, conf > 0.5): {', '.join(short_candidates) if short_candidates else 'None'}")
