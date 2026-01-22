import logging
from typing import Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LLM_WEIGHT = 1.0
DETERMINISTIC_WEIGHT = 0.0
LONG_THRESHOLD = 0.15
SHORT_THRESHOLD = -0.15
MIN_CONFIDENCE = 0.30
HIGH_CONFIDENCE_BOOST = 0.15
DIVERGENCE_PENALTY = 0.20
VOLATILITY_PENALTY_MAX = 0.30

def fuse_signals(
    tier1_result: Dict[str, Any],
    llm_result: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    LLM-primary signal fusion with deterministic gates, penalties, and confidence modifiers.
    
    Philosophy:
    - LLM generates base signal [-1, 1]
    - Deterministic logic acts as:
        1. Hard gates (liquidity, execution checks)
        2. Soft penalties (divergence, volatility)
        3. Confidence modifiers (boost/reduce based on agreement)
    
    Returns:
        {
            "signal_final": float,           # [-1, 1] final signal
            "confidence_final": float,       # [0, 1] overall confidence
            "decision": str,                 # "LONG", "SHORT", "NEUTRAL"
            "components": dict,              # Breakdown of signal sources
            "agreement": str,                # "aligned", "divergent", "mixed"
            "interpretation": str            # Human-readable
        }
    """
    # Extract component signals
    signal_deterministic = tier1_result['tier1_score']
    confidence_deterministic = tier1_result['confidence']
    regime = tier1_result.get('regime', 'unknown')
    
    signal_llm = llm_result.get('score', 0.0)
    confidence_llm = llm_result.get('confidence', 0.5)
    liquidity_ok = llm_result.get('liquidity_ok', True)
    min_exec_checks = llm_result.get('min_exec_checks', {})
    
    if verbose:
        logger.info(f"LLM: signal={signal_llm:+.3f}, conf={confidence_llm:.3f}")
        logger.info(f"Deterministic: signal={signal_deterministic:+.3f}, conf={confidence_deterministic:.3f}")
        logger.info(f"Regime: {regime}")
    
    # =========================================================================
    # LLM-PRIMARY FUSION: LLM as base, deterministic as gates/penalties
    # =========================================================================
    
    # Start with LLM signal as base
    base_signal = signal_llm * LLM_WEIGHT
    
    # Add deterministic component (if weight > 0)
    if DETERMINISTIC_WEIGHT > 0:
        base_signal += signal_deterministic * DETERMINISTIC_WEIGHT
    
    # HARD GATES: Force to neutral if critical filters fail
    if not liquidity_ok:
        if verbose:
            logger.info("GATE: Liquidity failed → forcing neutral")
        base_signal = 0.0
        confidence_llm = 0.1
    
    if min_exec_checks:
        if not min_exec_checks.get('min_dollar_volume_filter', True):
            if verbose:
                logger.info("GATE: Min dollar volume failed → forcing neutral")
            base_signal = 0.0
            confidence_llm = 0.1
    
    # Apply risk adjustment - high risk_score means LOW risk, so invert it
    # Risk score is already constructed where high score = low risk
    # But we want high risk to REDUCE the signal, so we subtract (1 - risk_score)
    risk_score = tier1_result.get('risk_score', 0.5)
    risk_penalty = (1.0 - risk_score) * 0.15  # High risk → larger penalty
    base_signal = base_signal * (1.0 - risk_penalty)
    
    if verbose and risk_penalty > 0.05:
        logger.info(f"PENALTY: Risk adjustment → reducing signal by {risk_penalty:.1%}")
    
    # Check for agreement/divergence
    agreement_type, agreement_score = _assess_agreement(
        signal_deterministic, signal_llm, confidence_deterministic, confidence_llm
    )
    
    # SOFT PENALTY: Divergence between LLM and deterministic
    divergence_penalty = 0.0
    opposite_signs = (signal_deterministic * signal_llm) < 0
    significant_deterministic = abs(signal_deterministic) > 0.1
    
    if agreement_type == "divergent" and (opposite_signs or significant_deterministic):
        divergence_penalty = DIVERGENCE_PENALTY
        base_signal *= (1.0 - divergence_penalty)
        if verbose:
            reason = "opposite signs" if opposite_signs else "large magnitude difference"
            logger.info(f"PENALTY: Divergence detected ({reason}) → reducing signal by {divergence_penalty:.1%}")
    
    # SOFT PENALTY: High volatility regime
    volatility_penalty = 0.0
    if 'high_vol' in regime:
        volatility_penalty = VOLATILITY_PENALTY_MAX * 0.5
        base_signal *= (1.0 - volatility_penalty)
        if verbose:
            logger.info(f"PENALTY: High volatility → reducing signal by {volatility_penalty:.1%}")
    
    # =========================================================================
    # CONFIDENCE TRACKING (METADATA ONLY - DOES NOT AFFECT SIGNAL)
    # =========================================================================
    # Confidence is tracked separately for post-hoc analysis.
    # It does NOT scale or modify the directional signal.
    # Signal = what we believe (direction/conviction)
    # Confidence = how certain we are (for later analysis/sizing if proven useful)
    
    base_confidence = confidence_llm
    
    # Track confidence modifiers WITHOUT affecting signal
    if confidence_llm > 0.75:
        confidence_boost = HIGH_CONFIDENCE_BOOST
        base_confidence = min(1.0, base_confidence + confidence_boost)
        if verbose:
            logger.info(f"CONFIDENCE: High LLM confidence → +{confidence_boost:.1%} (metadata only)")
    
    elif confidence_llm < MIN_CONFIDENCE:
        base_confidence *= 0.5
        if verbose:
            logger.info(f"CONFIDENCE: Low LLM confidence → -50% confidence (metadata only)")
    
    # Factor in deterministic confidence for agreement
    if agreement_type == "aligned":
        # Aligned signals boost confidence (metadata)
        conf_boost = 0.1 * agreement_score
        base_confidence = min(1.0, base_confidence + conf_boost)
        if verbose:
            logger.info(f"CONFIDENCE: Signals aligned → +{conf_boost:.2f} confidence (metadata only)")
    elif agreement_type == "divergent":
        # Divergence reduces confidence (metadata)
        base_confidence *= 0.7
        if verbose:
            logger.info(f"CONFIDENCE: Signals divergent → -30% confidence (metadata only)")
    
    signal_final = base_signal
    confidence_final = base_confidence
    
    # Make decision
    decision, decision_strength = _make_decision(signal_final, confidence_final)
    
    # Generate interpretation
    interpretation = _generate_fusion_interpretation(
        decision=decision,
        signal_final=signal_final,
        confidence_final=confidence_final,
        agreement_type=agreement_type,
        signal_deterministic=signal_deterministic,
        signal_llm=signal_llm,
        regime=regime
    )
    
    result = {
        "signal_final": signal_final,
        "confidence_final": confidence_final,
        "decision": decision,
        "decision_strength": decision_strength,
        "components": {
            "signal_llm": signal_llm,
            "confidence_llm": confidence_llm,
            "signal_deterministic": signal_deterministic,
            "confidence_deterministic": confidence_deterministic,
            "divergence_penalty": divergence_penalty,
            "volatility_penalty": volatility_penalty,
            "base_signal": base_signal
        },
        "agreement": agreement_type,
        "agreement_score": agreement_score,
        "regime": regime,
        "interpretation": interpretation
    }
    
    if verbose:
        logger.info(f"Final Signal: {signal_final:+.3f}")
        logger.info(f"Final Confidence: {confidence_final:.3f}")
        logger.info(f"Decision: {decision} ({decision_strength})")
    
    return result


def _assess_agreement(
    signal_det: float,
    signal_llm: float,
    conf_det: float,
    conf_llm: float
) -> Tuple[str, float]:
    """
    Assess agreement between deterministic and LLM signals.
    
    Returns:
        (agreement_type, agreement_score)
        - agreement_type: "aligned", "divergent", "mixed"
        - agreement_score: [0, 1] where 1 = perfect agreement
    """
    # Check if signals have same direction
    same_direction = (signal_det * signal_llm) > 0
    
    # Calculate absolute difference
    diff = abs(signal_det - signal_llm)
    
    # Normalized agreement score [0, 1]
    # 0 difference = 1.0 score, 2.0 difference = 0.0 score
    agreement_score = max(0.0, 1.0 - (diff / 2.0))
    
    # Determine agreement type
    if same_direction and diff < 0.3:
        agreement_type = "aligned"
        # Boost score if both are confident
        if conf_det > 0.6 and conf_llm > 0.6:
            agreement_score = min(1.0, agreement_score + 0.2)
    elif same_direction and diff < 0.6:
        agreement_type = "mixed"
    else:
        agreement_type = "divergent"
        # Penalize divergence
        agreement_score *= 0.5
    
    return agreement_type, agreement_score


def _adjust_confidence(
    base_confidence: float,
    agreement_score: float,
    agreement_type: str
) -> float:
    """
    Adjust confidence based on signal agreement.
    
    Strong agreement = confidence boost
    Divergence = confidence penalty
    """
    adjusted = base_confidence
    
    if agreement_type == "aligned":
        # Boost confidence when signals align
        adjusted += HIGH_CONFIDENCE_BOOST * agreement_score
    elif agreement_type == "divergent":
        # Penalize confidence when signals diverge
        adjusted -= DIVERGENCE_PENALTY
    else:
        # Slight penalty for mixed signals
        adjusted -= 0.05
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, adjusted))


def _make_decision(
    signal: float,
    confidence: float
) -> Tuple[str, str]:
    """
    Make trading decision based on signal and confidence.
    
    Returns:
        (decision, strength)
        - decision: "LONG", "SHORT", "NEUTRAL"
        - strength: "strong", "moderate", "weak"
    """
    # Check confidence threshold first
    if confidence < MIN_CONFIDENCE:
        return "NEUTRAL", "low_confidence"
    
    # Determine direction and strength
    abs_signal = abs(signal)
    
    if signal > LONG_THRESHOLD:
        decision = "LONG"
        if abs_signal > 0.6 and confidence > 0.7:
            strength = "strong"
        elif abs_signal > 0.4 and confidence > 0.6:
            strength = "moderate"
        else:
            strength = "weak"
    
    elif signal < SHORT_THRESHOLD:
        decision = "SHORT"
        if abs_signal > 0.6 and confidence > 0.7:
            strength = "strong"
        elif abs_signal > 0.4 and confidence > 0.6:
            strength = "moderate"
        else:
            strength = "weak"
    
    else:
        decision = "NEUTRAL"
        strength = "neutral_zone"
    
    return decision, strength


def _generate_fusion_interpretation(
    decision: str,
    signal_final: float,
    confidence_final: float,
    agreement_type: str,
    signal_deterministic: float,
    signal_llm: float,
    regime: str = "unknown"
) -> str:
    """Generate human-readable interpretation of fused signal."""
    
    # Build interpretation string
    parts = []
    
    # Decision
    parts.append(f"Decision: {decision}")
    
    # Signal strength
    abs_signal = abs(signal_final)
    if abs_signal > 0.6:
        signal_strength = "very strong"
    elif abs_signal > 0.4:
        signal_strength = "strong"
    elif abs_signal > 0.25:
        signal_strength = "moderate"
    else:
        signal_strength = "weak"
    
    parts.append(f"Signal: {signal_strength} ({signal_final:+.3f})")
    
    # Confidence
    if confidence_final > 0.75:
        conf_desc = "high confidence"
    elif confidence_final > 0.6:
        conf_desc = "good confidence"
    elif confidence_final > 0.4:
        conf_desc = "moderate confidence"
    else:
        conf_desc = "low confidence"
    
    parts.append(f"Confidence: {conf_desc} ({confidence_final:.3f})")
    
    # Agreement
    parts.append(f"Agreement: {agreement_type}")
    
    # Regime
    parts.append(f"Regime: {regime}")
    
    return " | ".join(parts)


# =============================================================================
# Batch Processing
# =============================================================================

def fuse_universe_signals(
    tier1_scores: Dict[str, Dict[str, Any]],
    llm_results: Dict[str, Dict[str, Any]],
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Fuse signals for multiple stocks.
    
    Args:
        tier1_scores: Tier-1 deterministic scores for each symbol
        llm_results: LLM verdicts for each symbol
        verbose: If True, log results
    
    Returns:
        Dictionary mapping symbol -> fused signal result
    """
    fused_signals = {}
    
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info(f"SIGNAL FUSION: Processing {len(tier1_scores)} symbols")
        logger.info(f"{'='*80}\n")
    
    for symbol in tier1_scores.keys():
        if symbol not in llm_results:
            logger.warning(f"Skipping {symbol}: missing LLM result")
            continue
        
        fused = fuse_signals(
            tier1_result=tier1_scores[symbol],
            llm_result=llm_results[symbol],
            verbose=False
        )
        
        fused['symbol'] = symbol
        fused_signals[symbol] = fused
        
        if verbose:
            decision = fused['decision']
            signal = fused['signal_final']
            conf = fused['confidence_final']
            agreement = fused['agreement']
            
            # Format with emoji
            emoji = "🟢" if decision == "LONG" else ("🔴" if decision == "SHORT" else "⚪")
            
            logger.info(f"{emoji} {symbol}: {decision} | Signal: {signal:+.3f} | Conf: {conf:.2f} | {agreement}")
    
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info(f"FUSION COMPLETE")
        logger.info(f"{'='*80}")
    
    return fused_signals


def filter_and_rank_signals(
    fused_signals: Dict[str, Dict[str, Any]],
    min_confidence: float = MIN_CONFIDENCE,
    min_signal_strength: float = 0.3
) -> Dict[str, list]:
    """
    Filter and rank signals by confidence and strength.
    
    Args:
        fused_signals: Dictionary of fused signal results
        min_confidence: Minimum confidence threshold
        min_signal_strength: Minimum absolute signal strength
    
    Returns:
        {
            "longs": [(symbol, signal, confidence), ...],  
            "shorts": [(symbol, signal, confidence), ...], 
            "neutral": [(symbol, signal, confidence), ...]
        }
    """
    longs = []
    shorts = []
    neutral = []
    
    for symbol, result in fused_signals.items():
        signal = result['signal_final']
        confidence = result['confidence_final']
        decision = result['decision']
        
        # Filter by confidence and strength
        if confidence < min_confidence:
            continue
        
        if abs(signal) < min_signal_strength and decision != "NEUTRAL":
            continue
        
        entry = (symbol, signal, confidence)
        
        if decision == "LONG":
            longs.append(entry)
        elif decision == "SHORT":
            shorts.append(entry)
        else:
            neutral.append(entry)
    
    # Sort longs by signal descending (best first)
    longs.sort(key=lambda x: x[1], reverse=True)
    
    # Sort shorts by signal ascending (most negative first)
    shorts.sort(key=lambda x: x[1])
    
    return {
        "longs": longs,
        "shorts": shorts,
        "neutral": neutral
    }

