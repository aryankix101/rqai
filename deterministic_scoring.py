import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REGIME_WEIGHTS = {
    "uptrend_low_vol": {"momentum": 0.35, "trend": 0.30, "relative_strength": 0.20, "volume": 0.10, "risk": 0.05},
    "uptrend_neutral_vol": {"momentum": 0.35, "trend": 0.25, "relative_strength": 0.20, "volume": 0.15, "risk": 0.05},
    "uptrend_high_vol": {"momentum": 0.30, "trend": 0.25, "relative_strength": 0.20, "volume": 0.15, "risk": 0.10},

    "downtrend_low_vol": {"momentum": 0.20, "trend": 0.20, "relative_strength": 0.15, "volume": 0.10, "risk": 0.35},
    "downtrend_neutral_vol": {"momentum": 0.20, "trend": 0.15, "relative_strength": 0.15, "volume": 0.10, "risk": 0.40},
    "downtrend_high_vol": {"momentum": 0.15, "trend": 0.15, "relative_strength": 0.10, "volume": 0.10, "risk": 0.50},

    "choppy_low_vol": {"momentum": 0.25, "trend": 0.20, "relative_strength": 0.25, "volume": 0.15, "risk": 0.15},
    "choppy_neutral_vol": {"momentum": 0.25, "trend": 0.25, "relative_strength": 0.25, "volume": 0.15, "risk": 0.10},
    "choppy_high_vol": {"momentum": 0.20, "trend": 0.20, "relative_strength": 0.20, "volume": 0.15, "risk": 0.25},
}

DEFAULT_WEIGHTS = {"momentum": 0.30, "trend": 0.25, "relative_strength": 0.20, "volume": 0.15, "risk": 0.10}


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _safe_get(d: Dict[str, Any], k: str, default: Any = 0.0) -> Any:
    v = d.get(k, default)
    return default if v is None else v


def _renorm_weights(weights: Dict[str, float], keys: Dict[str, bool]) -> Dict[str, float]:
    """
    Renormalize a weight dict given a mask of which keys are enabled (True) vs disabled (False).
    Any disabled key weight is set to 0 and remaining weights are scaled to sum to 1.
    """
    out = dict(weights)
    for k, enabled in keys.items():
        if not enabled and k in out:
            out[k] = 0.0
    s = sum(out.values())
    if s <= 1e-12:
        return out
    return {k: (v / s) for k, v in out.items()}


def _compute_quality(
    risk_score_0_1: float,
    volume_score_0_1: float,
    regime_score_0_1: float,
) -> float:
    # Quality is NOT directional. It should represent "permission to size up", not a sign.
    # Weighted blend, clamped to [0,1].
    q = 0.45 * risk_score_0_1 + 0.25 * volume_score_0_1 + 0.30 * regime_score_0_1
    return _clamp(q, 0.0, 1.0)


def _size_multiplier_from_quality(q: float, lo: float = 0.55, hi: float = 1.35) -> float:
    # Nonlinear map q∈[0,1] -> [lo, hi] with q^1.3 to emphasize higher quality
    q_clamped = _clamp(q, 0.0, 1.0)
    return lo + (hi - lo) * (q_clamped ** 1.3)


def calculate_tier1_master_score(
    composite_scores: Dict[str, Dict[str, Any]],
    regime: Dict[str, Any],
    verbose: bool = False,
    # Gates (risk as gate, not multiplier)
    min_risk_quality: float = 0.35,
    min_volume_quality: float = 0.35,
    # If gate fails, we don't flip direction — we cap sizing and confidence.
    gated_size_cap: float = 0.85,
) -> Dict[str, Any]:
    """
    Tier-1 deterministic output split into:
      - tier1_direction: [-1,1] directional context (mom/trend/RS later)
      - tier1_quality: [0,1] environment/permission (risk + volume + regime)
      - size_multiplier: sizing scalar derived from quality (and gates)
    """

    momentum_score = float(_safe_get(composite_scores.get("momentum_composite", {}), "score", 0.0))
    trend_score = float(_safe_get(composite_scores.get("trend_composite", {}), "score", 0.0))
    rs_score = float(_safe_get(composite_scores.get("rs_composite", {}), "score", 0.0))
    volume_score = float(_safe_get(composite_scores.get("volume_composite", {}), "score", 0.5))  # [0,1]
    risk_score = float(_safe_get(composite_scores.get("risk_composite", {}), "score", 0.5))      # [0,1]

    combined_regime = regime.get("combined_regime", "unknown")
    regime_score = float(_safe_get(regime, "regime_score", 0.5))  # [0,1]
    weights = REGIME_WEIGHTS.get(combined_regime, DEFAULT_WEIGHTS)

    # RS is not implemented yet in your pipeline (currently 0 / dead). Don’t let it dilute direction.
    # Check the "available" flag from RS composite
    rs_composite = composite_scores.get("rs_composite", {})
    rs_available = rs_composite.get("available", False)

    # Direction weights: only directional components should contribute to tier1_direction.
    # Volume and risk are quality/permission; keep them OUT of direction.
    dir_weights_raw = {
        "momentum": float(weights.get("momentum", 0.0)),
        "trend": float(weights.get("trend", 0.0)),
        "relative_strength": float(weights.get("relative_strength", 0.0)),
    }
    dir_weights = _renorm_weights(
        dir_weights_raw,
        keys={"relative_strength": rs_available, "momentum": True, "trend": True},
    )

    tier1_direction = (
        dir_weights["momentum"] * momentum_score +
        dir_weights["trend"] * trend_score +
        dir_weights["relative_strength"] * (rs_score if rs_available else 0.0)
    )
    tier1_direction = _clamp(tier1_direction, -1.0, 1.0)

    # Quality is permission to size/act, NOT a directional contribution.
    tier1_quality = _compute_quality(risk_score_0_1=risk_score, volume_score_0_1=volume_score, regime_score_0_1=regime_score)

    gates = {
        "risk_ok": risk_score >= min_risk_quality,
        "volume_ok": volume_score >= min_volume_quality,
        "rs_available": rs_available,
    }

    size_multiplier = _size_multiplier_from_quality(tier1_quality)
    if not gates["risk_ok"] or not gates["volume_ok"]:
        size_multiplier = min(size_multiplier, gated_size_cap)

    # Keep the legacy "tier1_score" as the directional score (so it can be used for sign agreement).
    tier1_score = tier1_direction

    contributions = {
        "direction": {
            "momentum": dir_weights["momentum"] * momentum_score,
            "trend": dir_weights["trend"] * trend_score,
            "relative_strength": dir_weights["relative_strength"] * (rs_score if rs_available else 0.0),
        },
        "quality": {
            "risk_score_0_1": risk_score,
            "volume_score_0_1": volume_score,
            "regime_score_0_1": regime_score,
        },
    }

    confidence = _calculate_confidence(
        tier1_direction=tier1_direction,
        tier1_quality=tier1_quality,
        composite_scores=composite_scores,
        rs_available=rs_available,
    )

    # If gates fail, confidence should also be capped (permission failure).
    if not gates["risk_ok"] or not gates["volume_ok"]:
        confidence = min(confidence, 0.55)

    interpretation = _generate_interpretation(
        score=tier1_score,
        confidence=confidence,
        regime=combined_regime,
        tier1_quality=tier1_quality,
        size_multiplier=size_multiplier,
        gates=gates,
    )

    result = {
        "tier1_score": tier1_score,                 # [-1,1] directional (legacy key)
        "tier1_direction": tier1_direction,         # [-1,1] explicit
        "tier1_quality": tier1_quality,             # [0,1]
        "size_multiplier": size_multiplier,         # scalar for sizing
        "gates": gates,
        "weights_used": weights,
        "direction_weights_used": dir_weights,
        "contributions": contributions,
        "regime": combined_regime,
        "interpretation": interpretation,
        "confidence": confidence,
    }

    if verbose:
        logger.info(f"Using weights for regime: {combined_regime}")
        logger.info(f"Raw regime weights: {weights}")
        logger.info(f"Directional weights used: {dir_weights}")
        logger.info(f"Tier-1 Direction: {tier1_direction:+.3f}")
        logger.info(f"Tier-1 Quality:   {tier1_quality:.3f}")
        logger.info(f"Size Mult:        {size_multiplier:.2f} (gates={gates})")
        logger.info(f"Confidence:       {confidence:.3f}")
        logger.info(f"Interpretation:   {interpretation}")

    return result


def _calculate_confidence(
    tier1_direction: float,
    tier1_quality: float,
    composite_scores: Dict[str, Dict[str, Any]],
    rs_available: bool,
) -> float:
    """
    Confidence in [0,1]:
      - Stronger direction magnitude => more confident
      - Higher quality => more confident
      - Agreement between directional components => more confident
    """
    signal_strength = _clamp(abs(tier1_direction), 0.0, 1.0)
    quality_component = _clamp(tier1_quality, 0.0, 1.0)

    momentum = float(_safe_get(composite_scores.get("momentum_composite", {}), "score", 0.0))
    trend = float(_safe_get(composite_scores.get("trend_composite", {}), "score", 0.0))
    rs = float(_safe_get(composite_scores.get("rs_composite", {}), "score", 0.0)) if rs_available else 0.0

    # Agreement is continuous-ish and ignores RS if not available.
    def sgn(x: float, eps: float = 0.05) -> int:
        if x > eps:
            return 1
        if x < -eps:
            return -1
        return 0

    sm = sgn(momentum)
    st = sgn(trend)
    sr = sgn(rs) if rs_available else 0

    # Base agreement from mom/trend
    if sm == 0 or st == 0:
        agreement_mt = 0.55  # weak/unclear signals
    elif sm == st:
        agreement_mt = 0.85
    else:
        agreement_mt = 0.35

    # If RS is available, incorporate it softly (don’t hard-require it).
    if rs_available and (sr != 0) and (sm != 0) and (st != 0):
        if sr == sm == st:
            agreement = min(1.0, agreement_mt + 0.10)
        elif sr != sm and sr != st:
            agreement = max(0.20, agreement_mt - 0.10)
        else:
            agreement = agreement_mt
    else:
        agreement = agreement_mt

    confidence = (
        0.40 * signal_strength +
        0.35 * quality_component +
        0.25 * agreement
    )
    return _clamp(confidence, 0.0, 1.0)


def _generate_interpretation(
    score: float,
    confidence: float,
    regime: str,
    tier1_quality: float,
    size_multiplier: float,
    gates: Dict[str, bool],
) -> str:
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

    if confidence > 0.75:
        conf_qual = "high confidence"
    elif confidence > 0.6:
        conf_qual = "good confidence"
    elif confidence > 0.4:
        conf_qual = "moderate confidence"
    else:
        conf_qual = "low confidence"

    gate_note = ""
    if not gates.get("risk_ok", True) or not gates.get("volume_ok", True):
        gate_note = " [GATED]"

    return (
        f"{signal} ({conf_qual}, {regime})"
        f" | quality={tier1_quality:.2f}, size_mult={size_multiplier:.2f}{gate_note}"
    )
