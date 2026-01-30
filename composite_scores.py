"""
Composite Score Calculation Module
"""

import logging
from typing import Dict, Any, Optional, Union
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clip(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    if value is None:
        return 0.0
    return max(min_val, min(max_val, value))


def normalize(value: Optional[float], min_val: float, max_val: float) -> float:
    if value is None:
        return 0.0
    if max_val == min_val:
        return 0.0
    normalized = (value - min_val) / (max_val - min_val)
    return clip(normalized * 2 - 1, -1.0, 1.0)


def safe_get(data: Dict, key: str, default: Any = 0.0) -> Any:
    val = data.get(key, default)
    return default if val is None else val


def squash_tanh(x: float, scale: float = 1.0, out_min: float = -1.0, out_max: float = 1.0) -> float:
    if x is None:
        return 0.0
    y = np.tanh(float(x) / max(1e-9, float(scale)))
    if out_min == -1.0 and out_max == 1.0:
        return float(y)
    # map [-1,1] -> [out_min,out_max]
    return float((y + 1.0) * 0.5 * (out_max - out_min) + out_min)


# =============================================================================
# Tier-1 helper: split direction vs quality
# =============================================================================

def tier1_quality_from_risk_volume_regime(
    risk_score_0_1: float,
    volume_score_0_1: float,
    regime_score_0_1: float = 0.5
) -> float:
    q = 0.45 * float(risk_score_0_1) + 0.25 * float(volume_score_0_1) + 0.30 * float(regime_score_0_1)
    return float(clip(q, 0.0, 1.0))


def size_multiplier_from_quality(q: float, lo: float = 0.55, hi: float = 1.35) -> float:
    q = float(clip(q, 0.0, 1.0))
    # Nonlinear mapping: emphasize higher quality more
    return float(lo + (hi - lo) * (q ** 1.3))


# =============================================================================
# Composite: Momentum
# =============================================================================

def calculate_momentum_composite(indicators: Dict[str, Any]) -> Dict[str, Any]:
    mom = indicators.get("momentum", {})
    trend = indicators.get("trend", {})

    mom_1m_z = mom.get("mom_1m_zscore", None)
    mom_3m_z = mom.get("mom_3m_zscore", None)
    mom_6m_z = mom.get("mom_6m_zscore", None)
    mom_12m_z = mom.get("mom_12m_zscore", None)

    if mom_3m_z is not None:
        mom_1m_norm = clip((mom_1m_z if mom_1m_z is not None else 0.0) / 3.0, -1, 1)
        mom_3m_norm = clip((mom_3m_z if mom_3m_z is not None else 0.0) / 3.0, -1, 1)
        mom_6m_norm = clip((mom_6m_z if mom_6m_z is not None else 0.0) / 3.0, -1, 1)
        mom_12m_norm = clip((mom_12m_z if mom_12m_z is not None else 0.0) / 3.0, -1, 1)
    else:
        mom_1m = safe_get(mom, "mom_1m")
        mom_3m = safe_get(mom, "mom_3m")
        mom_6m = safe_get(mom, "mom_6m")
        mom_12m = safe_get(mom, "mom_12m")
        mom_1m_norm = normalize(mom_1m, -0.10, 0.10)
        mom_3m_norm = normalize(mom_3m, -0.20, 0.20)
        mom_6m_norm = normalize(mom_6m, -0.30, 0.30)
        mom_12m_norm = normalize(mom_12m, -0.50, 0.50)

    mom_score = 0.40 * mom_3m_norm + 0.30 * mom_1m_norm + 0.20 * mom_6m_norm + 0.10 * mom_12m_norm

    macd_hist = safe_get(trend, "macd_histogram", 0.0)
    volatility = indicators.get("volatility", {})
    atr_pct = safe_get(volatility, "atr_14d_pct", 0.02)
    current_price = safe_get(indicators.get("id", {}), "price", safe_get(trend, "ma20", 100.0))

    # Guard against ultra-low ATR% producing huge scaled MACD
    atr_pct_floor = 0.005
    atr_pct_use = max(float(atr_pct), atr_pct_floor)

    if atr_pct_use > 0.001 and current_price > 0:
        atr_price = atr_pct_use * current_price
        raw = macd_hist / max(1e-9, (atr_price * 2))
        macd_signal = squash_tanh(raw, scale=1.0, out_min=-1.0, out_max=1.0)
    else:
        raw = macd_hist / max(1e-9, (current_price * 0.02)) if current_price > 0 else 0.0
        macd_signal = squash_tanh(raw, scale=1.0, out_min=-1.0, out_max=1.0)

    rsi = safe_get(trend, "rsi_14d", 50.0)
    rsi_mod = 0.0
    if abs(mom_score) < 0.4:
        if rsi > 75:
            rsi_mod = -0.15
        elif rsi < 25:
            rsi_mod = 0.15
    rsi_mod = clip(rsi_mod, -0.2, 0.2)

    volume_liq = indicators.get("volume_liquidity", {})
    vol_spike_z = safe_get(volume_liq, "volume_spike_z", 0.0)
    vol_mod = 0.0
    if mom_score > 0.3 and vol_spike_z < -1.5:
        vol_mod = -0.1
    elif vol_spike_z > 2.0:
        vol_mod = 0.1
    vol_mod = clip(vol_mod, -0.15, 0.15)

    final_score = clip(0.65 * mom_score + 0.20 * macd_signal + 0.10 * rsi_mod + 0.05 * vol_mod)

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
            "rsi_moderation": rsi_mod,
            "volume_mod": vol_mod,
        },
        "interpretation": interpretation,
    }


# =============================================================================
# Composite: Trend
# =============================================================================

def calculate_trend_composite(indicators: Dict[str, Any]) -> Dict[str, Any]:
    trend = indicators.get("trend", {})

    price_above_ma20 = trend.get("price_above_ma20")
    price_above_ma75 = trend.get("price_above_ma75")
    price_above_ma200 = trend.get("price_above_ma200")

    ma_score = 0.0
    if price_above_ma20 and price_above_ma75 and price_above_ma200:
        ma_score = 1.0
    elif (price_above_ma20 is False) and (price_above_ma75 is False) and (price_above_ma200 is False):
        ma_score = -1.0
    elif price_above_ma75 and price_above_ma200:
        ma_score = 0.6
    elif price_above_ma200:
        ma_score = 0.3
    elif (price_above_ma200 is False) and price_above_ma75:
        ma_score = -0.3
    elif (price_above_ma200 is False) and (price_above_ma75 is False) and price_above_ma20:
        ma_score = -0.6

    golden_cross = trend.get("golden_cross_recent", False)
    death_cross = trend.get("death_cross_recent", False)
    cross_boost = 0.3 if golden_cross else (-0.3 if death_cross else 0.0)

    adx = safe_get(trend, "adx_14d", 20.0)
    adx_level_score = normalize(adx, 0, 40)
    adx_level_score = (adx_level_score + 1) / 2
    adx_level_score = max(0.3, adx_level_score)

    structure = trend.get("higher_highs_lows", "mixed")
    structure_score = 0.8 if structure == "uptrend" else (-0.8 if structure == "downtrend" else 0.0)

    base_trend = 0.6 * ma_score + 0.4 * structure_score

    macd_line = safe_get(trend, "macd_line", 0.0)
    consistency_multiplier = 1.0
    if ma_score > 0 and structure_score > 0 and macd_line > 0:
        consistency_multiplier = 1.2
    elif ma_score < 0 and structure_score < 0 and macd_line < 0:
        consistency_multiplier = 1.2

    if adx < 20 and abs(ma_score) < 0.5:
        adx_dampener = 0.5
    elif adx < 20:
        adx_dampener = 0.7
    else:
        adx_dampener = 1.0

    final_score = clip(base_trend * consistency_multiplier * adx_level_score * adx_dampener + cross_boost)

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
            "cross_boost": cross_boost,
        },
        "interpretation": interpretation,
    }


# =============================================================================
# Composite: Relative Strength (placeholder for now)
# =============================================================================

def calculate_relative_strength_composite(indicators: Dict[str, Any]) -> Dict[str, Any]:
    # Check if benchmark data is available for single-stock RS calculation
    benchmark_data = indicators.get("benchmark_returns", {})
    
    if benchmark_data and len(benchmark_data) > 0:
        # Compute simple RS vs benchmark
        try:
            # Get stock momentum data
            momentum = indicators.get("momentum", {})
            stock_ret_3m = safe_get(momentum, "ret_3m", 0.0)
            stock_ret_1m = safe_get(momentum, "ret_1m", 0.0)
            stock_ret_6m = safe_get(momentum, "ret_6m", 0.0)
            
            # Get benchmark returns (assume same timeframes)
            bench_ret_3m = safe_get(benchmark_data, "ret_3m", 0.0)
            bench_ret_1m = safe_get(benchmark_data, "ret_1m", 0.0) 
            bench_ret_6m = safe_get(benchmark_data, "ret_6m", 0.0)
            
            # Calculate relative performance
            rel_3m = stock_ret_3m - bench_ret_3m
            rel_1m = stock_ret_1m - bench_ret_1m
            rel_6m = stock_ret_6m - bench_ret_6m
            
            # Normalize to [-1, 1] with reasonable bounds (e.g., ±20% differential = ±1.0)
            # Use tanh for soft normalization
            rel_score = np.tanh(rel_3m / 0.10)  # 10% differential = score of ~0.76
            
            # Weight recent performance more heavily
            weighted_rel = 0.5 * rel_3m + 0.3 * rel_1m + 0.2 * rel_6m
            final_score = clip(np.tanh(weighted_rel / 0.08), -1.0, 1.0)
            
            return {
                "score": float(final_score),
                "available": True,
                "components": {
                    "rs_capture": float(np.tanh(rel_3m / 0.10)),
                    "rs_trend_bonus": 0.0,  # Not implemented yet
                    "rs_relative_perf": float(rel_3m),
                    "relative_performance_level": float(weighted_rel),
                    "capture_ratio_score": 0.0,  # Not implemented yet
                },
                "interpretation": f"RS vs benchmark: {final_score:+.2f} (3M rel: {rel_3m:+.1%})",
            }
        except Exception as e:
            # Fallback if calculation fails
            return {
                "score": 0.0,
                "available": False,
                "components": {
                    "rs_capture": 0.0,
                    "rs_trend_bonus": 0.0,
                    "rs_relative_perf": 0.0,
                    "relative_performance_level": 0.0,
                    "capture_ratio_score": 0.0,
                },
                "interpretation": f"RS calculation failed: {e}",
            }
    
    # Default: RS unavailable
    return {
        "score": 0.0,
        "available": False,
        "components": {
            "rs_capture": 0.0,
            "rs_trend_bonus": 0.0,
            "rs_relative_perf": 0.0,
            "relative_performance_level": 0.0,
            "capture_ratio_score": 0.0,
        },
        "interpretation": "RS unavailable (cross-sectional not enabled, no benchmark provided)",
    }


# =============================================================================
# Composite: Volume Confirmation
# =============================================================================

def calculate_volume_confirmation_composite(indicators: Dict[str, Any]) -> Dict[str, Any]:
    vol_liq = indicators.get("volume_liquidity", {})
    momentum = indicators.get("momentum", {})
    events = indicators.get("events", {})  # kept for backward compat if present
    price_events = indicators.get("price_events", {})  # new-ish path in your alpha composite

    obv_trend = vol_liq.get("obv_trend_21d", "flat")
    obv_score = 1.0 if obv_trend == "up" else (0.0 if obv_trend == "down" else 0.5)

    ad_trend = vol_liq.get("acc_dist_trend_21d", "flat")
    ad_score = 1.0 if ad_trend == "up" else (0.0 if ad_trend == "down" else 0.5)

    vol_spike_z = safe_get(vol_liq, "volume_spike_z", 0.0)
    spike_score = 0.5
    if vol_spike_z >= 3:
        spike_score = 0.6
    elif vol_spike_z >= 2:
        spike_score = 1.0
    elif vol_spike_z > 0:
        spike_score = 0.7
    elif vol_spike_z > -1:
        spike_score = 0.3
    else:
        spike_score = 0.1

    ret_1d = safe_get(momentum, "ret_1d", 0.0)
    ret_3d = safe_get(momentum, "ret_3d", 0.0)
    intraday_ret = safe_get(price_events, "intraday_ret", safe_get(events, "intraday_ret", 0.0))

    short_move = 0.5 * ret_1d + 0.3 * ret_3d + 0.2 * intraday_ret

    confirmation_bonus = 0.0
    if short_move > 0.01 and obv_trend == "up" and ad_trend == "up":
        confirmation_bonus = 0.2
    elif short_move < -0.01 and obv_trend == "down" and ad_trend == "down":
        confirmation_bonus = 0.1
    elif short_move > 0.01 and (obv_trend == "down" or ad_trend == "down"):
        confirmation_bonus = -0.2
    elif short_move < -0.01 and (obv_trend == "up" or ad_trend == "up"):
        confirmation_bonus = 0.2

    base_score = 0.4 * obv_score + 0.4 * ad_score + 0.2 * spike_score
    final_score = clip(base_score + confirmation_bonus, 0.0, 1.0)

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
            "confirmation_bonus": confirmation_bonus,
        },
        "interpretation": interpretation,
    }


# =============================================================================
# Composite: Risk
# =============================================================================

def calculate_risk_composite(indicators: Dict[str, Any]) -> Dict[str, Any]:
    risk = indicators.get("risk", {})
    vol = indicators.get("volatility", {})

    sharpe = safe_get(risk, "sharpe_1y", None)
    if sharpe is not None:
        sharpe_z = (float(sharpe) - 0.5) / 0.8
        sharpe_score = 0.5 * (np.tanh(np.clip(sharpe_z, -3, 3)) + 1)
    else:
        sharpe_score = 0.5

    sortino = safe_get(risk, "sortino_1y", None)
    if sortino is not None:
        sortino_z = (float(sortino) - 0.7) / 1.0
        sortino_score = 0.5 * (np.tanh(np.clip(sortino_z, -3, 3)) + 1)
    else:
        sortino_score = 0.5

    max_dd = safe_get(risk, "max_drawdown_1y", 0.3)
    if max_dd < 0.05:
        dd_score = 1.0
    else:
        dd_normalized = float(max_dd) / 0.25
        dd_score = 1.0 - np.tanh(dd_normalized) ** 1.5
        dd_score = max(0.0, min(1.0, float(dd_score)))

    realized_vol = safe_get(vol, "realized_vol_252d", 0.3)
    rv = float(realized_vol)
    if rv < 0.12:
        vol_score = 1.0
    elif rv < 0.20:
        vol_score = 0.85
    elif rv < 0.30:
        vol_score = 0.60
    elif rv < 0.45:
        vol_score = 0.30
    else:
        vol_score = 0.10

    # Beta is often nearly constant for large caps under coarse bucketing.
    # Keep it smooth + very low weight so it can't dominate or become a constant "+1".
    beta = safe_get(risk, "beta_1y", None)
    if beta is None:
        beta_score = 0.5
    else:
        b = float(beta)
        # peak at 1.0, decay with distance; ~0.9 at 0.2 away, ~0.6 at 0.7 away
        beta_score = float(np.exp(-((b - 1.0) / 0.6) ** 2))
        beta_score = max(0.0, min(1.0, beta_score))

    final_score = clip(
        0.32 * sharpe_score +
        0.28 * dd_score +
        0.22 * sortino_score +
        0.16 * vol_score +
        0.02 * beta_score,
        0.0,
        1.0,
    )

    if final_score > 0.75:
        interpretation = "Excellent risk profile"
    elif final_score > 0.6:
        interpretation = "Good risk profile"
    elif final_score > 0.4:
        interpretation = "Moderate risk profile"
    else:
        interpretation = "Unfavorable risk profile"

    return {
        "score": float(final_score),
        "components": {
            "sharpe": float(sharpe_score),
            "sortino": float(sortino_score),
            "max_drawdown": float(dd_score),
            "volatility": float(vol_score),
            "beta": float(beta_score),
        },
        "interpretation": interpretation,
    }


# =============================================================================
# Tier-1 confirmation for alpha (direction vs quality)
# =============================================================================

Tier1Context = Union[float, Dict[str, Any], None]


def calculate_tier1_alpha_confirmation(
    alpha_score: float,
    tier1_context: Tier1Context,
    alpha_threshold_for_size: float = 0.08,  
    tier1_dir_threshold: float = 0.08,     
) -> Dict[str, Any]:
    """
    Uses Tier-1 *direction* for agreement and Tier-1 *quality* for sizing.
    Backward compatible:
      - if tier1_context is float: treated as tier1_direction, quality defaults to 0.5
      - if dict: expects keys like tier1_direction, tier1_quality, size_multiplier (optional)
    """
    alpha_score = float(alpha_score)

    tier1_direction = 0.0
    tier1_quality = 0.5
    base_size_mult = 1.0
    gates = {}

    if isinstance(tier1_context, dict):
        tier1_direction = float(tier1_context.get("tier1_direction", tier1_context.get("tier1_score", 0.0)))
        tier1_quality = float(tier1_context.get("tier1_quality", 0.5))
        base_size_mult = float(tier1_context.get("size_multiplier", size_multiplier_from_quality(tier1_quality)))
        gates = tier1_context.get("gates", {}) or {}
    elif isinstance(tier1_context, (int, float)):
        tier1_direction = float(tier1_context)
        tier1_quality = 0.5
        base_size_mult = 1.0
    else:
        tier1_direction = 0.0
        tier1_quality = 0.5
        base_size_mult = 1.0

    alpha_strong_for_size = abs(alpha_score) >= float(alpha_threshold_for_size)
    tier1_direction_meaningful = abs(tier1_direction) >= float(tier1_dir_threshold)

    agreement = None
    if alpha_strong_for_size and tier1_direction_meaningful:
        agreement = (np.sign(alpha_score) == np.sign(tier1_direction))

    # Sizing: start from quality-derived base
    size_multiplier = float(base_size_mult)

    # Confidence based on tier1 quality and agreement, not alpha magnitude
    confidence = "neutral"
    
    if tier1_quality >= 0.40 and (agreement is True or not tier1_direction_meaningful):
        confidence = "high"
        size_multiplier = min(1.25, size_multiplier * 1.05)  # 5% boost for high confidence
    elif tier1_quality <= 0.35 or (agreement is False and tier1_direction_meaningful):
        confidence = "low"
        size_multiplier = max(0.60, size_multiplier * 0.75)  # 25% penalty for low confidence
    # else: remains "neutral"

    # If Tier-1 gates indicate "permission" issues, cap sizing regardless
    if isinstance(gates, dict):
        if gates.get("risk_ok") is False or gates.get("volume_ok") is False:
            size_multiplier = min(size_multiplier, 0.85)

    return {
        "confidence": confidence,
        "size_multiplier": float(size_multiplier),
        "agreement": agreement,
        "alpha_strong": bool(alpha_strong_for_size),
        "tier1_strong": bool(tier1_direction_meaningful),
        "tier1_direction": float(tier1_direction),
        "tier1_quality": float(tier1_quality),
    }


# =============================================================================
# Next-day alpha composite
# =============================================================================

def calculate_next_day_alpha_composite(
    indicators: Dict[str, Any],
    long_threshold: float = 0.15,  # Lowered from 0.30
    short_threshold: float = -0.15,  # Raised from -0.30
    liquidity_min_score: float = 30.0,
    liquidity_min_dollar_vol: float = 5_000_000.0,
    tier1_context: Tier1Context = None,
) -> Dict[str, Any]:
    momentum = indicators.get("momentum", {})
    trend = indicators.get("trend", {})
    price_events = indicators.get("price_events", {})
    volatility = indicators.get("volatility", {})
    vol_liq = indicators.get("volume_liquidity", {})

    ret_1d = safe_get(momentum, "ret_1d", 0.0)
    ret_3d = safe_get(momentum, "ret_3d", 0.0)
    ret_5d = safe_get(momentum, "ret_5d", 0.0)
    ret_10d = safe_get(momentum, "ret_10d", 0.0)

    ret_1d_z = safe_get(momentum, "ret_1d_zscore", None)
    ret_3d_z = safe_get(momentum, "ret_3d_zscore", None)
    ret_5d_z = safe_get(momentum, "ret_5d_zscore", None)
    ret_10d_z = safe_get(momentum, "ret_10d_zscore", None)

    gap_size = safe_get(price_events, "gap_size_pct", 0.0)
    intraday_ret = safe_get(price_events, "intraday_ret", 0.0)
    range_pct = safe_get(price_events, "range_pct", 0.02)
    close_loc = safe_get(price_events, "close_loc_10d", 0.5)

    gap_size_z = safe_get(price_events, "gap_size_zscore", None)
    intraday_ret_z = safe_get(price_events, "intraday_ret_zscore", None)
    range_pct_z = safe_get(price_events, "range_pct_zscore", None)

    rsi_2d = safe_get(trend, "rsi_2d", 50.0)
    rsi_3d = safe_get(trend, "rsi_3d", 50.0)

    rsi_reversion = 0.0
    if rsi_2d < 10:
        rsi_reversion = 0.8
    elif rsi_2d < 20:
        rsi_reversion = 0.5
    elif rsi_2d < 30:
        rsi_reversion = 0.2
    elif rsi_2d > 90:
        rsi_reversion = -0.8
    elif rsi_2d > 80:
        rsi_reversion = -0.5
    elif rsi_2d > 70:
        rsi_reversion = -0.2

    close_loc_reversion = (0.5 - close_loc) * 2
    close_loc_continuation = (close_loc - 0.5) * 2

    vol_ratio = safe_get(volatility, "vol_ratio_21v252", 1.0)
    adx = safe_get(trend, "adx_14d", 20.0)
    ma200_slope_up = trend.get("ma200_slope_up", None)

    autocorr_5d = safe_get(momentum, "autocorr_5d", 0.0)
    autocorr_10d = safe_get(momentum, "autocorr_10d", 0.0)
    autocorr_21d = safe_get(momentum, "autocorr_21d", 0.0)

    autocorr_avg = 0.5 * autocorr_5d + 0.3 * autocorr_10d + 0.2 * autocorr_21d

    reversion_weight = 0.5
    continuation_weight = 0.5

    if autocorr_avg < -0.08:
        reversion_weight = 0.70
        continuation_weight = 0.30
        regime = "mean_reversion"
    elif autocorr_avg > 0.08:
        reversion_weight = 0.30
        continuation_weight = 0.70
        regime = "continuation"
    elif vol_ratio > 1.5:
        reversion_weight = 0.65
        continuation_weight = 0.35
        regime = "mean_reversion"
    elif vol_ratio < 0.8 and adx > 20 and ma200_slope_up is True:
        reversion_weight = 0.35
        continuation_weight = 0.65
        regime = "continuation"
    elif vol_ratio < 0.8 and adx > 20 and ma200_slope_up is False:
        reversion_weight = 0.35
        continuation_weight = 0.65
        regime = "continuation"
    else:
        regime = "mixed"

    if ret_1d_z is not None:
        short_ret_z_signal = clip(float(ret_1d_z) / 2.0, -1, 1)
    elif range_pct > 0.001:
        short_ret_z_signal = clip(float(ret_1d) / float(range_pct) / 3.0, -1, 1)
    else:
        short_ret_z_signal = 0.0

    if gap_size_z is not None:
        gap_signal = clip(float(gap_size_z) / 2.0, -1, 1)
    else:
        gap_signal = clip(float(gap_size) * 10, -1, 1)

    if ret_3d_z is not None:
        ret_3d_signal = clip(float(ret_3d_z) / 2.0, -1, 1)
    else:
        ret_3d_signal = clip(float(ret_3d) * 10, -1, 1)

    reversion_signal = (
        0.30 * rsi_reversion +
        0.30 * close_loc_reversion +
        0.25 * (-short_ret_z_signal) +
        0.15 * (-gap_signal)
    )

    continuation_signal = (
        0.35 * short_ret_z_signal +
        0.30 * close_loc_continuation +
        0.20 * ret_3d_signal +
        0.15 * gap_signal
    )

    raw_alpha = reversion_weight * reversion_signal + continuation_weight * continuation_signal

    liquidity_score = safe_get(vol_liq, "liquidity_score_0_100", 50.0)
    avg_dollar_vol = safe_get(vol_liq, "avg_dollar_vol_21d", 10_000_000.0)

    gates = {
        "liquidity_ok": liquidity_score >= liquidity_min_score,
        "dollar_vol_ok": avg_dollar_vol >= liquidity_min_dollar_vol,
        "long_permitted": True,
        "short_permitted": True,
    }

    if ma200_slope_up is False:
        gates["long_permitted"] = raw_alpha > 0.4
        if 0 < raw_alpha <= 0.4:
            raw_alpha *= 0.5

    if ma200_slope_up is True:
        gates["short_permitted"] = raw_alpha < -0.4
        if -0.4 <= raw_alpha < 0:
            raw_alpha *= 0.5

    final_score = clip(raw_alpha, -1, 1)

    if (not gates["liquidity_ok"]) or (not gates["dollar_vol_ok"]):
        decision = "HOLD"
        decision_reason = "Insufficient liquidity"
    elif final_score > long_threshold and gates["long_permitted"]:
        decision = "LONG"
        decision_reason = f"Alpha score {final_score:.3f} > LONG threshold {long_threshold}"
    elif final_score < short_threshold and gates["short_permitted"]:
        decision = "SHORT"
        decision_reason = f"Alpha score {final_score:.3f} < SHORT threshold {short_threshold}"
    else:
        decision = "HOLD"
        decision_reason = (
            f"Alpha score {final_score:.3f} in HOLD zone (LONG>{long_threshold}, SHORT<{short_threshold})"
        )

    # ==========================================================================
    # 9. Tier-1 Confirmation (direction vs quality)
    # ==========================================================================
    tier1_confirmation = None
    if tier1_context is not None:
        tier1_confirmation = calculate_tier1_alpha_confirmation(
            alpha_score=final_score,
            tier1_context=tier1_context,
            alpha_threshold_for_size=0.08,  # Lowered from 0.15
            tier1_dir_threshold=0.08,       # Lowered from 0.12
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

    # Append Tier-1 confidence if available
    if tier1_confirmation and tier1_confirmation.get("confidence") != "neutral":
        interpretation += f" [Tier-1 {tier1_confirmation['confidence']} confidence]"

    return {
        "score": float(final_score),
        "decision": decision,
        "decision_reason": decision_reason,
        "regime": regime,
        "components": {
            "reversion_signal": float(reversion_signal),
            "continuation_signal": float(continuation_signal),
            "reversion_weight": float(reversion_weight),
            "continuation_weight": float(continuation_weight),
            "autocorr_avg": float(autocorr_avg),
            "autocorr_5d": float(autocorr_5d),
            "autocorr_10d": float(autocorr_10d),
            "autocorr_21d": float(autocorr_21d),
            "vol_ratio": float(vol_ratio),
            "rsi_reversion": float(rsi_reversion),
            "close_loc_reversion": float(close_loc_reversion),
            "short_ret_z_signal": float(short_ret_z_signal),
            "gap_signal_z": float(gap_signal),
            "ret_3d_signal_z": float(ret_3d_signal),
            "using_zscore_ret_1d": ret_1d_z is not None,
            "using_zscore_gap": gap_size_z is not None,
            "using_zscore_ret_3d": ret_3d_z is not None,
        },
        "gates": gates,
        "tier1_confirmation": tier1_confirmation,
        "interpretation": interpretation,
    }


# =============================================================================
# Master Function: Calculate All Composites
# =============================================================================

def calculate_all_composites(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all composite scores for a stock.
    """
    symbol = indicators.get("id", {}).get("symbol", "UNKNOWN")

    return {
        "symbol": symbol,
        "momentum_composite": calculate_momentum_composite(indicators),
        "trend_composite": calculate_trend_composite(indicators),
        "rs_composite": calculate_relative_strength_composite(indicators),
        "volume_composite": calculate_volume_confirmation_composite(indicators),
        "risk_composite": calculate_risk_composite(indicators),
        # NOTE: tier1_context is optional; pass it in from your Tier-1 scoring module if desired
        "next_day_alpha": calculate_next_day_alpha_composite(indicators, tier1_context=None),
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
        raise SystemExit(1)

    for json_file in output_dir.glob("*_technical_indicators.json"):
        with open(json_file) as f:
            indicators = json.load(f)

        symbol = indicators.get("id", {}).get("symbol", "UNKNOWN")
        composites = calculate_all_composites(indicators)

        print(f"\n{'='*80}")
        print(f"{symbol} - COMPOSITE SCORES")
        print(f"{'='*80}")
        print(
            f"Momentum:  {composites['momentum_composite']['score']:+.3f} - {composites['momentum_composite']['interpretation']}"
        )
        print(
            f"Trend:     {composites['trend_composite']['score']:+.3f} - {composites['trend_composite']['interpretation']}"
        )
        print(
            f"Rel Str:   {composites['rs_composite']['score']:+.3f} - {composites['rs_composite']['interpretation']}"
        )
        print(
            f"Volume:    {composites['volume_composite']['score']: .3f} - {composites['volume_composite']['interpretation']}"
        )
        print(
            f"Risk:      {composites['risk_composite']['score']: .3f} - {composites['risk_composite']['interpretation']}"
        )
        print(
            f"Alpha(T+1): {composites['next_day_alpha']['score']:+.3f} - {composites['next_day_alpha']['interpretation']} ({composites['next_day_alpha']['decision']})"
        )

