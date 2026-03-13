import os
import json
import re
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import httpx
import numpy as np

# Load secrets from secrets.json
def _load_secrets():
    """Load API keys from secrets.json."""
    secrets_path = Path(__file__).parent / "secrets.json"
    if secrets_path.exists():
        with open(secrets_path, "r") as f:
            return json.load(f)
    return {}

_SECRETS = _load_secrets()

# DeepSeek API Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY = _SECRETS.get("DEEPSEEK_API_KEY", "")
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.15
REQUEST_TIMEOUT = 300.0
PORTFOLIO_REQUEST_TIMEOUT = 300.0  

# =============================================================================
# API Configuration (DeepSeek)
# =============================================================================

def get_deepseek_url() -> str:
    """Get DeepSeek API URL."""
    return DEEPSEEK_API_URL

def get_deepseek_api_key() -> str:
    """Get DeepSeek API key from environment."""
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
    return DEEPSEEK_API_KEY


# =============================================================================
# Validation & Hardening
# =============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def validate_and_deduplicate_portfolio(
    long_basket: list,
    short_basket: list,
    previous_portfolio: dict = None,
) -> Tuple[list, list]:
    """
    Enforce portfolio state machine:
    - No duplicate tickers within each leg
    - No ticker in both long AND short
    - No ticker in both kept AND dropped states
    - Rebuild final weights after deduplication

    Args:
        long_basket: [{ticker, weight, rationale}, ...]
        short_basket: [{ticker, weight, rationale}, ...]
        previous_portfolio: Previous portfolio dict (optional, for state validation)

    Returns:
        (clean_long_basket, clean_short_basket)
    """
    seen_tickers = set()
    clean_long = []
    clean_short = []

    # De-duplicate long basket — first occurrence wins
    for pos in long_basket:
        ticker = pos.get("ticker", "").upper()
        if ticker and ticker not in seen_tickers:
            pos["ticker"] = ticker
            clean_long.append(pos)
            seen_tickers.add(ticker)

    # De-duplicate short basket — skip any ticker already in long
    for pos in short_basket:
        ticker = pos.get("ticker", "").upper()
        if ticker and ticker not in seen_tickers:
            pos["ticker"] = ticker
            clean_short.append(pos)
            seen_tickers.add(ticker)

    # Re-normalize weights to sum to 1.0 per leg
    long_total = sum(p.get("weight", 0) for p in clean_long)
    short_total = sum(p.get("weight", 0) for p in clean_short)

    if long_total > 0:
        for p in clean_long:
            p["weight"] = p.get("weight", 0) / long_total
    if short_total > 0:
        for p in clean_short:
            p["weight"] = p.get("weight", 0) / short_total

    return clean_long, clean_short


def validate_portfolio_signals(
    long_basket: list,
    short_basket: list,
    universe: dict,
    previous_portfolio: dict = None,
) -> Tuple[bool, str]:
    """
    Pre-trade validation: reject portfolios with CLEAR signal violations.

    Rules:
    1. All longs must have direction_score > -5 (not clearly bearish)
    2. All shorts must have direction_score < +5 (not clearly bullish)
    3. No cost-driven overrides when direction signal CLEARLY flips (>15 point swing)

    Rationale: Stocks with direction ∈ (-5, +5) are ambiguous and can be traded either way.
    This avoids false positives on near-zero signals, allowing the LLM to make discretionary trades.

    Args:
        long_basket: [{ticker, weight, rationale}, ...]
        short_basket: [{ticker, weight, rationale}, ...]
        universe: {ticker: {direction_score, ...}}
        previous_portfolio: Previous portfolio for sign-flip detection

    Returns:
        (is_valid: bool, reason: str)
    """
    DIRECTION_THRESHOLD = 5.0  # Ambiguous zone: [-5, +5]
    FLIP_THRESHOLD = 15.0      # Clear flip: >15 point swing

    # Check rule 1: all longs must not be clearly bearish
    for pos in long_basket:
        ticker = pos.get("ticker", "").upper()
        if ticker not in universe:
            return False, f"Long position {ticker} not in universe"
        
        direction = universe[ticker].get("direction_score", 0)
        if direction < -DIRECTION_THRESHOLD:  # CLEARLY bearish (< -5)
            return False, f"Long position {ticker} has clearly bearish direction ({direction:.1f})"

    # Check rule 2: all shorts must not be clearly bullish
    for pos in short_basket:
        ticker = pos.get("ticker", "").upper()
        if ticker not in universe:
            return False, f"Short position {ticker} not in universe"
        
        direction = universe[ticker].get("direction_score", 0)
        if direction > DIRECTION_THRESHOLD:  # CLEARLY bullish (> +5)
            return False, f"Short position {ticker} has clearly bullish direction ({direction:.1f})"

    # Check rule 3: no cost-driven overrides when signal CLEARLY flips (>15 point swing)
    if previous_portfolio:
        prev_longs = {p.get("ticker", "").upper(): p for p in previous_portfolio.get("long_basket", [])}
        prev_shorts = {p.get("ticker", "").upper(): p for p in previous_portfolio.get("short_basket", [])}
        
        for pos in long_basket:
            ticker = pos.get("ticker", "").upper()
            
            # Check if this was previously short (sign flip)
            if ticker in prev_shorts:
                rationale = pos.get("rationale", "").upper()
                if "COST_DRIVEN" in rationale:
                    # Only block if it's a CLEAR flip (e.g., was -40, now +10)
                    current_direction = universe[ticker].get("direction_score", 0)
                    prev_direction = prev_shorts[ticker].get("direction_score", 0)
                    if (current_direction - prev_direction) > FLIP_THRESHOLD:
                        return False, f"Cannot keep {ticker} on cost override: flipped {prev_direction:.1f} → {current_direction:.1f}"

        for pos in short_basket:
            ticker = pos.get("ticker", "").upper()
            
            # Check if this was previously long (sign flip)
            if ticker in prev_longs:
                rationale = pos.get("rationale", "").upper()
                if "COST_DRIVEN" in rationale:
                    # Only block if it's a CLEAR flip (e.g., was +40, now -10)
                    current_direction = universe[ticker].get("direction_score", 0)
                    prev_direction = prev_longs[ticker].get("direction_score", 0)
                    if (prev_direction - current_direction) > FLIP_THRESHOLD:
                        return False, f"Cannot keep {ticker} on cost override: flipped {prev_direction:.1f} → {current_direction:.1f}"

    return True, "All signal validations passed"


def categorize_by_magnitude_mode(universe: dict) -> dict:
    """
    Segment universe into three magnitude modes.

    Returns:
        {
            "low":  {ticker: data, ...},   # mean-reversion scalps
            "mid":  {ticker: data, ...},   # noisy, generally avoid
            "high": {ticker: data, ...},   # momentum / breakout mode
            "thresholds": (p33, p67)
        }
    """
    mags = [d.get("magnitude_score", 0) for d in universe.values()]
    if len(mags) < 10:
        return {"low": {}, "mid": dict(universe), "high": {}, "thresholds": (0, 100)}

    p33 = float(np.percentile(mags, 33))
    p67 = float(np.percentile(mags, 67))

    return {
        "low":  {t: d for t, d in universe.items() if d.get("magnitude_score", 0) <= p33},
        "mid":  {t: d for t, d in universe.items() if p33 < d.get("magnitude_score", 0) <= p67},
        "high": {t: d for t, d in universe.items() if d.get("magnitude_score", 0) > p67},
        "thresholds": (p33, p67),
    }


# =============================================================================
# Portfolio Construction LLM — Universe-Level Decision Making
# =============================================================================

PORTFOLIO_SYSTEM_PROMPT = """
You are an elite quantitative portfolio manager constructing a long/short portfolio.
You are given a universe of stocks with deterministic technical scores, raw indicators,
and magnitude-mode labels on a single rebalance date.
Your job is to BEAT the naive mechanical baseline.

MECHANICAL BASELINE (what you are competing against):
- Sort all stocks by magnitude_score descending
- Take the top N as candidate pool
- LONG the strongest direction_scores, SHORT the weakest
- Equal-weight all positions (1/N per leg)
- This baseline ignores signal quality, cost, correlation, and context

--------------------------------------------------

YOUR EDGE:

1. INCLUDE / EXCLUDE
   Drop candidates with red flags (signal conflict, weak technicals).
   Promote stocks from outside the top-N magnitude pool if they have
   exceptionally clean directional signals.

2. SIGNAL QUALITY
   Require confluence: momentum + trend + volume alignment.
   Penalise divergences (strong momentum but weakening trend, overbought RSI
   in a weak trend, etc.).

3. MAGNITUDE MODE AWARENESS
   You will see each stock labelled as LOW / MID / HIGH magnitude.
   - HIGH magnitude: best for momentum / breakout trades
   - LOW magnitude:  best for mean-reversion scalps
   - MID magnitude:  noisy — generally AVOID unless direction is extreme
   Prefer stocks from HIGH or LOW buckets.

4. COST MINIMISATION & RETENTION RULES (DIFFERENT PER LEG)
   Turnover costs ~15 bps one-way (~30 bps round-trip per swap).

   LONGS — HOLD LONGER:
   - Default: KEEP a long position if it is still in top-50 percent by direction_score.
   - Only drop a long if the signal has materially deteriorated.
   - Kept longs historically outperform newly added longs.

   SHORTS — ROTATE AGGRESSIVELY:
   - Default: REPLACE shorts each rebalance unless the direction_score has become
     even MORE negative since last period.
   - Keeping stale shorts is dangerous — negative direction + high magnitude
     often means "strong stock pausing before another move higher".
   - Fresh short signals outperform kept shorts.

   COST-DRIVEN TRANSPARENCY:
   If you keep a position PRIMARILY to save turnover cost and the signal is weak,
   you MUST tag the rationale with the prefix "COST_DRIVEN: ".
   Example: "COST_DRIVEN: keeping to avoid 30 bps swap but signal is marginal".
   The system will track and potentially downweight these.

5. CORRELATION AWARENESS
   Avoid loading multiple stocks with similar technical profiles.
   Diversify across momentum, value, and volatility characteristics.

6. REGIME AWARENESS
   Adjust aggressiveness based on the given market regime:
   - crisis / high_vol: fewer positions, lower conviction, favour shorts
   - grind_up / low_vol: more aggressive on longs
   - normal: balanced approach

--------------------------------------------------

SCORING CONTEXT:
- direction_score: ~[-100, +100]. Positive = bullish, negative = bearish.
- magnitude_score: ~[0, 100]. Higher = bigger expected move.
- Raw indicators are also provided.

STRICT CONSTRAINTS:
- Pick between 5 and 30 stocks per leg.
- Weights per leg MUST sum to exactly 1.0.
- ONLY use tickers from the provided universe.
- Do NOT invent tickers or data.
- Do NOT use any knowledge about specific calendar dates — the date is masked.
- Each ticker may appear in ONLY ONE leg (no duplicates, no cross-leg overlap).
- Do not select positive direction scores as SHORTS or negative direction scores as LONGS without a very strong rationale.

--------------------------------------------------

CONFIDENCE SCORING (0.0 to 1.0):

Confidence reflects how aligned your portfolio is with the data:

0.9-1.0 = Exceptional confluence
  - 4+ indicators align on EVERY position
  - Strong direction scores (|score| > 40)
  - All picks from HIGH or LOW magnitude buckets
  - Zero cost-driven overrides
  - Portfolio deviates significantly from mechanical baseline

0.7-0.8 = Good confidence
  - 3+ indicators align on most positions
  - Direction scores mostly strong (|score| > 20)
  - Mix of HIGH/LOW magnitude with occasional MID
  - <10% cost-driven positions
  - Some deviation from baseline (added value)

0.5-0.7 = Moderate confidence
  - 2+ indicators align
  - Direction scores mixed (some <20)
  - Heavy use of MID magnitude (noisy)
  - 10-30% cost-driven positions
  - Minor deviation from baseline

0.3-0.5 = Low confidence
  - Weak indicator confluence (<2 aligned)
  - Weak direction scores (many <10)
  - Mostly cost-driven overrides (>30%)
  - Portfolio very similar to baseline

0.0-0.3 = Very low confidence (AVOID)
  - Contradictory signals
  - Weak or missing directionality
  - Excessive cost-driven inertia

OUTPUT FORMAT (STRICT JSON):
{
  "long_basket": [
    {"ticker": "AAPL", "weight": 0.05, "rationale": "Strong momentum + trend confluence"},
    ...
  ],
  "short_basket": [
    {"ticker": "XYZ", "weight": 0.04, "rationale": "Weak direction, high magnitude"},
    ...
  ],
  "kept_from_previous": ["AAPL", "MSFT"],
  "dropped_from_previous": ["TSLA"],
  "reasoning": "2-3 sentence portfolio construction rationale",
  "regime_adjustment": "How you adjusted for current regime",
  "confidence": 0.65
}
"""

PORTFOLIO_OUTPUT_SCHEMA = {
    "long_basket": [{"ticker": "str", "weight": "float [0,1]", "rationale": "str"}],
    "short_basket": [{"ticker": "str", "weight": "float [0,1]", "rationale": "str"}],
    "kept_from_previous": ["str"],
    "dropped_from_previous": ["str"],
    "reasoning": "str",
    "regime_adjustment": "str",
    "confidence": "float [0, 1], increments of 0.01"
}


def _format_universe_for_prompt(
    universe: dict,
    previous_portfolio: dict = None,
    regime: str = "unknown",
    rebalance_id: str = "w_00001",
    top_n_magnitude: int = 60,
) -> str:
    """Format universe data into a readable prompt for the LLM with PRE-FILTERED candidate pools.

    Separates stocks by direction signal to guide LLM toward correct long/short assignments.

    Args:
        universe: {ticker: {direction_score, magnitude_score, indicators...}}
        previous_portfolio: {long_basket: [{ticker, weight}], short_basket: [{ticker, weight}]} or None
        regime: Current market regime string
        rebalance_id: Opaque date identifier
        top_n_magnitude: How many top-magnitude stocks to show per candidate pool

    Returns:
        Formatted string for the LLM prompt with separated candidate pools
    """
    DIRECTION_THRESHOLD = 5.0
    
    # Separate into candidate pools by direction signal
    clear_long_candidates = {}
    clear_short_candidates = {}
    ambiguous_candidates = {}
    
    for ticker, data in universe.items():
        direction = data.get("direction_score", 0)
        if direction > DIRECTION_THRESHOLD:
            clear_long_candidates[ticker] = data
        elif direction < -DIRECTION_THRESHOLD:
            clear_short_candidates[ticker] = data
        else:
            ambiguous_candidates[ticker] = data
    
    # Get top N by magnitude from each pool
    top_clear_longs = sorted(
        clear_long_candidates.items(),
        key=lambda x: x[1].get("magnitude_score", 0),
        reverse=True
    )[:top_n_magnitude]
    
    top_clear_shorts = sorted(
        clear_short_candidates.items(),
        key=lambda x: x[1].get("magnitude_score", 0),
        reverse=True
    )[:top_n_magnitude]
    
    top_ambiguous = sorted(
        ambiguous_candidates.items(),
        key=lambda x: x[1].get("magnitude_score", 0),
        reverse=True
    )[:20]

    # Magnitude mode labels
    mag_modes = categorize_by_magnitude_mode(universe)
    p33, p67 = mag_modes["thresholds"]

    def mag_label(score):
        if score <= p33:
            return "LOW"
        elif score <= p67:
            return "MID"
        return "HIGH"

    lines = []
    lines.append(f"REBALANCE ID: {rebalance_id}")
    lines.append(f"MARKET REGIME: {regime}")
    lines.append(f"UNIVERSE SIZE: {len(universe)} stocks")
    lines.append(f"COST MODEL: 15 bps one-way (30 bps round-trip per position change)")
    lines.append("")

    lines.append(f"MAGNITUDE DISTRIBUTION:")
    lines.append(f"  LOW  (≤{p33:.1f}): {len(mag_modes['low'])} stocks  — mean-reversion mode")
    lines.append(f"  MID  ({p33:.1f}-{p67:.1f}): {len(mag_modes['mid'])} stocks  — noisy, generally AVOID")
    lines.append(f"  HIGH (>{p67:.1f}): {len(mag_modes['high'])} stocks  — momentum/breakout mode")
    lines.append("")

    # Previous portfolio for cost awareness
    if previous_portfolio:
        prev_longs = [p["ticker"] for p in previous_portfolio.get("long_basket", [])]
        prev_shorts = [p["ticker"] for p in previous_portfolio.get("short_basket", [])]
        lines.append(f"PREVIOUS LONG POSITIONS ({len(prev_longs)}): {', '.join(prev_longs)}")
        lines.append(f"  → LONGS: prefer KEEPING unless signal materially deteriorated")
        lines.append(f"PREVIOUS SHORT POSITIONS ({len(prev_shorts)}): {', '.join(prev_shorts)}")
        lines.append(f"  → SHORTS: prefer ROTATING — only keep if direction became MORE negative")
        lines.append("")

    # NEW: Candidate pool summary
    lines.append("=== CANDIDATE POOLS ===")
    lines.append(f"CLEAR LONG CANDIDATES (direction > {DIRECTION_THRESHOLD}): {len(clear_long_candidates)} stocks")
    lines.append(f"CLEAR SHORT CANDIDATES (direction < -{DIRECTION_THRESHOLD}): {len(clear_short_candidates)} stocks")
    lines.append(f"AMBIGUOUS CANDIDATES (direction ∈ [-{DIRECTION_THRESHOLD}, {DIRECTION_THRESHOLD}]): {len(ambiguous_candidates)} stocks")
    lines.append("")

    # Top clear longs
    lines.append(f"=== TOP {len(top_clear_longs)} CLEAR LONG CANDIDATES (direction > {DIRECTION_THRESHOLD}) ===")
    if top_clear_longs:
        # Header for the 17 unique indicators requested
        lines.append(f"{'Ticker':<8} {'Dir':>6} {'Mag':>6} {'Mode':<5} {'Mom3m':>8} {'RSI5':>6} {'ADX':>6} {'VolPctl':>8} {'VIX':>6} {'VixPctl':>8} {'VolImb':>7} {'M2/M1':>7} {'ATR':>7} {'RV10':>7} {'VAT_MA':>7} {'Rng55':>7} {'52WH':>7} {'Rng%':>7} {'AutoC':>7} {'Vel/V':>7} {'D52WH':>7} {'VATSlo':>7} {'%MA100':>7}")
        lines.append("-" * 165)
        for ticker, d in top_clear_longs:
            ind = d.get("indicators", {})
            m_score = d.get('magnitude_score', 0)
            lines.append(
                f"{ticker:<8} "
                f"{d.get('direction_score', 0):>6.1f} "
                f"{m_score:>6.1f} "
                f"{mag_label(m_score):<5} "
                # Row of 17 indicators
                f"{ind.get('mom_3m', 0):>8.3f} "
                f"{ind.get('rsi_5d', 0):>6.1f} "
                f"{ind.get('adx_14d', 0):>6.1f} "
                f"{ind.get('vol_percentile_504d', 0):>8.3f} "
                f"{ind.get('vix_level', 0):>6.1f} "
                f"{ind.get('vix_percentile_252d', 0):>8.3f} "
                f"{ind.get('vol_imbalance_20d', 0):>7.3f} "
                f"{ind.get('m2_m1_ratio', 0):>7.3f} "
                f"{ind.get('atr_14d_pct', 0):>7.3f} "
                f"{ind.get('realized_vol_10d', 0):>7.3f} "
                f"{ind.get('vol_adjusted_trend_ma100', 0):>7.3f} "
                f"{ind.get('range_pos_55d', 0):>7.3f} "
                f"{ind.get('pct_from_52w_high', 0):>7.3f} "
                f"{ind.get('range_pct', 0):>7.3f} "
                f"{ind.get('autocorr_5d', 0):>7.3f} "
                f"{ind.get('vel_10d_over_vol', 0):>7.3f} "
                f"{ind.get('days_since_52w_high', 0):>7.0f} "
                f"{ind.get('vol_adjusted_trend_ma100_slope', 0):>7.3f} "
                f"{ind.get('pct_from_ma100', 0):>7.3f}"
            )
    lines.append("")

    # Top clear shorts
    lines.append(f"=== TOP {len(top_clear_shorts)} CLEAR SHORT CANDIDATES (direction < -{DIRECTION_THRESHOLD}) ===")
    if top_clear_shorts:
        lines.append(f"{'Ticker':<8} {'Dir':>6} {'Mag':>6} {'Mode':<5} {'Mom3m':>8} {'RSI5':>6} {'ADX':>6} {'VolPctl':>8} {'VIX':>6} {'VixPctl':>8} {'VolImb':>7} {'M2/M1':>7} {'ATR':>7} {'RV10':>7} {'VAT_MA':>7} {'Rng55':>7} {'52WH':>7} {'Rng%':>7} {'AutoC':>7} {'Vel/V':>7} {'D52WH':>7} {'VATSlo':>7} {'%MA100':>7}")
        lines.append("-" * 165)
        for ticker, d in top_clear_shorts:
            ind = d.get("indicators", {})
            m_score = d.get('magnitude_score', 0)
            lines.append(
                f"{ticker:<8} "
                f"{d.get('direction_score', 0):>6.1f} "
                f"{m_score:>6.1f} "
                f"{mag_label(m_score):<5} "
                f"{ind.get('mom_3m', 0):>8.3f} "
                f"{ind.get('rsi_5d', 0):>6.1f} "
                f"{ind.get('adx_14d', 0):>6.1f} "
                f"{ind.get('vol_percentile_504d', 0):>8.3f} "
                f"{ind.get('vix_level', 0):>6.1f} "
                f"{ind.get('vix_percentile_252d', 0):>8.3f} "
                f"{ind.get('vol_imbalance_20d', 0):>7.3f} "
                f"{ind.get('m2_m1_ratio', 0):>7.3f} "
                f"{ind.get('atr_14d_pct', 0):>7.3f} "
                f"{ind.get('realized_vol_10d', 0):>7.3f} "
                f"{ind.get('vol_adjusted_trend_ma100', 0):>7.3f} "
                f"{ind.get('range_pos_55d', 0):>7.3f} "
                f"{ind.get('pct_from_52w_high', 0):>7.3f} "
                f"{ind.get('range_pct', 0):>7.3f} "
                f"{ind.get('autocorr_5d', 0):>7.3f} "
                f"{ind.get('vel_10d_over_vol', 0):>7.3f} "
                f"{ind.get('days_since_52w_high', 0):>7.0f} "
                f"{ind.get('vol_adjusted_trend_ma100_slope', 0):>7.3f} "
                f"{ind.get('pct_from_ma100', 0):>7.3f}"
            )
    lines.append("")

    # Ambiguous candidates (discretionary)
    if top_ambiguous:
        lines.append(f"=== TOP {len(top_ambiguous)} AMBIGUOUS CANDIDATES (discretionary trading, direction ∈ [-{DIRECTION_THRESHOLD}, {DIRECTION_THRESHOLD}]) ===")
        lines.append(f"{'Ticker':<8} {'Dir':>6} {'Mag':>6} {'Mode':<5} {'Mom3m':>8} {'RSI5':>6} {'ADX':>6} {'VolPctl':>8} {'VIX':>6} {'VixPctl':>8} {'VolImb':>7} {'M2/M1':>7} {'ATR':>7} {'RV10':>7} {'VAT_MA':>7} {'Rng55':>7} {'52WH':>7} {'Rng%':>7} {'AutoC':>7} {'Vel/V':>7} {'D52WH':>7} {'VATSlo':>7} {'%MA100':>7}")
        lines.append("-" * 165)
        for ticker, d in top_ambiguous:
            ind = d.get("indicators", {})
            m_score = d.get('magnitude_score', 0)
            lines.append(
                f"{ticker:<8} "
                f"{d.get('direction_score', 0):>6.1f} "
                f"{m_score:>6.1f} "
                f"{mag_label(m_score):<5} "
                f"{ind.get('mom_3m', 0):>8.3f} "
                f"{ind.get('rsi_5d', 0):>6.1f} "
                f"{ind.get('adx_14d', 0):>6.1f} "
                f"{ind.get('vol_percentile_504d', 0):>8.3f} "
                f"{ind.get('vix_level', 0):>6.1f} "
                f"{ind.get('vix_percentile_252d', 0):>8.3f} "
                f"{ind.get('vol_imbalance_20d', 0):>7.3f} "
                f"{ind.get('m2_m1_ratio', 0):>7.3f} "
                f"{ind.get('atr_14d_pct', 0):>7.3f} "
                f"{ind.get('realized_vol_10d', 0):>7.3f} "
                f"{ind.get('vol_adjusted_trend_ma100', 0):>7.3f} "
                f"{ind.get('range_pos_55d', 0):>7.3f} "
                f"{ind.get('pct_from_52w_high', 0):>7.3f} "
                f"{ind.get('range_pct', 0):>7.3f} "
                f"{ind.get('autocorr_5d', 0):>7.3f} "
                f"{ind.get('vel_10d_over_vol', 0):>7.3f} "
                f"{ind.get('days_since_52w_high', 0):>7.0f} "
                f"{ind.get('vol_adjusted_trend_ma100_slope', 0):>7.3f} "
                f"{ind.get('pct_from_ma100', 0):>7.3f}"
            )
        lines.append("")

    # Rules
    lines.append("PORTFOLIO CONSTRUCTION RULES:")
    lines.append(f"1. PRIMARY: Pick 5-30 from CLEAR LONG CANDIDATES for longs, 5-30 from CLEAR SHORT CANDIDATES for shorts")
    lines.append(f"2. DISCRETIONARY: May use up to 5 from AMBIGUOUS pool only if VERY high magnitude + strong confluence")
    lines.append(f"3. NO CROSS-LEG OVERLAP: Each ticker in only ONE leg")
    lines.append(f"4. WEIGHTS: Must sum to exactly 1.0 per leg")
    lines.append(f"5. COST-AWARE: Prefer keeping previous positions (saves 30bps per swap)")
    lines.append(f"6. SIGNAL QUALITY: Avoid weak signals, prioritize HIGH magnitude trades")

    return "\n".join(lines)



async def construct_portfolio_from_universe(
    universe: dict,
    rebalance_date: str,
    previous_portfolio: dict = None,
    regime: str = "unknown",
    mask_date: bool = True,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    top_n_magnitude: int = 60,
    use_equal_weight: bool = False,
) -> dict:
    """Send the full universe to the LLM and get back a weighted long/short portfolio.

    Args:
        universe: {ticker: {direction_score, magnitude_score, indicators: {...}}}
        rebalance_date: ISO date string (will be masked if mask_date=True)
        previous_portfolio: Previous period's portfolio dict for cost-aware rebalancing
        regime: Market regime string (crisis/squeeze/grind_up/slow_bleed/normal/unknown)
        mask_date: If True, replace date with opaque ID
        model: model name
        temperature: Lower = more deterministic (0.3 recommended for portfolio decisions)
        top_n_magnitude: How many top-magnitude stocks to show in detail
        use_equal_weight: If True, override LLM weights with equal weight per leg

    Returns:
        {
            "rebalance_date": str,
            "long_basket": [{"ticker": str, "weight": float, "rationale": str}],
            "short_basket": [{"ticker": str, "weight": float, "rationale": str}],
            "kept_from_previous": [str],
            "dropped_from_previous": [str],
            "reasoning": str,
            "regime_adjustment": str,
            "confidence": float,
            "raw_llm_response": dict
        }
    """

    if mask_date:
        rebalance_id = f"w_{abs(hash(rebalance_date)) % 100000:05d}"
    else:
        rebalance_id = rebalance_date

    # Format universe data
    universe_text = _format_universe_for_prompt(
        universe, previous_portfolio, regime, rebalance_id, top_n_magnitude
    )

    valid_tickers = set(universe.keys())

    user_content = f"""Construct a long/short portfolio from this universe.

{universe_text}

RULES:
- Pick 5-30 stocks for LONG leg, 5-30 for SHORT leg
- Weights MUST sum to 1.0 per leg
- Only use tickers from the universe above
- If previous positions are shown, prefer keeping them unless the signal has materially deteriorated (saves ~30 bps per swap)
- Return ONLY valid JSON matching the OUTPUT_SCHEMA
"""

    messages = [
        {"role": "system", "content": PORTFOLIO_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }

    url = get_deepseek_url()
    api_key = get_deepseek_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}

    validation_fail_count = 0  # Track consecutive validation failures to avoid API cost blowup

    async with httpx.AsyncClient(timeout=PORTFOLIO_REQUEST_TIMEOUT) as client:
        for attempt in range(5):
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                
                # DeepSeek response format: {"choices": [{"message": {"content": "..."}}]}
                if "choices" in result and result["choices"]:
                    content = result["choices"][0].get("message", {}).get("content", "")
                else:
                    raise ValueError(f"No content in response: {result}")
                
                if not content:
                    raise ValueError(f"No content in response: {result}")
                
                # Try to extract JSON from response
                try:
                    raw_output = json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, try to find JSON block in content
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        raw_output = json.loads(json_match.group())
                    else:
                        raise json.JSONDecodeError("No JSON found", content, 0)

                # Validate and clean the output
                long_basket = raw_output.get("long_basket", [])
                short_basket = raw_output.get("short_basket", [])

                if not isinstance(long_basket, list) or not isinstance(short_basket, list):
                    raise ValueError("long_basket and short_basket must be lists")
                
                # Filter to valid tickers only
                long_basket_before = len(long_basket)
                short_basket_before = len(short_basket)
                
                long_basket = [
                    p for p in long_basket
                    if isinstance(p, dict) and p.get("ticker", "").upper() in valid_tickers
                ]
                short_basket = [
                    p for p in short_basket
                    if isinstance(p, dict) and p.get("ticker", "").upper() in valid_tickers
                ]
                
                # Normalize ticker case
                for p in long_basket:
                    p["ticker"] = p["ticker"].upper()
                    p["weight"] = abs(float(p.get("weight", 1.0 / max(len(long_basket), 1))))
                    p["rationale"] = str(p.get("rationale", ""))
                for p in short_basket:
                    p["ticker"] = p["ticker"].upper()
                    p["weight"] = abs(float(p.get("weight", 1.0 / max(len(short_basket), 1))))
                    p["rationale"] = str(p.get("rationale", ""))

                # Enforce count limits
                if len(long_basket) < 5 or len(short_basket) < 5:
                    raise ValueError(
                        f"Need ≥5 per leg, got {len(long_basket)} long / {len(short_basket)} short"
                    )
                long_basket = long_basket[:30]
                short_basket = short_basket[:30]

                # ── State machine: deduplicate and enforce no cross-leg overlap ──
                long_basket, short_basket = validate_and_deduplicate_portfolio(
                    long_basket, short_basket, previous_portfolio
                )

                # Re-check counts after deduplication
                if len(long_basket) < 5 or len(short_basket) < 5:
                    raise ValueError(
                        f"Post-dedup: {len(long_basket)} long / {len(short_basket)} short (need ≥5)"
                    )

                # ── PRE-TRADE VALIDATION: Signal integrity check ──
                is_valid, validation_reason = validate_portfolio_signals(
                    long_basket, short_basket, universe, previous_portfolio
                )
                if not is_valid:
                    print(f"      ⚠️  Portfolio signal validation failed: {validation_reason}")
                    raise ValueError(f"Signal validation failed: {validation_reason}")

                # ── Equal-weight override if requested ──
                if use_equal_weight:
                    n_long = len(long_basket)
                    n_short = len(short_basket)
                    for p in long_basket:
                        p["weight"] = 1.0 / n_long
                    for p in short_basket:
                        p["weight"] = 1.0 / n_short

                # ── Count cost-driven positions ──
                cost_driven_count = sum(
                    1 for p in long_basket + short_basket
                    if p.get("rationale", "").upper().startswith("COST_DRIVEN")
                )

                return {
                    "rebalance_date": rebalance_date,
                    "rebalance_id": rebalance_id,
                    "regime": regime,
                    "long_basket": long_basket,
                    "short_basket": short_basket,
                    "kept_from_previous": raw_output.get("kept_from_previous", []),
                    "dropped_from_previous": raw_output.get("dropped_from_previous", []),
                    "reasoning": str(raw_output.get("reasoning", "")),
                    "regime_adjustment": str(raw_output.get("regime_adjustment", "")),
                    "confidence": clamp(float(raw_output.get("confidence", 0.5)), 0.0, 1.0),
                    "cost_driven_positions": cost_driven_count,
                    "raw_llm_response": raw_output,
                }

            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                error_msg = str(e)
                is_validation_error = "non-bullish direction" in error_msg.lower() or "non-bearish direction" in error_msg.lower()
                
                # Track validation failures to avoid API cost blowup
                if is_validation_error:
                    validation_fail_count += 1
                else:
                    validation_fail_count = 0  # Reset on non-validation errors
                
                print(f"   ⚠️  Attempt {attempt+1}/5 failed: {e}")
                
                # Early fallback if validation fails 2+ times (saves API cost)
                if validation_fail_count >= 2:
                    print(f"   ❌ Portfolio signal validation failed {validation_fail_count}× in a row, falling back to mechanical")
                    return _mechanical_fallback(universe, rebalance_date, rebalance_id, regime)
                
                if attempt < 4:
                    # Wait longer between retries — llama4 might be crashing
                    wait_time = 3 * (2 ** attempt)
                    print(f"   ⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    
                    # Build error-specific retry instruction
                    if is_validation_error:
                        if "non-bullish direction" in error_msg.lower():
                            # Extract ticker from error message: "Long position TICKER has non-bullish..."
                            ticker_match = error_msg.split("Long position ")
                            if len(ticker_match) > 1:
                                ticker = ticker_match[1].split(" ")[0]
                                retry_instruction = (
                                    f"ERROR: Stock {ticker} in your LONG basket has direction_score ≤ 0. "
                                    f"LONGS must have direction_score > 0 (bullish signals). "
                                    f"Either remove {ticker} from LONG, or move it to SHORT if its signal is bearish. "
                                    f"Return corrected JSON only."
                                )
                            else:
                                retry_instruction = (
                                    "ERROR: You placed a stock with non-bullish (negative) direction_score in LONG basket. "
                                    "LONGS require direction_score > 0. Remove or move the stock. Return corrected JSON."
                                )
                        else:  # non-bearish direction
                            # Extract ticker from error message: "Short position TICKER has non-bearish..."
                            ticker_match = error_msg.split("Short position ")
                            if len(ticker_match) > 1:
                                ticker = ticker_match[1].split(" ")[0]
                                retry_instruction = (
                                    f"ERROR: Stock {ticker} in your SHORT basket has direction_score ≥ 0. "
                                    f"SHORTS must have direction_score < 0 (bearish signals). "
                                    f"Either remove {ticker} from SHORT, or move it to LONG if its signal is bullish. "
                                    f"Return corrected JSON only."
                                )
                            else:
                                retry_instruction = (
                                    "ERROR: You placed a stock with non-bearish (positive) direction_score in SHORT basket. "
                                    "SHORTS require direction_score < 0. Remove or move the stock. Return corrected JSON."
                                )
                    else:
                        # Generic JSON/format error
                        retry_instruction = (
                            "Your previous response was invalid. Return ONLY valid JSON with "
                            "long_basket and short_basket arrays. Each entry needs ticker, weight, "
                            "rationale. Weights per leg must sum to 1.0. Min 5 stocks per leg."
                        )
                    
                    # Add retry instruction
                    payload["messages"].append({
                        "role": "user",
                        "content": retry_instruction,
                    })
                    continue

                # All retries exhausted — return equal-weight fallback from top candidates
                print(f"   ❌ All retries failed for {rebalance_id}, using mechanical fallback")
                return _mechanical_fallback(universe, rebalance_date, rebalance_id, regime)

            except httpx.HTTPStatusError as e:
                print(f"   ❌ HTTP {e.response.status_code}: {e.response.text[:200]}")
                if attempt < 4:
                    wait_time = 5 * (2 ** attempt)
                    print(f"   ⏳ Waiting {wait_time}s before retry (model may be restarting)...")
                    await asyncio.sleep(wait_time)
                    continue
                return _mechanical_fallback(universe, rebalance_date, rebalance_id, regime)

            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
                if attempt < 4:
                    wait_time = 5 * (2 ** attempt)
                    print(f"   ⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                return _mechanical_fallback(universe, rebalance_date, rebalance_id, regime)

    return _mechanical_fallback(universe, rebalance_date, rebalance_id, regime)


def _mechanical_fallback(universe: dict, rebalance_date: str, rebalance_id: str, regime: str) -> dict:
    """Fallback: equal-weight top 30 long / bottom 30 short by direction_score from top magnitude pool."""
    sorted_by_mag = sorted(
        universe.keys(),
        key=lambda t: universe[t].get("magnitude_score", 0),
        reverse=True
    )
    candidates = sorted_by_mag[:60]
    candidates_by_dir = sorted(
        candidates,
        key=lambda t: universe[t].get("direction_score", 0),
        reverse=True
    )
    long_tickers = candidates_by_dir[:30]
    short_tickers = candidates_by_dir[-30:]
    w = 1.0 / 30

    return {
        "rebalance_date": rebalance_date,
        "rebalance_id": rebalance_id,
        "regime": regime,
        "long_basket": [{"ticker": t, "weight": w, "rationale": "mechanical fallback"} for t in long_tickers],
        "short_basket": [{"ticker": t, "weight": w, "rationale": "mechanical fallback"} for t in short_tickers],
        "kept_from_previous": [],
        "dropped_from_previous": [],
        "reasoning": "LLM failed, using mechanical top/bottom 30 by direction from top 60 magnitude",
        "regime_adjustment": "none (fallback)",
        "confidence": 0.0,
        "raw_llm_response": {},
    }


# =============================================================================
# CLI Test Interface
# =============================================================================

async def main():
    """Test the function with a sample technical payload."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run technical LLM analysis on JSON file")
    parser.add_argument("--input", required=True, help="Path to technical JSON file (e.g., out/UNH_latest.json)")
    parser.add_argument("--output", help="Optional output file for verdict (default: print to stdout)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Load input
    with open(args.input, "r") as f:
        input_data = json.load(f)
    
    print(f"Analyzing {input_data.get('metadata', {}).get('symbol', 'UNKNOWN')}...")
    
    # Run analysis
    verdict = await run_technical_llm(input_data, model=args.model, temperature=args.temperature)
    
    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(verdict, f, indent=2)
        print(f"Verdict written to {args.output}")
    else:
        print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    asyncio.run(main())