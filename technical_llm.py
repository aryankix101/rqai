import os
import json
import re
import asyncio
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Tuple
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
DEFAULT_MODEL = "deepseek-reasoner"
DEFAULT_TEMPERATURE = 0.6
REQUEST_TIMEOUT = 300.0
PORTFOLIO_REQUEST_TIMEOUT = 300.0  
MAX_PORTFOLIO_RETRIES = 5

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


def model_supports_temperature(model: str) -> bool:
    """DeepSeek reasoner ignores temperature; omit it to match the API contract."""
    return str(model or "").strip() != "deepseek-reasoner"


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
    Relaxed validation: only ensure selected tickers are still in the provided universe.

    The LLM is allowed to choose contrarian longs/shorts when broader indicator context,
    regime, retention effects, or other portfolio-level considerations justify it.

    Args:
        long_basket: [{ticker, weight, rationale}, ...]
        short_basket: [{ticker, weight, rationale}, ...]
        universe: {ticker: {direction_score, ...}}
        previous_portfolio: Unused, kept for API compatibility

    Returns:
        (is_valid: bool, reason: str)
    """
    for pos in long_basket:
        ticker = pos.get("ticker", "").upper()
        if ticker not in universe:
            return False, f"Long position {ticker} not in universe"

    for pos in short_basket:
        ticker = pos.get("ticker", "").upper()
        if ticker not in universe:
            return False, f"Short position {ticker} not in universe"

    return True, "Relaxed validation passed"


def categorize_by_magnitude_mode(universe: dict) -> dict:
    """
    Segment universe into three magnitude modes.

    Returns:
        {
            "low":  {ticker: data, ...},   # mean-reversion scalps
            "mid":  {ticker: data, ...},   # mixed / noisy, use with caution
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

SIGNAL_STAGE_SYSTEM_PROMPT_TEMPLATE = """
ALPHA DISCOVERY STAGE
You are smarter than Claude Shannon. You are trying to outperform his portfolio which outperformed Warren Buffet's portfolio. 
Use only the provided point-in-time inputs to identify the strongest current long
and short candidates for the next rebalance horizon.
Treat tickers as opaque identifiers. Do not use outside knowledge, calendar knowledge, or ticker familiarity.

MECHANICAL BASELINE:
- Sort by magnitude_score descending
- Take the top-N names
- LONG the strongest direction_scores, SHORT the weakest
- Equal-weight each leg
- Ignore signal quality, regime context, and implementation details

ALPHA RULES:
1. Signal quality first:
   Prefer clean confluence across direction_score, momentum, trend,
   participation, indicators, and volatility context.
   Penalize unstable or conflicting setups.

2. Directional discipline:
   Longs should come from positive direction candidates and shorts from
   negative direction candidates.
   Contrarian selections are allowed only when the broader indicator context
   clearly supports them.

__PIPELINE_CONTEXT__

3. Regime and portfolio risk:
   Adjust aggressiveness to the provided market regime.
   In stressed or high-volatility regimes, demand cleaner setups and lower
   concentration.
   Avoid filling the slate with many names that look technically identical.

4. Loss memory:
   If similar prior loss cases are shown, treat them as cautionary analogs.
   Do not blindly avoid the trade, but require clearly better signal quality than
   the failed analogs before repeating the setup.

OUTPUT FORMAT (STRICT JSON):
{
  "long_candidates": [
    {"ticker": "AAPL", "edge_score": 88, "rationale": "Strong trend + momentum confluence"},
    ...
  ],
  "short_candidates": [
    {"ticker": "XYZ", "edge_score": 84, "rationale": "Weak trend + negative momentum"},
    ...
  ],
  "reasoning": "2-3 sentence summary of the alpha slate",
  "regime_adjustment": "How the regime affected your candidate quality threshold",
}
"""

SIGNAL_STAGE_SIGNAL_ONLY_CONTEXT = """
4. PIPELINE CONTEXT
   This is the only decision stage for this run.
   Ignore previous holdings, turnover, and transaction costs.
   Optimize for current cross-sectional edge and expected gross information ratio.
"""

SIGNAL_STAGE_TURNOVER_CONTEXT = """
4. PIPELINE CONTEXT
   This is stage 1 of 2.
   Ignore previous holdings, turnover, and transaction costs in this stage.
   Your job here is raw alpha discovery only; a later implementation overlay
   will decide which incumbents to keep or replace.
"""

TURNOVER_OVERLAY_SYSTEM_PROMPT = """
TURNOVER OVERLAY STAGE
You are NOT discovering alpha from scratch in this stage.
The alpha slate has already been produced. Your job is only to decide the final
implemented long and short books by comparing current incumbents against the
provided challengers.

PRIMARY OBJECTIVE:
Maximize expected net risk-adjusted return after implementation frictions.

DECISION RULES:
1. Treat previous holdings as incumbents.
2. A swap is justified only when the challenger is clearly better after:
   - about 15 bps one-way / 30 bps round-trip turnover cost
   - an uncertainty buffer for model error
3. LONGS:
   - Bias toward keeping if signal quality remains acceptable.
   - Do not keep a deteriorating long just to save cost.
4. SHORTS:
   - Bias toward replacing stale shorts.
   - Keep a short only if the bearish case is still strong or stronger.
5. Only use tickers from the provided incumbent and challenger pools.
6. Keep rationales short. If you keep a position mainly because replacement edge
   is too small after costs, prefix the rationale with "COST_DRIVEN: ".

OUTPUT FORMAT (STRICT JSON):
{
  "long_basket": [
    {"ticker": "AAPL", "rationale": "Still acceptable edge after costs"},
    ...
  ],
  "short_basket": [
    {"ticker": "XYZ", "rationale": "Challenger clearly better after costs"},
    ...
  ],
  "reasoning": "2-3 sentence implementation summary",
  "regime_adjustment": "How the regime affected keep-vs-replace decisions",
}
"""

SIGNAL_STAGE_OUTPUT_SCHEMA = {
    "long_candidates": [{"ticker": "str", "edge_score": "float [0,100]", "rationale": "str"}],
    "short_candidates": [{"ticker": "str", "edge_score": "float [0,100]", "rationale": "str"}],
    "reasoning": "str",
    "regime_adjustment": "str",
}

OVERLAY_STAGE_OUTPUT_SCHEMA = {
    "long_basket": [{"ticker": "str", "rationale": "str"}],
    "short_basket": [{"ticker": "str", "rationale": "str"}],
    "reasoning": "str",
    "regime_adjustment": "str",
}

PORTFOLIO_OUTPUT_SCHEMA = {
    "long_basket": [{"ticker": "str", "weight": "float [0,1]", "rationale": "str"}],
    "short_basket": [{"ticker": "str", "weight": "float [0,1]", "rationale": "str"}],
    "kept_from_previous": ["str"],
    "dropped_from_previous": ["str"],
    "reasoning": "str",
    "regime_adjustment": "str",
}


def build_signal_stage_system_prompt(turnover_aware: bool) -> str:
    """Build the stage-1 alpha discovery prompt."""
    pipeline_context = (
        SIGNAL_STAGE_TURNOVER_CONTEXT if turnover_aware else SIGNAL_STAGE_SIGNAL_ONLY_CONTEXT
    )
    return SIGNAL_STAGE_SYSTEM_PROMPT_TEMPLATE.replace("__PIPELINE_CONTEXT__", pipeline_context.strip())


def build_turnover_overlay_system_prompt() -> str:
    """Build the stage-2 turnover overlay prompt."""
    return TURNOVER_OVERLAY_SYSTEM_PROMPT


def build_portfolio_system_prompt(turnover_aware: bool) -> str:
    """Backward-compatible wrapper for the stage-1 prompt builder."""
    return build_signal_stage_system_prompt(turnover_aware)


def _normalize_ticker(value: Any) -> str:
    return str(value or "").upper().strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _deduplicate_cross_leg_entries(long_entries: list, short_entries: list) -> Tuple[list, list]:
    """Keep first occurrence per ticker, with long side winning cross-leg collisions."""
    seen = set()
    clean_long = []
    clean_short = []

    for entry in long_entries:
        ticker = _normalize_ticker(entry.get("ticker"))
        if ticker and ticker not in seen:
            normalized = dict(entry)
            normalized["ticker"] = ticker
            clean_long.append(normalized)
            seen.add(ticker)

    for entry in short_entries:
        ticker = _normalize_ticker(entry.get("ticker"))
        if ticker and ticker not in seen:
            normalized = dict(entry)
            normalized["ticker"] = ticker
            clean_short.append(normalized)
            seen.add(ticker)

    return clean_long, clean_short


def _candidate_utility(edge_score: float) -> float:
    return max(clamp(edge_score, 0.0, 100.0), 1e-6)


def _fallback_signal_utility(signal_data: dict, side: str) -> float:
    direction = _safe_float(signal_data.get("direction_score"), 0.0)
    magnitude = clamp(_safe_float(signal_data.get("magnitude_score"), 0.0) / 100.0, 0.0, 1.0)
    signed_strength = direction / 100.0 if side == "LONG" else -direction / 100.0
    directional = clamp(signed_strength, 0.0, 1.0)
    return max(0.05, 0.7 * directional + 0.3 * magnitude)


def _parse_json_content(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise


def _aggregate_llm_usage(stage_usages: Dict[str, dict]) -> dict:
    aggregate = {
        "stages": {},
        "provider_usage": {},
        "prompt_chars": 0,
        "completion_chars": 0,
        "approx_prompt_tokens": 0,
        "approx_completion_tokens": 0,
        "approx_total_tokens": 0,
    }

    for stage_name, usage in stage_usages.items():
        if not usage:
            continue
        aggregate["stages"][stage_name] = usage
        for key in ("prompt_chars", "completion_chars", "approx_prompt_tokens", "approx_completion_tokens", "approx_total_tokens"):
            aggregate[key] += int(usage.get(key, 0))
        provider_usage = usage.get("provider_usage", {})
        if isinstance(provider_usage, dict):
            for key, value in provider_usage.items():
                if isinstance(value, (int, float)):
                    aggregate["provider_usage"][key] = aggregate["provider_usage"].get(key, 0) + value

    return aggregate


async def _call_llm_json_with_validation(
    client: httpx.AsyncClient,
    model: str,
    temperature: float,
    base_messages: list,
    retry_instruction: str,
    stage_label: str,
    validate_output: Callable[[dict], dict],
) -> dict:
    """Call the LLM, parse JSON, validate the stage-specific contract, and retry if needed."""
    payload = {
        "model": model,
        "messages": list(base_messages),
        "stream": False,
    }
    if model_supports_temperature(model):
        payload["temperature"] = temperature

    url = get_deepseek_url()
    api_key = get_deepseek_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(MAX_PORTFOLIO_RETRIES):
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            provider_usage = result.get("usage", {})

            if "choices" in result and result["choices"]:
                content = result["choices"][0].get("message", {}).get("content", "")
            else:
                raise ValueError(f"No content in response: {result}")

            if not content:
                raise ValueError(f"No content in response: {result}")

            raw_output = _parse_json_content(content)
            cleaned_output = validate_output(raw_output)

            prompt_chars = sum(len(str(message.get("content", ""))) for message in payload["messages"])
            completion_chars = len(content)
            cleaned_output["raw_llm_response"] = raw_output
            cleaned_output["llm_usage"] = {
                "attempt_used": attempt + 1,
                "provider_usage": provider_usage if isinstance(provider_usage, dict) else {},
                "prompt_chars": prompt_chars,
                "completion_chars": completion_chars,
                "approx_prompt_tokens": (prompt_chars + 3) // 4,
                "approx_completion_tokens": (completion_chars + 3) // 4,
                "approx_total_tokens": (prompt_chars + completion_chars + 3) // 4,
            }
            return cleaned_output

        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            print(f"   ⚠️  {stage_label} attempt {attempt+1}/{MAX_PORTFOLIO_RETRIES} failed: {e}")
            if attempt < MAX_PORTFOLIO_RETRIES - 1:
                wait_time = 3 * (2 ** attempt)
                print(f"   ⏳ Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
                payload["messages"] = list(base_messages) + [{
                    "role": "user",
                    "content": retry_instruction,
                }]
                continue
            raise RuntimeError(
                f"{stage_label} failed after {MAX_PORTFOLIO_RETRIES} attempts: {e}"
            ) from e

        except httpx.HTTPStatusError as e:
            print(f"   ❌ {stage_label} HTTP {e.response.status_code}: {e.response.text[:200]}")
            if attempt < MAX_PORTFOLIO_RETRIES - 1:
                wait_time = 5 * (2 ** attempt)
                print(f"   ⏳ Waiting {wait_time}s before retry (model may be restarting)...")
                await asyncio.sleep(wait_time)
                continue
            raise RuntimeError(
                f"{stage_label} failed after {MAX_PORTFOLIO_RETRIES} attempts: HTTP {e.response.status_code}"
            ) from e


def derive_portfolio_transition(
    previous_portfolio: dict,
    long_basket: list,
    short_basket: list,
) -> Tuple[list, list]:
    """Derive kept/dropped names from the executed baskets."""
    if not previous_portfolio:
        return [], []

    prev_longs = [
        str(position.get("ticker", "")).upper()
        for position in previous_portfolio.get("long_basket", [])
        if position.get("ticker")
    ]
    prev_shorts = [
        str(position.get("ticker", "")).upper()
        for position in previous_portfolio.get("short_basket", [])
        if position.get("ticker")
    ]
    new_longs = {
        str(position.get("ticker", "")).upper()
        for position in long_basket
        if position.get("ticker")
    }
    new_shorts = {
        str(position.get("ticker", "")).upper()
        for position in short_basket
        if position.get("ticker")
    }

    kept = [ticker for ticker in prev_longs if ticker in new_longs]
    kept.extend(ticker for ticker in prev_shorts if ticker in new_shorts)

    dropped = [ticker for ticker in prev_longs if ticker not in new_longs]
    dropped.extend(ticker for ticker in prev_shorts if ticker not in new_shorts)
    return kept, dropped


def _format_universe_for_prompt(
    universe: dict,
    previous_portfolio: dict = None,
    regime: str = "unknown",
    rebalance_id: str = "w_00001",
    top_n_magnitude: int = 60,
    turnover_aware: bool = False,
    loss_memory_context: str = "",
) -> str:
    """Format universe data into a readable prompt for stage-1 alpha discovery.

    Separates stocks by direction signal to guide LLM toward correct long/short assignments.

    Args:
        universe: {ticker: {direction_score, magnitude_score, indicators...}}
        previous_portfolio: Unused. Kept for backwards-compatible call sites.
        regime: Current market regime string
        rebalance_id: Opaque date identifier
        top_n_magnitude: How many top-magnitude stocks to show per candidate pool
        turnover_aware: If True, note that a stage-2 overlay will handle turnover later
        loss_memory_context: Optional prompt appendix with retrieved failure cases

    Returns:
        Formatted string for the LLM prompt with separated candidate pools
    """
    DIRECTION_THRESHOLD = 5.0
    
    # Separate into candidate pools by direction signal
    clear_long_candidates = {}
    clear_short_candidates = {}
    
    for ticker, data in universe.items():
        direction = data.get("direction_score", 0)
        if direction > DIRECTION_THRESHOLD:
            clear_long_candidates[ticker] = data
        elif direction < -DIRECTION_THRESHOLD:
            clear_short_candidates[ticker] = data
    
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
    if turnover_aware:
        lines.append("PIPELINE MODE: stage 1 of 2 (alpha discovery first, turnover overlay later)")
        lines.append("TURNOVER OBJECTIVE: ignore trading costs in this stage; implementation will be handled later")
    else:
        lines.append("PIPELINE MODE: single-stage alpha discovery")
        lines.append("TURNOVER OBJECTIVE: ignore trading costs and optimize gross signal quality only")
    lines.append("")

    lines.append(f"MAGNITUDE DISTRIBUTION:")
    lines.append(f"  LOW  (≤{p33:.1f}): {len(mag_modes['low'])} stocks  — mean-reversion mode")
    lines.append(f"  MID  ({p33:.1f}-{p67:.1f}): {len(mag_modes['mid'])} stocks  — mixed / noisy, soft penalty only")
    lines.append(f"  HIGH (>{p67:.1f}): {len(mag_modes['high'])} stocks  — momentum/breakout mode")
    lines.append("")

    if loss_memory_context:
        lines.append("=== SIMILAR PRIOR LOSS CASES ===")
        lines.append(loss_memory_context)
        lines.append("")

    # NEW: Candidate pool summary
    lines.append("=== CANDIDATE POOLS ===")
    lines.append(f"CLEAR LONG CANDIDATES (direction > {DIRECTION_THRESHOLD}): {len(clear_long_candidates)} stocks")
    lines.append(f"CLEAR SHORT CANDIDATES (direction < -{DIRECTION_THRESHOLD}): {len(clear_short_candidates)} stocks")
    lines.append("")
    lines.append("INDICATOR SCALE: non-score indicator columns below are percentile-normalized to [0,1] when available.")
    lines.append("")

    indicator_header = (
        f"{'Ticker':<8} {'Dir':>6} {'Mag':>6} {'Mode':<5} "
        f"{'Mom3m':>8} {'RSI5':>7} {'ADX':>7} {'VolPctl':>8} {'VIX':>7} {'VixPctl':>8} "
        f"{'VolImb':>7} {'M2/M1':>7} {'ATR':>7} {'RV10':>7} {'VAT_MA':>7} {'Rng55':>7} "
        f"{'52WH':>7} {'Rng%':>7} {'AutoC':>7} {'Vel/V':>7} {'D52WH':>7} {'VATSlo':>7} {'%MA100':>7}"
    )

    def indicator_value(indicators: dict, key: str) -> float:
        return _safe_float(indicators.get(key), 0.0)

    def format_candidate_row(ticker: str, payload: dict) -> str:
        ind = payload.get("indicators", {})
        magnitude = payload.get("magnitude_score", 0)
        return (
            f"{ticker:<8} "
            f"{payload.get('direction_score', 0):>6.1f} "
            f"{magnitude:>6.1f} "
            f"{mag_label(magnitude):<5} "
            f"{indicator_value(ind, 'mom_3m'):>8.3f} "
            f"{indicator_value(ind, 'rsi_5d'):>7.3f} "
            f"{indicator_value(ind, 'adx_14d'):>7.3f} "
            f"{indicator_value(ind, 'vol_percentile_504d'):>8.3f} "
            f"{indicator_value(ind, 'vix_level'):>7.3f} "
            f"{indicator_value(ind, 'vix_percentile_252d'):>8.3f} "
            f"{indicator_value(ind, 'vol_imbalance_20d'):>7.3f} "
            f"{indicator_value(ind, 'm2_m1_ratio'):>7.3f} "
            f"{indicator_value(ind, 'atr_14d_pct'):>7.3f} "
            f"{indicator_value(ind, 'realized_vol_10d'):>7.3f} "
            f"{indicator_value(ind, 'vol_adjusted_trend_ma100'):>7.3f} "
            f"{indicator_value(ind, 'range_pos_55d'):>7.3f} "
            f"{indicator_value(ind, 'pct_from_52w_high'):>7.3f} "
            f"{indicator_value(ind, 'range_pct'):>7.3f} "
            f"{indicator_value(ind, 'autocorr_5d'):>7.3f} "
            f"{indicator_value(ind, 'vel_10d_over_vol'):>7.3f} "
            f"{indicator_value(ind, 'days_since_52w_high'):>7.3f} "
            f"{indicator_value(ind, 'vol_adjusted_trend_ma100_slope'):>7.3f} "
            f"{indicator_value(ind, 'pct_from_ma100'):>7.3f}"
        )

    # Top clear longs
    lines.append(f"=== TOP {len(top_clear_longs)} CLEAR LONG CANDIDATES (direction > {DIRECTION_THRESHOLD}) ===")
    if top_clear_longs:
        # Header for the configured indicator set
        lines.append(indicator_header)
        lines.append("-" * 165)
        for ticker, d in top_clear_longs:
            lines.append(format_candidate_row(ticker, d))
    lines.append("")

    # Top clear shorts
    lines.append(f"=== TOP {len(top_clear_shorts)} CLEAR SHORT CANDIDATES (direction < -{DIRECTION_THRESHOLD}) ===")
    if top_clear_shorts:
        lines.append(indicator_header)
        lines.append("-" * 165)
        for ticker, d in top_clear_shorts:
            lines.append(format_candidate_row(ticker, d))
    lines.append("")

    lines.append("STAGE-1 CANDIDATE RULES:")
    lines.append("1. Return 5-30 long candidates and 5-30 short candidates")
    lines.append("2. No cross-leg overlap")
    if turnover_aware:
        lines.append("3. Ignore incumbents here; stage 2 will handle keep-vs-replace decisions")
    else:
        lines.append("3. Ignore previous holdings, turnover, and transaction costs")
    lines.append("4. Prefer signal quality over tiny sizing differences")
    lines.append("5. Do not assign weights in this stage")

    return "\n".join(lines)


def _coerce_signal_candidate_list(raw_candidates: Any, valid_tickers: set) -> list:
    if not isinstance(raw_candidates, list):
        raise ValueError("candidate arrays must be lists")

    cleaned = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        ticker = _normalize_ticker(raw.get("ticker"))
        if ticker not in valid_tickers:
            continue
        edge_score = clamp(_safe_float(raw.get("edge_score", raw.get("score", 50.0)), 50.0), 0.0, 100.0)
        cleaned.append(
            {
                "ticker": ticker,
                "edge_score": edge_score,
                "rationale": str(raw.get("rationale", "")).strip(),
                "utility": _candidate_utility(edge_score),
            }
        )
    return cleaned


def _validate_signal_stage_output(raw_output: dict, valid_tickers: set) -> dict:
    long_candidates = _coerce_signal_candidate_list(raw_output.get("long_candidates", []), valid_tickers)
    short_candidates = _coerce_signal_candidate_list(raw_output.get("short_candidates", []), valid_tickers)
    long_candidates, short_candidates = _deduplicate_cross_leg_entries(long_candidates, short_candidates)

    if len(long_candidates) < 5 or len(short_candidates) < 5:
        raise ValueError(
            f"Need >=5 signal candidates per side, got {len(long_candidates)} long / {len(short_candidates)} short"
        )

    long_candidates = long_candidates[:30]
    short_candidates = short_candidates[:30]
    for index, candidate in enumerate(long_candidates, start=1):
        candidate["rank"] = index
    for index, candidate in enumerate(short_candidates, start=1):
        candidate["rank"] = index

    return {
        "long_candidates": long_candidates,
        "short_candidates": short_candidates,
        "reasoning": str(raw_output.get("reasoning", "")).strip(),
        "regime_adjustment": str(raw_output.get("regime_adjustment", "")).strip(),
    }


def _build_candidate_lookup(signal_stage_output: dict) -> Dict[str, Dict[str, dict]]:
    return {
        "LONG": {entry["ticker"]: dict(entry) for entry in signal_stage_output.get("long_candidates", [])},
        "SHORT": {entry["ticker"]: dict(entry) for entry in signal_stage_output.get("short_candidates", [])},
    }


def _selection_from_candidates(signal_stage_output: dict) -> dict:
    return {
        "long_basket": [
            {"ticker": entry["ticker"], "rationale": entry.get("rationale", ""), "selection_stage": "signal_stage"}
            for entry in signal_stage_output.get("long_candidates", [])
        ],
        "short_basket": [
            {"ticker": entry["ticker"], "rationale": entry.get("rationale", ""), "selection_stage": "signal_stage"}
            for entry in signal_stage_output.get("short_candidates", [])
        ],
    }


def _format_overlay_prompt_context(
    previous_portfolio: dict,
    signal_stage_output: dict,
    universe: dict,
    regime: str,
    rebalance_id: str,
) -> str:
    previous_portfolio = previous_portfolio or {}
    current_universe_tickers = set(universe.keys())
    prev_long_entries = [
        entry for entry in previous_portfolio.get("long_basket", [])
        if _normalize_ticker(entry.get("ticker")) in current_universe_tickers
    ]
    prev_short_entries = [
        entry for entry in previous_portfolio.get("short_basket", [])
        if _normalize_ticker(entry.get("ticker")) in current_universe_tickers
    ]
    prev_longs = {_normalize_ticker(entry.get("ticker")) for entry in prev_long_entries}
    prev_shorts = {_normalize_ticker(entry.get("ticker")) for entry in prev_short_entries}

    lines = []
    lines.append(f"REBALANCE ID: {rebalance_id}")
    lines.append(f"MARKET REGIME: {regime}")
    lines.append("IMPLEMENTATION COST ASSUMPTION: about 15 bps one-way / 30 bps round-trip")
    lines.append("")

    lines.append("=== PREVIOUS LONG HOLDINGS ===")
    if prev_long_entries:
        lines.append(f"{'Ticker':<8} {'PrevW':>7} {'Dir':>6} {'Mag':>6}")
        lines.append("-" * 32)
        for entry in prev_long_entries:
            ticker = _normalize_ticker(entry.get("ticker"))
            signal = universe.get(ticker, {})
            prev_weight = abs(_safe_float(entry.get("weight_raw", entry.get("weight_exec", entry.get("weight", 0.0))), 0.0))
            lines.append(
                f"{ticker:<8} "
                f"{prev_weight:>7.3f} "
                f"{_safe_float(signal.get('direction_score'), 0.0):>6.1f} "
                f"{_safe_float(signal.get('magnitude_score'), 0.0):>6.1f}"
            )
    else:
        lines.append("None")
    lines.append("")

    lines.append("=== PREVIOUS SHORT HOLDINGS ===")
    if prev_short_entries:
        lines.append(f"{'Ticker':<8} {'PrevW':>7} {'Dir':>6} {'Mag':>6}")
        lines.append("-" * 32)
        for entry in prev_short_entries:
            ticker = _normalize_ticker(entry.get("ticker"))
            signal = universe.get(ticker, {})
            prev_weight = abs(_safe_float(entry.get("weight_raw", entry.get("weight_exec", entry.get("weight", 0.0))), 0.0))
            lines.append(
                f"{ticker:<8} "
                f"{prev_weight:>7.3f} "
                f"{_safe_float(signal.get('direction_score'), 0.0):>6.1f} "
                f"{_safe_float(signal.get('magnitude_score'), 0.0):>6.1f}"
            )
    else:
        lines.append("None")
    lines.append("")

    lines.append("=== ALPHA STAGE LONG CHALLENGERS ===")
    long_candidates = signal_stage_output.get("long_candidates", [])
    if long_candidates:
        lines.append(f"{'Rank':<5} {'Ticker':<8} {'Edge':>6} {'Util':>8} {'Held':>5}  Rationale")
        lines.append("-" * 103)
        for entry in long_candidates:
            held_flag = "yes" if entry["ticker"] in prev_longs else "no"
            lines.append(
                f"{int(entry.get('rank', 0)):<5} "
                f"{entry['ticker']:<8} "
                f"{entry.get('edge_score', 0):>6.1f} "
                f"{entry.get('utility', 0):>8.2f} "
                f"{held_flag:>5}  "
                f"{entry.get('rationale', '')}"
            )
    else:
        lines.append("None")
    lines.append("")

    lines.append("=== ALPHA STAGE SHORT CHALLENGERS ===")
    short_candidates = signal_stage_output.get("short_candidates", [])
    if short_candidates:
        lines.append(f"{'Rank':<5} {'Ticker':<8} {'Edge':>6} {'Util':>8} {'Held':>5}  Rationale")
        lines.append("-" * 103)
        for entry in short_candidates:
            held_flag = "yes" if entry["ticker"] in prev_shorts else "no"
            lines.append(
                f"{int(entry.get('rank', 0)):<5} "
                f"{entry['ticker']:<8} "
                f"{entry.get('edge_score', 0):>6.1f} "
                f"{entry.get('utility', 0):>8.2f} "
                f"{held_flag:>5}  "
                f"{entry.get('rationale', '')}"
            )
    else:
        lines.append("None")
    lines.append("")

    lines.append("OVERLAY RULES:")
    lines.append("1. Output the final implemented long and short books only")
    lines.append("2. Keep incumbents only when replacement edge is not large enough after costs and uncertainty")
    lines.append("3. Do not assign weights; sizing will be handled by deterministic code")
    lines.append("4. Prefix a rationale with COST_DRIVEN when cost is the main reason to keep a marginal incumbent")
    return "\n".join(lines)


def _coerce_overlay_entries(raw_entries: Any, allowed_tickers: set) -> list:
    if not isinstance(raw_entries, list):
        raise ValueError("overlay baskets must be lists")

    cleaned = []
    for raw in raw_entries:
        if isinstance(raw, str):
            ticker = _normalize_ticker(raw)
            rationale = ""
        elif isinstance(raw, dict):
            ticker = _normalize_ticker(raw.get("ticker"))
            rationale = str(raw.get("rationale", "")).strip()
        else:
            continue
        if ticker in allowed_tickers:
            cleaned.append({"ticker": ticker, "rationale": rationale})
    return cleaned


def _validate_overlay_stage_output(
    raw_output: dict,
    allowed_long_tickers: set,
    allowed_short_tickers: set,
) -> dict:
    long_entries = _coerce_overlay_entries(raw_output.get("long_basket", []), allowed_long_tickers)
    short_entries = _coerce_overlay_entries(raw_output.get("short_basket", []), allowed_short_tickers)
    long_entries, short_entries = _deduplicate_cross_leg_entries(long_entries, short_entries)

    if len(long_entries) < 5 or len(short_entries) < 5:
        raise ValueError(
            f"Need >=5 overlay selections per side, got {len(long_entries)} long / {len(short_entries)} short"
        )

    return {
        "long_basket": long_entries[:30],
        "short_basket": short_entries[:30],
        "reasoning": str(raw_output.get("reasoning", "")).strip(),
        "regime_adjustment": str(raw_output.get("regime_adjustment", "")).strip(),
    }


def _build_weighted_side_from_selection(
    selected_entries: list,
    side: str,
    candidate_lookup: Dict[str, Dict[str, dict]],
    universe: dict,
    previous_portfolio: dict = None,
    use_equal_weight: bool = False,
) -> list:
    previous_side_key = "long_basket" if side == "LONG" else "short_basket"
    previous_weight_map = {}
    if previous_portfolio:
        for entry in previous_portfolio.get(previous_side_key, []):
            ticker = _normalize_ticker(entry.get("ticker"))
            previous_weight_map[ticker] = abs(
                _safe_float(entry.get("weight_raw", entry.get("weight_exec", entry.get("weight", 0.0))), 0.0)
            )

    utilities = []
    candidate_map = candidate_lookup.get(side, {})
    for entry in selected_entries:
        ticker = entry["ticker"]
        alpha_candidate = candidate_map.get(ticker)
        utility = alpha_candidate.get("utility") if alpha_candidate else None
        if utility is None:
            utility = max(previous_weight_map.get(ticker, 0.0), _fallback_signal_utility(universe.get(ticker, {}), side))
        utilities.append(max(float(utility), 1e-6))

    if not utilities:
        return []

    total_utility = sum(utilities)
    raw_weights = [u / total_utility for u in utilities] if total_utility > 0 else [1.0 / len(utilities)] * len(utilities)
    exec_weights = [1.0 / len(utilities)] * len(utilities) if use_equal_weight else list(raw_weights)

    weighted_entries = []
    for entry, raw_weight, exec_weight, utility in zip(selected_entries, raw_weights, exec_weights, utilities):
        ticker = entry["ticker"]
        alpha_candidate = candidate_map.get(ticker, {})
        signal = universe.get(ticker, {})
        rationale = str(entry.get("rationale") or alpha_candidate.get("rationale", "")).strip()
        weighted_entry = {
            "ticker": ticker,
            "weight_raw": float(raw_weight),
            "weight_exec": float(exec_weight),
            "weight": float(exec_weight),
            "rationale": rationale,
            "side": side,
            "direction_score": float(signal.get("direction_score", 0.0)),
            "magnitude_score": float(signal.get("magnitude_score", 0.0)),
            "llm_source": "llm",
            "selection_stage": entry.get("selection_stage", "signal_stage"),
            "alpha_edge_score": alpha_candidate.get("edge_score"),
            "alpha_rank": alpha_candidate.get("rank"),
            "alpha_utility": float(utility),
        }
        if alpha_candidate.get("rationale"):
            weighted_entry["alpha_rationale"] = str(alpha_candidate.get("rationale", "")).strip()
        if entry.get("selection_stage") == "overlay_stage":
            weighted_entry["overlay_rationale"] = rationale
        weighted_entries.append(weighted_entry)

    return weighted_entries


def _build_weighted_portfolio(
    selection: dict,
    candidate_lookup: Dict[str, Dict[str, dict]],
    universe: dict,
    previous_portfolio: dict = None,
    use_equal_weight: bool = False,
) -> Tuple[list, list]:
    long_basket = _build_weighted_side_from_selection(
        selection.get("long_basket", []),
        "LONG",
        candidate_lookup,
        universe,
        previous_portfolio=previous_portfolio,
        use_equal_weight=use_equal_weight,
    )
    short_basket = _build_weighted_side_from_selection(
        selection.get("short_basket", []),
        "SHORT",
        candidate_lookup,
        universe,
        previous_portfolio=previous_portfolio,
        use_equal_weight=use_equal_weight,
    )
    long_basket, short_basket = validate_and_deduplicate_portfolio(long_basket, short_basket, previous_portfolio)
    if len(long_basket) < 5 or len(short_basket) < 5:
        raise ValueError(
            f"Weighted portfolio needs >=5 names per side, got {len(long_basket)} long / {len(short_basket)} short"
        )
    return long_basket[:30], short_basket[:30]


async def construct_signal_candidates_from_universe(
    client: httpx.AsyncClient,
    universe: dict,
    regime: str,
    rebalance_id: str,
    model: str,
    temperature: float,
    top_n_magnitude: int,
    turnover_aware: bool,
    loss_memory_context: str = "",
) -> dict:
    """Stage 1: discover the best current long and short candidate slates."""
    valid_tickers = set(universe.keys())
    universe_text = _format_universe_for_prompt(
        universe=universe,
        previous_portfolio=None,
        regime=regime,
        rebalance_id=rebalance_id,
        top_n_magnitude=top_n_magnitude,
        turnover_aware=turnover_aware,
        loss_memory_context=loss_memory_context,
    )

    user_content = f"""Build ranked long and short candidate slates from this universe.

{universe_text}

RULES:
- Return 5-30 long candidates and 5-30 short candidates
- Ignore previous holdings, turnover, and transaction costs in this stage
- Only use tickers from the universe above
- Do NOT assign weights
- Return ONLY valid JSON matching the signal-stage schema
"""

    base_messages = [
        {"role": "system", "content": build_signal_stage_system_prompt(turnover_aware)},
        {"role": "user", "content": user_content},
    ]
    retry_instruction = (
        "Your previous response was invalid. Return ONLY valid JSON with long_candidates "
        "and short_candidates arrays. Each entry needs ticker, edge_score, "
        "and rationale. Pick 5-30 per side. Do not assign weights."
    )

    return await _call_llm_json_with_validation(
        client=client,
        model=model,
        temperature=temperature,
        base_messages=base_messages,
        retry_instruction=retry_instruction,
        stage_label="Signal stage",
        validate_output=lambda raw_output: _validate_signal_stage_output(raw_output, valid_tickers),
    )


async def apply_turnover_overlay_to_portfolio(
    client: httpx.AsyncClient,
    previous_portfolio: dict,
    signal_stage_output: dict,
    universe: dict,
    regime: str,
    rebalance_id: str,
    model: str,
    temperature: float,
) -> dict:
    """Stage 2: keep-vs-replace overlay using only incumbents plus stage-1 challengers."""
    current_universe_tickers = set(universe.keys())
    allowed_long_tickers = {
        _normalize_ticker(entry.get("ticker"))
        for entry in previous_portfolio.get("long_basket", [])
        if _normalize_ticker(entry.get("ticker")) in current_universe_tickers
    }
    allowed_short_tickers = {
        _normalize_ticker(entry.get("ticker"))
        for entry in previous_portfolio.get("short_basket", [])
        if _normalize_ticker(entry.get("ticker")) in current_universe_tickers
    }
    allowed_long_tickers.update(entry["ticker"] for entry in signal_stage_output.get("long_candidates", []))
    allowed_short_tickers.update(entry["ticker"] for entry in signal_stage_output.get("short_candidates", []))

    overlay_text = _format_overlay_prompt_context(
        previous_portfolio=previous_portfolio,
        signal_stage_output=signal_stage_output,
        universe=universe,
        regime=regime,
        rebalance_id=rebalance_id,
    )

    user_content = f"""Apply the turnover-aware overlay and produce the final implemented portfolio.

{overlay_text}

RULES:
- Output 5-30 long positions and 5-30 short positions
- Only use tickers from the incumbent and challenger pools above
- No cross-leg overlap
- Do NOT assign weights; sizing will be handled later by deterministic code
- Keep rationales short; use COST_DRIVEN prefix if cost is the main reason to keep a marginal incumbent
- Return ONLY valid JSON matching the overlay-stage schema
"""

    base_messages = [
        {"role": "system", "content": build_turnover_overlay_system_prompt()},
        {"role": "user", "content": user_content},
    ]
    retry_instruction = (
        "Your previous response was invalid. Return ONLY valid JSON with long_basket "
        "and short_basket arrays. Each entry needs ticker and rationale. Pick 5-30 "
        "per side, use only provided tickers, and do not assign weights."
    )

    return await _call_llm_json_with_validation(
        client=client,
        model=model,
        temperature=min(temperature, 0.2),
        base_messages=base_messages,
        retry_instruction=retry_instruction,
        stage_label="Overlay stage",
        validate_output=lambda raw_output: _validate_overlay_stage_output(
            raw_output,
            allowed_long_tickers,
            allowed_short_tickers,
        ),
    )


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
    turnover_aware: bool = False,
    loss_memory_context: str = "",
) -> dict:
    """Run the two-stage LLM portfolio pipeline and return the final weighted portfolio."""

    if mask_date:
        rebalance_id = f"w_{abs(hash(rebalance_date)) % 100000:05d}"
    else:
        rebalance_id = rebalance_date

    async with httpx.AsyncClient(timeout=PORTFOLIO_REQUEST_TIMEOUT) as client:
        signal_stage_output = await construct_signal_candidates_from_universe(
            client=client,
            universe=universe,
            regime=regime,
            rebalance_id=rebalance_id,
            model=model,
            temperature=temperature,
            top_n_magnitude=top_n_magnitude,
            turnover_aware=turnover_aware,
            loss_memory_context=loss_memory_context,
        )

        candidate_lookup = _build_candidate_lookup(signal_stage_output)
        preliminary_selection = _selection_from_candidates(signal_stage_output)
        preliminary_long_basket, preliminary_short_basket = _build_weighted_portfolio(
            preliminary_selection,
            candidate_lookup,
            universe,
            previous_portfolio=previous_portfolio,
            use_equal_weight=use_equal_weight,
        )

        overlay_stage_output = None
        final_selection = preliminary_selection
        if turnover_aware and previous_portfolio:
            overlay_stage_output = await apply_turnover_overlay_to_portfolio(
                client=client,
                previous_portfolio=previous_portfolio,
                signal_stage_output=signal_stage_output,
                universe=universe,
                regime=regime,
                rebalance_id=rebalance_id,
                model=model,
                temperature=temperature,
            )
            final_selection = {
                "long_basket": [
                    {**entry, "selection_stage": "overlay_stage"}
                    for entry in overlay_stage_output.get("long_basket", [])
                ],
                "short_basket": [
                    {**entry, "selection_stage": "overlay_stage"}
                    for entry in overlay_stage_output.get("short_basket", [])
                ],
            }

        long_basket, short_basket = _build_weighted_portfolio(
            final_selection,
            candidate_lookup,
            universe,
            previous_portfolio=previous_portfolio,
            use_equal_weight=use_equal_weight,
        )

    kept_from_previous, dropped_from_previous = derive_portfolio_transition(
        previous_portfolio,
        long_basket,
        short_basket,
    )

    cost_driven_count = (
        sum(
            1
            for entry in long_basket + short_basket
            if str(entry.get("rationale", "")).upper().startswith("COST_DRIVEN")
        )
        if turnover_aware
        else 0
    )

    llm_usage = _aggregate_llm_usage(
        {
            "signal_stage": signal_stage_output.get("llm_usage"),
            "overlay_stage": overlay_stage_output.get("llm_usage") if overlay_stage_output else None,
        }
    )

    execution_mode = "equal_weight" if use_equal_weight else "llm_weighted"
    signal_stage_summary = {
        key: signal_stage_output.get(key)
        for key in ("long_candidates", "short_candidates", "reasoning", "regime_adjustment")
    }
    overlay_stage_summary = None
    if overlay_stage_output:
        overlay_stage_summary = {
            key: overlay_stage_output.get(key)
            for key in ("long_basket", "short_basket", "reasoning", "regime_adjustment")
        }

    return {
        "rebalance_date": rebalance_date,
        "rebalance_id": rebalance_id,
        "regime": regime,
        "pipeline_mode": "two_stage_overlay" if overlay_stage_output else "two_stage_signal_only",
        "long_basket": long_basket,
        "short_basket": short_basket,
        "preliminary_portfolio": {
            "long_basket": preliminary_long_basket,
            "short_basket": preliminary_short_basket,
        },
        "signal_stage": signal_stage_summary,
        "overlay_stage": overlay_stage_summary,
        "kept_from_previous": kept_from_previous,
        "dropped_from_previous": dropped_from_previous,
        "reasoning": (
            overlay_stage_output.get("reasoning")
            if overlay_stage_output and overlay_stage_output.get("reasoning")
            else signal_stage_output.get("reasoning", "")
        ),
        "regime_adjustment": (
            overlay_stage_output.get("regime_adjustment")
            if overlay_stage_output and overlay_stage_output.get("regime_adjustment")
            else signal_stage_output.get("regime_adjustment", "")
        ),
        "cost_driven_positions": cost_driven_count,
        "turnover_aware": turnover_aware,
        "llm_source": "llm",
        "execution_mode": execution_mode,
        "max_retries": MAX_PORTFOLIO_RETRIES,
        "llm_usage": llm_usage,
        "raw_llm_response": {
            "signal_stage": signal_stage_output.get("raw_llm_response", {}),
            "overlay_stage": overlay_stage_output.get("raw_llm_response", {}) if overlay_stage_output else {},
        },
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
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model name")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature (ignored by deepseek-reasoner)")
    
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
