"""
Broad-universe mean-reversion LLM portfolio construction.

This module mirrors the public portfolio-construction interface from
`technical_llm.py`, but stage-1 candidate discovery follows the weekly
mean-reversion shape:

- long recent losers / oversold names
- short recent winners / overbought names
- no hard direction-score thresholding before ranking

The turnover overlay and deterministic weighting layers are intentionally
shared with the deterministic LLM pipeline so we only change the alpha slate.
"""

from typing import Any, Dict, Optional, Tuple

import httpx
import numpy as np

from technical_llm import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_PORTFOLIO_RETRIES,
    PORTFOLIO_REQUEST_TIMEOUT,
    _aggregate_llm_usage,
    _build_candidate_lookup,
    _compact_signal_stage_summary,
    _extract_stage_trace,
    _build_weighted_portfolio,
    _call_llm_json_with_validation,
    _candidate_utility,
    _normalize_ticker,
    _safe_float,
    _selection_from_candidates,
    apply_turnover_overlay_to_portfolio,
    clamp,
    derive_portfolio_transition,
    max_positions_per_side,
    position_count_rule_text,
    validate_position_counts,
)


TRAIL_LOOKBACK_DAYS = 5
TRAIL_RETURN_KEY = f"trail_return_{TRAIL_LOOKBACK_DAYS}d"


MEAN_REVERSION_SIGNAL_STAGE_SYSTEM_PROMPT_TEMPLATE = """
MEAN REVERSION DISCOVERY STAGE
Use only the provided point-in-time inputs to identify the strongest current
mean-reversion longs and shorts for the next rebalance horizon.
Treat tickers as opaque identifiers. Do not use outside knowledge, calendar
knowledge, or ticker familiarity.

MECHANICAL BASELINE:
- Compute mean_reversion_signal = -(trailing 5-trading-day return)
- Rank the full universe cross-sectionally by this signal
- LONG the highest signal names = biggest recent losers
- SHORT the lowest signal names = biggest recent winners
- Equal-weight each leg
- Ignore signal quality, regime context, and implementation details

ALPHA RULES:
1. Reversal quality first:
   Prefer names showing short-term exhaustion or stretch with a plausible snapback setup.
   Longs should look washed out, extended lower, and crowded to the downside.
   Shorts should look overheated, extended higher, and crowded to the upside.

2. Broad-universe discipline:
   Candidate pools come from the full ranked universe.
   Do not require positive direction scores for longs or negative direction scores for shorts.
   A strong mean-reversion setup can be contrarian to the deterministic direction composite.

__PIPELINE_CONTEXT__

3. Trend and regime awareness:
   Penalize fades that are fighting extremely strong medium-term continuation without
   any sign of exhaustion.
   In crisis or squeeze regimes, demand cleaner reversal evidence before fading extremes.

4. Loss memory:
   If similar prior loss cases are shown, treat them as cautionary analogs.
   Do not blindly avoid the trade, but require clearly better stretch/exhaustion quality
   than the failed analogs before repeating the setup.

OUTPUT FORMAT (STRICT JSON):
{
  "long_candidates": [
    {"ticker": "AAPL", "edge_score": 88, "rationale": "Oversold stretch with snapback setup"},
    ...
  ],
  "short_candidates": [
    {"ticker": "XYZ", "edge_score": 84, "rationale": "Overbought extension with fade risk"},
    ...
  ],
  "reasoning": "2-3 sentence summary of the alpha slate",
  "regime_adjustment": "How the regime affected your candidate quality threshold"
}
"""

SIGNAL_STAGE_SIGNAL_ONLY_CONTEXT = """
4. PIPELINE CONTEXT
   This is the only decision stage for this run.
   Ignore previous holdings, turnover, and transaction costs.
   Optimize for current cross-sectional reversal edge and expected gross information ratio.
"""

SIGNAL_STAGE_TURNOVER_CONTEXT = """
4. PIPELINE CONTEXT
   This is stage 1 of 2.
   Ignore previous holdings, turnover, and transaction costs in this stage.
   Your job here is raw mean-reversion alpha discovery only; a later implementation
   overlay will decide which incumbents to keep or replace.
"""


def build_signal_stage_system_prompt(turnover_aware: bool) -> str:
    """Build the stage-1 mean-reversion alpha discovery prompt."""
    pipeline_context = (
        SIGNAL_STAGE_TURNOVER_CONTEXT if turnover_aware else SIGNAL_STAGE_SIGNAL_ONLY_CONTEXT
    )
    return MEAN_REVERSION_SIGNAL_STAGE_SYSTEM_PROMPT_TEMPLATE.replace(
        "__PIPELINE_CONTEXT__",
        pipeline_context.strip(),
    )


def build_turnover_overlay_system_prompt() -> str:
    """Use the same stage-2 turnover overlay as the deterministic LLM pipeline."""
    from technical_llm import build_turnover_overlay_system_prompt as build_overlay_prompt

    return build_overlay_prompt()


def build_portfolio_system_prompt(turnover_aware: bool) -> str:
    """Backward-compatible wrapper for the stage-1 prompt builder."""
    return build_signal_stage_system_prompt(turnover_aware)


def _build_signal_percentiles(signal_rows: list[dict]) -> dict[str, float]:
    if not signal_rows:
        return {}

    values = np.asarray([row["mean_reversion_signal"] for row in signal_rows], dtype=float)
    if len(values) == 1 or np.allclose(values, values[0]):
        return {row["ticker"]: 0.5 for row in signal_rows}

    sorted_positions = np.argsort(values, kind="mergesort")
    percentiles = np.empty(len(values), dtype=float)
    percentiles[sorted_positions] = np.linspace(0.0, 1.0, num=len(values))
    return {
        signal_rows[idx]["ticker"]: float(percentiles[idx])
        for idx in range(len(signal_rows))
    }


def _build_mean_reversion_candidate_views(universe: dict, top_n_candidates: int = 60) -> dict:
    scored_rows = []
    for ticker, payload in universe.items():
        mean_reversion_signal = _safe_float(payload.get("mean_reversion_signal"), np.nan)
        trail_return = _safe_float(payload.get(TRAIL_RETURN_KEY), np.nan)
        if np.isnan(mean_reversion_signal) or np.isnan(trail_return):
            continue
        scored_rows.append(
            {
                "ticker": ticker,
                "payload": payload,
                "mean_reversion_signal": float(mean_reversion_signal),
                "trail_return": float(trail_return),
            }
        )

    signal_percentiles = _build_signal_percentiles(scored_rows)
    for row in scored_rows:
        long_score = clamp(signal_percentiles.get(row["ticker"], 0.5), 0.0, 1.0)
        row["long_score"] = long_score
        row["short_score"] = 1.0 - long_score

    candidate_n = max(1, min(int(top_n_candidates), len(scored_rows)))

    long_ranked = sorted(
        scored_rows,
        key=lambda row: row["mean_reversion_signal"],
        reverse=True,
    )
    short_ranked = sorted(
        scored_rows,
        key=lambda row: row["mean_reversion_signal"],
    )

    top_longs = long_ranked[:candidate_n]
    excluded_for_shorts = {row["ticker"] for row in top_longs}
    top_shorts = []
    for row in short_ranked:
        if row["ticker"] in excluded_for_shorts:
            continue
        top_shorts.append(row)
        if len(top_shorts) >= candidate_n:
            break

    return {
        "signal_percentiles": signal_percentiles,
        "long_candidates": top_longs,
        "short_candidates": top_shorts,
    }


def _format_universe_for_prompt(
    universe: dict,
    previous_portfolio: dict = None,
    regime: str = "unknown",
    rebalance_id: str = "w_00001",
    top_n_magnitude: int = 60,
    target_positions_per_side: Optional[int] = None,
    turnover_aware: bool = False,
    loss_memory_context: str = "",
) -> str:
    """Format a full-universe mean-reversion ranking for stage-1 alpha discovery."""
    del previous_portfolio  # Unused. Kept for backwards-compatible call sites.

    candidate_views = _build_mean_reversion_candidate_views(universe, top_n_candidates=top_n_magnitude)
    top_longs = candidate_views["long_candidates"]
    top_shorts = candidate_views["short_candidates"]

    def indicator_value(indicators: dict, key: str) -> float:
        return _safe_float(indicators.get(key), 0.0)

    def format_candidate_row(entry: dict) -> str:
        payload = entry["payload"]
        indicators = payload.get("indicators", {})
        return (
            f"{entry['ticker']:<8} "
            f"{entry['long_score'] * 100.0:>6.1f} "
            f"{entry['short_score'] * 100.0:>6.1f} "
            f"{entry['trail_return'] * 100.0:>8.2f} "
            f"{entry['mean_reversion_signal'] * 100.0:>8.2f} "
            f"{indicator_value(indicators, 'rsi_5d'):>7.3f} "
            f"{indicator_value(indicators, 'range_pos_55d'):>7.3f} "
            f"{indicator_value(indicators, 'vel_10d_over_vol'):>7.3f} "
            f"{indicator_value(indicators, 'pct_from_ma100'):>7.3f} "
            f"{indicator_value(indicators, 'pct_from_52w_high'):>7.3f} "
            f"{indicator_value(indicators, 'mom_3m'):>7.3f} "
            f"{indicator_value(indicators, 'autocorr_5d'):>7.3f} "
            f"{indicator_value(indicators, 'adx_14d'):>7.3f} "
            f"{indicator_value(indicators, 'vol_percentile_504d'):>8.3f}"
        )

    lines = []
    lines.append(f"REBALANCE ID: {rebalance_id}")
    lines.append(f"MARKET REGIME: {regime}")
    lines.append(f"UNIVERSE SIZE: {len(universe)} stocks")
    if turnover_aware:
        lines.append("PIPELINE MODE: stage 1 of 2 (raw mean-reversion discovery first, turnover overlay later)")
        lines.append("TURNOVER OBJECTIVE: ignore trading costs in this stage; implementation will be handled later")
    else:
        lines.append("PIPELINE MODE: single-stage mean-reversion alpha discovery")
        lines.append("TURNOVER OBJECTIVE: ignore trading costs and optimize gross snapback quality only")
    lines.append("")

    lines.append("CANDIDATE CONSTRUCTION:")
    lines.append(f"  Broad full-universe mean-reversion ranking using {TRAIL_LOOKBACK_DAYS}-trading-day trailing return.")
    lines.append("  Signal is mean_reversion_signal = -(trailing return), so longs are recent losers and shorts are recent winners.")
    lines.append("")

    if loss_memory_context:
        lines.append("=== SIMILAR PRIOR LOSS CASES ===")
        lines.append(loss_memory_context)
        lines.append("")

    lines.append("INDICATOR SCALE: non-score indicator columns below are percentile-normalized to [0,1] when available.")
    lines.append("")

    header = (
        f"{'Ticker':<8} {'MRL':>6} {'MRS':>6} {'Trail5':>8} {'MRSig':>8} "
        f"{'RSI5':>7} {'Rng55':>7} {'Vel/V':>7} {'%MA100':>7} {'52WH':>7} "
        f"{'Mom3m':>7} {'AutoC':>7} {'ADX':>7} {'VolPctl':>8}"
    )

    lines.append(f"=== TOP {len(top_longs)} OVERSOLD LONG CANDIDATES ===")
    if top_longs:
        lines.append(header)
        lines.append("-" * 126)
        for entry in top_longs:
            lines.append(format_candidate_row(entry))
    lines.append("")

    lines.append(f"=== TOP {len(top_shorts)} OVERBOUGHT SHORT CANDIDATES ===")
    if top_shorts:
        lines.append(header)
        lines.append("-" * 126)
        for entry in top_shorts:
            lines.append(format_candidate_row(entry))
    lines.append("")

    lines.append("STAGE-1 CANDIDATE RULES:")
    lines.append(f"1. {position_count_rule_text(target_positions_per_side, 'candidates')}")
    lines.append("2. No cross-leg overlap")
    if turnover_aware:
        lines.append("3. Ignore incumbents here; stage 2 will handle keep-vs-replace decisions")
    else:
        lines.append("3. Ignore previous holdings, turnover, and transaction costs")
    lines.append("4. Prefer reversal quality over tiny sizing differences")
    lines.append("5. Do not assign weights in this stage")
    lines.append("6. Keep rationales short fragments, ideally <= 8 words")

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


def _deduplicate_cross_leg_entries(long_entries: list, short_entries: list) -> Tuple[list, list]:
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


def _validate_signal_stage_output(
    raw_output: dict,
    valid_tickers: set,
    target_positions_per_side: Optional[int] = None,
) -> dict:
    long_candidates = _coerce_signal_candidate_list(raw_output.get("long_candidates", []), valid_tickers)
    short_candidates = _coerce_signal_candidate_list(raw_output.get("short_candidates", []), valid_tickers)
    long_candidates, short_candidates = _deduplicate_cross_leg_entries(long_candidates, short_candidates)

    validate_position_counts(
        len(long_candidates),
        len(short_candidates),
        target_positions_per_side,
        "signal candidates",
    )

    max_names = max_positions_per_side(target_positions_per_side)
    long_candidates = long_candidates[:max_names]
    short_candidates = short_candidates[:max_names]
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


async def construct_signal_candidates_from_universe(
    client: httpx.AsyncClient,
    universe: dict,
    regime: str,
    rebalance_id: str,
    model: str,
    temperature: float,
    top_n_magnitude: int,
    target_positions_per_side: Optional[int],
    turnover_aware: bool,
    loss_memory_context: str = "",
    log_llm_trace: bool = False,
) -> dict:
    """Stage 1: discover the best current mean-reversion candidate slates."""
    valid_tickers = set(universe.keys())
    effective_top_n = max(top_n_magnitude, max_positions_per_side(target_positions_per_side))
    universe_text = _format_universe_for_prompt(
        universe=universe,
        previous_portfolio=None,
        regime=regime,
        rebalance_id=rebalance_id,
        top_n_magnitude=effective_top_n,
        target_positions_per_side=target_positions_per_side,
        turnover_aware=turnover_aware,
        loss_memory_context=loss_memory_context,
    )

    user_content = f"""Build ranked long and short candidate slates from this mean-reversion universe.

{universe_text}

RULES:
- {position_count_rule_text(target_positions_per_side, 'candidates')}
- Ignore previous holdings, turnover, and transaction costs in this stage
- Only use tickers from the universe above
- Do NOT assign weights
- Keep rationales very short
- Return ONLY valid JSON matching the signal-stage schema
"""

    base_messages = [
        {"role": "system", "content": build_signal_stage_system_prompt(turnover_aware)},
        {"role": "user", "content": user_content},
    ]
    retry_instruction = (
        "Your previous response was invalid. Return ONLY valid JSON with long_candidates "
        "and short_candidates arrays. Each entry needs ticker, edge_score, "
        "and rationale. "
        f"{position_count_rule_text(target_positions_per_side, 'candidates')}. "
        "Do not assign weights."
    )

    return await _call_llm_json_with_validation(
        client=client,
        model=model,
        temperature=temperature,
        base_messages=base_messages,
        retry_instruction=retry_instruction,
        stage_label="Mean-reversion signal stage",
        validate_output=lambda raw_output: _validate_signal_stage_output(
            raw_output,
            valid_tickers,
            target_positions_per_side=target_positions_per_side,
        ),
        capture_trace=log_llm_trace,
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
    target_positions_per_side: Optional[int] = None,
    use_equal_weight: bool = False,
    turnover_aware: bool = False,
    loss_memory_context: str = "",
    log_llm_trace: bool = False,
) -> dict:
    """Run the two-stage LLM portfolio pipeline using mean-reversion alpha discovery."""
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
            target_positions_per_side=target_positions_per_side,
            turnover_aware=turnover_aware,
            loss_memory_context=loss_memory_context,
            log_llm_trace=log_llm_trace,
        )

        candidate_lookup = _build_candidate_lookup(signal_stage_output)
        preliminary_selection = _selection_from_candidates(signal_stage_output)
        preliminary_long_basket, preliminary_short_basket = _build_weighted_portfolio(
            preliminary_selection,
            candidate_lookup,
            universe,
            previous_portfolio=previous_portfolio,
            use_equal_weight=use_equal_weight,
            target_positions_per_side=target_positions_per_side,
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
                target_positions_per_side=target_positions_per_side,
                log_llm_trace=log_llm_trace,
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
            target_positions_per_side=target_positions_per_side,
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
    signal_stage_summary = _compact_signal_stage_summary(signal_stage_output)
    overlay_stage_summary = None
    if overlay_stage_output:
        overlay_stage_summary = {
            key: overlay_stage_output.get(key)
            for key in ("long_basket", "short_basket", "reasoning", "regime_adjustment")
        }

    result = {
        "rebalance_date": rebalance_date,
        "rebalance_id": rebalance_id,
        "regime": regime,
        "pipeline_mode": "two_stage_overlay" if overlay_stage_output else "two_stage_signal_only",
        "long_basket": long_basket,
        "short_basket": short_basket,
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
    }
    if overlay_stage_output:
        result["preliminary_portfolio"] = {
            "long_basket": preliminary_long_basket,
            "short_basket": preliminary_short_basket,
        }
    if log_llm_trace:
        result["llm_trace"] = {
            "signal_stage": _extract_stage_trace(signal_stage_output),
            "overlay_stage": _extract_stage_trace(overlay_stage_output),
        }
    return result


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "apply_turnover_overlay_to_portfolio",
    "build_portfolio_system_prompt",
    "build_signal_stage_system_prompt",
    "build_turnover_overlay_system_prompt",
    "construct_portfolio_from_universe",
    "construct_signal_candidates_from_universe",
]
