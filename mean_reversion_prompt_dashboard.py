"""
Static prompt-engineering dashboard for mean-reversion LLM experiments.

This is a planning/reporting artifact only. It reads prompt constants and
completed CSV/JSON outputs; it does not run backtests or make LLM calls.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

import mean_reversion_llm as legacy
import mean_reversion_llm_v1 as v1
import mean_reversion_llm_v2 as v2
import mean_reversion_llm_v2_phase2_validation as phase2


DEFAULT_RESULTS_ROOT = Path("backtest_results/llm_reversal_signal_v2_2009_2014")
DEFAULT_CACHE_DIR = Path("backtest_results/llm_reversal_signal_v2_cache")
DEFAULT_OUTPUT_NAME = "prompt_engineering_dashboard.html"
DEFAULT_HUB_OUTPUT_NAME = "mean_reversion_research_dashboard.html"
DEFAULT_FUTURE_OUTPUT_NAME = "future_experiments_dashboard.html"
DEFAULT_PHASE2_OUTPUT_NAME = "phase2_validation_dashboard.html"
DEFAULT_AUTORESEARCH_REPORT = Path("autoresearch_v2_cache_sweep/autoresearch_report.html")
DEFAULT_AUTORESEARCH_CAMPAIGNS_ROOT = Path("backtest_results/autoresearch_campaigns")
DEFAULT_AUTORESEARCH_CAMPAIGN_OUTPUT_NAME = "campaign_dashboard.html"
DEFAULT_THESIS_PACKAGE_ROOT = Path("backtest_results/research_thesis_package_20260808")
DEFAULT_CONSENSUS_ROOT = Path("backtest_results/v2_three_run_consensus_2009_2014")
DEFAULT_SIDE_EXPERIMENT_ROOT = Path("backtest_results/v2_dedicated_side_prompts_2009_2014")
DEFAULT_G0_CROSS_ROOT = Path("backtest_results/v2_prompt_cross_g0_01_g0_05_2009_2014")
DEFAULT_G0_CAMPAIGN_ROOT = Path(
    "backtest_results/autoresearch_campaigns/prompt_search_ticker_optional_resumed_20260806"
)
DEFAULT_V2_ANON_CONTROL_ROOT = Path("backtest_results/llm_reversal_signal_v2_anon_control_2009_2014")
DEFAULT_V2_STRUCTURED_ROOT = Path("backtest_results/llm_reversal_signal_v2_anon_structured_2009_2014")
DEFAULT_V21_ROOT = Path("backtest_results/llm_reversal_signal_v2_1_anon_2009_2014")
DEFAULT_V22_VISIBLE_ROOT = Path("backtest_results/llm_reversal_signal_v2_2_visible_2009_2014")
DEFAULT_V23_ROOT = Path("backtest_results/llm_reversal_signal_v2_3_anon_2009_2014")
DEFAULT_V2_TEMP0_ROOT = Path("backtest_results/llm_reversal_signal_v2_temp0_anon_2009_2014")
DEFAULT_V2_TEMP06_ROOT = Path("backtest_results/llm_reversal_signal_v2_temp0p6_anon_2009_2014")
DEFAULT_V2_TEMP1_ROOT = Path("backtest_results/llm_reversal_signal_v2_temp1_anon_2009_2014")
DEFAULT_V2_TEMP15_ROOT = Path("backtest_results/llm_reversal_signal_v2_temp1p5_anon_2009_2014")
DEFAULT_V2_TEMP2_ROOT = Path("backtest_results/llm_reversal_signal_v2_temp2_anon_2009_2014")
DEFAULT_V24_ROOT = Path("backtest_results/llm_reversal_signal_v2_4_anon_2009_2014")
DEFAULT_V25_ROOT = Path("backtest_results/llm_reversal_signal_v2_5_anon_2009_2014")
DEFAULT_V26_ROOT = Path("backtest_results/v2_prompt_ab_2009_2014/v26")
DEFAULT_V2_MINIMAL_CHANGE_ROOT = Path("backtest_results/v2_prompt_ab_2009_2014/minimal_change")
DEFAULT_V2_MINIMAL_CHANGE_TEMP1_ROOT = Path(
    "backtest_results/v2_prompt_ab_2009_2014/minimal_change_temp1"
)
DEFAULT_V2_MINIMAL_CHANGE_TEMP15_ROOT = Path(
    "backtest_results/v2_prompt_ab_2009_2014/minimal_change_temp1p5"
)
DEFAULT_V2_MINIMAL_CHANGE_TEMP15_INCIDENTS = Path(
    "backtest_results/v2_prompt_ab_2009_2014/minimal_change_temp1p5_incidents.json"
)
DEFAULT_V2_MINIMAL_CHANGE_TEMP06_SECOND_ROOT = Path(
    "backtest_results/v2_minimal_change_temp06_fresh_20260804"
)
DEFAULT_V2_CURRENT_EPOCH_TEMP1_ROOT = Path(
    "backtest_results/v2_system_prompt_temp1_v4flash0731_2009_2014"
)
DEFAULT_V2MC_CURRENT_EPOCH_TEMP1_ROOT = Path(
    "backtest_results/v2_minimal_change_temp1_v4flash0731_second_2009_2014"
)
DEFAULT_V2_CURRENT_EPOCH_TEMP1_SECOND_ROOT = Path(
    "backtest_results/v2_system_prompt_temp1_v4flash0731_second_2009_2014"
)
DEFAULT_V2_CURRENT_EPOCH_TEMP1_THIRD_ROOT = Path(
    "backtest_results/v2_system_prompt_temp1_v4flash0731_third_2009_2014"
)
DEFAULT_V2_CURRENT_EPOCH_TEMP1_VISIBLE_ROOT = Path(
    "backtest_results/v2_system_prompt_temp1_v4flash0731_ticker_visible_2009_2014"
)
DEFAULT_V2_NO_REGIME_ROOT = Path(
    "backtest_results/v2_no_regime_temp1_second_v4flash0731_2009_2014"
)
DEFAULT_PHASE2_SCORECARD = DEFAULT_RESULTS_ROOT / "phase2_validation_tracker/phase2_scorecard.csv"
DEFAULT_REASONING_VISIBLE_PATH = Path("backtest_results/deepseek_reasoning_peek_2009_01_06/reasoning_content.txt")
DEFAULT_REASONING_ANON_PATH = Path("backtest_results/deepseek_reasoning_peek_anon_2009_01_06/reasoning_content.txt")
DEFAULT_REASONING_GLOSSARY_VISIBLE_PATH = Path(
    "backtest_results/deepseek_reasoning_peek_glossary_visible_real_2009_01_06/reasoning_content.txt"
)
DEFAULT_REASONING_GLOSSARY_ANON_PATH = Path(
    "backtest_results/deepseek_reasoning_peek_glossary_anon_real_2009_01_06/reasoning_content.txt"
)
DEFAULT_REASONING_V22_VISIBLE_PATH = Path(
    "backtest_results/deepseek_reasoning_peek_v22_visible_2009_01_06/reasoning_content.txt"
)


@dataclass(frozen=True)
class PromptCard:
    title: str
    family: str
    schema: str
    asks_for: str
    reasoning_visibility: str
    pool_process: str
    accounting: str
    cache_replay: str
    status: str
    prompt_text: str


@dataclass(frozen=True)
class ConceptCard:
    concept: str
    hypothesis: str
    why_it_might_help: str
    risk: str
    expected_token_cost: str
    priority: str
    status: str


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def relative_href(from_path: Path, to_path: Path) -> str:
    return Path(os.path.relpath(Path(to_path), start=Path(from_path).parent)).as_posix()


def slug(value: Any) -> str:
    text = str(value).lower()
    return "".join(ch if ch.isalnum() else "-" for ch in text).strip("-")


def fmt_num(value: Any, digits: int = 2) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return f"{float(numeric):.{digits}f}"


def fmt_pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    prefix = "+" if signed and float(numeric) > 0 else ""
    return f"{prefix}{float(numeric):.{digits}f}%"


def fmt_pp(value: Any, digits: int = 1, signed: bool = True) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    prefix = "+" if signed and float(numeric) > 0 else ""
    return f"{prefix}{float(numeric):.{digits}f}pp"


def fmt_decimal_pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return fmt_pct(float(numeric) * 100.0, digits=digits, signed=signed)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json_object_if_exists(path: Path) -> dict[str, Any]:
    """Read a JSON object without breaking a live, read-only dashboard refresh."""
    try:
        value = read_json_if_exists(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_campaign_events(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    """Return only complete JSONL event objects, newest retained last."""
    if not path.exists() or path.is_symlink():
        return []
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    events.append(value)
    except (OSError, UnicodeError):
        return []
    return events[-max(1, int(limit)) :]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ", ".join(_display_value(item) for item in value)
    return str(value)


ARCHIVED_V21_DISCRIMINATIVE_SYSTEM_PROMPT = """
MEAN-REVERSION DISCRIMINATIVE SCORE PROMPT V2.1

CONTEXT
The candidate pools are already selected by extreme 5-trading-day moves.
Trail5 and RSI5 are useful context, but they are not enough by themselves.
Your job is to score the incremental quality of the mean-reversion setup
beyond the obvious recent move and short-term overbought/oversold reading.

TASK
Score two side-relative pools over the next 5 trading days.

  LONG POOL  = biggest 5-day losers after Russell 1000, price, and ADV filters.
               Higher score means the loser looks more like exhausted selling
               and less like a falling knife / continuation breakdown.

  SHORT POOL = biggest 5-day winners after Russell 1000, price, and ADV filters.
               Higher score means the winner looks more like exhausted buying
               and less like a squeeze / momentum continuation.

For each ticker in each pool, output one SCORE from 0 to 100.

SCORING PRINCIPLES
- Start from the fact that every name is already an extreme Trail5 candidate.
- Do not simply rank by Trail5 magnitude or RSI5 extremity.
- Use Trail5 and RSI5 as baseline context, then discriminate using the other
  columns to judge exhaustion versus continuation risk.
- Use the full 0-100 range when the cross-section supports it.
- Compare long names only to long names, and short names only to short names.

LONG-SIDE GUIDANCE
- Reward losers where secondary features support snap-back/exhaustion,
  especially AutoC, MRSig, and Rng55.
- Trail5 and RSI5 can support a high score only when secondary features also
  look favorable.
- Penalize names where the move looks like persistent downside or unstable
  high-volatility continuation, especially weak MRS, high VolPctl, poor Mom3m,
  poor 52WH, or other continuation-like evidence.

SHORT-SIDE GUIDANCE
- Reward winners where secondary features support overbought fade/exhaustion,
  especially Vel/V, %MA100, MRL, and MRS.
- Trail5 and RSI5 can support a high score only when secondary features also
  look favorable.
- Penalize names where the move looks like squeeze/continuation risk,
  especially unfavorable MRSig, AutoC, VolPctl, 52WH, or other continuation-like
  evidence.

RULES
- Use only the indicator columns provided. No outside knowledge, no ticker
  familiarity, no calendar awareness.
- Treat tickers as opaque identifiers.
- Do not produce ranks, weights, portfolio sizes, rationales, labels, or
  commentary.
- Return every provided ticker exactly once on its original side.

OUTPUT FORMAT (STRICT JSON, no other text, no markdown fences):
{
  "long_scores":  [{"ticker": "SEC_0001", "score": 72}, {"ticker": "SEC_0002", "score": 58}],
  "short_scores": [{"ticker": "SEC_0061", "score": 84}, {"ticker": "SEC_0062", "score": 61}]
}
""".strip()


def prompt_cards() -> list[PromptCard]:
    return [
        PromptCard(
            title="Legacy Discovery Stage",
            family="Legacy",
            schema="long_candidates / short_candidates with edge_score and rationale",
            asks_for="Candidate selection plus edge score",
            reasoning_visibility="Allows rationale plus slate/regime summary",
            pool_process="Full-universe mean-reversion ranking, then LLM candidate slate",
            accounting="Legacy factor harness; now replayed as gross fixed 5-day C2C for apples-to-apples review",
            cache_replay="Legacy cached selections/scores can be replayed offline",
            status="Historical reference",
            prompt_text=legacy.MEAN_REVERSION_SIGNAL_STAGE_SYSTEM_PROMPT_TEMPLATE,
        ),
        PromptCard(
            title="Legacy Signal Ranking",
            family="Legacy",
            schema="long_rankings / short_rankings with rank and confidence_score",
            asks_for="Side-relative rank plus confidence",
            reasoning_visibility="Allows 1-2 sentence ranking summary",
            pool_process="D1/D10 candidate buckets ranked within side",
            accounting="Legacy schedule/replay context",
            cache_replay="Rank/confidence outputs converted into score columns for replay",
            status="Historical reference",
            prompt_text=legacy.build_signal_ranking_system_prompt(),
        ),
        PromptCard(
            title="Raw Confidence",
            family="V1/Legacy bridge",
            schema="long_scores / short_scores with score 0-100",
            asks_for="Confidence that setup is better than same-side peers",
            reasoning_visibility="Strict JSON only; no commentary",
            pool_process="Mean-reversion candidates after Russell, price, and ADV filters",
            accounting="V1-style fixed 5-day accounting when run through V1/V2 harnesses",
            cache_replay="Cache one score per ticker/date/pool; replay books offline",
            status="Tested",
            prompt_text=v1.RAW_CONFIDENCE_SYSTEM_PROMPT,
        ),
        PromptCard(
            title="Rank + Confidence",
            family="V1/Legacy bridge",
            schema="rank plus confidence_score, converted to combined score",
            asks_for="Complete side-relative ordering and confidence",
            reasoning_visibility="Strict JSON only; no commentary",
            pool_process="Same candidate construction as V1, alternate scoring contract",
            accounting="V1-style fixed 5-day accounting",
            cache_replay="Replayable after conversion to score",
            status="Tested, weaker than raw confidence in legacy comparison",
            prompt_text=v1.RANK_CONFIDENCE_SYSTEM_PROMPT,
        ),
        PromptCard(
            title="V2 Score Prompt",
            family="V2",
            schema="long_scores / short_scores with score 0-100",
            asks_for="Setup strength over next 5 trading days",
            reasoning_visibility="Strict JSON only; no reasoning captured",
            pool_process="60/60 candidates; 30/30 selected book; offset 3 lab bench",
            accounting="Gross fixed 5-trading-day C2C headline, O2O audit available",
            cache_replay="One score pass per signal date; book/overlay/concentration replay offline",
            status="Current winner",
            prompt_text=v2.V2_SYSTEM_PROMPT,
        ),
        PromptCard(
            title="V2 Minimal-Change Control",
            family="V2 minimal change",
            schema="unchanged long_scores / short_scores with score 0-100",
            asks_for="Same V2 judgment with only conditional score-range separation",
            reasoning_visibility="Strict JSON only; no reasoning captured",
            pool_process="Same anonymized offset-3 V2 lab bench: 60/60 candidates, 30/30 selected",
            accounting="Same gross fixed 5-trading-day C2C/O2O replay as incumbent V2",
            cache_replay="Separate hash; all portfolio construction remains unchanged",
            status="Development winner; validation required",
            prompt_text=v2.V2_MINIMAL_CHANGE_CONTROL_PROMPT,
        ),
        PromptCard(
            title="V2.6 Production-Grade Rewrite",
            family="V2.6",
            schema="unchanged long_scores / short_scores with score 0-100",
            asks_for="Side-relative ordinal mean-reversion setup strength with clarified feature semantics",
            reasoning_visibility="Strict JSON only; no reasoning captured",
            pool_process="Same anonymized offset-3 V2 lab bench: 60/60 candidates, 30/30 selected",
            accounting="Same gross fixed 5-trading-day C2C/O2O replay as incumbent V2",
            cache_replay="Separate hash; bundled prompt treatment",
            status="Development screen failed",
            prompt_text=v2.V26_SYSTEM_PROMPT,
        ),
        PromptCard(
            title="Archived V2.1 Discriminative Prompt",
            family="Archived V2.1",
            schema="long_scores / short_scores with score 0-100",
            asks_for="Incremental reversal quality beyond Trail5 / RSI5",
            reasoning_visibility="Strict JSON only; no reasoning captured",
            pool_process="Same V2 lab bench: 60/60 candidates, 30/30 selected book, offset 3",
            accounting="Same gross fixed 5-trading-day C2C/O2O replay as V2",
            cache_replay="Separate hashed cache; no V2 score reuse",
            status="Tested, failed",
            prompt_text=ARCHIVED_V21_DISCRIMINATIVE_SYSTEM_PROMPT,
        ),
        PromptCard(
            title="V2.3 Rank-Only Prompt",
            family="V2.3",
            schema="long_ranking / short_ranking complete ordered ticker lists",
            asks_for="Ordinal ordering only; no calibrated 0-100 score",
            reasoning_visibility="Strict JSON only; no reasoning captured",
            pool_process="Same V2 lab bench: anonymized 60/60 candidates, 30/30 selected book, offset 3",
            accounting="Same gross fixed 5-trading-day C2C/O2O replay as V2",
            cache_replay="Separate hashed cache; rank order converted to deterministic replay score offline",
            status="Tested, slight Trail5 lift but weaker than V2 score prompt",
            prompt_text=v2.V23_RANK_ONLY_SYSTEM_PROMPT,
        ),
        PromptCard(
            title="V2.4 Raw Market Context",
            family="V2.4",
            schema="long_scores / short_scores with score 0-100",
            asks_for="Historical no-glossary V2 scoring with raw market measurements",
            reasoning_visibility="Strict JSON only; no reasoning captured",
            pool_process="Anonymized 60/60 candidates; semantic regime label replaced by SPY/VIX measurements",
            accounting="Locked offset 3, 30/30 book, gross fixed 5-trading-day C2C",
            cache_replay="Separate prompt/context hash; paired Trail5 replay built into the harness",
            status="Completed; near-flat versus Trail5 with better IC and drawdown",
            prompt_text=v2.V24_FROZEN_SYSTEM_PROMPT,
        ),
        PromptCard(
            title="V2.5 Balanced Outcome Memory",
            family="V2.5",
            schema="unchanged V2 0-100 scores plus HistN / HistWin / HistRet inputs",
            asks_for="Current glossary-backed V2 judgment informed by balanced past-only analog outcomes",
            reasoning_visibility="Strict JSON only; no reasoning captured",
            pool_process="20 nearest prior same-side analogs; max 4 per date; same ticker excluded",
            accounting="Locked offset 3 with Trail5, HistRet-only, and fresh-V2 paired controls",
            cache_replay="Memory-spec hash and enriched-pool hash isolate the treatment cache",
            status="Completed; memory reduced drawdown but underperformed both fresh V2 and Trail5",
            prompt_text=v2.V25_BALANCED_MEMORY_SYSTEM_PROMPT,
        ),
    ]


def concept_cards() -> list[ConceptCard]:
    return [
        ConceptCard(
            "Score semantics",
            "A soft wording change may sharpen the 5-day target without demoting the core reversal signal.",
            "It points the model directly at the realized holding-window outcome.",
            "V2.1 showed that a hard residual/incremental wording can suppress the actual edge.",
            "Same as V2 baseline",
            "High",
            "Revise after V2.1 failure",
        ),
        ConceptCard(
            "Ranking contract",
            "Rank-only or rank+confidence may force cleaner relative ordering than independent scores.",
            "The book is built from cross-sectional ordering, so ranking discipline matters.",
            "Prior rank+confidence underperformed raw confidence in legacy-style rows.",
            "Same to slightly higher completion",
            "Medium",
            "Tested / deprioritized",
        ),
        ConceptCard(
            "Reasoning visibility",
            "Capturing concise rationale labels can reveal why scores diverge from Trail5.",
            "Useful for prompt debugging and manager review.",
            "More tokens and possible rationalization noise; do only on small samples first.",
            "Higher completion cost",
            "Medium",
            "Proposed diagnostic",
        ),
        ConceptCard(
            "Anti-falling-knife rules",
            "A light caution about continuation/falling-knife setups may reduce bad long-side reversal attempts.",
            "Mean reversion loses when the move is repricing, not exhaustion.",
            "V2.1 overdid this and removed too many extreme names.",
            "Same as V2 baseline",
            "High",
            "Use only as a soft nudge",
        ),
        ConceptCard(
            "Side specialization",
            "Longs and shorts may need different scoring criteria.",
            "Recent diagnostics suggested more lift on longs than shorts.",
            "Separate rules may reduce comparability and add prompt complexity.",
            "Same if one combined prompt; higher if split calls",
            "High",
            "Proposed",
        ),
        ConceptCard(
            "Regime awareness",
            "The model may need stricter reversal evidence in crisis/squeeze regimes.",
            "Offset dispersion and year-level variation suggest regime dependence.",
            "May overcondition on noisy regime labels.",
            "Same as V2 baseline",
            "Medium",
            "Partially used",
        ),
        ConceptCard(
            "Feature framing",
            "Better feature wording or grouped feature blocks may help the model use indicators more consistently.",
            "The same raw data can be made easier for the model to parse.",
            "Prompt bloat increases cost and may distract from Trail5 core.",
            "Same to moderate increase",
            "Medium",
            "Proposed",
        ),
        ConceptCard(
            "Balanced outcome memory",
            "Past-only outcomes from technically similar candidates may help the model distinguish exhaustion from continuation.",
            "It adds empirical analog evidence without changing the original V2 score contract.",
            "Nearest-neighbor summaries can dominate judgment or encode unstable similarity; require a fresh V2 paired control.",
            "Higher input tokens; same output schema",
            "High",
            "V2.5 paid paired A/B complete; memory hurt CAGR",
        ),
        ConceptCard(
            "Model / hyperparameters",
            "Reasoner vs chat/flash and temperature may change ranking quality and cost.",
            "This is the cleanest cost/performance tradeoff test once the harness is locked.",
            "Changing model breaks direct comparability with existing DeepSeek runs.",
            "Variable",
            "Medium",
            "Proposed A/B",
        ),
    ]


def find_v2_cache(cache_dir: Path) -> dict[str, Any]:
    candidates = sorted(cache_dir.glob("llm_reversal_signal_v2_*.json"))
    if not candidates:
        return {"path": "", "metadata": {}, "signal_count": 0, "avg_total_tokens": None, "model": ""}
    path = candidates[-1]
    data = read_json_if_exists(path)
    signals = data.get("signals", {})
    token_values = []
    for entry in signals.values():
        usage = entry.get("llm_usage", {}) if isinstance(entry, dict) else {}
        provider = usage.get("provider_usage", {})
        total = provider.get("total_tokens") or usage.get("approx_total_tokens")
        numeric = pd.to_numeric(total, errors="coerce")
        if not pd.isna(numeric):
            token_values.append(float(numeric))
    metadata = data.get("metadata", {})
    return {
        "path": str(path),
        "metadata": metadata,
        "signal_count": len(signals),
        "avg_total_tokens": sum(token_values) / len(token_values) if token_values else None,
        "model": metadata.get("model", ""),
    }


def _metric(row: pd.Series, capital_col: str, raw_col: str) -> Any:
    value = row.get(capital_col)
    if pd.notna(pd.to_numeric(value, errors="coerce")):
        return value
    return row.get(raw_col)


def _conclusion_for(row: pd.Series, source: str) -> str:
    run = str(row.get("run", "")).lower()
    family = str(row.get("family", "")).lower()
    if source == "main" and "v2" in family and "best offset" in run and "llm" in run:
        return "Winner"
    if source == "overlay" and run == "llm":
        return "Winner"
    if source == "concentration":
        return "Diagnostic only"
    if source == "long_breadth":
        if run == "current_top30":
            return "Paired control"
        if "broad_core" in run or "reserve" in run or "equal_all60" in run or "rank_decay" in run:
            return "Useful but weaker"
    if source == "side_decay":
        if run == "control_top30_top30":
            return "Paired control"
        if "long_decay" in run and "short_top30" in run:
            return "Winner"
        return "Useful but weaker"
    if any(token in run for token in ("residual", "swap", "long_", "_short")):
        return "Useful but weaker"
    if "baseline" in run or "trail5" in run:
        return "Paired control"
    if "llm" in run:
        return "Useful but weaker"
    return "Diagnostic only"


def build_main_result_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "gross_legacy_v1_v2_comparison.csv"
    data = read_csv_if_exists(path)
    if data.empty:
        return []
    data = data[
        data.get("execution_model", pd.Series(dtype=str)).astype(str).str.upper().eq("C2C")
        & pd.to_numeric(data.get("cost_bps", 0.0), errors="coerce").fillna(0.0).eq(0.0)
    ].copy()
    data = data.sort_values("capital_cagr_pct", ascending=False)
    best_run = str(data.iloc[0].get("run", "")) if not data.empty else ""
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        rows.append(
            {
                "Run": row.get("run", ""),
                "Family": row.get("family", ""),
                "Pool/Book": row.get("pool_book", ""),
                "Schedule/Anchor": row.get("anchor_offset", ""),
                "Model": "cached LLM" if "llm" in str(row.get("run", "")).lower() else "none",
                "Schema": row.get("prompt_schema", ""),
                "CAGR": fmt_pct(_metric(row, "capital_cagr_pct", "cagr_pct")),
                "Sharpe": fmt_num(row.get("capital_sharpe", row.get("sharpe")), 2),
                "Max DD": fmt_pct(_metric(row, "capital_max_drawdown_pct", "max_drawdown_pct")),
                "IC": fmt_num(row.get("ic_mean"), 4),
                "Conclusion": _conclusion_for(row, "main"),
            }
        )
    return rows


def build_g0_cross_result_rows(
    cross_root: Path = DEFAULT_G0_CROSS_ROOT,
    campaign_root: Path = DEFAULT_G0_CAMPAIGN_ROOT,
) -> list[dict[str, Any]]:
    """Compare the completed G0_01 x G0_05 cross with its frozen references."""
    cross = read_json_if_exists(Path(cross_root) / "full/result.json")
    if not cross:
        return []
    campaign_root = Path(campaign_root)
    g0_01 = read_json_if_exists(campaign_root / "candidates/G0_01/full/result.json")
    g0_05 = read_json_if_exists(campaign_root / "candidates/G0_05/medium/result.json")
    controls = cross.get("controls", {}) if isinstance(cross.get("controls"), dict) else {}

    def result_row(
        label: str,
        payload: dict[str, Any],
        evidence: str,
        gate: str,
        read: str,
    ) -> dict[str, Any]:
        metrics = payload.get("metrics", payload)
        if not isinstance(metrics, dict):
            metrics = {}
        return {
            "Run": label,
            "Evidence": evidence,
            "Dates": metrics.get("n_periods", ""),
            "CAGR": fmt_pct(metrics.get("cagr_pct"), 2),
            "Sharpe": fmt_num(metrics.get("sharpe"), 3),
            "Max DD": fmt_pct(metrics.get("max_drawdown_pct"), 2),
            "Ann. lift vs control": fmt_pp(metrics.get("paired_annualized_lift_pct"), 2)
            if metrics.get("paired_annualized_lift_pct") is not None
            else "—",
            "Long lift": fmt_pp(metrics.get("long_paired_lift_pct"), 2)
            if metrics.get("long_paired_lift_pct") is not None
            else "—",
            "Short lift": fmt_pp(metrics.get("short_paired_lift_pct"), 2)
            if metrics.get("short_paired_lift_pct") is not None
            else "—",
            "Gate": gate,
            "Read": read,
        }

    cross_gate = cross.get("gate", {}) if isinstance(cross.get("gate"), dict) else {}
    rows = [
        result_row(
            "G0_01 × G0_05 cross",
            cross,
            "Full development",
            "PASS" if cross_gate.get("pass") else "FAIL",
            "Top-feature diagnostic reduced CAGR and Sharpe versus both reused controls.",
        )
    ]
    if g0_01:
        gate = g0_01.get("gate", {}) if isinstance(g0_01.get("gate"), dict) else {}
        rows.append(
            result_row(
                "G0_01",
                g0_01,
                "Full development",
                "PASS" if gate.get("pass") else "FAIL",
                "Autoresearch development winner and full-period parent reference.",
            )
        )
    if g0_05:
        gate = g0_05.get("gate", {}) if isinstance(g0_05.get("gate"), dict) else {}
        rows.append(
            result_row(
                "G0_05",
                g0_05,
                "Medium screen only",
                "PASS" if gate.get("pass") else "FAIL",
                "80-date diagnostic parent evidence; not a full-period comparison.",
            )
        )
    incumbent = controls.get("fresh_incumbent")
    if isinstance(incumbent, dict):
        rows.append(
            result_row(
                "Minimal-change control",
                incumbent,
                "Full reused control",
                "CONTROL",
                "Frozen anonymous control; no scorer calls rerun.",
            )
        )
    trail5 = controls.get("trail5")
    if isinstance(trail5, dict):
        rows.append(
            result_row(
                "Trail5",
                trail5,
                "Full reused baseline",
                "CONTROL",
                "Frozen deterministic baseline; no scorer calls.",
            )
        )
    return rows


def build_prompt_ab_decision_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in build_prompt_ab_rows():
        if str(row.get("Run", "")).startswith("V2.4"):
            continue
        rows.append(
            {
                "Run": row.get("Run", ""),
                "Family": "Ticker visibility / prompt A/B",
                "Pool/Book": "60/60 -> 30/30",
                "Schedule/Anchor": "offset 3",
                "Model": "deepseek-chat cache" if row.get("Run") != "Trail5 baseline" else "none",
                "Schema": row.get("Prompt / Process", ""),
                "CAGR": row.get("50/50 CAGR", ""),
                "Sharpe": row.get("Sharpe", ""),
                "Max DD": row.get("50/50 Max DD", ""),
                "IC": row.get("IC", ""),
                "Conclusion": row.get("Conclusion", ""),
            }
        )
    return rows


def build_temperature_decision_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in build_temperature_experiment_rows():
        rows.append(
            {
                "Run": row.get("Run", ""),
                "Family": "Temperature A/B",
                "Pool/Book": "60/60 -> 30/30",
                "Schedule/Anchor": "offset 3",
                "Model": "deepseek-chat" if row.get("Temperature") != "n/a" else "none",
                "Schema": row.get("Prompt", ""),
                "CAGR": row.get("50/50 CAGR", ""),
                "Sharpe": row.get("Sharpe", ""),
                "Max DD": row.get("Max DD", ""),
                "IC": "",
                "Conclusion": row.get("Read", ""),
            }
        )
    return rows


def _prompt_treatment_primary_row(root: Path, run_contains: str) -> pd.Series | None:
    data = read_csv_if_exists(Path(root) / "comparison_scorecard.csv")
    if data.empty:
        return None
    mask = (
        data.get("execution_model", pd.Series(index=data.index, dtype=str)).astype(str).str.upper().eq("C2C")
        & pd.to_numeric(data.get("cost_bps", 0.0), errors="coerce").fillna(0.0).eq(0.0)
        & ~data.get("is_anchor_average", pd.Series(False, index=data.index)).fillna(False).astype(bool)
        & data.get("run", pd.Series(index=data.index, dtype=str)).astype(str).str.contains(run_contains, regex=False)
    )
    selected = data.loc[mask]
    return None if selected.empty else selected.iloc[0]


def build_v26_minimal_result_rows(
    incumbent_root: Path = DEFAULT_V2_TEMP06_ROOT,
    minimal_root: Path = DEFAULT_V2_MINIMAL_CHANGE_ROOT,
    v26_root: Path = DEFAULT_V26_ROOT,
    minimal_temp1_root: Path = DEFAULT_V2_MINIMAL_CHANGE_TEMP1_ROOT,
    minimal_temp15_root: Path = DEFAULT_V2_MINIMAL_CHANGE_TEMP15_ROOT,
    minimal_temp15_incidents: Path = DEFAULT_V2_MINIMAL_CHANGE_TEMP15_INCIDENTS,
    minimal_temp06_second_root: Path = DEFAULT_V2_MINIMAL_CHANGE_TEMP06_SECOND_ROOT,
) -> list[dict[str, Any]]:
    incumbent = _prompt_treatment_primary_row(incumbent_root, "LLM-Pool v2")
    minimal = _prompt_treatment_primary_row(minimal_root, "LLM-Pool v2-minimal-change-control")
    minimal_temp1 = _prompt_treatment_primary_row(
        minimal_temp1_root, "LLM-Pool v2-minimal-change-control"
    )
    minimal_temp15 = _prompt_treatment_primary_row(
        minimal_temp15_root, "LLM-Pool v2-minimal-change-control"
    )
    minimal_temp06_second = _prompt_treatment_primary_row(
        minimal_temp06_second_root, "LLM-Pool v2-minimal-change-control"
    )
    v26_row = _prompt_treatment_primary_row(v26_root, "LLM-Pool v2.6")
    trail5 = _prompt_treatment_primary_row(minimal_root, "Baseline-Pool")
    if trail5 is None:
        trail5 = _prompt_treatment_primary_row(v26_root, "Baseline-Pool")

    specs = [
        ("Minimal change · temp 1.0", minimal_temp1, minimal_temp1_root, "Preferred challenger setting"),
        ("Minimal change · temp 0.6", minimal, minimal_root, "Best Sharpe and IC; near-tie with temp 1.0"),
        (
            "Minimal change · temp 0.6 · second run",
            minimal_temp06_second,
            minimal_temp06_second_root,
            "Independent rerun; materially below the first temp-0.6 realization",
        ),
        ("Minimal change · temp 1.5", minimal_temp15, minimal_temp15_root, "Failed; below Trail5 and frozen V2"),
        ("Frozen V2 incumbent", incumbent, incumbent_root, "Incumbent control"),
        ("V2.6 rewrite", v26_row, v26_root, "Failed development screen"),
        ("Trail5", trail5, minimal_root, "Paired deterministic baseline"),
    ]
    trail_cagr = pd.to_numeric(trail5.get("capital_cagr_pct"), errors="coerce") if trail5 is not None else float("nan")
    trail_sharpe = pd.to_numeric(trail5.get("capital_sharpe", trail5.get("sharpe")), errors="coerce") if trail5 is not None else float("nan")
    incumbent_cagr = (
        pd.to_numeric(incumbent.get("capital_cagr_pct"), errors="coerce")
        if incumbent is not None
        else float("nan")
    )
    minimal_cagr = (
        pd.to_numeric(minimal.get("capital_cagr_pct"), errors="coerce")
        if minimal is not None
        else float("nan")
    )
    incident_data = read_json_if_exists(minimal_temp15_incidents)
    incidents = incident_data.get("incidents", []) if isinstance(incident_data, dict) else []
    incident_note = (
        "Rejected score 800; resumed cleanly"
        if any("800" in str(item.get("error", "")) for item in incidents if isinstance(item, dict))
        else "No recorded schema incident"
    )
    rows: list[dict[str, Any]] = []
    for label, result, root, conclusion in specs:
        if result is None:
            continue
        config = read_json_if_exists(Path(root) / "run_config.json")
        cagr = pd.to_numeric(result.get("capital_cagr_pct"), errors="coerce")
        sharpe = pd.to_numeric(result.get("capital_sharpe", result.get("sharpe")), errors="coerce")
        rows.append(
            {
                "Run": label,
                "Prompt Hash": "deterministic" if label == "Trail5" else config.get("prompt_hash", "historical"),
                "Model": "none" if label == "Trail5" else config.get("model", "historical cache"),
                "Temperature": "—" if label == "Trail5" else fmt_num(config.get("temperature"), 1),
                "50/50 CAGR": fmt_pct(cagr, 2),
                "Lift vs Trail5": fmt_pp(cagr - trail_cagr, 2) if label != "Trail5" else "—",
                "Lift vs V2": fmt_pp(cagr - incumbent_cagr, 2) if label not in {"Frozen V2 incumbent", "Trail5"} else "—",
                "Δ vs Minimal 0.6": fmt_pp(cagr - minimal_cagr, 2)
                if label in {
                    "Minimal change · temp 1.0",
                    "Minimal change · temp 0.6 · second run",
                    "Minimal change · temp 1.5",
                }
                else "—",
                "Sharpe": fmt_num(sharpe, 3),
                "Sharpe vs Trail5": fmt_num(sharpe - trail_sharpe, 3) if label != "Trail5" else "—",
                "50/50 Max DD": fmt_pct(result.get("capital_max_drawdown_pct"), 2),
                "IC": fmt_num(result.get("ic_mean"), 4),
                "Schema / Reliability": incident_note if label == "Minimal change · temp 1.5" else "No recorded incident",
                "Conclusion": conclusion,
            }
        )
    return rows


def build_v26_minimal_read(
    incumbent_root: Path = DEFAULT_V2_TEMP06_ROOT,
    minimal_root: Path = DEFAULT_V2_MINIMAL_CHANGE_ROOT,
    v26_root: Path = DEFAULT_V26_ROOT,
    minimal_temp1_root: Path = DEFAULT_V2_MINIMAL_CHANGE_TEMP1_ROOT,
    minimal_temp15_root: Path = DEFAULT_V2_MINIMAL_CHANGE_TEMP15_ROOT,
    minimal_temp15_incidents: Path = DEFAULT_V2_MINIMAL_CHANGE_TEMP15_INCIDENTS,
    minimal_temp06_second_root: Path = DEFAULT_V2_MINIMAL_CHANGE_TEMP06_SECOND_ROOT,
) -> str:
    rows = {
        row["Run"]: row
        for row in build_v26_minimal_result_rows(
            incumbent_root,
            minimal_root,
            v26_root,
            minimal_temp1_root,
            minimal_temp15_root,
            minimal_temp15_incidents,
            minimal_temp06_second_root,
        )
    }
    minimal = rows.get("Minimal change · temp 0.6")
    minimal_temp1 = rows.get("Minimal change · temp 1.0")
    minimal_temp15 = rows.get("Minimal change · temp 1.5")
    minimal_temp06_second = rows.get("Minimal change · temp 0.6 · second run")
    incumbent = rows.get("Frozen V2 incumbent")
    v26_row = rows.get("V2.6 rewrite")
    if not minimal or not minimal_temp1 or not minimal_temp15 or not incumbent or not v26_row:
        return "V2.6/minimal-change result artifacts are incomplete."
    rerun_read = (
        f" A second independent temperature-0.6 run produced {minimal_temp06_second['50/50 CAGR']} CAGR "
        f"and {minimal_temp06_second['Sharpe']} Sharpe ({minimal_temp06_second['Δ vs Minimal 0.6']} versus "
        "the first temperature-0.6 run), showing material sampling variation."
        if minimal_temp06_second
        else ""
    )
    return (
        f"At temperature 1.0, minimal change reached {minimal_temp1['50/50 CAGR']} CAGR "
        f"({minimal_temp1['Δ vs Minimal 0.6']} versus temperature 0.6) with {minimal_temp1['Sharpe']} Sharpe "
        f"and {minimal_temp1['50/50 Max DD']} max drawdown. Temperature 0.6 retained slightly higher "
        f"Sharpe ({minimal['Sharpe']}) and IC ({minimal['IC']} versus {minimal_temp1['IC']}). "
        f"Temperature 1.0 remains the preferred challenger setting. Temperature 1.5 fell to "
        f"{minimal_temp15['50/50 CAGR']} CAGR, {minimal_temp15['Sharpe']} Sharpe, and "
        f"{minimal_temp15['50/50 Max DD']} max drawdown ({minimal_temp15['Lift vs Trail5']} versus Trail5); "
        f"it also produced a rejected score of 800 before completing through a clean cache resume. "
        f"V2.6 produced {v26_row['50/50 CAGR']} CAGR and {v26_row['Sharpe']} Sharpe, "
        f"trailing frozen V2 by {v26_row['Lift vs V2']}."
        f"{rerun_read}"
    )


def build_current_v4flash0731_rows(
    v2_root: Path = DEFAULT_V2_CURRENT_EPOCH_TEMP1_ROOT,
    minimal_root: Path = DEFAULT_V2MC_CURRENT_EPOCH_TEMP1_ROOT,
    v2_second_root: Path = DEFAULT_V2_CURRENT_EPOCH_TEMP1_SECOND_ROOT,
    v2_third_root: Path = DEFAULT_V2_CURRENT_EPOCH_TEMP1_THIRD_ROOT,
    v2_fourth_root: Path = DEFAULT_V2_CURRENT_EPOCH_TEMP1_VISIBLE_ROOT,
) -> list[dict[str, Any]]:
    """Show only completed, directly comparable post-July-31 prompt runs."""
    specs = (
        ("V2 system prompt · temp 1.0 · run 1", v2_root, "LLM-Pool v2", "Current-epoch leader"),
        (
            "V2 system prompt · temp 1.0 · run 2",
            v2_second_root,
            "LLM-Pool v2",
            "Independent anonymous rerun; materially weaker",
        ),
        (
            "V2 system prompt · temp 1.0 · run 3",
            v2_third_root,
            "LLM-Pool v2",
            "Independent anonymous rerun; weakest of the four",
        ),
        (
            "V2 system prompt · temp 1.0 · run 4",
            v2_fourth_root,
            "LLM-Pool v2",
            "Independent anonymous rerun; folder name is misleading",
        ),
        (
            "Minimal change · temp 1.0",
            minimal_root,
            "LLM-Pool v2-minimal-change-control",
            "Weaker than V2 in the current epoch",
        ),
    )
    rows: list[dict[str, Any]] = []
    for label, root, run_contains, conclusion in specs:
        result = _prompt_treatment_primary_row(root, run_contains)
        if result is None:
            continue
        config = read_json_if_exists(Path(root) / "run_config.json")
        rows.append(
            {
                "Run": label,
                "Model Epoch": "V4-Flash-0731",
                "Prompt Hash": config.get("prompt_hash", "—"),
                "Ticker Visibility": "Anonymous" if config.get("anonymize_tickers") else "Visible",
                "Temperature": fmt_num(config.get("temperature"), 1),
                "50/50 CAGR": fmt_pct(result.get("capital_cagr_pct"), 2),
                "Sharpe": fmt_num(result.get("capital_sharpe", result.get("sharpe")), 3),
                "50/50 Max DD": fmt_pct(result.get("capital_max_drawdown_pct"), 2),
                "IC": fmt_num(result.get("ic_mean"), 4),
                "Conclusion": conclusion,
            }
        )
    return rows


def build_current_v4flash0731_read() -> str:
    rows = {row["Run"]: row for row in build_current_v4flash0731_rows()}
    v2_row = rows.get("V2 system prompt · temp 1.0 · run 1")
    v2_second = rows.get("V2 system prompt · temp 1.0 · run 2")
    v2_third = rows.get("V2 system prompt · temp 1.0 · run 3")
    v2_fourth = rows.get("V2 system prompt · temp 1.0 · run 4")
    minimal_row = rows.get("Minimal change · temp 1.0")
    if not v2_row or not v2_second or not v2_third or not v2_fourth or not minimal_row:
        return "The four current-epoch V2 reruns and prompt comparisons are not all complete."
    return (
        f"On V4-Flash-0731, anonymous V2 temperature-1 runs ranged from {v2_third['50/50 CAGR']} to "
        f"{v2_row['50/50 CAGR']} CAGR ({v2_third['Sharpe']} to {v2_row['Sharpe']} Sharpe), confirming "
        f"material run-to-run variation across four independent anonymous runs. Minimal change produced "
        f"{minimal_row['50/50 CAGR']} CAGR and "
        f"{minimal_row['Sharpe']} Sharpe. These current-epoch runs must not be pooled with Preview-era results."
    )


def build_model_epoch_comparison_rows() -> list[dict[str, Any]]:
    """Compact, like-for-like epoch comparison for the manager summary."""
    preview = {
        row["Run"]: row for row in build_v26_minimal_result_rows()
    }.get("Minimal change · temp 1.0")
    current = {
        row["Run"]: row for row in build_current_v4flash0731_rows()
    }.get("Minimal change · temp 1.0")
    if preview is None or current is None:
        return []
    return [
        {
            "Model epoch": "V4-Flash Preview (historical)",
            "Run timing": "Before July 31, 2026",
            "Prompt / settings": "Minimal change · temp 1.0 · anonymous",
            "CAGR": preview["50/50 CAGR"],
            "Sharpe": preview["Sharpe"],
            "Interpretation": "Strong historical result; frozen reference only",
        },
        {
            "Model epoch": "V4-Flash-0731 (current)",
            "Run timing": "July 31, 2026 or later",
            "Prompt / settings": "Minimal change · temp 1.0 · anonymous",
            "CAGR": current["50/50 CAGR"],
            "Sharpe": current["Sharpe"],
            "Interpretation": "Same prompt rerun on the current backend",
        },
    ]


def build_model_epoch_read() -> str:
    rows = build_model_epoch_comparison_rows()
    if len(rows) != 2:
        return "The like-for-like minimal-change epoch comparison is incomplete."
    preview, current = rows
    cagr_drop = float(str(preview["CAGR"]).rstrip("%")) - float(str(current["CAGR"]).rstrip("%"))
    sharpe_drop = float(preview["Sharpe"]) - float(current["Sharpe"])
    return (
        "The prior model epoch performed better in the like-for-like rerun: the same anonymous "
        f"minimal-change prompt at temperature 1.0 produced {preview['CAGR']} CAGR and "
        f"{preview['Sharpe']} Sharpe on V4-Flash Preview, versus {current['CAGR']} CAGR and "
        f"{current['Sharpe']} Sharpe on V4-Flash-0731—a {cagr_drop:.2f} percentage-point CAGR drop "
        f"and {sharpe_drop:.3f} Sharpe drop. This is a model-backend break, so results "
        "from the two epochs are historical references—not pooled or directly ranked experiments."
    )


def build_no_regime_result_rows(
    root: Path = DEFAULT_V2_NO_REGIME_ROOT,
) -> list[dict[str, Any]]:
    """Current-epoch paired V2 market-regime input ablation."""
    treatment = _prompt_treatment_primary_row(root, "LLM-Pool v2-no-regime")
    control = _prompt_treatment_primary_row(root, "LLM-Pool v2 Fresh Control")
    trail5 = _prompt_treatment_primary_row(root, "Baseline-Pool")
    if treatment is None or control is None or trail5 is None:
        return []
    config = read_json_if_exists(Path(root) / "run_config.json")
    control_cagr = float(pd.to_numeric(control.get("capital_cagr_pct"), errors="coerce"))
    control_sharpe = float(
        pd.to_numeric(control.get("capital_sharpe", control.get("sharpe")), errors="coerce")
    )
    control_dd = float(pd.to_numeric(control.get("capital_max_drawdown_pct"), errors="coerce"))
    specs = (
        (
            "V2 without Market Regime",
            treatment,
            config.get("prompt_hash", "—"),
            "Omitted from glossary and date input",
            "Treatment result from the completed paired rerun",
        ),
        (
            "Fresh V2 control",
            control,
            "02e28abb83b6",
            "Included",
            "Contemporaneous paired control",
        ),
        ("Trail5", trail5, "deterministic", "Not applicable", "Deterministic benchmark"),
    )
    rows: list[dict[str, Any]] = []
    for label, result, prompt_hash, regime_input, conclusion in specs:
        cagr = float(pd.to_numeric(result.get("capital_cagr_pct"), errors="coerce"))
        sharpe = float(
            pd.to_numeric(result.get("capital_sharpe", result.get("sharpe")), errors="coerce")
        )
        drawdown = float(pd.to_numeric(result.get("capital_max_drawdown_pct"), errors="coerce"))
        rows.append(
            {
                "Run": label,
                "Regime information": regime_input,
                "Prompt hash": prompt_hash,
                "CAGR": fmt_pct(cagr, 2),
                "CAGR change vs fresh V2": fmt_pct(cagr - control_cagr, 2, signed=True)
                if label != "Fresh V2 control"
                else "—",
                "Sharpe": fmt_num(sharpe, 3),
                "Sharpe change vs fresh V2": fmt_num(sharpe - control_sharpe, 3)
                if label != "Fresh V2 control"
                else "—",
                "Max drawdown": fmt_pct(drawdown, 2),
                "Drawdown change vs fresh V2": fmt_pct(drawdown - control_dd, 2, signed=True)
                if label != "Fresh V2 control"
                else "—",
                "Conclusion": conclusion,
            }
        )
    return rows


def build_no_regime_result_read(root: Path = DEFAULT_V2_NO_REGIME_ROOT) -> str:
    rows = {row["Run"]: row for row in build_no_regime_result_rows(root)}
    treatment = rows.get("V2 without Market Regime")
    control = rows.get("Fresh V2 control")
    if treatment is None or control is None:
        return "The paired no-regime experiment is incomplete."
    drawdown_change = str(treatment["Drawdown change vs fresh V2"]).lstrip("+-")
    return (
        f"Removing Market Regime from both the feature glossary and date input produced "
        f"{treatment['CAGR']} CAGR and {treatment['Sharpe']} Sharpe, versus {control['CAGR']} "
        f"and {control['Sharpe']} for the fresh V2 control. That is a {treatment['CAGR change vs fresh V2']} "
        f"CAGR change and {treatment['Sharpe change vs fresh V2']} Sharpe change, with drawdown "
        f"{drawdown_change} {'deeper' if str(treatment['Drawdown change vs fresh V2']).startswith('-') else 'shallower'}. "
        "This is one stochastic paired experiment and should be interpreted as current-epoch evidence, "
        "not as a confirmed replacement."
    )
def build_v22_decision_rows(root: Path = DEFAULT_V22_VISIBLE_ROOT) -> list[dict[str, Any]]:
    path = root / "comparison_scorecard.csv"
    data = read_csv_if_exists(path)
    if data.empty:
        return [
            {
                "Run": "V2.2 lambda grid",
                "Family": "V2.2 contextual adjustment",
                "Pool/Book": "60/60 -> 30/30",
                "Schedule/Anchor": "offset 3",
                "Model": "n/a",
                "Schema": "Optional artifact not found",
                "CAGR": "",
                "Sharpe": "",
                "Max DD": "",
                "IC": "",
                "Conclusion": "Not found",
            }
        ]
    data = data[
        data.get("execution_model", pd.Series(dtype=str)).astype(str).str.upper().eq("C2C")
        & pd.to_numeric(data.get("cost_bps", 0.0), errors="coerce").fillna(0.0).eq(0.0)
        & data.get("anchor_offset", pd.Series(dtype=str)).astype(str).eq("3")
        & ~data.get("is_anchor_average", pd.Series(dtype=bool)).astype(bool)
    ].copy()
    keep = data[
        data.get("run", pd.Series(dtype=str)).astype(str).str.contains("Baseline-Pool|LLM-Pool v2.2", regex=True)
    ].copy()
    if keep.empty:
        return []
    keep = keep.sort_values("capital_cagr_pct", ascending=False)
    best_v22 = keep[keep["run"].astype(str).str.contains("v2.2", case=False, regex=False)]
    best_v22_run = str(best_v22.iloc[0].get("run", "")) if not best_v22.empty else ""
    rows: list[dict[str, Any]] = []
    for _, row in keep.iterrows():
        run = str(row.get("run", ""))
        is_v22 = "v2.2" in run
        rows.append(
            {
                "Run": run,
                "Family": "V2.2 contextual adjustment" if is_v22 else "V2.2 paired baseline",
                "Pool/Book": "60/60 -> 30/30",
                "Schedule/Anchor": "offset 3",
                "Model": "deepseek-chat adjustment cache" if is_v22 else "none",
                "Schema": (
                    "base_score = Trail5 rank; final_score = base_score + lambda * context_adjustment"
                    if is_v22
                    else "deterministic Trail5 baseline"
                ),
                "CAGR": fmt_pct(_metric(row, "capital_cagr_pct", "cagr_pct")),
                "Sharpe": fmt_num(row.get("capital_sharpe", row.get("sharpe")), 2),
                "Max DD": fmt_pct(_metric(row, "capital_max_drawdown_pct", "max_drawdown_pct")),
                "IC": fmt_num(row.get("ic_mean"), 4),
                "Conclusion": (
                    "Best V2.2 but weaker than Trail5"
                    if run == best_v22_run
                    else ("Paired control" if not is_v22 else "Weaker than Trail5")
                ),
            }
        )
    return rows


def build_v23_decision_rows(root: Path = DEFAULT_V23_ROOT) -> list[dict[str, Any]]:
    path = root / "comparison_scorecard.csv"
    data = read_csv_if_exists(path)
    if data.empty:
        return [
            {
                "Run": "V2.3 rank-only",
                "Family": "V2.3 rank-only",
                "Pool/Book": "60/60 -> 30/30",
                "Schedule/Anchor": "offset 3",
                "Model": "n/a",
                "Schema": "Optional V2.3 artifact not found",
                "CAGR": "",
                "Sharpe": "",
                "Max DD": "",
                "IC": "",
                "Conclusion": "Not found",
            }
        ]
    data = data[
        data.get("execution_model", pd.Series(dtype=str)).astype(str).str.upper().eq("C2C")
        & pd.to_numeric(data.get("cost_bps", 0.0), errors="coerce").fillna(0.0).eq(0.0)
        & data.get("anchor_offset", pd.Series(dtype=str)).astype(str).eq("3")
        & ~data.get("is_anchor_average", pd.Series(dtype=bool)).astype(bool)
    ].copy()
    keep = data[
        data.get("run", pd.Series(dtype=str)).astype(str).str.contains("Baseline-Pool|LLM-Pool v2.3", regex=True)
    ].copy()
    if keep.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in keep.sort_values("run").iterrows():
        run = str(row.get("run", ""))
        is_rank = "v2.3" in run
        rows.append(
            {
                "Run": run,
                "Family": "V2.3 rank-only" if is_rank else "V2.3 paired baseline",
                "Pool/Book": "60/60 -> 30/30",
                "Schedule/Anchor": "offset 3",
                "Model": "deepseek-chat rank-order cache" if is_rank else "none",
                "Schema": "complete ordered long/short lists; Python takes top 30" if is_rank else "deterministic Trail5 baseline",
                "CAGR": fmt_pct(_metric(row, "capital_cagr_pct", "cagr_pct")),
                "Sharpe": fmt_num(row.get("capital_sharpe", row.get("sharpe")), 2),
                "Max DD": fmt_pct(_metric(row, "capital_max_drawdown_pct", "max_drawdown_pct")),
                "IC": fmt_num(row.get("ic_mean"), 4),
                "Conclusion": "Useful but weaker" if is_rank else "Paired control",
            }
        )
    return rows


def build_overlay_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "overlay_experiments_pool60_pos30_anchor3" / "comparison_scorecard.csv"
    data = read_csv_if_exists(path)
    if data.empty:
        return [
            {
                "Run": "Overlay experiments",
                "Family": "V2 overlay",
                "Pool/Book": "60/60 -> 30/30",
                "Schedule/Anchor": "offset 3",
                "Model": "n/a",
                "Schema": "Optional artifact not found",
                "CAGR": "",
                "Sharpe": "",
                "Max DD": "",
                "IC": "",
                "Conclusion": "Not found",
            }
        ]
    data = data[
        data.get("execution_model", pd.Series(dtype=str)).astype(str).str.upper().eq("C2C")
        & pd.to_numeric(data.get("cost_bps", 0.0), errors="coerce").fillna(0.0).eq(0.0)
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        rows.append(
            {
                "Run": row.get("run", ""),
                "Family": "V2 overlay",
                "Pool/Book": "60/60 -> 30/30",
                "Schedule/Anchor": "offset 3",
                "Model": "replay from V2 cache",
                "Schema": row.get("description", ""),
                "CAGR": fmt_pct(row.get("capital_cagr_pct")),
                "Sharpe": fmt_num(row.get("capital_sharpe", row.get("sharpe")), 2),
                "Max DD": fmt_pct(row.get("capital_max_drawdown_pct")),
                "IC": "",
                "Conclusion": _conclusion_for(row, "overlay"),
            }
        )
    return rows


def build_concentration_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "concentration_experiments_pool60_anchor3" / "comparison_scorecard.csv"
    data = read_csv_if_exists(path)
    if data.empty:
        return [
            {
                "Run": "Concentration experiments",
                "Family": "V2 concentration",
                "Pool/Book": "60/60 -> variable",
                "Schedule/Anchor": "offset 3",
                "Model": "n/a",
                "Schema": "Optional artifact not found",
                "CAGR": "",
                "Sharpe": "",
                "Max DD": "",
                "IC": "",
                "Conclusion": "Not found",
            }
        ]
    data = data[
        data.get("execution_model", pd.Series(dtype=str)).astype(str).str.upper().eq("C2C")
        & pd.to_numeric(data.get("cost_bps", 0.0), errors="coerce").fillna(0.0).eq(0.0)
    ].copy()
    keep_books = {5, 10, 20, 30, 40, 60}
    data = data[pd.to_numeric(data.get("book_size"), errors="coerce").isin(keep_books)]
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        book = int(pd.to_numeric(row.get("book_size"), errors="coerce"))
        rows.append(
            {
                "Run": f"{row.get('run', '')} book {book}",
                "Family": "V2 concentration",
                "Pool/Book": f"60/60 -> {book}/{book}",
                "Schedule/Anchor": "offset 3",
                "Model": "replay from V2 cache",
                "Schema": row.get("description", ""),
                "CAGR": fmt_pct(row.get("capital_cagr_pct")),
                "Sharpe": fmt_num(row.get("capital_sharpe", row.get("sharpe")), 2),
                "Max DD": fmt_pct(row.get("capital_max_drawdown_pct")),
                "IC": "",
                "Conclusion": _conclusion_for(row, "concentration"),
            }
        )
    return rows


def build_long_breadth_rows(root: Path = DEFAULT_V2_ANON_CONTROL_ROOT) -> list[dict[str, Any]]:
    path = root / "long_breadth_experiments_pool60_pos30_anchor3" / "comparison_scorecard.csv"
    data = read_csv_if_exists(path)
    if data.empty:
        return [
            {
                "Run": "Long breadth experiments",
                "Family": "V2 long breadth",
                "Pool/Book": "60 long candidates; short top-30 fixed",
                "Schedule/Anchor": "offset 3",
                "Model": "n/a",
                "Schema": "Optional anonymized artifact not found",
                "CAGR": "",
                "Sharpe": "",
                "Max DD": "",
                "IC": "",
                "Conclusion": "Not found",
            }
        ]
    data = data[
        data.get("execution_model", pd.Series(dtype=str)).astype(str).str.upper().eq("C2C")
        & pd.to_numeric(data.get("cost_bps", 0.0), errors="coerce").fillna(0.0).eq(0.0)
    ].copy()
    data = data.sort_values("capital_cagr_pct", ascending=False)
    best_run = str(data.iloc[0].get("run", "")) if not data.empty else ""
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        top_weight = pd.to_numeric(row.get("long_top30_weight"), errors="coerce")
        reserve_weight = pd.to_numeric(row.get("long_ranks31_60_weight"), errors="coerce")
        schema = (
            f"Long top-30 {fmt_pct(float(top_weight) * 100.0) if pd.notna(top_weight) else ''}; "
            f"long ranks 31-60 {fmt_pct(float(reserve_weight) * 100.0) if pd.notna(reserve_weight) else ''}; "
            "short top-30 unchanged"
        )
        rows.append(
            {
                "Run": row.get("run", ""),
                "Family": "V2 long breadth",
                "Pool/Book": "60/60 pool; 30 shorts fixed",
                "Schedule/Anchor": "offset 3",
                "Model": "replay from anonymized V2 cache",
                "Schema": schema,
                "CAGR": fmt_pct(row.get("capital_cagr_pct")),
                "Sharpe": fmt_num(row.get("capital_sharpe", row.get("sharpe")), 2),
                "Max DD": fmt_pct(row.get("capital_max_drawdown_pct")),
                "IC": "",
                "Conclusion": "Winner" if str(row.get("run", "")) == best_run else _conclusion_for(row, "long_breadth"),
            }
        )
    return rows


def build_side_decay_rows(root: Path = DEFAULT_V2_ANON_CONTROL_ROOT) -> list[dict[str, Any]]:
    path = root / "side_decay_experiments_pool60_pos30_anchor3" / "comparison_scorecard.csv"
    data = read_csv_if_exists(path)
    if data.empty:
        return [
            {
                "Run": "Side decay experiments",
                "Family": "V2 side decay",
                "Pool/Book": "60/60 pool; side decay variants",
                "Schedule/Anchor": "offset 3",
                "Model": "n/a",
                "Schema": "Optional anonymized artifact not found",
                "CAGR": "",
                "Sharpe": "",
                "Max DD": "",
                "IC": "",
                "Conclusion": "Not found",
            }
        ]
    data = data[
        data.get("execution_model", pd.Series(dtype=str)).astype(str).str.upper().eq("C2C")
        & pd.to_numeric(data.get("cost_bps", 0.0), errors="coerce").fillna(0.0).eq(0.0)
    ].copy()
    data = data.sort_values("capital_cagr_pct", ascending=False)
    best_run = str(data.iloc[0].get("run", "")) if not data.empty else ""
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        schema = (
            f"long top-30 {fmt_pct(float(row.get('long_top30_weight')) * 100.0)}; "
            f"short top-30 {fmt_pct(float(row.get('short_top30_weight')) * 100.0)}"
        )
        rows.append(
            {
                "Run": row.get("run", ""),
                "Family": "V2 side decay",
                "Pool/Book": "60/60 pool; 30/30 control",
                "Schedule/Anchor": "offset 3",
                "Model": "replay from anonymized V2 cache",
                "Schema": schema,
                "CAGR": fmt_pct(row.get("capital_cagr_pct")),
                "Sharpe": fmt_num(row.get("capital_sharpe", row.get("sharpe")), 2),
                "Max DD": fmt_pct(row.get("capital_max_drawdown_pct")),
                "IC": "",
                "Conclusion": "Winner" if str(row.get("run", "")) == best_run else _conclusion_for(row, "side_decay"),
            }
        )
    return rows


def build_side_process_hybrid_rows(root: Path = DEFAULT_V22_VISIBLE_ROOT) -> list[dict[str, Any]]:
    path = root / "side_process_hybrid_experiments_pool60_pos30_anchor3" / "comparison_scorecard_expanded.csv"
    data = read_csv_if_exists(path)
    if data.empty:
        return [
            {
                "Run": "Side-process hybrids",
                "Family": "V2/V2.2 side split",
                "Pool/Book": "60/60 -> 30/30",
                "Schedule/Anchor": "offset 3",
                "Model": "n/a",
                "Schema": "Optional hybrid artifact not found",
                "CAGR": "",
                "Sharpe": "",
                "Max DD": "",
                "IC": "",
                "Conclusion": "Not found",
            }
        ]
    data = data.sort_values("cagr_pct", ascending=False).head(8)
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        long_process = str(row.get("long_process", ""))
        short_process = str(row.get("short_process", ""))
        is_control = long_process == "v2_top30" and short_process == "v2_top30"
        lift = pd.to_numeric(row.get("cagr_lift_vs_v2_pct"), errors="coerce")
        conclusion = "Diagnostic only"
        if pd.notna(lift) and lift > 0.25:
            conclusion = "Useful but small"
        if is_control:
            conclusion = "Control"
        rows.append(
            {
                "Run": f"L={long_process}; S={short_process}",
                "Family": "V2/V2.2 side split",
                "Pool/Book": "60/60 -> 30/30",
                "Schedule/Anchor": "offset 3",
                "Model": "replay from ticker-visible V2 + V2.2 caches",
                "Schema": "long and short processes selected independently",
                "CAGR": fmt_pct(row.get("cagr_pct")),
                "Sharpe": fmt_num(row.get("sharpe"), 2),
                "Max DD": fmt_pct(row.get("max_drawdown_pct")),
                "IC": "",
                "Conclusion": conclusion,
            }
        )
    return rows


def html_table(rows: list[dict[str, Any]], css_class: str = "") -> str:
    if not rows:
        return '<p class="muted">No rows available.</p>'
    columns = list(rows[0].keys())
    head = "".join(f"<th>{_escape(column)}</th>" for column in columns)
    body_parts = []
    for row in rows:
        cells = []
        row_class = f' class="row-{slug(row.get("Conclusion", ""))}"' if "Conclusion" in row else ""
        for column in columns:
            value = row.get(column, "")
            class_attr = ""
            if column == "Conclusion":
                class_attr = f' class="tag-cell tag-{slug(value)}"'
            cells.append(f"<td{class_attr}>{_escape(value)}</td>")
        body_parts.append(f"<tr{row_class}>" + "".join(cells) + "</tr>")
    return f'<table class="{_escape(css_class)}"><thead><tr>{head}</tr></thead><tbody>{"".join(body_parts)}</tbody></table>'


def build_consensus_and_side_experiments_html(
    consensus_root: Path = DEFAULT_CONSENSUS_ROOT,
    side_root: Path = DEFAULT_SIDE_EXPERIMENT_ROOT,
) -> str:
    """Render completed derived artifacts; never run a scorer or backtest."""
    consensus_root, side_root = Path(consensus_root), Path(side_root)
    sections: list[str] = []
    summary_path = consensus_root / "summary.json"
    triplet_path = consensus_root / "triplet_comparison.csv"
    curve_path = consensus_root / "ensemble_size_curve.csv"
    if summary_path.is_file() and triplet_path.is_file() and curve_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        triplets = pd.read_csv(triplet_path)
        curve = pd.read_csv(curve_path)
        triplet_rows = [
            {
                "Triplet": row["source_runs"],
                "CAGR": f"{float(row['cagr_pct']):.2f}%",
                "Sharpe": f"{float(row['sharpe']):.3f}",
                "Max DD": f"{float(row['max_drawdown_pct']):.2f}%",
                "Lift vs member mean": f"{float(row['paired_lift_vs_member_average_pct']):+.2f}pp",
                "Lift vs Trail5": f"{float(row.get('paired_lift_vs_trail5_pct', float('nan'))):+.2f}pp",
                "Primary": "Yes" if row["source_runs"] == "run1+run2+run3" else "No",
            }
            for row in triplets.to_dict("records")
        ]
        curve_rows = [
            {
                "Runs averaged": int(row["ensemble_size"]),
                "Combinations": int(row["combinations"]),
                "Mean CAGR": f"{float(row['mean_cagr_pct']):.2f}%",
                "CAGR range": f"{float(row['min_cagr_pct']):.2f}% to {float(row['max_cagr_pct']):.2f}%",
                "Mean Sharpe": f"{float(row['mean_sharpe']):.3f}",
            }
            for row in curve.to_dict("records")
        ]
        hypothesis = summary.get("hypothesis", {})
        provenance = summary.get("provenance", {})
        sections.append(
            '<div class="section-heading"><div><div class="section-label">Offline consensus</div>'
            '<h2>V2 Three-Run Rank Consensus</h2><p>Four predeclared 3-of-4 ensembles and the full ensemble-size curve, derived only from immutable V2 caches.</p></div></div>'
            f'<div class="callout"><strong>Decision:</strong> {_escape(str(hypothesis.get("label", "unknown")).replace("_", " ").title())}. '
            f'No-regime controls included: {_escape(provenance.get("no_regime_paired_controls_included", False))}; '
            f'new LLM calls: {_escape(provenance.get("llm_calls_made", 0))}. '
            '<a href="../v2_three_run_consensus_2009_2014/report.html">Open the full consensus report</a>.</div>'
            f'<div class="table-wrap">{html_table(triplet_rows, "wide-table")}</div>'
            f'<div class="table-wrap">{html_table(curve_rows, "wide-table")}</div>'
        )
    else:
        sections.append('<div class="callout"><strong>V2 consensus:</strong> Optional artifact not found.</div>')

    side_result = side_root / "fixed_prompt_result.json"
    side_scorecard = side_root / "fixed_prompt_hybrid_scorecard.csv"
    if side_result.is_file() and side_scorecard.is_file():
        result = json.loads(side_result.read_text(encoding="utf-8"))
        rows = [
            {
                "Configuration": row["configuration"],
                "CAGR": f"{float(row['cagr_pct']):.2f}%",
                "Sharpe": f"{float(row['sharpe']):.3f}",
                "Max DD": f"{float(row['max_drawdown_pct']):.2f}%",
                "Long contribution": f"{float(row['annualized_long_contribution_pct']):.2f}%",
                "Short contribution": f"{float(row['annualized_short_contribution_pct']):.2f}%",
            }
            for row in pd.read_csv(side_scorecard).to_dict("records")
        ]
        sections.append(
            '<div class="section-heading" style="margin-top:22px"><div><div class="section-label">Fixed side test</div>'
            '<h2>Dedicated Long And Short Scorers</h2><p>Each hybrid keeps the untouched side equal to the completed joint minimal-change book.</p></div></div>'
            f'<div class="callout"><strong>Qualified sides:</strong> {_escape(", ".join(result.get("autoresearch_sides", [])) or "none")}.</div>'
            f'<div class="table-wrap">{html_table(rows, "wide-table")}</div>'
        )
    else:
        sections.append('<div class="callout"><strong>Dedicated-side fixed test:</strong> Awaiting the two one-run side caches; the joint baseline will not be rerun.</div>')
    combined_rows: list[dict[str, Any]] = []
    for period in ("development", "validation"):
        path = side_root / "combined" / f"combined_{period}_result.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            combined_rows.append(
                {
                    "Period": period.title(),
                    "Winning sides": ", ".join(payload.get("sides", [])),
                    "CAGR": f"{float(metrics.get('cagr_pct', float('nan'))):.2f}%",
                    "Sharpe": f"{float(metrics.get('sharpe', float('nan'))):.3f}",
                    "Max DD": f"{float(metrics.get('max_drawdown_pct', float('nan'))):.2f}%",
                    "Other-side policy": payload.get("opposite_side_policy", ""),
                }
            )
    if combined_rows:
        sections.append(
            '<div class="section-heading" style="margin-top:22px"><div><div class="section-label">Frozen winners</div>'
            '<h2>Combined Side-Campaign Performance</h2></div></div>'
            f'<div class="table-wrap">{html_table(combined_rows, "wide-table")}</div>'
        )
    return "\n".join(sections)


def prompt_card_html(card: PromptCard) -> str:
    rows = [
        ("Family", card.family),
        ("Schema", card.schema),
        ("Asks for", card.asks_for),
        ("Reasoning visibility", card.reasoning_visibility),
        ("Pool/process", card.pool_process),
        ("Accounting", card.accounting),
        ("Cache/replay", card.cache_replay),
        ("Status", card.status),
    ]
    meta = "".join(f"<dt>{_escape(label)}</dt><dd>{_escape(value)}</dd>" for label, value in rows)
    return f"""
      <article class="prompt-card prompt-{slug(card.family)}">
        <div class="card-kicker">
          <span>{_escape(card.family)}</span>
          <span>{_escape(card.status)}</span>
        </div>
        <h3>{_escape(card.title)}</h3>
        <dl>{meta}</dl>
        <details>
          <summary>Full prompt</summary>
          <pre>{_escape(card.prompt_text.strip())}</pre>
        </details>
      </article>
    """


def concept_card_html(card: ConceptCard) -> str:
    return f"""
      <article class="concept-card">
        <div class="concept-topline">
          <h3>{_escape(card.concept)}</h3>
          <span class="pill priority-{slug(card.priority)}">{_escape(card.priority)}</span>
        </div>
        <p><strong>Hypothesis:</strong> {_escape(card.hypothesis)}</p>
        <p><strong>Why it might help:</strong> {_escape(card.why_it_might_help)}</p>
        <p><strong>Risk:</strong> {_escape(card.risk)}</p>
        <div class="card-footer">
          <span>{_escape(card.expected_token_cost)}</span>
          <span>{_escape(card.status)}</span>
        </div>
      </article>
    """


def build_experiment_gameplan_rows() -> list[dict[str, str]]:
    return [
        {
            "Priority": "1",
            "Experiment": "V2 unchanged control",
            "Prompt change": "None",
            "Harness": "offset 3, 60/60 -> 30/30, gross 5-day C2C",
            "Success read": "Anchor/lab control for all prompt A/Bs",
        },
        {
            "Priority": "2",
            "Experiment": "V2.2 soft multi-feature nudge",
            "Prompt change": "Keep V2 wording; add one line: Trail5/RSI5 matter, but other features can confirm or contradict the setup",
            "Harness": "Same as control",
            "Success read": "Keeps V2 score/Trail5 relationship while improving IC or drawdown",
        },
        {
            "Priority": "3",
            "Experiment": "V2.3 rank-only ordering",
            "Prompt change": "Keep the V2 setup framing but ask for complete ordered lists instead of 0-100 scores",
            "Harness": "Same as control",
            "Success read": "Tests whether ordinal judgment beats numeric score calibration",
        },
        {
            "Priority": "4",
            "Experiment": "V2.4 raw market context (complete)",
            "Prompt change": "Keep the frozen no-glossary prompt exact; replace regime label with raw SPY 200-day-MA and VIX measurements",
            "Harness": "Same as control; pair with frozen no-glossary scores",
            "Success read": "Measured near-flat versus Trail5, with better IC/drawdown but no CAGR lift",
        },
        {
            "Priority": "5",
            "Experiment": "V2.5 balanced outcome memory (complete)",
            "Prompt change": "Keep current V2 contract; add past-only HistN, HistWin, and HistRet analog summaries",
            "Harness": "Same as control; fresh V2, Trail5, and HistRet-only paired controls",
            "Success read": "Did not pass: memory-ready CAGR was -1.7pp versus fresh V2 and -2.6pp versus Trail5, despite a smaller drawdown",
        },
        {
            "Priority": "6",
            "Experiment": "Expected 5-day reversal strength",
            "Prompt change": "Define score as setup strength for the next 5 trading days without saying to ignore Trail5/RSI5",
            "Harness": "Same as control",
            "Success read": "CAGR/Sharpe lift without worse drawdown",
        },
        {
            "Priority": "7",
            "Experiment": "Soft continuation caution",
            "Prompt change": "One caution line about repricing/falling-knife/squeeze risk; no feature-specific hard gates",
            "Harness": "Same as control",
            "Success read": "Improves long-side contribution and drawdown without killing Trail5 exposure",
        },
        {
            "Priority": "8",
            "Experiment": "Long/short specialized criteria",
            "Prompt change": "Use separate criteria for snap-back longs and fade shorts inside same JSON contract",
            "Harness": "Same as control",
            "Success read": "Side attribution shows lift on the weaker side without hurting the stronger side",
        },
        {
            "Priority": "9",
            "Experiment": "Ticker identity audit",
            "Prompt change": "Compare ticker-visible, anonymized, and shuffled ticker-label score stability; optional label audit includes a ticker_identity option",
            "Harness": "Same as control; preferably score-only first, labels on a small sample",
            "Success read": "If shuffled names move scores materially, ticker identity is influencing the model",
        },
        {
            "Priority": "10",
            "Experiment": "Small-sample explanation capture",
            "Prompt change": "Add short rationale labels for 20-30 dates only",
            "Harness": "Diagnostic sample, not production scoring",
            "Success read": "Clear failure taxonomy for next prompt iteration",
        },
    ]


def build_experiment_purpose_rows() -> list[dict[str, str]]:
    return [
        {
            "Experiment": "V2 pure LLM vs Trail5",
            "What It Tests": "Whether the unchanged V2 score prompt adds stock-selection value beyond the deterministic Trail5 baseline under the exact same pool, book, anchor, and accounting.",
            "Useful Read Even If It Fails": "If LLM loses, the prompt is not adding incremental ranking quality; if it wins, the LLM is improving selection inside the reversal pool.",
            "Current Takeaway": "V2 LLM beats Trail5 on the key offset 3 lab bench, especially in the ticker-visible historical run and still in the anonymized control.",
        },
        {
            "Experiment": "Ticker-visible vs anonymized V2",
            "What It Tests": "Whether showing real tickers gives the model helpful or dangerous identity priors versus forcing it to use only the numeric setup features.",
            "Useful Read Even If It Fails": "A performance drop after masking means ticker identity was contributing something; that could be leakage-like prior knowledge or legitimate sector/company context not otherwise provided.",
            "Current Takeaway": "Masked V2 still works, but ticker-visible V2 was stronger, so ticker identity likely mattered.",
        },
        {
            "Experiment": "Structured labels / feature attribution",
            "What It Tests": "Whether forcing the model to name setup labels and primary features improves selection and reveals which inputs it claims to rely on.",
            "Useful Read Even If It Fails": "Even if performance drops, the labels help debug prompt behavior, feature emphasis, and which cited features are associated with good or bad outcomes.",
            "Current Takeaway": "Useful diagnostic, weaker as a production prompt; labels are for insight, not proof of internal reasoning.",
        },
        {
            "Experiment": "V2.1 discriminative prompt",
            "What It Tests": "Whether asking for incremental reversal quality beyond Trail5/RSI5 makes the model find subtler setup quality instead of just re-ranking obvious reversal extremes.",
            "Useful Read Even If It Fails": "A failure tells us the prompt over-penalized the core reversal signal or made the model too conservative.",
            "Current Takeaway": "It failed; V2.1 moved away from the strongest reversal exposure.",
        },
        {
            "Experiment": "V2.3 rank-only ordering",
            "What It Tests": "Whether the LLM is better at relative ordering than producing calibrated 0-100 scores.",
            "Useful Read Even If It Fails": "If rank-only loses to score-only, the numeric V2 scoring contract is not just noise; its score scale or calibration still contains useful selection information.",
            "Current Takeaway": "It slightly beat Trail5, but did not beat anonymized V2 score-only on CAGR or total return.",
        },
        {
            "Experiment": "V2.4 raw SPY/VIX context",
            "What It Tests": "Whether replacing the coarse semantic regime label with raw SPY 200-day-MA and VIX measurements improves the exact frozen no-glossary prompt.",
            "Useful Read Even If It Fails": "A failure would show that raw market context alone does not recover the glossary prompt's primary equal-weight advantage.",
            "Current Takeaway": "Completed on 2009-2014: 13.0% CAGR versus 13.2% for Trail5, essentially equal Sharpe, better drawdown, and higher IC. Raw context improved ranking diagnostics but did not add CAGR.",
        },
        {
            "Experiment": "V2.5 balanced outcome memory",
            "What It Tests": "Whether past-only outcomes from similar same-side candidates improve V2 scoring beyond a fresh same-run V2 control.",
            "Useful Read Even If It Fails": "The HistRet-only diagnostic separates retrieval quality from the LLM's ability to use the retrieved evidence.",
            "Current Takeaway": "The paid paired test is complete. On memory-ready dates V2.5 returned 10.5% CAGR versus 12.2% for fresh V2 and 13.0% for Trail5; drawdown improved to -7.4%.",
        },
        {
            "Experiment": "LLM-long / Trail5-short",
            "What It Tests": "Whether the LLM edge is mainly on selecting rebound longs while keeping deterministic shorts fixed.",
            "Useful Read Even If It Fails": "It isolates side contribution; if this works but the reverse does not, future prompt work should focus on the long side.",
            "Current Takeaway": "Close to pure V2 but did not beat it; supports that long-side LLM selection carries useful information.",
        },
        {
            "Experiment": "Trail5-long / LLM-short",
            "What It Tests": "Whether the LLM edge is mainly on the short book, especially avoiding squeeze/continuation names.",
            "Useful Read Even If It Fails": "A weak result says short-side LLM changes are not enough by themselves or may be less important than long selection.",
            "Current Takeaway": "Weaker than pure V2; short-side changes alone were not the main answer.",
        },
        {
            "Experiment": "Top-k LLM swaps",
            "What It Tests": "Whether small surgical LLM substitutions into a Trail5 book add value without letting the LLM fully control the portfolio.",
            "Useful Read Even If It Fails": "It tells whether LLM alpha is concentrated in a few high-conviction disagreements or requires full-book ranking authority.",
            "Current Takeaway": "Helpful diagnostic, but none beat pure V2; LLM seems better as full selector than a tiny patch on Trail5.",
        },
        {
            "Experiment": "Residual LLM score",
            "What It Tests": "Whether the part of the LLM ranking that differs from Trail5 contains independent alpha after removing the obvious Trail5 component.",
            "Useful Read Even If It Fails": "It separates 'LLM is just rediscovering Trail5' from 'LLM has useful disagreement.' Lambda behavior shows how much disagreement the book can tolerate.",
            "Current Takeaway": "Some signal exists, but pure V2 still held up better than residual wrappers.",
        },
        {
            "Experiment": "Concentration / book-size curve",
            "What It Tests": "How many LLM-ranked names should actually be trusted: 5/5, 10/10, 20/20, 30/30, 40/40, or 60/60.",
            "Useful Read Even If It Fails": "It shows whether the signal is strongest at the very top, needs diversification, or decays slowly through the pool.",
            "Current Takeaway": "30/30 remains the best core V2 book; too concentrated gets noisy, too broad dilutes edge.",
        },
        {
            "Experiment": "Long rank curve",
            "What It Tests": "Whether V2's long ranking actually separates winners from losers across ranks 1-10, 11-20, 21-30, and rejected 31-60 names.",
            "Useful Read Even If It Fails": "If rejected buckets still work, the 30-name cutoff may be arbitrary or too harsh; if rank order declines smoothly, top-30 selection is justified.",
            "Current Takeaway": "In masked V2, top 1-10 is strong, 21-30 is weak, and 41-50 rebounds; the curve is not cleanly monotonic.",
        },
        {
            "Experiment": "Broad long core + LLM tilt",
            "What It Tests": "Whether spreading some long capital across rejected longs improves the portfolio while keeping the V2 top-30 shorts unchanged.",
            "Useful Read Even If It Fails": "It distinguishes alpha from risk smoothing: reserve sleeves may reduce drawdown even if they do not improve CAGR.",
            "Current Takeaway": "On masked V2, linear rank decay beats current top-30 on CAGR and Sharpe, but bootstrap evidence is positive rather than decisive.",
        },
        {
            "Experiment": "Side decay / short rank curve",
            "What It Tests": "Whether the hard short cutoff is justified, and whether linear decay should be applied to longs only, shorts only, or both sides.",
            "Useful Read Even If It Fails": "If short decay lowers CAGR but improves drawdown, the short ranking is doing useful cutoff work while broader shorts mainly diversify risk.",
            "Current Takeaway": "On masked V2, long decay wins; short decay reduces CAGR but improves drawdown, and lower short ranks have weaker short P&L.",
        },
        {
            "Experiment": "Anchor/offset robustness",
            "What It Tests": "Whether the result depends on one lucky rebalance schedule or survives across all five fixed 5-trading-day offsets.",
            "Useful Read Even If It Fails": "Large offset dispersion warns that timing/seasonality matters and that a single-schedule result should not be overread.",
            "Current Takeaway": "Offset 3 is strongest and legacy-aligned; anchor average is more conservative.",
        },
    ]


def v2_model_question_grid_html() -> str:
    questions = [
        "How extreme was the 5-day move?",
        "Is the stock oversold/overbought?",
        "Is it near range extremes?",
        "Is the move fast relative to vol?",
        "Is it stretched versus MA / 52-week high?",
        "Is medium-term momentum helping or hurting?",
        "Does the stock tend to reverse or continue?",
        "Is trend strength high?",
        "Is current volatility unusually high?",
        "What market regime are we in?",
    ]
    items = []
    for index, question in enumerate(questions, start=1):
        items.append(
            f"""
            <div class="question-card">
              <span class="question-num">{index}</span>
              <span>{_escape(question)}</span>
            </div>
            """
        )
    return "\n".join(items)


def build_prompt_diff_rows() -> list[dict[str, str]]:
    return [
        {
            "Dimension": "Core ask",
            "V2": "Score mean-reversion setup strength.",
            "V2.1": "Score incremental setup quality beyond Trail5 / RSI5.",
            "Lesson": "V2.1 demoted the core reversal signal instead of lightly broadening attention.",
        },
        {
            "Dimension": "Trail5 / RSI5 treatment",
            "V2": "Part of the general extreme-indicator context.",
            "V2.1": "Explicitly says they are not enough and should not be simply ranked.",
            "Lesson": "The model moved away from the strongest known signal.",
        },
        {
            "Dimension": "Risk language",
            "V2": "Minimal caution language.",
            "V2.1": "Falling knife, continuation, squeeze, high-vol continuation warnings.",
            "Lesson": "The prompt became too conservative for an intentionally ugly reversal book.",
        },
        {
            "Dimension": "Feature steering",
            "V2": "Broad: extreme move + exhaustion + volume/range/regime.",
            "V2.1": "Harder steering toward named secondary features.",
            "Lesson": "The diagnostics were overfit into the prompt.",
        },
        {
            "Dimension": "Harness / schema",
            "V2": "60/60 -> 30/30, offset 3, gross fixed 5-day C2C, score-only JSON.",
            "V2.1": "Same.",
            "Lesson": "The performance delta is from prompt/score behavior, not accounting or replay mechanics.",
        },
    ]


def _load_best_offset_row(root: Path, run_contains: str) -> pd.Series | None:
    path = root / "gross_legacy_v1_v2_comparison.csv"
    data = read_csv_if_exists(path)
    if data.empty:
        path = root / "apples_to_apples_gross_c2c_comparison.csv"
        data = read_csv_if_exists(path)
    if data.empty:
        return None
    mask = (
        data.get("execution_model", pd.Series(dtype=str)).astype(str).str.upper().eq("C2C")
        & pd.to_numeric(data.get("cost_bps", 0.0), errors="coerce").fillna(0.0).eq(0.0)
        & data.get("run", pd.Series(dtype=str)).astype(str).str.contains(run_contains, case=False, regex=False)
        & data.get("anchor_offset", pd.Series(dtype=str)).astype(str).eq("3")
    )
    rows = data[mask].copy()
    if rows.empty:
        return None
    rows = rows[~rows.get("family", pd.Series(dtype=str)).astype(str).str.contains("Reference", case=False, na=False)]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_prompt_ab_rows(
    original_root: Path = DEFAULT_RESULTS_ROOT,
    anon_control_root: Path = DEFAULT_V2_ANON_CONTROL_ROOT,
    structured_root: Path = DEFAULT_V2_STRUCTURED_ROOT,
    v21_root: Path = DEFAULT_V21_ROOT,
    v22_root: Path = DEFAULT_V22_VISIBLE_ROOT,
    v23_root: Path = DEFAULT_V23_ROOT,
    v24_root: Path = DEFAULT_V24_ROOT,
) -> list[dict[str, Any]]:
    specs = [
        ("Trail5 baseline", anon_control_root, "Baseline-Pool", "No LLM; deterministic Trail5 control", "paired control"),
        ("V2 ticker-visible", original_root, "LLM-Pool v2 Best Offset", "Original V2 score prompt; real ticker symbols visible", "historical winner"),
        ("V2 anonymized", anon_control_root, "LLM-Pool v2", "Same V2 score prompt; tickers replaced with SEC_ IDs", "best clean control"),
        ("V2 anonymized + labels", structured_root, "LLM-Pool v2", "Same anonymized setup plus structured setup labels/features", "diagnostic, weaker"),
        ("V2.1 anonymized", v21_root, "LLM-Pool v2.1", "Anonymized discriminative/residual prompt", "failed"),
        ("V2.2 visible lambda=2", v22_root, "LLM-Pool v2.2 lambda=2", "Trail5 base_score plus LLM -3..+3 contextual adjustment; lambda 2", "weaker than Trail5"),
        ("V2.2 visible lambda=4", v22_root, "LLM-Pool v2.2 lambda=4", "Trail5 base_score plus LLM -3..+3 contextual adjustment; lambda 4", "weaker than Trail5"),
        ("V2.2 visible lambda=6", v22_root, "LLM-Pool v2.2 lambda=6", "Trail5 base_score plus LLM -3..+3 contextual adjustment; lambda 6", "weaker than Trail5"),
        ("V2.2 visible lambda=8", v22_root, "LLM-Pool v2.2 lambda=8", "Trail5 base_score plus LLM -3..+3 contextual adjustment; lambda 8", "best V2.2, still weaker"),
        ("V2.3 anonymized rank-only", v23_root, "LLM-Pool v2.3", "Complete side-relative rank ordering only; Python selects top 30", "useful but weaker than V2 score"),
        (
            "V2.4 anonymized raw SPY/VIX",
            v24_root,
            "LLM-Pool v2.4",
            "Frozen no-glossary score prompt; raw SPY/VIX measurements replace the semantic regime label",
            "near-flat versus Trail5; better IC and drawdown",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for label, root, run_contains, prompt, conclusion in specs:
        row = _load_best_offset_row(root, run_contains)
        if row is None:
            rows.append(
                {
                    "Run": label,
                    "Prompt / Process": prompt,
                    "Raw CAGR": "",
                    "50/50 CAGR": "",
                    "Sharpe": "",
                    "50/50 Max DD": "",
                    "IC": "",
                    "Conclusion": "Not found",
                }
            )
            continue
        rows.append(
            {
                "Run": label,
                "Prompt / Process": prompt,
                "Raw CAGR": fmt_pct(row.get("cagr_pct")),
                "50/50 CAGR": fmt_pct(_metric(row, "capital_cagr_pct", "cagr_pct")),
                "Sharpe": fmt_num(row.get("capital_sharpe", row.get("sharpe")), 3),
                "50/50 Max DD": fmt_pct(_metric(row, "capital_max_drawdown_pct", "max_drawdown_pct")),
                "IC": fmt_num(row.get("ic_mean"), 4),
                "Conclusion": conclusion,
            }
        )
    return rows


def build_phase2_glossary_ablation_rows(
    scorecard_path: Path = DEFAULT_PHASE2_SCORECARD,
) -> list[dict[str, Any]]:
    data = read_csv_if_exists(scorecard_path)
    if data.empty:
        return [
            {
                "Construction": "2015-2019 prompt ablation",
                "Trail5 CAGR": "",
                "No-glossary CAGR": "",
                "No-glossary lift": "",
                "Glossary CAGR": "",
                "Glossary lift": "",
                "Glossary minus no glossary": "",
                "Conclusion": "Pending",
            }
        ]
    glossary = data[data.get("period", pd.Series(dtype=str)).eq(phase2.GLOSSARY_VALIDATION_PERIOD)]
    no_glossary = data[data.get("period", pd.Series(dtype=str)).eq(phase2.NO_GLOSSARY_VALIDATION_PERIOD)]
    glossary_by_id = glossary.set_index("test_id") if not glossary.empty else pd.DataFrame()
    no_glossary_by_id = no_glossary.set_index("test_id") if not no_glossary.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for test in phase2.FROZEN_TESTS:
        if test.test_id not in glossary_by_id.index or test.test_id not in no_glossary_by_id.index:
            continue
        glossary_row = glossary_by_id.loc[test.test_id]
        no_glossary_row = no_glossary_by_id.loc[test.test_id]
        prompt_delta = float(glossary_row["llm_cagr_pct"] - no_glossary_row["llm_cagr_pct"])
        rows.append(
            {
                "Construction": test.label,
                "Trail5 CAGR": fmt_pct(glossary_row["trail5_cagr_pct"]),
                "No-glossary CAGR": fmt_pct(no_glossary_row["llm_cagr_pct"]),
                "No-glossary lift": fmt_pct(no_glossary_row["cagr_lift_pp"], signed=True),
                "Glossary CAGR": fmt_pct(glossary_row["llm_cagr_pct"]),
                "Glossary lift": fmt_pct(glossary_row["cagr_lift_pp"], signed=True),
                "Glossary minus no glossary": fmt_pct(prompt_delta, signed=True),
                "Conclusion": "Glossary stronger" if prompt_delta > 0 else "No glossary stronger",
            }
        )
    return rows


def build_phase2_glossary_ablation_read(
    scorecard_path: Path = DEFAULT_PHASE2_SCORECARD,
) -> str:
    data = read_csv_if_exists(scorecard_path)
    if data.empty:
        return "The 2015-2019 glossary ablation is pending."
    subset = data[data.get("test_id", pd.Series(dtype=str)).eq("top30_equal")]
    by_period = subset.set_index("period") if not subset.empty else pd.DataFrame()
    if (
        by_period.empty
        or phase2.GLOSSARY_VALIDATION_PERIOD not in by_period.index
        or phase2.NO_GLOSSARY_VALIDATION_PERIOD not in by_period.index
    ):
        return "The 2015-2019 glossary ablation is pending."
    glossary = by_period.loc[phase2.GLOSSARY_VALIDATION_PERIOD]
    no_glossary = by_period.loc[phase2.NO_GLOSSARY_VALIDATION_PERIOD]
    delta = float(glossary["llm_cagr_pct"] - no_glossary["llm_cagr_pct"])
    return (
        "Original equal-weight 30/30: no glossary returned "
        f"{fmt_pct(no_glossary['llm_cagr_pct'])} CAGR versus {fmt_pct(no_glossary['trail5_cagr_pct'])} "
        f"for Trail5 ({fmt_pct(no_glossary['cagr_lift_pp'], signed=True)} lift); glossary returned "
        f"{fmt_pct(glossary['llm_cagr_pct'])} ({fmt_pct(glossary['cagr_lift_pp'], signed=True)} lift). "
        f"Glossary therefore added {fmt_pct(delta, signed=True)} versus no glossary on the primary book. "
        "No glossary was stronger on five of the six concentrated constructions, so the effect depends on the wrapper."
    )


def build_v24_v25_architecture_rows(
    v24_root: Path = DEFAULT_V24_ROOT,
    v25_root: Path = DEFAULT_V25_ROOT,
) -> list[dict[str, Any]]:
    v24_row = _load_best_offset_row(v24_root, "LLM-Pool v2.4")
    v24_complete = v24_row is not None
    v24_config = read_json_if_exists(v24_root / "run_config.json")
    memory_dir = v25_root / "v25_balanced_outcome_memory"
    coverage = read_csv_if_exists(memory_dir / "coverage_by_date.csv")
    v25_results = read_csv_if_exists(memory_dir / "coverage_adjusted_paired_scorecard.csv")
    completed_runs = set(v25_results.get("run", pd.Series(dtype=str)).astype(str))
    v25_complete = {"V2 Fresh Control", "V2.5 Balanced Memory"}.issubset(completed_runs)
    ready = int(coverage.get("memory_ready", pd.Series(dtype=bool)).astype(bool).sum()) if not coverage.empty else 0
    total = len(coverage)
    return [
        {
            "Version": "V2.4",
            "Prompt base": "Exact frozen historical no-glossary 0-100 prompt",
            "Isolated addition": "Raw SPY relation/distance to 200-day MA plus VIX level and 252-day percentile; no regime label",
            "Output": "Unchanged long_scores / short_scores",
            "Required control": "Matched Trail5 is complete; a frozen no-context LLM control would isolate raw-context value",
            "Hash": v2.prompt_hash_for_variant(v2.PROMPT_VARIANT_V24),
            "Status": (
                f"Paid run complete on {v24_config.get('model', 'recorded model')}"
                if v24_complete
                else "Implemented; paid run pending"
            ),
        },
        {
            "Version": "V2.5",
            "Prompt base": "Current glossary-backed V2 0-100 prompt",
            "Isolated addition": "HistN, HistWin, HistRet from balanced past-only same-side analogs",
            "Output": "Unchanged long_scores / short_scores",
            "Required control": "Fresh V2 scores in the same run, plus Trail5 and HistRet-only diagnostics",
            "Hash": read_json_if_exists(memory_dir / "memory_spec.json").get("spec_hash", v2.v25m.memory_spec_hash()),
            "Status": (
                "Paid paired run complete"
                if v25_complete
                else f"Memory audit complete ({ready}/{total} dates ready); paid A/B pending"
            ),
        },
    ]


def build_v25_status_read(v25_root: Path = DEFAULT_V25_ROOT) -> str:
    path = v25_root / "v25_balanced_outcome_memory/coverage_adjusted_paired_scorecard.csv"
    metrics = read_csv_if_exists(path)
    completed_runs = set(metrics.get("run", pd.Series(dtype=str)).astype(str))
    required = {"Trail5", "HistRet-Only", "V2 Fresh Control", "V2.5 Balanced Memory"}
    if not required.issubset(completed_runs):
        present = ", ".join(sorted(completed_runs)) if completed_runs else "none"
        missing = ", ".join(sorted(required - completed_runs))
        return (
            f"Memory construction is complete, but the paid paired comparison is incomplete. "
            f"Present report rows: {present}. Missing: {missing}. Do not treat the scorecard as a V2.5 result yet."
        )
    ready = metrics[
        metrics.get("scope", pd.Series(dtype=str)).eq("memory_ready")
        & metrics.get("run", pd.Series(dtype=str)).eq("V2.5 Balanced Memory")
    ]
    if ready.empty:
        return "The paid V2.5 rows exist, but the memory-ready treatment row is missing."
    row = ready.iloc[0]
    return (
        f"Memory-ready V2.5 produced {fmt_pct(row['cagr_pct'])} CAGR and {fmt_num(row['sharpe'], 2)} Sharpe, "
        f"with {fmt_pp(row['cagr_lift_vs_trail5_pp'])} CAGR lift versus Trail5 and "
        f"{fmt_pp(row['cagr_lift_vs_v2_pp'])} versus the fresh V2 control."
    )


def build_v25_paired_result_rows(v25_root: Path = DEFAULT_V25_ROOT) -> list[dict[str, Any]]:
    path = v25_root / "v25_balanced_outcome_memory/coverage_adjusted_paired_scorecard.csv"
    metrics = read_csv_if_exists(path)
    if metrics.empty:
        return [
            {
                "Scope": "Pending",
                "Run": "No paired rows found",
                "Periods": "",
                "CAGR": "",
                "Sharpe": "",
                "Max DD": "",
                "Lift vs Trail5": "",
                "Lift vs fresh V2": "",
                "Evidence": "Pending",
            }
        ]
    rows = []
    for row in metrics.itertuples(index=False):
        run = str(row.run)
        rows.append(
            {
                "Scope": str(row.scope).replace("_", " "),
                "Run": run,
                "Periods": int(row.n_periods),
                "CAGR": fmt_pct(row.cagr_pct),
                "Sharpe": fmt_num(row.sharpe, 2),
                "Max DD": fmt_pct(row.max_drawdown_pct),
                "Lift vs Trail5": fmt_pp(getattr(row, "cagr_lift_vs_trail5_pp", float("nan"))),
                "Lift vs fresh V2": fmt_pp(getattr(row, "cagr_lift_vs_v2_pp", float("nan"))),
                "Evidence": (
                    "Treatment"
                    if run == "V2.5 Balanced Memory"
                    else "Paired control"
                    if run in {"Trail5", "V2 Fresh Control"}
                    else "Deterministic diagnostic"
                ),
            }
        )
    return rows


def build_v25_memory_coverage_rows(v25_root: Path = DEFAULT_V25_ROOT) -> list[dict[str, Any]]:
    memory_dir = v25_root / "v25_balanced_outcome_memory"
    coverage = read_csv_if_exists(memory_dir / "coverage_by_date.csv")
    candidates = read_csv_if_exists(memory_dir / "candidate_memory.csv")
    neighbors = read_csv_if_exists(memory_dir / "neighbor_audit.csv")
    cases = read_csv_if_exists(memory_dir / "case_history.csv")
    if coverage.empty:
        return [{"Audit metric": "V2.5 memory audit", "Value": "Not found", "Interpretation": "Run the zero-token memory audit first."}]
    total = len(coverage)
    ready = int(coverage["memory_ready"].astype(bool).sum())
    full = int(coverage["fully_populated"].astype(bool).sum())
    return [
        {"Audit metric": "Rebalance dates", "Value": f"{total:,}", "Interpretation": "2009-2014 offset-3 pools audited"},
        {"Audit metric": "Memory-ready dates", "Value": f"{ready:,}/{total:,} ({ready / total:.1%})", "Interpretation": "At least 10 analogs across at least 3 prior dates for every candidate"},
        {"Audit metric": "Fully populated dates", "Value": f"{full:,}/{total:,} ({full / total:.1%})", "Interpretation": "20 analogs across at least 5 prior dates for every candidate"},
        {"Audit metric": "Candidate-date rows", "Value": f"{len(candidates):,}", "Interpretation": "HistN / HistWin / HistRet summaries available for audit"},
        {"Audit metric": "Neighbor links", "Value": f"{len(neighbors):,}", "Interpretation": "Auditable point-in-time analog assignments"},
        {"Audit metric": "Historical cases", "Value": f"{len(cases):,}", "Interpretation": "Prior candidates with realized side-adjusted five-day outcomes"},
    ]


def build_v25_postmortem_read(v25_root: Path = DEFAULT_V25_ROOT) -> str:
    path = v25_root / "v25_balanced_outcome_memory/memory_postmortem/postmortem_summary.json"
    summary = read_json_if_exists(path)
    if not summary:
        return "The zero-token memory postmortem has not been generated."
    return (
        f"Memory changed scores materially (mean absolute memory-to-score rank correlation "
        f"{fmt_num(summary.get('mean_absolute_spearman_memory_vs_score_delta'), 3)}), but its strongest "
        f"absolute correlation with future side-adjusted return was only "
        f"{fmt_num(summary.get('strongest_absolute_spearman_memory_vs_future'), 3)}. "
        f"V2.5-added names versus dropped names produced {fmt_pct(summary.get('v25_added_minus_dropped_pct'), 2, signed=True)} "
        f"per rebalance, with a 95% interval from {fmt_pct(summary.get('v25_added_minus_dropped_ci_low_pct'), 2)} "
        f"to {fmt_pct(summary.get('v25_added_minus_dropped_ci_high_pct'), 2)}. "
        f"{summary.get('recommendation', '')}"
    )


def build_v25_postmortem_correlation_rows(v25_root: Path = DEFAULT_V25_ROOT) -> list[dict[str, Any]]:
    path = v25_root / "v25_balanced_outcome_memory/memory_postmortem/memory_score_correlations.csv"
    frame = read_csv_if_exists(path)
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        future = float(row.spearman_memory_vs_future_side_pnl)
        score = float(row.spearman_memory_vs_v25_score_delta)
        rows.append(
            {
                "Side": row.side,
                "Memory field": str(row.memory_field).replace("hist_", "Hist").replace("_pct", ""),
                "Memory -> future P&L": fmt_num(future, 3),
                "Memory -> V2.5 score change": fmt_num(score, 3),
                "Read": (
                    "Model reacted; memory did not predict"
                    if abs(score) >= 0.15 and abs(future) < 0.05
                    else "Inspect"
                ),
            }
        )
    return rows or [{"Side": "", "Memory field": "Not found", "Memory -> future P&L": "", "Memory -> V2.5 score change": "", "Read": ""}]


def build_v25_postmortem_tail_rows(v25_root: Path = DEFAULT_V25_ROOT) -> list[dict[str, Any]]:
    path = v25_root / "v25_balanced_outcome_memory/memory_postmortem/extreme_negative_tail_tests.csv"
    frame = read_csv_if_exists(path)
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        difference = float(row.paired_date_difference_pct)
        ci_low = float(row.bootstrap_ci_low_pct)
        ci_high = float(row.bootstrap_ci_high_pct)
        supported = ci_high < 0.0
        rows.append(
            {
                "Side": row.side,
                "Field": str(row.memory_field).replace("hist_", "Hist").replace("_pct", ""),
                "Worst tail": f"bottom {int(row.bottom_tail_pct)}%",
                "Tail side P&L": fmt_pct(row.extreme_avg_side_pnl_pct, 2),
                "Rest side P&L": fmt_pct(row.rest_avg_side_pnl_pct, 2),
                "Paired-date difference": fmt_pct(difference, 2, signed=True),
                "95% interval": f"{fmt_pct(ci_low, 2)} to {fmt_pct(ci_high, 2)}",
                "Veto evidence": "Supported" if supported else "Not supported",
            }
        )
    return rows or [{"Side": "", "Field": "Not found", "Worst tail": "", "Tail side P&L": "", "Rest side P&L": "", "Paired-date difference": "", "95% interval": "", "Veto evidence": ""}]


def build_v25_postmortem_swap_rows(v25_root: Path = DEFAULT_V25_ROOT) -> list[dict[str, Any]]:
    path = v25_root / "v25_balanced_outcome_memory/memory_postmortem/selection_swap_bootstrap.csv"
    frame = read_csv_if_exists(path)
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        estimate = float(row.mean_added_minus_dropped_pct)
        rows.append(
            {
                "Side": row.side,
                "Dates": int(row.n_dates),
                "V2.5 added minus V2 dropped": fmt_pct(estimate, 2, signed=True),
                "95% interval": (
                    f"{fmt_pct(row.bootstrap_ci_low_pct, 2)} to "
                    f"{fmt_pct(row.bootstrap_ci_high_pct, 2)}"
                ),
                "Read": "Helped" if estimate > 0 else "Hurt",
            }
        )
    return rows or [{"Side": "", "Dates": "", "V2.5 added minus V2 dropped": "Not found", "95% interval": "", "Read": ""}]


def build_v24_v25_decision_rows(
    v24_root: Path = DEFAULT_V24_ROOT,
    v25_root: Path = DEFAULT_V25_ROOT,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v24 = _load_best_offset_row(v24_root, "LLM-Pool v2.4")
    v24_config = read_json_if_exists(v24_root / "run_config.json")
    v24_baseline = _load_best_offset_row(v24_root, "Baseline-Pool")
    v24_lift = (
        _metric(v24, "capital_cagr_pct", "cagr_pct")
        - _metric(v24_baseline, "capital_cagr_pct", "cagr_pct")
        if v24 is not None and v24_baseline is not None
        else float("nan")
    )
    rows.append(
        {
            "Run": "V2.4 raw SPY/VIX context",
            "Family": "V2.4 context ablation",
            "Pool/Book": "60/60 -> 30/30",
            "Schedule/Anchor": "offset 3",
            "Model": v24_config.get("model", "recorded model") if v24 is not None else "pending",
            "Schema": "frozen no-glossary prompt + raw SPY/VIX user context",
            "CAGR": "" if v24 is None else fmt_pct(_metric(v24, "capital_cagr_pct", "cagr_pct")),
            "Sharpe": "" if v24 is None else fmt_num(v24.get("capital_sharpe", v24.get("sharpe")), 2),
            "Max DD": "" if v24 is None else fmt_pct(_metric(v24, "capital_max_drawdown_pct", "max_drawdown_pct")),
            "IC": "" if v24 is None else fmt_num(v24.get("ic_mean"), 4),
            "Conclusion": (
                "Paid run pending"
                if v24 is None
                else f"Completed; {v24_lift:+.1f}pp CAGR lift versus matched Trail5"
            ),
        }
    )
    v25_path = v25_root / "v25_balanced_outcome_memory/coverage_adjusted_paired_scorecard.csv"
    v25_metrics = read_csv_if_exists(v25_path)
    v25_row = pd.DataFrame()
    if not v25_metrics.empty:
        v25_row = v25_metrics[
            v25_metrics.get("scope", pd.Series(dtype=str)).eq("memory_ready")
            & v25_metrics.get("run", pd.Series(dtype=str)).eq("V2.5 Balanced Memory")
        ]
    row = None if v25_row.empty else v25_row.iloc[0]
    rows.append(
        {
            "Run": "V2.5 balanced outcome memory",
            "Family": "V2.5 memory ablation",
            "Pool/Book": "60/60 -> 30/30",
            "Schedule/Anchor": "offset 3",
            "Model": (
                read_json_if_exists(v25_root / "run_config.json").get("model", "recorded model")
                if row is not None
                else "pending"
            ),
            "Schema": "V2 glossary prompt + HistN / HistWin / HistRet",
            "CAGR": "" if row is None else fmt_pct(row.get("cagr_pct")),
            "Sharpe": "" if row is None else fmt_num(row.get("sharpe"), 2),
            "Max DD": "" if row is None else fmt_pct(row.get("max_drawdown_pct")),
            "IC": "",
            "Conclusion": (
                "Memory audit complete; paid A/B pending"
                if row is None
                else (
                    f"Paid paired run complete; {float(row.get('cagr_lift_vs_v2_pp')):+.1f}pp vs fresh V2 and "
                    f"{float(row.get('cagr_lift_vs_trail5_pp')):+.1f}pp vs Trail5"
                )
            ),
        }
    )
    return rows


def build_v24_status_read(v24_root: Path = DEFAULT_V24_ROOT) -> str:
    llm = _load_best_offset_row(v24_root, "LLM-Pool v2.4")
    trail5 = _load_best_offset_row(v24_root, "Baseline-Pool")
    if llm is None or trail5 is None:
        return "The V2.4 paid score run is pending."
    llm_cagr = _metric(llm, "capital_cagr_pct", "cagr_pct")
    trail5_cagr = _metric(trail5, "capital_cagr_pct", "cagr_pct")
    llm_dd = _metric(llm, "capital_max_drawdown_pct", "max_drawdown_pct")
    trail5_dd = _metric(trail5, "capital_max_drawdown_pct", "max_drawdown_pct")
    return (
        f"Development period 2009-2014: V2.4 produced {fmt_pct(llm_cagr)} CAGR versus "
        f"{fmt_pct(trail5_cagr)} for matched Trail5 ({llm_cagr - trail5_cagr:+.1f}pp lift). "
        f"Sharpe was {fmt_num(llm.get('capital_sharpe', llm.get('sharpe')), 2)} versus "
        f"{fmt_num(trail5.get('capital_sharpe', trail5.get('sharpe')), 2)}; max drawdown improved from "
        f"{fmt_pct(trail5_dd)} to {fmt_pct(llm_dd)}, and IC improved from "
        f"{fmt_num(trail5.get('ic_mean'), 4)} to {fmt_num(llm.get('ic_mean'), 4)}. "
        "This removes the semantic regime label but still supplies raw SPY/VIX measurements; it is not a no-market-context run."
    )


def _temperature_label(config: dict[str, Any], fallback: str) -> str:
    value = config.get("temperature", None)
    if value is None:
        return fallback
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return str(value)
    return fmt_num(numeric, 1 if float(numeric) else 0)


def build_temperature_experiment_rows(
    temp0_root: Path = DEFAULT_V2_TEMP0_ROOT,
    temp06_root: Path = DEFAULT_V2_TEMP06_ROOT,
    temp1_root: Path = DEFAULT_V2_TEMP1_ROOT,
    temp15_root: Path = DEFAULT_V2_TEMP15_ROOT,
    temp2_root: Path = DEFAULT_V2_TEMP2_ROOT,
) -> list[dict[str, Any]]:
    specs = [
        ("Current V2 temp 0", temp0_root, "LLM-Pool v2", "current glossary-backed V2 prompt"),
        ("Current V2 temp 0 Trail5", temp0_root, "Baseline-Pool", "paired deterministic baseline from same artifact"),
        ("Current V2 temp 0.6", temp06_root, "LLM-Pool v2", "matching current-prompt temp 0.6 control"),
        ("Current V2 temp 1", temp1_root, "LLM-Pool v2", "current glossary-backed V2 prompt"),
        ("Current V2 temp 1.5", temp15_root, "LLM-Pool v2", "current glossary-backed V2 prompt"),
        ("Current V2 temp 2", temp2_root, "LLM-Pool v2", "current glossary-backed V2 prompt"),
    ]
    rows: list[dict[str, Any]] = []
    for label, root, run_contains, prompt_note in specs:
        row = _load_best_offset_row(root, run_contains)
        config = read_json_if_exists(root / "run_config.json")
        if row is None:
            rows.append(
                {
                    "Run": label,
                    "Temperature": "0.6" if "0.6" in label else "",
                    "Prompt": prompt_note,
                    "Raw CAGR": "",
                    "50/50 CAGR": "",
                    "Sharpe": "",
                    "Max DD": "",
                    "Lift vs Trail5": "",
                    "Read": "Not found yet",
                }
            )
            continue
        is_llm = "llm" in str(row.get("run", "")).lower()
        paired_baseline = _load_best_offset_row(root, "Baseline-Pool")
        baseline_cagr = _metric(paired_baseline, "capital_cagr_pct", "cagr_pct") if paired_baseline is not None else None
        row_cagr = _metric(row, "capital_cagr_pct", "cagr_pct")
        lift = float(pd.to_numeric(row_cagr, errors="coerce") - pd.to_numeric(baseline_cagr, errors="coerce")) if (
            is_llm and baseline_cagr is not None
        ) else None
        read = "Paired baseline"
        if is_llm:
            if lift is None:
                read = "Needs matching baseline"
            elif lift > 1.0:
                read = "Strongest positive lift so far" if float(pd.to_numeric(row_cagr, errors="coerce")) > 14.0 else "Positive lift"
            elif lift > 0:
                read = "Small positive lift"
            else:
                read = "Underperformed Trail5"
        rows.append(
            {
                "Run": label,
                "Temperature": _temperature_label(config, "n/a" if not is_llm else ""),
                "Prompt": prompt_note,
                "Raw CAGR": fmt_pct(row.get("cagr_pct")),
                "50/50 CAGR": fmt_pct(row_cagr),
                "Sharpe": fmt_num(row.get("capital_sharpe", row.get("sharpe")), 3),
                "Max DD": fmt_pct(_metric(row, "capital_max_drawdown_pct", "max_drawdown_pct")),
                "Lift vs Trail5": fmt_pct(lift, signed=True) if lift is not None else "paired control",
                "Read": read,
            }
        )
    historical = _load_best_offset_row(DEFAULT_V2_ANON_CONTROL_ROOT, "LLM-Pool v2")
    if historical is not None:
        rows.append(
            {
                "Run": "Historical anonymized V2",
                "Temperature": "not recorded",
                "Prompt": "older V1/V2 score prompt; included as context, not clean temp A/B",
                "Raw CAGR": fmt_pct(historical.get("cagr_pct")),
                "50/50 CAGR": fmt_pct(_metric(historical, "capital_cagr_pct", "cagr_pct")),
                "Sharpe": fmt_num(historical.get("capital_sharpe", historical.get("sharpe")), 3),
                "Max DD": fmt_pct(_metric(historical, "capital_max_drawdown_pct", "max_drawdown_pct")),
                "Lift vs Trail5": "",
                "Read": "Context only because prompt/temperature metadata differ",
            }
        )
    return rows


def build_temperature_control_read(
    temp0_root: Path = DEFAULT_V2_TEMP0_ROOT,
    temp06_root: Path = DEFAULT_V2_TEMP06_ROOT,
) -> str:
    """Summarize the clean temp-0 versus temp-0.6 control, including stability."""
    temp0 = _load_best_offset_row(temp0_root, "LLM-Pool v2")
    temp06 = _load_best_offset_row(temp06_root, "LLM-Pool v2")
    if temp0 is None or temp06 is None:
        return (
            "The clean temperature control is incomplete. Both current-prompt temp 0 and temp 0.6 "
            "artifacts are required."
        )
    cagr0 = float(pd.to_numeric(_metric(temp0, "capital_cagr_pct", "cagr_pct"), errors="coerce"))
    cagr06 = float(pd.to_numeric(_metric(temp06, "capital_cagr_pct", "cagr_pct"), errors="coerce"))
    sharpe0 = float(pd.to_numeric(temp0.get("capital_sharpe", temp0.get("sharpe")), errors="coerce"))
    sharpe06 = float(pd.to_numeric(temp06.get("capital_sharpe", temp06.get("sharpe")), errors="coerce"))

    selection_path = Path("pool60_pos30/anchor_3/llm_pool_v2/portfolio_selections.csv")
    selection0 = read_csv_if_exists(temp0_root / selection_path)
    selection06 = read_csv_if_exists(temp06_root / selection_path)
    stability = ""
    keys = ["rebalance_date", "position_type", "ticker"]
    if not selection0.empty and not selection06.empty and set(keys + ["score"]).issubset(selection0.columns) and set(keys + ["score"]).issubset(selection06.columns):
        common = selection0.merge(selection06, on=keys, suffixes=("_t0", "_t06"))
        overlap = len(common) / len(selection0)
        score_corr = common["score_t0"].corr(common["score_t06"], method="spearman")
        stability = f" Selected-book overlap was {overlap * 100:.1f}% and common-name score-rank correlation was {score_corr:.3f}."

    return (
        f"Control complete: temp 0.6 produced {fmt_pct(cagr06)} CAGR versus {fmt_pct(cagr0)} at temp 0 "
        f"({cagr06 - cagr0:+.2f}pp), with Sharpe {sharpe06:.3f} versus {sharpe0:.3f} "
        f"({sharpe06 - sharpe0:+.3f}).{stability} The economic result is effectively unchanged, "
        "so temperature 0 is not being credited with a lucky one-off advantage."
    )


def build_run_taxonomy_rows() -> list[dict[str, str]]:
    return [
        {
            "Run Name": "V2 ticker-visible",
            "Ticker Identity": "Real symbols visible",
            "Prompt": "Original V2 / V1 score prompt",
            "Output": "score only",
            "Purpose": "Historical best run; may include ticker-name prior.",
        },
        {
            "Run Name": "V2 anonymized",
            "Ticker Identity": "Masked as SEC_ IDs",
            "Prompt": "Original V2 / V1 score prompt",
            "Output": "score only",
            "Purpose": "Clean control for whether V2 still works without ticker names.",
        },
        {
            "Run Name": "V2 anonymized + labels",
            "Ticker Identity": "Masked as SEC_ IDs",
            "Prompt": "Original V2 prompt plus structured label contract",
            "Output": "score + setup_label + primary_features",
            "Purpose": "Diagnostic run for feature/label analysis; not the production winner.",
        },
        {
            "Run Name": "V2.1 anonymized",
            "Ticker Identity": "Masked as SEC_ IDs",
            "Prompt": "Discriminative prompt asking for incremental quality beyond Trail5/RSI5",
            "Output": "score only",
            "Purpose": "Prompt variant test; failed because it moved away from the core reversal signal.",
        },
        {
            "Run Name": "V2.2 ticker-visible",
            "Ticker Identity": "Real symbols visible",
            "Prompt": "V2 glossary prompt plus contextual-adjustment override",
            "Output": "context_adjustment -3..+3",
            "Purpose": "Test whether the LLM works better as a small contextual tilt on top of deterministic Trail5 rank.",
        },
        {
            "Run Name": "V2.3 anonymized rank-only",
            "Ticker Identity": "Masked as SEC_ IDs",
            "Prompt": "Original V2 feature framing, but asks for ordered lists instead of 0-100 scores",
            "Output": "complete long_ranking / short_ranking lists",
            "Purpose": "Test whether the LLM is better at ordinal selection than numeric score calibration.",
        },
        {
            "Run Name": "V2.4 anonymized raw context",
            "Ticker Identity": "Masked as SEC_ IDs",
            "Prompt": "Exact frozen no-glossary score prompt; raw SPY/VIX context replaces semantic regime",
            "Output": "score only",
            "Purpose": "Isolate whether measured market context repairs the no-glossary prompt without changing its core wording.",
        },
        {
            "Run Name": "V2.5 anonymized balanced memory",
            "Ticker Identity": "Masked as SEC_ IDs",
            "Prompt": "Current glossary V2 plus past-only balanced analog summaries",
            "Output": "score only; inputs add HistN / HistWin / HistRet",
            "Purpose": "Test whether empirical outcome memory adds value beyond a fresh same-run V2 control.",
        },
    ]


def _forensic_path(root: Path, filename: str) -> Path:
    return root / "cache_forensics" / filename


def build_cache_insight_cards(
    v21_root: Path = DEFAULT_V21_ROOT,
    phase2_scorecard_path: Path = DEFAULT_PHASE2_SCORECARD,
) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    bucket = read_csv_if_exists(_forensic_path(v21_root, "selection_bucket_outcomes.csv"))
    corr = read_csv_if_exists(_forensic_path(v21_root, "score_feature_correlations.csv"))
    labels = read_csv_if_exists(_forensic_path(v21_root, "structured_label_outcomes.csv"))
    weekly = read_csv_if_exists(_forensic_path(v21_root, "weekly_v2_vs_v21_deltas.csv"))

    ablation_rows = build_phase2_glossary_ablation_rows(phase2_scorecard_path)
    completed_ablation = [row for row in ablation_rows if row.get("Conclusion") != "Pending"]
    if completed_ablation:
        primary = next(
            (row for row in completed_ablation if row.get("Construction") == "Top 30 / 30 equal weight"),
            completed_ablation[0],
        )
        no_glossary_wins = sum(row.get("Conclusion") == "No glossary stronger" for row in completed_ablation)
        cards.append(
            {
                "Title": "Glossary Changes Where The Edge Lives",
                "Finding": (
                    f"On equal-weight 30/30, glossary CAGR was {primary.get('Glossary CAGR')} versus "
                    f"{primary.get('No-glossary CAGR')} without the glossary and {primary.get('Trail5 CAGR')} "
                    f"for Trail5. No glossary was stronger on {no_glossary_wins} of "
                    f"{len(completed_ablation)} concentrated constructions."
                ),
                "Implication": (
                    "Working hypothesis: the glossary helps the diversified primary book but may smooth or "
                    "overthink the highest-conviction ordering. In plain terms, glossary may help diversification "
                    "while no glossary may help concentration. Because every wrapper was examined on the same "
                    "2015-2019 period, treat this as a hypothesis to validate, not a settled conclusion."
                ),
                "Artifact": str(phase2_scorecard_path),
            }
        )

    if not bucket.empty:
        long_v2 = bucket[(bucket["pool_side"].eq("LONG")) & (bucket["selection_bucket"].eq("v2_only"))]
        long_v21 = bucket[(bucket["pool_side"].eq("LONG")) & (bucket["selection_bucket"].eq("v21_only"))]
        short_v2 = bucket[(bucket["pool_side"].eq("SHORT")) & (bucket["selection_bucket"].eq("v2_only"))]
        short_v21 = bucket[(bucket["pool_side"].eq("SHORT")) & (bucket["selection_bucket"].eq("v21_only"))]
        cards.append(
            {
                "Title": "Selection Replacement",
                "Finding": (
                    f"V2-only longs averaged {fmt_decimal_pct(long_v2.iloc[0].get('avg_trail_ret'))} Trail5; "
                    f"V2.1-only longs averaged {fmt_decimal_pct(long_v21.iloc[0].get('avg_trail_ret'))}."
                    if not long_v2.empty and not long_v21.empty
                    else "Selection bucket file found, but expected long rows were missing."
                ),
                "Implication": "V2.1 replaced uglier/extreme reversal names with cleaner, less extreme names.",
                "Artifact": str(_forensic_path(v21_root, "selection_bucket_outcomes.csv")),
            }
        )
        cards.append(
            {
                "Title": "Short Book Read",
                "Finding": (
                    f"V2-only shorts had avg fwd C2C {fmt_decimal_pct(short_v2.iloc[0].get('avg_fwd_c2c'))}; "
                    f"V2.1-only shorts had {fmt_decimal_pct(short_v21.iloc[0].get('avg_fwd_c2c'))}."
                    if not short_v2.empty and not short_v21.empty
                    else "Selection bucket file found, but expected short rows were missing."
                ),
                "Implication": "Short-side changes were less catastrophic than long-side changes, but did not create enough lift.",
                "Artifact": str(_forensic_path(v21_root, "selection_bucket_outcomes.csv")),
            }
        )

    if not corr.empty:
        def top_vars(side: str, score: str) -> str:
            sub = corr[(corr["pool_side"].eq(side)) & (corr["score_field"].eq(score))].copy()
            if sub.empty:
                return "n/a"
            sub = sub.sort_values("spearman", ascending=False).head(3)
            return ", ".join(f"{str(r.variable).replace('feature_', '')} ({float(r.spearman):.2f})" for r in sub.itertuples())

        cards.append(
            {
                "Title": "What V2 Scored",
                "Finding": f"Long top correlations: {top_vars('LONG', 'score_v2')}. Short top correlations: {top_vars('SHORT', 'score_v2')}.",
                "Implication": "V2 naturally kept a strong reversal-ranking shape while still using context.",
                "Artifact": str(_forensic_path(v21_root, "score_feature_correlations.csv")),
            }
        )
        cards.append(
            {
                "Title": "What V2.1 Scored",
                "Finding": f"Long top correlations: {top_vars('LONG', 'score_v21')}. Short top correlations: {top_vars('SHORT', 'score_v21')}.",
                "Implication": "V2.1 over-rotated into secondary cleanliness features and lost the core reversal exposure.",
                "Artifact": str(_forensic_path(v21_root, "score_feature_correlations.csv")),
            }
        )

    if not labels.empty:
        top = labels.sort_values("n", ascending=False).head(3)
        finding = "; ".join(
            f"{r.pool_side} {r.setup_label}: n={int(r.n)}, avg fwd {fmt_decimal_pct(r.avg_fwd_c2c)}"
            for r in top.itertuples()
        )
        cards.append(
            {
                "Title": "Structured Labels",
                "Finding": finding,
                "Implication": "Labels are useful for debugging feature emphasis, but not proof of hidden reasoning.",
                "Artifact": str(_forensic_path(v21_root, "structured_label_outcomes.csv")),
            }
        )

    if not weekly.empty:
        cards.append(
            {
                "Title": "Weekly Damage Pattern",
                "Finding": (
                    f"Mean V2.1 minus V2 weekly LS C2C delta: {fmt_pct(weekly['ls_c2c_delta_v21_minus_v2'].mean() * 100)}; "
                    f"long overlap {fmt_pct(weekly['long_selection_overlap'].mean() * 100)}, "
                    f"short overlap {fmt_pct(weekly['short_selection_overlap'].mean() * 100)}."
                ),
                "Implication": "The prompt changed the book materially, especially on the long side.",
                "Artifact": str(_forensic_path(v21_root, "weekly_v2_vs_v21_deltas.csv")),
            }
        )

    if not cards:
        return [
            {
                "Title": "Cache Forensics",
                "Finding": "Forensic CSVs not found.",
                "Implication": "Run the cache-forensics extraction after V2/V2.1 artifacts are available.",
                "Artifact": str(v21_root / "cache_forensics"),
            }
        ]
    return cards


def cache_insight_cards_html(cards: list[dict[str, str]]) -> str:
    parts = []
    for card in cards:
        parts.append(
            f"""
            <article class="insight-card">
              <h3>{_escape(card.get("Title", ""))}</h3>
              <p><strong>Finding:</strong> {_escape(card.get("Finding", ""))}</p>
              <p><strong>Implication:</strong> {_escape(card.get("Implication", ""))}</p>
              <p class="artifact-path">{_escape(card.get("Artifact", ""))}</p>
            </article>
            """
        )
    return "\n".join(parts)


def _limited_rows(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    return rows[:limit] if len(rows) > limit else rows


def build_structured_label_rows(v21_root: Path = DEFAULT_V21_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_forensic_path(v21_root, "structured_label_outcomes.csv"))
    if data.empty:
        return [{"Side": "", "Setup Label": "not found", "N": "", "Avg Score": "", "Raw Stock C2C": "", "Side P&L": "", "Side Hit Rate": ""}]
    data = data.sort_values(["pool_side", "n"], ascending=[True, False])
    rows = []
    for row in data.itertuples(index=False):
        side = str(row.pool_side).upper()
        raw_c2c = float(row.avg_fwd_c2c)
        raw_hit = float(row.hit_rate_c2c)
        side_pnl = -raw_c2c if side == "SHORT" else raw_c2c
        side_hit = 1.0 - raw_hit if side == "SHORT" else raw_hit
        rows.append(
            {
                "Side": side,
                "Setup Label": row.setup_label,
                "N": int(row.n),
                "Avg Score": fmt_num(row.avg_score, 1),
                "Raw Stock C2C": fmt_decimal_pct(raw_c2c),
                "Side P&L": fmt_decimal_pct(side_pnl),
                "Side Hit Rate": fmt_decimal_pct(side_hit),
            }
        )
    return rows


def build_attribution_outcome_rows(
    structured_root: Path = DEFAULT_V2_STRUCTURED_ROOT,
) -> list[dict[str, Any]]:
    path = structured_root / "pool60_pos30" / "anchor_3" / "llm_pool_v2" / "attribution_outcome_summary.csv"
    data = read_csv_if_exists(path)
    if data.empty:
        return [{"Side": "", "Feature": "not found", "N": "", "Raw Stock C2C": "", "Side P&L": "", "Side Hit Rate": "", "Read": ""}]
    if "avg_side_adjusted_c2c_pct" not in data.columns:
        side_mult = data["side"].astype(str).str.upper().map(lambda side: -1.0 if side == "SHORT" else 1.0)
        data["avg_side_adjusted_c2c_pct"] = pd.to_numeric(data["avg_c2c_return_pct"], errors="coerce") * side_mult
    if "side_hit_rate_c2c_pct" not in data.columns:
        raw_hit = pd.to_numeric(data["hit_rate_c2c_pct"], errors="coerce")
        data["side_hit_rate_c2c_pct"] = [
            100.0 - hit if str(side).upper() == "SHORT" and pd.notna(hit) else hit
            for side, hit in zip(data["side"], raw_hit, strict=False)
        ]
    data = data.sort_values(["side", "avg_side_adjusted_c2c_pct"], ascending=[True, False])
    rows = []
    for row in data.itertuples(index=False):
        side_pnl = float(row.avg_side_adjusted_c2c_pct)
        rows.append(
            {
                "Side": row.side,
                "Feature": row.feature,
                "N": int(row.n),
                "Raw Stock C2C": fmt_pct(row.avg_c2c_return_pct),
                "Side P&L": fmt_pct(side_pnl),
                "Side Hit Rate": fmt_pct(row.side_hit_rate_c2c_pct),
                "Read": "helped" if side_pnl > 0 else "hurt",
            }
        )
    return rows


def build_score_correlation_rows(v21_root: Path = DEFAULT_V21_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_forensic_path(v21_root, "score_feature_correlations.csv"))
    if data.empty:
        return [{"Side": "", "Score Field": "not found", "Top Positive": "", "Top Negative": ""}]
    labels = {
        "score_v2": "V2 score",
        "score_v21": "V2.1 score",
        "score_delta_v21_minus_v2": "V2.1 - V2 delta",
    }
    rows = []
    for side in ["LONG", "SHORT"]:
        for field in ["score_v2", "score_v21", "score_delta_v21_minus_v2"]:
            sub = data[(data["pool_side"].eq(side)) & (data["score_field"].eq(field))].copy()
            if sub.empty:
                continue
            positive = sub.sort_values("spearman", ascending=False).head(5)
            negative = sub.sort_values("spearman", ascending=True).head(5)
            rows.append(
                {
                    "Side": side,
                    "Score Field": labels.get(field, field),
                    "Top Positive": "; ".join(
                        f"{str(r.variable).replace('feature_', '')} {float(r.spearman):+.2f}" for r in positive.itertuples()
                    ),
                    "Top Negative": "; ".join(
                        f"{str(r.variable).replace('feature_', '')} {float(r.spearman):+.2f}" for r in negative.itertuples()
                    ),
                }
            )
    return rows


def build_selection_bucket_rows(v21_root: Path = DEFAULT_V21_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_forensic_path(v21_root, "selection_bucket_outcomes.csv"))
    if data.empty:
        return [{"Side": "", "Bucket": "not found", "N": "", "V2 Score": "", "V2.1 Score": "", "Avg Trail5": "", "Avg Fwd C2C": "", "Hit Rate": ""}]
    order = {"selected_both": 0, "v2_only": 1, "v21_only": 2, "selected_neither": 3}
    data = data.assign(_order=data["selection_bucket"].map(order).fillna(99)).sort_values(["pool_side", "_order"])
    rows = []
    for row in data.itertuples(index=False):
        rows.append(
            {
                "Side": row.pool_side,
                "Bucket": row.selection_bucket,
                "N": int(row.n),
                "V2 Score": fmt_num(row.avg_score_v2, 1),
                "V2.1 Score": fmt_num(row.avg_score_v21, 1),
                "Avg Trail5": fmt_decimal_pct(row.avg_trail_ret),
                "Avg Fwd C2C": fmt_decimal_pct(row.avg_fwd_c2c),
                "Hit Rate": fmt_decimal_pct(row.hit_rate_c2c),
            }
        )
    return rows


def build_largest_score_change_rows(v21_root: Path = DEFAULT_V21_ROOT, limit_per_side: int = 12) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_forensic_path(v21_root, "largest_score_changes.csv"))
    if data.empty:
        return [{"Direction": "not found", "Date": "", "Ticker": "", "Side": "", "V2": "", "V2.1": "", "Delta": "", "Bucket": "", "Trail5": "", "Fwd C2C": "", "Notable Features": ""}]
    boosted = data.sort_values("score_delta_v21_minus_v2", ascending=False).head(limit_per_side).copy()
    dropped = data.sort_values("score_delta_v21_minus_v2", ascending=True).head(limit_per_side).copy()
    rows = []
    for direction, frame in [("V2.1 boosted", boosted), ("V2.1 dropped", dropped)]:
        for _, row in frame.iterrows():
            notable = (
                f"Rng55 {float(row.get('feature_Rng55')):.2f}, "
                f"%MA100 {float(row.get('feature_%MA100')):.1f}, "
                f"Mom3m {float(row.get('feature_Mom3m')):.1f}, "
                f"MRSig {float(row.get('feature_MRSig')):.1f}"
            )
            rows.append(
                {
                    "Direction": direction,
                    "Date": row.get("date"),
                    "Ticker": row.get("ticker"),
                    "Side": row.get("pool_side"),
                    "V2": fmt_num(row.get("score_v2"), 0),
                    "V2.1": fmt_num(row.get("score_v21"), 0),
                    "Delta": fmt_num(row.get("score_delta_v21_minus_v2"), 0),
                    "Bucket": row.get("selection_bucket"),
                    "Trail5": fmt_decimal_pct(row.get("trail_ret")),
                    "Fwd C2C": fmt_decimal_pct(row.get("fwd_c2c")),
                    "Notable Features": notable,
                }
            )
    return rows


def build_weekly_delta_rows(v21_root: Path = DEFAULT_V21_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_forensic_path(v21_root, "weekly_v2_vs_v21_deltas.csv"))
    if data.empty:
        return [{"View": "not found", "LS Delta": "", "Long Delta": "", "Short Delta": "", "Long Overlap": "", "Short Overlap": ""}]
    rows = []
    summary = data[
        [
            "ls_c2c_delta_v21_minus_v2",
            "long_ret_delta_v21_minus_v2",
            "short_ret_delta_v21_minus_v2",
            "long_selection_overlap",
            "short_selection_overlap",
        ]
    ].describe(percentiles=[0.25, 0.5, 0.75])
    for stat in ["mean", "25%", "50%", "75%", "min", "max"]:
        if stat in summary.index:
            rows.append(
                {
                    "View": stat,
                    "LS Delta": fmt_decimal_pct(summary.loc[stat, "ls_c2c_delta_v21_minus_v2"], signed=True),
                    "Long Delta": fmt_decimal_pct(summary.loc[stat, "long_ret_delta_v21_minus_v2"], signed=True),
                    "Short Delta": fmt_decimal_pct(summary.loc[stat, "short_ret_delta_v21_minus_v2"], signed=True),
                    "Long Overlap": fmt_decimal_pct(summary.loc[stat, "long_selection_overlap"]),
                    "Short Overlap": fmt_decimal_pct(summary.loc[stat, "short_selection_overlap"]),
                }
            )
    worst = data.sort_values("ls_c2c_delta_v21_minus_v2").head(5)
    for row in worst.itertuples(index=False):
        rows.append(
            {
                "View": f"worst {row.date}",
                "LS Delta": fmt_decimal_pct(row.ls_c2c_delta_v21_minus_v2, signed=True),
                "Long Delta": fmt_decimal_pct(row.long_ret_delta_v21_minus_v2, signed=True),
                "Short Delta": fmt_decimal_pct(row.short_ret_delta_v21_minus_v2, signed=True),
                "Long Overlap": fmt_decimal_pct(row.long_selection_overlap),
                "Short Overlap": fmt_decimal_pct(row.short_selection_overlap),
            }
        )
    return rows


def _long_breadth_path(root: Path, filename: str) -> Path:
    return root / "long_breadth_experiments_pool60_pos30_anchor3" / filename


def build_long_breadth_variant_rows(root: Path = DEFAULT_V2_ANON_CONTROL_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_long_breadth_path(root, "comparison_scorecard.csv"))
    if data.empty:
        return [
            {
                "Variant": "not found",
                "Top30 Wt": "",
                "Ranks31-60 Wt": "",
                "Sharpe": "",
                "50/50 CAGR": "",
                "50/50 Total": "",
                "50/50 Max DD": "",
                "CAGR Lift": "",
                "Read": "",
            }
        ]
    data = data[data.get("execution_model", pd.Series(dtype=str)).astype(str).str.upper().eq("C2C")].copy()
    data = data.sort_values("capital_cagr_pct", ascending=False)
    best_run = str(data.iloc[0].get("run", "")) if not data.empty else ""
    rows: list[dict[str, Any]] = []
    for row in data.itertuples(index=False):
        lift = getattr(row, "capital_cagr_lift_vs_current_pct", None)
        run = str(getattr(row, "run"))
        if run == best_run and run == "current_top30":
            read = "Best CAGR / total return"
        elif run == best_run:
            read = "Best CAGR here; promising wrapper"
        elif run == "current_top30":
            read = "Original masked V2 control"
        elif pd.to_numeric(lift, errors="coerce") < 0:
            read = "Smoother, lower CAGR"
        else:
            read = "Potential improvement"
        rows.append(
            {
                "Variant": run,
                "Top30 Wt": fmt_pct(float(getattr(row, "long_top30_weight")) * 100.0),
                "Ranks31-60 Wt": fmt_pct(float(getattr(row, "long_ranks31_60_weight")) * 100.0),
                "Sharpe": fmt_num(getattr(row, "capital_sharpe", getattr(row, "sharpe", "")), 3),
                "50/50 CAGR": fmt_pct(getattr(row, "capital_cagr_pct")),
                "50/50 Total": fmt_pct(getattr(row, "capital_total_return_pct")),
                "50/50 Max DD": fmt_pct(getattr(row, "capital_max_drawdown_pct")),
                "CAGR Lift": fmt_pct(lift, signed=True),
                "Read": read,
            }
        )
    return rows


def build_long_rank_curve_rows(root: Path = DEFAULT_V2_ANON_CONTROL_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_long_breadth_path(root, "long_rank_curve_summary.csv"))
    if data.empty:
        return [
            {
                "Rank Bucket": "not found",
                "Avg Ret": "",
                "Median Ret": "",
                "Hit Rate": "",
                "Ann Vol": "",
                "Max DD": "",
                "Avg Trail5": "",
                "Diff vs Top30": "",
            }
        ]
    rows: list[dict[str, Any]] = []
    data = data.sort_values("bucket_start")
    for row in data.itertuples(index=False):
        rows.append(
            {
                "Rank Bucket": getattr(row, "rank_bucket"),
                "Avg Ret": fmt_pct(getattr(row, "avg_return_pct")),
                "Median Ret": fmt_pct(getattr(row, "median_return_pct")),
                "Hit Rate": fmt_pct(getattr(row, "hit_rate_pct")),
                "Ann Vol": fmt_pct(getattr(row, "annual_vol_pct")),
                "Max DD": fmt_pct(getattr(row, "max_drawdown_pct")),
                "Avg Trail5": fmt_pct(getattr(row, "avg_trail5_pct")),
                "Diff vs Top30": fmt_pct(getattr(row, "avg_diff_vs_top30_pct"), signed=True),
            }
        )
    return rows


def build_long_rank_bootstrap_rows(root: Path = DEFAULT_V2_ANON_CONTROL_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_long_breadth_path(root, "long_rank_curve_block_bootstrap.csv"))
    if data.empty:
        return [{"Rank Bucket": "not found", "Mean Diff": "", "95% CI": "", "p two-sided": ""}]
    rows: list[dict[str, Any]] = []
    for row in data.sort_values("bucket_start").itertuples(index=False):
        rows.append(
            {
                "Rank Bucket": getattr(row, "rank_bucket"),
                "Mean Diff": fmt_pct(getattr(row, "mean_pct"), signed=True),
                "95% CI": f"{fmt_pct(getattr(row, 'ci_low_pct'))} to {fmt_pct(getattr(row, 'ci_high_pct'))}",
                "p two-sided": fmt_num(getattr(row, "p_two_sided"), 3),
            }
        )
    return rows


def build_long_variant_bootstrap_rows(root: Path = DEFAULT_V2_ANON_CONTROL_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_long_breadth_path(root, "variant_block_bootstrap_vs_current.csv"))
    if data.empty:
        return [{"Variant": "not found", "Mean LS Delta": "", "95% CI": "", "p two-sided": ""}]
    order = {
        "current_top30": 0,
        "light_reserve_85_15": 1,
        "core_reserve_75_25": 2,
        "broad_core_67_33": 3,
        "equal_all60_50_50": 4,
        "linear_rank_decay_60_to_1": 5,
    }
    data = data.assign(_order=data["run"].map(order).fillna(99)).sort_values("_order")
    rows: list[dict[str, Any]] = []
    for row in data.itertuples(index=False):
        rows.append(
            {
                "Variant": getattr(row, "run"),
                "Mean LS Delta": fmt_pct(getattr(row, "mean_pct"), signed=True),
                "95% CI": f"{fmt_pct(getattr(row, 'ci_low_pct'))} to {fmt_pct(getattr(row, 'ci_high_pct'))}",
                "p two-sided": fmt_num(getattr(row, "p_two_sided"), 3),
            }
        )
    return rows


def build_long_rank_year_rows(root: Path = DEFAULT_V2_ANON_CONTROL_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_long_breadth_path(root, "long_rank_curve_by_year.csv"))
    if data.empty:
        return [{"Year": "not found", "Rank Bucket": "", "Avg Ret": "", "Hit Rate": "", "Diff vs Top30": ""}]
    rows: list[dict[str, Any]] = []
    data = data.sort_values(["year", "bucket_start"])
    for row in data.itertuples(index=False):
        rows.append(
            {
                "Year": int(getattr(row, "year")),
                "Rank Bucket": getattr(row, "rank_bucket"),
                "Avg Ret": fmt_pct(getattr(row, "avg_return_pct")),
                "Hit Rate": fmt_pct(getattr(row, "hit_rate_pct")),
                "Diff vs Top30": fmt_pct(getattr(row, "avg_diff_vs_top30_pct"), signed=True),
            }
        )
    return rows


def build_long_breadth_html(root: Path = DEFAULT_V2_ANON_CONTROL_ROOT) -> str:
    sections = [
        (
            "Long Allocation Variants",
            "Shorts stay fixed as V2 top-30. Only long capital is split between ranks 1-30 and 31-60.",
            html_table(build_long_breadth_variant_rows(root), "long-breadth-table"),
        ),
        (
            "True V2 Long Rank Curve",
            "Forward 5-day C2C behavior by LLM long rank bucket. Diff vs Top30 is measured by signal date.",
            html_table(build_long_rank_curve_rows(root), "long-rank-table"),
        ),
        (
            "Block Bootstrap: Rank Buckets vs Top30",
            "Date-block bootstrap, so same-date stocks are not treated as independent evidence.",
            html_table(build_long_rank_bootstrap_rows(root), "long-bootstrap-table"),
        ),
        (
            "Block Bootstrap: Variant LS Delta vs Current",
            "Full long-short portfolio delta versus current V2 top-30 longs plus top-30 shorts.",
            html_table(build_long_variant_bootstrap_rows(root), "long-bootstrap-table"),
        ),
        (
            "Rank Curve By Year",
            "Year-level check for whether the rank-curve read is stable or period-specific.",
            html_table(build_long_rank_year_rows(root), "long-year-table"),
        ),
    ]
    parts = []
    for title, description, table in sections:
        parts.append(
            f"""
            <details class="drilldown-block" open>
              <summary>{_escape(title)}</summary>
              <p>{_escape(description)}</p>
              <div class="table-wrap">{table}</div>
            </details>
            """
        )
    return "\n".join(parts)


def build_linear_decay_visibility_rows() -> list[dict[str, Any]]:
    specs = [
        ("ticker-visible V2", DEFAULT_RESULTS_ROOT, "historical ticker-visible cache"),
        ("ticker-masked V2", DEFAULT_V2_ANON_CONTROL_ROOT, "anonymized control cache"),
    ]
    rows: list[dict[str, Any]] = []
    for label, root, source in specs:
        data = read_csv_if_exists(_long_breadth_path(root, "comparison_scorecard.csv"))
        if data.empty:
            rows.append(
                {
                    "Cache": label,
                    "Source": source,
                    "Linear CAGR": "not found",
                    "Sharpe": "",
                    "Max DD": "",
                    "Lift vs Top30": "",
                    "Read": "",
                }
            )
            continue
        row = data[
            data.get("execution_model", pd.Series(dtype=str)).astype(str).str.upper().eq("C2C")
            & data.get("run", pd.Series(dtype=str)).astype(str).eq("linear_rank_decay_60_to_1")
        ]
        if row.empty:
            continue
        r = row.iloc[0]
        lift = pd.to_numeric(r.get("capital_cagr_lift_vs_current_pct"), errors="coerce")
        rows.append(
            {
                "Cache": label,
                "Source": source,
                "Linear CAGR": fmt_pct(r.get("capital_cagr_pct")),
                "Sharpe": fmt_num(r.get("capital_sharpe"), 3),
                "Max DD": fmt_pct(r.get("capital_max_drawdown_pct")),
                "Lift vs Top30": fmt_pct(lift, signed=True),
                "Read": "helps masked V2" if pd.notna(lift) and lift > 0 else "hurts ticker-visible V2",
            }
        )
    return rows


def _side_decay_path(root: Path, filename: str) -> Path:
    return root / "side_decay_experiments_pool60_pos30_anchor3" / filename


def build_side_decay_variant_rows(root: Path = DEFAULT_V2_ANON_CONTROL_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_side_decay_path(root, "comparison_scorecard.csv"))
    if data.empty:
        return [{"Variant": "not found", "Longs": "", "Shorts": "", "Sharpe": "", "50/50 CAGR": "", "50/50 Max DD": "", "Lift": "", "Read": ""}]
    data = data[data.get("execution_model", pd.Series(dtype=str)).astype(str).str.upper().eq("C2C")].copy()
    data = data.sort_values("capital_cagr_pct", ascending=False)
    rows: list[dict[str, Any]] = []
    for row in data.itertuples(index=False):
        run = str(row.run)
        if run == "long_decay_short_top30":
            read = "Best CAGR: broaden longs only"
        elif run == "long_top30_short_decay":
            read = "Short broadening costs CAGR, helps DD"
        elif run == "both_decay":
            read = "Drawdown improves, CAGR flat/down"
        else:
            read = "Masked V2 control"
        rows.append(
            {
                "Variant": run,
                "Longs": "linear decay" if row.long_ranks31_60_weight > 0 else "top 30",
                "Shorts": "linear decay" if row.short_ranks31_60_weight > 0 else "top 30",
                "Sharpe": fmt_num(row.capital_sharpe, 3),
                "50/50 CAGR": fmt_pct(row.capital_cagr_pct),
                "50/50 Max DD": fmt_pct(row.capital_max_drawdown_pct),
                "Lift": fmt_pct(getattr(row, "capital_cagr_lift_vs_control_pct"), signed=True),
                "Read": read,
            }
        )
    return rows


def build_short_rank_curve_rows(root: Path = DEFAULT_V2_ANON_CONTROL_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_side_decay_path(root, "short_rank_curve_summary.csv"))
    if data.empty:
        return [{"Rank Bucket": "not found", "Avg Short P&L": "", "Hit Rate": "", "Max DD": "", "Avg Trail5": "", "Diff vs Top30": ""}]
    rows: list[dict[str, Any]] = []
    for row in data.sort_values("bucket_start").itertuples(index=False):
        rows.append(
            {
                "Rank Bucket": row.rank_bucket,
                "Avg Short P&L": fmt_pct(row.avg_short_pnl_pct),
                "Median Short P&L": fmt_pct(row.median_short_pnl_pct),
                "Hit Rate": fmt_pct(row.short_hit_rate_pct),
                "Max DD": fmt_pct(row.max_drawdown_pct),
                "Avg Trail5": fmt_pct(row.avg_trail5_pct),
                "Diff vs Top30": fmt_pct(row.avg_diff_vs_top30_short_pnl_pct, signed=True),
            }
        )
    return rows


def build_short_rank_bootstrap_rows(root: Path = DEFAULT_V2_ANON_CONTROL_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_side_decay_path(root, "short_rank_curve_block_bootstrap.csv"))
    if data.empty:
        return [{"Rank Bucket": "not found", "Mean Diff": "", "95% CI": "", "p two-sided": ""}]
    rows: list[dict[str, Any]] = []
    for row in data.sort_values("bucket_start").itertuples(index=False):
        rows.append(
            {
                "Rank Bucket": row.rank_bucket,
                "Mean Diff": fmt_pct(row.mean_pct, signed=True),
                "95% CI": f"{fmt_pct(row.ci_low_pct)} to {fmt_pct(row.ci_high_pct)}",
                "p two-sided": fmt_num(row.p_two_sided, 3),
            }
        )
    return rows


def build_side_decay_bootstrap_rows(root: Path = DEFAULT_V2_ANON_CONTROL_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_side_decay_path(root, "variant_block_bootstrap_vs_control.csv"))
    if data.empty:
        return [{"Variant": "not found", "Mean LS Delta": "", "95% CI": "", "p two-sided": ""}]
    order = {"control_top30_top30": 0, "long_decay_short_top30": 1, "long_top30_short_decay": 2, "both_decay": 3}
    data = data.assign(_order=data["run"].map(order).fillna(99)).sort_values("_order")
    rows: list[dict[str, Any]] = []
    for row in data.itertuples(index=False):
        rows.append(
            {
                "Variant": row.run,
                "Mean LS Delta": fmt_pct(row.mean_pct, signed=True),
                "95% CI": f"{fmt_pct(row.ci_low_pct)} to {fmt_pct(row.ci_high_pct)}",
                "p two-sided": fmt_num(row.p_two_sided, 3),
            }
        )
    return rows


def build_side_decay_html(root: Path = DEFAULT_V2_ANON_CONTROL_ROOT) -> str:
    sections = [
        (
            "Ticker-Visible vs Masked Linear Decay",
            "Same linear-decay formula applied to the historical ticker-visible cache and the anonymized control cache.",
            html_table(build_linear_decay_visibility_rows(), "side-decay-table"),
        ),
        (
            "Long / Short Decay Variants",
            "Control, long decay only, short decay only, and both-side decay on ticker-masked V2.",
            html_table(build_side_decay_variant_rows(root), "side-decay-table"),
        ),
        (
            "Short Rank Curve, Measured As Short P&L",
            "Positive means the short bucket made money. This checks whether lower-ranked shorts should receive capital.",
            html_table(build_short_rank_curve_rows(root), "short-rank-table"),
        ),
        (
            "Block Bootstrap: Short Buckets vs Top30",
            "Date-block bootstrap on short P&L differences versus the top-30 short sleeve.",
            html_table(build_short_rank_bootstrap_rows(root), "short-rank-table"),
        ),
        (
            "Block Bootstrap: Side Decay Variants vs Control",
            "Date-block bootstrap on full long-short return delta versus top30/top30 control.",
            html_table(build_side_decay_bootstrap_rows(root), "side-decay-table"),
        ),
    ]
    parts: list[str] = []
    for title, description, table in sections:
        parts.append(
            f"""
            <details class="drilldown-block" open>
              <summary>{_escape(title)}</summary>
              <p>{_escape(description)}</p>
              <div class="table-wrap">{table}</div>
            </details>
            """
        )
    return "\n".join(parts)


def _side_process_hybrid_path(root: Path, filename: str) -> Path:
    return root / "side_process_hybrid_experiments_pool60_pos30_anchor3" / filename


def build_side_process_focus_rows(
    root: Path = DEFAULT_V22_VISIBLE_ROOT,
    filename: str = "comparison_scorecard_expanded.csv",
    limit: int = 12,
) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_side_process_hybrid_path(root, filename))
    if data.empty:
        return [{"Long Process": "not found", "Short Process": "", "Sharpe": "", "50/50 CAGR": "", "Max DD": "", "Lift vs V2": "", "Long Avg": "", "Short Side P&L": ""}]
    data = data.sort_values("cagr_pct", ascending=False).head(limit)
    rows: list[dict[str, Any]] = []
    for row in data.itertuples(index=False):
        rows.append(
            {
                "Long Process": row.long_process,
                "Short Process": row.short_process,
                "Sharpe": fmt_num(row.sharpe, 3),
                "50/50 CAGR": fmt_pct(row.cagr_pct),
                "Max DD": fmt_pct(row.max_drawdown_pct),
                "Lift vs V2": fmt_pct(row.cagr_lift_vs_v2_pct, signed=True),
                "Long Avg": fmt_pct(row.avg_long_raw_c2c_pct),
                "Short Side P&L": fmt_pct(row.avg_short_side_pnl_pct),
            }
        )
    return rows


def build_side_process_summary_rows(root: Path = DEFAULT_V22_VISIBLE_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_side_process_hybrid_path(root, "side_process_summary_expanded.csv"))
    if data.empty:
        return [{"Side": "not found", "Process": "", "Avg Side P&L": "", "Hit Rate": "", "Side Vol": ""}]
    data = data.sort_values(["side", "avg_side_pnl_c2c_pct"], ascending=[True, False])
    rows: list[dict[str, Any]] = []
    for row in data.itertuples(index=False):
        rows.append(
            {
                "Side": row.side,
                "Process": row.process,
                "Family": row.family,
                "Avg Side P&L": fmt_pct(row.avg_side_pnl_c2c_pct),
                "Hit Rate": fmt_pct(row.side_hit_rate_pct),
                "Side Vol": fmt_pct(row.side_vol_pct),
            }
        )
    return rows


def build_side_process_hybrid_html(root: Path = DEFAULT_V22_VISIBLE_ROOT) -> str:
    sections = [
        (
            "Best Side-Process Hybrids",
            "All-vs-all cross of long-process choice and short-process choice. Uses ticker-visible V2 and V2.2 caches only; no LLM calls.",
            html_table(build_side_process_focus_rows(root, "comparison_scorecard_expanded.csv", limit=12), "side-process-table"),
        ),
        (
            "Long Process Test, Holding V2 Shorts Fixed",
            "This asks whether any V2.2-style long modifier beats the original V2 long ranking when the short book is unchanged.",
            html_table(build_side_process_focus_rows(root, "long_process_test_with_v2_shorts.csv", limit=14), "side-process-table"),
        ),
        (
            "Short Process Test, Holding V2 Longs Fixed",
            "This asks whether changing only the short process improves the original V2 book.",
            html_table(build_side_process_focus_rows(root, "short_process_test_with_v2_longs.csv", limit=14), "side-process-table"),
        ),
        (
            "Side-Only Process Quality",
            "Side P&L flips shorts, so positive means the process made money for that side.",
            html_table(build_side_process_summary_rows(root), "side-process-table"),
        ),
    ]
    parts: list[str] = []
    for title, description, table in sections:
        parts.append(
            f"""
            <details class="drilldown-block" open>
              <summary>{_escape(title)}</summary>
              <p>{_escape(description)}</p>
              <div class="table-wrap">{table}</div>
            </details>
            """
        )
    return "\n".join(parts)


def _v22_analysis_path(root: Path, filename: str) -> Path:
    return root / "v22_adjustment_analysis" / filename


def build_v22_adjustment_distribution_rows(root: Path = DEFAULT_V22_VISIBLE_ROOT) -> list[dict[str, Any]]:
    data = read_csv_if_exists(_v22_analysis_path(root, "adjustment_distribution.csv"))
    if data.empty:
        return [{"Side": "", "Adjustment": "not found", "N": "", "Share": ""}]
    data = data.sort_values(["pool_side", "context_adjustment"])
    rows = []
    for row in data.itertuples(index=False):
        rows.append(
            {
                "Side": row.pool_side,
                "Adjustment": int(row.context_adjustment),
                "N": int(row.n),
                "Share": fmt_pct(row.pct_side),
            }
        )
    return rows


def build_v22_adjustment_outcome_rows(
    root: Path = DEFAULT_V22_VISIBLE_ROOT,
    selected_only: bool = False,
) -> list[dict[str, Any]]:
    filename = "adjustment_bucket_outcomes_selected.csv" if selected_only else "adjustment_bucket_outcomes_all_candidates.csv"
    data = read_csv_if_exists(_v22_analysis_path(root, filename))
    if data.empty:
        return [{"Side": "", "Adjustment": "not found", "N": "", "Base": "", "Final": "", "Raw Stock C2C": "", "Side P&L": "", "Side Hit": ""}]
    side_col = "position_type" if selected_only else "pool_side"
    data = data.sort_values([side_col, "context_adjustment"])
    rows = []
    for row in data.itertuples(index=False):
        side = getattr(row, side_col)
        raw = getattr(row, "avg_raw_stock_c2c_pct", None)
        if raw is None:
            raw = getattr(row, "avg_raw_stock_c2c") * 100.0
        side_pnl = getattr(row, "avg_side_pnl_c2c") * 100.0
        rows.append(
            {
                "Side": side,
                "Adjustment": int(row.context_adjustment),
                "N": int(row.n),
                "Base": fmt_num(row.avg_base_score, 1),
                "Final": fmt_num(row.avg_final_score, 1),
                "Raw Stock C2C": fmt_pct(raw),
                "Side P&L": fmt_pct(side_pnl),
                "Side Hit": fmt_pct(row.side_hit_rate * 100.0),
            }
        )
    return rows


def build_v22_adjustment_html(root: Path = DEFAULT_V22_VISIBLE_ROOT) -> str:
    sections = [
        (
            "Adjustment Distribution",
            "How often DeepSeek used each -3..+3 adjustment value. The model mostly stayed near 0/+1 and rarely used extremes.",
            html_table(build_v22_adjustment_distribution_rows(root), "v22-adjustment-table"),
        ),
        (
            "Adjustment Bucket Outcomes: All Candidates",
            "Side P&L flips shorts, so positive is good for the intended side. This shows whether higher adjustments actually mapped to better outcomes.",
            html_table(build_v22_adjustment_outcome_rows(root, selected_only=False), "v22-adjustment-table"),
        ),
        (
            "Adjustment Bucket Outcomes: Selected Book",
            "Same read, but only for names that entered the 30/30 selected book at lambda 8.",
            html_table(build_v22_adjustment_outcome_rows(root, selected_only=True), "v22-adjustment-table"),
        ),
    ]
    parts: list[str] = []
    for title, description, table in sections:
        parts.append(
            f"""
            <details class="drilldown-block" open>
              <summary>{_escape(title)}</summary>
              <p>{_escape(description)}</p>
              <div class="table-wrap">{table}</div>
            </details>
            """
        )
    return "\n".join(parts)


def build_cache_drilldown_html() -> str:
    sections = [
        (
            "Setup Labels From Structured Run",
            "How each model-produced setup label performed out of sample. Raw Stock C2C is underlying return; Side P&L flips shorts so positive is good for the portfolio.",
            html_table(build_structured_label_rows(), "drill-table"),
        ),
        (
            "Cited Feature Outcome Analysis",
            "When a feature was cited in structured reasoning, how did those candidates do next period? Use Side P&L / Side Hit Rate for helped-versus-hurt reads.",
            html_table(build_attribution_outcome_rows(), "drill-table"),
        ),
        (
            "Score vs Feature Correlations",
            "What V2, V2.1, and the V2.1-minus-V2 score delta actually loaded on.",
            html_table(build_score_correlation_rows(), "drill-table wide-table"),
        ),
        (
            "Selection Bucket Outcomes",
            "Candidates selected by both prompts, V2 only, V2.1 only, or neither.",
            html_table(build_selection_bucket_rows(), "drill-table"),
        ),
        (
            "Largest Score Changes",
            "Specific examples of names V2.1 heavily boosted or dropped versus V2.",
            html_table(build_largest_score_change_rows(), "drill-table wide-table"),
        ),
        (
            "Weekly V2.1 vs V2 Damage",
            "Distribution of week-by-week return deltas and the worst weeks.",
            html_table(build_weekly_delta_rows(), "drill-table"),
        ),
    ]
    parts = []
    for title, description, table in sections:
        parts.append(
            f"""
            <details class="drilldown-block" open>
              <summary>{_escape(title)}</summary>
              <p>{_escape(description)}</p>
              <div class="table-wrap">{table}</div>
            </details>
            """
        )
    return "\n".join(parts)


def reasoning_panel_html(title: str, path: Path, note: str) -> str:
    text = read_text_if_exists(path)
    if text:
        line_count = text.count("\n") + 1
        size_kb = path.stat().st_size / 1024.0
        meta = f"{line_count:,} lines, {size_kb:.1f} KB"
        body = f'<pre class="reasoning-pre">{_escape(text)}</pre>'
    else:
        meta = "not found"
        body = '<p class="muted">Reasoning text file not found. Generate it from the debug cache first.</p>'
    return f"""
      <article class="reasoning-panel">
        <div class="reasoning-panel-head">
          <div>
            <h3>{_escape(title)}</h3>
            <p>{_escape(note)}</p>
          </div>
          <span class="pill">{_escape(meta)}</span>
        </div>
        <p class="artifact-path">{_escape(str(path))}</p>
        {body}
      </article>
    """


def build_reasoning_peek_html() -> str:
    return "\n".join(
        [
            reasoning_panel_html(
                "Before Definitions: Ticker-Visible",
                DEFAULT_REASONING_VISIBLE_PATH,
                "One-date V2 score call with real ticker symbols visible, before the feature glossary was added.",
            ),
            reasoning_panel_html(
                "Before Definitions: Ticker-Masked",
                DEFAULT_REASONING_ANON_PATH,
                "One-date V2 score call with SEC_ identifiers, before the feature glossary was added.",
            ),
            reasoning_panel_html(
                "After Definitions: Ticker-Visible",
                DEFAULT_REASONING_GLOSSARY_VISIBLE_PATH,
                "Same one-date V2-style reasoning peek after adding concise definitions for the input features.",
            ),
            reasoning_panel_html(
                "After Definitions: Ticker-Masked",
                DEFAULT_REASONING_GLOSSARY_ANON_PATH,
                "Same one-date V2-style reasoning peek with SEC_ identifiers after adding concise feature definitions.",
            ),
            reasoning_panel_html(
                "V2.2 Adjustment: Ticker-Visible",
                DEFAULT_REASONING_V22_VISIBLE_PATH,
                "One-date V2.2 contextual-adjustment run: Trail5-rank base score is fixed, DeepSeek returns only -3..+3 adjustment.",
            ),
        ]
    )


def specialized_view_rows() -> list[dict[str, str]]:
    return [
        {
            "View": "Anonymous technical",
            "Responsibility": "Rank stocks",
            "What It Should Add": "Clean technical ordering without ticker identity priors.",
            "Failure Rule": "Delete or demote if it cannot beat Trail5/random-within-pool after costs and block tests.",
        },
        {
            "View": "Ticker-visible",
            "Responsibility": "Identify identity-based exceptions",
            "What It Should Add": "Flag cases where company identity creates genuine exception risk, not broad score averaging.",
            "Failure Rule": "Delete if visible-only conviction keeps diluting anonymized long ranks.",
        },
        {
            "View": "Market-context",
            "Responsibility": "Choose concentration level",
            "What It Should Add": "Select how aggressively to use rank concentration based on market state.",
            "Failure Rule": "Delete if it cannot beat a simple deterministic regime rule.",
        },
        {
            "View": "Adversarial critic",
            "Responsibility": "Flag likely ranking failures",
            "What It Should Add": "Veto or haircut names with continuation, squeeze, or data-quality failure patterns.",
            "Failure Rule": "Delete if vetoes reduce returns or mostly remove winners.",
        },
    ]


def sequential_layer_rows() -> list[dict[str, str]]:
    return [
        {
            "Step": "1",
            "System": "Anonymous ranker alone",
            "Question": "How much value comes from pure technical ordinal ranking?",
            "Decision Metric": "CAGR/Sharpe lift vs Trail5 and random-within-pool; long/short attribution.",
        },
        {
            "Step": "2",
            "System": "Add visible critic",
            "Question": "Does ticker identity improve exception handling without diluting anonymized conviction?",
            "Decision Metric": "Incremental out-of-sample lift after anonymous ranker; directional disagreement buckets.",
        },
        {
            "Step": "3",
            "System": "Add regime allocator",
            "Question": "Does market context improve concentration timing?",
            "Decision Metric": "Share of oracle switching value captured; beat simple deterministic rule.",
        },
        {
            "Step": "4",
            "System": "Add adversarial veto",
            "Question": "Can a critic remove likely failures without over-vetoing winners?",
            "Decision Metric": "Drawdown and tail-loss reduction with no meaningful CAGR decay.",
        },
    ]


def regime_allocator_rows() -> list[dict[str, str]]:
    return [
        {
            "Allocator": "Fixed",
            "Rule": "Always h10",
            "Purpose": "Current clean benchmark for anonymized exponential ranking.",
            "Tradable": "Yes",
        },
        {
            "Allocator": "Simple deterministic",
            "Rule": "Stressed market: h10; calm market: h20",
            "Purpose": "Two-line rule using high VIX, weak breadth, SPY below 200d, and recent selloff evidence.",
            "Tradable": "Yes",
        },
        {
            "Allocator": "LLM-selected",
            "Rule": "LLM chooses one of {5, 10, 20, 40} from market context only",
            "Purpose": "Test whether the LLM can allocate concentration better than a deterministic rule.",
            "Tradable": "Yes",
        },
        {
            "Allocator": "Oracle upper bound",
            "Rule": "Choose the best half-life after observing future returns",
            "Purpose": "Not tradable; measures maximum theoretical value of perfect switching.",
            "Tradable": "No",
        },
    ]


def short_side_objective_rows() -> list[dict[str, str]]:
    return [
        {
            "Role": "Alpha shorts",
            "Prompt Objective": "Identify recent winners likely to reverse downward.",
            "Correct Benchmark": "Trail5 shorts, random shorts, and current anonymized short rank.",
            "What It Tells Us": "Whether the short book has true standalone alpha.",
        },
        {
            "Role": "Hedge shorts",
            "Prompt Objective": "Choose names that best offset the long book's market, sector, or factor exposure.",
            "Correct Benchmark": "SPY hedge, sector ETF hedge, factor-neutral basket, random candidate hedge.",
            "What It Tells Us": "Whether shorts are useful as cheaper risk control even if raw short alpha is weak.",
        },
        {
            "Role": "Avoidance shorts",
            "Prompt Objective": "Identify recent winners least likely to keep squeezing upward.",
            "Correct Benchmark": "Did selected shorts lose less than random/Trail5 shorts?",
            "What It Tells Us": "Whether the model is mainly avoiding bad shorts rather than finding profitable shorts.",
        },
    ]


def factorial_model_rows() -> list[dict[str, str]]:
    return [
        {"Model": "DeepSeek", "Anonymous": "yes", "Visible": "yes", "Measure": "Baseline incumbent plus masking interaction."},
        {"Model": "Gemini", "Anonymous": "yes", "Visible": "yes", "Measure": "Model effect and whether masking helps another architecture."},
        {"Model": "Claude", "Anonymous": "yes", "Visible": "yes", "Measure": "Long-rank quality, short protection, complementary errors."},
        {"Model": "GPT", "Anonymous": "yes", "Visible": "yes", "Measure": "Information-design effect versus model-family effect."},
    ]


CAMPAIGN_STAGE_ORDER = ("static", "smoke", "mini", "medium", "full")


def find_latest_campaign(
    campaigns_root: Path = DEFAULT_AUTORESEARCH_CAMPAIGNS_ROOT,
) -> Path | None:
    """Find the latest campaign by reading only immediate status projections."""
    campaigns_root = Path(campaigns_root)
    if not campaigns_root.is_dir() or campaigns_root.is_symlink():
        return None

    ranked: list[tuple[int, int, str, Path]] = []
    try:
        children = list(campaigns_root.iterdir())
    except OSError:
        return None
    for child in children:
        if child.is_symlink() or not child.is_dir():
            continue
        status_path = child / "status.json"
        if status_path.is_symlink() or not status_path.is_file():
            continue
        status = _read_json_object_if_exists(status_path)
        if not status:
            continue
        raw_timestamp = status.get("last_event_at") or status.get("updated_at")
        timestamp = pd.to_datetime(raw_timestamp, utc=True, errors="coerce")
        try:
            modified_ns = status_path.stat().st_mtime_ns
        except OSError:
            modified_ns = 0
        timestamp_ns = int(timestamp.value) if not pd.isna(timestamp) else modified_ns
        ranked.append((timestamp_ns, modified_ns, child.name, child))
    return max(ranked)[-1] if ranked else None


def _campaign_period_is_safe(campaign: dict[str, Any]) -> bool:
    """Campaign dashboards may never inspect artifacts declaring a 2020+ period."""
    cutoff = pd.Timestamp("2020-01-01", tz="UTC")
    for section_name in ("development", "validation"):
        section = _as_dict(campaign.get(section_name))
        for key in ("start", "end"):
            value = section.get(key)
            if value in (None, ""):
                continue
            parsed = pd.to_datetime(value, utc=True, errors="coerce")
            if pd.isna(parsed) or parsed >= cutoff:
                return False
    return True


def _campaign_period_label(section: Any) -> str:
    values = _as_dict(section)
    start = values.get("start")
    end = values.get("end")
    if not start and not end:
        return "not configured"
    return f"{_display_value(start) or '?'} to {_display_value(end) or '?'}"


def _candidate_ids(campaign: dict[str, Any]) -> list[str]:
    raw_candidates = campaign.get("candidates", [])
    if isinstance(raw_candidates, dict):
        raw_candidates = [dict(value, candidate_id=key) if isinstance(value, dict) else key for key, value in raw_candidates.items()]
    if not isinstance(raw_candidates, list):
        return []
    candidate_ids: list[str] = []
    for item in raw_candidates:
        candidate_id = item.get("candidate_id") if isinstance(item, dict) else item
        if not isinstance(candidate_id, str):
            continue
        if candidate_id != Path(candidate_id).name or candidate_id in {"", ".", ".."}:
            continue
        if candidate_id not in candidate_ids:
            candidate_ids.append(candidate_id)
    return candidate_ids


def _load_campaign_candidates(
    campaign_dir: Path,
    campaign: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    for candidate_id in _candidate_ids(campaign):
        candidate_dir = campaign_dir / "candidates" / candidate_id
        if candidate_dir.is_symlink() or not candidate_dir.is_dir():
            candidates.append({"candidate_id": candidate_id})
            continue
        spec_path = candidate_dir / "prompt_spec.json"
        spec = {} if spec_path.is_symlink() else _read_json_object_if_exists(spec_path)
        candidate = {"candidate_id": candidate_id, **spec}
        candidate["candidate_id"] = candidate_id
        candidates.append(candidate)
        for stage in CAMPAIGN_STAGE_ORDER:
            result_path = candidate_dir / stage / "result.json"
            if result_path.is_symlink():
                continue
            result = _read_json_object_if_exists(result_path)
            if result:
                stage_result = {**result, "candidate_id": candidate_id, "stage": stage}
                if stage_result.get("status") == "failed":
                    runner_log = result_path.parent / "runner.log"
                    if not runner_log.is_symlink() and runner_log.is_file():
                        try:
                            stage_result["runner_log_tail"] = runner_log.read_text(
                                encoding="utf-8", errors="replace"
                            )[-4000:]
                        except OSError:
                            pass
                stages.append(stage_result)

    validation_path = campaign_dir / "validation" / "result.json"
    if not validation_path.is_symlink():
        validation = _read_json_object_if_exists(validation_path)
        if validation:
            validation_candidate = (
                validation.get("candidate_id") or campaign.get("winner_id") or "winner"
            )
            stages.append(
                {
                    **validation,
                    "candidate_id": validation_candidate,
                    "stage": "validation",
                }
            )
    return candidates, stages


def load_campaign_dashboard_data(campaign_dir: Path) -> dict[str, Any]:
    """Load a campaign's fixed read model; never calls an LLM or changes state."""
    campaign_dir = Path(campaign_dir)
    campaign_path = campaign_dir / "campaign.json"
    status_path = campaign_dir / "status.json"
    campaign = {} if campaign_path.is_symlink() else _read_json_object_if_exists(campaign_path)
    status = {} if status_path.is_symlink() else _read_json_object_if_exists(status_path)
    safe = bool(campaign) and _campaign_period_is_safe(campaign)
    data: dict[str, Any] = {
        "campaign_dir": campaign_dir,
        "campaign": campaign,
        "status": status,
        "events": [],
        "rejected_proposals": [],
        "candidates": [],
        "stages": [],
        "control_metadata": {},
        "control_verifications": [],
        "firewall_blocked": not safe,
        "error": "",
    }
    if not campaign:
        data["error"] = "Campaign metadata is missing or invalid."
        return data
    if not safe:
        data["error"] = "Campaign artifact reads blocked: research periods must end before 2020."
        return data
    data["events"] = _read_campaign_events(campaign_dir / "events.jsonl")
    data["rejected_proposals"] = _read_campaign_events(
        campaign_dir / "rejected_proposals.jsonl",
        limit=200,
    )
    candidates, stages = _load_campaign_candidates(campaign_dir, campaign)
    data["candidates"] = candidates
    data["stages"] = stages
    control_path = campaign_dir / "shared_controls" / "development" / "v2mc.json"
    if not control_path.is_symlink():
        control_cache = _read_json_object_if_exists(control_path)
        data["control_metadata"] = _as_dict(control_cache.get("metadata"))
    # Newer campaigns reuse immutable control artifacts directly instead of
    # copying the old shared_controls metadata shim.  Verify the real paired
    # files and hashes so absence of that legacy shim is not mislabeled as a
    # mismatch.  This audit is development-only and never reads blind-period
    # returns.
    development_end = pd.to_datetime(
        _as_dict(campaign.get("development")).get("end"), errors="coerce"
    )
    if not pd.isna(development_end) and development_end < pd.Timestamp("2015-01-01"):
        try:
            from mean_reversion_research_diagnostics import (
                paired_artifacts,
                verify_paired_artifacts,
            )

            full_ids = sorted(
                {
                    str(row.get("candidate_id"))
                    for row in stages
                    if row.get("stage") == "full" and row.get("status") == "completed"
                }
            )
            for candidate_id in full_ids:
                try:
                    data["control_verifications"].append(
                        verify_paired_artifacts(
                            paired_artifacts(campaign_dir, candidate_id, stage="full")
                        )
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    data["control_verifications"].append(
                        {
                            "candidate_id": candidate_id,
                            "pass": False,
                            "status": "audit_error",
                            "limitation": str(exc),
                            "checks": [],
                        }
                    )
        except ImportError:
            pass
    return data


_CAMPAIGN_METRIC_FIELDS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("Gross CAGR", ("gross_cagr_pct", "capital_cagr_pct", "cagr_pct"), "pct"),
    (
        "Annualized return",
        ("annualized_arithmetic_return_pct", "annualized_return_pct", "annual_return_pct"),
        "pct",
    ),
    ("Sharpe", ("sample_sharpe", "capital_sharpe", "sharpe"), "num"),
    ("Max drawdown", ("capital_max_drawdown_pct", "max_drawdown_pct"), "pct"),
    (
        "Paired annualized lift",
        ("paired_annualized_lift_pct", "paired_lift_pct", "paired_lift_pp", "paired_lift"),
        "pp",
    ),
    ("Paired hit rate", ("paired_hit_rate_pct", "paired_hit_rate"), "num"),
    (
        "Worst-decile expected shortfall",
        ("worst_decile_expected_shortfall_pct", "expected_shortfall_pct", "downside_tail_pct"),
        "pct",
    ),
    ("Turnover", ("avg_turnover_total", "turnover_pct", "turnover"), "num"),
    ("Long paired lift", ("long_paired_lift_pct", "long_contribution_pct", "long_contribution"), "pp"),
    ("Short paired lift", ("short_paired_lift_pct", "short_contribution_pct", "short_contribution"), "pp"),
    ("Block stability", ("chronological_block_stability_pct", "block_stability", "chronological_stability"), "pct"),
    ("Bootstrap lower bound", ("bootstrap_lower_bound", "bootstrap_lcb"), "num"),
    ("Bootstrap upper bound", ("bootstrap_upper_bound", "bootstrap_ucb"), "num"),
)


def _format_campaign_metric(value: Any, kind: str) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return _display_value(value)
    if kind == "pct":
        return f"{float(numeric):.2f}%"
    if kind == "pp":
        return f"{float(numeric):+.2f}pp"
    return f"{float(numeric):.3f}".rstrip("0").rstrip(".")


def _format_campaign_delta(value: Any, *, percent: bool = False) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "—"
    if percent:
        return f"{float(numeric):+.2f}%"
    return f"{float(numeric):+.3f}".rstrip("0").rstrip(".")


def _campaign_metrics_summary(raw_metrics: Any) -> str:
    metrics = _as_dict(raw_metrics)
    values: list[str] = []
    for label, keys, kind in _CAMPAIGN_METRIC_FIELDS:
        for key in keys:
            if key in metrics:
                values.append(f"{label}: {_format_campaign_metric(metrics[key], kind)}")
                break
    return " · ".join(values) or "—"


def _campaign_gate_label(raw_gate: Any) -> str:
    if not isinstance(raw_gate, dict):
        return _display_value(raw_gate) or "—"
    status = (
        raw_gate.get("status")
        or raw_gate.get("result")
        or raw_gate.get("label")
        or raw_gate.get("terminal_status")
    )
    if status is None and "pass" in raw_gate:
        status = "passed" if raw_gate.get("pass") else "failed"
    if status is None and "passed" in raw_gate:
        status = "passed" if raw_gate.get("passed") else "failed"
    reason = raw_gate.get("reason") or raw_gate.get("message")
    return " — ".join(part for part in (_display_value(status), _display_value(reason)) if part) or "—"


def _format_campaign_time(value: Any) -> str:
    """Keep campaign timestamps readable without discarding their date."""
    if value in (None, ""):
        return "—"
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return _display_value(value)
    return parsed.strftime("%Y-%m-%d %H:%M")


def _campaign_checkpoint_sequence(data: dict[str, Any]) -> list[tuple[str, int]]:
    campaign = _as_dict(data.get("campaign"))
    development = _as_dict(campaign.get("development"))
    configured = _as_dict(development.get("manifest_sizes"))
    if configured:
        rows = []
        for stage, size in configured.items():
            numeric = pd.to_numeric(size, errors="coerce")
            if not pd.isna(numeric):
                rows.append((str(stage), int(numeric)))
        if rows:
            return sorted(rows, key=lambda row: (row[1], row[0]))
    return [("smoke", 5), ("early", 12), ("mini", 24), ("reassess", 48), ("medium", 80), ("full", 251)]


def _campaign_result_passed(result: dict[str, Any]) -> bool:
    gate = _as_dict(result.get("gate"))
    passed = gate.get("pass") if "pass" in gate else gate.get("passed")
    return result.get("status") == "completed" and bool(passed)


def _campaign_frozen_rank(
    results: list[dict[str, Any]],
    stage: str,
) -> list[str]:
    """Mirror the frozen campaign's deterministic ordinal rank for display only."""
    if not results:
        return []
    keys = [
        ("paired_annualized_lift_pct", 2),
        ("sharpe_improvement", 2),
        ("tail_improvement_pct", 1),
        ("long_paired_lift_pct", 1),
        ("short_paired_lift_pct", 1),
    ]
    if stage == "medium":
        keys.extend(
            [("paired_hit_rate_pct", 1), ("chronological_block_stability_pct", 1)]
        )
    scores = {str(row.get("candidate_id")): 0.0 for row in results}

    def finite_metric(row: dict[str, Any], key: str) -> float:
        value = pd.to_numeric(_as_dict(row.get("metrics")).get(key), errors="coerce")
        return float(value) if not pd.isna(value) else float("-inf")

    for key, weight in keys:
        values = sorted({finite_metric(row, key) for row in results}, reverse=True)
        ranks = {value: rank for rank, value in enumerate(values, start=1)}
        for row in results:
            scores[str(row.get("candidate_id"))] += ranks[finite_metric(row, key)] * weight
    reliability = {
        str(row.get("candidate_id")): (
            not bool(_as_dict(row.get("reliability")).get("complete_coverage", True)),
            int(_as_dict(row.get("reliability")).get("schema_retries", 0) or 0),
            int(_as_dict(row.get("reliability")).get("transient_retries", 0) or 0),
        )
        for row in results
    }
    reliability_ranks = {
        value: rank for rank, value in enumerate(sorted(set(reliability.values())), start=1)
    }
    for candidate_id, value in reliability.items():
        scores[candidate_id] += reliability_ranks[value]
    return sorted(scores, key=lambda candidate_id: (scores[candidate_id], candidate_id))


def build_campaign_next_steps(data: dict[str, Any]) -> dict[str, Any]:
    """Explain the remaining frozen research path without making a decision."""
    campaign = _as_dict(data.get("campaign"))
    status = _as_dict(data.get("status"))
    state = str(status.get("status") or campaign.get("status") or "unknown").lower()
    current_stage = str(status.get("current_stage") or "")
    current_candidate = str(status.get("current_candidate") or "")
    stages = list(data.get("stages", []))
    candidates = list(data.get("candidates", []))
    completed_by_stage = {
        stage: [row for row in stages if row.get("stage") == stage and row.get("status") == "completed"]
        for stage, _dates in _campaign_checkpoint_sequence(data)
    }
    passed_by_stage = {
        stage: [row for row in rows if _campaign_result_passed(row)]
        for stage, rows in completed_by_stage.items()
    }

    sequence = _campaign_checkpoint_sequence(data)
    targets: dict[str, int] = {}
    prior_target = len(candidates)
    for stage, _dates in sequence:
        if stage == "medium":
            prior = passed_by_stage.get("reassess") or passed_by_stage.get("mini", [])
            prior_target = min(3, len(prior))
        elif stage == "full":
            prior_target = min(2, len(passed_by_stage.get("medium", [])))
        elif stage not in {"smoke", "early", "mini", "reassess"}:
            prior_target = len(passed_by_stage.get(stage, []))
        targets[stage] = prior_target

    path_parts = [f"Proposals: {len(candidates)} accepted"]
    for stage, date_count in sequence:
        completed = len(completed_by_stage.get(stage, []))
        target = targets.get(stage, 0)
        marker = "running" if current_stage == stage else ("done" if target and completed >= target else "next")
        path_parts.append(f"{date_count} dates: {completed}/{target} {marker}")
    winner = campaign.get("winner_id") or status.get("winner_id")
    path_parts.append("Manual validation approval: waiting" if winner else "Manual validation approval: locked")

    if state in {"awaiting_validation_approval", "validation_approved"} or (
        winner and not current_stage
    ):
        next_steps = [
            f"Review the frozen development winner {winner}.",
            "Nothing else runs automatically; validation starts only after your explicit approval.",
        ]
    elif state in {"no_winner", "statistically_confirmed", "validated_winner", "failed_validation"}:
        next_steps = ["Nothing else is scheduled; the campaign is terminal."]
    elif state in {"stopped", "error", "failed"}:
        next_steps = ["Nothing is running. Resolve the recorded stop/error before resuming this frozen campaign."]
    elif current_stage == "full":
        ranked = _campaign_frozen_rank(passed_by_stage.get("medium", []), "medium")[:2]
        remaining = [candidate_id for candidate_id in ranked if candidate_id != current_candidate and not any(
            row.get("candidate_id") == candidate_id for row in completed_by_stage.get("full", [])
        )]
        next_steps = [f"Finish the 251-date full-development run for {current_candidate}."]
        if remaining:
            next_steps.append(
                f"Run the other full-development finalist, {remaining[0]}, on the same 251 dates."
            )
        next_steps.extend(
            [
                "Apply the frozen full-development gates and compare all qualifying finalists.",
                "Freeze at most one winner—or end with no winner. Validation will not start automatically.",
            ]
        )
    elif current_stage == "medium":
        next_steps = [
            f"Finish the 80-date medium run for {current_candidate}.",
            "Complete any remaining medium finalists, rank the passing results, and send at most two to 251 dates.",
            "After the full runs, freeze at most one winner and stop for manual validation approval.",
        ]
    elif current_stage:
        next_steps = [
            f"Finish {current_candidate or 'the current candidate'} at the {current_stage} checkpoint.",
            "Continue through the remaining frozen checkpoints, pruning and ranking at each configured decision point.",
            "Run at most two full-development finalists, then freeze at most one winner and stop for approval.",
        ]
    else:
        next_steps = ["The supervisor is active and will continue from the next incomplete frozen checkpoint."]
    return {"path_parts": path_parts, "next_steps": next_steps}


def build_campaign_plain_summary(data: dict[str, Any]) -> dict[str, str]:
    """Answer the operational questions a campaign owner asks first."""
    campaign = _as_dict(data.get("campaign"))
    status = _as_dict(data.get("status"))
    state = str(status.get("status") or campaign.get("status") or "unknown").lower()
    supervisor_pid = status.get("supervisor_pid")
    runner_pid = status.get("runner_pid")
    is_running = state in {"running", "starting", "stopping"} and bool(supervisor_pid or runner_pid)
    stages = list(data.get("stages", []))
    candidates = list(data.get("candidates", []))
    failed = sum(result.get("status") == "failed" for result in stages)
    terminal_reason = status.get("terminal_reason") or campaign.get("terminal_reason")
    current = " / ".join(
        str(value)
        for value in (status.get("current_candidate"), status.get("current_stage"))
        if value
    )
    if is_running:
        activity = f"The supervisor is working on {current}." if current else "The supervisor is active."
        headline = "Yes — research is running"
    else:
        activity = "No supervisor or experiment runner is active."
        headline = "No — nothing is running"
    checkpoint_counts = []
    for checkpoint, date_count in _campaign_checkpoint_sequence(data):
        completed = sum(
            result.get("stage") == checkpoint and result.get("status") == "completed"
            for result in stages
        )
        checkpoint_counts.append(f"{date_count} dates {completed}")
    progress = (
        f"The Pro researcher produced {len(candidates)} accepted prompt proposals. "
        f"Completed checkpoints: {', '.join(checkpoint_counts)}."
    )
    if failed:
        progress += f" {failed} stage run{'s' if failed != 1 else ''} failed."
    outcome = (
        f"The campaign stopped because: {terminal_reason}."
        if terminal_reason
        else "The campaign has not recorded a terminal reason."
    )
    winner = campaign.get("winner_id") or status.get("winner_id")
    decision = (
        f"Frozen development winner: {winner}. Validation still requires explicit approval."
        if winner
        else "No development winner was selected, and validation did not run."
    )
    return {
        "headline": headline,
        "activity": activity,
        "progress": progress,
        "outcome": outcome,
        "decision": decision,
        "updated": _format_campaign_time(status.get("last_event_at")),
    }


def build_campaign_winner_panel(data: dict[str, Any]) -> str:
    """Render a concise development decision once the full runs are complete."""
    campaign = _as_dict(data.get("campaign"))
    status = _as_dict(data.get("status"))
    winner_id = _display_value(campaign.get("winner_id") or status.get("winner_id"))
    full_results = [
        result
        for result in data.get("stages", [])
        if result.get("stage") == "full" and result.get("status") == "completed"
    ]
    if not winner_id or not full_results:
        return ""

    candidates = {
        _display_value(candidate.get("candidate_id")): candidate
        for candidate in data.get("candidates", [])
    }
    winner = next(
        (result for result in full_results if _display_value(result.get("candidate_id")) == winner_id),
        None,
    )
    if winner is None:
        return ""

    def metric(result: dict[str, Any], key: str, digits: int = 2, signed: bool = False) -> str:
        value = pd.to_numeric(_as_dict(result.get("metrics")).get(key), errors="coerce")
        if pd.isna(value):
            return "—"
        sign = "+" if signed else ""
        return f"{float(value):{sign}.{digits}f}"

    winner_spec = _as_dict(candidates.get(winner_id))
    control_metrics = _as_dict(_as_dict(winner.get("controls")).get("fresh_incumbent"))
    descriptor = (
        winner_spec.get("brief_descriptor")
        or winner_spec.get("descriptor")
        or winner_spec.get("hypothesis")
        or "See the prompt-change section for the exact mutation."
    )
    finalist_rows = []
    for result in sorted(full_results, key=lambda row: _display_value(row.get("candidate_id"))):
        candidate_id = _display_value(result.get("candidate_id"))
        passed = _campaign_result_passed(result)
        gate = _as_dict(result.get("gate"))
        if passed:
            outcome = "Winner · passed all development gates" if candidate_id == winner_id else "Qualified"
        else:
            failures = []
            if gate.get("cagr_or_sharpe_vs_incumbent") is False:
                failures.append("did not improve CAGR or Sharpe vs control")
            if gate.get("beats_trail5_cagr_and_sharpe") is False:
                failures.append("did not beat Trail5 on both CAGR and Sharpe")
            if gate.get("drawdown_ok") is False:
                failures.append("drawdown gate failed")
            lift = pd.to_numeric(_as_dict(result.get("metrics")).get("paired_annualized_lift_pct"), errors="coerce")
            if not pd.isna(lift) and float(lift) <= 0:
                failures.append("lift vs control was not positive")
            outcome = "Pruned · " + ("; ".join(failures) or "development gate failed")
        finalist_rows.append(
            f"""
            <tr class="{'winner-row' if candidate_id == winner_id else ''}">
              <td><strong>{_escape(candidate_id)}</strong></td>
              <td>{_escape(metric(result, 'cagr_pct'))}%</td>
              <td>{_escape(metric(result, 'sharpe', 3))}</td>
              <td>{_escape(metric(result, 'paired_annualized_lift_pct', signed=True))}%</td>
              <td>{_escape(metric(result, 'sharpe_improvement', 3, signed=True))}</td>
              <td>{_escape(outcome)}</td>
            </tr>
            """
        )

    validation = _as_dict(campaign.get("validation"))
    approved = bool(validation.get("approved"))
    action = (
        "Validation approval is recorded. Run the resume command below to start the frozen 2015–2019 validation."
        if approved
        else "Review this decision, then record approval before starting the frozen 2015–2019 validation."
    )
    command = (
        ".venv/bin/python mean_reversion_llm_v2_autoresearch.py campaign resume "
        if approved
        else ".venv/bin/python mean_reversion_llm_v2_autoresearch.py campaign approve-validation "
    ) + _display_value(campaign.get("campaign_id") or status.get("campaign_id"))
    return f"""
      <section id="decision" class="winner-panel">
        <div class="eyebrow">Development decision</div>
        <div class="winner-heading">
          <div>
            <h2>Winner: {_escape(winner_id)}</h2>
            <p>{_escape(descriptor)}</p>
          </div>
          <span class="winner-badge">Ready for validation</span>
        </div>
        <div class="winner-metrics">
          <div><span>Gross CAGR</span><strong>{_escape(metric(winner, 'cagr_pct'))}%</strong></div>
          <div><span>Sharpe</span><strong>{_escape(metric(winner, 'sharpe', 3))}</strong></div>
          <div><span>Lift vs control</span><strong>{_escape(metric(winner, 'paired_annualized_lift_pct', signed=True))}%</strong></div>
          <div><span>Sharpe change</span><strong>{_escape(metric(winner, 'sharpe_improvement', 3, signed=True))}</strong></div>
          <div><span>Max drawdown</span><strong>{_escape(metric(winner, 'max_drawdown_pct'))}%</strong></div>
        </div>
        <div class="control-comparison">
          <strong>v2-minimal-change-control</strong>
          <span>Gross CAGR: {_escape(_format_campaign_metric(control_metrics.get('cagr_pct'), 'pct'))}</span>
          <span>Sharpe: {_escape(_format_campaign_metric(control_metrics.get('sharpe'), 'num'))}</span>
        </div>
        <p class="decision-explanation"><strong>Why it won:</strong> It passed every frozen full-development economic gate and ranked first among qualifying finalists. Metrics are gross and use only the 2009–2014 development period.</p>
        <div class="table-wrap">
          <table class="finalist-table">
            <thead><tr><th>Finalist</th><th>CAGR</th><th>Sharpe</th><th>Lift vs control</th><th>Sharpe change</th><th>Decision</th></tr></thead>
            <tbody>{''.join(finalist_rows)}</tbody>
          </table>
        </div>
        <div class="validation-action">
          <strong>Next action:</strong> {_escape(action)}
          <code>{_escape(command)}</code>
        </div>
      </section>
    """


def build_campaign_summary_rows(data: dict[str, Any]) -> list[dict[str, str]]:
    campaign = _as_dict(data.get("campaign"))
    status = _as_dict(data.get("status"))
    validation = _as_dict(campaign.get("validation"))
    progress = " / ".join(
        value
        for value in (
            _display_value(status.get("completed_dates")),
            _display_value(status.get("planned_dates")),
        )
        if value
    ) or "—"
    values = [
        ("Campaign", campaign.get("campaign_id") or status.get("campaign_id")),
        ("Status", status.get("status") or campaign.get("status")),
        ("Phase", status.get("phase") or campaign.get("phase")),
        ("Current candidate", status.get("current_candidate")),
        ("Current stage", status.get("current_stage")),
        ("Date progress", progress),
        ("Development period", _campaign_period_label(campaign.get("development"))),
        ("Validation period", _campaign_period_label(validation)),
        ("Validation approved", validation.get("approved", False)),
        ("Validation status", validation.get("status") or "not run"),
        ("Frozen winner", campaign.get("winner_id") or "none"),
        ("Terminal reason", campaign.get("terminal_reason") or "—"),
        ("Last event", status.get("last_event_at") or "—"),
    ]
    return [{"Field": label, "Value": _display_value(value)} for label, value in values]


def _campaign_usage_scope(usage: dict[str, Any], scope: str) -> dict[str, Any]:
    nested = usage.get(scope)
    if isinstance(nested, dict):
        return nested
    if any(isinstance(usage.get(name), dict) for name in ("development", "validation")):
        return {}
    return usage if scope == "development" else {}


def build_campaign_budget_rows(data: dict[str, Any]) -> list[dict[str, str]]:
    campaign = _as_dict(data.get("campaign"))
    status = _as_dict(data.get("status"))
    budgets = _as_dict(campaign.get("budgets"))
    usage = _as_dict(status.get("usage") or campaign.get("usage"))
    rows: list[dict[str, str]] = []
    for scope in ("development", "validation"):
        limits = _as_dict(budgets.get(scope))
        used_values = _campaign_usage_scope(usage, scope)
        resources = set(used_values)
        resources.update(
            key[4:] if str(key).startswith("max_") else str(key)
            for key in limits
        )
        for resource in sorted(resources):
            limit = limits.get(resource, limits.get(f"max_{resource}"))
            used = used_values.get(resource, 0)
            if isinstance(limit, dict):
                limit = limit.get("limit") or limit.get("maximum")
            if isinstance(used, dict):
                used = used.get("used") or used.get("value") or 0
            used_num = pd.to_numeric(used, errors="coerce")
            limit_num = pd.to_numeric(limit, errors="coerce")
            remaining = ""
            if not pd.isna(used_num) and not pd.isna(limit_num):
                remaining = _display_value(float(limit_num) - float(used_num))
            rows.append(
                {
                    "Scope": scope.title(),
                    "Resource": str(resource).replace("_", " ").title(),
                    "Used": _display_value(used),
                    "Limit": _display_value(limit),
                    "Remaining": remaining,
                }
            )
    return rows


def build_campaign_candidate_rows(data: dict[str, Any]) -> list[dict[str, str]]:
    stage_results: dict[str, dict[str, dict[str, Any]]] = {}
    for result in data.get("stages", []):
        candidate_id = _display_value(result.get("candidate_id"))
        stage = _display_value(result.get("stage"))
        stage_results.setdefault(candidate_id, {})[stage] = result

    status = _as_dict(data.get("status"))
    active_candidate = _display_value(status.get("current_candidate"))
    active_stage = _display_value(status.get("current_stage"))
    campaign_running = str(status.get("status", "")).lower() == "running"

    def stage_line(candidate_id: str, stage: str, dates: int) -> str:
        result = stage_results.get(candidate_id, {}).get(stage)
        if not result and campaign_running and candidate_id == active_candidate and stage == active_stage:
            completed = _display_value(status.get("completed_dates")) or "0"
            planned = _display_value(status.get("planned_dates")) or str(dates)
            return f"{stage.title()} ({dates}): Running — {completed}/{planned} dates"
        if not result:
            return f"{stage.title()} ({dates}): Not reached"
        result_status = _display_value(result.get("status")).title()
        gate = _as_dict(result.get("gate"))
        reason = _display_value(gate.get("reason"))
        if result_status == "Failed":
            detail = f" — {reason.replace('_', ' ')}" if reason else ""
            return f"{stage.title()} ({dates}): Failed{detail}"
        if result_status == "Completed" and gate.get("pass") is False:
            detail = f" — {reason.replace('_', ' ')}" if reason else ""
            return f"{stage.title()} ({dates}): Pruned{detail}"
        metrics = _as_dict(result.get("metrics"))
        lift = metrics.get("paired_annualized_lift_pct")
        sharpe_change = metrics.get("sharpe_improvement")
        if lift is not None or sharpe_change is not None:
            lift_text = _format_campaign_delta(lift, percent=True) if lift is not None else "—"
            sharpe_text = _format_campaign_delta(sharpe_change) if sharpe_change is not None else "—"
            survivor_note = (
                "; advanced by minimum-survivor protection"
                if reason == "minimum_survivor_protection"
                else ""
            )
            return (
                f"{stage.title()} ({dates}): Completed — return lift {lift_text}; "
                f"Sharpe change {sharpe_text}{survivor_note}"
            )
        return f"{stage.title()} ({dates}): {result_status or 'Recorded'}"

    rows: list[dict[str, str]] = []
    for candidate in data.get("candidates", []):
        candidate_id = _display_value(candidate.get("candidate_id"))
        stage_progress = "\n".join(
            stage_line(candidate_id, stage, dates)
            for stage, dates in _campaign_checkpoint_sequence(data)
        )
        rows.append(
            {
                "Candidate": candidate_id,
                "Prompt idea": _display_value(candidate.get("hypothesis")),
                "Stage performance": stage_progress,
            }
        )
    return rows


def build_campaign_control_panel(data: dict[str, Any]) -> str:
    campaign = _as_dict(data.get("campaign"))
    harness = _as_dict(campaign.get("harness"))
    incumbent = _as_dict(campaign.get("incumbent"))
    metadata = _as_dict(data.get("control_metadata"))
    expected = {
        "prompt_variant": "v2-minimal-change-control",
        "prompt_hash": incumbent.get("prompt_hash"),
        "model": harness.get("model"),
        "temperature": harness.get("temperature"),
        "thinking_mode": harness.get("thinking_mode"),
    }
    if "anonymize_tickers" in harness:
        expected["anonymize_tickers"] = harness.get("anonymize_tickers")
    artifact_audits = list(data.get("control_verifications", []))
    verified = bool(artifact_audits) and all(bool(row.get("pass")) for row in artifact_audits)
    legacy_verified = bool(metadata) and all(
        metadata.get(key) == value for key, value in expected.items() if value is not None
    )
    if verified:
        has_limitation = any(bool(row.get("limitation")) for row in artifact_audits)
        verification = (
            "Frozen pairs verified; backend build ID was not recorded"
            if has_limitation
            else "Verified from frozen paired artifacts"
        )
    elif legacy_verified:
        verification = "Verified from control cache"
    elif artifact_audits:
        failed_ids = ", ".join(
            str(row.get("candidate_id")) for row in artifact_audits if not row.get("pass")
        )
        verification = f"Paired artifact verification failed: {failed_ids}"
    else:
        verification = "Control metadata mismatch or unavailable"
    verification_class = "benchmark-ok" if (verified or legacy_verified) else "benchmark-warning"
    audited_visibility = sorted(
        {
            str(check.get("detail", "")).split(" vs ")[0].strip("'")
            for audit in artifact_audits
            for check in audit.get("checks", [])
            if check.get("invariant") == "ticker visibility" and check.get("pass")
        }
    )
    values = [
        ("Prompt", metadata.get("prompt_variant") or expected["prompt_variant"]),
        ("Model", metadata.get("model") or harness.get("model")),
        ("Temperature", metadata.get("temperature", harness.get("temperature"))),
        ("Thinking", metadata.get("thinking_mode") or harness.get("thinking_mode")),
        (
            "Tickers",
            (
                "candidate-selected; each full pair audited"
                if audited_visibility
                else (
                    "anonymous"
                    if metadata.get("anonymize_tickers", harness.get("anonymize_tickers"))
                    else "visible"
                )
            ),
        ),
        ("Dates", "identical and paired within every stage"),
        ("Candidate pools", "identical"),
        ("Portfolio construction", "identical"),
        ("Returns", str(harness.get("return_basis", "gross fixed-horizon C2C")).replace("_", " ")),
        ("Prompt hash", metadata.get("prompt_hash") or incumbent.get("prompt_hash")),
    ]
    items = "".join(
        f"<div><span>{_escape(label)}</span><strong>{_escape(_display_value(value))}</strong></div>"
        for label, value in values
    )
    return f"""
      <div class="benchmark-heading">
        <div>
          <h2>Benchmark Behind Every Lift</h2>
          <p>Every return lift and Sharpe change below compares the candidate with this fresh control.</p>
        </div>
        <span class="benchmark-verification {verification_class}">{_escape(verification)}</span>
      </div>
      <div class="benchmark-grid">{items}</div>
    """


def build_campaign_stage_diagnostics(data: dict[str, Any]) -> str:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for result in data.get("stages", []):
        if result.get("stage") not in {"smoke", "early", "mini", "reassess", "medium", "full"}:
            continue
        by_candidate.setdefault(_display_value(result.get("candidate_id")), []).append(result)

    def metric(metrics: dict[str, Any], key: str, *, percent: bool = False) -> str:
        value = metrics.get(key)
        if value is None:
            return "—"
        return _format_campaign_delta(value, percent=percent) if key in {
            "paired_annualized_lift_pct", "long_paired_lift_pct", "short_paired_lift_pct"
        } else _format_campaign_metric(value, "pct" if percent else "num")

    cards: list[str] = []
    order = {
        stage: index
        for index, (stage, _dates) in enumerate(_campaign_checkpoint_sequence(data))
    }
    for candidate in data.get("candidates", []):
        candidate_id = _display_value(candidate.get("candidate_id"))
        stage_blocks: list[str] = []
        for result in sorted(by_candidate.get(candidate_id, []), key=lambda row: order.get(row.get("stage"), 99)):
            stage = _display_value(result.get("stage")).title()
            status = _display_value(result.get("status")).title()
            metrics = _as_dict(result.get("metrics"))
            diagnostics = [
                ("Return lift", metric(metrics, "paired_annualized_lift_pct", percent=True)),
                ("Sharpe change", metric(metrics, "sharpe_improvement")),
                ("Paired hit rate", metric(metrics, "paired_hit_rate_pct", percent=True)),
                ("Long-side lift", metric(metrics, "long_paired_lift_pct", percent=True)),
                ("Short-side lift", metric(metrics, "short_paired_lift_pct", percent=True)),
                ("Block stability", metric(metrics, "chronological_block_stability_pct", percent=True)),
                ("Long top-30 overlap", metric(metrics, "long_top_book_overlap_pct", percent=True)),
                ("Short top-30 overlap", metric(metrics, "short_top_book_overlap_pct", percent=True)),
                ("Long score-rank correlation", metric(metrics, "long_score_rank_correlation")),
                ("Short score-rank correlation", metric(metrics, "short_score_rank_correlation")),
            ]
            visible = [(label, value) for label, value in diagnostics if value != "—"]
            diagnostics_html = "".join(
                f"<div><span>{_escape(label)}</span><strong>{_escape(value)}</strong></div>"
                for label, value in visible
            ) or '<p class="muted">No diagnostics were produced for this stage.</p>'
            stage_blocks.append(
                f'<div class="diagnostic-stage"><h4>{_escape(stage)} · {_escape(status)}</h4>'
                f'<div class="diagnostic-grid">{diagnostics_html}</div></div>'
            )
        if stage_blocks:
            cards.append(
                f'<details class="campaign-diagnostics"><summary>{_escape(candidate_id)} stage diagnostics</summary>'
                + "".join(stage_blocks)
                + "</details>"
            )
    return "".join(cards) or '<p class="muted">No completed-stage diagnostics yet.</p>'


def build_campaign_prompt_change_cards(data: dict[str, Any]) -> str:
    """Render immutable prompt proposals with escaped before/after text and full prompts."""
    cards: list[str] = []
    for candidate in data.get("candidates", []):
        candidate_id = _display_value(candidate.get("candidate_id"))
        generation = _display_value(candidate.get("generation"))
        mutation = _display_value(candidate.get("mutation_mode"))
        score_contract = (
            f"{_display_value(candidate.get('score_min', 0))} to {_display_value(candidate.get('score_max', 100))}, "
            f"{_display_value(candidate.get('score_direction', 'higher_is_better')).replace('_', ' ')}"
        )
        hypothesis = _display_value(candidate.get("hypothesis"))
        mechanism = _display_value(candidate.get("mechanism"))
        expected_failure = _display_value(candidate.get("expected_failure"))
        patch = _as_dict(candidate.get("patch"))
        old_text = _display_value(patch.get("old_text"))
        new_text = _display_value(patch.get("new_text"))
        prompt_text = _display_value(candidate.get("prompt_text"))
        descriptor = hypothesis if hypothesis not in {"", "—"} else f"{mutation} prompt mutation"
        cards.append(
            f"""
            <details class="campaign-prompt-change">
              <summary>{_escape(candidate_id)} · {_escape(descriptor)}</summary>
              <div class="campaign-prompt-meta">
                <span><strong>Generation:</strong> {_escape(generation)}</span>
                <span><strong>Mutation:</strong> {_escape(mutation)}</span>
                <span><strong>Native score contract:</strong> {_escape(score_contract)}</span>
              </div>
              <p><strong>What changed:</strong> {_escape(descriptor)}</p>
              <p><strong>Expected mechanism:</strong> {_escape(mechanism)}</p>
              <p><strong>Expected failure:</strong> {_escape(expected_failure)}</p>
              <div class="campaign-prompt-diff">
                <div>
                  <h3>Replaced text</h3>
                  <pre>{_escape(old_text)}</pre>
                </div>
                <div>
                  <h3>New text</h3>
                  <pre>{_escape(new_text)}</pre>
                </div>
              </div>
              <details class="campaign-full-prompt">
                <summary>Show complete resulting prompt</summary>
                <pre>{_escape(prompt_text)}</pre>
              </details>
            </details>
            """
        )
    if not cards:
        return '<p class="muted">No prompt proposals have been persisted yet.</p>'
    return "".join(cards)


def build_campaign_rejected_prompt_cards(data: dict[str, Any]) -> str:
    """Render sanitized rejected proposals without opening raw provider responses."""
    cards: list[str] = []
    for record in data.get("rejected_proposals", []):
        proposal = _as_dict(record.get("proposal"))
        patch = _as_dict(proposal.get("patch"))
        rejection_id = _display_value(record.get("rejection_id"))
        reason = _display_value(record.get("reason"))
        hypothesis = _display_value(proposal.get("hypothesis"))
        descriptor = hypothesis if hypothesis not in {"", "—"} else "Rejected prompt proposal"
        cards.append(
            f"""
            <details class="campaign-prompt-change campaign-rejected-prompt">
              <summary>{_escape(rejection_id)} · {_escape(descriptor)}</summary>
              <div class="campaign-prompt-meta">
                <span><strong>Time:</strong> {_escape(_format_campaign_time(record.get('timestamp')))}</span>
                <span><strong>Generation:</strong> {_escape(_display_value(record.get('generation')))}</span>
                <span><strong>Mutation:</strong> {_escape(_display_value(proposal.get('mutation_mode')))}</span>
              </div>
              <p><strong>Rejected because:</strong> {_escape(reason)}</p>
              <p><strong>Expected mechanism:</strong> {_escape(_display_value(proposal.get('mechanism')))}</p>
              <p><strong>Expected failure:</strong> {_escape(_display_value(proposal.get('expected_failure')))}</p>
              <div class="campaign-prompt-diff">
                <div><h3>Proposed replaced text</h3><pre>{_escape(_display_value(patch.get('old_text')))}</pre></div>
                <div><h3>Proposed new text</h3><pre>{_escape(_display_value(patch.get('new_text')))}</pre></div>
              </div>
            </details>
            """
        )
    if not cards:
        return '<p class="muted">No sanitized rejected proposals have been persisted.</p>'
    return "".join(cards)


def build_campaign_stage_rows(data: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "Candidate": _display_value(result.get("candidate_id")),
            "Stage": _display_value(result.get("stage")),
            "Status": _display_value(result.get("status")),
            "Gross metrics": _campaign_metrics_summary(result.get("metrics")),
            "Recorded gate": _campaign_gate_label(result.get("gate")),
        }
        for result in data.get("stages", [])
    ]


def _campaign_event_detail(event: dict[str, Any]) -> str:
    payload = _as_dict(event.get("payload"))
    for key in ("error", "message", "reason", "result_class", "status", "permitted_action"):
        if payload.get(key) not in (None, ""):
            return _display_value(payload[key])[:500]
    return ""


def build_campaign_event_rows(data: dict[str, Any], limit: int = 25) -> list[dict[str, str]]:
    return [
        {
            "Time": _display_value(event.get("timestamp")),
            "Event": _display_value(event.get("event")),
            "Candidate": _display_value(event.get("candidate_id")),
            "Stage": _display_value(event.get("stage")),
            "Detail": _campaign_event_detail(event),
        }
        for event in data.get("events", [])[-max(1, int(limit)) :]
    ]


def build_campaign_failure_rows(data: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    status_error = _as_dict(data.get("status")).get("error")
    if status_error:
        rows.append({"Time": "Current", "Candidate": "", "Stage": "", "What went wrong": _display_value(status_error)})
    for result in data.get("stages", []):
        if result.get("status") != "failed" and not result.get("error"):
            continue
        candidate = _display_value(result.get("candidate_id"))
        stage = _display_value(result.get("stage"))
        runner_log_tail = _display_value(result.get("runner_log_tail"))
        if "init_sys_streams: can't initialize sys standard streams" in runner_log_tail and "Bad file descriptor" in runner_log_tail:
            detail = "Python subprocess could not start: its standard input/output file descriptor was invalid. No experiment dates ran."
        else:
            detail = _display_value(result.get("error"))
        if not detail:
            detail = _campaign_gate_label(result.get("gate"))
        key = (candidate, stage, detail)
        seen.add(key)
        rows.append(
            {
                "Time": "—",
                "Candidate": candidate,
                "Stage": stage,
                "What went wrong": detail,
            }
        )
    for event in data.get("events", []):
        event_name = _display_value(event.get("event"))
        detail = _campaign_event_detail(event)
        payload = _as_dict(event.get("payload"))
        named_failure = any(word in event_name.lower() for word in ("fail", "error", "retry"))
        if not named_failure and payload.get("error") in (None, ""):
            continue
        candidate = _display_value(event.get("candidate_id"))
        stage = _display_value(event.get("stage"))
        failure = detail or event_name.replace("_", " ")
        key = (candidate, stage, failure)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "Time": _format_campaign_time(event.get("timestamp")),
                "Candidate": candidate,
                "Stage": stage,
                "What went wrong": failure,
            }
        )
    return rows[-25:]


def shared_suite_css() -> str:
    return """
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #17222d;
      --muted: #687589;
      --line: #d8e0e8;
      --head: #edf2f6;
      --accent: #247665;
      --accent-soft: #edf8f5;
      --warn-soft: #fff7e8;
      --shadow: 0 14px 34px rgba(22, 33, 44, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    main { max-width: 1180px; margin: 0 auto; padding: 28px; }
    a { color: inherit; }
    h1 { margin: 0 0 8px; font-size: clamp(30px, 4vw, 46px); line-height: 1.05; letter-spacing: 0; }
    h2 { margin: 0 0 8px; font-size: 22px; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 17px; letter-spacing: 0; }
    p { margin: 0 0 10px; color: var(--muted); }
    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      margin-bottom: 22px;
    }
    .brand { font-weight: 900; color: #203129; }
    .nav { display: flex; flex-wrap: wrap; gap: 8px; }
    .nav a {
      text-decoration: none;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 13px;
      font-weight: 800;
      color: #2a3b34;
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(280px, .8fr);
      gap: 18px;
      align-items: stretch;
      margin-bottom: 18px;
    }
    .hero-card, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: var(--shadow);
      padding: 22px;
    }
    .eyebrow {
      color: var(--accent);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: .08em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .lead { color: #334252; font-size: 18px; max-width: 850px; }
    .summary-list {
      margin: 12px 0 0;
      padding-left: 18px;
      color: #334252;
      font-weight: 650;
    }
    .summary-list li { margin: 8px 0; }
    section { margin-bottom: 18px; }
    .callout {
      background: var(--accent-soft);
      border-left: 5px solid var(--accent);
      color: #244d44;
      padding: 13px 14px;
      margin: 14px 0;
      font-weight: 750;
    }
    .warning { background: var(--warn-soft); border-left-color: #b5812d; color: #6f4b00; }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      margin-top: 12px;
    }
    table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; }
    th {
      text-align: left;
      background: var(--head);
      color: #263647;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .05em;
      white-space: nowrap;
    }
    th, td { padding: 10px 11px; border-bottom: 1px solid var(--line); vertical-align: top; }
    tbody tr:nth-child(even) { background: #fafcfd; }
    @media (max-width: 860px) {
      main { padding: 18px; }
      .topbar, .hero { display: block; }
      .nav { margin-top: 12px; }
      .hero-card { margin-bottom: 14px; }
    }
    """


def write_autoresearch_campaign_dashboard(
    campaign_dir: Path,
    output_path: Path | None = None,
    hub_path: Path | None = None,
) -> Path:
    """Render one campaign from persisted state without running research."""
    campaign_dir = Path(campaign_dir)
    output_path = Path(output_path) if output_path else campaign_dir / DEFAULT_AUTORESEARCH_CAMPAIGN_OUTPUT_NAME
    hub_path = Path(hub_path) if hub_path else DEFAULT_RESULTS_ROOT / DEFAULT_HUB_OUTPUT_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_campaign_dashboard_data(campaign_dir)
    campaign = _as_dict(data.get("campaign"))
    status = _as_dict(data.get("status"))
    campaign_id = campaign.get("campaign_id") or status.get("campaign_id") or campaign_dir.name
    status_label = status.get("status") or campaign.get("status") or "unknown"
    phase_label = status.get("phase") or campaign.get("phase") or "unknown"
    hub_href = relative_href(output_path, hub_path)
    checkpoint_dates = [dates for _stage, dates in _campaign_checkpoint_sequence(data)]
    if checkpoint_dates == [5, 12, 24, 48, 80, 251]:
        race_description = (
            "Each candidate follows the shared 5 → 12 → 24 → 48 → 80 → 251-date race. "
            "Five dates are mechanical-only; 12/24/48 are futility screens; 80 dates "
            "are the first meaningful economic comparison; 251 dates are full paired inference."
        )
    else:
        race_description = (
            "This campaign used its frozen "
            + " → ".join(str(value) for value in checkpoint_dates)
            + "-date stage sequence: 5 is mechanical-only, 24 is futility-only, "
            "80 is the first economic comparison, and 251 is full paired inference."
        )

    error_html = ""
    if data.get("error"):
        error_html = f'<div class="callout warning">{_escape(data["error"])}</div>'
    plain_summary = build_campaign_plain_summary(data)
    next_steps = build_campaign_next_steps(data)
    winner_panel = build_campaign_winner_panel(data)
    control_panel = build_campaign_control_panel(data)
    candidate_table = html_table(build_campaign_candidate_rows(data), "campaign-lineage")
    stage_diagnostics = build_campaign_stage_diagnostics(data)
    prompt_change_cards = build_campaign_prompt_change_cards(data)
    rejected_prompt_cards = build_campaign_rejected_prompt_cards(data)
    failure_table = html_table(build_campaign_failure_rows(data), "campaign-failures")

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Autoresearch Campaign {_escape(campaign_id)}</title>
  <style>
    {shared_suite_css()}
    .hero {{ display: block; }}
    .state-pill {{
      display: inline-block;
      border: 1px solid #9fc9bd;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #24594d;
      padding: 6px 10px;
      margin: 4px 6px 0 0;
      font-size: 13px;
      font-weight: 850;
    }}
    .table-wrap + .table-wrap {{ margin-top: 16px; }}
    .campaign-prompt-change {{
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      background: var(--panel);
    }}
    .campaign-prompt-change > summary,
    .campaign-full-prompt > summary {{ cursor: pointer; color: var(--accent); font-weight: 800; }}
    .campaign-prompt-meta {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 12px 0; }}
    .campaign-prompt-diff {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .campaign-prompt-diff pre,
    .campaign-full-prompt pre {{ white-space: pre-wrap; overflow-wrap: anywhere; max-height: 34rem; overflow: auto; }}
    .campaign-full-prompt {{ margin-top: 14px; }}
    .campaign-rejected-prompt {{ border-left: 5px solid #b55252; }}
    .campaign-lineage td:last-child {{ white-space: pre-line; line-height: 1.75; min-width: 390px; }}
    .benchmark-heading {{ display: flex; justify-content: space-between; gap: 18px; align-items: start; }}
    .benchmark-verification {{ border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 850; white-space: nowrap; }}
    .benchmark-ok {{ color: #24594d; background: var(--accent-soft); border: 1px solid #9fc9bd; }}
    .benchmark-warning {{ color: #6f4b00; background: var(--warn-soft); border: 1px solid #d9b567; }}
    .benchmark-grid, .diagnostic-grid {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; margin-top: 12px; }}
    .benchmark-grid div, .diagnostic-grid div {{ display: flex; justify-content: space-between; gap: 14px; background: #fff; padding: 9px 11px; }}
    .benchmark-grid span, .diagnostic-grid span {{ color: var(--muted); }}
    .campaign-diagnostics {{ margin-top: 10px; border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }}
    .campaign-diagnostics > summary {{ cursor: pointer; color: var(--accent); font-weight: 850; }}
    .diagnostic-stage {{ margin-top: 14px; }}
    .diagnostic-stage h4 {{ margin: 0 0 6px; }}
    .plain-summary {{ display: grid; gap: 11px; margin-top: 18px; }}
    .plain-summary p {{ margin: 0; color: #334252; font-size: 16px; }}
    .plain-summary strong {{ color: var(--ink); }}
    .now-card {{ border-left: 6px solid var(--accent); }}
    .stopped-card {{ border-left-color: #b55252; }}
    .research-path {{ display: flex; flex-wrap: wrap; align-items: center; gap: 7px; margin-top: 12px; }}
    .path-step {{ border: 1px solid var(--line); border-radius: 999px; padding: 6px 9px; background: #fff; font-size: 12px; font-weight: 800; }}
    .path-arrow {{ color: var(--muted); font-weight: 900; }}
    .next-actions {{ margin: 8px 0 0; padding-left: 22px; color: #334252; }}
    .next-actions li {{ margin: 7px 0; }}
    .winner-panel {{ border: 2px solid #75ae9f; box-shadow: 0 16px 38px rgba(36, 118, 101, .14); }}
    .winner-heading {{ display: flex; justify-content: space-between; align-items: start; gap: 18px; }}
    .winner-heading h2 {{ font-size: 30px; color: #174f43; }}
    .winner-badge {{ border-radius: 999px; padding: 7px 12px; color: #174f43; background: #dff4ed; border: 1px solid #75ae9f; font-size: 13px; font-weight: 900; white-space: nowrap; }}
    .winner-metrics {{ display: grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap: 10px; margin: 18px 0; }}
    .winner-metrics div {{ display: grid; gap: 3px; padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: var(--accent-soft); }}
    .winner-metrics span {{ color: var(--muted); font-size: 12px; font-weight: 750; }}
    .winner-metrics strong {{ color: #174f43; font-size: 20px; }}
    .control-comparison {{ display: flex; flex-wrap: wrap; gap: 10px 22px; align-items: center; margin: 0 0 16px; padding: 12px 14px; border: 1px solid #aebbc7; border-radius: 8px; background: #f4f7f9; color: #334252; }}
    .control-comparison strong {{ color: var(--ink); }}
    .decision-explanation {{ color: #334252; }}
    .winner-row {{ background: #edf8f5 !important; }}
    .validation-action {{ display: grid; gap: 8px; margin-top: 16px; padding: 14px; border-left: 5px solid var(--accent); background: var(--accent-soft); color: #244d44; }}
    .validation-action code {{ display: block; overflow-x: auto; padding: 10px 12px; border-radius: 6px; background: #17222d; color: #f5faf8; white-space: nowrap; }}
    @media (max-width: 860px) {{
      .campaign-prompt-diff, .benchmark-grid, .diagnostic-grid {{ grid-template-columns: 1fr; }}
      .benchmark-heading {{ display: block; }}
      .benchmark-verification {{ display: inline-block; margin-top: 8px; }}
      .winner-heading {{ display: block; }}
      .winner-badge {{ display: inline-block; margin-top: 8px; }}
      .winner-metrics {{ grid-template-columns: repeat(2, minmax(130px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <div class="brand">RQA Mean Reversion Research Suite</div>
      <nav class="nav">
        <a href="{_escape(hub_href)}">Research Hub</a>
        <a href="#results">What Was Tested</a>
        <a href="#prompts">Prompt Changes</a>
        <a href="#failures">What Stopped</a>
      </nav>
    </div>

    <div class="hero">
      <section class="hero-card now-card {'stopped-card' if status_label in {'stopped', 'error', 'failed'} else ''}">
        <div class="eyebrow">What is happening now?</div>
        <h1>{_escape(campaign_id)}</h1>
        <p class="lead"><strong>{_escape(plain_summary['headline'])}</strong></p>
        <span class="state-pill">Status: {_escape(status_label)}</span>
        <span class="state-pill">Phase: {_escape(phase_label)}</span>
        {error_html}
        <div class="plain-summary">
          <p><strong>Current activity:</strong> {_escape(plain_summary['activity'])}</p>
          <p><strong>What completed:</strong> {_escape(plain_summary['progress'])}</p>
          <p><strong>Why it ended:</strong> {_escape(plain_summary['outcome'])}</p>
          <p><strong>Result:</strong> {_escape(plain_summary['decision'])}</p>
          <p><strong>Last update:</strong> {_escape(plain_summary['updated'])}</p>
        </div>
        <h3>How far down the research tree?</h3>
        <div class="research-path">
          {'<span class="path-arrow">→</span>'.join(f'<span class="path-step">{_escape(part)}</span>' for part in next_steps['path_parts'])}
        </div>
        <h3>What happens next?</h3>
        <ol class="next-actions">
          {''.join(f'<li>{_escape(step)}</li>' for step in next_steps['next_steps'])}
        </ol>
      </section>
    </div>

    {winner_panel}

    <section id="benchmark">
      {control_panel}
    </section>

    <section id="results">
      <h2>What Was Tested and How Far It Got</h2>
      <p>{_escape(race_description)} Return lift is the annualized return difference versus the fresh minimal-change control; Sharpe change is candidate Sharpe minus control Sharpe.</p>
      <div class="table-wrap">{candidate_table}</div>
      <h3>Detailed Stage Diagnostics</h3>
      <p>Expand a candidate to inspect side contributions, consistency, and ranking behavior without widening the main table.</p>
      {stage_diagnostics}
    </section>

    <section id="prompts">
      <h3>Proposed Prompt Changes</h3>
      <p>Open a candidate to see the Researcher's idea, the exact before/after wording, and the complete resulting prompt.</p>
      {prompt_change_cards}
      <h3>Sanitized Rejected Proposals</h3>
      <p>These ideas were proposed but blocked by a static rule before any backtest.</p>
      {rejected_prompt_cards}
    </section>

    <section id="failures">
      <h2>What Interrupted the Campaign</h2>
      <p>Only failures and retries are shown here. The campaign stops after three consecutive provider or runner failures.</p>
      <div class="table-wrap">{failure_table}</div>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(report_html, encoding="utf-8")
    return output_path


def write_future_experiments_page(results_root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    results_root = Path(results_root)
    output_path = results_root / DEFAULT_FUTURE_OUTPUT_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hub_href = relative_href(output_path, results_root / DEFAULT_HUB_OUTPUT_NAME)
    prompt_href = relative_href(output_path, results_root / DEFAULT_OUTPUT_NAME)
    autoresearch_href = relative_href(output_path, results_root / DEFAULT_AUTORESEARCH_REPORT)
    phase2_href = relative_href(output_path, results_root / DEFAULT_PHASE2_OUTPUT_NAME)
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    specialized_table = html_table(specialized_view_rows(), "future-table")
    sequential_table = html_table(sequential_layer_rows(), "future-table")
    regime_table = html_table(regime_allocator_rows(), "future-table")
    short_table = html_table(short_side_objective_rows(), "future-table")
    factorial_table = html_table(factorial_model_rows(), "future-table")

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mean-Reversion Future Experiments</title>
  <style>{shared_suite_css()}</style>
</head>
<body>
  <main>
    <div class="topbar">
      <div class="brand">RQA Mean Reversion Research Suite</div>
      <nav class="nav">
        <a href="{_escape(hub_href)}">Hub</a>
        <a href="{_escape(prompt_href)}">Prompt Dashboard</a>
        <a href="{_escape(autoresearch_href)}">Autoresearch</a>
        <a href="{_escape(phase2_href)}">Out-of-Sample Testing</a>
        <a href="#specialized">Specialized Views</a>
        <a href="#regime">Regime Allocator</a>
        <a href="#shorts">Short Side</a>
        <a href="#models">Models</a>
      </nav>
    </div>

    <div class="hero">
      <section class="hero-card">
        <div class="eyebrow">Future experiments</div>
        <h1>Test Layers, Not More Blends</h1>
        <p class="lead">The next research question is whether each specialized LLM view adds incremental out-of-sample value after the previous layer. This page keeps the planning work separate from the dense evidence pages.</p>
        <div class="callout">Core rule: do not let every view independently score every stock and then average them. Add one layer at a time; if a layer does not add value, delete it.</div>
      </section>
      <section class="hero-card">
        <h2>Locked Lab Bench</h2>
        <ul class="summary-list">
          <li>Primary setup: offset 3, 60/60 -> 30/30.</li>
          <li>Headline accounting: gross fixed 5-trading-day C2C.</li>
          <li>Primary ranker: anonymized technical view unless the experiment says otherwise.</li>
          <li>Generated {_escape(generated_at)}; no LLM calls or backtests are run here.</li>
        </ul>
      </section>
    </div>

    <section id="specialized">
      <h2>Specialized LLM Views</h2>
      <p>Use the same underlying model in isolated jobs. Each view gets a narrow responsibility and is judged by incremental value, not by sounding smart.</p>
      <div class="table-wrap">{specialized_table}</div>
      <div class="callout">Research question: does each specialized view add incremental out-of-sample value after the previous layer?</div>
      <div class="table-wrap">{sequential_table}</div>
    </section>

    <section id="regime">
      <h2>Regime Allocator: LLM vs Deterministic Rule</h2>
      <p>The current evidence says the strategy's incremental edge is more concentrated in stressed environments: high VIX, weak breadth, SPY below its 200-day average, and recent market selloffs. It is weaker in low-volatility grind-up markets.</p>
      <div class="table-wrap">{regime_table}</div>
      <div class="callout">The key metric is not just whether dynamic allocation beats fixed h10. The key metric is how much of the oracle's achievable switching value it captures, and whether it beats a two-line deterministic rule.</div>
      <div class="callout warning">Decision rule: if the LLM allocator cannot beat the simple deterministic regime rule, do not use the LLM for regime allocation.</div>
    </section>

    <section id="shorts">
      <h2>Short-Side Specific Experiments</h2>
      <p>The short side likely needs a different objective. The anonymized model can improve shorts relative to Trail5, but the short leg may still be weak in absolute expectancy. That means the short book might be a hedge or avoidance mechanism rather than a standalone alpha engine.</p>
      <div class="table-wrap">{short_table}</div>
      <div class="callout">Correct short-side question: did the model create a cheaper, safer hedge than random, Trail5, SPY, or sector ETFs? A short sleeve can be useful even if it does not have standalone positive alpha.</div>
    </section>

    <section id="models">
      <h2>Multi-Model Research Should Be Factorial</h2>
      <p>When testing other LLMs, do not run a simple horse race for highest CAGR. Run a factorial design so we can estimate model effect, masking effect, model x masking interaction, ranking agreement, and complementary errors.</p>
      <div class="table-wrap">{factorial_table}</div>
      <div class="callout">A stronger conclusion would be: masking helps every model, or only certain models use ticker identity intelligently, or one model ranks longs while another protects shorts. That is more useful than “Model X won the backtest.”</div>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(report_html, encoding="utf-8")
    return output_path


def write_research_hub(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    campaigns_root: Path = DEFAULT_AUTORESEARCH_CAMPAIGNS_ROOT,
) -> Path:
    results_root = Path(results_root)
    campaigns_root = Path(campaigns_root)
    output_path = results_root / DEFAULT_HUB_OUTPUT_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_href = relative_href(output_path, results_root / DEFAULT_OUTPUT_NAME)
    autoresearch_href = relative_href(output_path, results_root / DEFAULT_AUTORESEARCH_REPORT)
    future_href = relative_href(output_path, results_root / DEFAULT_FUTURE_OUTPUT_NAME)
    phase2_href = relative_href(output_path, results_root / DEFAULT_PHASE2_OUTPUT_NAME)
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    latest_campaign = find_latest_campaign(campaigns_root)
    campaign_nav = ""
    campaign_card = ""
    if latest_campaign is not None:
        latest_status = _read_json_object_if_exists(latest_campaign / "status.json")
        campaign_href = relative_href(
            output_path,
            latest_campaign / DEFAULT_AUTORESEARCH_CAMPAIGN_OUTPUT_NAME,
        )
        campaign_id = latest_status.get("campaign_id") or latest_campaign.name
        campaign_state = latest_status.get("status") or "unknown"
        campaign_phase = latest_status.get("phase") or "unknown"
        campaign_nav = f'<a href="{_escape(campaign_href)}">Latest Campaign</a>'
        campaign_card = f"""
      <a class="hub-card" href="{_escape(campaign_href)}">
        <div class="label">Live autoresearch</div>
        <h2>{_escape(campaign_id)}</h2>
        <p>Read-only supervisor status, candidate lineage, completed gross metrics, budgets, failures, and validation state.</p>
        <p><strong>Status:</strong> {_escape(campaign_state)} · <strong>Phase:</strong> {_escape(campaign_phase)}</p>
        <div class="next-link">Open latest campaign</div>
      </a>
"""

    interaction_root = DEFAULT_THESIS_PACKAGE_ROOT / "prompt_interaction"
    interaction_rows: list[dict[str, Any]] = []
    for arm, decision in (("B", "RETAIN — development candidate"), ("C", "REJECT")):
        summary = _read_json_object_if_exists(
            interaction_root / f"arm_{arm.lower()}" / "diagnostics" / "summary.json"
        )
        if not summary:
            continue
        bootstrap = summary.get("bootstrap", {})
        interaction_rows.append(
            {
                "Arm": arm,
                "Status": "Complete · 251/251",
                "Decision": decision,
                "Annualized lift vs A": fmt_pct(
                    summary.get("annualized_paired_lift_pct"), digits=2, signed=True
                ),
                "95% paired CI": (
                    f"{fmt_pct(bootstrap.get('ci_95_low_pct'), digits=2, signed=True)} to "
                    f"{fmt_pct(bootstrap.get('ci_95_high_pct'), digits=2, signed=True)}"
                ),
                "P(lift > 0)": fmt_pct(
                    100.0 * float(bootstrap.get("probability_lift_gt_zero", float("nan"))),
                    digits=2,
                ),
                "Block stability": fmt_pct(
                    summary.get("chronological_block_stability_pct"), digits=1
                ),
                "Best-5%-trimmed lift": fmt_pct(
                    summary.get("trimmed_5pct_annualized_lift_pct"), digits=2, signed=True
                ),
                "Long lift": fmt_pct(
                    summary.get("long_annualized_lift_pct"), digits=2, signed=True
                ),
                "Short lift": fmt_pct(
                    summary.get("short_annualized_lift_pct"), digits=2, signed=True
                ),
            }
        )
    d_cache_files = sorted((interaction_root / "arm_d" / "cache").glob("*.json"))
    d_signal_count = 0
    d_calls = 0
    d_tokens = 0
    if d_cache_files:
        d_cache = _read_json_object_if_exists(d_cache_files[-1])
        d_signal_count = len(d_cache.get("signals", {}))
        d_metadata = d_cache.get("metadata", {})
        d_calls = int(d_metadata.get("run_calls_used", 0))
        d_tokens = int(d_metadata.get("run_output_tokens_used", 0))
    interaction_rows.append(
        {
            "Arm": "D",
            "Status": f"Incomplete · {d_signal_count}/251",
            "Decision": "PRUNE — interaction futility; not a statistical rejection",
            "Annualized lift vs A": "Not computed",
            "95% paired CI": "Not computed",
            "P(lift > 0)": "Not computed",
            "Block stability": "Not computed",
            "Best-5%-trimmed lift": "Not computed",
            "Long lift": "Not computed",
            "Short lift": "Not computed",
        }
    )
    interaction_table = html_table(interaction_rows, "wide-table")
    thesis_href = relative_href(
        output_path, DEFAULT_THESIS_PACKAGE_ROOT / "research_dashboard.html"
    )

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mean-Reversion Research Hub</title>
  <style>
    {shared_suite_css()}
    .hub-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(260px, 1fr));
      gap: 14px;
      margin-top: 16px;
    }}
    .hub-card {{
      display: block;
      min-height: 210px;
      text-decoration: none;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: var(--shadow);
      padding: 20px;
    }}
    .hub-card:hover {{ border-color: #9fc9bd; transform: translateY(-1px); }}
    .hub-card .label {{
      color: var(--accent);
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: .06em;
      margin-bottom: 10px;
    }}
    .hub-card h2 {{ margin-bottom: 10px; }}
    .next-link {{
      margin-top: 18px;
      color: var(--accent);
      font-weight: 900;
    }}
    .synthesis {{
      display: grid;
      grid-template-columns: repeat(2, minmax(260px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .synthesis article {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 16px;
    }}
    .synthesis strong {{ color: var(--ink); }}
    @media (max-width: 860px) {{
      .hub-grid, .synthesis {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <div class="brand">RQA Mean Reversion Research Suite</div>
      <nav class="nav">
        <a href="{_escape(prompt_href)}">Prompt Dashboard</a>
        <a href="{_escape(autoresearch_href)}">Autoresearch</a>
        <a href="{_escape(future_href)}">Future Experiments</a>
        <a href="{_escape(phase2_href)}">Out-of-Sample Testing</a>
        {campaign_nav}
        <a href="#prompt-interaction">A/B/C/D Update</a>
      </nav>
    </div>

    <div class="hero">
      <section class="hero-card">
        <div class="eyebrow">Research hub</div>
        <h1>Mean-Reversion LLM Research, Split Into Readable Pages</h1>
        <p class="lead">This is the front door. The prompt dashboard explains how the LLM is asked to think, the autoresearch page shows zero-token cache experiments, and the future page lays out the next testable layers.</p>
        <div class="callout">Current synthesis: Trail5 is excellent for building the reversal pool, while the LLM is most interesting as an ordinal selector and risk-aware exception layer inside that pool.</div>
        <p class="muted">Generated {_escape(generated_at)}.</p>
      </section>
    </div>

    <div class="hub-grid">
      <a class="hub-card" href="{_escape(prompt_href)}">
        <div class="label">Prompt/process</div>
        <h2>Prompt Engineering Dashboard</h2>
        <p>Legacy vs V2 prompt map, prompt A/Bs, cache forensics, reasoning peeks, and manager-readable purpose notes for each experiment.</p>
        <div class="next-link">Open prompt dashboard</div>
      </a>
      <a class="hub-card" href="{_escape(autoresearch_href)}">
        <div class="label">Zero-token evidence</div>
        <h2>Autoresearch Cache Sweep</h2>
        <p>Offline replay experiments from existing caches: rank weighting, visible vs anonymized agreement, disagreement tests, placebo, walk-forward, and regime cuts.</p>
        <div class="next-link">Open autoresearch</div>
      </a>
      <a class="hub-card" href="{_escape(future_href)}">
        <div class="label">Next research</div>
        <h2>Future Experiments</h2>
        <p>Sequential specialized-view roadmap: anonymous ranker, visible critic, regime allocator, adversarial veto, short-side roles, and factorial model tests.</p>
        <div class="next-link">Open future plan</div>
      </a>
      <a class="hub-card" href="{_escape(phase2_href)}">
        <div class="label">Out-of-sample evidence</div>
        <h2>2015–2019 Out-of-Sample Testing</h2>
        <p>Tracks development evidence, locked validation, final holdout status, exploratory robustness, and individual experiments including the glossary ablation.</p>
        <div class="next-link">Open out-of-sample testing</div>
      </a>
      {campaign_card}
    </div>

    <section id="prompt-interaction">
      <div class="section-heading">
        <div>
          <div class="section-label">Latest development evidence</div>
          <h2>Shared-Control Prompt Interaction: A / B / C / D</h2>
          <p>This is the latest prompt work folded into the main research record. It is development-only; the blind period remains untouched.</p>
        </div>
      </div>
      <div class="callout"><strong>Correct control design:</strong> A was collected once alongside B, frozen, and reused unchanged for C and D. C and D did not receive fresh controls.</div>
      <div class="table-wrap">{interaction_table}</div>
      <div class="synthesis">
        <article>
          <h3>A — Shared Control</h3>
          <p>Capital-normalized results: <strong>12.66% CAGR</strong>, <strong>1.094 Sharpe</strong>, and <strong>-8.76% maximum drawdown</strong>.</p>
        </article>
        <article>
          <h3>B — De-prescribed Context</h3>
          <p><strong>14.54% CAGR, 1.195 Sharpe, -8.40% drawdown.</strong> The +1.71-point lift came primarily from a <strong>+2.11-point short-side improvement</strong>; longs declined 0.40 points. Retain for its mechanism, not headline CAGR alone. Its CI crosses zero and trimmed lift is negative, so it is not validated.</p>
        </article>
        <article>
          <h3>C — Proportional Separation</h3>
          <p><strong>11.64% CAGR, 1.002 Sharpe, -10.37% drawdown.</strong> It lost 0.90 points versus A, hurt both sides, achieved only 25% chronological-block stability, and is rejected.</p>
        </article>
        <article>
          <h3>D — B × C Interaction</h3>
          <p>D combined B's neutral context with: <strong>“Use the full 0–100 range. Two tickers that differ on a signal must receive distinct scores proportional to the signal's magnitude.”</strong></p>
          <p>It cached <strong>{d_signal_count}/251 dates</strong> through 2011-09-16 using {d_calls} calls and {d_tokens:,} output tokens, then stopped on DeepSeek 402. No D return series was produced, so no CAGR or lift is claimed. Prune as an incomplete interaction-futility decision—not a statistical rejection—and preserve the cache.</p>
        </article>
      </div>
      <div class="callout">Selection remains robustness- and mechanism-aware: short lift, long-side damage, stability, concentration, uncertainty, and weak-state behavior matter alongside CAGR. This preserves the earlier G0_02/G0_01 lesson that useful short-side improvement can matter even when headline CAGR is not the sole winner-selection key.</div>
      <p><a href="{_escape(thesis_href)}">Open the detailed thesis-package dashboard</a></p>
    </section>

    <section>
      <h2>Executive Insights</h2>
      <p>Short version, with the dense evidence kept behind the linked pages.</p>
      <div class="synthesis">
        <article>
          <h3>Visible Model = Classifier</h3>
          <p><strong>Ticker-visible scoring</strong> appears better at making the broad yes/no distinction: roughly deciding which half belongs in the book.</p>
        </article>
        <article>
          <h3>Anonymized Model = Ranker</h3>
          <p><strong>Anonymized scoring</strong> appears better at ordering the highest-conviction ideas once the candidate pool is already technical.</p>
        </article>
        <article>
          <h3>Disagreement Is Directional</h3>
          <p>Generic disagreement penalties are too blunt. The interesting case is when the anonymized model ranks a stock high and ticker-visible does not.</p>
        </article>
        <article>
          <h3>Shorts Need A Different Job</h3>
          <p>The short book may be more useful as avoidance or hedge construction than as pure short alpha. Test that explicitly.</p>
        </article>
      </div>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(report_html, encoding="utf-8")
    return output_path


def write_prompt_dashboard(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_path: Path | None = None,
) -> Path:
    results_root = Path(results_root)
    cache_dir = Path(cache_dir)
    output_path = output_path or (results_root / DEFAULT_OUTPUT_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hub_href = relative_href(output_path, results_root / DEFAULT_HUB_OUTPUT_NAME)
    autoresearch_href = relative_href(output_path, results_root / DEFAULT_AUTORESEARCH_REPORT)
    future_href = relative_href(output_path, results_root / DEFAULT_FUTURE_OUTPUT_NAME)
    phase2_href = relative_href(output_path, results_root / DEFAULT_PHASE2_OUTPUT_NAME)

    run_config = read_json_if_exists(results_root / "run_config.json")
    cache_info = find_v2_cache(cache_dir)
    prompts = prompt_cards()
    concepts = concept_cards()
    result_rows = (
        build_current_v4flash0731_rows()
        + build_v26_minimal_result_rows()
        + build_prompt_ab_decision_rows()
        + build_temperature_decision_rows()
        + build_main_result_rows(results_root)
        + build_v22_decision_rows(DEFAULT_V22_VISIBLE_ROOT)
        + build_v23_decision_rows(DEFAULT_V23_ROOT)
        + build_v24_v25_decision_rows(DEFAULT_V24_ROOT, DEFAULT_V25_ROOT)
        + build_overlay_rows(results_root)
        + build_concentration_rows(results_root)
        + build_long_breadth_rows(DEFAULT_V2_ANON_CONTROL_ROOT)
        + build_side_decay_rows(DEFAULT_V2_ANON_CONTROL_ROOT)
        + build_side_process_hybrid_rows(DEFAULT_V22_VISIBLE_ROOT)
    )
    prompt_diff_table = html_table(build_prompt_diff_rows(), "prompt-diff-table")
    run_taxonomy_table = html_table(build_run_taxonomy_rows(), "run-taxonomy-table")
    prompt_ab_table = html_table(build_prompt_ab_rows(results_root), "prompt-ab-table")
    v26_minimal_table = html_table(build_v26_minimal_result_rows(), "prompt-ab-table wide-table")
    v26_minimal_read = build_v26_minimal_read()
    current_epoch_table = html_table(build_current_v4flash0731_rows(), "prompt-ab-table wide-table")
    current_epoch_read = build_current_v4flash0731_read()
    model_epoch_table = html_table(build_model_epoch_comparison_rows(), "epoch-table")
    model_epoch_read = build_model_epoch_read()
    no_regime_table = html_table(build_no_regime_result_rows(), "prompt-ab-table wide-table")
    no_regime_read = build_no_regime_result_read()
    g0_cross_table = html_table(build_g0_cross_result_rows(), "prompt-ab-table wide-table")
    g0_cross_report = DEFAULT_G0_CROSS_ROOT / "full/llm_reversal_v2_report.html"
    g0_cross_report_href = relative_href(output_path, g0_cross_report)
    glossary_ablation_table = html_table(
        build_phase2_glossary_ablation_rows(),
        "prompt-ab-table wide-table",
    )
    glossary_ablation_read = build_phase2_glossary_ablation_read()
    v24_v25_architecture_table = html_table(
        build_v24_v25_architecture_rows(),
        "wide-table",
    )
    v24_status_read = build_v24_status_read()
    v25_memory_coverage_table = html_table(
        build_v25_memory_coverage_rows(),
        "wide-table",
    )
    v25_status_read = build_v25_status_read()
    v25_paired_results_table = html_table(
        build_v25_paired_result_rows(),
        "wide-table",
    )
    v25_postmortem_read = build_v25_postmortem_read()
    v25_postmortem_correlation_table = html_table(
        build_v25_postmortem_correlation_rows(),
        "wide-table",
    )
    v25_postmortem_tail_table = html_table(
        build_v25_postmortem_tail_rows(),
        "wide-table",
    )
    v25_postmortem_swap_table = html_table(
        build_v25_postmortem_swap_rows(),
        "wide-table",
    )
    v25_postmortem_report = (
        DEFAULT_V25_ROOT
        / "v25_balanced_outcome_memory/memory_postmortem/memory_postmortem_report.html"
    )
    v25_postmortem_href = relative_href(output_path, v25_postmortem_report)
    temperature_table = html_table(build_temperature_experiment_rows(), "temperature-table")
    temperature_control_read = build_temperature_control_read()
    cache_insight_grid = cache_insight_cards_html(build_cache_insight_cards())
    cache_drilldown_html = build_cache_drilldown_html()
    reasoning_peek_html = build_reasoning_peek_html()
    v22_adjustment_html = build_v22_adjustment_html(DEFAULT_V22_VISIBLE_ROOT)
    side_process_hybrid_html = build_side_process_hybrid_html(DEFAULT_V22_VISIBLE_ROOT)
    consensus_side_html = build_consensus_and_side_experiments_html()
    experiment_purpose_table = html_table(build_experiment_purpose_rows(), "purpose-table wide-table")
    long_breadth_html = build_long_breadth_html(DEFAULT_V2_ANON_CONTROL_ROOT)
    side_decay_html = build_side_decay_html(DEFAULT_V2_ANON_CONTROL_ROOT)

    interaction_root = DEFAULT_THESIS_PACKAGE_ROOT / "prompt_interaction"
    prompt_interaction_rows: list[dict[str, Any]] = []
    for arm, decision in (("B", "RETAIN — development candidate"), ("C", "REJECT")):
        summary = _read_json_object_if_exists(
            interaction_root / f"arm_{arm.lower()}" / "diagnostics" / "summary.json"
        )
        if not summary:
            continue
        bootstrap = summary.get("bootstrap", {})
        prompt_interaction_rows.append(
            {
                "Arm": arm,
                "Status": "Complete · 251/251",
                "Decision": decision,
                "Annualized lift vs A": fmt_pct(
                    summary.get("annualized_paired_lift_pct"), digits=2, signed=True
                ),
                "95% paired CI": (
                    f"{fmt_pct(bootstrap.get('ci_95_low_pct'), digits=2, signed=True)} to "
                    f"{fmt_pct(bootstrap.get('ci_95_high_pct'), digits=2, signed=True)}"
                ),
                "P(lift > 0)": fmt_pct(
                    100.0 * float(bootstrap.get("probability_lift_gt_zero", float("nan"))),
                    digits=2,
                ),
                "Block stability": fmt_pct(
                    summary.get("chronological_block_stability_pct"), digits=1
                ),
                "Best-5%-trimmed lift": fmt_pct(
                    summary.get("trimmed_5pct_annualized_lift_pct"), digits=2, signed=True
                ),
                "Long lift": fmt_pct(
                    summary.get("long_annualized_lift_pct"), digits=2, signed=True
                ),
                "Short lift": fmt_pct(
                    summary.get("short_annualized_lift_pct"), digits=2, signed=True
                ),
            }
        )
    d_cache_files = sorted((interaction_root / "arm_d" / "cache").glob("*.json"))
    d_signal_count = d_calls = d_tokens = 0
    if d_cache_files:
        d_cache = _read_json_object_if_exists(d_cache_files[-1])
        d_signal_count = len(d_cache.get("signals", {}))
        d_metadata = d_cache.get("metadata", {})
        d_calls = int(d_metadata.get("run_calls_used", 0))
        d_tokens = int(d_metadata.get("run_output_tokens_used", 0))
    prompt_interaction_rows.append(
        {
            "Arm": "D",
            "Status": f"Incomplete · {d_signal_count}/251",
            "Decision": "PRUNE — interaction futility; not statistically rejected",
            "Annualized lift vs A": "Not computed",
            "95% paired CI": "Not computed",
            "P(lift > 0)": "Not computed",
            "Block stability": "Not computed",
            "Best-5%-trimmed lift": "Not computed",
            "Long lift": "Not computed",
            "Short lift": "Not computed",
        }
    )
    prompt_interaction_table = html_table(prompt_interaction_rows, "wide-table")
    improvement_change_table = html_table(
        [
            {
                "Run / finding": "Fresh B · de-prescribed context",
                "What changed exactly": (
                    "Replaced the prescriptive mean-reversion explanation with: "
                    "‘We are evaluating two pools of stocks that have experienced extreme "
                    "5-day moves, identifying those most likely to partially reverse over "
                    "the next 5 trading days based on exhaustion, positioning, and regime "
                    "signals.’ The calibrated no-manufactured-separation rule stayed fixed."
                ),
                "Observed improvement": "+1.71 pp annualized vs shared A; shorts +2.11 pp; longs -0.40 pp.",
                "Interpretation / caveat": "Retain as a short-side mechanism clue; CI crosses zero and trimmed lift is negative.",
            },
            {
                "Run / finding": "Autoresearch G0_01 · proportional separation",
                "What changed exactly": (
                    "Replaced ‘Use the 0–100 range only when the supplied cross-section "
                    "supports it. Do not manufacture separation’ with ‘Use the full 0–100 "
                    "range. Two tickers that differ on a signal must receive distinct scores "
                    "proportional to the signal’s magnitude.’"
                ),
                "Observed improvement": "Won its original autoresearch campaign.",
                "Interpretation / caveat": "Fresh arm C repeated this exact change and failed; do not treat the original win as durable.",
            },
            {
                "Run / finding": "Autoresearch G0_02 · expected-magnitude scoring",
                "What changed exactly": (
                    "Replaced the fixed 100/50/0 setup labels with a score estimating expected "
                    "next-5-day return magnitude, scaled to a within-pool percentile; explicitly "
                    "removed 50 as a mandatory average anchor."
                ),
                "Observed improvement": "Produced useful short-side lift even though it was not selected on headline CAGR alone.",
                "Interpretation / caveat": "Keep the expected-magnitude/short-selection mechanism in future prompt design and judge both sleeves separately.",
            },
            {
                "Run / finding": "Market-regime ablation",
                "What changed exactly": "Removed only the Market Regime glossary block and the date-level regime value.",
                "Observed improvement": "The ablation hurt; therefore the improvement belongs to retaining regime context.",
                "Interpretation / caveat": "This isolates an input-context contribution, not a general endorsement of longer prompts.",
            },
            {
                "Run / finding": "Historical V2 ticker-visible",
                "What changed exactly": "Used the same V2 score prompt but exposed real ticker symbols instead of opaque SEC_#### identifiers.",
                "Observed improvement": "Higher historical headline CAGR than Trail5 and the anonymized run.",
                "Interpretation / caveat": "Historical model epoch; ticker visibility is confounded with learned company identity and is not a clean wording change.",
            },
            {
                "Run / finding": "V2.3 rank-only",
                "What changed exactly": "Changed the output job from independent 0–100 scores to a complete side-relative ordering; Python selected the top 30.",
                "Observed improvement": "Useful ranking result, but weaker than the original V2 score prompt.",
                "Interpretation / caveat": "Evidence that ordinal ranking is viable, not evidence that rank-only replaced V2.",
            },
            {
                "Run / finding": "Current-model temperature 1",
                "What changed exactly": "No prompt wording changed; only the sampling temperature differed within the current glossary-backed V2 setup.",
                "Observed improvement": "Strongest positive lift in the current temperature sweep so far.",
                "Interpretation / caveat": "Attribute this to sampling configuration and run variance, not prompt text.",
            },
        ],
        "wide-table",
    )
    interaction_thesis_href = relative_href(
        output_path, DEFAULT_THESIS_PACKAGE_ROOT / "research_dashboard.html"
    )

    prompt_grid = "\n".join(prompt_card_html(card) for card in prompts)
    v2_question_grid = v2_model_question_grid_html()
    concept_grid = "\n".join(concept_card_html(card) for card in concepts)
    result_table = html_table(result_rows, "result-table")
    gameplan_table = html_table(build_experiment_gameplan_rows(), "gameplan-table")

    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    prompt_source = run_config.get("prompt_source", "mean_reversion_llm_v1.V1_SYSTEM_PROMPT")
    prompt_style = run_config.get("prompt_style", v2.V2_PROMPT_STYLE)
    pool_sizes = ", ".join(str(x) for x in run_config.get("pool_sizes", [60]))
    book_sizes = ", ".join(str(x) for x in run_config.get("book_sizes", [30]))
    anchors = ", ".join(str(x) for x in run_config.get("anchor_offsets", [3]))
    avg_tokens = fmt_num(cache_info.get("avg_total_tokens"), 0)

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mean-Reversion Prompt Engineering Dashboard</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --panel-2: #fbfcfd;
      --ink: #16212c;
      --muted: #687589;
      --line: #d8e0e8;
      --head: #edf2f6;
      --accent: #247665;
      --accent-2: #b5812d;
      --accent-soft: #edf8f5;
      --warn-soft: #fff7e8;
      --bad-soft: #fff1f1;
      --good-soft: #e7f6ef;
      --info-soft: #eef5fb;
      --shadow: 0 16px 40px rgba(22, 33, 44, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0 0 auto 0;
      height: 270px;
      background: #eaf0f3;
      border-bottom: 1px solid #dde5ed;
      z-index: -1;
    }}
    main {{ max-width: 1360px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(30px, 4vw, 46px); line-height: 1.03; letter-spacing: 0; }}
    h2 {{ margin: 0 0 10px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; letter-spacing: 0; }}
    p {{ margin: 0 0 10px; color: var(--muted); }}
    a {{ color: inherit; }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 18px;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
      color: #344255;
      font-weight: 800;
      letter-spacing: .02em;
    }}
    .brand-mark {{
      width: 28px;
      height: 28px;
      border-radius: 8px;
      background: #247665;
      box-shadow: inset 0 -8px 0 rgba(0, 0, 0, .08);
    }}
    .nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}
    .nav a {{
      text-decoration: none;
      color: #425166;
      background: rgba(255, 255, 255, .7);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 11px;
      font-size: 13px;
      font-weight: 700;
    }}
    header.hero-header {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(360px, .85fr);
      gap: 22px;
      align-items: stretch;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--accent);
      background: var(--accent-soft);
      border: 1px solid #cfe7df;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: .06em;
      margin-bottom: 14px;
    }}
    .lead {{
      max-width: 820px;
      font-size: 16px;
      color: #536176;
    }}
    .header-side {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: var(--panel-2);
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      margin: 18px 0;
      box-shadow: 0 4px 18px rgba(22, 33, 44, 0.045);
      scroll-margin-top: 18px;
    }}
    .section-heading {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .section-heading p {{ max-width: 840px; }}
    .section-label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: .06em;
      text-transform: uppercase;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }}
    .callout {{
      background: var(--accent-soft);
      border-left: 4px solid var(--accent);
      border-radius: 0 8px 8px 0;
      padding: 13px 15px;
      color: #24483f;
      margin-top: 12px;
    }}
    .decision-banner {{
      background: var(--warn-soft);
      border: 1px solid #ecd6a9;
      border-left: 6px solid var(--accent-2);
      border-radius: 8px;
      padding: 16px 18px;
      color: #5e4317;
      font-size: 16px;
      margin: 14px 0;
    }}
    .decision-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .decision-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 15px;
      background: var(--panel-2);
    }}
    .decision-card h3 {{ margin-bottom: 7px; }}
    .research-appendix {{ margin: 18px 0; }}
    .research-appendix > summary {{
      list-style: none;
      background: #263746;
      color: white;
      border-radius: 8px;
      padding: 17px 20px;
      font-size: 17px;
      box-shadow: var(--shadow);
    }}
    .research-appendix > summary::-webkit-details-marker {{ display: none; }}
    .research-appendix > summary::after {{ content: " +"; float: right; }}
    .research-appendix[open] > summary::after {{ content: " −"; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 12px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--panel);
      min-height: 112px;
      box-shadow: 0 2px 10px rgba(22, 33, 44, .04);
    }}
    .metric-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .metric-value {{ font-size: 22px; font-weight: 800; margin-top: 5px; }}
    .metric-note {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    .question-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(260px, 1fr));
      gap: 10px;
    }}
    .question-card {{
      display: flex;
      align-items: center;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 12px;
      color: #273747;
      font-weight: 700;
    }}
    .question-num {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      flex: 0 0 28px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 13px;
      font-weight: 900;
    }}
    .prompt-grid, .concept-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(300px, 1fr));
      gap: 14px;
    }}
    .insight-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(260px, 1fr));
      gap: 12px;
    }}
    .reasoning-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 14px;
    }}
    .prompt-card, .concept-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 16px;
      position: relative;
      overflow: hidden;
    }}
    .prompt-card::before, .concept-card::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 4px;
      background: var(--accent);
    }}
    .prompt-v2::before {{ background: var(--accent); }}
    .prompt-v2-1::before {{ background: #9f2727; }}
    .prompt-legacy::before {{ background: #59687a; }}
    .prompt-v1-legacy-bridge::before {{ background: var(--accent-2); }}
    .insight-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 15px;
      min-height: 170px;
    }}
    .reasoning-panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 15px;
      min-width: 0;
    }}
    .reasoning-panel-head {{
      display: flex;
      gap: 12px;
      justify-content: space-between;
      align-items: flex-start;
    }}
    .insight-card h3 {{ color: #213041; margin-bottom: 8px; }}
    .artifact-path {{
      margin-top: 12px;
      color: #536176;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      overflow-wrap: anywhere;
      background: var(--info-soft);
      border: 1px solid #d7e6f2;
      border-radius: 6px;
      padding: 7px;
    }}
    .drilldown-block {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 12px;
      margin-top: 12px;
    }}
    .drilldown-block summary {{
      cursor: pointer;
      color: #213041;
      font-size: 15px;
      font-weight: 900;
    }}
    .drilldown-block p {{ margin: 8px 0 10px; }}
    .card-kicker {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 10px;
    }}
    .card-kicker span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      background: white;
      color: #536176;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 800;
    }}
    dl {{
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 7px 12px;
      margin: 12px 0 0;
      font-size: 13px;
    }}
    dt {{ color: var(--muted); font-weight: 700; }}
    dd {{ margin: 0; }}
    details {{ margin-top: 12px; }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 800; }}
    pre {{
      margin: 10px 0 0;
      white-space: pre-wrap;
      font-size: 12px;
      line-height: 1.5;
      background: #f2f5f8;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      max-height: 320px;
      overflow: auto;
    }}
    .prompt-card details pre {{
      max-height: none;
      overflow: visible;
    }}
    .reasoning-pre {{
      max-height: 680px;
      background: #0f1720;
      color: #e9eef5;
      border-color: #263445;
      font-size: 12px;
      line-height: 1.55;
    }}
    .concept-topline {{
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
    }}
    .pill, .tag-cell {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }}
    .priority-high, .tag-winner {{ background: var(--good-soft); color: #0e6245; }}
    .priority-medium, .tag-useful-but-weaker {{ background: var(--warn-soft); color: #835400; }}
    .tag-diagnostic-only, .tag-paired-control {{ background: #edf1f6; color: #405064; }}
    .tag-do-not-prioritize, .tag-not-found {{ background: var(--bad-soft); color: #9f2727; }}
    .card-footer {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
      margin-top: 10px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; background: white; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 11px; text-align: left; vertical-align: top; }}
    th {{
      position: sticky;
      top: 0;
      background: var(--head);
      color: #273747;
      font-weight: 900;
      white-space: nowrap;
      z-index: 1;
    }}
    tr:nth-child(even) td {{ background: #fafbfc; }}
    tr:hover td {{ background: #f5faf8; }}
    .row-winner td {{ background: #f4fbf8; }}
    .row-paired-control td {{ color: #405064; }}
    .result-table td:nth-child(1), .result-table td:nth-child(6) {{ max-width: 320px; }}
    .prompt-diff-table td:nth-child(2), .prompt-diff-table td:nth-child(3), .prompt-diff-table td:nth-child(4) {{ max-width: 360px; }}
    .prompt-ab-table td:nth-child(2) {{ max-width: 330px; }}
    .run-taxonomy-table td:nth-child(5) {{ max-width: 360px; }}
    .wide-table td {{ min-width: 150px; }}
    .drill-table td:first-child {{ font-weight: 700; }}
    .gameplan-table td:nth-child(1) {{ font-weight: 900; color: var(--accent); }}
    td:nth-last-child(n+4):nth-child(n+7), th:nth-last-child(n+4):nth-child(n+7) {{ text-align: right; }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 920px) {{
      main {{ padding: 18px; }}
      .topbar {{ align-items: flex-start; flex-direction: column; }}
      header.hero-header, .hero, .prompt-grid, .concept-grid, .insight-grid, .reasoning-grid, .metric-grid, .question-grid {{ grid-template-columns: 1fr; }}
      .section-heading {{ flex-direction: column; }}
      .decision-grid {{ grid-template-columns: 1fr; }}
      dl {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <div class="brand"><span class="brand-mark"></span><span>RQA Mean Reversion</span></div>
      <nav class="nav">
        <a href="{_escape(hub_href)}">Hub</a>
        <a href="{_escape(autoresearch_href)}">Autoresearch</a>
        <a href="{_escape(phase2_href)}">Out-of-Sample Testing</a>
        <a href="{_escape(future_href)}">Future Experiments</a>
        <a href="#manager-read">Manager Summary</a>
        <a href="#shared-control-abcd">A/B/C/D</a>
        <a href="#appendix">Research Appendix</a>
      </nav>
    </div>

    <header class="hero-header" id="summary">
      <div>
        <div class="eyebrow">Prompt engineering dashboard</div>
        <h1>Mean-Reversion Prompt Research</h1>
        <p class="lead">A manager-facing summary of the model-epoch change, the current evidence, and what the historical prompt experiments mean.</p>
        <p class="muted">Generated {_escape(generated_at)} from local prompt constants and completed result artifacts. No LLM calls or backtests are run by this dashboard.</p>
      </div>
      <div class="header-side">
        <h2>Bottom Line</h2>
        <p><strong>A backend model change occurred July 31.</strong></p>
        <p>The strong 16%+ minimal-change results belong to the prior V4-Flash Preview epoch.</p>
        <p>Post-switch reruns were weaker and more variable. Current research must be rebased on V4-Flash-0731.</p>
      </div>
    </header>

    <section class="section hero" id="manager-read">
      <div class="section-heading">
        <div>
          <div class="section-label">Manager Summary</div>
          <h2>The July 31 Model Change Reset The Baseline</h2>
        </div>
      </div>
      <div>
        <div class="decision-banner"><strong>{_escape(model_epoch_read)}</strong></div>
        <div class="table-wrap">{model_epoch_table}</div>
        <div class="decision-grid">
          <div class="decision-card"><h3>What changed</h3><p>DeepSeek changed the backend serving the same <code>deepseek-v4-flash</code> name on July 31, 2026. The API label stayed the same; the experimental model did not.</p></div>
          <div class="decision-card"><h3>What the reruns show</h3><p>{_escape(current_epoch_read)}</p></div>
          <div class="decision-card"><h3>Decision</h3><p>Freeze Preview results as historical evidence. Use repeated V4-Flash-0731 controls for all new prompt comparisons, and do not combine the two epochs.</p></div>
        </div>
        <details>
          <summary>See all completed V4-Flash-0731 reruns</summary>
          <p>These are current-model runs only. The four anonymous V2 rows are independent repeats and make the run-to-run variance visible. The fourth run is stored in a misleadingly named <code>ticker_visible</code> folder, but its frozen config and cache both confirm anonymization.</p>
          <div class="table-wrap">{current_epoch_table}</div>
        </details>
        <div class="section-heading" style="margin-top: 22px;">
          <div>
            <div class="section-label">Latest isolated experiment</div>
            <h2>Removing Market-Regime Information Hurt This Run</h2>
            <p>Current V4-Flash-0731 experiment using the same 251 dates and frozen harness. The treatment removes only the Market Regime glossary block and date-level regime value.</p>
          </div>
        </div>
        <div class="callout"><strong>{_escape(no_regime_read)}</strong></div>
        <div class="table-wrap">{no_regime_table}</div>
        <div class="section-heading" style="margin-top: 22px;">
          <div>
            <div class="section-label">Completed prompt cross</div>
            <h2>G0_01 × G0_05 Full Development Result</h2>
            <p>The G0_01 proportional-separation winner was crossed with G0_05's forced top-feature diagnostic. All 251 dates use the frozen anonymous offset-3 harness; the minimal-change and Trail5 rows are reused artifacts.</p>
          </div>
        </div>
        <div class="callout"><strong>Decision:</strong> reject the cross. It failed the full gate and underperformed both reused controls; retain G0_01 as the development winner. <a href="{_escape(g0_cross_report_href)}">Open the full crossed-run report.</a></div>
        <div class="table-wrap">{g0_cross_table}</div>
        <div class="callout"><strong>Measurement:</strong> all headline figures here use gross fixed five-trading-day C2C returns and 50/50 capital-normalized CAGR, Sharpe, and drawdown.</div>
        <div id="shared-control-abcd" class="section-heading" style="margin-top: 22px;">
          <div>
            <div class="section-label">Latest prompt interaction</div>
            <h2>Shared-Control A / B / C / D Experiment</h2>
            <p>Development-only evidence using the same 251-date manifest, anonymous tickers, offset-3 60/60 → 30/30 construction, temperature 1, and unchanged accounting.</p>
          </div>
        </div>
        <div class="callout"><strong>Control correction:</strong> A was collected once alongside B, frozen, and reused unchanged for C and D. The arms did not receive new controls each time.</div>
        <div class="table-wrap">{prompt_interaction_table}</div>
        <div class="decision-grid">
          <div class="decision-card">
            <h3>A — Shared Fresh Control</h3>
            <p><strong>12.66% CAGR · 1.094 Sharpe · -8.76% max drawdown.</strong> This exact A artifact is the comparison anchor for B, C, and D.</p>
          </div>
          <div class="decision-card">
            <h3>B — De-prescribed Context</h3>
            <p><strong>Exact prompt change:</strong> the old explanatory context was replaced with: “We are evaluating two pools of stocks that have experienced extreme 5-day moves, identifying those most likely to partially reverse over the next 5 trading days based on exhaustion, positioning, and regime signals.” The rule <em>“Use the 0–100 range only when the supplied cross-section supports it. Do not manufacture separation”</em> stayed unchanged.</p>
            <p><strong>14.54% CAGR · 1.195 Sharpe · -8.40% drawdown.</strong> Lift versus A was +1.71 points: longs -0.40, shorts +2.11. Paired hit rate was 52.59%, block stability 75%, P(lift &gt; 0) 84.64%, and the 95% CI was -1.49 to +4.98 points.</p>
            <p><strong>Decision:</strong> retain as a development candidate because it materially improved the short sleeve—not because CAGR alone won. The best-5%-trimmed lift was -1.60 points, so it is promising but concentrated and unvalidated.</p>
          </div>
          <div class="decision-card">
            <h3>C — Proportional Score Separation</h3>
            <p><strong>11.64% CAGR · 1.002 Sharpe · -10.37% drawdown.</strong> Lift versus A was -0.90 points: longs -0.58, shorts -0.31. Stability was 25%, P(lift &gt; 0) 29.96%, CI -4.24 to +2.28, and trimmed lift -3.88 points.</p>
            <p><strong>Decision:</strong> reject. Forced proportional separation damaged both sides and worsened drawdown.</p>
          </div>
          <div class="decision-card">
            <h3>D — B × C Interaction</h3>
            <p>D combined B's neutral context with C's instruction: <strong>“Use the full 0–100 range. Two tickers that differ on a signal must receive distinct scores proportional to the signal's magnitude.”</strong> The feature glossary, task, strict JSON contract, and anonymous tickers stayed fixed.</p>
            <p>D cached <strong>{d_signal_count}/251 dates</strong> through 2011-09-16 using {d_calls} calls and {d_tokens:,} output tokens, then stopped on DeepSeek 402. No D return series exists, so no CAGR, lift, or statistical rejection is claimed.</p>
            <p><strong>Decision:</strong> prune for incomplete interaction futility: D's remaining proposition was that B could rescue C's already harmful rule. Preserve the cache.</p>
          </div>
        </div>
        <div class="callout"><strong>How candidates are judged:</strong> CAGR is not the sole selection key. Short lift, long-side damage, chronological stability, concentration, paired uncertainty, drawdown, and weak-state behavior all matter. This explicitly carries forward the G0_02/G0_01 lesson and the autoresearch finding that a prompt can remain useful when it materially improves the short sleeve.</div>
        <div class="callout"><strong>Validation status:</strong> B is retained only for further development; C is rejected; D is pruned incomplete. No 2015+ blind artifact was read or touched. <a href="{_escape(interaction_thesis_href)}">Open the detailed thesis-package record.</a></div>
        <div class="section-heading" style="margin-top: 22px;">
          <div>
            <div class="section-label">Change attribution audit</div>
            <h2>When A Run Improved, What Actually Changed?</h2>
            <p>This table makes the treatment explicit and separates prompt wording, ticker visibility, input ablations, output contracts, and sampling parameters. A performance change is not automatically attributed to prompt text.</p>
          </div>
        </div>
        <div class="table-wrap">{improvement_change_table}</div>
        <div class="section-heading" style="margin-top: 22px;"><div><div class="section-label">New experiments</div><h2>Consensus And Side-Specialized Scoring</h2></div></div>
        {consensus_side_html}
      </div>
      <div class="metric-grid">
        <div class="metric"><div class="metric-label">Frozen period</div><div class="metric-value">2009–2014</div><div class="metric-note">251 offset-3 dates</div></div>
        <div class="metric"><div class="metric-label">Construction</div><div class="metric-value">60/60 → 30/30</div><div class="metric-note">equal weight, anonymous unless noted</div></div>
        <div class="metric"><div class="metric-label">Current model</div><div class="metric-value">V4-Flash-0731</div><div class="metric-note">temperature 1.0; thinking disabled</div></div>
        <div class="metric"><div class="metric-label">Historical model</div><div class="metric-value">V4-Flash Preview</div><div class="metric-note">reference only; not directly comparable</div></div>
      </div>
    </section>

    <details class="research-appendix" id="appendix">
      <summary>Open detailed research appendix — prompts, experiments, diagnostics, and full results</summary>

    <section class="section" id="prompts">
      <div class="section-heading">
        <div>
          <div class="section-label">Prompt Map</div>
          <h2>Legacy Through V2.6 Prompt / Process Map</h2>
          <p>These cards intentionally separate the LLM-facing contract from the backtest/accounting harness.</p>
        </div>
      </div>
      <div class="prompt-grid">{prompt_grid}</div>
    </section>

    <section class="section" id="prompt-ab">
      <div class="section-heading">
        <div>
          <div class="section-label">Prompt A/B</div>
          <h2>Run Taxonomy: Tickers, Labels, And Prompt Variants</h2>
          <p>This separates ticker visibility from prompt changes. The confusing part is that “anonymized,” “structured labels,” and “V2.1” are three different axes, not one thing.</p>
        </div>
      </div>
      <div class="callout"><strong>Epoch warning:</strong> unless a row is explicitly labeled V4-Flash-0731, treat DeepSeek V4-Flash results in this appendix as historical Preview-era evidence. Do not rank them directly against current-model runs.</div>
      <div class="table-wrap">{run_taxonomy_table}</div>
      <div class="section-heading" style="margin-top: 18px;">
        <div>
          <h2>V2 vs V2.1: What Actually Changed</h2>
          <p>V2.1 used the same harness, anonymized tickers, offset 3, 60/60 -> 30/30 book, score-only schema, and gross fixed 5-day C2C replay. The performance delta came from prompt-driven score behavior.</p>
        </div>
      </div>
      <div class="table-wrap">{prompt_diff_table}</div>
      <div class="callout">Working interpretation: V2.1 was not a light reminder to inspect more columns. It became a hard instruction to find incremental evidence beyond Trail5/RSI5, which pushed the model away from the core reversal exposure.</div>
      <div class="section-heading" style="margin-top: 18px;">
        <div>
          <h2>Prompt Experiment Performance</h2>
          <p>Same lab bench where available: offset 3, 60/60 -> 30/30, gross fixed 5-trading-day C2C. The table shows both raw spread CAGR and 50/50 capital-normalized CAGR.</p>
        </div>
      </div>
      <div class="table-wrap">{prompt_ab_table}</div>
      <div class="section-heading" style="margin-top: 18px;">
        <div>
          <h2>Historical Preview-Epoch V2.6 And Minimal-Change Results</h2>
          <p>Historical 2009-2014 development runs from the prior model epoch. They remain useful evidence about Preview behavior, but they are not directly comparable with V4-Flash-0731. The minimal treatment changes one range-separation instruction; V2.6 changes the full prompt bundle.</p>
        </div>
      </div>
      <div class="callout">{_escape(v26_minimal_read)}</div>
      <div class="table-wrap">{v26_minimal_table}</div>
      <div class="section-heading" style="margin-top: 18px;">
        <div>
          <h2>2015-2019 Glossary Ablation</h2>
          <p>Both prompt variants use the same dates, candidate pools, accounting, and byte-identical Trail5 artifacts. This separates prompt wording from portfolio construction.</p>
        </div>
      </div>
      <div class="callout">{_escape(glossary_ablation_read)}</div>
      <div class="table-wrap">{glossary_ablation_table}</div>
      <div class="section-heading" style="margin-top: 18px;">
        <div>
          <h2>Temperature Check</h2>
          <p>DeepSeek Chat temperature can change score stability and ranking spread. Temp 0 and temp 0.6 now form a clean current-prompt control; the other completed temperatures extend the sensitivity sweep.</p>
        </div>
      </div>
      <div class="callout">{_escape(temperature_control_read)}</div>
      <div class="table-wrap">{temperature_table}</div>
      <div class="section-heading" style="margin-top: 18px;">
        <div>
          <h2>V2.2 Adjustment Distribution And Outcomes</h2>
          <p>Why the contextual-adjustment run underperformed: the model's -3..+3 buckets did not consistently map to better side-adjusted P&amp;L, especially on shorts.</p>
        </div>
      </div>
      <div class="callout">Read: positive Side P&amp;L is good for the intended side. V2.2 had some useful long-side separation, but positive short adjustments often selected stocks that rose, which hurts shorts.</div>
      {v22_adjustment_html}
      <div class="section-heading" style="margin-top: 18px;">
        <div>
          <h2>Side-Specific Process Hybrids</h2>
          <p>This tests whether longs and shorts should use different selection processes, using existing ticker-visible V2 and V2.2 caches only.</p>
        </div>
      </div>
      <div class="callout">Read: the original V2 long ranking remains the best long process. The only small improvement came from using the V2.2 adjustment as a short-side overlay on top of V2 scores, but the lift is small enough to treat as a prompt-design clue, not a new winner yet.</div>
      {side_process_hybrid_html}
    </section>

    <section class="section" id="v24-v25">
      <div class="section-heading">
        <div>
          <div class="section-label">New Prompt Treatments</div>
          <h2>V2.4 Raw Context And V2.5 Balanced Memory</h2>
          <p>These are isolated prompt/process treatments. Both paid development runs are complete; V2.5 is shown against a fresh same-model V2 control, Trail5, and a deterministic HistRet-only diagnostic.</p>
        </div>
      </div>
      <div class="table-wrap">{v24_v25_architecture_table}</div>
      <div class="callout"><strong>V2.4 result:</strong> {_escape(v24_status_read)}</div>
      <div class="section-heading" style="margin-top: 18px;">
        <div>
          <h2>V2.5 Point-In-Time Memory Audit</h2>
          <p>V2.5 keeps the glossary-backed V2 score definition and adds only three fields: HistN, HistWin, and HistRet. Retrieval uses prior same-side candidates, excludes same-ticker history, caps representation from any one rebalance date, and never uses an outcome unavailable on the signal date.</p>
        </div>
      </div>
      <div class="table-wrap">{v25_memory_coverage_table}</div>
      <div class="callout"><strong>Status:</strong> {_escape(v25_status_read)}</div>
      <div class="section-heading" style="margin-top: 18px;">
        <div>
          <h2>V2.5 Paired Results</h2>
          <p>Full includes all 251 dates. Memory-ready excludes the three warm-up dates without adequate analog coverage; fully populated requires every candidate to have the complete 20-neighbor memory. Positive lift means the row beat its paired benchmark.</p>
        </div>
      </div>
      <div class="table-wrap">{v25_paired_results_table}</div>
      <div class="section-heading" style="margin-top: 18px;">
        <div>
          <h2>Why V2.5 Failed: Zero-Token Memory Postmortem</h2>
          <p>The diagnostic asks whether HistWin/HistRet actually predicted future side-adjusted returns, whether V2.5 reacted to those fields, and whether the names it added beat the fresh-V2 names it dropped. Statistical intervals resample whole rebalance dates.</p>
        </div>
      </div>
      <div class="callout"><strong>Conclusion:</strong> {_escape(v25_postmortem_read)} <a href="{_escape(v25_postmortem_href)}">Open the full postmortem</a>.</div>
      <h3>Signal Validity</h3>
      <div class="table-wrap">{v25_postmortem_correlation_table}</div>
      <h3>Extreme Negative Memory</h3>
      <div class="table-wrap">{v25_postmortem_tail_table}</div>
      <h3>Names Added Versus Dropped</h3>
      <div class="table-wrap">{v25_postmortem_swap_table}</div>
    </section>

    <section class="section" id="cache-insights">
      <div class="section-heading">
        <div>
          <div class="section-label">Cache Forensics</div>
          <h2>What We Can Learn From The Cache</h2>
          <p>The cache and replay files let us inspect candidate-level score changes, selected-name overlap, feature emphasis, structured labels, and week-by-week damage without rerunning the LLM.</p>
        </div>
      </div>
      <div class="insight-grid">{cache_insight_grid}</div>
      <div class="callout">Important: structured labels are a diagnostic artifact, not hidden chain-of-thought. They are useful for feature-emphasis debugging and prompt iteration, not proof of the model's internal reasoning.</div>
    </section>

    <section class="section" id="cache-drilldown">
      <div class="section-heading">
        <div>
          <div class="section-label">Cache Drill-Down</div>
          <h2>Label, Feature, Score, And Portfolio Diagnostics</h2>
          <p>These tables expose the analysis behind the cache-insight cards: which labels/features worked, how each prompt loaded on inputs, which names were promoted or dropped, and when V2.1 diverged most from V2.</p>
        </div>
      </div>
      {cache_drilldown_html}
    </section>

    <section class="section" id="reasoning-peek">
      <div class="section-heading">
        <div>
          <div class="section-label">DeepSeek Debug Read</div>
          <h2>One-Day Reasoning Content Peek</h2>
          <p>These are raw DeepSeek <code>reasoning_content</code> dumps from one-date V2 score calls, shown before and after adding concise feature definitions to the prompt. Use them as qualitative diagnostics for ticker identity behavior and feature-meaning confusion, not as proof of the true causal reason for a score.</p>
        </div>
      </div>
      <div class="callout">Read this as a debugging lens: first compare visible vs masked ticker behavior, then compare before vs after definitions to see whether the model stops guessing what MRL, MRS, MRSig, AutoC, and regime mean.</div>
      <div class="reasoning-grid">{reasoning_peek_html}</div>
    </section>

    <section class="section" id="experiment-purpose">
      <div class="section-heading">
        <div>
          <div class="section-label">How To Read The Experiments</div>
          <h2>Experiment Purpose Guide</h2>
          <p>Each experiment is useful even when it does not win. This table says what question the test is designed to answer and what kind of insight to extract from it.</p>
        </div>
      </div>
      <div class="table-wrap">{experiment_purpose_table}</div>
    </section>

    <section class="section" id="long-breadth">
      <div class="section-heading">
        <div>
          <div class="section-label">Zero-Cost Replay</div>
          <h2>Broad Long Core + LLM Tilt</h2>
          <p>This tests whether ticker-masked V2 was too aggressive on the long side by keeping offset 3, the same 60/60 pool, same fixed 5-day C2C accounting, and V2 top-30 shorts unchanged. No LLM calls are used.</p>
        </div>
      </div>
      <div class="callout">Read: this section uses the anonymized V2 control cache, not the historical ticker-visible cache. In the masked run, linear rank-decay across all 60 longs has the best 50/50 CAGR and Sharpe, but the date-block bootstrap is positive rather than decisive. Treat it as a promising wrapper to validate, not a settled replacement.</div>
      {long_breadth_html}
    </section>

    <section class="section" id="side-decay">
      <div class="section-heading">
        <div>
          <div class="section-label">Zero-Cost Replay</div>
          <h2>Long Decay vs Short Decay</h2>
          <p>This tests the ticker-masked V2 cache with decay applied to longs only, shorts only, or both. The short rank curve is measured as short P&L, so positive means the short bucket made money.</p>
        </div>
      </div>
      <div class="callout">Read: linear decay helps when applied to masked V2 longs, but short decay reduces CAGR. Short ranks 31-60 generally have worse short P&L than the top-30 short sleeve, so the hard short cutoff looks more justified than the hard long cutoff.</div>
      {side_decay_html}
    </section>

    <section class="section" id="v2-inputs">
      <div class="section-heading">
        <div>
          <div class="section-label">V2 Inputs</div>
          <h2>What The V2 LLM Is Asked To Judge</h2>
          <p>Each score is based on the same feature row per candidate plus the market regime label. The model is effectively asked to synthesize these checks into one side-relative 0-100 mean-reversion setup score.</p>
        </div>
      </div>
      <div class="question-grid">{v2_question_grid}</div>
    </section>

    <section class="section" id="concepts">
      <div class="section-heading">
        <div>
          <div class="section-label">Iteration Concepts</div>
          <h2>Prompt Engineering Concept Board</h2>
          <p>Each concept is framed as a testable prompt or process lever, not a conclusion.</p>
        </div>
      </div>
      <div class="concept-grid">{concept_grid}</div>
    </section>

    <section class="section" id="gameplan">
      <div class="section-heading">
        <div>
          <div class="section-label">Next Tests</div>
          <h2>Experiment Gameplan</h2>
          <p>All prompt A/Bs should keep the same locked harness unless the experiment explicitly says otherwise: offset 3, 60/60 -> 30/30, gross fixed 5-day C2C, paired Trail5 baseline.</p>
        </div>
      </div>
      <div class="table-wrap">{gameplan_table}</div>
    </section>

    <section class="section" id="results">
      <div class="section-heading">
        <div>
          <div class="section-label">Evidence</div>
          <h2>Results / Decision Table</h2>
          <p>One table for what we have learned so far: legacy, V1, V2, the 2015-2019 glossary ablation, V2.4/V2.5 status, overlays, concentration, long-breadth, and side-decay outputs where available.</p>
        </div>
      </div>
      <div class="table-wrap">{result_table}</div>
    </section>

    <section class="section">
      <h2>Run Context</h2>
      <p>Detected V2 run shape: pool sizes {_escape(pool_sizes)}, book sizes {_escape(book_sizes)}, anchors {_escape(anchors)}.</p>
      <p>Detected cache file: {_escape(cache_info.get("path") or "not found")}.</p>
    </section>
    </details>
  </main>
</body>
</html>
"""
    output_path.write_text(report_html, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--campaigns-root",
        type=Path,
        default=DEFAULT_AUTORESEARCH_CAMPAIGNS_ROOT,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_output = write_prompt_dashboard(args.results_root, args.cache_dir, args.output)
    future_output = write_future_experiments_page(args.results_root)
    phase2_output = phase2.write_dashboard(
        suite_root=args.results_root,
        development_panel=args.results_root / "autoresearch_v2_cache_sweep/candidate_panel.parquet",
        validation_root=phase2.DEFAULT_VALIDATION_ROOT,
    )
    latest_campaign = find_latest_campaign(args.campaigns_root)
    campaign_output = (
        write_autoresearch_campaign_dashboard(
            latest_campaign,
            hub_path=args.results_root / DEFAULT_HUB_OUTPUT_NAME,
        )
        if latest_campaign is not None
        else None
    )
    hub_output = write_research_hub(args.results_root, args.campaigns_root)
    print(f"Wrote research hub: {hub_output}")
    print(f"Wrote prompt engineering dashboard: {prompt_output}")
    print(f"Wrote future experiments dashboard: {future_output}")
    print(f"Wrote out-of-sample testing dashboard: {phase2_output}")
    if campaign_output is not None:
        print(f"Wrote latest autoresearch campaign dashboard: {campaign_output}")


if __name__ == "__main__":
    main()
