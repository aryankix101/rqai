"""Frozen prompts and offline evaluation for the three sleeve-splice tests.

The paid scorer calls are supervised by
``run_mean_reversion_sleeve_splice_suite.py``.  This module only materializes
prompt specifications, validates completed caches, combines the two scorers
used by Experiment 1, and replays cached scores through the existing V2
portfolio harness.  Importing or executing this module never calls an LLM.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import gc
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import mean_reversion_llm_v1 as v1
import mean_reversion_llm_v2 as runner
import run_backtest_v3 as rb


ORIGINAL_RANGE_RULE = "- Use the full 0-100 range. Do not bunch scores around 50."
MINIMAL_RANGE_RULE = (
    "- Use the 0-100 range only when the supplied cross-section supports it. "
    "Do not manufacture separation."
)
PROPORTIONAL_RANGE_LANGUAGE = (
    "Two tickers that differ on a signal must receive distinct scores proportional "
    "to the signal's magnitude."
)
ORIGINAL_CONTEXT = """CONTEXT
Mean reversion is the tendency of stocks that have moved sharply in one
direction to partially reverse over the next 5 trading days, especially
when the move shows signs of exhaustion (extreme indicators, volume
climax, range positioning). The strongest reversion setups combine an
extreme recent move with exhaustion signals and a supportive regime."""
B_CONTEXT = """CONTEXT
We are evaluating two pools of stocks that have experienced extreme 5-day moves, identifying those most likely to partially reverse over the next 5 trading days based on exhaustion, positioning, and regime signals."""
B_SHORT_FRAMING = (
    "The SHORT POOL contains stocks that have experienced extreme positive 5-day "
    "moves. Identify those most likely to partially reverse over the next 5 trading "
    "days based on exhaustion, positioning, and regime signals."
)

ORIGINAL_V2_PROMPT = runner.V2_SYSTEM_PROMPT
B_PROMPT = runner.G0_05_DEPRESCRIBED_CONTEXT_SYSTEM_PROMPT


def _replace_once(parent: str, old: str, new: str, *, label: str) -> str:
    if parent.count(old) != 1:
        raise RuntimeError(f"{label}: expected exactly one replace target")
    if old == new:
        raise RuntimeError(f"{label}: replacement must change the prompt")
    return parent.replace(old, new, 1)


B_CONTEXT_ORIGINAL_RANGE_PROMPT = _replace_once(
    B_PROMPT,
    MINIMAL_RANGE_RULE,
    ORIGINAL_RANGE_RULE,
    label="Experiment 2 range rule",
)


def build_asymmetric_hybrid_prompt() -> str:
    """Build the one-call prompt with intentionally different side philosophies."""
    glossary = runner._feature_glossary_block(ORIGINAL_V2_PROMPT)
    return f"""MEAN-REVERSION CANDIDATE SCORING

LONG-SIDE FRAMING
Mean reversion is the tendency of stocks that have moved sharply in one
direction to partially reverse over the next 5 trading days, especially
when the move shows signs of exhaustion (extreme indicators, volume
climax, range positioning). The strongest reversion setups combine an
extreme recent move with exhaustion signals and a supportive regime.
Apply this original V2 philosophy to the LONG POOL only.

SHORT-SIDE FRAMING
{B_SHORT_FRAMING}
Apply this de-prescribed B philosophy to the SHORT POOL only.

{glossary}

TASK
You are scoring two side-specific pools over the next 5 trading days.

  LONG POOL  = the biggest 5-day losers in the Russell 1000 candidate universe.
               Look for an upward snap-back / oversold-bounce setup.

  SHORT POOL = the biggest 5-day winners in the Russell 1000 candidate universe.
               Look for a downward fade / overbought-reversal setup.

LONG-SIDE SCORE CONTRACT
- Higher LONG score means a stronger expected upward snap-back over the next 5 trading days.
- Use the full 0-100 range. Do not bunch scores around 50.

SHORT-SIDE SCORE CONTRACT
- Higher SHORT score means a stronger expected downward fade over the next 5 trading days.
- Use the 0-100 range only when the supplied short-side cross-section supports it. Do not manufacture separation.

ASYMMETRIC RULES
- LONG and SHORT scores are side-relative.
- LONG scores must follow the LONG-SIDE SCORE CONTRACT.
- SHORT scores must follow the SHORT-SIDE SCORE CONTRACT.
- The two score-range rules intentionally differ.
- Do not compare scores across sides.
- Return every ticker exactly once on its original side.
- Use only the supplied indicator columns. Use no outside knowledge, calendar awareness, or ticker familiarity.
- Treat tickers as opaque identifiers.
- Return no rationale, weights, portfolio construction, or commentary.
- Return strict JSON only, with no markdown fences or other text.

OUTPUT FORMAT
{{
  "long_scores": [
    {{"ticker": "SEC_0001", "score": 72}}
  ],
  "short_scores": [
    {{"ticker": "SEC_0061", "score": 84}}
  ]
}}
""".strip()


ASYMMETRIC_HYBRID_PROMPT = build_asymmetric_hybrid_prompt()

EXPECTED_PROMPT_SHA256 = {
    "original_v2": "02e28abb83b608524088cce18e18aeabd08171484e1998bcdcafb16f8bf28877",
    "b_deprescribed": "384e11633b3135afba73b9b723b46ef629df807a44f0ff5c596800f313583ac0",
    "b_original_range": "95f008b9ac256f0a5044b714b7a937e44c8e55e1f0e3f47adaf6d2bf8120618a",
    "asymmetric_hybrid": "b966d76a8487eb7386d5b2f066934f46efa9215a88662603683e9c50000978ae",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_prompt_contracts() -> None:
    """Refuse setup when any canonical prompt or requested isolation has drifted."""
    actual = {
        "original_v2": sha256_text(ORIGINAL_V2_PROMPT),
        "b_deprescribed": sha256_text(B_PROMPT),
        "b_original_range": sha256_text(B_CONTEXT_ORIGINAL_RANGE_PROMPT),
        "asymmetric_hybrid": sha256_text(ASYMMETRIC_HYBRID_PROMPT),
    }
    if actual != EXPECTED_PROMPT_SHA256:
        raise RuntimeError(f"Frozen prompt hash drift: {actual}")
    if _replace_once(
        runner.V2_MINIMAL_CHANGE_CONTROL_PROMPT,
        MINIMAL_RANGE_RULE,
        ORIGINAL_RANGE_RULE,
        label="Original V2 reconstruction",
    ) != ORIGINAL_V2_PROMPT:
        raise RuntimeError("Original V2 is no longer the expected one-line range-rule treatment")
    if _replace_once(
        runner.V2_MINIMAL_CHANGE_CONTROL_PROMPT,
        ORIGINAL_CONTEXT,
        B_CONTEXT,
        label="B reconstruction",
    ) != B_PROMPT:
        raise RuntimeError("B is no longer the expected de-prescribed-context treatment")
    if B_CONTEXT_ORIGINAL_RANGE_PROMPT.count(ORIGINAL_RANGE_RULE) != 1:
        raise RuntimeError("Experiment 2 must contain the Original V2 range rule exactly once")
    if MINIMAL_RANGE_RULE in B_CONTEXT_ORIGINAL_RANGE_PROMPT:
        raise RuntimeError("Experiment 2 retained B's minimal-change range rule")
    if PROPORTIONAL_RANGE_LANGUAGE in B_CONTEXT_ORIGINAL_RANGE_PROMPT:
        raise RuntimeError("Experiment 2 accidentally contains the C/D proportional rule")

    hybrid = ASYMMETRIC_HYBRID_PROMPT
    required = (
        B_SHORT_FRAMING,
        "LONG and SHORT scores are side-relative.",
        "LONG scores must follow the LONG-SIDE SCORE CONTRACT.",
        "SHORT scores must follow the SHORT-SIDE SCORE CONTRACT.",
        "The two score-range rules intentionally differ.",
        "Do not compare scores across sides.",
        "Return every ticker exactly once on its original side.",
        "Return no rationale, weights, portfolio construction, or commentary.",
        '"long_scores"',
        '"short_scores"',
    )
    missing = [clause for clause in required if clause not in hybrid]
    if missing:
        raise RuntimeError(f"Experiment 3 is missing required clauses: {missing}")
    if PROPORTIONAL_RANGE_LANGUAGE in hybrid:
        raise RuntimeError("Experiment 3 accidentally contains the C/D proportional rule")
    if runner._feature_glossary_block(hybrid) != runner._feature_glossary_block(
        ORIGINAL_V2_PROMPT
    ):
        raise RuntimeError("Experiment 3 changed the frozen feature glossary")


def _prompt_spec_payload(
    *,
    candidate_id: str,
    generation: str,
    parent_id: str,
    parent_prompt: str,
    old_text: str,
    new_text: str,
    prompt_text: str,
    mutation_mode: str,
    hypothesis: str,
    mechanism: str,
    expected_failure: str,
    prior_experiment_ids: Sequence[str],
) -> dict[str, Any]:
    materialized = _replace_once(parent_prompt, old_text, new_text, label=candidate_id)
    if materialized != prompt_text:
        raise RuntimeError(f"{candidate_id}: prompt does not match its declared patch")
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "generation": generation,
        "parent_id": parent_id,
        "mutation_mode": mutation_mode,
        "side_scope": "both",
        "replicate_id": "r1",
        "parent_prompt_text": parent_prompt,
        "parent_prompt_hash": sha256_text(parent_prompt),
        "patch": {
            "operation": "replace_once",
            "old_text": old_text,
            "new_text": new_text,
        },
        "prompt_text": prompt_text,
        "prompt_hash": sha256_text(prompt_text),
        "hypothesis": hypothesis,
        "mechanism": mechanism,
        "expected_failure": expected_failure,
        "prior_experiment_ids": list(prior_experiment_ids),
        "output_contract": runner.AUTORESEARCH_OUTPUT_CONTRACT,
        "score_min": 0.0,
        "score_max": 100.0,
        "score_direction": "higher_is_better",
        "anonymize_tickers": True,
    }


def prompt_spec_payloads() -> dict[str, dict[str, Any]]:
    """Return the four fresh scorer legs needed by the three portfolio tests."""
    assert_prompt_contracts()
    minimal = runner.V2_MINIMAL_CHANGE_CONTROL_PROMPT
    return {
        "original_v2": _prompt_spec_payload(
            candidate_id="SLEEVE_E1_ORIGINAL_V2_FULL",
            generation="G0",
            parent_id="v2-minimal-change-control",
            parent_prompt=minimal,
            old_text=MINIMAL_RANGE_RULE,
            new_text=ORIGINAL_RANGE_RULE,
            prompt_text=ORIGINAL_V2_PROMPT,
            mutation_mode="substitutive",
            hypothesis="The original V2 scorer supplies the stronger long sleeve for a two-call splice.",
            mechanism="Preserve the exact tested Original V2 two-sided scoring task and retain only its long scores offline.",
            expected_failure="A fresh stochastic replicate may not reproduce the historical long-sleeve advantage.",
            prior_experiment_ids=("original-v2",),
        ),
        "b_deprescribed": _prompt_spec_payload(
            candidate_id="SLEEVE_E1_B_FULL",
            generation="G0",
            parent_id="v2-minimal-change-control",
            parent_prompt=minimal,
            old_text=ORIGINAL_CONTEXT,
            new_text=B_CONTEXT,
            prompt_text=B_PROMPT,
            mutation_mode="substitutive",
            hypothesis="The de-prescribed B scorer supplies the stronger short sleeve for a two-call splice.",
            mechanism="Preserve the exact tested B two-sided scoring task and retain only its short scores offline.",
            expected_failure="A fresh stochastic replicate may not reproduce the historical short-sleeve advantage.",
            prior_experiment_ids=("smart_prompt_search_20260802:G0_05", "prompt_interaction:B"),
        ),
        "b_original_range": _prompt_spec_payload(
            candidate_id="SLEEVE_E2_B_CONTEXT_ORIGINAL_RANGE",
            generation="G1",
            parent_id="SLEEVE_E1_B_FULL",
            parent_prompt=B_PROMPT,
            old_text=MINIMAL_RANGE_RULE,
            new_text=ORIGINAL_RANGE_RULE,
            prompt_text=B_CONTEXT_ORIGINAL_RANGE_PROMPT,
            mutation_mode="substitutive",
            hypothesis="B's de-prescribed framing remains useful when paired with Original V2's score-range rule.",
            mechanism="Change only B's score-separation sentence to isolate its interaction with context.",
            expected_failure="B's behavior may depend on the minimal-change instruction not to manufacture separation.",
            prior_experiment_ids=("prompt_interaction:B", "original-v2"),
        ),
        "asymmetric_hybrid": _prompt_spec_payload(
            candidate_id="SLEEVE_E3_ONE_CALL_ASYMMETRIC_HYBRID",
            generation="G1",
            parent_id="original-v2",
            parent_prompt=ORIGINAL_V2_PROMPT,
            old_text=ORIGINAL_V2_PROMPT,
            new_text=ASYMMETRIC_HYBRID_PROMPT,
            prompt_text=ASYMMETRIC_HYBRID_PROMPT,
            mutation_mode="rewrite",
            hypothesis="One call can preserve Original V2 long behavior and B short behavior through explicit side asymmetry.",
            mechanism="Give each side its own framing and range rule while forbidding cross-side score comparison.",
            expected_failure="Joint-context interaction may degrade one or both specialized sleeves.",
            prior_experiment_ids=("original-v2", "prompt_interaction:B"),
        ),
    }


def write_prompt_specs(directory: Path) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for arm, payload in prompt_spec_payloads().items():
        path = directory / f"{arm}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        runner.load_experimental_prompt_spec(path)
        paths[arm] = path.resolve()
    return paths


def splice_score_entries(
    original_scores: Mapping[str, dict[str, Any]],
    b_scores: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Use only Original V2 longs and B shorts, preserving full source provenance."""
    if set(original_scores) != set(b_scores):
        missing_original = sorted(set(b_scores) - set(original_scores))
        missing_b = sorted(set(original_scores) - set(b_scores))
        raise ValueError(
            "Experiment 1 source caches cover different dates: "
            f"missing_original={missing_original}, missing_b={missing_b}"
        )
    combined: dict[str, dict[str, Any]] = {}
    for storage_key in sorted(original_scores):
        original = original_scores[storage_key]
        b_entry = b_scores[storage_key]
        if original.get("universe_hash") != b_entry.get("universe_hash"):
            raise ValueError(f"Experiment 1 source universe mismatch at {storage_key}")
        if original.get("rebalance_date") != b_entry.get("rebalance_date"):
            raise ValueError(f"Experiment 1 source date mismatch at {storage_key}")
        long_rows = original.get("long_scores")
        short_rows = b_entry.get("short_scores")
        if not isinstance(long_rows, list) or not isinstance(short_rows, list):
            raise ValueError(f"Experiment 1 source score arrays are missing at {storage_key}")
        combined[storage_key] = {
            "long_scores": [dict(row) for row in long_rows],
            "short_scores": [dict(row) for row in short_rows],
            "rebalance_date": original.get("rebalance_date"),
            "universe_hash": original.get("universe_hash"),
            "prompt_variant": "two_call_sleeve_splice",
            "source_prompt_hashes": {
                "long": original.get("prompt_hash"),
                "short": b_entry.get("prompt_hash"),
            },
            "source_cache_keys": {
                "long": original.get("cache_key"),
                "short": b_entry.get("cache_key"),
            },
        }
    return combined


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _capital_return(weekly: pd.DataFrame, cost_bps: float) -> pd.Series:
    column = f"ls_c2c_{rb.format_bps_label(float(cost_bps))}"
    if column not in weekly:
        raise ValueError(f"Missing cost overlay {column}")
    return pd.to_numeric(weekly[column], errors="coerce") * runner.CAPITAL_RETURN_SCALE


def _side_contributions(weekly: pd.DataFrame, cost_bps: float) -> tuple[pd.Series, pd.Series]:
    rate = float(cost_bps) / 10_000.0
    long_net = pd.to_numeric(weekly["long_ret_c2c"], errors="coerce") - (
        pd.to_numeric(weekly["turnover_long"], errors="coerce").fillna(0.0) * 2.0 * rate
    )
    short_net = -pd.to_numeric(weekly["short_ret_c2c"], errors="coerce") - (
        pd.to_numeric(weekly["turnover_short"], errors="coerce").fillna(0.0) * 2.0 * rate
    )
    scale = runner.CAPITAL_RETURN_SCALE
    return long_net * scale, short_net * scale


def _return_metrics(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    periods = rb.DEFAULT_PERIODS_PER_YEAR
    annual_return = float(clean.mean() * periods) if not clean.empty else np.nan
    annual_vol = float(clean.std() * np.sqrt(periods)) if len(clean) > 1 else np.nan
    return {
        "n_periods": int(len(clean)),
        "annual_return_pct": annual_return * 100.0,
        "cagr_pct": rb.safe_cagr(clean, periods_per_year=periods) if not clean.empty else np.nan,
        "sharpe": annual_return / annual_vol if np.isfinite(annual_vol) and annual_vol > 0 else np.nan,
        "max_drawdown_pct": rb.compute_max_drawdown_pct(clean) if not clean.empty else np.nan,
    }


def _paired_comparison_row(
    label: str,
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    cost_bps: float,
) -> dict[str, Any]:
    treatment_capital = _capital_return(treatment, cost_bps)
    control_capital = _capital_return(control, cost_bps)
    paired = pd.concat(
        [treatment_capital.rename("treatment"), control_capital.rename("control")],
        axis=1,
        join="inner",
    ).dropna()
    treatment_long, treatment_short = _side_contributions(treatment, cost_bps)
    control_long, control_short = _side_contributions(control, cost_bps)
    long_delta = (treatment_long - control_long).reindex(paired.index).dropna()
    short_delta = (treatment_short - control_short).reindex(paired.index).dropna()
    total_delta = paired["treatment"] - paired["control"]
    period_scale = rb.DEFAULT_PERIODS_PER_YEAR * 100.0
    yearly = total_delta.groupby(total_delta.index.year).mean()
    treatment_metrics = _return_metrics(paired["treatment"])
    control_metrics = _return_metrics(paired["control"])
    return {
        "comparison": label,
        "cost_bps_per_traded_side": float(cost_bps),
        "n_periods": int(len(paired)),
        "treatment_cagr_pct": treatment_metrics["cagr_pct"],
        "control_cagr_pct": control_metrics["cagr_pct"],
        "cagr_lift_pp": float(treatment_metrics["cagr_pct"] - control_metrics["cagr_pct"]),
        "treatment_sharpe": treatment_metrics["sharpe"],
        "control_sharpe": control_metrics["sharpe"],
        "sharpe_lift": float(treatment_metrics["sharpe"] - control_metrics["sharpe"]),
        "treatment_max_drawdown_pct": treatment_metrics["max_drawdown_pct"],
        "control_max_drawdown_pct": control_metrics["max_drawdown_pct"],
        "annualized_total_paired_lift_pct": float(total_delta.mean() * period_scale),
        "annualized_long_lift_contribution_pct": float(long_delta.mean() * period_scale),
        "annualized_short_lift_contribution_pct": float(short_delta.mean() * period_scale),
        "positive_both_sleeves": bool(long_delta.mean() > 0.0 and short_delta.mean() > 0.0),
        "positive_lift_years": int((yearly > 0.0).sum()),
        "years_tested": int(len(yearly)),
    }


def _assert_splice_attribution(
    experiment_1: v1.ExperimentBundle,
    original: v1.ExperimentBundle,
    b_bundle: v1.ExperimentBundle,
) -> dict[str, Any]:
    e1_weekly = experiment_1.weekly_returns
    original_weekly = original.weekly_returns
    b_weekly = b_bundle.weekly_returns
    if not e1_weekly.index.equals(original_weekly.index) or not e1_weekly.index.equals(b_weekly.index):
        raise ValueError("Experiment 1 and its two sources do not cover identical dates")
    if not np.allclose(e1_weekly["long_ret_c2c"], original_weekly["long_ret_c2c"], equal_nan=True):
        raise ValueError("Experiment 1 long sleeve does not exactly reproduce Original V2")
    if not np.allclose(e1_weekly["short_ret_c2c"], b_weekly["short_ret_c2c"], equal_nan=True):
        raise ValueError("Experiment 1 short sleeve does not exactly reproduce B")

    def selected_pairs(bundle: v1.ExperimentBundle, side: str) -> set[tuple[pd.Timestamp, str]]:
        frame = bundle.portfolio_selections
        selected = frame[frame["position_type"].eq(side)]
        return {
            (pd.Timestamp(row.signal_date), str(row.ticker).upper())
            for row in selected[["signal_date", "ticker"]].itertuples(index=False)
        }

    if selected_pairs(experiment_1, "LONG") != selected_pairs(original, "LONG"):
        raise ValueError("Experiment 1 long selections changed during the splice")
    if selected_pairs(experiment_1, "SHORT") != selected_pairs(b_bundle, "SHORT"):
        raise ValueError("Experiment 1 short selections changed during the splice")
    return {
        "long_scores_source": "fresh Original V2 full two-sided calls",
        "short_scores_source": "fresh B full two-sided calls",
        "discarded_original_short_scores": True,
        "discarded_b_long_scores": True,
        "long_returns_and_selections_exact": True,
        "short_returns_and_selections_exact": True,
    }


def evaluate_completed_suite(
    *,
    output_dir: Path,
    cache_payloads: Mapping[str, dict[str, Any]],
    signal_manifest: runner.SignalDateManifest,
    config: rb.BenchmarkConfig,
) -> dict[str, Any]:
    """Replay all cached treatments and write the development-stage gross comparisons."""
    required = {"original_v2", "b_deprescribed", "b_original_range"}
    allowed = required | {"asymmetric_hybrid"}
    if not required.issubset(cache_payloads) or not set(cache_payloads).issubset(allowed):
        raise ValueError(
            f"Expected required cache payloads {sorted(required)} and optional asymmetric_hybrid; "
            f"got {sorted(cache_payloads)}"
        )
    signals = {
        arm: payload.get("signals")
        for arm, payload in cache_payloads.items()
    }
    if any(not isinstance(entries, dict) for entries in signals.values()):
        raise ValueError("Every completed cache must contain a signals object")
    spliced = splice_score_entries(signals["original_v2"], signals["b_deprescribed"])

    evaluation_config = replace(config, cost_bps_levels=(0.0,))
    _, membership_lookup, market_frames, feature_frames, spy_df = runner.load_experiment_inputs(
        evaluation_config
    )
    common = {
        "market_frames": market_frames,
        "feature_frames": feature_frames,
        "membership_lookup": membership_lookup,
        "spy_df": spy_df,
        "config": evaluation_config,
        "pool_per_side": 60,
        "positions_per_side": 30,
        "anchor_offset": 3,
        "llm_source_pool_per_side": 60,
        "allowed_signal_dates": signal_manifest.date_set,
    }
    score_sources: dict[str, tuple[str, Mapping[str, dict[str, Any]] | None]] = {
        "Trail5": ("trail5", None),
        "Fresh Original V2": ("llm", signals["original_v2"]),
        "Fresh B": ("llm", signals["b_deprescribed"]),
        "Experiment 1 — Original V2 Long + B Short — Two-Call Splice": ("llm", spliced),
        "Experiment 2 — B Context + Original V2 Range": ("llm", signals["b_original_range"]),
    }
    if "asymmetric_hybrid" in signals:
        score_sources["Experiment 3 — One-Call Asymmetric V2-Long / B-Short Hybrid"] = (
            "llm",
            signals["asymmetric_hybrid"],
        )
    bundles: dict[str, v1.ExperimentBundle] = {}
    analysis_root = Path(output_dir) / "analysis"
    run_root = analysis_root / "runs"
    for label, (provider, entries) in score_sources.items():
        weekly, checks, selections, pools = runner.run_v2_pool_strategy(
            **common,
            score_provider=provider,
            llm_scores_by_key=dict(entries) if entries is not None else None,
        )
        bundle = runner.build_v2_bundle(weekly, checks, selections, evaluation_config)
        bundles[label] = bundle
        slug = {
            "Trail5": "trail5",
            "Fresh Original V2": "fresh_original_v2",
            "Fresh B": "fresh_b",
            "Experiment 1 — Original V2 Long + B Short — Two-Call Splice": "experiment_1_two_call_splice",
            "Experiment 2 — B Context + Original V2 Range": "experiment_2_b_original_range",
            "Experiment 3 — One-Call Asymmetric V2-Long / B-Short Hybrid": "experiment_3_one_call_hybrid",
        }[label]
        runner.save_v2_bundle(bundle, run_root / slug)
        if provider == "llm":
            runner.save_v2_score_diagnostics(checks, run_root / slug)
        # Each replay returns the full 251-date pool objects, including feature
        # rows. They are not needed after the bundle is built and retaining the
        # previous arm while constructing the next can exceed local memory.
        del pools
        gc.collect()

    attribution = _assert_splice_attribution(
        bundles["Experiment 1 — Original V2 Long + B Short — Two-Call Splice"],
        bundles["Fresh Original V2"],
        bundles["Fresh B"],
    )
    variant = runner.V2Variant(pool_per_side=60, positions_per_side=30, anchor_offset=3)
    scorecard = pd.concat(
        [
            runner.scorecard_for_bundle(
                label,
                bundle,
                variant,
                scorer="trail5" if label == "Trail5" else "llm_score",
                score_pool_per_side=None if label == "Trail5" else 60,
            )
            for label, bundle in bundles.items()
        ],
        ignore_index=True,
    )
    analysis_root.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(analysis_root / "experiment_scorecard.csv", index=False)

    comparisons = [
        (
            "Experiment 1 vs Trail5 (primary)",
            "Experiment 1 — Original V2 Long + B Short — Two-Call Splice",
            "Trail5",
        ),
        (
            "Experiment 1 vs Fresh Original V2",
            "Experiment 1 — Original V2 Long + B Short — Two-Call Splice",
            "Fresh Original V2",
        ),
        (
            "Experiment 1 vs Fresh B",
            "Experiment 1 — Original V2 Long + B Short — Two-Call Splice",
            "Fresh B",
        ),
        (
            "Experiment 2 vs Fresh B (range-rule interaction)",
            "Experiment 2 — B Context + Original V2 Range",
            "Fresh B",
        ),
        (
            "Experiment 2 vs Fresh Original V2 (context contrast)",
            "Experiment 2 — B Context + Original V2 Range",
            "Fresh Original V2",
        ),
    ]
    if "asymmetric_hybrid" in signals:
        comparisons.extend(
            [
                (
                    "Experiment 3 vs Experiment 1 (one-call preservation)",
                    "Experiment 3 — One-Call Asymmetric V2-Long / B-Short Hybrid",
                    "Experiment 1 — Original V2 Long + B Short — Two-Call Splice",
                ),
                (
                    "Experiment 3 vs Trail5",
                    "Experiment 3 — One-Call Asymmetric V2-Long / B-Short Hybrid",
                    "Trail5",
                ),
            ]
        )
    paired_rows = [
        _paired_comparison_row(label, bundles[treatment], bundles[control], cost_bps)
        for label, treatment, control in comparisons
        for cost_bps in (0.0,)
    ]
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(analysis_root / "paired_comparisons.csv", index=False)
    (analysis_root / "experiment_1_attribution_checks.json").write_text(
        json.dumps(attribution, indent=2, sort_keys=True), encoding="utf-8"
    )
    result = {
        "status": "completed",
        "paid_calls_during_analysis": 0,
        "period": "development_2009_2013",
        "signal_dates": len(signal_manifest.dates),
        "portfolio_experiments": 3 if "asymmetric_hybrid" in signals else 2,
        "cost_bps_levels": [0.0],
        "transaction_cost_analysis_deferred": True,
        "attribution_checks": attribution,
        "artifacts": {
            "scorecard": str((analysis_root / "experiment_scorecard.csv").resolve()),
            "paired_comparisons": str((analysis_root / "paired_comparisons.csv").resolve()),
            "runs": str(run_root.resolve()),
        },
    }
    (analysis_root / "analysis_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


assert_prompt_contracts()
