"""Dedicated long/short prompt experiment and hybrid portfolio evaluation.

Scoring is performed by ``mean_reversion_llm_v2.py --phase llm-cache`` using
the immutable side-scoped prompt specs created here.  This module combines the
resulting caches with the already-completed joint minimal-change baseline; it
never reruns that baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import mean_reversion_llm_v2 as runner
from mean_reversion_llm_v2_consensus import (
    BOOK_PER_SIDE,
    PERIODS_PER_YEAR,
    deterministic_tiebreak,
    moving_block_bootstrap,
    return_metrics,
)


DEFAULT_BASELINE_ROOT = Path("backtest_results/v2_minimal_change_temp1_v4flash0731_second_2009_2014")
DEFAULT_OUTPUT_DIR = Path("backtest_results/v2_dedicated_side_prompts_2009_2014")
DEFAULT_SPEC_DIR = DEFAULT_OUTPUT_DIR / "prompt_specs"
DEFAULT_LONG_CACHE = DEFAULT_OUTPUT_DIR / "cache/dedicated_long_r1.json"
DEFAULT_SHORT_CACHE = DEFAULT_OUTPUT_DIR / "cache/dedicated_short_r1.json"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixed_prompt_spec_payload(side_scope: str, replicate_id: str = "r1") -> dict[str, Any]:
    side = str(side_scope).strip().lower()
    if side not in {"long", "short"}:
        raise ValueError("Fixed dedicated prompt side must be long or short")
    prompt = runner.dedicated_side_system_prompt(side)
    parent = runner.V2_MINIMAL_CHANGE_CONTROL_PROMPT
    return {
        "schema_version": 1,
        "candidate_id": f"dedicated-{side}-parent",
        "generation": "G0",
        "parent_id": "v2-minimal-change-control",
        "mutation_mode": "rewrite",
        "side_scope": side,
        "replicate_id": replicate_id,
        "parent_prompt_text": parent,
        "parent_prompt_hash": _sha256(parent),
        "patch": {"operation": "replace_once", "old_text": parent, "new_text": prompt},
        "prompt_text": prompt,
        "prompt_hash": _sha256(prompt),
        "hypothesis": (
            f"A dedicated {side}-only objective improves five-day {side}-side mean-reversion selection "
            "relative to the completed joint minimal-change scorer."
        ),
        "mechanism": "Remove the opposite-side objective so the scorer need not compromise between asymmetric reversal problems.",
        "expected_failure": "Separate wording and separate calls may add no signal or amplify sampling noise.",
        "prior_experiment_ids": ["v2-minimal-change-control"],
        "output_contract": runner.AUTORESEARCH_OUTPUT_CONTRACT,
        "score_min": 0.0,
        "score_max": 100.0,
        "score_direction": "higher_is_better",
        "anonymize_tickers": True,
    }


def write_fixed_prompt_specs(directory: Path = DEFAULT_SPEC_DIR) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output: dict[str, Path] = {}
    for side in ("long", "short"):
        path = directory / f"dedicated_{side}_r1.json"
        path.write_text(json.dumps(fixed_prompt_spec_payload(side), indent=2, sort_keys=True), encoding="utf-8")
        # Parse immediately so generated artifacts cannot drift from the runner contract.
        runner.load_experimental_prompt_spec(path)
        output[side] = path
    return output


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def resolve_side_cache(path: Path, side_scope: str) -> Path:
    """Resolve an explicit cache or discover the runner-generated immutable identity."""
    path = Path(path)
    if path.is_file():
        return path
    candidates: list[Path] = []
    if path.parent.is_dir():
        for candidate in sorted(path.parent.glob("*.json")):
            try:
                metadata = _load_json(candidate).get("metadata", {})
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if metadata.get("side_scope") == side_scope and metadata.get("replicate_id") == "r1":
                candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {side_scope} r1 cache in {path.parent}, found {len(candidates)}"
        )
    return candidates[0]


def load_side_cache(path: Path, side_scope: str) -> dict[str, Any]:
    path = Path(path)
    payload = _load_json(path)
    metadata = payload.get("metadata", {})
    side = str(side_scope).lower()
    expected = {
        "model": runner.AUTORESEARCH_MODEL,
        "temperature": runner.AUTORESEARCH_TEMPERATURE,
        "thinking_mode": runner.AUTORESEARCH_THINKING_MODE,
        "anonymize_tickers": True,
        "side_scope": side,
        "replicate_id": "r1",
        "pool_count": 251,
        "prompt_hash": _sha256(runner.dedicated_side_system_prompt(side))[:12],
    }
    bad = {key: (value, metadata.get(key)) for key, value in expected.items() if metadata.get(key) != value}
    signals = payload.get("signals")
    if bad or not isinstance(signals, dict) or len(signals) != 251:
        raise ValueError(f"Invalid {side} side cache: metadata={bad}, signals={len(signals or {})}")
    for storage_key, entry in signals.items():
        if entry.get("side_scope") != side or entry.get("replicate_id") != "r1":
            raise ValueError(f"Side/replicate identity mismatch at {storage_key}")
        rows = entry.get("scores")
        if not isinstance(rows, list) or len(rows) != 60:
            raise ValueError(f"{storage_key} must contain exactly 60 side scores")
        tickers = [str(row.get("ticker", "")).upper() for row in rows]
        scores = pd.to_numeric([row.get("score") for row in rows], errors="coerce")
        if len(set(tickers)) != 60 or any(not ticker for ticker in tickers):
            raise ValueError(f"{storage_key} contains missing or duplicated side tickers")
        if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 100.0):
            raise ValueError(f"{storage_key} contains out-of-range side scores")
    return payload


def side_rank_panel(cache: dict[str, Any], side_scope: str) -> pd.DataFrame:
    side = str(side_scope).upper()
    rows: list[dict[str, Any]] = []
    for storage_key, entry in sorted(cache["signals"].items()):
        date = pd.Timestamp(entry.get("rebalance_date", storage_key.split("|", 1)[0]))
        for item in entry["scores"]:
            rows.append(
                {
                    "date": date,
                    "side": side,
                    "ticker": str(item["ticker"]).upper(),
                    "score": float(item["score"]),
                    "anonymous_ticker": item.get("anonymous_ticker", ""),
                }
            )
    panel = pd.DataFrame(rows)
    panel["score_rank"] = panel.groupby("date")["score"].rank(method="average", ascending=False)
    panel["blind_tiebreak"] = [
        deterministic_tiebreak(date, side, ticker)
        for date, ticker in zip(panel["date"], panel["ticker"])
    ]
    panel = panel.sort_values(["date", "score_rank", "blind_tiebreak"], kind="mergesort")
    panel["selection_rank"] = panel.groupby("date").cumcount() + 1
    panel["selected"] = panel["selection_rank"].le(BOOK_PER_SIDE)
    return panel


def consensus_side_rank_panel(
    caches: Sequence[dict[str, Any]], side_scope: str, expected_dates: Sequence[Any]
) -> pd.DataFrame:
    """Build a one-run rank panel or a three-run finalist mean-rank panel."""
    if len(caches) not in {1, 3}:
        raise ValueError("Side campaigns support one screening run or three finalist runs")
    expected = {pd.Timestamp(value).normalize() for value in expected_dates}
    panels: list[pd.DataFrame] = []
    identities: set[str] = set()
    prompt_hashes: set[str] = set()
    for cache in caches:
        metadata = cache.get("metadata", {})
        replicate = str(metadata.get("replicate_id", ""))
        if not replicate or replicate in identities:
            raise ValueError("Finalist caches require distinct replicate identities")
        identities.add(replicate)
        prompt_hashes.add(str(metadata.get("prompt_hash", "")))
        panel = side_rank_panel(cache, side_scope)
        panel = panel[panel["date"].dt.normalize().isin(expected)].copy()
        if set(panel["date"].dt.normalize()) != expected:
            raise ValueError(f"Replicate {replicate} does not cover the exact stage dates")
        panel = panel.rename(columns={"score_rank": f"rank_{replicate}", "score": f"score_{replicate}"})
        panels.append(panel[["date", "side", "ticker", f"rank_{replicate}", f"score_{replicate}"]])
    if len(prompt_hashes) != 1:
        raise ValueError("Finalist replicate prompt hashes differ")
    merged = panels[0]
    for panel in panels[1:]:
        merged = merged.merge(panel, on=["date", "side", "ticker"], how="inner", validate="one_to_one")
    rank_columns = [column for column in merged if column.startswith("rank_")]
    merged["score_rank"] = merged[rank_columns].mean(axis=1)
    merged["median_rank"] = merged[rank_columns].median(axis=1)
    merged["worst_rank"] = merged[rank_columns].max(axis=1)
    merged["blind_tiebreak"] = [
        deterministic_tiebreak(date, side, ticker)
        for date, side, ticker in zip(merged["date"], merged["side"], merged["ticker"])
    ]
    merged = merged.sort_values(
        ["date", "score_rank", "median_rank", "worst_rank", "blind_tiebreak"], kind="mergesort"
    )
    merged["selection_rank"] = merged.groupby("date").cumcount() + 1
    merged["selected"] = merged["selection_rank"].le(BOOK_PER_SIDE)
    merged["replicate_count"] = len(caches)
    return merged


def _baseline_paths(root: Path) -> tuple[Path, Path, Path]:
    base = Path(root) / "pool60_pos30/anchor_3"
    treatment = base / "llm_pool_v2_minimal_change_control"
    return (
        base / "baseline_pool/signal_checks.csv",
        treatment / "portfolio_selections.csv",
        treatment / "weekly_returns.csv",
    )


def load_baseline(root: Path = DEFAULT_BASELINE_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcome_path, selection_path, weekly_path = _baseline_paths(root)
    outcomes = pd.read_csv(outcome_path, parse_dates=["date"])
    outcomes["ticker"] = outcomes["ticker"].astype(str).str.upper()
    outcomes["pool_side"] = outcomes["pool_side"].astype(str).str.upper()
    selections = pd.read_csv(selection_path, parse_dates=["signal_date"])
    selections["ticker"] = selections["ticker"].astype(str).str.upper()
    selections["position_type"] = selections["position_type"].astype(str).str.upper()
    weekly = pd.read_csv(weekly_path, parse_dates=["date"]).set_index("date").sort_index()
    return outcomes, selections, weekly


def _dedicated_selected(panel: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    selected = panel[panel["selected"]].merge(
        outcomes[["date", "ticker", "pool_side", "fwd_c2c"]],
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    if selected["pool_side"].isna().any() or not selected["pool_side"].eq(selected["side"]).all():
        raise ValueError("Dedicated selection failed point-in-time side membership validation")
    return selected


def _baseline_selected(selections: pd.DataFrame, outcomes: pd.DataFrame, side: str) -> pd.DataFrame:
    chosen = selections[selections["position_type"].eq(side)][["signal_date", "ticker", "position_type", "rank"]].rename(
        columns={"signal_date": "date", "position_type": "side", "rank": "selection_rank"}
    )
    chosen = chosen.merge(
        outcomes[["date", "ticker", "pool_side", "fwd_c2c"]], on=["date", "ticker"], how="left", validate="one_to_one"
    )
    if chosen["pool_side"].isna().any():
        raise ValueError("Existing minimal selection could not be reconstructed")
    return chosen


def _side_series(selected: pd.DataFrame, side: str) -> pd.Series:
    raw = selected.groupby("date")["fwd_c2c"].mean().sort_index()
    return raw if side == "LONG" else -raw


def _hybrid_weekly(long_side: pd.Series, short_side: pd.Series, label: str) -> pd.DataFrame:
    weekly = pd.concat([long_side.rename("long_side_pnl"), short_side.rename("short_side_pnl")], axis=1).dropna()
    weekly["long_contribution"] = 0.5 * weekly["long_side_pnl"]
    weekly["short_contribution"] = 0.5 * weekly["short_side_pnl"]
    weekly["capital_c2c"] = weekly["long_contribution"] + weekly["short_contribution"]
    weekly["configuration"] = label
    return weekly


def _chronological_positive_blocks(delta: pd.Series) -> int:
    return int(sum(float(block.mean()) > 0.0 for block in np.array_split(delta.to_numpy(), 4)))


def _side_drawdown(side_returns: pd.Series) -> float:
    return float(return_metrics(side_returns).get("max_drawdown_pct", np.nan))


def evaluate_fixed_prompts(
    long_cache_path: Path = DEFAULT_LONG_CACHE,
    short_cache_path: Path = DEFAULT_SHORT_CACHE,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    long_cache_path = resolve_side_cache(long_cache_path, "long")
    short_cache_path = resolve_side_cache(short_cache_path, "short")
    long_cache = load_side_cache(long_cache_path, "long")
    short_cache = load_side_cache(short_cache_path, "short")
    if set(long_cache["signals"]) != set(short_cache["signals"]):
        raise ValueError("Dedicated long and short caches do not cover identical dates")
    for key in long_cache["signals"]:
        if long_cache["signals"][key].get("universe_hash") != short_cache["signals"][key].get("universe_hash"):
            raise ValueError(f"Dedicated side universe mismatch at {key}")

    outcomes, baseline_selections, baseline_weekly = load_baseline(baseline_root)
    long_panel = side_rank_panel(long_cache, "long")
    short_panel = side_rank_panel(short_cache, "short")
    dedicated_long = _dedicated_selected(long_panel, outcomes)
    dedicated_short = _dedicated_selected(short_panel, outcomes)
    baseline_long = _baseline_selected(baseline_selections, outcomes, "LONG")
    baseline_short = _baseline_selected(baseline_selections, outcomes, "SHORT")
    if set(map(tuple, baseline_long[["date", "ticker"]].to_numpy())) != set(
        map(tuple, baseline_selections[baseline_selections["position_type"].eq("LONG")][["signal_date", "ticker"]].to_numpy())
    ):
        raise ValueError("Untouched baseline long book changed during reconstruction")
    if set(map(tuple, baseline_short[["date", "ticker"]].to_numpy())) != set(
        map(tuple, baseline_selections[baseline_selections["position_type"].eq("SHORT")][["signal_date", "ticker"]].to_numpy())
    ):
        raise ValueError("Untouched baseline short book changed during reconstruction")

    series = {
        "baseline_long": _side_series(baseline_long, "LONG"),
        "baseline_short": _side_series(baseline_short, "SHORT"),
        "dedicated_long": _side_series(dedicated_long, "LONG"),
        "dedicated_short": _side_series(dedicated_short, "SHORT"),
    }
    configurations = {
        "existing_minimal": (series["baseline_long"], series["baseline_short"]),
        "dedicated_long__existing_short": (series["dedicated_long"], series["baseline_short"]),
        "existing_long__dedicated_short": (series["baseline_long"], series["dedicated_short"]),
        "dedicated_long__dedicated_short": (series["dedicated_long"], series["dedicated_short"]),
    }
    weekly_frames: list[pd.DataFrame] = []
    score_rows: list[dict[str, Any]] = []
    for label, (long_side, short_side) in configurations.items():
        weekly = _hybrid_weekly(long_side, short_side, label)
        weekly_frames.append(weekly.reset_index())
        score_rows.append(
            {
                "configuration": label,
                **return_metrics(weekly["capital_c2c"]),
                "annualized_long_contribution_pct": float(weekly["long_contribution"].mean() * PERIODS_PER_YEAR * 100.0),
                "annualized_short_contribution_pct": float(weekly["short_contribution"].mean() * PERIODS_PER_YEAR * 100.0),
            }
        )

    side_results: dict[str, Any] = {}
    for side, treatment_key, control_key in (
        ("long", "dedicated_long", "baseline_long"),
        ("short", "dedicated_short", "baseline_short"),
    ):
        treatment = series[treatment_key]
        control = series[control_key]
        delta = treatment - control
        annualized_lift = float(delta.mean() * PERIODS_PER_YEAR * 100.0)
        positive_blocks = _chronological_positive_blocks(delta)
        treatment_dd = _side_drawdown(treatment)
        control_dd = _side_drawdown(control)
        gate = {
            "positive_target_side_annualized_lift": annualized_lift > 0.0,
            "positive_in_at_least_two_of_four_blocks": positive_blocks >= 2,
            "drawdown_within_two_points": treatment_dd >= control_dd - 2.0,
        }
        side_results[side] = {
            "qualifies_for_autoresearch": all(gate.values()),
            "gate": gate,
            "annualized_lift_pct": annualized_lift,
            "positive_chronological_blocks": positive_blocks,
            "treatment_max_drawdown_pct": treatment_dd,
            "control_max_drawdown_pct": control_dd,
            "paired_bootstrap": moving_block_bootstrap(treatment, control),
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard = pd.DataFrame(score_rows)
    all_weekly = pd.concat(weekly_frames, ignore_index=True)
    long_panel.to_csv(output_dir / "dedicated_long_scores_and_ranks.csv", index=False)
    short_panel.to_csv(output_dir / "dedicated_short_scores_and_ranks.csv", index=False)
    dedicated_long.to_csv(output_dir / "dedicated_long_selections.csv", index=False)
    dedicated_short.to_csv(output_dir / "dedicated_short_selections.csv", index=False)
    scorecard.to_csv(output_dir / "fixed_prompt_hybrid_scorecard.csv", index=False)
    all_weekly.to_csv(output_dir / "fixed_prompt_weekly_returns.csv", index=False)
    result = {
        "status": "completed",
        "baseline_reused": str(Path(baseline_root).resolve()),
        "baseline_rerun": False,
        "cache_provenance": {
            "long": {"path": str(Path(long_cache_path).resolve()), "sha256": runner._file_sha256(Path(long_cache_path))},
            "short": {"path": str(Path(short_cache_path).resolve()), "sha256": runner._file_sha256(Path(short_cache_path))},
        },
        "long": side_results["long"],
        "short": side_results["short"],
        "autoresearch_sides": [side for side in ("long", "short") if side_results[side]["qualifies_for_autoresearch"]],
    }
    (output_dir / "fixed_prompt_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def evaluate_campaign_side_stage(
    candidate_cache_paths: Sequence[Path],
    parent_cache_path: Path,
    *,
    side_scope: str,
    expected_dates: Sequence[Any],
    stage: str,
    output_dir: Path,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
) -> dict[str, Any]:
    """Evaluate one side candidate while keeping the existing minimal opposite book fixed."""
    side = str(side_scope).strip().lower()
    side_upper = side.upper()
    candidate_caches = [_load_json(Path(path)) for path in candidate_cache_paths]
    for cache in candidate_caches:
        metadata = cache.get("metadata", {})
        if metadata.get("side_scope") != side:
            raise ValueError("Candidate cache side_scope mismatch")
    parent = _load_json(parent_cache_path)
    if parent.get("metadata", {}).get("side_scope") != side:
        raise ValueError("Parent cache side_scope mismatch")
    candidate_panel = consensus_side_rank_panel(candidate_caches, side, expected_dates)
    parent_panel = consensus_side_rank_panel([parent], side, expected_dates)
    outcomes, baseline_selections, _ = load_baseline(baseline_root)
    expected = {pd.Timestamp(value).normalize() for value in expected_dates}
    outcomes = outcomes[outcomes["date"].dt.normalize().isin(expected)]
    candidate_selected = _dedicated_selected(candidate_panel, outcomes)
    parent_selected = _dedicated_selected(parent_panel, outcomes)
    opposite = "SHORT" if side_upper == "LONG" else "LONG"
    opposite_selected = _baseline_selected(baseline_selections, outcomes, opposite)
    candidate_side = _side_series(candidate_selected, side_upper)
    parent_side = _side_series(parent_selected, side_upper)
    opposite_side = _side_series(opposite_selected, opposite)
    if side == "long":
        treatment = _hybrid_weekly(candidate_side, opposite_side, "candidate")
        control = _hybrid_weekly(parent_side, opposite_side, "dedicated_parent")
    else:
        treatment = _hybrid_weekly(opposite_side, candidate_side, "candidate")
        control = _hybrid_weekly(opposite_side, parent_side, "dedicated_parent")
    paired = moving_block_bootstrap(treatment["capital_c2c"], control["capital_c2c"], n_boot=10_000 if stage == "full" else 5_000)
    treatment_metrics = return_metrics(treatment["capital_c2c"])
    control_metrics = return_metrics(control["capital_c2c"])
    target_treatment_metrics = return_metrics(candidate_side)
    target_control_metrics = return_metrics(parent_side)
    target_delta = candidate_side - parent_side
    target_lift = float(target_delta.mean() * PERIODS_PER_YEAR * 100.0)
    positive_blocks = _chronological_positive_blocks(target_delta)
    target_drawdown_ok = float(target_treatment_metrics["max_drawdown_pct"]) >= float(
        target_control_metrics["max_drawdown_pct"]
    ) - 2.0
    if stage == "smoke":
        score_columns = [column for column in candidate_panel if column.startswith("score_")]
        dispersion = float(candidate_panel[score_columns].stack().groupby(candidate_panel.set_index(["date", "ticker"]).index).std().mean()) if len(score_columns) > 1 else float(candidate_panel.groupby("date")["score_rank"].std().mean())
        gate = {"pass": bool(np.isfinite(dispersion) and dispersion > 0.0), "reason": "coverage_and_dispersion"}
    elif stage in {"mini", "medium"}:
        gate = {"pass": float(paired["upper_90_pct"]) > 0.0, "reason": "one_sided_90pct_upper_bound"}
    else:
        gate_checks = {
            "positive_target_side_annualized_lift": target_lift > 0.0,
            "target_side_sharpe_improved": float(target_treatment_metrics["sharpe"]) > float(target_control_metrics["sharpe"]),
            "positive_in_at_least_two_of_four_blocks": positive_blocks >= 2,
            "target_side_drawdown_within_two_points": target_drawdown_ok,
            "three_replicates": len(candidate_caches) == 3,
        }
        gate = {"pass": all(gate_checks.values()), "reason": "three_run_side_finalist", **gate_checks}
    metrics = {
        **treatment_metrics,
        "paired_annualized_lift_pct": paired["mean_annualized_delta_pct"],
        "sharpe_improvement": float(treatment_metrics["sharpe"]) - float(control_metrics["sharpe"]),
        "target_side_annualized_lift_pct": target_lift,
        "target_side_sharpe_improvement": float(target_treatment_metrics["sharpe"]) - float(target_control_metrics["sharpe"]),
        "tail_improvement_pct": float(treatment_metrics["max_drawdown_pct"]) - float(control_metrics["max_drawdown_pct"]),
        "long_paired_lift_pct": target_lift if side == "long" else 0.0,
        "short_paired_lift_pct": target_lift if side == "short" else 0.0,
        "chronological_block_stability_pct": positive_blocks * 25.0,
        "replicate_count": len(candidate_caches),
    }
    run_dir = Path(output_dir) / "pool60_pos30/anchor_3/llm_pool_experimental"
    run_dir.mkdir(parents=True, exist_ok=True)
    treatment.reset_index().to_csv(run_dir / "weekly_returns.csv", index=False)
    candidate_panel.to_csv(run_dir / "signal_checks.csv", index=False)
    candidate_selected.to_csv(run_dir / "portfolio_selections.csv", index=False)
    pd.DataFrame([metrics]).to_csv(run_dir / "headline_metrics.csv", index=False)
    result = {
        "status": "completed",
        "stage": stage,
        "side_scope": side,
        "metrics": metrics,
        "target_side_treatment": target_treatment_metrics,
        "target_side_control": target_control_metrics,
        "controls": {"dedicated_parent": control_metrics},
        "bootstrap_vs_incumbent": paired,
        "gate": gate,
        "artifacts": {
            "treatment": str(run_dir),
            "parent_cache": str(Path(parent_cache_path).resolve()),
            "candidate_caches": [str(Path(path).resolve()) for path in candidate_cache_paths],
        },
    }
    return result


def approve_side_validation_batch(
    campaign_dirs: Sequence[Path], output_path: Path
) -> Path:
    """Freeze one or both side winners in one manual, pre-reveal approval action."""
    import mean_reversion_llm_v2_autoresearch as research

    paths = [Path(path).resolve() for path in campaign_dirs]
    if not 1 <= len(paths) <= 2 or len(set(paths)) != len(paths):
        raise ValueError("A side-validation batch requires one or two distinct campaigns")
    states = [research.campaign_load(path) for path in paths]
    sides = [str(state.get("side_scope", "both")) for state in states]
    if any(side not in {"long", "short"} for side in sides) or len(set(sides)) != len(sides):
        raise ValueError("Batch campaigns must contain distinct long/short side scopes")
    if any(state.get("status") != "awaiting_validation_approval" for state in states):
        raise ValueError("Every side campaign must be awaiting validation approval")
    freezes = [research.campaign_approve_validation(path, approver="manual-side-batch") for path in paths]
    payload = {
        "schema_version": 1,
        "artifact_type": "side_validation_batch_freeze",
        "approved": True,
        "approved_at": research.campaign_utc_now(),
        "holdout_start": "2020-01-01",
        "campaigns": [
            {
                "campaign_id": path.name,
                "side_scope": side,
                "winner_id": state["winner_id"],
                "validation_freeze": str(freeze.resolve()),
                "validation_freeze_hash": research.campaign_file_hash(freeze),
            }
            for path, side, state, freeze in zip(paths, sides, states, freezes)
        ],
    }
    output_path = Path(output_path).resolve()
    research.campaign_atomic_json(output_path, payload)
    batch_hash = research.campaign_file_hash(output_path)
    for path in paths:
        with research.campaign_lock(path):
            state = research.campaign_load(path)
            state["validation"].update(
                {"batch_freeze_path": str(output_path), "batch_freeze_hash": batch_hash}
            )
            research.campaign_save(path, state, "side_validation_batch_frozen")
    return output_path


def combine_side_campaign_results(
    campaign_dirs: Sequence[Path], *, validation: bool, output_dir: Path
) -> dict[str, Any]:
    """Combine independently frozen winning sides, or retain the fixed opposite book."""
    import mean_reversion_llm_v2_autoresearch as research

    rows: dict[str, tuple[dict[str, Any], pd.DataFrame]] = {}
    for campaign_dir in map(Path, campaign_dirs):
        state = research.campaign_load(campaign_dir)
        side = str(state.get("side_scope", "both"))
        if side not in {"long", "short"} or side in rows:
            raise ValueError("Combined results require distinct side campaigns")
        result_path = (
            campaign_dir / "validation/result.json"
            if validation
            else research.campaign_stage_result_path(campaign_dir, str(state["winner_id"]), "full")
        )
        result = research.campaign_read_json(result_path)
        if result.get("status") != "completed" or not result.get("gate", {}).get("pass"):
            raise ValueError(f"{side} winner does not have a passing completed result")
        weekly_path = Path(result["artifacts"]["treatment"]) / "weekly_returns.csv"
        weekly = pd.read_csv(weekly_path, parse_dates=["date"]).set_index("date").sort_index()
        rows[side] = (result, weekly)
    if not rows or len(rows) > 2:
        raise ValueError("Provide one or two winning side campaigns")
    if len(rows) == 1:
        side, (result, weekly) = next(iter(rows.items()))
        combined = weekly[["long_contribution", "short_contribution", "capital_c2c"]].copy()
    else:
        long_weekly = rows["long"][1]
        short_weekly = rows["short"][1]
        combined = pd.concat(
            [
                long_weekly["long_contribution"].rename("long_contribution"),
                short_weekly["short_contribution"].rename("short_contribution"),
            ],
            axis=1,
            join="inner",
        )
        combined["capital_c2c"] = combined["long_contribution"] + combined["short_contribution"]
    metrics = return_metrics(combined["capital_c2c"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = "validation" if validation else "development"
    combined.reset_index().to_csv(output_dir / f"combined_{label}_weekly_returns.csv", index=False)
    result = {
        "status": "completed",
        "period": label,
        "sides": sorted(rows),
        "metrics": metrics,
        "campaigns": [str(Path(path).resolve()) for path in campaign_dirs],
        "opposite_side_policy": "existing_minimal" if len(rows) == 1 else "other_side_winner",
    }
    (output_dir / f"combined_{label}_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dedicated long/short prompt experiment utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-specs")
    init.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--long-cache", type=Path, default=DEFAULT_LONG_CACHE)
    evaluate.add_argument("--short-cache", type=Path, default=DEFAULT_SHORT_CACHE)
    evaluate.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    freeze = sub.add_parser("freeze-validation")
    freeze.add_argument("campaign_dirs", nargs="+", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    combine = sub.add_parser("combine-campaigns")
    combine.add_argument("campaign_dirs", nargs="+", type=Path)
    combine.add_argument("--validation", action="store_true")
    combine.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "combined")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "init-specs":
        print(json.dumps({key: str(value) for key, value in write_fixed_prompt_specs(args.spec_dir).items()}, indent=2))
    elif args.command == "evaluate":
        print(
            json.dumps(
                evaluate_fixed_prompts(args.long_cache, args.short_cache, args.baseline_root, args.output_dir),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "freeze-validation":
        print(approve_side_validation_batch(args.campaign_dirs, args.output))
    else:
        print(
            json.dumps(
                combine_side_campaign_results(
                    args.campaign_dirs, validation=args.validation, output_dir=args.output_dir
                ),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
