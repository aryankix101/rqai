"""Analyze fixed-prompt minimal-change scoring replicas against matched Trail5.

This is an offline research script. It uses only completed development-period
artifacts and refuses to continue unless all LLM runs share their candidate
pools, features, and matched deterministic baseline. The tabular scores are
reused only after their candidate rows are verified against the same panel.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import mean_reversion_research_diagnostics as diagnostics


DEFAULT_REPLICAS = {
    "replica_1": Path(
        "backtest_results/v2_prompt_ab_2009_2014/minimal_change_temp1"
    ),
    "replica_2": Path(
        "backtest_results/v2_minimal_change_temp1_v4flash0731_second_2009_2014"
    ),
}
DEFAULT_REPLICA_3 = Path(
    "backtest_results/v2_minimal_change_temp1_replica3_20261027"
)
TABULAR_DIR = Path(
    "backtest_results/research_thesis_package_20260808/same_information_tabular"
)
REPLICA_SCORER_DIR = "llm_pool_v2_minimal_change_control"
BOOK_SIZE = 30
PERIODS_PER_YEAR = 52.0
FEATURES = (
    "feature_MRL",
    "feature_MRS",
    "feature_Trail5",
    "feature_MRSig",
    "feature_RSI5",
    "feature_Rng55",
    "feature_Vel/V",
    "feature_%MA100",
    "feature_52WH",
    "feature_Mom3m",
    "feature_AutoC",
    "feature_ADX",
    "feature_VolPctl",
)


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(diagnostics._json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_paths(root: Path) -> dict[str, Path]:
    anchor = root / "pool60_pos30/anchor_3"
    return {
        "config": root / "run_config.json",
        "treatment": anchor / REPLICA_SCORER_DIR,
        "trail5": anchor / "baseline_pool",
    }


def _read_run(root: Path) -> dict[str, Any]:
    paths = _run_paths(root)
    config = diagnostics._read_json(paths["config"])
    treatment_checks = diagnostics.load_checks(paths["treatment"])
    treatment_selections = diagnostics.load_selections(paths["treatment"])
    treatment_weekly = diagnostics._capitalize_weekly(
        diagnostics._read_csv_development(paths["treatment"] / "weekly_returns.csv")
    )
    trail_checks = diagnostics.load_checks(paths["trail5"])
    trail_selections = diagnostics.load_selections(paths["trail5"])
    trail_weekly = diagnostics._capitalize_weekly(
        diagnostics._read_csv_development(paths["trail5"] / "weekly_returns.csv")
    )
    return {
        "root": root,
        "paths": paths,
        "config": config,
        "checks": treatment_checks,
        "selections": treatment_selections,
        "weekly": treatment_weekly,
        "trail_checks": trail_checks,
        "trail_selections": trail_selections,
        "trail_weekly": trail_weekly,
    }


def _canonical_checks(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "pool_side", "ticker", "fwd_c2c", *FEATURES]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise diagnostics.ResearchDiagnosticError(
            f"Candidate checks lack required columns: {sorted(missing)}"
        )
    result = frame[columns].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["pool_side"] = result["pool_side"].astype(str).str.upper()
    result["ticker"] = result["ticker"].astype(str)
    return result.sort_values(["date", "pool_side", "ticker"]).reset_index(drop=True)


def _assert_frames_match(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
    if len(reference) != len(candidate):
        raise diagnostics.ResearchDiagnosticError(
            f"{label} has {len(candidate)} rows; expected {len(reference)}"
        )
    key_cols = ["date", "pool_side", "ticker"]
    ref = _canonical_checks(reference)
    got = _canonical_checks(candidate)
    if not ref[key_cols].equals(got[key_cols]):
        raise diagnostics.ResearchDiagnosticError(f"{label} candidate keys do not match")
    numeric_cols = ["fwd_c2c", *FEATURES]
    for column in numeric_cols:
        left = pd.to_numeric(ref[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(got[column], errors="coerce").to_numpy(float)
        if not np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True):
            raise diagnostics.ResearchDiagnosticError(
                f"{label} differs from reference in {column}"
            )


def _weekly_capital(weekly: pd.DataFrame) -> pd.DataFrame:
    frame = weekly[["date", "long_ret_c2c", "short_ret_c2c"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["capital_return"] = 0.5 * (
        pd.to_numeric(frame["long_ret_c2c"], errors="coerce")
        - pd.to_numeric(frame["short_ret_c2c"], errors="coerce")
    )
    return frame[["date", "capital_return"]].sort_values("date").reset_index(drop=True)


def _weekly_from_ranks(
    predictions: pd.DataFrame,
    score_col: str,
    *,
    book_size: int = BOOK_SIZE,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, date_group in predictions.groupby("date", sort=True):
        side_values: dict[str, float] = {}
        for side, side_group in date_group.groupby("pool_side", sort=True):
            usable = side_group.dropna(subset=[score_col, "fwd_c2c"]).copy()
            chosen = usable.sort_values(
                [score_col, "ticker"], ascending=[False, True], kind="mergesort"
            ).head(book_size)
            sign = -1.0 if side == "SHORT" else 1.0
            side_values[side] = float((sign * chosen["fwd_c2c"]).mean())
        if set(side_values) != {"LONG", "SHORT"}:
            continue
        rows.append(
            {
                "date": pd.Timestamp(date),
                "capital_return": 0.5 * (side_values["LONG"] + side_values["SHORT"]),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _rank_combo(checks: pd.DataFrame, features: tuple[str, ...], label: str) -> pd.DataFrame:
    panel = checks.copy()
    panel["score"] = 0.0
    for feature in features:
        raw = pd.to_numeric(panel[feature], errors="coerce")
        oriented = np.where(panel["pool_side"].eq("SHORT"), raw, -raw)
        panel[f"_oriented_{feature}"] = oriented
        ranks = panel.groupby(["date", "pool_side"])[f"_oriented_{feature}"].rank(
            method="average", ascending=True
        )
        counts = panel.groupby(["date", "pool_side"])[f"_oriented_{feature}"].transform(
            "count"
        )
        panel["score"] += (ranks - 1.0) / (counts - 1.0).clip(lower=1.0)
    panel["score"] /= len(features)
    return panel[["date", "pool_side", "ticker", "fwd_c2c", "score"]].assign(
        model=label
    )


def _paired_stats(
    candidate: pd.DataFrame,
    control: pd.DataFrame,
    *,
    label: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    joined = candidate.merge(
        control, on="date", how="inner", suffixes=("_candidate", "_control"), validate="one_to_one"
    )
    if len(joined) != len(candidate) or len(joined) != len(control):
        raise diagnostics.ResearchDiagnosticError(
            f"{label} and Trail5 dates do not match exactly"
        )
    delta = joined["capital_return_candidate"] - joined["capital_return_control"]
    boot = diagnostics.moving_block_bootstrap(
        delta, n_boot=n_boot, block_size=4, seed=seed
    )
    cand_metrics = diagnostics.return_metrics(joined["capital_return_candidate"])
    base_metrics = diagnostics.return_metrics(joined["capital_return_control"])
    return {
        "model": label,
        "n_dates": int(len(joined)),
        "annual_return_pct": cand_metrics["annual_return_pct"],
        "cagr_pct": cand_metrics["cagr_pct"],
        "sharpe": cand_metrics["sharpe"],
        "max_drawdown_pct": cand_metrics["max_drawdown_pct"],
        "trail5_annual_return_pct": base_metrics["annual_return_pct"],
        "paired_lift_vs_trail5_pp_per_year": boot["mean_annualized_lift_pct"],
        "ci95_low_pp_per_year": boot["ci_95_low_pct"],
        "ci95_high_pp_per_year": boot["ci_95_high_pct"],
        "probability_lift_gt_zero": boot["probability_lift_gt_zero"],
        "positive_week_fraction_pct": float((delta > 0.0).mean() * 100.0),
        "delta": pd.DataFrame({"date": joined["date"], "delta": delta}),
        "joined": joined,
    }


def _replica_block_bootstrap(
    delta_matrix: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    block_size: int = 4,
) -> dict[str, Any]:
    """Resample both independent scoring runs and circular weekly blocks."""
    values = np.asarray(delta_matrix, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise diagnostics.ResearchDiagnosticError("Cannot bootstrap an empty replica matrix")
    n_dates, n_replicas = values.shape
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot), dtype=float)
    for draw_idx in range(int(n_boot)):
        date_indices: list[int] = []
        while len(date_indices) < n_dates:
            start = int(rng.integers(0, n_dates))
            date_indices.extend((start + offset) % n_dates for offset in range(block_size))
        replica_indices = rng.integers(0, n_replicas, size=n_replicas)
        draws[draw_idx] = float(
            values[np.asarray(date_indices[:n_dates])][:, replica_indices].mean()
            * PERIODS_PER_YEAR
            * 100.0
        )
    point = float(values.mean() * PERIODS_PER_YEAR * 100.0)
    return {
        "n_boot": int(n_boot),
        "block_size": int(block_size),
        "n_replicas": int(n_replicas),
        "mean_annualized_lift_pct": point,
        "ci_95_low_pct": float(np.quantile(draws, 0.025)),
        "ci_95_high_pct": float(np.quantile(draws, 0.975)),
        "probability_lift_gt_zero": float(np.mean(draws > 0.0)),
    }


def _selection_attribution(
    run: dict[str, Any],
    trail5_weekly: pd.DataFrame,
    *,
    replica: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    treatment = run["selections"].copy()
    control = run["trail_selections"].copy()
    treatment["date"] = pd.to_datetime(treatment["date"]).dt.normalize()
    control["date"] = pd.to_datetime(control["date"]).dt.normalize()
    treatment["pool_side"] = treatment["pool_side"].astype(str).str.upper()
    control["pool_side"] = control["pool_side"].astype(str).str.upper()
    regime = trail5_weekly[["date", "regime"]].copy()
    regime["date"] = pd.to_datetime(regime["date"]).dt.normalize()
    regime_map = regime.set_index("date")["regime"].to_dict()
    by_key_t = {
        key: group for key, group in treatment.groupby(["date", "pool_side"], sort=True)
    }
    by_key_c = {
        key: group for key, group in control.groupby(["date", "pool_side"], sort=True)
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(set(by_key_t) & set(by_key_c)):
        date, side = key
        t_group = by_key_t[key]
        c_group = by_key_c[key]
        t_names = set(t_group["ticker"].astype(str))
        c_names = set(c_group["ticker"].astype(str))
        added = sorted(t_names - c_names)
        dropped = sorted(c_names - t_names)
        if len(added) != len(dropped):
            raise diagnostics.ResearchDiagnosticError(
                f"Unequal added/dropped names on {date} {side}"
            )
        sign = -1.0 if side == "SHORT" else 1.0
        t_group = t_group.assign(ticker=t_group["ticker"].astype(str))
        c_group = c_group.assign(ticker=c_group["ticker"].astype(str))
        # Selection artifacts store returns in percentage units (e.g. 2.0 means 2%).
        t_group["realized_c2c_return_pct"] = pd.to_numeric(
            t_group["realized_c2c_return_pct"], errors="coerce"
        )
        c_group["realized_c2c_return_pct"] = pd.to_numeric(
            c_group["realized_c2c_return_pct"], errors="coerce"
        )
        added_rows = t_group[t_group["ticker"].isin(added)]
        dropped_rows = c_group[c_group["ticker"].isin(dropped)]
        add_returns = added_rows["realized_c2c_return_pct"].dropna().to_numpy(float)
        drop_returns = dropped_rows["realized_c2c_return_pct"].dropna().to_numpy(float)
        add_side = sign * add_returns
        drop_side = sign * drop_returns
        treatment_valid_n = int(t_group["realized_c2c_return_pct"].notna().sum())
        control_valid_n = int(c_group["realized_c2c_return_pct"].notna().sum())
        if treatment_valid_n == 0 or control_valid_n == 0:
            raise diagnostics.ResearchDiagnosticError(
                f"No valid realized returns on {date} {side}"
            )
        total_lift = 0.5 * sign * (
            t_group["realized_c2c_return_pct"].mean()
            - c_group["realized_c2c_return_pct"].mean()
        ) / 100.0
        changed_names_lift = (
            0.5
            * sign
            * (
                added_rows["realized_c2c_return_pct"].sum()
                / treatment_valid_n
                - dropped_rows["realized_c2c_return_pct"].sum()
                / control_valid_n
            )
            / 100.0
            if added
            else 0.0
        )
        rows.append(
            {
                "replica": replica,
                "date": pd.Timestamp(date),
                "year": int(pd.Timestamp(date).year),
                "regime": regime_map.get(date, "unknown"),
                "pool_side": side,
                "n_added": len(added),
                "n_dropped": len(dropped),
                "added_mean_raw_return_pct": float(add_returns.mean()) if len(added) else np.nan,
                "dropped_mean_raw_return_pct": float(drop_returns.mean()) if len(dropped) else np.nan,
                "added_mean_directional_return_pct": float(add_side.mean()) if len(added) else np.nan,
                "dropped_mean_directional_return_pct": float(drop_side.mean()) if len(dropped) else np.nan,
                "added_minus_dropped_directional_pct": float(add_side.mean() - drop_side.mean()) if len(added) else 0.0,
                "capital_lift": total_lift,
                "changed_names_only_capital_lift": changed_names_lift,
                "shared_name_reweight_capital_lift": total_lift - changed_names_lift,
                "treatment_valid_positions": treatment_valid_n,
                "trail5_valid_positions": control_valid_n,
                "added_names": ";".join(added),
                "dropped_names": ";".join(dropped),
            }
        )
    attribution = pd.DataFrame(rows)
    # Verify that name substitutions account for the full equal-weighted return lift.
    total_by_date = attribution.groupby("date", as_index=False)["capital_lift"].sum()
    l = _weekly_capital(run["weekly"])
    b = _weekly_capital(run["trail_weekly"])
    actual = l.merge(b, on="date", suffixes=("_llm", "_trail"), validate="one_to_one")
    actual = actual.merge(total_by_date, on="date", validate="one_to_one")
    if not np.allclose(
        actual["capital_return_llm"] - actual["capital_return_trail"],
        actual["capital_lift"],
        atol=1e-10,
        rtol=0.0,
    ):
        raise diagnostics.ResearchDiagnosticError(
            f"Added-versus-dropped attribution does not reconcile to weekly return for {replica}"
        )
    return attribution, total_by_date


def _summarize_groups(
    frame: pd.DataFrame,
    group_col: str,
    *,
    output_col: str = "capital_lift",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(group_col, dropna=False, sort=True):
        lifted = pd.to_numeric(group[output_col], errors="coerce").dropna()
        rows.append(
            {
                group_col: value,
                "n_dates": int(len(lifted)),
                "annualized_lift_pp_per_year": float(lifted.mean() * PERIODS_PER_YEAR * 100.0),
                "positive_date_fraction_pct": float((lifted > 0.0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def _trimmed_lift(delta: pd.Series, trim_fraction: float = 0.05) -> tuple[float, float]:
    ordered = pd.to_numeric(delta, errors="coerce").dropna().sort_values(ascending=False)
    count = max(1, int(math.ceil(len(ordered) * trim_fraction)))
    top_share = float(ordered.iloc[:count].sum() / ordered.sum() * 100.0) if ordered.sum() else np.nan
    remaining = ordered.iloc[count:]
    trimmed = float(remaining.mean() * PERIODS_PER_YEAR * 100.0) if len(remaining) else np.nan
    return trimmed, top_share


def analyze(
    roots: dict[str, Path],
    output_dir: Path,
    *,
    n_boot: int = 5000,
) -> dict[str, Any]:
    loaded = {name: _read_run(root) for name, root in roots.items()}
    required_config = (
        "prompt_hash",
        "prompt_source",
        "temperature",
        "model",
        "thinking_mode",
        "thinking_call_policy",
        "llm_max_output_tokens",
        "llm_max_attempts",
        "output_contract",
        "prompt_variant",
        "prompt_version",
        "anonymize_tickers",
        "schedule",
        "anchor_offsets",
        "pool_sizes",
        "book_sizes",
        "llm_pool_size",
        "cost_bps_levels",
    )
    reference_config = loaded[next(iter(loaded))]["config"]
    construction_fields = (
        "candidate_pool_per_side",
        "selected_per_side",
        "long_short_capital_split",
        "within_side_weighting",
        "score_proportional_weighting",
        "concentrated_shorts",
        "ticker_visibility",
    )
    reference_construction = reference_config.get("incumbent_spec", {})
    for name, run in loaded.items():
        config = run["config"]
        for key in required_config:
            if config.get(key) != reference_config.get(key):
                raise diagnostics.ResearchDiagnosticError(
                    f"{name} config differs from replica 1 on {key}: "
                    f"{config.get(key)!r} != {reference_config.get(key)!r}"
                )
        construction = config.get("incumbent_spec", {})
        for key in construction_fields:
            if construction.get(key) != reference_construction.get(key):
                raise diagnostics.ResearchDiagnosticError(
                    f"{name} portfolio construction differs on {key}"
                )
        if config.get("prompt_hash") != "5300b8966c29":
            raise diagnostics.ResearchDiagnosticError(
                f"{name} does not use the frozen minimal-change prompt hash"
            )

    first_name = next(iter(loaded))
    first = loaded[first_name]
    canonical = _canonical_checks(first["checks"])
    canonical_trail = _canonical_checks(first["trail_checks"])
    for name, run in list(loaded.items())[1:]:
        _assert_frames_match(first["checks"], run["checks"], f"{name} treatment panel")
        _assert_frames_match(first["trail_checks"], run["trail_checks"], f"{name} Trail5 panel")
    _assert_frames_match(first["checks"], first["trail_checks"], "treatment versus Trail5 candidate panel")

    base_weekly = _weekly_capital(first["trail_weekly"])
    for name, run in loaded.items():
        other_base = _weekly_capital(run["trail_weekly"])
        paired_base = base_weekly.merge(other_base, on="date", suffixes=("_a", "_b"), validate="one_to_one")
        if len(paired_base) != len(base_weekly) or not np.allclose(
            paired_base["capital_return_a"], paired_base["capital_return_b"], atol=1e-12, rtol=0.0
        ):
            raise diagnostics.ResearchDiagnosticError(f"{name} has a different matched Trail5 series")

    tabular_root = PROJECT_ROOT / TABULAR_DIR
    tabular_predictions = pd.read_csv(tabular_root / "walk_forward_predictions.csv")
    tabular_predictions["date"] = pd.to_datetime(tabular_predictions["date"]).dt.normalize()
    tabular_predictions["pool_side"] = tabular_predictions["pool_side"].astype(str).str.upper()
    tabular_predictions["ticker"] = tabular_predictions["ticker"].astype(str)
    model_panel = tabular_predictions[
        [
            "date",
            "pool_side",
            "ticker",
            "fwd_c2c",
            *FEATURES,
            "score_ridge",
            "score_hist_gradient_boosting",
        ]
    ].copy()
    # The reused predictions must cover exactly the common warm-up-complete panel.
    model_keys = model_panel[["date", "pool_side", "ticker"]].sort_values(
        ["date", "pool_side", "ticker"]
    ).reset_index(drop=True)
    first_eval = canonical[canonical["date"].isin(model_keys["date"].unique())].copy()
    first_eval_keys = first_eval[["date", "pool_side", "ticker"]].sort_values(
        ["date", "pool_side", "ticker"]
    ).reset_index(drop=True)
    if not first_eval_keys.equals(model_keys):
        raise diagnostics.ResearchDiagnosticError(
            "The existing tabular predictions do not cover the exact minimal-change evaluation pools"
        )
    model_join = first_eval.merge(
        model_panel,
        on=["date", "pool_side", "ticker"],
        suffixes=("_minimal", "_tabular"),
        validate="one_to_one",
    )
    if not np.allclose(
        model_join["fwd_c2c_minimal"],
        model_join["fwd_c2c_tabular"],
        atol=1e-12,
        rtol=0.0,
        equal_nan=True,
    ):
        raise diagnostics.ResearchDiagnosticError("Tabular baseline forward returns do not match")
    for feature in FEATURES:
        if not np.allclose(
            model_join[f"{feature}_minimal"],
            model_join[f"{feature}_tabular"],
            atol=1e-12,
            rtol=0.0,
            equal_nan=True,
        ):
            raise diagnostics.ResearchDiagnosticError(
                f"Tabular baseline candidate features do not match in {feature}"
            )
    if "regime" in tabular_predictions.columns:
        model_regime = tabular_predictions[["date", "regime"]].drop_duplicates()
        if model_regime["date"].duplicated().any():
            raise diagnostics.ResearchDiagnosticError(
                "Tabular baseline has inconsistent regime labels within a date"
            )
        baseline_regime = first["trail_weekly"][["date", "regime"]].copy()
        baseline_regime["date"] = pd.to_datetime(baseline_regime["date"]).dt.normalize()
        regime_pair = model_regime.merge(
            baseline_regime,
            on="date",
            how="left",
            suffixes=("_tabular", "_minimal"),
            validate="one_to_one",
        )
        if regime_pair["regime_tabular"].astype(str).ne(
            regime_pair["regime_minimal"].astype(str)
        ).any():
            raise diagnostics.ResearchDiagnosticError(
                "Tabular baseline market-regime context differs from the minimal-change panel"
            )
    leakage_audit = pd.read_csv(tabular_root / "leakage_audit.csv")
    if not leakage_audit["strictly_past"].astype(bool).all():
        raise diagnostics.ResearchDiagnosticError("Prior-data model leakage audit failed")

    evaluation_dates = sorted(model_panel["date"].unique())
    base_common = base_weekly[base_weekly["date"].isin(evaluation_dates)].reset_index(drop=True)
    if len(base_common) != len(evaluation_dates):
        raise diagnostics.ResearchDiagnosticError("Matched Trail5 misses model evaluation dates")
    rows: list[dict[str, Any]] = []
    delta_by_model: dict[str, pd.DataFrame] = {}
    seed_idx = 0
    for name, run in loaded.items():
        weekly = _weekly_capital(run["weekly"])
        weekly = weekly[weekly["date"].isin(evaluation_dates)].reset_index(drop=True)
        stats = _paired_stats(
            weekly,
            base_common,
            label=name,
            n_boot=n_boot,
            seed=3101 + seed_idx,
        )
        seed_idx += 1
        rows.append({k: v for k, v in stats.items() if k not in {"delta", "joined"}})
        delta_by_model[name] = stats["delta"]

    model_frame = first_eval.merge(
        model_panel,
        on=["date", "pool_side", "ticker", "fwd_c2c"],
        validate="one_to_one",
    )
    model_frame["score_ridge"] = pd.to_numeric(model_frame["score_ridge"], errors="coerce")
    model_frame["score_hist_gradient_boosting"] = pd.to_numeric(
        model_frame["score_hist_gradient_boosting"], errors="coerce"
    )
    baseline_panels: list[tuple[str, pd.DataFrame]] = [
        ("simple_trail5_rsi5_equal_rank", _rank_combo(first_eval, ("feature_Trail5", "feature_RSI5"), "simple_trail5_rsi5_equal_rank")),
        ("simple_trail5_rsi5_rng55_equal_rank", _rank_combo(first_eval, ("feature_Trail5", "feature_RSI5", "feature_Rng55"), "simple_trail5_rsi5_rng55_equal_rank")),
    ]
    for score in ("score_ridge", "score_hist_gradient_boosting"):
        baseline_panels.append((score.removeprefix("score_"), model_frame[["date", "pool_side", "ticker", "fwd_c2c", score]].rename(columns={score: "score"})))
    for label, panel in baseline_panels:
        candidate_weekly = _weekly_from_ranks(panel, "score")
        stats = _paired_stats(
            candidate_weekly,
            base_common,
            label=label,
            n_boot=n_boot,
            seed=4101 + seed_idx,
        )
        seed_idx += 1
        rows.append({k: v for k, v in stats.items() if k not in {"delta", "joined"}})
        delta_by_model[label] = stats["delta"]

    trail_metrics_common = diagnostics.return_metrics(base_common["capital_return"])
    rows.append(
        {
            "model": "Trail5_matched_control",
            "n_dates": int(len(base_common)),
            "annual_return_pct": trail_metrics_common["annual_return_pct"],
            "cagr_pct": trail_metrics_common["cagr_pct"],
            "sharpe": trail_metrics_common["sharpe"],
            "max_drawdown_pct": trail_metrics_common["max_drawdown_pct"],
            "trail5_annual_return_pct": trail_metrics_common["annual_return_pct"],
            "paired_lift_vs_trail5_pp_per_year": 0.0,
            "ci95_low_pp_per_year": np.nan,
            "ci95_high_pp_per_year": np.nan,
            "probability_lift_gt_zero": np.nan,
            "positive_week_fraction_pct": np.nan,
        }
    )

    # A transparent average across all predeclared replicas; no winner is chosen.
    rep_delta_frames = [
        frame.rename(columns={"delta": name})
        for name, frame in delta_by_model.items()
        if name in loaded
    ]
    rep_delta_panel = rep_delta_frames[0]
    for frame in rep_delta_frames[1:]:
        rep_delta_panel = rep_delta_panel.merge(
            frame, on="date", how="inner", validate="one_to_one"
        )
    rep_delta_panel["delta"] = rep_delta_panel[list(loaded)].mean(axis=1)
    rep_boot = _replica_block_bootstrap(
        rep_delta_panel[list(loaded)].to_numpy(float), n_boot=n_boot, seed=5101
    )
    replica_point_lifts = [
        float(frame["delta"].mean() * PERIODS_PER_YEAR * 100.0)
        for name, frame in delta_by_model.items()
        if name in loaded
    ]
    aggregate_row = {
        "model": "replica_mean_no_selection",
        "n_dates": int(len(rep_delta_panel)),
        "annual_return_pct": float(
            np.mean(
                [
                    row["annual_return_pct"]
                    for row in rows
                    if row["model"] in loaded
                ]
            )
        ),
        "cagr_pct": np.nan,
        "sharpe": np.nan,
        "max_drawdown_pct": np.nan,
        "trail5_annual_return_pct": diagnostics.return_metrics(base_common["capital_return"])["annual_return_pct"],
        "paired_lift_vs_trail5_pp_per_year": rep_boot["mean_annualized_lift_pct"],
        "ci95_low_pp_per_year": rep_boot["ci_95_low_pct"],
        "ci95_high_pp_per_year": rep_boot["ci_95_high_pct"],
        "probability_lift_gt_zero": rep_boot["probability_lift_gt_zero"],
        "positive_week_fraction_pct": float((rep_delta_panel["delta"] > 0.0).mean() * 100.0),
        "replica_lift_sd_pp_per_year": float(np.std(replica_point_lifts, ddof=1)),
        "replica_lift_min_pp_per_year": float(np.min(replica_point_lifts)),
        "replica_lift_max_pp_per_year": float(np.max(replica_point_lifts)),
    }
    rows.append(aggregate_row)
    comparison = pd.DataFrame(rows)

    # Estimate each row's pairwise uncertainty against the unselected replica mean.
    # Build the equally weighted treatment stream from the three replicas on the common model window.
    treatment_streams = []
    for name, run in loaded.items():
        stream = _weekly_capital(run["weekly"])
        stream = stream[stream["date"].isin(evaluation_dates)].rename(
            columns={"capital_return": name}
        )
        treatment_streams.append(stream)
    mean_treatment = treatment_streams[0]
    for stream in treatment_streams[1:]:
        mean_treatment = mean_treatment.merge(stream, on="date", validate="one_to_one")
    mean_treatment["capital_return"] = mean_treatment[list(loaded)].mean(axis=1)
    mean_treatment = mean_treatment[["date", "capital_return"]]
    treatment_matrix_frame = treatment_streams[0]
    for stream in treatment_streams[1:]:
        treatment_matrix_frame = treatment_matrix_frame.merge(
            stream, on="date", validate="one_to_one"
        )
    treatment_matrix = treatment_matrix_frame[list(loaded)].to_numpy(float)
    comparison_vs_llm: list[dict[str, Any]] = []
    for label, delta_frame in delta_by_model.items():
        base_return = base_common.rename(columns={"capital_return": "trail"})
        candidate_stream = delta_frame.merge(base_return, on="date", validate="one_to_one")
        candidate_stream["capital_return"] = candidate_stream["delta"] + candidate_stream["trail"]
        candidate_stream = candidate_stream.sort_values("date").reset_index(drop=True)
        if not candidate_stream["date"].equals(treatment_matrix_frame["date"]):
            raise diagnostics.ResearchDiagnosticError(
                f"{label} dates do not match the replica average"
            )
        pair_boot = _replica_block_bootstrap(
            candidate_stream["capital_return"].to_numpy(float)[:, None]
            - treatment_matrix,
            n_boot=n_boot,
            seed=6101 + len(comparison_vs_llm),
        )
        comparison_vs_llm.append(
            {
                "model": label,
                "paired_lift_vs_replica_mean_pp_per_year": pair_boot["mean_annualized_lift_pct"],
                "ci95_low_vs_replica_mean_pp_per_year": pair_boot["ci_95_low_pct"],
                "ci95_high_vs_replica_mean_pp_per_year": pair_boot["ci_95_high_pct"],
            }
        )
    comparison = comparison.merge(pd.DataFrame(comparison_vs_llm), on="model", how="left")

    trail_weekly_full = first["trail_weekly"].copy()
    trail_weekly_full["date"] = pd.to_datetime(trail_weekly_full["date"]).dt.normalize()
    attribution_parts: list[pd.DataFrame] = []
    lift_parts: list[pd.DataFrame] = []
    replica_summary: dict[str, Any] = {}
    attribution_ci_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    year_side_rows: list[dict[str, Any]] = []
    regime_side_rows: list[dict[str, Any]] = []
    full_replica_rows: list[dict[str, Any]] = []
    full_delta_frames: list[pd.DataFrame] = []
    for idx, (name, run) in enumerate(loaded.items()):
        attr, daily_lift = _selection_attribution(run, trail_weekly_full, replica=name)
        attribution_parts.append(attr)
        lift_parts.append(daily_lift.assign(replica=name))
        actual_full = _weekly_capital(run["weekly"]).merge(
            _weekly_capital(run["trail_weekly"]), on="date", suffixes=("_llm", "_trail"), validate="one_to_one"
        )
        delta = actual_full["capital_return_llm"] - actual_full["capital_return_trail"]
        boot = diagnostics.moving_block_bootstrap(delta, n_boot=n_boot, block_size=4, seed=7101 + idx)
        trimmed, top_share = _trimmed_lift(delta)
        replica_summary[name] = {
            "n_dates": int(len(actual_full)),
            "annual_lift_pp_per_year": boot["mean_annualized_lift_pct"],
            "ci95_low_pp_per_year": boot["ci_95_low_pct"],
            "ci95_high_pp_per_year": boot["ci_95_high_pct"],
            "probability_lift_gt_zero": boot["probability_lift_gt_zero"],
            "positive_week_fraction_pct": float((delta > 0).mean() * 100.0),
            "lift_after_removing_best_5pct_dates_pp_per_year": trimmed,
            "best_5pct_dates_share_of_net_lift_pct": top_share,
        }
        full_candidate_metrics = diagnostics.return_metrics(actual_full["capital_return_llm"])
        full_control_metrics = diagnostics.return_metrics(actual_full["capital_return_trail"])
        full_replica_rows.append(
            {
                "model": name,
                "n_dates": len(actual_full),
                "annual_return_pct": full_candidate_metrics["annual_return_pct"],
                "cagr_pct": full_candidate_metrics["cagr_pct"],
                "trail5_annual_return_pct": full_control_metrics["annual_return_pct"],
                "paired_lift_vs_trail5_pp_per_year": boot["mean_annualized_lift_pct"],
                "ci95_low_pp_per_year": boot["ci_95_low_pct"],
                "ci95_high_pp_per_year": boot["ci_95_high_pct"],
                "probability_lift_gt_zero": boot["probability_lift_gt_zero"],
                "positive_week_fraction_pct": float((delta > 0.0).mean() * 100.0),
                "lift_after_removing_best_5pct_dates_pp_per_year": trimmed,
                "best_5pct_dates_share_of_net_lift_pct": top_share,
            }
        )
        full_delta_frames.append(
            actual_full[["date"]].assign(**{name: delta.to_numpy(float)})
        )
        # Attribution by side, including a block interval for side contribution.
        for side in ("LONG", "SHORT"):
            side_frame = attr[attr["pool_side"].eq(side)]
            side_lift = side_frame.groupby("date", as_index=False)["capital_lift"].sum()
            changed_lift = side_frame.groupby("date", as_index=False)[
                "changed_names_only_capital_lift"
            ].sum()
            side_boot = diagnostics.moving_block_bootstrap(
                side_lift["capital_lift"], n_boot=n_boot, block_size=4, seed=8101 + idx * 2 + (side == "SHORT")
            )
            changed_boot = diagnostics.moving_block_bootstrap(
                changed_lift["changed_names_only_capital_lift"],
                n_boot=n_boot,
                block_size=4,
                seed=8201 + idx * 2 + (side == "SHORT"),
            )
            attribution_ci_rows.append(
                {
                    "replica": name,
                    "pool_side": side,
                    "n_dates": len(side_lift),
                    "mean_added_directional_return_pct": float(side_frame["added_mean_directional_return_pct"].mean()),
                    "mean_dropped_directional_return_pct": float(side_frame["dropped_mean_directional_return_pct"].mean()),
                    "annualized_capital_contribution_pp": side_boot["mean_annualized_lift_pct"],
                    "ci95_low_pp": side_boot["ci_95_low_pct"],
                    "ci95_high_pp": side_boot["ci_95_high_pct"],
                    "added_dropped_only_contribution_pp": changed_boot["mean_annualized_lift_pct"],
                    "added_dropped_only_ci95_low_pp": changed_boot["ci_95_low_pct"],
                    "added_dropped_only_ci95_high_pp": changed_boot["ci_95_high_pct"],
                    "shared_name_reweight_contribution_pp": float(
                        side_frame["shared_name_reweight_capital_lift"].mean()
                        * PERIODS_PER_YEAR
                        * 100.0
                    ),
                    "positive_substitution_date_fraction_pct": float((side_lift["capital_lift"] > 0).mean() * 100.0),
                }
            )
        by_year = daily_lift.merge(
            pd.DataFrame({"date": daily_lift["date"], "year": daily_lift["date"].dt.year}),
            on="date",
            validate="one_to_one",
        )
        for year, group in by_year.groupby("year", sort=True):
            year_rows.append(
                {
                    "replica": name,
                    "year": int(year),
                    "n_dates": len(group),
                    "annualized_lift_pp_per_year": float(group["capital_lift"].mean() * PERIODS_PER_YEAR * 100.0),
                    "positive_date_fraction_pct": float((group["capital_lift"] > 0.0).mean() * 100.0),
                }
            )
        for (side, year), group in attr.groupby(["pool_side", "year"], sort=True):
            year_side_rows.append(
                {
                    "replica": name,
                    "pool_side": side,
                    "year": int(year),
                    "n_dates": len(group),
                    "annualized_lift_pp_per_year": float(group["capital_lift"].mean() * PERIODS_PER_YEAR * 100.0),
                    "positive_date_fraction_pct": float((group["capital_lift"] > 0.0).mean() * 100.0),
                }
            )
        regime_group = daily_lift.merge(
            trail_weekly_full[["date", "regime"]], on="date", validate="one_to_one"
        )
        for regime_value, group in regime_group.groupby("regime", dropna=False, sort=True):
            regime_rows.append(
                {
                    "replica": name,
                    "regime": regime_value,
                    "n_dates": len(group),
                    "annualized_lift_pp_per_year": float(group["capital_lift"].mean() * PERIODS_PER_YEAR * 100.0),
                    "positive_date_fraction_pct": float((group["capital_lift"] > 0.0).mean() * 100.0),
                }
            )
        for (side, regime_value), group in attr.groupby(["pool_side", "regime"], dropna=False, sort=True):
            regime_side_rows.append(
                {
                    "replica": name,
                    "pool_side": side,
                    "regime": regime_value,
                    "n_dates": len(group),
                    "annualized_lift_pp_per_year": float(group["capital_lift"].mean() * PERIODS_PER_YEAR * 100.0),
                    "positive_date_fraction_pct": float((group["capital_lift"] > 0.0).mean() * 100.0),
                }
            )

    attribution = pd.concat(attribution_parts, ignore_index=True)
    daily_lifts = pd.concat(lift_parts, ignore_index=True)
    for side in ("LONG", "SHORT"):
        side_replica_frames = []
        changed_replica_frames = []
        for replica in loaded:
            side_dates = attribution[
                attribution["replica"].eq(replica) & attribution["pool_side"].eq(side)
            ].groupby("date", as_index=False)[
                ["capital_lift", "changed_names_only_capital_lift", "shared_name_reweight_capital_lift"]
            ].sum()
            side_replica_frames.append(
                side_dates[["date", "capital_lift"]].rename(columns={"capital_lift": replica})
            )
            changed_replica_frames.append(
                side_dates[["date", "changed_names_only_capital_lift"]].rename(
                    columns={"changed_names_only_capital_lift": replica}
                )
            )
        total_matrix_frame = side_replica_frames[0]
        changed_matrix_frame = changed_replica_frames[0]
        for frame in side_replica_frames[1:]:
            total_matrix_frame = total_matrix_frame.merge(frame, on="date", validate="one_to_one")
        for frame in changed_replica_frames[1:]:
            changed_matrix_frame = changed_matrix_frame.merge(frame, on="date", validate="one_to_one")
        total_matrix = total_matrix_frame[list(loaded)].to_numpy(float)
        changed_matrix = changed_matrix_frame[list(loaded)].to_numpy(float)
        total_boot = _replica_block_bootstrap(
            total_matrix, n_boot=n_boot, seed=10101 + (side == "SHORT")
        )
        changed_boot = _replica_block_bootstrap(
            changed_matrix, n_boot=n_boot, seed=10201 + (side == "SHORT")
        )
        mean_side_rows = attribution[attribution["pool_side"].eq(side)]
        attribution_ci_rows.append(
            {
                "replica": "replica_mean_no_selection",
                "pool_side": side,
                "n_dates": int(total_matrix.shape[0]),
                "mean_added_directional_return_pct": float(
                    mean_side_rows["added_mean_directional_return_pct"].mean()
                ),
                "mean_dropped_directional_return_pct": float(
                    mean_side_rows["dropped_mean_directional_return_pct"].mean()
                ),
                "annualized_capital_contribution_pp": total_boot["mean_annualized_lift_pct"],
                "ci95_low_pp": total_boot["ci_95_low_pct"],
                "ci95_high_pp": total_boot["ci_95_high_pct"],
                "added_dropped_only_contribution_pp": changed_boot["mean_annualized_lift_pct"],
                "added_dropped_only_ci95_low_pp": changed_boot["ci_95_low_pct"],
                "added_dropped_only_ci95_high_pp": changed_boot["ci_95_high_pct"],
                "shared_name_reweight_contribution_pp": float(
                    mean_side_rows["shared_name_reweight_capital_lift"].mean()
                    * PERIODS_PER_YEAR
                    * 100.0
                ),
                "positive_substitution_date_fraction_pct": float(
                    (total_matrix.mean(axis=1) > 0.0).mean() * 100.0
                ),
            }
        )
    side_summary = pd.DataFrame(attribution_ci_rows)
    by_year = pd.DataFrame(year_rows)
    by_regime = pd.DataFrame(regime_rows)
    by_year_side = pd.DataFrame(year_side_rows)
    by_regime_side = pd.DataFrame(regime_side_rows)
    full_delta_panel = full_delta_frames[0]
    for frame in full_delta_frames[1:]:
        full_delta_panel = full_delta_panel.merge(frame, on="date", validate="one_to_one")
    full_delta_panel["delta"] = full_delta_panel[list(loaded)].mean(axis=1)
    full_mean_boot = _replica_block_bootstrap(
        full_delta_panel[list(loaded)].to_numpy(float), n_boot=n_boot, seed=9101
    )
    full_control_metrics = diagnostics.return_metrics(_weekly_capital(first["trail_weekly"])["capital_return"])
    full_replica_rows.append(
        {
            "model": "Trail5_matched_control",
            "n_dates": int(len(full_delta_panel)),
            "annual_return_pct": full_control_metrics["annual_return_pct"],
            "cagr_pct": full_control_metrics["cagr_pct"],
            "trail5_annual_return_pct": full_control_metrics["annual_return_pct"],
            "paired_lift_vs_trail5_pp_per_year": 0.0,
            "ci95_low_pp_per_year": np.nan,
            "ci95_high_pp_per_year": np.nan,
            "probability_lift_gt_zero": np.nan,
            "positive_week_fraction_pct": np.nan,
            "lift_after_removing_best_5pct_dates_pp_per_year": np.nan,
            "best_5pct_dates_share_of_net_lift_pct": np.nan,
        }
    )
    full_replica_rows.append(
        {
            "model": "replica_mean_no_selection",
            "n_dates": int(len(full_delta_panel)),
            "annual_return_pct": float(
                np.mean([row["annual_return_pct"] for row in full_replica_rows if row["model"] in loaded])
            ),
            "cagr_pct": np.nan,
            "trail5_annual_return_pct": full_control_metrics["annual_return_pct"],
            "paired_lift_vs_trail5_pp_per_year": full_mean_boot["mean_annualized_lift_pct"],
            "ci95_low_pp_per_year": full_mean_boot["ci_95_low_pct"],
            "ci95_high_pp_per_year": full_mean_boot["ci_95_high_pct"],
            "probability_lift_gt_zero": full_mean_boot["probability_lift_gt_zero"],
            "positive_week_fraction_pct": float((full_delta_panel["delta"] > 0.0).mean() * 100.0),
            "lift_after_removing_best_5pct_dates_pp_per_year": _trimmed_lift(full_delta_panel["delta"])[0],
            "best_5pct_dates_share_of_net_lift_pct": _trimmed_lift(full_delta_panel["delta"])[1],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "comparison_table.csv", index=False)
    pd.DataFrame(full_replica_rows).to_csv(output_dir / "replica_full_window.csv", index=False)
    attribution.to_csv(output_dir / "added_dropped_attribution.csv", index=False)
    side_summary.to_csv(output_dir / "attribution_by_side.csv", index=False)
    by_year.to_csv(output_dir / "attribution_by_year.csv", index=False)
    by_regime.to_csv(output_dir / "attribution_by_market_regime.csv", index=False)
    by_year_side.to_csv(output_dir / "attribution_by_year_and_side.csv", index=False)
    by_regime_side.to_csv(output_dir / "attribution_by_regime_and_side.csv", index=False)
    daily_lifts.to_csv(output_dir / "paired_lifts_by_date.csv", index=False)
    _json_write(
        output_dir / "summary.json",
        {
            "configuration": {key: reference_config.get(key) for key in required_config},
            "portfolio_construction": {
                key: reference_construction.get(key) for key in construction_fields
            },
            "backend_revision_recorded": all(
                bool(run["config"].get("backend_revision")) for run in loaded.values()
            ),
            "replica_count": len(loaded),
            "replicas": replica_summary,
            "common_baseline_evaluation_dates": len(evaluation_dates),
            "full_replica_dates": int(len(first["weekly"])),
            "tabular_source": str(TABULAR_DIR / "walk_forward_predictions.csv"),
            "tabular_all_training_strictly_past": bool(leakage_audit["strictly_past"].astype(bool).all()),
            "tabular_panel_verified_exact": True,
            "bootstrap": {
                "n_boot": n_boot,
                "block_size_weeks": 4,
                "replica_mean_ci_resamples_scoring_runs": True,
            },
            "return_lift_unit": "annualized arithmetic capital return percentage points",
            "simple_baselines": [
                "Trail5 + RSI5, equal within-side percentile ranks",
                "Trail5 + RSI5 + Rng55, equal within-side percentile ranks",
            ],
            "replica_average": {
                "annualized_lift_pp_per_year": rep_boot["mean_annualized_lift_pct"],
                "ci95_low_pp_per_year": rep_boot["ci_95_low_pct"],
                "ci95_high_pp_per_year": rep_boot["ci_95_high_pct"],
                "between_replica_sd_pp_per_year": float(np.std(replica_point_lifts, ddof=1)),
                "individual_lifts_pp_per_year": replica_point_lifts,
            },
        },
    )
    return {
        "comparison_table": str(output_dir / "comparison_table.csv"),
        "replica_full_window": str(output_dir / "replica_full_window.csv"),
        "attribution_by_side": str(output_dir / "attribution_by_side.csv"),
        "attribution_by_year": str(output_dir / "attribution_by_year.csv"),
        "attribution_by_market_regime": str(output_dir / "attribution_by_market_regime.csv"),
        "attribution_by_year_and_side": str(output_dir / "attribution_by_year_and_side.csv"),
        "attribution_by_regime_and_side": str(output_dir / "attribution_by_regime_and_side.csv"),
        "summary": str(output_dir / "summary.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("backtest_results/minimal_change_replica_analysis_20261027"))
    parser.add_argument("--n-boot", type=int, default=5000)
    for name, default in DEFAULT_REPLICAS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    parser.add_argument("--replica-3", type=Path, default=DEFAULT_REPLICA_3)
    args = parser.parse_args()
    roots = {
        name: getattr(args, name).resolve()
        for name in DEFAULT_REPLICAS
    }
    replica_3 = args.replica_3.resolve()
    if (replica_3 / "run_config.json").is_file():
        roots["replica_3"] = replica_3
    result = analyze(roots, args.output_dir, n_boot=args.n_boot)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
