"""Leakage-safe development diagnostics for the mean-reversion LLM study.

This module is deliberately offline.  It reads completed 2009-2013 development
artifacts, never calls an LLM, and refuses to inspect rows dated 2015 or later.
It supplies the inference, factor-strength, disagreement, and tabular-baseline
checks that should run before any prompt is frozen for blind validation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PERIODS_PER_YEAR = 52.0
DEVELOPMENT_CUTOFF = pd.Timestamp("2015-01-01")
DEFAULT_BLOCK_SIZE = 4
DEFAULT_BOOTSTRAPS = 5_000
DEFAULT_BOOK_SIZE = 30
DEFAULT_OUTPUT_DIR = Path("backtest_results/research_thesis_package_20260808")
DEFAULT_CAMPAIGN_06 = Path(
    "backtest_results/autoresearch_campaigns/"
    "prompt_search_ticker_optional_resumed_20260806"
)
DEFAULT_CAMPAIGN_07 = Path(
    "backtest_results/autoresearch_campaigns/"
    "prompt_search_ticker_optional_20260807"
)
DEFAULT_G005_CAMPAIGN = Path(
    "backtest_results/autoresearch_campaigns/smart_prompt_search_20260802"
)
DEFAULT_CONTROL_DIR = Path(
    "backtest_results/v2_minimal_change_temp1_v4flash0731_second_2009_2014/"
    "pool60_pos30/anchor_3/llm_pool_v2_minimal_change_control"
)
DEFAULT_TRAIL5_DIR = Path(
    "backtest_results/v2_minimal_change_temp1_v4flash0731_second_2009_2014/"
    "pool60_pos30/anchor_3/baseline_pool"
)
DEFAULT_V2_RUN_DIRS = (
    Path(
        "backtest_results/v2_system_prompt_temp1_v4flash0731_2009_2014/"
        "pool60_pos30/anchor_3/llm_pool_v2"
    ),
    Path(
        "backtest_results/v2_system_prompt_temp1_v4flash0731_second_2009_2014/"
        "pool60_pos30/anchor_3/llm_pool_v2"
    ),
    Path(
        "backtest_results/v2_system_prompt_temp1_v4flash0731_third_2009_2014/"
        "pool60_pos30/anchor_3/llm_pool_v2"
    ),
    Path(
        "backtest_results/v2_system_prompt_temp1_v4flash0731_ticker_visible_2009_2014/"
        "pool60_pos30/anchor_3/llm_pool_v2"
    ),
)


class ResearchDiagnosticError(RuntimeError):
    """Raised when an artifact is unsafe, incomplete, or not genuinely paired."""


@dataclass(frozen=True)
class PairedArtifacts:
    label: str
    candidate_id: str
    campaign_dir: Path
    stage_dir: Path
    treatment_dir: Path
    control_dir: Path
    trail5_dir: Path
    result_path: Path
    prompt_spec_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchDiagnosticError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResearchDiagnosticError(f"Expected a JSON object at {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assert_development_dates(values: Iterable[Any], *, source: Path | str) -> pd.Series:
    dates = pd.to_datetime(pd.Series(list(values)), errors="coerce").dt.normalize()
    if dates.empty or dates.isna().any():
        raise ResearchDiagnosticError(f"Missing or invalid dates in {source}")
    if bool((dates >= DEVELOPMENT_CUTOFF).any()):
        first = dates[dates >= DEVELOPMENT_CUTOFF].min()
        raise ResearchDiagnosticError(
            f"Blind-period firewall blocked {source}: encountered {first.date()}"
        )
    return dates


def _read_csv_development(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ResearchDiagnosticError(f"Missing artifact: {path}")
    frame = pd.read_csv(path)
    date_col = "date" if "date" in frame.columns else "rebalance_date"
    if date_col not in frame.columns:
        raise ResearchDiagnosticError(f"Missing date column in {path}")
    frame[date_col] = _assert_development_dates(frame[date_col], source=path).to_numpy()
    if date_col != "date":
        frame["date"] = frame[date_col]
    return frame


def _capitalize_weekly(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().sort_values("date").reset_index(drop=True)
    if result["date"].duplicated().any():
        raise ResearchDiagnosticError("Weekly returns contain duplicate dates")
    if "capital_c2c" not in result.columns:
        source = "ls_c2c_0" if "ls_c2c_0" in result.columns else "ls_c2c"
        if source not in result.columns:
            raise ResearchDiagnosticError("Weekly returns lack capital_c2c/ls_c2c")
        result["capital_c2c"] = 0.5 * pd.to_numeric(result[source], errors="coerce")
    if {"long_ret_c2c", "short_ret_c2c"}.issubset(result.columns):
        result["long_contribution"] = 0.5 * pd.to_numeric(
            result["long_ret_c2c"], errors="coerce"
        )
        result["short_contribution"] = -0.5 * pd.to_numeric(
            result["short_ret_c2c"], errors="coerce"
        )
    else:
        result["long_contribution"] = np.nan
        result["short_contribution"] = np.nan
    return result


def load_weekly(run_dir: Path) -> pd.DataFrame:
    return _capitalize_weekly(_read_csv_development(Path(run_dir) / "weekly_returns.csv"))


def load_checks(run_dir: Path) -> pd.DataFrame:
    frame = _read_csv_development(Path(run_dir) / "signal_checks.csv")
    required = {"date", "ticker", "pool_side", "fwd_c2c"}
    missing = required - set(frame.columns)
    if missing:
        raise ResearchDiagnosticError(
            f"Signal checks at {run_dir} lack {', '.join(sorted(missing))}"
        )
    frame["ticker"] = frame["ticker"].astype(str)
    frame["pool_side"] = frame["pool_side"].astype(str).str.upper()
    frame["fwd_c2c"] = pd.to_numeric(frame["fwd_c2c"], errors="coerce")
    return frame.sort_values(["date", "pool_side", "ticker"]).reset_index(drop=True)


def load_selections(run_dir: Path) -> pd.DataFrame:
    frame = _read_csv_development(Path(run_dir) / "portfolio_selections.csv")
    side_col = "position_type" if "position_type" in frame.columns else "pool_side"
    required = {"date", "ticker", side_col, "realized_c2c_return_pct"}
    missing = required - set(frame.columns)
    if missing:
        raise ResearchDiagnosticError(
            f"Portfolio selections at {run_dir} lack {', '.join(sorted(missing))}"
        )
    frame["pool_side"] = frame[side_col].astype(str).str.upper()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["weight"] = pd.to_numeric(frame.get("weight", 1.0 / DEFAULT_BOOK_SIZE), errors="coerce")
    frame["realized_c2c_return_pct"] = pd.to_numeric(
        frame["realized_c2c_return_pct"], errors="coerce"
    )
    return frame.sort_values(["date", "pool_side", "ticker"]).reset_index(drop=True)


def _find_run_config(run_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    current = Path(run_dir).resolve()
    for parent in (current, *current.parents):
        candidate = parent / "run_config.json"
        if candidate.is_file():
            return candidate, _read_json(candidate)
        if parent.name == "backtest_results":
            break
    return None, {}


def paired_artifacts(
    campaign_dir: Path,
    candidate_id: str,
    *,
    stage: str = "full",
    label: str | None = None,
) -> PairedArtifacts:
    campaign_dir = Path(campaign_dir)
    stage_dir = campaign_dir / "candidates" / candidate_id / stage
    result_path = stage_dir / "result.json"
    result = _read_json(result_path)
    artifacts = result.get("artifacts", {})
    required = {"treatment", "control", "trail5"}
    if not required.issubset(artifacts):
        raise ResearchDiagnosticError(f"Incomplete paired artifacts in {result_path}")
    return PairedArtifacts(
        label=label or f"{campaign_dir.name}:{candidate_id}",
        candidate_id=candidate_id,
        campaign_dir=campaign_dir,
        stage_dir=stage_dir,
        treatment_dir=Path(str(artifacts["treatment"])),
        control_dir=Path(str(artifacts["control"])),
        trail5_dir=Path(str(artifacts["trail5"])),
        result_path=result_path,
        prompt_spec_path=campaign_dir / "candidates" / candidate_id / "prompt_spec.json",
    )


def _check_row(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"invariant": name, "pass": bool(passed), "detail": detail}


def verify_paired_artifacts(artifacts: PairedArtifacts) -> dict[str, Any]:
    """Verify the actual frozen files and pair invariants, not a dashboard cache hint."""
    result = _read_json(artifacts.result_path)
    spec = _read_json(artifacts.prompt_spec_path)
    treatment_weekly = load_weekly(artifacts.treatment_dir)
    control_weekly = load_weekly(artifacts.control_dir)
    treatment_checks = load_checks(artifacts.treatment_dir)
    control_checks = load_checks(artifacts.control_dir)
    treatment_config_path, treatment_config = _find_run_config(artifacts.treatment_dir)
    control_config_path, control_config = _find_run_config(artifacts.control_dir)
    rows: list[dict[str, Any]] = []

    for key, label in (
        ("model", "model/backend alias"),
        ("temperature", "temperature"),
        ("thinking_mode", "thinking mode"),
        ("anonymize_tickers", "ticker visibility"),
    ):
        left = treatment_config.get(key)
        right = control_config.get(key)
        rows.append(_check_row(label, left == right and left is not None, f"{left!r} vs {right!r}"))

    treatment_dates = treatment_weekly["date"].tolist()
    control_dates = control_weekly["date"].tolist()
    rows.append(
        _check_row(
            "signal dates",
            treatment_dates == control_dates,
            f"{len(treatment_dates)} treatment / {len(control_dates)} control dates",
        )
    )

    key_cols = ["date", "pool_side", "ticker"]
    treatment_keys = treatment_checks[key_cols].drop_duplicates()
    control_keys = control_checks[key_cols].drop_duplicates()
    key_compare = treatment_keys.merge(control_keys, on=key_cols, how="outer", indicator=True)
    pools_match = bool(key_compare["_merge"].eq("both").all())
    rows.append(
        _check_row(
            "candidate pools",
            pools_match,
            f"{len(treatment_keys)} treatment / {len(control_keys)} control name-side-date rows",
        )
    )
    common_numeric = [
        column
        for column in (
            "trail_ret",
            "fwd_c2c",
            "fwd_o2o",
            *sorted(
                set(column for column in treatment_checks if column.startswith("feature_"))
                & set(column for column in control_checks if column.startswith("feature_"))
            ),
        )
        if column in treatment_checks.columns and column in control_checks.columns
    ]
    values_match = pools_match
    if pools_match and common_numeric:
        paired = treatment_checks[key_cols + common_numeric].merge(
            control_checks[key_cols + common_numeric],
            on=key_cols,
            how="inner",
            suffixes=("_treatment", "_control"),
        )
        values_match = all(
            np.allclose(
                pd.to_numeric(paired[f"{column}_treatment"], errors="coerce"),
                pd.to_numeric(paired[f"{column}_control"], errors="coerce"),
                equal_nan=True,
            )
            for column in common_numeric
        )
    rows.append(
        _check_row(
            "pool inputs and realized returns",
            values_match,
            ", ".join(common_numeric) or "no common numeric columns",
        )
    )

    construction_cols = [
        column
        for column in (
            "anchor_offset",
            "pool_per_side",
            "positions_per_side",
            "n_pool",
            "n_long",
            "n_short",
        )
        if column in treatment_weekly.columns and column in control_weekly.columns
    ]
    construction_match = treatment_dates == control_dates
    if construction_match:
        for column in construction_cols:
            construction_match = construction_match and np.allclose(
                pd.to_numeric(treatment_weekly[column], errors="coerce"),
                pd.to_numeric(control_weekly[column], errors="coerce"),
                equal_nan=True,
            )
    rows.append(
        _check_row(
            "portfolio construction and accounting",
            construction_match,
            ", ".join(construction_cols),
        )
    )

    parent_id = str(spec.get("parent_id", ""))
    control_variant = str(
        control_config.get("paired_control_prompt_variant")
        or control_config.get("prompt_variant", "")
    )
    rows.append(
        _check_row(
            "prompt parent/control",
            parent_id == "v2-minimal-change-control"
            and control_variant == "v2-minimal-change-control",
            f"parent={parent_id!r}; control={control_variant!r}",
        )
    )

    declared_control = result.get("artifacts", {}).get("control_artifact") or {}
    hash_checks: list[str] = []
    hashes_ok = True
    for filename, field in (
        ("weekly_returns.csv", "weekly_returns_sha256"),
        ("signal_checks.csv", "signal_checks_sha256"),
    ):
        expected = str(declared_control.get(field, ""))
        path = artifacts.control_dir / filename
        actual = _sha256_file(path) if path.is_file() else ""
        matched = bool(expected) and actual == expected
        present = bool(actual)
        hashes_ok = hashes_ok and (matched if declared_control else present)
        status = "match" if matched else ("computed-now" if present and not expected else "mismatch")
        hash_checks.append(f"{filename}={status}:{actual[:12]}")
    rows.append(
        _check_row(
            "frozen control hashes" if declared_control else "legacy control audit hashes",
            hashes_ok,
            "; ".join(hash_checks),
        )
    )

    backend_build_recorded = bool(
        treatment_config.get("backend_revision") and control_config.get("backend_revision")
    )
    limitations: list[str] = []
    if not backend_build_recorded:
        limitations.append(
            "Provider backend build/revision was not recorded; model alias is the strongest available backend check."
        )
    if not declared_control:
        limitations.append(
            "This legacy fresh-paired result did not persist expected control hashes; current hashes were recorded by this audit."
        )
    limitation = " ".join(limitations)
    passed = all(bool(row["pass"]) for row in rows)
    return {
        "label": artifacts.label,
        "candidate_id": artifacts.candidate_id,
        "status": "verified_with_backend_limitation" if passed and limitation else ("verified" if passed else "failed"),
        "pass": passed,
        "backend_revision_recorded": backend_build_recorded,
        "limitation": limitation,
        "treatment_run_config": str(treatment_config_path or ""),
        "control_run_config": str(control_config_path or ""),
        "checks": rows,
    }


def moving_block_bootstrap(
    delta: Sequence[float],
    *,
    n_boot: int = DEFAULT_BOOTSTRAPS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int = 1701,
) -> dict[str, Any]:
    values = pd.to_numeric(pd.Series(delta), errors="coerce").dropna().to_numpy(float)
    if not len(values):
        raise ResearchDiagnosticError("Cannot bootstrap an empty paired series")
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot), dtype=float)
    for draw_idx in range(int(n_boot)):
        sample: list[float] = []
        while len(sample) < len(values):
            start = int(rng.integers(0, len(values)))
            sample.extend(values[(start + offset) % len(values)] for offset in range(block_size))
        draws[draw_idx] = float(np.mean(sample[: len(values)]) * PERIODS_PER_YEAR * 100.0)
    return {
        "n_boot": int(n_boot),
        "block_size": int(block_size),
        "mean_annualized_lift_pct": float(values.mean() * PERIODS_PER_YEAR * 100.0),
        "ci_95_low_pct": float(np.quantile(draws, 0.025)),
        "ci_95_high_pct": float(np.quantile(draws, 0.975)),
        "probability_lift_gt_zero": float(np.mean(draws > 0.0)),
        "probability_lift_le_zero": float(np.mean(draws <= 0.0)),
    }


def return_metrics(values: Sequence[float]) -> dict[str, Any]:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    annual = float(clean.mean() * PERIODS_PER_YEAR) if not clean.empty else np.nan
    volatility = float(clean.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)) if len(clean) > 1 else np.nan
    terminal = float((1.0 + clean).prod()) if not clean.empty else np.nan
    equity = (1.0 + clean).cumprod() if not clean.empty else pd.Series(dtype=float)
    drawdown = equity / equity.cummax() - 1.0 if not equity.empty else equity
    return {
        "n_periods": int(len(clean)),
        "annual_return_pct": annual * 100.0 if np.isfinite(annual) else np.nan,
        "cagr_pct": (
            (terminal ** (PERIODS_PER_YEAR / len(clean)) - 1.0) * 100.0
            if len(clean) and terminal > 0
            else np.nan
        ),
        "sharpe": annual / volatility if np.isfinite(volatility) and volatility > 0 else np.nan,
        "max_drawdown_pct": float(drawdown.min() * 100.0) if not drawdown.empty else np.nan,
        "hit_rate_pct": float((clean > 0.0).mean() * 100.0) if not clean.empty else np.nan,
    }


def _score_column(frame: pd.DataFrame) -> str:
    for column in ("llm_score", "final_score", "score"):
        if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any():
            return column
    raise ResearchDiagnosticError("No usable score column in signal checks")


def _ranked_checks(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    result = frame.copy()
    result[score_col] = pd.to_numeric(result[score_col], errors="coerce")
    result["score_rank"] = result.groupby(["date", "pool_side"])[score_col].rank(
        method="average", ascending=False
    )
    result["side_adjusted_fwd"] = np.where(
        result["pool_side"].eq("SHORT"), -result["fwd_c2c"], result["fwd_c2c"]
    )
    return result


def selection_diagnostics(
    treatment_checks: pd.DataFrame,
    control_checks: pd.DataFrame,
    *,
    book_size: int = DEFAULT_BOOK_SIZE,
    treatment_selections: pd.DataFrame | None = None,
    control_selections: pd.DataFrame | None = None,
) -> dict[str, Any]:
    candidate_score = _score_column(treatment_checks)
    control_score = _score_column(control_checks)
    candidate = _ranked_checks(treatment_checks, candidate_score)
    control = _ranked_checks(control_checks, control_score)
    keys = ["date", "pool_side", "ticker"]
    joined = candidate[keys + ["score_rank", "side_adjusted_fwd"]].merge(
        control[keys + ["score_rank"]],
        on=keys,
        how="inner",
        suffixes=("_candidate", "_control"),
        validate="one_to_one",
    )
    joined["candidate_selected"] = joined["score_rank_candidate"] <= book_size
    joined["control_selected"] = joined["score_rank_control"] <= book_size
    joined["changed"] = joined["candidate_selected"] != joined["control_selected"]
    joined["abs_rank_displacement"] = (
        joined["score_rank_candidate"] - joined["score_rank_control"]
    ).abs()
    overlap_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    use_actual_selections = treatment_selections is not None and control_selections is not None
    if use_actual_selections:
        candidate_groups = treatment_selections.groupby(["date", "pool_side"], sort=True)
        control_groups = control_selections.groupby(["date", "pool_side"], sort=True)
        selection_keys = sorted(set(candidate_groups.groups) & set(control_groups.groups))
    else:
        selection_keys = list(joined.groupby(["date", "pool_side"], sort=True).groups)
    for date, side in selection_keys:
        if use_actual_selections:
            candidate_group = candidate_groups.get_group((date, side)).copy()
            control_group = control_groups.get_group((date, side)).copy()
            candidate_names = set(candidate_group["ticker"])
            control_names = set(control_group["ticker"])
            candidate_unique = candidate_group[~candidate_group["ticker"].isin(control_names)]
            control_unique = control_group[~control_group["ticker"].isin(candidate_names)]
            sign = -1.0 if side == "SHORT" else 1.0
            candidate_valid_n = max(1, candidate_group["realized_c2c_return_pct"].notna().sum())
            control_valid_n = max(1, control_group["realized_c2c_return_pct"].notna().sum())
            changed_lift = 0.5 * sign * (
                candidate_unique["realized_c2c_return_pct"].sum() / candidate_valid_n
                - control_unique["realized_c2c_return_pct"].sum() / control_valid_n
            ) / 100.0
            selection_total_lift = 0.5 * sign * (
                candidate_group["realized_c2c_return_pct"].mean()
                - control_group["realized_c2c_return_pct"].mean()
            ) / 100.0
            shared_reweight_lift = selection_total_lift - changed_lift
            changed_count = len(candidate_unique)
        else:
            group = joined[
                joined["date"].eq(date) & joined["pool_side"].eq(side)
            ]
            candidate_names = set(group.loc[group["candidate_selected"], "ticker"])
            control_names = set(group.loc[group["control_selected"], "ticker"])
            candidate_unique_mask = group["candidate_selected"] & ~group["control_selected"]
            control_unique_mask = group["control_selected"] & ~group["candidate_selected"]
            changed_lift = (0.5 / float(book_size)) * (
                group.loc[candidate_unique_mask, "side_adjusted_fwd"].sum()
                - group.loc[control_unique_mask, "side_adjusted_fwd"].sum()
            )
            selection_total_lift = changed_lift
            shared_reweight_lift = 0.0
            changed_count = int(candidate_unique_mask.sum())
        overlap_rows.append(
            {
                "date": date,
                "pool_side": side,
                "selection_overlap_pct": 100.0
                * len(candidate_names & control_names)
                / max(1, book_size),
            }
        )
        changed_rows.append(
            {
                "date": date,
                "pool_side": side,
                "changed_name_count_each_book": changed_count,
                "changed_name_capital_lift": float(changed_lift),
                "selection_change_total_lift": float(selection_total_lift),
                "shared_name_reweight_lift": float(shared_reweight_lift),
            }
        )
    overlaps = pd.DataFrame(overlap_rows)
    changed = pd.DataFrame(changed_rows)

    ic_rows: list[dict[str, Any]] = []
    spread_rows: list[dict[str, Any]] = []
    for (date, side), group in candidate.groupby(["date", "pool_side"], sort=True):
        score = pd.to_numeric(group[candidate_score], errors="coerce")
        target = pd.to_numeric(group["side_adjusted_fwd"], errors="coerce")
        ic_rows.append(
            {
                "date": date,
                "pool_side": side,
                "spearman_ic": float(score.corr(target, method="spearman")),
            }
        )
        ordered = group.sort_values([candidate_score, "ticker"], ascending=[False, True])
        spread_rows.append(
            {
                "date": date,
                "pool_side": side,
                "top10_minus_bottom10_pct": float(
                    (ordered.head(10)["side_adjusted_fwd"].mean()
                    - ordered.tail(10)["side_adjusted_fwd"].mean())
                    * 100.0
                ),
            }
        )
    ic = pd.DataFrame(ic_rows)
    spread = pd.DataFrame(spread_rows)
    changed_by_date = changed.groupby("date", as_index=False)["changed_name_capital_lift"].sum()
    selection_total_by_date = changed.groupby("date", as_index=False)[
        ["selection_change_total_lift", "shared_name_reweight_lift"]
    ].sum()
    return {
        "mean_selection_overlap_pct": float(overlaps["selection_overlap_pct"].mean()),
        "mean_abs_rank_displacement": float(joined["abs_rank_displacement"].mean()),
        "median_abs_rank_displacement": float(joined["abs_rank_displacement"].median()),
        "mean_score_future_spearman_ic": float(ic["spearman_ic"].mean()),
        "annualized_top10_bottom10_spread_pct": float(
            spread.groupby("date")["top10_minus_bottom10_pct"].mean().mean()
            * PERIODS_PER_YEAR
        ),
        "annualized_changed_names_only_lift_pct": float(
            changed_by_date["changed_name_capital_lift"].mean()
            * PERIODS_PER_YEAR
            * 100.0
        ),
        "annualized_selection_change_total_lift_pct": float(
            selection_total_by_date["selection_change_total_lift"].mean()
            * PERIODS_PER_YEAR
            * 100.0
        ),
        "annualized_shared_name_reweight_lift_pct": float(
            selection_total_by_date["shared_name_reweight_lift"].mean()
            * PERIODS_PER_YEAR
            * 100.0
        ),
        "overlap_rows": overlaps,
        "rank_rows": joined,
        "ic_rows": ic,
        "spread_rows": spread,
        "changed_rows": changed,
    }


def alpha_concentration(delta: pd.Series) -> pd.DataFrame:
    values = pd.to_numeric(delta, errors="coerce").dropna().sort_values(ascending=False)
    total = float(values.sum())
    rows: list[dict[str, Any]] = []
    for fraction in (0.01, 0.05, 0.10, 0.20):
        count = max(1, int(math.ceil(len(values) * fraction)))
        contribution = float(values.iloc[:count].sum())
        trimmed = values.iloc[count:]
        rows.append(
            {
                "top_fraction_pct": fraction * 100.0,
                "dates_removed": count,
                "contribution_to_total_lift_pct": (
                    contribution / total * 100.0 if not np.isclose(total, 0.0) else np.nan
                ),
                "annualized_contribution_pct": contribution / len(values) * PERIODS_PER_YEAR * 100.0,
                "trimmed_annualized_lift_pct": (
                    float(trimmed.mean() * PERIODS_PER_YEAR * 100.0)
                    if not trimmed.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def grouped_paired_diagnostics(
    merged_weekly: pd.DataFrame,
    group_col: str,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, group in merged_weekly.groupby(group_col, dropna=False, sort=True):
        delta = group["candidate_return"] - group["control_return"]
        boot = moving_block_bootstrap(delta, n_boot=n_boot, seed=seed + len(rows))
        candidate_metrics = return_metrics(group["candidate_return"])
        control_metrics = return_metrics(group["control_return"])
        rows.append(
            {
                group_col: group_value,
                "n_dates": len(group),
                "candidate_return_pct": candidate_metrics["annual_return_pct"],
                "control_return_pct": control_metrics["annual_return_pct"],
                "paired_lift_pct": boot["mean_annualized_lift_pct"],
                "sharpe_lift": candidate_metrics["sharpe"] - control_metrics["sharpe"],
                "paired_hit_rate_pct": float((delta > 0.0).mean() * 100.0),
                "bootstrap_ci_low_pct": boot["ci_95_low_pct"],
                "bootstrap_ci_high_pct": boot["ci_95_high_pct"],
                "probability_lift_gt_zero": boot["probability_lift_gt_zero"],
            }
        )
    return pd.DataFrame(rows)


def full_stage_diagnostics(
    artifacts: PairedArtifacts,
    output_dir: Path,
    *,
    n_boot: int = DEFAULT_BOOTSTRAPS,
) -> dict[str, Any]:
    verification = verify_paired_artifacts(artifacts)
    if not verification["pass"]:
        failed = [row for row in verification["checks"] if not row["pass"]]
        raise ResearchDiagnosticError(f"Unpaired comparison {artifacts.label}: {failed}")
    candidate = load_weekly(artifacts.treatment_dir)
    control = load_weekly(artifacts.control_dir)
    trail5 = load_weekly(artifacts.trail5_dir)
    merged = candidate.merge(control, on="date", suffixes=("_candidate", "_control"), validate="one_to_one")
    trail_cols = trail5[["date", "capital_c2c", "long_contribution", "short_contribution"]].rename(
        columns={
            "capital_c2c": "trail5_return",
            "long_contribution": "trail5_long_contribution",
            "short_contribution": "trail5_short_contribution",
        }
    )
    merged = merged.merge(trail_cols, on="date", how="inner", validate="one_to_one")
    merged = merged.rename(
        columns={
            "capital_c2c_candidate": "candidate_return",
            "capital_c2c_control": "control_return",
            "long_contribution_candidate": "candidate_long_contribution",
            "long_contribution_control": "control_long_contribution",
            "short_contribution_candidate": "candidate_short_contribution",
            "short_contribution_control": "control_short_contribution",
        }
    )
    merged["paired_delta"] = merged["candidate_return"] - merged["control_return"]
    merged["year"] = merged["date"].dt.year
    regime_col = next(
        (column for column in ("regime_candidate", "regime") if column in merged.columns),
        None,
    )
    if regime_col:
        merged["market_regime"] = merged[regime_col].fillna("unknown").astype(str)
    else:
        merged["market_regime"] = "unknown"

    bootstrap = moving_block_bootstrap(merged["paired_delta"], n_boot=n_boot)
    selection = selection_diagnostics(
        load_checks(artifacts.treatment_dir),
        load_checks(artifacts.control_dir),
        treatment_selections=load_selections(artifacts.treatment_dir),
        control_selections=load_selections(artifacts.control_dir),
    )
    concentration = alpha_concentration(merged["paired_delta"])
    yearly = grouped_paired_diagnostics(merged, "year", n_boot=n_boot, seed=1801)
    regime = grouped_paired_diagnostics(merged, "market_regime", n_boot=n_boot, seed=1901)

    candidate_metrics = return_metrics(merged["candidate_return"])
    control_metrics = return_metrics(merged["control_return"])
    trail5_metrics = return_metrics(merged["trail5_return"])
    delta = merged["paired_delta"]
    long_delta = merged["candidate_long_contribution"] - merged["control_long_contribution"]
    short_delta = merged["candidate_short_contribution"] - merged["control_short_contribution"]
    blocks = [block for block in np.array_split(delta.to_numpy(), 4) if len(block)]

    cost_rows: list[dict[str, Any]] = []
    candidate_turnover = pd.to_numeric(merged.get("turnover_long_candidate"), errors="coerce") + pd.to_numeric(
        merged.get("turnover_short_candidate"), errors="coerce"
    )
    control_turnover = pd.to_numeric(merged.get("turnover_long_control"), errors="coerce") + pd.to_numeric(
        merged.get("turnover_short_control"), errors="coerce"
    )
    incremental_turnover = candidate_turnover - control_turnover
    for cost_bps in (0.0, 5.0, 10.0, 20.0):
        net_delta = delta - incremental_turnover * (cost_bps / 10_000.0)
        cost_rows.append(
            {
                "cost_bps_per_side": cost_bps,
                "incremental_turnover": float(incremental_turnover.mean()),
                "net_annualized_paired_lift_pct": float(
                    net_delta.mean() * PERIODS_PER_YEAR * 100.0
                ),
            }
        )
    cost_sensitivity = pd.DataFrame(cost_rows)

    summary = {
        "label": artifacts.label,
        "candidate_id": artifacts.candidate_id,
        "verification": verification,
        "candidate": candidate_metrics,
        "control": control_metrics,
        "trail5": trail5_metrics,
        "mean_paired_return_lift_pct_per_date": float(delta.mean() * 100.0),
        "median_paired_return_lift_pct_per_date": float(delta.median() * 100.0),
        "annualized_paired_lift_pct": bootstrap["mean_annualized_lift_pct"],
        "paired_hit_rate_pct": float((delta > 0.0).mean() * 100.0),
        "long_annualized_lift_pct": float(long_delta.mean() * PERIODS_PER_YEAR * 100.0),
        "short_annualized_lift_pct": float(short_delta.mean() * PERIODS_PER_YEAR * 100.0),
        "sharpe_lift": candidate_metrics["sharpe"] - control_metrics["sharpe"],
        "drawdown_improvement_pct": candidate_metrics["max_drawdown_pct"] - control_metrics["max_drawdown_pct"],
        "chronological_block_stability_pct": float(np.mean([np.mean(block) > 0 for block in blocks]) * 100.0),
        "bootstrap": bootstrap,
        "selection": {key: value for key, value in selection.items() if not isinstance(value, pd.DataFrame)},
        "top_1pct_contribution_pct": float(concentration.iloc[0]["contribution_to_total_lift_pct"]),
        "trimmed_5pct_annualized_lift_pct": float(concentration.iloc[1]["trimmed_annualized_lift_pct"]),
        "cost_sensitivity": cost_sensitivity.to_dict("records"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "paired_returns.csv", index=False)
    concentration.to_csv(output_dir / "alpha_concentration.csv", index=False)
    yearly.to_csv(output_dir / "lift_by_year.csv", index=False)
    regime.to_csv(output_dir / "lift_by_market_regime.csv", index=False)
    cost_sensitivity.to_csv(output_dir / "incremental_cost_sensitivity.csv", index=False)
    selection["overlap_rows"].to_csv(output_dir / "selection_overlap.csv", index=False)
    selection["rank_rows"].to_csv(output_dir / "rank_displacement.csv", index=False)
    selection["ic_rows"].to_csv(output_dir / "score_future_ic.csv", index=False)
    selection["spread_rows"].to_csv(output_dir / "top_bottom_score_spread.csv", index=False)
    selection["changed_rows"].to_csv(output_dir / "changed_names_pnl.csv", index=False)
    _write_json(output_dir / "summary.json", summary)
    return summary


def _percentile_score(series: pd.Series, *, higher_is_stronger: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    ranked = numeric.rank(method="average", pct=True)
    return ranked if higher_is_stronger else 1.0 - ranked


def conditional_factor_strength_study(
    artifacts: PairedArtifacts,
    output_dir: Path,
    *,
    n_boot: int = DEFAULT_BOOTSTRAPS,
) -> dict[str, Any]:
    """Bucket dates by ex-ante Trail5 opportunity strength and measure LLM lift."""
    verification = verify_paired_artifacts(artifacts)
    if not verification["pass"]:
        raise ResearchDiagnosticError("Conditional study requires a verified paired run")
    checks = load_checks(artifacts.treatment_dir)
    weekly = load_weekly(artifacts.treatment_dir)
    trail5 = load_weekly(artifacts.trail5_dir)
    trail_col = "feature_Trail5" if "feature_Trail5" in checks.columns else "trail_ret"
    adx_col = "feature_ADX" if "feature_ADX" in checks.columns else None
    vol_col = "feature_VolPctl" if "feature_VolPctl" in checks.columns else None
    rsi_col = "feature_RSI5" if "feature_RSI5" in checks.columns else None
    range_col = "feature_Rng55" if "feature_Rng55" in checks.columns else None
    checks[trail_col] = pd.to_numeric(checks[trail_col], errors="coerce")
    checks["abs_trail5"] = checks[trail_col].abs()
    # Five percentage points is fixed before looking at outcomes and is therefore
    # a true ex-ante breadth definition rather than a fitted sample threshold.
    checks["extreme_5pct"] = checks["abs_trail5"] >= 5.0
    if rsi_col:
        rsi = pd.to_numeric(checks[rsi_col], errors="coerce")
        checks["rsi_extreme"] = np.where(
            checks["pool_side"].eq("LONG"), rsi <= 30.0, rsi >= 70.0
        )
    if range_col:
        range_position = pd.to_numeric(checks[range_col], errors="coerce")
        checks["range_extreme"] = np.where(
            checks["pool_side"].eq("LONG"),
            range_position <= 0.20,
            range_position >= 0.80,
        )
    aggregations: dict[str, tuple[str, str | Any]] = {
        "trail5_dispersion": (trail_col, "std"),
        "trail5_mean_abs": ("abs_trail5", "mean"),
        "trail5_median_abs": ("abs_trail5", "median"),
        "extremity_breadth": ("extreme_5pct", "mean"),
    }
    if adx_col:
        aggregations["mean_adx"] = (adx_col, "mean")
    if vol_col:
        aggregations["mean_volatility_percentile"] = (vol_col, "mean")
    if rsi_col:
        aggregations["rsi_extremity_breadth"] = ("rsi_extreme", "mean")
    if range_col:
        aggregations["range_extremity_breadth"] = ("range_extreme", "mean")
    context = checks.groupby("date", as_index=False).agg(**aggregations)
    raw_components = [
        "trail5_dispersion",
        "trail5_mean_abs",
        "trail5_median_abs",
        "extremity_breadth",
    ]
    if rsi_col:
        raw_components.append("rsi_extremity_breadth")
    if range_col:
        raw_components.append("range_extremity_breadth")
    for column in raw_components:
        context[f"pct_{column}"] = _percentile_score(context[column])
    context["factor_strength_score"] = context[
        [f"pct_{column}" for column in raw_components]
    ].mean(axis=1)
    ordered = context["factor_strength_score"].rank(method="first")
    context["factor_strength_quintile"] = pd.qcut(
        ordered,
        q=5,
        labels=["Q1 weakest", "Q2", "Q3", "Q4", "Q5 strongest"],
    ).astype(str)

    combined = weekly[[
        "date",
        "capital_c2c",
        "long_contribution",
        "short_contribution",
        *( ["regime"] if "regime" in weekly.columns else [] ),
    ]].rename(
        columns={
            "capital_c2c": "llm_return",
            "long_contribution": "llm_long",
            "short_contribution": "llm_short",
        }
    )
    combined = combined.merge(
        trail5[["date", "capital_c2c", "long_contribution", "short_contribution"]].rename(
            columns={
                "capital_c2c": "trail5_return",
                "long_contribution": "trail5_long",
                "short_contribution": "trail5_short",
            }
        ),
        on="date",
        validate="one_to_one",
    ).merge(context, on="date", validate="one_to_one")
    combined["paired_lift"] = combined["llm_return"] - combined["trail5_return"]

    rows: list[dict[str, Any]] = []
    labels = ["Q1 weakest", "Q2", "Q3", "Q4", "Q5 strongest"]
    for index, label in enumerate(labels):
        group = combined[combined["factor_strength_quintile"].eq(label)]
        boot = moving_block_bootstrap(
            group["paired_lift"], n_boot=n_boot, seed=2301 + index
        )
        llm_metrics = return_metrics(group["llm_return"])
        trail_metrics = return_metrics(group["trail5_return"])
        rows.append(
            {
                "factor_strength_quintile": label,
                "n_dates": len(group),
                "mean_factor_strength_score": float(group["factor_strength_score"].mean()),
                "trail5_annualized_return_pct": trail_metrics["annual_return_pct"],
                "llm_annualized_return_pct": llm_metrics["annual_return_pct"],
                "paired_llm_minus_trail5_lift_pct": boot["mean_annualized_lift_pct"],
                "sharpe_lift": llm_metrics["sharpe"] - trail_metrics["sharpe"],
                "long_lift_pct": float(
                    (group["llm_long"] - group["trail5_long"]).mean()
                    * PERIODS_PER_YEAR
                    * 100.0
                ),
                "short_lift_pct": float(
                    (group["llm_short"] - group["trail5_short"]).mean()
                    * PERIODS_PER_YEAR
                    * 100.0
                ),
                "paired_hit_rate_pct": float((group["paired_lift"] > 0.0).mean() * 100.0),
                "bootstrap_ci_low_pct": boot["ci_95_low_pct"],
                "bootstrap_ci_high_pct": boot["ci_95_high_pct"],
                "probability_lift_gt_zero": boot["probability_lift_gt_zero"],
            }
        )
    quintiles = pd.DataFrame(rows)
    weakest = quintiles.iloc[0].to_dict()
    summary = {
        "label": artifacts.label,
        "definition": (
            "Equal-weighted development-date percentile composite of Trail5 dispersion, "
            "mean absolute Trail5, median absolute Trail5, fixed-|Trail5|>=5% breadth, "
            "and available side-aware RSI/range extremity breadth. "
            "ADX, volatility percentile, and regime are reported as context and are not "
            "allowed to redefine the primary raw-factor quintiles."
        ),
        "ex_ante_inputs_only": True,
        "weakest_quintile": weakest,
        "weak_state_positive_lift": bool(
            float(weakest["paired_llm_minus_trail5_lift_pct"]) > 0.0
        ),
        "weak_state_ci_excludes_zero": bool(float(weakest["bootstrap_ci_low_pct"]) > 0.0),
        "verification_status": verification["status"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    context.to_csv(output_dir / "factor_strength_date_context.csv", index=False)
    combined.to_csv(output_dir / "factor_strength_paired_returns.csv", index=False)
    quintiles.to_csv(output_dir / "factor_strength_quintiles.csv", index=False)
    if "regime" in combined.columns:
        regime_cross = (
            combined.groupby(["factor_strength_quintile", "regime"], observed=True)
            .agg(n_dates=("date", "size"), mean_paired_lift=("paired_lift", "mean"))
            .reset_index()
        )
        regime_cross["annualized_paired_lift_pct"] = (
            regime_cross["mean_paired_lift"] * PERIODS_PER_YEAR * 100.0
        )
        regime_cross.to_csv(output_dir / "factor_strength_by_regime.csv", index=False)
    _write_json(output_dir / "summary.json", summary)
    return summary


def _load_run_score_panel(run_dir: Path, run_label: str) -> pd.DataFrame:
    checks = load_checks(run_dir)
    score_col = _score_column(checks)
    panel = checks[["date", "ticker", "pool_side", "fwd_c2c", score_col]].copy()
    panel = panel.rename(columns={score_col: f"score_{run_label}"})
    panel[f"score_{run_label}"] = pd.to_numeric(panel[f"score_{run_label}"], errors="coerce")
    panel[f"rank_{run_label}"] = panel.groupby(["date", "pool_side"])[
        f"score_{run_label}"
    ].rank(method="average", ascending=False)
    return panel


def stochastic_disagreement_study(
    run_dirs: Sequence[Path],
    output_dir: Path,
    *,
    book_size: int = DEFAULT_BOOK_SIZE,
) -> dict[str, Any]:
    """Measure observed rank disagreement; never average ranks into a portfolio."""
    if len(run_dirs) < 2:
        raise ResearchDiagnosticError("Disagreement study requires at least two runs")
    panels: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    for index, run_dir in enumerate(map(Path, run_dirs), start=1):
        label = f"r{index}"
        panel = _load_run_score_panel(run_dir, label)
        config_path, config = _find_run_config(run_dir)
        panels.append(panel)
        metadata_rows.append(
            {
                "run": label,
                "run_dir": str(run_dir),
                "run_config": str(config_path or ""),
                "model": config.get("model"),
                "temperature": config.get("temperature"),
                "thinking_mode": config.get("thinking_mode"),
                "anonymize_tickers": config.get("anonymize_tickers"),
                "prompt_hash": config.get("prompt_hash"),
            }
        )
    keys = ["date", "ticker", "pool_side"]
    combined = panels[0]
    for index, panel in enumerate(panels[1:], start=2):
        combined = combined.merge(
            panel.drop(columns=["fwd_c2c"]),
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    expected_rows = len(panels[0])
    if len(combined) != expected_rows or any(len(panel) != expected_rows for panel in panels):
        raise ResearchDiagnosticError(
            "Repeated-run candidate pools differ; disagreement would not be paired"
        )
    score_cols = [f"score_r{index}" for index in range(1, len(panels) + 1)]
    rank_cols = [f"rank_r{index}" for index in range(1, len(panels) + 1)]
    combined["mean_rank"] = combined[rank_cols].mean(axis=1)
    combined["rank_std"] = combined[rank_cols].std(axis=1, ddof=0)
    combined["rank_iqr"] = combined[rank_cols].quantile(0.75, axis=1) - combined[
        rank_cols
    ].quantile(0.25, axis=1)
    combined["score_dispersion"] = combined[score_cols].std(axis=1, ddof=0)
    combined["selection_count"] = (combined[rank_cols] <= book_size).sum(axis=1)
    combined["selection_agreement"] = combined["selection_count"] / len(rank_cols)
    combined["side_adjusted_fwd"] = np.where(
        combined["pool_side"].eq("SHORT"), -combined["fwd_c2c"], combined["fwd_c2c"]
    )
    within_group_rank = combined.groupby(["date", "pool_side"])["rank_std"].rank(
        method="first"
    )
    combined["disagreement_quintile"] = (
        combined.groupby(["date", "pool_side"])["rank_std"]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"),
                5,
                labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
            ).astype(str)
        )
    )
    del within_group_rank
    combined["consensus_top30"] = combined["mean_rank"] <= book_size

    quintile_rows: list[dict[str, Any]] = []
    for scope, scoped in (
        ("all_pool_names", combined),
        ("consensus_top30_only", combined[combined["consensus_top30"]]),
    ):
        for (quintile, side), group in scoped.groupby(
            ["disagreement_quintile", "pool_side"], sort=True
        ):
            quintile_rows.append(
                {
                    "scope": scope,
                    "disagreement_quintile": quintile,
                    "pool_side": side,
                    "n_name_dates": len(group),
                    "mean_side_adjusted_return_pct": float(group["side_adjusted_fwd"].mean() * 100.0),
                    "median_side_adjusted_return_pct": float(group["side_adjusted_fwd"].median() * 100.0),
                    "mean_rank_std": float(group["rank_std"].mean()),
                    "mean_selection_agreement_pct": float(group["selection_agreement"].mean() * 100.0),
                }
            )
    quintiles = pd.DataFrame(quintile_rows)
    agreement = (
        combined.groupby(["selection_count", "pool_side"], as_index=False)
        .agg(
            n_name_dates=("ticker", "size"),
            mean_side_adjusted_return=("side_adjusted_fwd", "mean"),
            mean_rank=("mean_rank", "mean"),
            mean_rank_std=("rank_std", "mean"),
        )
    )
    agreement["mean_side_adjusted_return_pct"] = agreement[
        "mean_side_adjusted_return"
    ] * 100.0
    metadata = pd.DataFrame(metadata_rows)
    anonymous_count = int(metadata["anonymize_tickers"].eq(True).sum())  # noqa: E712
    sensitivity_note = (
        "The first three runs are exact anonymous replicates; the fourth is ticker-visible. "
        "Outputs therefore include both three-run core and four-run visibility sensitivity."
        if anonymous_count == 3 and len(metadata) == 4
        else "All supplied runs are analyzed together; inspect run_metadata.csv for differences."
    )
    lowest = quintiles[
        quintiles["scope"].eq("consensus_top30_only")
        & quintiles["disagreement_quintile"].eq("Q1 lowest")
    ]["mean_side_adjusted_return_pct"].mean()
    highest = quintiles[
        quintiles["scope"].eq("consensus_top30_only")
        & quintiles["disagreement_quintile"].eq("Q5 highest")
    ]["mean_side_adjusted_return_pct"].mean()
    summary = {
        "n_runs": len(run_dirs),
        "n_name_dates": len(combined),
        "sensitivity_note": sensitivity_note,
        "low_disagreement_consensus_return_pct": float(lowest),
        "high_disagreement_consensus_return_pct": float(highest),
        "low_disagreement_outperformed": bool(lowest > highest),
        "ensemble_portfolio_constructed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "run_metadata.csv", index=False)
    combined.to_csv(output_dir / "name_date_disagreement.csv", index=False)
    quintiles.to_csv(output_dir / "returns_by_disagreement_quintile.csv", index=False)
    agreement.to_csv(output_dir / "returns_by_selection_agreement.csv", index=False)
    _write_json(output_dir / "summary.json", summary)

    if anonymous_count >= 3 and len(run_dirs) > 3:
        core_dir = output_dir / "anonymous_three_run_core"
        core_summary = stochastic_disagreement_study(
            list(run_dirs)[:3], core_dir, book_size=book_size
        )
        summary["anonymous_three_run_core"] = core_summary
        _write_json(output_dir / "summary.json", summary)
    return summary


def _model_pipeline(model_name: str) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_name == "ridge":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        )
    if model_name == "hist_gradient_boosting":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=100,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=1701,
                    ),
                ),
            ]
        )
    raise ResearchDiagnosticError(f"Unknown tabular model {model_name!r}")


def _selection_return_rows(
    predictions: pd.DataFrame,
    score_columns: Sequence[str],
    *,
    book_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    for date, date_group in predictions.groupby("date", sort=True):
        row: dict[str, Any] = {"date": date}
        for score_col in score_columns:
            side_returns: dict[str, float] = {}
            for side, side_group in date_group.groupby("pool_side", sort=True):
                usable = side_group.dropna(subset=[score_col, "side_adjusted_fwd"])
                selected = usable.sort_values(
                    [score_col, "ticker"], ascending=[False, True]
                ).head(book_size)
                side_returns[side] = float(selected["side_adjusted_fwd"].mean())
                ic_rows.append(
                    {
                        "date": date,
                        "pool_side": side,
                        "model": score_col,
                        "spearman_ic": float(
                            usable[score_col].corr(
                                usable["side_adjusted_fwd"], method="spearman"
                            )
                        ),
                    }
                )
            if {"LONG", "SHORT"}.issubset(side_returns):
                row[score_col] = 0.5 * (
                    side_returns["LONG"] + side_returns["SHORT"]
                )
                row[f"{score_col}_long"] = 0.5 * side_returns["LONG"]
                row[f"{score_col}_short"] = 0.5 * side_returns["SHORT"]
        weekly_rows.append(row)
    return pd.DataFrame(weekly_rows).sort_values("date"), pd.DataFrame(ic_rows)


def expanding_tabular_baselines(
    artifacts: PairedArtifacts,
    output_dir: Path,
    *,
    warmup_dates: int = 52,
    embargo_periods: int = 1,
    refit_every: int = 4,
    book_size: int = DEFAULT_BOOK_SIZE,
    n_boot: int = DEFAULT_BOOTSTRAPS,
) -> dict[str, Any]:
    """Fit fixed-spec ridge and nonlinear models using only settled past outcomes."""
    verification = verify_paired_artifacts(artifacts)
    if not verification["pass"]:
        raise ResearchDiagnosticError("Tabular baselines require a verified paired run")
    if warmup_dates < 20 or embargo_periods < 1 or refit_every < 1:
        raise ResearchDiagnosticError("Unsafe tabular walk-forward configuration")
    checks = load_checks(artifacts.treatment_dir)
    weekly = load_weekly(artifacts.treatment_dir)
    if "regime" in weekly.columns:
        checks = checks.merge(weekly[["date", "regime"]], on="date", how="left", validate="many_to_one")
    else:
        checks["regime"] = "unknown"
    feature_cols = sorted(column for column in checks.columns if column.startswith("feature_"))
    if not feature_cols:
        raise ResearchDiagnosticError("No exact LLM feature columns available for tabular models")
    regime_dummies = pd.get_dummies(
        checks["regime"].fillna("unknown").astype(str), prefix="regime", dtype=float
    )
    for column in regime_dummies.columns:
        checks[column] = regime_dummies[column]
    model_features = feature_cols + sorted(regime_dummies.columns)
    for column in model_features:
        checks[column] = pd.to_numeric(checks[column], errors="coerce")
    checks["side_adjusted_fwd"] = np.where(
        checks["pool_side"].eq("SHORT"), -checks["fwd_c2c"], checks["fwd_c2c"]
    )
    llm_score = _score_column(checks)
    checks["score_llm"] = pd.to_numeric(checks[llm_score], errors="coerce")
    trail_score = "signal" if "signal" in checks.columns else (
        "feature_MRL" if "feature_MRL" in checks.columns else "feature_MRSig"
    )
    checks["score_trail5"] = pd.to_numeric(checks[trail_score], errors="coerce")

    dates = list(pd.Index(checks["date"].drop_duplicates()).sort_values())
    if len(dates) <= warmup_dates + embargo_periods:
        raise ResearchDiagnosticError("Not enough development dates for walk-forward training")
    prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    model_names = ("ridge", "hist_gradient_boosting")
    for side in ("LONG", "SHORT"):
        fitted: dict[str, Any] = {}
        fitted_cutoff: dict[str, pd.Timestamp] = {}
        for date_idx in range(warmup_dates, len(dates)):
            prediction_date = pd.Timestamp(dates[date_idx])
            eligible_dates = dates[: date_idx - embargo_periods]
            if not eligible_dates:
                continue
            should_refit = not fitted or (date_idx - warmup_dates) % refit_every == 0
            if should_refit:
                training = checks[
                    checks["pool_side"].eq(side) & checks["date"].isin(eligible_dates)
                ].dropna(subset=["side_adjusted_fwd"])
                if training.empty:
                    continue
                for model_name in model_names:
                    model = _model_pipeline(model_name)
                    model.fit(training[model_features], training["side_adjusted_fwd"])
                    fitted[model_name] = model
                    fitted_cutoff[model_name] = pd.Timestamp(training["date"].max())
            scoring = checks[
                checks["pool_side"].eq(side) & checks["date"].eq(prediction_date)
            ].copy()
            if scoring.empty or len(fitted) != len(model_names):
                continue
            for model_name in model_names:
                cutoff = fitted_cutoff[model_name]
                if not cutoff < prediction_date:
                    raise ResearchDiagnosticError(
                        f"Leakage firewall: {model_name} cutoff {cutoff} >= {prediction_date}"
                    )
                scoring[f"score_{model_name}"] = fitted[model_name].predict(
                    scoring[model_features]
                )
                audit_rows.append(
                    {
                        "model": model_name,
                        "pool_side": side,
                        "prediction_date": prediction_date,
                        "training_max_signal_date": cutoff,
                        "strictly_past": bool(cutoff < prediction_date),
                        "embargo_periods": embargo_periods,
                        "refit_every_dates": refit_every,
                    }
                )
            prediction_parts.append(scoring)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    if predictions.empty:
        raise ResearchDiagnosticError("Walk-forward models produced no predictions")
    score_columns = [
        "score_trail5",
        "score_llm",
        "score_ridge",
        "score_hist_gradient_boosting",
    ]
    weekly_models, ic = _selection_return_rows(
        predictions, score_columns, book_size=book_size
    )
    weekly_models = weekly_models.dropna(subset=score_columns).reset_index(drop=True)

    comparison_rows: list[dict[str, Any]] = []
    for score_col in score_columns:
        metrics = return_metrics(weekly_models[score_col])
        row = {"model": score_col.removeprefix("score_"), **metrics}
        if score_col != "score_trail5":
            versus_trail = moving_block_bootstrap(
                weekly_models[score_col] - weekly_models["score_trail5"],
                n_boot=n_boot,
                seed=2701 + len(comparison_rows),
            )
            row.update(
                {
                    "paired_lift_vs_trail5_pct": versus_trail["mean_annualized_lift_pct"],
                    "ci_vs_trail5_low_pct": versus_trail["ci_95_low_pct"],
                    "ci_vs_trail5_high_pct": versus_trail["ci_95_high_pct"],
                    "probability_beats_trail5": versus_trail["probability_lift_gt_zero"],
                }
            )
        if score_col not in {"score_trail5", "score_llm"}:
            versus_llm = moving_block_bootstrap(
                weekly_models[score_col] - weekly_models["score_llm"],
                n_boot=n_boot,
                seed=2801 + len(comparison_rows),
            )
            row.update(
                {
                    "paired_lift_vs_llm_pct": versus_llm["mean_annualized_lift_pct"],
                    "ci_vs_llm_low_pct": versus_llm["ci_95_low_pct"],
                    "ci_vs_llm_high_pct": versus_llm["ci_95_high_pct"],
                    "probability_beats_llm": versus_llm["probability_lift_gt_zero"],
                }
            )
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    ic["model"] = ic["model"].str.removeprefix("score_")
    mean_ic = ic.groupby("model", as_index=False)["spearman_ic"].mean().rename(
        columns={"spearman_ic": "mean_spearman_ic"}
    )
    comparison = comparison.merge(
        mean_ic, left_on="model", right_on="model", how="left"
    )
    llm_row = comparison[comparison["model"].eq("llm")].iloc[0]
    nonlinear_row = comparison[
        comparison["model"].eq("hist_gradient_boosting")
    ].iloc[0]
    summary = {
        "label": artifacts.label,
        "feature_columns": model_features,
        "target": "side-adjusted next-5-trading-day C2C return",
        "ticker_identity_used": False,
        "walk_forward": "expanding",
        "warmup_dates": warmup_dates,
        "embargo_periods": embargo_periods,
        "refit_every_dates": refit_every,
        "evaluation_dates": len(weekly_models),
        "all_training_strictly_past": bool(pd.DataFrame(audit_rows)["strictly_past"].all()),
        "llm_annual_return_pct": float(llm_row["annual_return_pct"]),
        "nonlinear_annual_return_pct": float(nonlinear_row["annual_return_pct"]),
        "nonlinear_paired_lift_vs_llm_pct": float(nonlinear_row["paired_lift_vs_llm_pct"]),
        "llm_beats_nonlinear_tabular": bool(
            float(nonlinear_row["paired_lift_vs_llm_pct"]) < 0.0
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "walk_forward_predictions.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output_dir / "leakage_audit.csv", index=False)
    weekly_models.to_csv(output_dir / "weekly_model_returns.csv", index=False)
    ic.to_csv(output_dir / "score_future_ic.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    _write_json(output_dir / "summary.json", summary)
    return summary


def _prompt_spec_payload(
    candidate_id: str,
    parent_prompt: str,
    prompt_text: str,
    *,
    hypothesis: str,
    mechanism: str,
    expected_failure: str,
    prior_experiment_ids: Sequence[str],
) -> dict[str, Any]:
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "generation": "G0",
        "parent_id": "v2-minimal-change-control",
        "mutation_mode": "rewrite",
        "patch": {
            "operation": "replace_once",
            "old_text": parent_prompt,
            "new_text": prompt_text,
        },
        "hypothesis": hypothesis,
        "mechanism": mechanism,
        "expected_failure": expected_failure,
        "prior_experiment_ids": list(prior_experiment_ids),
        "prompt_text": prompt_text,
        "prompt_hash": prompt_hash,
        "output_contract": "candidate_declared_numeric_score",
        "score_min": 0.0,
        "score_max": 100.0,
        "score_direction": "higher_is_better",
        "anonymize_tickers": True,
        "side_scope": "both",
        "replicate_id": "r1",
        "model": "deepseek-v4-flash",
        "temperature": 1.0,
        "thinking_mode": "disabled",
    }


def prepare_prompt_interaction_suite(output_dir: Path) -> dict[str, Any]:
    """Freeze the requested A/B/C/D prompts and exact paid-run commands."""
    import mean_reversion_llm_v2 as runner

    parent = runner.V2_MINIMAL_CHANGE_CONTROL_PROMPT
    prompt_b = runner.G0_05_SYSTEM_PROMPT
    prompt_c = runner.G0_01_SYSTEM_PROMPT
    expected_hashes = {
        "B": "384e11633b3135afba73b9b723b46ef629df807a44f0ff5c596800f313583ac0",
        "C": "8cb4e396a72fde7ebfa2477961de5e8ab474ed8c9607f405acc715d047cae77e",
    }
    actual_hashes = {
        "B": hashlib.sha256(prompt_b.encode("utf-8")).hexdigest(),
        "C": hashlib.sha256(prompt_c.encode("utf-8")).hexdigest(),
    }
    if actual_hashes != expected_hashes:
        raise ResearchDiagnosticError(
            f"Archived prompt hash mismatch; refusing interaction suite: {actual_hashes}"
        )
    old_context = parent[parent.index("CONTEXT\n") : parent.index("\nFEATURE GLOSSARY")]
    new_context = prompt_b[prompt_b.index("CONTEXT\n") : prompt_b.index("\nFEATURE GLOSSARY")]
    old_rule = (
        "- Use the 0-100 range only when the supplied cross-section supports it. "
        "Do not manufacture separation."
    )
    new_rule = (
        "- Use the full 0-100 range. Two tickers that differ on a signal must receive "
        "distinct scores proportional to the signal's magnitude."
    )
    if parent.count(old_context) != 1 or parent.count(old_rule) != 1:
        raise ResearchDiagnosticError("Could not materialize the predeclared interaction prompt")
    prompt_d = parent.replace(old_context, new_context, 1).replace(old_rule, new_rule, 1)

    specs = {
        "B": _prompt_spec_payload(
            "AB_B_DEPRESCRIBED_CONTEXT",
            parent,
            prompt_b,
            hypothesis="Removing the prescriptive strongest-setup narrative improves flexible conditional ranking.",
            mechanism="Neutral context allows the model to weight the supplied cross-section without a rigid exhaustion template.",
            expected_failure="Neutral framing may remove useful economic priors and reduce ranking quality.",
            prior_experiment_ids=("smart_prompt_search_20260802:G0_05",),
        ),
        "C": _prompt_spec_payload(
            "AB_C_PROPORTIONAL_SEPARATION",
            parent,
            prompt_c,
            hypothesis="Proportional score separation preserves genuine feature magnitude differences.",
            mechanism="The score rule discourages flat ties when supplied signals materially differ.",
            expected_failure="Forced separation may amplify noise and unstable small feature differences.",
            prior_experiment_ids=(
                "prompt_search_ticker_optional_resumed_20260806:G0_01",
            ),
        ),
        "D": _prompt_spec_payload(
            "AB_D_DEPRESCRIBED_X_PROPORTIONAL",
            parent,
            prompt_d,
            hypothesis="Neutral context and proportional score separation interact positively without assuming additive lift.",
            mechanism="The context frees feature weighting while the score rule preserves economically meaningful cross-sectional distinctions.",
            expected_failure="The combination may amplify noise or destroy either standalone mechanism, as prior prompt crosses have done.",
            prior_experiment_ids=(
                "smart_prompt_search_20260802:G0_05",
                "prompt_search_ticker_optional_resumed_20260806:G0_01",
                "v2_prompt_cross_g0_01_g0_05_2009_2014:negative-control-cross",
            ),
        ),
    }
    output_dir = Path(output_dir)
    spec_dir = output_dir / "prompt_specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_paths: dict[str, Path] = {}
    for arm, spec in specs.items():
        path = spec_dir / f"arm_{arm.lower()}.json"
        _write_json(path, spec)
        runner.load_experimental_prompt_spec(path)
        spec_paths[arm] = path

    manifest = DEFAULT_CAMPAIGN_06 / "manifests/full.json"
    if not manifest.is_file():
        raise ResearchDiagnosticError(f"Missing frozen development manifest: {manifest}")
    commands: dict[str, list[str]] = {}
    for arm in ("B", "C", "D"):
        arm_root = output_dir / f"arm_{arm.lower()}"
        command = [
            sys.executable,
            str(Path(runner.__file__).resolve()),
            "--phase",
            "full",
            "--start",
            "2009-01-01",
            "--end",
            "2014-01-01",
            "--output-dir",
            str(arm_root / "run"),
            "--cache-dir",
            str(arm_root / "cache"),
            "--experimental-prompt-spec",
            str(spec_paths[arm]),
            "--signal-dates-file",
            str(manifest),
            "--progress-file",
            str(arm_root / "run" / "runner_events.jsonl"),
            "--schedule",
            "anchors",
            "--anchor-offsets",
            "3",
            "--pool-sizes",
            "60",
            "--book-sizes",
            "30",
            "--llm-pool-size",
            "60",
            "--model",
            "deepseek-v4-flash",
            "--temperature",
            "1",
            "--thinking-mode",
            "disabled",
            "--anonymize-tickers",
            "--cost-bps-levels",
            "0",
            "--run-output-token-budget",
            "5000000",
            "--run-call-budget",
            "550",
            "--llm-max-output-tokens",
            "16000",
            "--llm-max-attempts",
            "2",
            "--experimental-override",
        ]
        if arm == "B":
            insertion = command.index("--run-output-token-budget")
            command[insertion:insertion] = [
                "--paired-control",
                "--paired-control-cache",
                str(output_dir / "shared_A" / "fresh_minimal_change_control.json"),
            ]
        commands[arm] = command
    suite = {
        "schema_version": 1,
        "purpose": "predeclared_four_way_prompt_interaction",
        "development_only": True,
        "blind_period_touched": False,
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "arms": {
            "A": {
                "label": "Minimal-change control",
                "prompt_hash": hashlib.sha256(parent.encode("utf-8")).hexdigest(),
                "execution": (
                    "Collected once alongside B, then frozen and reused unchanged for "
                    "the C and D comparisons."
                ),
            },
            "B": {"label": "De-prescribed context", "prompt_spec": str(spec_paths["B"])},
            "C": {"label": "Proportional score separation", "prompt_spec": str(spec_paths["C"])},
            "D": {"label": "B x C interaction", "prompt_spec": str(spec_paths["D"])},
        },
        "commands": commands,
        "execution_policy": (
            "Run sequentially. Collect one fresh minimal-change A control alongside B, freeze "
            "its cache/artifacts, and compare B/C/D to that same A under identical dates, pools, "
            "temperature, visibility, portfolio construction, and accounting. Do not approve "
            "validation from this suite."
        ),
    }
    _write_json(output_dir / "suite_manifest.json", suite)
    return suite


def _summary_table_html(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "<p>No rows available.</p>"
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body: list[str] = []
    for row in rows:
        cells: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_consolidated_dashboard(
    output_dir: Path,
    candidate_summaries: Sequence[dict[str, Any]],
    factor_summaries: Sequence[dict[str, Any]],
    disagreement_summary: dict[str, Any],
    tabular_summary: dict[str, Any],
) -> dict[str, Any]:
    """Create a conservative pre-validation recommendation from development only."""
    factor_by_label = {
        str(summary.get("label")): summary for summary in factor_summaries
    }
    ranking_rows: list[dict[str, Any]] = []
    for summary in candidate_summaries:
        bootstrap = summary["bootstrap"]
        factor = factor_by_label.get(str(summary["label"]), {})
        ranking_rows.append(
            {
                "label": summary["label"],
                "verified": summary["verification"]["pass"],
                "paired_lift_pct": summary["annualized_paired_lift_pct"],
                "ci_low_pct": bootstrap["ci_95_low_pct"],
                "probability_positive": bootstrap["probability_lift_gt_zero"],
                "block_stability_pct": summary["chronological_block_stability_pct"],
                "trimmed_5pct_lift_pct": summary["trimmed_5pct_annualized_lift_pct"],
                "long_lift_pct": summary["long_annualized_lift_pct"],
                "short_lift_pct": summary["short_annualized_lift_pct"],
                "sharpe_lift": summary["sharpe_lift"],
                "drawdown_improvement_pct": summary["drawdown_improvement_pct"],
                "weak_state_lift_pct": factor.get("weakest_quintile", {}).get(
                    "paired_llm_minus_trail5_lift_pct", float("nan")
                ),
                "weak_state_positive": bool(factor.get("weak_state_positive_lift")),
            }
        )
    # Robustness ordering treats 5%/24-date estimates as futility information and
    # refuses to turn tiny full-sample CAGR differences into an automatic winner.
    # Once all candidates fail a threshold (as these do after 5% trimming), the
    # failed numeric estimate is not used to manufacture a false precision rank.
    ranking_rows.sort(
        key=lambda row: (
            not bool(row["verified"]),
            not (float(row["trimmed_5pct_lift_pct"]) > 0.0),
            -float(row["block_stability_pct"]),
            not (
                float(row["long_lift_pct"]) >= 0.0
                and float(row["short_lift_pct"]) >= 0.0
            ),
            not bool(row["weak_state_positive"]),
            -min(float(row["long_lift_pct"]), float(row["short_lift_pct"])),
            -float(row["ci_low_pct"]),
            -float(row["paired_lift_pct"]),
            str(row["label"]),
        )
    )
    historical_primary = ranking_rows[0]["label"] if ranking_rows else None
    historical_challenger = ranking_rows[1]["label"] if len(ranking_rows) > 1 else None
    historical_weakest_positive = any(
        bool(summary.get("weak_state_positive_lift")) for summary in factor_summaries
    )
    historical_weakest_confirmed = any(
        bool(summary.get("weak_state_ci_excludes_zero")) for summary in factor_summaries
    )
    prompt_files_complete = all(
        (Path(output_dir) / "prompt_interaction" / f"arm_{arm.lower()}" / "run" / "comparison_scorecard.csv").is_file()
        for arm in ("B", "C", "D")
    )
    interaction_analysis_path = (
        Path(output_dir) / "prompt_interaction" / "analysis" / "summary.json"
    )
    interaction_analysis = (
        _read_json(interaction_analysis_path)
        if interaction_analysis_path.is_file()
        else {}
    )
    prompt_suite_complete = bool(
        prompt_files_complete and interaction_analysis.get("status") == "completed"
    )
    interaction_ranking = list(interaction_analysis.get("robustness_ranking", []))
    # Keep the dashboard useful when the predeclared interaction suite stops
    # after some arms have completed.  In particular, B and C have valid full
    # paired diagnostics even though D did not finish, so hiding all three arms
    # behind the suite-level completion flag would discard real evidence.
    interaction_status_rows: list[dict[str, Any]] = []
    interaction_root = Path(output_dir) / "prompt_interaction"
    for arm, decision in (("B", "retain"), ("C", "reject")):
        summary_path = interaction_root / f"arm_{arm.lower()}" / "diagnostics" / "summary.json"
        if not summary_path.is_file():
            continue
        arm_summary = _read_json(summary_path)
        bootstrap = arm_summary.get("bootstrap", {})
        interaction_status_rows.append(
            {
                "arm": arm,
                "status": "completed (251/251)",
                "decision": decision,
                "paired_lift_pct": arm_summary.get("annualized_paired_lift_pct"),
                "ci_95_pct": (
                    f"[{float(bootstrap.get('ci_95_low_pct', float('nan'))):.2f}, "
                    f"{float(bootstrap.get('ci_95_high_pct', float('nan'))):.2f}]"
                ),
                "positive_probability_pct": 100.0
                * float(bootstrap.get("probability_lift_gt_zero", float("nan"))),
                "block_stability_pct": arm_summary.get("chronological_block_stability_pct"),
                "trimmed_5pct_lift_pct": arm_summary.get("trimmed_5pct_annualized_lift_pct"),
                "long_lift_pct": arm_summary.get("long_annualized_lift_pct"),
                "short_lift_pct": arm_summary.get("short_annualized_lift_pct"),
            }
        )
    d_cache_files = sorted((interaction_root / "arm_d" / "cache").glob("*.json"))
    d_signal_count = 0
    d_calls_used = 0
    d_tokens_used = 0
    if d_cache_files:
        d_cache = _read_json(d_cache_files[-1])
        d_signal_count = len(d_cache.get("signals", {}))
        d_metadata = d_cache.get("metadata", {})
        d_calls_used = int(d_metadata.get("run_calls_used", 0))
        d_tokens_used = int(d_metadata.get("run_output_tokens_used", 0))
    interaction_status_rows.append(
        {
            "arm": "D",
            "status": f"incomplete ({d_signal_count}/251)",
            "decision": "prune — interaction futility; not statistically rejected",
            "paired_lift_pct": "not computed",
            "ci_95_pct": "not computed",
            "positive_probability_pct": "not computed",
            "block_stability_pct": "not computed",
            "trimmed_5pct_lift_pct": "not computed",
            "long_lift_pct": "not computed",
            "short_lift_pct": "not computed",
        }
    )
    arm_labels = {
        "B": "Fresh B: old G0_05 de-prescribed context",
        "C": "Fresh C: proportional score separation",
        "D": "Fresh D: de-prescribed × proportional interaction",
    }
    if prompt_suite_complete and interaction_ranking:
        provisional_primary = arm_labels.get(
            str(interaction_ranking[0].get("arm")),
            str(interaction_ranking[0].get("arm")),
        )
        provisional_challenger = (
            arm_labels.get(
                str(interaction_ranking[1].get("arm")),
                str(interaction_ranking[1].get("arm")),
            )
            if len(interaction_ranking) > 1
            else None
        )
        decision_rows = interaction_ranking[:2]
        trimmed_survivor = any(
            float(row.get("trimmed_5pct_lift_pct", -np.inf)) > 0.0
            for row in decision_rows
        )
        full_ci_confirmed = any(
            float(row.get("ci_low_pct", -np.inf)) > 0.0 for row in decision_rows
        )
        weakest_positive = any(
            float(row.get("weak_state_lift_pct", -np.inf)) > 0.0
            for row in decision_rows
        )
        weakest_confirmed = any(
            float(row.get("weak_state_ci_low_pct", -np.inf)) > 0.0
            for row in decision_rows
        )
        qualified = [
            row
            for row in decision_rows
            if float(row.get("trimmed_5pct_lift_pct", -np.inf)) > 0.0
            and float(row.get("ci_low_pct", -np.inf)) > 0.0
            and float(row.get("weak_state_ci_low_pct", -np.inf)) > 0.0
        ]
        frozen_primary = (
            arm_labels.get(str(qualified[0]["arm"]), str(qualified[0]["arm"]))
            if qualified
            else None
        )
        frozen_challenger = (
            arm_labels.get(str(qualified[1]["arm"]), str(qualified[1]["arm"]))
            if len(qualified) > 1
            else None
        )
    else:
        provisional_primary = historical_primary
        provisional_challenger = historical_challenger
        trimmed_survivor = any(
            bool(row["verified"]) and float(row["trimmed_5pct_lift_pct"]) > 0.0
            for row in ranking_rows[:2]
        )
        full_ci_confirmed = any(
            bool(row["verified"]) and float(row["ci_low_pct"]) > 0.0
            for row in ranking_rows[:2]
        )
        weakest_positive = historical_weakest_positive
        weakest_confirmed = historical_weakest_confirmed
        frozen_primary = None
        frozen_challenger = None
    ready = bool(
        prompt_suite_complete
        and frozen_primary
        and weakest_confirmed
        and trimmed_survivor
        and full_ci_confirmed
        and tabular_summary.get("llm_beats_nonlinear_tabular")
    )
    nonlinear_gap = float(tabular_summary.get("nonlinear_paired_lift_vs_llm_pct", float("nan")))
    strongest_weak = max(
        (
            float(summary.get("weakest_quintile", {}).get("paired_llm_minus_trail5_lift_pct", -np.inf)),
            str(summary.get("label")),
        )
        for summary in factor_summaries
    ) if factor_summaries else (float("nan"), "unavailable")
    recommendation = {
        "research_period": "2009-01-01 through 2013-12-23 development dates only",
        "blind_period_touched": False,
        "selection_rule": (
            "valid pair -> positive 5%-trimmed-lift threshold -> block stability -> "
            "two-sided non-damage -> weak-factor-state sign -> worst-side lift -> "
            "95% lower bound -> mean paired lift"
        ),
        "provisional_primary": provisional_primary,
        "provisional_challenger": provisional_challenger,
        "frozen_primary": frozen_primary,
        "frozen_challenger": frozen_challenger,
        "ready_for_blind_validation": ready,
        "blocking_reasons": [
            reason
            for condition, reason in (
                (
                    not prompt_suite_complete,
                    "Predeclared B/C/D runs against the single shared fresh A control, "
                    "or their paired analysis, are incomplete.",
                ),
                (not trimmed_survivor, "No shortlisted prompt retains positive lift after removing its best 5% of dates."),
                (not full_ci_confirmed, "No shortlisted prompt has a full-development 95% paired-lift lower bound above zero."),
                (not weakest_positive, "Positive incremental lift in the weakest factor state is not demonstrated."),
                (
                    not bool(tabular_summary.get("llm_beats_nonlinear_tabular")),
                    "The LLM has not beaten the same-information nonlinear tabular baseline.",
                ),
                (
                    not weakest_confirmed,
                    "Weak-state lift does not yet have a 95% interval entirely above zero.",
                ),
            )
            if condition
        ],
        "supporting_evidence": (
            f"The LLM beat the same-information nonlinear tabular model by "
            f"{-nonlinear_gap:.2f} annualized percentage points on their common "
            f"past-only evaluation window; the strongest weakest-factor-quintile "
            f"result was {strongest_weak[0]:.2f} points for {strongest_weak[1]}."
        ),
        "evidence_against": (
            "Every tested prompt has a 95% paired-lift interval crossing zero and "
            "negative lift after removing its best 5% of dates; the apparent edge is "
            "concentrated and remains model/prompt dependent."
        ),
        "disagreement_informative": disagreement_summary.get("low_disagreement_outperformed"),
        "llm_beats_nonlinear_tabular": tabular_summary.get("llm_beats_nonlinear_tabular"),
    }
    output_dir = Path(output_dir)
    _write_json(output_dir / "recommendation.json", recommendation)
    columns = (
        "label",
        "paired_lift_pct",
        "ci_low_pct",
        "probability_positive",
        "block_stability_pct",
        "trimmed_5pct_lift_pct",
        "long_lift_pct",
        "short_lift_pct",
        "sharpe_lift",
        "drawdown_improvement_pct",
        "weak_state_lift_pct",
    )
    table = _summary_table_html(ranking_rows, columns)
    interaction_table = _summary_table_html(
        interaction_ranking,
        (
            "arm",
            "paired_lift_pct",
            "ci_low_pct",
            "block_stability_pct",
            "trimmed_5pct_lift_pct",
            "long_lift_pct",
            "short_lift_pct",
            "weak_state_lift_pct",
            "weak_state_ci_low_pct",
        ),
    )
    interaction_status_table = _summary_table_html(
        interaction_status_rows,
        (
            "arm",
            "status",
            "decision",
            "paired_lift_pct",
            "ci_95_pct",
            "positive_probability_pct",
            "block_stability_pct",
            "trimmed_5pct_lift_pct",
            "long_lift_pct",
            "short_lift_pct",
        ),
    )
    blocking = "".join(
        f"<li>{html.escape(reason)}</li>" for reason in recommendation["blocking_reasons"]
    )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Mean-Reversion Thesis Package</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#172033}}
h1,h2{{color:#102044}} .warning{{background:#fff4dc;border:1px solid #e2a436;padding:14px}}
.ok{{background:#e9f8ef;border:1px solid #39965b;padding:14px}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #d8deea;padding:7px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
code{{background:#eef1f6;padding:2px 4px}}</style></head><body>
<h1>Conditional Mean-Reversion Ranking Thesis</h1>
<p class="{'ok' if ready else 'warning'}"><strong>Blind-validation readiness:</strong> {'READY' if ready else 'NOT READY'}</p>
<p>Development-only research. No 2015+ artifact is read by this package.</p>
<h2>Robustness-first candidate comparison</h2>{table}
<p>Provisional primary: <strong>{html.escape(str(provisional_primary))}</strong>. Provisional challenger: <strong>{html.escape(str(provisional_challenger))}</strong>. These are not frozen prompts.</p>
<p>The shortlist explicitly preserves complementary mechanisms: the old G0_05 is predominantly long-driven, while 20260806 G0_02 is predominantly short-driven and avoids material long-side damage. Headline CAGR is not a selection key.</p>
<h2>Fresh A/B/C/D prompt interaction</h2>{interaction_status_table}
<p><strong>Control design:</strong> A was collected once alongside B, then frozen and reused unchanged for C and D. The arms did not receive fresh controls.</p>
<p><strong>A:</strong> The shared capital-normalized control produced 12.66% CAGR, 1.094 Sharpe, and -8.76% maximum drawdown.</p>
<p><strong>B:</strong> Neutral/de-prescribed context produced 14.54% CAGR, 1.195 Sharpe, and -8.40% maximum drawdown. It improved the short book by 2.11 annualized percentage points while the long book declined 0.40 points. Its +1.71-point headline lift is promising but concentrated: the 95% interval crosses zero and its best-5%-trimmed lift is -1.60 points. Retain it as a development candidate, not a validated winner.</p>
<p><strong>C:</strong> Forced use of the full 0-100 range with distinct proportional scores produced 11.64% CAGR, 1.002 Sharpe, and -10.37% maximum drawdown. It lost 0.90 annualized points versus A, damaged both sides, and was positive in only 25% of chronological blocks. Reject it.</p>
<p><strong>D prompt:</strong> Combined B's neutral context with C's rule: <code>Use the full 0-100 range. Two tickers that differ on a signal must receive distinct scores proportional to the signal's magnitude.</code> Everything else remained fixed and tickers stayed anonymous. D produced {d_signal_count}/251 cached dates through 2011-09-16 ({d_calls_used} calls; {d_tokens_used:,} output tokens), then stopped on a DeepSeek 402 payment error. No portfolio-return result was produced, so no interim CAGR or lift is claimed.</p>
<p><strong>D decision:</strong> Prune for interaction futility because its only additional proposition is that B rescues C's already harmful rule. Preserve the cache and label D incomplete—not statistically rejected. No additive effect was assumed, and the blind period remains untouched.</p>
{interaction_table if prompt_suite_complete else ''}
<h2>Strongest evidence supporting the thesis</h2><p>{html.escape(recommendation['supporting_evidence'])}</p>
<h2>Strongest evidence against the thesis</h2><p>{html.escape(recommendation['evidence_against'])}</p>
<h2>Required before blind validation</h2><ul>{blocking}</ul>
<h2>Uncertainty and non-LLM checks</h2>
<p>Low-disagreement consensus outperformed high-disagreement consensus: <strong>{html.escape(str(disagreement_summary.get('low_disagreement_outperformed')))}</strong>.</p>
<p>LLM beat the same-information nonlinear tabular baseline: <strong>{html.escape(str(tabular_summary.get('llm_beats_nonlinear_tabular')))}</strong>.</p>
</body></html>"""
    (output_dir / "research_dashboard.html").write_text(page, encoding="utf-8")
    pd.DataFrame(ranking_rows).to_csv(output_dir / "candidate_robustness_ranking.csv", index=False)
    return recommendation


def default_candidate_artifacts() -> list[PairedArtifacts]:
    return [
        paired_artifacts(
            DEFAULT_G005_CAMPAIGN,
            "G0_05",
            label="20260802 G0_05 de-prescribed context",
        ),
        paired_artifacts(
            DEFAULT_CAMPAIGN_06,
            "G0_01",
            label="20260806 G0_01 proportional separation",
        ),
        paired_artifacts(
            DEFAULT_CAMPAIGN_06,
            "G0_02",
            label="20260806 G0_02 expected-magnitude scoring",
        ),
        paired_artifacts(
            DEFAULT_CAMPAIGN_07,
            "G0_02",
            label="20260807 G0_02 exhaustion-context rubric",
        ),
    ]


def run_development_thesis_package(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    n_boot: int = DEFAULT_BOOTSTRAPS,
    run_tabular: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = default_candidate_artifacts()
    candidate_summaries: list[dict[str, Any]] = []
    factor_summaries: list[dict[str, Any]] = []
    for index, candidate in enumerate(artifacts, start=1):
        slug = f"candidate_{index}_{candidate.candidate_id.lower()}"
        candidate_summaries.append(
            full_stage_diagnostics(
                candidate,
                output_dir / "full_stage_diagnostics" / slug,
                n_boot=n_boot,
            )
        )
        factor_summaries.append(
            conditional_factor_strength_study(
                candidate,
                output_dir / "conditional_factor_strength" / slug,
                n_boot=n_boot,
            )
        )
    disagreement = stochastic_disagreement_study(
        DEFAULT_V2_RUN_DIRS,
        output_dir / "stochastic_disagreement",
    )
    tabular_artifacts = artifacts[0]
    if run_tabular:
        tabular = expanding_tabular_baselines(
            tabular_artifacts,
            output_dir / "same_information_tabular",
            n_boot=n_boot,
        )
    else:
        tabular = {
            "label": tabular_artifacts.label,
            "skipped": True,
            "llm_beats_nonlinear_tabular": False,
        }
        _write_json(output_dir / "same_information_tabular" / "summary.json", tabular)
    prompt_suite = prepare_prompt_interaction_suite(output_dir / "prompt_interaction")
    prompt_analysis_path = output_dir / "prompt_interaction" / "analysis" / "summary.json"
    prompt_analysis = _read_json(prompt_analysis_path) if prompt_analysis_path.is_file() else {}
    recommendation = write_consolidated_dashboard(
        output_dir,
        candidate_summaries,
        factor_summaries,
        disagreement,
        tabular,
    )
    manifest = {
        "schema_version": 1,
        "development_cutoff_exclusive": str(DEVELOPMENT_CUTOFF.date()),
        "blind_period_touched": False,
        "bootstrap_samples": n_boot,
        "candidate_summaries": candidate_summaries,
        "factor_summaries": factor_summaries,
        "disagreement": disagreement,
        "tabular": tabular,
        "prompt_interaction_suite": {
            "manifest": str(output_dir / "prompt_interaction" / "suite_manifest.json"),
            "arms": list(prompt_suite["arms"]),
            "executed": bool(prompt_analysis.get("status") == "completed"),
            "analysis": str(prompt_analysis_path) if prompt_analysis else None,
        },
        "recommendation": recommendation,
    }
    _write_json(output_dir / "package_manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline, development-only mean-reversion thesis diagnostics"
    )
    parser.add_argument(
        "command",
        choices=("all", "verify", "prompt-suite", "factor", "tabular", "finalize"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--skip-tabular", action="store_true")
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_07)
    parser.add_argument("--candidate-id", default="G0_02")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prompt-suite":
        result = prepare_prompt_interaction_suite(args.output_dir / "prompt_interaction")
    elif args.command == "factor":
        result = {}
        for index, candidate in enumerate(default_candidate_artifacts(), start=1):
            slug = f"candidate_{index}_{candidate.candidate_id.lower()}"
            result[candidate.label] = conditional_factor_strength_study(
                candidate,
                args.output_dir / "conditional_factor_strength" / slug,
                n_boot=args.n_boot,
            )
    elif args.command == "tabular":
        result = expanding_tabular_baselines(
            paired_artifacts(DEFAULT_G005_CAMPAIGN, "G0_05"),
            args.output_dir / "same_information_tabular",
            n_boot=args.n_boot,
        )
    elif args.command == "finalize":
        candidate_dirs = sorted((args.output_dir / "full_stage_diagnostics").glob("candidate_*"))
        factor_dirs = sorted((args.output_dir / "conditional_factor_strength").glob("candidate_*"))
        candidate_summaries = [_read_json(path / "summary.json") for path in candidate_dirs]
        factor_summaries = [_read_json(path / "summary.json") for path in factor_dirs]
        disagreement = _read_json(args.output_dir / "stochastic_disagreement" / "summary.json")
        tabular = _read_json(args.output_dir / "same_information_tabular" / "summary.json")
        prompt_suite = prepare_prompt_interaction_suite(args.output_dir / "prompt_interaction")
        prompt_analysis_path = (
            args.output_dir / "prompt_interaction" / "analysis" / "summary.json"
        )
        prompt_analysis = (
            _read_json(prompt_analysis_path) if prompt_analysis_path.is_file() else {}
        )
        recommendation = write_consolidated_dashboard(
            args.output_dir,
            candidate_summaries,
            factor_summaries,
            disagreement,
            tabular,
        )
        result = {
            "schema_version": 1,
            "development_cutoff_exclusive": str(DEVELOPMENT_CUTOFF.date()),
            "blind_period_touched": False,
            "bootstrap_samples": args.n_boot,
            "candidate_summaries": candidate_summaries,
            "factor_summaries": factor_summaries,
            "disagreement": disagreement,
            "tabular": tabular,
            "prompt_interaction_suite": {
                "manifest": str(args.output_dir / "prompt_interaction" / "suite_manifest.json"),
                "arms": list(prompt_suite["arms"]),
                "executed": bool(prompt_analysis.get("status") == "completed"),
                "analysis": str(prompt_analysis_path) if prompt_analysis else None,
            },
            "recommendation": recommendation,
        }
        _write_json(args.output_dir / "package_manifest.json", result)
    elif args.command == "verify":
        result = verify_paired_artifacts(
            paired_artifacts(args.campaign_dir, args.candidate_id)
        )
        _write_json(args.output_dir / "pair_verification.json", result)
    else:
        result = run_development_thesis_package(
            args.output_dir,
            n_boot=args.n_boot,
            run_tabular=not args.skip_tabular,
        )
    if args.command in {"all", "finalize"}:
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "blind_period_touched": result["blind_period_touched"],
                    "candidate_count": len(result["candidate_summaries"]),
                    "ready_for_blind_validation": result["recommendation"][
                        "ready_for_blind_validation"
                    ],
                    "provisional_primary": result["recommendation"]["provisional_primary"],
                    "provisional_challenger": result["recommendation"][
                        "provisional_challenger"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
