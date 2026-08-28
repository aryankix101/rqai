"""Development and out-of-sample tracker for anonymized V2 prompt variants.

This is a zero-token evaluator. It reads completed V2 artifacts, applies the
six pre-registered portfolio constructions, and compares each one with its
matched Trail5 control under identical gross fixed 5-day C2C accounting.  It
also compares the glossary-backed prompt with the frozen no-glossary prompt on
the same 2015-2019 out-of-sample dates and candidate pools.
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import mean_reversion_llm_v2 as v2
import mean_reversion_llm_v2_autoresearch as ar


DEFAULT_SUITE_ROOT = Path("backtest_results/llm_reversal_signal_v2_2009_2014")
DEFAULT_DEVELOPMENT_PANEL = DEFAULT_SUITE_ROOT / "autoresearch_v2_cache_sweep/candidate_panel.parquet"
DEFAULT_DEVELOPMENT_GLOSSARY_ROOT = Path(
    "backtest_results/llm_reversal_signal_v2_temp0p6_anon_2009_2014"
)
DEFAULT_VALIDATION_ROOT = Path("backtest_results/llm_reversal_signal_v2_anon_validation_2015_2020")
DEFAULT_NO_GLOSSARY_VALIDATION_ROOT = Path(
    "backtest_results/llm_reversal_signal_v2_frozen_anon_validation_2015_2020"
)
DEFAULT_FINAL_ROOT = Path("backtest_results/llm_reversal_signal_v2_anon_final_2020_2025")
DEFAULT_EXPLORATORY_FINAL_ROOT = Path("backtest_results/llm_reversal_signal_v2_anon_offset2_exploratory_2020_2025")
DEFAULT_MINIMAL_CHANGE_DEVELOPMENT_ROOT = Path(
    "backtest_results/v2_prompt_ab_2009_2014/minimal_change_temp1"
)
DEFAULT_MINIMAL_CHANGE_VISIBLE_DEVELOPMENT_ROOT = Path(
    "backtest_results/v2_minimal_change_visible_temp1_2009_2014"
)
DEFAULT_MINIMAL_CHANGE_VALIDATION_ROOT = Path(
    "backtest_results/v2_prompt_ab_2015_2020/minimal_change_temp1"
)
DEFAULT_SLEEVE_SPLICE_DEVELOPMENT_ROOT = Path(
    "backtest_results/v2_sleeve_splice_in_sample_2009_2014/analysis/runs"
)
DEFAULT_SLEEVE_SPLICE_OOS_ROOT = Path(
    "backtest_results/v2_sleeve_splice_oos_2015_2020/analysis/runs"
)
DEFAULT_OUTPUT_NAME = "phase2_validation_dashboard.html"
DEFAULT_ARTIFACT_DIR = "phase2_validation_tracker"
ANCHOR = "3"
POOL = 60
BOOK = 30
GLOSSARY_VALIDATION_PERIOD = "Out-of-sample 2015-2019 — glossary"
NO_GLOSSARY_VALIDATION_PERIOD = "Out-of-sample 2015-2019 — no glossary"
GLOSSARY_DEVELOPMENT_PERIOD = "Development 2009-2013 — glossary"
NO_GLOSSARY_DEVELOPMENT_PERIOD = "Development 2009-2013 — no glossary"


@dataclass(frozen=True)
class FrozenTest:
    test_id: str
    label: str
    purpose: str
    llm_long: str
    llm_short: str
    control_long: str
    control_short: str


FROZEN_TESTS = (
    FrozenTest(
        "top30_equal",
        "Top 30 / 30 equal weight",
        "Does anonymized V2 improve the original selected book versus Trail5 with no weighting change?",
        "anon_eq30", "anon_eq30", "trail_eq30", "trail_eq30",
    ),
    FrozenTest(
        "exp60_h10",
        "Exponential 60, h10",
        "Does the LLM add ranking value when both methods use the same robust exponential concentration?",
        "anon_exp60_h10", "anon_exp60_h10", "trail_exp60_h10", "trail_exp60_h10",
    ),
    FrozenTest(
        "exp60_h5",
        "Exponential 60, h5",
        "Does the more aggressive development winner retain lift versus identically weighted Trail5?",
        "anon_exp60_h5", "anon_exp60_h5", "trail_exp60_h5", "trail_exp60_h5",
    ),
    FrozenTest(
        "top10_long",
        "Top-10 long selector",
        "Is the anonymized LLM's strongest incremental value concentrated in its ten highest-ranked longs?",
        "anon_eq10", "trail_eq30", "trail_eq10", "trail_eq30",
    ),
    FrozenTest(
        "short_rank1",
        "Short rank 1",
        "Does the LLM identify a uniquely useful first-ranked short beyond the same Trail5 concentration?",
        "trail_eq30", "anon_eq1", "trail_eq30", "trail_eq1",
    ),
    FrozenTest(
        "short_top10_h1",
        "Short top-10, h1",
        "Does the extreme short-tail ordering add value after holding the long sleeve fixed?",
        "trail_eq30", "anon_exp10_h1", "trail_eq30", "trail_exp10_h1",
    ),
)
FINAL_TESTS = (FROZEN_TESTS[0],)
# Keep the dashboard focused on the primary 30/30 construction. The complete
# pre-registered test ladder remains in the generated CSV artifacts.
DASHBOARD_TESTS = FINAL_TESTS

# Shared portfolio rules that can be replayed identically from every completed
# anonymized-V2 cache.  The cross-period search deliberately does not introduce
# feature tweaks or period-specific parameters.
CROSS_PERIOD_WEIGHT_MODELS = (
    "eq1", "eq5", "eq10", "eq15", "eq20", "eq25", "eq30", "eq35", "eq40", "eq50", "eq60",
    "lin60",
    "exp10_h1", "exp10_h2", "exp10_h3", "exp10_h5", "exp10_h10",
    "exp60_h5", "exp60_h10", "exp60_h20", "exp60_h40",
    "reserve10", "reserve20", "reserve30", "reserve40", "reserve50",
)


PROCESS_SPECS = {
    "trail_eq1": ar.ProcessSpec("trail_eq1", "score_trail5", "eq1", "phase2", "Trail5 rank 1"),
    "trail_eq10": ar.ProcessSpec("trail_eq10", "score_trail5", "eq10", "phase2", "Trail5 top 10"),
    "trail_eq30": ar.ProcessSpec("trail_eq30", "score_trail5", "eq30", "phase2", "Trail5 top 30"),
    "trail_exp10_h1": ar.ProcessSpec("trail_exp10_h1", "score_trail5", "exp10_h1", "phase2", "Trail5 top-10 exponential h1"),
    "trail_exp60_h5": ar.ProcessSpec("trail_exp60_h5", "score_trail5", "exp60_h5", "phase2", "Trail5 exponential h5"),
    "trail_exp60_h10": ar.ProcessSpec("trail_exp60_h10", "score_trail5", "exp60_h10", "phase2", "Trail5 exponential h10"),
    "anon_eq1": ar.ProcessSpec("anon_eq1", "score_v2_anon", "eq1", "phase2", "Anonymized V2 rank 1"),
    "anon_eq10": ar.ProcessSpec("anon_eq10", "score_v2_anon", "eq10", "phase2", "Anonymized V2 top 10"),
    "anon_eq30": ar.ProcessSpec("anon_eq30", "score_v2_anon", "eq30", "phase2", "Anonymized V2 top 30"),
    "anon_exp10_h1": ar.ProcessSpec("anon_exp10_h1", "score_v2_anon", "exp10_h1", "phase2", "Anonymized V2 top-10 exponential h1"),
    "anon_exp60_h5": ar.ProcessSpec("anon_exp60_h5", "score_v2_anon", "exp60_h5", "phase2", "Anonymized V2 exponential h5"),
    "anon_exp60_h10": ar.ProcessSpec("anon_exp60_h10", "score_v2_anon", "exp60_h10", "phase2", "Anonymized V2 exponential h10"),
}


def load_development_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing development candidate panel: {path}")
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"])
    required = {"date", "ticker", "pool_side", "score_trail5", "score_v2_anon", "fwd_c2c", "fwd_o2o"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Development panel missing columns: {sorted(missing)}")
    return panel


def load_period_panel(
    root: Path,
    start: str,
    end: str,
    anchor: str = ANCHOR,
    prompt_variant: str = v2.PROMPT_VARIANT_V2,
) -> pd.DataFrame | None:
    baseline = ar.signal_path(root, anchor, "baseline_pool", POOL, BOOK)
    llm = ar.signal_path(root, anchor, "llm_pool_v2", POOL, BOOK)
    if not baseline.exists() or not llm.exists():
        return None
    config_path = root / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Period artifacts exist without run_config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "start": start,
        "end": end,
        "pool_sizes": [60],
        "book_sizes": [30],
        "schedule": "anchors",
        "anchor_offsets": [int(anchor)],
        "llm_pool_size": 60,
        "cost_bps_levels": [0.0],
        "prompt_variant": prompt_variant,
        "prompt_hash": v2.prompt_hash_for_variant(prompt_variant),
        "anonymize_tickers": True,
        "temperature": 0.6,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Run does not match the locked validation protocol: {mismatches}")
    model = str(config.get("model", ""))
    thinking_mode = str(config.get("thinking_mode", "disabled"))
    if model not in {"deepseek-v4-flash", "deepseek-chat"}:
        raise ValueError(f"Run does not match the locked validation protocol: unsupported model {model!r}")
    if model == "deepseek-v4-flash" and thinking_mode != "disabled":
        raise ValueError(
            "Run does not match the locked validation protocol: "
            f"expected thinking_mode='disabled', got {thinking_mode!r}"
        )
    panel = ar.load_base_panel(root, anchor, POOL, BOOK)
    prompt_label = "No-glossary frozen prompt" if prompt_variant == v2.PROMPT_VARIANT_V2_FROZEN else "Glossary-backed V2 prompt"
    source = ar.ScoreSource(
        "v2_anon", root, "llm_pool_v2", "llm_score", prompt_label, v2.prompt_source_for_variant(prompt_variant),
    )
    panel = ar.merge_score_source(panel, source, anchor, POOL, BOOK)
    if "score_v2_anon" not in panel.columns or panel["score_v2_anon"].isna().any():
        raise ValueError("Validation LLM score coverage is incomplete")
    return panel


def assert_shared_trail5_baseline(glossary_panel: pd.DataFrame, no_glossary_panel: pd.DataFrame) -> None:
    """Require the prompt A/B to use exactly the same Trail5 observations."""
    keys = ["date", "ticker", "pool_side"]
    values = ["score_trail5", "fwd_c2c", "fwd_o2o"]
    glossary = glossary_panel[keys + values].sort_values(keys).reset_index(drop=True)
    no_glossary = no_glossary_panel[keys + values].sort_values(keys).reset_index(drop=True)
    if not glossary.equals(no_glossary):
        raise ValueError(
            "Glossary and no-glossary validation runs do not share the same Trail5 baseline; "
            "a direct prompt comparison would be invalid"
        )


def combine_sides(long_weekly: pd.DataFrame, short_weekly: pd.DataFrame) -> pd.DataFrame:
    merged = long_weekly[["date", "raw_c2c"]].rename(columns={"raw_c2c": "long_raw_c2c"}).merge(
        short_weekly[["date", "raw_c2c"]].rename(columns={"raw_c2c": "short_raw_c2c"}),
        on="date",
        how="inner",
        validate="one_to_one",
    )
    merged["capital_c2c"] = 0.5 * (merged["long_raw_c2c"] - merged["short_raw_c2c"])
    return merged.sort_values("date").reset_index(drop=True)


def bootstrap_lift(delta: pd.Series, n_boot: int = 3000, block_size: int = 4) -> tuple[float, float]:
    values = pd.to_numeric(delta, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(20260721)
    starts = np.arange(len(values))
    means = np.empty(n_boot)
    n_blocks = int(np.ceil(len(values) / block_size))
    for idx in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([values[(start + np.arange(block_size)) % len(values)] for start in chosen])[: len(values)]
        means[idx] = sample.mean() * ar.PERIODS_PER_YEAR * 100.0
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def evaluate_panel(
    panel: pd.DataFrame,
    period: str,
    tests: tuple[FrozenTest, ...] = FROZEN_TESTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    side_weekly: dict[tuple[str, str], pd.DataFrame] = {}
    for name, spec in PROCESS_SPECS.items():
        for side in ("LONG", "SHORT"):
            side_weekly[(name, side)] = ar.compute_side_process(panel, spec, side)[0]

    rows: list[dict[str, Any]] = []
    weekly_rows: list[pd.DataFrame] = []
    yearly_rows: list[dict[str, Any]] = []
    for test in tests:
        llm = combine_sides(side_weekly[(test.llm_long, "LONG")], side_weekly[(test.llm_short, "SHORT")])
        control = combine_sides(
            side_weekly[(test.control_long, "LONG")], side_weekly[(test.control_short, "SHORT")]
        )
        paired = llm.rename(
            columns={
                "capital_c2c": "llm_return",
                "long_raw_c2c": "llm_long_raw_c2c",
                "short_raw_c2c": "llm_short_raw_c2c",
            }
        ).merge(
            control[["date", "capital_c2c", "long_raw_c2c", "short_raw_c2c"]].rename(
                columns={
                    "capital_c2c": "trail5_return",
                    "long_raw_c2c": "trail5_long_raw_c2c",
                    "short_raw_c2c": "trail5_short_raw_c2c",
                }
            ),
            on="date",
            how="inner",
            validate="one_to_one",
        )
        paired["lift"] = paired["llm_return"] - paired["trail5_return"]
        paired["long_lift_contribution"] = 0.5 * (
            paired["llm_long_raw_c2c"] - paired["trail5_long_raw_c2c"]
        )
        paired["short_lift_contribution"] = 0.5 * (
            paired["trail5_short_raw_c2c"] - paired["llm_short_raw_c2c"]
        )
        paired["period"] = period
        paired["test_id"] = test.test_id
        paired["test"] = test.label
        weekly_rows.append(paired)

        llm_metrics = ar.metric_values(paired["llm_return"])
        control_metrics = ar.metric_values(paired["trail5_return"])
        years = sorted(paired["date"].dt.year.unique())
        positive_years = 0
        for year in years:
            year_data = paired[paired["date"].dt.year.eq(year)]
            llm_year = ar.metric_values(year_data["llm_return"])
            control_year = ar.metric_values(year_data["trail5_return"])
            year_lift = llm_year["cagr_pct"] - control_year["cagr_pct"]
            positive_years += int(np.isfinite(year_lift) and year_lift > 0)
            yearly_rows.append(
                {
                    "period": period,
                    "test_id": test.test_id,
                    "test": test.label,
                    "year": int(year),
                    "llm_cagr_pct": llm_year["cagr_pct"],
                    "trail5_cagr_pct": control_year["cagr_pct"],
                    "cagr_lift_pp": year_lift,
                }
            )
        ci_low, ci_high = bootstrap_lift(paired["lift"])
        rows.append(
            {
                "period": period,
                "test_id": test.test_id,
                "test": test.label,
                "purpose": test.purpose,
                "llm_long": test.llm_long,
                "llm_short": test.llm_short,
                "control_long": test.control_long,
                "control_short": test.control_short,
                "n_periods": len(paired),
                "llm_cagr_pct": llm_metrics["cagr_pct"],
                "trail5_cagr_pct": control_metrics["cagr_pct"],
                "cagr_lift_pp": llm_metrics["cagr_pct"] - control_metrics["cagr_pct"],
                "llm_sharpe": llm_metrics["sharpe"],
                "trail5_sharpe": control_metrics["sharpe"],
                "sharpe_lift": llm_metrics["sharpe"] - control_metrics["sharpe"],
                "llm_max_dd_pct": llm_metrics["max_drawdown_pct"],
                "trail5_max_dd_pct": control_metrics["max_drawdown_pct"],
                "max_dd_change_pp": llm_metrics["max_drawdown_pct"] - control_metrics["max_drawdown_pct"],
                "annualized_paired_lift_pct": paired["lift"].mean() * ar.PERIODS_PER_YEAR * 100.0,
                "annualized_long_lift_contribution_pct": (
                    paired["long_lift_contribution"].mean() * ar.PERIODS_PER_YEAR * 100.0
                ),
                "annualized_short_lift_contribution_pct": (
                    paired["short_lift_contribution"].mean() * ar.PERIODS_PER_YEAR * 100.0
                ),
                "rebalance_win_rate_pct": paired["lift"].gt(0.0).mean() * 100.0,
                "paired_lift_ci_low_pct": ci_low,
                "paired_lift_ci_high_pct": ci_high,
                "positive_lift_years": positive_years,
                "years_tested": len(years),
            }
        )
    return pd.DataFrame(rows), pd.concat(weekly_rows, ignore_index=True), pd.DataFrame(yearly_rows)


def evaluate_cross_period_search(panels: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Search one fixed long/short weighting rule across every supplied period.

    Each LLM construction is compared with Trail5 using the identical long and
    short weighting models, so lift is attributable to score ordering rather
    than a mismatched portfolio wrapper.
    """
    period_rows: list[dict[str, Any]] = []
    for period, panel in panels.items():
        weekly: dict[tuple[str, str, str], pd.DataFrame] = {}
        for score_col in ("score_v2_anon", "score_trail5"):
            for model in CROSS_PERIOD_WEIGHT_MODELS:
                spec = ar.ProcessSpec(f"{score_col}__{model}", score_col, model, "cross_period", model)
                for side in ("LONG", "SHORT"):
                    weekly[(score_col, model, side)] = ar.compute_side_process(panel, spec, side)[0]

        for long_model in CROSS_PERIOD_WEIGHT_MODELS:
            for short_model in CROSS_PERIOD_WEIGHT_MODELS:
                llm = combine_sides(
                    weekly[("score_v2_anon", long_model, "LONG")],
                    weekly[("score_v2_anon", short_model, "SHORT")],
                )
                control = combine_sides(
                    weekly[("score_trail5", long_model, "LONG")],
                    weekly[("score_trail5", short_model, "SHORT")],
                )
                llm_metrics = ar.metric_values(llm["capital_c2c"])
                control_metrics = ar.metric_values(control["capital_c2c"])
                period_rows.append(
                    {
                        "period": period,
                        "long_model": long_model,
                        "short_model": short_model,
                        "n_periods": len(llm),
                        "llm_cagr_pct": llm_metrics["cagr_pct"],
                        "trail5_cagr_pct": control_metrics["cagr_pct"],
                        "cagr_lift_pp": llm_metrics["cagr_pct"] - control_metrics["cagr_pct"],
                        "llm_sharpe": llm_metrics["sharpe"],
                        "trail5_sharpe": control_metrics["sharpe"],
                        "sharpe_lift": llm_metrics["sharpe"] - control_metrics["sharpe"],
                        "llm_max_dd_pct": llm_metrics["max_drawdown_pct"],
                    }
                )

    detail = pd.DataFrame(period_rows)
    summary_rows: list[dict[str, Any]] = []
    metric_cols = (
        "llm_cagr_pct", "trail5_cagr_pct", "cagr_lift_pp", "llm_sharpe",
        "trail5_sharpe", "sharpe_lift", "llm_max_dd_pct",
    )
    for (long_model, short_model), group in detail.groupby(["long_model", "short_model"], sort=False):
        row: dict[str, Any] = {"long_model": long_model, "short_model": short_model}
        for item in group.itertuples(index=False):
            prefix = ar.slug(item.period)
            for metric in metric_cols:
                row[f"{prefix}__{metric}"] = getattr(item, metric)
        row["min_period_cagr_pct"] = group["llm_cagr_pct"].min()
        row["mean_period_cagr_pct"] = group["llm_cagr_pct"].mean()
        row["min_period_sharpe"] = group["llm_sharpe"].min()
        row["min_period_lift_pp"] = group["cagr_lift_pp"].min()
        row["positive_cagr_all_periods"] = bool(group["llm_cagr_pct"].gt(0).all())
        row["positive_lift_all_periods"] = bool(group["cagr_lift_pp"].gt(0).all())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["min_period_cagr_pct", "mean_period_cagr_pct", "min_period_sharpe"],
        ascending=False,
    ).reset_index(drop=True)
    summary.insert(0, "robust_rank", np.arange(1, len(summary) + 1))
    return detail, summary


def esc(value: Any) -> str:
    return html.escape(str(value))


def pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return "Pending"
    return f"{float(value):+.{digits}f}%" if signed else f"{float(value):.{digits}f}%"


def num(value: Any, digits: int = 2, signed: bool = False) -> str:
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return "Pending"
    return f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"


def result_table(
    scorecard: pd.DataFrame,
    period: str,
    tests: tuple[FrozenTest, ...] = DASHBOARD_TESTS,
) -> str:
    subset = scorecard[scorecard["period"].eq(period)] if not scorecard.empty else pd.DataFrame()
    by_id = subset.set_index("test_id") if not subset.empty else pd.DataFrame()
    rows = []
    for test in tests:
        display_label = f"V2 · {test.label}" if test.test_id == "top30_equal" else test.label
        row = by_id.loc[test.test_id] if not by_id.empty and test.test_id in by_id.index else None
        lift = np.nan if row is None else row["cagr_lift_pp"]
        lift_class = "good" if pd.notna(lift) and lift > 0 else "bad" if pd.notna(lift) else "pending"
        positive_years = (
            "Pending"
            if row is None
            else f"{int(row['positive_lift_years'])}/{int(row['years_tested'])}"
        )
        rows.append(
            f"<tr><td><strong>{esc(display_label)}</strong><small>{esc(test.purpose)}</small></td>"
            f"<td>{pct(np.nan if row is None else row['llm_cagr_pct'])}</td>"
            f"<td>{pct(np.nan if row is None else row['trail5_cagr_pct'])}</td>"
            f"<td class='{lift_class}'><strong>{pct(lift, signed=True)}</strong></td>"
            f"<td>{pct(np.nan if row is None else row['annualized_long_lift_contribution_pct'], signed=True)}</td>"
            f"<td>{pct(np.nan if row is None else row['annualized_short_lift_contribution_pct'], signed=True)}</td>"
            f"<td>{num(np.nan if row is None else row['llm_sharpe'])}</td>"
            f"<td>{num(np.nan if row is None else row['sharpe_lift'], signed=True)}</td>"
            f"<td>{pct(np.nan if row is None else row['llm_max_dd_pct'])}</td>"
            f"<td>{pct(np.nan if row is None else row['max_dd_change_pp'], signed=True)}</td>"
            f"<td>{pct(np.nan if row is None else row['rebalance_win_rate_pct'])}</td>"
            f"<td>{positive_years}</td></tr>"
        )
    return "".join(rows)


def prompt_comparison_table(
    scorecard: pd.DataFrame,
    glossary_period: str = GLOSSARY_VALIDATION_PERIOD,
    no_glossary_period: str = NO_GLOSSARY_VALIDATION_PERIOD,
) -> str:
    """Render a same-period glossary ablation from cached score panels."""
    glossary = scorecard[scorecard["period"].eq(glossary_period)] if not scorecard.empty else pd.DataFrame()
    no_glossary = scorecard[scorecard["period"].eq(no_glossary_period)] if not scorecard.empty else pd.DataFrame()
    glossary_by_id = glossary.set_index("test_id") if not glossary.empty else pd.DataFrame()
    no_glossary_by_id = no_glossary.set_index("test_id") if not no_glossary.empty else pd.DataFrame()
    rows = []
    for test in DASHBOARD_TESTS:
        glossary_row = (
            glossary_by_id.loc[test.test_id]
            if not glossary_by_id.empty and test.test_id in glossary_by_id.index
            else None
        )
        no_glossary_row = (
            no_glossary_by_id.loc[test.test_id]
            if not no_glossary_by_id.empty and test.test_id in no_glossary_by_id.index
            else None
        )
        trail5_cagr = np.nan if glossary_row is None else glossary_row["trail5_cagr_pct"]
        no_glossary_lift = np.nan if no_glossary_row is None else no_glossary_row["cagr_lift_pp"]
        glossary_lift = np.nan if glossary_row is None else glossary_row["cagr_lift_pp"]
        prompt_delta = (
            np.nan
            if glossary_row is None or no_glossary_row is None
            else glossary_row["llm_cagr_pct"] - no_glossary_row["llm_cagr_pct"]
        )
        no_glossary_class = (
            "good" if pd.notna(no_glossary_lift) and no_glossary_lift > 0 else "bad" if pd.notna(no_glossary_lift) else "pending"
        )
        glossary_class = "good" if pd.notna(glossary_lift) and glossary_lift > 0 else "bad" if pd.notna(glossary_lift) else "pending"
        prompt_delta_class = "good" if pd.notna(prompt_delta) and prompt_delta > 0 else "bad" if pd.notna(prompt_delta) else "pending"
        rows.append(
            f"<tr><td><strong>{esc(test.label)}</strong></td>"
            f"<td>{pct(trail5_cagr)}</td>"
            f"<td>{pct(np.nan if no_glossary_row is None else no_glossary_row['llm_cagr_pct'])}</td>"
            f"<td class='{no_glossary_class}'><strong>{pct(no_glossary_lift, signed=True)}</strong></td>"
            f"<td>{pct(np.nan if glossary_row is None else glossary_row['llm_cagr_pct'])}</td>"
            f"<td class='{glossary_class}'><strong>{pct(glossary_lift, signed=True)}</strong></td>"
            f"<td class='{prompt_delta_class}'><strong>{pct(prompt_delta, signed=True)}</strong></td></tr>"
        )
    return "".join(rows)


def prompt_comparison_summary(
    scorecard: pd.DataFrame,
    glossary_period: str = GLOSSARY_VALIDATION_PERIOD,
    no_glossary_period: str = NO_GLOSSARY_VALIDATION_PERIOD,
) -> str:
    """State the original equal-weight question in plain language."""
    if scorecard.empty:
        return "The glossary ablation is pending."
    subset = scorecard[
        scorecard["period"].isin([glossary_period, no_glossary_period])
        & scorecard["test_id"].eq("top30_equal")
    ]
    by_period = subset.set_index("period") if not subset.empty else pd.DataFrame()
    if (
        by_period.empty
        or glossary_period not in by_period.index
        or no_glossary_period not in by_period.index
    ):
        return "The glossary ablation is pending."
    glossary = by_period.loc[glossary_period]
    no_glossary = by_period.loc[no_glossary_period]
    no_glossary_lift = float(no_glossary["cagr_lift_pp"])
    glossary_lift = float(glossary["cagr_lift_pp"])
    prompt_delta = float(glossary["llm_cagr_pct"] - no_glossary["llm_cagr_pct"])
    no_glossary_read = "beat" if no_glossary_lift > 0 else "did not beat"
    relative_read = "helped" if prompt_delta > 0 else "hurt"
    return (
        f"On the original equal-weight top-30/30 book, the no-glossary prompt <strong>{no_glossary_read} "
        f"Trail5</strong>: {pct(no_glossary['llm_cagr_pct'])} versus {pct(no_glossary['trail5_cagr_pct'])} "
        f"CAGR ({pct(no_glossary_lift, signed=True)} lift). The glossary prompt produced "
        f"{pct(glossary['llm_cagr_pct'])} CAGR ({pct(glossary_lift, signed=True)} versus Trail5), so the glossary "
        f"<strong>{relative_read} by {pct(prompt_delta, signed=True)}</strong>."
    )


def load_minimal_change_validation(root: Path) -> tuple[pd.Series | None, pd.Series | None, dict[str, Any]]:
    """Load the primary treatment and Trail5 rows from a completed prompt run."""
    root = Path(root)
    scorecard_path = root / "comparison_scorecard.csv"
    config_path = root / "run_config.json"
    if not scorecard_path.exists():
        return None, None, {}
    scorecard = pd.read_csv(scorecard_path)
    if scorecard.empty:
        return None, None, {}
    anchor = scorecard.get("anchor_offset", pd.Series(index=scorecard.index, dtype=str)).astype(str)
    is_average = (
        scorecard.get("is_anchor_average", pd.Series(False, index=scorecard.index))
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    costs = pd.to_numeric(
        scorecard.get("cost_bps", pd.Series(0.0, index=scorecard.index)),
        errors="coerce",
    ).fillna(0.0)
    eligible = scorecard[
        scorecard.get("execution_model", pd.Series(index=scorecard.index, dtype=str))
        .astype(str)
        .str.upper()
        .eq("C2C")
        & costs.eq(0.0)
        & anchor.eq(ANCHOR)
        & ~is_average
    ]
    runs = eligible.get("run", pd.Series(index=eligible.index, dtype=str)).astype(str)
    treatment_rows = eligible[runs.str.contains("minimal-change-control", case=False, regex=False)]
    baseline_rows = eligible[runs.eq("Baseline-Pool")]
    treatment = None if treatment_rows.empty else treatment_rows.iloc[0]
    baseline = None if baseline_rows.empty else baseline_rows.iloc[0]
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return treatment, baseline, config


def minimal_change_validation_summary(root: Path = DEFAULT_MINIMAL_CHANGE_VALIDATION_ROOT) -> str:
    treatment, baseline, config = load_minimal_change_validation(root)
    if treatment is None or baseline is None:
        return "The minimal-change temperature-1 validation result is pending."
    lift = float(treatment["capital_cagr_pct"] - baseline["capital_cagr_pct"])
    return (
        f"The minimal-change prompt at temperature {num(config.get('temperature'), 1)} produced "
        f"<strong>{pct(treatment['capital_cagr_pct'])} capital CAGR</strong> versus "
        f"{pct(baseline['capital_cagr_pct'])} for matched Trail5 ({pct(lift, signed=True)} lift), "
        f"with Sharpe improving from {num(baseline['capital_sharpe'])} to "
        f"{num(treatment['capital_sharpe'])}."
    )


def minimal_change_paired_diagnostics(root: Path) -> dict[str, Any]:
    """Calculate the paired diagnostics used by the standard Phase 2 columns."""
    run_root = Path(root) / "pool60_pos30" / f"anchor_{ANCHOR}"
    llm_path = run_root / "llm_pool_v2_minimal_change_control" / "weekly_returns.csv"
    trail5_path = run_root / "baseline_pool" / "weekly_returns.csv"
    if not llm_path.exists() or not trail5_path.exists():
        return {}
    columns = ["date", "long_ret_c2c", "short_ret_c2c"]
    llm = pd.read_csv(llm_path, usecols=columns).rename(
        columns={
            "long_ret_c2c": "llm_long",
            "short_ret_c2c": "llm_short",
        }
    )
    trail5 = pd.read_csv(trail5_path, usecols=columns).rename(
        columns={
            "long_ret_c2c": "trail5_long",
            "short_ret_c2c": "trail5_short",
        }
    )
    paired = llm.merge(trail5, on="date", how="inner", validate="one_to_one")
    if paired.empty:
        return {}
    paired["date"] = pd.to_datetime(paired["date"])
    paired["llm_return"] = 0.5 * (paired["llm_long"] - paired["llm_short"])
    paired["trail5_return"] = 0.5 * (paired["trail5_long"] - paired["trail5_short"])
    paired["lift"] = paired["llm_return"] - paired["trail5_return"]
    positive_years = 0
    years = sorted(paired["date"].dt.year.unique())
    for year in years:
        year_rows = paired[paired["date"].dt.year.eq(year)]
        llm_metrics = ar.metric_values(year_rows["llm_return"])
        trail5_metrics = ar.metric_values(year_rows["trail5_return"])
        positive_years += int(llm_metrics["cagr_pct"] > trail5_metrics["cagr_pct"])
    return {
        "annualized_long_lift_contribution_pct": (
            0.5 * (paired["llm_long"] - paired["trail5_long"]).mean()
            * ar.PERIODS_PER_YEAR
            * 100.0
        ),
        "annualized_short_lift_contribution_pct": (
            0.5 * (paired["trail5_short"] - paired["llm_short"]).mean()
            * ar.PERIODS_PER_YEAR
            * 100.0
        ),
        "rebalance_win_rate_pct": paired["lift"].gt(0.0).mean() * 100.0,
        "positive_lift_years": positive_years,
        "years_tested": len(years),
    }


def minimal_change_validation_table(
    root: Path = DEFAULT_MINIMAL_CHANGE_VALIDATION_ROOT,
    label: str = "Minimal change · temperature 1.0",
    purpose: str = "Does the minimal-change prompt improve the original selected book versus matched Trail5?",
) -> str:
    treatment, baseline, _ = load_minimal_change_validation(root)
    if treatment is None or baseline is None:
        return "<tr><td colspan='12' class='pending'>Pending: completed scorecard not found.</td></tr>"
    diagnostics = minimal_change_paired_diagnostics(root)
    cagr_lift = float(treatment["capital_cagr_pct"] - baseline["capital_cagr_pct"])
    sharpe_lift = float(treatment["capital_sharpe"] - baseline["capital_sharpe"])
    dd_change = float(treatment["capital_max_drawdown_pct"] - baseline["capital_max_drawdown_pct"])
    positive_years = (
        "Pending"
        if not diagnostics
        else f"{diagnostics['positive_lift_years']}/{diagnostics['years_tested']}"
    )
    return (
        f"<tr><td><strong>{esc(label)}</strong>"
        f"<small>{esc(purpose)}</small></td>"
        f"<td>{pct(treatment['capital_cagr_pct'])}</td>"
        f"<td>{pct(baseline['capital_cagr_pct'])}</td>"
        f"<td class='good'><strong>{pct(cagr_lift, signed=True)}</strong></td>"
        f"<td>{pct(diagnostics.get('annualized_long_lift_contribution_pct'), signed=True)}</td>"
        f"<td>{pct(diagnostics.get('annualized_short_lift_contribution_pct'), signed=True)}</td>"
        f"<td>{num(treatment['capital_sharpe'])}</td>"
        f"<td class='good'><strong>{num(sharpe_lift, signed=True)}</strong></td>"
        f"<td>{pct(treatment['capital_max_drawdown_pct'])}</td>"
        f"<td class='good'><strong>{pct(dd_change, signed=True)}</strong></td>"
        f"<td>{pct(diagnostics.get('rebalance_win_rate_pct'))}</td>"
        f"<td>{positive_years}</td></tr>"
    )


def lift_bars(scorecard: pd.DataFrame, period: str) -> str:
    subset = scorecard[scorecard["period"].eq(period)] if not scorecard.empty else pd.DataFrame()
    subset = subset[subset["test_id"].isin(test.test_id for test in DASHBOARD_TESTS)]
    if subset.empty:
        return f"<div class='empty'>Pending: no rows found yet for {esc(period)}.</div>"
    max_abs = max(float(subset["cagr_lift_pp"].abs().max()), 1.0)
    parts = []
    for row in subset.itertuples():
        lift = float(row.cagr_lift_pp)
        width = 10.0 + 90.0 * abs(lift) / max_abs
        cls = "positive" if lift >= 0 else "negative"
        parts.append(
            f"<div class='bar-row'><div>{esc(row.test)}</div><div class='bar-track'>"
            f"<div class='bar {cls}' style='width:{width:.1f}%'></div></div>"
            f"<strong class='{('good' if lift >= 0 else 'bad')}'>{lift:+.1f}pp</strong></div>"
        )
    return "".join(parts)


def equity_svg(weekly: pd.DataFrame, period: str) -> str:
    data = weekly[weekly["period"].eq(period)].copy() if not weekly.empty else pd.DataFrame()
    if data.empty:
        return f"<div class='empty'>Equity chart will appear after rows exist for {esc(period)}.</div>"
    test_ids = [test.test_id for test in DASHBOARD_TESTS]
    colors = {"top30_equal": "#247665"}
    series = []
    all_values = []
    all_dates = []
    for test_id in test_ids:
        item = data[data["test_id"].eq(test_id)].sort_values("date")
        if item.empty:
            continue
        equity = (1.0 + item["llm_return"]).cumprod()
        dates = pd.to_datetime(item["date"]).reset_index(drop=True)
        series.append((test_id, dates, equity.to_numpy()))
        all_values.extend(equity.tolist())
        all_dates.extend(dates.tolist())
    if not all_values:
        return "<div class='empty'>No chartable validation returns.</div>"
    width, height = 900, 300
    left, right, top, bottom = 46, 18, 22, 44
    low, high = min(all_values), max(all_values)
    span = max(high - low, 1e-9)
    start = min(all_dates)
    end = max(all_dates)
    start_ns = pd.Timestamp(start).value
    end_ns = pd.Timestamp(end).value
    date_span = max(end_ns - start_ns, 1)
    paths = []
    for test_id, dates, values in series:
        date_ns = pd.to_datetime(dates).astype("int64").to_numpy()
        xs = left + (date_ns - start_ns) / date_span * (width - left - right)
        ys = height - bottom - (values - low) / span * (height - top - bottom)
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        paths.append(f"<polyline points='{points}' fill='none' stroke='{colors[test_id]}' stroke-width='2.5'/>")
    tick_parts = [
        f"<line x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' stroke='#d8e0e8'/>",
        f"<text x='{left}' y='{top+4}' fill='#687589' font-size='11'>{high:.2f}x</text>",
        f"<text x='{left}' y='{height-bottom-6}' fill='#687589' font-size='11'>{low:.2f}x</text>",
        f"<text x='{left}' y='{height-17}' text-anchor='start' fill='#687589' font-size='11'>{pd.Timestamp(start).strftime('%Y-%m-%d')}</text>",
    ]
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        tick = pd.Timestamp(year=year, month=1, day=1)
        if tick < pd.Timestamp(start) or tick > pd.Timestamp(end):
            continue
        x = left + (tick.value - start_ns) / date_span * (width - left - right)
        tick_parts.append(f"<line x1='{x:.1f}' y1='{height-bottom}' x2='{x:.1f}' y2='{height-bottom+5}' stroke='#9aa8b6'/>")
        tick_parts.append(f"<text x='{x:.1f}' y='{height-17}' text-anchor='middle' fill='#687589' font-size='11'>{year}</text>")
    tick_parts.append(
        f"<text x='{width-right}' y='{height-17}' text-anchor='end' fill='#687589' font-size='11'>{pd.Timestamp(end).strftime('%Y-%m-%d')}</text>"
    )
    legend = "".join(
        f"<span><i style='background:{colors[test_id]}'></i>{esc(next(t.label for t in FROZEN_TESTS if t.test_id == test_id))}</span>"
        for test_id, _, _ in series
    )
    return (
        f"<div class='legend'>{legend}</div>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='LLM equity curves for {esc(period)}'>"
        f"{''.join(tick_parts)}{''.join(paths)}</svg>"
    )


def minimal_change_weekly_returns(root: Path) -> pd.DataFrame:
    path = (
        Path(root)
        / "pool60_pos30"
        / f"anchor_{ANCHOR}"
        / "llm_pool_v2_minimal_change_control"
        / "weekly_returns.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path, usecols=["date", "long_ret_c2c", "short_ret_c2c"])
    data["date"] = pd.to_datetime(data["date"])
    data["return"] = 0.5 * (data["long_ret_c2c"] - data["short_ret_c2c"])
    return data[["date", "return"]].sort_values("date").reset_index(drop=True)


def sleeve_splice_weekly_returns(root: Path, run_slug: str) -> pd.DataFrame:
    path = Path(root) / run_slug / "weekly_returns.csv"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path, usecols=["date", "long_ret_c2c", "short_ret_c2c"])
    data["date"] = pd.to_datetime(data["date"])
    data["return"] = 0.5 * (data["long_ret_c2c"] - data["short_ret_c2c"])
    return data[["date", "return"]].sort_values("date").reset_index(drop=True)


def challenger_comparison_equity_svg(
    weekly: pd.DataFrame,
    period: str,
    challenger_root: Path,
    visible_challenger_root: Path | None = None,
    sleeve_splice_root: Path | None = None,
) -> str:
    """Plot Trail5, V2, and minimal-change equity on shared date and wealth axes."""
    v2_data = weekly[
        weekly["period"].eq(period) & weekly["test_id"].eq("top30_equal")
    ][["date", "llm_return", "trail5_return"]].copy() if not weekly.empty else pd.DataFrame()
    challenger_data = minimal_change_weekly_returns(challenger_root)
    if v2_data.empty or challenger_data.empty:
        return f"<div class='empty'>Comparison chart pending for {esc(period)}.</div>"
    v2_data["date"] = pd.to_datetime(v2_data["date"])
    v2_data = v2_data.sort_values("date")
    trail5_data = v2_data[["date", "trail5_return"]].rename(columns={"trail5_return": "return"})
    v2_data = v2_data[["date", "llm_return"]].rename(columns={"llm_return": "return"})
    colors = {
        "Trail5": "#7b8794",
        "V2 · Top 30 / 30": "#247665",
        "Minimal change · temp 1.0": "#b5812d",
        "Minimal change · temp 1.0 · ticker visible": "#2e6fbb",
        "Exp. 1 · V2 Long + B Short": "#7b3fc6",
        "Exp. 2 · B Context + V2 Range": "#c24e72",
    }
    series = []
    requested_series = [
        ("Trail5", trail5_data),
        ("V2 · Top 30 / 30", v2_data),
        ("Minimal change · temp 1.0", challenger_data),
    ]
    if visible_challenger_root is not None:
        visible_data = minimal_change_weekly_returns(visible_challenger_root)
        if not visible_data.empty:
            requested_series.append(
                ("Minimal change · temp 1.0 · ticker visible", visible_data)
            )
    if sleeve_splice_root is not None:
        for label, slug in (
            ("Exp. 1 · V2 Long + B Short", "experiment_1_two_call_splice"),
            ("Exp. 2 · B Context + V2 Range", "experiment_2_b_original_range"),
        ):
            splice_data = sleeve_splice_weekly_returns(sleeve_splice_root, slug)
            if not splice_data.empty:
                requested_series.append((label, splice_data))
    for label, data in requested_series:
        returns = pd.to_numeric(data["return"], errors="coerce").fillna(0.0)
        equity = (1.0 + returns).cumprod().to_numpy()
        dates = pd.to_datetime(data["date"]).reset_index(drop=True)
        metrics = ar.metric_values(returns)
        series.append((label, dates, equity, metrics["cagr_pct"]))

    width, height = 1100, 360
    left, right, top, bottom = 58, 28, 28, 48
    all_values = [value for _, _, values, _ in series for value in values]
    all_dates = [date for _, dates, _, _ in series for date in dates]
    raw_low, raw_high = min(min(all_values), 1.0), max(max(all_values), 1.0)
    padding = max((raw_high - raw_low) * 0.08, 0.02)
    low, high = raw_low - padding, raw_high + padding
    span = high - low
    start, end = min(all_dates), max(all_dates)
    start_ns, end_ns = pd.Timestamp(start).value, pd.Timestamp(end).value
    date_span = max(end_ns - start_ns, 1)

    grid = []
    for value in np.linspace(low, high, 5):
        y = height - bottom - (value - low) / span * (height - top - bottom)
        grid.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#e7edf2'/>"
            f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' fill='#687589' font-size='11'>{value:.2f}x</text>"
        )
    if low <= 1.0 <= high:
        y_one = height - bottom - (1.0 - low) / span * (height - top - bottom)
        grid.append(
            f"<line x1='{left}' y1='{y_one:.1f}' x2='{width-right}' y2='{y_one:.1f}' "
            "stroke='#9aa8b6' stroke-dasharray='5 5'/>"
        )

    paths = []
    for label, dates, values, _ in series:
        date_ns = dates.astype("int64").to_numpy()
        xs = left + (date_ns - start_ns) / date_span * (width - left - right)
        ys = height - bottom - (values - low) / span * (height - top - bottom)
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        color = colors[label]
        paths.append(
            f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='3' "
            "stroke-linejoin='round' stroke-linecap='round'/>"
            f"<circle cx='{xs[-1]:.1f}' cy='{ys[-1]:.1f}' r='4' fill='{color}'/>"
        )

    date_ticks = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        tick = pd.Timestamp(year=year, month=1, day=1)
        if tick < pd.Timestamp(start) or tick > pd.Timestamp(end):
            continue
        x = left + (tick.value - start_ns) / date_span * (width - left - right)
        date_ticks.append(
            f"<line x1='{x:.1f}' y1='{height-bottom}' x2='{x:.1f}' y2='{height-bottom+5}' stroke='#9aa8b6'/>"
            f"<text x='{x:.1f}' y='{height-20}' text-anchor='middle' fill='#687589' font-size='11'>{year}</text>"
        )
    legend = "".join(
        f"<span><i style='background:{colors[label]}'></i>{esc(label)} "
        f"<strong>{pct(cagr)} CAGR</strong></span>"
        for label, _, _, cagr in series
    )
    return (
        f"<div class='legend comparison-legend'>{legend}</div>"
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='V2 versus minimal-change equity for {esc(period)}'>"
        f"{''.join(grid)}{''.join(date_ticks)}{''.join(paths)}</svg>"
    )


def challenger_comparison_block(
    weekly: pd.DataFrame,
    period: str,
    challenger_root: Path,
    title: str,
    note: str,
    visible_challenger_root: Path | None = None,
    sleeve_splice_root: Path | None = None,
) -> str:
    return (
        f"<div class='comparison-card'><h3>{esc(title)}</h3><p>{esc(note)}</p>"
        f"{challenger_comparison_equity_svg(weekly, period, challenger_root, visible_challenger_root, sleeve_splice_root)}</div>"
    )


def period_visual_block(scorecard: pd.DataFrame, weekly: pd.DataFrame, period: str, title: str, note: str) -> str:
    return (
        f"<div class='period-visual'><div class='period-copy'><h3>{esc(title)}</h3><p>{esc(note)}</p></div>"
        f"<div class='chart-grid'><div><h4>CAGR Lift vs Trail5</h4>{lift_bars(scorecard, period)}</div>"
        f"<div><h4>LLM Equity Curves</h4>{equity_svg(weekly, period)}</div></div></div>"
    )


def write_dashboard(
    suite_root: Path,
    development_panel: Path,
    validation_root: Path,
    final_root: Path = DEFAULT_FINAL_ROOT,
    exploratory_final_root: Path = DEFAULT_EXPLORATORY_FINAL_ROOT,
    no_glossary_validation_root: Path = DEFAULT_NO_GLOSSARY_VALIDATION_ROOT,
    development_glossary_root: Path = DEFAULT_DEVELOPMENT_GLOSSARY_ROOT,
    n_boot: int = 3000,
    minimal_change_development_root: Path = DEFAULT_MINIMAL_CHANGE_DEVELOPMENT_ROOT,
    minimal_change_visible_development_root: Path = DEFAULT_MINIMAL_CHANGE_VISIBLE_DEVELOPMENT_ROOT,
    minimal_change_validation_root: Path = DEFAULT_MINIMAL_CHANGE_VALIDATION_ROOT,
) -> Path:
    del n_boot  # Bootstrap count is intentionally fixed for reproducible tracker output.
    del final_root, exploratory_final_root  # Retained as CLI-compatible inputs; omitted from this dashboard.
    artifact_dir = suite_root / DEFAULT_ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)

    development_no_glossary_panel = load_development_panel(development_panel)
    dev_no_glossary_scorecard, dev_no_glossary_weekly, dev_no_glossary_yearly = evaluate_panel(
        development_no_glossary_panel,
        NO_GLOSSARY_DEVELOPMENT_PERIOD,
    )
    development_glossary_panel = load_period_panel(
        development_glossary_root,
        "2009-01-01",
        "2014-01-01",
    )
    if development_glossary_panel is None:
        dev_glossary_scorecard = pd.DataFrame()
        dev_glossary_weekly = pd.DataFrame()
        dev_glossary_yearly = pd.DataFrame()
    else:
        dev_glossary_scorecard, dev_glossary_weekly, dev_glossary_yearly = evaluate_panel(
            development_glossary_panel,
            GLOSSARY_DEVELOPMENT_PERIOD,
        )
        assert_shared_trail5_baseline(development_glossary_panel, development_no_glossary_panel)
    validation_panel = load_period_panel(validation_root, "2015-01-01", "2020-01-01")
    if validation_panel is None:
        val_scorecard = pd.DataFrame()
        val_weekly = pd.DataFrame()
        val_yearly = pd.DataFrame()
    else:
        val_scorecard, val_weekly, val_yearly = evaluate_panel(validation_panel, GLOSSARY_VALIDATION_PERIOD)

    no_glossary_panel = load_period_panel(
        no_glossary_validation_root,
        "2015-01-01",
        "2020-01-01",
        prompt_variant=v2.PROMPT_VARIANT_V2_FROZEN,
    )
    if no_glossary_panel is None:
        no_glossary_scorecard = pd.DataFrame()
        no_glossary_weekly = pd.DataFrame()
        no_glossary_yearly = pd.DataFrame()
    else:
        no_glossary_scorecard, no_glossary_weekly, no_glossary_yearly = evaluate_panel(
            no_glossary_panel,
            NO_GLOSSARY_VALIDATION_PERIOD,
        )
    if validation_panel is not None and no_glossary_panel is not None:
        assert_shared_trail5_baseline(validation_panel, no_glossary_panel)

    scorecard = pd.concat(
        [
            dev_no_glossary_scorecard,
            dev_glossary_scorecard,
            val_scorecard,
            no_glossary_scorecard,
        ],
        ignore_index=True,
    )
    weekly = pd.concat(
        [
            dev_no_glossary_weekly,
            dev_glossary_weekly,
            val_weekly,
            no_glossary_weekly,
        ],
        ignore_index=True,
    )
    yearly = pd.concat(
        [
            dev_no_glossary_yearly,
            dev_glossary_yearly,
            val_yearly,
            no_glossary_yearly,
        ],
        ignore_index=True,
    )
    scorecard.to_csv(artifact_dir / "phase2_scorecard.csv", index=False)
    weekly.to_csv(artifact_dir / "phase2_weekly_paired_returns.csv", index=False)
    yearly.to_csv(artifact_dir / "phase2_yearly_lift.csv", index=False)

    challenger_visuals = "".join(
        [
            challenger_comparison_block(
                weekly,
                NO_GLOSSARY_DEVELOPMENT_PERIOD,
                minimal_change_development_root,
                "Development: 2009-2013",
                "Cumulative 50/50 capital growth for Trail5, V2, minimal-change, and the two selected sleeve experiments.",
                visible_challenger_root=minimal_change_visible_development_root,
                sleeve_splice_root=DEFAULT_SLEEVE_SPLICE_DEVELOPMENT_ROOT,
            ),
            challenger_comparison_block(
                weekly,
                GLOSSARY_VALIDATION_PERIOD,
                minimal_change_validation_root,
                "Out of sample: 2015-2019",
                "Cumulative 50/50 capital growth for Trail5, V2, minimal-change, and the two selected sleeve experiments.",
                sleeve_splice_root=DEFAULT_SLEEVE_SPLICE_OOS_ROOT,
            ),
        ]
    )
    output = suite_root / DEFAULT_OUTPUT_NAME
    report = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Mean-Reversion Out-of-Sample Testing</title>
<style>
:root{{--bg:#f4f6f8;--panel:#fff;--ink:#172331;--muted:#687589;--line:#d8e0e8;--head:#edf2f6;--accent:#247665;--good:#167052;--bad:#a83232;--blue:#2e6fbb;--gold:#b5812d}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.45}}
main{{max-width:1320px;margin:auto;padding:28px}} h1{{font-size:40px;line-height:1.05;margin:6px 0 10px}} h2{{margin:0 0 8px}} p{{color:var(--muted)}}
.nav{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}} .nav a{{background:#fff;border:1px solid var(--line);border-radius:999px;padding:7px 12px;text-decoration:none;color:#425166;font-weight:700;font-size:13px}}
.hero,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:22px;margin-bottom:16px;box-shadow:0 12px 30px rgba(22,33,44,.06)}}
.eyebrow{{color:var(--accent);font-size:12px;font-weight:900;text-transform:uppercase;letter-spacing:.06em}} .lock{{border-left:5px solid var(--gold);background:#fff8e9;padding:14px 16px;font-weight:750;color:#634813;margin-top:16px}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin-top:14px}} table{{width:100%;border-collapse:collapse;font-size:13px}} th{{background:var(--head);text-align:left;text-transform:uppercase;font-size:11px;letter-spacing:.05em;white-space:nowrap}} th,td{{padding:10px;border-bottom:1px solid var(--line);vertical-align:top}} tbody tr:nth-child(even){{background:#fafcfd}} td small{{display:block;color:var(--muted);max-width:390px;margin-top:4px}} .good{{color:var(--good)}} .bad{{color:var(--bad)}} .pending{{color:var(--muted)}}
.bar-row{{display:grid;grid-template-columns:220px 1fr 70px;gap:12px;align-items:center;margin:10px 0}} .bar-track{{height:14px;background:#edf1f4;border-radius:3px;overflow:hidden}} .bar{{height:100%}} .bar.positive{{background:var(--good)}} .bar.negative{{background:var(--bad)}}
.grid,.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .visual-stack{{display:grid;gap:22px;margin-top:16px}} .period-visual{{border-top:1px solid var(--line);padding-top:18px}} .period-copy{{display:flex;justify-content:space-between;gap:18px;align-items:flex-end;margin-bottom:12px}} .period-copy h3{{margin:0;font-size:22px}} .period-copy p{{margin:0;max-width:620px}} .comparison-card{{border-top:1px solid var(--line);padding-top:20px}} .comparison-card h3{{font-size:22px;margin:0 0 4px}} .comparison-card p{{margin:0 0 14px}} .comparison-legend{{font-size:14px;gap:24px}} h4{{margin:0 0 10px;font-size:15px;text-transform:uppercase;letter-spacing:.04em;color:#425166}}
svg{{width:100%;height:auto;background:#fbfcfd;border:1px solid var(--line);border-radius:8px}} .legend{{display:flex;gap:15px;flex-wrap:wrap;margin-bottom:8px;font-size:12px;font-weight:750}} .legend i{{display:inline-block;width:10px;height:10px;margin-right:5px;border-radius:2px}} .empty{{padding:24px;background:#f7f9fa;color:var(--muted);border:1px dashed var(--line);border-radius:8px}}
code{{background:#eef3f5;padding:2px 5px;border-radius:4px}} @media(max-width:850px){{main{{padding:16px}} .grid,.chart-grid{{grid-template-columns:1fr}} .bar-row{{grid-template-columns:1fr}} .period-copy{{display:block}}}}
</style></head><body><main>
<nav class='nav'><a href='mean_reversion_research_dashboard.html'>Research Hub</a><a href='prompt_engineering_dashboard.html'>Prompt Dashboard</a><a href='autoresearch_v2_cache_sweep/autoresearch_report.html'>Autoresearch</a><a href='future_experiments_dashboard.html'>Future Experiments</a></nav>
<header class='hero'><div class='eyebrow'>Locked evaluation</div><h1>Out-of-Sample Testing</h1><p>A focused comparison of prompt and portfolio evidence across one five-year development window and the separate 2015–2019 out-of-sample window.</p></header>

<section class='panel'><div class='eyebrow'>Evidence map</div><h2>How To Read The Results</h2><p><strong>Development 2009-2013:</strong> hypothesis generation and cache-based experiments, using dates from 2009-01-01 up to the exclusive 2014-01-01 endpoint. <strong>Out of sample 2015-2019:</strong> the separate five-year evaluation window, using dates from 2015-01-01 up to the exclusive 2020-01-01 endpoint.</p><p>Core harness: 60 long and 60 short candidates, 30/30 primary book, fixed five-trading-day C2C, zero costs, and 50/50 capital normalization.</p></section>

<section class='panel'><div class='eyebrow'>Out-of-sample evidence</div><h2>Out-of-Sample Results: 2015-2019</h2><p>The V2 top-30/30 result and the minimal-change temperature-1 challenger are shown against their matched Trail5 controls. Positive CAGR and Sharpe lift are better; negative drawdown change means the LLM drawdown became worse.</p>
<div class='table-wrap'><table><thead><tr><th>Frozen test</th><th>LLM CAGR</th><th>Trail5 CAGR</th><th>LLM lift</th><th>Long lift</th><th>Short lift</th><th>LLM Sharpe</th><th>Sharpe lift</th><th>LLM DD</th><th>DD change</th><th>Rebalances won</th><th>Positive years</th></tr></thead><tbody>{result_table(scorecard, GLOSSARY_VALIDATION_PERIOD)}{minimal_change_validation_table(minimal_change_validation_root)}</tbody></table></div><div class='lock'>Both rows use offset 3, a 60/60 candidate pool, a 30/30 equal-weight book, gross fixed five-day C2C accounting, and matched Trail5 controls. The minimal-change row uses temperature 1.0.</div></section>

<section class='panel'><div class='eyebrow'>Development reference</div><h2>Development Results: 2009-2013</h2><p>The original anonymized V2 top-30/30 result and the cached minimal-change temperature-1 challenger are shown together.</p>
<div class='table-wrap'><table><thead><tr><th>Frozen test</th><th>LLM CAGR</th><th>Trail5 CAGR</th><th>LLM lift</th><th>Long lift</th><th>Short lift</th><th>LLM Sharpe</th><th>Sharpe lift</th><th>LLM DD</th><th>DD change</th><th>Rebalances won</th><th>Positive years</th></tr></thead><tbody>{result_table(scorecard, NO_GLOSSARY_DEVELOPMENT_PERIOD)}{minimal_change_validation_table(minimal_change_development_root)}{minimal_change_validation_table(minimal_change_visible_development_root, label='Minimal change · temperature 1.0 · ticker visible', purpose='Does exposing ticker identity improve the minimal-change prompt versus matched Trail5?')}</tbody></table></div></section>

<section class='panel'><div class='eyebrow'>Head-to-head charts</div><h2>Trail5 And Phase 2 Challengers</h2><p>Every line uses the same top-30/30 equal-weight construction and 50/50 capital normalization, including the two sleeve experiments selected for OOS validation.</p><div class='visual-stack'>{challenger_visuals}</div></section>

</main></body></html>"""
    output.write_text(report, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--development-panel", type=Path, default=DEFAULT_DEVELOPMENT_PANEL)
    parser.add_argument(
        "--development-glossary-root",
        type=Path,
        default=DEFAULT_DEVELOPMENT_GLOSSARY_ROOT,
    )
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument(
        "--no-glossary-validation-root",
        type=Path,
        default=DEFAULT_NO_GLOSSARY_VALIDATION_ROOT,
    )
    parser.add_argument("--final-root", type=Path, default=DEFAULT_FINAL_ROOT)
    parser.add_argument("--exploratory-final-root", type=Path, default=DEFAULT_EXPLORATORY_FINAL_ROOT)
    parser.add_argument(
        "--minimal-change-development-root",
        type=Path,
        default=DEFAULT_MINIMAL_CHANGE_DEVELOPMENT_ROOT,
    )
    parser.add_argument(
        "--minimal-change-visible-development-root",
        type=Path,
        default=DEFAULT_MINIMAL_CHANGE_VISIBLE_DEVELOPMENT_ROOT,
    )
    parser.add_argument(
        "--minimal-change-validation-root",
        type=Path,
        default=DEFAULT_MINIMAL_CHANGE_VALIDATION_ROOT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_dashboard(
        args.suite_root,
        args.development_panel,
        args.validation_root,
        args.final_root,
        args.exploratory_final_root,
        args.no_glossary_validation_root,
        args.development_glossary_root,
        minimal_change_development_root=args.minimal_change_development_root,
        minimal_change_visible_development_root=args.minimal_change_visible_development_root,
        minimal_change_validation_root=args.minimal_change_validation_root,
    )
    print(f"Wrote out-of-sample testing dashboard: {output}")


if __name__ == "__main__":
    main()
