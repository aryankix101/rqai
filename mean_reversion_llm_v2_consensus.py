"""Offline rank-consensus experiment over four immutable V2 score caches.

This module never calls an LLM.  It validates the four current-epoch V2
caches, constructs every three-of-four mean-rank portfolio, and writes a
fully auditable set of derived research artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


EXPECTED_PROMPT_HASH = "02e28abb83b6"
EXPECTED_MODEL = "deepseek-v4-flash"
EXPECTED_TEMPERATURE = 1.0
EXPECTED_DATES = 251
POOL_PER_SIDE = 60
BOOK_PER_SIDE = 30
PERIODS_PER_YEAR = 52.0
PRIMARY_RUNS = ("run1", "run2", "run3")

DEFAULT_RUN_ROOTS = {
    "run1": Path("backtest_results/v2_system_prompt_temp1_v4flash0731_2009_2014"),
    "run2": Path("backtest_results/v2_system_prompt_temp1_v4flash0731_second_2009_2014"),
    "run3": Path("backtest_results/v2_system_prompt_temp1_v4flash0731_third_2009_2014"),
    # The historical folder name is misleading; its run_config and cache are anonymous.
    "run4": Path("backtest_results/v2_system_prompt_temp1_v4flash0731_ticker_visible_2009_2014"),
}
DEFAULT_OUTPUT_DIR = Path("backtest_results/v2_three_run_consensus_2009_2014")


@dataclass(frozen=True)
class SourceRun:
    run_id: str
    root: Path
    cache_path: Path
    cache_sha256: str
    config_sha256: str
    signals: dict[str, dict[str, Any]]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def deterministic_tiebreak(date: Any, side: str, ticker: str) -> str:
    """Outcome-blind final ordering key for otherwise exact consensus ties."""
    value = f"{pd.Timestamp(date).date()}|{str(side).upper()}|{str(ticker).upper()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def midranks(scores: pd.Series) -> pd.Series:
    """One is best; equal scores receive the arithmetic mean of occupied ranks."""
    numeric = pd.to_numeric(scores, errors="raise")
    return numeric.rank(method="average", ascending=False)


def _single_cache(root: Path) -> Path:
    matches = sorted((root / "cache").glob(f"*v2-{EXPECTED_PROMPT_HASH}_anon.json"))
    excluded = sorted((root / "cache").glob(f"*paired-control-v2-{EXPECTED_PROMPT_HASH}.json"))
    if excluded and not matches:
        raise ValueError(f"No-regime paired-control caches are excluded: {excluded[0]}")
    if len(matches) != 1:
        raise ValueError(f"Expected one exact V2 anonymous cache under {root}, found {len(matches)}")
    if "paired-control" in matches[0].name or "v2nr-" in matches[0].name:
        raise ValueError(f"No-regime paired-control caches are excluded: {matches[0]}")
    return matches[0]


def load_source_run(run_id: str, root: Path) -> SourceRun:
    root = Path(root)
    config_path = root / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config = {
        "model": EXPECTED_MODEL,
        "temperature": EXPECTED_TEMPERATURE,
        "thinking_mode": "disabled",
        "prompt_variant": "v2",
        "prompt_hash": EXPECTED_PROMPT_HASH,
        "anonymize_tickers": True,
        "pool_sizes": [POOL_PER_SIDE],
        "book_sizes": [BOOK_PER_SIDE],
        "llm_pool_size": POOL_PER_SIDE,
        "anchor_offsets": [3],
        "schedule": "anchors",
        "cost_bps_levels": [0.0],
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Incompatible run config for {run_id}: {mismatches}")

    cache_path = _single_cache(root)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    expected_metadata = {
        "model": EXPECTED_MODEL,
        "temperature": EXPECTED_TEMPERATURE,
        "thinking_mode": "disabled",
        "prompt_hash": EXPECTED_PROMPT_HASH,
        "anonymize_tickers": True,
        "pool_count": EXPECTED_DATES,
    }
    bad_metadata = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    signals = payload.get("signals")
    if bad_metadata or not isinstance(signals, dict) or len(signals) != EXPECTED_DATES:
        raise ValueError(
            f"Incompatible cache for {run_id}: metadata={bad_metadata}, "
            f"signal_count={len(signals) if isinstance(signals, dict) else None}"
        )
    return SourceRun(
        run_id=run_id,
        root=root,
        cache_path=cache_path,
        cache_sha256=file_sha256(cache_path),
        config_sha256=file_sha256(config_path),
        signals=signals,
    )


def validate_sources(sources: Sequence[SourceRun]) -> None:
    if len(sources) != 4 or len({source.run_id for source in sources}) != 4:
        raise ValueError("Consensus experiment requires four uniquely named source runs")
    resolved = [source.cache_path.resolve() for source in sources]
    hashes = [source.cache_sha256 for source in sources]
    if len(set(resolved)) != 4 or len(set(hashes)) != 4:
        raise ValueError("Consensus sources must be four distinct cache files with distinct contents")

    reference_keys = set(sources[0].signals)
    reference_dates = {
        str(entry.get("rebalance_date", key.split("|", 1)[0]))
        for key, entry in sources[0].signals.items()
    }
    if len(reference_dates) != EXPECTED_DATES:
        raise ValueError("Consensus source does not contain 251 distinct rebalance dates")
    for source in sources[1:]:
        if set(source.signals) != reference_keys:
            raise ValueError(f"Signal-date/storage-key coverage differs for {source.run_id}")
    for storage_key in sorted(reference_keys):
        reference = sources[0].signals[storage_key]
        reference_hash = reference.get("universe_hash")
        for source in sources:
            entry = source.signals[storage_key]
            for side in ("long_scores", "short_scores"):
                values = entry.get(side)
                if not isinstance(values, list) or len(values) != POOL_PER_SIDE:
                    raise ValueError(f"{source.run_id} {storage_key} must contain exactly 60 {side}")
                tickers = [str(row.get("ticker", "")).upper() for row in values]
                if any(not ticker for ticker in tickers) or len(set(tickers)) != POOL_PER_SIDE:
                    raise ValueError(f"{source.run_id} {storage_key} contains missing/duplicate {side} tickers")
                scores = pd.to_numeric([row.get("score") for row in values], errors="coerce")
                if not np.isfinite(scores).all():
                    raise ValueError(f"{source.run_id} {storage_key} contains invalid {side} scores")
        reference_members = {
            side: {str(row.get("ticker", "")).upper() for row in reference.get(side, [])}
            for side in ("long_scores", "short_scores")
        }
        if any(len(values) != POOL_PER_SIDE for values in reference_members.values()):
            raise ValueError(f"{storage_key} does not contain 60 candidates on each side")
        for source in sources[1:]:
            entry = source.signals[storage_key]
            if entry.get("universe_hash") != reference_hash:
                raise ValueError(f"Universe hash mismatch for {storage_key} in {source.run_id}")
            members = {
                side: {str(row.get("ticker", "")).upper() for row in entry.get(side, [])}
                for side in ("long_scores", "short_scores")
            }
            if members != reference_members:
                raise ValueError(f"Candidate membership mismatch for {storage_key} in {source.run_id}")


def rank_panel(sources: Sequence[SourceRun]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in sources:
        for storage_key, entry in sorted(source.signals.items()):
            date = pd.Timestamp(entry.get("rebalance_date", storage_key.split("|", 1)[0]))
            for key, side in (("long_scores", "LONG"), ("short_scores", "SHORT")):
                for item in entry[key]:
                    rows.append(
                        {
                            "date": date,
                            "storage_key": storage_key,
                            "side": side,
                            "ticker": str(item["ticker"]).upper(),
                            "anonymous_ticker": str(item.get("anonymous_ticker", "")),
                            "run_id": source.run_id,
                            "raw_score": float(item["score"]),
                            "universe_hash": str(entry["universe_hash"]),
                        }
                    )
    panel = pd.DataFrame(rows)
    panel["run_rank"] = panel.groupby(["date", "side", "run_id"])["raw_score"].transform(midranks)
    counts = panel.groupby(["date", "side", "run_id"])["ticker"].nunique()
    if not counts.eq(POOL_PER_SIDE).all():
        raise ValueError("Rank panel lost candidate coverage")
    return panel.sort_values(["date", "side", "ticker", "run_id"], kind="mergesort")


def aggregate_ranks(panel: pd.DataFrame, run_ids: Sequence[str], label: str) -> pd.DataFrame:
    selected = panel[panel["run_id"].isin(run_ids)].copy()
    if selected["run_id"].nunique() != len(run_ids):
        raise ValueError(f"Missing requested source run for {label}")
    wide = selected.pivot_table(
        index=["date", "side", "ticker"], columns="run_id", values="run_rank", aggfunc="first"
    ).reset_index()
    run_columns = list(run_ids)
    if wide[run_columns].isna().any().any():
        raise ValueError(f"Incomplete rank coverage for {label}")
    wide["average_rank"] = wide[run_columns].mean(axis=1)
    wide["median_rank"] = wide[run_columns].median(axis=1)
    wide["worst_rank"] = wide[run_columns].max(axis=1)
    wide["rank_std"] = wide[run_columns].std(axis=1, ddof=0)
    wide["blind_tiebreak"] = [
        deterministic_tiebreak(date, side, ticker)
        for date, side, ticker in zip(wide["date"], wide["side"], wide["ticker"])
    ]
    wide = wide.sort_values(
        ["date", "side", "average_rank", "median_rank", "worst_rank", "blind_tiebreak"],
        kind="mergesort",
    )
    wide["consensus_rank"] = wide.groupby(["date", "side"]).cumcount() + 1
    wide["selected"] = wide["consensus_rank"].le(BOOK_PER_SIDE)
    wide["ensemble"] = label
    wide["source_runs"] = "+".join(run_ids)
    return wide


def load_outcome_panel(root: Path) -> pd.DataFrame:
    path = root / "pool60_pos30/anchor_3/baseline_pool/signal_checks.csv"
    data = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", "pool_side", "fwd_c2c", "fwd_o2o", "trail_ret"}
    if missing := required - set(data.columns):
        raise ValueError(f"Outcome panel is missing {sorted(missing)}")
    data["ticker"] = data["ticker"].astype(str).str.upper()
    data["pool_side"] = data["pool_side"].astype(str).str.upper()
    return data


def portfolio_from_ranks(ranks: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = ranks.merge(
        outcomes[["date", "ticker", "pool_side", "fwd_c2c", "fwd_o2o", "trail_ret"]],
        left_on=["date", "ticker", "side"],
        right_on=["date", "ticker", "pool_side"],
        how="left",
        validate="one_to_one",
    )
    if merged["pool_side"].isna().any():
        raise ValueError("Consensus ranks could not be joined to the point-in-time candidate panel")
    chosen = merged[merged["selected"]].copy()
    counts = chosen.groupby(["date", "side"])["ticker"].size()
    if not counts.eq(BOOK_PER_SIDE).all():
        raise ValueError("Consensus portfolio does not contain exactly 30 names per side")
    chosen["weight"] = 1.0 / BOOK_PER_SIDE
    chosen["side_c2c"] = np.where(chosen["side"].eq("SHORT"), -chosen["fwd_c2c"], chosen["fwd_c2c"])
    chosen["side_o2o"] = np.where(chosen["side"].eq("SHORT"), -chosen["fwd_o2o"], chosen["fwd_o2o"])

    side = chosen.groupby(["date", "side"], as_index=False).agg(
        raw_c2c=("fwd_c2c", "mean"), raw_o2o=("fwd_o2o", "mean")
    )
    long = side[side["side"].eq("LONG")].set_index("date")["raw_c2c"]
    short = side[side["side"].eq("SHORT")].set_index("date")["raw_c2c"]
    weekly = pd.DataFrame({"long_ret_c2c": long, "short_ret_c2c": short}).sort_index()
    weekly["long_contribution"] = 0.5 * weekly["long_ret_c2c"]
    weekly["short_contribution"] = -0.5 * weekly["short_ret_c2c"]
    weekly["capital_c2c"] = weekly["long_contribution"] + weekly["short_contribution"]

    for side_name, column in (("LONG", "turnover_long"), ("SHORT", "turnover_short")):
        previous: set[str] = set()
        values: dict[pd.Timestamp, float] = {}
        for date, group in chosen[chosen["side"].eq(side_name)].groupby("date", sort=True):
            current = set(group["ticker"])
            values[pd.Timestamp(date)] = 1.0 if not previous else 1.0 - len(previous & current) / BOOK_PER_SIDE
            previous = current
        weekly[column] = pd.Series(values)
    return weekly.reset_index(), chosen


def return_metrics(returns: Iterable[float]) -> dict[str, float | int]:
    values = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"n_periods": 0}
    equity = (1.0 + values).cumprod()
    terminal = float(equity.iloc[-1])
    ann_return = float(values.mean() * PERIODS_PER_YEAR)
    ann_vol = float(values.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(values) > 1 else np.nan
    return {
        "n_periods": int(len(values)),
        "annual_return_pct": ann_return * 100.0,
        "cagr_pct": (terminal ** (PERIODS_PER_YEAR / len(values)) - 1.0) * 100.0 if terminal > 0 else np.nan,
        "annual_vol_pct": ann_vol * 100.0,
        "sharpe": ann_return / ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else np.nan,
        "max_drawdown_pct": float((equity / equity.cummax() - 1.0).min() * 100.0),
        "hit_rate_pct": float(values.gt(0).mean() * 100.0),
    }


def moving_block_bootstrap(
    treatment: pd.Series,
    control: pd.Series,
    *,
    block_size: int = 8,
    n_boot: int = 10_000,
    seed: int = 1701,
) -> dict[str, float | int]:
    paired = pd.concat([treatment.rename("t"), control.rename("c")], axis=1).dropna()
    delta = (paired["t"] - paired["c"]).to_numpy(dtype=float)
    if not len(delta):
        raise ValueError("Cannot bootstrap an empty paired series")
    rng = np.random.default_rng(seed)
    starts = np.arange(len(delta))
    samples = np.empty(n_boot, dtype=float)
    blocks_needed = int(np.ceil(len(delta) / block_size))
    offsets = np.arange(block_size)
    for idx in range(n_boot):
        picked_starts = rng.choice(starts, size=blocks_needed, replace=True)
        positions = ((picked_starts[:, None] + offsets) % len(delta)).ravel()[: len(delta)]
        samples[idx] = float(delta[positions].mean() * PERIODS_PER_YEAR * 100.0)
    observed = float(delta.mean() * PERIODS_PER_YEAR * 100.0)
    return {
        "n_periods": int(len(delta)),
        "n_boot": int(n_boot),
        "block_size": int(block_size),
        "mean_annualized_delta_pct": observed,
        "lower_90_pct": float(np.quantile(samples, 0.05)),
        "upper_90_pct": float(np.quantile(samples, 0.95)),
        "lower_95_pct": float(np.quantile(samples, 0.025)),
        "upper_95_pct": float(np.quantile(samples, 0.975)),
        "probability_positive_pct": float(np.mean(samples > 0.0) * 100.0),
    }


def source_weekly(source: SourceRun) -> pd.DataFrame:
    path = source.root / "pool60_pos30/anchor_3/llm_pool_v2/weekly_returns.csv"
    data = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if "capital_c2c" not in data:
        if "ls_c2c_0" not in data:
            raise ValueError(f"Source weekly return file lacks a capital return column: {path}")
        # Legacy ls_c2c_0 is a 100% long plus 100% short spread.  All experiment
        # reporting is capital-normalized to a 50/50 long/short allocation.
        data["capital_c2c"] = 0.5 * data["ls_c2c_0"]
    data["long_contribution"] = 0.5 * data["long_ret_c2c"]
    data["short_contribution"] = -0.5 * data["short_ret_c2c"]
    return data


def trail5_weekly(source: SourceRun) -> pd.DataFrame:
    path = source.root / "pool60_pos30/anchor_3/baseline_pool/weekly_returns.csv"
    data = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if "capital_c2c" not in data:
        column = "ls_c2c_0" if "ls_c2c_0" in data else "ls_c2c"
        data["capital_c2c"] = 0.5 * data[column]
    return data


def stability_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlation_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(panel["run_id"].unique()), 2):
        pair = panel[panel["run_id"].isin([left, right])].pivot_table(
            index=["date", "side", "ticker"], columns="run_id", values="run_rank", aggfunc="first"
        ).reset_index()
        for side, group in pair.groupby("side"):
            correlations = group.groupby("date").apply(
                lambda rows: rows[left].corr(rows[right], method="spearman"), include_groups=False
            )
            correlation_rows.append(
                {"run_left": left, "run_right": right, "side": side, "mean_spearman": correlations.mean()}
            )
            for date, rows in group.groupby("date"):
                a = set(rows.nsmallest(BOOK_PER_SIDE, left)["ticker"])
                b = set(rows.nsmallest(BOOK_PER_SIDE, right)["ticker"])
                overlap_rows.append(
                    {
                        "date": date,
                        "side": side,
                        "run_left": left,
                        "run_right": right,
                        "overlap_count": len(a & b),
                        "jaccard": len(a & b) / len(a | b),
                    }
                )
    return pd.DataFrame(correlation_rows), pd.DataFrame(overlap_rows)


def render_report(summary: dict[str, Any], scorecard: pd.DataFrame, curve: pd.DataFrame) -> str:
    def table(frame: pd.DataFrame) -> str:
        return frame.to_html(index=False, border=0, classes="data", float_format=lambda value: f"{value:.3f}")

    decision = summary["hypothesis"]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>V2 Three-Run Consensus</title>
<style>body{{font:14px system-ui;margin:32px;color:#18212b}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.callout{{padding:14px;background:#eef5ff;border-left:4px solid #3977c3}}</style></head>
<body><h1>V2 Three-Run Rank Consensus</h1>
<div class="callout"><b>Decision:</b> {decision['label']}. No LLM calls were made.</div>
<h2>Three-run ensembles</h2>{table(scorecard)}
<h2>Ensemble-size curve</h2>{table(curve)}
<p>Primary ensemble: run1+run2+run3. Run 4 is used only in prespecified robustness triplets. No no-regime paired-control cache is included.</p>
</body></html>"""


def run_experiment(
    run_roots: dict[str, Path] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    n_boot: int = 10_000,
) -> dict[str, Any]:
    roots = run_roots or DEFAULT_RUN_ROOTS
    sources = [load_source_run(run_id, roots[run_id]) for run_id in sorted(roots)]
    validate_sources(sources)
    panel = rank_panel(sources)
    outcomes = load_outcome_panel(sources[0].root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_returns = {source.run_id: source_weekly(source) for source in sources}
    trail5_returns = trail5_weekly(sources[0])
    trail5_metrics = return_metrics(trail5_returns["capital_c2c"])
    all_weekly: list[pd.DataFrame] = []
    all_ranks: list[pd.DataFrame] = []
    all_selections: list[pd.DataFrame] = []
    score_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    triplet_positive = 0
    primary_metrics: dict[str, Any] = {}
    primary_bootstrap: dict[str, Any] = {}

    for size in range(1, 5):
        for combo in itertools.combinations(sorted(roots), size):
            label = f"k{size}_" + "_".join(combo)
            ranks = aggregate_ranks(panel, combo, label)
            weekly, selections = portfolio_from_ranks(ranks, outcomes)
            weekly["ensemble"] = label
            selections["ensemble"] = label
            all_ranks.append(ranks)
            all_weekly.append(weekly)
            all_selections.append(selections)
            metrics = return_metrics(weekly["capital_c2c"])
            row = {
                "ensemble": label,
                "ensemble_size": size,
                "source_runs": "+".join(combo),
                **metrics,
                "annualized_long_contribution_pct": float(weekly["long_contribution"].mean() * PERIODS_PER_YEAR * 100.0),
                "annualized_short_contribution_pct": float(weekly["short_contribution"].mean() * PERIODS_PER_YEAR * 100.0),
                "avg_turnover_long": float(weekly["turnover_long"].mean()),
                "avg_turnover_short": float(weekly["turnover_short"].mean()),
            }
            trail_boot = moving_block_bootstrap(
                weekly.set_index("date")["capital_c2c"],
                trail5_returns["capital_c2c"],
                n_boot=n_boot,
            )
            row.update(
                {
                    "paired_lift_vs_trail5_pct": trail_boot["mean_annualized_delta_pct"],
                    "trail5_cagr_pct": trail5_metrics["cagr_pct"],
                    "trail5_sharpe": trail5_metrics["sharpe"],
                }
            )
            bootstrap_rows.append({"ensemble": label, "comparison": "Trail5", **trail_boot})
            if size == 3:
                member_mean = pd.concat(
                    [source_returns[run_id]["capital_c2c"].rename(run_id) for run_id in combo], axis=1
                ).mean(axis=1)
                treatment = weekly.set_index("date")["capital_c2c"]
                boot = moving_block_bootstrap(treatment, member_mean, n_boot=n_boot)
                row.update(
                    {
                        "paired_lift_vs_member_average_pct": boot["mean_annualized_delta_pct"],
                        "bootstrap_lower_95_pct": boot["lower_95_pct"],
                        "bootstrap_probability_positive_pct": boot["probability_positive_pct"],
                    }
                )
                bootstrap_rows.append(
                    {"ensemble": label, "comparison": "mean_constituent", **boot}
                )
                triplet_positive += int(float(boot["mean_annualized_delta_pct"]) > 0.0)
                if tuple(combo) == PRIMARY_RUNS:
                    primary_metrics = row
                    primary_bootstrap = boot
            score_rows.append(row)

    scorecard = pd.DataFrame(score_rows)
    triplets = scorecard[scorecard["ensemble_size"].eq(3)].copy()
    source_metric_rows = [
        {"run_id": run_id, **return_metrics(frame["capital_c2c"])}
        for run_id, frame in source_returns.items()
    ]
    source_metric_rows.append({"run_id": "Trail5", **trail5_metrics})
    source_metrics = pd.DataFrame(source_metric_rows)
    primary_member_metrics = source_metrics[source_metrics["run_id"].isin(PRIMARY_RUNS)]
    drawdown_tolerance = float(primary_member_metrics["max_drawdown_pct"].median()) - 2.0
    checks = {
        "positive_paired_lift": float(primary_bootstrap.get("mean_annualized_delta_pct", np.nan)) > 0.0,
        "sharpe_above_constituent_median": float(primary_metrics.get("sharpe", np.nan))
        > float(primary_member_metrics["sharpe"].median()),
        "drawdown_within_two_points": float(primary_metrics.get("max_drawdown_pct", -np.inf)) >= drawdown_tolerance,
        "at_least_three_positive_triplets": triplet_positive >= 3,
    }
    supported = all(checks.values())
    statistically_strong = supported and float(primary_bootstrap.get("lower_95_pct", -np.inf)) > 0.0
    label = "statistically_strong" if statistically_strong else ("supported" if supported else "not_supported")

    correlation, overlap = stability_tables(panel)
    curve = (
        scorecard.groupby("ensemble_size", as_index=False)
        .agg(
            combinations=("ensemble", "size"),
            mean_cagr_pct=("cagr_pct", "mean"),
            min_cagr_pct=("cagr_pct", "min"),
            max_cagr_pct=("cagr_pct", "max"),
            mean_sharpe=("sharpe", "mean"),
            mean_max_drawdown_pct=("max_drawdown_pct", "mean"),
        )
    )
    provenance = {
        "schema_version": 1,
        "artifact_type": "offline_v2_three_run_rank_consensus",
        "llm_calls_made": 0,
        "prompt_hash": EXPECTED_PROMPT_HASH,
        "model": EXPECTED_MODEL,
        "temperature": EXPECTED_TEMPERATURE,
        "thinking_mode": "disabled",
        "primary_runs": list(PRIMARY_RUNS),
        "no_regime_paired_controls_included": False,
        "sources": [
            {
                "run_id": source.run_id,
                "root": str(source.root.resolve()),
                "cache_path": str(source.cache_path.resolve()),
                "cache_sha256": source.cache_sha256,
                "config_sha256": source.config_sha256,
            }
            for source in sources
        ],
        "derivation_hash": hashlib.sha256(
            canonical_json({source.run_id: source.cache_sha256 for source in sources})
        ).hexdigest(),
    }
    summary = {
        "provenance": provenance,
        "hypothesis": {"label": label, "supported": supported, "statistically_strong": statistically_strong, "checks": checks},
        "primary_metrics": primary_metrics,
        "primary_bootstrap": primary_bootstrap,
        "positive_triplets": triplet_positive,
        "triplet_count": 4,
    }

    pd.concat(all_ranks, ignore_index=True).to_csv(output_dir / "candidate_ranks.csv", index=False)
    pd.concat(all_selections, ignore_index=True).to_csv(output_dir / "portfolio_selections.csv", index=False)
    pd.concat(all_weekly, ignore_index=True).to_csv(output_dir / "weekly_returns.csv", index=False)
    triplets.to_csv(output_dir / "triplet_comparison.csv", index=False)
    scorecard.to_csv(output_dir / "all_ensemble_scorecard.csv", index=False)
    curve.to_csv(output_dir / "ensemble_size_curve.csv", index=False)
    source_metrics.to_csv(output_dir / "source_run_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output_dir / "moving_block_bootstrap.csv", index=False)
    correlation.to_csv(output_dir / "pairwise_rank_correlation.csv", index=False)
    overlap.to_csv(output_dir / "top30_overlap.csv", index=False)
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (output_dir / "report.html").write_text(render_report(summary, triplets, curve), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the zero-call four-cache V2 consensus experiment")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-boot", type=int, default=10_000)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = run_experiment(output_dir=args.output_dir, n_boot=args.n_boot)
    print(json.dumps(summary["hypothesis"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
