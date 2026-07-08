#!/usr/bin/env python3
"""Generate a combined report for the legacy composite and v1 reversal runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from html import escape
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any


LEGACY_ROOT = Path("backtest_results/factor_backtest/composite_scores_llm_signal_2009_2014_5d_horizon")
V1_ROOT = Path("backtest_results/llm_reversal_signal_v1_2009_2014")
DEFAULT_OUTPUT = Path("backtest_results/llm_reversal_2009_2014_combined_report.html")

LEGACY_DEFAULT_KEY = "legacy_2p5"
LEGACY_COST_OPTIONS = [
    {"key": "legacy_0", "label": "Legacy daily, 0 bps cost", "costBps": 0.0},
    {"key": "legacy_2p5", "label": "Legacy daily, 2.5 bps cost", "costBps": 2.5},
    {"key": "legacy_5", "label": "Legacy daily, 5 bps cost", "costBps": 5.0},
]
TRADING_DAYS = 252
WEEKLY_PERIODS = 52

LEGACY_LABELS = {
    "llm_raw_confidence_score": "Legacy LLM Raw Confidence",
    "llm_rank_score": "Legacy LLM Rank",
    "llm_rank_confidence_score": "Legacy LLM Rank + Confidence",
    "mean_reversion_raw_score": "Legacy Det Raw",
}

LEGACY_SIGNAL_DESCRIPTIONS = {
    "llm_raw_confidence_score": "LLM confidence only",
    "llm_rank_score": "LLM rank only",
    "llm_rank_confidence_score": "LLM rank + confidence",
    "mean_reversion_raw_score": "deterministic raw mean-reversion score",
}

EXCLUDED_LEGACY_SCORE_TYPES = {"mean_reversion_rank_score"}

V1_RUNS = [
    {
        "dir": "baseline_universe",
        "scorecard_run": "Baseline-Universe",
        "label": "V1 Baseline Universe",
        "include_in_narrow_replays": False,
    },
    {
        "dir": "baseline_pool",
        "scorecard_run": "Baseline-Pool",
        "label": "V1 Baseline Pool {book}",
        "include_in_narrow_replays": True,
    },
    {
        "dir": "llm_pool_v1",
        "scorecard_run": "LLM-Pool v1",
        "label": "V1 LLM Pool {book}",
        "include_in_narrow_replays": True,
    },
]

V1_NARROW_REPLAYS = [
    {"suffix": "_pos30", "variant_id": "pos30", "book": "30/30", "positions_per_side": 30},
    {"suffix": "_pos50", "variant_id": "pos50", "book": "50/50", "positions_per_side": 50},
]

V1_EXTRA_VARIANTS = [
    {
        "root": Path("backtest_results/llm_reversal_raw_confidence_v1_2009_2014_pool60_pos30"),
        "variant_id": "raw_conf_pool60_pos30",
        "book": "30/30",
        "positions_per_side": 30,
        "pool_per_side": 60,
        "include_universe": False,
        "family": "LLM Reversal v1 Raw Confidence",
        "label_prefix": "V1 Raw Confidence",
        "protocol_label": "Raw confidence prompt",
    },
    {
        "root": Path("backtest_results/llm_reversal_rank_confidence_v1_2009_2014_pool60_pos30"),
        "variant_id": "rank_conf_pool60_pos30",
        "book": "30/30",
        "positions_per_side": 30,
        "pool_per_side": 60,
        "include_universe": False,
        "family": "LLM Reversal v1 Rank + Confidence",
        "label_prefix": "V1 Rank + Confidence",
        "protocol_label": "Rank + confidence prompt",
    },
]

V1_RETURN_OPTIONS = [
    {
        "key": "ls_c2c_0",
        "label": "V1 close-to-close, 0 bps cost",
        "scorecard": ("C2C", 0.0),
        "yearly": "LS_C2C_Gross",
    },
    {
        "key": "ls_c2c_2p5",
        "label": "V1 close-to-close, 2.5 bps cost",
        "scorecard": ("C2C", 2.5),
        "yearly": "LS_C2C_2.5bps",
    },
    {
        "key": "ls_c2c_5",
        "label": "V1 close-to-close, 5 bps cost",
        "scorecard": ("C2C", 5.0),
        "yearly": "LS_C2C_5bps",
    },
    {
        "key": "ls_o2o_0",
        "label": "V1 open-to-open, 0 bps cost",
        "scorecard": ("O2O", 0.0),
        "yearly": "LS_O2O_Gross",
    },
    {
        "key": "ls_o2o_2p5",
        "label": "V1 open-to-open, 2.5 bps cost",
        "scorecard": ("O2O", 2.5),
        "yearly": "LS_O2O_2.5bps",
    },
    {
        "key": "ls_o2o_5",
        "label": "V1 open-to-open, 5 bps cost",
        "scorecard": ("O2O", 5.0),
        "yearly": "LS_O2O_5bps",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fh:
        return json.load(fh)


def to_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def pct_decimal_to_pct(value: float) -> float:
    return value * 100.0


def max_drawdown_from_growth(points: list[dict[str, Any]], key: str) -> float:
    peak = 0.0
    worst = 0.0
    for point in points:
        growth = float(point["values"][key]["growth"])
        peak = max(peak, growth)
        if peak:
            worst = min(worst, growth / peak - 1.0)
    return worst * 100.0


def compound_return(returns: list[float]) -> float:
    growth = 1.0
    for ret in returns:
        growth *= 1.0 + ret
    return (growth - 1.0) * 100.0


def annual_vol_pct(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    return stdev(returns) * sqrt(TRADING_DAYS) * 100.0


def hit_rate_pct(returns: list[float]) -> float | None:
    if not returns:
        return None
    return sum(1 for ret in returns if ret > 0.0) / len(returns) * 100.0


def annualized_mean_pct(returns: list[float], periods_per_year: float) -> float | None:
    if not returns:
        return None
    return mean(returns) * periods_per_year * 100.0


def sharpe_ratio(returns: list[float], periods_per_year: float) -> float:
    if len(returns) < 2:
        return 0.0
    vol = stdev(returns)
    return mean(returns) / vol * sqrt(periods_per_year) if vol > 0 else 0.0


def safe_cagr_pct(returns: list[float], periods_per_year: float) -> float:
    if not returns:
        return 0.0
    cumulative = 1.0
    for ret in returns:
        cumulative *= 1.0 + ret
    if cumulative <= 0:
        return -100.0
    return (cumulative ** (periods_per_year / len(returns)) - 1.0) * 100.0


def drawdown_points(points: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    peak = 1.0
    rows: list[dict[str, Any]] = []
    for point in points:
        growth = float(point["values"][key]["growth"])
        peak = max(peak, growth)
        drawdown = growth / peak - 1.0 if peak else 0.0
        rows.append({"date": point["date"], "drawdownPct": drawdown * 100.0})
    return rows


def rolling_mean_points(rows: list[dict[str, str]], column: str, window: int) -> list[dict[str, Any]]:
    values: list[tuple[str, float]] = []
    for row in rows:
        if row.get(column) in (None, ""):
            continue
        values.append((row["date"], to_float(row.get(column))))
    output: list[dict[str, Any]] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        subset = [value for _, value in values[start : idx + 1]]
        if len(subset) >= max(3, min(window, 6)):
            output.append({"date": values[idx][0], "value": mean(subset)})
    return output


def regime_breakdown(
    rows: list[dict[str, str]],
    return_columns: list[str],
    periods_per_year: float,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        regime = row.get("regime") or "unknown"
        for column in return_columns:
            if column in row and row.get(column) not in (None, ""):
                grouped[column][regime].append(to_float(row.get(column)))

    output: dict[str, list[dict[str, Any]]] = {}
    for column in return_columns:
        output[column] = [
            {
                "regime": regime,
                "n": len(values),
                "annualReturnPct": annualized_mean_pct(values, periods_per_year),
                "sharpe": sharpe_ratio(values, periods_per_year),
                "hitRatePct": hit_rate_pct(values),
            }
            for regime, values in sorted(grouped[column].items())
            if values
        ]
    return output


def top_week_rows(rows: list[dict[str, str]], return_columns: list[str], limit: int = 4) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for column in return_columns:
        parsed: list[dict[str, Any]] = []
        for row in rows:
            if column not in row or row.get(column) in (None, ""):
                continue
            is_c2c = "_c2c" in column
            parsed.append(
                {
                    "date": row["date"],
                    "returnPct": to_float(row.get(column)) * 100.0,
                    "longPct": to_float(row.get("long_ret_c2c" if is_c2c else "long_ret_o2o")) * 100.0,
                    "shortPct": -to_float(row.get("short_ret_c2c" if is_c2c else "short_ret_o2o")) * 100.0,
                    "ic": to_float(row.get("ic_c2c" if is_c2c else "ic_o2o")),
                    "turnover": to_float(row.get("turnover_long")) + to_float(row.get("turnover_short")),
                    "regime": row.get("regime") or "unknown",
                }
            )
        output[column] = {
            "best": sorted(parsed, key=lambda item: item["returnPct"], reverse=True)[:limit],
            "worst": sorted(parsed, key=lambda item: item["returnPct"])[:limit],
        }
    return output


def top_period_rows(
    rows_by_key: dict[str, list[dict[str, Any]]],
    limit: int = 4,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        key: {
            "best": sorted(rows, key=lambda item: item["returnPct"], reverse=True)[:limit],
            "worst": sorted(rows, key=lambda item: item["returnPct"])[:limit],
        }
        for key, rows in rows_by_key.items()
    }


def decile_profile(rows: list[dict[str, str]], periods_per_year: float) -> dict[str, list[dict[str, Any]]]:
    profiles: dict[str, list[dict[str, Any]]] = {"c2c": [], "o2o": []}
    for execution in ("c2c", "o2o"):
        for decile in range(1, 11):
            column = f"dec_{decile}_{execution}"
            values = [to_float(row.get(column)) for row in rows if row.get(column) not in (None, "")]
            profiles[execution].append(
                {
                    "decile": decile,
                    "annualReturnPct": annualized_mean_pct(values, periods_per_year),
                    "n": len(values),
                }
            )
    return profiles


def concentration_from_selections(rows: list[dict[str, str]], limit: int = 8) -> dict[str, Any]:
    by_side: dict[str, dict[str, int]] = {"LONG": defaultdict(int), "SHORT": defaultdict(int)}
    dates_by_side: dict[str, set[str]] = {"LONG": set(), "SHORT": set()}
    for row in rows:
        side = (row.get("position_type") or row.get("side") or "").upper()
        ticker = (row.get("ticker") or "").upper()
        date = row.get("rebalance_date") or row.get("date") or row.get("signal_date") or ""
        if side in by_side and ticker:
            by_side[side][ticker] += 1
            if date:
                dates_by_side[side].add(date)
    return {
        side.lower(): [
            {
                "ticker": ticker,
                "count": count,
                "periodPct": count / max(len(dates_by_side[side]), 1) * 100.0,
            }
            for ticker, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        ]
        for side, counts in by_side.items()
    }


def copy_like_summary(root: Path, run_dir: str) -> dict[str, Any] | None:
    path = root / run_dir / "score_trail5_correlations.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    parsed = [
        {
            "date": row.get("date"),
            "side": row.get("pool_side"),
            "corr": to_float(row.get("score_trail5_spearman")),
        }
        for row in rows
        if row.get("score_trail5_spearman") not in (None, "")
    ]
    if not parsed:
        return None
    by_side: dict[str, list[float]] = defaultdict(list)
    for row in parsed:
        by_side[str(row["side"])].append(float(row["corr"]))
    return {
        "n": len(parsed),
        "avgAbsCorr": mean(abs(float(row["corr"])) for row in parsed),
        "highAbsCorrCount": sum(1 for row in parsed if abs(float(row["corr"])) > 0.9),
        "bySide": [
            {
                "side": side,
                "avgCorr": mean(values),
                "avgAbsCorr": mean(abs(value) for value in values),
                "highAbsCorrCount": sum(1 for value in values if abs(value) > 0.9),
            }
            for side, values in sorted(by_side.items())
        ],
        "top": sorted(parsed, key=lambda row: abs(float(row["corr"])), reverse=True)[:10],
    }


def score_spread_summary(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    llm_rows = [
        row
        for row in rows
        if (row.get("score_provider") or "").lower() == "llm" and row.get("llm_score") not in (None, "")
    ]
    if not llm_rows:
        return None
    by_side: dict[str, list[float]] = defaultdict(list)
    by_date: dict[str, list[float]] = defaultdict(list)
    for row in llm_rows:
        score = to_float(row.get("llm_score"))
        by_side[row.get("pool_side") or "unknown"].append(score)
        by_date[row["date"]].append(score)
    date_spreads = [max(values) - min(values) for values in by_date.values() if values]
    return {
        "n": len(llm_rows),
        "avgDateSpread": mean(date_spreads) if date_spreads else None,
        "bySide": [
            {
                "side": side,
                "avgScore": mean(values),
                "minScore": min(values),
                "maxScore": max(values),
            }
            for side, values in sorted(by_side.items())
        ],
    }


def build_legacy_series(root: Path) -> list[dict[str, Any]]:
    config = read_json(root / "run_config.json")
    leaderboard = read_csv(root / "compact_leaderboard.csv")
    configured_cost_bps = float(config.get("cost_bps", 0.0) or 0.0)
    series: list[dict[str, Any]] = []

    for row in leaderboard:
        score_type = row["score_type"]
        if score_type in EXCLUDED_LEGACY_SCORE_TYPES:
            continue
        pnl_path = root / score_type / "factor_pnl.csv"
        pnl_rows = read_csv(pnl_path)
        selection_path = root / score_type / "portfolio_selections.csv"
        selection_rows = read_csv(selection_path) if selection_path.exists() else []
        returns_by_key: dict[str, list[float]] = {option["key"]: [] for option in LEGACY_COST_OPTIONS}
        long_leg_returns_by_key: dict[str, list[float]] = {option["key"]: [] for option in LEGACY_COST_OPTIONS}
        short_leg_returns_by_key: dict[str, list[float]] = {option["key"]: [] for option in LEGACY_COST_OPTIONS}
        top_periods_by_key: dict[str, list[dict[str, Any]]] = {
            option["key"]: [] for option in LEGACY_COST_OPTIONS
        }
        yearly_returns: dict[str, dict[str, list[float]]] = {
            option["key"]: defaultdict(list) for option in LEGACY_COST_OPTIONS
        }
        regime_returns: dict[str, dict[str, list[float]]] = {
            option["key"]: defaultdict(list) for option in LEGACY_COST_OPTIONS
        }
        growth_by_key: dict[str, float] = {option["key"]: 1.0 for option in LEGACY_COST_OPTIONS}
        points: list[dict[str, Any]] = []

        for pnl in pnl_rows:
            gross_ret_bps = to_float(pnl.get("gross_daily_ret_bps"))
            turnover_total = to_float(pnl.get("turnover_total"))
            turnover_long = to_float(pnl.get("turnover_long"))
            turnover_short = to_float(pnl.get("turnover_short"))
            gross_long_bps = to_float(pnl.get("long_pnl_bps")) + turnover_long * configured_cost_bps
            gross_short_bps = to_float(pnl.get("short_pnl_bps")) + turnover_short * configured_cost_bps
            year = pnl["date"][:4]
            regime = pnl.get("regime") or "unknown"
            values: dict[str, Any] = {}
            for option in LEGACY_COST_OPTIONS:
                key = option["key"]
                cost_bps = float(option["costBps"])
                ret = (gross_ret_bps - turnover_total * cost_bps) / 10000.0
                long_ret = (gross_long_bps - turnover_long * cost_bps) / 10000.0
                short_ret = (gross_short_bps - turnover_short * cost_bps) / 10000.0
                returns_by_key[key].append(ret)
                long_leg_returns_by_key[key].append(long_ret)
                short_leg_returns_by_key[key].append(short_ret)
                yearly_returns[key][year].append(ret)
                regime_returns[key][regime].append(ret)
                top_periods_by_key[key].append(
                    {
                        "date": pnl["date"],
                        "returnPct": ret * 100.0,
                        "longPct": long_ret * 100.0,
                        "shortPct": short_ret * 100.0,
                        "ic": None,
                        "turnover": turnover_total,
                        "regime": regime,
                        "basis": "daily",
                    }
                )
                growth_by_key[key] *= 1.0 + ret
                values[key] = {
                    "ret": ret,
                    "growth": growth_by_key[key],
                    "returnPct": pct_decimal_to_pct(growth_by_key[key] - 1.0),
                }
            points.append({"date": pnl["date"], "values": values})

        label = LEGACY_LABELS.get(score_type, score_type.replace("_", " ").title())
        metrics: dict[str, Any] = {}
        yearly: dict[str, list[dict[str, Any]]] = {}
        components: dict[str, dict[str, float | None]] = {}
        for option in LEGACY_COST_OPTIONS:
            key = option["key"]
            returns = returns_by_key[key]
            metrics[key] = {
                "sharpe": sharpe_ratio(returns, TRADING_DAYS),
                "annualReturnPct": annualized_mean_pct(returns, TRADING_DAYS),
                "totalReturnPct": compound_return(returns),
                "cagrPct": safe_cagr_pct(returns, TRADING_DAYS),
                "icMean": to_float(row.get("ic_mean")),
                "annualVolPct": annual_vol_pct(returns),
                "maxDrawdownPct": max_drawdown_from_growth(points, key),
                "hitRatePct": hit_rate_pct(returns),
                "nPeriods": len(returns),
            }
            yearly[key] = [
                {"year": year, "returnPct": compound_return(year_returns)}
                for year, year_returns in sorted(yearly_returns[key].items())
            ]
            components[key] = {
                "longAnnPct": annualized_mean_pct(long_leg_returns_by_key[key], TRADING_DAYS),
                "shortAnnPct": annualized_mean_pct(short_leg_returns_by_key[key], TRADING_DAYS),
            }

        top_n = config.get("top_n", "n/a")
        bottom_n = config.get("bottom_n", "n/a")
        cost_bps = config.get("cost_bps", "n/a")
        weighting = config.get("weighting_mode", "n/a")
        setup = f"{top_n}/{bottom_n} per side, {weighting}; legacy costs reconstructed from gross P&L and turnover"
        setup_details = [
            {"label": "Harness", "value": "legacy factor backtest"},
            {"label": "Scorer", "value": LEGACY_SIGNAL_DESCRIPTIONS.get(score_type, label)},
            {
                "label": "Pool -> book",
                "value": f"{config.get('candidate_pool', {}).get('top_n_magnitude_per_side', 'n/a')}/{config.get('candidate_pool', {}).get('top_n_magnitude_per_side', 'n/a')} candidates -> {top_n}/{bottom_n} held",
            },
            {"label": "Cadence", "value": "daily MTM; rebalance every 5 trading days"},
            {"label": "Costs", "value": "gross P&L plus turnover reconstruction"},
        ]

        series.append(
            {
                "id": f"legacy:{score_type}",
                "label": label,
                "family": "Prior Rank/Confidence Composite",
                "kind": "legacy",
                "setup": setup,
                "setupDetails": setup_details,
                "periodBasis": "Daily marked-to-market rows from factor_pnl.csv; rebalanced every 5 trading days.",
                "points": points,
                "metrics": metrics,
                "yearly": yearly,
                "components": components,
                "diagnostics": {
                    "drawdowns": {
                        option["key"]: drawdown_points(points, str(option["key"]))
                        for option in LEGACY_COST_OPTIONS
                    },
                    "regimes": {
                        option["key"]: [
                            {
                                "regime": regime,
                                "n": len(values),
                                "annualReturnPct": annualized_mean_pct(values, TRADING_DAYS),
                                "sharpe": sharpe_ratio(values, TRADING_DAYS),
                                "hitRatePct": hit_rate_pct(values),
                            }
                            for regime, values in sorted(regime_returns[str(option["key"])].items())
                        ]
                        for option in LEGACY_COST_OPTIONS
                    },
                    "topPeriods": top_period_rows(top_periods_by_key),
                    "concentration": concentration_from_selections(selection_rows),
                },
            }
        )

    return series


def normalize_v1_scorecard(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    numeric_columns = {
        "cost_bps",
        "annual_return_pct",
        "annual_vol_pct",
        "sharpe",
        "total_return_pct",
        "cagr_pct",
        "max_drawdown_pct",
        "hit_rate_pct",
        "ic_mean",
        "ic_std",
        "ic_tstat",
        "decile_spread_pct",
        "avg_turnover_total",
        "n_periods",
    }
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        for column in numeric_columns:
            if column in item:
                item[column] = to_float(item[column])
        normalized.append(item)
    return normalized


def v1_scorecard_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, float], dict[str, Any]]:
    lookup: dict[tuple[str, str, float], dict[str, Any]] = {}
    for row in rows:
        lookup[(row["run"], row["execution_model"], float(row["cost_bps"]))] = row
    return lookup


def has_v1_outputs(root: Path) -> bool:
    return (
        (root / "comparison_scorecard.csv").exists()
        and (root / "baseline_pool" / "weekly_returns.csv").exists()
        and (root / "llm_pool_v1" / "weekly_returns.csv").exists()
    )


def discover_v1_variants(v1_root: Path) -> list[dict[str, Any]]:
    variants = [
        {
            "root": v1_root,
            "variant_id": "pos75",
            "book": "75/75",
            "positions_per_side": 75,
            "pool_per_side": 150,
            "include_universe": True,
            "family": "LLM Reversal v1",
            "label_prefix": "V1",
            "protocol_label": "Standard v1 score prompt",
        }
    ]
    for replay in V1_NARROW_REPLAYS:
        candidate = v1_root.parent / f"{v1_root.name}{replay['suffix']}"
        if has_v1_outputs(candidate):
            variants.append(
                {
                    "root": candidate,
                    "variant_id": replay["variant_id"],
                    "book": replay["book"],
                    "positions_per_side": replay["positions_per_side"],
                    "pool_per_side": 150,
                    "include_universe": False,
                    "family": "LLM Reversal v1",
                    "label_prefix": "V1",
                    "protocol_label": "Standard v1 score prompt",
                }
            )
    for extra in V1_EXTRA_VARIANTS:
        if has_v1_outputs(extra["root"]):
            variants.append(dict(extra))
    return variants


def build_v1_series(
    root: Path,
    variant_id: str,
    book: str,
    positions_per_side: int,
    include_universe: bool,
    pool_per_side: int = 150,
    family: str = "LLM Reversal v1",
    label_prefix: str = "V1",
    protocol_label: str = "Standard v1 score prompt",
) -> list[dict[str, Any]]:
    scorecard_rows = normalize_v1_scorecard(read_csv(root / "comparison_scorecard.csv"))
    scorecard = v1_scorecard_lookup(scorecard_rows)
    series: list[dict[str, Any]] = []

    for run in V1_RUNS:
        if not include_universe and not run["include_in_narrow_replays"]:
            continue
        run_dir = str(run["dir"])
        scorecard_run = str(run["scorecard_run"])
        base_label = str(run["label"]).removeprefix("V1 ").format(book=book)
        label = f"{label_prefix} {base_label}" if label_prefix != "V1" else str(run["label"]).format(book=book)
        weekly_rows = read_csv(root / run_dir / "weekly_returns.csv")
        signal_rows = read_csv(root / run_dir / "signal_checks.csv") if (root / run_dir / "signal_checks.csv").exists() else []
        selection_rows = (
            read_csv(root / run_dir / "portfolio_selections.csv")
            if (root / run_dir / "portfolio_selections.csv").exists()
            else []
        )
        growth_by_key = {option["key"]: 1.0 for option in V1_RETURN_OPTIONS}
        leg_returns_by_key = {
            str(option["key"]): {"long": [], "short": []}
            for option in V1_RETURN_OPTIONS
        }
        points: list[dict[str, Any]] = []

        for row in weekly_rows:
            values: dict[str, Any] = {}
            for option in V1_RETURN_OPTIONS:
                key = str(option["key"])
                ret = to_float(row.get(key))
                if "_c2c_" in key:
                    leg_returns_by_key[key]["long"].append(to_float(row.get("long_ret_c2c")))
                    leg_returns_by_key[key]["short"].append(-to_float(row.get("short_ret_c2c")))
                else:
                    leg_returns_by_key[key]["long"].append(to_float(row.get("long_ret_o2o")))
                    leg_returns_by_key[key]["short"].append(-to_float(row.get("short_ret_o2o")))
                growth_by_key[key] *= 1.0 + ret
                values[key] = {
                    "ret": ret,
                    "growth": growth_by_key[key],
                    "returnPct": pct_decimal_to_pct(growth_by_key[key] - 1.0),
                }
            points.append({"date": row["date"], "values": values})

        yearly_rows = read_csv(root / run_dir / "yearly_returns.csv")
        yearly: dict[str, list[dict[str, Any]]] = {str(option["key"]): [] for option in V1_RETURN_OPTIONS}
        for row in yearly_rows:
            for option in V1_RETURN_OPTIONS:
                yearly[str(option["key"])].append(
                    {
                        "year": row["year"],
                        "returnPct": pct_decimal_to_pct(to_float(row.get(str(option["yearly"])))),
                    }
                )

        metrics: dict[str, Any] = {}
        for option in V1_RETURN_OPTIONS:
            key = str(option["key"])
            execution_model, cost_bps = option["scorecard"]
            row = scorecard[(scorecard_run, str(execution_model), float(cost_bps))]
            metrics[key] = {
                "sharpe": row["sharpe"],
                "annualReturnPct": row["annual_return_pct"],
                "totalReturnPct": row["total_return_pct"],
                "cagrPct": row["cagr_pct"],
                "icMean": row["ic_mean"],
                "annualVolPct": row["annual_vol_pct"],
                "maxDrawdownPct": row["max_drawdown_pct"],
                "hitRatePct": row["hit_rate_pct"],
                "nPeriods": int(row["n_periods"]),
                "icTstat": row["ic_tstat"],
                "avgTurnoverTotal": row["avg_turnover_total"],
            }

        if run_dir == "baseline_universe":
            scorer = "deterministic decile spread, no LLM pool"
            pool_text = "full valid R1K-filtered universe"
            book_text = "D1 losers long / D10 winners short"
        elif run_dir == "baseline_pool":
            scorer = "deterministic Trail5 rank inside the candidate pool"
            pool_text = f"{pool_per_side} losers + {pool_per_side} winners"
            book_text = f"{book} selected by baseline rank"
        else:
            scorer = "LLM integer score inside the candidate pool"
            pool_text = f"{pool_per_side} losers + {pool_per_side} winners"
            book_text = f"{book} selected by LLM score"

        setup = (
            f"{protocol_label}; pool {pool_per_side}/{pool_per_side}, {book} held book"
            if run_dir != "baseline_universe"
            else f"{protocol_label}; full valid universe decile legs"
        )
        setup_details = [
            {"label": "Protocol", "value": protocol_label},
            {"label": "Scorer", "value": scorer},
            {"label": "Pool", "value": pool_text},
            {"label": "Book", "value": book_text},
            {"label": "Cadence", "value": "weekly Friday; 5-trading-day hold"},
        ]
        series.append(
            {
                "id": f"v1:{variant_id}:{run_dir}",
                "label": label,
                "family": family,
                "kind": "v1",
                "setup": setup,
                "setupDetails": setup_details,
                "periodBasis": "Weekly 5-trading-day holding-window rows from weekly_returns.csv.",
                "points": points,
                "metrics": metrics,
                "yearly": yearly,
                "components": {
                    str(option["key"]): {
                        "longAnnPct": annualized_mean_pct(leg_returns_by_key[str(option["key"])]["long"], WEEKLY_PERIODS),
                        "shortAnnPct": annualized_mean_pct(leg_returns_by_key[str(option["key"])]["short"], WEEKLY_PERIODS),
                    }
                    for option in V1_RETURN_OPTIONS
                },
                "diagnostics": {
                    "drawdowns": {
                        str(option["key"]): drawdown_points(points, str(option["key"]))
                        for option in V1_RETURN_OPTIONS
                    },
                    "regimes": regime_breakdown(
                        weekly_rows,
                        [str(option["key"]) for option in V1_RETURN_OPTIONS],
                        WEEKLY_PERIODS,
                    ),
                    "rollingIc": {
                        "c2c": rolling_mean_points(weekly_rows, "ic_c2c", 13),
                        "o2o": rolling_mean_points(weekly_rows, "ic_o2o", 13),
                    },
                    "deciles": decile_profile(weekly_rows, WEEKLY_PERIODS),
                    "topWeeks": top_week_rows(weekly_rows, [str(option["key"]) for option in V1_RETURN_OPTIONS]),
                    "concentration": concentration_from_selections(selection_rows),
                    "copyLike": copy_like_summary(root, run_dir),
                    "scoreSpread": score_spread_summary(signal_rows),
                },
            }
        )

    return series


def build_payload(legacy_root: Path, v1_root: Path) -> dict[str, Any]:
    legacy_config = read_json(legacy_root / "run_config.json")
    success = read_json(v1_root / "success_evaluation.json")
    v1_variants = discover_v1_variants(v1_root)
    v1_series: list[dict[str, Any]] = []
    for variant in v1_variants:
        v1_series.extend(build_v1_series(**variant))
    return {
        "legacyDefaultKey": LEGACY_DEFAULT_KEY,
        "legacyCostOptions": LEGACY_COST_OPTIONS,
        "v1Options": V1_RETURN_OPTIONS,
        "series": build_legacy_series(legacy_root) + v1_series,
        "success": success,
        "legacyMeta": {
            "root": str(legacy_root),
            "configuredCostBps": legacy_config.get("cost_bps"),
            "candidatePerSide": legacy_config.get("candidate_pool", {}).get("top_n_magnitude_per_side"),
            "topN": legacy_config.get("top_n"),
            "bottomN": legacy_config.get("bottom_n"),
            "weightingMode": legacy_config.get("weighting_mode"),
        },
        "v1Meta": {
            "root": str(v1_root),
            "variants": [
                {
                    "root": str(variant["root"]),
                    "book": variant["book"],
                    "positionsPerSide": variant["positions_per_side"],
                    "poolPerSide": variant.get("pool_per_side"),
                    "family": variant.get("family"),
                    "protocol": variant.get("protocol_label"),
                }
                for variant in v1_variants
            ],
        },
    }


def html_page(payload: dict[str, Any], output: Path) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_label = escape(str(output))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Reversal 2009-2014 Combined Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17212f;
      --muted: #627084;
      --line: #d9e0ea;
      --soft: #f6f8fb;
      --good: #13715f;
      --warn: #a55f13;
      --bad: #b23a48;
      --bg: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #fbfcfe;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 28px 22px 44px;
    }}
    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 7px;
      font-size: clamp(28px, 4vw, 43px);
      line-height: 1.05;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 9px;
      flex-wrap: wrap;
    }}
    label {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }}
    select {{
      min-width: 180px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--ink);
      padding: 8px 10px;
      font: inherit;
      font-size: 14px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(5, minmax(155px, 1fr));
      gap: 10px;
      margin: 18px 0 14px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      padding: 12px 13px;
      min-height: 86px;
    }}
    .card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      margin-bottom: 7px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .card strong {{
      display: block;
      font-size: 22px;
      line-height: 1.12;
      letter-spacing: 0;
    }}
    .card small {{
      display: block;
      color: var(--muted);
      margin-top: 6px;
      font-size: 12px;
    }}
    .good strong {{ color: var(--good); }}
    .warn strong {{ color: var(--warn); }}
    .method-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 0 0 14px;
    }}
    .method-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 13px 14px;
    }}
    .method-card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .04em;
      text-transform: uppercase;
      margin-bottom: 7px;
    }}
    .method-card strong {{
      display: block;
      font-size: 17px;
      margin-bottom: 5px;
    }}
    .method-card p {{
      font-size: 13px;
    }}
    .analysis-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin: 0 0 14px;
    }}
    .analysis-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 15px;
    }}
    .analysis-card h2 {{
      margin-bottom: 6px;
    }}
    .analysis-card p {{
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .analysis-list {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.38;
    }}
    .analysis-list li {{
      margin: 5px 0;
    }}
    .trace-control {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin: 10px 0 14px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
    }}
    .trace-control label {{
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .04em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .trace-control select {{
      min-width: min(520px, 100%);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      color: var(--ink);
      background: #fff;
      font-weight: 750;
    }}
    .trace-grid {{
      display: grid;
      grid-template-columns: 1.1fr 1.1fr .9fr;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .trace-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }}
    .trace-card h3 {{
      font-size: 16px;
      margin-bottom: 8px;
    }}
    .trace-row {{
      display: grid;
      grid-template-columns: 112px 1fr;
      gap: 10px;
      padding: 8px 0;
      border-top: 1px solid #eef2f7;
      font-size: 13px;
      line-height: 1.35;
    }}
    .trace-row:first-of-type {{
      border-top: 0;
    }}
    .trace-row span {{
      color: var(--muted);
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .03em;
      font-size: 11px;
    }}
    .verdict-box {{
      border-radius: 8px;
      padding: 12px;
      font-size: 14px;
      line-height: 1.4;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .verdict-box.good {{
      border-color: #9bcdbf;
      background: #f0faf6;
    }}
    .verdict-box.warn {{
      border-color: #e5c785;
      background: #fff8e7;
    }}
    .verdict-box.bad {{
      border-color: #df9aa4;
      background: #fff1f3;
    }}
    .guide-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin: 0 0 14px;
    }}
    .guide-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 16px;
    }}
    .guide-card h2 {{
      margin-bottom: 8px;
    }}
    .guide-card p {{
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .guide-list {{
      display: grid;
      gap: 7px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .guide-list li {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      padding-left: 14px;
      position: relative;
    }}
    .guide-list li::before {{
      content: "";
      position: absolute;
      left: 0;
      top: .58em;
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: var(--good);
    }}
    .run-pill {{
      display: inline-block;
      margin-bottom: 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      color: var(--muted);
      background: var(--soft);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .02em;
      text-transform: uppercase;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.65fr) minmax(360px, .95fr);
      gap: 14px;
      align-items: start;
      margin-top: 14px;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
      margin-top: 14px;
      align-items: start;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 15px;
    }}
    .panel-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 8px;
    }}
    svg {{
      width: 100%;
      display: block;
      overflow: visible;
    }}
    #growthChart {{ height: 475px; }}
    #qualityChart, #legChart, #decileChart, #costChart {{ height: 340px; }}
    #rollingIcChart {{ height: 520px; }}
    #copyChart {{ height: 310px; }}
    .axis {{ fill: var(--muted); font-size: 11px; }}
    .gridline {{ stroke: #e9eef4; stroke-width: 1; }}
    .axis-line {{ stroke: #a8b4c3; stroke-width: 1; }}
    .path {{ fill: none; stroke-width: 2.35; opacity: .74; }}
    .path.v1 {{ stroke-width: 3.1; opacity: .95; }}
    .path.det {{ stroke-dasharray: 5 4; }}
    .point-label {{ fill: var(--ink); font-size: 10px; font-weight: 700; }}
    .legend {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(235px, 1fr));
      gap: 8px 14px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 7px;
      min-width: 0;
    }}
    .swatch {{
      width: 22px;
      height: 3px;
      border-radius: 2px;
      flex: 0 0 auto;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    #leaderboard table {{ min-width: 1420px; table-layout: fixed; }}
    #yearly table {{ min-width: 760px; }}
    #leaderboard th:nth-child(1), #leaderboard td:nth-child(1) {{ width: 220px; }}
    #leaderboard th:nth-child(2), #leaderboard td:nth-child(2) {{ width: 190px; text-align: left; }}
    #leaderboard th:nth-child(3), #leaderboard td:nth-child(3) {{ width: 74px; }}
    #leaderboard th:nth-child(4), #leaderboard td:nth-child(4) {{ width: 92px; }}
    #leaderboard th:nth-child(5), #leaderboard td:nth-child(5) {{ width: 82px; }}
    #leaderboard th:nth-child(6), #leaderboard td:nth-child(6) {{ width: 86px; }}
    #leaderboard th:nth-child(7), #leaderboard td:nth-child(7) {{ width: 86px; }}
    #leaderboard th:nth-child(8), #leaderboard td:nth-child(8) {{ width: 88px; }}
    #leaderboard th:nth-child(9), #leaderboard td:nth-child(9) {{ width: 86px; }}
    #leaderboard th:nth-child(10), #leaderboard td:nth-child(10) {{ width: 74px; }}
    #leaderboard th:nth-child(11), #leaderboard td:nth-child(11) {{ width: 440px; }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
      position: sticky;
      left: 0;
      background: inherit;
      z-index: 1;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
      background: var(--soft);
    }}
    tr:last-child td {{ border-bottom: 0; }}
    tbody tr:nth-child(even) td {{ background: #fafbfc; }}
    td.name {{
      min-width: 230px;
      font-weight: 650;
    }}
    .family {{
      color: var(--muted);
      font-size: 12px;
    }}
    .chip {{
      display: inline-block;
      max-width: 170px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      color: var(--muted);
      background: var(--soft);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      vertical-align: middle;
      white-space: nowrap;
    }}
    .setup-list {{
      display: grid;
      gap: 4px;
      text-align: left;
      white-space: normal;
      line-height: 1.25;
    }}
    .setup-line {{
      display: grid;
      grid-template-columns: 72px minmax(0, 1fr);
      gap: 7px;
      align-items: baseline;
    }}
    .setup-label {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .setup-value {{
      color: var(--ink);
      overflow-wrap: anywhere;
    }}
    .best {{
      color: var(--good);
      font-weight: 800;
    }}
    .note {{
      margin-top: 11px;
      color: var(--muted);
      font-size: 12px;
    }}
    .meta {{
      display: grid;
      gap: 9px;
      color: var(--muted);
      font-size: 13px;
    }}
    .meta-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
    }}
    .meta-row strong {{
      color: var(--ink);
      text-align: right;
      max-width: 62%;
      overflow-wrap: anywhere;
    }}
    .year-cell {{
      font-variant-numeric: tabular-nums;
      font-weight: 650;
    }}
    .mini-table {{
      width: 100%;
      min-width: 0;
      font-size: 12px;
    }}
    .mini-table th, .mini-table td {{
      padding: 7px 8px;
      white-space: normal;
    }}
    .mini-table td:first-child, .mini-table th:first-child {{
      position: static;
    }}
    .split-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .delta-pos {{ color: var(--good); font-weight: 800; }}
    .delta-neg {{ color: var(--bad); font-weight: 800; }}
    .regime-matrix {{
      display: grid;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .regime-row {{
      display: grid;
      grid-template-columns: 145px repeat(var(--regime-count), minmax(74px, 1fr));
      gap: 6px;
      align-items: stretch;
    }}
    .regime-cell {{
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 7px 8px;
      min-height: 42px;
      font-size: 12px;
      line-height: 1.15;
    }}
    .regime-cell strong {{
      display: block;
      color: var(--ink);
      font-size: 13px;
    }}
    .table-swatch {{
      display: inline-block;
      width: 13px;
      height: 13px;
      border-radius: 50%;
      margin-right: 8px;
      vertical-align: -1px;
      border: 2px solid #fff;
      box-shadow: 0 0 0 1px var(--line);
    }}
    .regime-head {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .04em;
      text-transform: uppercase;
      background: var(--soft);
    }}
    @media (max-width: 980px) {{
      header, .layout, .detail-grid, .guide-grid, .analysis-grid, .split-grid, .trace-grid {{ display: block; }}
      .controls {{ margin-top: 14px; }}
      .cards, .method-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .panel {{ margin-top: 14px; }}
      #growthChart {{ height: 390px; }}
    }}
    @media (max-width: 680px) {{
      .cards, .method-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>LLM Reversal 2009-2014 Combined Report</h1>
        <p>Compares the earlier rank/confidence LLM signal experiment against the revised v1 reversal system. Legacy lines use the matching selected cost level where available; v1 lines follow the selected execution and cost view.</p>
      </div>
      <div class="controls">
        <label for="metricSelect">V1 execution/cost view</label>
        <select id="metricSelect"></select>
      </div>
    </header>

    <section class="cards" id="summaryCards" aria-label="Best-of summary"></section>
    <section class="method-grid" id="methodCards" aria-label="Methodology notes"></section>
    <section class="analysis-grid" id="analysisCards" aria-label="Interpretive analysis notes"></section>

    <section class="guide-grid" aria-label="Experiment guide">
      <div class="guide-card">
        <span class="run-pill">Prior Composite</span>
        <h2>Old Rank/Confidence System</h2>
        <p>The earlier LLM experiment kept portfolio construction in the factor harness, but the LLM output was richer: ranks and confidence scores for D1 loser and D10 winner candidates.</p>
        <ul class="guide-list">
          <li>Variants: deterministic raw, deterministic rank, LLM raw confidence, LLM rank, and LLM rank plus confidence.</li>
          <li>The LLM/cache used a larger candidate pool, 60 names per side in this saved run, then the harness held the top/bottom 30/30 equal-weight book.</li>
          <li>The saved result is a daily marked-to-market factor P&L path, so it has many more observations.</li>
        </ul>
      </div>
      <div class="guide-card">
        <span class="run-pill">V1 System</span>
        <h2>New V1 Protocol</h2>
        <p>V1 is the revised system: fixed pool, strict prompt, and one score per name. The LLM is no longer asked for ranks, confidence, weights, or portfolio language.</p>
        <ul class="guide-list">
          <li>The standard v1 rows build a 300-name pool: 150 biggest 5-day losers and 150 biggest 5-day winners after R1K, price, and ADV filters.</li>
          <li>The LLM returns one integer score from 0 to 100 for every pool ticker.</li>
          <li>The benchmark runs are Baseline-Universe, Baseline-Pool, and LLM-Pool v1, all under the same v1 harness. Narrower or alternate prompt cached replays appear as separate rows when available.</li>
        </ul>
      </div>
      <div class="guide-card">
        <span class="run-pill">Reading It</span>
        <h2>Clean Comparisons</h2>
        <p>The clean apples-to-apples comparison is inside the v1 family. The legacy family is still useful, but it answers a slightly different question.</p>
        <ul class="guide-list">
          <li>Use v1 vs v1 to judge whether the new LLM score beats the deterministic v1 baselines.</li>
          <li>Use legacy vs v1 as broader signal context, not a perfect apples-to-apples backtest.</li>
          <li>The dropdown changes v1 execution/cost assumptions. Legacy rows use the matching 0, 2.5, or 5 bps cost reconstruction, but do not switch between C2C and O2O.</li>
        </ul>
      </div>
    </section>

    <section class="panel" style="margin-top:14px;">
      <div class="panel-head">
        <div>
          <h2>Run Explainability / Sanity Check</h2>
          <p>Pick one run to see exactly what changed, what stayed controlled, and which comparison should be used for attribution.</p>
        </div>
      </div>
      <div class="trace-control">
        <label for="traceRunSelect">Run To Explain</label>
        <select id="traceRunSelect"></select>
      </div>
      <div class="trace-grid">
        <div class="trace-card" id="traceSummary"></div>
        <div class="trace-card" id="controlSummary"></div>
        <div class="trace-card" id="verdictSummary"></div>
      </div>
      <div class="table-wrap">
        <table class="mini-table" id="controlDiffTable"></table>
      </div>
      <details style="margin-top:12px;">
        <summary style="cursor:pointer;font-weight:800;color:var(--ink);">Show deterministic 5-day MR baseline rows</summary>
        <div class="table-wrap" style="margin-top:10px;">
          <table class="mini-table" id="baselineTable"></table>
        </div>
      </details>
      <p class="note">Rule of thumb: only compare an LLM row to its matching deterministic control when universe, candidate pool, book size, rebalance, and P&L path are the same.</p>
    </section>

    <section class="layout">
      <div class="panel">
        <div class="panel-head">
          <div>
            <h2>Growth of $1</h2>
            <p id="growthSubtitle"></p>
          </div>
        </div>
        <svg id="growthChart" role="img" aria-label="Combined growth path comparison"></svg>
        <div id="legend" class="legend"></div>
      </div>

      <div class="panel">
        <h2>Run Context</h2>
        <div class="meta" id="contextRows"></div>
        <p class="note">This is a compare-and-contrast view, not a claim that the two harnesses are identical. The old composite run is useful for relative signal quality; v1 is the standardized protocol benchmark path.</p>
      </div>
    </section>

    <section class="detail-grid">
      <div class="panel">
        <div class="panel-head">
          <div>
            <h2>Sharpe vs IC Map</h2>
            <p>Higher and farther right is better. V1 rows use the selected execution/cost view.</p>
          </div>
        </div>
        <svg id="qualityChart" role="img" aria-label="Sharpe versus IC scatter plot"></svg>
        <div class="table-wrap" style="margin-top:10px;">
          <table class="mini-table" id="qualityTable"></table>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <div>
            <h2>Long vs Short Contribution</h2>
            <p>Annualized average gross P&L contribution. V1 short-basket raw returns are sign-flipped here to show short-book contribution.</p>
          </div>
        </div>
        <svg id="legChart" role="img" aria-label="Long versus short annualized contribution chart"></svg>
      </div>
    </section>

    <section class="detail-grid">
      <div class="panel">
        <div class="panel-head">
          <div>
            <h2>LLM Delta vs Baselines</h2>
            <p>Positive means the LLM row beats the matching deterministic baseline in the selected view.</p>
          </div>
        </div>
        <div class="table-wrap">
          <table class="mini-table" id="deltaTable"></table>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <div>
            <h2>Cost Fragility</h2>
            <p>Gross, 2.5 bps, and 5 bps Sharpe for the visible V1 runs under the selected execution family.</p>
          </div>
        </div>
        <svg id="costChart" role="img" aria-label="Cost sensitivity chart"></svg>
      </div>
    </section>

    <section class="detail-grid">
      <div class="panel">
        <div class="panel-head">
          <div>
            <h2>Rolling IC</h2>
            <p id="focusRollingNote">13-week rolling mean IC for every V1 run with saved weekly IC diagnostics. Legacy rows do not have comparable rolling IC paths in the old harness.</p>
          </div>
        </div>
        <svg id="rollingIcChart" role="img" aria-label="Rolling IC chart"></svg>
      </div>
      <div class="panel">
        <div class="panel-head">
          <div>
            <h2>IC Readout</h2>
            <p id="focusIcNote">Latest and average rolling IC values for every V1 run with saved weekly IC diagnostics.</p>
          </div>
        </div>
        <div class="table-wrap">
          <table class="mini-table" id="rollingIcTable"></table>
        </div>
      </div>
    </section>

    <section class="detail-grid">
      <div class="panel">
        <div class="panel-head">
          <div>
            <h2>Score Decile Monotonicity</h2>
            <p id="focusDecileNote">Decile spread for every V1 run with saved score-decile buckets. Legacy rows do not have comparable decile bucket diagnostics.</p>
          </div>
        </div>
        <svg id="decileChart" role="img" aria-label="Score decile monotonicity chart"></svg>
        <div class="table-wrap"><table class="mini-table" id="decileTable"></table></div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <div>
            <h2>Copy-Likeness</h2>
            <p>LLM score rank correlation to Trail5 inside the candidate pool. High absolute values mean less independent signal.</p>
          </div>
        </div>
        <svg id="copyChart" role="img" aria-label="LLM copy-likeness chart"></svg>
      </div>
    </section>

    <section class="detail-grid">
      <div class="panel">
        <div class="panel-head">
          <div>
            <h2>Regime Breakdown</h2>
            <p>Heatmap shows annualized return by regime. The table keeps the exact Sharpe, hit rate, and sample size.</p>
          </div>
        </div>
        <div class="regime-matrix" id="regimeMatrix"></div>
        <div class="table-wrap">
          <table class="mini-table" id="regimeTable"></table>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <div>
            <h2>Ticker Concentration</h2>
            <p>Most repeated names by side for every run with saved portfolio selections.</p>
          </div>
        </div>
        <div class="table-wrap"><table class="mini-table" id="concentrationTable"></table></div>
      </div>
    </section>

    <section class="panel" style="margin-top:14px;">
      <div class="panel-head">
        <div>
          <h2>Best and Worst Weeks</h2>
          <p>Largest positive and negative periods across all runs. V1 rows are weekly windows; legacy rows are daily MTM rows.</p>
        </div>
      </div>
      <div class="split-grid">
        <div class="table-wrap"><table class="mini-table" id="bestWeeksTable"></table></div>
        <div class="table-wrap"><table class="mini-table" id="worstWeeksTable"></table></div>
      </div>
    </section>

    <section class="panel" id="leaderboard" style="margin-top:14px;">
      <div class="panel-head">
        <div>
          <h2>Leaderboard</h2>
          <p id="tableSubtitle"></p>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Family</th>
              <th>Sharpe</th>
              <th>Total Ret</th>
              <th>CAGR</th>
              <th>Max DD</th>
              <th>Ann Vol</th>
              <th>Hit Rate</th>
              <th>IC Mean</th>
              <th>Obs</th>
              <th>Setup</th>
            </tr>
          </thead>
          <tbody id="leaderboardBody"></tbody>
        </table>
      </div>
    </section>

    <section class="panel" id="yearly" style="margin-top:14px;">
      <div class="panel-head">
        <div>
          <h2>Yearly Return Heatmap</h2>
          <p>Calendar-year long-short returns. Green means stronger positive year, red means negative year.</p>
        </div>
      </div>
      <div class="table-wrap">
        <table id="yearlyTable"></table>
      </div>
    </section>

    <p class="note">Generated {generated_at}. Output: <code>{output_label}</code>.</p>
  </main>

  <script>
    const DATA = {payload_json};
    const LEGACY_DEFAULT_KEY = DATA.legacyDefaultKey;
    const COLORS = [
      "#13715f", "#345995", "#b66b18", "#7c4d9d", "#b23a48",
      "#0c7894", "#8a731b", "#5f6f7c", "#924c2e"
    ];

    const metricSelect = document.getElementById("metricSelect");
    const summaryCards = document.getElementById("summaryCards");
    const methodCards = document.getElementById("methodCards");
    const analysisCards = document.getElementById("analysisCards");
    const traceRunSelect = document.getElementById("traceRunSelect");
    const traceSummary = document.getElementById("traceSummary");
    const controlSummary = document.getElementById("controlSummary");
    const verdictSummary = document.getElementById("verdictSummary");
    const controlDiffTable = document.getElementById("controlDiffTable");
    const baselineTable = document.getElementById("baselineTable");
    const growthSubtitle = document.getElementById("growthSubtitle");
    const tableSubtitle = document.getElementById("tableSubtitle");
    const leaderboardBody = document.getElementById("leaderboardBody");
    const yearlyTable = document.getElementById("yearlyTable");
    const contextRows = document.getElementById("contextRows");
    const focusRollingNote = document.getElementById("focusRollingNote");
    const focusIcNote = document.getElementById("focusIcNote");
    const focusDecileNote = document.getElementById("focusDecileNote");
    const qualityTable = document.getElementById("qualityTable");
    const deltaTable = document.getElementById("deltaTable");
    const rollingIcTable = document.getElementById("rollingIcTable");
    const decileTable = document.getElementById("decileTable");
    const regimeMatrix = document.getElementById("regimeMatrix");
    const regimeTable = document.getElementById("regimeTable");
    const concentrationTable = document.getElementById("concentrationTable");
    const bestWeeksTable = document.getElementById("bestWeeksTable");
    const worstWeeksTable = document.getElementById("worstWeeksTable");

    function selectedKey(series) {{
      if (series.kind !== "legacy") return metricSelect.value;
      if (metricSelect.value.endsWith("_0")) return "legacy_0";
      if (metricSelect.value.endsWith("_2p5")) return "legacy_2p5";
      if (metricSelect.value.endsWith("_5")) return "legacy_5";
      return LEGACY_DEFAULT_KEY;
    }}

    function metric(series) {{
      return series.metrics[selectedKey(series)];
    }}

    function yearly(series) {{
      return series.yearly[selectedKey(series)] || [];
    }}

    function component(series) {{
      return series.components[selectedKey(series)] || {{}};
    }}

    function diagnostics(series) {{
      return series.diagnostics || {{}};
    }}

    function executionFamily() {{
      return metricSelect.value.includes("_o2o_") ? "o2o" : "c2c";
    }}

    function costKey(costLabel) {{
      const family = executionFamily();
      return `ls_${{family}}_${{costLabel}}`;
    }}

    function renderFocusNotes() {{
      focusRollingNote.textContent = "All V1 rows with weekly rolling IC diagnostics. Legacy rows are excluded because factor_pnl.csv only has daily P&L and aggregate IC, not a saved date-by-date IC series.";
      focusIcNote.textContent = "All V1 rows with weekly rolling IC diagnostics. This is a V1-only diagnostic because the legacy harness did not save comparable rolling IC paths.";
      focusDecileNote.textContent = "All V1 rows with saved score-decile buckets. This is V1-only because the legacy harness did not save comparable score-decile bucket returns.";
    }}

    function matchingV1Baseline(series, kind) {{
      if (!series) return null;
      const parts = series.id.split(":");
      if (parts.length < 3) return null;
      const variant = parts[1];
      const runDir = kind === "pool" ? "baseline_pool" : "baseline_universe";
      return DATA.series.find((candidate) => candidate.id === `v1:${{variant}}:${{runDir}}`)
        || DATA.series.find((candidate) => candidate.id === `v1:pos75:${{runDir}}`);
    }}

    function pct(value, digits = 1) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
      const sign = value > 0 ? "+" : "";
      return `${{sign}}${{value.toFixed(digits)}}%`;
    }}

    function num(value, digits = 2) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
      return value.toFixed(digits);
    }}

    function renderSetup(series) {{
      const details = series.setupDetails || [];
      if (!details.length) return series.setup;
      return `
        <div class="setup-list">
          ${{details.map((item) => `
            <div class="setup-line">
              <span class="setup-label">${{item.label}}</span>
              <span class="setup-value">${{item.value}}</span>
            </div>
          `).join("")}}
        </div>
      `;
    }}

    function optionLabel() {{
      const option = DATA.v1Options.find((item) => item.key === metricSelect.value);
      return option ? option.label : "selected v1 view";
    }}

    function sortedSeries() {{
      return [...DATA.series].sort((a, b) => metric(b).sharpe - metric(a).sharpe);
    }}

    function renderControls() {{
      DATA.v1Options.forEach((option) => {{
        const el = document.createElement("option");
        el.value = option.key;
        el.textContent = option.label;
        metricSelect.appendChild(el);
      }});
      metricSelect.value = "ls_c2c_5";
      metricSelect.addEventListener("change", renderAll);
      renderTraceControls();
    }}

    function bestBy(field) {{
      return sortedSeries().reduce((best, item) => {{
        if (!best || metric(item)[field] > metric(best)[field]) return item;
        return best;
      }}, null);
    }}

    function renderSummary() {{
      const bestSharpe = bestBy("sharpe");
      const bestReturn = bestBy("totalReturnPct");
      const bestIc = bestBy("icMean");
      const success = DATA.success || {{}};
      const cards = [
        {{
          label: "Best Sharpe",
          value: `${{bestSharpe.label}}`,
          detail: `${{num(metric(bestSharpe).sharpe, 2)}} Sharpe`,
          klass: "good"
        }},
        {{
          label: "Best Total Return",
          value: `${{bestReturn.label}}`,
          detail: `${{pct(metric(bestReturn).totalReturnPct)}} total`,
          klass: "good"
        }},
        {{
          label: "Best IC Mean",
          value: `${{bestIc.label}}`,
          detail: `${{num(metric(bestIc).icMean, 4)}} IC`,
          klass: ""
        }},
        {{
          label: "V1 75/75 Win Rule",
          value: success.is_win ? "Pass" : "No",
          detail: "LLM-Pool > Universe at C2C 5 bps",
          klass: success.is_win ? "good" : "warn"
        }},
        {{
          label: "V1 Copy-Like Periods",
          value: success.copy_like_correlation_periods ?? "n/a",
          detail: "score-vs-Trail5 high-corr periods",
          klass: success.copy_like_correlation_periods ? "warn" : ""
        }}
      ];

      summaryCards.innerHTML = cards.map((card) => `
        <div class="card ${{card.klass}}">
          <span>${{card.label}}</span>
          <strong>${{card.value}}</strong>
          <small>${{card.detail}}</small>
        </div>
      `).join("");
    }}

    function renderMethodCards() {{
      const legacy = DATA.series.find((series) => series.kind === "legacy");
      const v1 = DATA.series.find((series) => series.kind === "v1");
      const legacyPeriods = legacy ? metric(legacy).nPeriods : "n/a";
      const v1Periods = v1 ? metric(v1).nPeriods : "n/a";
      const cards = [
        {{
          label: "Period counts",
          title: `${{legacyPeriods}} daily vs ${{v1Periods}} weekly`,
          text: "Legacy writes daily marked-to-market P&L rows. V1 writes one 5-trading-day holding-window row per Friday-style rebalance."
        }},
        {{
          label: "Comparison hygiene",
          title: "V1 answers the new-system question",
          text: "The revised system changed the pool, prompt, output schema, book size, and reporting path. Use Baseline-Universe, Baseline-Pool, and LLM-Pool v1 for protocol conclusions."
        }},
        {{
          label: "Book shape",
          title: "Book size matters",
          text: "The prior composite harness held 30/30. The original v1 run held 75/75, and cached narrower v1 replays are shown separately when pulled locally."
        }}
      ];
      methodCards.innerHTML = cards.map((card) => `
        <div class="method-card">
          <span>${{card.label}}</span>
          <strong>${{card.title}}</strong>
          <p>${{card.text}}</p>
        </div>
      `).join("");
    }}

    function renderBaselineTable() {{
      const rows = DATA.series
        .filter((series) => (
          series.id.includes("mean_reversion_")
          || series.id.includes(":baseline_universe")
          || series.id.includes(":baseline_pool")
        ))
        .sort((a, b) => {{
          const kindRank = a.kind === b.kind ? 0 : (a.kind === "legacy" ? -1 : 1);
          return kindRank || metric(b).sharpe - metric(a).sharpe;
        }});
      const baselineType = (series) => {{
        if (series.id.includes("mean_reversion_raw_score")) return "Legacy deterministic raw MR";
        if (series.id.includes(":baseline_universe")) return "V1 deterministic full-universe deciles";
        if (series.id.includes(":baseline_pool")) return "V1 deterministic Trail5 inside pool";
        return "Deterministic baseline";
      }};
      const head = `<thead><tr><th>Baseline</th><th>What It Tests</th><th>Harness</th><th>Sharpe</th><th>CAGR</th><th>IC</th><th>Max DD</th><th>N</th></tr></thead>`;
      const body = rows.map((series) => {{
        const m = metric(series);
        return `
          <tr>
            <td class="name">${{compactLabel(series.label)}}</td>
            <td>${{baselineType(series)}}</td>
            <td>${{series.kind === "legacy" ? "legacy daily factor path" : "V1 weekly path"}}</td>
            <td>${{num(m.sharpe, 2)}}</td>
            <td>${{pct(m.cagrPct)}}</td>
            <td>${{num(m.icMean, 4)}}</td>
            <td>${{pct(m.maxDrawdownPct)}}</td>
            <td>${{m.nPeriods}}</td>
          </tr>
        `;
      }}).join("");
      baselineTable.innerHTML = `${{head}}<tbody>${{body}}</tbody>`;
    }}

    function traceRowsHtml(rows) {{
      return rows.map(([label, value]) => `
        <div class="trace-row"><span>${{label}}</span><strong>${{value}}</strong></div>
      `).join("");
    }}

    function traceDetails(series) {{
      const details = Object.fromEntries((series.setupDetails || []).map((row) => [row.label, row.value]));
      const id = series.id;
      const isLegacy = series.kind === "legacy";
      const isLegacyLlm = isLegacy && id.includes("llm_");
      const isLegacyDet = isLegacy && id.includes("mean_reversion_");
      const isV1 = series.kind === "v1";
      const isV1Llm = isV1 && id.includes(":llm_pool_v1");
      const isV1Pool = isV1 && id.includes(":baseline_pool");
      const isV1Universe = isV1 && id.includes(":baseline_universe");
      const variant = isV1 ? id.split(":")[1] : "";
      const poolText = details.Pool || (variant.includes("pool60") ? "60 losers + 60 winners" : "150 losers + 150 winners");
      const bookText = details.Book || (isLegacy ? `${{DATA.legacyMeta.topN}}/${{DATA.legacyMeta.bottomN}} equal-weight book` : "matched V1 book");

      if (isLegacyDet) {{
        return {{
          title: compactLabel(series.label),
          universe: "Legacy Russell membership-gated composite-score universe.",
          candidates: `${{DATA.legacyMeta.candidatePerSide}} oversold + ${{DATA.legacyMeta.candidatePerSide}} overbought from the 5-day mean-reversion signal.`,
          inputs: "No LLM. Deterministic raw MR score or deterministic MR rank.",
          output: "Score column consumed directly by the legacy factor harness.",
          book: `${{DATA.legacyMeta.topN}}/${{DATA.legacyMeta.bottomN}} equal-weight book.`,
          path: "Rebalance every 5 trading days; daily marked-to-market P&L; costs reconstructed from gross P&L plus turnover.",
          use: "Old-harness deterministic 5-day MR baseline."
        }};
      }}

      if (isLegacyLlm) {{
        return {{
          title: compactLabel(series.label),
          universe: "Same legacy Russell membership-gated composite-score universe.",
          candidates: `${{DATA.legacyMeta.candidatePerSide}} D1 losers + ${{DATA.legacyMeta.candidatePerSide}} D10 winners in the saved run.`,
          inputs: "Ticker, MRL, MRS, Trail5, MRSig, VixPctl, VolPctl, VIX, VolImb, M2/M1, ATR, RV10, VAT_MA, Rng55, 52WH, Rng%, AutoC, Mom3m, RSI5, Vel/V, D52WH, VATSlo, %MA100, ADX.",
          output: "LLM returns rank + confidence_score; writer derives raw-confidence, rank, and rank*confidence score columns.",
          book: `${{DATA.legacyMeta.topN}}/${{DATA.legacyMeta.bottomN}} equal-weight book.`,
          path: "Rebalance every 5 trading days; daily marked-to-market P&L; costs reconstructed from gross P&L plus turnover.",
          use: "Legacy-internal LLM-vs-deterministic signal comparison, not clean legacy-vs-V1 attribution."
        }};
      }}

      if (isV1Universe) {{
        return {{
          title: compactLabel(series.label),
          universe: "V1 Russell 1000 filtered universe with price, liquidity, and data-validity screens.",
          candidates: "Full valid universe decile legs: D1 losers long and D10 winners short.",
          inputs: "No LLM. Deterministic 5-day reversal signal from Trail5/decile construction.",
          output: "Deterministic decile selection.",
          book: "Full decile basket, not an LLM-scored pool.",
          path: "Weekly Friday-style rebalance; fixed 5-trading-day holding windows; direct gross/2.5/5 bps overlays.",
          use: "V1 full-universe deterministic 5-day MR baseline."
        }};
      }}

      if (isV1Pool) {{
        return {{
          title: compactLabel(series.label),
          universe: "Same V1 filtered universe as its paired LLM run.",
          candidates: poolText,
          inputs: "No LLM. Deterministic Trail5 rank inside the candidate pool.",
          output: "Deterministic pool ranking.",
          book: bookText,
          path: "Same weekly 5-trading-day return path as paired LLM row.",
          use: "Main control for whether a paired V1 LLM scorer adds value."
        }};
      }}

      if (isV1Llm) {{
        return {{
          title: compactLabel(series.label),
          universe: "Same V1 filtered universe as the paired Baseline-Pool.",
          candidates: poolText,
          inputs: "Ticker, MRL, MRS, Trail5, MRSig, RSI5, Rng55, Vel/V, %MA100, 52WH, Mom3m, AutoC, ADX, VolPctl.",
          output: series.family.includes("Rank + Confidence")
            ? "LLM returns side-relative rank + confidence_score, then normalizes to score."
            : "LLM returns one 0-100 score for every candidate ticker.",
          book: bookText,
          path: "Weekly Friday-style rebalance; fixed 5-trading-day holding windows; direct gross/2.5/5 bps overlays.",
          use: "Compare to matching V1 Baseline-Pool to isolate LLM scoring."
        }};
      }}

      return {{
        title: compactLabel(series.label),
        universe: series.setup || "n/a",
        candidates: "n/a",
        inputs: "n/a",
        output: "n/a",
        book: "n/a",
        path: series.periodBasis || "n/a",
        use: "n/a"
      }};
    }}

    function controlFor(series) {{
      if (!series) return null;
      const id = series.id;
      if (series.kind === "v1" && id.includes(":llm_pool_v1")) return matchingV1Baseline(series, "pool");
      if (series.kind === "v1" && id.includes(":baseline_pool")) return matchingV1Baseline(series, "universe");
      if (series.kind === "legacy" && id.includes("llm_raw_confidence_score")) return runById("mean_reversion_raw_score");
      if (series.kind === "legacy" && id.includes("llm_rank")) return runById("mean_reversion_raw_score");
      return null;
    }}

    function verdictFor(series, control) {{
      if (!series) return {{ klass: "warn", title: "No run selected", text: "Choose a run to see attribution guidance." }};
      const m = metric(series);
      if (series.kind === "v1" && series.id.includes(":llm_pool_v1") && control) {{
        const b = metric(control);
        const ds = m.sharpe - b.sharpe;
        return {{
          klass: ds >= 0 ? "good" : "warn",
          title: "Clean V1 LLM Attribution",
          text: `Compare this run to ${{compactLabel(control.label)}}. Universe, pool, book, rebalance, and P&L path are controlled; the main changed layer is LLM scoring. Current delta: ${{ds >= 0 ? "+" : ""}}${{ds.toFixed(2)}} Sharpe and ${{pct(m.cagrPct - b.cagrPct)}} CAGR.`
        }};
      }}
      if (series.kind === "v1" && series.id.includes(":baseline_pool")) {{
        return {{
          klass: "warn",
          title: "Pool/Book Control",
          text: control
            ? `Compare to ${{compactLabel(control.label)}} to understand pool-filter effects, but note the book construction is not identical. This is not an LLM-alpha comparison.`
            : "This is a deterministic pool baseline. Its best use is as the paired control for the matching LLM-Pool row."
        }};
      }}
      if (series.kind === "v1" && series.id.includes(":baseline_universe")) {{
        return {{
          klass: "good",
          title: "Pure V1 MR Baseline",
          text: "No LLM is involved. This is the clean full-universe 5-day reversal benchmark inside the V1 weekly harness."
        }};
      }}
      if (series.kind === "legacy" && series.id.includes("llm_")) {{
        return {{
          klass: "warn",
          title: "Legacy-Internal LLM Comparison",
          text: control
            ? `Compare to ${{compactLabel(control.label)}} for old-harness context. Do not use this as proof versus V1: legacy has richer LLM features, rank/confidence output, daily MTM accounting, and a different harness.`
            : "Legacy LLM row. Useful for historical context, but not directly comparable to V1 for isolated LLM alpha."
        }};
      }}
      if (series.kind === "legacy") {{
        return {{
          klass: "good",
          title: "Pure Legacy MR Baseline",
          text: "No LLM is involved. This is the old factor-harness deterministic 5-day mean-reversion baseline."
        }};
      }}
      return {{ klass: "warn", title: "Attribution unclear", text: "Use the closest matching deterministic control before making an alpha claim." }};
    }}

    function renderTraceControls() {{
      const previous = traceRunSelect.value;
      const rows = [...DATA.series].sort((a, b) => {{
        const groupA = a.kind === "legacy" ? 0 : 1;
        const groupB = b.kind === "legacy" ? 0 : 1;
        return groupA - groupB || compactLabel(a.label).localeCompare(compactLabel(b.label));
      }});
      traceRunSelect.innerHTML = rows.map((series) => `
        <option value="${{series.id}}">${{compactLabel(series.label)}} · ${{series.kind === "legacy" ? "Legacy" : "V1"}}</option>
      `).join("");
      const preferred = rows.find((series) => series.id.includes("rank_conf_pool60_pos30:llm_pool_v1"))
        || rows.find((series) => series.id.includes(":llm_pool_v1"))
        || rows[0];
      traceRunSelect.value = previous && DATA.series.some((series) => series.id === previous) ? previous : preferred.id;
      if (!traceRunSelect.dataset.bound) {{
        traceRunSelect.addEventListener("change", renderRunTrace);
        traceRunSelect.dataset.bound = "1";
      }}
    }}

    function renderRunTrace() {{
      const series = DATA.series.find((item) => item.id === traceRunSelect.value) || DATA.series[0];
      const control = controlFor(series);
      const trace = traceDetails(series);
      const controlTrace = control ? traceDetails(control) : null;
      const verdict = verdictFor(series, control);
      const m = metric(series);
      traceSummary.innerHTML = `
        <h3>${{trace.title}}</h3>
        ${{traceRowsHtml([
          ["Sharpe", `${{num(m.sharpe, 2)}} / CAGR ${{pct(m.cagrPct)}}`],
          ["Universe", trace.universe],
          ["Candidates", trace.candidates],
          ["Inputs", trace.inputs],
          ["Output", trace.output],
          ["Book", trace.book],
          ["Rebalance", trace.path],
        ])}}
      `;
      controlSummary.innerHTML = controlTrace ? `
        <h3>Best Control: ${{controlTrace.title}}</h3>
        ${{traceRowsHtml([
          ["Why", trace.use],
          ["Control", controlTrace.use],
          ["Universe", controlTrace.universe],
          ["Candidates", controlTrace.candidates],
          ["Inputs", controlTrace.inputs],
          ["Book", controlTrace.book],
          ["Rebalance", controlTrace.path],
        ])}}
      ` : `
        <h3>No Direct Control</h3>
        ${{traceRowsHtml([
          ["Why", trace.use],
          ["Read As", "Baseline or context row, not an isolated LLM-alpha test."],
          ["Next Step", "Compare an LLM row to its matching deterministic pool baseline."],
        ])}}
      `;
      verdictSummary.innerHTML = `
        <h3>Attribution Verdict</h3>
        <div class="verdict-box ${{verdict.klass}}">
          <strong>${{verdict.title}}</strong><br>
          ${{verdict.text}}
        </div>
      `;

      const diffRows = controlTrace ? [
        ["Universe", trace.universe, controlTrace.universe, trace.universe === controlTrace.universe ? "Controlled" : "Different"],
        ["Candidate Pool", trace.candidates, controlTrace.candidates, trace.candidates === controlTrace.candidates ? "Controlled" : "Different"],
        ["Signals / Scorer", trace.inputs, controlTrace.inputs, "Changed by design"],
        ["Output", trace.output, controlTrace.output, "Changed by design"],
        ["Book", trace.book, controlTrace.book, trace.book === controlTrace.book ? "Controlled" : "Different"],
        ["Rebalance / P&L", trace.path, controlTrace.path, trace.path === controlTrace.path ? "Controlled" : "Different"],
      ] : [];
      if (!diffRows.length) {{
        controlDiffTable.innerHTML = "";
        return;
      }}
      const head = `<thead><tr><th>Dimension</th><th>This Run</th><th>Control</th><th>Status</th></tr></thead>`;
      const body = diffRows.map(([dim, left, right, status]) => `
        <tr>
          <td class="name">${{dim}}</td>
          <td>${{left}}</td>
          <td>${{right}}</td>
          <td>${{status}}</td>
        </tr>
      `).join("");
      controlDiffTable.innerHTML = `${{head}}<tbody>${{body}}</tbody>`;
    }}

    function runById(fragment) {{
      return DATA.series.find((series) => series.id.includes(fragment));
    }}

    function deltaText(llm, baseline) {{
      if (!llm || !baseline) return "n/a";
      const dm = metric(llm).sharpe - metric(baseline).sharpe;
      const dc = metric(llm).cagrPct - metric(baseline).cagrPct;
      return `${{dm >= 0 ? "+" : ""}}${{dm.toFixed(2)}} Sharpe, ${{dc >= 0 ? "+" : ""}}${{dc.toFixed(1)}} pct CAGR`;
    }}

    function copyText(series) {{
      const copy = diagnostics(series)?.copyLike;
      if (!copy) return "No Trail5 copy-likeness file for this run.";
      return `avg |corr| ${{copy.avgAbsCorr.toFixed(2)}}; ${{copy.highAbsCorrCount}} / ${{copy.n}} side-periods above 0.90.`;
    }}

    function costSharpeDrop(series) {{
      if (!series || series.kind !== "v1") return null;
      const gross = series.metrics[costKey("0")]?.sharpe;
      const net = series.metrics[costKey("5")]?.sharpe;
      if (gross === undefined || net === undefined) return null;
      return gross - net;
    }}

    function promptDeltaBullet(series) {{
      const baseline = matchingV1Baseline(series, "pool");
      return `${{series.label}} vs pool: ${{deltaText(series, baseline)}}; copy-likeness ${{copyText(series)}}`;
    }}

    function renderAnalysisCards() {{
      const standardLlm75 = runById("pos75:llm_pool_v1");
      const standardPool75 = runById("pos75:baseline_pool");
      const standardLlm30 = runById("pos30:llm_pool_v1");
      const standardPool30 = runById("pos30:baseline_pool");
      const rawLlm = runById("raw_conf_pool60_pos30:llm_pool_v1");
      const rawPool = runById("raw_conf_pool60_pos30:baseline_pool");
      const rankLlm = runById("rank_conf_pool60_pos30:llm_pool_v1");
      const rankPool = runById("rank_conf_pool60_pos30:baseline_pool");
      const best = sortedSeries()[0];
      const v1LlmRows = DATA.series.filter((series) => series.kind === "v1" && series.id.includes(":llm_pool_v1"));
      const bestV1Llm = [...v1LlmRows].sort((a, b) => metric(b).sharpe - metric(a).sharpe)[0];
      const worstV1Llm = [...v1LlmRows].sort((a, b) => metric(a).sharpe - metric(b).sharpe)[0];
      const copyRows = [...v1LlmRows]
        .filter((series) => diagnostics(series).copyLike)
        .sort((a, b) => diagnostics(b).copyLike.avgAbsCorr - diagnostics(a).copyLike.avgAbsCorr);
      const costRows = [...v1LlmRows]
        .map((series) => ({{ series, drop: costSharpeDrop(series) }}))
        .filter((row) => row.drop !== null)
        .sort((a, b) => b.drop - a.drop);
      const cards = [
        {{
          title: "All-Run Read",
          body: `Across every row, the report separates broad historical context from the clean v1-vs-v1 comparison. The legacy rows still look strongest in this selected view, but the standardized v1 lesson is more important: LLM scoring only helps in some prompt/book setups.`,
          bullets: [
            `Current best row: ${{best.label}} with Sharpe ${{num(metric(best).sharpe, 2)}} and CAGR ${{pct(metric(best).cagrPct)}}.`,
            `Best v1 LLM row: ${{bestV1Llm?.label || "n/a"}} with Sharpe ${{bestV1Llm ? num(metric(bestV1Llm).sharpe, 2) : "n/a"}}.`,
            `Weakest v1 LLM row: ${{worstV1Llm?.label || "n/a"}} with Sharpe ${{worstV1Llm ? num(metric(worstV1Llm).sharpe, 2) : "n/a"}}.`
          ]
        }},
        {{
          title: "Prompt Comparisons",
          body: `The prompt format matters. The standard v1 score prompt is the only one that can look competitive against its deterministic pool baseline; raw-confidence and rank-confidence both degrade the pool comparison in the current selected view.`,
          bullets: [
            `Standard v1 75/75 LLM vs pool: ${{deltaText(standardLlm75, standardPool75)}}.`,
            `Standard v1 30/30 LLM vs pool: ${{deltaText(standardLlm30, standardPool30)}}.`,
            `Raw-confidence LLM vs pool: ${{deltaText(rawLlm, rawPool)}}.`,
            `Rank-confidence LLM vs pool: ${{deltaText(rankLlm, rankPool)}}.`
          ]
        }},
        {{
          title: "Copy-Likeness Insight",
          body: `Copy-likeness is a key trace. Rank-confidence mostly reconstructs Trail5; raw-confidence is less copy-like but still does not produce better alpha. So both extremes are informative: copying is not enough, and being different is not enough either.`,
          bullets: [
            ...copyRows.map((series) => `${{series.label}}: ${{copyText(series)}}`),
            `Interpretation: the useful signal must improve forward-return ordering, not merely resemble or avoid Trail5.`
          ]
        }},
        {{
          title: "Cost And Robustness",
          body: `The v1 rows are meaningfully cost-sensitive because turnover is high. A run can look acceptable gross and become weak at 5 bps, so the 5 bps view is the harsh but more useful read.`,
          bullets: [
            ...costRows.map((row) => `${{row.series.label}}: Sharpe drops ${{num(row.drop, 2)}} from gross to 5 bps in the selected execution family.`),
            `Interpretation: any LLM edge has to clear turnover drag, not just improve gross IC.`
          ]
        }},
        {{
          title: "Why Some Runs Win",
          body: `The stronger runs combine decent IC with less harmful book construction and survivable costs. The weaker LLM runs either copy the baseline without improving it, or produce independent-looking scores that do not rank future returns well enough.`,
          bullets: [
            `Legacy rows win the broad leaderboard, but they are from the older factor-backtest/composite-score harness: daily marked-to-market P&L, richer LLM rank/confidence composite inputs, 60/60 candidates -> 30/30 held, and costs reconstructed from gross P&L plus turnover.`,
            `V1 rows are the standardized reversal protocol: weekly Friday-style rebalance rows, fixed 5-trading-day hold, explicit Baseline-Universe/Baseline-Pool/LLM-Pool comparisons, and direct cost overlays on the same weekly return path.`,
            `Standard v1 LLM rows are the cleanest evidence that the LLM can sometimes help inside the v1 setup.`,
            `Rank-confidence is the clearest negative result: high Trail5 agreement plus worse net performance.`
          ]
        }},
        {{
          title: "What To Use LLMs For",
          body: `The evidence argues against using the LLM as a standalone numeric scorer for opaque technical rows. It may still be useful as a meta-layer around deterministic signals.`,
          bullets: [
            `Promising next uses: regime gating, basket review, anomaly filtering, or deciding when to trust/reduce deterministic reversal exposure.`,
            `Less promising use: asking for one 0-100 alpha score per ticker with no feedback loop.`,
            `The lower rolling-IC and decile panels intentionally focus on one LLM family at a time; this analysis block is the all-run interpretation.`
          ]
        }}
      ];
      analysisCards.innerHTML = cards.map((card) => `
        <div class="analysis-card">
          <h2>${{card.title}}</h2>
          <p>${{card.body}}</p>
          <ul class="analysis-list">
            ${{card.bullets.map((item) => `<li>${{item}}</li>`).join("")}}
          </ul>
        </div>
      `).join("");
    }}

    function svgEl(tag, attrs = {{}}, text = "") {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
      if (text) el.textContent = text;
      return el;
    }}

    function renderGrowthChart() {{
      const svg = document.getElementById("growthChart");
      svg.innerHTML = "";
      const width = 1040;
      const height = 475;
      const margin = {{ top: 18, right: 44, bottom: 42, left: 62 }};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);

      const all = [];
      DATA.series.forEach((series) => {{
        const key = selectedKey(series);
        series.points.forEach((point) => {{
          all.push({{ date: new Date(point.date + "T00:00:00"), value: point.values[key].growth }});
        }});
      }});

      const minX = Math.min(...all.map((p) => p.date.getTime()));
      const maxX = Math.max(...all.map((p) => p.date.getTime()));
      const maxY = Math.max(...all.map((p) => p.value), 1);
      const minY = Math.min(...all.map((p) => p.value), 1);
      const yPad = Math.max((maxY - minY) * 0.08, 0.04);
      const y0 = Math.max(0, minY - yPad);
      const y1 = maxY + yPad;
      const x = (date) => margin.left + ((date.getTime() - minX) / (maxX - minX)) * innerW;
      const y = (value) => margin.top + (1 - ((value - y0) / (y1 - y0))) * innerH;

      for (let i = 0; i <= 5; i++) {{
        const yy = margin.top + (innerH / 5) * i;
        const value = y1 - ((y1 - y0) / 5) * i;
        svg.appendChild(svgEl("line", {{ x1: margin.left, x2: width - margin.right, y1: yy, y2: yy, class: "gridline" }}));
        svg.appendChild(svgEl("text", {{ x: margin.left - 10, y: yy + 4, "text-anchor": "end", class: "axis" }}, `${{value.toFixed(2)}}x`));
      }}

      const years = [...new Set(all.map((p) => p.date.getFullYear()))];
      years.forEach((year) => {{
        const xx = x(new Date(`${{year}}-01-01T00:00:00`));
        if (xx >= margin.left && xx <= width - margin.right) {{
          svg.appendChild(svgEl("text", {{ x: xx, y: height - 15, "text-anchor": "middle", class: "axis" }}, String(year)));
        }}
      }});

      DATA.series.forEach((series, index) => {{
        const key = selectedKey(series);
        const d = series.points.map((point, pointIndex) => {{
          const command = pointIndex === 0 ? "M" : "L";
          return `${{command}}${{x(new Date(point.date + "T00:00:00")).toFixed(1)}},${{y(point.values[key].growth).toFixed(1)}}`;
        }}).join(" ");
        const cls = [
          "path",
          series.kind === "v1" ? "v1" : "",
          series.label.includes("Det") ? "det" : ""
        ].join(" ");
        svg.appendChild(svgEl("path", {{ d, class: cls, stroke: COLORS[index % COLORS.length] }}));
      }});

      document.getElementById("legend").innerHTML = DATA.series.map((series, index) => {{
        const key = selectedKey(series);
        const last = series.points[series.points.length - 1].values[key];
        return `
          <div class="legend-item">
            <span class="swatch" style="background:${{COLORS[index % COLORS.length]}}"></span>
            <strong>${{series.label}}</strong>
            <span>${{last.growth.toFixed(2)}}x (${{pct(last.returnPct)}})</span>
          </div>
        `;
      }}).join("");
    }}

    function renderLeaderboard() {{
      const rows = sortedSeries();
      tableSubtitle.textContent = `Legacy rows use the matching selected cost level on their daily path; v1 rows use ${{optionLabel()}}.`;
      const maxSharpe = Math.max(...rows.map((series) => metric(series).sharpe));
      const maxReturn = Math.max(...rows.map((series) => metric(series).totalReturnPct));
      const maxIc = Math.max(...rows.map((series) => metric(series).icMean));

      leaderboardBody.innerHTML = rows.map((series) => {{
        const m = metric(series);
        return `
          <tr>
            <td class="name">${{series.label}}</td>
            <td><span class="chip" title="${{series.family}}">${{series.family}}</span></td>
            <td class="${{m.sharpe === maxSharpe ? "best" : ""}}">${{num(m.sharpe, 2)}}</td>
            <td class="${{m.totalReturnPct === maxReturn ? "best" : ""}}">${{pct(m.totalReturnPct)}}</td>
            <td>${{pct(m.cagrPct)}}</td>
            <td>${{pct(m.maxDrawdownPct)}}</td>
            <td>${{pct(m.annualVolPct)}}</td>
            <td>${{pct(m.hitRatePct)}}</td>
            <td class="${{m.icMean === maxIc ? "best" : ""}}">${{num(m.icMean, 4)}}</td>
            <td>${{m.nPeriods}}<br><span class="family">${{series.kind === "legacy" ? "daily" : "weekly"}}</span></td>
            <td>${{renderSetup(series)}}</td>
          </tr>
        `;
      }}).join("");
    }}

    function heatColor(value) {{
      const clamped = Math.max(-40, Math.min(40, value));
      if (clamped >= 0) {{
        const alpha = 0.08 + Math.abs(clamped) / 40 * 0.34;
        return `rgba(19, 113, 95, ${{alpha}})`;
      }}
      const alpha = 0.08 + Math.abs(clamped) / 40 * 0.34;
      return `rgba(178, 58, 72, ${{alpha}})`;
    }}

    function renderYearly() {{
      const years = [...new Set(DATA.series.flatMap((series) => yearly(series).map((row) => row.year)))].sort();
      const head = `<thead><tr><th>Run</th>${{years.map((year) => `<th>${{year}}</th>`).join("")}}</tr></thead>`;
      const body = sortedSeries().map((series) => {{
        const rows = yearly(series);
        const byYear = Object.fromEntries(rows.map((row) => [row.year, row.returnPct]));
        const cells = years.map((year) => {{
          const value = byYear[year];
          if (value === undefined) return "<td></td>";
          return `<td class="year-cell" style="background:${{heatColor(value)}}">${{pct(value)}}</td>`;
        }}).join("");
        return `<tr><td class="name">${{series.label}}</td>${{cells}}</tr>`;
      }}).join("");
      yearlyTable.innerHTML = `${{head}}<tbody>${{body}}</tbody>`;
    }}

    function compactLabel(label) {{
      return label
        .replace("Legacy LLM Raw Confidence", "Legacy raw conf")
        .replace("Legacy LLM Rank + Confidence", "Legacy rank+conf")
        .replace("Legacy LLM Rank", "Legacy rank")
        .replace("Legacy Det Raw", "Legacy det raw")
        .replace("V1 Baseline Universe", "V1 universe")
        .replace("V1 Raw Confidence Baseline Pool", "Raw conf pool")
        .replace("V1 Raw Confidence LLM Pool", "Raw conf LLM")
        .replace("V1 Baseline Pool", "V1 pool")
        .replace("V1 LLM Pool", "V1 LLM");
    }}

    function renderQualityChart() {{
      const svg = document.getElementById("qualityChart");
      svg.innerHTML = "";
      const width = 620;
      const height = 340;
      const margin = {{ top: 18, right: 28, bottom: 48, left: 58 }};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);

      const rows = DATA.series.map((series, index) => ({{
        series,
        index,
        x: metric(series).icMean,
        y: metric(series).sharpe,
        r: Math.max(5, Math.min(13, Math.sqrt(Math.max(metric(series).totalReturnPct, 0)) * 0.8))
      }}));
      const xMinRaw = Math.min(...rows.map((row) => row.x));
      const xMaxRaw = Math.max(...rows.map((row) => row.x));
      const yMinRaw = Math.min(...rows.map((row) => row.y));
      const yMaxRaw = Math.max(...rows.map((row) => row.y));
      const xPad = Math.max((xMaxRaw - xMinRaw) * 0.16, 0.002);
      const yPad = Math.max((yMaxRaw - yMinRaw) * 0.18, 0.08);
      const xMin = xMinRaw - xPad;
      const xMax = xMaxRaw + xPad;
      const yMin = Math.max(0, yMinRaw - yPad);
      const yMax = yMaxRaw + yPad;
      const xScale = (value) => margin.left + ((value - xMin) / (xMax - xMin)) * innerW;
      const yScale = (value) => margin.top + (1 - ((value - yMin) / (yMax - yMin))) * innerH;

      for (let i = 0; i <= 4; i++) {{
        const yy = margin.top + (innerH / 4) * i;
        const yValue = yMax - ((yMax - yMin) / 4) * i;
        svg.appendChild(svgEl("line", {{ x1: margin.left, x2: width - margin.right, y1: yy, y2: yy, class: "gridline" }}));
        svg.appendChild(svgEl("text", {{ x: margin.left - 8, y: yy + 4, "text-anchor": "end", class: "axis" }}, yValue.toFixed(2)));
      }}
      for (let i = 0; i <= 4; i++) {{
        const xx = margin.left + (innerW / 4) * i;
        const xValue = xMin + ((xMax - xMin) / 4) * i;
        svg.appendChild(svgEl("text", {{ x: xx, y: height - 18, "text-anchor": "middle", class: "axis" }}, xValue.toFixed(3)));
      }}
      svg.appendChild(svgEl("line", {{ x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom, class: "axis-line" }}));
      svg.appendChild(svgEl("line", {{ x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, class: "axis-line" }}));
      svg.appendChild(svgEl("text", {{ x: margin.left + innerW / 2, y: height - 2, "text-anchor": "middle", class: "axis" }}, "Mean IC"));
      svg.appendChild(svgEl("text", {{ x: 14, y: margin.top + innerH / 2, transform: `rotate(-90 14 ${{margin.top + innerH / 2}})`, "text-anchor": "middle", class: "axis" }}, "Sharpe"));

      rows.forEach((row) => {{
        const cx = xScale(row.x);
        const cy = yScale(row.y);
        svg.appendChild(svgEl("circle", {{
          cx,
          cy,
          r: row.r,
          fill: COLORS[row.index % COLORS.length],
          opacity: row.series.kind === "v1" ? "0.92" : "0.74",
          stroke: "#fff",
          "stroke-width": "2"
        }}));
      }});
      renderQualityTable(rows);
    }}

    function renderQualityTable(rows) {{
      const sorted = [...rows].sort((a, b) => b.y - a.y);
      const head = `<thead><tr><th>Run</th><th>Family</th><th>Sharpe</th><th>Mean IC</th><th>CAGR</th><th>Max DD</th></tr></thead>`;
      const body = sorted.map((row) => {{
        const m = metric(row.series);
        const color = COLORS[row.index % COLORS.length];
        return `
          <tr>
            <td class="name"><span class="table-swatch" style="background:${{color}}"></span>${{compactLabel(row.series.label)}}</td>
            <td>${{row.series.kind === "legacy" ? "Legacy" : "V1"}}</td>
            <td>${{num(m.sharpe, 2)}}</td>
            <td>${{num(m.icMean, 4)}}</td>
            <td>${{pct(m.cagrPct)}}</td>
            <td>${{pct(m.maxDrawdownPct)}}</td>
          </tr>
        `;
      }}).join("");
      qualityTable.innerHTML = `${{head}}<tbody>${{body}}</tbody>`;
    }}

    function renderLegChart() {{
      const svg = document.getElementById("legChart");
      svg.innerHTML = "";
      const rows = sortedSeries().map((series) => ({{
        series,
        longValue: component(series).longAnnPct || 0,
        shortValue: component(series).shortAnnPct || 0
      }}));
      const width = 640;
      const rowH = 31;
      const height = Math.max(340, 58 + rows.length * rowH);
      const margin = {{ top: 18, right: 34, bottom: 30, left: 150 }};
      const innerW = width - margin.left - margin.right;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);

      const maxAbs = Math.max(8, ...rows.flatMap((row) => [Math.abs(row.longValue), Math.abs(row.shortValue)]));
      const x = (value) => margin.left + ((value + maxAbs) / (maxAbs * 2)) * innerW;
      const zeroX = x(0);
      svg.appendChild(svgEl("line", {{ x1: zeroX, x2: zeroX, y1: margin.top, y2: height - margin.bottom, class: "axis-line" }}));

      [-1, -0.5, 0, 0.5, 1].forEach((fraction) => {{
        const value = maxAbs * fraction;
        const xx = x(value);
        svg.appendChild(svgEl("line", {{ x1: xx, x2: xx, y1: margin.top, y2: height - margin.bottom, class: "gridline" }}));
        svg.appendChild(svgEl("text", {{ x: xx, y: height - 9, "text-anchor": "middle", class: "axis" }}, `${{value.toFixed(0)}}%`));
      }});

      rows.forEach((row, index) => {{
        const yBase = margin.top + index * rowH + 8;
        svg.appendChild(svgEl("text", {{ x: margin.left - 10, y: yBase + 8, "text-anchor": "end", class: "axis" }}, compactLabel(row.series.label)));
        [
          {{ value: row.longValue, color: "#13715f", y: yBase, label: "L" }},
          {{ value: row.shortValue, color: "#345995", y: yBase + 12, label: "S" }}
        ].forEach((bar) => {{
          const x0 = x(Math.min(0, bar.value));
          const x1 = x(Math.max(0, bar.value));
          svg.appendChild(svgEl("rect", {{
            x: x0,
            y: bar.y,
            width: Math.max(1, x1 - x0),
            height: 8,
            rx: 2,
            fill: bar.color,
            opacity: row.series.kind === "v1" ? "0.92" : "0.68"
          }}));
        }});
      }});

      svg.appendChild(svgEl("text", {{ x: margin.left, y: 10, class: "axis" }}, "Green = long P&L, blue = short P&L"));
    }}

    function deltaClass(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "";
      return value >= 0 ? "delta-pos" : "delta-neg";
    }}

    function renderDeltaTable() {{
      const llmRows = DATA.series.filter((series) => series.kind === "v1" && series.id.includes(":llm_pool_v1"));
      const rows = llmRows.flatMap((series) => ["pool", "universe"].map((baselineKind) => {{
        const baseline = matchingV1Baseline(series, baselineKind);
        if (!baseline) return null;
        const m = metric(series);
        const b = metric(baseline);
        return {{
          run: compactLabel(series.label),
          baseline: baselineKind === "pool" ? "Pool" : "Universe",
          sharpe: m.sharpe - b.sharpe,
          cagr: m.cagrPct - b.cagrPct,
          ic: m.icMean - b.icMean,
          maxDd: m.maxDrawdownPct - b.maxDrawdownPct,
          llmSharpe: m.sharpe,
          baseSharpe: b.sharpe
        }};
      }}).filter(Boolean));
      const head = `<thead><tr><th>LLM Run</th><th>Vs</th><th>LLM Sh</th><th>Base Sh</th><th>Delta Sh</th><th>Delta CAGR</th><th>Delta IC</th><th>DD Diff</th></tr></thead>`;
      const body = rows.map((row) => `
        <tr>
          <td class="name">${{row.run}}</td>
          <td>${{row.baseline}}</td>
          <td>${{num(row.llmSharpe, 2)}}</td>
          <td>${{num(row.baseSharpe, 2)}}</td>
          <td class="${{deltaClass(row.sharpe)}}">${{num(row.sharpe, 2)}}</td>
          <td class="${{deltaClass(row.cagr)}}">${{pct(row.cagr)}}</td>
          <td class="${{deltaClass(row.ic)}}">${{num(row.ic, 4)}}</td>
          <td class="${{deltaClass(-row.maxDd)}}">${{pct(row.maxDd)}}</td>
        </tr>
      `).join("");
      deltaTable.innerHTML = `${{head}}<tbody>${{body}}</tbody>`;
    }}

    function renderCostChart() {{
      const svg = document.getElementById("costChart");
      svg.innerHTML = "";
      const rows = DATA.series.filter((series) => series.kind === "v1").slice(-8);
      const width = 640;
      const height = 340;
      const margin = {{ top: 18, right: 26, bottom: 84, left: 50 }};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      const keys = [costKey("0"), costKey("2p5"), costKey("5")];
      const vals = rows.flatMap((series) => keys.map((key) => series.metrics[key]?.sharpe ?? 0));
      const yMax = Math.max(0.25, ...vals) * 1.15;
      const y = (value) => margin.top + (1 - value / yMax) * innerH;
      for (let i = 0; i <= 4; i++) {{
        const value = yMax * i / 4;
        const yy = y(value);
        svg.appendChild(svgEl("line", {{ x1: margin.left, x2: width - margin.right, y1: yy, y2: yy, class: "gridline" }}));
        svg.appendChild(svgEl("text", {{ x: margin.left - 7, y: yy + 4, "text-anchor": "end", class: "axis" }}, value.toFixed(2)));
      }}
      const groupW = innerW / Math.max(rows.length, 1);
      const barW = Math.min(16, groupW / 4);
      rows.forEach((series, index) => {{
        const cx = margin.left + groupW * index + groupW / 2;
        keys.forEach((key, kIndex) => {{
          const value = series.metrics[key]?.sharpe ?? 0;
          const x = cx + (kIndex - 1) * (barW + 3);
          svg.appendChild(svgEl("rect", {{ x, y: y(value), width: barW, height: height - margin.bottom - y(value), rx: 2, fill: COLORS[kIndex], opacity: "0.85" }}));
        }});
        svg.appendChild(svgEl("text", {{ x: cx, y: height - 64, transform: `rotate(35 ${{cx}} ${{height - 64}})`, "text-anchor": "start", class: "axis" }}, compactLabel(series.label)));
      }});
      svg.appendChild(svgEl("text", {{ x: width - 152, y: 14, class: "axis" }}, "0 / 2.5 / 5 bps"));
    }}

    function renderRollingIcChart() {{
      const svg = document.getElementById("rollingIcChart");
      svg.innerHTML = "";
      const width = 640;
      const family = executionFamily();
      const rows = DATA.series.filter((series) => diagnostics(series).rollingIc?.[family]?.length);
      const all = rows.flatMap((series) => diagnostics(series).rollingIc[family].map((point) => ({{ date: new Date(point.date + "T00:00:00"), value: point.value }})));
      if (!all.length) return;
      const cols = 3;
      const cellW = width / cols;
      const cellH = 142;
      const height = 34 + Math.ceil(rows.length / cols) * cellH;
      const pad = {{ top: 26, right: 14, bottom: 24, left: 38 }};
      const innerW = cellW - pad.left - pad.right;
      const innerH = cellH - pad.top - pad.bottom;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      const minX = Math.min(...all.map((p) => p.date.getTime()));
      const maxX = Math.max(...all.map((p) => p.date.getTime()));
      const minY = Math.min(-0.1, ...all.map((p) => p.value));
      const maxY = Math.max(0.1, ...all.map((p) => p.value));
      rows.forEach((series, index) => {{
        const col = index % cols;
        const row = Math.floor(index / cols);
        const ox = col * cellW;
        const oy = 18 + row * cellH;
        const x = (date) => ox + pad.left + ((date.getTime() - minX) / Math.max(maxX - minX, 1)) * innerW;
        const y = (value) => oy + pad.top + (1 - ((value - minY) / Math.max(maxY - minY, 0.0001))) * innerH;
        [minY, 0, maxY].forEach((value) => {{
          const yy = y(value);
          svg.appendChild(svgEl("line", {{ x1: ox + pad.left, x2: ox + cellW - pad.right, y1: yy, y2: yy, class: value === 0 ? "axis-line" : "gridline" }}));
        }});
        if (col === 0) {{
          [maxY, 0, minY].forEach((value) => {{
            const yy = y(value);
            svg.appendChild(svgEl("text", {{ x: ox + pad.left - 7, y: yy + 4, "text-anchor": "end", class: "axis" }}, value.toFixed(2)));
          }});
        }}
        const points = diagnostics(series).rollingIc[family];
        const d = points.map((point, pointIndex) => `${{pointIndex ? "L" : "M"}}${{x(new Date(point.date + "T00:00:00")).toFixed(1)}},${{y(point.value).toFixed(1)}}`).join(" ");
        const isLlm = series.id.includes(":llm_pool_v1");
        svg.appendChild(svgEl("path", {{
          d,
          class: "path v1",
          stroke: COLORS[index % COLORS.length],
          "stroke-dasharray": isLlm ? "0" : "5 4",
          opacity: isLlm ? "0.95" : "0.72"
        }}));
        svg.appendChild(svgEl("text", {{ x: ox + pad.left, y: oy + 13, class: "point-label" }}, compactLabel(series.label)));
        const vals = points.map((point) => point.value);
        const latest = vals[vals.length - 1];
        const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
        svg.appendChild(svgEl("text", {{ x: ox + pad.left, y: oy + cellH - 6, class: "axis" }}, `latest ${{num(latest, 3)}} / avg ${{num(avg, 3)}}`));
      }});
      renderRollingIcTable(family);
    }}

    function renderRollingIcTable(family = executionFamily()) {{
      const rows = sortedSeries().filter((series) => diagnostics(series).rollingIc?.[family]?.length);
      const head = `<thead><tr><th>Run</th><th>Family</th><th>Latest 13w IC</th><th>Avg 13w IC</th><th>Best</th><th>Worst</th></tr></thead>`;
      const body = rows.map((series) => {{
        const points = diagnostics(series).rollingIc[family] || [];
        const vals = points.map((point) => point.value);
        const latest = vals[vals.length - 1];
        const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
        return `
          <tr>
            <td class="name">${{compactLabel(series.label)}}</td>
            <td>${{series.family}}</td>
            <td>${{num(latest, 4)}}</td>
            <td>${{num(avg, 4)}}</td>
            <td>${{vals.length ? num(Math.max(...vals), 4) : "n/a"}}</td>
            <td>${{vals.length ? num(Math.min(...vals), 4) : "n/a"}}</td>
          </tr>
        `;
      }}).join("");
      rollingIcTable.innerHTML = `${{head}}<tbody>${{body}}</tbody>`;
    }}

    function renderDecileChart() {{
      const svg = document.getElementById("decileChart");
      svg.innerHTML = "";
      const family = executionFamily();
      const rows = sortedSeries()
        .filter((series) => diagnostics(series).deciles?.[family]?.length)
        .map((series) => {{
          const deciles = diagnostics(series).deciles[family] || [];
          const first = deciles[0]?.annualReturnPct ?? null;
          const last = deciles[deciles.length - 1]?.annualReturnPct ?? null;
          return {{ series, spread: first !== null && last !== null ? first - last : null }};
        }})
        .filter((row) => row.spread !== null);
      const width = 640;
      const height = 340;
      const margin = {{ top: 18, right: 28, bottom: 112, left: 58 }};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      if (!rows.length) return;
      const minY = Math.min(0, ...rows.map((row) => row.spread ?? 0));
      const maxY = Math.max(0, ...rows.map((row) => row.spread ?? 0));
      const pad = Math.max((maxY - minY) * 0.15, 2);
      const y0 = minY - pad;
      const y1 = maxY + pad;
      const x = (idx) => margin.left + innerW * (idx / Math.max(rows.length - 1, 1));
      const y = (value) => margin.top + (1 - ((value - y0) / (y1 - y0))) * innerH;
      const zeroY = y(0);
      [y0, 0, y1].forEach((value) => {{
        const yy = y(value);
        svg.appendChild(svgEl("line", {{ x1: margin.left, x2: width - margin.right, y1: yy, y2: yy, class: value === 0 ? "axis-line" : "gridline" }}));
        svg.appendChild(svgEl("text", {{ x: margin.left - 8, y: yy + 4, "text-anchor": "end", class: "axis" }}, pct(value)));
      }});
      svg.appendChild(svgEl("line", {{ x1: margin.left, x2: width - margin.right, y1: zeroY, y2: zeroY, class: "axis-line" }}));
      rows.forEach((row, idx) => {{
        const value = row.spread ?? 0;
        const barW = innerW / rows.length * 0.62;
        svg.appendChild(svgEl("rect", {{ x: x(idx) - barW / 2, y: y(Math.max(0, value)), width: barW, height: Math.abs(y(value) - zeroY), rx: 2, fill: COLORS[idx % COLORS.length], opacity: "0.86" }}));
        svg.appendChild(svgEl("text", {{ x: x(idx), y: height - 88, transform: `rotate(35 ${{x(idx)}} ${{height - 88}})`, "text-anchor": "start", class: "axis" }}, compactLabel(row.series.label)));
      }});
      svg.appendChild(svgEl("text", {{ x: margin.left, y: 10, class: "axis" }}, "D1 minus D10 annualized return by run"));
      renderDecileTable();
    }}

    function renderDecileTable() {{
      const family = executionFamily();
      const rows = sortedSeries()
        .filter((series) => diagnostics(series).deciles?.[family]?.length)
        .map((series) => {{
          const deciles = diagnostics(series).deciles[family] || [];
          const first = deciles[0]?.annualReturnPct ?? null;
          const last = deciles[deciles.length - 1]?.annualReturnPct ?? null;
          const values = deciles.map((row) => row.annualReturnPct).filter((value) => value !== null && value !== undefined && !Number.isNaN(value));
          const spread = first !== null && last !== null ? first - last : null;
          const positiveSteps = deciles.slice(1).reduce((count, row, index) => count + (row.annualReturnPct >= deciles[index].annualReturnPct ? 1 : 0), 0);
          return {{ series, first, last, spread, best: values.length ? Math.max(...values) : null, worst: values.length ? Math.min(...values) : null, positiveSteps }};
        }});
      const head = `<thead><tr><th>Run</th><th>Family</th><th>D1</th><th>D10</th><th>D1-D10</th><th>Best Decile</th><th>Worst Decile</th><th>Up Steps</th></tr></thead>`;
      const body = rows.map((row) => `
        <tr>
          <td class="name">${{compactLabel(row.series.label)}}</td>
          <td>${{row.series.family}}</td>
          <td>${{pct(row.first)}}</td>
          <td>${{pct(row.last)}}</td>
          <td>${{pct(row.spread)}}</td>
          <td>${{pct(row.best)}}</td>
          <td>${{pct(row.worst)}}</td>
          <td>${{row.positiveSteps}}/9</td>
        </tr>
      `).join("");
      decileTable.innerHTML = `${{head}}<tbody>${{body}}</tbody>`;
    }}

    function renderCopyChart() {{
      const svg = document.getElementById("copyChart");
      svg.innerHTML = "";
      const rows = DATA.series.filter((series) => diagnostics(series).copyLike).map((series) => ({{
        label: compactLabel(series.label),
        value: diagnostics(series).copyLike.avgAbsCorr,
        high: diagnostics(series).copyLike.highAbsCorrCount
      }}));
      const width = 640;
      const rowH = 32;
      const height = Math.max(310, 52 + rows.length * rowH);
      const margin = {{ top: 18, right: 42, bottom: 28, left: 170 }};
      const innerW = width - margin.left - margin.right;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      rows.forEach((row, index) => {{
        const y = margin.top + index * rowH + 7;
        svg.appendChild(svgEl("text", {{ x: margin.left - 8, y: y + 10, "text-anchor": "end", class: "axis" }}, row.label));
        svg.appendChild(svgEl("rect", {{ x: margin.left, y, width: innerW * Math.min(row.value, 1), height: 14, rx: 2, fill: row.value > 0.9 ? "#b23a48" : "#345995", opacity: "0.82" }}));
        svg.appendChild(svgEl("text", {{ x: margin.left + innerW * Math.min(row.value, 1) + 6, y: y + 11, class: "axis" }}, `${{row.value.toFixed(2)}} | >.90: ${{row.high}}`));
      }});
    }}

    function renderRegimeTable() {{
      const rows = sortedSeries().flatMap((series) => {{
        const regimes = diagnostics(series).regimes?.[selectedKey(series)] || [];
        return regimes.map((row) => ({{ series, ...row }}));
      }});
      const regimes = [...new Set(rows.map((row) => row.regime))].sort();
      const matrixRows = sortedSeries().filter((series) => (diagnostics(series).regimes?.[selectedKey(series)] || []).length);
      regimeMatrix.style.setProperty("--regime-count", regimes.length);
      const header = `
        <div class="regime-row">
          <div class="regime-cell regime-head">Run</div>
          ${{regimes.map((regime) => `<div class="regime-cell regime-head">${{regime}}</div>`).join("")}}
        </div>
      `;
      const matrixBody = matrixRows.map((series) => {{
        const byRegime = Object.fromEntries((diagnostics(series).regimes?.[selectedKey(series)] || []).map((row) => [row.regime, row]));
        const cells = regimes.map((regime) => {{
          const row = byRegime[regime];
          if (!row) return `<div class="regime-cell"></div>`;
          return `
            <div class="regime-cell" style="background:${{heatColor(row.annualReturnPct || 0)}}">
              <strong>${{pct(row.annualReturnPct)}}</strong>
              Sh ${{num(row.sharpe, 2)}} · n ${{row.n}}
            </div>
          `;
        }}).join("");
        return `
          <div class="regime-row">
            <div class="regime-cell"><strong>${{compactLabel(series.label)}}</strong></div>
            ${{cells}}
          </div>
        `;
      }}).join("");
      regimeMatrix.innerHTML = header + matrixBody;

      const head = `<thead><tr><th>Run</th><th>Regime</th><th>Ann Ret</th><th>Sharpe</th><th>Hit</th><th>N</th></tr></thead>`;
      const body = rows
        .sort((a, b) => compactLabel(a.series.label).localeCompare(compactLabel(b.series.label)) || a.regime.localeCompare(b.regime))
        .map((row) => `
        <tr>
          <td class="name">${{compactLabel(row.series.label)}}</td>
          <td>${{row.regime}}</td>
          <td>${{pct(row.annualReturnPct)}}</td>
          <td>${{num(row.sharpe, 2)}}</td>
          <td>${{pct(row.hitRatePct)}}</td>
          <td>${{row.n}}</td>
        </tr>
      `).join("");
      regimeTable.innerHTML = `${{head}}<tbody>${{body}}</tbody>`;
    }}

    function renderConcentrationTable() {{
      const rows = sortedSeries().flatMap((series) => {{
        const conc = diagnostics(series)?.concentration || {{}};
        return ["long", "short"].map((side) => {{
          const names = (conc[side] || []).slice(0, 5);
          if (!names.length) return null;
          return {{
            series,
            side: side === "long" ? "Long" : "Short",
            tickers: names.map((row) => `${{row.ticker}} ${{row.count}} (${{row.periodPct.toFixed(0)}}%)`).join(", ")
          }};
        }}).filter(Boolean);
      }});
      const head = `<thead><tr><th>Run</th><th>Family</th><th>Side</th><th>Top Repeated Tickers</th></tr></thead>`;
      const body = rows.map((row) => `
        <tr>
          <td class="name">${{compactLabel(row.series.label)}}</td>
          <td>${{row.series.family}}</td>
          <td>${{row.side}}</td>
          <td>${{row.tickers}}</td>
        </tr>
      `).join("");
      concentrationTable.innerHTML = `${{head}}<tbody>${{body}}</tbody>`;
    }}

    function renderWeekTables() {{
      const periodRows = (which) => sortedSeries().flatMap((series) => {{
        const key = selectedKey(series);
        const periods = diagnostics(series)?.topWeeks?.[key] || diagnostics(series)?.topPeriods?.[key] || {{ best: [], worst: [] }};
        return (periods[which] || []).map((row) => ({{
          ...row,
          series,
          periodBasis: row.basis || (series.kind === "legacy" ? "daily" : "weekly")
        }}));
      }}).sort((a, b) => a.returnPct - b.returnPct);
      const bestRows = periodRows("best").sort((a, b) => b.returnPct - a.returnPct).slice(0, 18);
      const worstRows = periodRows("worst").slice(0, 18);
      const table = (title, rows) => {{
        const head = `<thead><tr><th>${{title}}</th><th>Run</th><th>Basis</th><th>Ret</th><th>L</th><th>S</th><th>IC</th><th>Turn</th><th>Regime</th></tr></thead>`;
        const body = rows.map((row) => `
          <tr>
            <td class="name">${{row.date}}</td>
            <td>${{compactLabel(row.series.label)}}</td>
            <td>${{row.periodBasis}}</td>
            <td>${{pct(row.returnPct)}}</td>
            <td>${{pct(row.longPct)}}</td>
            <td>${{pct(row.shortPct)}}</td>
            <td>${{num(row.ic, 2)}}</td>
            <td>${{num(row.turnover, 1)}}</td>
            <td>${{row.regime}}</td>
          </tr>
        `).join("");
        return `${{head}}<tbody>${{body}}</tbody>`;
      }};
      bestWeeksTable.innerHTML = table("Best Period", bestRows);
      worstWeeksTable.innerHTML = table("Worst Period", worstRows);
    }}

    function renderContext() {{
      const success = DATA.success || {{}};
      const v1Books = (DATA.v1Meta?.variants || [])
        .map((variant) => variant.poolPerSide ? `${{variant.poolPerSide}}/${{variant.poolPerSide}} pool -> ${{variant.book}}` : variant.book)
        .join(", ") || "75/75";
      const rows = [
        ["Prior composite", `rank/confidence variants, ${{DATA.legacyMeta.candidatePerSide}} candidates per side, ${{DATA.legacyMeta.topN}}/${{DATA.legacyMeta.bottomN}} held book`],
        ["V1 system", `one 0-100 LLM score per ticker; pool/book views: ${{v1Books}}`],
        ["Diagnostic coverage", "Best/worst periods and ticker concentration include legacy plus V1. Rolling IC and decile diagnostics are V1-only because the legacy harness did not save comparable source series."],
        ["Selected view", `${{optionLabel()}}; legacy rows use matching cost only`],
        ["Period column", "Legacy counts daily P&L rows; v1 counts weekly holding windows."],
        ["V1 75/75 win rule", success.is_win ? "Pass" : "No"],
        ["V1 75/75 stop rule", success.is_stop_failure ? "Hit" : "Clear"],
        ["Use this report for", "Broad signal comparison. Use v1-v1 rows for standardized protocol conclusions."]
      ];
      contextRows.innerHTML = rows.map(([label, value]) => `
        <div class="meta-row"><span>${{label}}</span><strong>${{value}}</strong></div>
      `).join("");
    }}

    function renderAll() {{
      growthSubtitle.textContent = `Legacy = daily composite path at matching cost level; v1 = ${{optionLabel()}}.`;
      renderSummary();
      renderMethodCards();
      renderBaselineTable();
      renderRunTrace();
      renderAnalysisCards();
      renderFocusNotes();
      renderGrowthChart();
      renderQualityChart();
      renderLegChart();
      renderDeltaTable();
      renderCostChart();
      renderRollingIcChart();
      renderDecileChart();
      renderCopyChart();
      renderRegimeTable();
      renderConcentrationTable();
      renderWeekTables();
      renderLeaderboard();
      renderYearly();
      renderContext();
    }}

    renderControls();
    renderAll();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_ROOT)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_payload(args.legacy_root, args.v1_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_page(payload, args.output), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
