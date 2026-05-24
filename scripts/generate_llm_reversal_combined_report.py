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
    "mean_reversion_rank_score": "Legacy Det Rank",
    "mean_reversion_raw_score": "Legacy Det Raw",
}

LEGACY_SIGNAL_DESCRIPTIONS = {
    "llm_raw_confidence_score": "LLM confidence only",
    "llm_rank_score": "LLM rank only",
    "llm_rank_confidence_score": "LLM rank + confidence",
    "mean_reversion_rank_score": "deterministic mean-reversion rank",
    "mean_reversion_raw_score": "deterministic raw mean-reversion score",
}

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


def build_legacy_series(root: Path) -> list[dict[str, Any]]:
    config = read_json(root / "run_config.json")
    leaderboard = read_csv(root / "compact_leaderboard.csv")
    configured_cost_bps = float(config.get("cost_bps", 0.0) or 0.0)
    series: list[dict[str, Any]] = []

    for row in leaderboard:
        score_type = row["score_type"]
        pnl_path = root / score_type / "factor_pnl.csv"
        pnl_rows = read_csv(pnl_path)
        returns_by_key: dict[str, list[float]] = {option["key"]: [] for option in LEGACY_COST_OPTIONS}
        long_leg_returns_by_key: dict[str, list[float]] = {option["key"]: [] for option in LEGACY_COST_OPTIONS}
        short_leg_returns_by_key: dict[str, list[float]] = {option["key"]: [] for option in LEGACY_COST_OPTIONS}
        yearly_returns: dict[str, dict[str, list[float]]] = {
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
    #qualityChart, #legChart {{ height: 340px; }}
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
    @media (max-width: 980px) {{
      header, .layout, .detail-grid, .guide-grid {{ display: block; }}
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
    const growthSubtitle = document.getElementById("growthSubtitle");
    const tableSubtitle = document.getElementById("tableSubtitle");
    const leaderboardBody = document.getElementById("leaderboardBody");
    const yearlyTable = document.getElementById("yearlyTable");
    const contextRows = document.getElementById("contextRows");

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
        .replace("Legacy Det Rank", "Legacy det rank")
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
        svg.appendChild(svgEl("text", {{
          x: cx + row.r + 4,
          y: cy + 4,
          class: "point-label"
        }}, compactLabel(row.series.label)));
      }});
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

    function renderContext() {{
      const success = DATA.success || {{}};
      const v1Books = (DATA.v1Meta?.variants || [])
        .map((variant) => variant.poolPerSide ? `${{variant.poolPerSide}}/${{variant.poolPerSide}} pool -> ${{variant.book}}` : variant.book)
        .join(", ") || "75/75";
      const rows = [
        ["Prior composite", `rank/confidence variants, ${{DATA.legacyMeta.candidatePerSide}} candidates per side, ${{DATA.legacyMeta.topN}}/${{DATA.legacyMeta.bottomN}} held book`],
        ["V1 system", `one 0-100 LLM score per ticker; pool/book views: ${{v1Books}}`],
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
      renderGrowthChart();
      renderQualityChart();
      renderLegChart();
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
