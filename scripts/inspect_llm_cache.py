#!/usr/bin/env python3
"""Summarize and audit LLM portfolio-construction cache files."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_source_config import DEFAULT_COMPOSITE_SCORES_DIR, RUSSELL_MEMBERSHIP_PATH
from portfolio_constraints import filter_symbol_hygiene_frame, has_min_trade_price


COMPOSITE_FILENAME_RE = re.compile(
    r"^composite_scores_(?P<ticker>.+)_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}$"
)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ticker_set(rows: list[Any]) -> set[str]:
    tickers: set[str] = set()
    for row in rows:
        if isinstance(row, dict) and row.get("ticker"):
            tickers.add(str(row["ticker"]))
    return tickers


def _field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return ""


def _usage_field(usage: dict[str, Any], field: str) -> Any:
    provider_usage = _as_dict(usage.get("provider_usage"))
    if field in provider_usage:
        return provider_usage[field]
    return usage.get(field, "")


def _iter_portfolios(cache: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    portfolios = cache.get("portfolios", [])
    if isinstance(portfolios, dict):
        return [
            (str(date), portfolio)
            for date, portfolio in sorted(portfolios.items())
            if isinstance(portfolio, dict)
        ]
    if isinstance(portfolios, list):
        rows = []
        for index, portfolio in enumerate(portfolios):
            if not isinstance(portfolio, dict):
                continue
            date = portfolio.get("rebalance_date") or portfolio.get("date") or str(index)
            rows.append((str(date), portfolio))
        return rows
    return []


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_truthy_membership(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    numeric = _safe_float(value)
    if numeric is not None:
        return numeric != 0.0
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _boolish_or_blank(value: Any) -> bool | str:
    if value in ("", None):
        return ""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return str(value)


def _selection_return_format(fieldnames: list[str] | None) -> str:
    fields = set(fieldnames or [])
    if {"realized_c2c_return_pct", "realized_o2o_return_pct"} & fields:
        return "raw_underlying_return"
    return "signed_position_pnl"


def _signed_selection_return(side: str, realized: float, return_format: str) -> float:
    if side == "SHORT" and return_format == "raw_underlying_return":
        return -realized
    return realized


def _composite_ticker_from_path(path: Path) -> str:
    match = COMPOSITE_FILENAME_RE.match(path.stem)
    if match:
        return match.group("ticker").upper()
    parts = path.stem.split("_")
    return parts[2].upper() if len(parts) >= 3 else path.stem.upper()


def _build_composite_file_index(composite_dir: Path | None) -> dict[str, Path]:
    if composite_dir is None or not composite_dir.exists():
        return {}
    return {_composite_ticker_from_path(path): path for path in sorted(composite_dir.glob("*.csv"))}


def _load_clean_composite_frame(
    ticker: str,
    composite_index: dict[str, Path],
    frame_cache: dict[str, pd.DataFrame | None],
) -> pd.DataFrame | None:
    ticker = str(ticker).upper()
    if ticker in frame_cache:
        return frame_cache[ticker]
    path = composite_index.get(ticker)
    if path is None:
        frame_cache[ticker] = None
        return None
    try:
        header = pd.read_csv(path, nrows=0)
        needed_cols = [
            col
            for col in ("date", "adj_close", "is_historical_suffix", "collision_count")
            if col in header.columns
        ]
        if "date" not in needed_cols or "adj_close" not in needed_cols:
            frame_cache[ticker] = None
            return None
        df = pd.read_csv(path, usecols=needed_cols, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        cleaned = filter_symbol_hygiene_frame(df)
        if cleaned is None or cleaned.empty:
            frame_cache[ticker] = None
        else:
            cleaned = cleaned.sort_values("date").reset_index(drop=True)
            dates = pd.to_datetime(cleaned["date"])
            cleaned.attrs["date_to_pos"] = {
                pd.Timestamp(date).strftime("%Y-%m-%d"): idx
                for idx, date in enumerate(dates)
            }
            cleaned.attrs["dates_ns"] = dates.astype("int64").to_numpy()
            frame_cache[ticker] = cleaned
    except Exception:
        frame_cache[ticker] = None
    return frame_cache[ticker]


def _exact_composite_row(df: pd.DataFrame | None, date: Any) -> pd.Series | None:
    if df is None or df.empty:
        return None
    date_to_pos = df.attrs.get("date_to_pos", {})
    pos = date_to_pos.get(pd.Timestamp(date).strftime("%Y-%m-%d"))
    if pos is None:
        return None
    return df.iloc[int(pos)]


def _ffill_composite_row(df: pd.DataFrame | None, date: Any) -> pd.Series | None:
    if df is None or df.empty:
        return None
    dates_ns = df.attrs.get("dates_ns")
    if dates_ns is None:
        dates_ns = pd.to_datetime(df["date"]).astype("int64").to_numpy()
    idx = dates_ns.searchsorted(pd.Timestamp(date).value, side="right") - 1
    if idx < 0:
        return None
    return df.iloc[int(idx)]


def _row_price(row: pd.Series | None) -> float | None:
    if row is None:
        return None
    value = _safe_float(row.get("adj_close"))
    return value if value is not None else None


def _load_membership_lookup(
    path: Path | None,
    date_values: set[str] | None = None,
) -> tuple[tuple[pd.Timestamp, ...], dict[pd.Timestamp, set[str]]]:
    if path is None or not path.exists():
        return (), {}
    min_date = min((pd.Timestamp(date) for date in date_values), default=None) if date_values else None
    max_date = max((pd.Timestamp(date) for date in date_values), default=None) if date_values else None
    if path.suffix.lower() == ".parquet":
        filters = None
        if min_date is not None and max_date is not None:
            filters = [("date", "<=", max_date), ("date", ">=", min_date - pd.Timedelta(days=370))]
        membership = pd.read_parquet(path, columns=["date", "symbol", "in_r1000"], filters=filters)
    else:
        membership = pd.read_csv(path)
    if "date" not in membership.columns or "symbol" not in membership.columns:
        return (), {}
    membership = membership.copy()
    membership["date"] = pd.to_datetime(membership["date"], errors="coerce")
    membership = membership.dropna(subset=["date", "symbol"])
    if "in_r1000" in membership.columns:
        membership = membership[membership["in_r1000"].map(_is_truthy_membership)]
    membership["symbol"] = membership["symbol"].astype(str).str.upper().str.strip().str.split("-", n=1).str[0]
    grouped = membership.groupby("date")["symbol"].agg(lambda values: set(values)).to_dict()
    dates = tuple(sorted(pd.Timestamp(date) for date in grouped))
    return dates, grouped


def _is_member_on_or_before(
    ticker: str,
    date: Any,
    membership_dates: tuple[pd.Timestamp, ...],
    membership_by_date: dict[pd.Timestamp, set[str]],
) -> bool | str:
    if not membership_dates:
        return ""
    idx = bisect_right(membership_dates, pd.Timestamp(date)) - 1
    if idx < 0:
        return False
    return str(ticker).upper().split("-", 1)[0] in membership_by_date.get(membership_dates[idx], set())


def _load_next_review_dates(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    rows = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            signal_date = row.get("signal_date") or ""
            review_date = row.get("review_date") or ""
            if signal_date and review_date:
                rows.append((pd.Timestamp(signal_date), pd.Timestamp(review_date)))
    rows = sorted(set(rows))
    next_by_signal: dict[str, str] = {}
    for idx, (signal_date, _review_date) in enumerate(rows[:-1]):
        next_by_signal[signal_date.strftime("%Y-%m-%d")] = rows[idx + 1][1].strftime("%Y-%m-%d")
    return next_by_signal


def _load_factor_period_performance(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}

    rows: list[dict[str, str]] = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    periods: dict[str, dict[str, Any]] = {}
    current_date = ""
    for row in rows:
        signal_date = row.get("rebalance_signal_date") or ""
        if signal_date:
            current_date = signal_date
        if not current_date:
            continue

        period = periods.setdefault(
            current_date,
            {
                "pnl_days": 0,
                "period_net_return_pct": 0.0,
                "period_gross_return_pct": 0.0,
                "period_long_pnl_bps": 0.0,
                "period_short_pnl_bps": 0.0,
                "period_cost_bps": 0.0,
                "period_turnover": 0.0,
                "period_start": row.get("date", ""),
                "period_end": row.get("date", ""),
            },
        )
        net = _safe_float(row.get("net_daily_ret_bps", row.get("daily_ret_bps")))
        gross = _safe_float(row.get("gross_daily_ret_bps"))
        long_bps = _safe_float(row.get("long_pnl_bps"))
        short_bps = _safe_float(row.get("short_pnl_bps"))
        cost = _safe_float(row.get("cost_bps_applied"))
        turnover = _safe_float(row.get("turnover_total"))
        if net is not None:
            period["period_net_return_pct"] += net / 100.0
        if gross is not None:
            period["period_gross_return_pct"] += gross / 100.0
        if long_bps is not None:
            period["period_long_pnl_bps"] += long_bps
        if short_bps is not None:
            period["period_short_pnl_bps"] += short_bps
        if cost is not None:
            period["period_cost_bps"] += cost
        if turnover is not None:
            period["period_turnover"] += turnover
        period["pnl_days"] += 1
        period["period_end"] = row.get("date", period["period_end"])

    return periods


def _load_selection_performance(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}

    grouped: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        return_format = _selection_return_format(reader.fieldnames)
        for row in reader:
            date = row.get("rebalance_date") or row.get("signal_date") or ""
            if not date:
                continue
            side = str(row.get("position_type", "")).upper()
            realized = _safe_float(row.get("realized_return_pct"))
            weight = abs(_safe_float(row.get("weight")) or 0.0)
            if realized is None:
                continue
            signed_return = _signed_selection_return(side, realized, return_format)
            bucket = grouped.setdefault(
                date,
                {
                    "selection_realized_return_pct": 0.0,
                    "selection_long_realized_pct": 0.0,
                    "selection_short_realized_pct": 0.0,
                    "selection_long_count": 0,
                    "selection_short_count": 0,
                },
            )
            bucket["selection_realized_return_pct"] += weight * signed_return
            if side == "LONG":
                bucket["selection_long_realized_pct"] += weight * signed_return
                bucket["selection_long_count"] += 1
            elif side == "SHORT":
                bucket["selection_short_realized_pct"] += weight * signed_return
                bucket["selection_short_count"] += 1
    return grouped


def _load_selection_detail(path: Path | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    if path is None:
        return {}

    detail: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        return_format = _selection_return_format(reader.fieldnames)
        for row in reader:
            date = row.get("rebalance_date") or row.get("signal_date") or ""
            side = str(row.get("position_type", row.get("side", ""))).upper()
            ticker = str(row.get("ticker", "")).upper()
            if not date or side not in {"LONG", "SHORT"} or not ticker:
                continue
            realized = _safe_float(row.get("realized_return_pct"))
            weight = abs(_safe_float(row.get("weight")) or 0.0)
            signed_return = ""
            contribution = ""
            if realized is not None:
                signed_return = _signed_selection_return(side, realized, return_format)
                contribution = weight * signed_return
            detail[(date, side, ticker)] = {
                "date": date,
                "side": side,
                "ticker": ticker,
                "rank": row.get("rank", ""),
                "score": row.get("score", ""),
                "weight": weight,
                "realized_return_pct": realized if realized is not None else "",
                "raw_or_source_return_pct": realized if realized is not None else "",
                "signed_return_pct": signed_return,
                "position_pnl_return_pct": signed_return,
                "contribution_pct": contribution,
                "entry_date": row.get("entry_date", ""),
                "exit_date": row.get("exit_date", ""),
                "return_format": return_format,
            }
    return detail


def _load_composite_coverage(path: Path | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    detail: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            date = row.get("date") or row.get("signal_date") or row.get("rebalance_date") or ""
            side = str(row.get("side", row.get("position_type", ""))).upper()
            ticker = str(row.get("ticker", "")).upper()
            if not date or side not in {"LONG", "SHORT"} or not ticker:
                continue
            detail[(date, side, ticker)] = {
                "date": date,
                "side": side,
                "ticker": ticker,
                "has_composite_file": row.get("has_composite_file", ""),
                "has_composite_row": row.get("has_composite_row", ""),
                "coverage_reason": row.get("coverage_reason", ""),
            }
    return detail


def _load_selection_audit(path: Path | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    detail: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            date = row.get("signal_date") or row.get("rebalance_date") or row.get("date") or ""
            side = str(row.get("side", row.get("position_type", ""))).upper()
            ticker = str(row.get("ticker", "")).upper()
            if not date or side not in {"LONG", "SHORT"} or not ticker:
                continue
            detail[(date, side, ticker)] = {
                "date": date,
                "side": side,
                "ticker": ticker,
                "trade_date": row.get("trade_date", ""),
                "survived_to_execution": row.get("survived_to_execution", ""),
                "drop_reason": row.get("drop_reason", ""),
                "trade_price": row.get("trade_price", ""),
            }
    return detail


def _summarize_baseline_comparison(
    llm_detail: dict[tuple[str, str, str], dict[str, Any]],
    baseline_detail: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not baseline_detail:
        return [], []

    comparison_rows: list[dict[str, Any]] = []
    summary_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    all_keys = sorted(set(llm_detail) | set(baseline_detail))
    for key in all_keys:
        date, side, ticker = key
        llm_row = llm_detail.get(key)
        base_row = baseline_detail.get(key)
        if llm_row and base_row:
            decision = "kept"
            source = llm_row
        elif llm_row:
            decision = "added_by_llm"
            source = llm_row
        else:
            decision = "dropped_by_llm"
            source = base_row or {}

        contribution = source.get("contribution_pct", "")
        summary = summary_by_key.setdefault(
            (date, side),
            {
                "date": date,
                "side": side,
                "n_kept_vs_baseline": 0,
                "n_added_by_llm": 0,
                "n_dropped_by_llm": 0,
                "kept_contribution_pct": 0.0,
                "added_contribution_pct": 0.0,
                "dropped_contribution_pct": 0.0,
            },
        )
        if decision == "kept":
            summary["n_kept_vs_baseline"] += 1
            if isinstance(contribution, float):
                summary["kept_contribution_pct"] += contribution
        elif decision == "added_by_llm":
            summary["n_added_by_llm"] += 1
            if isinstance(contribution, float):
                summary["added_contribution_pct"] += contribution
        else:
            summary["n_dropped_by_llm"] += 1
            if isinstance(contribution, float):
                summary["dropped_contribution_pct"] += contribution

        comparison_rows.append(
            {
                "date": date,
                "side": side,
                "ticker": ticker,
                "decision": decision,
                "return_source": "llm_realized" if llm_row else "deterministic_baseline",
                "return_format": source.get("return_format", ""),
                "llm_rank": (llm_row or {}).get("rank", ""),
                "baseline_rank": (base_row or {}).get("rank", ""),
                "llm_weight": (llm_row or {}).get("weight", ""),
                "baseline_weight": (base_row or {}).get("weight", ""),
                "realized_return_pct": source.get("realized_return_pct", ""),
                "raw_or_source_return_pct": source.get("raw_or_source_return_pct", ""),
                "signed_return_pct": source.get("signed_return_pct", ""),
                "position_pnl_return_pct": source.get("position_pnl_return_pct", ""),
                "contribution_pct": contribution,
                "entry_date": source.get("entry_date", ""),
                "exit_date": source.get("exit_date", ""),
            }
        )

    summary_rows = sorted(summary_by_key.values(), key=lambda row: (row["date"], row["side"]))
    for row in summary_rows:
        row["llm_minus_baseline_selection_contribution_pct"] = (
            row["added_contribution_pct"] - row["dropped_contribution_pct"]
        )
    return comparison_rows, summary_rows


def _raw_cache_selection_detail(portfolios: list[tuple[str, dict[str, Any]]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    detail: dict[tuple[str, str, str], dict[str, Any]] = {}
    for date, portfolio in portfolios:
        for side, basket_key in (("LONG", "long_basket"), ("SHORT", "short_basket")):
            rows = _as_list(portfolio.get(basket_key))
            for rank, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    continue
                ticker = str(row.get("ticker", "")).upper()
                if not ticker:
                    continue
                weight = _field(row, "weight", "weight_exec", "weight_raw")
                detail[(date, side, ticker)] = {
                    "date": date,
                    "side": side,
                    "ticker": ticker,
                    "rank": row.get("alpha_rank", row.get("rank", rank)),
                    "score": _field(row, "alpha_edge_score", "edge_score"),
                    "weight": weight,
                    "rationale": _field(row, "alpha_rationale", "rationale"),
                }
    return detail


def _cache_candidate_detail(portfolios: list[tuple[str, dict[str, Any]]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    detail: dict[tuple[str, str, str], dict[str, Any]] = {}
    for date, portfolio in portfolios:
        signal_stage = _as_dict(portfolio.get("signal_stage"))
        for side, candidate_key in (("LONG", "long_candidates"), ("SHORT", "short_candidates")):
            rows = _as_list(signal_stage.get(candidate_key))
            for index, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    continue
                ticker = str(row.get("ticker", "")).upper()
                if not ticker:
                    continue
                detail[(date, side, ticker)] = {
                    "date": date,
                    "side": side,
                    "ticker": ticker,
                    "candidate_rank": row.get("rank", index),
                    "candidate_edge_score": row.get("edge_score", ""),
                    "candidate_rationale": row.get("rationale", ""),
                }
    return detail


def _summarize_raw_baseline_overlap(
    raw_llm_detail: dict[tuple[str, str, str], dict[str, Any]],
    baseline_detail: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not baseline_detail:
        return [], []

    rows: list[dict[str, Any]] = []
    summary_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(set(raw_llm_detail) | set(baseline_detail)):
        date, side, ticker = key
        raw_row = raw_llm_detail.get(key)
        base_row = baseline_detail.get(key)
        if raw_row and base_row:
            decision = "kept"
        elif raw_row:
            decision = "added_by_llm_raw"
        else:
            decision = "dropped_by_llm_raw"

        summary = summary_by_key.setdefault(
            (date, side),
            {
                "date": date,
                "side": side,
                "raw_llm_selected": 0,
                "baseline_selected": 0,
                "n_kept_vs_baseline_raw": 0,
                "n_added_by_llm_raw": 0,
                "n_dropped_by_llm_raw": 0,
            },
        )
        if raw_row:
            summary["raw_llm_selected"] += 1
        if base_row:
            summary["baseline_selected"] += 1
        if decision == "kept":
            summary["n_kept_vs_baseline_raw"] += 1
        elif decision == "added_by_llm_raw":
            summary["n_added_by_llm_raw"] += 1
        else:
            summary["n_dropped_by_llm_raw"] += 1

        rows.append(
            {
                "date": date,
                "side": side,
                "ticker": ticker,
                "decision": decision,
                "raw_llm_rank": (raw_row or {}).get("rank", ""),
                "baseline_rank": (base_row or {}).get("rank", ""),
                "raw_llm_weight": (raw_row or {}).get("weight", ""),
                "baseline_weight": (base_row or {}).get("weight", ""),
                "raw_llm_score": (raw_row or {}).get("score", ""),
                "baseline_score": (base_row or {}).get("score", ""),
                "baseline_realized_return_pct": (base_row or {}).get("realized_return_pct", ""),
                "baseline_contribution_pct": (base_row or {}).get("contribution_pct", ""),
            }
        )

    return rows, sorted(summary_by_key.values(), key=lambda row: (row["date"], row["side"]))


def _summarize_survival(
    raw_llm_detail: dict[tuple[str, str, str], dict[str, Any]],
    realized_llm_detail: dict[tuple[str, str, str], dict[str, Any]],
    selection_audit_detail: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    composite_coverage_detail: dict[tuple[str, str, str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(raw_llm_detail):
        date, side, ticker = key
        raw_row = raw_llm_detail[key]
        realized_row = realized_llm_detail.get(key)
        audit_row = (selection_audit_detail or {}).get(key, {})
        coverage_row = (composite_coverage_detail or {}).get(key, {})
        survived = realized_row is not None
        drop_reason = audit_row.get("drop_reason", "")
        if not drop_reason:
            drop_reason = coverage_row.get("coverage_reason", "")
        if not drop_reason:
            if survived:
                drop_reason = "survived_to_realized"
            elif str(audit_row.get("survived_to_execution", "")).lower() in {"true", "1"}:
                drop_reason = "missing_entry_or_exit_price"
            else:
                drop_reason = "missing_from_realized_attribution_no_audit"
        summary = summary_by_key.setdefault(
            (date, side),
            {
                "date": date,
                "side": side,
                "raw_selected_count": 0,
                "realized_survivor_count": 0,
                "missing_from_realized_count": 0,
                "missing_from_realized_pct": 0.0,
            },
        )
        summary["raw_selected_count"] += 1
        if survived:
            summary["realized_survivor_count"] += 1
        else:
            summary["missing_from_realized_count"] += 1
        rows.append(
            {
                "date": date,
                "side": side,
                "ticker": ticker,
                "raw_rank": raw_row.get("rank", ""),
                "raw_weight": raw_row.get("weight", ""),
                "survived_to_realized": survived,
                "drop_reason": drop_reason,
                "audit_survived_to_execution": audit_row.get("survived_to_execution", ""),
                "audit_trade_price": audit_row.get("trade_price", ""),
                "composite_coverage_reason": coverage_row.get("coverage_reason", ""),
                "has_composite_file": coverage_row.get("has_composite_file", ""),
                "has_composite_row": coverage_row.get("has_composite_row", ""),
                "realized_weight": (realized_row or {}).get("weight", ""),
                "realized_return_pct": (realized_row or {}).get("realized_return_pct", ""),
            }
        )
    summary_rows = sorted(summary_by_key.values(), key=lambda row: (row["date"], row["side"]))
    for row in summary_rows:
        raw_count = int(row["raw_selected_count"] or 0)
        row["missing_from_realized_pct"] = (
            float(row["missing_from_realized_count"]) / raw_count if raw_count else 0.0
        )
    return rows, summary_rows


def _build_decision_flow(
    candidate_detail: dict[tuple[str, str, str], dict[str, Any]],
    raw_llm_detail: dict[tuple[str, str, str], dict[str, Any]],
    realized_llm_detail: dict[tuple[str, str, str], dict[str, Any]],
    baseline_detail: dict[tuple[str, str, str], dict[str, Any]],
    survival_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    survival_detail = {
        (
            str(row.get("date", "")),
            str(row.get("side", "")).upper(),
            str(row.get("ticker", "")).upper(),
        ): row
        for row in survival_rows
        if row.get("date") and row.get("side") and row.get("ticker")
    }
    keys = sorted(set(candidate_detail) | set(raw_llm_detail) | set(realized_llm_detail))
    rows: list[dict[str, Any]] = []
    for key in keys:
        date, side, ticker = key
        candidate_row = candidate_detail.get(key, {})
        raw_row = raw_llm_detail.get(key, {})
        realized_row = realized_llm_detail.get(key, {})
        baseline_row = baseline_detail.get(key, {})
        survival_row = survival_detail.get(key, {})
        rows.append(
            {
                "date": date,
                "side": side,
                "prompt_bucket": "D1 oversold losers" if side == "LONG" else "D10 overbought winners",
                "ticker": ticker,
                "candidate_rank": candidate_row.get("candidate_rank", ""),
                "candidate_edge_score": candidate_row.get("candidate_edge_score", ""),
                "llm_selected": bool(raw_row),
                "llm_rank": raw_row.get("rank", ""),
                "llm_raw_weight": raw_row.get("weight", ""),
                "survived_to_realized": survival_row.get("survived_to_realized", bool(realized_row)),
                "drop_reason": survival_row.get("drop_reason", ""),
                "realized_weight": realized_row.get("weight", ""),
                "position_pnl_return_pct": realized_row.get("position_pnl_return_pct", ""),
                "in_det_top50": bool(baseline_row),
                "det_rank": baseline_row.get("rank", ""),
                "det_position_pnl_return_pct": baseline_row.get("position_pnl_return_pct", ""),
            }
        )
    return rows


def _build_fast_validity_audit(
    raw_llm_detail: dict[tuple[str, str, str], dict[str, Any]],
    realized_llm_detail: dict[tuple[str, str, str], dict[str, Any]],
    baseline_detail: dict[tuple[str, str, str], dict[str, Any]],
    review_log_path: Path | None,
    membership_path: Path | None = None,
    selection_audit_detail: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    composite_coverage_detail: dict[tuple[str, str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Cheap audit from cache truth plus realized exports.

    This deliberately avoids rereading the full composite/membership universe. For old
    runs without llm_selection_audit.csv, the cache proves the name was shown to and
    selected by the LLM, but it does not prove the exact later engine filter.
    """

    next_review_dates = _load_next_review_dates(review_log_path)
    membership_dates, membership_by_date = _load_membership_lookup(
        membership_path,
        {key[0] for key in raw_llm_detail},
    )
    realized_dates = {key[0] for key in realized_llm_detail}
    max_realized_date = max((pd.Timestamp(date) for date in realized_dates), default=None)
    rows: list[dict[str, Any]] = []
    for key in sorted(raw_llm_detail):
        date, side, ticker = key
        raw_row = raw_llm_detail[key]
        realized_row = realized_llm_detail.get(key, {})
        baseline_row = baseline_detail.get(key, {})
        audit_row = (selection_audit_detail or {}).get(key, {})
        coverage_row = (composite_coverage_detail or {}).get(key, {})

        audit_drop_reason = audit_row.get("drop_reason", "")
        audit_survived = _boolish_or_blank(audit_row.get("survived_to_execution", ""))
        coverage_reason = coverage_row.get("coverage_reason", "")
        has_composite_file = _boolish_or_blank(coverage_row.get("has_composite_file", ""))
        has_composite_row = _boolish_or_blank(coverage_row.get("has_composite_row", ""))

        if has_composite_file == "":
            has_composite_file = "cache_implied_true"
        if has_composite_row == "":
            has_composite_row = "cache_implied_true"

        if audit_row:
            loaded_by_factor_engine: bool | str = True
            if audit_drop_reason == "not_russell_member":
                russell_member: bool | str = False
            else:
                russell_member = True

            if audit_drop_reason == "missing_trade_price":
                trade_price_found: bool | str = False
            elif audit_drop_reason in {"below_min_price", "survived_to_execution"}:
                trade_price_found = True
            else:
                trade_price_found = ""

            if audit_drop_reason == "below_min_price":
                min_price_pass: bool | str = False
            elif audit_drop_reason == "survived_to_execution":
                min_price_pass = True
            elif audit_drop_reason == "missing_trade_price":
                min_price_pass = ""
            else:
                min_price_pass = ""
        else:
            loaded_by_factor_engine = True if realized_row else "unknown_without_selection_audit"
            russell_member = _is_member_on_or_before(
                ticker,
                date,
                membership_dates,
                membership_by_date,
            )
            if russell_member == "":
                russell_member = "unknown_without_membership_file"
            trade_price_found = "cache_implied_true_at_prompt"
            min_price_pass = "cache_implied_true_at_prompt"

        written_to_portfolio = bool(realized_row)
        written_to_realized = bool(realized_row and realized_row.get("position_pnl_return_pct", "") != "")
        exit_date = realized_row.get("exit_date", "") or next_review_dates.get(date, "")
        exit_price_found: bool | str = True if written_to_realized else "unknown_without_realized_row"

        if written_to_realized:
            likely_drop_reason = "survived_to_realized"
        elif audit_drop_reason and audit_drop_reason != "survived_to_execution":
            likely_drop_reason = audit_drop_reason
        elif coverage_reason:
            likely_drop_reason = coverage_reason
        elif has_composite_file is False:
            likely_drop_reason = "missing_composite_file"
        elif has_composite_row is False:
            likely_drop_reason = "missing_composite_row"
        elif russell_member is False:
            likely_drop_reason = "not_russell_member"
        elif max_realized_date is not None and pd.Timestamp(date) > max_realized_date:
            likely_drop_reason = "missing_exit_window_final_rebalance"
        else:
            likely_drop_reason = "cache_valid_but_missing_from_realized_attribution"

        rows.append(
            {
                "date": date,
                "side": side,
                "ticker": ticker,
                "cache_selected": True,
                "llm_rank": raw_row.get("rank", ""),
                "intended_weight": raw_row.get("weight", ""),
                "composite_file_exists": has_composite_file,
                "composite_row_exists": has_composite_row,
                "loaded_by_factor_engine": loaded_by_factor_engine,
                "russell_member_at_trade_date": russell_member,
                "trade_price_found": trade_price_found,
                "trade_price": audit_row.get("trade_price", ""),
                "min_price_pass": min_price_pass,
                "trail_return_found": "cache_implied_true",
                "entry_price_found": trade_price_found,
                "exit_date": exit_date,
                "exit_price_found": exit_price_found,
                "exit_price": "",
                "assigned_weight": realized_row.get("weight", ""),
                "written_to_portfolio_selections": written_to_portfolio,
                "written_to_realized_attribution": written_to_realized,
                "in_det_top50": bool(baseline_row),
                "det_rank": baseline_row.get("rank", ""),
                "det_position_pnl_return_pct": baseline_row.get("position_pnl_return_pct", ""),
                "prompt_prefilter_pass": "cache_implied_true",
                "measurable_in_backtest": written_to_realized,
                "likely_drop_reason": likely_drop_reason,
            }
        )
    return rows


def _build_validity_audit(
    raw_llm_detail: dict[tuple[str, str, str], dict[str, Any]],
    realized_llm_detail: dict[tuple[str, str, str], dict[str, Any]],
    baseline_detail: dict[tuple[str, str, str], dict[str, Any]],
    review_log_path: Path | None,
    composite_dir: Path | None,
    membership_path: Path | None,
    validity_mode: str = "cache_fast",
    selection_audit_detail: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    composite_coverage_detail: dict[tuple[str, str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if validity_mode == "cache_fast":
        return _build_fast_validity_audit(
            raw_llm_detail,
            realized_llm_detail,
            baseline_detail,
            review_log_path,
            membership_path,
            selection_audit_detail,
            composite_coverage_detail,
        )
    if validity_mode != "exact":
        raise ValueError(f"Unknown validity audit mode: {validity_mode}")

    composite_index = _build_composite_file_index(composite_dir)
    composite_cache: dict[str, pd.DataFrame | None] = {}
    membership_dates, membership_by_date = _load_membership_lookup(
        membership_path,
        {key[0] for key in raw_llm_detail},
    )
    next_review_dates = _load_next_review_dates(review_log_path)

    rows: list[dict[str, Any]] = []
    for key in sorted(raw_llm_detail):
        date, side, ticker = key
        raw_row = raw_llm_detail[key]
        realized_row = realized_llm_detail.get(key, {})
        baseline_row = baseline_detail.get(key, {})
        composite_file_exists = ticker in composite_index
        composite_df = _load_clean_composite_frame(ticker, composite_index, composite_cache)
        loaded_by_factor_engine = composite_df is not None and not composite_df.empty

        signal_row = _exact_composite_row(composite_df, date)
        composite_row_exists = signal_row is not None
        trade_row = _ffill_composite_row(composite_df, date)
        trade_price = _row_price(trade_row)
        trade_price_found = trade_price is not None and pd.notna(trade_price) and trade_price > 0
        min_price_pass = has_min_trade_price(trade_price)

        exit_date = next_review_dates.get(date, "")
        exit_row = _ffill_composite_row(composite_df, exit_date) if exit_date else None
        exit_price = _row_price(exit_row)
        exit_price_found = exit_price is not None and pd.notna(exit_price) and exit_price > 0

        trail_return_found = False
        if composite_df is not None and signal_row is not None:
            signal_idx = int(signal_row.name)
            if signal_idx >= 5:
                prior_price = _row_price(composite_df.iloc[signal_idx - 5])
                signal_price = _row_price(signal_row)
                trail_return_found = (
                    prior_price is not None
                    and signal_price is not None
                    and prior_price > 0
                    and signal_price > 0
                )

        russell_member = _is_member_on_or_before(
            ticker,
            date,
            membership_dates,
            membership_by_date,
        )
        prompt_prefilter_pass = bool(
            composite_row_exists
            and loaded_by_factor_engine
            and russell_member is not False
            and trade_price_found
            and min_price_pass
            and trail_return_found
        )
        measurable_in_backtest = bool(prompt_prefilter_pass and exit_price_found)
        written_to_portfolio = bool(realized_row)
        written_to_realized = bool(realized_row and realized_row.get("position_pnl_return_pct", "") != "")

        likely_drop_reason = ""
        if written_to_realized:
            likely_drop_reason = "survived_to_realized"
        elif not composite_file_exists:
            likely_drop_reason = "missing_composite_file"
        elif not loaded_by_factor_engine:
            likely_drop_reason = "not_loaded_after_symbol_hygiene"
        elif not composite_row_exists:
            likely_drop_reason = "missing_composite_row"
        elif russell_member is False:
            likely_drop_reason = "not_russell_member"
        elif not trade_price_found:
            likely_drop_reason = "missing_trade_price"
        elif not min_price_pass:
            likely_drop_reason = "below_min_price"
        elif not trail_return_found:
            likely_drop_reason = "missing_trailing_return"
        elif not exit_price_found:
            likely_drop_reason = "missing_entry_or_exit_price"
        else:
            likely_drop_reason = "eligible_but_missing_from_realized_attribution"

        rows.append(
            {
                "date": date,
                "side": side,
                "ticker": ticker,
                "cache_selected": True,
                "llm_rank": raw_row.get("rank", ""),
                "intended_weight": raw_row.get("weight", ""),
                "composite_file_exists": composite_file_exists,
                "composite_row_exists": composite_row_exists,
                "loaded_by_factor_engine": loaded_by_factor_engine,
                "russell_member_at_trade_date": russell_member,
                "trade_price_found": trade_price_found,
                "trade_price": trade_price if trade_price is not None else "",
                "min_price_pass": min_price_pass,
                "trail_return_found": trail_return_found,
                "entry_price_found": trade_price_found,
                "exit_date": exit_date,
                "exit_price_found": exit_price_found,
                "exit_price": exit_price if exit_price is not None else "",
                "assigned_weight": realized_row.get("weight", ""),
                "written_to_portfolio_selections": written_to_portfolio,
                "written_to_realized_attribution": written_to_realized,
                "in_det_top50": bool(baseline_row),
                "det_rank": baseline_row.get("rank", ""),
                "det_position_pnl_return_pct": baseline_row.get("position_pnl_return_pct", ""),
                "prompt_prefilter_pass": prompt_prefilter_pass,
                "measurable_in_backtest": measurable_in_backtest,
                "likely_drop_reason": likely_drop_reason,
            }
        )
    return rows


def _html_value(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if value is None:
        return ""
    return html.escape(str(value))


def _html_table(rows: list[dict[str, Any]], fieldnames: list[str], limit: int | None = None) -> str:
    displayed = rows if limit is None else rows[:limit]
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fieldnames)
    body = []
    for row in displayed:
        cells = "".join(f"<td>{_html_value(row.get(field, ''))}</td>" for field in fieldnames)
        body.append(f"<tr>{cells}</tr>")
    if not body:
        body.append(f"<tr><td colspan=\"{len(fieldnames)}\" class=\"muted\">No rows</td></tr>")
    suffix = ""
    if limit is not None and len(rows) > limit:
        suffix = (
            f"<p class=\"muted\">Showing {limit:,} of {len(rows):,} rows. "
            "Use the CSV files for the full table.</p>"
        )
    return f"<div class=\"table-wrap\"><table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>{suffix}"


def _render_html_report(
    cache_path: Path,
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    anomaly_rows: list[dict[str, Any]],
    raw_sections: list[dict[str, Any]],
    raw_baseline_rows: list[dict[str, Any]],
    raw_baseline_summary_rows: list[dict[str, Any]],
    decision_flow_rows: list[dict[str, Any]],
    validity_audit_rows: list[dict[str, Any]],
    survival_rows: list[dict[str, Any]],
    survival_summary_rows: list[dict[str, Any]],
    baseline_comparison_rows: list[dict[str, Any]],
    baseline_summary_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    has_performance: bool,
) -> str:
    total_dates = len(summary_rows)
    trace_dates = sum(1 for row in summary_rows if row.get("has_trace"))
    total_longs = sum(int(row.get("n_long_basket") or 0) for row in summary_rows)
    total_shorts = sum(int(row.get("n_short_basket") or 0) for row in summary_rows)
    avg_longs = total_longs / total_dates if total_dates else 0.0
    avg_shorts = total_shorts / total_dates if total_dates else 0.0
    high_anomalies = sum(1 for row in anomaly_rows if row.get("severity") == "high")
    medium_anomalies = sum(1 for row in anomaly_rows if row.get("severity") == "medium")
    avg_period_net = 0.0
    perf_rows = [row for row in summary_rows if row.get("period_net_return_pct") not in ("", None)]
    if perf_rows:
        avg_period_net = sum(float(row.get("period_net_return_pct") or 0.0) for row in perf_rows) / len(perf_rows)

    summary_fields = [
        "date",
        "regime",
        "period_net_return_pct",
        "period_gross_return_pct",
        "period_long_pnl_bps",
        "period_short_pnl_bps",
        "period_cost_bps",
        "n_long_basket",
        "n_short_basket",
        "n_long_candidates",
        "n_short_candidates",
        "has_trace",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning",
    ]
    selection_fields = [
        "date",
        "side",
        "rank",
        "ticker",
        "weight",
        "alpha_edge_score",
        "rationale",
    ]
    candidate_fields = [
        "date",
        "side",
        "rank",
        "ticker",
        "edge_score",
        "selected",
        "rationale",
    ]
    baseline_summary_fields = [
        "date",
        "side",
        "n_kept_vs_baseline",
        "n_added_by_llm",
        "n_dropped_by_llm",
        "kept_contribution_pct",
        "added_contribution_pct",
        "dropped_contribution_pct",
        "llm_minus_baseline_selection_contribution_pct",
    ]
    baseline_comparison_fields = [
        "date",
        "side",
        "ticker",
        "decision",
        "return_source",
        "return_format",
        "llm_rank",
        "baseline_rank",
        "realized_return_pct",
        "raw_or_source_return_pct",
        "signed_return_pct",
        "position_pnl_return_pct",
        "contribution_pct",
    ]
    raw_baseline_summary_fields = [
        "date",
        "side",
        "raw_llm_selected",
        "baseline_selected",
        "n_kept_vs_baseline_raw",
        "n_added_by_llm_raw",
        "n_dropped_by_llm_raw",
    ]
    raw_baseline_fields = [
        "date",
        "side",
        "ticker",
        "decision",
        "raw_llm_rank",
        "baseline_rank",
        "raw_llm_weight",
        "baseline_weight",
    ]
    decision_flow_fields = [
        "date",
        "side",
        "prompt_bucket",
        "ticker",
        "candidate_rank",
        "candidate_edge_score",
        "llm_selected",
        "llm_rank",
        "llm_raw_weight",
        "survived_to_realized",
        "drop_reason",
        "realized_weight",
        "position_pnl_return_pct",
        "in_det_top50",
        "det_rank",
        "det_position_pnl_return_pct",
    ]
    validity_audit_fields = [
        "date",
        "side",
        "ticker",
        "cache_selected",
        "llm_rank",
        "intended_weight",
        "composite_file_exists",
        "composite_row_exists",
        "loaded_by_factor_engine",
        "russell_member_at_trade_date",
        "trade_price_found",
        "trade_price",
        "min_price_pass",
        "trail_return_found",
        "entry_price_found",
        "exit_date",
        "exit_price_found",
        "exit_price",
        "assigned_weight",
        "written_to_portfolio_selections",
        "written_to_realized_attribution",
        "in_det_top50",
        "det_rank",
        "det_position_pnl_return_pct",
        "prompt_prefilter_pass",
        "measurable_in_backtest",
        "likely_drop_reason",
    ]
    survival_summary_fields = [
        "date",
        "side",
        "raw_selected_count",
        "realized_survivor_count",
        "missing_from_realized_count",
        "missing_from_realized_pct",
    ]
    survival_fields = [
        "date",
        "side",
        "ticker",
        "raw_rank",
        "raw_weight",
        "survived_to_realized",
        "drop_reason",
        "audit_survived_to_execution",
        "audit_trade_price",
        "composite_coverage_reason",
        "has_composite_file",
        "has_composite_row",
        "realized_weight",
        "realized_return_pct",
    ]

    raw_html = []
    for section in raw_sections[:50]:
        raw_payload = json.dumps(section.get("raw_payload", {}), indent=2, sort_keys=True)
        perf_label = section.get("perf_label") or ""
        raw_html.append(
            "<details>"
            f"<summary>{_html_value(section.get('date'))}: "
            f"{_html_value(section.get('title'))}"
            f"{' | ' + _html_value(perf_label) if perf_label else ''}</summary>"
            f"<pre>{html.escape(raw_payload[:60000])}</pre>"
            "</details>"
        )
    if len(raw_sections) > 50:
        raw_html.append(
            f"<p class=\"muted\">Showing 50 of {len(raw_sections):,} raw sections. "
            "Use raw_responses.md for the rest.</p>"
        )

    metadata_html = ""
    if metadata:
        metadata_html = (
            "<details><summary>Cache Metadata</summary>"
            f"<pre>{html.escape(json.dumps(metadata, indent=2, sort_keys=True)[:60000])}</pre>"
            "</details>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Cache Inspection</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #20242d;
      --muted: #657080;
      --line: #dce1ea;
      --accent: #1463ff;
      --warn: #a15c00;
      --bad: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; letter-spacing: 0; }}
    main {{ padding: 18px 32px 40px; }}
    .muted {{ color: var(--muted); }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 24px; font-weight: 650; margin-top: 4px; }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 900px; }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #eef2f8;
      font-size: 12px;
      color: #394150;
      z-index: 1;
    }}
    td {{ max-width: 460px; }}
    tr:last-child td {{ border-bottom: 0; }}
    details {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }}
    summary {{ cursor: pointer; font-weight: 650; }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #101828;
      color: #e6edf7;
      padding: 12px;
      border-radius: 6px;
      max-height: 640px;
      overflow: auto;
    }}
    .links a {{ color: var(--accent); margin-right: 14px; }}
  </style>
</head>
<body>
  <header>
    <h1>LLM Cache Inspection</h1>
    <div class="muted">Cache: {html.escape(str(cache_path))}</div>
    <div class="muted">Output: {html.escape(str(output_dir))}</div>
    <div class="links">
      <a href="cache_summary.csv">cache_summary.csv</a>
      <a href="selections.csv">selections.csv</a>
      <a href="signal_candidates.csv">signal_candidates.csv</a>
      <a href="llm_decision_flow.csv">llm_decision_flow.csv</a>
      <a href="llm_vs_det_validity_audit.csv">llm_vs_det_validity_audit.csv</a>
      <a href="raw_baseline_comparison_summary.csv">raw_baseline_comparison_summary.csv</a>
      <a href="raw_baseline_comparison.csv">raw_baseline_comparison.csv</a>
      <a href="selection_survival_summary.csv">selection_survival_summary.csv</a>
      <a href="selection_survival.csv">selection_survival.csv</a>
      <a href="baseline_comparison_summary.csv">baseline_comparison_summary.csv</a>
      <a href="baseline_comparison.csv">baseline_comparison.csv</a>
      <a href="anomalies.csv">anomalies.csv</a>
      <a href="raw_responses.md">raw_responses.md</a>
    </div>
    <div class="cards">
      <div class="card"><div class="label">Dates</div><div class="value">{total_dates:,}</div></div>
      <div class="card"><div class="label">Avg Longs</div><div class="value">{avg_longs:.1f}</div></div>
      <div class="card"><div class="label">Avg Shorts</div><div class="value">{avg_shorts:.1f}</div></div>
      <div class="card"><div class="label">Trace Dates</div><div class="value">{trace_dates:,}</div></div>
      <div class="card"><div class="label">Avg Period Net</div><div class="value">{avg_period_net:.2f}%</div></div>
      <div class="card"><div class="label">High Issues</div><div class="value">{high_anomalies:,}</div></div>
      <div class="card"><div class="label">Medium Issues</div><div class="value">{medium_anomalies:,}</div></div>
    </div>
  </header>
  <main>
    {metadata_html}
    {'<p class="muted">Performance columns are joined from the supplied backtest CSVs.</p>' if has_performance else '<p class="muted">No performance CSV supplied. Add --factor-pnl and/or --portfolio-selections to join realized returns by rebalance date.</p>'}
    <h2>Anomalies</h2>
    {_html_table(anomaly_rows, ["date", "severity", "issue", "detail"], limit=500)}
    <h2>Summary</h2>
    {_html_table(summary_rows, summary_fields, limit=500)}
    <h2>Selections</h2>
    <p class="muted">Mean-reversion LLM selections. Legacy engine-compatibility score columns are omitted here.</p>
    {_html_table(selection_rows, selection_fields, limit=1000)}
    <h2>Signal Candidates</h2>
    {_html_table(candidate_rows, candidate_fields, limit=1000)}
    <h2>LLM Decision Flow</h2>
    <p class="muted">Reconstructed from cache output: D1/D10 candidate scores → selected 50/50 → realized-attribution survival. Exact prompt text is unavailable for this run because log_llm_trace=false.</p>
    {_html_table(decision_flow_rows, decision_flow_fields, limit=1500)}
    <h2>LLM vs Det Validity Audit</h2>
    <p class="muted">One row per raw LLM pick. Default mode uses cache-selected names, realized exports, and the Russell membership file when available. Old runs without llm_selection_audit.csv still cannot prove every price/exit filter, but membership drops and final-window misses are now labeled directly.</p>
    {_html_table(validity_audit_rows, validity_audit_fields, limit=1500)}
    <h2>Raw LLM 50/50 vs Deterministic Baseline</h2>
    <p class="muted">Name overlap from the cache before downstream tradability/return filtering. This section is count-based, not return attribution.</p>
    {_html_table(raw_baseline_summary_rows, raw_baseline_summary_fields, limit=1000)}
    <h2>Raw Baseline Name-Level Compare</h2>
    {_html_table(raw_baseline_rows, raw_baseline_fields, limit=1500)}
    <h2>LLM Selection Survival</h2>
    <p class="muted">Raw cache-selected names versus names that survived into portfolio_selections.csv realized attribution.</p>
    {_html_table(survival_summary_rows, survival_summary_fields, limit=1000)}
    <h2>Selection Survival Name-Level</h2>
    {_html_table(survival_rows, survival_fields, limit=1500)}
    <h2>Executable/Realized LLM vs Deterministic Baseline</h2>
    <p class="muted">Return attribution using names that survived into portfolio_selections.csv.</p>
    {_html_table(baseline_summary_rows, baseline_summary_fields, limit=1000)}
    <h2>Executable/Realized Name-Level Compare</h2>
    {_html_table(baseline_comparison_rows, baseline_comparison_fields, limit=1500)}
    <h2>Raw Responses / Traces</h2>
    {''.join(raw_html) if raw_html else '<p class="muted">No raw responses found.</p>'}
  </main>
</body>
</html>
"""


def build_reports(
    cache_path: Path,
    output_dir: Path,
    focus_date: str | None = None,
    factor_pnl_path: Path | None = None,
    portfolio_selections_path: Path | None = None,
    baseline_selections_path: Path | None = None,
    selection_audit_path: Path | None = None,
    composite_coverage_path: Path | None = None,
    review_log_path: Path | None = None,
    composite_dir: Path | None = None,
    membership_path: Path | None = None,
    validity_mode: str = "cache_fast",
) -> None:
    cache = json.loads(cache_path.read_text())
    metadata = _as_dict(cache.get("metadata"))
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_performance = _load_factor_period_performance(factor_pnl_path)
    selection_performance = _load_selection_performance(portfolio_selections_path)
    llm_selection_detail = _load_selection_detail(portfolio_selections_path)
    baseline_selection_detail = _load_selection_detail(baseline_selections_path)
    if selection_audit_path is None and portfolio_selections_path is not None:
        inferred_audit_path = portfolio_selections_path.parent / "llm_selection_audit.csv"
        if inferred_audit_path.exists():
            selection_audit_path = inferred_audit_path
    if review_log_path is None and portfolio_selections_path is not None:
        inferred_review_log_path = portfolio_selections_path.parent / "review_log.csv"
        if inferred_review_log_path.exists():
            review_log_path = inferred_review_log_path
    if composite_dir is None:
        composite_dir = Path(metadata.get("composite_scores_dir") or DEFAULT_COMPOSITE_SCORES_DIR)
    if membership_path is None:
        membership_path = RUSSELL_MEMBERSHIP_PATH
    selection_audit_detail = _load_selection_audit(selection_audit_path)
    composite_coverage_detail = _load_composite_coverage(composite_coverage_path)

    summary_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    anomaly_rows: list[dict[str, Any]] = []
    raw_md: list[str] = [f"# LLM Cache Inspection\n\nCache: `{cache_path}`\n"]
    raw_sections: list[dict[str, Any]] = []

    if metadata:
        raw_md.append("\n## Metadata\n\n```json\n")
        raw_md.append(json.dumps(metadata, indent=2, sort_keys=True)[:12000])
        raw_md.append("\n```\n")

    portfolios = _iter_portfolios(cache)
    if focus_date:
        portfolios = [(date, row) for date, row in portfolios if date == focus_date]
    portfolio_dates = {date for date, _ in portfolios}
    raw_llm_selection_detail = _raw_cache_selection_detail(portfolios)
    candidate_detail = _cache_candidate_detail(portfolios)
    if portfolio_dates:
        llm_selection_detail = {
            key: row for key, row in llm_selection_detail.items() if key[0] in portfolio_dates
        }
        baseline_selection_detail = {
            key: row for key, row in baseline_selection_detail.items() if key[0] in portfolio_dates
        }
        selection_audit_detail = {
            key: row for key, row in selection_audit_detail.items() if key[0] in portfolio_dates
        }
        composite_coverage_detail = {
            key: row for key, row in composite_coverage_detail.items() if key[0] in portfolio_dates
        }
    raw_baseline_rows, raw_baseline_summary_rows = _summarize_raw_baseline_overlap(
        raw_llm_selection_detail,
        baseline_selection_detail,
    )
    survival_rows, survival_summary_rows = _summarize_survival(
        raw_llm_selection_detail,
        llm_selection_detail,
        selection_audit_detail,
        composite_coverage_detail,
    )
    decision_flow_rows = _build_decision_flow(
        candidate_detail,
        raw_llm_selection_detail,
        llm_selection_detail,
        baseline_selection_detail,
        survival_rows,
    )
    validity_audit_rows = _build_validity_audit(
        raw_llm_selection_detail,
        llm_selection_detail,
        baseline_selection_detail,
        review_log_path,
        composite_dir,
        membership_path,
        validity_mode=validity_mode,
        selection_audit_detail=selection_audit_detail,
        composite_coverage_detail=composite_coverage_detail,
    )
    baseline_comparison_rows, baseline_summary_rows = _summarize_baseline_comparison(
        llm_selection_detail,
        baseline_selection_detail,
    )

    for date, portfolio in portfolios:
        signal_stage = _as_dict(portfolio.get("signal_stage"))
        trace = _as_dict(portfolio.get("llm_trace"))
        raw_response = _as_dict(portfolio.get("raw_llm_response"))
        long_basket = _as_list(portfolio.get("long_basket"))
        short_basket = _as_list(portfolio.get("short_basket"))
        long_candidates = _as_list(signal_stage.get("long_candidates"))
        short_candidates = _as_list(signal_stage.get("short_candidates"))
        usage = _as_dict(portfolio.get("llm_usage"))
        long_tickers = _ticker_set(long_basket)
        short_tickers = _ticker_set(short_basket)
        overlap = sorted(long_tickers & short_tickers)
        perf = {
            **selection_performance.get(date, {}),
            **factor_performance.get(date, {}),
        }

        summary_rows.append(
            {
                "date": date,
                "regime": portfolio.get("regime", ""),
                "execution_mode": portfolio.get("execution_mode", ""),
                "turnover_aware": portfolio.get("turnover_aware", ""),
                "period_start": perf.get("period_start", ""),
                "period_end": perf.get("period_end", ""),
                "pnl_days": perf.get("pnl_days", ""),
                "period_net_return_pct": perf.get("period_net_return_pct", ""),
                "period_gross_return_pct": perf.get("period_gross_return_pct", ""),
                "period_long_pnl_bps": perf.get("period_long_pnl_bps", ""),
                "period_short_pnl_bps": perf.get("period_short_pnl_bps", ""),
                "period_cost_bps": perf.get("period_cost_bps", ""),
                "period_turnover": perf.get("period_turnover", ""),
                "selection_realized_return_pct": perf.get("selection_realized_return_pct", ""),
                "selection_long_realized_pct": perf.get("selection_long_realized_pct", ""),
                "selection_short_realized_pct": perf.get("selection_short_realized_pct", ""),
                "n_long_basket": len(long_basket),
                "n_short_basket": len(short_basket),
                "n_long_candidates": len(long_candidates),
                "n_short_candidates": len(short_candidates),
                "n_kept": len(_as_list(portfolio.get("kept_from_previous"))),
                "n_dropped": len(_as_list(portfolio.get("dropped_from_previous"))),
                "n_cross_leg_overlap": len(overlap),
                "overlap_tickers": " ".join(overlap),
                "has_trace": bool(trace),
                "prompt_tokens": _usage_field(usage, "prompt_tokens"),
                "completion_tokens": _usage_field(usage, "completion_tokens"),
                "total_tokens": _usage_field(usage, "total_tokens"),
                "reasoning_tokens": _as_dict(
                    _usage_field(usage, "completion_tokens_details")
                ).get("reasoning_tokens", ""),
                "reasoning": signal_stage.get("reasoning", portfolio.get("reasoning", "")),
                "regime_adjustment": signal_stage.get(
                    "regime_adjustment", portfolio.get("regime_adjustment", "")
                ),
            }
        )

        if overlap:
            anomaly_rows.append(
                {"date": date, "severity": "high", "issue": "ticker appears in both legs", "detail": " ".join(overlap)}
            )
        if len(long_basket) != len(short_basket):
            anomaly_rows.append(
                {
                    "date": date,
                    "severity": "medium",
                    "issue": "uneven long/short count",
                    "detail": f"long={len(long_basket)} short={len(short_basket)}",
                }
            )
        if not trace:
            anomaly_rows.append(
                {
                    "date": date,
                    "severity": "info",
                    "issue": "no prompt trace",
                    "detail": "rerun with --log-llm-trace to inspect exact input prompts",
                }
            )

        for side, rows in (("LONG", long_basket), ("SHORT", short_basket)):
            for rank, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    continue
                selection_rows.append(
                    {
                        "date": date,
                        "side": side,
                        "rank": rank,
                        "ticker": row.get("ticker", ""),
                        "weight": row.get("weight", row.get("weight_exec", "")),
                        "alpha_edge_score": _field(row, "alpha_edge_score", "edge_score"),
                        "direction_score": row.get("direction_score", ""),
                        "magnitude_score": row.get("magnitude_score", ""),
                        "selection_stage": row.get("selection_stage", ""),
                        "rationale": _field(row, "alpha_rationale", "rationale"),
                    }
                )

        for side, rows in (("LONG", long_candidates), ("SHORT", short_candidates)):
            for rank, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    continue
                candidate_rows.append(
                    {
                        "date": date,
                        "side": side,
                        "rank": row.get("rank", rank),
                        "ticker": row.get("ticker", ""),
                        "edge_score": row.get("edge_score", ""),
                        "utility": row.get("utility", ""),
                        "selected": row.get("ticker", "") in (long_tickers if side == "LONG" else short_tickers),
                        "rationale": row.get("rationale", ""),
                    }
                )

        raw_md.append(f"\n## {date}\n")
        raw_md.append(
            f"\nLongs: {len(long_basket)} | Shorts: {len(short_basket)} | "
            f"Signal candidates: {len(long_candidates)}/{len(short_candidates)} | "
            f"Trace: {bool(trace)}\n"
        )
        if perf:
            raw_md.append(
                "\nPerformance: "
                f"net {perf.get('period_net_return_pct', '')}% | "
                f"gross {perf.get('period_gross_return_pct', '')}% | "
                f"long {perf.get('period_long_pnl_bps', '')} bps | "
                f"short {perf.get('period_short_pnl_bps', '')} bps | "
                f"cost {perf.get('period_cost_bps', '')} bps\n"
            )
        if signal_stage.get("reasoning"):
            raw_md.append(f"\nReasoning: {signal_stage['reasoning']}\n")
        if signal_stage.get("regime_adjustment"):
            raw_md.append(f"\nRegime adjustment: {signal_stage['regime_adjustment']}\n")
        if trace:
            raw_md.append("\n### Trace\n\n```json\n")
            raw_md.append(json.dumps(trace, indent=2, sort_keys=True)[:30000])
            raw_md.append("\n```\n")
            raw_sections.append(
                {
                    "date": date,
                    "title": "trace",
                    "perf_label": (
                        f"net {perf.get('period_net_return_pct', ''):.2f}%"
                        if isinstance(perf.get("period_net_return_pct"), float)
                        else ""
                    ),
                    "raw_payload": trace,
                }
            )
        elif raw_response:
            raw_md.append("\n### Raw Response\n\n```json\n")
            raw_md.append(json.dumps(raw_response, indent=2, sort_keys=True)[:30000])
            raw_md.append("\n```\n")
            raw_sections.append(
                {
                    "date": date,
                    "title": "raw response",
                    "perf_label": (
                        f"net {perf.get('period_net_return_pct', ''):.2f}%"
                        if isinstance(perf.get("period_net_return_pct"), float)
                        else ""
                    ),
                    "raw_payload": raw_response,
                }
            )

    _write_csv(
        output_dir / "cache_summary.csv",
        summary_rows,
        [
            "date",
            "regime",
            "execution_mode",
            "turnover_aware",
            "period_start",
            "period_end",
            "pnl_days",
            "period_net_return_pct",
            "period_gross_return_pct",
            "period_long_pnl_bps",
            "period_short_pnl_bps",
            "period_cost_bps",
            "period_turnover",
            "selection_realized_return_pct",
            "selection_long_realized_pct",
            "selection_short_realized_pct",
            "n_long_basket",
            "n_short_basket",
            "n_long_candidates",
            "n_short_candidates",
            "n_kept",
            "n_dropped",
            "n_cross_leg_overlap",
            "overlap_tickers",
            "has_trace",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
            "reasoning",
            "regime_adjustment",
        ],
    )
    _write_csv(
        output_dir / "selections.csv",
        selection_rows,
        [
            "date",
            "side",
            "rank",
            "ticker",
            "weight",
            "alpha_edge_score",
            "direction_score",
            "magnitude_score",
            "selection_stage",
            "rationale",
        ],
    )
    _write_csv(
        output_dir / "signal_candidates.csv",
        candidate_rows,
        ["date", "side", "rank", "ticker", "edge_score", "utility", "selected", "rationale"],
    )
    _write_csv(
        output_dir / "llm_decision_flow.csv",
        decision_flow_rows,
        [
            "date",
            "side",
            "prompt_bucket",
            "ticker",
            "candidate_rank",
            "candidate_edge_score",
            "llm_selected",
            "llm_rank",
            "llm_raw_weight",
            "survived_to_realized",
            "drop_reason",
            "realized_weight",
            "position_pnl_return_pct",
            "in_det_top50",
            "det_rank",
            "det_position_pnl_return_pct",
        ],
    )
    _write_csv(
        output_dir / "llm_vs_det_validity_audit.csv",
        validity_audit_rows,
        [
            "date",
            "side",
            "ticker",
            "cache_selected",
            "llm_rank",
            "intended_weight",
            "composite_file_exists",
            "composite_row_exists",
            "loaded_by_factor_engine",
            "russell_member_at_trade_date",
            "trade_price_found",
            "trade_price",
            "min_price_pass",
            "trail_return_found",
            "entry_price_found",
            "exit_date",
            "exit_price_found",
            "exit_price",
            "assigned_weight",
            "written_to_portfolio_selections",
            "written_to_realized_attribution",
            "in_det_top50",
            "det_rank",
            "det_position_pnl_return_pct",
            "prompt_prefilter_pass",
            "measurable_in_backtest",
            "likely_drop_reason",
        ],
    )
    _write_csv(
        output_dir / "baseline_comparison_summary.csv",
        baseline_summary_rows,
        [
            "date",
            "side",
            "n_kept_vs_baseline",
            "n_added_by_llm",
            "n_dropped_by_llm",
            "kept_contribution_pct",
            "added_contribution_pct",
            "dropped_contribution_pct",
            "llm_minus_baseline_selection_contribution_pct",
        ],
    )
    _write_csv(
        output_dir / "raw_baseline_comparison_summary.csv",
        raw_baseline_summary_rows,
        [
            "date",
            "side",
            "raw_llm_selected",
            "baseline_selected",
            "n_kept_vs_baseline_raw",
            "n_added_by_llm_raw",
            "n_dropped_by_llm_raw",
        ],
    )
    _write_csv(
        output_dir / "raw_baseline_comparison.csv",
        raw_baseline_rows,
        [
            "date",
            "side",
            "ticker",
            "decision",
            "raw_llm_rank",
            "baseline_rank",
            "raw_llm_weight",
            "baseline_weight",
            "raw_llm_score",
            "baseline_score",
            "baseline_realized_return_pct",
            "baseline_contribution_pct",
        ],
    )
    _write_csv(
        output_dir / "selection_survival_summary.csv",
        survival_summary_rows,
        [
            "date",
            "side",
            "raw_selected_count",
            "realized_survivor_count",
            "missing_from_realized_count",
            "missing_from_realized_pct",
        ],
    )
    _write_csv(
        output_dir / "selection_survival.csv",
        survival_rows,
        [
            "date",
            "side",
            "ticker",
            "raw_rank",
            "raw_weight",
            "survived_to_realized",
            "drop_reason",
            "audit_survived_to_execution",
            "audit_trade_price",
            "composite_coverage_reason",
            "has_composite_file",
            "has_composite_row",
            "realized_weight",
            "realized_return_pct",
        ],
    )
    _write_csv(
        output_dir / "baseline_comparison.csv",
        baseline_comparison_rows,
        [
            "date",
            "side",
            "ticker",
            "decision",
            "return_source",
            "return_format",
            "llm_rank",
            "baseline_rank",
            "llm_weight",
            "baseline_weight",
            "realized_return_pct",
            "raw_or_source_return_pct",
            "signed_return_pct",
            "position_pnl_return_pct",
            "contribution_pct",
            "entry_date",
            "exit_date",
        ],
    )
    _write_csv(output_dir / "anomalies.csv", anomaly_rows, ["date", "severity", "issue", "detail"])
    (output_dir / "raw_responses.md").write_text("".join(raw_md))
    html_report = _render_html_report(
        cache_path=cache_path,
        output_dir=output_dir,
        summary_rows=summary_rows,
        selection_rows=selection_rows,
        candidate_rows=candidate_rows,
        anomaly_rows=anomaly_rows,
        raw_sections=raw_sections,
        raw_baseline_rows=raw_baseline_rows,
        raw_baseline_summary_rows=raw_baseline_summary_rows,
        decision_flow_rows=decision_flow_rows,
        validity_audit_rows=validity_audit_rows,
        survival_rows=survival_rows,
        survival_summary_rows=survival_summary_rows,
        baseline_comparison_rows=baseline_comparison_rows,
        baseline_summary_rows=baseline_summary_rows,
        metadata=metadata,
        has_performance=bool(factor_performance or selection_performance),
    )
    (output_dir / "index.html").write_text(html_report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path, help="LLM portfolio cache JSON")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for audit outputs")
    parser.add_argument("--date", help="Optional rebalance date to inspect, e.g. 2020-03-20")
    parser.add_argument("--factor-pnl", type=Path, help="Optional factor_pnl.csv to join period returns")
    parser.add_argument(
        "--portfolio-selections",
        type=Path,
        help="Optional portfolio_selections.csv to join per-name realized returns",
    )
    parser.add_argument(
        "--baseline-selections",
        type=Path,
        help="Optional deterministic portfolio_selections.csv for LLM kept/dropped/added attribution",
    )
    parser.add_argument(
        "--selection-audit",
        type=Path,
        help="Optional llm_selection_audit.csv with execution survivor/drop reasons",
    )
    parser.add_argument(
        "--composite-coverage",
        type=Path,
        help="Optional llm_composite_coverage.csv from llm_composite_writer.py",
    )
    parser.add_argument(
        "--review-log",
        type=Path,
        help="Optional review_log.csv for mapping each signal date to its next exit date",
    )
    parser.add_argument(
        "--composite-dir",
        type=Path,
        help=f"Base composite score directory for validity audit (default: cache metadata or {DEFAULT_COMPOSITE_SCORES_DIR})",
    )
    parser.add_argument(
        "--membership-path",
        type=Path,
        help=f"Russell membership file for validity audit (default: {RUSSELL_MEMBERSHIP_PATH})",
    )
    parser.add_argument(
        "--validity-mode",
        choices=("cache_fast", "exact"),
        default="cache_fast",
        help=(
            "Validity audit mode. cache_fast uses cache-selected names plus realized exports; "
            "exact rereads composite/membership data and is much slower."
        ),
    )
    args = parser.parse_args()

    build_reports(
        args.cache,
        args.output_dir,
        args.date,
        factor_pnl_path=args.factor_pnl,
        portfolio_selections_path=args.portfolio_selections,
        baseline_selections_path=args.baseline_selections,
        selection_audit_path=args.selection_audit,
        composite_coverage_path=args.composite_coverage,
        review_log_path=args.review_log,
        composite_dir=args.composite_dir,
        membership_path=args.membership_path,
        validity_mode=args.validity_mode,
    )
    print(f"Wrote cache inspection reports to {args.output_dir}")


if __name__ == "__main__":
    main()
