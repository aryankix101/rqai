"""Build leakage-safe historical-analogue memory features for V2.5 prompts.

This module finds prior feature-neighbor outcomes, summarizes their historical
win/return evidence, and exposes the optional memory fields consumed by V2.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import mean_reversion_llm_v1 as v1


V25_MEMORY_COLUMNS = ("HistN", "HistWin", "HistRet")
V25_SIMILARITY_FEATURES = (
    "Trail5",
    "RSI5",
    "Rng55",
    "Vel/V",
    "%MA100",
    "52WH",
    "Mom3m",
    "AutoC",
    "ADX",
    "VolPctl",
)
V25_NEIGHBOR_COUNT = 20
V25_MAX_NEIGHBORS_PER_DATE = 4
V25_READY_MIN_NEIGHBORS = 10
V25_READY_MIN_DATES = 3
V25_FULL_MIN_NEIGHBORS = 20
V25_FULL_MIN_DATES = 5


def memory_spec() -> dict[str, Any]:
    return {
        "name": "v2.5_balanced_outcome_memory",
        "version": 1,
        "similarity_features": list(V25_SIMILARITY_FEATURES),
        "duplicate_features_excluded": ["MRL", "MRS", "MRSig"],
        "standardization": "past-only side-specific z-score; missing values use past-only median",
        "distance": "equal-weight Euclidean",
        "regime_policy": "not used",
        "neighbor_count": V25_NEIGHBOR_COUNT,
        "max_neighbors_per_rebalance_date": V25_MAX_NEIGHBORS_PER_DATE,
        "same_ticker_history": "excluded",
        "case_universe": "all prior candidates on the same side, selected and unselected",
        "outcome": "five-trading-day side-adjusted C2C return",
        "eligibility": "case exit_date <= current signal_date",
        "llm_fields": list(V25_MEMORY_COLUMNS),
        "hist_score_included": False,
    }


def memory_spec_hash() -> str:
    blob = json.dumps(memory_spec(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class MemoryCase:
    ticker: str
    side: str
    signal_date: pd.Timestamp
    exit_date: pd.Timestamp
    side_adjusted_return: float
    features: dict[str, float]


@dataclass(frozen=True)
class V25MemoryBuild:
    pools_by_key: dict[str, v1.CandidatePool]
    candidate_memory: pd.DataFrame
    neighbor_audit: pd.DataFrame
    case_history: pd.DataFrame
    coverage_by_date: pd.DataFrame
    spec_hash: str


def _finite_float(value: Any) -> float:
    numeric = v1._safe_float(value)
    return float(numeric) if pd.notna(numeric) and np.isfinite(numeric) else np.nan


def _c2c_return(
    close_df: pd.DataFrame,
    ticker: str,
    signal_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> float:
    if ticker not in close_df.columns or signal_date not in close_df.index or exit_date not in close_df.index:
        return np.nan
    start = _finite_float(close_df.at[signal_date, ticker])
    end = _finite_float(close_df.at[exit_date, ticker])
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0.0:
        return np.nan
    return float(end / start - 1.0)


def _case_rows(
    pools: list[v1.CandidatePool],
    close_df: pd.DataFrame,
) -> tuple[list[MemoryCase], pd.DataFrame]:
    cases: list[MemoryCase] = []
    rows: list[dict[str, Any]] = []
    for pool in pools:
        for side, tickers in (("LONG", pool.long_tickers), ("SHORT", pool.short_tickers)):
            for ticker in tickers:
                raw_return = _c2c_return(close_df, ticker, pool.signal_date, pool.exit_date)
                if not np.isfinite(raw_return):
                    continue
                side_return = float(raw_return if side == "LONG" else -raw_return)
                features = {
                    column: _finite_float(pool.feature_rows[ticker].get(column))
                    for column in V25_SIMILARITY_FEATURES
                }
                case = MemoryCase(
                    ticker=str(ticker).upper(),
                    side=side,
                    signal_date=pd.Timestamp(pool.signal_date),
                    exit_date=pd.Timestamp(pool.exit_date),
                    side_adjusted_return=side_return,
                    features=features,
                )
                cases.append(case)
                rows.append(
                    {
                        "ticker": case.ticker,
                        "side": side,
                        "signal_date": case.signal_date,
                        "exit_date": case.exit_date,
                        "raw_c2c_return_pct": raw_return * 100.0,
                        "side_adjusted_return_pct": side_return * 100.0,
                        **{f"feature_{key}": value for key, value in features.items()},
                    }
                )
    return cases, pd.DataFrame(rows)


def _standardized_case_and_query_matrices(
    query_rows: list[dict[str, Any]],
    eligible: list[MemoryCase],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [[case.features.get(column, np.nan) for column in V25_SIMILARITY_FEATURES] for case in eligible],
        dtype=float,
    )
    query_values = np.asarray(
        [[_finite_float(row.get(column)) for column in V25_SIMILARITY_FEATURES] for row in query_rows],
        dtype=float,
    )
    medians = np.zeros(matrix.shape[1], dtype=float)
    means = np.zeros(matrix.shape[1], dtype=float)
    stds = np.ones(matrix.shape[1], dtype=float)
    for idx in range(matrix.shape[1]):
        valid = matrix[:, idx][np.isfinite(matrix[:, idx])]
        if valid.size == 0:
            continue
        medians[idx] = float(np.median(valid))
        means[idx] = float(np.mean(valid))
        std = float(np.std(valid))
        if std >= 1e-8:
            stds[idx] = std
    matrix = np.where(np.isfinite(matrix), matrix, medians)
    query_values = np.where(np.isfinite(query_values), query_values, medians[None, :])
    standardized = (matrix - means) / stds
    standardized_query = (query_values - means) / stds
    return standardized, standardized_query


def _select_neighbors_for_side(
    query_tickers: list[str],
    feature_rows: dict[str, dict[str, Any]],
    side: str,
    signal_date: pd.Timestamp,
    cases: list[MemoryCase],
) -> dict[str, list[tuple[MemoryCase, float]]]:
    eligible = [
        case
        for case in cases
        if case.side == side
        and case.signal_date < signal_date
        and case.exit_date <= signal_date
    ]
    if not eligible:
        return {ticker: [] for ticker in query_tickers}
    case_matrix, query_matrix = _standardized_case_and_query_matrices(
        [feature_rows[ticker] for ticker in query_tickers],
        eligible,
    )
    distances = np.sqrt(
        np.mean(np.square(case_matrix[:, None, :] - query_matrix[None, :, :]), axis=2)
    )
    indices_by_date: dict[pd.Timestamp, np.ndarray] = {}
    for idx, case in enumerate(eligible):
        indices_by_date.setdefault(pd.Timestamp(case.signal_date), []).append(idx)
    indices_by_date = {
        date: np.asarray(indices, dtype=int) for date, indices in indices_by_date.items()
    }
    output: dict[str, list[tuple[MemoryCase, float]]] = {}
    for query_idx, ticker in enumerate(query_tickers):
        ticker_upper = str(ticker).upper()
        candidates: list[tuple[MemoryCase, float]] = []
        for date in sorted(indices_by_date):
            date_indices = indices_by_date[date]
            allowed = np.asarray(
                [idx for idx in date_indices if eligible[int(idx)].ticker != ticker_upper],
                dtype=int,
            )
            if allowed.size == 0:
                continue
            local_distances = distances[allowed, query_idx]
            keep_count = min(V25_MAX_NEIGHBORS_PER_DATE, int(allowed.size))
            if keep_count < allowed.size:
                positions = np.argpartition(local_distances, keep_count - 1)[:keep_count]
                allowed = allowed[positions]
            candidates.extend((eligible[int(idx)], float(distances[int(idx), query_idx])) for idx in allowed)
        candidates.sort(key=lambda item: (item[1], item[0].signal_date, item[0].ticker))
        output[ticker] = candidates[:V25_NEIGHBOR_COUNT]
    return output


def hash_memory_enriched_pool(
    base_universe_hash: str,
    long_tickers: list[str],
    short_tickers: list[str],
    feature_rows: dict[str, dict[str, Any]],
) -> str:
    payload = {
        "base_universe_hash": base_universe_hash,
        "memory_spec_hash": memory_spec_hash(),
        "long": [
            {"ticker": ticker, **{column: feature_rows[ticker].get(column) for column in V25_MEMORY_COLUMNS}}
            for ticker in long_tickers
        ],
        "short": [
            {"ticker": ticker, **{column: feature_rows[ticker].get(column) for column in V25_MEMORY_COLUMNS}}
            for ticker in short_tickers
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _coverage_by_date(candidate_memory: pd.DataFrame) -> pd.DataFrame:
    if candidate_memory.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "candidate_count",
                "min_hist_n",
                "min_hist_date_n",
                "memory_ready",
                "fully_populated",
            ]
        )
    rows: list[dict[str, Any]] = []
    for signal_date, group in candidate_memory.groupby("signal_date", sort=True):
        min_hist_n = int(pd.to_numeric(group["hist_n"], errors="coerce").min())
        min_hist_dates = int(pd.to_numeric(group["hist_date_n"], errors="coerce").min())
        rows.append(
            {
                "signal_date": pd.Timestamp(signal_date),
                "candidate_count": int(len(group)),
                "min_hist_n": min_hist_n,
                "min_hist_date_n": min_hist_dates,
                "memory_ready": bool(
                    min_hist_n >= V25_READY_MIN_NEIGHBORS and min_hist_dates >= V25_READY_MIN_DATES
                ),
                "fully_populated": bool(
                    min_hist_n >= V25_FULL_MIN_NEIGHBORS and min_hist_dates >= V25_FULL_MIN_DATES
                ),
            }
        )
    return pd.DataFrame(rows)


def build_balanced_outcome_memory(
    pools_by_key: dict[str, v1.CandidatePool],
    close_df: pd.DataFrame,
) -> V25MemoryBuild:
    ordered_items = sorted(
        pools_by_key.items(),
        key=lambda item: (pd.Timestamp(item[1].signal_date), item[0]),
    )
    signal_dates = [pd.Timestamp(pool.signal_date) for _, pool in ordered_items]
    if len(signal_dates) != len(set(signal_dates)):
        raise ValueError("V2.5 memory requires exactly one source pool per signal date")
    pools = [pool for _, pool in ordered_items]
    cases, case_history = _case_rows(pools, close_df)
    enriched: dict[str, v1.CandidatePool] = {}
    candidate_rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []

    for storage_key, pool in ordered_items:
        feature_rows = {ticker: dict(row) for ticker, row in pool.feature_rows.items()}
        for side, tickers in (("LONG", pool.long_tickers), ("SHORT", pool.short_tickers)):
            neighbors_by_ticker = _select_neighbors_for_side(
                query_tickers=tickers,
                feature_rows=feature_rows,
                side=side,
                signal_date=pd.Timestamp(pool.signal_date),
                cases=cases,
            )
            for ticker in tickers:
                neighbors = neighbors_by_ticker[ticker]
                returns = np.asarray([case.side_adjusted_return for case, _ in neighbors], dtype=float)
                hist_n = int(len(neighbors))
                hist_win = float(np.mean(returns > 0.0) * 100.0) if hist_n else np.nan
                hist_ret = float(np.mean(returns) * 100.0) if hist_n else np.nan
                hist_dates = {pd.Timestamp(case.signal_date) for case, _ in neighbors}
                max_exit = max((case.exit_date for case, _ in neighbors), default=pd.NaT)
                feature_rows[ticker].update(
                    {
                        "HistN": hist_n,
                        "HistWin": hist_win,
                        "HistRet": hist_ret,
                    }
                )
                candidate_rows.append(
                    {
                        "storage_key": storage_key,
                        "signal_date": pd.Timestamp(pool.signal_date),
                        "ticker": str(ticker).upper(),
                        "side": side,
                        "hist_n": hist_n,
                        "hist_win_pct": hist_win,
                        "hist_ret_pct": hist_ret,
                        "hist_date_n": int(len(hist_dates)),
                        "max_analog_exit_date": max_exit,
                        "memory_ready": bool(
                            hist_n >= V25_READY_MIN_NEIGHBORS
                            and len(hist_dates) >= V25_READY_MIN_DATES
                        ),
                        "fully_populated": bool(
                            hist_n >= V25_FULL_MIN_NEIGHBORS
                            and len(hist_dates) >= V25_FULL_MIN_DATES
                        ),
                    }
                )
                for rank, (case, distance) in enumerate(neighbors, start=1):
                    neighbor_rows.append(
                        {
                            "storage_key": storage_key,
                            "query_signal_date": pd.Timestamp(pool.signal_date),
                            "query_ticker": str(ticker).upper(),
                            "side": side,
                            "neighbor_rank": rank,
                            "neighbor_ticker": case.ticker,
                            "neighbor_signal_date": case.signal_date,
                            "neighbor_exit_date": case.exit_date,
                            "distance": distance,
                            "side_adjusted_return_pct": case.side_adjusted_return * 100.0,
                        }
                    )
        universe_hash = hash_memory_enriched_pool(
            pool.universe_hash,
            pool.long_tickers,
            pool.short_tickers,
            feature_rows,
        )
        enriched[storage_key] = replace(pool, feature_rows=feature_rows, universe_hash=universe_hash)

    candidate_memory = pd.DataFrame(candidate_rows)
    neighbor_audit = pd.DataFrame(neighbor_rows)
    return V25MemoryBuild(
        pools_by_key=enriched,
        candidate_memory=candidate_memory,
        neighbor_audit=neighbor_audit,
        case_history=case_history,
        coverage_by_date=_coverage_by_date(candidate_memory),
        spec_hash=memory_spec_hash(),
    )


def save_memory_artifacts(build: V25MemoryBuild, output_dir: Path) -> Path:
    target = Path(output_dir) / "v25_balanced_outcome_memory"
    target.mkdir(parents=True, exist_ok=True)
    build.candidate_memory.to_csv(target / "candidate_memory.csv", index=False)
    build.neighbor_audit.to_csv(target / "neighbor_audit.csv", index=False)
    build.case_history.to_csv(target / "case_history.csv", index=False)
    build.coverage_by_date.to_csv(target / "coverage_by_date.csv", index=False)
    (target / "memory_spec.json").write_text(
        json.dumps({**memory_spec(), "spec_hash": build.spec_hash}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target
