"""
Offline autoresearch sweep for V2 mean-reversion LLM caches.

This script is intentionally zero-token: it reads completed V2/V2.1/V2.2
signal_checks artifacts, creates many ranking/weighting/hybrid variants, and
scores them under the same gross fixed 5-trading-day C2C/O2O accounting.
"""

from __future__ import annotations

import argparse
import difflib
import fcntl
import hashlib
import html
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import httpx
import numpy as np
import pandas as pd

import technical_llm


DEFAULT_V2_ROOT = Path("backtest_results/llm_reversal_signal_v2_2009_2014")
DEFAULT_V2_ANON_ROOT = Path("backtest_results/llm_reversal_signal_v2_anon_control_2009_2014")
DEFAULT_V2_STRUCTURED_ROOT = Path("backtest_results/llm_reversal_signal_v2_anon_structured_2009_2014")
DEFAULT_V21_ROOT = Path("backtest_results/llm_reversal_signal_v2_1_anon_2009_2014")
DEFAULT_V22_ROOT = Path("backtest_results/llm_reversal_signal_v2_2_visible_2009_2014")
DEFAULT_V23_ROOT = Path("backtest_results/llm_reversal_signal_v2_3_anon_2009_2014")
DEFAULT_OUTPUT_DIR = DEFAULT_V2_ROOT / "autoresearch_v2_cache_sweep"
DEFAULT_SIDE_HYBRID_DIR = DEFAULT_V22_ROOT / "side_process_hybrid_experiments_pool60_pos30_anchor3"
DEFAULT_V24_V25_FOLLOWUP_DIR = DEFAULT_OUTPUT_DIR / "v24_v25_followup"
SPY_SIGNAL_DATA_PATH = Path("backtest_results/signal_data_dump/signal_data_dump_SPY_1993-01-01_2025-12-31.csv")
RUSSELL_WIDE_PATH = Path("RusselData/russell_data_loader.parquet")

PERIODS_PER_YEAR = 52.0
CAPITAL_SCALE = 0.5
FEATURE_PREFIX = "feature_"
TOP10_EXP_HALF_LIVES = (1, 2, 3, 5, 10)
SHORT_RANK_NULL_SIMS = 20_000
TOP10_EXP_SCORE_LABELS = {
    "score_trail5": "Trail5",
    "score_v2_visible": "Ticker-visible V2",
    "score_v2_anon": "Anonymized V2",
    "score_v2_structured": "Structured V2",
    "score_v21_anon": "Anonymized V2.1",
    "score_v22_lam2": "V2.2 lambda 2",
    "score_v22_lam4": "V2.2 lambda 4",
    "score_v22_lam6": "V2.2 lambda 6",
    "score_v22_lam8": "V2.2 lambda 8",
    "score_v23_anon": "Anonymized V2.3",
}


@dataclass(frozen=True)
class ScoreSource:
    name: str
    root: Path
    run_dir: str
    score_col: str
    source_family: str
    description: str


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    score_col: str
    weight_model: str
    family: str
    description: str


def slug(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def fmt_num(value: Any, digits: int = 3) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return f"{float(numeric):.{digits}f}"


def fmt_pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    prefix = "+" if signed and float(numeric) > 0 else ""
    return f"{prefix}{float(numeric):.{digits}f}%"


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_cagr(returns: pd.Series, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    terminal = float((1.0 + clean).prod())
    if terminal <= 0.0:
        return np.nan
    return (terminal ** (periods_per_year / len(clean)) - 1.0) * 100.0


def max_drawdown_pct(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min() * 100.0)


def metric_values(returns: pd.Series, periods_per_year: float = PERIODS_PER_YEAR) -> dict[str, Any]:
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    ann_ret = float(clean.mean() * periods_per_year) if not clean.empty else np.nan
    ann_vol = float(clean.std(ddof=1) * np.sqrt(periods_per_year)) if len(clean) > 1 else np.nan
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    return {
        "n_periods": int(len(clean)),
        "annual_return_pct": ann_ret * 100.0 if np.isfinite(ann_ret) else np.nan,
        "annual_vol_pct": ann_vol * 100.0 if np.isfinite(ann_vol) else np.nan,
        "sharpe": sharpe,
        "total_return_pct": float(((1.0 + clean).prod() - 1.0) * 100.0) if not clean.empty else np.nan,
        "cagr_pct": safe_cagr(clean, periods_per_year),
        "max_drawdown_pct": max_drawdown_pct(clean),
        "hit_rate_pct": float((clean > 0).mean() * 100.0) if not clean.empty else np.nan,
    }


def weighted_turnover(old_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    if not old_weights:
        return 1.0 if new_weights else 0.0
    tickers = set(old_weights) | set(new_weights)
    return float(sum(abs(new_weights.get(ticker, 0.0) - old_weights.get(ticker, 0.0)) for ticker in tickers) / 2.0)


def signal_path(root: Path, anchor: str, run_dir: str, pool: int = 60, book: int = 30) -> Path:
    return root / f"pool{pool}_pos{book}" / f"anchor_{anchor}" / run_dir / "signal_checks.csv"


def weekly_path(root: Path, anchor: str, run_dir: str, pool: int = 60, book: int = 30) -> Path:
    return root / f"pool{pool}_pos{book}" / f"anchor_{anchor}" / run_dir / "weekly_returns.csv"


def load_base_panel(root: Path, anchor: str, pool: int = 60, book: int = 30) -> pd.DataFrame:
    path = signal_path(root, anchor, "baseline_pool", pool, book)
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline signal_checks: {path}")
    data = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", "pool_side", "signal", "trail_ret", "fwd_c2c", "fwd_o2o"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Baseline signal checks missing: {', '.join(sorted(missing))}")
    data["ticker"] = data["ticker"].astype(str)
    data["pool_side"] = data["pool_side"].astype(str).str.upper()
    data["signal"] = pd.to_numeric(data["signal"], errors="coerce")
    data["trail_ret"] = pd.to_numeric(data["trail_ret"], errors="coerce")
    data["fwd_c2c"] = pd.to_numeric(data["fwd_c2c"], errors="coerce")
    data["fwd_o2o"] = pd.to_numeric(data["fwd_o2o"], errors="coerce")
    data["score_trail5"] = np.where(data["pool_side"].eq("SHORT"), -data["signal"], data["signal"])
    weekly = weekly_path(root, anchor, "baseline_pool", pool, book)
    if weekly.exists():
        regimes = pd.read_csv(weekly, parse_dates=["date"])
        if "regime" in regimes.columns:
            data = data.merge(regimes[["date", "regime"]], on="date", how="left")
    if "regime" not in data.columns:
        data["regime"] = "unknown"
    return data


def merge_score_source(panel: pd.DataFrame, source: ScoreSource, anchor: str, pool: int = 60, book: int = 30) -> pd.DataFrame:
    path = signal_path(source.root, anchor, source.run_dir, pool, book)
    if not path.exists():
        print(f"Skipping missing source {source.name}: {path}")
        return panel
    data = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", "pool_side", source.score_col}
    missing = required - set(data.columns)
    if missing:
        print(f"Skipping source {source.name}; missing {sorted(missing)}")
        return panel
    data["ticker"] = data["ticker"].astype(str)
    data["pool_side"] = data["pool_side"].astype(str).str.upper()
    score_name = f"score_{source.name}"
    data[score_name] = pd.to_numeric(data[source.score_col], errors="coerce")
    if data[score_name].isna().all():
        print(f"Skipping source {source.name}; score is all NaN")
        return panel
    before = len(panel)
    merged = panel.merge(data[["date", "ticker", "pool_side", score_name]], on=["date", "ticker", "pool_side"], how="left")
    missing_count = int(merged[score_name].isna().sum())
    if missing_count:
        print(f"Source {source.name} has {missing_count}/{before} missing rows after merge")
    return merged


def default_sources() -> list[ScoreSource]:
    return [
        ScoreSource("v2_visible", DEFAULT_V2_ROOT, "llm_pool_v2", "llm_score", "V2 visible", "Original ticker-visible V2 score."),
        ScoreSource("v2_anon", DEFAULT_V2_ANON_ROOT, "llm_pool_v2", "llm_score", "V2 anonymized", "Same V2 score prompt with anonymized ticker IDs."),
        ScoreSource("v2_structured", DEFAULT_V2_STRUCTURED_ROOT, "llm_pool_v2", "llm_score", "V2 structured", "V2 with structured labels/features in the scoring call."),
        ScoreSource("v21_anon", DEFAULT_V21_ROOT, "llm_pool_v2", "llm_score", "V2.1 anonymized", "Discriminative/residual prompt variant."),
        ScoreSource("v22_lam2", DEFAULT_V22_ROOT, "llm_pool_v22_lambda2", "final_score", "V2.2", "Trail5 base plus 2x context adjustment."),
        ScoreSource("v22_lam4", DEFAULT_V22_ROOT, "llm_pool_v22_lambda4", "final_score", "V2.2", "Trail5 base plus 4x context adjustment."),
        ScoreSource("v22_lam6", DEFAULT_V22_ROOT, "llm_pool_v22_lambda6", "final_score", "V2.2", "Trail5 base plus 6x context adjustment."),
        ScoreSource("v22_lam8", DEFAULT_V22_ROOT, "llm_pool_v22_lambda8", "final_score", "V2.2", "Trail5 base plus 8x context adjustment."),
        ScoreSource("v22_adj", DEFAULT_V22_ROOT, "llm_pool_v22_lambda8", "context_adjustment", "V2.2", "Raw -3..+3 contextual adjustment."),
        ScoreSource("v23_anon", DEFAULT_V23_ROOT, "llm_pool_v23", "llm_score", "V2.3 anonymized", "Rank-only ordering prompt with anonymized ticker IDs."),
    ]


def add_rank_columns(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in panel.columns:
            continue
        panel[f"rank_{col}"] = panel.groupby(["date", "pool_side"])[col].rank(method="average", pct=True)
    return panel


def zscore_by_date_side(panel: pd.DataFrame, col: str) -> pd.Series:
    numeric = pd.to_numeric(panel[col], errors="coerce")

    def _z(group: pd.Series) -> pd.Series:
        std = float(group.std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            return pd.Series(0.0, index=group.index)
        return (group - float(group.mean())) / std

    return numeric.groupby([panel["date"], panel["pool_side"]]).transform(_z)


def add_derived_scores(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], list[str], list[str]]:
    descriptions: dict[str, str] = {
        "score_trail5": "Deterministic Trail5 side-relative baseline.",
    }
    base_score_cols = ["score_trail5"] + sorted(
        col for col in panel.columns if col.startswith("score_") and col != "score_trail5"
    )
    panel = add_rank_columns(panel, base_score_cols)

    score_cols = list(base_score_cols)
    core_cols = [
        col
        for col in [
            "score_trail5",
            "score_v2_visible",
            "score_v2_anon",
            "score_v2_structured",
            "score_v21_anon",
            "score_v22_lam2",
            "score_v22_lam4",
            "score_v22_lam6",
            "score_v22_lam8",
            "score_v23_anon",
        ]
        if col in panel.columns
    ]
    for col in core_cols:
        descriptions.setdefault(col, col.replace("score_", ""))

    blend_pairs = [
        ("score_v2_visible", "score_trail5"),
        ("score_v2_visible", "score_v2_anon"),
        ("score_v2_visible", "score_v2_structured"),
        ("score_v2_visible", "score_v21_anon"),
        ("score_v2_visible", "score_v22_lam8"),
        ("score_v2_anon", "score_v2_structured"),
    ]
    for left, right in blend_pairs:
        if left not in panel.columns or right not in panel.columns:
            continue
        for pct in range(0, 101, 10):
            weight = pct / 100.0
            name = f"score_blend_{slug(left)}_{slug(right)}_w{pct:03d}"
            panel[name] = weight * panel[f"rank_{left}"] + (1.0 - weight) * panel[f"rank_{right}"]
            score_cols.append(name)
            descriptions[name] = f"Rank blend: {weight:.1f} {left} + {1.0 - weight:.1f} {right}."

    if "score_v2_visible" in panel.columns and "score_v22_adj" in panel.columns:
        for gamma in (-20, -16, -12, -10, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20):
            name = f"score_v2_visible_plus_adj_{gamma:+d}".replace("+", "p").replace("-", "m")
            panel[name] = panel["score_v2_visible"] + float(gamma) * panel["score_v22_adj"]
            score_cols.append(name)
            descriptions[name] = f"Ticker-visible V2 score plus {gamma}x V2.2 context adjustment."

    feature_cols = [col for col in panel.columns if col.startswith(FEATURE_PREFIX)]
    feature_score_cols: list[str] = []
    for feature_col in feature_cols:
        clean = slug(feature_col.removeprefix(FEATURE_PREFIX))
        z_col = f"z_{clean}"
        panel[z_col] = zscore_by_date_side(panel, feature_col)
        if "rank_score_v2_visible" not in panel.columns:
            continue
        for coeff in (-0.30, -0.20, -0.10, 0.10, 0.20, 0.30):
            label = str(coeff).replace("-", "m").replace(".", "p")
            name = f"score_v2_feature_{clean}_{label}"
            panel[name] = panel["rank_score_v2_visible"] + coeff * panel[z_col]
            score_cols.append(name)
            feature_score_cols.append(name)
            descriptions[name] = f"V2 rank plus {coeff:+.2f}x within-date/side z-score of {feature_col}."

    return panel, descriptions, score_cols, core_cols


def parse_weight_model(model: str) -> tuple[str, float | int | tuple[int, float] | None]:
    if model.startswith("eq"):
        return "equal", int(model.removeprefix("eq"))
    if model == "lin60":
        return "linear", 60
    if model.startswith("exp") and "_h" in model:
        rank_limit, half_life = model.removeprefix("exp").split("_h", maxsplit=1)
        return "exp", (int(rank_limit), float(half_life))
    if model.startswith("reserve"):
        return "reserve", float(model.removeprefix("reserve")) / 100.0
    raise ValueError(f"Unknown weight model: {model}")


def weights_for_rank(rank: pd.Series, model: str) -> pd.Series:
    kind, value = parse_weight_model(model)
    ranks = pd.to_numeric(rank, errors="coerce")
    weights = pd.Series(0.0, index=rank.index)
    if kind == "equal":
        n = int(value or 0)
        mask = ranks.le(n)
        if mask.any():
            weights.loc[mask] = 1.0 / int(mask.sum())
        return weights
    if kind == "linear":
        max_rank = int(value or 60)
        mask = ranks.le(max_rank)
        raw = (max_rank + 1 - ranks.loc[mask]).clip(lower=0.0)
        weights.loc[mask] = raw / float(raw.sum())
        return weights
    if kind == "exp":
        rank_limit, half_life = value if isinstance(value, tuple) else (60, float(value or 10.0))
        mask = ranks.le(rank_limit)
        raw = np.exp(-(ranks.loc[mask] - 1.0) / half_life)
        weights.loc[mask] = raw / float(raw.sum())
        return weights
    if kind == "reserve":
        reserve = float(value or 0.0)
        top = ranks.le(30)
        bottom = ranks.gt(30) & ranks.le(60)
        if top.any():
            weights.loc[top] = (1.0 - reserve) / int(top.sum())
        if bottom.any() and reserve > 0:
            weights.loc[bottom] = reserve / int(bottom.sum())
        return weights
    return weights


def vector_weights_for_rank(dates: pd.Series, rank: pd.Series, model: str) -> pd.Series:
    kind, value = parse_weight_model(model)
    ranks = pd.to_numeric(rank, errors="coerce")
    raw = pd.Series(0.0, index=rank.index, dtype=float)
    if kind == "equal":
        n = int(value or 0)
        raw.loc[ranks.le(n)] = 1.0
    elif kind == "linear":
        max_rank = int(value or 60)
        mask = ranks.le(max_rank)
        raw.loc[mask] = (max_rank + 1 - ranks.loc[mask]).clip(lower=0.0)
    elif kind == "exp":
        rank_limit, half_life = value if isinstance(value, tuple) else (60, float(value or 10.0))
        mask = ranks.le(rank_limit)
        raw.loc[mask] = np.exp(-(ranks.loc[mask] - 1.0) / half_life)
    elif kind == "reserve":
        reserve = float(value or 0.0)
        raw.loc[ranks.le(30)] = 1.0 - reserve
        raw.loc[ranks.gt(30) & ranks.le(60)] = reserve
    else:
        raise ValueError(f"Unknown weight model: {model}")
    denom = raw.groupby(dates).transform("sum").replace(0.0, np.nan)
    return (raw / denom).fillna(0.0)


def build_process_specs(score_cols: list[str], core_cols: list[str], descriptions: dict[str, str]) -> list[ProcessSpec]:
    specs: list[ProcessSpec] = []
    core_set = set(core_cols)
    core_set.update(col for col in score_cols if col.startswith("score_v2_visible_plus_adj_"))
    core_set.update(
        col
        for col in score_cols
        if col.startswith("score_blend_score_v2_visible_score_trail5")
        or col.startswith("score_blend_score_v2_visible_score_v2_anon")
    )
    for col in score_cols:
        family = "feature_perturbation" if col.startswith("score_v2_feature_") else "score_rank"
        specs.append(
            ProcessSpec(
                name=f"{slug(col)}__eq30",
                score_col=col,
                weight_model="eq30",
                family=family,
                description=f"Top-30 equal-weight by {descriptions.get(col, col)}",
            )
        )
        if col in core_set:
            for n in (5, 10, 15, 20, 25, 35, 40, 50, 60):
                specs.append(
                    ProcessSpec(
                        name=f"{slug(col)}__eq{n}",
                        score_col=col,
                        weight_model=f"eq{n}",
                        family="concentration",
                        description=f"Top-{n} equal-weight by {descriptions.get(col, col)}",
                    )
                )
            for model in ("lin60", "exp60_h5", "exp60_h10", "exp60_h20", "exp60_h40"):
                specs.append(
                    ProcessSpec(
                        name=f"{slug(col)}__{model}",
                        score_col=col,
                        weight_model=model,
                        family="rank_weighting",
                        description=f"{model} weighting by {descriptions.get(col, col)}",
                    )
                )
            if col in TOP10_EXP_SCORE_LABELS:
                for half_life in TOP10_EXP_HALF_LIVES:
                    model = f"exp10_h{half_life}"
                    specs.append(
                        ProcessSpec(
                            name=f"{slug(col)}__{model}",
                            score_col=col,
                            weight_model=model,
                            family="truncated_exponential",
                            description=(
                                f"Top-10 only with exponential half-life {half_life} by "
                                f"{descriptions.get(col, col)}"
                            ),
                        )
                    )
            for reserve in (10, 20, 30, 40, 50):
                specs.append(
                    ProcessSpec(
                        name=f"{slug(col)}__reserve{reserve}",
                        score_col=col,
                        weight_model=f"reserve{reserve}",
                        family="reserve_sleeve",
                        description=f"Top-30 plus {reserve}% reserve to ranks 31-60 by {descriptions.get(col, col)}",
                    )
                )
    unique: dict[str, ProcessSpec] = {}
    for spec in specs:
        unique[spec.name] = spec
    return list(unique.values())


def compute_side_process(
    panel: pd.DataFrame,
    spec: ProcessSpec,
    side: str,
    return_selections: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    side_data = panel[panel["pool_side"].eq(side)].copy()
    if spec.score_col not in side_data.columns:
        raise ValueError(f"Missing score column {spec.score_col}")
    if side_data[spec.score_col].isna().any():
        side_data = side_data.dropna(subset=[spec.score_col])
    ordered = side_data.sort_values(["date", spec.score_col, "ticker"], ascending=[True, False, True], kind="mergesort").copy()
    ordered["rank"] = ordered.groupby("date").cumcount() + 1
    ordered["portfolio_weight"] = vector_weights_for_rank(ordered["date"], ordered["rank"], spec.weight_model)
    selected = ordered[ordered["portfolio_weight"].gt(0.0)].copy()
    selected["weighted_c2c"] = selected["portfolio_weight"] * selected["fwd_c2c"]
    selected["weighted_o2o"] = selected["portfolio_weight"] * selected["fwd_o2o"]
    selected["top30_weight_part"] = np.where(selected["rank"].le(30), selected["portfolio_weight"], 0.0)
    selected["ranks31_60_weight_part"] = np.where(selected["rank"].gt(30), selected["portfolio_weight"], 0.0)
    regime = selected.groupby("date")["regime"].first()
    weekly = (
        selected.groupby("date", as_index=False)
        .agg(
            raw_c2c=("weighted_c2c", "sum"),
            raw_o2o=("weighted_o2o", "sum"),
            n_selected=("ticker", "size"),
            avg_rank=("rank", "mean"),
            top30_weight=("top30_weight_part", "sum"),
            ranks31_60_weight=("ranks31_60_weight_part", "sum"),
        )
        .merge(regime.rename("regime"), on="date", how="left")
    )
    weekly["side"] = side
    weekly["process"] = spec.name
    weekly["score_col"] = spec.score_col
    weekly["weight_model"] = spec.weight_model
    weekly["side_pnl_c2c"] = weekly["raw_c2c"] if side == "LONG" else -weekly["raw_c2c"]
    weekly["side_pnl_o2o"] = weekly["raw_o2o"] if side == "LONG" else -weekly["raw_o2o"]
    weekly["turnover"] = np.nan
    selections = pd.DataFrame()
    if return_selections:
        keep_cols = [
            "date",
            "ticker",
            "pool_side",
            "rank",
            "portfolio_weight",
            spec.score_col,
            "trail_ret",
            "fwd_c2c",
            "fwd_o2o",
        ]
        selections = selected[keep_cols].rename(columns={spec.score_col: "score"}).copy()
        selections["side"] = side
        selections["process"] = spec.name
    return weekly, selections


def build_variant_specs(processes: list[ProcessSpec]) -> list[tuple[str, str, str, str]]:
    process_by_name = {spec.name: spec for spec in processes}
    v2_control = "score_v2_visible__eq30"
    if v2_control not in process_by_name:
        v2_control = "score_trail5__eq30"
    variants: dict[str, tuple[str, str, str, str]] = {}
    for spec in processes:
        variants[f"same::{spec.name}"] = (f"same::{spec.name}", spec.name, spec.name, "same_process")
        variants[f"long::{spec.name}|short::{v2_control}"] = (
            f"long::{spec.name}|short::{v2_control}",
            spec.name,
            v2_control,
            "one_side_long_vs_v2",
        )
        variants[f"long::{v2_control}|short::{spec.name}"] = (
            f"long::{v2_control}|short::{spec.name}",
            v2_control,
            spec.name,
            "one_side_short_vs_v2",
        )
    matrix_processes = [
        spec.name
        for spec in processes
        if spec.weight_model == "eq30"
        and (
            spec.score_col
            in {
                "score_trail5",
                "score_v2_visible",
                "score_v2_anon",
                "score_v2_structured",
                "score_v21_anon",
                "score_v22_lam2",
                "score_v22_lam4",
                "score_v22_lam6",
                "score_v22_lam8",
            }
            or spec.score_col.startswith("score_v2_visible_plus_adj_")
            or spec.score_col.startswith("score_blend_score_v2_visible_score_trail5")
        )
    ]
    for long_proc in matrix_processes:
        for short_proc in matrix_processes:
            variants[f"matrix::{long_proc}|{short_proc}"] = (
                f"matrix::{long_proc}|{short_proc}",
                long_proc,
                short_proc,
                "side_matrix",
            )
    return list(variants.values())


def build_scorecard(
    variants: list[tuple[str, str, str, str]],
    side_weekly: dict[tuple[str, str], pd.DataFrame],
    process_by_name: dict[str, ProcessSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    weekly_parts: list[pd.DataFrame] = []
    for run_name, long_proc, short_proc, family in variants:
        long_weekly = side_weekly.get((long_proc, "LONG"))
        short_weekly = side_weekly.get((short_proc, "SHORT"))
        if long_weekly is None or short_weekly is None or long_weekly.empty or short_weekly.empty:
            continue
        merged = long_weekly.rename(
            columns={
                "raw_c2c": "long_raw_c2c",
                "raw_o2o": "long_raw_o2o",
                "turnover": "long_turnover",
                "top30_weight": "long_top30_weight",
                "ranks31_60_weight": "long_ranks31_60_weight",
            }
        ).merge(
            short_weekly.rename(
                columns={
                    "raw_c2c": "short_raw_c2c",
                    "raw_o2o": "short_raw_o2o",
                    "turnover": "short_turnover",
                    "top30_weight": "short_top30_weight",
                    "ranks31_60_weight": "short_ranks31_60_weight",
                }
            )[
                [
                    "date",
                    "short_raw_c2c",
                    "short_raw_o2o",
                    "short_turnover",
                    "short_top30_weight",
                    "short_ranks31_60_weight",
                ]
            ],
            on="date",
            how="inner",
        )
        if merged.empty:
            continue
        merged["ls_c2c"] = merged["long_raw_c2c"] - merged["short_raw_c2c"]
        merged["ls_o2o"] = merged["long_raw_o2o"] - merged["short_raw_o2o"]
        merged["capital_c2c"] = CAPITAL_SCALE * merged["ls_c2c"]
        merged["capital_o2o"] = CAPITAL_SCALE * merged["ls_o2o"]
        metrics = metric_values(merged["capital_c2c"])
        row = {
            "run": run_name,
            "family": family,
            "long_process": long_proc,
            "short_process": short_proc,
            "long_score_col": process_by_name[long_proc].score_col,
            "short_score_col": process_by_name[short_proc].score_col,
            "long_weight_model": process_by_name[long_proc].weight_model,
            "short_weight_model": process_by_name[short_proc].weight_model,
            "long_description": process_by_name[long_proc].description,
            "short_description": process_by_name[short_proc].description,
            "avg_long_raw_c2c_pct": float(merged["long_raw_c2c"].mean() * 100.0),
            "avg_short_raw_c2c_pct": float(merged["short_raw_c2c"].mean() * 100.0),
            "avg_short_side_pnl_pct": float((-merged["short_raw_c2c"]).mean() * 100.0),
            "avg_turnover_long": float(merged["long_turnover"].mean()),
            "avg_turnover_short": float(merged["short_turnover"].mean()),
            "avg_turnover_total": float((merged["long_turnover"] + merged["short_turnover"]).mean()),
            "long_top30_weight": float(merged["long_top30_weight"].mean()),
            "long_ranks31_60_weight": float(merged["long_ranks31_60_weight"].mean()),
            "short_top30_weight": float(merged["short_top30_weight"].mean()),
            "short_ranks31_60_weight": float(merged["short_ranks31_60_weight"].mean()),
            **metrics,
        }
        rows.append(row)
        keep = merged[
            [
                "date",
                "long_raw_c2c",
                "short_raw_c2c",
                "ls_c2c",
                "capital_c2c",
                "long_raw_o2o",
                "short_raw_o2o",
                "ls_o2o",
                "capital_o2o",
                "regime",
            ]
        ].copy()
        keep["run"] = run_name
        keep["long_process"] = long_proc
        keep["short_process"] = short_proc
        weekly_parts.append(keep)
    scorecard = pd.DataFrame(rows)
    weekly_returns = pd.concat(weekly_parts, ignore_index=True) if weekly_parts else pd.DataFrame()
    if not scorecard.empty:
        controls = {
            "v2": ("score_v2_visible__eq30", "score_v2_visible__eq30"),
            "trail5": ("score_trail5__eq30", "score_trail5__eq30"),
            "v2_anon": ("score_v2_anon__eq30", "score_v2_anon__eq30"),
        }
        for label, (long_proc, short_proc) in controls.items():
            control = scorecard[scorecard["long_process"].eq(long_proc) & scorecard["short_process"].eq(short_proc)]
            if control.empty:
                continue
            cagr = float(control.iloc[0]["cagr_pct"])
            sharpe = float(control.iloc[0]["sharpe"])
            scorecard[f"cagr_lift_vs_{label}_pct"] = scorecard["cagr_pct"] - cagr
            scorecard[f"sharpe_lift_vs_{label}"] = scorecard["sharpe"] - sharpe
    return scorecard.sort_values(["cagr_pct", "sharpe"], ascending=[False, False]), weekly_returns


def side_process_summary(side_weekly: dict[tuple[str, str], pd.DataFrame], processes: list[ProcessSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    process_by_name = {spec.name: spec for spec in processes}
    for (process, side), data in side_weekly.items():
        if data.empty:
            continue
        pnl = pd.to_numeric(data["side_pnl_c2c"], errors="coerce")
        metrics = metric_values(pnl)
        rows.append(
            {
                "side": side,
                "process": process,
                "score_col": process_by_name[process].score_col,
                "weight_model": process_by_name[process].weight_model,
                "family": process_by_name[process].family,
                "avg_raw_stock_c2c_pct": float(data["raw_c2c"].mean() * 100.0),
                "avg_side_pnl_c2c_pct": float(data["side_pnl_c2c"].mean() * 100.0),
                "side_hit_rate_pct": float((data["side_pnl_c2c"] > 0).mean() * 100.0),
                "side_vol_pct": float(data["side_pnl_c2c"].std(ddof=1) * 100.0),
                **{f"side_{key}": value for key, value in metrics.items()},
            }
        )
    return pd.DataFrame(rows).sort_values(["side", "avg_side_pnl_c2c_pct"], ascending=[True, False])


def block_bootstrap(
    weekly_returns: pd.DataFrame,
    scorecard: pd.DataFrame,
    control_run: str,
    output_runs: list[str],
    n_boot: int = 3000,
    seed: int = 7,
) -> pd.DataFrame:
    if weekly_returns.empty or not output_runs:
        return pd.DataFrame()
    pivot = weekly_returns.pivot_table(index="date", columns="run", values="capital_c2c", aggfunc="first").sort_index()
    if control_run not in pivot.columns:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    n = len(pivot)
    rows: list[dict[str, Any]] = []
    for run in output_runs:
        if run not in pivot.columns or run == control_run:
            continue
        delta = (pivot[run] - pivot[control_run]).dropna().to_numpy()
        if len(delta) == 0:
            continue
        samples = rng.integers(0, len(delta), size=(n_boot, len(delta)))
        boot = delta[samples].mean(axis=1) * PERIODS_PER_YEAR * 100.0
        actual = float(delta.mean() * PERIODS_PER_YEAR * 100.0)
        p_two = float(2.0 * min((boot <= 0.0).mean(), (boot >= 0.0).mean()))
        rows.append(
            {
                "run": run,
                "mean_annualized_delta_pct": actual,
                "ci_low_pct": float(np.quantile(boot, 0.025)),
                "ci_high_pct": float(np.quantile(boot, 0.975)),
                "p_two_sided": min(1.0, p_two),
            }
        )
    bootstrap = pd.DataFrame(rows)
    if bootstrap.empty:
        return bootstrap
    keep_cols = ["run", "cagr_pct", "sharpe", "max_drawdown_pct", "cagr_lift_vs_v2_pct"]
    available = [col for col in keep_cols if col in scorecard.columns]
    return bootstrap.merge(scorecard[available], on="run", how="left").sort_values(
        ["mean_annualized_delta_pct", "cagr_pct"], ascending=[False, False]
    )


def year_regime_metrics(weekly_returns: pd.DataFrame, top_runs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if weekly_returns.empty or not top_runs:
        return pd.DataFrame(), pd.DataFrame()
    data = weekly_returns[weekly_returns["run"].isin(top_runs)].copy()
    data["year"] = pd.to_datetime(data["date"]).dt.year
    year_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for (run, year), group in data.groupby(["run", "year"], sort=True):
        row = {"run": run, "year": int(year), **metric_values(group["capital_c2c"])}
        year_rows.append(row)
    for (run, regime), group in data.groupby(["run", "regime"], sort=True):
        row = {"run": run, "regime": regime, **metric_values(group["capital_c2c"])}
        regime_rows.append(row)
    return pd.DataFrame(year_rows), pd.DataFrame(regime_rows)


def load_side_hybrid_sections(root: Path = DEFAULT_SIDE_HYBRID_DIR) -> list[dict[str, Any]]:
    section_specs = [
        {
            "title": "Prior Side-Process Hybrids",
            "description": (
                "All-vs-all cross of long-process choice and short-process choice from the earlier V2/V2.2 "
                "side-process experiment. This is included here because it is also zero-token cache research."
            ),
            "filename": "comparison_scorecard_expanded.csv",
            "columns": [
                "long_process",
                "short_process",
                "cagr_pct",
                "sharpe",
                "max_drawdown_pct",
                "cagr_lift_vs_v2_pct",
                "avg_long_raw_c2c_pct",
                "avg_short_side_pnl_pct",
            ],
            "sort": ["cagr_pct", "sharpe"],
            "ascending": [False, False],
            "limit": 20,
        },
        {
            "title": "Long Process Test, V2 Shorts Fixed",
            "description": (
                "Tests whether changing only the long-side process improves the original V2 book while the "
                "short book stays exactly V2."
            ),
            "filename": "long_process_test_with_v2_shorts.csv",
            "columns": [
                "long_process",
                "short_process",
                "cagr_pct",
                "sharpe",
                "max_drawdown_pct",
                "cagr_lift_vs_v2_pct",
                "avg_long_raw_c2c_pct",
                "avg_short_side_pnl_pct",
            ],
            "sort": ["cagr_pct", "sharpe"],
            "ascending": [False, False],
            "limit": 16,
        },
        {
            "title": "Short Process Test, V2 Longs Fixed",
            "description": (
                "Tests whether changing only the short-side process improves the original V2 book while the "
                "long book stays exactly V2. Short side P&L flips stock returns, so less negative is better."
            ),
            "filename": "short_process_test_with_v2_longs.csv",
            "columns": [
                "long_process",
                "short_process",
                "cagr_pct",
                "sharpe",
                "max_drawdown_pct",
                "cagr_lift_vs_v2_pct",
                "avg_long_raw_c2c_pct",
                "avg_short_side_pnl_pct",
            ],
            "sort": ["cagr_pct", "sharpe"],
            "ascending": [False, False],
            "limit": 16,
        },
        {
            "title": "Side-Only Process Quality",
            "description": (
                "Ranks each side process by its own side P&L. For SHORT rows, side P&L is already sign-adjusted."
            ),
            "filename": "side_process_summary_expanded.csv",
            "columns": ["side", "process", "family", "avg_side_pnl_c2c_pct", "side_hit_rate_pct", "side_vol_pct"],
            "sort": ["side", "avg_side_pnl_c2c_pct"],
            "ascending": [True, False],
            "limit": 80,
        },
    ]
    sections: list[dict[str, Any]] = []
    for spec in section_specs:
        path = root / str(spec["filename"])
        data = read_csv_if_exists(path)
        if data.empty:
            sections.append({**spec, "path": path, "data": pd.DataFrame()})
            continue
        sort_cols = [col for col in spec["sort"] if col in data.columns]
        if sort_cols:
            ascending = spec["ascending"][: len(sort_cols)]
            data = data.sort_values(sort_cols, ascending=ascending)
        sections.append({**spec, "path": path, "data": data})
    return sections


def load_v24_v25_followup_sections(root: Path = DEFAULT_V24_V25_FOLLOWUP_DIR) -> list[dict[str, Any]]:
    section_specs = [
        {
            "title": "V2.4 Weighting And Side Isolation",
            "description": (
                "Matched Trail5 versus V2.4 under top-10/20/30 equal weight and exponential h5/h10/h20, "
                "including one-side-at-a-time attribution. V2.4 remains exploratory, not a finalist."
            ),
            "filename": "v24_weighting_and_side_attribution.csv",
            "columns": [
                "construction",
                "long_ranker",
                "short_ranker",
                "cagr_pct",
                "sharpe",
                "max_drawdown_pct",
                "cagr_lift_vs_matched_trail5_pct",
                "sharpe_lift_vs_matched_trail5",
                "annualized_mean_delta_pct",
                "bootstrap_ci_low_pct",
                "bootstrap_ci_high_pct",
                "paired_win_rate_pct",
                "avg_long_raw_c2c_pct",
                "avg_short_side_pnl_pct",
            ],
            "sort": ["construction", "cagr_pct"],
            "ascending": [True, False],
            "limit": 30,
        },
        {
            "title": "V2.4 Standalone Long / Short Attribution",
            "description": (
                "Standalone side P&L by ranking and weighting. Short returns are sign-adjusted, so higher is better."
            ),
            "filename": "v24_side_only_attribution.csv",
            "columns": [
                "side",
                "process",
                "weight_model",
                "avg_side_pnl_c2c_pct",
                "side_cagr_pct",
                "side_sharpe",
                "side_max_drawdown_pct",
            ],
            "sort": ["side", "avg_side_pnl_c2c_pct"],
            "ascending": [True, False],
            "limit": 30,
        },
        {
            "title": "V2.5 Final Salvage: Memory Weighting, Side Hybrids, And Rank Blends",
            "description": (
                "Fresh V2 and V2.5 are compared under matched equal top-30 and exponential h5/h10/h20 weighting. "
                "The table also isolates each side and tests 25/50/75% V2.5 influence on long ranks with fresh "
                "V2 shorts fixed. All rows are post-hoc exploratory."
            ),
            "filename": "v25_side_hybrid_and_rank_blend_salvage.csv",
            "columns": [
                "construction",
                "run",
                "cagr_pct",
                "sharpe",
                "max_drawdown_pct",
                "cagr_lift_vs_fresh_v2_pct",
                "sharpe_lift_vs_fresh_v2",
                "annualized_mean_delta_pct",
                "bootstrap_ci_low_pct",
                "bootstrap_ci_high_pct",
                "paired_win_rate_pct",
                "avg_long_raw_c2c_pct",
                "avg_short_side_pnl_pct",
                "clear_improvement",
            ],
            "sort": ["cagr_pct", "sharpe"],
            "ascending": [False, False],
            "limit": 35,
        },
        {
            "title": "V2.5 Salvage Paired Bootstrap",
            "description": (
                "Rebalance-date bootstrap of each salvage row's weekly return difference versus fresh V2 both sides. "
                "A clear improvement requires the lower confidence bound above zero."
            ),
            "filename": "v25_salvage_paired_bootstrap.csv",
            "columns": [
                "run",
                "annualized_mean_delta_pct",
                "bootstrap_ci_low_pct",
                "bootstrap_ci_high_pct",
                "bootstrap_p_not_positive",
                "paired_win_rate_pct",
            ],
            "sort": ["annualized_mean_delta_pct"],
            "ascending": [False],
            "limit": 20,
        },
    ]
    sections: list[dict[str, Any]] = []
    for spec in section_specs:
        path = root / str(spec["filename"])
        data = read_csv_if_exists(path)
        sort_cols = [col for col in spec["sort"] if col in data.columns]
        if sort_cols:
            data = data.sort_values(sort_cols, ascending=spec["ascending"][: len(sort_cols)])
        sections.append({**spec, "path": path, "data": data})
    return sections


def build_exponential_2x2_rows(
    weekly_returns: pd.DataFrame,
    llm_run_prefix: str = "score_v2_visible",
    llm_label: str = "Ticker-visible LLM",
) -> pd.DataFrame:
    if weekly_returns.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    combos = [
        ("Trail5", "Trail5", "Pure deterministic control"),
        ("LLM", "LLM", "Full LLM ranking under the same exponential weighting"),
        ("LLM", "Trail5", "LLM value on longs only"),
        ("Trail5", "LLM", "LLM value on shorts only"),
    ]
    source_run = {
        "Trail5": "same::score_trail5__exp60_h{half_life}",
        "LLM": f"same::{llm_run_prefix}__exp60_h{{half_life}}",
    }
    for half_life in (5, 10, 20, 40):
        source_data: dict[str, pd.DataFrame] = {}
        for ranking, template in source_run.items():
            run = template.format(half_life=half_life)
            data = weekly_returns[weekly_returns["run"].eq(run)].copy()
            if data.empty:
                source_data = {}
                break
            source_data[ranking] = data
        if not source_data:
            continue
        baseline_cagr = np.nan
        combo_rows: list[dict[str, Any]] = []
        for long_ranking, short_ranking, question in combos:
            long_data = source_data[long_ranking][["date", "long_raw_c2c", "long_raw_o2o", "regime"]].copy()
            short_data = source_data[short_ranking][["date", "short_raw_c2c", "short_raw_o2o"]].copy()
            merged = long_data.merge(short_data, on="date", how="inner")
            if merged.empty:
                continue
            merged["capital_c2c"] = CAPITAL_SCALE * (merged["long_raw_c2c"] - merged["short_raw_c2c"])
            metrics = metric_values(merged["capital_c2c"])
            row = {
                "half_life": half_life,
                "llm_source": llm_label,
                "long_ranking": long_ranking,
                "short_ranking": short_ranking,
                "question": question,
                "avg_long_raw_c2c_pct": float(merged["long_raw_c2c"].mean() * 100.0),
                "avg_short_raw_c2c_pct": float(merged["short_raw_c2c"].mean() * 100.0),
                "avg_short_side_pnl_pct": float((-merged["short_raw_c2c"]).mean() * 100.0),
                **metrics,
            }
            if long_ranking == "Trail5" and short_ranking == "Trail5":
                baseline_cagr = float(metrics["cagr_pct"])
            combo_rows.append(row)
        for row in combo_rows:
            row["lift_vs_trail5_exp_pct"] = (
                float(row["cagr_pct"]) - baseline_cagr if np.isfinite(baseline_cagr) else np.nan
            )
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    data = pd.DataFrame(rows)
    return data.sort_values(["llm_source", "half_life", "cagr_pct"], ascending=[True, True, False])


def build_all_exponential_2x2_rows(weekly_returns: pd.DataFrame) -> pd.DataFrame:
    parts = [
        build_exponential_2x2_rows(weekly_returns, "score_v2_visible", "Ticker-visible V2"),
        build_exponential_2x2_rows(weekly_returns, "score_v2_anon", "Anonymized V2"),
        build_exponential_2x2_rows(weekly_returns, "score_v23_anon", "V2.3 rank-only"),
    ]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(
        ["llm_source", "half_life", "cagr_pct"], ascending=[True, True, False]
    )


def build_top10_exponential_summary(weekly_returns: pd.DataFrame) -> pd.DataFrame:
    """Compare equal and exponentially weighted books with identical top-10 membership."""
    if weekly_returns.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    models = ["eq10", *(f"exp10_h{half_life}" for half_life in TOP10_EXP_HALF_LIVES)]
    for score_col, source_label in TOP10_EXP_SCORE_LABELS.items():
        for weight_model in models:
            run = f"same::{score_col}__{weight_model}"
            data = weekly_returns[weekly_returns["run"].eq(run)].sort_values("date").copy()
            if data.empty:
                continue
            year_cagrs = {
                int(year): safe_cagr(group["capital_c2c"])
                for year, group in data.groupby(pd.to_datetime(data["date"]).dt.year)
            }
            metrics = metric_values(data["capital_c2c"])
            rows.append(
                {
                    "score_source": source_label,
                    "score_col": score_col,
                    "weight_model": weight_model,
                    "half_life": np.nan if weight_model == "eq10" else float(weight_model.split("_h", 1)[1]),
                    "cagr_2013_pct": year_cagrs.get(2013, np.nan),
                    "worst_year_cagr_pct": min(year_cagrs.values()) if year_cagrs else np.nan,
                    "positive_years": sum(value > 0 for value in year_cagrs.values()),
                    **metrics,
                }
            )
    if not rows:
        return pd.DataFrame()
    summary = pd.DataFrame(rows)
    equal_cagr = summary[summary["weight_model"].eq("eq10")].set_index("score_col")["cagr_pct"]
    trail5_cagr = summary[summary["score_col"].eq("score_trail5")].set_index("weight_model")["cagr_pct"]
    summary["cagr_lift_vs_same_score_eq10_pct"] = summary.apply(
        lambda row: float(row["cagr_pct"] - equal_cagr.get(row["score_col"], np.nan)), axis=1
    )
    summary["cagr_lift_vs_trail5_same_weight_pct"] = summary.apply(
        lambda row: float(row["cagr_pct"] - trail5_cagr.get(row["weight_model"], np.nan)), axis=1
    )
    return summary.sort_values(["score_source", "cagr_pct"], ascending=[True, False]).reset_index(drop=True)


def build_short_rank_audit(
    panel: pd.DataFrame,
    n_sims: int = SHORT_RANK_NULL_SIMS,
    seed: int = 20_260_721,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit exact anonymized short ranks and adjust the concentration ladder for search."""
    required = {"date", "pool_side", "ticker", "score_v2_anon", "fwd_c2c"}
    if panel.empty or not required.issubset(panel.columns):
        return pd.DataFrame(), pd.DataFrame()
    shorts = panel[panel["pool_side"].eq("SHORT")].copy()
    shorts["date"] = pd.to_datetime(shorts["date"])
    shorts = shorts.sort_values(
        ["date", "score_v2_anon", "ticker"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    shorts["rank"] = shorts.groupby("date").cumcount() + 1
    top10 = shorts[shorts["rank"].le(10)].copy()
    side_returns = (
        -top10.pivot(index="date", columns="rank", values="fwd_c2c")
        .sort_index()
        .reindex(columns=range(1, 11))
        .fillna(0.0)
    )
    if side_returns.empty or side_returns.shape[1] != 10:
        return pd.DataFrame(), pd.DataFrame()
    matrix = side_returns.to_numpy(dtype=float)

    rank_specs = [
        ("Rank 1 alone", [0], [1.0]),
        ("Rank 2 alone", [1], [1.0]),
        ("Rank 3 alone", [2], [1.0]),
        ("Equal ranks 1-3", [0, 1, 2], [1.0 / 3.0] * 3),
        ("Equal ranks 2-3", [1, 2], [0.5, 0.5]),
    ]
    slice_rows: list[dict[str, Any]] = []
    for construction, indices, weights in rank_specs:
        returns = matrix[:, indices] @ np.asarray(weights, dtype=float)
        slice_rows.append(
            {
                "construction": construction,
                "interpretation": (
                    "Ranks 1-3 excluding rank 1" if construction == "Equal ranks 2-3" else construction
                ),
                "avg_weekly_side_pnl_pct": float(returns.mean() * 100.0),
                "worst_week_pct": float(returns.min() * 100.0),
                **metric_values(pd.Series(returns)),
            }
        )
    slices = pd.DataFrame(slice_rows)

    models = ["eq10", "exp10_h10", "exp10_h5", "exp10_h3", "exp10_h2", "exp10_h1"]
    ranks = pd.Series(np.arange(1, 11, dtype=float))
    weight_matrix = np.vstack([weights_for_rank(ranks, model).to_numpy(dtype=float) for model in models])
    observed_returns = matrix @ weight_matrix.T
    observed_cagrs = np.asarray(
        [safe_cagr(pd.Series(observed_returns[:, idx])) for idx in range(len(models))], dtype=float
    )
    observed_best_idx = int(np.nanargmax(observed_cagrs))
    observed_best_cagr = float(observed_cagrs[observed_best_idx])
    observed_best_mean = float(observed_returns.mean(axis=0).max() * 100.0)

    rng = np.random.default_rng(seed)
    random_best_cagrs = np.empty(int(n_sims), dtype=float)
    random_best_means = np.empty(int(n_sims), dtype=float)
    for sim in range(int(n_sims)):
        permutation = np.argsort(rng.random(matrix.shape), axis=1)
        randomized = np.take_along_axis(matrix, permutation, axis=1) @ weight_matrix.T
        randomized_cagrs = [safe_cagr(pd.Series(randomized[:, idx])) for idx in range(len(models))]
        random_best_cagrs[sim] = float(np.nanmax(randomized_cagrs))
        random_best_means[sim] = float(randomized.mean(axis=0).max() * 100.0)
    null = pd.DataFrame(
        [
            {
                "null_hypothesis": "Ranks 1-10 are exchangeable within each date; best of six concentration levels retained",
                "n_dates": int(matrix.shape[0]),
                "n_concentration_levels": len(models),
                "n_randomizations": int(n_sims),
                "observed_best_model": models[observed_best_idx],
                "observed_best_cagr_pct": observed_best_cagr,
                "random_best_cagr_p50_pct": float(np.percentile(random_best_cagrs, 50)),
                "random_best_cagr_p95_pct": float(np.percentile(random_best_cagrs, 95)),
                "random_best_cagr_p99_pct": float(np.percentile(random_best_cagrs, 99)),
                "observed_cagr_percentile_pct": float((random_best_cagrs <= observed_best_cagr).mean() * 100.0),
                "one_sided_p_pct": float((random_best_cagrs >= observed_best_cagr).mean() * 100.0),
                "observed_best_avg_weekly_pct": observed_best_mean,
                "random_best_avg_weekly_p50_pct": float(np.percentile(random_best_means, 50)),
                "random_best_avg_weekly_p95_pct": float(np.percentile(random_best_means, 95)),
                "observed_avg_weekly_percentile_pct": float((random_best_means <= observed_best_mean).mean() * 100.0),
            }
        ]
    )
    return slices, null


def add_ordinal_rank(data: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    if out_col in data.columns:
        return data
    ordered = data.sort_values(["date", "pool_side", score_col, "ticker"], ascending=[True, True, False, True], kind="mergesort")
    ranks = ordered.groupby(["date", "pool_side"]).cumcount() + 1
    data[out_col] = pd.Series(ranks.to_numpy(), index=ordered.index).reindex(data.index)
    return data


def add_research_scores(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    diagnostic_score_cols = [
        "score_v2_visible",
        "score_v2_anon",
        "score_v23_anon",
        "score_trail5",
    ]
    for col in diagnostic_score_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data = add_ordinal_rank(data, col, f"ord_{col}")
            data[f"pct_{col}"] = (61.0 - data[f"ord_{col}"]) / 60.0
    if {"score_v2_visible", "score_v2_anon"}.issubset(data.columns):
        data["score_consensus_avg_raw"] = (data["score_v2_visible"] + data["score_v2_anon"]) / 2.0
        data["score_consensus_avg_pct"] = (data["pct_score_v2_visible"] + data["pct_score_v2_anon"]) / 2.0
        data["score_consensus_borda"] = -(data["ord_score_v2_visible"] + data["ord_score_v2_anon"])
        data["visible_anon_rank_diff"] = (data["ord_score_v2_visible"] - data["ord_score_v2_anon"]).abs()
        for col in ("score_consensus_avg_raw", "score_consensus_avg_pct", "score_consensus_borda"):
            data = add_ordinal_rank(data, col, f"ord_{col}")
            data[f"pct_{col}"] = (61.0 - data[f"ord_{col}"]) / 60.0
    return data


def side_weekly_from_panel(
    panel: pd.DataFrame,
    side: str,
    score_col: str,
    weight_model: str,
    disagreement_power: float = 0.0,
) -> pd.DataFrame:
    data = panel[panel["pool_side"].eq(side)].copy()
    if data.empty or score_col not in data.columns:
        return pd.DataFrame()
    data = data.dropna(subset=[score_col, "fwd_c2c", "fwd_o2o"])
    if data.empty:
        return pd.DataFrame()
    rank_col = f"ord_{score_col}"
    if rank_col not in data.columns:
        data = add_ordinal_rank(data, score_col, rank_col)
    ranks = pd.to_numeric(data[rank_col], errors="coerce")
    if weight_model == "eq_all60":
        raw = pd.Series(1.0, index=data.index)
    elif weight_model == "eq_top30":
        raw = ranks.le(30).astype(float)
    elif weight_model.startswith("exp_h"):
        half_life = float(weight_model.removeprefix("exp_h"))
        raw = np.exp(-(ranks - 1.0) / half_life)
    elif weight_model == "raw_score":
        raw = pd.to_numeric(data[score_col], errors="coerce").clip(lower=0.0).fillna(0.0)
    elif weight_model == "percentile_score":
        pct_col = f"pct_{score_col}"
        if pct_col not in data.columns:
            data[pct_col] = (61.0 - ranks) / 60.0
        raw = pd.to_numeric(data[pct_col], errors="coerce").clip(lower=0.0).fillna(0.0)
    else:
        raise ValueError(f"Unknown research weight model: {weight_model}")
    if disagreement_power > 0 and "visible_anon_rank_diff" in data.columns:
        penalty = (1.0 - pd.to_numeric(data["visible_anon_rank_diff"], errors="coerce").fillna(60.0) / 59.0).clip(0.0, 1.0)
        raw = raw * np.power(penalty, disagreement_power)
    denom = raw.groupby(data["date"]).transform("sum").replace(0.0, np.nan)
    data["portfolio_weight"] = (raw / denom).fillna(0.0)
    data["weighted_c2c"] = data["portfolio_weight"] * data["fwd_c2c"]
    data["weighted_o2o"] = data["portfolio_weight"] * data["fwd_o2o"]
    regime = data.groupby("date")["regime"].first() if "regime" in data.columns else pd.Series(dtype=object)
    weekly = (
        data.groupby("date", as_index=False)
        .agg(
            raw_c2c=("weighted_c2c", "sum"),
            raw_o2o=("weighted_o2o", "sum"),
            n_selected=("portfolio_weight", lambda x: int((x > 0).sum())),
        )
        .merge(regime.rename("regime"), on="date", how="left")
    )
    weekly["side"] = side
    weekly["score_col"] = score_col
    weekly["weight_model"] = weight_model
    weekly["side_pnl_c2c"] = weekly["raw_c2c"] if side == "LONG" else -weekly["raw_c2c"]
    return weekly


def combine_research_sides(
    panel: pd.DataFrame,
    long_score_col: str,
    short_score_col: str,
    long_weight_model: str,
    short_weight_model: str,
    label: str,
    long_disagreement_power: float = 0.0,
    short_disagreement_power: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    long_weekly = side_weekly_from_panel(panel, "LONG", long_score_col, long_weight_model, long_disagreement_power)
    short_weekly = side_weekly_from_panel(panel, "SHORT", short_score_col, short_weight_model, short_disagreement_power)
    if long_weekly.empty or short_weekly.empty:
        return {}, pd.DataFrame()
    merged = long_weekly.rename(columns={"raw_c2c": "long_raw_c2c", "raw_o2o": "long_raw_o2o"}).merge(
        short_weekly.rename(columns={"raw_c2c": "short_raw_c2c", "raw_o2o": "short_raw_o2o"})[
            ["date", "short_raw_c2c", "short_raw_o2o"]
        ],
        on="date",
        how="inner",
    )
    if merged.empty:
        return {}, pd.DataFrame()
    merged["capital_c2c"] = CAPITAL_SCALE * (merged["long_raw_c2c"] - merged["short_raw_c2c"])
    merged["run"] = label
    row = {
        "run": label,
        "long_score_col": long_score_col,
        "short_score_col": short_score_col,
        "long_weight_model": long_weight_model,
        "short_weight_model": short_weight_model,
        "long_disagreement_power": long_disagreement_power,
        "short_disagreement_power": short_disagreement_power,
        "avg_long_raw_c2c_pct": float(merged["long_raw_c2c"].mean() * 100.0),
        "avg_short_raw_c2c_pct": float(merged["short_raw_c2c"].mean() * 100.0),
        "avg_short_side_pnl_pct": float((-merged["short_raw_c2c"]).mean() * 100.0),
        **metric_values(merged["capital_c2c"]),
    }
    return row, merged


def save_rank_vs_score_tests(panel: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tests = [
        ("score_v2_visible", "Visible V2"),
        ("score_v2_anon", "Anonymized V2"),
        ("score_v23_anon", "V2.3 rank-only"),
        ("score_trail5", "Trail5"),
    ]
    weights = [
        ("eq_top30", "Rank-only top 30"),
        ("exp_h10", "Rank-only exponential h10"),
        ("raw_score", "Direct raw-score proportional"),
        ("percentile_score", "Percentile-normalized score proportional"),
    ]
    for score_col, source in tests:
        if score_col not in panel.columns:
            continue
        for weight_model, question in weights:
            row, _ = combine_research_sides(panel, score_col, score_col, weight_model, weight_model, f"{source} | {question}")
            if row:
                row.update({"source": source, "test": question})
                rows.append(row)
    data = pd.DataFrame(rows).sort_values(["source", "cagr_pct"], ascending=[True, False]) if rows else pd.DataFrame()
    data.to_csv(output_dir / "rank_vs_raw_score_tests.csv", index=False)
    return data


def save_visible_anon_agreement(panel: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    required = {"ord_score_v2_visible", "ord_score_v2_anon", "fwd_c2c"}
    if not required.issubset(panel.columns):
        data = pd.DataFrame()
        data.to_csv(output_dir / "visible_anon_top10_agreement.csv", index=False)
        return data
    data = panel.copy()
    visible_top = data["ord_score_v2_visible"].le(10)
    anon_top = data["ord_score_v2_anon"].le(10)
    data["agreement_bucket"] = np.select(
        [visible_top & anon_top, visible_top & ~anon_top, ~visible_top & anon_top],
        ["both_top10", "visible_only_top10", "anon_only_top10"],
        default="neither_top10",
    )
    data["side_pnl_c2c"] = np.where(data["pool_side"].eq("SHORT"), -data["fwd_c2c"], data["fwd_c2c"])
    grouped = (
        data.groupby(["pool_side", "agreement_bucket"], as_index=False)
        .agg(
            n=("ticker", "size"),
            avg_visible_rank=("ord_score_v2_visible", "mean"),
            avg_anon_rank=("ord_score_v2_anon", "mean"),
            avg_raw_c2c_pct=("fwd_c2c", lambda x: float(x.mean() * 100.0)),
            avg_side_pnl_c2c_pct=("side_pnl_c2c", lambda x: float(x.mean() * 100.0)),
            side_hit_rate_pct=("side_pnl_c2c", lambda x: float((x > 0).mean() * 100.0)),
            side_vol_pct=("side_pnl_c2c", lambda x: float(x.std(ddof=1) * 100.0)),
        )
        .sort_values(["pool_side", "avg_side_pnl_c2c_pct"], ascending=[True, False])
    )
    grouped.to_csv(output_dir / "visible_anon_top10_agreement.csv", index=False)
    return grouped


def save_consensus_rank_tests(panel: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tests = [
        ("score_v2_visible", "Visible raw score/rank"),
        ("score_v2_anon", "Anonymized raw score/rank"),
        ("score_consensus_avg_raw", "Average raw score"),
        ("score_consensus_avg_pct", "Average percentile rank"),
        ("score_consensus_borda", "Borda consensus rank"),
    ]
    for score_col, source in tests:
        if score_col not in panel.columns:
            continue
        for weight_model in ("eq_top30", "exp_h10", "percentile_score"):
            row, _ = combine_research_sides(panel, score_col, score_col, weight_model, weight_model, f"{source} | {weight_model}")
            if row:
                row.update({"source": source, "test": weight_model})
                rows.append(row)
    data = pd.DataFrame(rows).sort_values(["cagr_pct", "sharpe"], ascending=[False, False]) if rows else pd.DataFrame()
    data.to_csv(output_dir / "consensus_rank_tests.csv", index=False)
    return data


def save_rank_monotonicity(panel: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = [
        ("score_v2_visible", "Visible V2"),
        ("score_v2_anon", "Anonymized V2"),
        ("score_v23_anon", "V2.3 rank-only"),
        ("score_trail5", "Trail5"),
        ("score_consensus_avg_pct", "Consensus percentile"),
    ]
    for score_col, source in sources:
        rank_col = f"ord_{score_col}"
        if rank_col not in panel.columns:
            continue
        data = panel.copy()
        data["rank_bucket_start"] = (((pd.to_numeric(data[rank_col], errors="coerce") - 1) // 5) * 5 + 1).astype("Int64")
        data["rank_bucket"] = data["rank_bucket_start"].astype(str) + "-" + (data["rank_bucket_start"] + 4).astype(str)
        data["side_pnl_c2c"] = np.where(data["pool_side"].eq("SHORT"), -data["fwd_c2c"], data["fwd_c2c"])
        grouped = (
            data.groupby(["pool_side", "rank_bucket_start", "rank_bucket"], as_index=False)
            .agg(
                n=("ticker", "size"),
                avg_raw_c2c_pct=("fwd_c2c", lambda x: float(x.mean() * 100.0)),
                avg_side_pnl_c2c_pct=("side_pnl_c2c", lambda x: float(x.mean() * 100.0)),
                side_hit_rate_pct=("side_pnl_c2c", lambda x: float((x > 0).mean() * 100.0)),
                side_vol_pct=("side_pnl_c2c", lambda x: float(x.std(ddof=1) * 100.0)),
            )
            .sort_values(["pool_side", "rank_bucket_start"])
        )
        grouped["source"] = source
        rows.extend(grouped.to_dict("records"))
    data = pd.DataFrame(rows)
    if not data.empty:
        data = data[["source", "pool_side", "rank_bucket", "n", "avg_raw_c2c_pct", "avg_side_pnl_c2c_pct", "side_hit_rate_pct", "side_vol_pct"]]
    data.to_csv(output_dir / "rank_monotonicity.csv", index=False)
    return data


def save_disagreement_tests(panel: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "visible_anon_rank_diff" not in panel.columns:
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "visible_anon_disagreement_risk.csv", index=False)
        empty.to_csv(output_dir / "disagreement_penalty_strategy.csv", index=False)
        return empty, empty
    data = panel.copy()
    bins = [-0.1, 5, 10, 20, 40, 60]
    labels = ["0-5", "6-10", "11-20", "21-40", "41-60"]
    data["disagreement_bucket"] = pd.cut(data["visible_anon_rank_diff"], bins=bins, labels=labels)
    data["side_pnl_c2c"] = np.where(data["pool_side"].eq("SHORT"), -data["fwd_c2c"], data["fwd_c2c"])
    risk = (
        data.groupby(["pool_side", "disagreement_bucket"], observed=False, as_index=False)
        .agg(
            n=("ticker", "size"),
            avg_rank_diff=("visible_anon_rank_diff", "mean"),
            avg_raw_c2c_pct=("fwd_c2c", lambda x: float(x.mean() * 100.0)),
            avg_side_pnl_c2c_pct=("side_pnl_c2c", lambda x: float(x.mean() * 100.0)),
            side_hit_rate_pct=("side_pnl_c2c", lambda x: float((x > 0).mean() * 100.0)),
            side_vol_pct=("side_pnl_c2c", lambda x: float(x.std(ddof=1) * 100.0)),
            worst_side_return_pct=("side_pnl_c2c", lambda x: float(x.min() * 100.0)),
        )
        .sort_values(["pool_side", "avg_rank_diff"])
    )
    risk.to_csv(output_dir / "visible_anon_disagreement_risk.csv", index=False)

    rows: list[dict[str, Any]] = []
    for target, long_power, short_power in [
        ("none", 0.0, 0.0),
        ("both_p1", 1.0, 1.0),
        ("both_p2", 2.0, 2.0),
        ("both_p4", 4.0, 4.0),
        ("long_only_p1", 1.0, 0.0),
        ("long_only_p2", 2.0, 0.0),
        ("short_only_p1", 0.0, 1.0),
        ("short_only_p2", 0.0, 2.0),
    ]:
        row, _ = combine_research_sides(
            panel,
            "score_v2_visible",
            "score_v2_visible",
            "exp_h10",
            "exp_h10",
            f"Visible V2 exp h10 disagreement penalty {target}",
            long_power,
            short_power,
        )
        if row:
            row["penalty_target"] = target
            rows.append(row)
    strategy = pd.DataFrame(rows).sort_values(["cagr_pct", "sharpe"], ascending=[False, False]) if rows else pd.DataFrame()
    strategy.to_csv(output_dir / "disagreement_penalty_strategy.csv", index=False)
    return risk, strategy


def save_directional_disagreement(panel: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    required = {"ord_score_v2_visible", "ord_score_v2_anon", "fwd_c2c"}
    if not required.issubset(panel.columns):
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "directional_visible_anon_disagreement.csv", index=False)
        return empty
    data = panel.copy()
    data["visible_minus_anon_rank_gap"] = (
        pd.to_numeric(data["ord_score_v2_visible"], errors="coerce")
        - pd.to_numeric(data["ord_score_v2_anon"], errors="coerce")
    )
    labels = [
        "visible_higher_41_plus",
        "visible_higher_21_40",
        "visible_higher_11_20",
        "visible_higher_6_10",
        "aligned_0_5",
        "anon_higher_6_10",
        "anon_higher_11_20",
        "anon_higher_21_40",
        "anon_higher_41_plus",
    ]
    data["directional_gap_bucket"] = pd.cut(
        data["visible_minus_anon_rank_gap"],
        bins=[-np.inf, -40, -20, -10, -5, 5, 10, 20, 40, np.inf],
        labels=labels,
        include_lowest=True,
    )
    data["side_pnl_c2c"] = np.where(data["pool_side"].eq("SHORT"), -data["fwd_c2c"], data["fwd_c2c"])
    grouped = (
        data.groupby(["pool_side", "directional_gap_bucket"], observed=False, as_index=False)
        .agg(
            n=("ticker", "size"),
            avg_visible_rank=("ord_score_v2_visible", "mean"),
            avg_anon_rank=("ord_score_v2_anon", "mean"),
            avg_signed_gap=("visible_minus_anon_rank_gap", "mean"),
            avg_raw_c2c_pct=("fwd_c2c", lambda x: float(x.mean() * 100.0)),
            avg_side_pnl_c2c_pct=("side_pnl_c2c", lambda x: float(x.mean() * 100.0)),
            side_hit_rate_pct=("side_pnl_c2c", lambda x: float((x > 0).mean() * 100.0)),
            side_vol_pct=("side_pnl_c2c", lambda x: float(x.std(ddof=1) * 100.0)),
            worst_side_return_pct=("side_pnl_c2c", lambda x: float(x.min() * 100.0)),
        )
        .sort_values(["pool_side", "avg_signed_gap"])
    )
    grouped.to_csv(output_dir / "directional_visible_anon_disagreement.csv", index=False)
    return grouped


def random_strategy_returns(panel: pd.DataFrame, weight_model: str, n_sims: int, seed: int = 11) -> np.ndarray:
    dates = pd.Index(sorted(panel["date"].dropna().unique()))
    date_pos = {date: idx for idx, date in enumerate(dates)}
    rng = np.random.default_rng(seed)
    sim_returns = np.zeros((n_sims, len(dates)), dtype=float)
    for (date, side), group in panel.groupby(["date", "pool_side"], sort=True):
        n = len(group)
        if n == 0:
            continue
        if weight_model == "eq_top30":
            weights = np.zeros(n, dtype=float)
            weights[: min(30, n)] = 1.0 / min(30, n)
        elif weight_model == "exp_h10":
            ranks = np.arange(n, dtype=float)
            weights = np.exp(-ranks / 10.0)
            weights = weights / weights.sum()
        else:
            raise ValueError(f"Unknown random placebo weight model: {weight_model}")
        returns = pd.to_numeric(group["fwd_c2c"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        random_keys = rng.random((n_sims, n))
        perms = np.argsort(random_keys, axis=1)
        side_returns = returns[perms] @ weights
        sign = 1.0 if side == "LONG" else -1.0
        sim_returns[:, date_pos[date]] += CAPITAL_SCALE * sign * side_returns
    return sim_returns


def save_random_placebo(panel: pd.DataFrame, output_dir: Path, n_sims: int = 2000) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for weight_model in ("eq_top30", "exp_h10"):
        sims = random_strategy_returns(panel, weight_model, n_sims=n_sims)
        cagr = np.array([safe_cagr(pd.Series(row)) for row in sims], dtype=float)
        sharpe = np.array([metric_values(pd.Series(row))["sharpe"] for row in sims], dtype=float)
        drawdown = np.array([max_drawdown_pct(pd.Series(row)) for row in sims], dtype=float)
        for score_col, label in [
            ("score_trail5", "Trail5"),
            ("score_v2_visible", "Visible V2"),
            ("score_v2_anon", "Anonymized V2"),
            ("score_v23_anon", "V2.3 rank-only"),
        ]:
            if score_col not in panel.columns:
                continue
            real_row, _ = combine_research_sides(panel, score_col, score_col, weight_model, weight_model, f"{label} {weight_model}")
            if not real_row:
                continue
            real_cagr = float(real_row["cagr_pct"])
            rows.append(
                {
                    "weight_model": weight_model,
                    "real_ranking": label,
                    "n_random_sims": n_sims,
                    "real_cagr_pct": real_cagr,
                    "real_sharpe": float(real_row["sharpe"]),
                    "real_max_drawdown_pct": float(real_row["max_drawdown_pct"]),
                    "random_mean_cagr_pct": float(np.nanmean(cagr)),
                    "random_p05_cagr_pct": float(np.nanquantile(cagr, 0.05)),
                    "random_p50_cagr_pct": float(np.nanquantile(cagr, 0.50)),
                    "random_p95_cagr_pct": float(np.nanquantile(cagr, 0.95)),
                    "random_max_cagr_pct": float(np.nanmax(cagr)),
                    "real_cagr_percentile_pct": float((cagr <= real_cagr).mean() * 100.0),
                    "random_mean_sharpe": float(np.nanmean(sharpe)),
                    "random_p95_sharpe": float(np.nanquantile(sharpe, 0.95)),
                    "random_p05_max_drawdown_pct": float(np.nanquantile(drawdown, 0.05)),
                }
            )
    data = pd.DataFrame(rows).sort_values(["weight_model", "real_cagr_pct"], ascending=[True, False]) if rows else pd.DataFrame()
    data.to_csv(output_dir / "random_ranking_placebo.csv", index=False)
    return data


def save_pool_selector_decomposition(panel: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    tests = [
        ("Pool only: equal all 60", "score_trail5", "eq_all60"),
        ("Trail5 selector: top30 equal", "score_trail5", "eq_top30"),
        ("Trail5 selector: exponential h10", "score_trail5", "exp_h10"),
        ("V2 selector: top30 equal", "score_v2_visible", "eq_top30"),
        ("V2 selector: exponential h10", "score_v2_visible", "exp_h10"),
        ("Anon V2 selector: top30 equal", "score_v2_anon", "eq_top30"),
        ("Anon V2 selector: exponential h10", "score_v2_anon", "exp_h10"),
        ("V2.3 rank-only selector: top30 equal", "score_v23_anon", "eq_top30"),
        ("V2.3 rank-only selector: exponential h5", "score_v23_anon", "exp_h5"),
        ("V2.3 rank-only selector: exponential h10", "score_v23_anon", "exp_h10"),
        ("Consensus percentile: top30 equal", "score_consensus_avg_pct", "eq_top30"),
        ("Consensus percentile: exponential h10", "score_consensus_avg_pct", "exp_h10"),
    ]
    rows: list[dict[str, Any]] = []
    for label, score_col, weight_model in tests:
        if score_col not in panel.columns:
            continue
        row, _ = combine_research_sides(panel, score_col, score_col, weight_model, weight_model, label)
        if row:
            row["decomposition_step"] = label
            rows.append(row)
    data = pd.DataFrame(rows).sort_values(["cagr_pct", "sharpe"], ascending=[False, False]) if rows else pd.DataFrame()
    data.to_csv(output_dir / "candidate_pool_selector_decomposition.csv", index=False)
    return data


def save_walk_forward_selection(weekly_returns: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if weekly_returns.empty:
        data = pd.DataFrame()
        data.to_csv(output_dir / "walk_forward_parameter_selection.csv", index=False)
        return data
    candidate_runs = [
        "same::score_trail5__eq30",
        "same::score_v2_visible__eq30",
        "same::score_v2_anon__eq30",
        "same::score_v23_anon__eq30",
    ]
    for score in ("score_trail5", "score_v2_visible", "score_v2_anon", "score_v23_anon"):
        for half_life in (5, 10, 20, 40):
            candidate_runs.append(f"same::{score}__exp60_h{half_life}")
    data = weekly_returns[weekly_returns["run"].isin(candidate_runs)].copy()
    if data.empty:
        out = pd.DataFrame()
        out.to_csv(output_dir / "walk_forward_parameter_selection.csv", index=False)
        return out
    data["year"] = pd.to_datetime(data["date"]).dt.year
    years = sorted(data["year"].dropna().unique())
    rows: list[dict[str, Any]] = []
    for selection_metric in ("cagr_pct", "sharpe"):
        for test_year in years:
            train = data[data["year"].lt(test_year)]
            test = data[data["year"].eq(test_year)]
            if train["year"].nunique() < 2 or test.empty:
                continue
            train_rows = []
            for run, group in train.groupby("run"):
                train_rows.append({"run": run, **metric_values(group["capital_c2c"])})
            train_metrics = pd.DataFrame(train_rows).dropna(subset=[selection_metric])
            if train_metrics.empty:
                continue
            best = train_metrics.sort_values([selection_metric, "cagr_pct"], ascending=[False, False]).iloc[0]
            selected_test = test[test["run"].eq(best["run"])]
            v2_test = test[test["run"].eq("same::score_v2_visible__eq30")]
            trail5_test = test[test["run"].eq("same::score_trail5__eq30")]
            selected_metrics = metric_values(selected_test["capital_c2c"])
            v2_metrics = metric_values(v2_test["capital_c2c"]) if not v2_test.empty else {}
            trail5_metrics = metric_values(trail5_test["capital_c2c"]) if not trail5_test.empty else {}
            rows.append(
                {
                    "selection_metric": selection_metric,
                    "train_years": f"{int(min(train['year']))}-{int(max(train['year']))}",
                    "test_year": int(test_year),
                    "selected_run": best["run"],
                    "train_cagr_pct": float(best["cagr_pct"]),
                    "train_sharpe": float(best["sharpe"]),
                    "test_cagr_pct": selected_metrics["cagr_pct"],
                    "test_sharpe": selected_metrics["sharpe"],
                    "test_max_drawdown_pct": selected_metrics["max_drawdown_pct"],
                    "fixed_v2_test_cagr_pct": v2_metrics.get("cagr_pct", np.nan),
                    "fixed_trail5_test_cagr_pct": trail5_metrics.get("cagr_pct", np.nan),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "walk_forward_parameter_selection.csv", index=False)
    return out


REGIME_GATE_ARMS = {
    "top30": "same::score_v2_anon__eq30",
    "h20": "same::score_v2_anon__exp60_h20",
    "h10": "same::score_v2_anon__exp60_h10",
    "h5": "same::score_v2_anon__exp60_h5",
}


def _regime_gate_arm_panel(weekly_returns: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "run", "capital_c2c", "long_raw_c2c", "short_raw_c2c", "regime"}
    if weekly_returns.empty or not required.issubset(weekly_returns.columns):
        return pd.DataFrame()
    selected = weekly_returns[weekly_returns["run"].isin(REGIME_GATE_ARMS.values())].copy()
    selected["date"] = pd.to_datetime(selected["date"])
    if selected.empty:
        return pd.DataFrame()
    selected["arm"] = selected["run"].map({run: arm for arm, run in REGIME_GATE_ARMS.items()})
    counts = selected.groupby("arm")["date"].nunique()
    if any(counts.get(arm, 0) == 0 for arm in REGIME_GATE_ARMS):
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for value_col in ("capital_c2c", "long_raw_c2c", "short_raw_c2c"):
        part = selected.pivot_table(index="date", columns="arm", values=value_col, aggfunc="first")
        part = part.rename(columns={arm: f"{arm}_{value_col}" for arm in part.columns})
        parts.append(part)
    panel = pd.concat(parts, axis=1).reset_index()
    regimes = selected.groupby("date")["regime"].first().rename("regime").reset_index()
    panel = panel.merge(regimes, on="date", how="left").sort_values("date").reset_index(drop=True)
    panel["regime"] = panel["regime"].fillna("unknown").astype(str)
    panel["year"] = panel["date"].dt.year
    return panel


def _apply_regime_gate(
    panel: pd.DataFrame,
    mapping: dict[str, str],
    fallback_arm: str,
    rule_name: str,
    rule_type: str,
) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    chosen = panel.copy()
    chosen["selected_arm"] = chosen["regime"].map(mapping).fillna(fallback_arm)
    return _materialize_selected_arms(chosen, rule_name, rule_type)


def _materialize_selected_arms(panel: pd.DataFrame, rule_name: str, rule_type: str) -> pd.DataFrame:
    chosen = panel.copy()
    invalid = sorted(set(chosen["selected_arm"].dropna()) - set(REGIME_GATE_ARMS))
    if invalid:
        raise ValueError(f"Unknown regime-gate arms: {', '.join(invalid)}")
    chosen["rule"] = rule_name
    chosen["rule_type"] = rule_type
    for value_col in ("capital_c2c", "long_raw_c2c", "short_raw_c2c"):
        chosen[value_col] = np.nan
        for arm in REGIME_GATE_ARMS:
            mask = chosen["selected_arm"].eq(arm)
            chosen.loc[mask, value_col] = chosen.loc[mask, f"{arm}_{value_col}"]
    return chosen[
        [
            "date",
            "year",
            "regime",
            "rule",
            "rule_type",
            "selected_arm",
            "capital_c2c",
            "long_raw_c2c",
            "short_raw_c2c",
        ]
    ]


def _regime_gate_score_row(rule: str, rule_type: str, data: pd.DataFrame) -> dict[str, Any]:
    metrics = metric_values(data["capital_c2c"])
    counts = data["selected_arm"].value_counts(normalize=True)
    return {
        "rule": rule,
        "rule_type": rule_type,
        "date_start": data["date"].min(),
        "date_end": data["date"].max(),
        "avg_weekly_pct": float(data["capital_c2c"].mean() * 100.0),
        "avg_long_contribution_pct": float(data["long_raw_c2c"].mean() * CAPITAL_SCALE * 100.0),
        "avg_short_contribution_pct": float(-data["short_raw_c2c"].mean() * CAPITAL_SCALE * 100.0),
        "h5_usage_pct": float(counts.get("h5", 0.0) * 100.0),
        "h10_usage_pct": float(counts.get("h10", 0.0) * 100.0),
        "h20_usage_pct": float(counts.get("h20", 0.0) * 100.0),
        "top30_usage_pct": float(counts.get("top30", 0.0) * 100.0),
        **metrics,
    }


def _learn_regime_mapping(
    train: pd.DataFrame,
    arms: tuple[str, ...],
    min_regime_dates: int,
) -> tuple[dict[str, str], str, list[dict[str, Any]]]:
    global_means = {arm: float(train[f"{arm}_capital_c2c"].mean()) for arm in arms}
    fallback = max(global_means, key=global_means.get)
    mapping: dict[str, str] = {}
    details: list[dict[str, Any]] = []
    for regime in sorted(train["regime"].dropna().astype(str).unique()):
        group = train[train["regime"].eq(regime)]
        means = {arm: float(group[f"{arm}_capital_c2c"].mean()) for arm in arms}
        enough_history = len(group) >= min_regime_dates
        selected_arm = max(means, key=means.get) if enough_history else fallback
        mapping[regime] = selected_arm
        detail = {
            "regime": regime,
            "n_train_dates": int(len(group)),
            "selected_arm": selected_arm,
            "used_regime_specific_choice": bool(enough_history),
            "fallback_arm": fallback,
        }
        detail.update({f"train_{arm}_avg_weekly_pct": means[arm] * 100.0 for arm in arms})
        details.append(detail)
    return mapping, fallback, details


def _walk_forward_regime_gate(
    panel: pd.DataFrame,
    arms: tuple[str, ...],
    gate_name: str,
    min_regime_dates: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly_parts: list[pd.DataFrame] = []
    detail_rows: list[dict[str, Any]] = []
    for test_year in sorted(panel["year"].dropna().unique()):
        train = panel[panel["year"].lt(test_year)]
        test = panel[panel["year"].eq(test_year)]
        if train["year"].nunique() < 2 or test.empty:
            continue
        mapping, fallback, details = _learn_regime_mapping(train, arms, min_regime_dates)
        applied = _apply_regime_gate(
            test,
            mapping,
            fallback,
            gate_name,
            "walk_forward",
        )
        weekly_parts.append(applied)
        for detail in details:
            detail_rows.append(
                {
                    "gate": gate_name,
                    "train_years": f"{int(train['year'].min())}-{int(train['year'].max())}",
                    "test_year": int(test_year),
                    "candidate_arms": ", ".join(arms),
                    **detail,
                }
            )
    weekly = pd.concat(weekly_parts, ignore_index=True) if weekly_parts else pd.DataFrame()
    details = pd.DataFrame(detail_rows)
    return weekly, details


def _regime_gate_placebo(
    gate_weekly: pd.DataFrame,
    panel: pd.DataFrame,
    aggressive_arm: str,
    defensive_arm: str,
    n_sims: int,
    seed: int,
) -> dict[str, Any]:
    if gate_weekly.empty:
        return {}
    source = panel[panel["date"].isin(gate_weekly["date"])].copy().sort_values("date")
    actual = gate_weekly.sort_values("date")
    real_metrics = metric_values(actual["capital_c2c"])
    rng = np.random.default_rng(seed)
    random_cagrs = np.empty(n_sims)
    random_sharpes = np.empty(n_sims)
    for sim in range(n_sims):
        simulated_parts: list[pd.Series] = []
        for year, year_data in source.groupby("year", sort=True):
            actual_year = actual[actual["year"].eq(year)]
            n_aggressive = int(actual_year["selected_arm"].eq(aggressive_arm).sum())
            choose = np.zeros(len(year_data), dtype=bool)
            if n_aggressive:
                selected_positions = rng.choice(len(year_data), size=n_aggressive, replace=False)
                choose[selected_positions] = True
            values = np.where(
                choose,
                year_data[f"{aggressive_arm}_capital_c2c"].to_numpy(),
                year_data[f"{defensive_arm}_capital_c2c"].to_numpy(),
            )
            simulated_parts.append(pd.Series(values))
        simulated = pd.concat(simulated_parts, ignore_index=True)
        metrics = metric_values(simulated)
        random_cagrs[sim] = metrics["cagr_pct"]
        random_sharpes[sim] = metrics["sharpe"]
    return {
        "gate": str(actual["rule"].iloc[0]),
        "aggressive_arm": aggressive_arm,
        "defensive_arm": defensive_arm,
        "n_sims": int(n_sims),
        "real_cagr_pct": real_metrics["cagr_pct"],
        "real_sharpe": real_metrics["sharpe"],
        "random_cagr_p50_pct": float(np.quantile(random_cagrs, 0.50)),
        "random_cagr_p95_pct": float(np.quantile(random_cagrs, 0.95)),
        "real_cagr_percentile_pct": float((random_cagrs <= real_metrics["cagr_pct"]).mean() * 100.0),
        "random_sharpe_p50": float(np.quantile(random_sharpes, 0.50)),
        "random_sharpe_p95": float(np.quantile(random_sharpes, 0.95)),
        "real_sharpe_percentile_pct": float((random_sharpes <= real_metrics["sharpe"]).mean() * 100.0),
    }


def _paired_regime_gate_bootstrap(
    target: pd.DataFrame,
    control: pd.DataFrame,
    comparison: str,
    n_sims: int,
    seed: int,
) -> dict[str, Any]:
    paired = target[["date", "capital_c2c"]].merge(
        control[["date", "capital_c2c"]],
        on="date",
        how="inner",
        suffixes=("_target", "_control"),
    )
    delta = (paired["capital_c2c_target"] - paired["capital_c2c_control"]).dropna().to_numpy()
    if len(delta) == 0:
        return {}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(delta), size=(n_sims, len(delta)))
    annualized = delta[samples].mean(axis=1) * PERIODS_PER_YEAR * 100.0
    actual = float(delta.mean() * PERIODS_PER_YEAR * 100.0)
    p_two = float(2.0 * min((annualized <= 0.0).mean(), (annualized >= 0.0).mean()))
    return {
        "target_rule": str(target["rule"].iloc[0]),
        "control_rule": str(control["rule"].iloc[0]),
        "comparison": comparison,
        "n_dates": int(len(delta)),
        "mean_annualized_delta_pct": actual,
        "ci_low_pct": float(np.quantile(annualized, 0.025)),
        "ci_high_pct": float(np.quantile(annualized, 0.975)),
        "p_two_sided": min(1.0, p_two),
        "probability_target_beats_control_pct": float((annualized > 0.0).mean() * 100.0),
    }


def _ordered_observed_regimes(panel: pd.DataFrame) -> tuple[str, ...]:
    preferred = ("normal", "grind_up", "crisis", "squeeze", "slow_bleed", "unknown")
    observed = set(panel["regime"].dropna().astype(str))
    ordered = [regime for regime in preferred if regime in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return tuple(ordered)


def _mapping_description(mapping: dict[str, str], regimes: tuple[str, ...]) -> str:
    return " | ".join(f"{regime}={mapping[regime]}" for regime in regimes)


def _evaluate_regime_mapping_sweep(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regimes = _ordered_observed_regimes(panel)
    arms = tuple(REGIME_GATE_ARMS)
    rows: list[dict[str, Any]] = []
    mapping_by_id: dict[str, dict[str, str]] = {}
    for combination in product(arms, repeat=len(regimes)):
        mapping = dict(zip(regimes, combination))
        mapping_id = _mapping_description(mapping, regimes)
        mapping_by_id[mapping_id] = mapping
        weekly = _apply_regime_gate(panel, mapping, "top30", mapping_id, "development_mapping")
        row = _regime_gate_score_row(mapping_id, "development_mapping", weekly)
        year_cagrs: list[float] = []
        for year, group in weekly.groupby("year", sort=True):
            year_metrics = metric_values(group["capital_c2c"])
            year_cagr = float(year_metrics["cagr_pct"])
            row[f"cagr_{int(year)}_pct"] = year_cagr
            row[f"max_dd_{int(year)}_pct"] = float(year_metrics["max_drawdown_pct"])
            year_cagrs.append(year_cagr)
        row["mapping_id"] = mapping_id
        for regime in regimes:
            row[f"arm_{regime}"] = mapping[regime]
        row["calmar_ratio"] = (
            float(row["cagr_pct"]) / abs(float(row["max_drawdown_pct"]))
            if pd.notna(row["max_drawdown_pct"]) and float(row["max_drawdown_pct"]) < 0.0
            else np.nan
        )
        row["worst_year_cagr_pct"] = min(year_cagrs) if year_cagrs else np.nan
        row["year_cagr_std_pct"] = float(np.std(year_cagrs, ddof=1)) if len(year_cagrs) > 1 else np.nan
        row["positive_years"] = int(sum(value > 0.0 for value in year_cagrs))
        rows.append(row)

    sweep = pd.DataFrame(rows)
    if sweep.empty:
        return sweep, pd.DataFrame(), pd.DataFrame()
    sweep["cagr_rank"] = sweep["cagr_pct"].rank(method="min", ascending=False).astype("Int64")
    sweep["sharpe_rank"] = sweep["sharpe"].rank(method="min", ascending=False).astype("Int64")
    sweep["calmar_rank"] = sweep["calmar_ratio"].rank(method="min", ascending=False).astype("Int64")
    sweep["worst_year_rank"] = sweep["worst_year_cagr_pct"].rank(method="min", ascending=False).astype("Int64")

    fixed_h5_id = _mapping_description({regime: "h5" for regime in regimes}, regimes)
    fixed_h5 = sweep[sweep["mapping_id"].eq(fixed_h5_id)].iloc[0]
    no_worse_dd = sweep[sweep["max_drawdown_pct"].ge(float(fixed_h5["max_drawdown_pct"]))]
    champion_specs = [
        ("Best development CAGR", sweep.sort_values(["cagr_pct", "sharpe"], ascending=False).iloc[0]),
        ("Best development Sharpe", sweep.sort_values(["sharpe", "cagr_pct"], ascending=False).iloc[0]),
        ("Best development Calmar", sweep.sort_values(["calmar_ratio", "cagr_pct"], ascending=False).iloc[0]),
        (
            "Best worst-year CAGR",
            sweep.sort_values(["worst_year_cagr_pct", "cagr_pct"], ascending=False).iloc[0],
        ),
        (
            "Best CAGR with DD no worse than fixed h5",
            no_worse_dd.sort_values(["cagr_pct", "sharpe"], ascending=False).iloc[0],
        ),
        ("Fixed h5 reference", fixed_h5),
    ]
    champion_rows: list[dict[str, Any]] = []
    champion_year_rows: list[dict[str, Any]] = []
    for objective, source in champion_specs:
        mapping_id = str(source["mapping_id"])
        mapping = mapping_by_id[mapping_id]
        champion_rows.append({"objective": objective, **source.to_dict()})
        weekly = _apply_regime_gate(panel, mapping, "top30", objective, "development_champion")
        for year, group in weekly.groupby("year", sort=True):
            champion_year_rows.append(
                {
                    "objective": objective,
                    "mapping_id": mapping_id,
                    "year": int(year),
                    **metric_values(group["capital_c2c"]),
                }
            )
    champions = pd.DataFrame(champion_rows)
    champion_years = pd.DataFrame(champion_year_rows)
    return sweep.sort_values(["cagr_pct", "sharpe"], ascending=False), champions, champion_years


def _regime_mapping_search_placebo(
    panel: pd.DataFrame,
    observed_values: dict[str, float],
    n_sims: int,
    seed: int = 907,
) -> pd.DataFrame:
    if panel.empty or n_sims <= 0:
        return pd.DataFrame()
    regimes = np.asarray(_ordered_observed_regimes(panel), dtype=object)
    labels = panel["regime"].astype(str).to_numpy()
    arms = tuple(REGIME_GATE_ARMS)
    returns = np.column_stack([panel[f"{arm}_capital_c2c"].to_numpy(dtype=float) for arm in arms])
    if np.any(returns <= -1.0):
        raise ValueError("Regime mapping placebo requires all weekly returns to be greater than -100%")
    combinations = np.asarray(list(product(range(len(arms)), repeat=len(regimes))), dtype=int)
    regime_to_idx = {str(regime): idx for idx, regime in enumerate(regimes)}
    years = panel["year"].to_numpy(dtype=int)
    unique_years = np.unique(years)
    rng = np.random.default_rng(seed)
    best_cagrs = np.empty(n_sims, dtype=float)
    best_sharpes = np.empty(n_sims, dtype=float)
    best_worst_years = np.empty(n_sims, dtype=float)
    n_dates = len(panel)
    for sim in range(n_sims):
        shuffled = rng.permutation(labels)
        regime_idx = np.asarray([regime_to_idx[str(label)] for label in shuffled], dtype=int)
        selected_arm_idx = combinations[:, regime_idx]
        selected = returns[np.arange(n_dates)[None, :], selected_arm_idx]
        log_selected = np.log1p(selected)
        cagrs = (np.exp(log_selected.sum(axis=1) * PERIODS_PER_YEAR / n_dates) - 1.0) * 100.0
        means = selected.mean(axis=1)
        vols = selected.std(axis=1, ddof=1)
        sharpes = np.divide(
            means * PERIODS_PER_YEAR,
            vols * np.sqrt(PERIODS_PER_YEAR),
            out=np.full_like(means, np.nan),
            where=vols > 0.0,
        )
        year_cagrs = []
        for year in unique_years:
            mask = years == year
            year_cagrs.append(
                (np.exp(log_selected[:, mask].sum(axis=1) * PERIODS_PER_YEAR / int(mask.sum())) - 1.0) * 100.0
            )
        worst_year = np.min(np.column_stack(year_cagrs), axis=1)
        best_cagrs[sim] = float(np.nanmax(cagrs))
        best_sharpes[sim] = float(np.nanmax(sharpes))
        best_worst_years[sim] = float(np.nanmax(worst_year))

    rows: list[dict[str, Any]] = []
    for objective, values in (
        ("Best development CAGR", best_cagrs),
        ("Best development Sharpe", best_sharpes),
        ("Best worst-year CAGR", best_worst_years),
    ):
        observed = float(observed_values.get(objective, np.nan))
        rows.append(
            {
                "objective": objective,
                "n_regimes": int(len(regimes)),
                "n_arms": int(len(arms)),
                "n_mappings_searched": int(len(combinations)),
                "n_permutations": int(n_sims),
                "observed_value": observed,
                "shuffled_best_p50": float(np.quantile(values, 0.50)),
                "shuffled_best_p95": float(np.quantile(values, 0.95)),
                "shuffled_best_p99": float(np.quantile(values, 0.99)),
                "shuffled_best_max": float(values.max()),
                "observed_percentile_pct": float((values <= observed).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def save_regime_gate_experiments(
    weekly_returns: pd.DataFrame,
    output_dir: Path,
    random_sims: int = 2000,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = _regime_gate_arm_panel(weekly_returns)
    empty = pd.DataFrame()
    paths = {
        "regime_gate_scorecard": output_dir / "regime_gate_scorecard.csv",
        "regime_gate_by_regime": output_dir / "regime_gate_arm_performance.csv",
        "regime_gate_by_year": output_dir / "regime_gate_year_metrics.csv",
        "regime_gate_walk_forward": output_dir / "regime_gate_walk_forward_choices.csv",
        "regime_gate_weekly": output_dir / "regime_gate_weekly_returns.csv",
        "regime_gate_placebo": output_dir / "regime_gate_random_timing_placebo.csv",
        "regime_gate_bootstrap": output_dir / "regime_gate_paired_bootstrap.csv",
        "regime_mapping_sweep": output_dir / "regime_mapping_development_sweep.csv",
        "regime_mapping_champions": output_dir / "regime_mapping_champions.csv",
        "regime_mapping_champion_years": output_dir / "regime_mapping_champion_year_metrics.csv",
        "regime_mapping_search_placebo": output_dir / "regime_mapping_search_placebo.csv",
    }
    if panel.empty:
        for path in paths.values():
            empty.to_csv(path, index=False)
        return {key: empty for key in paths}

    mapping_sweep, mapping_champions, mapping_champion_years = _evaluate_regime_mapping_sweep(panel)
    observed_search_values = {
        "Best development CAGR": float(
            mapping_champions.loc[mapping_champions["objective"].eq("Best development CAGR"), "cagr_pct"].iloc[0]
        ),
        "Best development Sharpe": float(
            mapping_champions.loc[mapping_champions["objective"].eq("Best development Sharpe"), "sharpe"].iloc[0]
        ),
        "Best worst-year CAGR": float(
            mapping_champions.loc[mapping_champions["objective"].eq("Best worst-year CAGR"), "worst_year_cagr_pct"].iloc[0]
        ),
    }
    mapping_search_placebo = _regime_mapping_search_placebo(
        panel,
        observed_values=observed_search_values,
        n_sims=random_sims,
    )

    weekly_runs: list[pd.DataFrame] = []
    for arm in REGIME_GATE_ARMS:
        weekly_runs.append(_apply_regime_gate(panel, {}, arm, f"Fixed {arm}", "fixed_control"))

    primary_mapping = {
        "normal": "h5",
        "crisis": "h5",
        "grind_up": "top30",
        "squeeze": "top30",
        "slow_bleed": "top30",
        "unknown": "top30",
    }
    robust_mapping = {regime: ("h10" if arm == "h5" else arm) for regime, arm in primary_mapping.items()}
    weekly_runs.append(
        _apply_regime_gate(
            panel,
            primary_mapping,
            "top30",
            "Regime rule: h5 normal/crisis, top30 grind-up/squeeze",
            "full_sample_hypothesis",
        )
    )
    weekly_runs.append(
        _apply_regime_gate(
            panel,
            robust_mapping,
            "top30",
            "Regime rule robustness: h10 normal/crisis, top30 grind-up/squeeze",
            "full_sample_hypothesis",
        )
    )

    weekly_all = pd.concat(weekly_runs, ignore_index=True)
    score_rows = [
        _regime_gate_score_row(rule, str(group["rule_type"].iloc[0]), group)
        for rule, group in weekly_all.groupby("rule", sort=False)
    ]
    scorecard = pd.DataFrame(score_rows)
    scorecard["cagr_lift_vs_period_top30_pct"] = np.nan
    scorecard["cagr_lift_vs_period_h5_pct"] = np.nan
    scorecard["cagr_lift_vs_period_h10_pct"] = np.nan
    period_mask = scorecard["rule_type"].isin({"fixed_control", "full_sample_hypothesis"})
    for arm, control_name in {"top30": "Fixed top30", "h5": "Fixed h5", "h10": "Fixed h10"}.items():
        control = scorecard[scorecard["rule"].eq(control_name)]
        if control.empty:
            continue
        scorecard.loc[period_mask, f"cagr_lift_vs_period_{arm}_pct"] = (
            scorecard.loc[period_mask, "cagr_pct"] - float(control.iloc[0]["cagr_pct"])
        )

    regime_rows: list[dict[str, Any]] = []
    for regime, group in panel.groupby("regime", sort=True):
        row: dict[str, Any] = {"regime": regime, "n_dates": int(len(group))}
        for arm in REGIME_GATE_ARMS:
            row[f"{arm}_avg_weekly_pct"] = float(group[f"{arm}_capital_c2c"].mean() * 100.0)
        row["h5_minus_top30_annualized_pct"] = float(
            (group["h5_capital_c2c"] - group["top30_capital_c2c"]).mean() * PERIODS_PER_YEAR * 100.0
        )
        row["h5_beats_top30_dates_pct"] = float(
            (group["h5_capital_c2c"] > group["top30_capital_c2c"]).mean() * 100.0
        )
        row["h10_minus_top30_annualized_pct"] = float(
            (group["h10_capital_c2c"] - group["top30_capital_c2c"]).mean() * PERIODS_PER_YEAR * 100.0
        )
        row["h10_beats_top30_dates_pct"] = float(
            (group["h10_capital_c2c"] > group["top30_capital_c2c"]).mean() * 100.0
        )
        row["best_full_sample_arm"] = max(
            REGIME_GATE_ARMS,
            key=lambda arm: group[f"{arm}_capital_c2c"].mean(),
        )
        regime_rows.append(row)
    by_regime = pd.DataFrame(regime_rows)

    year_rows: list[dict[str, Any]] = []
    for (rule, year), group in weekly_all.groupby(["rule", "year"], sort=True):
        year_rows.append(
            {
                "rule": rule,
                "rule_type": str(group["rule_type"].iloc[0]),
                "year": int(year),
                "h5_usage_pct": float(group["selected_arm"].eq("h5").mean() * 100.0),
                "h10_usage_pct": float(group["selected_arm"].eq("h10").mean() * 100.0),
                **metric_values(group["capital_c2c"]),
            }
        )
    by_year = pd.DataFrame(year_rows)

    details = empty
    placebo = empty

    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_specs = (
        (
            "Regime rule: h5 normal/crisis, top30 grind-up/squeeze",
            "Fixed top30",
            "Full-sample readable rule vs defensive top30",
        ),
        (
            "Regime rule: h5 normal/crisis, top30 grind-up/squeeze",
            "Fixed h5",
            "Full-sample readable rule vs always-on h5",
        ),
    )
    for idx, (target_name, control_name, comparison) in enumerate(bootstrap_specs):
        target = weekly_all[weekly_all["rule"].eq(target_name)]
        control = weekly_all[weekly_all["rule"].eq(control_name)]
        if target.empty or control.empty:
            continue
        row = _paired_regime_gate_bootstrap(
            target,
            control,
            comparison,
            n_sims=random_sims,
            seed=173 + idx,
        )
        if row:
            bootstrap_rows.append(row)
    gate_bootstrap = pd.DataFrame(bootstrap_rows)

    scorecard.to_csv(paths["regime_gate_scorecard"], index=False)
    by_regime.to_csv(paths["regime_gate_by_regime"], index=False)
    by_year.to_csv(paths["regime_gate_by_year"], index=False)
    details.to_csv(paths["regime_gate_walk_forward"], index=False)
    weekly_all.to_csv(paths["regime_gate_weekly"], index=False)
    placebo.to_csv(paths["regime_gate_placebo"], index=False)
    gate_bootstrap.to_csv(paths["regime_gate_bootstrap"], index=False)
    mapping_sweep.to_csv(paths["regime_mapping_sweep"], index=False)
    mapping_champions.to_csv(paths["regime_mapping_champions"], index=False)
    mapping_champion_years.to_csv(paths["regime_mapping_champion_years"], index=False)
    mapping_search_placebo.to_csv(paths["regime_mapping_search_placebo"], index=False)
    return {
        "regime_gate_scorecard": scorecard,
        "regime_gate_by_regime": by_regime,
        "regime_gate_by_year": by_year,
        "regime_gate_walk_forward": details,
        "regime_gate_weekly": weekly_all,
        "regime_gate_placebo": placebo,
        "regime_gate_bootstrap": gate_bootstrap,
        "regime_mapping_sweep": mapping_sweep,
        "regime_mapping_champions": mapping_champions,
        "regime_mapping_champion_years": mapping_champion_years,
        "regime_mapping_search_placebo": mapping_search_placebo,
    }


def tertile_bucket(series: pd.Series, low_label: str, mid_label: str, high_label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series("unknown", index=series.index, dtype=object)
    valid = numeric.dropna()
    if valid.empty:
        return out
    if valid.nunique() < 3:
        out.loc[numeric.notna()] = "single_bucket"
        return out
    q1, q2 = valid.quantile([1.0 / 3.0, 2.0 / 3.0]).to_numpy()
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 >= q2:
        out.loc[numeric.notna()] = "single_bucket"
        return out
    buckets = pd.cut(
        numeric.loc[numeric.notna()],
        bins=[-np.inf, q1, q2, np.inf],
        labels=[low_label, mid_label, high_label],
        include_lowest=True,
    )
    out.loc[buckets.index] = buckets.astype(str)
    return out


def load_spy_market_context(signal_dates: pd.Series | pd.Index | list[Any]) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(pd.Index(signal_dates).dropna().unique()), name="date").sort_values()
    context = pd.DataFrame({"date": dates})
    if context.empty or not SPY_SIGNAL_DATA_PATH.exists():
        return context
    spy = pd.read_csv(SPY_SIGNAL_DATA_PATH, parse_dates=["date"]).sort_values("date")
    spy["spy_ma200"] = pd.to_numeric(spy["adj_close"], errors="coerce").rolling(200, min_periods=200).mean()
    spy["spy_trail5_pct"] = pd.to_numeric(spy["adj_close"], errors="coerce").pct_change(5) * 100.0
    spy["spy_200d_bucket"] = np.where(
        pd.to_numeric(spy["adj_close"], errors="coerce") >= spy["spy_ma200"],
        "above_200d",
        "below_200d",
    )
    spy.loc[spy["spy_ma200"].isna(), "spy_200d_bucket"] = "unknown"
    spy["vix_percentile_252d"] = pd.to_numeric(spy.get("vix_percentile_252d"), errors="coerce")
    spy["vix_pctl_bucket"] = pd.cut(
        spy["vix_percentile_252d"],
        bins=[-np.inf, 33.0, 66.0, np.inf],
        labels=["low_vix_pctl", "mid_vix_pctl", "high_vix_pctl"],
        include_lowest=True,
    ).astype(object)
    spy["vix_pctl_bucket"] = spy["vix_pctl_bucket"].fillna("unknown")
    spy["spy_trail5_bucket"] = pd.cut(
        spy["spy_trail5_pct"],
        bins=[-np.inf, -2.0, 0.0, 2.0, np.inf],
        labels=["spy_selloff_lt_-2pct", "spy_mild_down", "spy_mild_up", "spy_rally_gt_2pct"],
        include_lowest=True,
    ).astype(object)
    spy["spy_trail5_bucket"] = spy["spy_trail5_bucket"].fillna("unknown")
    cols = [
        "date",
        "adj_close",
        "spy_ma200",
        "spy_200d_bucket",
        "spy_trail5_pct",
        "spy_trail5_bucket",
        "vix_level",
        "vix_percentile_252d",
        "vix_pctl_bucket",
        "realized_vol_10d",
        "range_pos_55d",
        "rsi_5d",
    ]
    available = [col for col in cols if col in spy.columns]
    return pd.merge_asof(
        context.sort_values("date"),
        spy[available].sort_values("date"),
        on="date",
        direction="backward",
    )


def load_russell_market_breadth(signal_dates: pd.Series | pd.Index | list[Any]) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(pd.Index(signal_dates).dropna().unique()), name="date").sort_values()
    context = pd.DataFrame({"date": dates})
    if context.empty or not RUSSELL_WIDE_PATH.exists():
        return context
    try:
        import pyarrow.parquet as pq

        names = pq.ParquetFile(RUSSELL_WIDE_PATH).schema_arrow.names
        close_cols = [name for name in names if name.endswith("_Close")]
        if not close_cols:
            return context
        read_cols = (["Date"] if "Date" in names else []) + close_cols
        closes = pd.read_parquet(RUSSELL_WIDE_PATH, columns=read_cols)
    except Exception as exc:
        print(f"Skipping Russell breadth context: {exc}")
        return context

    if "Date" in closes.columns:
        date_index = pd.to_datetime(closes.pop("Date"))
    else:
        date_index = pd.to_datetime(closes.index)
    closes.index = date_index
    closes = closes.sort_index()
    trail5 = closes.pct_change(5, fill_method=None) * 100.0
    valid = trail5.notna()
    coverage = valid.sum(axis=1).replace(0, np.nan)
    breadth = pd.DataFrame(
        {
            "date": trail5.index,
            "market_breadth_5d_up_pct": ((trail5 > 0.0) & valid).sum(axis=1) / coverage * 100.0,
            "market_breadth_5d_down_pct": ((trail5 < 0.0) & valid).sum(axis=1) / coverage * 100.0,
            "market_avg_trail5_pct": trail5.mean(axis=1),
            "market_median_trail5_pct": trail5.median(axis=1),
            "market_breadth_coverage_names": coverage,
        }
    )
    return pd.merge_asof(
        context.sort_values("date"),
        breadth.sort_values("date"),
        on="date",
        direction="backward",
    )


def build_candidate_signal_context(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or "date" not in panel.columns:
        return pd.DataFrame()
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["trail_abs_pct"] = pd.to_numeric(data["trail_ret"], errors="coerce").abs() * 100.0
    data["side_pnl_c2c"] = np.where(data["pool_side"].eq("SHORT"), -data["fwd_c2c"], data["fwd_c2c"])
    rows: list[dict[str, Any]] = []
    for date, group in data.groupby("date", sort=True):
        long_group = group[group["pool_side"].eq("LONG")]
        short_group = group[group["pool_side"].eq("SHORT")]
        rows.append(
            {
                "date": date,
                "candidate_avg_abs_trail5_pct": float(group["trail_abs_pct"].mean()),
                "long_pool_avg_abs_trail5_pct": float(long_group["trail_abs_pct"].mean()),
                "short_pool_avg_abs_trail5_pct": float(short_group["trail_abs_pct"].mean()),
                "candidate_forward_raw_up_pct": float((pd.to_numeric(group["fwd_c2c"], errors="coerce") > 0.0).mean() * 100.0),
                "candidate_forward_side_hit_pct": float((pd.to_numeric(group["side_pnl_c2c"], errors="coerce") > 0.0).mean() * 100.0),
                "candidate_forward_side_pnl_pct": float(pd.to_numeric(group["side_pnl_c2c"], errors="coerce").mean() * 100.0),
                "long_pool_fwd_c2c_pct": float(pd.to_numeric(long_group["fwd_c2c"], errors="coerce").mean() * 100.0),
                "short_pool_raw_fwd_c2c_pct": float(pd.to_numeric(short_group["fwd_c2c"], errors="coerce").mean() * 100.0),
                "short_pool_side_pnl_pct": float((-pd.to_numeric(short_group["fwd_c2c"], errors="coerce")).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def aggregate_failure_dimension(data: pd.DataFrame, dimension_col: str, dimension_label: str) -> pd.DataFrame:
    if data.empty or dimension_col not in data.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for bucket, group in data.dropna(subset=[dimension_col]).groupby(dimension_col, sort=True):
        if group.empty:
            continue
        def mean_col(col: str) -> float:
            if col not in group.columns:
                return np.nan
            return float(pd.to_numeric(group[col], errors="coerce").mean())

        rows.append(
            {
                "dimension": dimension_label,
                "bucket": str(bucket),
                "n_dates": int(len(group)),
                "target_avg_weekly_pct": float(group["target_capital_c2c"].mean() * 100.0),
                "v2_avg_weekly_pct": float(group["v2_capital_c2c"].mean() * 100.0),
                "avg_excess_weekly_pct": float(group["excess_c2c"].mean() * 100.0),
                "annualized_excess_pct": float(group["excess_c2c"].mean() * PERIODS_PER_YEAR * 100.0),
                "excess_hit_rate_pct": float((group["excess_c2c"] > 0.0).mean() * 100.0),
                "target_bucket_cagr_pct": safe_cagr(group["target_capital_c2c"]),
                "v2_bucket_cagr_pct": safe_cagr(group["v2_capital_c2c"]),
                "target_bucket_sharpe": metric_values(group["target_capital_c2c"])["sharpe"],
                "v2_bucket_sharpe": metric_values(group["v2_capital_c2c"])["sharpe"],
                "avg_long_excess_contrib_pct": float(group["long_excess_contrib"].mean() * 100.0),
                "avg_short_excess_contrib_pct": float(group["short_excess_contrib"].mean() * 100.0),
                "worst_excess_weekly_pct": float(group["excess_c2c"].min() * 100.0),
                "avg_spy_trail5_pct": mean_col("spy_trail5_pct"),
                "avg_vix_percentile_252d": mean_col("vix_percentile_252d"),
                "avg_market_breadth_5d_up_pct": mean_col("market_breadth_5d_up_pct"),
                "avg_candidate_abs_trail5_pct": mean_col("candidate_avg_abs_trail5_pct"),
                "avg_candidate_forward_side_hit_pct": mean_col("candidate_forward_side_hit_pct"),
            }
        )
    return pd.DataFrame(rows)


def save_anon_exp_vs_v2_regime_failure(
    weekly_returns: pd.DataFrame,
    panel: pd.DataFrame,
    output_dir: Path,
    target_run: str = "same::score_v2_anon__exp60_h5",
    v2_run: str = "same::score_v2_visible__eq30",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame()
    breakdown_path = output_dir / "anon_exp_vs_v2_regime_breakdown.csv"
    worst_path = output_dir / "anon_exp_vs_v2_failure_dates.csv"
    context_path = output_dir / "anon_exp_vs_v2_weekly_context.csv"
    if weekly_returns.empty:
        empty.to_csv(breakdown_path, index=False)
        empty.to_csv(worst_path, index=False)
        empty.to_csv(context_path, index=False)
        return empty, empty, empty

    target = weekly_returns[weekly_returns["run"].eq(target_run)].copy()
    control = weekly_returns[weekly_returns["run"].eq(v2_run)].copy()
    if target.empty or control.empty:
        empty.to_csv(breakdown_path, index=False)
        empty.to_csv(worst_path, index=False)
        empty.to_csv(context_path, index=False)
        return empty, empty, empty

    target["date"] = pd.to_datetime(target["date"])
    control["date"] = pd.to_datetime(control["date"])
    keep = ["date", "long_raw_c2c", "short_raw_c2c", "capital_c2c", "regime"]
    data = target[keep].rename(
        columns={
            "long_raw_c2c": "target_long_raw_c2c",
            "short_raw_c2c": "target_short_raw_c2c",
            "capital_c2c": "target_capital_c2c",
            "regime": "market_regime",
        }
    ).merge(
        control[keep].rename(
            columns={
                "long_raw_c2c": "v2_long_raw_c2c",
                "short_raw_c2c": "v2_short_raw_c2c",
                "capital_c2c": "v2_capital_c2c",
                "regime": "v2_regime",
            }
        ),
        on="date",
        how="inner",
    )
    if data.empty:
        empty.to_csv(breakdown_path, index=False)
        empty.to_csv(worst_path, index=False)
        empty.to_csv(context_path, index=False)
        return empty, empty, empty

    data["year"] = data["date"].dt.year.astype(str)
    data["excess_c2c"] = data["target_capital_c2c"] - data["v2_capital_c2c"]
    data["long_excess_contrib"] = CAPITAL_SCALE * (data["target_long_raw_c2c"] - data["v2_long_raw_c2c"])
    data["short_excess_contrib"] = CAPITAL_SCALE * ((-data["target_short_raw_c2c"]) - (-data["v2_short_raw_c2c"]))
    data["target_short_side_pnl_c2c"] = -data["target_short_raw_c2c"]
    data["v2_short_side_pnl_c2c"] = -data["v2_short_raw_c2c"]
    data["weekly_excess_outcome"] = np.where(data["excess_c2c"] >= 0.0, "anon_exp_beat_v2", "anon_exp_lagged_v2")

    market_context = load_spy_market_context(data["date"])
    breadth_context = load_russell_market_breadth(data["date"])
    candidate_context = build_candidate_signal_context(panel)
    for context in (market_context, breadth_context, candidate_context):
        if not context.empty:
            context = context.copy()
            context["date"] = pd.to_datetime(context["date"])
            data = data.merge(context, on="date", how="left")

    if "market_breadth_5d_up_pct" in data.columns:
        data["market_breadth_bucket"] = tertile_bucket(
            data["market_breadth_5d_up_pct"],
            "low_market_breadth",
            "mid_market_breadth",
            "high_market_breadth",
        )
    if "candidate_avg_abs_trail5_pct" in data.columns:
        data["candidate_move_strength_bucket"] = tertile_bucket(
            data["candidate_avg_abs_trail5_pct"],
            "low_candidate_move",
            "mid_candidate_move",
            "high_candidate_move",
        )
    if "candidate_forward_side_hit_pct" in data.columns:
        data["candidate_forward_breadth_bucket"] = tertile_bucket(
            data["candidate_forward_side_hit_pct"],
            "low_forward_candidate_breadth",
            "mid_forward_candidate_breadth",
            "high_forward_candidate_breadth",
        )

    for source_col, pct_col in {
        "target_capital_c2c": "target_weekly_pct",
        "v2_capital_c2c": "v2_weekly_pct",
        "excess_c2c": "excess_weekly_pct",
        "long_excess_contrib": "long_excess_contrib_pct",
        "short_excess_contrib": "short_excess_contrib_pct",
        "target_long_raw_c2c": "target_long_raw_c2c_pct",
        "v2_long_raw_c2c": "v2_long_raw_c2c_pct",
        "target_short_side_pnl_c2c": "target_short_side_pnl_c2c_pct",
        "v2_short_side_pnl_c2c": "v2_short_side_pnl_c2c_pct",
    }.items():
        if source_col in data.columns:
            data[pct_col] = pd.to_numeric(data[source_col], errors="coerce") * 100.0

    dimensions = [
        ("weekly_excess_outcome", "Weekly Excess Outcome"),
        ("year", "Calendar Year"),
        ("market_regime", "Market Regime"),
        ("spy_200d_bucket", "SPY vs 200D"),
        ("vix_pctl_bucket", "VIX Percentile"),
        ("spy_trail5_bucket", "SPY Prior 5D Move"),
        ("market_breadth_bucket", "Russell Prior 5D Breadth"),
        ("candidate_move_strength_bucket", "Candidate Prior 5D Move Strength"),
        ("candidate_forward_breadth_bucket", "Candidate Forward Breadth (Post-Hoc)"),
    ]
    breakdown_parts = [aggregate_failure_dimension(data, col, label) for col, label in dimensions]
    breakdown_parts = [part for part in breakdown_parts if not part.empty]
    breakdown = pd.concat(breakdown_parts, ignore_index=True) if breakdown_parts else empty
    if not breakdown.empty:
        breakdown = breakdown.sort_values(["dimension", "annualized_excess_pct"], ascending=[True, False])

    worst_cols = [
        "date",
        "year",
        "market_regime",
        "spy_200d_bucket",
        "vix_percentile_252d",
        "vix_pctl_bucket",
        "spy_trail5_pct",
        "market_breadth_5d_up_pct",
        "candidate_avg_abs_trail5_pct",
        "target_weekly_pct",
        "v2_weekly_pct",
        "excess_weekly_pct",
        "long_excess_contrib_pct",
        "short_excess_contrib_pct",
        "target_long_raw_c2c_pct",
        "v2_long_raw_c2c_pct",
        "target_short_side_pnl_c2c_pct",
        "v2_short_side_pnl_c2c_pct",
    ]
    available_worst_cols = [col for col in worst_cols if col in data.columns]
    worst = data.sort_values("excess_c2c", ascending=True)[available_worst_cols].head(25)

    data.to_csv(context_path, index=False)
    breakdown.to_csv(breakdown_path, index=False)
    worst.to_csv(worst_path, index=False)
    return breakdown, worst, data


def build_offline_diagnostics(
    output_dir: Path,
    panel: pd.DataFrame | None = None,
    weekly_returns: pd.DataFrame | None = None,
    random_sims: int = 2000,
) -> dict[str, pd.DataFrame]:
    panel_path = output_dir / "candidate_panel.parquet"
    weekly_path = output_dir / "weekly_returns.parquet"
    if panel is None:
        panel = pd.read_parquet(panel_path) if panel_path.exists() else pd.DataFrame()
    if weekly_returns is None:
        weekly_returns = pd.read_parquet(weekly_path) if weekly_path.exists() else pd.DataFrame()
    if panel.empty:
        empty = pd.DataFrame()
        short_rank_slices, short_rank_null = build_short_rank_audit(panel)
        diagnostics = {
            "rank_vs_score": empty,
            "agreement": empty,
            "consensus": empty,
            "monotonicity": empty,
            "disagreement_risk": empty,
            "directional_disagreement": empty,
            "disagreement_penalty": empty,
            "random_placebo": empty,
            "pool_selector": empty,
            "anon_exp_regime_breakdown": empty,
            "anon_exp_failure_dates": empty,
            "top10_exponential": build_top10_exponential_summary(weekly_returns),
            "short_rank_slices": short_rank_slices,
            "short_rank_search_null": short_rank_null,
        }
        diagnostics["top10_exponential"].to_csv(output_dir / "top10_exponential_weighting.csv", index=False)
        diagnostics["short_rank_slices"].to_csv(output_dir / "short_rank_slice_audit.csv", index=False)
        diagnostics["short_rank_search_null"].to_csv(
            output_dir / "short_rank_concentration_search_null.csv", index=False
        )
        diagnostics.update(save_regime_gate_experiments(weekly_returns, output_dir, random_sims=random_sims))
        return diagnostics
    research_panel = add_research_scores(panel)
    diagnostics = {
        "rank_vs_score": save_rank_vs_score_tests(research_panel, output_dir),
        "agreement": save_visible_anon_agreement(research_panel, output_dir),
        "consensus": save_consensus_rank_tests(research_panel, output_dir),
        "monotonicity": save_rank_monotonicity(research_panel, output_dir),
    }
    disagreement_risk, disagreement_penalty = save_disagreement_tests(research_panel, output_dir)
    diagnostics["disagreement_risk"] = disagreement_risk
    diagnostics["directional_disagreement"] = save_directional_disagreement(research_panel, output_dir)
    diagnostics["disagreement_penalty"] = disagreement_penalty
    diagnostics["random_placebo"] = save_random_placebo(research_panel, output_dir, n_sims=random_sims)
    diagnostics["pool_selector"] = save_pool_selector_decomposition(research_panel, output_dir)
    diagnostics["top10_exponential"] = build_top10_exponential_summary(weekly_returns)
    diagnostics["top10_exponential"].to_csv(output_dir / "top10_exponential_weighting.csv", index=False)
    short_rank_slices, short_rank_null = build_short_rank_audit(research_panel)
    diagnostics["short_rank_slices"] = short_rank_slices
    diagnostics["short_rank_search_null"] = short_rank_null
    short_rank_slices.to_csv(output_dir / "short_rank_slice_audit.csv", index=False)
    short_rank_null.to_csv(output_dir / "short_rank_concentration_search_null.csv", index=False)
    regime_breakdown, failure_dates, _ = save_anon_exp_vs_v2_regime_failure(weekly_returns, research_panel, output_dir)
    diagnostics["anon_exp_regime_breakdown"] = regime_breakdown
    diagnostics["anon_exp_failure_dates"] = failure_dates
    diagnostics.update(save_regime_gate_experiments(weekly_returns, output_dir, random_sims=random_sims))
    return diagnostics


def _diagnostic_filename(key: str) -> str:
    return {
        "rank_vs_score": "rank_vs_raw_score_tests.csv",
        "agreement": "visible_anon_top10_agreement.csv",
        "consensus": "consensus_rank_tests.csv",
        "monotonicity": "rank_monotonicity.csv",
        "disagreement_risk": "visible_anon_disagreement_risk.csv",
        "directional_disagreement": "directional_visible_anon_disagreement.csv",
        "disagreement_penalty": "disagreement_penalty_strategy.csv",
        "random_placebo": "random_ranking_placebo.csv",
        "pool_selector": "candidate_pool_selector_decomposition.csv",
        "walk_forward": "walk_forward_parameter_selection.csv",
        "anon_exp_regime_breakdown": "anon_exp_vs_v2_regime_breakdown.csv",
        "anon_exp_failure_dates": "anon_exp_vs_v2_failure_dates.csv",
        "anon_exp_weekly_context": "anon_exp_vs_v2_weekly_context.csv",
        "regime_gate_scorecard": "regime_gate_scorecard.csv",
        "regime_gate_by_regime": "regime_gate_arm_performance.csv",
        "regime_gate_by_year": "regime_gate_year_metrics.csv",
        "regime_gate_walk_forward": "regime_gate_walk_forward_choices.csv",
        "regime_gate_weekly": "regime_gate_weekly_returns.csv",
        "regime_gate_placebo": "regime_gate_random_timing_placebo.csv",
        "regime_gate_bootstrap": "regime_gate_paired_bootstrap.csv",
        "regime_mapping_sweep": "regime_mapping_development_sweep.csv",
        "regime_mapping_champions": "regime_mapping_champions.csv",
        "regime_mapping_champion_years": "regime_mapping_champion_year_metrics.csv",
        "regime_mapping_search_placebo": "regime_mapping_search_placebo.csv",
        "top10_exponential": "top10_exponential_weighting.csv",
        "short_rank_slices": "short_rank_slice_audit.csv",
        "short_rank_search_null": "short_rank_concentration_search_null.csv",
    }.get(key, f"{key}.csv")


def write_html_report(
    output_dir: Path,
    scorecard: pd.DataFrame,
    side_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    top_by_family: pd.DataFrame,
    control_checks: dict[str, Any],
    side_hybrid_sections: list[dict[str, Any]] | None = None,
    exp_2x2: pd.DataFrame | None = None,
    diagnostics: dict[str, pd.DataFrame] | None = None,
) -> Path:
    def display_label(col: str) -> str:
        labels = {
            "cagr_pct": "CAGR",
            "sharpe": "Sharpe",
            "max_drawdown_pct": "Max DD",
            "total_return_pct": "Total",
            "cagr_lift_vs_v2_pct": "Lift vs V2",
            "lift_vs_trail5_exp_pct": "Lift vs Trail5 Exp",
            "cagr_lift_vs_same_score_eq10_pct": "Lift vs Equal 10/10",
            "cagr_lift_vs_trail5_same_weight_pct": "Lift vs Trail5",
            "cagr_lift_pp": "CAGR Lift",
            "min_period_cagr_pct": "Worst-Period CAGR",
            "mean_period_cagr_pct": "Mean Period CAGR",
            "min_period_lift_pp": "Worst-Period Lift",
            "cagr_2013_pct": "2013 CAGR",
            "worst_year_cagr_pct": "Worst-Year CAGR",
            "avg_long_raw_c2c_pct": "Long Avg",
            "avg_short_side_pnl_pct": "Short P&L",
            "avg_side_pnl_c2c_pct": "Side P&L",
            "side_hit_rate_pct": "Hit Rate",
            "side_cagr_pct": "Standalone CAGR",
            "avg_weekly_side_pnl_pct": "Avg Weekly Short P&L",
            "worst_week_pct": "Worst Week",
            "observed_best_cagr_pct": "Observed Best CAGR",
            "random_best_cagr_p50_pct": "Random Best Median",
            "random_best_cagr_p95_pct": "Random Best 95%",
            "random_best_cagr_p99_pct": "Random Best 99%",
            "observed_cagr_percentile_pct": "Observed Percentile",
            "one_sided_p_pct": "Search-Adjusted p",
            "observed_best_avg_weekly_pct": "Observed Best Avg Week",
            "random_best_avg_weekly_p50_pct": "Random Best Avg Week Median",
            "random_best_avg_weekly_p95_pct": "Random Best Avg Week 95%",
            "real_cagr_percentile_pct": "Random Percentile",
            "real_cagr_pct": "Real CAGR",
            "random_p50_cagr_pct": "Random Median",
            "random_p95_cagr_pct": "Random 95%",
            "random_max_cagr_pct": "Random Max",
            "target_avg_weekly_pct": "Anon Exp Avg Wk",
            "v2_avg_weekly_pct": "V2 Avg Wk",
            "avg_excess_weekly_pct": "Avg Excess Wk",
            "annualized_excess_pct": "Ann Excess",
            "excess_hit_rate_pct": "Excess Hit",
            "target_bucket_cagr_pct": "Anon Exp CAGR",
            "v2_bucket_cagr_pct": "V2 CAGR",
            "target_bucket_sharpe": "Anon Exp Sharpe",
            "v2_bucket_sharpe": "V2 Sharpe",
            "avg_long_excess_contrib_pct": "Long Excess",
            "avg_short_excess_contrib_pct": "Short Excess",
            "worst_excess_weekly_pct": "Worst Excess Wk",
            "avg_spy_trail5_pct": "Avg SPY Prior 5D",
            "avg_vix_percentile_252d": "Avg VIX Pctl",
            "avg_market_breadth_5d_up_pct": "Avg Russell Breadth",
            "avg_candidate_abs_trail5_pct": "Avg Candidate Move",
            "avg_candidate_forward_side_hit_pct": "Forward Side Hit",
            "target_weekly_pct": "Anon Exp Wk",
            "v2_weekly_pct": "V2 Wk",
            "excess_weekly_pct": "Excess Wk",
            "long_excess_contrib_pct": "Long Excess",
            "short_excess_contrib_pct": "Short Excess",
            "target_long_raw_c2c_pct": "Anon Long",
            "v2_long_raw_c2c_pct": "V2 Long",
            "target_short_side_pnl_c2c_pct": "Anon Short P&L",
            "v2_short_side_pnl_c2c_pct": "V2 Short P&L",
            "market_breadth_5d_up_pct": "Russell Breadth",
            "candidate_avg_abs_trail5_pct": "Candidate Move",
            "spy_trail5_pct": "SPY Prior 5D",
            "vix_percentile_252d": "VIX Pctl",
        }
        return labels.get(col, col.replace("_", " ").title())

    def is_pct_col(col: str) -> bool:
        return col.endswith("_pct") or col.endswith("_pp") or col in {
            "cagr_pct",
            "max_drawdown_pct",
            "total_return_pct",
            "avg_long_raw_c2c_pct",
            "avg_short_side_pnl_pct",
            "mean_annualized_delta_pct",
            "ci_low_pct",
            "ci_high_pct",
            "target_weekly_pct",
            "v2_weekly_pct",
            "excess_weekly_pct",
            "long_excess_contrib_pct",
            "short_excess_contrib_pct",
            "target_long_raw_c2c_pct",
            "v2_long_raw_c2c_pct",
            "target_short_side_pnl_c2c_pct",
            "v2_short_side_pnl_c2c_pct",
            "market_breadth_5d_up_pct",
            "candidate_avg_abs_trail5_pct",
            "spy_trail5_pct",
            "vix_percentile_252d",
            "avg_vix_percentile_252d",
        }

    def table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
        if df.empty:
            return "<p class='muted'>No rows.</p>"
        data = df[columns].head(limit).copy() if limit else df[columns].copy()
        formatted = data.copy()
        for col in formatted.columns:
            if is_pct_col(col):
                formatted[col] = formatted[col].map(lambda x: fmt_pct(x, 2))
            elif col in {
                "sharpe",
                "p_two_sided",
                "cagr_lift_vs_v2_pct",
                "calmar_ratio",
                "observed_value",
                "shuffled_best_p50",
                "shuffled_best_p95",
                "shuffled_best_p99",
                "shuffled_best_max",
            } or col.endswith("_sharpe") or col.endswith("sharpe"):
                formatted[col] = formatted[col].map(lambda x: fmt_num(x, 3))
            elif col.startswith("avg_") or col.endswith("_rank") or col.endswith("_diff"):
                formatted[col] = formatted[col].map(lambda x: fmt_num(x, 2))
            else:
                formatted[col] = formatted[col].astype(str)

        best_by_col: dict[str, float] = {}
        high_is_good = {
            "cagr_pct",
            "sharpe",
            "total_return_pct",
            "cagr_lift_vs_v2_pct",
            "lift_vs_trail5_exp_pct",
            "avg_long_raw_c2c_pct",
            "avg_short_side_pnl_pct",
            "avg_side_pnl_c2c_pct",
            "side_hit_rate_pct",
            "side_cagr_pct",
            "real_cagr_percentile_pct",
            "real_cagr_pct",
            "cagr_lift_vs_same_score_eq10_pct",
            "cagr_lift_vs_trail5_same_weight_pct",
        }
        low_abs_is_good = {"p_two_sided", "one_sided_p_pct"}
        for col in data.columns:
            numeric = pd.to_numeric(data[col], errors="coerce")
            if numeric.notna().sum() == 0:
                continue
            if col == "max_drawdown_pct":
                best_by_col[col] = float(numeric.max())
            elif col in high_is_good:
                best_by_col[col] = float(numeric.max())
            elif col in low_abs_is_good:
                best_by_col[col] = float(numeric.min())

        head = "".join(f"<th>{html.escape(display_label(str(col)))}</th>" for col in formatted.columns)
        body = []
        for idx, row in formatted.iterrows():
            cells = []
            for col, value in row.items():
                raw = pd.to_numeric(data.loc[idx, col], errors="coerce")
                classes = []
                if is_pct_col(str(col)) or str(col) == "sharpe" or str(col).endswith("_sharpe"):
                    classes.append("metric")
                    if pd.notna(raw):
                        if str(col) == "max_drawdown_pct":
                            classes.append("bad" if raw < -10 else "ok")
                        elif str(col) == "one_sided_p_pct":
                            classes.append("good" if raw < 5.0 else "bad")
                        elif raw > 0:
                            classes.append("good")
                        elif raw < 0:
                            classes.append("bad")
                if str(col) in best_by_col and pd.notna(raw) and np.isclose(float(raw), best_by_col[str(col)], rtol=1e-9, atol=1e-9):
                    classes.append("best")
                class_attr = f" class='{' '.join(classes)}'" if classes else ""
                cells.append(f"<td{class_attr}>{html.escape(str(value))}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    def pick_row(data: pd.DataFrame | None, **criteria: Any) -> pd.Series | None:
        if data is None or data.empty:
            return None
        mask = pd.Series(True, index=data.index)
        for col, expected in criteria.items():
            if col not in data.columns:
                return None
            mask &= data[col].astype(str).eq(str(expected))
        matches = data[mask]
        if matches.empty:
            return None
        return matches.iloc[0]

    def row_pct(row: pd.Series | None, col: str, digits: int = 1) -> str:
        if row is None or col not in row:
            return "n/a"
        return fmt_pct(row[col], digits)

    def row_num(row: pd.Series | None, col: str, digits: int = 2) -> str:
        if row is None or col not in row:
            return "n/a"
        return fmt_num(row[col], digits)

    def insight_card(title: str, thesis: str, evidence: str, next_step: str, tone: str = "good") -> str:
        label = "Watch" if tone == "warn" else "Finding"
        return f"""
        <article class="insight-item {html.escape(tone)}">
          <div class="insight-index"></div>
          <div class="insight-copy">
            <div class="insight-kicker">{html.escape(label)}</div>
            <h3>{html.escape(title)}</h3>
            <p class="thesis">{html.escape(thesis)}</p>
            <p><strong>Evidence:</strong> {html.escape(evidence)}</p>
            <p class="next-step"><strong>Next:</strong> {html.escape(next_step)}</p>
          </div>
        </article>
        """

    def insights_html(items: dict[str, pd.DataFrame] | None, exp_rows: pd.DataFrame | None) -> str:
        items = items or {}
        rank_score = items.get("rank_vs_score", pd.DataFrame())
        agreement = items.get("agreement", pd.DataFrame())
        directional = items.get("directional_disagreement", pd.DataFrame())
        placebo = items.get("random_placebo", pd.DataFrame())
        pool_selector = items.get("pool_selector", pd.DataFrame())
        mapping_champions = items.get("regime_mapping_champions", pd.DataFrame())
        mapping_search_placebo = items.get("regime_mapping_search_placebo", pd.DataFrame())
        short_rank_slices = items.get("short_rank_slices", pd.DataFrame())
        short_rank_search_null = items.get("short_rank_search_null", pd.DataFrame())

        visible_top30 = pick_row(rank_score, source="Visible V2", test="Rank-only top 30")
        visible_exp = pick_row(rank_score, source="Visible V2", test="Rank-only exponential h10")
        visible_raw = pick_row(rank_score, source="Visible V2", test="Direct raw-score proportional")
        anon_top30 = pick_row(rank_score, source="Anonymized V2", test="Rank-only top 30")
        anon_exp = pick_row(rank_score, source="Anonymized V2", test="Rank-only exponential h10")
        anon_raw = pick_row(rank_score, source="Anonymized V2", test="Direct raw-score proportional")

        long_anon_only = pick_row(agreement, pool_side="LONG", agreement_bucket="anon_only_top10")
        long_both = pick_row(agreement, pool_side="LONG", agreement_bucket="both_top10")
        long_visible_only = pick_row(agreement, pool_side="LONG", agreement_bucket="visible_only_top10")
        long_aligned = pick_row(directional, pool_side="LONG", directional_gap_bucket="aligned_0_5")
        long_anon_21_40 = pick_row(directional, pool_side="LONG", directional_gap_bucket="anon_higher_21_40")
        long_anon_41_plus = pick_row(directional, pool_side="LONG", directional_gap_bucket="anon_higher_41_plus")
        long_visible_21_40 = pick_row(directional, pool_side="LONG", directional_gap_bucket="visible_higher_21_40")

        visible_placebo = pick_row(placebo, weight_model="eq_top30", real_ranking="Visible V2")
        anon_placebo = pick_row(placebo, weight_model="exp_h10", real_ranking="Anonymized V2")

        pool_only = pick_row(pool_selector, decomposition_step="Pool only: equal all 60")
        trail5_top30 = pick_row(pool_selector, decomposition_step="Trail5 selector: top30 equal")
        v2_top30 = pick_row(pool_selector, decomposition_step="V2 selector: top30 equal")
        anon_exp_selector = pick_row(pool_selector, decomposition_step="Anon V2 selector: exponential h10")
        best_mapping_cagr = pick_row(mapping_champions, objective="Best development CAGR")
        best_mapping_sharpe = pick_row(mapping_champions, objective="Best development Sharpe")
        best_mapping_worst_year = pick_row(mapping_champions, objective="Best worst-year CAGR")
        fixed_h5_mapping = pick_row(mapping_champions, objective="Fixed h5 reference")
        mapping_placebo = pick_row(mapping_search_placebo, objective="Best development CAGR")
        mapping_sharpe_placebo = pick_row(mapping_search_placebo, objective="Best development Sharpe")
        mapping_worst_placebo = pick_row(mapping_search_placebo, objective="Best worst-year CAGR")

        llm_long_only = pick_row(exp_rows, llm_source="Ticker-visible V2", half_life=10, long_ranking="LLM", short_ranking="Trail5")
        llm_short_only = pick_row(exp_rows, llm_source="Ticker-visible V2", half_life=10, long_ranking="Trail5", short_ranking="LLM")
        anon_llm_both = pick_row(exp_rows, llm_source="Anonymized V2", half_life=10, long_ranking="LLM", short_ranking="LLM")
        anon_llm_long_only = pick_row(exp_rows, llm_source="Anonymized V2", half_life=10, long_ranking="LLM", short_ranking="Trail5")
        anon_llm_short_only = pick_row(exp_rows, llm_source="Anonymized V2", half_life=10, long_ranking="Trail5", short_ranking="LLM")
        short_exp10_h1 = pick_row(side_summary, side="SHORT", process="score_v2_anon__exp10_h1")
        short_rank1 = pick_row(short_rank_slices, construction="Rank 1 alone")
        short_rank_null = short_rank_search_null.iloc[0] if not short_rank_search_null.empty else None
        n_short_processes = int(side_summary["side"].eq("SHORT").sum()) if "side" in side_summary.columns else 0
        n_positive_short_processes = (
            int((side_summary["side"].eq("SHORT") & side_summary["side_cagr_pct"].gt(0.0)).sum())
            if {"side", "side_cagr_pct"}.issubset(side_summary.columns)
            else 0
        )

        cards = [
            insight_card(
                "Visible model = classifier. Anonymized model = ranker.",
                "Ticker-visible scoring is better at making a broad yes/no distinction; anonymized scoring is better at ordering the highest-conviction ideas.",
                (
                    f"Visible top30 did {row_pct(visible_top30, 'cagr_pct')} CAGR vs visible exponential h10 at "
                    f"{row_pct(visible_exp, 'cagr_pct')}. Anonymized exponential h10 did "
                    f"{row_pct(anon_exp, 'cagr_pct')} vs anonymized top30 at {row_pct(anon_top30, 'cagr_pct')}."
                ),
                "Treat ticker-visible V2 as the conservative selection baseline; test anonymized rank-weighting as the next serious portfolio construction candidate.",
            ),
            insight_card(
                "Anonymized-only conviction was strongest on longs.",
                "When anonymized V2 loved a long but ticker-visible V2 did not, those names had the best forward long returns.",
                (
                    f"LONG anon-only top10 averaged {row_pct(long_anon_only, 'avg_side_pnl_c2c_pct', 2)} C2C, "
                    f"both-top10 averaged {row_pct(long_both, 'avg_side_pnl_c2c_pct', 2)}, and visible-only top10 averaged "
                    f"{row_pct(long_visible_only, 'avg_side_pnl_c2c_pct', 2)}."
                ),
                "Build a long-side rule that overweights anonymized top-rank names, especially when visible V2 is less excited.",
            ),
            insight_card(
                "The 0-100 score magnitude is not trustworthy enough to size directly.",
                "The LLM's ordering contains signal, but proportional raw-score weighting generally weakens the result.",
                (
                    f"Visible rank-only top30 did {row_pct(visible_top30, 'cagr_pct')} CAGR, while visible raw-score weighting did "
                    f"{row_pct(visible_raw, 'cagr_pct')}. Anonymized exp rank did {row_pct(anon_exp, 'cagr_pct')}, while anonymized raw-score weighting did "
                    f"{row_pct(anon_raw, 'cagr_pct')}."
                ),
                "Use ranks, percentile ranks, or buckets for portfolio construction; avoid treating 90 vs 70 as a calibrated probability gap.",
                "warn",
            ),
            insight_card(
                "The LLM is not just riding the reversal pool.",
                "Random-ranking placebo says the selector adds real information inside the same 60/60 candidate pool.",
                (
                    f"Visible V2 top30 landed at the {row_pct(visible_placebo, 'real_cagr_percentile_pct', 1)} percentile vs random top30 rankings. "
                    f"Anonymized exp h10 landed at the {row_pct(anon_placebo, 'real_cagr_percentile_pct', 1)} percentile vs random exponential rankings."
                ),
                "Keep measuring every new selector against random-within-pool, not just Trail5, so we know whether ranking actually matters.",
            ),
            insight_card(
                "Disagreement is directional, not automatically bad.",
                "Moderate cases where anonymized V2 ranks a long much higher than ticker-visible V2 performed better than aligned names.",
                (
                    f"LONG aligned 0-5 rank-gap names averaged {row_pct(long_aligned, 'avg_side_pnl_c2c_pct', 2)} C2C. "
                    f"Anon-higher 21-40 gap names averaged {row_pct(long_anon_21_40, 'avg_side_pnl_c2c_pct', 2)}. "
                    f"Extreme anon-higher 41+ gap names fell back to {row_pct(long_anon_41_plus, 'avg_side_pnl_c2c_pct', 2)}, while visible-higher 21-40 was "
                    f"{row_pct(long_visible_21_40, 'avg_side_pnl_c2c_pct', 2)}."
                ),
                "Replace generic disagreement penalties with a directional rule: overweight moderate anon-over-visible long disagreements, but do not trust extreme disagreements blindly.",
            ),
            insight_card(
                "Most of the next upside is long-side process design.",
                "The clean 2x2 exponential tests say LLM value shows up more clearly on longs than shorts.",
                (
                    f"Ticker-visible exp h10: LLM longs + Trail5 shorts did {row_pct(llm_long_only, 'cagr_pct')} CAGR; "
                    f"Trail5 longs + LLM shorts did {row_pct(llm_short_only, 'cagr_pct')}. "
                    f"Anonymized exp h10: LLM/LLM did {row_pct(anon_llm_both, 'cagr_pct')}; "
                    f"anon longs + Trail5 shorts did {row_pct(anon_llm_long_only, 'cagr_pct')}; "
                    f"Trail5 longs + anon shorts did {row_pct(anon_llm_short_only, 'cagr_pct')}."
                ),
                "Separate the future long prompt/process from the short prompt/process instead of forcing one symmetric LLM ranking rule.",
            ),
            insight_card(
                "The pool matters, but the selector matters more.",
                "The base reversal pool has value, Trail5 selection adds some value, and V2/anonymized selection adds more.",
                (
                    f"Equal all-60 pool did {row_pct(pool_only, 'cagr_pct')}; Trail5 top30 did {row_pct(trail5_top30, 'cagr_pct')}; "
                    f"V2 top30 did {row_pct(v2_top30, 'cagr_pct')}; anonymized exp h10 did {row_pct(anon_exp_selector, 'cagr_pct')}."
                ),
                "Frame future research as pool construction plus selector design plus sizing, not just prompt A/Bs.",
            ),
            insight_card(
                "Regime switching adds only a small amount to maximum CAGR.",
                "The best of all 256 development mappings is nearly always-on h5; it only replaces h5 with top30 during squeeze dates.",
                (
                    f"Best mapping: {'' if best_mapping_cagr is None else str(best_mapping_cagr.get('mapping_id', ''))}. "
                    f"It produced {row_pct(best_mapping_cagr, 'cagr_pct')} CAGR versus fixed h5 at "
                    f"{row_pct(fixed_h5_mapping, 'cagr_pct')}."
                ),
                "Treat regime switching primarily as a risk-allocation tool; it is not a large source of incremental return by itself.",
                "warn",
            ),
            insight_card(
                "The strongest risk-balanced mapping uses a real aggression ladder.",
                "Maximum aggression belongs in crisis, moderate concentration in normal/grind-up, and defensive top30 in squeezes.",
                (
                    f"Best worst-year mapping: {'' if best_mapping_worst_year is None else str(best_mapping_worst_year.get('mapping_id', ''))}. "
                    f"It produced {row_pct(best_mapping_worst_year, 'cagr_pct')} full-period CAGR and "
                    f"{row_pct(best_mapping_worst_year, 'cagr_2013_pct')} in 2013 versus fixed h5 at "
                    f"{row_pct(fixed_h5_mapping, 'cagr_2013_pct')}."
                ),
                "Carry both the maximum-CAGR mapping and this risk-balanced mapping into the final development shortlist.",
            ),
            insight_card(
                "Searching 256 mappings creates a real optimization tax.",
                "The regime-label permutation test asks whether the winning mapping is better than winners found after shuffling the same regime counts.",
                (
                    f"Search-adjusted percentiles were {row_pct(mapping_placebo, 'observed_percentile_pct', 1)} for CAGR, "
                    f"{row_pct(mapping_sharpe_placebo, 'observed_percentile_pct', 1)} for Sharpe, and "
                    f"{row_pct(mapping_worst_placebo, 'observed_percentile_pct', 1)} for worst-year CAGR."
                ),
                "Use this period freely for development, but recognize that small mapping improvements may be search noise until the separate five-year validation.",
                "warn",
            ),
            insight_card(
                "The short result is a rank-1 lead, not six independent confirmations.",
                "The concentration ladder reuses the same names and dates; its clean shape is largely explained by progressively overweighting one profitable rank-1 series.",
                (
                    f"Only {n_positive_short_processes} of {n_short_processes} short processes had positive standalone CAGR: h1 at "
                    f"{row_pct(short_exp10_h1, 'side_cagr_pct', 3)}. Rank 1 alone produced {row_pct(short_rank1, 'cagr_pct', 2)}, "
                    f"but the six-level rank-randomization test placed the observed winner at the "
                    f"{row_num(short_rank_null, 'observed_cagr_percentile_pct', 1)}th percentile with "
                    f"p={row_pct(short_rank_null, 'one_sided_p_pct', 2)}."
                ),
                "Carry rank 1 as a short-specific research hypothesis, not as established alpha; the separate validation period must decide whether it survives.",
                "warn",
            ),
        ]
        return f"""
    <section id="insights">
      <h2>Executive Insights</h2>
      <p>Plain-English research conclusions from the offline diagnostics. These are the guide rails for what to test next.</p>
      <div class="insight-list">{''.join(cards)}</div>
    </section>
        """

    def exp_2x2_html(data: pd.DataFrame | None) -> str:
        cols = [
            "llm_source",
            "half_life",
            "long_ranking",
            "short_ranking",
            "question",
            "cagr_pct",
            "sharpe",
            "max_drawdown_pct",
            "total_return_pct",
            "lift_vs_trail5_exp_pct",
            "avg_long_raw_c2c_pct",
            "avg_short_side_pnl_pct",
        ]
        if data is None or data.empty:
            body = "<p class='muted'>No exponential 2x2 rows found. Re-run the autoresearch sweep if weekly returns are missing.</p>"
        else:
            body = table(data, [col for col in cols if col in data.columns], None)
        return f"""
    <section id="exp2x2">
      <h2>2x2 Exponential Weighting Isolation</h2>
      <p>This isolates ranking value from weighting value. Same anchor 3, same 60/60 candidate pool, same gross fixed 5-day C2C; only the long/short ranking source changes while both legs use exponential rank weighting across 60 candidates.</p>
      <div class="callout">Read: each LLM source gets its own 2x2. LLM/Trail5 isolates long-side value; Trail5/LLM isolates short-side value. The anonymized matrix is the key one for checking where the 19% CAGR came from.</div>
      <div class="table-wrap">{body}</div>
    </section>
        """

    def top10_exponential_html(items: dict[str, pd.DataFrame] | None) -> str:
        items = items or {}
        data = items.get("top10_exponential", pd.DataFrame())
        rank_slices = items.get("short_rank_slices", pd.DataFrame())
        rank_null = items.get("short_rank_search_null", pd.DataFrame())
        if data.empty:
            return ""
        anon = data[data["score_source"].eq("Anonymized V2")].copy()
        anon_equal = pick_row(anon, weight_model="eq10")
        anon_weighted = anon[anon["weight_model"].str.startswith("exp10_h", na=False)]
        best_cagr = anon_weighted.sort_values("cagr_pct", ascending=False).iloc[0] if not anon_weighted.empty else None
        best_sharpe = anon_weighted.sort_values("sharpe", ascending=False).iloc[0] if not anon_weighted.empty else None
        primary = data[
            data["score_source"].isin(["Trail5", "Anonymized V2", "Anonymized V2.1"])
        ].copy()
        primary["source_order"] = primary["score_source"].map(
            {"Trail5": 0, "Anonymized V2": 1, "Anonymized V2.1": 2}
        )
        primary["model_order"] = primary["weight_model"].map(
            {"eq10": 0, **{f"exp10_h{half_life}": idx + 1 for idx, half_life in enumerate(TOP10_EXP_HALF_LIVES)}}
        )
        primary = primary.sort_values(["source_order", "model_order"])
        cols = [
            "score_source",
            "weight_model",
            "cagr_pct",
            "sharpe",
            "max_drawdown_pct",
            "cagr_2013_pct",
            "worst_year_cagr_pct",
            "positive_years",
            "cagr_lift_vs_same_score_eq10_pct",
            "cagr_lift_vs_trail5_same_weight_pct",
        ]
        short_specs = [
            ("Equal top 30", "score_v2_anon__eq30"),
            ("Equal top 10", "score_v2_anon__eq10"),
            ("Exp h10 within top 10", "score_v2_anon__exp10_h10"),
            ("Exp h5 within top 10", "score_v2_anon__exp10_h5"),
            ("Exp h3 within top 10", "score_v2_anon__exp10_h3"),
            ("Exp h2 within top 10", "score_v2_anon__exp10_h2"),
            ("Exp h1 within top 10", "score_v2_anon__exp10_h1"),
        ]
        short_rows: list[dict[str, Any]] = []
        for construction, process in short_specs:
            row = pick_row(side_summary, side="SHORT", process=process)
            if row is None:
                continue
            short_rows.append(
                {
                    "construction": construction,
                    "avg_side_pnl_c2c_pct": row.get("avg_side_pnl_c2c_pct", np.nan),
                    "side_hit_rate_pct": row.get("side_hit_rate_pct", np.nan),
                    "side_cagr_pct": row.get("side_cagr_pct", np.nan),
                }
            )
        short_progression = pd.DataFrame(short_rows)
        summary = (
            f"Equal-weight anonymized V2 10/10 produced {row_pct(anon_equal, 'cagr_pct')} CAGR. "
            f"Its best truncated-exponential CAGR was {row_pct(best_cagr, 'cagr_pct')} with "
            f"{'' if best_cagr is None else str(best_cagr.get('weight_model', ''))}; the best Sharpe was "
            f"{row_num(best_sharpe, 'sharpe')} with {'' if best_sharpe is None else str(best_sharpe.get('weight_model', ''))}."
        )
        return f"""
    <section id="top10-exp">
      <h2>Top-10 Exponential Sizing</h2>
      <p><strong>Purpose:</strong> hold top-10 membership fixed and test whether the LLM's ordering within those ten names supports unequal position sizing.</p>
      <div class="callout">{html.escape(summary)} This is a sizing test, not a new selector or prompt test.</div>
      <p><strong>Weight convention:</strong> these labels preserve the existing implementation, <code>w(r) proportional to exp(-(r-1)/h)</code>. Therefore <code>h1</code> puts 63.2% of each leg in rank 1 and 95.0% in ranks 1-3; <code>h</code> is an exponential decay scale, not a literal mathematical half-life.</p>
      <div class="table-wrap">{table(primary, cols, None)}</div>
      <h3>Short-book progression</h3>
      <p>Signed short P&amp;L is positive when the selected stocks subsequently fall. The progression tests whether the best anonymized short ranks are materially better than the broader short book.</p>
      <div class="table-wrap">{table(short_progression, ['construction', 'avg_side_pnl_c2c_pct', 'side_hit_rate_pct', 'side_cagr_pct'], None)}</div>
      <div class="callout"><strong>Read:</strong> concentration steadily improved the short book, but these are highly correlated portfolios rather than independent confirmations. The exact-rank and randomization audits below test whether rank 1, rather than a broad short tail, explains the shape.</div>
      <h3>Exact short-rank audit</h3>
      <p>Rank 1 is positive, rank 2 is weak, and rank 3 is strongly negative. This is not a smooth monotonic short-ranking curve.</p>
      <p class="path">{html.escape(str(output_dir / _diagnostic_filename('short_rank_slices')))}</p>
      <div class="table-wrap">{table(rank_slices, ['construction', 'interpretation', 'avg_weekly_side_pnl_pct', 'cagr_pct', 'sharpe', 'max_drawdown_pct', 'hit_rate_pct', 'worst_week_pct'], None)}</div>
      <h3>Search-adjusted rank-randomization null</h3>
      <p>Within every date, positions 1-10 are randomly reordered. All six concentration levels are then evaluated and only the best is retained, matching the optimization applied to the observed ladder.</p>
      <p class="path">{html.escape(str(output_dir / _diagnostic_filename('short_rank_search_null')))}</p>
      <div class="table-wrap">{table(rank_null, ['n_dates', 'n_concentration_levels', 'n_randomizations', 'observed_best_model', 'observed_best_cagr_pct', 'random_best_cagr_p50_pct', 'random_best_cagr_p95_pct', 'random_best_cagr_p99_pct', 'observed_cagr_percentile_pct', 'one_sided_p_pct'], None)}</div>
      <div class="callout"><strong>Conclusion:</strong> the observed ladder is suggestive but does not clear a conventional 5% search-adjusted threshold. Standalone short CAGR is geometric return, not factor-adjusted alpha.</div>
      <details>
        <summary>All prompt variants</summary>
        <p class="path">{html.escape(str(output_dir / _diagnostic_filename('top10_exponential')))}</p>
        <div class="table-wrap">{table(data, cols, None)}</div>
      </details>
    </section>
        """

    def regime_gate_html(items: dict[str, pd.DataFrame] | None) -> str:
        items = items or {}
        scorecard = items.get("regime_gate_scorecard", pd.DataFrame())
        by_regime = items.get("regime_gate_by_regime", pd.DataFrame())
        gate_bootstrap = items.get("regime_gate_bootstrap", pd.DataFrame())
        mapping_sweep = items.get("regime_mapping_sweep", pd.DataFrame())
        champions = items.get("regime_mapping_champions", pd.DataFrame())
        champion_years = items.get("regime_mapping_champion_years", pd.DataFrame())
        search_placebo = items.get("regime_mapping_search_placebo", pd.DataFrame())
        if scorecard.empty or champions.empty:
            return ""

        fixed_h5 = pick_row(champions, objective="Fixed h5 reference")
        best_cagr = pick_row(champions, objective="Best development CAGR")
        best_sharpe = pick_row(champions, objective="Best development Sharpe")
        best_worst_year = pick_row(champions, objective="Best worst-year CAGR")
        placebo_row = search_placebo.iloc[0] if not search_placebo.empty else None
        summary = (
            f"All 256 mappings were searched over the full development period. Best CAGR was {row_pct(best_cagr, 'cagr_pct')} "
            f"with {'' if best_cagr is None else str(best_cagr.get('mapping_id', ''))}, versus fixed h5 at "
            f"{row_pct(fixed_h5, 'cagr_pct')}. Best Sharpe was {row_num(best_sharpe, 'sharpe')} with "
            f"{'' if best_sharpe is None else str(best_sharpe.get('mapping_id', ''))}. The best worst-year mapping reduced "
            f"2013 to {row_pct(best_worst_year, 'cagr_2013_pct')} while retaining {row_pct(best_worst_year, 'cagr_pct')} full-period CAGR. "
            f"After correcting for the 256-way search, the observed best-CAGR mapping ranked at only the "
            f"{row_pct(placebo_row, 'observed_percentile_pct', 1)} percentile of shuffled-regime searches."
        )

        def gate_table(data: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
            available = [col for col in cols if col in data.columns]
            return table(data, available, limit) if not data.empty and available else "<p class='muted'>No rows.</p>"

        return f"""
    <section id="regime-gate">
      <h2>Development Regime-to-Aggression Autoresearch</h2>
      <p>This zero-token search uses the complete 2009-2014 development period. Each observed regime independently chooses top30, h20, h10, or h5, producing 4^4 = 256 mappings. These are development results, not out-of-sample claims.</p>
      <div class="callout"><strong>Result:</strong> {html.escape(summary)}</div>
      <h3>Champion Mappings</h3>
      <p>Different objectives expose the return/risk trade-off instead of pretending one optimized mapping is universally best.</p>
      <div class="table-wrap">{gate_table(champions, ["objective", "mapping_id", "cagr_pct", "sharpe", "max_drawdown_pct", "total_return_pct", "calmar_ratio", "worst_year_cagr_pct", "cagr_2013_pct", "year_cagr_std_pct", "h5_usage_pct", "h10_usage_pct", "h20_usage_pct", "top30_usage_pct"], None)}</div>
      <details open>
        <summary>Champion performance by year</summary>
        <p>This includes 2013 explicitly so the cost of maximizing full-period CAGR is visible.</p>
        <p class="path">{html.escape(str(output_dir / _diagnostic_filename("regime_mapping_champion_years")))}</p>
        <div class="table-wrap">{gate_table(champion_years, ["objective", "mapping_id", "year", "cagr_pct", "sharpe", "max_drawdown_pct", "total_return_pct"], None)}</div>
      </details>
      <details>
        <summary>Top 25 of all 256 mappings</summary>
        <p>The full CSV contains every mapping and objective rank.</p>
        <p class="path">{html.escape(str(output_dir / _diagnostic_filename("regime_mapping_sweep")))}</p>
        <div class="table-wrap">{gate_table(mapping_sweep, ["mapping_id", "arm_normal", "arm_grind_up", "arm_crisis", "arm_squeeze", "cagr_pct", "sharpe", "max_drawdown_pct", "calmar_ratio", "worst_year_cagr_pct", "cagr_2013_pct", "cagr_rank", "sharpe_rank", "calmar_rank", "worst_year_rank"], 25)}</div>
      </details>
      <details>
        <summary>Which regimes rewarded aggression?</summary>
        <p>Conditional development-sample averages for each fixed weighting arm.</p>
        <p class="path">{html.escape(str(output_dir / _diagnostic_filename("regime_gate_by_regime")))}</p>
        <div class="table-wrap">{gate_table(by_regime, ["regime", "n_dates", "top30_avg_weekly_pct", "h20_avg_weekly_pct", "h10_avg_weekly_pct", "h5_avg_weekly_pct", "h5_minus_top30_annualized_pct", "h5_beats_top30_dates_pct", "h10_minus_top30_annualized_pct", "h10_beats_top30_dates_pct", "best_full_sample_arm"], None)}</div>
      </details>
      <details>
        <summary>Search-adjusted regime-label placebo</summary>
        <p>Regime labels are shuffled while preserving their counts, and the best-CAGR mapping is re-optimized each time. This measures how easily a 256-way search finds an apparently strong winner by construction.</p>
        <p class="path">{html.escape(str(output_dir / _diagnostic_filename("regime_mapping_search_placebo")))}</p>
        <div class="table-wrap">{gate_table(search_placebo, ["objective", "n_regimes", "n_arms", "n_mappings_searched", "n_permutations", "observed_value", "shuffled_best_p50", "shuffled_best_p95", "shuffled_best_p99", "shuffled_best_max", "observed_percentile_pct"], None)}</div>
      </details>
      <details>
        <summary>Original readable-rule audit</summary>
        <p>The first hand-written regime rule and fixed controls remain here for comparison. No internal years are labeled out of sample.</p>
        <div class="table-wrap">{gate_table(scorecard, ["rule", "h5_usage_pct", "h10_usage_pct", "h20_usage_pct", "top30_usage_pct", "cagr_pct", "sharpe", "max_drawdown_pct", "total_return_pct", "cagr_lift_vs_period_top30_pct", "cagr_lift_vs_period_h5_pct"], None)}</div>
        <p class="path">{html.escape(str(output_dir / _diagnostic_filename("regime_gate_bootstrap")))}</p>
        <div class="table-wrap">{gate_table(gate_bootstrap, ["comparison", "n_dates", "mean_annualized_delta_pct", "ci_low_pct", "ci_high_pct", "p_two_sided", "probability_target_beats_control_pct"], None)}</div>
      </details>
    </section>
        """

    def diagnostics_html(items: dict[str, pd.DataFrame] | None) -> str:
        if not items:
            return ""

        def _block(title: str, description: str, key: str, cols: list[str], limit: int | None = None) -> str:
            data = items.get(key, pd.DataFrame())
            available = [col for col in cols if col in data.columns]
            if data.empty or not available:
                body = "<p class='muted'>No rows.</p>"
            else:
                body = table(data, available, limit)
            return f"""
      <details>
        <summary>{html.escape(title)}</summary>
        <p>{html.escape(description)}</p>
        <p class="path">{html.escape(str(output_dir / _diagnostic_filename(key)))}</p>
        <div class="table-wrap">{body}</div>
      </details>
            """

        return f"""
    <section id="diagnostics">
      <h2>Additional Offline Diagnostics</h2>
      <p>These are all zero-token tests built from cached candidate-level scores and realized five-day C2C outcomes.</p>
      {_block(
          "Rank-Only vs Raw-Score LLM",
          "Tests whether the LLM's numeric 0-100 distances matter, or whether only the order matters.",
          "rank_vs_score",
          ["source", "test", "cagr_pct", "sharpe", "max_drawdown_pct", "avg_long_raw_c2c_pct", "avg_short_side_pnl_pct"],
          None,
      )}
      {_block(
          "Visible/Anonymized Top-10 Agreement",
          "Checks whether agreement between ticker-visible and anonymized rankings predicts better forward returns.",
          "agreement",
          ["pool_side", "agreement_bucket", "n", "avg_visible_rank", "avg_anon_rank", "avg_raw_c2c_pct", "avg_side_pnl_c2c_pct", "side_hit_rate_pct", "side_vol_pct"],
          None,
      )}
      {_block(
          "Average Rank / Borda Consensus",
          "Tests consensus rank construction without letting one model's score scale dominate.",
          "consensus",
          ["source", "test", "cagr_pct", "sharpe", "max_drawdown_pct", "avg_long_raw_c2c_pct", "avg_short_side_pnl_pct"],
          None,
      )}
      {_block(
          "Rank Monotonicity",
          "Bucketed rank curve. We want better ranks to produce better side-adjusted returns, not one lucky bucket.",
          "monotonicity",
          ["source", "pool_side", "rank_bucket", "n", "avg_raw_c2c_pct", "avg_side_pnl_c2c_pct", "side_hit_rate_pct", "side_vol_pct"],
          120,
      )}
      {_block(
          "Disagreement As A Risk Signal",
          "Measures whether ticker-visible and anonymized rank disagreement is associated with worse outcomes, volatility, or large losses.",
          "disagreement_risk",
          ["pool_side", "disagreement_bucket", "n", "avg_rank_diff", "avg_raw_c2c_pct", "avg_side_pnl_c2c_pct", "side_hit_rate_pct", "side_vol_pct", "worst_side_return_pct"],
          None,
      )}
      {_block(
          "Directional Visible vs Anonymized Disagreement",
          "Splits disagreement by direction. Positive anon-higher buckets mean anonymized V2 ranked the stock better than ticker-visible V2; visible-higher means the opposite.",
          "directional_disagreement",
          ["pool_side", "directional_gap_bucket", "n", "avg_visible_rank", "avg_anon_rank", "avg_signed_gap", "avg_raw_c2c_pct", "avg_side_pnl_c2c_pct", "side_hit_rate_pct", "side_vol_pct", "worst_side_return_pct"],
          None,
      )}
      {_block(
          "Disagreement-Penalized Strategy",
          "Keeps V2 ranking but reduces weights when visible and anonymized ranks strongly disagree.",
          "disagreement_penalty",
          ["penalty_target", "cagr_pct", "sharpe", "max_drawdown_pct", "avg_long_raw_c2c_pct", "avg_short_side_pnl_pct"],
          None,
      )}
      {_block(
          "Random-Ranking Placebo",
          "Randomizes ranking inside the same 60/60 reversal pools to estimate how much performance comes from the pool itself versus the selector.",
          "random_placebo",
          ["weight_model", "real_ranking", "real_cagr_pct", "real_sharpe", "real_max_drawdown_pct", "random_p50_cagr_pct", "random_p95_cagr_pct", "random_max_cagr_pct", "real_cagr_percentile_pct"],
          None,
      )}
      {_block(
          "Candidate Pool vs Selector Value",
          "Decomposes equal-weight pool value, top-30 selection, and weighting effects.",
          "pool_selector",
          ["decomposition_step", "cagr_pct", "sharpe", "max_drawdown_pct", "avg_long_raw_c2c_pct", "avg_short_side_pnl_pct"],
          None,
      )}
      {_block(
          "Anon Exponential h5 vs V2 Regime Failure Breakdown",
          "Explains when anonymized exponential h5 beat or lagged the original ticker-visible V2 top30 control. Excess is capital-normalized C2C; long and short excess contributions sum to the weekly excess.",
          "anon_exp_regime_breakdown",
          ["dimension", "bucket", "n_dates", "target_avg_weekly_pct", "v2_avg_weekly_pct", "avg_excess_weekly_pct", "annualized_excess_pct", "excess_hit_rate_pct", "target_bucket_cagr_pct", "v2_bucket_cagr_pct", "avg_long_excess_contrib_pct", "avg_short_excess_contrib_pct", "worst_excess_weekly_pct", "avg_spy_trail5_pct", "avg_vix_percentile_252d", "avg_market_breadth_5d_up_pct", "avg_candidate_abs_trail5_pct", "avg_candidate_forward_side_hit_pct"],
          None,
      )}
      {_block(
          "Worst Anon Exponential Failure Dates",
          "The worst individual weeks for anonymized exponential h5 versus V2, with market context and long/short attribution.",
          "anon_exp_failure_dates",
          ["date", "year", "market_regime", "spy_200d_bucket", "vix_percentile_252d", "vix_pctl_bucket", "spy_trail5_pct", "market_breadth_5d_up_pct", "candidate_avg_abs_trail5_pct", "target_weekly_pct", "v2_weekly_pct", "excess_weekly_pct", "long_excess_contrib_pct", "short_excess_contrib_pct", "target_long_raw_c2c_pct", "v2_long_raw_c2c_pct", "target_short_side_pnl_c2c_pct", "v2_short_side_pnl_c2c_pct"],
          25,
      )}
    </section>
        """

    def side_hybrid_html(sections: list[dict[str, Any]] | None) -> str:
        if not sections:
            return ""
        parts = [
            """
    <section id="hybrids">
      <h2>Additional Zero-Token Cache Experiments</h2>
      <p>These include prior side-process hybrids and the V2.4/V2.5 follow-up sweeps. They use no new LLM calls and retain the same anchor 3, 60/60 pool, and gross fixed 5-day C2C lab bench.</p>
            """
        ]
        for section in sections:
            data = section["data"]
            cols = [col for col in section["columns"] if col in data.columns]
            if data.empty or not cols:
                body = f"<p class='muted'>Not found: {html.escape(str(section['path']))}</p>"
            else:
                body = table(data, cols, int(section["limit"]))
            parts.append(
                f"""
      <details>
        <summary>{html.escape(str(section["title"]))}</summary>
        <p>{html.escape(str(section["description"]))}</p>
        <p class="path">{html.escape(str(section["path"]))}</p>
        <div class="table-wrap">{body}</div>
      </details>
                """
            )
        parts.append("</section>")
        return "".join(parts)

    top_cols = [
        "family",
        "long_process",
        "short_process",
        "cagr_pct",
        "sharpe",
        "max_drawdown_pct",
        "cagr_lift_vs_v2_pct",
        "avg_long_raw_c2c_pct",
        "avg_short_side_pnl_pct",
        "avg_turnover_total",
    ]
    side_cols = ["side", "process", "score_col", "weight_model", "avg_side_pnl_c2c_pct", "side_hit_rate_pct", "side_vol_pct"]
    boot_cols = ["run", "mean_annualized_delta_pct", "ci_low_pct", "ci_high_pct", "p_two_sided", "cagr_pct", "sharpe", "max_drawdown_pct"]
    family_cols = ["family", "run", "cagr_pct", "sharpe", "max_drawdown_pct", "cagr_lift_vs_v2_pct"]
    cross_root = output_dir.parent / "phase2_validation_tracker"
    cross_summary = read_csv_if_exists(cross_root / "cross_period_autoresearch_summary.csv")
    cross_detail = read_csv_if_exists(cross_root / "cross_period_autoresearch_detail.csv")
    if cross_summary.empty or cross_detail.empty:
        cross_period_html = """<section id="cross-period"><h2>Cross-Period Robustness Search</h2>
        <p class="muted">Run the out-of-sample testing dashboard to generate the cross-period shared-rule sweep.</p></section>"""
    else:
        champion = cross_summary.iloc[0]
        champion_detail = cross_detail[
            cross_detail["long_model"].eq(champion["long_model"])
            & cross_detail["short_model"].eq(champion["short_model"])
        ]
        champion_cols = [
            "period", "llm_cagr_pct", "trail5_cagr_pct", "cagr_lift_pp",
            "llm_sharpe", "sharpe_lift", "llm_max_dd_pct",
        ]
        ranking_cols = [
            "robust_rank", "long_model", "short_model", "min_period_cagr_pct",
            "mean_period_cagr_pct", "min_period_sharpe", "min_period_lift_pp",
            "positive_lift_all_periods",
        ]
        cross_period_html = f"""<section id="cross-period"><h2>Cross-Period Robustness Search</h2>
        <p><strong>Winner: {html.escape(str(champion['long_model']))} longs / {html.escape(str(champion['short_model']))} shorts.</strong>
        This is the maximin-CAGR rule across 2009-2014, 2015-2019, and the completed exploratory 2020-2024 offset-2 cache.
        The search evaluates one fixed rule in every period and matches each LLM row to Trail5 with identical sleeve weights.</p>
        <div class="callout">The winner's weakest-period CAGR is {fmt_pct(champion['min_period_cagr_pct'])}; its weakest matched Trail5 lift is {fmt_pct(champion['min_period_lift_pp'])}. The 2020-2024 leg is exploratory because the available cache uses offset 2, not the locked offset 3.</div>
        <h3>Winner by period</h3><div class="table-wrap">{table(champion_detail, champion_cols, None)}</div>
        <h3>Top robust rules</h3><div class="table-wrap">{table(cross_summary, ranking_cols, 12)}</div>
        <p class="path">phase2_validation_tracker/cross_period_autoresearch_detail.csv · cross_period_autoresearch_summary.csv</p></section>"""
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V2 Cache Autoresearch Sweep</title>
  <style>
    :root {{
      --ink: #182029;
      --muted: #667384;
      --line: #d9e0e7;
      --panel: #ffffff;
      --paper: #f7f5ef;
      --soft: #f1f5f3;
      --head: #21352d;
      --good: #0f6b4f;
      --good-bg: #e5f4ee;
      --bad: #a12b2f;
      --bad-bg: #fae7e7;
      --warn: #8a5b00;
      --warn-bg: #fff3ce;
      --accent: #287d6b;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 36px 44px 30px;
      background: linear-gradient(135deg, #21352d 0%, #2f4a3f 54%, #5b4b2c 100%);
      color: #fff;
      border-bottom: 1px solid rgba(255,255,255,.18);
    }}
    header h1 {{ margin: 0 0 8px; font-size: clamp(30px, 4vw, 48px); letter-spacing: 0; line-height: 1.05; }}
    header p {{ color: rgba(255,255,255,.82); max-width: 980px; margin: 0; font-size: 16px; }}
    .hero-meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border: 1px solid rgba(255,255,255,.25);
      border-radius: 999px;
      background: rgba(255,255,255,.12);
      color: #fff;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .06em;
    }}
    .toc {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      gap: 10px;
      overflow-x: auto;
      padding: 10px 44px;
      background: rgba(247,245,239,.92);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--line);
    }}
    .toc a {{
      flex: 0 0 auto;
      color: #2a3b34;
      text-decoration: none;
      font-size: 13px;
      font-weight: 800;
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid transparent;
    }}
    .toc a:hover {{ background: #fff; border-color: var(--line); }}
    main {{ padding: 24px 44px 56px; max-width: 1720px; margin: 0 auto; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 22px;
      box-shadow: 0 14px 34px rgba(38, 48, 55, .08);
    }}
    h2 {{ margin: 0 0 6px; font-size: 24px; letter-spacing: 0; }}
    h3 {{ margin: 0; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }}
    p {{ color: var(--muted); margin: 8px 0 14px; }}
    .insight-list {{
      counter-reset: insight;
      margin-top: 16px;
      border-top: 1px solid var(--line);
    }}
    .insight-item {{
      counter-increment: insight;
      display: grid;
      grid-template-columns: 52px minmax(0, 1fr);
      gap: 14px;
      padding: 18px 0;
      border-bottom: 1px solid var(--line);
    }}
    .insight-index::before {{
      content: counter(insight);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 34px;
      height: 34px;
      border-radius: 999px;
      background: var(--good-bg);
      color: var(--good);
      font-weight: 900;
      font-variant-numeric: tabular-nums;
    }}
    .insight-item.warn .insight-index::before {{
      background: var(--warn-bg);
      color: var(--warn);
    }}
    .insight-kicker {{
      color: var(--accent);
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 8px;
    }}
    .insight-item.warn .insight-kicker {{ color: #8a5b00; }}
    .insight-item h3 {{
      color: var(--ink);
      font-size: 18px;
      line-height: 1.2;
      text-transform: none;
      letter-spacing: 0;
      margin-bottom: 10px;
    }}
    .insight-item p {{ margin: 8px 0; color: #4f5d69; max-width: 1120px; }}
    .insight-item .thesis {{ color: var(--ink); font-weight: 800; }}
    .insight-item strong {{ color: #203129; }}
    .next-step {{
      background: #eef7f3;
      border: 1px solid #d5e8df;
      border-radius: 8px;
      padding: 10px;
      max-width: 1120px;
    }}
    .insight-item.warn .next-step {{ background: #fff3ce; border-color: #efd58c; }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 12.5px; }}
    thead th {{
      position: sticky;
      top: 44px;
      z-index: 4;
      text-align: left;
      background: #edf3ef;
      color: #27372f;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .06em;
      white-space: nowrap;
    }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    tbody tr:nth-child(even) {{ background: #fafbfa; }}
    tbody tr:hover {{ background: #fff8e5; }}
    td {{ color: #22303a; }}
    td.metric {{ font-variant-numeric: tabular-nums; font-weight: 800; white-space: nowrap; }}
    td.good {{ color: var(--good); }}
    td.bad {{ color: var(--bad); }}
    td.ok {{ color: #405467; }}
    td.best {{ background: var(--good-bg); box-shadow: inset 3px 0 0 var(--good); }}
    details {{
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 14px 0;
      padding: 0;
      background: #fbfcfb;
      overflow: hidden;
    }}
    summary {{
      cursor: pointer;
      font-weight: 900;
      color: #22342d;
      padding: 14px 16px;
      background: #f0f5f2;
      border-bottom: 1px solid var(--line);
    }}
    details p, details .path, details .table-wrap {{ margin-left: 16px; margin-right: 16px; }}
    details .table-wrap {{ margin-bottom: 16px; }}
    .callout {{
      border-left: 5px solid var(--accent);
      background: #edf8f3;
      padding: 14px 16px;
      margin: 16px 0;
      color: #244d44;
      font-weight: 700;
    }}
    .muted {{ color: #7b8797; }}
    .path {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      color: #4f5f6d;
      background: #eef3f5;
      border: 1px solid #dce5ea;
      border-radius: 8px;
      padding: 8px 10px;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 760px) {{
      header, main, .toc {{ padding-left: 18px; padding-right: 18px; }}
      section {{ padding: 18px; }}
      .insight-item {{ grid-template-columns: 1fr; gap: 8px; }}
      thead th {{ top: 42px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>V2 Cache Autoresearch Sweep</h1>
    <p>Zero-token offline sweep across cached V2/V2.1/V2.2 scores. All headline metrics use 50/50 capital-normalized gross fixed 5-day C2C.</p>
    <div class="hero-meta">
      <span class="pill">Zero Token</span>
      <span class="pill">Anchor 3</span>
      <span class="pill">60/60 -> 30/30</span>
      <span class="pill">Gross C2C</span>
      <span class="pill">Hypothesis Lab</span>
    </div>
  </header>
  <nav class="toc">
    <a href="../mean_reversion_research_dashboard.html">Hub</a>
    <a href="../prompt_engineering_dashboard.html">Prompt Dashboard</a>
    <a href="../future_experiments_dashboard.html">Future Experiments</a>
    <a href="../phase2_validation_dashboard.html">Out-of-Sample Testing</a>
    <a href="#insights">Insights</a>
    <a href="#cross-period">Cross-Period</a>
    <a href="#top">Top Variants</a>
    <a href="#bootstrap">Bootstrap</a>
    <a href="#families">Families</a>
    <a href="#top10-exp">Top-10 Exp</a>
    <a href="#exp2x2">2x2 Exp</a>
    <a href="#regime-gate">Regime Gate</a>
    <a href="#diagnostics">Diagnostics</a>
    <a href="#hybrids">Prior Hybrids</a>
    <a href="#side">Side Quality</a>
  </nav>
  <main>
    {insights_html(diagnostics, exp_2x2)}
    {cross_period_html}
    <section id="top">
      <h2>Top Variants By CAGR</h2>
      <p>Top rows only. Full detail remains in the CSV artifacts.</p>
      <div class="table-wrap">{table(scorecard, top_cols, 25)}</div>
    </section>
    <section id="bootstrap">
      <h2>Bootstrap Vs V2 Control</h2>
      <p>Date-block bootstrap of weekly capital-return delta against the original ticker-visible V2 top30/top30 control.</p>
      <div class="table-wrap">{table(bootstrap, boot_cols, 25)}</div>
    </section>
    <section id="families">
      <h2>Best By Experiment Family</h2>
      <div class="table-wrap">{table(top_by_family, family_cols, None)}</div>
    </section>
    {top10_exponential_html(diagnostics)}
    {exp_2x2_html(exp_2x2)}
    {regime_gate_html(diagnostics)}
    {diagnostics_html(diagnostics)}
    {side_hybrid_html(side_hybrid_sections)}
    <section id="side">
      <h2>Side-Only Process Quality</h2>
      <p>For shorts, positive side P&amp;L means the short book made money. Showing the top rows only; full detail is in <span class="path">side_process_summary.csv</span>.</p>
      <div class="table-wrap">{table(side_summary, side_cols, 40)}</div>
    </section>
  </main>
</body>
</html>
"""
    path = output_dir / "autoresearch_report.html"
    path.write_text(report, encoding="utf-8")
    return path


def run_autoresearch(
    output_dir: Path,
    anchor: str = "3",
    pool: int = 60,
    book: int = 30,
    n_boot: int = 3000,
    random_placebo_runs: int = 2000,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_base_panel(DEFAULT_V2_ROOT, anchor, pool, book)
    for source in default_sources():
        panel = merge_score_source(panel, source, anchor, pool, book)
    available_score_sources = [col for col in panel.columns if col.startswith("score_")]
    panel, descriptions, score_cols, core_cols = add_derived_scores(panel)
    panel = panel.copy()
    panel.to_parquet(output_dir / "candidate_panel.parquet", index=False)

    processes = build_process_specs(score_cols, core_cols, descriptions)
    process_by_name = {spec.name: spec for spec in processes}
    pd.DataFrame([spec.__dict__ for spec in processes]).to_csv(output_dir / "process_specs.csv", index=False)

    side_weekly: dict[tuple[str, str], pd.DataFrame] = {}
    side_selection_samples: list[pd.DataFrame] = []
    sample_processes = {"score_v2_visible__eq30", "score_trail5__eq30", "score_v2_visible__lin60"}
    for idx, spec in enumerate(processes, start=1):
        if idx % 100 == 0:
            print(f"  computed side processes for {idx}/{len(processes)} process specs")
        for side in ("LONG", "SHORT"):
            weekly, selections = compute_side_process(panel, spec, side, return_selections=spec.name in sample_processes)
            side_weekly[(spec.name, side)] = weekly
            if spec.name in sample_processes and not selections.empty:
                side_selection_samples.append(selections)
    if side_selection_samples:
        pd.concat(side_selection_samples, ignore_index=True).to_csv(output_dir / "selection_samples.csv", index=False)

    variants = build_variant_specs(processes)
    pd.DataFrame(variants, columns=["run", "long_process", "short_process", "family"]).to_csv(
        output_dir / "variant_specs.csv", index=False
    )
    scorecard, weekly_returns = build_scorecard(variants, side_weekly, process_by_name)
    scorecard.to_csv(output_dir / "comparison_scorecard.csv", index=False)
    weekly_returns.to_parquet(output_dir / "weekly_returns.parquet", index=False)
    exp_2x2 = build_all_exponential_2x2_rows(weekly_returns)
    exp_2x2.to_csv(output_dir / "exponential_2x2_isolation.csv", index=False)
    exp_2x2[exp_2x2["llm_source"].eq("Anonymized V2")].to_csv(
        output_dir / "anonymized_exponential_2x2_isolation.csv", index=False
    )
    diagnostics = build_offline_diagnostics(output_dir, panel=panel, weekly_returns=weekly_returns, random_sims=random_placebo_runs)
    side_summary = side_process_summary(side_weekly, processes)
    side_summary.to_csv(output_dir / "side_process_summary.csv", index=False)
    top_by_family = scorecard.sort_values(["family", "cagr_pct", "sharpe"], ascending=[True, False, False]).groupby(
        "family", as_index=False
    ).head(3)
    top_by_family.to_csv(output_dir / "top_by_family.csv", index=False)
    top_runs = (
        pd.concat(
            [
                scorecard.sort_values("cagr_pct", ascending=False).head(80),
                scorecard.sort_values("sharpe", ascending=False).head(40),
                scorecard[scorecard["run"].eq("same::score_v2_visible__eq30")],
            ],
            ignore_index=True,
        )["run"]
        .drop_duplicates()
        .tolist()
    )
    bootstrap = block_bootstrap(weekly_returns, scorecard, "same::score_v2_visible__eq30", top_runs, n_boot=n_boot)
    bootstrap.to_csv(output_dir / "block_bootstrap_vs_v2.csv", index=False)
    year_metrics, regime_metrics = year_regime_metrics(weekly_returns, top_runs[:30])
    year_metrics.to_csv(output_dir / "year_metrics_top.csv", index=False)
    regime_metrics.to_csv(output_dir / "regime_metrics_top.csv", index=False)

    control_checks: dict[str, Any] = {
        "Candidate rows": f"{len(panel):,}",
        "Base score columns": ", ".join(available_score_sources),
        "Derived scores": f"{len(score_cols):,}",
        "Processes": f"{len(processes):,}",
        "Variants": f"{len(scorecard):,}",
    }
    for label, run in {
        "V2 control": "same::score_v2_visible__eq30",
        "Trail5 control": "same::score_trail5__eq30",
        "Best variant": scorecard.iloc[0]["run"] if not scorecard.empty else "",
    }.items():
        row = scorecard[scorecard["run"].eq(run)]
        if not row.empty:
            r = row.iloc[0]
            control_checks[label] = f"CAGR {fmt_pct(r['cagr_pct'])}, Sharpe {fmt_num(r['sharpe'])}, DD {fmt_pct(r['max_drawdown_pct'])}"
    report_path = write_html_report(
        output_dir,
        scorecard,
        side_summary,
        bootstrap,
        top_by_family,
        control_checks,
        load_side_hybrid_sections() + load_v24_v25_followup_sections(),
        exp_2x2,
        diagnostics,
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "anchor": anchor,
                "pool_per_side": pool,
                "positions_per_side": book,
                "n_boot": n_boot,
                "random_placebo_runs": random_placebo_runs,
                "output_dir": str(output_dir),
                "warning": "Offline in-sample autoresearch sweep; treat as hypothesis generation.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir,
        "scorecard": output_dir / "comparison_scorecard.csv",
        "bootstrap": output_dir / "block_bootstrap_vs_v2.csv",
        "side_summary": output_dir / "side_process_summary.csv",
        "report": report_path,
    }


def rewrite_report_from_existing(output_dir: Path, random_placebo_runs: int = 2000) -> dict[str, Path]:
    scorecard = read_csv_if_exists(output_dir / "comparison_scorecard.csv")
    side_summary = read_csv_if_exists(output_dir / "side_process_summary.csv")
    bootstrap = read_csv_if_exists(output_dir / "block_bootstrap_vs_v2.csv")
    top_by_family = read_csv_if_exists(output_dir / "top_by_family.csv")
    if scorecard.empty:
        raise FileNotFoundError(f"Missing existing autoresearch scorecard: {output_dir / 'comparison_scorecard.csv'}")
    control_checks: dict[str, Any] = {
        "Candidate rows": "existing report-only refresh",
        "Base score columns": "existing cached artifacts",
        "Derived scores": "existing cached artifacts",
        "Processes": "existing cached artifacts",
        "Variants": f"{len(scorecard):,}",
    }
    for label, run in {
        "V2 control": "same::score_v2_visible__eq30",
        "Trail5 control": "same::score_trail5__eq30",
        "Best variant": scorecard.iloc[0]["run"] if not scorecard.empty else "",
    }.items():
        row = scorecard[scorecard["run"].eq(run)]
        if not row.empty:
            r = row.iloc[0]
            control_checks[label] = f"CAGR {fmt_pct(r['cagr_pct'])}, Sharpe {fmt_num(r['sharpe'])}, DD {fmt_pct(r['max_drawdown_pct'])}"
    weekly_returns_path = output_dir / "weekly_returns.parquet"
    weekly_returns = pd.read_parquet(weekly_returns_path) if weekly_returns_path.exists() else pd.DataFrame()
    exp_2x2 = build_all_exponential_2x2_rows(weekly_returns)
    exp_2x2.to_csv(output_dir / "exponential_2x2_isolation.csv", index=False)
    if not exp_2x2.empty and "llm_source" in exp_2x2.columns:
        exp_2x2[exp_2x2["llm_source"].eq("Anonymized V2")].to_csv(
            output_dir / "anonymized_exponential_2x2_isolation.csv", index=False
        )
    diagnostics = build_offline_diagnostics(output_dir, weekly_returns=weekly_returns, random_sims=random_placebo_runs)
    report_path = write_html_report(
        output_dir,
        scorecard,
        side_summary,
        bootstrap,
        top_by_family,
        control_checks,
        load_side_hybrid_sections() + load_v24_v25_followup_sections(),
        exp_2x2,
        diagnostics,
    )
    return {
        "output_dir": output_dir,
        "scorecard": output_dir / "comparison_scorecard.csv",
        "bootstrap": output_dir / "block_bootstrap_vs_v2.csv",
        "side_summary": output_dir / "side_process_summary.csv",
        "report": report_path,
    }


CAMPAIGN_SCHEMA_VERSION = 1
CAMPAIGN_ROOT = Path("backtest_results/autoresearch_campaigns")
CAMPAIGN_DEVELOPMENT_START = pd.Timestamp("2009-01-01")
CAMPAIGN_DEVELOPMENT_END = pd.Timestamp("2014-01-01")
CAMPAIGN_VALIDATION_START = pd.Timestamp("2015-01-01")
CAMPAIGN_VALIDATION_END = pd.Timestamp("2019-12-31")
CAMPAIGN_HOLDOUT_START = pd.Timestamp("2020-01-01")
CAMPAIGN_STAGE_SIZES = {"smoke": 5, "mini": 24, "medium": 80, "full": 251}
CAMPAIGN_RUNNER = Path(__file__).with_name("mean_reversion_llm_v2.py")
CAMPAIGN_DATE_SOURCE = (
    Path("backtest_results/v2_minimal_change_visible_temp1_2009_2014")
    / "pool60_pos30/anchor_3/baseline_pool/weekly_returns.csv"
)
CAMPAIGN_DEVELOPMENT_ANON_CONTROL = (
    Path("backtest_results/v2_minimal_change_temp1_v4flash0731_second_2009_2014")
    / "pool60_pos30/anchor_3/llm_pool_v2_minimal_change_control"
)
CAMPAIGN_DEVELOPMENT_VISIBLE_CONTROL = (
    Path("backtest_results/v2_minimal_change_visible_temp1_2009_2014")
    / "pool60_pos30/anchor_3/llm_pool_v2_minimal_change_control"
)
CAMPAIGN_VALIDATION_CONTROL = (
    Path("backtest_results/v2_prompt_ab_2015_2020/minimal_change_temp1")
    / "pool60_pos30/anchor_3/llm_pool_v2_minimal_change_control"
)
CAMPAIGN_DEVELOPMENT_TRAIL5 = (
    Path("backtest_results/v2_minimal_change_temp1_v4flash0731_second_2009_2014")
    / "pool60_pos30/anchor_3/baseline_pool"
)
CAMPAIGN_VALIDATION_TRAIL5 = (
    Path("backtest_results/v2_prompt_ab_2015_2020/minimal_change_temp1")
    / "pool60_pos30/anchor_3/baseline_pool"
)
CAMPAIGN_OUTPUT_CONTRACT = "candidate_declared_numeric_score"
CAMPAIGN_RESEARCHER_MODEL = "deepseek-v4-pro"
CAMPAIGN_SCORER_MODEL = "deepseek-v4-flash"
CAMPAIGN_TEMPERATURE = 1.0
CAMPAIGN_THINKING_MODE = "disabled"
CAMPAIGN_MAX_CANDIDATES = 8
CAMPAIGN_MAX_PRO_CALLS = 4
CAMPAIGN_PRO_MAX_OUTPUT_TOKENS = 12_000
CAMPAIGN_MAX_FULL_RUNS = 2
CAMPAIGN_DEV_MAX_FLASH_CALLS = 1100
CAMPAIGN_DEV_MAX_FLASH_OUTPUT_TOKENS = 5_000_000
CAMPAIGN_VAL_MAX_FLASH_CALLS = 550
CAMPAIGN_VAL_MAX_FLASH_OUTPUT_TOKENS = 2_500_000
CAMPAIGN_PROMPT_MAX_TOKENS = 8_192


class CampaignError(RuntimeError):
    """Base error for the paid, bounded prompt-autoresearch campaign."""


class CampaignFirewallError(CampaignError):
    """Raised before an artifact or date can cross the development firewall."""


class CampaignStop(CampaignError):
    """Safe terminal/interruption condition; all completed artifacts remain resumable."""


def campaign_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def campaign_sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def campaign_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def campaign_atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def campaign_read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CampaignError(f"Expected a JSON object: {path}")
    return value


def campaign_append_event(campaign_dir: Path, event: str, **payload: Any) -> None:
    record = {
        "timestamp": campaign_utc_now(),
        "event": str(event),
        "campaign_id": campaign_dir.name,
        **{key: value for key, value in payload.items() if value is not None},
    }
    campaign_dir.mkdir(parents=True, exist_ok=True)
    with (campaign_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def campaign_path(root: Path, campaign_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", str(campaign_id)):
        raise CampaignError("campaign_id must use only letters, numbers, dot, dash, or underscore")
    root_resolved = Path(root).resolve()
    path = (root_resolved / str(campaign_id)).resolve()
    if path.parent != root_resolved:
        raise CampaignError("campaign directory escaped its configured root")
    return path


def campaign_load(campaign_dir: Path) -> dict[str, Any]:
    return campaign_read_json(campaign_dir / "campaign.json")


def campaign_status_projection(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": state["campaign_id"],
        "status": state.get("status"),
        "phase": state.get("phase"),
        "current_candidate": state.get("current_candidate"),
        "current_stage": state.get("current_stage"),
        "completed_dates": state.get("completed_dates", 0),
        "planned_dates": state.get("planned_dates", 0),
        "usage": state.get("usage", {}),
        "active_stage_usage": state.get("active_stage_usage", {}),
        "winner_id": state.get("winner_id"),
        "terminal_reason": state.get("terminal_reason"),
        "stop_requested": bool(state.get("stop_requested", False)),
        "error": state.get("error"),
        "supervisor_pid": state.get("supervisor_pid"),
        "runner_pid": state.get("runner_pid"),
        "last_event_at": state.get("updated_at"),
    }


def campaign_save(campaign_dir: Path, state: dict[str, Any], event: str | None = None, **payload: Any) -> None:
    state["updated_at"] = campaign_utc_now()
    campaign_atomic_json(campaign_dir / "campaign.json", state)
    campaign_atomic_json(campaign_dir / "status.json", campaign_status_projection(state))
    if event:
        campaign_append_event(campaign_dir, event, **payload)


@contextmanager
def campaign_lock(campaign_dir: Path) -> Iterator[None]:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    # Lock the one permitted runner itself so alternate --campaign-root values
    # cannot create concurrent cache writers.
    global_handle = CAMPAIGN_RUNNER.open("r", encoding="utf-8")
    handle = (campaign_dir / ".campaign.lock").open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(global_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError("Another autoresearch supervisor already owns the repo-native runner") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            try:
                fcntl.flock(global_handle.fileno(), fcntl.LOCK_UN)
            finally:
                global_handle.close()


def campaign_assert_dates(dates: Sequence[Any], *, validation: bool = False) -> list[pd.Timestamp]:
    parsed = [pd.Timestamp(value).normalize() for value in dates]
    if not parsed or len(parsed) != len(set(parsed)):
        raise CampaignFirewallError("A date manifest must be non-empty and contain unique dates")
    if any(date >= CAMPAIGN_HOLDOUT_START for date in parsed):
        raise CampaignFirewallError("Autoresearch cannot access signal or exit dates in 2020 or later")
    lower = CAMPAIGN_VALIDATION_START if validation else CAMPAIGN_DEVELOPMENT_START
    upper = CAMPAIGN_VALIDATION_END if validation else CAMPAIGN_DEVELOPMENT_END
    if any(date < lower or date > upper for date in parsed):
        label = "validation" if validation else "development"
        raise CampaignFirewallError(f"Date is outside the frozen {label} interval")
    return sorted(parsed)


def campaign_manifest_hash(dates: Sequence[Any]) -> str:
    normalized = [str(pd.Timestamp(value).date()) for value in dates]
    return campaign_sha256(campaign_json_bytes(normalized))


def campaign_spread_order(count: int) -> list[int]:
    """Deterministic maximin ordering; every stage is a nested chronological subset."""
    if count <= 0:
        return []
    chosen = [0]
    if count > 1:
        chosen.append(count - 1)
    while len(chosen) < count:
        candidates = [idx for idx in range(count) if idx not in chosen]
        next_idx = max(candidates, key=lambda idx: (min(abs(idx - old) for old in chosen), -idx))
        chosen.append(next_idx)
    return chosen


def campaign_build_nested_manifests(
    campaign_dir: Path,
    source_csv: Path = CAMPAIGN_DATE_SOURCE,
) -> dict[str, dict[str, Any]]:
    if not source_csv.exists():
        raise CampaignError(f"Missing development date source: {source_csv}")
    source = pd.read_csv(source_csv, usecols=["date"])
    dates = campaign_assert_dates(source["date"].drop_duplicates().tolist())
    if len(dates) != CAMPAIGN_STAGE_SIZES["full"]:
        raise CampaignError(f"Expected 251 development dates, found {len(dates)} in {source_csv}")
    priority = campaign_spread_order(len(dates))
    manifests: dict[str, dict[str, Any]] = {}
    target_dir = campaign_dir / "manifests"
    for stage, size in CAMPAIGN_STAGE_SIZES.items():
        selected = sorted(dates[idx] for idx in priority[:size])
        manifest = {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": campaign_dir.name,
            "stage": stage,
            "purpose": "development",
            "dates": [str(date.date()) for date in selected],
            "date_count": len(selected),
            "manifest_hash": campaign_manifest_hash(selected),
            "generated_before_candidates": True,
        }
        campaign_atomic_json(target_dir / f"{stage}.json", manifest)
        manifests[stage] = manifest
    return manifests


def campaign_build_validation_manifest(campaign_dir: Path, wide_path: Path = RUSSELL_WIDE_PATH) -> dict[str, Any]:
    """Build the frozen offset-3 schedule only after manual validation approval."""
    if not wide_path.exists():
        raise CampaignError(f"Missing Russell trading calendar: {wide_path}")
    trading_dates = pd.DatetimeIndex(pd.read_parquet(wide_path, columns=[]).index).sort_values().unique()
    pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for idx in range(5 + 3, len(trading_dates) - 5, 5):
        signal_date = pd.Timestamp(trading_dates[idx]).normalize()
        exit_date = pd.Timestamp(trading_dates[idx + 5]).normalize()
        if signal_date < CAMPAIGN_VALIDATION_START:
            continue
        if signal_date > CAMPAIGN_VALIDATION_END or exit_date > CAMPAIGN_VALIDATION_END:
            continue
        pairs.append((signal_date, exit_date))
    dates = campaign_assert_dates([pair[0] for pair in pairs], validation=True)
    campaign_assert_dates([pair[1] for pair in pairs], validation=True)
    manifest = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_dir.name,
        "stage": "validation",
        "purpose": "validation",
        "dates": [str(value.date()) for value in dates],
        "exit_dates": [str(value.date()) for _, value in pairs],
        "date_count": len(dates),
        "manifest_hash": campaign_manifest_hash(dates),
        "exit_date_hash": campaign_manifest_hash([value for _, value in pairs]),
    }
    campaign_atomic_json(campaign_dir / "manifests/validation.json", manifest)
    return manifest


def campaign_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def campaign_code_fingerprint(runner: Path = CAMPAIGN_RUNNER) -> str:
    import mean_reversion_llm_v2 as runner_module

    if Path(runner).resolve() != Path(runner_module.__file__).resolve():
        raise CampaignError("Validation fingerprints require the repo-native mean_reversion_llm_v2.py runner")
    return str(runner_module.autoresearch_code_fingerprint())


def campaign_harness(side_scope: str = "both", side_parent_cache: str | None = None) -> dict[str, Any]:
    harness = {
        "schedule": "anchors",
        "anchor_offsets": [3],
        "signal_lookback_days": 5,
        "holding_days": 5,
        "pool_sizes": [60],
        "book_sizes": [30],
        "llm_pool_size": 60,
        "capital_split": {"long": 0.5, "short": 0.5},
        "weighting": "equal",
        "return_basis": "gross_fixed_horizon_c2c",
        "cost_bps_levels": [0.0],
        "ticker_visibility": "candidate_selected",
        "allowed_anonymize_tickers": [True, False],
        "control_policy": "reuse_completed_read_only_v2_minimal_change",
        "control_artifacts": campaign_control_artifacts(),
        "trail5_policy": "reuse_completed_read_only",
        "trail5_artifacts": campaign_trail5_artifacts(),
        "model": CAMPAIGN_SCORER_MODEL,
        "temperature": CAMPAIGN_TEMPERATURE,
        "thinking_mode": CAMPAIGN_THINKING_MODE,
        "output_contract": CAMPAIGN_OUTPUT_CONTRACT,
    }
    if side_scope != "both":
        harness.update(
            {
                "side_scope": side_scope,
                "side_parent_cache": str(side_parent_cache or ""),
                "control_policy": "reuse_completed_dedicated_side_parent_and_fixed_opposite_minimal_book",
                "ticker_visibility": "anonymous",
                "allowed_anonymize_tickers": [True],
                "finalist_replicates": 3,
            }
        )
    return harness


def campaign_control_artifact(
    *,
    anonymize_tickers: bool,
    validation: bool,
) -> dict[str, Any]:
    if validation:
        run_dir = CAMPAIGN_VALIDATION_CONTROL
        matched_visibility = bool(anonymize_tickers)
    else:
        run_dir = (
            CAMPAIGN_DEVELOPMENT_ANON_CONTROL
            if anonymize_tickers
            else CAMPAIGN_DEVELOPMENT_VISIBLE_CONTROL
        )
        matched_visibility = True
    weekly = run_dir / "weekly_returns.csv"
    checks = run_dir / "signal_checks.csv"
    for path in (weekly, checks):
        if not path.is_file():
            raise CampaignError(f"Missing frozen reusable control artifact: {path}")
    return {
        "run_dir": str(run_dir),
        "weekly_returns_sha256": campaign_file_hash(weekly),
        "signal_checks_sha256": campaign_file_hash(checks),
        "requested_anonymize_tickers": bool(anonymize_tickers),
        "control_anonymize_tickers": True if validation else bool(anonymize_tickers),
        "visibility_matched": matched_visibility,
    }


def campaign_control_artifacts() -> dict[str, Any]:
    return {
        "development_anonymous": campaign_control_artifact(
            anonymize_tickers=True, validation=False
        ),
        "development_ticker_visible": campaign_control_artifact(
            anonymize_tickers=False, validation=False
        ),
        "validation_anonymous": campaign_control_artifact(
            anonymize_tickers=True, validation=True
        ),
        "validation_visible_candidate_fallback": campaign_control_artifact(
            anonymize_tickers=False, validation=True
        ),
    }


def campaign_trail5_artifact(*, validation: bool) -> dict[str, Any]:
    run_dir = CAMPAIGN_VALIDATION_TRAIL5 if validation else CAMPAIGN_DEVELOPMENT_TRAIL5
    weekly = run_dir / "weekly_returns.csv"
    if not weekly.is_file():
        raise CampaignError(f"Missing frozen reusable Trail5 artifact: {weekly}")
    return {
        "run_dir": str(run_dir),
        "weekly_returns_sha256": campaign_file_hash(weekly),
    }


def campaign_trail5_artifacts() -> dict[str, Any]:
    return {
        "development": campaign_trail5_artifact(validation=False),
        "validation": campaign_trail5_artifact(validation=True),
    }


def campaign_harness_hash(side_scope: str = "both", side_parent_cache: str | None = None) -> str:
    return campaign_sha256(campaign_json_bytes(campaign_harness(side_scope, side_parent_cache)))


def campaign_runner_harness_fingerprint(
    state: dict[str, Any],
    campaign_dir: Path,
    spec_path: Path,
    *,
    validation: bool,
) -> str:
    import mean_reversion_llm_v2 as runner_module

    command, _ = campaign_runner_command(
        state,
        campaign_dir,
        spec_path,
        "validation" if validation else "full",
        validation=validation,
    )
    args = runner_module.parse_args(command[2:])
    loaded_spec = runner_module.load_experimental_prompt_spec(spec_path)
    args.side_scope = loaded_spec.side_scope
    args.replicate_id = loaded_spec.replicate_id
    args.thinking_mode = technical_llm.effective_thinking_mode(args.model, args.thinking_mode)
    return str(runner_module.autoresearch_harness_fingerprint(args))


def campaign_incumbent_prompt() -> tuple[str, str]:
    import mean_reversion_llm_v2 as runner

    prompt = str(runner.V2_MINIMAL_CHANGE_CONTROL_PROMPT)
    prompt_hash = str(runner.prompt_hash_for_variant(runner.PROMPT_VARIANT_V2_MINIMAL_CHANGE_CONTROL))
    return prompt, prompt_hash


def campaign_catalog(backtest_root: Path = Path("backtest_results")) -> dict[str, Any]:
    """Catalog only explicitly development-labelled top-level trees; never descend holdout trees."""
    def safe_development_artifact(path: Path, root: Path) -> bool:
        try:
            relative = path.relative_to(root)
            resolved = path.resolve()
            resolved.relative_to(root.resolve())
        except (OSError, ValueError):
            return False
        parts = [part.lower() for part in relative.parts]
        if path.is_symlink() or any((root / Path(*relative.parts[:index])).is_symlink() for index in range(1, len(relative.parts))):
            return False
        blocked = re.compile(r"(?:20(?:2\d|[3-9]\d)|holdout|out[-_ ]?of[-_ ]?sample|\boos\b|validation|phase[-_ ]?2|blended)")
        return not any(blocked.search(part) for part in parts)

    expected_dates: set[str] = set()
    if CAMPAIGN_DATE_SOURCE.exists():
        try:
            expected_dates = {
                str(value.date())
                for value in campaign_assert_dates(
                    pd.read_csv(CAMPAIGN_DATE_SOURCE, usecols=["date"])["date"]
                    .drop_duplicates()
                    .tolist()
                )
            }
        except (OSError, ValueError, CampaignFirewallError):
            expected_dates = set()

    def complete_score_side(rows: Any) -> bool:
        if not isinstance(rows, list) or len(rows) != 60:
            return False
        tickers: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or not str(row.get("ticker", "")).strip():
                return False
            try:
                score = float(row.get("score"))
            except (TypeError, ValueError):
                return False
            if not np.isfinite(score):
                return False
            tickers.add(str(row["ticker"]).strip())
        return len(tickers) == 60

    def cache_records(root: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*.json")):
            relative_text = str(path.relative_to(root)).lower()
            if (
                not safe_development_artifact(path, root)
                or path.name == "run_config.json"
                or ("cache" not in relative_text and not path.name.startswith("llm_reversal_signal"))
            ):
                continue
            try:
                payload = campaign_read_json(path)
            except (OSError, json.JSONDecodeError, CampaignError):
                continue
            signals = payload.get("signals")
            if not isinstance(signals, dict) or not signals:
                continue
            raw_dates = [str(key).split("|", 1)[0] for key in signals]
            try:
                dates = {
                    str(value.date()) for value in campaign_assert_dates(raw_dates)
                }
            except CampaignFirewallError:
                continue
            scores_complete = all(
                isinstance(entry, dict)
                and complete_score_side(entry.get("long_scores"))
                and complete_score_side(entry.get("short_scores"))
                for entry in signals.values()
            )
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            records.append(
                {
                    "path": str(path),
                    "metadata": metadata,
                    "complete": bool(
                        expected_dates
                        and dates == expected_dates
                        and scores_complete
                        and not payload.get("partial_signals")
                    ),
                }
            )
        return records

    def qualitative_profile(config: dict[str, Any], path: Path) -> tuple[str, str]:
        variant = str(config.get("prompt_variant", path.name)).lower()
        location = str(path).lower()
        combined = f"{variant} {location}"
        if "v2.1" in combined or "v21" in combined:
            return (
                "anonymized_discriminative_residual_prompt",
                "discriminative_residual_rewrite_failed_to_improve",
            )
        if "v2.2" in combined or "v22" in combined:
            return (
                "trail5_base_plus_bounded_context_adjustment",
                "bounded_adjustment_remained_weaker_than_trail5",
            )
        if "v2.3" in combined or "v23" in combined:
            return (
                "side_relative_rank_only_output",
                "rank_only_was_usable_but_weaker_than_score_output",
            )
        if "v2.4" in combined or "v24" in combined:
            return (
                "frozen_prompt_with_raw_market_context",
                "raw_market_context_was_near_flat_with_better_tail_characteristics",
            )
        if "v2.5" in combined or "v25" in combined:
            return (
                "balanced_point_in_time_outcome_memory",
                "outcome_memory_added_complexity_without_robust_improvement",
            )
        if "v2.6" in combined or "v26" in combined:
            return (
                "full_score_prompt_rewrite",
                "full_rewrite_underperformed_the_frozen_incumbent",
            )
        if "minimal" in combined:
            if "1p5" in combined or np.isclose(
                float(pd.to_numeric(config.get("temperature"), errors="coerce")),
                1.5,
                equal_nan=False,
            ):
                return (
                    "minimal_change_prompt_high_temperature_control",
                    "higher_temperature_degraded_economics_and_schema_reliability",
                )
            if config.get("thinking_mode") == "enabled" or "reason" in combined:
                return (
                    "minimal_change_prompt_thinking_control",
                    "thinking_control_is_outside_the_frozen_nonthinking_harness",
                )
            if np.isclose(
                float(pd.to_numeric(config.get("temperature"), errors="coerce")),
                1.0,
                equal_nan=False,
            ):
                return (
                    "minimal_change_prompt_temperature_one",
                    "minimal_wording_change_at_temperature_one_is_the_clean_parent",
                )
            return (
                "minimal_change_prompt_control",
                "minimal_change_control_is_the_primary_prompt_reference",
            )
        if "structured" in combined or "attr" in combined:
            return (
                "expanded_labels_or_attribution_output",
                "expanded_output_fields_were_diagnostic_but_weaker",
            )
        if "reason" in combined or config.get("thinking_mode") == "enabled":
            return (
                "reasoning_mode_control",
                "thinking_mode_is_outside_the_frozen_nonthinking_harness",
            )
        if "temp" in combined:
            return (
                "temperature_control",
                "temperature_only_runs_are_controls_not_prompt_evidence",
            )
        if "v1" in combined:
            return ("legacy_v1_prompt", "legacy_reasoning_prompt_reference")
        if variant in {"v2", "v2-frozen"}:
            return (
                "frozen_v2_score_prompt",
                "frozen_v2_is_the_incumbent_score_prompt_reference",
            )
        return (
            "historical_prompt_experiment",
            "completed_prompt_reference" if config else "legacy_development_reference",
        )

    entries: list[dict[str, Any]] = []
    if backtest_root.exists():
        allowed = sorted(
            path for path in backtest_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and ("2009_2014" in path.name or "2009-2014" in path.name)
        )
        for root in allowed:
            if re.search(r"20(?:2\d|[3-9]\d)", str(root)):
                continue
            root_cache_records = cache_records(root)
            configs = sorted(
                path for path in root.rglob("run_config.json")
                if safe_development_artifact(path, root)
            )
            if not configs:
                weekly_files = [
                    path for path in root.rglob("weekly_returns.csv")
                    if "baseline_pool" not in str(path)
                    and safe_development_artifact(path, root)
                ]
                complete = False
                for weekly_path in weekly_files:
                    try:
                        complete = len(campaign_assert_dates(pd.read_csv(weekly_path, usecols=["date"])["date"].drop_duplicates().tolist())) == CAMPAIGN_STAGE_SIZES["full"]
                    except (ValueError, CampaignFirewallError):
                        continue
                    if complete:
                        break
                design, lesson = qualitative_profile({}, root)
                entries.append({
                    "source_path": str(root),
                    "prompt_variant": root.name,
                    "complete": complete,
                    "harness_compatible": False,
                    "leaderboard_eligible": False,
                    "experiment_design": design,
                    "qualitative_lesson": lesson if complete else "incomplete_reference",
                })
                continue
            for config_path in configs:
                config = campaign_read_json(config_path)
                start = pd.Timestamp(config.get("start", CAMPAIGN_DEVELOPMENT_START))
                end = pd.Timestamp(config.get("end", CAMPAIGN_DEVELOPMENT_END))
                if start != CAMPAIGN_DEVELOPMENT_START or end != CAMPAIGN_DEVELOPMENT_END:
                    continue
                weekly_files = [
                    path for path in config_path.parent.rglob("weekly_returns.csv")
                    if "baseline_pool" not in str(path)
                    and safe_development_artifact(path, root)
                ]
                complete = False
                for weekly_path in weekly_files:
                    try:
                        values = campaign_assert_dates(
                            pd.read_csv(weekly_path, usecols=["date"])["date"]
                            .drop_duplicates()
                            .tolist()
                        )
                        if expected_dates and {str(value.date()) for value in values} == expected_dates:
                            complete = True
                            break
                    except (ValueError, CampaignFirewallError):
                        continue
                frozen = {
                    key: config.get(key)
                    for key in (
                        "model",
                        "temperature",
                        "thinking_mode",
                        "pool_sizes",
                        "book_sizes",
                        "llm_pool_size",
                        "anchor_offsets",
                        "schedule",
                        "anonymize_tickers",
                        "cost_bps_levels",
                        "output_contract",
                    )
                }
                temperature = pd.to_numeric(config.get("temperature"), errors="coerce")
                pool_sizes = config.get("pool_sizes")
                book_sizes = config.get("book_sizes")
                anchor_offsets = config.get("anchor_offsets")
                cost_bps_levels = config.get("cost_bps_levels")
                llm_pool_size = pd.to_numeric(config.get("llm_pool_size"), errors="coerce")
                harness_compatible = bool(
                    bool(config.get("prompt_hash"))
                    and str(config.get("model")) == CAMPAIGN_SCORER_MODEL
                    and bool(np.isfinite(temperature))
                    and np.isclose(float(temperature), CAMPAIGN_TEMPERATURE)
                    and str(config.get("thinking_mode")) == CAMPAIGN_THINKING_MODE
                    and isinstance(config.get("anonymize_tickers"), bool)
                    and isinstance(pool_sizes, (list, tuple))
                    and tuple(pool_sizes) == (60,)
                    and isinstance(book_sizes, (list, tuple))
                    and tuple(book_sizes) == (30,)
                    and bool(np.isfinite(llm_pool_size))
                    and int(llm_pool_size) == 60
                    and isinstance(anchor_offsets, (list, tuple))
                    and tuple(anchor_offsets) == (3,)
                    and str(config.get("schedule", "anchors")) == "anchors"
                    and isinstance(cost_bps_levels, (list, tuple))
                    and tuple(float(value) for value in cost_bps_levels) == (0.0,)
                    and str(config.get("output_contract", CAMPAIGN_OUTPUT_CONTRACT))
                    == CAMPAIGN_OUTPUT_CONTRACT
                )
                matching_caches: list[dict[str, Any]] = []
                for cache in root_cache_records:
                    metadata = cache["metadata"]
                    cache_temperature = pd.to_numeric(
                        metadata.get("temperature"), errors="coerce"
                    )
                    prompt_hash = str(config.get("prompt_hash", ""))
                    cached_hash = str(metadata.get("prompt_hash", ""))
                    if (
                        prompt_hash
                        and cached_hash
                        and (prompt_hash.startswith(cached_hash) or cached_hash.startswith(prompt_hash))
                        and str(metadata.get("model")) == str(config.get("model"))
                        and bool(np.isfinite(cache_temperature))
                        and np.isclose(float(cache_temperature), float(temperature))
                        and str(metadata.get("thinking_mode")) == str(config.get("thinking_mode"))
                        and bool(metadata.get("anonymize_tickers")) == bool(config.get("anonymize_tickers"))
                        and str(metadata.get("output_contract")) == CAMPAIGN_OUTPUT_CONTRACT
                    ):
                        matching_caches.append(cache)
                cache_complete = any(bool(row["complete"]) for row in matching_caches)
                complete_cache_path = next(
                    (row["path"] for row in matching_caches if row["complete"]),
                    None,
                )
                experiment_design, qualitative_lesson = qualitative_profile(
                    config, config_path.parent
                )
                if not complete:
                    qualitative_lesson = "incomplete_reference"
                entries.append({
                    "source_path": str(config_path.parent),
                    "prompt_hash": config.get("prompt_hash"),
                    "prompt_variant": config.get("prompt_variant", "unknown"),
                    "config_hash": campaign_sha256(campaign_json_bytes(frozen)),
                    "complete": complete,
                    "cache_complete": bool(cache_complete),
                    "cache_path": complete_cache_path,
                    "harness_compatible": bool(harness_compatible),
                    "leaderboard_eligible": bool(complete and cache_complete and harness_compatible),
                    "model": config.get("model"),
                    "temperature": config.get("temperature"),
                    "thinking_mode": config.get("thinking_mode"),
                    "anonymize_tickers": bool(config.get("anonymize_tickers")),
                    "experiment_design": experiment_design,
                    "qualitative_lesson": qualitative_lesson,
                })
    groups: dict[tuple[Any, Any], list[int]] = {}
    for index, entry in enumerate(entries):
        key = (entry.get("prompt_hash"), entry.get("config_hash"))
        if all(key):
            groups.setdefault(key, []).append(index)
        entry["duplicate"] = False
    for indices in groups.values():
        canonical = min(
            indices,
            key=lambda index: (
                not bool(entries[index].get("leaderboard_eligible")),
                not bool(entries[index].get("complete")),
                not bool(entries[index].get("cache_complete")),
                str(entries[index].get("source_path", "")),
            ),
        )
        for index in indices:
            entries[index]["duplicate"] = index != canonical
            if index != canonical:
                entries[index]["leaderboard_eligible"] = False
    for entry in entries:
        entry["experiment_id"] = "catalog-" + campaign_sha256(
            campaign_json_bytes(
                {
                    "source_path": entry.get("source_path"),
                    "prompt_hash": entry.get("prompt_hash"),
                    "config_hash": entry.get("config_hash"),
                }
            )
        )[:12]
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "generated_at": campaign_utc_now(),
        "firewall": {"maximum_artifact_end": str(CAMPAIGN_DEVELOPMENT_END.date()), "holdout_opened": False},
        "summary": {"entries": len(entries), "complete_unique": sum(bool(row.get("leaderboard_eligible")) for row in entries)},
        "entries": entries,
    }


CAMPAIGN_PATCH_LEAKAGE = re.compile(
    r"(?:\b(?:19|20)\d{2}\b|\b\d{4}-\d{2}-\d{2}\b|\$[A-Z]{1,6}\b|"
    r"\b(?:ticker|symbol|company)\s*[=:]\s*[A-Z]{1,6}\b|"
    r"\b(?:AAPL|AMZN|GOOG|GOOGL|META|MSFT|NVDA|QQQ|SPY|TSLA)\b|"
    r"\bsecurity\s+(?:id|identifier)s?\s+(?:ending|starting|matching|with)\b|"
    r"\bSEC_\d+\b)",
    flags=re.IGNORECASE,
)
CAMPAIGN_FORMAT_REPAIR_APPENDICES = (
    "Return only one strict valid JSON object with exactly the required long_scores and "
    "short_scores keys; include no extra prose.",
    "Respond with a single parseable JSON object using exactly the long_scores and "
    "short_scores keys and no surrounding text.",
)


def campaign_sanitize_rejected_proposal(proposal: Any) -> dict[str, Any]:
    """Keep proposal intent for audit without retaining raw prompts or identifiers."""
    source = proposal if isinstance(proposal, dict) else {}

    def clean(value: Any, *, limit: int = 12_000) -> str:
        text = str(value or "")
        text = CAMPAIGN_PATCH_LEAKAGE.sub("[REDACTED]", text)
        return text[:limit]

    patch = source.get("patch") if isinstance(source.get("patch"), dict) else {}
    references = source.get("prior_experiment_ids", [])
    if not isinstance(references, list):
        references = []
    return {
        "mutation_mode": clean(source.get("mutation_mode"), limit=100),
        "score_min": source.get("score_min"),
        "score_max": source.get("score_max"),
        "score_direction": clean(source.get("score_direction"), limit=100),
        "anonymize_tickers": source.get("anonymize_tickers"),
        "parent_id": clean(source.get("parent_id"), limit=200),
        "patch": {
            "operation": clean(patch.get("operation") or patch.get("op"), limit=100),
            "old_text": clean(patch.get("old_text")),
            "new_text": clean(patch.get("new_text")),
        },
        "hypothesis": clean(source.get("hypothesis"), limit=2_000),
        "mechanism": clean(source.get("mechanism"), limit=2_000),
        "expected_failure": clean(source.get("expected_failure"), limit=2_000),
        "prior_experiment_ids": [clean(value, limit=300) for value in references[:20]],
    }


def campaign_record_rejected_proposal(
    campaign_dir: Path,
    proposal: Any,
    *,
    candidate_id: str,
    generation: str,
    parent_id: str,
    reason: str,
) -> None:
    timestamp = campaign_utc_now()
    proposal_hash = campaign_sha256(campaign_json_bytes(proposal))
    record = {
        "timestamp": timestamp,
        "rejection_id": "rejected-" + campaign_sha256(
            f"{timestamp}:{candidate_id}:{proposal_hash}:{reason}"
        )[:12],
        "candidate_id": candidate_id,
        "generation": generation,
        "parent_id": parent_id,
        "reason": str(reason)[:1_000],
        "proposal_hash": proposal_hash,
        "proposal": campaign_sanitize_rejected_proposal(proposal),
    }
    with (campaign_dir / "rejected_proposals.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def campaign_materialize_prompt_spec(
    proposal: dict[str, Any],
    *,
    candidate_id: str,
    generation: str,
    parent_id: str,
    parent_prompt: str,
    parent_hash: str,
    side_scope: str = "both",
    replicate_id: str = "r1",
) -> dict[str, Any]:
    if generation not in {"G0", "G1"}:
        raise CampaignError("generation must be G0 or G1")
    mutation_mode = str(proposal.get("mutation_mode", ""))
    if mutation_mode not in {"additive", "substitutive", "rewrite", "format_repair"}:
        raise CampaignError("mutation_mode must be additive, substitutive, rewrite, or format_repair")
    if mutation_mode == "format_repair" and generation != "G1":
        raise CampaignError("A format_repair is permitted only for a G1 child")
    side_scope = str(side_scope).strip().lower()
    if side_scope not in {"both", "long", "short"}:
        raise CampaignError("side_scope must be both, long, or short")
    if side_scope != "both" and mutation_mode == "format_repair":
        raise CampaignError("Side-only campaigns do not permit two-sided format-repair appendices")
    patch = proposal.get("patch")
    if not isinstance(patch, dict) or patch.get("operation") != "replace_once":
        raise CampaignError("Every candidate must contain one replace_once patch")
    old_text = str(patch.get("old_text", ""))
    new_text = str(patch.get("new_text", ""))
    if not old_text or not new_text or old_text == new_text or parent_prompt.count(old_text) != 1:
        raise CampaignError("replace_once old_text must occur exactly once and materially change")
    if mutation_mode == "format_repair":
        if new_text.count(old_text) != 1:
            raise CampaignError("A format_repair may only append a wrapper-compliance clarification")
        repair_text = new_text.replace(old_text, "", 1).strip()
        if repair_text not in CAMPAIGN_FORMAT_REPAIR_APPENDICES:
            raise CampaignError(
                "A format_repair must use one frozen JSON/schema wrapper clarification"
            )
    old_tokens = re.findall(r"\w+|[^\w\s]", old_text)
    new_tokens = re.findall(r"\w+|[^\w\s]", new_text)
    changed_blocks = 0
    changed_tokens = 0
    for tag, old_start, old_end, new_start, new_end in difflib.SequenceMatcher(
        a=old_tokens, b=new_tokens, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        changed_blocks += 1
        changed_tokens += max(old_end - old_start, new_end - new_start)
    prompt_text = parent_prompt.replace(old_text, new_text, 1)
    parent_leakage = {
        match.group(0).lower() for match in CAMPAIGN_PATCH_LEAKAGE.finditer(parent_prompt)
    }
    candidate_leakage = {
        match.group(0).lower() for match in CAMPAIGN_PATCH_LEAKAGE.finditer(prompt_text)
    }
    if not candidate_leakage.issubset(parent_leakage):
        raise CampaignError("Mutation contains a dated, ticker, or security-identifier reference")
    try:
        from mean_reversion_llm_v2 import _feature_glossary_block

        parent_glossary = _feature_glossary_block(parent_prompt)
        candidate_glossary = _feature_glossary_block(prompt_text)
    except ValueError as exc:
        raise CampaignError(str(exc)) from exc
    if candidate_glossary != parent_glossary:
        raise CampaignError("The feature glossary is immutable")
    if side_scope == "both":
        if "long_scores" not in prompt_text or "short_scores" not in prompt_text:
            raise CampaignError("Prompt must retain the long_scores and short_scores JSON arrays")
    elif '"scores"' not in prompt_text:
        raise CampaignError("A side-scoped prompt must retain its single scores JSON array")
    try:
        score_min = float(proposal.get("score_min"))
        score_max = float(proposal.get("score_max"))
    except (TypeError, ValueError) as exc:
        raise CampaignError("Candidate must declare numeric score_min and score_max") from exc
    score_direction = str(proposal.get("score_direction", "")).strip().lower()
    if not np.isfinite(score_min) or not np.isfinite(score_max) or score_max <= score_min:
        raise CampaignError("score_min and score_max must be finite with score_max > score_min")
    if score_direction not in {"higher_is_better", "lower_is_better"}:
        raise CampaignError("score_direction must be higher_is_better or lower_is_better")
    anonymize_tickers = proposal.get("anonymize_tickers", True)
    if not isinstance(anonymize_tickers, bool):
        raise CampaignError("anonymize_tickers must be a JSON boolean")
    approximate_prompt_tokens = (len(prompt_text) + 3) // 4
    if approximate_prompt_tokens > CAMPAIGN_PROMPT_MAX_TOKENS:
        raise CampaignError("Materialized prompt exceeds the frozen token-size limit")
    supplied = proposal.get("prompt_text")
    if supplied is not None and str(supplied) != prompt_text:
        raise CampaignError("Supplied prompt_text does not match its replace_once patch")
    prompt_hash = campaign_sha256(prompt_text)
    spec = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "generation": generation,
        "parent_id": parent_id,
        "mutation_mode": mutation_mode,
        "patch": {"operation": "replace_once", "old_text": old_text, "new_text": new_text},
        "hypothesis": str(proposal.get("hypothesis", "")).strip(),
        "mechanism": str(proposal.get("mechanism", "")).strip(),
        "expected_failure": str(proposal.get("expected_failure", "")).strip(),
        "prior_experiment_ids": [str(value).strip() for value in proposal.get("prior_experiment_ids", [])],
        "prompt_text": prompt_text,
        "prompt_hash": prompt_hash,
        "output_contract": CAMPAIGN_OUTPUT_CONTRACT,
        "score_min": score_min,
        "score_max": score_max,
        "score_direction": score_direction,
        "anonymize_tickers": anonymize_tickers,
        "side_scope": side_scope,
        "replicate_id": str(replicate_id),
        "model": CAMPAIGN_SCORER_MODEL,
        "temperature": CAMPAIGN_TEMPERATURE,
        "thinking_mode": CAMPAIGN_THINKING_MODE,
        "compliance": {
            "system_prompt_only": True,
            "fixed_harness": True,
            "ticker_visibility_candidate_selected": True,
            "ticker_identity_policy": (
                "opaque" if anonymize_tickers else "visible_identifier_familiarity_allowed"
            ),
            "no_external_knowledge_beyond_ticker_identity": True,
            "gross_only": True,
            "approximate_prompt_tokens": approximate_prompt_tokens,
            "changed_tokens": changed_tokens,
            "change_blocks": changed_blocks,
        },
    }
    if generation == "G1":
        spec["parent_prompt_text"] = parent_prompt
        spec["parent_prompt_hash"] = parent_hash
    if not spec["hypothesis"] or not spec["mechanism"] or not spec["expected_failure"]:
        raise CampaignError("Candidate hypothesis, mechanism, and expected_failure are required")
    if not spec["prior_experiment_ids"] or any(not value for value in spec["prior_experiment_ids"]):
        raise CampaignError("Candidate prior_experiment_ids must contain at least one reference")
    return spec


def campaign_researcher_payload(user_prompt: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": CAMPAIGN_RESEARCHER_MODEL,
        "temperature": CAMPAIGN_TEMPERATURE,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Pro prompt researcher improving cross-sectional mean-reversion ranking. "
                    "Diagnose behavioral weaknesses and propose coherent system-prompt rewrites. The supplied "
                    "feature glossary is factual and immutable: never edit it, reinterpret its definitions, "
                    "invent derived features, or claim access to unsupplied information. Ticker visibility is a "
                    "candidate-level choice: a visible candidate may use familiarity implied by the supplied ticker "
                    "identifier, but proposals must never hard-code named securities or dated facts. Ideas may be simple, "
                    "generic, unconventional, or extensive; empirical tests decide their merit. Return strict "
                    "JSON only. Never choose winners or request validation data."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "max_tokens": CAMPAIGN_PRO_MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
    }
    technical_llm.apply_deepseek_request_mode(payload, CAMPAIGN_RESEARCHER_MODEL, CAMPAIGN_THINKING_MODE)
    return payload


def campaign_call_researcher(user_prompt: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = campaign_researcher_payload(user_prompt)
    headers = {"Authorization": f"Bearer {technical_llm.get_deepseek_api_key()}"}
    with httpx.Client(timeout=technical_llm.PORTFOLIO_REQUEST_TIMEOUT) as client:
        response = client.post(technical_llm.get_deepseek_url(), json=payload, headers=headers)
        response.raise_for_status()
    result = response.json()
    content = str(result.get("choices", [{}])[0].get("message", {}).get("content", ""))
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    parsed = json.loads(match.group(0) if match else content)
    proposals = parsed.get("candidates", [])
    if not isinstance(proposals, list):
        raise CampaignError("Researcher response must contain a candidates list")
    usage = result.get("usage", {}) if isinstance(result.get("usage"), dict) else {}
    return proposals, {"calls": 1, "output_tokens": int(usage.get("completion_tokens", 0) or 0)}


def campaign_read_manifest(
    path: Path,
    *,
    validation: bool = False,
    expected_hash: str | None = None,
    expected_file_hash: str | None = None,
) -> dict[str, Any]:
    if expected_file_hash is not None and campaign_file_hash(path) != str(expected_file_hash):
        raise CampaignFirewallError(f"Date manifest file changed after it was frozen: {path}")
    manifest = campaign_read_json(path)
    dates = campaign_assert_dates(manifest.get("dates", []), validation=validation)
    if manifest.get("manifest_hash") != campaign_manifest_hash(dates):
        raise CampaignFirewallError(f"Date manifest hash mismatch: {path}")
    if expected_hash is not None and manifest.get("manifest_hash") != str(expected_hash):
        raise CampaignFirewallError(f"Date manifest differs from the campaign's frozen hash: {path}")
    expected_stage = "validation" if validation else path.stem
    expected_purpose = "validation" if validation else "development"
    expected_campaign_id = path.parent.parent.name
    if (
        str(manifest.get("stage")) != expected_stage
        or str(manifest.get("purpose")) != expected_purpose
        or str(manifest.get("campaign_id")) != expected_campaign_id
        or int(manifest.get("date_count", -1)) != len(dates)
    ):
        raise CampaignFirewallError(f"Date manifest metadata mismatch: {path}")
    if validation:
        exit_dates = campaign_assert_dates(manifest.get("exit_dates", []), validation=True)
        if len(exit_dates) != len(dates):
            raise CampaignFirewallError("Validation signal and exit date counts must match")
        if manifest.get("exit_date_hash") != campaign_manifest_hash(exit_dates):
            raise CampaignFirewallError("Validation exit-date hash mismatch")
    return manifest


def campaign_expected_manifest_hash(
    state: dict[str, Any], stage: str, *, validation: bool
) -> str:
    if validation:
        value = state.get("validation", {}).get("manifest_hash")
    else:
        value = state.get("development", {}).get("manifest_hashes", {}).get(stage)
    if not value:
        raise CampaignFirewallError(f"Campaign state has no frozen manifest hash for {stage}")
    return str(value)


def campaign_expected_manifest_file_hash(
    state: dict[str, Any], stage: str, *, validation: bool
) -> str:
    if validation:
        value = state.get("validation", {}).get("manifest_file_hash")
    else:
        value = state.get("development", {}).get("manifest_file_hashes", {}).get(stage)
    if not value:
        raise CampaignFirewallError(f"Campaign state has no frozen manifest-file hash for {stage}")
    return str(value)


def campaign_cache_budget_counters(paths: Sequence[Path]) -> tuple[int, int]:
    calls = 0
    output_tokens = 0
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = campaign_read_json(path)
        except (OSError, json.JSONDecodeError, CampaignError):
            continue
        records = payload.get("llm_call_usage", [])
        if not isinstance(records, list):
            continue
        calls += sum(isinstance(record, dict) for record in records)
        output_tokens += sum(
            int(record.get("output_tokens_charged", 0) or 0)
            for record in records if isinstance(record, dict)
        )
    return calls, output_tokens


def campaign_reconcile_cache_usage(
    state: dict[str, Any], usage_key: str, paths: Sequence[Path]
) -> None:
    """Charge each persisted provider-attempt record exactly once campaign-wide."""
    seen = set(state.setdefault("cache_usage_record_hashes", {}).get(usage_key, []))
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            records = campaign_read_json(path).get("llm_call_usage", [])
        except (OSError, json.JSONDecodeError, CampaignError):
            continue
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            identity = campaign_sha256(
                campaign_json_bytes({"cache": str(path.resolve()), "record": record})
            )
            if identity in seen:
                continue
            seen.add(identity)
            state["usage"][usage_key]["flash_calls"] += 1
            state["usage"][usage_key]["flash_output_tokens"] += int(
                record.get("output_tokens_charged", 0) or 0
            )
    state["cache_usage_record_hashes"][usage_key] = sorted(seen)


def campaign_runner_command(
    state: dict[str, Any],
    campaign_dir: Path,
    spec_path: Path,
    stage: str,
    *,
    validation: bool = False,
) -> tuple[list[str], Path]:
    manifest_path = campaign_dir / "manifests" / ("validation.json" if validation else f"{stage}.json")
    manifest = campaign_read_manifest(
        manifest_path,
        validation=validation,
        expected_hash=campaign_expected_manifest_hash(state, stage, validation=validation),
        expected_file_hash=campaign_expected_manifest_file_hash(state, stage, validation=validation),
    )
    spec = campaign_read_json(spec_path)
    side_scope = str(spec.get("side_scope", "both"))
    anonymize_tickers = spec.get("anonymize_tickers", True)
    if not isinstance(anonymize_tickers, bool):
        raise CampaignFirewallError("Prompt spec anonymize_tickers must be a JSON boolean")
    output_dir = campaign_dir / ("validation" if validation else f"candidates/{spec_path.parent.name}/{stage}")
    cache_dir = (
        campaign_dir / "validation/cache/treatment"
        if validation
        else campaign_dir / "candidates" / spec_path.parent.name / "cache"
    )
    progress_path = output_dir / "runner_events.jsonl"
    start = CAMPAIGN_VALIDATION_START if validation else CAMPAIGN_DEVELOPMENT_START
    end = CAMPAIGN_VALIDATION_END if validation else CAMPAIGN_DEVELOPMENT_END
    budget = state["budgets"]["validation" if validation else "development"]
    usage = state["usage"]["validation" if validation else "development"]
    treatment_paths = sorted(cache_dir.glob("*.json"))
    treatment_calls, treatment_tokens = campaign_cache_budget_counters(treatment_paths)
    remaining_calls = max(1, int(budget["max_flash_calls"]) - int(usage.get("flash_calls", 0)))
    remaining_tokens = max(1, int(budget["max_flash_output_tokens"]) - int(usage.get("flash_output_tokens", 0)))
    call_limit = max(1, remaining_calls + treatment_calls)
    token_limit = max(1, remaining_tokens + treatment_tokens)
    command = [
        sys.executable,
        str(Path(state.get("runner", CAMPAIGN_RUNNER)).resolve()),
        "--phase", "llm-cache" if side_scope != "both" else "full",
        "--start", str(start.date()),
        "--end", str(end.date()),
        "--output-dir", str(output_dir),
        "--cache-dir", str(cache_dir),
        "--experimental-prompt-spec", str(spec_path),
        "--signal-dates-file", str(manifest_path),
        "--progress-file", str(progress_path),
        "--schedule", "anchors",
        "--anchor-offsets", "3",
        "--pool-sizes", "60",
        "--book-sizes", "30",
        "--llm-pool-size", "60",
        "--model", CAMPAIGN_SCORER_MODEL,
        "--temperature", "1",
        "--thinking-mode", CAMPAIGN_THINKING_MODE,
        "--anonymize-tickers" if anonymize_tickers else "--no-anonymize-tickers",
        "--cost-bps-levels", "0",
        "--run-output-token-budget", str(token_limit),
        "--run-call-budget", str(call_limit),
        "--llm-max-attempts", "2",
        "--experimental-override",
    ]
    if validation:
        command.extend(["--validation-freeze", str(campaign_dir / "validation/validation_freeze.json")])
    if int(manifest["date_count"]) <= 0:
        raise CampaignError("Cannot launch an empty experiment")
    return command, output_dir


def campaign_progress_usage(path: Path) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "flash_calls": 0,
        "flash_output_tokens": 0,
        "schema_retries": 0,
        "transient_retries": 0,
        "schema_retry_dates": [],
        "format_only_dates": [],
        "format_only_retries": 0,
        "non_format_schema_retries": 0,
        "completed_dates": 0,
    }
    completed_dates: set[str] = set()
    if not path.exists():
        return usage
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = str(event.get("event", event.get("type", ""))).lower()
            payload = event.get("payload", event)
            prompt_identity = str(payload.get("prompt_identity", ""))
            if prompt_identity and prompt_identity != "autoresearch-experimental":
                continue
            if kind == "llm_call_started":
                usage["flash_calls"] += 1
            if kind == "llm_call_succeeded":
                usage["flash_output_tokens"] += int(payload.get("output_tokens", 0) or 0)
            if kind == "llm_call_failed":
                usage["flash_output_tokens"] += int(payload.get("output_tokens", 0) or 0)
                if bool(payload.get("schema_error", False)):
                    usage["schema_retries"] += 1
                    signal_date = str(payload.get("signal_date", ""))
                    if signal_date and signal_date not in usage["schema_retry_dates"]:
                        usage["schema_retry_dates"].append(signal_date)
                    wrapper_only = bool(payload.get("format_only_repair_eligible", False))
                    usage["format_only_retries" if wrapper_only else "non_format_schema_retries"] += 1
                    if wrapper_only and signal_date and signal_date not in usage["format_only_dates"]:
                        usage["format_only_dates"].append(signal_date)
                elif bool(payload.get("transient_error", False)):
                    usage["transient_retries"] += 1
            if kind == "cache_entry_saved":
                signal_date = str(payload.get("signal_date", ""))
                if signal_date:
                    completed_dates.add(signal_date)
    usage["completed_dates"] = len(completed_dates)
    return usage


def campaign_progress_has_event(path: Path, event_name: str) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                if str(json.loads(line).get("event", "")) == event_name:
                    return True
            except json.JSONDecodeError:
                continue
    return False


def campaign_latest_progress_event(path: Path, event_name: str) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(event.get("event", "")) == event_name:
                latest = event
    return latest


def campaign_run_subprocess(
    campaign_dir: Path,
    state: dict[str, Any],
    command: list[str],
    output_dir: Path,
    *,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "runner.log"
    campaign_append_event(campaign_dir, "runner_started", candidate_id=state.get("current_candidate"), stage=state.get("current_stage"), payload={"argv": command[1:], "log": str(log_path)})
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = popen(
            command,
            cwd=Path(__file__).parent,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        state["runner_pid"] = process.pid
        campaign_save(campaign_dir, state)
        while process.poll() is None:
            latest = campaign_load(campaign_dir)
            if latest.get("stop_requested") or (campaign_dir / "STOP").exists():
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                state["runner_pid"] = None
                campaign_save(campaign_dir, state, "runner_stopped")
                raise CampaignStop("Stop requested")
            active = campaign_progress_usage(output_dir / "runner_events.jsonl")
            state["active_stage_usage"] = active
            state["completed_dates"] = min(
                int(state.get("planned_dates", 0)), int(active.get("completed_dates", 0))
            )
            campaign_save(campaign_dir, state)
            time.sleep(1.0)
        state["runner_pid"] = None
        state["active_stage_usage"] = campaign_progress_usage(output_dir / "runner_events.jsonl")
        campaign_save(campaign_dir, state)
        if process.returncode != 0:
            raise CampaignError(f"V2 runner failed with exit code {process.returncode}; see {log_path}")


def campaign_resolve_artifact_dir(output_dir: Path, role: str) -> Path:
    config_path = output_dir / "run_config.json"
    config = campaign_read_json(config_path) if config_path.exists() else {}
    resolver = config.get("output_resolver", {}) if isinstance(config.get("output_resolver"), dict) else {}
    resolver_key = {"treatment": "treatment_dir", "control": "paired_control_dir", "trail5": "baseline_dir"}[role]
    resolver_raw = resolver.get(resolver_key)
    if resolver_raw:
        candidate = output_dir / str(resolver_raw)
        if (candidate / "weekly_returns.csv").exists():
            return candidate
    keys = {
        "treatment": ("treatment_output_dir", "experimental_treatment_dir", "treatment_run_dir"),
        "control": ("control_output_dir", "paired_control_output_dir", "fresh_control_run_dir"),
        "trail5": ("baseline_output_dir", "trail5_output_dir"),
    }[role]
    for key in keys:
        raw = config.get(key)
        if raw:
            candidate = Path(raw)
            candidate = candidate if candidate.is_absolute() else output_dir / candidate
            if (candidate / "weekly_returns.csv").exists():
                return candidate
    conventional = {
        "treatment": "llm_pool_experimental",
        "control": "llm_pool_v2_minimal_change_fresh_control",
        "trail5": "baseline_pool",
    }[role]
    candidate = output_dir / "pool60_pos30/anchor_3" / conventional
    if (candidate / "weekly_returns.csv").exists():
        return candidate
    choices = sorted((output_dir / "pool60_pos30/anchor_3").glob("*/weekly_returns.csv"))
    if role == "trail5":
        choices = [path for path in choices if "baseline" in path.parent.name]
    elif role == "control":
        choices = [path for path in choices if "control" in path.parent.name]
    else:
        choices = [path for path in choices if "baseline" not in path.parent.name and "control" not in path.parent.name]
    if len(choices) != 1:
        raise CampaignError(f"Could not resolve {role} weekly_returns.csv below {output_dir}")
    return choices[0].parent


def campaign_load_returns(
    run_dir: Path,
    expected_dates: Sequence[Any],
    *,
    validation: bool = False,
    allow_superset: bool = False,
) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "weekly_returns.csv")
    if "date" not in frame.columns:
        raise CampaignError(f"Missing date column in {run_dir / 'weekly_returns.csv'}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    campaign_assert_dates(frame["date"].tolist(), validation=validation)
    expected = {pd.Timestamp(value).normalize() for value in expected_dates}
    observed = set(frame["date"])
    coverage_ok = expected.issubset(observed) if allow_superset else observed == expected
    if not coverage_ok:
        raise CampaignError(f"Incomplete date coverage in {run_dir}: {len(observed)}/{len(expected)}")
    if allow_superset:
        frame = frame[frame["date"].isin(expected)].copy()
    if "capital_c2c" in frame.columns:
        frame["capital_c2c"] = pd.to_numeric(frame["capital_c2c"], errors="coerce")
    else:
        source = "ls_c2c_0" if "ls_c2c_0" in frame.columns else "ls_c2c"
        frame["capital_c2c"] = 0.5 * pd.to_numeric(frame[source], errors="coerce")
    if {"long_ret_c2c", "short_ret_c2c"}.issubset(frame.columns):
        frame["long_contribution"] = 0.5 * pd.to_numeric(frame["long_ret_c2c"], errors="coerce")
        frame["short_contribution"] = -0.5 * pd.to_numeric(frame["short_ret_c2c"], errors="coerce")
    else:
        frame["long_contribution"] = np.nan
        frame["short_contribution"] = np.nan
    return frame.sort_values("date").reset_index(drop=True)


def campaign_return_metrics(values: pd.Series, *, chronological: bool) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    annual = float(clean.mean() * PERIODS_PER_YEAR) if not clean.empty else np.nan
    volatility = float(clean.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(clean) > 1 else np.nan
    result: dict[str, float | int] = {
        "n_periods": int(len(clean)),
        "annual_return_pct": annual * 100.0,
        "sharpe": annual / volatility if volatility > 0 else np.nan,
        "hit_rate_pct": float((clean > 0).mean() * 100.0) if not clean.empty else np.nan,
        "worst_decile_expected_shortfall_pct": float(clean.nsmallest(max(1, int(np.ceil(len(clean) * 0.1)))).mean() * 100.0) if not clean.empty else np.nan,
    }
    if chronological:
        result["cagr_pct"] = safe_cagr(clean)
        result["max_drawdown_pct"] = max_drawdown_pct(clean)
        result["total_return_pct"] = float(((1.0 + clean).prod() - 1.0) * 100.0) if not clean.empty else np.nan
    return result


def campaign_moving_block_bootstrap(
    candidate: Sequence[float],
    control: Sequence[float],
    *,
    n_boot: int,
    block_size: int = 4,
    seed: int = 1701,
) -> dict[str, float | int]:
    delta = np.asarray(candidate, dtype=float) - np.asarray(control, dtype=float)
    delta = delta[np.isfinite(delta)]
    if not len(delta):
        raise CampaignError("Cannot bootstrap an empty paired return series")
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot), dtype=float)
    for sample_idx in range(int(n_boot)):
        sampled: list[float] = []
        while len(sampled) < len(delta):
            start = int(rng.integers(0, len(delta)))
            sampled.extend(delta[(start + offset) % len(delta)] for offset in range(block_size))
        means[sample_idx] = float(np.mean(sampled[:len(delta)]) * PERIODS_PER_YEAR * 100.0)
    return {
        "n_boot": int(n_boot),
        "block_size": int(block_size),
        "mean_annualized_delta_pct": float(delta.mean() * PERIODS_PER_YEAR * 100.0),
        "lower_90_pct": float(np.quantile(means, 0.10)),
        "upper_90_pct": float(np.quantile(means, 0.90)),
        "lower_one_sided_95_pct": float(np.quantile(means, 0.05)),
        "lower_one_sided_97_5_pct": float(np.quantile(means, 0.025)),
        "lower_95_pct": float(np.quantile(means, 0.025)),
        "upper_95_pct": float(np.quantile(means, 0.975)),
        "p_not_positive": float(np.mean(means <= 0.0)),
    }


def campaign_smoke_gate(signal_path: Path, expected_dates: Sequence[Any], control_signal_path: Path | None = None) -> dict[str, Any]:
    checks = pd.read_csv(signal_path)
    checks["date"] = pd.to_datetime(checks["date"]).dt.normalize()
    checks["pool_side"] = checks["pool_side"].astype(str).str.upper()
    score_col = next((name for name in ("llm_score", "final_score", "score") if name in checks.columns), None)
    if not score_col or "pool_side" not in checks.columns:
        raise CampaignError("Smoke signal_checks lacks score or pool_side")
    per_date: list[dict[str, Any]] = []
    hard_floor = True
    primary_passes = {"LONG": 0, "SHORT": 0}
    for (date, side), group in checks.groupby(["date", "pool_side"]):
        values = pd.to_numeric(group[score_col], errors="coerce").dropna()
        row = {"date": str(pd.Timestamp(date).date()), "side": str(side).upper(), "unique": int(values.nunique()), "std": float(values.std(ddof=0)), "range": float(values.max() - values.min())}
        per_date.append(row)
        hard_floor = hard_floor and row["unique"] >= 3 and row["range"] >= 5.0
        if row["unique"] >= 5 and row["std"] >= 3.0 and row["range"] >= 10.0:
            primary_passes[row["side"]] = primary_passes.get(row["side"], 0) + 1
    expected_keys = {(pd.Timestamp(date).normalize(), side) for date in expected_dates for side in ("LONG", "SHORT")}
    observed_keys = {(pd.Timestamp(row["date"]).normalize(), row["side"]) for row in per_date}
    complete_side_coverage = observed_keys == expected_keys
    pass_gate = complete_side_coverage and hard_floor and all(primary_passes.get(side, 0) >= 4 for side in ("LONG", "SHORT"))
    warning = False
    if control_signal_path and control_signal_path.exists():
        control = pd.read_csv(control_signal_path)
        if "date" in control.columns:
            expected_control_dates = {
                pd.Timestamp(value).normalize() for value in expected_dates
            }
            control_dates = pd.to_datetime(control["date"]).dt.normalize()
            control = control[control_dates.isin(expected_control_dates)].copy()
        control_score = next((name for name in ("llm_score", "final_score", "score") if name in control.columns), None)
        if control_score:
            candidate_std = np.median([row["std"] for row in per_date])
            control_std = pd.to_numeric(control[control_score], errors="coerce").groupby([pd.to_datetime(control["date"]), control["pool_side"]]).std(ddof=0).median()
            warning = bool(np.isfinite(control_std) and candidate_std < 0.5 * float(control_std))
    return {"pass": bool(pass_gate), "complete_side_coverage": bool(complete_side_coverage), "hard_floor": bool(hard_floor), "primary_passes": primary_passes, "low_dispersion_warning": warning, "per_date": per_date}


def _campaign_full_inference_diagnostics(
    result: dict[str, Any],
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    treatment_dir: Path,
    control_dir: Path,
    output_dir: Path,
) -> None:
    """Attach reporting-only full diagnostics without changing the gross judge."""
    from mean_reversion_research_diagnostics import (
        alpha_concentration,
        load_checks,
        selection_diagnostics,
    )

    delta_control = treatment["capital_c2c"] - control["capital_c2c"]
    result["bootstrap_long_vs_incumbent"] = campaign_moving_block_bootstrap(
        treatment["long_contribution"],
        control["long_contribution"],
        n_boot=5_000,
        seed=1703,
    )
    result["bootstrap_short_vs_incumbent"] = campaign_moving_block_bootstrap(
        treatment["short_contribution"],
        control["short_contribution"],
        n_boot=5_000,
        seed=1704,
    )
    concentration = alpha_concentration(delta_control)
    result["alpha_concentration"] = concentration.to_dict(orient="records")
    trimmed_5pct = concentration.loc[
        concentration["top_fraction_pct"].eq(5.0),
        "trimmed_annualized_lift_pct",
    ]
    result["metrics"]["trimmed_5pct_annualized_lift_pct"] = (
        float(trimmed_5pct.iloc[0]) if not trimmed_5pct.empty else np.nan
    )

    grouped = treatment[["date", "capital_c2c", "regime"]].rename(
        columns={"capital_c2c": "candidate_return"}
    ).merge(
        control[["date", "capital_c2c"]].rename(
            columns={"capital_c2c": "control_return"}
        ),
        on="date",
        validate="one_to_one",
    )
    grouped["year"] = grouped["date"].dt.year
    grouped["paired_lift"] = grouped["candidate_return"] - grouped["control_return"]

    def grouped_rows(column: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for value, group in grouped.groupby(column, dropna=False, sort=True):
            lift = group["paired_lift"]
            rows.append(
                {
                    column: value,
                    "n_dates": int(len(group)),
                    "mean_annualized_lift_pct": float(
                        lift.mean() * PERIODS_PER_YEAR * 100.0
                    ),
                    "median_return_lift_pct": float(lift.median() * 100.0),
                    "paired_hit_rate_pct": float((lift > 0.0).mean() * 100.0),
                }
            )
        return rows

    result["lift_by_year"] = grouped_rows("year")
    result["lift_by_market_regime"] = grouped_rows("regime")
    control_turnover = 0.5 * (
        pd.to_numeric(control["turnover_long"], errors="coerce").mean()
        + pd.to_numeric(control["turnover_short"], errors="coerce").mean()
    )
    incremental_turnover = float(
        result["metrics"]["avg_turnover_total"] - control_turnover
    )
    result["incremental_cost_sensitivity"] = [
        {
            "cost_bps_per_side": cost_bps,
            "incremental_turnover": incremental_turnover,
            "net_annualized_paired_lift_pct": float(
                result["bootstrap_vs_incumbent"]["mean_annualized_delta_pct"]
                - incremental_turnover
                * cost_bps
                / 10_000.0
                * PERIODS_PER_YEAR
                * 100.0
            ),
        }
        for cost_bps in (0.0, 5.0, 10.0, 20.0)
    ]

    selection = selection_diagnostics(
        load_checks(treatment_dir), load_checks(control_dir), book_size=30
    )
    result["selection_diagnostics"] = {
        key: value
        for key, value in selection.items()
        if not isinstance(value, pd.DataFrame)
    }
    diagnostic_dir = output_dir / "full_stage_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    concentration.to_csv(diagnostic_dir / "alpha_concentration.csv", index=False)
    for key, filename in (
        ("overlap_rows", "selection_overlap.csv"),
        ("rank_rows", "rank_displacement.csv"),
        ("ic_rows", "score_future_ic.csv"),
        ("spread_rows", "top_bottom_score_spread.csv"),
        ("changed_rows", "changed_names_pnl.csv"),
    ):
        selection[key].to_csv(diagnostic_dir / filename, index=False)


def campaign_evaluate_stage(
    output_dir: Path,
    manifest: dict[str, Any],
    stage: str,
    *,
    validation: bool = False,
    control_dir: Path | None = None,
    control_artifact: dict[str, Any] | None = None,
    trail_dir: Path | None = None,
    trail5_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    treatment_dir = campaign_resolve_artifact_dir(output_dir, "treatment")
    control_dir = control_dir or campaign_resolve_artifact_dir(output_dir, "control")
    trail_dir = trail_dir or campaign_resolve_artifact_dir(output_dir, "trail5")
    treatment = campaign_load_returns(treatment_dir, manifest["dates"], validation=validation)
    control = campaign_load_returns(
        control_dir,
        manifest["dates"],
        validation=validation,
        allow_superset=control_artifact is not None,
    )
    trail = campaign_load_returns(
        trail_dir,
        manifest["dates"],
        validation=validation,
        allow_superset=trail5_artifact is not None,
    )
    chronological = stage in {"full", "validation"}
    metrics = campaign_return_metrics(treatment["capital_c2c"], chronological=chronological)
    control_metrics = campaign_return_metrics(control["capital_c2c"], chronological=chronological)
    trail_metrics = campaign_return_metrics(trail["capital_c2c"], chronological=chronological)
    paired_control = campaign_moving_block_bootstrap(treatment["capital_c2c"], control["capital_c2c"], n_boot=10_000 if chronological else 5_000)
    paired_trail = campaign_moving_block_bootstrap(treatment["capital_c2c"], trail["capital_c2c"], n_boot=10_000 if chronological else 5_000, seed=1702)
    delta_control = treatment["capital_c2c"] - control["capital_c2c"]
    result: dict[str, Any] = {
        "status": "completed",
        "stage": stage,
        "metrics": {
            **metrics,
            "paired_annualized_lift_pct": paired_control["mean_annualized_delta_pct"],
            "paired_median_return_lift_pct": float(delta_control.median() * 100.0),
            "bootstrap_probability_lift_gt_zero": float(
                1.0 - paired_control["p_not_positive"]
            ),
            "paired_lift_vs_trail5_pct": paired_trail["mean_annualized_delta_pct"],
            "sharpe_improvement": float(metrics["sharpe"] - control_metrics["sharpe"]),
            "tail_improvement_pct": float(metrics["worst_decile_expected_shortfall_pct"] - control_metrics["worst_decile_expected_shortfall_pct"]),
            "paired_hit_rate_pct": float((delta_control > 0).mean() * 100.0),
            "long_paired_lift_pct": float((treatment["long_contribution"] - control["long_contribution"]).mean() * PERIODS_PER_YEAR * 100.0),
            "short_paired_lift_pct": float((treatment["short_contribution"] - control["short_contribution"]).mean() * PERIODS_PER_YEAR * 100.0),
            "chronological_block_stability_pct": float(np.mean([block.mean() > 0 for block in np.array_split(delta_control.to_numpy(), 4)]) * 100.0),
        },
        "controls": {"fresh_incumbent": control_metrics, "trail5": trail_metrics},
        "bootstrap_vs_incumbent": paired_control,
        "bootstrap_vs_trail5": paired_trail,
        "artifacts": {
            "treatment": str(treatment_dir),
            "control": str(control_dir),
            "control_reused": bool(control_artifact is not None),
            "control_artifact": control_artifact,
            "trail5": str(trail_dir),
            "trail5_reused": bool(trail5_artifact is not None),
            "trail5_artifact": trail5_artifact,
        },
    }
    if chronological:
        turnover_long = pd.to_numeric(treatment.get("turnover_long"), errors="coerce") if "turnover_long" in treatment else pd.Series(dtype=float)
        turnover_short = pd.to_numeric(treatment.get("turnover_short"), errors="coerce") if "turnover_short" in treatment else pd.Series(dtype=float)
        result["metrics"].update({
            "avg_turnover_long": float(turnover_long.mean()) if not turnover_long.empty else np.nan,
            "avg_turnover_short": float(turnover_short.mean()) if not turnover_short.empty else np.nan,
            "avg_turnover_total": float(0.5 * (turnover_long.mean() + turnover_short.mean())) if not turnover_long.empty and not turnover_short.empty else np.nan,
        })
    if stage == "full":
        _campaign_full_inference_diagnostics(
            result,
            treatment,
            control,
            treatment_dir,
            control_dir,
            output_dir,
        )
    if stage == "smoke":
        result["smoke"] = campaign_smoke_gate(treatment_dir / "signal_checks.csv", manifest["dates"], control_dir / "signal_checks.csv")
        result["gate"] = {"pass": bool(result["smoke"]["pass"]), "reason": "dispersion_and_schema"}
    elif stage == "mini":
        result["gate"] = {"pass": bool(paired_control["upper_90_pct"] > 0.0), "reason": "one_sided_90pct_upper_bound"}
    elif stage == "medium":
        result["gate"] = {"pass": bool(paired_control["upper_90_pct"] > 0.0 and metrics["n_periods"] == manifest["date_count"]), "reason": "positive_potential_and_complete"}
    elif stage == "full":
        cagr_or_sharpe = metrics["cagr_pct"] > control_metrics["cagr_pct"] or metrics["sharpe"] > control_metrics["sharpe"]
        beats_trail = metrics["cagr_pct"] > trail_metrics["cagr_pct"] and metrics["sharpe"] > trail_metrics["sharpe"]
        drawdown_ok = metrics["max_drawdown_pct"] >= control_metrics["max_drawdown_pct"] - 2.0
        passed = cagr_or_sharpe and beats_trail and paired_control["mean_annualized_delta_pct"] > 0 and drawdown_ok and metrics["n_periods"] == manifest["date_count"]
        label = "development_significant" if paired_control["lower_95_pct"] > 0 else ("positive_but_inconclusive" if paired_control["mean_annualized_delta_pct"] > 0 else "not_positive")
        result["gate"] = {"pass": bool(passed), "label": label, "cagr_or_sharpe_vs_incumbent": bool(cagr_or_sharpe), "beats_trail5_cagr_and_sharpe": bool(beats_trail), "drawdown_ok": bool(drawdown_ok)}
    return result


def campaign_rank_results(results: Sequence[dict[str, Any]], stage: str) -> list[str]:
    if not results:
        return []
    side_scoped = any(str(row.get("side_scope", "both")) in {"long", "short"} for row in results)
    if stage == "mini":
        # The 24-date stage is an early futility screen.  It deliberately uses
        # only coarse robustness indicators—not the magnitude of noisy lift or
        # CAGR estimates—to choose among survivors when the compute budget caps
        # the 80-date stage.
        def mini_key(row: dict[str, Any]) -> tuple[Any, ...]:
            metrics = row.get("metrics", {})
            reliability = row.get("reliability", {})

            def finite(key: str, default: float = -np.inf) -> float:
                try:
                    value = float(metrics.get(key, default))
                except (TypeError, ValueError):
                    return default
                return value if np.isfinite(value) else default

            return (
                not bool(reliability.get("complete_coverage", True)),
                int(reliability.get("schema_retries", 0)),
                int(reliability.get("transient_retries", 0)),
                -sum(
                    finite(key) >= 0.0
                    for key in ("long_paired_lift_pct", "short_paired_lift_pct")
                ),
                -(finite("chronological_block_stability_pct", 0.0) >= 50.0),
                -(finite("tail_improvement_pct", 0.0) >= 0.0),
                -(finite("paired_annualized_lift_pct", 0.0) >= 0.0),
                str(row.get("candidate_id")),
            )

        return [
            str(row["candidate_id"])
            for row in sorted(results, key=mini_key)
        ]

    keys = (
        [
            ("target_side_annualized_lift_pct", 3),
            ("target_side_sharpe_improvement", 3),
            ("tail_improvement_pct", 1),
            ("chronological_block_stability_pct", 1),
        ]
        if side_scoped
        else [
            ("paired_annualized_lift_pct", 2), ("sharpe_improvement", 2),
            ("tail_improvement_pct", 1), ("long_paired_lift_pct", 1), ("short_paired_lift_pct", 1),
        ]
    )
    if stage == "medium":
        keys.extend([("paired_hit_rate_pct", 1), ("chronological_block_stability_pct", 1)])
    scores = {str(row["candidate_id"]): 0.0 for row in results}

    def finite_metric(row: dict[str, Any], key: str) -> float:
        try:
            value = float(row.get("metrics", {}).get(key, -np.inf))
        except (TypeError, ValueError):
            return -np.inf
        return value if np.isfinite(value) else -np.inf

    for key, weight in keys:
        values = sorted({finite_metric(row, key) for row in results}, reverse=True)
        metric_rank = {value: rank for rank, value in enumerate(values, start=1)}
        for row in results:
            scores[str(row["candidate_id"])] += metric_rank[finite_metric(row, key)] * weight
    reliability_keys = {
        str(row["candidate_id"]): (
            not bool(row.get("reliability", {}).get("complete_coverage", True)),
            int(row.get("reliability", {}).get("schema_retries", 0)),
            int(row.get("reliability", {}).get("transient_retries", 0)),
        )
        for row in results
    }
    reliability_values = sorted(set(reliability_keys.values()))
    reliability_rank = {
        value: rank for rank, value in enumerate(reliability_values, start=1)
    }
    for candidate_id, value in reliability_keys.items():
        scores[candidate_id] += reliability_rank[value]
    return sorted(scores, key=lambda cid: (scores[cid], cid))


def campaign_sanitized_feedback(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    lift = float(metrics.get("paired_annualized_lift_pct", 0.0))
    result_class = "promising" if lift > 0 and result.get("gate", {}).get("pass") else ("inconclusive" if lift > 0 else "clearly_negative")
    labels: list[str] = []
    if float(metrics.get("long_paired_lift_pct", 0.0)) > 0:
        labels.append("long_side_improvement")
    if float(metrics.get("short_paired_lift_pct", 0.0)) < 0:
        labels.append("short_side_degradation")
    if float(metrics.get("tail_improvement_pct", 0.0)) < 0:
        labels.append("tail_risk_degradation")
    if result.get("smoke", {}).get("low_dispersion_warning"):
        labels.append("score_compression")
    return {
        "candidate": result.get("candidate_id"),
        "parent": result.get("parent_id"),
        "stage": result.get("stage"),
        "result_class": result_class,
        "qualitative_labels": labels,
        "permitted_action": "one_atomic_mutation" if result.get("gate", {}).get("pass") else "none",
    }


def campaign_initial_state(
    campaign_id: str,
    runner: Path,
    side_scope: str = "both",
    side_parent_cache: Path | None = None,
) -> dict[str, Any]:
    side_scope = str(side_scope).strip().lower()
    if side_scope not in {"both", "long", "short"}:
        raise CampaignError("side_scope must be both, long, or short")
    if side_scope == "both":
        prompt, prompt_hash = campaign_incumbent_prompt()
        incumbent_id = "v2-minimal-change-control"
    else:
        import mean_reversion_llm_v2 as runner_module

        prompt = runner_module.dedicated_side_system_prompt(side_scope)
        prompt_hash = campaign_sha256(prompt)
        incumbent_id = f"dedicated-{side_scope}-parent"
    now = campaign_utc_now()
    parent_cache_value = str(Path(side_parent_cache).resolve()) if side_parent_cache is not None else None
    frozen_harness = campaign_harness(side_scope, parent_cache_value)
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "side_scope": side_scope,
        "side_parent_cache": parent_cache_value,
        "created_at": now,
        "updated_at": now,
        "status": "initialized",
        "phase": "proposal",
        "runner": str(runner.resolve()),
        "supervisor_pid": None,
        "runner_pid": None,
        "stop_requested": False,
        "current_candidate": None,
        "current_stage": None,
        "completed_dates": 0,
        "planned_dates": 0,
        "consecutive_provider_failures": 0,
        "incumbent": {"candidate_id": incumbent_id, "prompt_text": prompt, "prompt_hash": prompt_hash},
        "harness": frozen_harness,
        "harness_hash": campaign_sha256(campaign_json_bytes(frozen_harness)),
        "code_fingerprint": campaign_code_fingerprint(runner),
        "supervisor_code_fingerprint": campaign_file_hash(Path(__file__)),
        "development": {"start": str(CAMPAIGN_DEVELOPMENT_START.date()), "end": str(CAMPAIGN_DEVELOPMENT_END.date()), "manifest_sizes": CAMPAIGN_STAGE_SIZES},
        "validation": {"start": str(CAMPAIGN_VALIDATION_START.date()), "end": str(CAMPAIGN_VALIDATION_END.date()), "approved": False},
        "budgets": {
            "development": {"max_candidates": CAMPAIGN_MAX_CANDIDATES, "max_full_runs": (1 if side_scope != "both" else CAMPAIGN_MAX_FULL_RUNS), "max_flash_calls": (1_800 if side_scope != "both" else CAMPAIGN_DEV_MAX_FLASH_CALLS), "max_flash_output_tokens": CAMPAIGN_DEV_MAX_FLASH_OUTPUT_TOKENS, "max_pro_calls": CAMPAIGN_MAX_PRO_CALLS},
            "validation": {"max_flash_calls": CAMPAIGN_VAL_MAX_FLASH_CALLS, "max_flash_output_tokens": CAMPAIGN_VAL_MAX_FLASH_OUTPUT_TOKENS},
        },
        "usage": {
            "development": {"candidates": 0, "full_runs": 0, "flash_calls": 0, "flash_output_tokens": 0, "pro_calls": 0, "pro_output_tokens": 0},
            "validation": {"flash_calls": 0, "flash_output_tokens": 0},
        },
        "candidates": {},
        "g1_generated": False,
        "winner_id": None,
        "terminal_reason": None,
        "error": None,
    }


def campaign_initialize(
    campaign_id: str,
    *,
    root: Path = CAMPAIGN_ROOT,
    runner: Path = CAMPAIGN_RUNNER,
    date_source: Path = CAMPAIGN_DATE_SOURCE,
    side_scope: str = "both",
    side_parent_cache: Path | None = None,
) -> Path:
    campaign_dir = campaign_path(root, campaign_id)
    if campaign_dir.exists() and any(campaign_dir.iterdir()):
        raise CampaignError(f"Campaign already exists: {campaign_dir}")
    if not runner.exists():
        raise CampaignError(f"Missing V2 runner: {runner}")
    if runner.resolve() != CAMPAIGN_RUNNER.resolve():
        raise CampaignError("Campaign experiments must use repo-native mean_reversion_llm_v2.py")
    if str(side_scope) != "both":
        if side_parent_cache is None or not Path(side_parent_cache).is_file():
            raise CampaignError("Side campaigns require --side-parent-cache from the qualifying fixed-prompt run")
        import mean_reversion_llm_v2_side_experiments as side_experiments

        parent_path = Path(side_parent_cache).resolve()
        side_experiments.load_side_cache(parent_path, str(side_scope))
        fixed_result_path = parent_path.parent.parent / "fixed_prompt_result.json"
        if not fixed_result_path.is_file():
            raise CampaignError("Side parent cache must remain beside its fixed-prompt qualification report")
        fixed_result = campaign_read_json(fixed_result_path)
        provenance = fixed_result.get("cache_provenance", {}).get(str(side_scope), {})
        if (
            not fixed_result.get(str(side_scope), {}).get("qualifies_for_autoresearch")
            or str(provenance.get("path", "")) != str(parent_path)
            or str(provenance.get("sha256", "")) != campaign_file_hash(parent_path)
        ):
            raise CampaignError("Dedicated side did not pass the frozen fixed-prompt qualification gate")
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state = campaign_initial_state(
        campaign_id,
        runner,
        side_scope=side_scope,
        side_parent_cache=side_parent_cache,
    )
    manifests = campaign_build_nested_manifests(campaign_dir, date_source)
    state["development"]["manifest_hashes"] = {stage: value["manifest_hash"] for stage, value in manifests.items()}
    state["development"]["manifest_file_hashes"] = {
        stage: campaign_file_hash(campaign_dir / f"manifests/{stage}.json")
        for stage in manifests
    }
    catalog = campaign_catalog()
    campaign_atomic_json(campaign_dir / "catalog.json", catalog)
    campaign_save(campaign_dir, state, "campaign_initialized", payload={"runner": str(runner), "catalog_entries": catalog["summary"]["entries"]})
    return campaign_dir


def campaign_migrate_cached_candidates(
    source_campaign_id: str,
    target_campaign_id: str,
    *,
    root: Path = CAMPAIGN_ROOT,
    reuse_completed_stages: bool = False,
) -> Path:
    """Start a newly fingerprinted campaign from immutable specs and paid score caches.

    Completed development stages may also be migrated after their immutable inputs and
    coverage are verified.  Their runtime fingerprints are rebound to the new campaign;
    failed or partial stages are never reused.
    """
    source_dir = campaign_path(root, source_campaign_id)
    target_dir = campaign_path(root, target_campaign_id)
    source_state = campaign_load(source_dir)
    if campaign_pid_is_alive(source_state.get("supervisor_pid")) or campaign_pid_is_alive(
        source_state.get("runner_pid")
    ):
        raise CampaignError("Cannot migrate a campaign while its supervisor or runner is alive")
    source_candidates = source_state.get("candidates", {})
    if not isinstance(source_candidates, dict) or not source_candidates:
        raise CampaignError("Source campaign has no cached candidates to migrate")

    campaign_initialize(target_campaign_id, root=root)
    target_state = campaign_load(target_dir)
    for stage in CAMPAIGN_STAGE_SIZES:
        source_hash = source_state.get("development", {}).get("manifest_hashes", {}).get(stage)
        target_hash = target_state.get("development", {}).get("manifest_hashes", {}).get(stage)
        if source_hash != target_hash:
            raise CampaignFirewallError(
                f"Cannot migrate across different frozen {stage} date manifests"
            )

    import mean_reversion_llm_v2 as runner_module

    migrated_candidates: dict[str, dict[str, Any]] = {}
    copied_caches: list[dict[str, Any]] = []
    copied_stages: list[dict[str, Any]] = []
    for candidate_id, source_candidate in sorted(source_candidates.items()):
        if not isinstance(source_candidate, dict):
            raise CampaignFirewallError(f"Invalid source candidate state: {candidate_id}")
        source_spec = source_dir / f"candidates/{candidate_id}/prompt_spec.json"
        if not source_spec.is_file() or campaign_file_hash(source_spec) != source_candidate.get(
            "prompt_spec_hash"
        ):
            raise CampaignFirewallError(f"Source prompt spec failed its frozen hash: {candidate_id}")
        target_spec = target_dir / f"candidates/{candidate_id}/prompt_spec.json"
        target_spec.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_spec, target_spec)
        loaded_spec = runner_module.load_experimental_prompt_spec(target_spec)
        if loaded_spec.prompt_sha256 != str(source_candidate.get("prompt_hash")):
            raise CampaignFirewallError(
                f"Current runner rejected migrated prompt identity: {candidate_id}"
            )
        target_spec_hash = campaign_file_hash(target_spec)
        migrated_candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "generation": source_candidate.get("generation"),
            "parent_id": source_candidate.get("parent_id"),
            "prompt_hash": source_candidate.get("prompt_hash"),
            "anonymize_tickers": loaded_spec.anonymize_tickers,
            "prompt_spec_hash": target_spec_hash,
            "status": "static_pass",
            "migrated_from": source_campaign_id,
        }
        campaign_static_result(target_dir, campaign_read_json(target_spec))

        source_cache_dir = source_dir / f"candidates/{candidate_id}/cache"
        target_cache_dir = target_dir / f"candidates/{candidate_id}/cache"
        for source_cache in sorted(source_cache_dir.glob("*.json")):
            payload = campaign_read_json(source_cache)
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict) or metadata.get(
                "experimental_prompt_spec_sha256"
            ) != loaded_spec.spec_sha256:
                raise CampaignFirewallError(
                    f"Cache does not match migrated prompt spec: {source_cache}"
                )
            cached_dates = campaign_cached_signal_dates([source_cache])
            full_dates = set(
                campaign_read_manifest(target_dir / "manifests/full.json").get("dates", [])
            )
            if not cached_dates.issubset(full_dates):
                raise CampaignFirewallError(
                    f"Cache contains dates outside the target development manifest: {source_cache}"
                )
            target_cache_dir.mkdir(parents=True, exist_ok=True)
            target_cache = target_cache_dir / source_cache.name
            shutil.copy2(source_cache, target_cache)
            copied_caches.append(
                {
                    "candidate_id": candidate_id,
                    "source": str(source_cache),
                    "target": str(target_cache),
                    "sha256": campaign_file_hash(target_cache),
                    "signals": len(payload.get("signals", {})),
                    "provider_attempts": len(payload.get("llm_call_usage", [])),
                }
            )

        if reuse_completed_stages:
            for stage in CAMPAIGN_STAGE_SIZES:
                source_stage_dir = source_dir / f"candidates/{candidate_id}/{stage}"
                source_result_path = source_stage_dir / "result.json"
                if not source_result_path.is_file():
                    continue
                source_result = campaign_read_json(source_result_path)
                if source_result.get("status") != "completed":
                    continue
                frozen = source_result.get("frozen_inputs")
                if not isinstance(frozen, dict):
                    raise CampaignFirewallError(
                        f"Completed source stage lacks frozen inputs: {candidate_id}/{stage}"
                    )
                manifest = campaign_read_manifest(target_dir / f"manifests/{stage}.json")
                expected_control = campaign_control_artifact(
                    anonymize_tickers=loaded_spec.anonymize_tickers,
                    validation=False,
                )
                expected_trail5 = campaign_trail5_artifact(validation=False)
                immutable_expected = {
                    "candidate_id": candidate_id,
                    "parent_id": source_candidate.get("parent_id"),
                    "prompt_hash": source_candidate.get("prompt_hash"),
                    "prompt_spec_hash": target_spec_hash,
                    "manifest_hash": manifest["manifest_hash"],
                    "anonymize_tickers": loaded_spec.anonymize_tickers,
                    "control_artifact": expected_control,
                    "trail5_artifact": expected_trail5,
                }
                for key, expected in immutable_expected.items():
                    if frozen.get(key) != expected:
                        raise CampaignFirewallError(
                            f"Completed source stage changed immutable {key}: "
                            f"{candidate_id}/{stage}"
                        )
                if int(source_result.get("metrics", {}).get("n_periods", -1)) != int(
                    manifest["date_count"]
                ):
                    raise CampaignFirewallError(
                        f"Completed source stage lacks full manifest coverage: "
                        f"{candidate_id}/{stage}"
                    )
                target_stage_dir = target_dir / f"candidates/{candidate_id}/{stage}"
                shutil.copytree(source_stage_dir, target_stage_dir)
                migrated_result = campaign_read_json(target_stage_dir / "result.json")
                migrated_result["frozen_inputs"].update(
                    {
                        "code_fingerprint": target_state["code_fingerprint"],
                        "supervisor_code_fingerprint": target_state[
                            "supervisor_code_fingerprint"
                        ],
                        "harness_hash": target_state["harness_hash"],
                    }
                )
                campaign_atomic_json(target_stage_dir / "result.json", migrated_result)
                migrated_candidates[candidate_id][stage] = bool(
                    migrated_result.get("gate", {}).get("pass", False)
                )
                copied_stages.append(
                    {
                        "candidate_id": candidate_id,
                        "stage": stage,
                        "passed": migrated_candidates[candidate_id][stage],
                        "source_result_sha256": campaign_file_hash(source_result_path),
                        "target_result_sha256": campaign_file_hash(
                            target_stage_dir / "result.json"
                        ),
                    }
                )

    source_usage = source_state.get("usage", {}).get("development", {})
    target_state["candidates"] = migrated_candidates
    target_state["usage"]["development"].update(
        {
            "candidates": len(migrated_candidates),
            "pro_calls": int(source_usage.get("pro_calls", 0)),
            "pro_output_tokens": int(source_usage.get("pro_output_tokens", 0)),
            "flash_calls": 0,
            "flash_output_tokens": 0,
            "full_runs": 0,
        }
    )
    target_state.update(
        {
            "status": "initialized",
            "phase": "migration_ready",
            "current_candidate": None,
            "current_stage": None,
            "completed_dates": 0,
            "planned_dates": 0,
            "g1_generated": False,
            "winner_id": None,
            "terminal_reason": None,
            "error": None,
            "migration": {
                "source_campaign_id": source_campaign_id,
                "source_state_sha256": campaign_file_hash(source_dir / "campaign.json"),
                "source_code_fingerprint": source_state.get("code_fingerprint"),
                "target_code_fingerprint": target_state.get("code_fingerprint"),
                "candidate_ids": sorted(migrated_candidates),
                "copied_cache_count": len(copied_caches),
                "copied_stage_count": len(copied_stages),
            },
        }
    )
    campaign_atomic_json(
        target_dir / "migration.json",
        {
            **target_state["migration"],
            "created_at": campaign_utc_now(),
            "copied_caches": copied_caches,
            "copied_stages": copied_stages,
            "old_stage_results_reused": bool(copied_stages),
            "policy": (
                "reuse immutable prompt specs, paid score caches, and verified completed "
                "development stages; recompute failed, partial, or missing stages"
                if reuse_completed_stages
                else "reuse immutable prompt specs and paid score caches; recompute all stage artifacts"
            ),
        },
    )
    campaign_save(
        target_dir,
        target_state,
        "campaign_migrated",
        payload={
            "source_campaign_id": source_campaign_id,
            "candidate_ids": sorted(migrated_candidates),
            "copied_cache_count": len(copied_caches),
            "copied_stage_count": len(copied_stages),
        },
    )
    return target_dir


def campaign_static_result(campaign_dir: Path, spec: dict[str, Any]) -> None:
    result_dir = campaign_dir / "candidates" / spec["candidate_id"] / "static"
    result = {
        "candidate_id": spec["candidate_id"],
        "parent_id": spec["parent_id"],
        "stage": "static",
        "status": "completed",
        "gate": {"pass": True, "reason": "lineage_patch_contract_harness"},
        "prompt_hash": spec["prompt_hash"],
        "anonymize_tickers": bool(spec.get("anonymize_tickers", True)),
    }
    campaign_atomic_json(result_dir / "result.json", result)


def campaign_store_proposals(
    campaign_dir: Path,
    state: dict[str, Any],
    proposals: Sequence[dict[str, Any]],
    *,
    generation: str,
    parents: dict[str, tuple[str, str]],
    limit: int,
) -> list[str]:
    stored: list[str] = []
    prefix = "G0" if generation == "G0" else "G1"
    catalog = campaign_read_json(campaign_dir / "catalog.json")
    catalog_rows = list(catalog.get("entries", []))
    catalog_identities = {
        (str(row.get("prompt_hash")), bool(row.get("anonymize_tickers", True)))
        for row in catalog_rows
        if row.get("prompt_hash") and not row.get("duplicate")
    }
    known_references = {"v2-minimal-change-control", state["incumbent"]["candidate_id"], *parents.keys()}
    for row in catalog_rows:
        for value in (
            row.get("prompt_hash"),
            row.get("experiment_id"),
        ):
            if value:
                known_references.add(str(value))

    def config_matches_catalog(prompt_hash: str, anonymize_tickers: bool) -> bool:
        return any(
            visibility == anonymize_tickers
            and (prompt_hash.startswith(known_hash) or known_hash.startswith(prompt_hash))
            for known_hash, visibility in catalog_identities
        )

    def reject(
        proposal: Any,
        *,
        candidate_id: str,
        parent_id: str,
        reason: str,
    ) -> None:
        campaign_record_rejected_proposal(
            campaign_dir,
            proposal,
            candidate_id=candidate_id,
            generation=generation,
            parent_id=parent_id,
            reason=reason,
        )
        campaign_append_event(
            campaign_dir,
            "proposal_rejected",
            candidate_id=candidate_id,
            payload={"reason": reason},
        )

    for proposal in proposals:
        if len(stored) >= limit or len(state["candidates"]) >= CAMPAIGN_MAX_CANDIDATES:
            break
        candidate_id = f"{prefix}_{len([key for key in state['candidates'] if key.startswith(prefix + '_')]) + 1:02d}"
        parent_id = state["incumbent"]["candidate_id"] if generation == "G0" else str(proposal.get("parent_id", ""))
        if parent_id not in parents:
            reject(
                proposal,
                candidate_id=candidate_id,
                parent_id=parent_id,
                reason="unknown_parent_id",
            )
            continue
        parent_prompt, parent_hash = parents[parent_id]
        try:
            spec = campaign_materialize_prompt_spec(
                dict(proposal), candidate_id=candidate_id, generation=generation, parent_id=parent_id,
                parent_prompt=parent_prompt, parent_hash=parent_hash,
                side_scope=str(state.get("side_scope", "both")),
            )
        except CampaignError as exc:
            reject(
                proposal,
                candidate_id=candidate_id,
                parent_id=parent_id,
                reason=str(exc),
            )
            continue
        unknown_references = sorted(set(spec["prior_experiment_ids"]) - known_references)
        if unknown_references:
            reject(
                proposal,
                candidate_id=candidate_id,
                parent_id=parent_id,
                reason="unknown_prior_experiment_ids",
            )
            continue
        if (
            config_matches_catalog(spec["prompt_hash"], spec["anonymize_tickers"])
            or any(
                existing.get("prompt_hash") == spec["prompt_hash"]
                and bool(existing.get("anonymize_tickers", True)) == spec["anonymize_tickers"]
                for existing in state["candidates"].values()
            )
        ):
            reject(
                proposal,
                candidate_id=candidate_id,
                parent_id=parent_id,
                reason="duplicate_prompt_or_config_hash",
            )
            continue
        candidate_dir = campaign_dir / "candidates" / candidate_id
        spec_path = candidate_dir / "prompt_spec.json"
        campaign_atomic_json(spec_path, spec)
        campaign_static_result(campaign_dir, spec)
        state["candidates"][candidate_id] = {
            "candidate_id": candidate_id,
            "generation": generation,
            "parent_id": parent_id,
            "prompt_hash": spec["prompt_hash"],
            "anonymize_tickers": spec["anonymize_tickers"],
            "prompt_spec_hash": campaign_file_hash(spec_path),
            "status": "static_pass",
        }
        state["usage"]["development"]["candidates"] += 1
        stored.append(candidate_id)
    campaign_save(campaign_dir, state, "proposals_stored", payload={"generation": generation, "candidate_ids": stored})
    return stored


def campaign_researcher_prompt_g0(state: dict[str, Any], catalog: dict[str, Any]) -> str:
    history = [
        {
            key: row.get(key)
            for key in (
                "prompt_variant",
                "prompt_hash",
                "experiment_id",
                "harness_compatible",
                "experiment_design",
                "qualitative_lesson",
                "anonymize_tickers",
            )
        }
        for row in catalog.get("entries", [])
        if row.get("complete") and not row.get("duplicate")
    ]
    side_scope = str(state.get("side_scope", "both"))
    output_instruction = (
        "the long_scores and short_scores JSON arrays"
        if side_scope == "both"
        else f"one scores JSON array for the {side_scope.upper()} pool only"
    )
    side_instruction = (
        ""
        if side_scope == "both"
        else (
            f"This is an independent {side_scope.upper()}-side campaign. Mutate only instructions for that side, "
            "never introduce the opposite-side objective, and optimize only its five-day side-relative ranking. "
        )
    )
    return (
        "Propose exactly six distinct G0 candidates directly from the parent below. Diagnose a behavioral weakness "
        "that may limit cross-sectional ranking, then propose a coherent replacement for any editable reasoning, TASK, "
        "RULES, scoring, or output-instruction text. The FEATURE GLOSSARY block must remain byte-for-byte identical. "
        "Do not reinterpret glossary definitions, invent derived features, or claim unsupplied information beyond "
        "ticker familiarity when anonymize_tickers is false. Never hard-code a named security or dated fact. A proposal "
        "may be a one-line change, generic wording, a focused rewrite, or a full rewrite of all non-glossary text. "
        "Each candidate must contain mutation_mode ('additive', 'substitutive', or 'rewrite'); one exact "
        "{operation:'replace_once',old_text:<exact parent substring>,new_text:<replacement>} patch; hypothesis; mechanism; "
        "expected_failure; prior_experiment_ids; numeric score_min and score_max; score_direction equal to "
        "'higher_is_better' or 'lower_is_better'; and anonymize_tickers as a JSON boolean. Set anonymize_tickers false "
        "when the thesis benefits from real ticker identifiers, or true when it benefits from opaque identifiers. "
        f"{side_instruction}Ticker visibility is a genuine candidate-level research choice. The prompt must request one numeric score per security in the declared "
        f"range using {output_instruction}. References must use a SANITIZED_HISTORY experiment_id, "
        f"its prompt_hash, v2-minimal-change-control, or {state['incumbent'].get('candidate_id', 'v2-minimal-change-control')}. Make the six research theses meaningfully distinct. "
        "Return {candidates:[...]}.\n"
        f"FROZEN_PARENT:\n{state['incumbent']['prompt_text']}\n"
        f"SANITIZED_HISTORY:{json.dumps(history, sort_keys=True)}"
    )


def campaign_researcher_prompt_g1(
    state: dict[str, Any],
    survivors: Sequence[str],
    feedback: Sequence[dict[str, Any]],
) -> str:
    parents = []
    for candidate_id in survivors:
        spec = campaign_read_json(Path(state["campaign_dir"]) / f"candidates/{candidate_id}/prompt_spec.json")
        parents.append({
            "parent_id": candidate_id,
            "prompt_text": spec["prompt_text"],
            "prompt_hash": spec["prompt_hash"],
            "anonymize_tickers": bool(spec.get("anonymize_tickers", True)),
        })
    side_scope = str(state.get("side_scope", "both"))
    side_instruction = (
        ""
        if side_scope == "both"
        else f"Keep every child strictly {side_scope.upper()}-only with one scores array. "
    )
    return (
        "Propose at most two G1 children total. Each must contain parent_id, mutation_mode, one replace_once patch, "
        "hypothesis, mechanism, expected_failure, score_min, score_max, score_direction, anonymize_tickers as a JSON boolean, and prior_experiment_ids referencing its supplied parent or "
        f"v2-minimal-change-control or {state['incumbent']['candidate_id']}. {side_instruction}Use one additive, substitutive, rewrite, or format_repair mutation. "
        "The FEATURE GLOSSARY must remain byte-for-byte identical. Non-glossary instructions may be rewritten freely. "
        "Ticker visibility is candidate-selected: false supplies real ticker identifiers and true supplies opaque identifiers. "
        "A format_repair must append exactly one of "
        f"{json.dumps(list(CAMPAIGN_FORMAT_REPAIR_APPENDICES))} to old_text, with no other wording. "
        "Do not create children of G1. Return {candidates:[...]}. Exact outcomes are intentionally withheld.\n"
        f"PARENTS:{json.dumps(parents, sort_keys=True)}\nSANITIZED_FEEDBACK:{json.dumps(list(feedback), sort_keys=True)}"
    )


def campaign_recent_rejection_feedback(
    campaign_dir: Path,
    *,
    generation: str,
    limit: int = 12,
) -> list[dict[str, str]]:
    """Return structural-only feedback from sanitized rejected-proposal records."""
    path = campaign_dir / "rejected_proposals.jsonl"
    if not path.exists() or path.is_symlink():
        return []
    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or record.get("generation") != generation:
                    continue
                proposal = record.get("proposal") if isinstance(record.get("proposal"), dict) else {}
                patch = proposal.get("patch") if isinstance(proposal.get("patch"), dict) else {}
                rows.append(
                    {
                        "reason": str(record.get("reason", ""))[:1_000],
                        "returned_mutation_mode": str(proposal.get("mutation_mode", ""))[:100],
                        "returned_patch_operation": str(patch.get("operation", ""))[:100],
                    }
                )
    except (OSError, UnicodeError):
        return []
    return rows[-max(1, int(limit)) :]


def campaign_use_researcher(
    campaign_dir: Path,
    state: dict[str, Any],
    prompt: str,
    researcher: Callable[[str], tuple[list[dict[str, Any]], dict[str, int]]],
) -> list[dict[str, Any]]:
    usage = state["usage"]["development"]
    maximum = int(state["budgets"]["development"]["max_pro_calls"])
    while int(usage["pro_calls"]) < maximum:
        if (campaign_dir / "STOP").exists():
            raise CampaignStop("Stop requested before Pro researcher call")
        usage["pro_calls"] += 1
        campaign_save(campaign_dir, state, "researcher_started")
        try:
            proposals, delta = researcher(prompt)
        except Exception as exc:
            state["consecutive_provider_failures"] += 1
            campaign_save(
                campaign_dir,
                state,
                "researcher_failed",
                payload={"error": f"{type(exc).__name__}: {exc}"},
            )
            if state["consecutive_provider_failures"] >= 3:
                raise CampaignStop("Three consecutive provider failures") from exc
            continue
        state["consecutive_provider_failures"] = 0
        usage["pro_output_tokens"] += int(delta.get("output_tokens", 0))
        campaign_save(campaign_dir, state, "researcher_completed", payload={"proposal_count": len(proposals)})
        return proposals
    raise CampaignStop("Pro researcher call budget exhausted")


def campaign_stage_result_path(campaign_dir: Path, candidate_id: str, stage: str) -> Path:
    return campaign_dir / "candidates" / candidate_id / stage / "result.json"


def campaign_verify_frozen_runtime(state: dict[str, Any]) -> None:
    runner = Path(state.get("runner", ""))
    if runner.resolve() != CAMPAIGN_RUNNER.resolve():
        raise CampaignFirewallError("Campaign runner no longer resolves to mean_reversion_llm_v2.py")
    if state.get("code_fingerprint") != campaign_code_fingerprint(runner):
        raise CampaignFirewallError("Runner/dependency fingerprint changed after campaign initialization")
    if state.get("supervisor_code_fingerprint") != campaign_file_hash(Path(__file__)):
        raise CampaignFirewallError(
            "Supervisor fingerprint changed after campaign initialization; initialize a new "
            "campaign ID after code changes instead of resuming this frozen campaign"
        )
    expected_harness = campaign_harness(
        str(state.get("side_scope", "both")),
        state.get("side_parent_cache"),
    )
    expected_harness_hash = campaign_sha256(campaign_json_bytes(expected_harness))
    if state.get("harness") != expected_harness or state.get("harness_hash") != expected_harness_hash:
        raise CampaignFirewallError("Frozen campaign harness changed")


def campaign_verify_prompt_spec(
    campaign_dir: Path,
    state: dict[str, Any],
    candidate_id: str,
) -> tuple[dict[str, Any], Path]:
    import mean_reversion_llm_v2 as runner_module

    candidate = state.get("candidates", {}).get(candidate_id)
    if not isinstance(candidate, dict):
        raise CampaignFirewallError(f"Unknown candidate {candidate_id}")
    path = campaign_dir / "candidates" / candidate_id / "prompt_spec.json"
    if not path.exists() or campaign_file_hash(path) != candidate.get("prompt_spec_hash"):
        raise CampaignFirewallError(f"Prompt spec changed after static validation: {candidate_id}")
    spec = campaign_read_json(path)
    for key in ("candidate_id", "generation", "parent_id", "prompt_hash"):
        if str(spec.get(key)) != str(candidate.get(key)):
            raise CampaignFirewallError(f"Prompt spec {key} differs from frozen candidate state")
    if bool(spec.get("anonymize_tickers", True)) != bool(
        candidate.get("anonymize_tickers", True)
    ):
        raise CampaignFirewallError(
            "Prompt spec anonymize_tickers differs from frozen candidate state"
        )
    independently_loaded = runner_module.load_experimental_prompt_spec(path)
    if independently_loaded.prompt_sha256 != str(candidate.get("prompt_hash")):
        raise CampaignFirewallError("Runner and supervisor disagree on the materialized prompt hash")
    if independently_loaded.anonymize_tickers != bool(
        candidate.get("anonymize_tickers", True)
    ):
        raise CampaignFirewallError("Runner and supervisor disagree on ticker visibility")
    if candidate.get("generation") == "G1":
        parent_id = str(candidate.get("parent_id"))
        parent = state.get("candidates", {}).get(parent_id)
        if not isinstance(parent, dict) or str(spec.get("parent_prompt_hash")) != str(parent.get("prompt_hash")):
            raise CampaignFirewallError("G1 parent lineage differs from frozen campaign state")
    elif candidate.get("parent_id") != state.get("incumbent", {}).get("candidate_id"):
        raise CampaignFirewallError("G0 candidate is not descended from the frozen incumbent")
    return spec, path


def campaign_candidate_lineage(state: dict[str, Any], candidate_id: str) -> list[dict[str, Any]]:
    lineage: list[dict[str, Any]] = []
    current = candidate_id
    seen: set[str] = set()
    while current != state.get("incumbent", {}).get("candidate_id"):
        if current in seen:
            raise CampaignFirewallError("Candidate lineage contains a cycle")
        seen.add(current)
        candidate = state.get("candidates", {}).get(current)
        if not isinstance(candidate, dict):
            raise CampaignFirewallError(f"Candidate lineage references unknown parent {current}")
        lineage.append(
            {
                key: candidate.get(key)
                for key in (
                    "candidate_id",
                    "generation",
                    "parent_id",
                    "prompt_hash",
                    "prompt_spec_hash",
                    "anonymize_tickers",
                )
            }
        )
        current = str(candidate.get("parent_id"))
    lineage.append(
        {
            "candidate_id": state["incumbent"]["candidate_id"],
            "generation": "parent",
            "parent_id": None,
            "prompt_hash": state["incumbent"]["prompt_hash"],
            "prompt_spec_hash": None,
        }
    )
    return lineage


def campaign_cache_paths(campaign_dir: Path, usage_key: str) -> list[Path]:
    if usage_key == "validation":
        return sorted(
            path
            for path in (campaign_dir / "validation/cache").rglob("*.json")
            if path.is_file()
        )
    return sorted(
        path for path in (campaign_dir / "candidates").glob("*/cache/*.json") if path.is_file()
    )


def campaign_reconcile_all_cache_usage(
    campaign_dir: Path, state: dict[str, Any], usage_key: str
) -> None:
    campaign_reconcile_cache_usage(state, usage_key, campaign_cache_paths(campaign_dir, usage_key))


def campaign_cached_signal_dates(paths: Sequence[Path]) -> set[str]:
    dates: set[str] = set()
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = campaign_read_json(path)
        except (OSError, json.JSONDecodeError, CampaignError):
            continue
        signals = payload.get("signals", {})
        if not isinstance(signals, dict):
            continue
        for key, value in signals.items():
            if isinstance(value, dict):
                dates.add(str(key).split("|", 1)[0])
    return dates


def campaign_execute_stage(
    campaign_dir: Path,
    state: dict[str, Any],
    candidate_id: str,
    stage: str,
    *,
    validation: bool = False,
    executor: Callable[[Path, dict[str, Any], list[str], Path], None] | None = None,
) -> dict[str, Any]:
    result_path = campaign_dir / "validation/result.json" if validation else campaign_stage_result_path(campaign_dir, candidate_id, stage)
    campaign_verify_frozen_runtime(state)
    spec, spec_path = campaign_verify_prompt_spec(campaign_dir, state, candidate_id)
    side_scope = str(state.get("side_scope", "both"))
    manifest_path = campaign_dir / "manifests" / ("validation.json" if validation else f"{stage}.json")
    manifest = campaign_read_manifest(
        manifest_path,
        validation=validation,
        expected_hash=campaign_expected_manifest_hash(state, stage, validation=validation),
        expected_file_hash=campaign_expected_manifest_file_hash(state, stage, validation=validation),
    )
    usage_key = "validation" if validation else "development"
    campaign_reconcile_all_cache_usage(campaign_dir, state, usage_key)
    if (campaign_dir / "STOP").exists():
        raise CampaignStop("Stop requested")
    usage = state["usage"][usage_key]
    budget = state["budgets"][usage_key]
    frozen_inputs = {
        "candidate_id": candidate_id,
        "parent_id": spec["parent_id"],
        "prompt_hash": spec["prompt_hash"],
        "prompt_spec_hash": state["candidates"][candidate_id]["prompt_spec_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "code_fingerprint": state["code_fingerprint"],
        "supervisor_code_fingerprint": state["supervisor_code_fingerprint"],
        "harness_hash": state["harness_hash"],
        "anonymize_tickers": bool(spec.get("anonymize_tickers", True)),
    }
    if side_scope != "both":
        frozen_inputs["side_scope"] = side_scope
    control_artifact: dict[str, Any] | None = None
    control_dir: Path | None = None
    if side_scope == "both":
        control_artifact = campaign_control_artifact(
            anonymize_tickers=bool(spec.get("anonymize_tickers", True)),
            validation=validation,
        )
        control_dir = Path(control_artifact["run_dir"])
        frozen_inputs["control_artifact"] = control_artifact
    else:
        if validation:
            parent_spec_path = campaign_dir / "validation/parent_prompt_spec.json"
            if not parent_spec_path.is_file():
                raise CampaignFirewallError("Frozen validation dedicated-parent spec is missing")
            frozen_inputs["side_parent_validation_spec"] = str(parent_spec_path.resolve())
            frozen_inputs["side_parent_validation_spec_hash"] = campaign_file_hash(parent_spec_path)
        else:
            parent_cache = Path(str(state.get("side_parent_cache", "")))
            if not parent_cache.is_file():
                raise CampaignFirewallError("Frozen dedicated side parent cache is missing")
            frozen_inputs["side_parent_cache"] = str(parent_cache.resolve())
            frozen_inputs["side_parent_cache_hash"] = campaign_file_hash(parent_cache)
    trail5_artifact = campaign_trail5_artifact(validation=validation)
    trail_dir = Path(trail5_artifact["run_dir"])
    frozen_inputs["trail5_artifact"] = trail5_artifact
    if result_path.exists():
        existing = campaign_read_json(result_path)
        if existing.get("frozen_inputs") != frozen_inputs:
            raise CampaignFirewallError(f"Existing {stage} result is not bound to the frozen inputs")
        if existing.get("status") == "completed":
            campaign_save(campaign_dir, state)
            return existing
        archive_dir = (
            campaign_dir / "validation/attempt_failures"
            if validation
            else result_path.parent / "attempt_failures"
        )
        archive_path = archive_dir / f"{campaign_utc_now().replace(':', '').replace('-', '')}.json"
        campaign_atomic_json(archive_path, existing)
        result_path.unlink()
    treatment_cache_dir = campaign_dir / ("validation/cache/treatment" if validation else f"candidates/{candidate_id}/cache")
    requested_dates = set(str(value) for value in manifest["dates"])
    treatment_cached = campaign_cached_signal_dates(sorted(treatment_cache_dir.glob("*.json")))
    minimum_calls = len(requested_dates - treatment_cached)
    if side_scope != "both" and (stage == "full" or validation):
        required = 3 * len(requested_dates)
        if validation:
            parent_cached = campaign_cached_signal_dates(
                sorted((campaign_dir / "validation/cache/parent").glob("*.json"))
            )
            required += len(requested_dates - parent_cached)
        minimum_calls = max(minimum_calls, required - len(treatment_cached))
    if usage["flash_calls"] + minimum_calls > budget["max_flash_calls"]:
        raise CampaignStop(f"{usage_key} Flash call budget would be exceeded")
    command, output_dir = campaign_runner_command(state, campaign_dir, spec_path, stage, validation=validation)
    state.update({"status": "running", "phase": "validation" if validation else stage, "current_candidate": candidate_id, "current_stage": stage, "planned_dates": manifest["date_count"], "completed_dates": 0})
    campaign_save(campaign_dir, state, "stage_started", candidate_id=candidate_id, stage=stage)
    progress_path = output_dir / "runner_events.jsonl"
    progress_usage: dict[str, Any] | None = None
    try:
        if executor:
            executor(campaign_dir, state, command, output_dir)
        else:
            if side_scope != "both" and validation:
                parent_spec_path = campaign_dir / "validation/parent_prompt_spec.json"
                parent_command, _ = campaign_runner_command(
                    state, campaign_dir, parent_spec_path, stage, validation=True
                )
                replacements = {
                    "--output-dir": str(campaign_dir / "validation/parent"),
                    "--cache-dir": str(campaign_dir / "validation/cache/parent"),
                    "--progress-file": str(campaign_dir / "validation/parent/runner_events.jsonl"),
                }
                for flag, value in replacements.items():
                    parent_command[parent_command.index(flag) + 1] = value
                campaign_run_subprocess(
                    campaign_dir, state, parent_command, campaign_dir / "validation/parent"
                )
            campaign_run_subprocess(campaign_dir, state, command, output_dir)
            if side_scope != "both" and (stage == "full" or validation):
                # Full-stage side finalists are confirmed with two additional immutable
                # provider runs; screening stages remain single-run.
                for replicate_id in ("r2", "r3"):
                    replica = dict(spec)
                    replica["replicate_id"] = replicate_id
                    replica_path = spec_path.parent / "replicates" / f"prompt_spec_{replicate_id}.json"
                    campaign_atomic_json(replica_path, replica)
                    # Validate the materialized prompt and cache identity before spending a call.
                    import mean_reversion_llm_v2 as runner_module

                    runner_module.load_experimental_prompt_spec(replica_path)
                    replica_command = list(command)
                    replica_command[replica_command.index("--experimental-prompt-spec") + 1] = str(replica_path)
                    campaign_run_subprocess(campaign_dir, state, replica_command, output_dir)
        progress_usage = campaign_progress_usage(progress_path)
        if validation:
            campaign_reconcile_all_cache_usage(campaign_dir, state, usage_key)
        else:
            campaign_reconcile_cache_usage(
                state,
                usage_key,
                sorted(treatment_cache_dir.glob("*.json")),
            )
        if campaign_progress_has_event(progress_path, "runner_stopped") and not campaign_progress_has_event(progress_path, "runner_completed"):
            stopped_event = campaign_latest_progress_event(progress_path, "runner_stopped") or {}
            stop_reason = str(stopped_event.get("reason", "V2 runner stopped safely"))
            campaign_save(campaign_dir, state, "runner_safe_stop_checkpointed", candidate_id=candidate_id, stage=stage, payload={"reason": stop_reason})
            if "budget" in stop_reason.lower():
                raise CampaignStop(stop_reason)
            raise CampaignError(stop_reason)
        if usage["flash_calls"] > budget["max_flash_calls"]:
            raise CampaignStop(f"{usage_key} Flash call budget exhausted")
        if usage["flash_output_tokens"] > budget["max_flash_output_tokens"]:
            raise CampaignStop(f"{usage_key} Flash output-token budget exhausted")
        if side_scope == "both":
            assert control_dir is not None and control_artifact is not None
            result = campaign_evaluate_stage(
                output_dir,
                manifest,
                stage,
                validation=validation,
                control_dir=control_dir,
                control_artifact=control_artifact,
                trail_dir=trail_dir,
                trail5_artifact=trail5_artifact,
            )
        else:
            import mean_reversion_llm_v2_side_experiments as side_experiments

            candidate_cache_paths = []
            for path in sorted(treatment_cache_dir.glob("*.json")):
                try:
                    metadata = campaign_read_json(path).get("metadata", {})
                except (CampaignError, json.JSONDecodeError, OSError):
                    continue
                if (
                    str(metadata.get("prompt_hash", "")) == str(spec["prompt_hash"])[:12]
                    and str(metadata.get("side_scope", "")) == side_scope
                ):
                    candidate_cache_paths.append(path)
            required_replicates = 3 if (stage == "full" or validation) else 1
            by_replicate = {
                str(campaign_read_json(path).get("metadata", {}).get("replicate_id")): path
                for path in candidate_cache_paths
            }
            required_ids = ("r1", "r2", "r3") if required_replicates == 3 else ("r1",)
            if any(replicate not in by_replicate for replicate in required_ids):
                raise CampaignError(f"Missing side finalist cache replicate(s): {required_ids}")
            if validation:
                parent_paths = sorted((campaign_dir / "validation/cache/parent").glob("*.json"))
                if len(parent_paths) != 1:
                    raise CampaignError("Validation requires exactly one dedicated-parent cache")
                evaluation_parent_cache = parent_paths[0]
            else:
                evaluation_parent_cache = Path(str(state["side_parent_cache"]))
            result = side_experiments.evaluate_campaign_side_stage(
                [by_replicate[replicate] for replicate in required_ids],
                evaluation_parent_cache,
                side_scope=side_scope,
                expected_dates=manifest["dates"],
                stage=stage,
                output_dir=output_dir,
                baseline_root=(
                    CAMPAIGN_VALIDATION_CONTROL.parents[2]
                    if validation
                    else side_experiments.DEFAULT_BASELINE_ROOT
                ),
            )
        result.update({
            "candidate_id": candidate_id,
            "parent_id": state["candidates"][candidate_id]["parent_id"],
            "frozen_inputs": frozen_inputs,
        })
        progress_usage["complete_coverage"] = bool(
            int(result.get("metrics", {}).get("n_periods", 0)) == int(manifest["date_count"])
        )
        result["reliability"] = progress_usage
        if stage == "smoke" and len(progress_usage["schema_retry_dates"]) > 1:
            result["gate"] = {"pass": False, "reason": "more_than_one_schema_retried_smoke_date"}
        state["consecutive_provider_failures"] = 0
    except CampaignStop:
        raise
    except Exception as exc:
        state["consecutive_provider_failures"] += 1
        failed_usage = progress_usage or campaign_progress_usage(progress_path)
        if validation:
            campaign_reconcile_all_cache_usage(campaign_dir, state, usage_key)
        else:
            campaign_reconcile_cache_usage(
                state,
                usage_key,
                sorted(treatment_cache_dir.glob("*.json")),
            )
        result = {"candidate_id": candidate_id, "parent_id": state["candidates"][candidate_id]["parent_id"], "stage": stage, "status": "failed", "gate": {"pass": False, "reason": "runner_or_artifact_failure"}, "reliability": failed_usage, "frozen_inputs": frozen_inputs, "error": str(exc)}
        if validation:
            failure_path = (
                campaign_dir
                / "validation/attempt_failures"
                / f"{campaign_utc_now().replace(':', '').replace('-', '')}.json"
            )
            campaign_atomic_json(failure_path, result)
            state.update({"current_candidate": None, "current_stage": None, "runner_pid": None})
            campaign_save(
                campaign_dir,
                state,
                "validation_attempt_failed",
                payload={"error": str(exc), "attempt_artifact": str(failure_path)},
            )
            reason = (
                "Three consecutive provider/runner failures"
                if state["consecutive_provider_failures"] >= 3
                else "Validation provider/runner failure checkpointed for resume"
            )
            raise CampaignStop(reason) from exc
        if state["consecutive_provider_failures"] >= 3:
            campaign_atomic_json(result_path, result)
            raise CampaignStop("Three consecutive provider/runner failures") from exc
    campaign_atomic_json(result_path, result)
    state["candidates"][candidate_id][stage] = result.get("gate", {}).get("pass", False)
    state.update({"current_candidate": None, "current_stage": None, "completed_dates": manifest["date_count"], "planned_dates": manifest["date_count"]})
    campaign_save(campaign_dir, state, "stage_completed", candidate_id=candidate_id, stage=stage, payload={"passed": result.get("gate", {}).get("pass", False)})
    return result


def campaign_results(campaign_dir: Path, candidate_ids: Sequence[str], stage: str) -> list[dict[str, Any]]:
    rows = []
    for candidate_id in candidate_ids:
        path = campaign_stage_result_path(campaign_dir, candidate_id, stage)
        if path.exists():
            rows.append(campaign_read_json(path))
    return rows


def campaign_finish(state: dict[str, Any], campaign_dir: Path, status: str, reason: str) -> None:
    state.update({"status": status, "phase": "terminal", "terminal_reason": reason, "current_candidate": None, "current_stage": None, "supervisor_pid": None, "runner_pid": None})
    campaign_save(campaign_dir, state, "campaign_terminal", payload={"status": status, "reason": reason})


def campaign_full_sort_key(result: dict[str, Any]) -> tuple[Any, ...]:
    metrics = result.get("metrics", {})
    control = result.get("controls", {}).get("fresh_incumbent", {})

    def finite(value: Any, default: float = -np.inf) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if np.isfinite(parsed) else default

    bootstrap = result.get("bootstrap_vs_incumbent", {})
    stability = finite(metrics.get("chronological_block_stability_pct"), 0.0)
    long_lift = finite(metrics.get("long_paired_lift_pct"))
    short_lift = finite(metrics.get("short_paired_lift_pct"))
    trimmed_lift = finite(metrics.get("trimmed_5pct_annualized_lift_pct"))
    return (
        -(trimmed_lift > 0.0),
        -stability,
        -(long_lift >= 0.0 and short_lift >= 0.0),
        -min(long_lift, short_lift),
        -finite(bootstrap.get("lower_95_pct")),
        -finite(metrics.get("paired_annualized_lift_pct")),
        -finite(metrics.get("sharpe_improvement")),
        -finite(metrics.get("max_drawdown_pct")),
        str(result.get("candidate_id")),
    )


def campaign_format_repair_parent_ids(
    smoke_results: Sequence[dict[str, Any]],
) -> list[str]:
    """Return at most one parent proven to have failed only strict wrapper parsing."""
    return [
        str(row["candidate_id"])
        for row in smoke_results
        if row.get("status") == "failed"
        and not row.get("gate", {}).get("pass")
        and row.get("gate", {}).get("reason") == "runner_or_artifact_failure"
        and 0 < len(row.get("reliability", {}).get("format_only_dates", [])) <= 1
        and int(row.get("reliability", {}).get("format_only_retries", 0)) > 0
        and int(row.get("reliability", {}).get("non_format_schema_retries", 0)) == 0
        and int(row.get("reliability", {}).get("transient_retries", 0)) == 0
    ][:1]


def campaign_run(
    campaign_dir: Path,
    *,
    researcher: Callable[[str], tuple[list[dict[str, Any]], dict[str, int]]] = campaign_call_researcher,
    executor: Callable[[Path, dict[str, Any], list[str], Path], None] | None = None,
) -> dict[str, Any]:
    with campaign_lock(campaign_dir):
        state = campaign_load(campaign_dir)
        campaign_verify_frozen_runtime(state)
        campaign_reconcile_all_cache_usage(campaign_dir, state, "development")
        state["usage"]["development"]["full_runs"] = sum(
            campaign_stage_result_path(campaign_dir, candidate_id, "full").exists()
            for candidate_id in state.get("candidates", {})
        )
        state["campaign_dir"] = str(campaign_dir)
        state["supervisor_pid"] = os.getpid()
        state["stop_requested"] = False
        state["error"] = None
        campaign_save(campaign_dir, state, "supervisor_started")
        try:
            if (campaign_dir / "STOP").exists():
                raise CampaignStop("Stop requested before campaign work")
            if state["status"] in {
                "statistically_confirmed",
                "validated_winner",
                "failed_validation",
                "no_winner",
                "awaiting_validation_approval",
                "validation_approved",
            }:
                state["supervisor_pid"] = None
                campaign_save(campaign_dir, state)
                return state
            g0_count = sum(
                row.get("generation") == "G0"
                for row in state["candidates"].values()
            )
            while g0_count < 6:
                catalog = campaign_read_json(campaign_dir / "catalog.json")
                proposal_prompt = campaign_researcher_prompt_g0(state, catalog)
                rejection_feedback = campaign_recent_rejection_feedback(
                    campaign_dir,
                    generation="G0",
                )
                if rejection_feedback:
                    proposal_prompt += (
                        "\nSTRUCTURAL_REJECTION_FEEDBACK_FROM_PRIOR_ATTEMPT:"
                        + json.dumps(rejection_feedback, sort_keys=True)
                        + "\nCorrect these schema/guard failures without changing the frozen harness."
                    )
                if g0_count:
                    proposal_prompt += (
                        f"\nThe frozen campaign already contains {g0_count} valid unique G0 candidates. "
                        f"Return exactly {6 - g0_count} additional candidates; do not repeat them."
                    )
                proposals = campaign_use_researcher(
                    campaign_dir, state, proposal_prompt, researcher
                )
                parents = {state["incumbent"]["candidate_id"]: (state["incumbent"]["prompt_text"], state["incumbent"]["prompt_hash"])}
                campaign_store_proposals(
                    campaign_dir,
                    state,
                    proposals,
                    generation="G0",
                    parents=parents,
                    limit=6 - g0_count,
                )
                next_count = sum(
                    row.get("generation") == "G0"
                    for row in state["candidates"].values()
                )
                if next_count == g0_count and int(
                    state["usage"]["development"]["pro_calls"]
                ) >= int(state["budgets"]["development"]["max_pro_calls"]):
                    break
                g0_count = next_count
            if g0_count != 6:
                raise CampaignStop("Researcher did not produce six valid G0 mutations")
            g0 = sorted(key for key, row in state["candidates"].items() if row["generation"] == "G0")
            smoke_g0 = [campaign_execute_stage(campaign_dir, state, cid, "smoke", executor=executor) for cid in g0]
            smoke_survivors = [row["candidate_id"] for row in smoke_g0 if row.get("gate", {}).get("pass")]
            format_repair_parents = campaign_format_repair_parent_ids(smoke_g0)
            if not smoke_survivors and not format_repair_parents:
                campaign_finish(state, campaign_dir, "no_winner", "no_smoke_survivors")
                return state
            mini_g0 = [campaign_execute_stage(campaign_dir, state, cid, "mini", executor=executor) for cid in smoke_survivors]
            mini_survivors = [row["candidate_id"] for row in mini_g0 if row.get("gate", {}).get("pass")]
            if not state.get("g1_generated"):
                g1_parents = mini_survivors + format_repair_parents
                pro_usage = int(state["usage"]["development"].get("pro_calls", 0))
                pro_limit = int(state["budgets"]["development"]["max_pro_calls"])
                if g1_parents and pro_usage < pro_limit:
                    feedback = [campaign_sanitized_feedback(row) for row in mini_g0]
                    feedback.extend({"candidate": cid, "stage": "smoke", "result_class": "schema_instability", "qualitative_labels": ["schema_instability"], "permitted_action": "format_repair_only"} for cid in format_repair_parents)
                    proposals = campaign_use_researcher(campaign_dir, state, campaign_researcher_prompt_g1(state, g1_parents, feedback), researcher)
                    parents = {}
                    for cid in g1_parents:
                        spec = campaign_read_json(campaign_dir / f"candidates/{cid}/prompt_spec.json")
                        parents[cid] = (spec["prompt_text"], spec["prompt_hash"])
                    repair_seen = False
                    bounded_proposals: list[dict[str, Any]] = []
                    for proposal in proposals:
                        is_repair = proposal.get("parent_id") in format_repair_parents
                        declared_repair = proposal.get("mutation_mode") == "format_repair"
                        if is_repair != declared_repair:
                            continue
                        if is_repair and repair_seen:
                            continue
                        repair_seen = repair_seen or is_repair
                        bounded_proposals.append(proposal)
                    campaign_store_proposals(campaign_dir, state, bounded_proposals, generation="G1", parents=parents, limit=2)
                elif g1_parents:
                    campaign_save(
                        campaign_dir,
                        state,
                        "g1_skipped",
                        payload={"reason": "Pro researcher budget exhausted"},
                    )
                state["g1_generated"] = True
                campaign_save(campaign_dir, state, "g1_closed")
            g1 = sorted(key for key, row in state["candidates"].items() if row["generation"] == "G1")
            smoke_g1 = [campaign_execute_stage(campaign_dir, state, cid, "smoke", executor=executor) for cid in g1]
            mini_g1 = [campaign_execute_stage(campaign_dir, state, row["candidate_id"], "mini", executor=executor) for row in smoke_g1 if row.get("gate", {}).get("pass")]
            combined = [row for row in mini_g0 + mini_g1 if row.get("gate", {}).get("pass")]
            if not combined:
                campaign_finish(state, campaign_dir, "no_winner", "no_mini_survivors")
                return state
            medium_ids = campaign_rank_results(combined, "mini")[:3]
            medium = [campaign_execute_stage(campaign_dir, state, cid, "medium", executor=executor) for cid in medium_ids]
            medium_pass = [row for row in medium if row.get("gate", {}).get("pass")]
            if not medium_pass:
                campaign_finish(state, campaign_dir, "no_winner", "no_medium_survivors")
                return state
            full_limit = int(state["budgets"]["development"]["max_full_runs"])
            full_ids = campaign_rank_results(medium_pass, "medium")[:full_limit]
            full = []
            for cid in full_ids:
                full.append(campaign_execute_stage(campaign_dir, state, cid, "full", executor=executor))
                state["usage"]["development"]["full_runs"] = sum(
                    campaign_stage_result_path(campaign_dir, candidate_id, "full").exists()
                    for candidate_id in state.get("candidates", {})
                )
            qualifiers = sorted([row for row in full if row.get("gate", {}).get("pass")], key=campaign_full_sort_key)
            if not qualifiers:
                campaign_finish(state, campaign_dir, "no_winner", "no_full_development_winner")
                return state
            winner = qualifiers[0]["candidate_id"]
            state.update({"winner_id": winner, "status": "awaiting_validation_approval", "phase": "approval", "current_candidate": None, "current_stage": None, "supervisor_pid": None})
            campaign_save(campaign_dir, state, "development_winner_frozen", candidate_id=winner)
            return state
        except CampaignStop as exc:
            state.update({"status": "stopped", "phase": "stopped", "terminal_reason": str(exc), "supervisor_pid": None, "runner_pid": None})
            campaign_save(campaign_dir, state, "campaign_stopped", payload={"reason": str(exc)})
            return state
        except Exception as exc:
            state.update({"status": "error", "phase": "error", "error": f"{type(exc).__name__}: {exc}", "supervisor_pid": None, "runner_pid": None})
            campaign_save(campaign_dir, state, "campaign_error", payload={"error": state["error"], "traceback": traceback.format_exc()})
            raise


def campaign_approve_validation(campaign_dir: Path, *, approver: str = "manual-cli") -> Path:
    with campaign_lock(campaign_dir):
        state = campaign_load(campaign_dir)
        campaign_verify_frozen_runtime(state)
        campaign_reconcile_all_cache_usage(campaign_dir, state, "development")
        if state.get("status") != "awaiting_validation_approval" or not state.get("winner_id"):
            raise CampaignError("Validation approval requires one frozen development winner")
        winner_id = str(state["winner_id"])
        spec, spec_path = campaign_verify_prompt_spec(campaign_dir, state, winner_id)
        full_result_path = campaign_stage_result_path(campaign_dir, winner_id, "full")
        full_result = campaign_execute_stage(campaign_dir, state, winner_id, "full")
        if not full_result.get("gate", {}).get("pass"):
            raise CampaignError("Frozen development winner no longer has a qualifying full result")
        manifest = campaign_build_validation_manifest(campaign_dir)
        state["validation"]["manifest_hash"] = manifest["manifest_hash"]
        state["validation"]["manifest_file_hash"] = campaign_file_hash(
            campaign_dir / "manifests/validation.json"
        )
        lineage = campaign_candidate_lineage(state, winner_id)
        lineage_hash = campaign_sha256(campaign_json_bytes(lineage))
        runner_harness_fingerprint = campaign_runner_harness_fingerprint(
            state, campaign_dir, spec_path, validation=True
        )
        development_results_hash = campaign_file_hash(full_result_path)
        side_scope = str(state.get("side_scope", "both"))
        allowed_prompt_specs: list[dict[str, Any]] | None = None
        if side_scope != "both":
            import mean_reversion_llm_v2 as runner_module
            import mean_reversion_llm_v2_side_experiments as side_experiments

            replica_paths: dict[str, Path] = {"r1": spec_path}
            for replicate_id in ("r2", "r3"):
                replica = dict(spec)
                replica["replicate_id"] = replicate_id
                replica_path = spec_path.parent / "replicates" / f"prompt_spec_{replicate_id}.json"
                campaign_atomic_json(replica_path, replica)
                runner_module.load_experimental_prompt_spec(replica_path)
                replica_paths[replicate_id] = replica_path
            parent_spec_path = campaign_dir / "validation/parent_prompt_spec.json"
            campaign_atomic_json(
                parent_spec_path, side_experiments.fixed_prompt_spec_payload(side_scope, "r1")
            )
            runner_module.load_experimental_prompt_spec(parent_spec_path)
            allowed_prompt_specs = []
            for role, candidate, paths in (
                ("winner", winner_id, replica_paths),
                ("dedicated_parent", f"dedicated-{side_scope}-parent", {"r1": parent_spec_path}),
            ):
                for replicate_id, allowed_path in paths.items():
                    allowed_spec = campaign_read_json(allowed_path)
                    allowed_prompt_specs.append(
                        {
                            "role": role,
                            "candidate_id": candidate,
                            "side_scope": side_scope,
                            "replicate_id": replicate_id,
                            "prompt_hash": allowed_spec["prompt_hash"],
                            "prompt_spec_hash": campaign_file_hash(allowed_path),
                            "harness_fingerprint": campaign_runner_harness_fingerprint(
                                state, campaign_dir, allowed_path, validation=True
                            ),
                        }
                    )
        freeze = {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": campaign_dir.name,
            "approved": True,
            "approval_status": "approved",
            "approved_at": campaign_utc_now(),
            "approved_by": str(approver),
            "candidate_id": winner_id,
            "prompt_hash": spec["prompt_hash"],
            "side_scope": side_scope,
            "replicate_id": "r1",
            "anonymize_tickers": bool(spec.get("anonymize_tickers", True)),
            "control_artifact": campaign_control_artifact(
                anonymize_tickers=bool(spec.get("anonymize_tickers", True)),
                validation=True,
            ),
            "trail5_artifact": campaign_trail5_artifact(validation=True),
            "prompt_text": spec["prompt_text"],
            "prompt_spec_path": str(spec_path.resolve()),
            "prompt_spec_hash": state["candidates"][winner_id]["prompt_spec_hash"],
            "lineage": lineage,
            "lineage_hash": lineage_hash,
            "code_fingerprint": state["code_fingerprint"],
            "supervisor_code_fingerprint": state["supervisor_code_fingerprint"],
            "harness_fingerprint": runner_harness_fingerprint,
            "development_results_hash": development_results_hash,
            "development_results_path": str(full_result_path.resolve()),
            "development_date_manifest_hash": state["development"]["manifest_hashes"]["full"],
            "development_date_manifest_file_hash": state["development"]["manifest_file_hashes"]["full"],
            "date_manifest_hash": manifest["manifest_hash"],
            "date_manifest_file_hash": state["validation"]["manifest_file_hash"],
            "exit_date_hash": manifest["exit_date_hash"],
            "temperature": CAMPAIGN_TEMPERATURE,
            "thinking_mode": CAMPAIGN_THINKING_MODE,
        }
        if allowed_prompt_specs is not None:
            freeze["allowed_prompt_specs"] = allowed_prompt_specs
        freeze_path = campaign_dir / "validation/validation_freeze.json"
        campaign_atomic_json(freeze_path, freeze)
        state["validation"].update({"approved": True, "approved_at": freeze["approved_at"], "freeze_hash": campaign_file_hash(freeze_path), "manifest_hash": manifest["manifest_hash"], "manifest_file_hash": state["validation"]["manifest_file_hash"], "harness_fingerprint": runner_harness_fingerprint})
        state.update({"status": "validation_approved", "phase": "validation"})
        campaign_save(campaign_dir, state, "validation_manually_approved", candidate_id=winner_id)
        return freeze_path


def campaign_holm_adjusted(p_values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(float(value) for value in p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(indexed)
    running = 0.0
    total = len(indexed)
    for rank, (original_idx, value) in enumerate(indexed):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[original_idx] = running
    return adjusted


def campaign_holm_adjusted_lower_bounds(
    bootstrap_results: Sequence[dict[str, Any]],
) -> list[float]:
    """Step-down simultaneous one-sided 95% lower bounds for two controls."""
    if len(bootstrap_results) != 2:
        raise CampaignError("Validation expects exactly two paired-control comparisons")
    ordering = sorted(
        range(2),
        key=lambda idx: (float(bootstrap_results[idx].get("p_not_positive", 1.0)), idx),
    )
    bounds = [-np.inf, -np.inf]
    # The smaller p-value must clear alpha/2; the second clears alpha.
    bounds[ordering[0]] = float(
        bootstrap_results[ordering[0]].get("lower_one_sided_97_5_pct", -np.inf)
    )
    bounds[ordering[1]] = float(
        bootstrap_results[ordering[1]].get("lower_one_sided_95_pct", -np.inf)
    )
    return bounds


def campaign_validation_decision(
    result: dict[str, Any], *, expected_periods: int
) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    controls = result.get("controls", {})
    incumbent = controls.get("fresh_incumbent", {})
    trail = controls.get("trail5", {})
    complete_coverage = int(metrics.get("n_periods", 0)) == int(expected_periods)
    economic = (
        complete_coverage
        and metrics.get("cagr_pct", -np.inf) > incumbent.get("cagr_pct", np.inf)
        and metrics.get("sharpe", -np.inf) > incumbent.get("sharpe", np.inf)
        and metrics.get("cagr_pct", -np.inf) > trail.get("cagr_pct", np.inf)
        and metrics.get("sharpe", -np.inf) > trail.get("sharpe", np.inf)
        and result.get("bootstrap_vs_incumbent", {}).get(
            "mean_annualized_delta_pct", -np.inf
        )
        > 0
        and result.get("bootstrap_vs_trail5", {}).get(
            "mean_annualized_delta_pct", -np.inf
        )
        > 0
        and metrics.get("max_drawdown_pct", -np.inf)
        >= incumbent.get("max_drawdown_pct", np.inf) - 2.0
    )
    bootstrap_results = [
        result["bootstrap_vs_incumbent"],
        result["bootstrap_vs_trail5"],
    ]
    adjusted_p = campaign_holm_adjusted(
        [row["p_not_positive"] for row in bootstrap_results]
    )
    adjusted_lower_bounds = campaign_holm_adjusted_lower_bounds(bootstrap_results)
    confirmed = (
        economic
        and all(value < 0.05 for value in adjusted_p)
        and all(value > 0.0 for value in adjusted_lower_bounds)
    )
    terminal = (
        "statistically_confirmed"
        if confirmed
        else ("validated_winner" if economic else "failed_validation")
    )
    return {
        "pass": bool(economic),
        "complete_coverage": bool(complete_coverage),
        "terminal_status": terminal,
        "holm_adjusted_p_not_positive": adjusted_p,
        "holm_adjusted_95pct_lower_bounds": {
            "fresh_minimal_change_control": adjusted_lower_bounds[0],
            "trail5": adjusted_lower_bounds[1],
        },
    }


def campaign_run_validation(
    campaign_dir: Path,
    *,
    executor: Callable[[Path, dict[str, Any], list[str], Path], None] | None = None,
) -> dict[str, Any]:
    with campaign_lock(campaign_dir):
        state = campaign_load(campaign_dir)
        terminal_statuses = {"statistically_confirmed", "validated_winner", "failed_validation"}
        if state.get("status") in terminal_statuses:
            return campaign_read_json(campaign_dir / "validation/result.json")
        if not state.get("validation", {}).get("approved"):
            raise CampaignError("Validation cannot run without explicit approval")
        if str(state.get("side_scope", "both")) != "both":
            batch_path_value = state.get("validation", {}).get("batch_freeze_path")
            batch_hash = state.get("validation", {}).get("batch_freeze_hash")
            if not batch_path_value or not batch_hash:
                raise CampaignError(
                    "Side validation requires one manually approved side-validation batch freeze"
                )
            batch_path = Path(str(batch_path_value))
            if not batch_path.is_file() or campaign_file_hash(batch_path) != batch_hash:
                raise CampaignFirewallError("Side-validation batch freeze changed after approval")
        campaign_verify_frozen_runtime(state)
        state.update(
            {
                "status": "validation_running",
                "phase": "validation",
                "supervisor_pid": os.getpid(),
                "runner_pid": None,
                "error": None,
            }
        )
        campaign_save(campaign_dir, state, "validation_supervisor_started")
        try:
            winner = str(state["winner_id"])
            spec, spec_path = campaign_verify_prompt_spec(campaign_dir, state, winner)
            freeze_path = campaign_dir / "validation/validation_freeze.json"
            freeze = campaign_read_json(freeze_path)
            if campaign_file_hash(freeze_path) != state["validation"].get("freeze_hash"):
                raise CampaignFirewallError("Validation-freeze file changed after approval")
            manifest = campaign_read_manifest(
                campaign_dir / "manifests/validation.json",
                validation=True,
                expected_hash=campaign_expected_manifest_hash(state, "validation", validation=True),
                expected_file_hash=campaign_expected_manifest_file_hash(
                    state, "validation", validation=True
                ),
            )
            lineage = campaign_candidate_lineage(state, winner)
            lineage_hash = campaign_sha256(campaign_json_bytes(lineage))
            full_result_path = campaign_stage_result_path(campaign_dir, winner, "full")
            expected = {
                "approved": True,
                "candidate_id": winner,
                "prompt_hash": spec["prompt_hash"],
                "side_scope": str(state.get("side_scope", "both")),
                "replicate_id": "r1",
                "anonymize_tickers": bool(spec.get("anonymize_tickers", True)),
                "control_artifact": campaign_control_artifact(
                    anonymize_tickers=bool(spec.get("anonymize_tickers", True)),
                    validation=True,
                ),
                "trail5_artifact": campaign_trail5_artifact(validation=True),
                "prompt_text": spec["prompt_text"],
                "prompt_spec_path": str(spec_path.resolve()),
                "prompt_spec_hash": state["candidates"][winner]["prompt_spec_hash"],
                "lineage": lineage,
                "lineage_hash": lineage_hash,
                "code_fingerprint": state["code_fingerprint"],
                "supervisor_code_fingerprint": state["supervisor_code_fingerprint"],
                "harness_fingerprint": campaign_runner_harness_fingerprint(
                    state, campaign_dir, spec_path, validation=True
                ),
                "development_results_hash": campaign_file_hash(full_result_path),
                "development_results_path": str(full_result_path.resolve()),
                "development_date_manifest_hash": state["development"]["manifest_hashes"]["full"],
                "development_date_manifest_file_hash": state["development"]["manifest_file_hashes"]["full"],
                "date_manifest_hash": manifest["manifest_hash"],
                "date_manifest_file_hash": state["validation"]["manifest_file_hash"],
                "exit_date_hash": manifest["exit_date_hash"],
                "temperature": CAMPAIGN_TEMPERATURE,
                "thinking_mode": CAMPAIGN_THINKING_MODE,
            }
            if str(state.get("side_scope", "both")) != "both":
                expected["allowed_prompt_specs"] = freeze.get("allowed_prompt_specs")
            if any(freeze.get(key) != value for key, value in expected.items()):
                raise CampaignFirewallError("Validation freeze fingerprint mismatch")

            campaign_reconcile_all_cache_usage(campaign_dir, state, "validation")
            result = campaign_execute_stage(
                campaign_dir,
                state,
                winner,
                "validation",
                validation=True,
                executor=executor,
            )
            if result.get("status") != "completed":
                raise CampaignStop("Validation runner did not produce a complete paired result")
            if str(state.get("side_scope", "both")) == "both":
                result["gate"] = campaign_validation_decision(
                    result, expected_periods=int(manifest["date_count"])
                )
            else:
                side_gate = dict(result.get("gate", {}))
                complete = int(result.get("metrics", {}).get("n_periods", 0)) == int(
                    manifest["date_count"]
                )
                passed = bool(side_gate.get("pass")) and complete
                side_gate.update(
                    {
                        "pass": passed,
                        "complete_coverage": complete,
                        "terminal_status": "validated_winner" if passed else "failed_validation",
                    }
                )
                result["gate"] = side_gate
            terminal = str(result["gate"]["terminal_status"])
            result_path = campaign_dir / "validation/result.json"
            campaign_atomic_json(result_path, result)
            inputs = {
                "candidate_id": winner,
                "prompt_hash": spec["prompt_hash"],
                "anonymize_tickers": bool(spec.get("anonymize_tickers", True)),
                "control_artifact": freeze["control_artifact"],
                "trail5_artifact": freeze["trail5_artifact"],
                "prompt_spec_path": str(spec_path.resolve()),
                "prompt_spec_hash": state["candidates"][winner]["prompt_spec_hash"],
                "validation_freeze": str(freeze_path.resolve()),
                "validation_freeze_hash": campaign_file_hash(freeze_path),
                "validation_result_hash": campaign_file_hash(result_path),
                "development_results_hash": freeze["development_results_hash"],
                "development_date_manifest_hash": freeze["development_date_manifest_hash"],
                "validation_date_manifest_hash": freeze["date_manifest_hash"],
                "validation_date_manifest_file_hash": freeze["date_manifest_file_hash"],
                "code_fingerprint": freeze["code_fingerprint"],
                "supervisor_code_fingerprint": freeze["supervisor_code_fingerprint"],
                "harness_fingerprint": freeze["harness_fingerprint"],
                "lineage_hash": freeze["lineage_hash"],
                "gross_only": True,
                "selection_files": sorted(
                    str(path.resolve())
                    for path in (campaign_dir / "validation").rglob("portfolio_selections.csv")
                ),
                "gross_return_files": sorted(
                    str(path.resolve())
                    for path in (campaign_dir / "validation").rglob("weekly_returns.csv")
                ),
                "turnover_source": "weekly_returns.csv columns turnover_long and turnover_short",
            }
            campaign_atomic_json(
                campaign_dir / "validation/actual_cost_experiment_inputs.json", inputs
            )
            campaign_finish(state, campaign_dir, terminal, terminal)
            return result
        except CampaignStop as exc:
            state.update(
                {
                    "status": "stopped",
                    "phase": "validation",
                    "terminal_reason": str(exc),
                    "supervisor_pid": None,
                    "runner_pid": None,
                }
            )
            campaign_save(
                campaign_dir,
                state,
                "validation_stopped",
                payload={"reason": str(exc)},
            )
            return {"status": "stopped", "reason": str(exc)}
        except Exception as exc:
            state.update(
                {
                    "status": "error",
                    "phase": "validation",
                    "error": f"{type(exc).__name__}: {exc}",
                    "supervisor_pid": None,
                    "runner_pid": None,
                }
            )
            campaign_save(
                campaign_dir,
                state,
                "validation_error",
                payload={"error": state["error"], "traceback": traceback.format_exc()},
            )
            raise


def campaign_request_stop(campaign_dir: Path) -> None:
    # The STOP sentinel is the race-free signal. Only the supervisor owns and
    # atomically replaces campaign.json while it is running.
    (campaign_dir / "STOP").touch()
    campaign_append_event(campaign_dir, "stop_requested")


def campaign_pid_is_alive(value: Any) -> bool:
    try:
        pid = int(value)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False


def campaign_record_startup_failure(campaign_dir: Path, exc: Exception) -> None:
    """Recover a detached launch that failed before campaign_run's inner handler."""
    try:
        state = campaign_load(campaign_dir)
    except (OSError, json.JSONDecodeError, CampaignError):
        return
    if not str(state.get("status", "")).startswith("starting"):
        return
    if campaign_pid_is_alive(state.get("supervisor_pid")):
        return
    state.update(
        {
            "status": "error",
            "phase": "startup",
            "error": f"{type(exc).__name__}: {exc}",
            "terminal_reason": None,
            "supervisor_pid": None,
            "runner_pid": None,
        }
    )
    campaign_save(
        campaign_dir,
        state,
        "supervisor_startup_failed",
        payload={"error": state["error"]},
    )


def campaign_write_dashboard(campaign_dir: Path) -> Path:
    from mean_reversion_prompt_dashboard import write_autoresearch_campaign_dashboard

    return Path(write_autoresearch_campaign_dashboard(campaign_dir))


def campaign_spawn(campaign_dir: Path, *, validation: bool = False) -> int:
    state = campaign_load(campaign_dir)
    if str(state.get("status", "")).startswith("starting"):
        launch_path = campaign_dir / ".supervisor_process.json"
        launch = campaign_read_json(launch_path) if launch_path.exists() else {}
        if campaign_pid_is_alive(launch.get("pid")):
            raise CampaignError("A supervisor is already starting for this campaign")
        state.update(
            {
                "status": "stopped",
                "phase": "startup",
                "terminal_reason": "Recovered stale detached-supervisor start",
                "supervisor_pid": None,
            }
        )
        campaign_save(campaign_dir, state, "stale_supervisor_start_recovered")
    if campaign_pid_is_alive(state.get("supervisor_pid")):
        raise CampaignError(f"Supervisor is already running as PID {state['supervisor_pid']}")
    state["stop_requested"] = False
    (campaign_dir / "STOP").unlink(missing_ok=True)
    state["supervisor_pid"] = None
    state["status"] = "starting_validation" if validation else "starting"
    campaign_save(campaign_dir, state)
    process_marker = campaign_dir / ".supervisor_process.json"
    campaign_atomic_json(
        process_marker,
        {"pid": os.getpid(), "role": "launcher", "created_at": campaign_utc_now()},
    )
    action = "_validate" if validation else "_run"
    command = [sys.executable, str(Path(__file__).resolve()), "campaign", action, campaign_dir.name, "--campaign-root", str(campaign_dir.parent)]
    log_handle = (campaign_dir / "supervisor.log").open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).parent,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
    except Exception as exc:
        log_handle.close()
        campaign_record_startup_failure(campaign_dir, exc)
        raise
    log_handle.close()
    campaign_atomic_json(
        process_marker,
        {"pid": process.pid, "role": "supervisor", "created_at": campaign_utc_now()},
    )
    campaign_append_event(campaign_dir, "supervisor_spawned", payload={"pid": process.pid, "validation": validation})
    return int(process.pid)


def campaign_cli(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="Run the bounded repo-native prompt autoresearch campaign.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(name: str) -> argparse.ArgumentParser:
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("campaign_id")
        command_parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT)
        return command_parser

    init = common("init")
    init.add_argument("--runner", type=Path, default=CAMPAIGN_RUNNER)
    init.add_argument("--date-source", type=Path, default=CAMPAIGN_DATE_SOURCE)
    init.add_argument("--side-scope", choices=("both", "long", "short"), default="both")
    init.add_argument("--side-parent-cache", type=Path)
    migrate = subparsers.add_parser("migrate")
    migrate.add_argument("source_campaign_id")
    migrate.add_argument("campaign_id", help="New target campaign ID")
    migrate.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT)
    migrate.add_argument(
        "--reuse-completed-stages",
        action="store_true",
        help="Verify, rebind, and copy completed development stages from the source",
    )
    start = common("start")
    start.add_argument("--foreground", action="store_true")
    resume = common("resume")
    resume.add_argument("--foreground", action="store_true")
    common("status")
    common("stop")
    common("report")
    approve = common("approve-validation")
    approve.add_argument("--approver", default="manual-cli")
    common("_run")
    common("_validate")
    args = parser.parse_args(list(argv))
    campaign_dir = campaign_path(args.campaign_root, args.campaign_id)
    if args.command == "init":
        print(
            campaign_initialize(
                args.campaign_id,
                root=args.campaign_root,
                runner=args.runner,
                date_source=args.date_source,
                side_scope=args.side_scope,
                side_parent_cache=args.side_parent_cache,
            )
        )
    elif args.command == "migrate":
        print(
            campaign_migrate_cached_candidates(
                args.source_campaign_id,
                args.campaign_id,
                root=args.campaign_root,
                reuse_completed_stages=args.reuse_completed_stages,
            )
        )
    elif args.command in {"start", "resume"}:
        state = campaign_load(campaign_dir)
        validation = bool(
            state.get("validation", {}).get("approved")
            and (campaign_dir / "validation/validation_freeze.json").exists()
        )
        if state.get("status") == "awaiting_validation_approval":
            raise CampaignError("Validation must be explicitly approved before start/resume")
        if args.foreground:
            (campaign_dir / "STOP").unlink(missing_ok=True)
            result = campaign_run_validation(campaign_dir) if validation else campaign_run(campaign_dir)
            print(json.dumps(campaign_status_projection(result if validation is False else campaign_load(campaign_dir)), indent=2, default=str))
        else:
            print(f"supervisor_pid: {campaign_spawn(campaign_dir, validation=validation)}")
    elif args.command == "status":
        print(json.dumps(campaign_read_json(campaign_dir / "status.json"), indent=2, default=str))
    elif args.command == "stop":
        campaign_request_stop(campaign_dir)
        print("stop requested")
    elif args.command == "report":
        print(campaign_write_dashboard(campaign_dir))
    elif args.command == "approve-validation":
        print(campaign_approve_validation(campaign_dir, approver=args.approver))
    elif args.command == "_run":
        try:
            campaign_run(campaign_dir)
        except Exception as exc:
            campaign_record_startup_failure(campaign_dir, exc)
            raise
    elif args.command == "_validate":
        try:
            campaign_run_validation(campaign_dir)
        except Exception as exc:
            campaign_record_startup_failure(campaign_dir, exc)
            raise


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "campaign":
        campaign_cli(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description="Run zero-token autoresearch sweep over completed V2 caches.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor", default="3")
    parser.add_argument("--pool-per-side", type=int, default=60)
    parser.add_argument("--positions-per-side", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--random-placebo-runs", type=int, default=2000)
    parser.add_argument("--report-only", action="store_true", help="Regenerate HTML from existing autoresearch CSVs.")
    args = parser.parse_args()
    if args.report_only:
        paths = rewrite_report_from_existing(
            output_dir=args.output_dir,
            random_placebo_runs=int(args.random_placebo_runs),
        )
    else:
        paths = run_autoresearch(
            output_dir=args.output_dir,
            anchor=str(args.anchor),
            pool=int(args.pool_per_side),
            book=int(args.positions_per_side),
            n_boot=int(args.n_boot),
            random_placebo_runs=int(args.random_placebo_runs),
        )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
