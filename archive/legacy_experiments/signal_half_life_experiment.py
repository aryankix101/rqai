"""
Signal half-life experiment on full-universe composite scores.

Outputs two decay views for a chosen score:
1. Structural decay: cross-sectional rank persistence of the score itself.
2. Economic decay: cross-sectional IC of a stale score versus future returns.

This script uses daily cross-sectional percentile ranks and row-wise Pearson
correlation across overlapping tickers, which is equivalent to a fast
rank-correlation approximation for large cross-sections.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

_MPL_DIR = Path("backtest_results/.mplconfig")
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR.resolve()))

import matplotlib.pyplot as plt


COMPOSITE_SCORES_DIR = Path("backtest_results/composite_scores")
OUTPUT_DIR = Path("backtest_results/half_life_experiment")
SUPPORTED_HORIZONS = (5, 10, 20, 30, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate structural and economic signal half-life from composite score CSVs."
    )
    parser.add_argument(
        "--score-type",
        choices=["direction_score", "magnitude_score"],
        default="direction_score",
        help="Score column to analyze (default: direction_score)",
    )
    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=COMPOSITE_SCORES_DIR,
        help="Directory containing composite score CSVs",
    )
    parser.add_argument(
        "--target-horizons",
        type=str,
        default="5,10,20,30,60",
        help="Comma-separated forward-return horizons for economic decay",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=60,
        help="Maximum lag in trading days for decay curves",
    )
    parser.add_argument(
        "--min-cross-section",
        type=int,
        default=50,
        help="Minimum overlapping tickers required for a date-level observation",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional inclusive start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional inclusive end date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional ticker filter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for CSV/JSON/PNG outputs",
    )
    return parser.parse_args()


def parse_horizons(raw: str) -> list[int]:
    horizons = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in SUPPORTED_HORIZONS:
            raise ValueError(f"Unsupported horizon {value}; choose from {SUPPORTED_HORIZONS}")
        horizons.append(value)
    if not horizons:
        raise ValueError("At least one target horizon is required")
    return sorted(set(horizons))


def load_matrices(
    scores_dir: Path,
    score_type: str,
    target_horizons: list[int],
    start_date: str | None,
    end_date: str | None,
    tickers: list[str] | None,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    if not scores_dir.exists():
        raise FileNotFoundError(f"Scores directory not found: {scores_dir}")

    ticker_filter = {t.upper() for t in tickers} if tickers else None
    score_series = {}
    return_series: dict[int, dict[str, pd.Series]] = {h: {} for h in target_horizons}

    usecols = ["date", score_type] + [
        f"fwd_abs_{h}d_ret" if score_type == "magnitude_score" else f"fwd_{h}d_ret"
        for h in target_horizons
    ]

    csv_files = sorted(scores_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No composite score CSVs found in {scores_dir}")

    print(f"Loading {len(csv_files)} composite score files...")

    loaded = 0
    for csv_file in csv_files:
        ticker = csv_file.stem.split("_")[2].upper()
        if ticker_filter and ticker not in ticker_filter:
            continue

        try:
            df = pd.read_csv(csv_file, usecols=usecols, parse_dates=["date"])
        except Exception as exc:
            print(f"  Skipping {csv_file.name}: {exc}")
            continue

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]
        if df.empty:
            continue

        indexed = df.sort_values("date").set_index("date")
        score_series[ticker] = indexed[score_type]
        for horizon in target_horizons:
            col = f"fwd_abs_{horizon}d_ret" if score_type == "magnitude_score" else f"fwd_{horizon}d_ret"
            return_series[horizon][ticker] = indexed[col]
        loaded += 1

    if not score_series:
        raise ValueError("No score data loaded for the requested filters")

    score_matrix = pd.DataFrame(score_series).sort_index()
    return_matrices = {
        horizon: pd.DataFrame(series_dict).reindex(score_matrix.index).sort_index()
        for horizon, series_dict in return_series.items()
    }

    print(f"Loaded {loaded} tickers")
    print(f"Date range: {score_matrix.index.min().date()} -> {score_matrix.index.max().date()}")
    print(f"Average available tickers/date: {score_matrix.notna().sum(axis=1).mean():.1f}")
    return score_matrix, return_matrices


def rowwise_masked_corr(a: np.ndarray, b: np.ndarray, min_obs: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    n_obs = mask.sum(axis=1)

    safe_n = np.where(n_obs > 0, n_obs, 1)
    a0 = np.where(mask, a, 0.0)
    b0 = np.where(mask, b, 0.0)

    mean_a = a0.sum(axis=1) / safe_n
    mean_b = b0.sum(axis=1) / safe_n

    centered_a = np.where(mask, a - mean_a[:, None], 0.0)
    centered_b = np.where(mask, b - mean_b[:, None], 0.0)

    cov = (centered_a * centered_b).sum(axis=1)
    var_a = (centered_a * centered_a).sum(axis=1)
    var_b = (centered_b * centered_b).sum(axis=1)
    denom = np.sqrt(var_a * var_b)

    corr = np.divide(cov, denom, out=np.full_like(cov, np.nan, dtype=float), where=denom > 0)
    corr[n_obs < min_obs] = np.nan
    return corr, n_obs


def build_rank_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, method="average", pct=True)


def summarize_curve(
    lags: np.ndarray,
    values: np.ndarray,
    counts: np.ndarray,
    avg_cross_section: np.ndarray,
    value_label: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lag_days": lags,
            value_label: values,
            "valid_periods": counts,
            "avg_cross_section": avg_cross_section,
        }
    )


def estimate_half_life(lags: np.ndarray, values: np.ndarray, baseline_lag: int) -> float:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.nan

    baseline_idx = np.where(lags == baseline_lag)[0]
    if len(baseline_idx) == 0:
        return np.nan

    baseline_val = values[baseline_idx[0]]
    if not np.isfinite(baseline_val) or baseline_val <= 0:
        return np.nan

    threshold = baseline_val * 0.5
    for idx in range(baseline_idx[0] + 1, len(lags)):
        prev_val = values[idx - 1]
        curr_val = values[idx]
        if not np.isfinite(prev_val) or not np.isfinite(curr_val):
            continue
        if curr_val <= threshold:
            prev_lag = lags[idx - 1]
            curr_lag = lags[idx]
            if prev_val == curr_val:
                return float(curr_lag)
            frac = (threshold - prev_val) / (curr_val - prev_val)
            return float(prev_lag + frac * (curr_lag - prev_lag))
    return np.nan


def compute_structural_decay(
    score_ranks: pd.DataFrame,
    max_lag: int,
    min_cross_section: int,
) -> tuple[pd.DataFrame, float]:
    values = score_ranks.to_numpy(dtype=float)
    lags = np.arange(1, max_lag + 1)
    mean_corrs = []
    valid_counts = []
    avg_cross = []

    print("Computing structural decay...")
    for lag in lags:
        left = values[:-lag]
        right = values[lag:]
        corr, n_obs = rowwise_masked_corr(left, right, min_cross_section)
        finite = np.isfinite(corr)
        mean_corrs.append(float(np.nanmean(corr)) if finite.any() else np.nan)
        valid_counts.append(int(finite.sum()))
        avg_cross.append(float(np.nanmean(n_obs[finite])) if finite.any() else np.nan)

    curve = summarize_curve(
        lags=lags,
        values=np.array(mean_corrs),
        counts=np.array(valid_counts),
        avg_cross_section=np.array(avg_cross),
        value_label="mean_rank_corr",
    )
    half_life = estimate_half_life(curve["lag_days"].to_numpy(), curve["mean_rank_corr"].to_numpy(), baseline_lag=1)
    return curve, half_life


def compute_economic_decay(
    score_ranks: pd.DataFrame,
    return_matrices: dict[int, pd.DataFrame],
    max_lag: int,
    min_cross_section: int,
) -> tuple[pd.DataFrame, dict[int, float]]:
    score_values = score_ranks.to_numpy(dtype=float)
    rows = []
    half_lives = {}

    print("Computing economic decay...")
    for horizon, ret_df in return_matrices.items():
        ret_ranks = build_rank_matrix(ret_df).to_numpy(dtype=float)
        lags = np.arange(0, max_lag + 1)
        ic_values = []

        for lag in lags:
            if lag == 0:
                left = score_values
                right = ret_ranks
            else:
                left = score_values[:-lag]
                right = ret_ranks[lag:]

            corr, n_obs = rowwise_masked_corr(left, right, min_cross_section)
            finite = np.isfinite(corr)
            mean_ic = float(np.nanmean(corr)) if finite.any() else np.nan
            ic_values.append(mean_ic)
            rows.append(
                {
                    "holding_days": horizon,
                    "lag_days": lag,
                    "mean_ic": mean_ic,
                    "valid_periods": int(finite.sum()),
                    "avg_cross_section": float(np.nanmean(n_obs[finite])) if finite.any() else np.nan,
                }
            )

        half_lives[horizon] = estimate_half_life(lags, np.array(ic_values), baseline_lag=0)

    return pd.DataFrame(rows), half_lives


def plot_structural_decay(curve: pd.DataFrame, half_life: float, output_path: Path, score_type: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(curve["lag_days"], curve["mean_rank_corr"], color="#2f7ed8", linewidth=2.2)
    ax.axhline(curve["mean_rank_corr"].iloc[0], color="#999999", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(curve["mean_rank_corr"].iloc[0] * 0.5, color="#d95f02", linestyle=":", linewidth=1.5, alpha=0.9)
    if np.isfinite(half_life):
        ax.axvline(half_life, color="#d95f02", linestyle="--", linewidth=1.5)
        ax.text(half_life + 0.8, curve["mean_rank_corr"].iloc[0] * 0.52, f"half-life ~ {half_life:.1f}d",
                color="#d95f02", fontsize=10)
    ax.set_title(f"Structural Decay: {score_type}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Mean cross-sectional rank correlation")
    ax.set_ylim(bottom=min(-0.05, curve["mean_rank_corr"].min() - 0.05), top=1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_economic_decay(curve: pd.DataFrame, half_lives: dict[int, float], output_path: Path, score_type: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = {
        5: "#1f77b4",
        10: "#ff7f0e",
        20: "#2ca02c",
        30: "#d62728",
        60: "#9467bd",
    }

    for horizon, sub in curve.groupby("holding_days"):
        sub = sub.sort_values("lag_days")
        ax.plot(sub["lag_days"], sub["mean_ic"], label=f"{horizon}d hold", color=palette.get(horizon), linewidth=2.0)
        hl = half_lives.get(int(horizon))
        if np.isfinite(hl):
            ax.axvline(hl, color=palette.get(int(horizon)), linestyle=":", linewidth=1.0, alpha=0.5)

    ax.axhline(0, color="#666666", linestyle="--", linewidth=1)
    ax.set_title(f"Economic Decay: stale {score_type} vs future returns", fontsize=14, fontweight="bold")
    ax.set_xlabel("Signal age at entry (trading days)")
    ax.set_ylabel("Mean cross-sectional IC")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary(
    structural_curve: pd.DataFrame,
    structural_half_life: float,
    economic_curve: pd.DataFrame,
    economic_half_lives: dict[int, float],
    output_path: Path,
    score_type: str,
    target_horizons: list[int],
) -> None:
    lag1 = structural_curve.loc[structural_curve["lag_days"] == 1, "mean_rank_corr"]
    summary = {
        "score_type": score_type,
        "target_horizons": target_horizons,
        "structural_half_life_days": structural_half_life,
        "structural_lag1_corr": float(lag1.iloc[0]) if not lag1.empty else None,
        "economic_half_life_days": {str(k): v for k, v in economic_half_lives.items()},
    }
    output_path.write_text(json.dumps(summary, indent=2))


def print_summary(structural_half_life: float, economic_half_lives: dict[int, float], structural_curve: pd.DataFrame) -> None:
    lag1 = structural_curve.loc[structural_curve["lag_days"] == 1, "mean_rank_corr"].iloc[0]
    lag5 = structural_curve.loc[structural_curve["lag_days"] == 5, "mean_rank_corr"]
    lag20 = structural_curve.loc[structural_curve["lag_days"] == 20, "mean_rank_corr"]

    print("\nHalf-life summary")
    print("-" * 60)
    print(f"Structural lag-1 persistence : {lag1:.4f}")
    if not lag5.empty:
        print(f"Structural lag-5 persistence : {lag5.iloc[0]:.4f}")
    if not lag20.empty:
        print(f"Structural lag-20 persistence: {lag20.iloc[0]:.4f}")
    print(f"Structural half-life         : {structural_half_life:.2f} trading days")
    for horizon in sorted(economic_half_lives):
        value = economic_half_lives[horizon]
        if np.isfinite(value):
            print(f"Economic half-life ({horizon:>2d}d hold): {value:.2f} trading days")
        else:
            print(f"Economic half-life ({horizon:>2d}d hold): n/a")


def main() -> None:
    args = parse_args()
    target_horizons = parse_horizons(args.target_horizons)

    score_matrix, return_matrices = load_matrices(
        scores_dir=args.scores_dir,
        score_type=args.score_type,
        target_horizons=target_horizons,
        start_date=args.start_date,
        end_date=args.end_date,
        tickers=args.tickers,
    )
    score_ranks = build_rank_matrix(score_matrix)

    structural_curve, structural_half_life = compute_structural_decay(
        score_ranks=score_ranks,
        max_lag=args.max_lag,
        min_cross_section=args.min_cross_section,
    )
    economic_curve, economic_half_lives = compute_economic_decay(
        score_ranks=score_ranks,
        return_matrices=return_matrices,
        max_lag=args.max_lag,
        min_cross_section=args.min_cross_section,
    )

    out_dir = args.output_dir / args.score_type
    out_dir.mkdir(parents=True, exist_ok=True)

    structural_csv = out_dir / "structural_decay.csv"
    economic_csv = out_dir / "economic_decay.csv"
    summary_json = out_dir / "half_life_summary.json"
    structural_png = out_dir / "structural_decay.png"
    economic_png = out_dir / "economic_decay.png"

    structural_curve.to_csv(structural_csv, index=False)
    economic_curve.to_csv(economic_csv, index=False)
    save_summary(
        structural_curve=structural_curve,
        structural_half_life=structural_half_life,
        economic_curve=economic_curve,
        economic_half_lives=economic_half_lives,
        output_path=summary_json,
        score_type=args.score_type,
        target_horizons=target_horizons,
    )
    plot_structural_decay(structural_curve, structural_half_life, structural_png, args.score_type)
    plot_economic_decay(economic_curve, economic_half_lives, economic_png, args.score_type)

    print_summary(structural_half_life, economic_half_lives, structural_curve)
    print("\nSaved outputs")
    print(f"  {structural_csv}")
    print(f"  {economic_csv}")
    print(f"  {summary_json}")
    print(f"  {structural_png}")
    print(f"  {economic_png}")


if __name__ == "__main__":
    main()
