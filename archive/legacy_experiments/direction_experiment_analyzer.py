#!/usr/bin/env python3
"""Summarize and visualize direction_score backtest variants."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.errors import EmptyDataError


sns.set_style("darkgrid")

BASE_DIR = Path(__file__).parent
DEFAULT_BACKTEST_DIR = BASE_DIR / "backtest_results" / "factor_backtest"
DEFAULT_OUTPUT_DIR = BASE_DIR / "backtest_results" / "analysis_output" / "direction_experiments"
REVIEW_INTERVAL_COLUMNS = [
    "review_normal_days",
    "review_grind_up_days",
    "review_slow_bleed_days",
    "review_squeeze_days",
    "review_crisis_days",
    "review_unknown_days",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze direction_score experiment variants.")
    parser.add_argument("--backtest-dir", type=Path, default=DEFAULT_BACKTEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sweep-summary",
        type=Path,
        default=None,
        help="Optional direction_sweep_summary.csv to analyze without raw factor_backtest folders.",
    )
    parser.add_argument("--horizon", type=int, default=10, help="Only analyze this horizon.")
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Comma-separated horizons to analyze together, e.g. 10,15,20.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Substring filter. Repeat to require multiple substrings in variant names.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Substring filter. Repeat to exclude matching variant names.",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Include llm_* variants. Default is DET-like runs only.",
    )
    return parser.parse_args()


def is_det_like_variant(name: str) -> bool:
    return not name.startswith("llm_")


def parse_horizons_arg(value: str | None, fallback_horizon: int) -> list[int]:
    if not value:
        return [fallback_horizon]
    horizons = [int(chunk.strip()) for chunk in value.split(",") if chunk.strip()]
    if not horizons:
        return [fallback_horizon]
    return list(dict.fromkeys(horizons))


def discover_direction_runs(
    backtest_dir: Path,
    horizons: list[int],
    include_filters: list[str],
    exclude_filters: list[str],
    include_llm: bool,
) -> list[Path]:
    score_dirs: list[Path] = []

    for variant_dir in sorted(backtest_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        name = variant_dir.name
        if not include_llm and not is_det_like_variant(name):
            continue
        if include_filters and not all(token in name for token in include_filters):
            continue
        if any(token in name for token in exclude_filters):
            continue

        for horizon in horizons:
            score_dir = variant_dir / f"{horizon}d_horizon" / "direction_score"
            if (score_dir / "factor_pnl.csv").exists():
                score_dirs.append(score_dir)

    return score_dirs


def safe_cagr(cumulative_series: pd.Series) -> float:
    if len(cumulative_series) < 2:
        return np.nan
    terminal = float(cumulative_series.iloc[-1])
    if not np.isfinite(terminal) or terminal <= 0:
        return np.nan
    return float((terminal ** (252.0 / len(cumulative_series)) - 1.0) * 100.0)


def compute_max_drawdown_pct(cumulative_series: pd.Series) -> float:
    if cumulative_series.empty:
        return np.nan
    running_max = cumulative_series.cummax()
    drawdown = (cumulative_series / running_max) - 1.0
    return float(drawdown.min() * 100.0)


def mean_decile_spread(decile_df: pd.DataFrame) -> float:
    if decile_df.empty:
        return np.nan
    pivot = decile_df.pivot(index="date", columns="decile", values="mean_ret")
    if 1 not in pivot.columns or 10 not in pivot.columns:
        return np.nan
    spread = pivot[10] - pivot[1]
    return float(spread.mean())


def load_manifest(score_dir: Path) -> dict:
    manifest_path = score_dir.parent / "run_config.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_optional_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except EmptyDataError:
        return pd.DataFrame()


def is_missing_value(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def format_label_value(value: object) -> str | None:
    if is_missing_value(value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


def compact_variant_name(name: str, max_chars: int = 72) -> str:
    pretty = name.replace("__", " | ")
    if len(pretty) <= max_chars:
        return pretty
    head = (max_chars - 3) // 2
    tail = max_chars - 3 - head
    return f"{pretty[:head]}...{pretty[-tail:]}"


def build_display_label(summary_row: pd.Series | dict[str, object], include_horizon: bool = True) -> str:
    horizon = summary_row.get("horizon")
    review_mode = summary_row.get("review_mode")
    hysteresis_mode = summary_row.get("hysteresis_mode")
    weighting_mode = summary_row.get("weighting_mode")
    top_n = format_label_value(summary_row.get("top_n"))
    bottom_n = format_label_value(summary_row.get("bottom_n"))
    pool_mode = summary_row.get("candidate_pool_mode")

    parts: list[str] = []

    if include_horizon and not is_missing_value(horizon):
        parts.append(str(horizon).replace("_horizon", ""))

    if review_mode == "adaptive":
        interval_values = [format_label_value(summary_row.get(column)) for column in REVIEW_INTERVAL_COLUMNS]
        if all(value is not None for value in interval_values):
            parts.append(f"adp {'/'.join(interval_values)}")
        else:
            parts.append("adp")
    elif not is_missing_value(review_mode):
        static_days = format_label_value(summary_row.get("review_static_days"))
        parts.append(f"static {static_days}d" if static_days is not None else str(review_mode))

    if hysteresis_mode == "combined":
        lb = format_label_value(summary_row.get("long_retain_buffer"))
        sbf = format_label_value(summary_row.get("short_retain_buffer"))
        gap = format_label_value(summary_row.get("replacement_gap"))
        if lb is not None and sbf is not None and gap is not None:
            parts.append(f"h{lb}/{sbf}/{gap}")
        else:
            parts.append("hyst")
    elif not is_missing_value(hysteresis_mode):
        parts.append(str(hysteresis_mode))

    if not is_missing_value(weighting_mode):
        parts.append(str(weighting_mode))

    if top_n is not None or bottom_n is not None:
        parts.append(f"{top_n or '?'}x{bottom_n or '?'}")

    if pool_mode == "llm_mirror":
        magnitude = format_label_value(summary_row.get("top_n_magnitude_per_side"))
        threshold = format_label_value(summary_row.get("direction_threshold"))
        pool_parts = ["mirror"]
        if magnitude is not None:
            pool_parts.append(f"m{magnitude}")
        if threshold is not None:
            pool_parts.append(f"d{threshold}")
        parts.append(" ".join(pool_parts))
    elif not is_missing_value(pool_mode):
        parts.append(str(pool_mode))

    if len(parts) <= 1:
        variant_name = str(summary_row.get("variant_name", ""))
        return compact_variant_name(variant_name)
    return " | ".join(parts)


def uniquify_labels(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique_labels: list[str] = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        suffix = counts[label]
        unique_labels.append(label if suffix == 1 else f"{label} #{suffix}")
    return unique_labels


def attach_plot_metadata(summary_df: pd.DataFrame) -> pd.DataFrame:
    labeled = summary_df.copy()
    labeled["display_label"] = uniquify_labels([build_display_label(row) for _, row in labeled.iterrows()])
    labeled["plot_id"] = [f"S{i:02d}" for i in range(1, len(labeled) + 1)]
    return labeled


def select_annotation_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    if len(summary_df) <= 12:
        return summary_df

    selected: set[int] = set()
    total_return = pd.to_numeric(summary_df["total_return_pct"], errors="coerce").dropna()
    sharpe = pd.to_numeric(summary_df["sharpe"], errors="coerce").dropna()
    cost = pd.to_numeric(summary_df["total_cost_bps"], errors="coerce").dropna()

    selected.update(total_return.nlargest(min(6, len(total_return))).index.tolist())
    selected.update(total_return.nsmallest(min(3, len(total_return))).index.tolist())
    selected.update(sharpe.nlargest(min(3, len(sharpe))).index.tolist())
    selected.update(cost.nsmallest(min(2, len(cost))).index.tolist())
    selected.update(cost.nlargest(min(2, len(cost))).index.tolist())

    return summary_df.loc[sorted(selected)]


def normalize_strategy_key(variant_name: str) -> str:
    return re.sub(r"__fd_\d+(?=__|$)", "", variant_name)


def build_cross_horizon_leaderboard(summary_df: pd.DataFrame) -> pd.DataFrame:
    enriched = summary_df.copy()
    enriched["strategy_key"] = enriched["variant_name"].map(normalize_strategy_key)
    enriched["horizon_tag"] = enriched["horizon"].astype(str).str.replace("_horizon", "", regex=False)
    unique_strategies = enriched.drop_duplicates("strategy_key").copy()
    unique_strategies["strategy_display_label"] = uniquify_labels(
        [build_display_label(row, include_horizon=False) for _, row in unique_strategies.iterrows()]
    )
    strategy_label_map = dict(
        zip(
            unique_strategies["strategy_key"],
            unique_strategies["strategy_display_label"],
            strict=True,
        )
    )
    enriched["strategy_display_label"] = enriched["strategy_key"].map(strategy_label_map)

    metrics = (
        enriched.pivot_table(
            index=["strategy_key", "strategy_display_label"],
            columns="horizon_tag",
            values=["total_return_pct", "sharpe", "total_cost_bps"],
            aggfunc="first",
        )
        .sort_index(axis=1)
    )
    metrics.columns = [f"{metric}_{horizon}" for metric, horizon in metrics.columns]
    leaderboard = metrics.reset_index()

    total_return_cols = [column for column in leaderboard.columns if column.startswith("total_return_pct_")]
    sharpe_cols = [column for column in leaderboard.columns if column.startswith("sharpe_")]
    cost_cols = [column for column in leaderboard.columns if column.startswith("total_cost_bps_")]

    leaderboard["n_horizons"] = leaderboard[total_return_cols].notna().sum(axis=1)
    leaderboard["avg_total_return_pct"] = leaderboard[total_return_cols].mean(axis=1)
    leaderboard["min_total_return_pct"] = leaderboard[total_return_cols].min(axis=1)
    leaderboard["max_total_return_pct"] = leaderboard[total_return_cols].max(axis=1)
    leaderboard["avg_sharpe"] = leaderboard[sharpe_cols].mean(axis=1)
    leaderboard["avg_total_cost_bps"] = leaderboard[cost_cols].mean(axis=1)
    leaderboard["horizon_return_range_pct"] = (
        leaderboard["max_total_return_pct"] - leaderboard["min_total_return_pct"]
    )

    return leaderboard.sort_values(
        ["avg_total_return_pct", "avg_sharpe", "min_total_return_pct"],
        ascending=[False, False, False],
    )


def plot_cross_horizon_heatmap(cross_horizon_df: pd.DataFrame, output_dir: Path) -> None:
    return_cols = [column for column in cross_horizon_df.columns if column.startswith("total_return_pct_")]
    if not return_cols:
        return

    top = cross_horizon_df.head(20).copy()
    heatmap_df = top.set_index("strategy_display_label")[return_cols]
    renamed_columns = {
        column: column.replace("total_return_pct_", "")
        for column in return_cols
    }
    heatmap_df = heatmap_df.rename(columns=renamed_columns)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(heatmap_df))))
    sns.heatmap(heatmap_df, cmap="RdYlGn", center=0, annot=True, fmt=".1f", ax=ax)
    ax.set_title("Direction Experiments — Cross-Horizon Return Leaderboard")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(output_dir / "cross_horizon_return_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_summary_from_sweep_csv(
    sweep_summary_path: Path,
    include_filters: list[str],
    exclude_filters: list[str],
) -> pd.DataFrame:
    summary_df = pd.read_csv(sweep_summary_path)
    if "status" in summary_df.columns:
        summary_df = summary_df[summary_df["status"] == "ok"].copy()
    if include_filters:
        mask = summary_df["variant_label"].astype(str).apply(lambda name: all(token in name for token in include_filters))
        summary_df = summary_df[mask].copy()
    if exclude_filters:
        mask = summary_df["variant_label"].astype(str).apply(lambda name: not any(token in name for token in exclude_filters))
        summary_df = summary_df[mask].copy()
    if summary_df.empty:
        return summary_df

    summary_df = summary_df.rename(
        columns={
            "variant_label": "variant_name",
            "forward_days": "forward_days",
            "top_n_magnitude": "top_n_magnitude_per_side",
        }
    )
    summary_df["horizon"] = summary_df["forward_days"].astype(int).astype(str) + "d_horizon"
    return summary_df


def summarize_run(score_dir: Path) -> tuple[dict[str, object], pd.Series]:
    variant_name = score_dir.parent.parent.name
    horizon_name = score_dir.parent.name

    pnl_df = pd.read_csv(score_dir / "factor_pnl.csv", parse_dates=["date"])
    review_path = score_dir / "review_log.csv"
    review_df = read_optional_csv(review_path)
    decile_path = score_dir / "decile_analysis.csv"
    decile_df = read_optional_csv(decile_path)
    manifest = load_manifest(score_dir)
    review_policy = manifest.get("review_policy", {})
    hysteresis_policy = manifest.get("hysteresis_policy", {})
    regime_intervals = review_policy.get("regime_intervals", {})
    candidate_pool = manifest.get("candidate_pool", {})

    ret_bps = pnl_df["daily_ret_bps"].astype(float)
    daily_ret = ret_bps / 10000.0
    cumulative = (1.0 + daily_ret).cumprod()

    gross_total_return = np.nan
    if "gross_daily_ret_bps" in pnl_df.columns:
        gross_daily_ret = pnl_df["gross_daily_ret_bps"].astype(float) / 10000.0
        gross_total_return = float(((1.0 + gross_daily_ret).prod() - 1.0) * 100.0)

    daily_vol_bps = float(ret_bps.std()) if len(ret_bps) > 1 else np.nan
    sharpe = float((ret_bps.mean() / daily_vol_bps) * np.sqrt(252)) if daily_vol_bps and daily_vol_bps > 0 else np.nan

    n_reviews = int(len(review_df))
    n_exec_reviews = int(review_df["executed"].fillna(False).astype(bool).sum()) if "executed" in review_df.columns else 0

    summary = {
        "variant_name": variant_name,
        "horizon": horizon_name,
        "total_return_pct": float((cumulative.iloc[-1] - 1.0) * 100.0) if len(cumulative) else np.nan,
        "gross_total_return_pct": gross_total_return,
        "cost_drag_pct": float(gross_total_return - ((cumulative.iloc[-1] - 1.0) * 100.0)) if np.isfinite(gross_total_return) and len(cumulative) else np.nan,
        "cagr_pct": safe_cagr(cumulative),
        "sharpe": sharpe,
        "daily_vol_bps": daily_vol_bps,
        "max_drawdown_pct": compute_max_drawdown_pct(cumulative),
        "total_cost_bps": float(pnl_df["cost_bps_applied"].sum()) if "cost_bps_applied" in pnl_df.columns else np.nan,
        "avg_turnover": float(review_df["turnover_total"].mean()) if "turnover_total" in review_df.columns and not review_df.empty else np.nan,
        "avg_replacements": float(review_df["replacements"].mean()) if "replacements" in review_df.columns and not review_df.empty else np.nan,
        "n_days": int(len(pnl_df)),
        "n_reviews": n_reviews,
        "n_exec_reviews": n_exec_reviews,
        "exec_review_rate_pct": float(n_exec_reviews / n_reviews * 100.0) if n_reviews else np.nan,
        "mean_decile_spread_pct": mean_decile_spread(decile_df),
        "review_mode": review_policy.get("mode"),
        "review_static_days": review_policy.get("static_review_days"),
        "review_normal_days": regime_intervals.get("normal"),
        "review_grind_up_days": regime_intervals.get("grind_up"),
        "review_slow_bleed_days": regime_intervals.get("slow_bleed"),
        "review_squeeze_days": regime_intervals.get("squeeze"),
        "review_crisis_days": regime_intervals.get("crisis"),
        "review_unknown_days": regime_intervals.get("unknown"),
        "hysteresis_mode": hysteresis_policy.get("mode"),
        "long_retain_buffer": hysteresis_policy.get("long_retain_buffer"),
        "short_retain_buffer": hysteresis_policy.get("short_retain_buffer"),
        "replacement_gap": hysteresis_policy.get("replacement_gap"),
        "top_n": manifest.get("top_n"),
        "bottom_n": manifest.get("bottom_n"),
        "candidate_pool_mode": candidate_pool.get("mode"),
        "direction_threshold": candidate_pool.get("direction_threshold"),
        "top_n_magnitude_per_side": candidate_pool.get("top_n_magnitude_per_side"),
        "weighting_mode": manifest.get("weighting_mode"),
        "score_dir": str(score_dir),
    }

    return summary, pd.Series(daily_ret.values, index=pnl_df["date"], name=variant_name)


def plot_total_return(summary_df: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary_df.sort_values("total_return_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(14, max(5, 0.5 * len(ordered))))
    colors = ["#2ca02c" if val >= 0 else "#d62728" for val in ordered["total_return_pct"]]
    ax.barh(ordered["display_label"], ordered["total_return_pct"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Direction Experiments — Total Return")
    ax.set_xlabel("Total Return (%)")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(output_dir / "leaderboard_total_return.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_risk_return(summary_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    sizes = np.clip(summary_df["total_cost_bps"].fillna(0).values / 50.0, 60, 500)
    scatter = ax.scatter(
        summary_df["daily_vol_bps"],
        summary_df["total_return_pct"],
        s=sizes,
        c=summary_df["sharpe"],
        cmap="RdYlGn",
        edgecolor="black",
        alpha=0.85,
    )
    for _, row in select_annotation_rows(summary_df).iterrows():
        ax.annotate(
            row["plot_id"],
            (row["daily_vol_bps"], row["total_return_pct"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
        )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Direction Experiments — Return vs Volatility")
    ax.set_xlabel("Daily Volatility (bps)")
    ax.set_ylabel("Total Return (%)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe")
    plt.tight_layout()
    fig.savefig(output_dir / "risk_return_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(corr_df: pd.DataFrame, output_dir: Path) -> None:
    if corr_df.empty:
        return
    annotate_values = len(corr_df) <= 12
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(corr_df)), max(6, 0.7 * len(corr_df))))
    sns.heatmap(corr_df, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=annotate_values, fmt=".2f", ax=ax)
    ax.set_title("Direction Experiments — Daily Return Correlation")
    plt.tight_layout()
    fig.savefig(output_dir / "return_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_vs_return(summary_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(summary_df["total_cost_bps"], summary_df["total_return_pct"], s=140, color="#1f77b4", alpha=0.85)
    for _, row in select_annotation_rows(summary_df).iterrows():
        ax.annotate(
            row["plot_id"],
            (row["total_cost_bps"], row["total_return_pct"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
        )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Direction Experiments — Cost Drag vs Total Return")
    ax.set_xlabel("Total Transaction Costs (bps)")
    ax.set_ylabel("Total Return (%)")
    plt.tight_layout()
    fig.savefig(output_dir / "cost_vs_return.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_horizons = parse_horizons_arg(args.horizons, args.horizon) if args.horizons is not None else None
    aligned_returns = pd.DataFrame()
    corr_df = pd.DataFrame()

    if args.sweep_summary is not None:
        summary_df = load_summary_from_sweep_csv(args.sweep_summary, args.include, args.exclude)
        if summary_df.empty:
            print("No matching rows found in sweep summary.")
            return
        if requested_horizons is not None:
            requested_horizon_tags = {f"{horizon}d_horizon" for horizon in requested_horizons}
            summary_df = summary_df[summary_df["horizon"].isin(requested_horizon_tags)].copy()
            if summary_df.empty:
                print("No matching horizons found in sweep summary.")
                return
        active_horizons = sorted(summary_df["horizon"].astype(str).unique().tolist())
    else:
        horizons = requested_horizons or [args.horizon]
        score_dirs = discover_direction_runs(
            args.backtest_dir,
            horizons,
            args.include,
            args.exclude,
            args.include_llm,
        )
        if not score_dirs:
            print("No matching direction_score runs found.")
            return

        summary_rows: list[dict[str, object]] = []
        return_series: list[pd.Series] = []
        for score_dir in score_dirs:
            summary, daily_ret = summarize_run(score_dir)
            summary_rows.append(summary)
            return_series.append(daily_ret)

        summary_df = pd.DataFrame(summary_rows)
        aligned_returns = pd.concat(return_series, axis=1).sort_index()
        aligned_returns.to_csv(output_dir / "aligned_daily_returns.csv", index_label="date")
        corr_df = aligned_returns.corr(min_periods=max(30, len(aligned_returns) // 10))
        corr_df.to_csv(output_dir / "return_correlation_matrix.csv", index=True)
        active_horizons = [f"{horizon}d_horizon" for horizon in horizons]

    summary_df = summary_df.sort_values(["total_return_pct", "sharpe"], ascending=[False, False])
    summary_df = attach_plot_metadata(summary_df)
    summary_df.to_csv(output_dir / "direction_experiment_summary.csv", index=False)
    label_key_columns = [
        "plot_id",
        "display_label",
        "variant_name",
        "total_return_pct",
        "sharpe",
        "daily_vol_bps",
        "total_cost_bps",
    ]
    summary_df[[column for column in label_key_columns if column in summary_df.columns]].to_csv(
        output_dir / "direction_experiment_label_key.csv",
        index=False,
    )

    plot_total_return(summary_df, output_dir)
    if "daily_vol_bps" in summary_df.columns and summary_df["daily_vol_bps"].notna().any():
        plot_risk_return(summary_df, output_dir)
    if not corr_df.empty:
        plot_corr_df = corr_df.rename(
            index=summary_df.set_index("variant_name")["plot_id"].to_dict(),
            columns=summary_df.set_index("variant_name")["plot_id"].to_dict(),
        )
        plot_correlation_heatmap(plot_corr_df, output_dir)
    if "total_cost_bps" in summary_df.columns and summary_df["total_cost_bps"].notna().any():
        plot_cost_vs_return(summary_df, output_dir)

    if len(active_horizons) > 1:
        cross_horizon_df = build_cross_horizon_leaderboard(summary_df)
        cross_horizon_df.to_csv(output_dir / "cross_horizon_strategy_leaderboard.csv", index=False)
        plot_cross_horizon_heatmap(cross_horizon_df, output_dir)

    print(f"\nSaved summary to: {output_dir / 'direction_experiment_summary.csv'}")
    print(f"Saved label key to: {output_dir / 'direction_experiment_label_key.csv'}")
    if not corr_df.empty:
        print(f"Saved correlations to: {output_dir / 'return_correlation_matrix.csv'}")
    if len(active_horizons) > 1:
        print(f"Saved cross-horizon leaderboard to: {output_dir / 'cross_horizon_strategy_leaderboard.csv'}")
    print("\nTop runs by total return:")
    for _, row in summary_df.head(10).iterrows():
        print(
            f"   {row['variant_name']:<70s} "
            f"ret={row['total_return_pct']:>8.2f}% "
            f"sharpe={row['sharpe']:>7.3f} "
            f"cost={row['total_cost_bps']:>9.1f}bps"
        )


if __name__ == "__main__":
    main()
