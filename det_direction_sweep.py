#!/usr/bin/env python3
"""Batch runner for deterministic direction_score experiments."""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).parent
BACKTEST_DIR = BASE_DIR / "backtest_results" / "factor_backtest"
SWEEP_DIR = BASE_DIR / "backtest_results" / "det_sweeps"


def parse_int_list(value: str) -> list[int]:
    values = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return values


def parse_float_list(value: str) -> list[float]:
    values = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float")
    return values


def parse_str_list(value: str) -> list[str]:
    values = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one string")
    return values


def default_python_bin() -> str:
    venv_python = BASE_DIR / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def fmt_tag(value: float | int | str) -> str:
    if isinstance(value, str):
        return value.replace("-", "_").replace(".", "p")
    if isinstance(value, int):
        return str(value)
    return str(value).replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class SweepSpec:
    forward_days: int
    review_mode: str
    static_review_days: int | None
    hysteresis_mode: str
    weighting_mode: str
    top_n: int
    bottom_n: int
    long_retain_buffer: int
    short_retain_buffer: int
    replacement_gap: int
    review_normal_days: int
    review_grind_up_days: int
    review_slow_bleed_days: int
    review_squeeze_days: int
    review_crisis_days: int
    review_unknown_days: int
    candidate_pool_mode: str
    direction_threshold: float
    top_n_magnitude: int


def build_variant_label(prefix: str, spec: SweepSpec) -> str:
    parts = [
        prefix,
        f"fd_{spec.forward_days}",
        f"review_{spec.review_mode}",
        f"hyst_{spec.hysteresis_mode}",
        f"w_{spec.weighting_mode}",
        f"top_{spec.top_n}",
        f"bot_{spec.bottom_n}",
    ]
    if spec.review_mode == "static":
        parts.append(f"rd_{spec.static_review_days}")
    else:
        parts.extend(
            [
                f"n_{spec.review_normal_days}",
                f"g_{spec.review_grind_up_days}",
                f"sb_{spec.review_slow_bleed_days}",
                f"sq_{spec.review_squeeze_days}",
                f"cr_{spec.review_crisis_days}",
                f"u_{spec.review_unknown_days}",
            ]
        )
    if spec.hysteresis_mode != "none":
        parts.extend(
            [
                f"lb_{spec.long_retain_buffer}",
                f"sbf_{spec.short_retain_buffer}",
                f"gap_{spec.replacement_gap}",
            ]
        )
    parts.append(f"pool_{spec.candidate_pool_mode}")
    if spec.candidate_pool_mode == "llm_mirror":
        parts.append(f"m_{fmt_tag(spec.top_n_magnitude)}")
        parts.append(f"d_{fmt_tag(spec.direction_threshold)}")
    return "__".join(parts)


def expand_specs(args: argparse.Namespace) -> list[SweepSpec]:
    specs: list[SweepSpec] = []
    for forward_days in args.forward_days_list:
        adaptive_grid = itertools.product(
            args.review_normal_days,
            args.review_grind_up_days,
            args.review_slow_bleed_days,
            args.review_squeeze_days,
            args.review_crisis_days,
            args.review_unknown_days,
        )
        adaptive_configs = list(adaptive_grid)
        static_days = args.static_review_days or [forward_days]

        for review_mode in args.review_modes:
            review_variants: list[tuple[int | None, int, int, int, int, int, int]]
            if review_mode == "static":
                review_variants = [(day, day, day, day, day, day, day) for day in static_days]
            else:
                review_variants = [(None, *cfg) for cfg in adaptive_configs]

            for (
                static_review_days,
                review_normal_days,
                review_grind_up_days,
                review_slow_bleed_days,
                review_squeeze_days,
                review_crisis_days,
                review_unknown_days,
            ) in review_variants:
                for (
                    hysteresis_mode,
                    weighting_mode,
                    top_n,
                    long_retain_buffer,
                    short_retain_buffer,
                    replacement_gap,
                    candidate_pool_mode,
                ) in itertools.product(
                    args.hysteresis_modes,
                    args.weighting_modes,
                    args.top_ns,
                    args.long_retain_buffers,
                    args.short_retain_buffers,
                    args.replacement_gaps,
                    args.candidate_pool_modes,
                ):
                    bottom_n = top_n
                    if hysteresis_mode == "none":
                        long_retain_buffer = args.long_retain_buffers[0]
                        short_retain_buffer = args.short_retain_buffers[0]
                        replacement_gap = args.replacement_gaps[0]
                    if candidate_pool_mode == "raw":
                        specs.append(
                            SweepSpec(
                                forward_days=forward_days,
                                review_mode=review_mode,
                                static_review_days=static_review_days,
                                hysteresis_mode=hysteresis_mode,
                                weighting_mode=weighting_mode,
                                top_n=top_n,
                                bottom_n=bottom_n,
                                long_retain_buffer=long_retain_buffer,
                                short_retain_buffer=short_retain_buffer,
                                replacement_gap=replacement_gap,
                                review_normal_days=review_normal_days,
                                review_grind_up_days=review_grind_up_days,
                                review_slow_bleed_days=review_slow_bleed_days,
                                review_squeeze_days=review_squeeze_days,
                                review_crisis_days=review_crisis_days,
                                review_unknown_days=review_unknown_days,
                                candidate_pool_mode=candidate_pool_mode,
                                direction_threshold=args.direction_thresholds[0],
                                top_n_magnitude=args.top_n_magnitudes[0],
                            )
                        )
                        continue

                    for direction_threshold, top_n_magnitude in itertools.product(
                        args.direction_thresholds,
                        args.top_n_magnitudes,
                    ):
                        specs.append(
                            SweepSpec(
                                forward_days=forward_days,
                                review_mode=review_mode,
                                static_review_days=static_review_days,
                                hysteresis_mode=hysteresis_mode,
                                weighting_mode=weighting_mode,
                                top_n=top_n,
                                bottom_n=bottom_n,
                                long_retain_buffer=long_retain_buffer,
                                short_retain_buffer=short_retain_buffer,
                                replacement_gap=replacement_gap,
                                review_normal_days=review_normal_days,
                                review_grind_up_days=review_grind_up_days,
                                review_slow_bleed_days=review_slow_bleed_days,
                                review_squeeze_days=review_squeeze_days,
                                review_crisis_days=review_crisis_days,
                                review_unknown_days=review_unknown_days,
                                candidate_pool_mode=candidate_pool_mode,
                                direction_threshold=direction_threshold,
                                top_n_magnitude=top_n_magnitude,
                            )
                        )

    deduped = list(dict.fromkeys(specs))
    if args.max_runs is not None:
        deduped = deduped[: args.max_runs]
    return deduped


def summarize_direction_run(variant_label: str, forward_days: int) -> dict[str, object]:
    score_dir = BACKTEST_DIR / variant_label / f"{forward_days}d_horizon" / "direction_score"
    pnl_df = pd.read_csv(score_dir / "factor_pnl.csv")
    review_df = pd.read_csv(score_dir / "review_log.csv") if (score_dir / "review_log.csv").exists() else pd.DataFrame()

    daily_ret = pnl_df["daily_ret_bps"] / 10000.0 if not pnl_df.empty else pd.Series(dtype=float)
    total_ret = float(((1.0 + daily_ret).prod() - 1.0) * 100.0) if len(daily_ret) > 0 else 0.0
    daily_vol = float(pnl_df["daily_ret_bps"].std()) if len(pnl_df) > 1 else 0.0
    sharpe = float((pnl_df["daily_ret_bps"].mean() / daily_vol) * (252 ** 0.5)) if daily_vol > 0 else 0.0

    return {
        "variant_label": variant_label,
        "forward_days": forward_days,
        "status": "ok",
        "total_return_pct": total_ret,
        "sharpe": sharpe,
        "total_cost_bps": float(pnl_df["cost_bps_applied"].sum()) if "cost_bps_applied" in pnl_df.columns else 0.0,
        "avg_turnover": float(review_df["turnover_total"].mean()) if "turnover_total" in review_df.columns and not review_df.empty else None,
        "score_dir": str(score_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep deterministic direction_score backtest configs.")
    parser.add_argument("--forward-days", type=int, default=10)
    parser.add_argument("--forward-days-list", type=parse_int_list, default=None)
    parser.add_argument("--start", type=str, default="2000-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--review-modes", type=parse_str_list, default=parse_str_list("adaptive"))
    parser.add_argument("--static-review-days", type=parse_int_list, default=None)
    parser.add_argument("--hysteresis-modes", type=parse_str_list, default=parse_str_list("combined"))
    parser.add_argument("--weighting-modes", type=parse_str_list, default=parse_str_list("equal,rank"))
    parser.add_argument("--top-ns", type=parse_int_list, default=parse_int_list("30,60,100,200"))
    parser.add_argument("--long-retain-buffers", type=parse_int_list, default=parse_int_list("20"))
    parser.add_argument("--short-retain-buffers", type=parse_int_list, default=parse_int_list("5"))
    parser.add_argument("--replacement-gaps", type=parse_int_list, default=parse_int_list("5"))
    parser.add_argument("--review-normal-days", type=parse_int_list, default=parse_int_list("20"))
    parser.add_argument("--review-grind-up-days", type=parse_int_list, default=parse_int_list("60"))
    parser.add_argument("--review-slow-bleed-days", type=parse_int_list, default=parse_int_list("20"))
    parser.add_argument("--review-squeeze-days", type=parse_int_list, default=parse_int_list("5"))
    parser.add_argument("--review-crisis-days", type=parse_int_list, default=parse_int_list("5"))
    parser.add_argument("--review-unknown-days", type=parse_int_list, default=parse_int_list("20"))
    parser.add_argument("--candidate-pool-modes", type=parse_str_list, default=parse_str_list("raw,llm_mirror"))
    parser.add_argument("--direction-thresholds", type=parse_float_list, default=parse_float_list("5,8"))
    parser.add_argument("--top-n-magnitudes", type=parse_int_list, default=parse_int_list("60,100,200"))
    parser.add_argument("--variant-prefix", type=str, default="det_dir_sweep")
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--python-bin", type=str, default=default_python_bin())
    parser.add_argument("--scores-dir", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--periodic-static", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()
    if args.forward_days_list is None:
        args.forward_days_list = [args.forward_days]

    horizon_tag = "x".join(f"{fd}d" for fd in args.forward_days_list)
    run_label = args.run_label or f"{args.variant_prefix}_{horizon_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = SWEEP_DIR / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(BASE_DIR / "backtest_results" / ".mplconfig"))

    specs = expand_specs(args)
    summary_rows: list[dict[str, object]] = []

    print(f"\n{'=' * 96}")
    print("DET DIRECTION SWEEP")
    print(f"{'=' * 96}")
    print(f"Run Label: {run_label}")
    print(f"Forward Horizons: {', '.join(f'{fd}d' for fd in args.forward_days_list)}")
    print(f"Date Range: {args.start} -> {args.end}")
    print(f"Python: {args.python_bin}")
    print(f"Total Runs: {len(specs)}")
    print(f"{'=' * 96}\n")

    for idx, spec in enumerate(specs, start=1):
        variant_label = build_variant_label(run_label, spec)
        result_root = BACKTEST_DIR / variant_label / f"{spec.forward_days}d_horizon" / "direction_score"
        print(f"[{idx}/{len(specs)}] {variant_label}")
        started = time.time()

        if args.skip_existing and (result_root / "factor_pnl.csv").exists():
            print("   ↳ Skipping existing result")
        else:
            cmd = [
                args.python_bin,
                "factor_backtest_engine.py",
                str(spec.forward_days),
                args.start,
                args.end,
                "--variant-label",
                variant_label,
                "--score-types",
                "direction_score",
                "--top-n",
                str(spec.top_n),
                "--bottom-n",
                str(spec.bottom_n),
                "--review-mode",
                spec.review_mode,
                "--hysteresis-mode",
                spec.hysteresis_mode,
                "--weighting-mode",
                spec.weighting_mode,
                "--long-retain-buffer",
                str(spec.long_retain_buffer),
                "--short-retain-buffer",
                str(spec.short_retain_buffer),
                "--replacement-gap",
                str(spec.replacement_gap),
                "--candidate-pool-mode",
                spec.candidate_pool_mode,
                "--candidate-top-n-magnitude",
                str(spec.top_n_magnitude),
                "--candidate-direction-threshold",
                str(spec.direction_threshold),
                "--skip-plots",
            ]
            if spec.review_mode == "static":
                cmd += ["--review-days", str(spec.static_review_days)]
                if args.periodic_static:
                    cmd += ["--periodic-static"]
            else:
                cmd += [
                    "--review-normal-days", str(spec.review_normal_days),
                    "--review-grind-up-days", str(spec.review_grind_up_days),
                    "--review-slow-bleed-days", str(spec.review_slow_bleed_days),
                    "--review-squeeze-days", str(spec.review_squeeze_days),
                    "--review-crisis-days", str(spec.review_crisis_days),
                    "--review-unknown-days", str(spec.review_unknown_days),
                ]
            if args.scores_dir:
                cmd += ["--scores-dir", args.scores_dir]
            subprocess.run(cmd, cwd=BASE_DIR, env=env, check=True)

        if not (result_root / "factor_pnl.csv").exists():
            print("   ↳ No factor_pnl.csv produced; marking run as failed")
            row = {
                "variant_label": variant_label,
                "status": "failed",
                "total_return_pct": None,
                "sharpe": None,
                "total_cost_bps": None,
                "avg_turnover": None,
                "score_dir": str(result_root),
            }
        else:
            row = summarize_direction_run(variant_label, spec.forward_days)
        row.update(
            {
                "runtime_sec": time.time() - started,
                "forward_days": spec.forward_days,
                "review_mode": spec.review_mode,
                "static_review_days": spec.static_review_days,
                "hysteresis_mode": spec.hysteresis_mode,
                "weighting_mode": spec.weighting_mode,
                "top_n": spec.top_n,
                "bottom_n": spec.bottom_n,
                "long_retain_buffer": spec.long_retain_buffer,
                "short_retain_buffer": spec.short_retain_buffer,
                "replacement_gap": spec.replacement_gap,
                "candidate_pool_mode": spec.candidate_pool_mode,
                "direction_threshold": spec.direction_threshold,
                "top_n_magnitude": spec.top_n_magnitude,
            }
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(["total_return_pct", "sharpe"], ascending=[False, False])
    summary_path = run_dir / "direction_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    analyzer_root = run_dir / "analysis"
    if summary_df["status"].eq("ok").any():
        successful_forward_days = sorted(summary_df.loc[summary_df["status"] == "ok", "forward_days"].dropna().unique())
        for forward_days in successful_forward_days:
            analyzer_output_dir = analyzer_root / f"{int(forward_days)}d"
            analyzer_cmd = [
                args.python_bin,
                "direction_experiment_analyzer.py",
                "--horizon",
                str(int(forward_days)),
                "--include",
                run_label,
                "--output-dir",
                str(analyzer_output_dir),
            ]
            subprocess.run(analyzer_cmd, cwd=BASE_DIR, env=env, check=True)

        if len(successful_forward_days) > 1:
            combined_output_dir = analyzer_root / "all_horizons"
            combined_cmd = [
                args.python_bin,
                "direction_experiment_analyzer.py",
                "--sweep-summary",
                str(summary_path),
                "--horizons",
                ",".join(str(int(forward_days)) for forward_days in successful_forward_days),
                "--output-dir",
                str(combined_output_dir),
            ]
            subprocess.run(combined_cmd, cwd=BASE_DIR, env=env, check=True)
    else:
        print("No successful runs were produced; skipping analyzer.")

    print(f"\n{'=' * 96}")
    print("DET DIRECTION SWEEP COMPLETE")
    print(f"Summary: {summary_path}")
    print(f"Analyzer output: {analyzer_root}")
    successful = summary_df[summary_df["status"] == "ok"]
    if not successful.empty:
        best = successful.iloc[0]
        print(
            f"Best run: {best['variant_label']} "
            f"ret={best['total_return_pct']:+.2f}% "
            f"sharpe={best['sharpe']:+.3f} "
            f"cost={best['total_cost_bps']:.1f}bps"
        )
    print(f"{'=' * 96}\n")


if __name__ == "__main__":
    main()
