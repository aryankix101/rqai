"""Publish selected sleeve-splice results into the Phase 2 dashboard.

This offline updater reads completed development and OOS analysis artifacts,
then refreshes the matching dashboard sections and validation tracker rows.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEV = ROOT / "backtest_results/v2_sleeve_splice_in_sample_2009_2014/analysis"
OOS = ROOT / "backtest_results/v2_sleeve_splice_oos_2015_2020/analysis"
DASHBOARD = ROOT / "backtest_results/llm_reversal_signal_v2_2009_2014/phase2_validation_dashboard.html"
TRACKER = ROOT / "backtest_results/llm_reversal_signal_v2_2009_2014/phase2_validation_tracker"
START = "<!-- SLEEVE_SPLICE_PHASE2_START -->"
END = "<!-- SLEEVE_SPLICE_PHASE2_END -->"
DEV_ROWS_START = "<!-- SLEEVE_SPLICE_DEV_ROWS_START -->"
DEV_ROWS_END = "<!-- SLEEVE_SPLICE_DEV_ROWS_END -->"
OOS_ROWS_START = "<!-- SLEEVE_SPLICE_OOS_ROWS_START -->"
OOS_ROWS_END = "<!-- SLEEVE_SPLICE_OOS_ROWS_END -->"
LABELS = {
    "Experiment 1 — Original V2 Long + B Short — Two-Call Splice": "Exp. 1 · V2 Long + B Short",
    "Experiment 2 — B Context + Original V2 Range": "Exp. 2 · B Context + V2 Range",
}
SLUGS = {
    "Trail5": "trail5",
    "Experiment 1 — Original V2 Long + B Short — Two-Call Splice": "experiment_1_two_call_splice",
    "Experiment 2 — B Context + Original V2 Range": "experiment_2_b_original_range",
}


def metrics(root: Path, period: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    scorecard = pd.read_csv(root / "experiment_scorecard.csv")
    scorecard = scorecard[
        scorecard["execution_model"].eq("C2C")
        & pd.to_numeric(scorecard["cost_bps"], errors="coerce").fillna(0.0).eq(0.0)
    ].set_index("run")
    weekly = {
        label: pd.read_csv(root / "runs" / slug / "weekly_returns.csv", parse_dates=["date"]).set_index("date")
        for label, slug in SLUGS.items()
    }
    base = weekly["Trail5"]
    rows = []
    for label in LABELS:
        frame = weekly[label]
        paired = pd.DataFrame(index=frame.index.intersection(base.index))
        paired["treatment"] = 0.5 * (
            frame.loc[paired.index, "long_ret_c2c"] - frame.loc[paired.index, "short_ret_c2c"]
        )
        paired["control"] = 0.5 * (
            base.loc[paired.index, "long_ret_c2c"] - base.loc[paired.index, "short_ret_c2c"]
        )
        paired["lift"] = paired["treatment"] - paired["control"]
        long_lift = 0.5 * (
            frame.loc[paired.index, "long_ret_c2c"] - base.loc[paired.index, "long_ret_c2c"]
        )
        short_lift = 0.5 * (
            base.loc[paired.index, "short_ret_c2c"] - frame.loc[paired.index, "short_ret_c2c"]
        )
        yearly = paired["lift"].groupby(paired.index.year).mean()
        row = scorecard.loc[label]
        trail = scorecard.loc["Trail5"]
        rows.append(
            {
                "period": period,
                "experiment": LABELS[label],
                "cagr_pct": float(row["capital_cagr_pct"]),
                "trail5_cagr_pct": float(trail["capital_cagr_pct"]),
                "cagr_lift_pp": float(row["capital_cagr_pct"] - trail["capital_cagr_pct"]),
                "long_lift_pct": float(long_lift.mean() * 52 * 100),
                "short_lift_pct": float(short_lift.mean() * 52 * 100),
                "sharpe": float(row["capital_sharpe"]),
                "sharpe_lift": float(row["capital_sharpe"] - trail["capital_sharpe"]),
                "max_dd_pct": float(row["capital_max_drawdown_pct"]),
                "max_dd_change_pp": float(
                    row["capital_max_drawdown_pct"] - trail["capital_max_drawdown_pct"]
                ),
                "rebalance_win_rate_pct": float(paired["lift"].gt(0).mean() * 100),
                "positive_years": int((yearly > 0).sum()),
                "years": int(len(yearly)),
            }
        )
    return pd.DataFrame(rows), weekly


def pct(value: float, signed: bool = False) -> str:
    return f"{value:+.1f}%" if signed else f"{value:.1f}%"


def table(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples(index=False):
        lift_class = "good" if row.cagr_lift_pp > 0 else "bad"
        rows.append(
            f"<tr><td><strong>{row.period}</strong></td><td>{row.experiment}</td>"
            f"<td>{pct(row.cagr_pct)}</td><td>{pct(row.trail5_cagr_pct)}</td>"
            f"<td class='{lift_class}'><strong>{pct(row.cagr_lift_pp, True)}</strong></td>"
            f"<td>{pct(row.long_lift_pct, True)}</td><td>{pct(row.short_lift_pct, True)}</td>"
            f"<td>{row.sharpe:.2f}</td><td>{row.sharpe_lift:+.2f}</td>"
            f"<td>{pct(row.max_dd_pct)}</td><td>{pct(row.max_dd_change_pp, True)}</td>"
            f"<td>{pct(row.rebalance_win_rate_pct)}</td>"
            f"<td>{row.positive_years}/{row.years}</td></tr>"
        )
    return "".join(rows)


def canonical_table_rows(frame: pd.DataFrame) -> str:
    rows = []
    purposes = {
        "Exp. 1 · V2 Long + B Short": "Does the development-selected two-call sleeve splice retain lift versus matched Trail5?",
        "Exp. 2 · B Context + V2 Range": "Does the development-selected B-context range treatment retain lift versus matched Trail5?",
    }
    for row in frame.itertuples(index=False):
        lift_class = "good" if row.cagr_lift_pp > 0 else "bad"
        dd_class = "good" if row.max_dd_change_pp > 0 else "bad"
        rows.append(
            f"<tr><td><strong>{row.experiment}</strong><small>{purposes[row.experiment]}</small></td>"
            f"<td>{pct(row.cagr_pct)}</td><td>{pct(row.trail5_cagr_pct)}</td>"
            f"<td class='{lift_class}'><strong>{pct(row.cagr_lift_pp, True)}</strong></td>"
            f"<td>{pct(row.long_lift_pct, True)}</td><td>{pct(row.short_lift_pct, True)}</td>"
            f"<td>{row.sharpe:.2f}</td><td>{row.sharpe_lift:+.2f}</td>"
            f"<td>{pct(row.max_dd_pct)}</td><td class='{dd_class}'><strong>{pct(row.max_dd_change_pp, True)}</strong></td>"
            f"<td>{pct(row.rebalance_win_rate_pct)}</td><td>{row.positive_years}/{row.years}</td></tr>"
        )
    return "".join(rows)


def inject_rows(
    html: str,
    heading: str,
    rows: str,
    start_marker: str,
    end_marker: str,
) -> str:
    marked = start_marker + rows + end_marker
    if start_marker in html and end_marker in html:
        return html.split(start_marker, 1)[0] + marked + html.split(end_marker, 1)[1]
    heading_at = html.index(heading)
    tbody_end = html.index("</tbody>", heading_at)
    return html[:tbody_end] + marked + html[tbody_end:]


def equity_svg(weekly: dict[str, pd.DataFrame], title: str) -> str:
    colors = {"Trail5": "#7b8794", list(LABELS)[0]: "#247665", list(LABELS)[1]: "#b5812d"}
    series = []
    for label, frame in weekly.items():
        values = (1 + 0.5 * (frame["long_ret_c2c"] - frame["short_ret_c2c"])).cumprod()
        series.append((label, frame.index, values))
    all_values = pd.concat([item[2] for item in series])
    low, high = float(all_values.min()), float(all_values.max())
    width, height, left, right, top, bottom = 900, 300, 48, 18, 20, 42
    span = max(high - low, 1e-9)
    start = min(item[1].min() for item in series)
    finish = max(item[1].max() for item in series)
    date_span = max((finish - start).days, 1)
    paths = []
    legend = []
    for label, dates, values in series:
        xs = [left + (date - start).days / date_span * (width - left - right) for date in dates]
        ys = [height - bottom - (value - low) / span * (height - top - bottom) for value in values]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        color = colors[label]
        paths.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        legend.append(f"<span><i style='background:{color}'></i>{LABELS.get(label, label)}</span>")
    return (
        f"<div class='comparison-card'><h3>{title}</h3><div class='legend comparison-legend'>"
        + "".join(legend)
        + "</div><svg viewBox='0 0 900 300' role='img' aria-label='Sleeve splice equity comparison'>"
        + f"<text x='8' y='24' fill='#687589' font-size='11'>{high:.2f}x</text>"
        + f"<text x='8' y='{height-bottom}' fill='#687589' font-size='11'>{low:.2f}x</text>"
        + "".join(paths)
        + "</svg></div>"
    )


def main() -> None:
    dev, dev_weekly = metrics(DEV, "Development 2009–2013")
    oos, oos_weekly = metrics(OOS, "OOS 2015–2019")
    combined = pd.concat([dev, oos], ignore_index=True)
    TRACKER.mkdir(parents=True, exist_ok=True)
    combined.to_csv(TRACKER / "sleeve_splice_development_oos_scorecard.csv", index=False)
    html = DASHBOARD.read_text(encoding="utf-8")
    # Remove the earlier standalone section; these belong in the canonical tables.
    if START in html and END in html:
        html = html.split(START, 1)[0] + html.split(END, 1)[1]
    html = inject_rows(
        html,
        "<h2>Out-of-Sample Results: 2015-2019</h2>",
        canonical_table_rows(oos),
        OOS_ROWS_START,
        OOS_ROWS_END,
    )
    html = inject_rows(
        html,
        "<h2>Development Results: 2009-2013</h2>",
        canonical_table_rows(dev),
        DEV_ROWS_START,
        DEV_ROWS_END,
    )
    DASHBOARD.write_text(html, encoding="utf-8")
    manifest = {
        "status": "completed",
        "period": "locked_oos_2015_2019",
        "paid_calls_during_dashboard_update": 0,
        "transaction_cost_analysis_deferred": True,
        "dashboard": str(DASHBOARD.resolve()),
        "combined_scorecard": str(
            (TRACKER / "sleeve_splice_development_oos_scorecard.csv").resolve()
        ),
    }
    (OOS / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    state_path = OOS.parent / "execution_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.update(
        {
            "status": "completed",
            "current_arm": None,
            "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "analysis": manifest,
        }
    )
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    print(combined.to_csv(index=False))


if __name__ == "__main__":
    main()
