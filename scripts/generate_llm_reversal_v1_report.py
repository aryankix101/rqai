#!/usr/bin/env python3
"""Generate a standalone HTML report for the LLM reversal v1 experiment."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("backtest_results/llm_reversal_signal_v1_2009_2014")
DEFAULT_OUTPUT = DEFAULT_ROOT / "llm_reversal_v1_report.html"

RUNS = [
    ("baseline_universe", "Baseline-Universe"),
    ("baseline_pool", "Baseline-Pool"),
    ("llm_pool_v1", "LLM-Pool v1"),
]

RETURN_OPTIONS = [
    {
        "key": "ls_c2c_0",
        "label": "Close-to-close, 0 bps cost",
        "scorecard": ("C2C", 0.0),
        "yearly": "LS_C2C_Gross",
    },
    {
        "key": "ls_c2c_2p5",
        "label": "Close-to-close, 2.5 bps cost",
        "scorecard": ("C2C", 2.5),
        "yearly": "LS_C2C_2.5bps",
    },
    {
        "key": "ls_c2c_5",
        "label": "Close-to-close, 5 bps cost",
        "scorecard": ("C2C", 5.0),
        "yearly": "LS_C2C_5bps",
    },
    {
        "key": "ls_o2o_0",
        "label": "Open-to-open, 0 bps cost",
        "scorecard": ("O2O", 0.0),
        "yearly": "LS_O2O_Gross",
    },
    {
        "key": "ls_o2o_2p5",
        "label": "Open-to-open, 2.5 bps cost",
        "scorecard": ("O2O", 2.5),
        "yearly": "LS_O2O_2.5bps",
    },
    {
        "key": "ls_o2o_5",
        "label": "Open-to-open, 5 bps cost",
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


def normalize_scorecard(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
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


def read_growth_series(root: Path, run_dir: str, label: str) -> dict[str, Any]:
    rows = read_csv(root / run_dir / "weekly_returns.csv")
    points: list[dict[str, Any]] = []
    growth_by_key = {option["key"]: 1.0 for option in RETURN_OPTIONS}

    for row in rows:
        point: dict[str, Any] = {"date": row["date"]}
        for option in RETURN_OPTIONS:
            key = str(option["key"])
            ret = to_float(row.get(key))
            growth_by_key[key] *= 1.0 + ret
            point[key] = {
                "ret": ret,
                "growth": growth_by_key[key],
                "returnPct": (growth_by_key[key] - 1.0) * 100.0,
            }
        points.append(point)

    return {"runDir": run_dir, "label": label, "points": points}


def read_yearly_returns(root: Path, run_dir: str, label: str) -> dict[str, Any]:
    rows = read_csv(root / run_dir / "yearly_returns.csv")
    yearly: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {"year": row["year"]}
        for option in RETURN_OPTIONS:
            item[str(option["key"])] = to_float(row.get(str(option["yearly"]))) * 100.0
        yearly.append(item)
    return {"runDir": run_dir, "label": label, "years": yearly}


def read_copy_correlation(root: Path) -> dict[str, Any]:
    path = root / "llm_pool_v1" / "score_trail5_correlations.csv"
    if not path.exists():
        return {"count": 0, "maxAbs": None, "avgAbs": None}

    values: list[float] = []
    for row in read_csv(path):
        value = row.get("score_trail5_spearman")
        if value not in (None, ""):
            values.append(abs(float(value)))

    if not values:
        return {"count": 0, "maxAbs": None, "avgAbs": None}
    return {
        "count": len(values),
        "maxAbs": max(values),
        "avgAbs": sum(values) / len(values),
    }


def build_payload(root: Path) -> dict[str, Any]:
    scorecard = normalize_scorecard(read_csv(root / "comparison_scorecard.csv"))
    return {
        "returnOptions": RETURN_OPTIONS,
        "scorecard": scorecard,
        "series": [read_growth_series(root, run_dir, label) for run_dir, label in RUNS],
        "yearly": [read_yearly_returns(root, run_dir, label) for run_dir, label in RUNS],
        "success": read_json(root / "success_evaluation.json"),
        "copyCorrelation": read_copy_correlation(root),
    }


def html_page(payload: dict[str, Any], root: Path) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = "LLM Reversal Signal v1 Report"
    root_label = escape(str(root))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18212d;
      --muted: #627084;
      --line: #d8dee7;
      --soft: #f5f7fa;
      --good: #16705f;
      --warn: #a65f17;
      --bad: #b23a48;
      --blue: #345995;
      --bg: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 22px 42px;
    }}
    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 7px;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.05;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
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
      gap: 10px;
      flex-wrap: wrap;
    }}
    label {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }}
    select {{
      min-width: 160px;
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
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      padding: 12px 13px;
      min-height: 82px;
    }}
    .card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 7px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .card strong {{
      display: block;
      font-size: 23px;
      line-height: 1.1;
      letter-spacing: 0;
    }}
    .card small {{
      display: block;
      color: var(--muted);
      margin-top: 6px;
      font-size: 12px;
    }}
    .status-good strong {{ color: var(--good); }}
    .status-warn strong {{ color: var(--warn); }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(320px, .9fr);
      gap: 14px;
      margin: 14px 0;
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
      gap: 14px;
      align-items: baseline;
      margin-bottom: 8px;
    }}
    svg {{
      width: 100%;
      display: block;
      overflow: visible;
    }}
    #growthChart {{ height: 430px; }}
    #yearChart {{ height: 330px; }}
    .axis {{ fill: var(--muted); font-size: 11px; }}
    .gridline {{ stroke: #e9edf3; stroke-width: 1; }}
    .zero {{ stroke: #9aa8ba; stroke-width: 1; }}
    .path {{ fill: none; stroke-width: 2.6; opacity: .78; }}
    .path.llm {{ stroke-width: 3.5; opacity: 1; }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
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
      min-width: 820px;
      border-collapse: collapse;
      font-size: 13px;
    }}
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
    .diagnostics {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 270px;
      gap: 14px;
      align-items: start;
      margin-top: 14px;
    }}
    .diag-list {{
      display: grid;
      gap: 10px;
    }}
    .diag-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 9px;
      color: var(--muted);
      font-size: 13px;
    }}
    .diag-row strong {{
      color: var(--ink);
      font-size: 14px;
    }}
    .hist {{
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
    }}
    .note {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 900px) {{
      header, .grid, .diagnostics {{ display: block; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .panel {{ margin-top: 12px; }}
      #growthChart {{ height: 360px; }}
      #yearChart {{ height: 280px; }}
      .hist {{ margin-top: 12px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>{title}</h1>
        <p>Visual readout for Baseline-Universe, Baseline-Pool, and LLM-Pool v1. Source folder: <code>{root_label}</code>.</p>
      </div>
      <div class="controls">
        <label for="metricSelect">Execution/cost view</label>
        <select id="metricSelect"></select>
      </div>
    </header>

    <section class="cards" id="summaryCards" aria-label="Summary metrics"></section>

    <section class="grid">
      <div class="panel">
        <div class="panel-head">
          <div>
            <h2>Growth of $1</h2>
            <p id="growthSubtitle"></p>
          </div>
        </div>
        <svg id="growthChart" role="img" aria-label="Growth of one dollar comparison"></svg>
        <div id="growthLegend" class="legend"></div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <div>
            <h2>Yearly Returns</h2>
            <p id="yearSubtitle"></p>
          </div>
        </div>
        <svg id="yearChart" role="img" aria-label="Yearly returns comparison"></svg>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <div>
          <h2>Scorecard</h2>
          <p id="tableSubtitle"></p>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Sharpe</th>
              <th>Total Ret</th>
              <th>CAGR</th>
              <th>Ann Vol</th>
              <th>Max DD</th>
              <th>Hit Rate</th>
              <th>IC Mean</th>
              <th>IC t-stat</th>
              <th>Turnover</th>
            </tr>
          </thead>
          <tbody id="scorecardBody"></tbody>
        </table>
      </div>
    </section>

    <section class="panel diagnostics">
      <div>
        <h2>LLM Diagnostics</h2>
        <div class="diag-list" id="diagnostics"></div>
        <p class="note">The copy-like correlation count is useful because a high score-vs-Trail5 correlation means the LLM may be mostly restating the deterministic pool sort.</p>
      </div>
      <img class="hist" src="llm_pool_v1/llm_score_histograms.png" alt="LLM score histogram by side">
    </section>

    <p class="note">Generated {generated_at}. Open <code>comparison_scorecard.csv</code> for the underlying table, or the per-run <code>weekly_returns.csv</code> files for the return paths.</p>
  </main>

  <script>
    const DATA = {payload_json};
    const COLORS = {{
      "Baseline-Universe": "#345995",
      "Baseline-Pool": "#b66b18",
      "LLM-Pool v1": "#16705f"
    }};

    const metricSelect = document.getElementById("metricSelect");
    const summaryCards = document.getElementById("summaryCards");
    const growthSubtitle = document.getElementById("growthSubtitle");
    const yearSubtitle = document.getElementById("yearSubtitle");
    const tableSubtitle = document.getElementById("tableSubtitle");
    const scorecardBody = document.getElementById("scorecardBody");
    const diagnostics = document.getElementById("diagnostics");

    function pct(value, digits = 1) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
      const sign = value > 0 ? "+" : "";
      return `${{sign}}${{value.toFixed(digits)}}%`;
    }}

    function num(value, digits = 2) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
      return value.toFixed(digits);
    }}

    function optionByKey(key) {{
      return DATA.returnOptions.find((option) => option.key === key);
    }}

    function scorecardRowsFor(key) {{
      const option = optionByKey(key);
      return DATA.scorecard.filter((row) =>
        row.execution_model === option.scorecard[0] && Number(row.cost_bps) === Number(option.scorecard[1])
      );
    }}

    function rowFor(rows, run) {{
      return rows.find((row) => row.run === run);
    }}

    function renderControls() {{
      DATA.returnOptions.forEach((option) => {{
        const el = document.createElement("option");
        el.value = option.key;
        el.textContent = option.label;
        metricSelect.appendChild(el);
      }});
      metricSelect.value = "ls_c2c_5";
      metricSelect.addEventListener("change", () => renderAll());
    }}

    function renderSummary(key) {{
      const rows = scorecardRowsFor(key);
      const llm = rowFor(rows, "LLM-Pool v1");
      const universe = rowFor(rows, "Baseline-Universe");
      const pool = rowFor(rows, "Baseline-Pool");
      const success = DATA.success || {{}};
      const winClass = success.is_win ? "status-good" : "status-warn";
      const failClass = success.is_stop_failure ? "status-warn" : "status-good";

      const cards = [
        {{
          label: "LLM Sharpe",
          value: llm ? num(llm.sharpe, 2) : "n/a",
          detail: llm ? `${{pct(llm.total_return_pct)}} total return` : "",
          klass: ""
        }},
        {{
          label: "Baseline-Universe",
          value: universe ? num(universe.sharpe, 2) : "n/a",
          detail: "same selected view",
          klass: ""
        }},
        {{
          label: "Baseline-Pool",
          value: pool ? num(pool.sharpe, 2) : "n/a",
          detail: "same selected view",
          klass: ""
        }},
        {{
          label: "Protocol Win Rule",
          value: success.is_win ? "Pass" : "No",
          detail: "LLM > universe at C2C 5 bps",
          klass: winClass
        }},
        {{
          label: "Stop Rule",
          value: success.is_stop_failure ? "Hit" : "Clear",
          detail: "gross LLM not >0.10 below pool",
          klass: failClass
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

    function svgEl(tag, attrs = {{}}, text = "") {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
      if (text) el.textContent = text;
      return el;
    }}

    function renderGrowthChart(key) {{
      const svg = document.getElementById("growthChart");
      svg.innerHTML = "";
      const width = 980;
      const height = 430;
      const margin = {{ top: 18, right: 42, bottom: 42, left: 62 }};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);

      const all = [];
      DATA.series.forEach((run) => {{
        run.points.forEach((point) => {{
          all.push({{ date: new Date(point.date), value: point[key].growth, run: run.label }});
        }});
      }});
      const minX = Math.min(...all.map((p) => p.date.getTime()));
      const maxX = Math.max(...all.map((p) => p.date.getTime()));
      const maxY = Math.max(...all.map((p) => p.value), 1);
      const minY = Math.min(...all.map((p) => p.value), 1);
      const yPad = Math.max((maxY - minY) * 0.08, 0.03);
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

      DATA.series.forEach((run) => {{
        const d = run.points.map((point, index) => {{
          const command = index === 0 ? "M" : "L";
          return `${{command}}${{x(new Date(point.date)).toFixed(1)}},${{y(point[key].growth).toFixed(1)}}`;
        }}).join(" ");
        svg.appendChild(svgEl("path", {{
          d,
          class: `path ${{run.label === "LLM-Pool v1" ? "llm" : ""}}`,
          stroke: COLORS[run.label]
        }}));
      }});

      const lastPoints = DATA.series.map((run) => {{
        const last = run.points[run.points.length - 1];
        return {{ label: run.label, value: last[key].growth, returnPct: last[key].returnPct }};
      }});
      document.getElementById("growthLegend").innerHTML = lastPoints.map((item) => `
        <div class="legend-item">
          <span class="swatch" style="background:${{COLORS[item.label]}}"></span>
          <strong>${{item.label}}</strong>
          <span>${{item.value.toFixed(2)}}x (${{pct(item.returnPct)}})</span>
        </div>
      `).join("");
    }}

    function renderYearChart(key) {{
      const svg = document.getElementById("yearChart");
      svg.innerHTML = "";
      const width = 600;
      const height = 330;
      const margin = {{ top: 18, right: 14, bottom: 42, left: 48 }};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);

      const years = DATA.yearly[0].years.map((row) => row.year);
      const values = [];
      DATA.yearly.forEach((run) => run.years.forEach((row) => values.push(row[key])));
      const minY = Math.min(...values, 0);
      const maxY = Math.max(...values, 0);
      const pad = Math.max((maxY - minY) * 0.12, 2);
      const y0 = minY - pad;
      const y1 = maxY + pad;
      const y = (value) => margin.top + (1 - ((value - y0) / (y1 - y0))) * innerH;
      const zeroY = y(0);
      const yearW = innerW / years.length;
      const barW = Math.min(24, (yearW - 12) / DATA.yearly.length);

      for (let i = 0; i <= 4; i++) {{
        const yy = margin.top + (innerH / 4) * i;
        const value = y1 - ((y1 - y0) / 4) * i;
        svg.appendChild(svgEl("line", {{ x1: margin.left, x2: width - margin.right, y1: yy, y2: yy, class: "gridline" }}));
        svg.appendChild(svgEl("text", {{ x: margin.left - 8, y: yy + 4, "text-anchor": "end", class: "axis" }}, `${{value.toFixed(0)}}%`));
      }}
      svg.appendChild(svgEl("line", {{ x1: margin.left, x2: width - margin.right, y1: zeroY, y2: zeroY, class: "zero" }}));

      years.forEach((year, yearIndex) => {{
        const groupX = margin.left + yearIndex * yearW;
        svg.appendChild(svgEl("text", {{ x: groupX + yearW / 2, y: height - 15, "text-anchor": "middle", class: "axis" }}, String(year)));
        DATA.yearly.forEach((run, runIndex) => {{
          const row = run.years.find((item) => item.year === year);
          const value = row ? row[key] : 0;
          const xPos = groupX + (yearW / 2) - (barW * DATA.yearly.length / 2) + runIndex * barW;
          const yPos = value >= 0 ? y(value) : zeroY;
          const barH = Math.abs(y(value) - zeroY);
          svg.appendChild(svgEl("rect", {{
            x: xPos,
            y: yPos,
            width: barW - 2,
            height: Math.max(barH, 1),
            fill: COLORS[run.label],
            opacity: run.label === "LLM-Pool v1" ? "0.95" : "0.72"
          }}));
        }});
      }});
    }}

    function renderScorecard(key) {{
      const option = optionByKey(key);
      const rows = scorecardRowsFor(key);
      tableSubtitle.textContent = `${{option.label}}. The success rule uses C2C 5 bps.`;
      scorecardBody.innerHTML = rows.map((row) => `
        <tr>
          <td>${{row.run}}</td>
          <td>${{num(row.sharpe, 2)}}</td>
          <td>${{pct(row.total_return_pct)}}</td>
          <td>${{pct(row.cagr_pct)}}</td>
          <td>${{pct(row.annual_vol_pct)}}</td>
          <td>${{pct(row.max_drawdown_pct)}}</td>
          <td>${{pct(row.hit_rate_pct)}}</td>
          <td>${{num(row.ic_mean, 4)}}</td>
          <td>${{num(row.ic_tstat, 2)}}</td>
          <td>${{num(row.avg_turnover_total, 2)}}</td>
        </tr>
      `).join("");
    }}

    function renderDiagnostics() {{
      const success = DATA.success || {{}};
      const corr = DATA.copyCorrelation || {{}};
      const rows = [
        ["LLM gross Sharpe", num(success.llm_pool_gross_sharpe, 2)],
        ["Baseline-Pool gross Sharpe", num(success.baseline_pool_gross_sharpe, 2)],
        ["LLM 5 bps Sharpe", num(success.llm_pool_5bps_sharpe, 2)],
        ["Baseline-Universe 5 bps Sharpe", num(success.baseline_universe_5bps_sharpe, 2)],
        ["Copy-like correlation periods", success.copy_like_correlation_periods ?? "n/a"],
        ["Max abs score-vs-Trail5 corr", corr.maxAbs === null || corr.maxAbs === undefined ? "n/a" : num(corr.maxAbs, 3)]
      ];
      diagnostics.innerHTML = rows.map(([label, value]) => `
        <div class="diag-row"><span>${{label}}</span><strong>${{value}}</strong></div>
      `).join("");
    }}

    function renderAll() {{
      const key = metricSelect.value;
      const option = optionByKey(key);
      growthSubtitle.textContent = `${{option.label}} long-short path, weekly rebalance.`;
      yearSubtitle.textContent = `${{option.label}} calendar-year return.`;
      renderSummary(key);
      renderGrowthChart(key);
      renderYearChart(key);
      renderScorecard(key);
      renderDiagnostics();
    }}

    renderControls();
    renderAll();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_payload(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_page(payload, args.root), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
