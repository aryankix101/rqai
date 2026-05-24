#!/usr/bin/env python3
"""Generate a standalone growth chart for a factor backtest output directory."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path("backtest_results/factor_backtest/composite_scores_llm_signal_2009_2014_5d_horizon")
OUTPUT = ROOT / "growth_path_chart.html"


def read_leaderboard(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_path(score_type: str) -> list[dict[str, float | str]]:
    path = ROOT / score_type / "factor_pnl.csv"
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "date": row["date"],
                    "growth": float(row["cumulative_ret"]),
                    "returnPct": float(row["cumulative_ret_pct"]),
                    "dailyRet": float(row["daily_ret"]),
                }
            )
    return rows


def max_drawdown(points: list[dict[str, float | str]]) -> float:
    peak = 0.0
    worst = 0.0
    for point in points:
        growth = float(point["growth"])
        peak = max(peak, growth)
        if peak:
            worst = min(worst, growth / peak - 1.0)
    return worst * 100.0


def fmt_pct(value: float) -> str:
    return f"{value:,.1f}%"


def main() -> None:
    leaderboard = read_leaderboard(ROOT / "compact_leaderboard.csv")
    series = []
    for row in leaderboard:
        score_type = row["score_type"]
        points = read_path(score_type)
        series.append(
            {
                "scoreType": score_type,
                "label": score_type.replace("_", " ").title(),
                "sharpe": float(row["sharpe"]),
                "icMean": float(row["ic_mean"]),
                "totalReturnPct": float(row["total_return_pct"]),
                "cagrPct": float(row["cagr_pct"]),
                "maxDrawdownPct": max_drawdown(points),
                "points": points,
            }
        )

    top = series[0]
    all_points = [point for item in series for point in item["points"]]
    min_date = min(str(point["date"]) for point in all_points)
    max_date = max(str(point["date"]) for point in all_points)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = json.dumps({"series": series}, separators=(",", ":"))

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Composite Scores LLM Signal Growth Path, 2009-2014</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1d2530;
      --muted: #657282;
      --line: #d8dee7;
      --panel: #f7f9fc;
      --accent: #0f7a70;
      --accent-2: #c24e2f;
      --bg: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 30px 24px 40px;
    }}
    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(26px, 4vw, 42px);
      line-height: 1.05;
      letter-spacing: 0;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 15px;
      margin: 0;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 10px;
      margin: 18px 0 24px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      background: var(--panel);
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .stat strong {{
      font-size: 22px;
      line-height: 1;
    }}
    .chart-wrap {{
      position: relative;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: #fff;
    }}
    svg {{
      width: 100%;
      height: 560px;
      display: block;
      overflow: visible;
    }}
    .grid {{ stroke: #e8edf3; stroke-width: 1; }}
    .axis {{ fill: var(--muted); font-size: 12px; }}
    .path {{ fill: none; stroke-width: 2.2; opacity: .7; }}
    .path.top {{ stroke-width: 3.4; opacity: 1; }}
    .legend {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px 18px;
      margin-top: 16px;
      font-size: 13px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }}
    .swatch {{
      width: 22px;
      height: 3px;
      border-radius: 2px;
      flex: 0 0 auto;
    }}
    .legend b {{ white-space: nowrap; }}
    .legend span {{ color: var(--muted); }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      min-width: 230px;
      max-width: 320px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,.96);
      box-shadow: 0 12px 30px rgba(29,37,48,.13);
      font-size: 12px;
      opacity: 0;
      transform: translate(-50%, -105%);
    }}
    .tooltip .date {{
      font-weight: 700;
      margin-bottom: 6px;
      font-size: 13px;
    }}
    .tip-row {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      padding: 2px 0;
    }}
    .tip-row span:first-child {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .note {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 760px) {{
      main {{ padding: 20px 14px 32px; }}
      header {{ display: block; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      svg {{ height: 460px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Composite Scores LLM Signal Growth Path</h1>
        <p class="subtitle">{min_date} through {max_date}. Growth of $1, net of costs, across score variants.</p>
      </div>
    </header>

    <section class="stats" aria-label="Top score summary">
      <div class="stat"><span>Highlighted Path</span><strong>{top["label"]}</strong></div>
      <div class="stat"><span>Total Return</span><strong>{fmt_pct(top["totalReturnPct"])}</strong></div>
      <div class="stat"><span>CAGR</span><strong>{fmt_pct(top["cagrPct"])}</strong></div>
      <div class="stat"><span>Sharpe</span><strong>{top["sharpe"]:.2f}</strong></div>
      <div class="stat"><span>Max Drawdown</span><strong>{fmt_pct(top["maxDrawdownPct"])}</strong></div>
    </section>

    <section class="chart-wrap">
      <svg id="chart" role="img" aria-label="Growth chart for composite scores LLM signal"></svg>
      <div id="tooltip" class="tooltip"></div>
      <div id="legend" class="legend"></div>
    </section>
    <p class="note">Generated {generated_at} from <code>{ROOT}/&lt;score_type&gt;/factor_pnl.csv</code> and <code>compact_leaderboard.csv</code>.</p>
  </main>

  <script>
    const DATA = {payload};
    const colors = ["#0f7a70", "#3659a7", "#c24e2f", "#8a6f18", "#7a4ca0"];
    const svg = document.getElementById("chart");
    const tooltip = document.getElementById("tooltip");
    const legend = document.getElementById("legend");
    const NS = "http://www.w3.org/2000/svg";

    const parseDate = d => new Date(d + "T00:00:00");
    DATA.series.forEach((s, i) => {{
      s.color = colors[i % colors.length];
      s.points.forEach(p => {{
        p.t = parseDate(p.date).getTime();
        p.growth = Number(p.growth);
        p.returnPct = Number(p.returnPct);
      }});
    }});

    const all = DATA.series.flatMap(s => s.points);
    const minT = Math.min(...all.map(p => p.t));
    const maxT = Math.max(...all.map(p => p.t));
    const minY = Math.min(0.9, ...all.map(p => p.growth));
    const maxY = Math.max(...all.map(p => p.growth)) * 1.06;

    function el(name, attrs = {{}}, parent = svg) {{
      const node = document.createElementNS(NS, name);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      parent.appendChild(node);
      return node;
    }}

    function linePath(points, x, y) {{
      return points.map((p, i) => `${{i ? "L" : "M"}}${{x(p.t).toFixed(1)}},${{y(p.growth).toFixed(1)}}`).join(" ");
    }}

    function nearestPoint(points, t) {{
      let lo = 0, hi = points.length - 1;
      while (lo < hi) {{
        const mid = Math.floor((lo + hi) / 2);
        if (points[mid].t < t) lo = mid + 1;
        else hi = mid;
      }}
      const a = points[Math.max(0, lo - 1)];
      const b = points[lo];
      return !a || Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a;
    }}

    function draw() {{
      svg.replaceChildren();
      const box = svg.getBoundingClientRect();
      const width = Math.max(640, box.width);
      const height = Math.max(420, box.height);
      const margin = {{ top: 28, right: 72, bottom: 48, left: 62 }};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);

      const x = t => margin.left + ((t - minT) / (maxT - minT)) * innerW;
      const y = v => margin.top + (1 - ((v - minY) / (maxY - minY))) * innerH;
      const g = el("g");

      const yTicks = [1, 1.5, 2, 2.5, 3, 3.5].filter(v => v >= minY && v <= maxY);
      yTicks.forEach(v => {{
        el("line", {{ x1: margin.left, x2: width - margin.right, y1: y(v), y2: y(v), class: "grid" }}, g);
        const txt = el("text", {{ x: margin.left - 10, y: y(v) + 4, "text-anchor": "end", class: "axis" }}, g);
        txt.textContent = `${{v.toFixed(1)}}x`;
      }});

      const years = [2009, 2010, 2011, 2012, 2013, 2014];
      years.forEach(year => {{
        const t = new Date(`${{year}}-01-01T00:00:00`).getTime();
        if (t < minT || t > maxT) return;
        el("line", {{ x1: x(t), x2: x(t), y1: margin.top, y2: height - margin.bottom, class: "grid" }}, g);
        const txt = el("text", {{ x: x(t), y: height - 16, "text-anchor": "middle", class: "axis" }}, g);
        txt.textContent = year;
      }});

      el("line", {{ x1: margin.left, x2: width - margin.right, y1: y(1), y2: y(1), stroke: "#8d98a6", "stroke-dasharray": "4 4" }}, g);

      DATA.series.slice().reverse().forEach((s, idx) => {{
        const isTop = s === DATA.series[0];
        el("path", {{
          d: linePath(s.points, x, y),
          stroke: s.color,
          class: `path${{isTop ? " top" : ""}}`,
        }}, g);
      }});

      const hoverLine = el("line", {{
        y1: margin.top,
        y2: height - margin.bottom,
        stroke: "#566170",
        "stroke-width": 1,
        opacity: 0,
      }}, g);
      const dots = DATA.series.map(s => el("circle", {{ r: s === DATA.series[0] ? 4.5 : 3.5, fill: s.color, opacity: 0 }}, g));

      const hit = el("rect", {{
        x: margin.left,
        y: margin.top,
        width: innerW,
        height: innerH,
        fill: "transparent",
      }}, g);
      hit.addEventListener("mousemove", event => {{
        const pt = svg.createSVGPoint();
        pt.x = event.clientX;
        pt.y = event.clientY;
        const local = pt.matrixTransform(svg.getScreenCTM().inverse());
        const t = minT + ((local.x - margin.left) / innerW) * (maxT - minT);
        const points = DATA.series.map(s => ({{ s, p: nearestPoint(s.points, t) }}));
        const hoverX = x(points[0].p.t);
        hoverLine.setAttribute("x1", hoverX);
        hoverLine.setAttribute("x2", hoverX);
        hoverLine.setAttribute("opacity", 1);
        points.forEach((item, i) => {{
          dots[i].setAttribute("cx", x(item.p.t));
          dots[i].setAttribute("cy", y(item.p.growth));
          dots[i].setAttribute("opacity", 1);
        }});
        tooltip.innerHTML = `<div class="date">${{points[0].p.date}}</div>` + points
          .map(item => `<div class="tip-row"><span style="color:${{item.s.color}}">${{item.s.label}}</span><b>${{item.p.growth.toFixed(2)}}x (${{item.p.returnPct.toFixed(1)}}%)</b></div>`)
          .join("");
        tooltip.style.opacity = 1;
        tooltip.style.left = `${{Math.min(Math.max(hoverX, 130), width - 130)}}px`;
        tooltip.style.top = `${{Math.max(y(points[0].p.growth), 90)}}px`;
      }});
      hit.addEventListener("mouseleave", () => {{
        hoverLine.setAttribute("opacity", 0);
        dots.forEach(dot => dot.setAttribute("opacity", 0));
        tooltip.style.opacity = 0;
      }});

      legend.innerHTML = DATA.series.map(s => `
        <div class="legend-item">
          <i class="swatch" style="background:${{s.color}}"></i>
          <b>${{s.label}}</b>
          <span>${{s.totalReturnPct.toFixed(1)}}% total, Sharpe ${{s.sharpe.toFixed(2)}}</span>
        </div>
      `).join("");
    }}

    draw();
    window.addEventListener("resize", draw);
  </script>
</body>
</html>
"""
    OUTPUT.write_text(html)
    print(OUTPUT)


if __name__ == "__main__":
    main()
