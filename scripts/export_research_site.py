"""Export the research dashboard suite as a self-contained static website.

The source dashboards live under the ignored ``backtest_results`` tree. This
utility copies only the curated HTML pages intended for publication, rewrites
their cross-page links, and writes the GitHub Pages bundle to ``docs/research``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs/research"
PUBLISHED_HEADER = """<!--
  Purpose: Publish a self-contained page from the mean-reversion LLM research
  dashboard suite. Regenerate with scripts/export_research_site.py.
-->
"""
HREF_PATTERN = re.compile(r'(?P<prefix>href=["\'])(?P<url>[^"\']+)(?P<suffix>["\'])')


@dataclass(frozen=True)
class PublishedPage:
    source: Path
    destination: str


PAGES = (
    PublishedPage(
        Path("backtest_results/llm_reversal_signal_v2_2009_2014/mean_reversion_research_dashboard.html"),
        "index.html",
    ),
    PublishedPage(
        Path("backtest_results/llm_reversal_signal_v2_2009_2014/prompt_engineering_dashboard.html"),
        "prompt-engineering.html",
    ),
    PublishedPage(
        Path("backtest_results/llm_reversal_signal_v2_2009_2014/autoresearch_v2_cache_sweep/autoresearch_report.html"),
        "autoresearch.html",
    ),
    PublishedPage(
        Path("backtest_results/llm_reversal_signal_v2_2009_2014/future_experiments_dashboard.html"),
        "future-experiments.html",
    ),
    PublishedPage(
        Path("backtest_results/llm_reversal_signal_v2_2009_2014/phase2_validation_dashboard.html"),
        "out-of-sample.html",
    ),
    PublishedPage(
        Path("backtest_results/autoresearch_campaigns/prompt_search_ticker_optional_20260807/campaign_dashboard.html"),
        "latest-campaign.html",
    ),
    PublishedPage(
        Path("backtest_results/research_thesis_package_20260808/research_dashboard.html"),
        "prompt-interaction.html",
    ),
    PublishedPage(
        Path("backtest_results/llm_reversal_signal_v2_5_anon_2009_2014/v25_balanced_outcome_memory/memory_postmortem/memory_postmortem_report.html"),
        "memory-postmortem.html",
    ),
    PublishedPage(
        Path("backtest_results/v2_prompt_cross_g0_01_g0_05_2009_2014/full/llm_reversal_v2_report.html"),
        "prompt-cross-report.html",
    ),
    PublishedPage(
        Path("backtest_results/v2_three_run_consensus_2009_2014/report.html"),
        "consensus-report.html",
    ),
)


def absolute_source(page: PublishedPage) -> Path:
    """Resolve a manifest source relative to the repository root."""
    return (PROJECT_ROOT / page.source).resolve()


def rewrite_links(source: Path, html: str, destinations: dict[Path, str]) -> str:
    """Replace links between manifest pages with flat publish-directory links."""

    def replace(match: re.Match[str]) -> str:
        url = match.group("url")
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match.group(0)

        linked_source = (source.parent / parsed.path).resolve()
        destination = destinations.get(linked_source)
        if destination is None:
            if parsed.path.lower().endswith(".html") and linked_source.is_file():
                raise ValueError(
                    f"{source.relative_to(PROJECT_ROOT)} links to unpublished page "
                    f"{linked_source.relative_to(PROJECT_ROOT)}"
                )
            return match.group(0)

        rewritten = urlunsplit(("", "", destination, parsed.query, parsed.fragment))
        return f'{match.group("prefix")}{rewritten}{match.group("suffix")}'

    return HREF_PATTERN.sub(replace, html)


def strip_trailing_whitespace(html: str) -> str:
    """Remove generator whitespace that would fail Git's diff checks."""
    trailing_newline = "\n" if html.endswith("\n") else ""
    return "\n".join(line.rstrip() for line in html.splitlines()) + trailing_newline


def render_pages() -> dict[str, str]:
    """Read, validate, and render every page declared in the publish manifest."""
    missing = [page.source for page in PAGES if not absolute_source(page).is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required dashboard sources are missing:\n{formatted}")

    destinations = {absolute_source(page): page.destination for page in PAGES}
    rendered: dict[str, str] = {}
    for page in PAGES:
        source = absolute_source(page)
        html = source.read_text(encoding="utf-8")
        html = rewrite_links(source, html, destinations)
        html = strip_trailing_whitespace(html)
        if not html.startswith(PUBLISHED_HEADER):
            html = PUBLISHED_HEADER + html.lstrip()
        rendered[page.destination] = html
    return rendered


def export_site(output_dir: Path, *, check: bool = False) -> None:
    """Write the site or verify that committed pages match current sources."""
    rendered = render_pages()
    output_dir = output_dir.resolve()

    if check:
        stale = [
            name
            for name, expected in rendered.items()
            if not (output_dir / name).is_file()
            or (output_dir / name).read_text(encoding="utf-8") != expected
        ]
        if stale:
            raise SystemExit("Published dashboard pages are stale: " + ", ".join(stale))
        print(f"Published dashboard suite is current ({len(rendered)} pages).")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, html in rendered.items():
        (output_dir / name).write_text(html, encoding="utf-8")
    print(f"Exported {len(rendered)} dashboard pages to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse the output location and optional freshness-check mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_site(args.output_dir, check=args.check)
