"""Test base rendering of the static prompt-engineering dashboard.

The dashboard must remain buildable from partial result sets without invoking
an LLM or requiring optional research artifacts to exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pandas as pd

import mean_reversion_prompt_dashboard as dashboard


def test_prompt_dashboard_writes_without_optional_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "results"
    cache = tmp_path / "cache"
    root.mkdir()
    cache.mkdir()
    (root / "run_config.json").write_text(
        json.dumps(
            {
                "prompt_source": "mean_reversion_llm_v1.V1_SYSTEM_PROMPT",
                "prompt_style": "v1_score",
                "pool_sizes": [60],
                "book_sizes": [30],
                "anchor_offsets": [3],
            }
        ),
        encoding="utf-8",
    )

    # Keep this test independent of optional artifacts that may exist in the
    # developer's ignored ``backtest_results`` directory.
    with mock.patch.object(
        dashboard,
        "build_consensus_and_side_experiments_html",
        return_value="<div>Optional artifact not found</div>",
    ):
        output = dashboard.write_prompt_dashboard(root, cache)

    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "Mean-Reversion Prompt Engineering Dashboard" in html
    assert "V2 Score Prompt" in html
    assert "Optional artifact not found" in html
    assert "No LLM calls or backtests are run" in html


def test_dashboard_detects_cache_metadata(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "llm_reversal_signal_v2_demo.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {"model": "deepseek-chat", "prompt_style": "v1_score"},
                "signals": {
                    "2009-01-02|pool60": {
                        "llm_usage": {
                            "provider_usage": {"total_tokens": 1000},
                            "approx_total_tokens": 900,
                        }
                    },
                    "2009-01-07|pool60": {
                        "llm_usage": {
                            "provider_usage": {"total_tokens": 1200},
                            "approx_total_tokens": 1100,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    info = dashboard.find_v2_cache(cache)

    assert info["signal_count"] == 2
    assert info["model"] == "deepseek-chat"
    assert info["avg_total_tokens"] == 1100


def test_consensus_dashboard_section_reads_completed_artifacts(tmp_path: Path) -> None:
    consensus = tmp_path / "consensus"
    side = tmp_path / "side"
    consensus.mkdir()
    side.mkdir()
    (consensus / "summary.json").write_text(
        json.dumps(
            {
                "hypothesis": {"label": "not_supported"},
                "provenance": {
                    "no_regime_paired_controls_included": False,
                    "llm_calls_made": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"source_runs": "run1+run2+run3", "cagr_pct": 12.9, "sharpe": 1.1, "max_drawdown_pct": -10.1, "paired_lift_vs_member_average_pct": -1.0}]
    ).to_csv(consensus / "triplet_comparison.csv", index=False)
    pd.DataFrame(
        [{"ensemble_size": 3, "combinations": 4, "mean_cagr_pct": 14.0, "min_cagr_pct": 12.9, "max_cagr_pct": 14.7, "mean_sharpe": 1.2}]
    ).to_csv(consensus / "ensemble_size_curve.csv", index=False)

    rendered = dashboard.build_consensus_and_side_experiments_html(consensus, side)

    assert "V2 Three-Run Rank Consensus" in rendered
    assert "run1+run2+run3" in rendered
    assert "Not Supported" in rendered
    assert "Awaiting the two one-run side caches" in rendered


def test_build_main_result_rows_uses_capital_metrics(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    pd.DataFrame(
        [
            {
                "family": "V2",
                "run": "LLM-Pool v2 Best Offset",
                "prompt_schema": "standard v1 score prompt",
                "pool_book": "60/60 -> 30/30",
                "anchor_offset": "3",
                "execution_model": "C2C",
                "cost_bps": 0.0,
                "capital_cagr_pct": 16.2,
                "cagr_pct": 33.2,
                "capital_sharpe": 1.33,
                "sharpe": 1.33,
                "capital_max_drawdown_pct": -7.7,
                "max_drawdown_pct": -15.4,
                "ic_mean": 0.0295,
            }
        ]
    ).to_csv(root / "gross_legacy_v1_v2_comparison.csv", index=False)

    rows = dashboard.build_main_result_rows(root)

    assert rows[0]["CAGR"] == "16.2%"
    assert rows[0]["Max DD"] == "-7.7%"
    assert rows[0]["Conclusion"] == "Winner"


def test_temperature_control_read_uses_completed_matching_artifacts(tmp_path: Path) -> None:
    roots = []
    for label, temperature, cagr, sharpe in (
        ("temp0", 0.0, 13.4, 1.16),
        ("temp06", 0.6, 13.6, 1.15),
    ):
        root = tmp_path / label
        root.mkdir()
        (root / "run_config.json").write_text(json.dumps({"temperature": temperature}), encoding="utf-8")
        pd.DataFrame(
            [{
                "run": "LLM-Pool v2",
                "family": "V2",
                "anchor_offset": "3",
                "execution_model": "C2C",
                "cost_bps": 0.0,
                "capital_cagr_pct": cagr,
                "capital_sharpe": sharpe,
            }]
        ).to_csv(root / "gross_legacy_v1_v2_comparison.csv", index=False)
        roots.append(root)

    read = dashboard.build_temperature_control_read(roots[0], roots[1])

    assert "Control complete" in read
    assert "13.6% CAGR versus 13.4%" in read
    assert "effectively unchanged" in read


def test_campaign_dashboard_renders_read_only_status_lineage_and_gross_metrics(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaigns" / "campaign-001"
    candidate_dir = campaign_dir / "candidates" / "g0-01"
    (candidate_dir / "mini").mkdir(parents=True)
    (campaign_dir / "validation").mkdir()
    (campaign_dir / "campaign.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "campaign-001",
                "status": "running",
                "phase": "development",
                "development": {
                    "start": "2009-01-01",
                    "end": "2014-01-01",
                    "manifest_sizes": {
                        "smoke": 5,
                        "early": 12,
                        "mini": 24,
                        "reassess": 48,
                        "medium": 80,
                        "full": 251,
                    },
                },
                "validation": {
                    "start": "2015-01-01",
                    "end": "2019-12-31",
                    "approved": False,
                    "status": "awaiting_approval",
                },
                "budgets": {
                    "development": {
                        "flash_calls": 1100,
                        "flash_output_tokens": 5_000_000,
                        "pro_calls": 4,
                    },
                    "validation": {"flash_calls": 550},
                },
                "candidates": ["g0-01"],
                "winner_id": "g0-01",
            }
        ),
        encoding="utf-8",
    )
    (campaign_dir / "status.json").write_text(
        json.dumps(
            {
                "campaign_id": "campaign-001",
                "status": "running",
                "phase": "development",
                "current_candidate": "g0-01",
                "current_stage": "medium",
                "supervisor_pid": 12345,
                "completed_dates": 24,
                "planned_dates": 80,
                "usage": {
                    "development": {
                        "flash_calls": 48,
                        "flash_output_tokens": 120000,
                        "pro_calls": 1,
                    },
                    "validation": {"flash_calls": 0},
                },
                "last_event_at": "2026-07-29T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "prompt_spec.json").write_text(
        json.dumps(
            {
                "candidate_id": "g0-01",
                "generation": "G0",
                "parent_id": "v2-minimal-change-control",
                "mutation_mode": "substitutive",
                "patch": {"op": "replace_once", "old_text": "old", "new_text": "new"},
                "hypothesis": "Separate exhaustion from continuation more clearly.",
                "mechanism": "Make the score distinguish exhaustion from persistent momentum.",
                "expected_failure": "The distinction may not improve ordering.",
                "prompt_text": "Complete newly materialized prompt.",
                "model": "deepseek-v4-flash",
                "temperature": 1.0,
                "thinking_mode": "disabled",
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "mini" / "result.json").write_text(
        json.dumps(
            {
                "candidate_id": "g0-01",
                "stage": "mini",
                "status": "completed",
                "metrics": {
                    "annualized_arithmetic_return_pct": 12.5,
                    "sample_sharpe": 1.25,
                    "paired_lift_pp": 2.75,
                    "net_return_pct": 777.77,
                    "spy_regression_alpha": 888.88,
                },
                "gate": {"passed": True, "reason": "promising"},
            }
        ),
        encoding="utf-8",
    )
    (campaign_dir / "validation" / "result.json").write_text(
        json.dumps(
            {
                "candidate_id": "g0-01",
                "status": "validated_winner",
                "metrics": {"capital_cagr_pct": 16.2, "capital_sharpe": 1.4},
            }
        ),
        encoding="utf-8",
    )
    (campaign_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-07-29T11:00:00Z",
                        "event": "stage_completed",
                        "campaign_id": "campaign-001",
                        "candidate_id": "g0-01",
                        "stage": "mini",
                        "payload": {"status": "completed", "raw_return": "DO_NOT_RENDER"},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-07-29T11:01:00Z",
                        "event": "provider_retry",
                        "campaign_id": "campaign-001",
                        "candidate_id": "g0-01",
                        "stage": "mini",
                        "payload": {"error": "temporary transport failure"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (campaign_dir / "rejected_proposals.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-29T10:59:00Z",
                "rejection_id": "rejected-a1b2c3",
                "candidate_id": "G0_01",
                "generation": "G0",
                "reason": "mutation_mode must use an allowed value",
                "proposal": {
                    "mutation_mode": "descriptive",
                    "patch": {
                        "operation": "replace_once",
                        "old_text": "sanitized old wording",
                        "new_text": "sanitized proposed wording",
                    },
                    "hypothesis": "Rejected but auditable idea.",
                    "mechanism": "Would change relative ordering.",
                    "expected_failure": "Could compress scores.",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = dashboard.write_autoresearch_campaign_dashboard(campaign_dir)

    rendered = output.read_text(encoding="utf-8")
    assert "campaign-001" in rendered
    assert "What Was Tested and How Far It Got" in rendered
    assert "Proposed Prompt Changes" in rendered
    assert "Separate exhaustion from continuation more clearly." in rendered
    assert "Make the score distinguish exhaustion from persistent momentum." in rendered
    assert "The distinction may not improve ordering." in rendered
    assert "Replaced text" in rendered
    assert "old" in rendered
    assert "New text" in rendered
    assert "new" in rendered
    assert "Show complete resulting prompt" in rendered
    assert "Complete newly materialized prompt." in rendered
    assert "Sanitized Rejected Proposals" in rendered
    assert "rejected-a1b2c3" in rendered
    assert "Rejected but auditable idea." in rendered
    assert "mutation_mode must use an allowed value" in rendered
    assert "sanitized proposed wording" in rendered
    assert "24 dates 1" in rendered
    assert "shared 5 → 12 → 24 → 48 → 80 → 251-date race" in rendered
    assert "No development winner was selected" not in rendered
    assert "Yes — research is running" in rendered
    assert "How far down the research tree?" in rendered
    assert "What happens next?" in rendered
    assert "80 dates: 0/1 running" in rendered
    assert "Finish the 80-date medium run for g0-01." in rendered
    assert "send at most two to 251 dates" in rendered
    assert "Manual validation approval: waiting" in rendered
    assert "2026-07-29 12:00" in rendered
    assert "temporary transport failure" in rendered
    assert "DO_NOT_RENDER" not in rendered
    assert "777.77" not in rendered
    assert "888.88" not in rendered
    assert "Display Contract" not in rendered
    assert "Budgets and Usage" not in rendered
    assert "Completed Stage Results" not in rendered
    assert "Recent Events" not in rendered
    assert 'http-equiv="refresh"' not in rendered


def test_campaign_winner_panel_highlights_decision_and_validation_action() -> None:
    rendered = dashboard.build_campaign_winner_panel(
        {
            "campaign": {
                "campaign_id": "campaign-001",
                "winner_id": "G0_05",
                "validation": {"approved": True},
            },
            "status": {"status": "awaiting_validation_approval"},
            "candidates": [
                {"candidate_id": "G0_05", "hypothesis": "Prefer clean exhaustion."},
                {"candidate_id": "G0_06", "hypothesis": "Prefer broad reversals."},
            ],
            "stages": [
                {
                    "candidate_id": "G0_05",
                    "stage": "full",
                    "status": "completed",
                    "gate": {"pass": True},
                    "controls": {"fresh_incumbent": {"cagr_pct": 13.10, "sharpe": 1.103}},
                    "metrics": {
                        "cagr_pct": 14.56,
                        "sharpe": 1.260,
                        "paired_annualized_lift_pct": 1.26,
                        "sharpe_improvement": 0.157,
                        "max_drawdown_pct": -7.62,
                    },
                },
                {
                    "candidate_id": "G0_06",
                    "stage": "full",
                    "status": "completed",
                    "gate": {
                        "pass": False,
                        "cagr_or_sharpe_vs_incumbent": True,
                        "beats_trail5_cagr_and_sharpe": False,
                        "drawdown_ok": True,
                    },
                    "metrics": {
                        "cagr_pct": 12.72,
                        "sharpe": 1.139,
                        "paired_annualized_lift_pct": -0.38,
                        "sharpe_improvement": 0.036,
                        "max_drawdown_pct": -6.54,
                    },
                },
            ],
        }
    )

    assert "Winner: G0_05" in rendered
    assert "Prefer clean exhaustion." in rendered
    assert "+1.26%" in rendered
    assert "v2-minimal-change-control" in rendered
    assert "Gross CAGR: 13.10%" in rendered
    assert "Sharpe: 1.103" in rendered
    assert "Pruned · did not beat Trail5 on both CAGR and Sharpe; lift vs control was not positive" in rendered
    assert "Validation approval is recorded" in rendered
    assert "campaign resume campaign-001" in rendered


def test_campaign_candidate_rows_explain_stage_progress() -> None:
    rows = dashboard.build_campaign_candidate_rows(
        {
            "candidates": [
                {
                    "candidate_id": "current-schema",
                    "hypothesis": "Clarify exhaustion.",
                },
                {
                    "candidate_id": "legacy-schema",
                    "hypothesis": "Emphasize reversal quality.",
                },
            ],
            "stages": [
                {
                    "candidate_id": "current-schema",
                    "stage": "smoke",
                    "status": "completed",
                    "metrics": {
                        "paired_annualized_lift_pct": 1.25,
                        "sharpe_improvement": -0.1,
                    },
                }
            ],
        }
    )

    assert "Smoke (5): Completed — return lift +1.25%; Sharpe change -0.1" in rows[0]["Stage performance"]
    assert "Early (12): Not reached" in rows[0]["Stage performance"]
    assert "Mini (24): Not reached" in rows[0]["Stage performance"]
    assert "Reassess (48): Not reached" in rows[0]["Stage performance"]
    assert "Medium (80): Not reached" in rows[0]["Stage performance"]
    assert "Full (251): Not reached" in rows[0]["Stage performance"]
    assert "Smoke (5): Not reached" in rows[1]["Stage performance"]


def test_campaign_budget_and_gate_rows_use_supervisor_schema() -> None:
    data = {
        "campaign": {
            "budgets": {
                "development": {
                    "max_flash_calls": 1100,
                    "max_flash_output_tokens": 5_000_000,
                },
                "validation": {"max_flash_calls": 550},
            }
        },
        "status": {
            "usage": {
                "development": {"flash_calls": 125, "flash_output_tokens": 4567},
                "validation": {"flash_calls": 0},
            }
        },
        "stages": [
            {
                "candidate_id": "G0_01",
                "stage": "mini",
                "status": "completed",
                "metrics": {
                    "annual_return_pct": 12.0,
                    "long_paired_lift_pct": 1.5,
                    "short_paired_lift_pct": -0.5,
                    "chronological_block_stability_pct": 75.0,
                },
                "gate": {"pass": False, "reason": "bootstrap_prune"},
            }
        ],
    }

    budgets = dashboard.build_campaign_budget_rows(data)
    flash = next(
        row
        for row in budgets
        if row["Scope"] == "Development" and row["Resource"] == "Flash Calls"
    )
    assert flash["Used"] == "125"
    assert flash["Limit"] == "1100"
    assert flash["Remaining"] == "975"

    stage = dashboard.build_campaign_stage_rows(data)[0]
    assert "Long paired lift: +1.50pp" in stage["Gross metrics"]
    assert "Block stability: 75.00%" in stage["Gross metrics"]
    assert stage["Recorded gate"] == "failed — bootstrap_prune"


def test_campaign_firewall_blocks_candidate_and_event_artifact_reads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign_dir = tmp_path / "campaigns" / "unsafe"
    campaign_dir.mkdir(parents=True)
    (campaign_dir / "campaign.json").write_text(
        json.dumps(
            {
                "campaign_id": "unsafe",
                "development": {"start": "2009-01-01", "end": "2014-01-01"},
                "validation": {"start": "2015-01-01", "end": "2020-01-01"},
                "candidates": ["must-not-open"],
            }
        ),
        encoding="utf-8",
    )
    (campaign_dir / "status.json").write_text(
        json.dumps({"campaign_id": "unsafe", "status": "blocked"}),
        encoding="utf-8",
    )

    def fail_read(*_args, **_kwargs):
        raise AssertionError("protected campaign artifact was opened")

    monkeypatch.setattr(dashboard, "_read_campaign_events", fail_read)
    monkeypatch.setattr(dashboard, "_load_campaign_candidates", fail_read)

    data = dashboard.load_campaign_dashboard_data(campaign_dir)
    output = dashboard.write_autoresearch_campaign_dashboard(campaign_dir)

    assert data["firewall_blocked"] is True
    assert data["events"] == []
    assert data["candidates"] == []
    assert "research periods must end before 2020" in output.read_text(encoding="utf-8")


def test_research_hub_links_latest_campaign_using_status_only(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    campaigns_root = tmp_path / "campaigns"
    for campaign_id, timestamp in (
        ("campaign-old", "2026-07-29T10:00:00Z"),
        ("campaign-new", "2026-07-29T12:00:00Z"),
    ):
        campaign_dir = campaigns_root / campaign_id
        campaign_dir.mkdir(parents=True)
        (campaign_dir / "status.json").write_text(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "status": "running",
                    "phase": "development",
                    "last_event_at": timestamp,
                }
            ),
            encoding="utf-8",
        )
        # Latest discovery must not depend on opening campaign metadata.
        (campaign_dir / "campaign.json").write_text("not-json", encoding="utf-8")

    assert dashboard.find_latest_campaign(campaigns_root).name == "campaign-new"

    hub = dashboard.write_research_hub(results_root, campaigns_root)
    rendered = hub.read_text(encoding="utf-8")
    expected_href = dashboard.relative_href(
        hub,
        campaigns_root / "campaign-new" / dashboard.DEFAULT_AUTORESEARCH_CAMPAIGN_OUTPUT_NAME,
    )
    assert "campaign-new" in rendered
    assert expected_href in rendered
    assert "campaign-old" not in rendered
