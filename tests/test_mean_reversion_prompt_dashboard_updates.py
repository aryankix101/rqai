"""Test incremental research sections added to the prompt dashboard.

These cases cover completed-artifact ingestion, frozen-result presentation,
and stable HTML updates as later V2 analyses become available.
"""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import mean_reversion_llm_v2_phase2_validation as phase2
import mean_reversion_prompt_dashboard as dashboard


class PromptDashboardUpdateTests(unittest.TestCase):
    def test_no_regime_rows_use_fresh_v2_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_config.json").write_text(
                json.dumps({"prompt_hash": "no-regime-hash"}),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "run": "Baseline-Pool", "execution_model": "C2C", "cost_bps": 0.0,
                        "is_anchor_average": False, "capital_cagr_pct": 13.0,
                        "capital_sharpe": 1.1, "capital_max_drawdown_pct": -9.0,
                    },
                    {
                        "run": "LLM-Pool v2 Fresh Control", "execution_model": "C2C", "cost_bps": 0.0,
                        "is_anchor_average": False, "capital_cagr_pct": 12.8,
                        "capital_sharpe": 1.14, "capital_max_drawdown_pct": -8.5,
                    },
                    {
                        "run": "LLM-Pool v2-no-regime", "execution_model": "C2C", "cost_bps": 0.0,
                        "is_anchor_average": False, "capital_cagr_pct": 15.9,
                        "capital_sharpe": 1.35, "capital_max_drawdown_pct": -9.7,
                    },
                ]
            ).to_csv(root / "comparison_scorecard.csv", index=False)

            rows = dashboard.build_no_regime_result_rows(root)
            read = dashboard.build_no_regime_result_read(root)

        by_run = {row["Run"]: row for row in rows}
        self.assertEqual(by_run["V2 without Market Regime"]["CAGR change vs fresh V2"], "+3.10%")
        self.assertEqual(by_run["V2 without Market Regime"]["Sharpe change vs fresh V2"], "0.210")
        self.assertEqual(by_run["V2 without Market Regime"]["Prompt hash"], "no-regime-hash")
        self.assertIn("one stochastic paired experiment", read)

    def test_v26_minimal_rows_compare_against_frozen_v2_and_trail5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incumbent = root / "incumbent"
            minimal = root / "minimal"
            minimal_temp1 = root / "minimal_temp1"
            minimal_temp15 = root / "minimal_temp15"
            minimal_temp06_second = root / "minimal_temp06_second"
            minimal_temp15_incidents = root / "minimal_temp15_incidents.json"
            v26 = root / "v26"
            for path, model, prompt_hash in (
                (incumbent, "deepseek-chat", "incumbent-hash"),
                (minimal, "deepseek-v4-flash", "minimal-hash"),
                (minimal_temp1, "deepseek-v4-flash", "minimal-hash"),
                (minimal_temp15, "deepseek-v4-flash", "minimal-hash"),
                (minimal_temp06_second, "deepseek-v4-flash", "minimal-hash"),
                (v26, "deepseek-v4-flash", "v26-hash"),
            ):
                path.mkdir()
                (path / "run_config.json").write_text(
                    json.dumps(
                        {
                            "model": model,
                            "prompt_hash": prompt_hash,
                            "temperature": 1.5 if path == minimal_temp15 else (1.0 if path == minimal_temp1 else 0.6),
                        }
                    ),
                    encoding="utf-8",
                )

            def write_rows(path: Path, run: str, cagr: float, sharpe: float) -> None:
                pd.DataFrame(
                    [
                        {
                            "run": "Baseline-Pool",
                            "execution_model": "C2C",
                            "cost_bps": 0.0,
                            "is_anchor_average": False,
                            "capital_cagr_pct": 13.0,
                            "capital_sharpe": 1.1,
                            "capital_max_drawdown_pct": -9.0,
                            "ic_mean": 0.02,
                        },
                        {
                            "run": run,
                            "execution_model": "C2C",
                            "cost_bps": 0.0,
                            "is_anchor_average": False,
                            "capital_cagr_pct": cagr,
                            "capital_sharpe": sharpe,
                            "capital_max_drawdown_pct": -8.0,
                            "ic_mean": 0.03,
                        },
                    ]
                ).to_csv(path / "comparison_scorecard.csv", index=False)

            write_rows(incumbent, "LLM-Pool v2", 14.0, 1.2)
            write_rows(minimal, "LLM-Pool v2-minimal-change-control", 16.0, 1.4)
            write_rows(minimal_temp1, "LLM-Pool v2-minimal-change-control", 16.2, 1.39)
            write_rows(minimal_temp15, "LLM-Pool v2-minimal-change-control", 12.0, 1.0)
            write_rows(minimal_temp06_second, "LLM-Pool v2-minimal-change-control", 13.0, 1.1)
            write_rows(v26, "LLM-Pool v2.6", 12.0, 1.0)
            minimal_temp15_incidents.write_text(
                json.dumps({"incidents": [{"error": "SEC_0061 has invalid score 800"}]}),
                encoding="utf-8",
            )
            rows = dashboard.build_v26_minimal_result_rows(
                incumbent, minimal, v26, minimal_temp1, minimal_temp15,
                minimal_temp15_incidents, minimal_temp06_second
            )
            read = dashboard.build_v26_minimal_read(
                incumbent, minimal, v26, minimal_temp1, minimal_temp15,
                minimal_temp15_incidents, minimal_temp06_second
            )

        by_run = {row["Run"]: row for row in rows}
        self.assertEqual(by_run["Minimal change · temp 0.6"]["Lift vs V2"], "+2.00pp")
        self.assertEqual(by_run["Minimal change · temp 1.0"]["Δ vs Minimal 0.6"], "+0.20pp")
        self.assertEqual(by_run["Minimal change · temp 1.0"]["Temperature"], "1.0")
        self.assertEqual(by_run["Minimal change · temp 0.6 · second run"]["Δ vs Minimal 0.6"], "-3.00pp")
        self.assertEqual(by_run["Minimal change · temp 1.5"]["Δ vs Minimal 0.6"], "-4.00pp")
        self.assertIn("Rejected score 800", by_run["Minimal change · temp 1.5"]["Schema / Reliability"])
        self.assertEqual(by_run["V2.6 rewrite"]["Conclusion"], "Failed development screen")
        self.assertIn("Temperature 1.0 remains the preferred challenger setting", read)
        self.assertIn("rejected score of 800", read)
        self.assertIn("second independent temperature-0.6 run", read)

    def test_prompt_cards_render_complete_prompt_text(self) -> None:
        marker = "END-OF-FULL-PROMPT"
        card = dashboard.PromptCard(
            title="Long prompt",
            family="Test",
            schema="Test",
            asks_for="Test",
            reasoning_visibility="Test",
            pool_process="Test",
            accounting="Test",
            cache_replay="Test",
            status="Test",
            prompt_text=("x" * 1_000) + marker,
        )

        rendered = dashboard.prompt_card_html(card)

        self.assertIn("<summary>Full prompt</summary>", rendered)
        self.assertIn(marker, rendered)
        self.assertNotIn("Prompt excerpt", rendered)

    def test_archived_v21_prompt_matches_tested_run_hash(self) -> None:
        prompt_hash = hashlib.sha256(
            dashboard.ARCHIVED_V21_DISCRIMINATIVE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()[:12]

        self.assertEqual(prompt_hash, "c6d0cc621aca")
        v21_card = next(card for card in dashboard.prompt_cards() if card.family == "Archived V2.1")
        self.assertEqual(v21_card.prompt_text, dashboard.ARCHIVED_V21_DISCRIMINATIVE_SYSTEM_PROMPT)

    def test_glossary_ablation_rows_compare_same_construction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase2_scorecard.csv"
            pd.DataFrame(
                [
                    {
                        "period": phase2.NO_GLOSSARY_VALIDATION_PERIOD,
                        "test_id": "top30_equal",
                        "llm_cagr_pct": 0.9,
                        "trail5_cagr_pct": 1.6,
                        "cagr_lift_pp": -0.7,
                    },
                    {
                        "period": phase2.GLOSSARY_VALIDATION_PERIOD,
                        "test_id": "top30_equal",
                        "llm_cagr_pct": 2.6,
                        "trail5_cagr_pct": 1.6,
                        "cagr_lift_pp": 1.0,
                    },
                ]
            ).to_csv(path, index=False)

            rows = dashboard.build_phase2_glossary_ablation_rows(path)
            read = dashboard.build_phase2_glossary_ablation_read(path)
            insights = dashboard.build_cache_insight_cards(Path(tmp) / "missing-forensics", path)

        self.assertEqual(rows[0]["Construction"], "Top 30 / 30 equal weight")
        self.assertEqual(rows[0]["Conclusion"], "Glossary stronger")
        self.assertEqual(rows[0]["Glossary minus no glossary"], "+1.7%")
        self.assertIn("No glossary was stronger on five", read)
        self.assertEqual(insights[0]["Title"], "Glossary Changes Where The Edge Lives")
        self.assertIn("glossary may help diversification", insights[0]["Implication"])
        self.assertIn("hypothesis to validate", insights[0]["Implication"])

    def test_v25_coverage_is_audit_not_paid_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "v25"
            memory = root / "v25_balanced_outcome_memory"
            memory.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"memory_ready": False, "fully_populated": False},
                    {"memory_ready": True, "fully_populated": False},
                    {"memory_ready": True, "fully_populated": True},
                ]
            ).to_csv(memory / "coverage_by_date.csv", index=False)
            pd.DataFrame([{"candidate": 1}, {"candidate": 2}]).to_csv(
                memory / "candidate_memory.csv", index=False
            )
            pd.DataFrame([{"neighbor": 1}]).to_csv(memory / "neighbor_audit.csv", index=False)
            pd.DataFrame([{"case": 1}]).to_csv(memory / "case_history.csv", index=False)
            (memory / "memory_spec.json").write_text(
                json.dumps({"spec_hash": "test-memory"}), encoding="utf-8"
            )
            pd.DataFrame(
                [
                    {
                        "scope": "memory_ready",
                        "run": "Trail5",
                        "n_periods": 2,
                        "cagr_pct": 10.0,
                        "sharpe": 1.0,
                        "max_drawdown_pct": -5.0,
                        "cagr_lift_vs_trail5_pp": 0.0,
                        "cagr_lift_vs_v2_pp": float("nan"),
                    },
                    {
                        "scope": "memory_ready",
                        "run": "HistRet-Only",
                        "n_periods": 2,
                        "cagr_pct": -3.0,
                        "sharpe": -0.3,
                        "max_drawdown_pct": -20.0,
                        "cagr_lift_vs_trail5_pp": -13.0,
                        "cagr_lift_vs_v2_pp": float("nan"),
                    },
                ]
            ).to_csv(memory / "coverage_adjusted_paired_scorecard.csv", index=False)

            coverage = dashboard.build_v25_memory_coverage_rows(root)
            architecture = dashboard.build_v24_v25_architecture_rows(Path(tmp) / "v24", root)
            status_read = dashboard.build_v25_status_read(root)
            paired_rows = dashboard.build_v25_paired_result_rows(root)

        self.assertEqual(coverage[1]["Value"], "2/3 (66.7%)")
        self.assertIn("paid A/B pending", architecture[1]["Status"])
        self.assertEqual(architecture[1]["Hash"], "test-memory")
        self.assertIn("paid paired comparison is incomplete", status_read)
        self.assertIn("V2 Fresh Control", status_read)
        self.assertEqual(paired_rows[1]["Run"], "HistRet-Only")
        self.assertEqual(paired_rows[1]["Evidence"], "Deterministic diagnostic")


if __name__ == "__main__":
    unittest.main()
