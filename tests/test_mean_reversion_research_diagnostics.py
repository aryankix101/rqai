"""Test leakage guards and offline diagnostics for V2 development research.

The suite verifies development-period enforcement, statistical summaries,
prompt-suite manifests, and tabular baseline outputs without model calls.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import mean_reversion_llm_v2 as runner
import mean_reversion_research_diagnostics as diagnostics


class ResearchDiagnosticsTests(unittest.TestCase):
    def test_blind_period_firewall_rejects_2015(self):
        with self.assertRaisesRegex(
            diagnostics.ResearchDiagnosticError, "Blind-period firewall"
        ):
            diagnostics._assert_development_dates(
                ["2014-12-31", "2015-01-02"], source="synthetic"
            )

    def test_moving_block_bootstrap_reports_probability_positive(self):
        result = diagnostics.moving_block_bootstrap(
            np.full(40, 0.001), n_boot=200, block_size=4, seed=7
        )

        self.assertAlmostEqual(result["mean_annualized_lift_pct"], 5.2)
        self.assertEqual(result["probability_lift_gt_zero"], 1.0)
        self.assertGreater(result["ci_95_low_pct"], 0.0)

    def test_alpha_concentration_trims_best_dates(self):
        result = diagnostics.alpha_concentration(
            pd.Series([0.10, 0.01, 0.01, 0.01, 0.01] * 20)
        )

        one_percent = result.iloc[0]
        five_percent = result.iloc[1]
        self.assertEqual(int(one_percent["dates_removed"]), 1)
        self.assertEqual(int(five_percent["dates_removed"]), 5)
        self.assertLess(
            five_percent["trimmed_annualized_lift_pct"],
            float(pd.Series([0.10, 0.01, 0.01, 0.01, 0.01] * 20).mean() * 52 * 100),
        )

    def test_changed_name_attribution_uses_signed_short_returns(self):
        rows = []
        for side in ("LONG", "SHORT"):
            for ticker, candidate_score, control_score, fwd in (
                ("A", 100.0, 0.0, 0.02),
                ("B", 0.0, 100.0, -0.01),
            ):
                rows.append(
                    {
                        "date": pd.Timestamp("2013-01-02"),
                        "pool_side": side,
                        "ticker": ticker,
                        "fwd_c2c": fwd,
                        "candidate": candidate_score,
                        "control": control_score,
                    }
                )
        frame = pd.DataFrame(rows)
        treatment = frame.rename(columns={"candidate": "llm_score"}).drop(
            columns=["control"]
        )
        control = frame.rename(columns={"control": "llm_score"}).drop(
            columns=["candidate"]
        )

        result = diagnostics.selection_diagnostics(
            treatment, control, book_size=1
        )
        changed = result["changed_rows"].set_index("pool_side")

        self.assertAlmostEqual(changed.loc["LONG", "changed_name_capital_lift"], 0.015)
        self.assertAlmostEqual(changed.loc["SHORT", "changed_name_capital_lift"], -0.015)
        self.assertAlmostEqual(result["annualized_changed_names_only_lift_pct"], 0.0)

    def test_prompt_suite_uses_old_g005_and_not_forced_feature_cross(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = diagnostics.prepare_prompt_interaction_suite(Path(tmp))
            prompt_b = json.loads(Path(suite["arms"]["B"]["prompt_spec"]).read_text())[
                "prompt_text"
            ]
            prompt_c = json.loads(Path(suite["arms"]["C"]["prompt_spec"]).read_text())[
                "prompt_text"
            ]
            prompt_d = json.loads(Path(suite["arms"]["D"]["prompt_spec"]).read_text())[
                "prompt_text"
            ]

            self.assertEqual(prompt_b, runner.G0_05_DEPRESCRIBED_CONTEXT_SYSTEM_PROMPT)
            self.assertEqual(prompt_c, runner.G0_01_PROPORTIONAL_SEPARATION_SYSTEM_PROMPT)
            self.assertIn("We are evaluating two pools", prompt_d)
            self.assertIn("distinct scores proportional", prompt_d)
            self.assertNotIn("top_feature", prompt_d)
            self.assertNotEqual(prompt_d, runner.G0_01_X_FORCED_TOP_FEATURE_SYSTEM_PROMPT)

    def test_consolidated_ranking_keeps_robust_g002_over_tiny_cagr_gap(self):
        def candidate(label, lift, stability, long_lift, short_lift, ci_low, trimmed):
            return {
                "label": label,
                "verification": {"pass": True},
                "bootstrap": {
                    "ci_95_low_pct": ci_low,
                    "probability_lift_gt_zero": 0.75,
                },
                "annualized_paired_lift_pct": lift,
                "chronological_block_stability_pct": stability,
                "trimmed_5pct_annualized_lift_pct": trimmed,
                "long_annualized_lift_pct": long_lift,
                "short_annualized_lift_pct": short_lift,
                "sharpe_lift": 0.1,
                "drawdown_improvement_pct": 1.0,
            }

        candidates = [
            candidate("G0_01", 1.16, 50.0, -0.12, 1.28, -1.87, -2.30),
            candidate("old G0_05", 1.26, 75.0, 1.08, 0.18, -1.92, -1.21),
            candidate("G0_02", 1.06, 75.0, 0.02, 1.03, -2.07, -2.38),
        ]
        factors = [
            {
                "label": label,
                "weak_state_positive_lift": label != "G0_01",
                "weak_state_ci_excludes_zero": False,
                "weakest_quintile": {
                    "paired_llm_minus_trail5_lift_pct": 2.0 if label != "G0_01" else -2.0
                },
            }
            for label in ("G0_01", "old G0_05", "G0_02")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            recommendation = diagnostics.write_consolidated_dashboard(
                Path(tmp),
                candidates,
                factors,
                {"low_disagreement_outperformed": True},
                {
                    "llm_beats_nonlinear_tabular": True,
                    "nonlinear_paired_lift_vs_llm_pct": -3.0,
                },
            )

        self.assertEqual(recommendation["provisional_primary"], "old G0_05")
        self.assertEqual(recommendation["provisional_challenger"], "G0_02")
        self.assertFalse(recommendation["ready_for_blind_validation"])


if __name__ == "__main__":
    unittest.main()
