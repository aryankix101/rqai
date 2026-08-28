"""Test Phase 2 development and out-of-sample validation reporting.

These cases verify matched-control comparisons, frozen prompt tracking, and
the tables emitted by the zero-token validation dashboard builder.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import mean_reversion_llm_v2_phase2_validation as phase2


def _prompt_row(period: str, llm_cagr: float, trail5_cagr: float) -> dict[str, object]:
    return {
        "period": period,
        "test_id": "top30_equal",
        "test": "Top 30 / 30 equal weight",
        "llm_cagr_pct": llm_cagr,
        "trail5_cagr_pct": trail5_cagr,
        "cagr_lift_pp": llm_cagr - trail5_cagr,
    }


class Phase2GlossaryAblationTests(unittest.TestCase):
    def test_prompt_comparison_summary_answers_glossary_ablation(self) -> None:
        scorecard = pd.DataFrame(
            [
                _prompt_row(phase2.NO_GLOSSARY_VALIDATION_PERIOD, 0.87, 1.52),
                _prompt_row(phase2.GLOSSARY_VALIDATION_PERIOD, 2.59, 1.52),
            ]
        )

        summary = phase2.prompt_comparison_summary(scorecard)

        self.assertIn("did not beat Trail5", summary)
        self.assertIn("0.9% versus 1.5%", summary)
        self.assertIn("helped by +1.7%", summary)

    def test_prompt_comparison_requires_shared_trail5_baseline(self) -> None:
        glossary = pd.DataFrame(
            [
                {
                    "date": "2015-01-06",
                    "ticker": "ABC",
                    "pool_side": "LONG",
                    "score_trail5": 1.0,
                    "fwd_c2c": 0.1,
                    "fwd_o2o": 0.2,
                }
            ]
        )
        no_glossary = glossary.copy()
        phase2.assert_shared_trail5_baseline(glossary, no_glossary)

        no_glossary.loc[0, "score_trail5"] = 2.0
        with self.assertRaisesRegex(ValueError, "do not share the same Trail5 baseline"):
            phase2.assert_shared_trail5_baseline(glossary, no_glossary)

    def test_development_glossary_comparison_uses_existing_period_labels(self) -> None:
        scorecard = pd.DataFrame(
            [
                _prompt_row(phase2.NO_GLOSSARY_DEVELOPMENT_PERIOD, 14.3, 13.2),
                _prompt_row(phase2.GLOSSARY_DEVELOPMENT_PERIOD, 13.6, 13.2),
            ]
        )

        summary = phase2.prompt_comparison_summary(
            scorecard,
            phase2.GLOSSARY_DEVELOPMENT_PERIOD,
            phase2.NO_GLOSSARY_DEVELOPMENT_PERIOD,
        )
        table = phase2.prompt_comparison_table(
            scorecard,
            phase2.GLOSSARY_DEVELOPMENT_PERIOD,
            phase2.NO_GLOSSARY_DEVELOPMENT_PERIOD,
        )

        self.assertIn("glossary <strong>hurt by -0.7%</strong>", summary)
        self.assertIn("+1.1%", table)
        self.assertIn("-0.7%", table)

    def test_dashboard_tables_only_show_primary_construction(self) -> None:
        rows = []
        for test in phase2.FROZEN_TESTS:
            rows.append(
                {
                    "period": phase2.GLOSSARY_VALIDATION_PERIOD,
                    "test_id": test.test_id,
                    "test": test.label,
                    "llm_cagr_pct": 2.0,
                    "trail5_cagr_pct": 1.0,
                    "cagr_lift_pp": 1.0,
                    "annualized_long_lift_contribution_pct": 0.5,
                    "annualized_short_lift_contribution_pct": 0.5,
                    "llm_sharpe": 0.4,
                    "sharpe_lift": 0.2,
                    "llm_max_dd_pct": -10.0,
                    "max_dd_change_pp": 2.0,
                    "rebalance_win_rate_pct": 52.0,
                    "positive_lift_years": 3,
                    "years_tested": 5,
                }
            )

        rendered = phase2.result_table(pd.DataFrame(rows), phase2.GLOSSARY_VALIDATION_PERIOD)

        self.assertIn("Top 30 / 30 equal weight", rendered)
        for test in phase2.FROZEN_TESTS[1:]:
            self.assertNotIn(test.label, rendered)

    def test_minimal_change_validation_renders_completed_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "run": "Baseline-Pool",
                        "execution_model": "C2C",
                        "cost_bps": 0.0,
                        "anchor_offset": 3,
                        "is_anchor_average": False,
                        "capital_cagr_pct": 1.52,
                        "capital_sharpe": 0.20,
                        "capital_max_drawdown_pct": -15.10,
                    },
                    {
                        "run": "LLM-Pool v2-minimal-change-control",
                        "execution_model": "C2C",
                        "cost_bps": 0.0,
                        "anchor_offset": 3,
                        "is_anchor_average": False,
                        "capital_cagr_pct": 3.61,
                        "capital_sharpe": 0.39,
                        "capital_max_drawdown_pct": -12.35,
                    },
                ]
            ).to_csv(root / "comparison_scorecard.csv", index=False)
            (root / "run_config.json").write_text('{"temperature": 1.0}', encoding="utf-8")

            summary = phase2.minimal_change_validation_summary(root)
            table = phase2.minimal_change_validation_table(root)

        self.assertIn("3.6% capital CAGR", summary)
        self.assertIn("+2.1% lift", summary)
        self.assertIn("Minimal change · temperature 1.0", table)
        self.assertIn("-12.3%", table)

    def test_challenger_chart_plots_v2_and_minimal_change(self) -> None:
        period = phase2.GLOSSARY_VALIDATION_PERIOD
        weekly = pd.DataFrame(
            {
                "date": pd.to_datetime(["2015-01-06", "2015-01-13", "2015-01-20"]),
                "period": [period] * 3,
                "test_id": ["top30_equal"] * 3,
                "llm_return": [0.01, -0.02, 0.03],
                "trail5_return": [0.005, -0.01, 0.01],
            }
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = (
                root
                / "pool60_pos30"
                / "anchor_3"
                / "llm_pool_v2_minimal_change_control"
            )
            run.mkdir(parents=True)
            pd.DataFrame(
                {
                    "date": ["2015-01-06", "2015-01-13", "2015-01-20"],
                    "long_ret_c2c": [0.02, -0.01, 0.04],
                    "short_ret_c2c": [0.00, 0.01, 0.00],
                }
            ).to_csv(run / "weekly_returns.csv", index=False)

            rendered = phase2.challenger_comparison_equity_svg(weekly, period, root)

        self.assertIn("V2 · Top 30 / 30", rendered)
        self.assertIn("Minimal change · temp 1.0", rendered)
        self.assertIn("Trail5", rendered)
        self.assertEqual(rendered.count("<polyline"), 3)
        self.assertIn("V2 versus minimal-change equity", rendered)


if __name__ == "__main__":
    unittest.main()
