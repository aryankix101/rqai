"""Test the autoresearch regime gate used to promote robust candidates.

Synthetic return histories exercise regime-level statistics and ensure a
candidate must satisfy the declared robustness rules before advancing.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import mean_reversion_llm_v2_autoresearch as autoresearch


def synthetic_weekly_returns() -> pd.DataFrame:
    rows = []
    observations = [
        ("2009-01-06", "normal", {"top30": 0.01, "h20": 0.015, "h10": 0.02, "h5": 0.03}),
        ("2009-01-13", "grind_up", {"top30": 0.03, "h20": 0.02, "h10": 0.01, "h5": 0.00}),
        ("2010-01-05", "normal", {"top30": 0.01, "h20": 0.015, "h10": 0.02, "h5": 0.03}),
        ("2010-01-12", "grind_up", {"top30": 0.03, "h20": 0.02, "h10": 0.01, "h5": 0.00}),
        # Test-year outcomes reverse. A valid walk-forward mapping must still use
        # h5 for normal and top30 for grind_up based only on 2009-2010.
        ("2011-01-04", "normal", {"top30": 0.04, "h20": 0.02, "h10": 0.01, "h5": -0.01}),
        ("2011-01-11", "grind_up", {"top30": -0.01, "h20": 0.00, "h10": 0.02, "h5": 0.04}),
    ]
    for date, regime, returns in observations:
        for arm, capital_return in returns.items():
            rows.append(
                {
                    "date": date,
                    "run": autoresearch.REGIME_GATE_ARMS[arm],
                    "capital_c2c": capital_return,
                    "long_raw_c2c": capital_return * 2.0,
                    "short_raw_c2c": 0.0,
                    "regime": regime,
                }
            )
    return pd.DataFrame(rows)


class RegimeGateExperimentTests(unittest.TestCase):
    def test_development_sweep_enumerates_every_regime_arm_mapping(self):
        panel = autoresearch._regime_gate_arm_panel(synthetic_weekly_returns())
        sweep, champions, champion_years = autoresearch._evaluate_regime_mapping_sweep(panel)

        self.assertEqual(len(sweep), 4**2)
        self.assertEqual(set(sweep["arm_normal"]), set(autoresearch.REGIME_GATE_ARMS))
        self.assertEqual(set(sweep["arm_grind_up"]), set(autoresearch.REGIME_GATE_ARMS))
        self.assertIn("Best development CAGR", set(champions["objective"]))
        self.assertEqual(set(champion_years["year"]), {2009, 2010, 2011})

    def test_saved_experiment_has_fixed_and_development_mapping_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = autoresearch.save_regime_gate_experiments(
                synthetic_weekly_returns(),
                Path(tmpdir),
                random_sims=20,
            )

            scorecard = outputs["regime_gate_scorecard"]
            expected = {
                "Fixed h5",
                "Fixed top30",
                "Regime rule: h5 normal/crisis, top30 grind-up/squeeze",
            }
            self.assertTrue(expected.issubset(set(scorecard["rule"])))
            self.assertTrue((Path(tmpdir) / "regime_gate_scorecard.csv").exists())
            self.assertTrue((Path(tmpdir) / "regime_gate_paired_bootstrap.csv").exists())
            self.assertTrue((Path(tmpdir) / "regime_mapping_development_sweep.csv").exists())
            self.assertEqual(len(outputs["regime_mapping_sweep"]), 4**2)


if __name__ == "__main__":
    unittest.main()
