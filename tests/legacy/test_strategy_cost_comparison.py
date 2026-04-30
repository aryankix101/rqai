import unittest
import tempfile
from pathlib import Path

import pandas as pd

from archive.legacy_experiments import run_strategy_cost_comparison as scc


class StrategyCostComparisonTests(unittest.TestCase):
    def test_det_cost_overlay_applies_turnover_costs(self):
        pnl_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-16", "2024-01-30"]),
                "gross_daily_ret_bps": [100.0, 50.0, 75.0],
                "daily_ret_bps": [100.0, 50.0, 75.0],
                "turnover_total": [1.0, 0.5, 2.0],
                "period_length_days": [10, 10, 10],
            }
        )
        config = scc.DetConfig(
            name="det_test",
            display_name="DET test",
            weighting_mode="rank",
            top_n=100,
            candidate_pool_mode="raw",
        )

        summary, timeseries = scc.build_det_cost_overlay_outputs(
            pnl_df,
            config=config,
            variant_label="det_test_variant",
            score_dir=Path("dummy"),
            cost_bps_levels=(0.0, 5.0),
        )

        five_bps = summary[summary["cost_bps_per_side"] == 5.0].iloc[0]
        self.assertAlmostEqual(five_bps["total_cost_bps"], 17.5)
        self.assertEqual(five_bps["invalid_periods_at_or_below_minus_100pct"], 0)

        first_net = timeseries[
            (timeseries["cost_bps_per_side"] == 5.0)
            & (timeseries["date"] == pd.Timestamp("2024-01-02"))
        ]["period_return"].iloc[0]
        self.assertAlmostEqual(first_net, 0.0095)

    def test_det_cost_overlay_degrades_monotonically(self):
        pnl_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-16", "2024-01-30"]),
                "gross_daily_ret_bps": [100.0, 80.0, 60.0],
                "turnover_total": [1.0, 1.0, 1.0],
                "period_length_days": [10, 10, 10],
            }
        )
        config = scc.DetConfig(
            name="det_test",
            display_name="DET test",
            weighting_mode="rank",
            top_n=100,
            candidate_pool_mode="raw",
        )

        summary, _ = scc.build_det_cost_overlay_outputs(
            pnl_df,
            config=config,
            variant_label="det_test_variant",
            score_dir=Path("dummy"),
            cost_bps_levels=(0.0, 2.5, 5.0, 15.0),
        )
        ordered = summary.sort_values("cost_bps_per_side")
        returns = ordered["total_return_pct"].tolist()

        self.assertEqual(returns, sorted(returns, reverse=True))

    def test_comparison_charts_are_written_from_synthetic_outputs(self):
        pnl_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-16", "2024-01-30"]),
                "gross_daily_ret_bps": [100.0, 80.0, 60.0],
                "turnover_total": [1.0, 1.0, 1.0],
                "period_length_days": [10, 10, 10],
            }
        )
        config = scc.DetConfig(
            name="det_test",
            display_name="DET test",
            weighting_mode="rank",
            top_n=100,
            candidate_pool_mode="raw",
        )
        summary, timeseries = scc.build_det_cost_overlay_outputs(
            pnl_df,
            config=config,
            variant_label="det_test_variant",
            score_dir=Path("dummy"),
            cost_bps_levels=(0.0, 5.0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            scc.save_comparison_charts(summary, timeseries, output_dir, (0.0, 5.0))

            self.assertTrue((output_dir / "leaderboard_total_return_by_cost.png").exists())
            self.assertTrue((output_dir / "cumulative_returns_5bps.png").exists())
            self.assertTrue((output_dir / "drawdown_5bps.png").exists())


if __name__ == "__main__":
    unittest.main()
