import tempfile
import unittest
from pathlib import Path

import pandas as pd

import direction_experiment_analyzer as dea


class DirectionExperimentAnalyzerTests(unittest.TestCase):
    def test_parse_horizons_arg_supports_multi_horizon(self):
        self.assertEqual(dea.parse_horizons_arg("10,15,20", 10), [10, 15, 20])
        self.assertEqual(dea.parse_horizons_arg(None, 15), [15])

    def test_build_display_label_from_structured_fields(self):
        label = dea.build_display_label(
            {
                "variant_name": "det_dir_batch_10_15_20__fd_20__review_adaptive__...",
                "horizon": "20d_horizon",
                "review_mode": "adaptive",
                "review_normal_days": 20,
                "review_grind_up_days": 60,
                "review_slow_bleed_days": 20,
                "review_squeeze_days": 5,
                "review_crisis_days": 5,
                "review_unknown_days": 20,
                "hysteresis_mode": "combined",
                "long_retain_buffer": 20,
                "short_retain_buffer": 5,
                "replacement_gap": 5,
                "weighting_mode": "rank",
                "top_n": 200,
                "bottom_n": 200,
                "candidate_pool_mode": "llm_mirror",
                "top_n_magnitude_per_side": 200,
                "direction_threshold": 10.0,
            }
        )
        self.assertEqual(label, "20d | adp 20/60/20/5/5/20 | h20/5/5 | rank | 200x200 | mirror m200 d10")

    def test_build_cross_horizon_leaderboard_pivots_by_strategy(self):
        summary_df = pd.DataFrame(
            [
                {
                    "variant_name": "det_batch__fd_10__review_adaptive__hyst_combined__w_rank__top_200__bot_200__pool_raw",
                    "horizon": "10d_horizon",
                    "review_mode": "adaptive",
                    "hysteresis_mode": "combined",
                    "weighting_mode": "rank",
                    "top_n": 200,
                    "bottom_n": 200,
                    "candidate_pool_mode": "raw",
                    "total_return_pct": 12.0,
                    "sharpe": 0.3,
                    "total_cost_bps": 100.0,
                },
                {
                    "variant_name": "det_batch__fd_15__review_adaptive__hyst_combined__w_rank__top_200__bot_200__pool_raw",
                    "horizon": "15d_horizon",
                    "review_mode": "adaptive",
                    "hysteresis_mode": "combined",
                    "weighting_mode": "rank",
                    "top_n": 200,
                    "bottom_n": 200,
                    "candidate_pool_mode": "raw",
                    "total_return_pct": 18.0,
                    "sharpe": 0.4,
                    "total_cost_bps": 120.0,
                },
                {
                    "variant_name": "det_batch__fd_20__review_adaptive__hyst_combined__w_rank__top_200__bot_200__pool_raw",
                    "horizon": "20d_horizon",
                    "review_mode": "adaptive",
                    "hysteresis_mode": "combined",
                    "weighting_mode": "rank",
                    "top_n": 200,
                    "bottom_n": 200,
                    "candidate_pool_mode": "raw",
                    "total_return_pct": 15.0,
                    "sharpe": 0.35,
                    "total_cost_bps": 110.0,
                },
            ]
        )
        leaderboard = dea.build_cross_horizon_leaderboard(summary_df)
        self.assertEqual(len(leaderboard), 1)
        row = leaderboard.iloc[0]
        self.assertEqual(row["strategy_key"], "det_batch__review_adaptive__hyst_combined__w_rank__top_200__bot_200__pool_raw")
        self.assertAlmostEqual(row["total_return_pct_10d"], 12.0)
        self.assertAlmostEqual(row["total_return_pct_15d"], 18.0)
        self.assertAlmostEqual(row["total_return_pct_20d"], 15.0)
        self.assertAlmostEqual(row["avg_total_return_pct"], 15.0)
        self.assertAlmostEqual(row["avg_sharpe"], 0.35)
        self.assertEqual(row["n_horizons"], 3)

    def test_attach_plot_metadata_uniquifies_duplicate_display_labels(self):
        summary_df = pd.DataFrame(
            [
                {
                    "variant_name": "det_a",
                    "horizon": "10d_horizon",
                    "review_mode": "adaptive",
                    "review_normal_days": 20,
                    "review_grind_up_days": 60,
                    "review_slow_bleed_days": 20,
                    "review_squeeze_days": 5,
                    "review_crisis_days": 5,
                    "review_unknown_days": 20,
                    "hysteresis_mode": "combined",
                    "long_retain_buffer": 20,
                    "short_retain_buffer": 5,
                    "replacement_gap": 5,
                    "weighting_mode": "rank",
                    "top_n": 200,
                    "bottom_n": 200,
                    "candidate_pool_mode": "raw",
                    "total_return_pct": 1.0,
                    "sharpe": 0.1,
                },
                {
                    "variant_name": "det_b",
                    "horizon": "10d_horizon",
                    "review_mode": "adaptive",
                    "review_normal_days": 20,
                    "review_grind_up_days": 60,
                    "review_slow_bleed_days": 20,
                    "review_squeeze_days": 5,
                    "review_crisis_days": 5,
                    "review_unknown_days": 20,
                    "hysteresis_mode": "combined",
                    "long_retain_buffer": 20,
                    "short_retain_buffer": 5,
                    "replacement_gap": 5,
                    "weighting_mode": "rank",
                    "top_n": 200,
                    "bottom_n": 200,
                    "candidate_pool_mode": "raw",
                    "total_return_pct": 0.5,
                    "sharpe": 0.2,
                },
            ]
        )
        labeled = dea.attach_plot_metadata(summary_df)
        self.assertEqual(labeled["plot_id"].tolist(), ["S01", "S02"])
        self.assertEqual(
            labeled["display_label"].tolist(),
            [
                "10d | adp 20/60/20/5/5/20 | h20/5/5 | rank | 200x200 | raw",
                "10d | adp 20/60/20/5/5/20 | h20/5/5 | rank | 200x200 | raw #2",
            ],
        )

    def test_discover_and_summarize_direction_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            score_dir = base / "det_example" / "10d_horizon" / "direction_score"
            score_dir.mkdir(parents=True)

            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                    "daily_ret_bps": [10.0, -5.0, 20.0],
                    "gross_daily_ret_bps": [12.0, -4.0, 22.0],
                    "cost_bps_applied": [2.0, 1.0, 2.0],
                }
            ).to_csv(score_dir / "factor_pnl.csv", index=False)

            pd.DataFrame(
                {
                    "executed": [True, False],
                    "turnover_total": [2.0, 0.0],
                    "replacements": [30, 0],
                }
            ).to_csv(score_dir / "review_log.csv", index=False)

            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-02"],
                    "decile": [1, 10],
                    "mean_ret": [1.0, 4.0],
                    "std_ret": [0.5, 0.5],
                    "n_stocks": [10, 10],
                }
            ).to_csv(score_dir / "decile_analysis.csv", index=False)

            found = dea.discover_direction_runs(base, [10], [], [], include_llm=False)
            self.assertEqual(found, [score_dir])

            summary, daily_ret = dea.summarize_run(score_dir)
            self.assertEqual(summary["variant_name"], "det_example")
            self.assertAlmostEqual(summary["total_cost_bps"], 5.0)
            self.assertAlmostEqual(summary["mean_decile_spread_pct"], 3.0)
            self.assertEqual(len(daily_ret), 3)

    def test_summarize_run_tolerates_empty_optional_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            score_dir = base / "det_empty_optional" / "15d_horizon" / "direction_score"
            score_dir.mkdir(parents=True)

            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "daily_ret_bps": [5.0, -2.0],
                    "cost_bps_applied": [1.0, 1.0],
                }
            ).to_csv(score_dir / "factor_pnl.csv", index=False)

            (score_dir / "review_log.csv").write_text("", encoding="utf-8")
            (score_dir / "decile_analysis.csv").write_text("", encoding="utf-8")

            summary, daily_ret = dea.summarize_run(score_dir)
            self.assertTrue(pd.isna(summary["mean_decile_spread_pct"]))
            self.assertEqual(summary["n_reviews"], 0)
            self.assertEqual(len(daily_ret), 2)


if __name__ == "__main__":
    unittest.main()
