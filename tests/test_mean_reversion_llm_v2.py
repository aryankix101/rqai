import tempfile
import unittest
from pathlib import Path

import pandas as pd

import mean_reversion_llm_v1 as v1
import mean_reversion_llm_v2 as v2
import run_backtest_v3 as rb
from tests.test_mean_reversion_llm_v1 import _make_histories, _membership


def _v2_config(start: str, end: str, tmpdir: str) -> rb.BenchmarkConfig:
    return rb.BenchmarkConfig(
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        output_dir=Path(tmpdir),
        membership_path=Path("unused.csv"),
        cost_bps_levels=(0.0, 2.5, 5.0),
        cadence_mode="trading_days",
        signal_lookback_days=5,
        rebalance_days=5,
        holding_days=5,
        min_universe_size=10,
    )


class MeanReversionLLMV2Tests(unittest.TestCase):
    def test_v2_reuses_standard_v1_prompt_contract(self):
        self.assertEqual(v2.V2_PROMPT_STYLE, v1.PROMPT_STYLE_V1_SCORE)
        self.assertEqual(v2.V2_PROMPT_VERSION, v1.PROMPT_VERSION)
        self.assertEqual(
            v1.system_prompt_for_style(v2.V2_PROMPT_STYLE),
            v1.V1_SYSTEM_PROMPT,
        )

    def test_v2_defaults_are_gross_only(self):
        args = v2.parse_args([])

        self.assertEqual(v2.DEFAULT_COST_BPS_LEVELS, (0.0,))
        self.assertEqual(tuple(args.cost_bps_levels), (0.0,))
        self.assertEqual(args.schedule, "anchors")
        self.assertEqual(tuple(args.pool_sizes), (60,))
        self.assertEqual(tuple(args.book_sizes), (30,))
        self.assertEqual(args.llm_pool_size, 60)

    def test_variant_grid_replays_valid_book_sizes_only(self):
        default_variants = v2.build_variant_grid((60, 150), (30, 75), (0, 1))
        default_labels = {variant.label for variant in default_variants}
        self.assertIn("pool60_pos30_anchor0", default_labels)
        self.assertIn("pool150_pos75_anchor1", default_labels)
        self.assertNotIn("pool60_pos75_anchor0", default_labels)

        friday_variants = v2.build_variant_grid((60, 150), (30, 75), (0, 1), schedule="friday")
        friday_labels = {variant.label for variant in friday_variants}

        self.assertIn("pool60_pos30_friday", friday_labels)
        self.assertIn("pool150_pos30_friday", friday_labels)
        self.assertIn("pool150_pos75_friday", friday_labels)
        self.assertNotIn("pool60_pos75_friday", friday_labels)

        variants = v2.build_variant_grid((60, 150), (30, 75), (0, 1), schedule="anchors")
        labels = {variant.label for variant in variants}

        self.assertIn("pool60_pos30_anchor0", labels)
        self.assertIn("pool150_pos30_anchor0", labels)
        self.assertIn("pool150_pos75_anchor1", labels)
        self.assertNotIn("pool60_pos75_anchor0", labels)

    def test_v2_gross_comparison_prefers_anchor_average_zero_cost_c2c(self):
        scorecard = pd.DataFrame(
            [
                {
                    "pool_per_side": 60,
                    "positions_per_side": 30,
                    "anchor_offset": 0,
                    "is_anchor_average": False,
                    "scorer": "trail5",
                    "run": "Baseline-Pool",
                    "execution_model": "C2C",
                    "cost_bps": 0.0,
                    "sharpe": 1.0,
                    "cagr_pct": 10.0,
                    "total_return_pct": 50.0,
                    "max_drawdown_pct": -8.0,
                    "hit_rate_pct": 55.0,
                    "ic_mean": 0.02,
                    "n_periods": 10,
                },
                {
                    "pool_per_side": 60,
                    "positions_per_side": 30,
                    "anchor_offset": "average",
                    "is_anchor_average": True,
                    "scorer": "trail5",
                    "run": "Baseline-Pool",
                    "execution_model": "C2C",
                    "cost_bps": 0.0,
                    "sharpe": 1.2,
                    "cagr_pct": 12.0,
                    "total_return_pct": 60.0,
                    "max_drawdown_pct": -7.0,
                    "hit_rate_pct": 56.0,
                    "ic_mean": 0.03,
                    "n_periods": 10,
                },
                {
                    "pool_per_side": 60,
                    "positions_per_side": 30,
                    "anchor_offset": "average",
                    "is_anchor_average": True,
                    "scorer": "trail5",
                    "run": "Baseline-Pool",
                    "execution_model": "C2C",
                    "cost_bps": 5.0,
                    "sharpe": 0.4,
                    "cagr_pct": 4.0,
                    "total_return_pct": 15.0,
                    "max_drawdown_pct": -9.0,
                    "hit_rate_pct": 50.0,
                    "ic_mean": 0.03,
                    "n_periods": 10,
                },
            ]
        )

        comparison = v2.v2_gross_comparison_rows(scorecard)

        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison.iloc[0]["family"], "V2")
        self.assertEqual(comparison.iloc[0]["anchor_offset"], "average")
        self.assertEqual(float(comparison.iloc[0]["cost_bps"]), 0.0)
        self.assertEqual(float(comparison.iloc[0]["sharpe"]), 1.2)
        self.assertEqual(comparison.iloc[0]["harness"], "v2 multi-anchor replay harness")
        self.assertEqual(comparison.iloc[0]["prompt_schema"], "none: deterministic Trail5 baseline")
        self.assertEqual(comparison.iloc[0]["candidate_pool"], "60 biggest 5-day losers + 60 biggest 5-day winners")
        self.assertIn("ADV20", comparison.iloc[0]["liquidity_filters"])

    def test_v2_gross_comparison_labels_friday_schedule(self):
        scorecard = pd.DataFrame(
            [
                {
                    "pool_per_side": 150,
                    "positions_per_side": 75,
                    "anchor_offset": "Friday",
                    "is_anchor_average": False,
                    "score_pool_per_side": 150,
                    "scorer": "llm_score",
                    "run": "LLM-Pool v2",
                    "execution_model": "C2C",
                    "cost_bps": 0.0,
                    "sharpe": 1.1,
                    "cagr_pct": 17.0,
                    "total_return_pct": 100.0,
                    "max_drawdown_pct": -12.0,
                    "hit_rate_pct": 55.0,
                    "ic_mean": 0.03,
                    "n_periods": 251,
                }
            ]
        )

        comparison = v2.v2_gross_comparison_rows(scorecard)

        self.assertEqual(comparison.iloc[0]["basis"], "Friday gross C2C")
        self.assertEqual(comparison.iloc[0]["harness"], "v2 Friday replay harness")
        self.assertEqual(comparison.iloc[0]["cadence"], "Friday-only weekly schedule")
        self.assertEqual(comparison.iloc[0]["accounting"], "gross Friday-to-Friday C2C")
        self.assertEqual(comparison.iloc[0]["prompt_schema"], "unchanged v1 0-100 score prompt")
        self.assertEqual(comparison.iloc[0]["score_source"], "LLM scored 150/150; row pool 150/150")
        self.assertIn("pool 150", comparison.iloc[0]["replay_cache"])

    def test_v2_can_slice_broad_llm_scores_for_smaller_pool(self):
        tickers = [f"T{i:02d}" for i in range(16)]
        histories = _make_histories(tickers, start="2019-01-01", end="2019-06-30")
        market = rb.build_market_frames(histories)
        features = v1.build_feature_frames(histories)
        membership = _membership(tickers)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _v2_config("2019-03-01", "2019-06-01", tmpdir)
            context = v2.build_v2_pool_context(market)
            signal_idx = int(context.trading_dates.get_loc(pd.Timestamp("2019-03-15")))
            pool2 = v2.build_candidate_pool_for_index(
                market,
                features,
                membership,
                spy_df=None,
                signal_idx=signal_idx,
                pool_per_side=2,
                config=config,
                context=context,
            )
            pool4 = v2.build_candidate_pool_for_index(
                market,
                features,
                membership,
                spy_df=None,
                signal_idx=signal_idx,
                pool_per_side=4,
                config=config,
                context=context,
            )

        assert pool2 is not None
        assert pool4 is not None
        broad_entry = {
            "long_scores": [{"ticker": ticker, "score": 80} for ticker in pool4.long_tickers],
            "short_scores": [{"ticker": ticker, "score": 70} for ticker in pool4.short_tickers],
        }
        validated = v2.validate_llm_scores_for_v2_pool(
            {v2.llm_score_storage_key(pool2, 4): broad_entry},
            pool2,
            pool_per_side=2,
            llm_source_pool_per_side=4,
        )

        self.assertEqual({row["ticker"] for row in validated["long_scores"]}, {ticker.upper() for ticker in pool2.long_tickers})
        self.assertEqual({row["ticker"] for row in validated["short_scores"]}, {ticker.upper() for ticker in pool2.short_tickers})

    def test_legacy_gross_comparison_prefers_anchor_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame(
                [
                    {
                        "run": "Anchor-Monday",
                        "anchor": "Monday",
                        "execution_model": "C2C",
                        "cost_bps": 0.0,
                        "n_periods": 10,
                        "sharpe": 0.5,
                        "cagr_pct": 5.0,
                        "total_return_pct": 8.0,
                        "max_drawdown_pct": -3.0,
                        "hit_rate_pct": 50.0,
                        "ic_mean": 0.01,
                    },
                    {
                        "run": "Anchor-Friday",
                        "anchor": "Friday",
                        "execution_model": "C2C",
                        "cost_bps": 0.0,
                        "n_periods": 10,
                        "sharpe": 0.9,
                        "cagr_pct": 9.0,
                        "total_return_pct": 15.0,
                        "max_drawdown_pct": -4.0,
                        "hit_rate_pct": 55.0,
                        "ic_mean": 0.02,
                    },
                    {
                        "run": "Anchor-Average metrics",
                        "anchor": "Average",
                        "execution_model": "C2C",
                        "cost_bps": 0.0,
                        "n_periods": 10,
                        "sharpe": 0.7,
                        "cagr_pct": 7.0,
                        "total_return_pct": 11.0,
                        "max_drawdown_pct": -5.0,
                        "hit_rate_pct": 53.0,
                        "ic_mean": 0.03,
                    },
                    {
                        "run": "Anchor-Friday",
                        "anchor": "Friday",
                        "execution_model": "C2C",
                        "cost_bps": 5.0,
                        "n_periods": 10,
                        "sharpe": 0.1,
                        "cagr_pct": 1.0,
                        "total_return_pct": 2.0,
                        "max_drawdown_pct": -6.0,
                        "hit_rate_pct": 45.0,
                        "ic_mean": 0.02,
                    },
                ]
            ).to_csv(root / "anchor_summary.csv", index=False)

            comparison = v2.legacy_gross_comparison_rows(root)

        self.assertEqual(set(comparison["anchor_offset"]), {"Friday", "Average"})
        self.assertEqual(set(comparison["cost_bps"]), {0.0})
        self.assertIn("Legacy Det Raw Friday", set(comparison["run"]))
        self.assertIn("Legacy Det Raw Anchor Average", set(comparison["run"]))
        self.assertEqual(set(comparison["harness"]), {"legacy raw anchor replay"})
        self.assertEqual(set(comparison["book"]), {"30/30 equal-weight long-short"})
        self.assertIn("price >= $5", comparison.iloc[0]["liquidity_filters"])

    def test_legacy_factor_rows_use_fixed_window_selection_accounting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            score_dir = root / "llm_raw_confidence_score"
            score_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"rebalance_date": "2020-01-03", "position_type": "LONG", "realized_return_pct": 10.0},
                    {"rebalance_date": "2020-01-03", "position_type": "SHORT", "realized_return_pct": 5.0},
                    {"rebalance_date": "2020-01-10", "position_type": "LONG", "realized_return_pct": 0.0},
                    {"rebalance_date": "2020-01-10", "position_type": "SHORT", "realized_return_pct": -10.0},
                ]
            ).to_csv(score_dir / "portfolio_selections.csv", index=False)
            pd.DataFrame(
                [
                    {"score_type": "llm_raw_confidence_score", "ic_mean": 0.12},
                ]
            ).to_csv(root / "compact_leaderboard.csv", index=False)
            pd.DataFrame(
                [
                    {"turnover_total": 2.0},
                    {"turnover_total": 1.0},
                ]
            ).to_csv(score_dir / "review_log.csv", index=False)

            comparison = v2.legacy_factor_gross_comparison_rows(root, include_deterministic=False)

        self.assertEqual(len(comparison), 1)
        row = comparison.iloc[0]
        self.assertEqual(row["execution_model"], "C2C")
        self.assertEqual(row["accounting"], "gross fixed 5-trading-day C2C")
        self.assertEqual(row["harness"], "legacy factor selections replay")
        self.assertEqual(row["n_periods"], 2)
        self.assertAlmostEqual(float(row["total_return_pct"]), 3.5)
        self.assertAlmostEqual(float(row["ic_mean"]), 0.12)
        self.assertAlmostEqual(float(row["avg_turnover_total"]), 1.5)

    def test_llm_cache_storage_key_separates_pool_sizes_on_same_date(self):
        tickers = [f"T{i:02d}" for i in range(12)]
        histories = _make_histories(tickers, start="2019-01-01", end="2019-06-30")
        market = rb.build_market_frames(histories)
        features = v1.build_feature_frames(histories)
        membership = _membership(tickers)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _v2_config("2019-03-01", "2019-06-01", tmpdir)
            context = v2.build_v2_pool_context(market)
            signal_idx = int(context.trading_dates.get_loc(pd.Timestamp("2019-03-15")))
            pool2 = v2.build_candidate_pool_for_index(
                market,
                features,
                membership,
                spy_df=None,
                signal_idx=signal_idx,
                pool_per_side=2,
                config=config,
                context=context,
            )
            pool3 = v2.build_candidate_pool_for_index(
                market,
                features,
                membership,
                spy_df=None,
                signal_idx=signal_idx,
                pool_per_side=3,
                config=config,
                context=context,
            )

        assert pool2 is not None
        assert pool3 is not None
        self.assertNotEqual(
            v2.llm_score_storage_key(pool2, 2),
            v2.llm_score_storage_key(pool3, 3),
        )

    def test_friday_deterministic_replay_matches_v1(self):
        tickers = [f"T{i:02d}" for i in range(24)]
        histories = _make_histories(tickers, start="2019-01-01", end="2019-09-30")
        market = rb.build_market_frames(histories)
        features = v1.build_feature_frames(histories)
        membership = _membership(tickers)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _v2_config("2019-04-01", "2019-08-31", tmpdir)
            v1_weekly, _, _, _ = v1.run_pool_strategy(
                market,
                features,
                membership,
                spy_df=None,
                config=config,
                pool_per_side=5,
                positions_per_side=2,
                score_provider="trail5",
            )

            v2_weekly, _, _, _ = v2.run_v2_pool_strategy(
                market,
                features,
                membership,
                spy_df=None,
                config=config,
                pool_per_side=5,
                positions_per_side=2,
                anchor_offset="Friday",
                score_provider="trail5",
            )

        columns = ["long_ret_c2c", "short_ret_c2c", "long_ret_o2o", "short_ret_o2o", "ic_c2c", "ic_o2o"]
        pd.testing.assert_index_equal(v1_weekly.index, v2_weekly.index)
        pd.testing.assert_frame_equal(v1_weekly[columns], v2_weekly[columns])


if __name__ == "__main__":
    unittest.main()
