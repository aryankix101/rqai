import unittest

from mean_reversion_llm import (
    _build_mean_reversion_candidate_views,
    _format_universe_for_prompt,
    _validate_signal_stage_output,
    build_portfolio_system_prompt,
)


class MeanReversionPromptTests(unittest.TestCase):
    def test_prompt_uses_full_universe_mean_reversion_language(self):
        prompt = build_portfolio_system_prompt(turnover_aware=False)
        self.assertIn("MEAN REVERSION DISCOVERY STAGE", prompt)
        self.assertIn("Rank the full universe cross-sectionally", prompt)
        self.assertIn("mean_reversion_signal = -(trailing 5-trading-day return)", prompt)

        universe = {
            "AAA": {
                "direction_score": -25.0,
                "magnitude_score": 90.0,
                "trail_return_5d": -0.12,
                "mean_reversion_signal": 0.12,
                "indicators": {
                    "rsi_5d": 0.05,
                    "range_pos_55d": 0.10,
                    "vel_10d_over_vol": 0.08,
                    "pct_from_ma100": 0.12,
                    "pct_from_52w_high": 0.09,
                    "mom_3m": 0.30,
                    "autocorr_5d": 0.20,
                    "adx_14d": 0.40,
                    "vol_percentile_504d": 0.65,
                },
            },
            "BBB": {
                "direction_score": 30.0,
                "magnitude_score": 88.0,
                "trail_return_5d": 0.11,
                "mean_reversion_signal": -0.11,
                "indicators": {
                    "rsi_5d": 0.95,
                    "range_pos_55d": 0.92,
                    "vel_10d_over_vol": 0.91,
                    "pct_from_ma100": 0.90,
                    "pct_from_52w_high": 0.96,
                    "mom_3m": 0.78,
                    "autocorr_5d": 0.61,
                    "adx_14d": 0.72,
                    "vol_percentile_504d": 0.58,
                },
            },
        }

        prompt_body = _format_universe_for_prompt(
            universe=universe,
            previous_portfolio=None,
            regime="normal",
            rebalance_id="w_00001",
            turnover_aware=False,
            top_n_magnitude=10,
        )
        self.assertIn("Broad full-universe mean-reversion ranking", prompt_body)
        self.assertIn("OVERSOLD LONG CANDIDATES", prompt_body)
        self.assertIn("OVERBOUGHT SHORT CANDIDATES", prompt_body)
        self.assertIn("single-stage mean-reversion alpha discovery", prompt_body)
        self.assertIn("Trail5", prompt_body)
        self.assertNotIn("MAGNITUDE DISTRIBUTION:", prompt_body)
        self.assertNotIn(" Dir ", prompt_body)
        self.assertNotIn(" Mag ", prompt_body)

    def test_candidate_views_do_not_apply_direction_thresholds(self):
        universe = {
            "LOSER": {
                "direction_score": -35.0,
                "magnitude_score": 95.0,
                "trail_return_5d": -0.22,
                "mean_reversion_signal": 0.22,
                "indicators": {
                    "rsi_5d": 0.02,
                    "range_pos_55d": 0.03,
                    "vel_10d_over_vol": 0.04,
                    "pct_from_ma100": 0.05,
                    "pct_from_52w_high": 0.06,
                    "mom_3m": 0.20,
                    "autocorr_5d": 0.30,
                    "adx_14d": 0.40,
                    "vol_percentile_504d": 0.70,
                },
            },
            "WINNER": {
                "direction_score": 40.0,
                "magnitude_score": 94.0,
                "trail_return_5d": 0.19,
                "mean_reversion_signal": -0.19,
                "indicators": {
                    "rsi_5d": 0.98,
                    "range_pos_55d": 0.97,
                    "vel_10d_over_vol": 0.96,
                    "pct_from_ma100": 0.95,
                    "pct_from_52w_high": 0.94,
                    "mom_3m": 0.80,
                    "autocorr_5d": 0.60,
                    "adx_14d": 0.75,
                    "vol_percentile_504d": 0.55,
                },
            },
            "UP_BUT_OVERBOUGHT": {
                "direction_score": 18.0,
                "magnitude_score": 80.0,
                "trail_return_5d": 0.14,
                "mean_reversion_signal": -0.14,
                "indicators": {
                    "rsi_5d": 0.92,
                    "range_pos_55d": 0.91,
                    "vel_10d_over_vol": 0.88,
                    "pct_from_ma100": 0.93,
                    "pct_from_52w_high": 0.90,
                    "mom_3m": 0.74,
                    "autocorr_5d": 0.57,
                    "adx_14d": 0.68,
                    "vol_percentile_504d": 0.52,
                },
            },
            "DOWN_BUT_OVERSOLD": {
                "direction_score": -12.0,
                "magnitude_score": 82.0,
                "trail_return_5d": -0.17,
                "mean_reversion_signal": 0.17,
                "indicators": {
                    "rsi_5d": 0.08,
                    "range_pos_55d": 0.11,
                    "vel_10d_over_vol": 0.09,
                    "pct_from_ma100": 0.10,
                    "pct_from_52w_high": 0.12,
                    "mom_3m": 0.24,
                    "autocorr_5d": 0.33,
                    "adx_14d": 0.45,
                    "vol_percentile_504d": 0.67,
                },
            },
            "NEUTRAL": {
                "direction_score": 1.0,
                "magnitude_score": 30.0,
                "trail_return_5d": 0.01,
                "mean_reversion_signal": -0.01,
                "indicators": {
                    "rsi_5d": 0.50,
                    "range_pos_55d": 0.50,
                    "vel_10d_over_vol": 0.50,
                    "pct_from_ma100": 0.50,
                    "pct_from_52w_high": 0.50,
                    "mom_3m": 0.50,
                    "autocorr_5d": 0.50,
                    "adx_14d": 0.50,
                    "vol_percentile_504d": 0.50,
                },
            },
        }

        candidate_views = _build_mean_reversion_candidate_views(universe, top_n_candidates=3)
        long_tickers = [entry["ticker"] for entry in candidate_views["long_candidates"]]
        short_tickers = [entry["ticker"] for entry in candidate_views["short_candidates"]]

        self.assertIn("LOSER", long_tickers)
        self.assertIn("DOWN_BUT_OVERSOLD", long_tickers)
        self.assertIn("WINNER", short_tickers)
        self.assertIn("UP_BUT_OVERBOUGHT", short_tickers)

    def test_candidate_views_keep_prompt_sides_disjoint_even_in_small_universe(self):
        universe = {
            "AAA": {
                "direction_score": -15.0,
                "magnitude_score": 80.0,
                "trail_return_5d": -0.09,
                "mean_reversion_signal": 0.09,
                "indicators": {
                    "rsi_5d": 0.10,
                    "range_pos_55d": 0.12,
                    "vel_10d_over_vol": 0.11,
                    "pct_from_ma100": 0.13,
                    "pct_from_52w_high": 0.14,
                },
            },
            "BBB": {
                "direction_score": 18.0,
                "magnitude_score": 82.0,
                "trail_return_5d": 0.07,
                "mean_reversion_signal": -0.07,
                "indicators": {
                    "rsi_5d": 0.90,
                    "range_pos_55d": 0.88,
                    "vel_10d_over_vol": 0.87,
                    "pct_from_ma100": 0.86,
                    "pct_from_52w_high": 0.85,
                },
            },
            "CCC": {
                "direction_score": 0.0,
                "magnitude_score": 35.0,
                "trail_return_5d": 0.00,
                "mean_reversion_signal": 0.00,
                "indicators": {
                    "rsi_5d": 0.50,
                    "range_pos_55d": 0.50,
                    "vel_10d_over_vol": 0.50,
                    "pct_from_ma100": 0.50,
                    "pct_from_52w_high": 0.50,
                },
            },
        }

        candidate_views = _build_mean_reversion_candidate_views(universe, top_n_candidates=3)
        long_tickers = {entry["ticker"] for entry in candidate_views["long_candidates"]}
        short_tickers = {entry["ticker"] for entry in candidate_views["short_candidates"]}
        self.assertTrue(long_tickers.isdisjoint(short_tickers))

    def test_actual_trailing_return_signal_drives_candidate_ranking(self):
        universe = {
            "PROXY_PRETTY_BUT_WEAK_SIGNAL": {
                "direction_score": -30.0,
                "magnitude_score": 90.0,
                "trail_return_5d": -0.03,
                "mean_reversion_signal": 0.03,
                "indicators": {
                    "rsi_5d": 0.01,
                    "range_pos_55d": 0.02,
                    "vel_10d_over_vol": 0.03,
                    "pct_from_ma100": 0.04,
                    "pct_from_52w_high": 0.05,
                },
            },
            "ACTUAL_BIG_LOSER": {
                "direction_score": 5.0,
                "magnitude_score": 40.0,
                "trail_return_5d": -0.18,
                "mean_reversion_signal": 0.18,
                "indicators": {
                    "rsi_5d": 0.60,
                    "range_pos_55d": 0.55,
                    "vel_10d_over_vol": 0.58,
                    "pct_from_ma100": 0.52,
                    "pct_from_52w_high": 0.50,
                },
            },
            "ACTUAL_BIG_WINNER": {
                "direction_score": -5.0,
                "magnitude_score": 42.0,
                "trail_return_5d": 0.16,
                "mean_reversion_signal": -0.16,
                "indicators": {
                    "rsi_5d": 0.40,
                    "range_pos_55d": 0.45,
                    "vel_10d_over_vol": 0.42,
                    "pct_from_ma100": 0.48,
                    "pct_from_52w_high": 0.50,
                },
            },
        }

        candidate_views = _build_mean_reversion_candidate_views(universe, top_n_candidates=1)
        self.assertEqual(candidate_views["long_candidates"][0]["ticker"], "ACTUAL_BIG_LOSER")
        self.assertEqual(candidate_views["short_candidates"][0]["ticker"], "ACTUAL_BIG_WINNER")

    def test_exact_target_positions_are_enforced(self):
        valid_tickers = {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"}
        raw_output = {
            "long_candidates": [
                {"ticker": "AAA", "edge_score": 90, "rationale": "A"},
                {"ticker": "BBB", "edge_score": 89, "rationale": "B"},
                {"ticker": "CCC", "edge_score": 88, "rationale": "C"},
                {"ticker": "DDD", "edge_score": 87, "rationale": "D"},
                {"ticker": "EEE", "edge_score": 86, "rationale": "E"},
            ],
            "short_candidates": [
                {"ticker": "FFF", "edge_score": 90, "rationale": "F"},
                {"ticker": "GGG", "edge_score": 89, "rationale": "G"},
                {"ticker": "HHH", "edge_score": 88, "rationale": "H"},
                {"ticker": "III", "edge_score": 87, "rationale": "I"},
                {"ticker": "JJJ", "edge_score": 86, "rationale": "J"},
            ],
        }

        validated = _validate_signal_stage_output(
            raw_output,
            valid_tickers,
            target_positions_per_side=5,
        )
        self.assertEqual(len(validated["long_candidates"]), 5)
        self.assertEqual(len(validated["short_candidates"]), 5)


if __name__ == "__main__":
    unittest.main()
