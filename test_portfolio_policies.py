import unittest

import pandas as pd

from portfolio_policies import (
    HysteresisPolicy,
    ReviewPolicy,
    build_target_weights,
    compute_trade_decision,
)


class ReviewPolicyTests(unittest.TestCase):
    def test_static_policy_only_reviews_on_scheduled_signal_dates(self):
        policy = ReviewPolicy(
            mode="static",
            static_review_days=10,
            scheduled_signal_dates=frozenset({pd.Timestamp("2024-01-10")}),
        )
        due, reason = policy.should_review(
            signal_date=pd.Timestamp("2024-01-10"),
            current_regime="normal",
            previous_regime="normal",
            days_since_last_review=3,
        )
        self.assertTrue(due)
        self.assertEqual(reason, "scheduled_interval")

    def test_adaptive_policy_reviews_on_regime_deterioration(self):
        policy = ReviewPolicy(mode="adaptive", static_review_days=10)
        due, reason = policy.should_review(
            signal_date=pd.Timestamp("2024-01-10"),
            current_regime="crisis",
            previous_regime="normal",
            days_since_last_review=2,
        )
        self.assertTrue(due)
        self.assertEqual(reason, "regime_deterioration")


class HysteresisPolicyTests(unittest.TestCase):
    def test_long_rank_31_is_retained_with_rank_band(self):
        policy = HysteresisPolicy(
            mode="rank_band_asymmetric",
            top_n=30,
            bottom_n=30,
            long_retain_buffer=15,
            short_retain_buffer=10,
        )
        current_longs = ["A"] + [f"L{i}" for i in range(2, 31)]
        current_shorts = [f"S{i}" for i in range(1, 31)]
        long_candidates = [f"L{i}" for i in range(1, 40)]
        short_candidates = [f"S{i}" for i in range(1, 40)]
        long_rank_map = {ticker: idx + 1 for idx, ticker in enumerate(long_candidates)}
        long_rank_map["A"] = 31
        short_rank_map = {ticker: idx + 1 for idx, ticker in enumerate(short_candidates)}

        target_longs, _ = policy.target_book(
            current_longs=current_longs,
            current_shorts=current_shorts,
            candidate_longs=long_candidates,
            candidate_shorts=short_candidates,
            long_rank_map=long_rank_map,
            short_rank_map=short_rank_map,
        )
        self.assertIn("A", target_longs)

    def test_short_rank_41_is_not_retained_with_tighter_short_band(self):
        policy = HysteresisPolicy(
            mode="rank_band_asymmetric",
            top_n=30,
            bottom_n=30,
            long_retain_buffer=15,
            short_retain_buffer=10,
        )
        current_longs = [f"L{i}" for i in range(1, 31)]
        current_shorts = ["Z"] + [f"S{i}" for i in range(2, 31)]
        long_candidates = [f"L{i}" for i in range(1, 40)]
        short_candidates = [f"S{i}" for i in range(1, 40)]
        long_rank_map = {ticker: idx + 1 for idx, ticker in enumerate(long_candidates)}
        short_rank_map = {ticker: idx + 1 for idx, ticker in enumerate(short_candidates)}
        short_rank_map["Z"] = 41

        _, target_shorts = policy.target_book(
            current_longs=current_longs,
            current_shorts=current_shorts,
            candidate_longs=long_candidates,
            candidate_shorts=short_candidates,
            long_rank_map=long_rank_map,
            short_rank_map=short_rank_map,
        )
        self.assertNotIn("Z", target_shorts)

    def test_replacement_gap_blocks_small_improvement(self):
        policy = HysteresisPolicy(
            mode="replacement_gap",
            top_n=3,
            bottom_n=3,
            replacement_gap=5,
        )
        current_longs = ["A", "B", "C"]
        current_shorts = ["X", "Y", "Z"]
        long_candidates = ["A", "B", "D"]
        short_candidates = ["X", "Y", "Z"]
        long_rank_map = {"A": 1, "B": 2, "D": 3, "C": 7}
        short_rank_map = {"X": 1, "Y": 2, "Z": 3}

        target_longs, _ = policy.target_book(
            current_longs=current_longs,
            current_shorts=current_shorts,
            candidate_longs=long_candidates,
            candidate_shorts=short_candidates,
            long_rank_map=long_rank_map,
            short_rank_map=short_rank_map,
        )
        self.assertNotIn("D", target_longs)
        self.assertIn("C", target_longs)


class TradeDecisionTests(unittest.TestCase):
    def test_no_trade_when_weights_unchanged(self):
        decision = compute_trade_decision(
            old_weights={"A": 0.5, "B": -0.5},
            target_weights={"A": 0.5, "B": -0.5},
            cost_rate=0.0015,
        )
        self.assertFalse(decision.executed)
        self.assertEqual(decision.turnover_total, 0.0)
        self.assertEqual(decision.cost_bps, 0.0)

    def test_min_weight_delta_does_not_block_exits(self):
        decision = compute_trade_decision(
            old_weights={"DEAD": -0.003},
            target_weights={},
            cost_rate=0.0005,
            min_weight_delta=0.005,
        )

        self.assertTrue(decision.executed)
        self.assertNotIn("DEAD", decision.target_weights)
        self.assertEqual(decision.short_exits, 1)
        self.assertAlmostEqual(decision.turnover_total, 0.003)

    def test_min_weight_delta_does_not_block_side_flips(self):
        decision = compute_trade_decision(
            old_weights={"FLIP": -0.002},
            target_weights={"FLIP": 0.002},
            cost_rate=0.0005,
            min_weight_delta=0.005,
        )

        self.assertTrue(decision.executed)
        self.assertEqual(decision.target_weights["FLIP"], 0.002)
        self.assertEqual(decision.short_exits, 1)
        self.assertEqual(decision.long_entries, 1)


class WeightingModeTests(unittest.TestCase):
    def test_rank_weighting_tilts_toward_stronger_names(self):
        weights = build_target_weights(
            longs=["L1", "L2", "L3"],
            shorts=["S1", "S2", "S3"],
            weighting_mode="rank",
        )

        self.assertAlmostEqual(weights["L1"], 3 / 6)
        self.assertAlmostEqual(weights["L2"], 2 / 6)
        self.assertAlmostEqual(weights["L3"], 1 / 6)
        self.assertAlmostEqual(weights["S1"], -(3 / 6))
        self.assertAlmostEqual(weights["S2"], -(2 / 6))
        self.assertAlmostEqual(weights["S3"], -(1 / 6))
        self.assertAlmostEqual(sum(weight for weight in weights.values() if weight > 0), 1.0)
        self.assertAlmostEqual(sum(weight for weight in weights.values() if weight < 0), -1.0)


if __name__ == "__main__":
    unittest.main()
