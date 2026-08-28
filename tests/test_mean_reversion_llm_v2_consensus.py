"""Test deterministic rank consensus over immutable V2 scorer caches.

The cases validate ranking, tie-breaking, cache compatibility checks, and the
offline portfolio artifacts produced by the consensus experiment.
"""

import json
import itertools
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import mean_reversion_llm_v2_consensus as consensus


class ConsensusRankingTests(unittest.TestCase):
    def test_midranks_tied_scores(self) -> None:
        ranks = consensus.midranks(pd.Series([90, 80, 80, 70]))
        self.assertEqual(ranks.tolist(), [1.0, 2.5, 2.5, 4.0])

    def test_average_rank_selects_consistent_name(self) -> None:
        rows = []
        ranks_by_run = {
            "run1": {"A": 2, "B": 1, "C": 3},
            "run2": {"A": 1, "B": 3, "C": 2},
            "run3": {"A": 2, "B": 3, "C": 1},
        }
        for run_id, ranks in ranks_by_run.items():
            for ticker, rank in ranks.items():
                rows.append(
                    {
                        "date": pd.Timestamp("2020-01-02"),
                        "side": "LONG",
                        "ticker": ticker,
                        "run_id": run_id,
                        "run_rank": rank,
                    }
                )
        result = consensus.aggregate_ranks(pd.DataFrame(rows), tuple(ranks_by_run), "test")
        self.assertEqual(result.iloc[0]["ticker"], "A")
        self.assertEqual(result.iloc[0]["average_rank"], 5 / 3)

    def test_deterministic_tiebreak_is_stable_and_side_specific(self) -> None:
        first = consensus.deterministic_tiebreak("2020-01-02", "LONG", "AAA")
        self.assertEqual(first, consensus.deterministic_tiebreak("2020-01-02", "LONG", "AAA"))
        self.assertNotEqual(first, consensus.deterministic_tiebreak("2020-01-02", "SHORT", "AAA"))

    def test_all_four_three_of_four_combinations_are_predeclared(self) -> None:
        combinations = list(itertools.combinations(("run1", "run2", "run3", "run4"), 3))
        self.assertEqual(len(combinations), 4)
        self.assertEqual(combinations[0], consensus.PRIMARY_RUNS)

    def test_short_selection_profit_has_positive_contribution(self) -> None:
        date = pd.Timestamp("2020-01-02")
        ranks = pd.DataFrame(
            [
                {
                    "date": date,
                    "side": side,
                    "ticker": f"{side[0]}{index:02d}",
                    "selected": index < consensus.BOOK_PER_SIDE,
                    "selection_rank": index + 1,
                    "average_rank": index + 1.0,
                }
                for side in ("LONG", "SHORT")
                for index in range(60)
            ]
        )
        outcomes = pd.DataFrame(
            [
                {
                    "date": date,
                    "side": side,
                    "pool_side": side,
                    "ticker": f"{side[0]}{index:02d}",
                    "fwd_c2c": 0.01 if side == "LONG" else -0.01,
                    "fwd_o2o": 0.0,
                    "trail_ret": 0.0,
                }
                for side in ("LONG", "SHORT")
                for index in range(60)
            ]
        )
        weekly, _ = consensus.portfolio_from_ranks(ranks, outcomes)
        self.assertAlmostEqual(float(weekly.iloc[0]["long_contribution"]), 0.005)
        self.assertAlmostEqual(float(weekly.iloc[0]["short_contribution"]), 0.005)
        self.assertAlmostEqual(float(weekly.iloc[0]["capital_c2c"]), 0.01)

    def test_no_regime_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            path = cache / (
                "scores_v2nr-deadbeef_anon_paired-control-v2-"
                f"{consensus.EXPECTED_PROMPT_HASH}.json"
            )
            path.write_text(json.dumps({}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No-regime paired-control"):
                consensus._single_cache(root)


if __name__ == "__main__":
    unittest.main()
