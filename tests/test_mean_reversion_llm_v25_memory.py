"""Test leakage-safe construction and integration of V2.5 memory features.

The cases cover historical-neighbor selection, memory summaries, and the
feature columns passed from the V1 data utilities into V2.5 prompts.
"""

import unittest

import numpy as np
import pandas as pd

import mean_reversion_llm_v1 as v1
import mean_reversion_llm_v25_memory as v25m


def _feature_row(ticker: str, date_idx: int, ticker_idx: int) -> dict[str, float | str]:
    trail = -10.0 + date_idx * 0.2 + ticker_idx * 0.05
    return {
        "Ticker": ticker,
        "MRL": 90.0,
        "MRS": 10.0,
        "Trail5": trail,
        "MRSig": -trail,
        "RSI5": 20.0 + ticker_idx,
        "Rng55": 0.1 + ticker_idx * 0.02,
        "Vel/V": -0.5 + date_idx * 0.01,
        "%MA100": -5.0 + ticker_idx * 0.1,
        "52WH": -20.0 + date_idx * 0.2,
        "Mom3m": -3.0 + ticker_idx * 0.1,
        "AutoC": -0.1 + date_idx * 0.01,
        "ADX": 25.0 + ticker_idx,
        "VolPctl": 60.0 + date_idx,
    }


def _memory_fixture(regime_prefix: str = "regime"):
    dates = pd.bdate_range("2020-01-06", periods=36)
    signal_dates = [dates[idx] for idx in range(0, 31, 5)]
    pools = {}
    all_tickers = {"L_SHARED", "S_SHARED"}
    for date_idx in range(len(signal_dates)):
        all_tickers.update(f"L{date_idx}_{idx}" for idx in range(4))
        all_tickers.update(f"S{date_idx}_{idx}" for idx in range(4))
    close = pd.DataFrame(100.0, index=dates, columns=sorted(all_tickers))

    for date_idx, signal_date in enumerate(signal_dates):
        exit_date = dates[int(dates.get_loc(signal_date)) + 5]
        long_tickers = ["L_SHARED", *[f"L{date_idx}_{idx}" for idx in range(4)]]
        short_tickers = ["S_SHARED", *[f"S{date_idx}_{idx}" for idx in range(4)]]
        for ticker in long_tickers[1:]:
            close.at[signal_date, ticker] = 100.0
            close.at[exit_date, ticker] = 101.0
        for ticker in short_tickers[1:]:
            close.at[signal_date, ticker] = 100.0
            close.at[exit_date, ticker] = 99.0
        close.at[signal_date, "L_SHARED"] = 100.0 * (1.01 ** date_idx)
        close.at[exit_date, "L_SHARED"] = 100.0 * (1.01 ** (date_idx + 1))
        close.at[signal_date, "S_SHARED"] = 100.0 * (0.99 ** date_idx)
        close.at[exit_date, "S_SHARED"] = 100.0 * (0.99 ** (date_idx + 1))
        tickers = long_tickers + short_tickers
        features = {
            ticker: _feature_row(ticker, date_idx, ticker_idx)
            for ticker_idx, ticker in enumerate(tickers)
        }
        pool = v1.CandidatePool(
            signal_date=signal_date,
            lookback_date=dates[max(0, int(dates.get_loc(signal_date)) - 5)],
            exit_date=exit_date,
            entry_open_date=None,
            exit_open_date=None,
            valid_tickers=tickers,
            long_tickers=long_tickers,
            short_tickers=short_tickers,
            trail_by_ticker={ticker: -0.1 if ticker in long_tickers else 0.1 for ticker in tickers},
            signal_by_ticker={ticker: 0.1 if ticker in long_tickers else -0.1 for ticker in tickers},
            feature_rows=features,
            universe_hash=f"pool-hash-{date_idx}",
            regime=f"{regime_prefix}-{date_idx % 2}",
        )
        pools[f"{signal_date.date()}|pool5"] = pool
    return pools, close, signal_dates


class V25BalancedOutcomeMemoryTests(unittest.TestCase):
    def test_memory_is_past_only_date_balanced_and_excludes_same_ticker(self):
        pools, close, signal_dates = _memory_fixture()
        build = v25m.build_balanced_outcome_memory(pools, close)
        last_date = signal_dates[-1]
        row = build.candidate_memory[
            (build.candidate_memory["signal_date"] == last_date)
            & (build.candidate_memory["ticker"] == "L_SHARED")
        ].iloc[0]
        neighbors = build.neighbor_audit[
            (build.neighbor_audit["query_signal_date"] == last_date)
            & (build.neighbor_audit["query_ticker"] == "L_SHARED")
        ]

        self.assertEqual(int(row["hist_n"]), 20)
        self.assertGreaterEqual(int(row["hist_date_n"]), 5)
        self.assertNotIn("L_SHARED", set(neighbors["neighbor_ticker"]))
        self.assertLessEqual(int(neighbors.groupby("neighbor_signal_date").size().max()), 4)
        self.assertTrue((pd.to_datetime(neighbors["neighbor_exit_date"]) <= last_date).all())

    def test_short_outcomes_are_side_adjusted(self):
        pools, close, signal_dates = _memory_fixture()
        build = v25m.build_balanced_outcome_memory(pools, close)
        rows = build.candidate_memory[
            (build.candidate_memory["signal_date"] == signal_dates[-1])
            & (build.candidate_memory["side"] == "SHORT")
        ]

        self.assertTrue((pd.to_numeric(rows["hist_ret_pct"], errors="coerce") > 0.0).all())
        self.assertTrue((pd.to_numeric(rows["hist_win_pct"], errors="coerce") == 100.0).all())

    def test_future_outcome_changes_cannot_change_earlier_memory(self):
        pools, close, signal_dates = _memory_fixture()
        original = v25m.build_balanced_outcome_memory(pools, close)
        changed = close.copy()
        future_pool = pools[f"{signal_dates[-1].date()}|pool5"]
        changed.loc[future_pool.exit_date, future_pool.pool_tickers] = 1_000.0
        rebuilt = v25m.build_balanced_outcome_memory(pools, changed)
        cutoff = signal_dates[-2]
        columns = ["signal_date", "ticker", "hist_n", "hist_win_pct", "hist_ret_pct", "hist_date_n"]
        left = original.candidate_memory.loc[original.candidate_memory["signal_date"] <= cutoff, columns]
        right = rebuilt.candidate_memory.loc[rebuilt.candidate_memory["signal_date"] <= cutoff, columns]

        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))

    def test_regime_does_not_affect_v25_memory(self):
        pools_a, close, _ = _memory_fixture("a")
        pools_b, _, _ = _memory_fixture("completely-different")
        build_a = v25m.build_balanced_outcome_memory(pools_a, close)
        build_b = v25m.build_balanced_outcome_memory(pools_b, close)
        columns = ["signal_date", "ticker", "side", "hist_n", "hist_win_pct", "hist_ret_pct", "hist_date_n"]

        pd.testing.assert_frame_equal(
            build_a.candidate_memory[columns].reset_index(drop=True),
            build_b.candidate_memory[columns].reset_index(drop=True),
        )
        self.assertEqual(v25m.memory_spec()["regime_policy"], "not used")

    def test_memory_hash_changes_when_memory_values_change(self):
        pools, close, _ = _memory_fixture()
        build = v25m.build_balanced_outcome_memory(pools, close)
        pool = next(reversed(build.pools_by_key.values()))
        rows = {ticker: dict(values) for ticker, values in pool.feature_rows.items()}
        original = v25m.hash_memory_enriched_pool(
            "base", pool.long_tickers, pool.short_tickers, rows
        )
        rows[pool.long_tickers[0]]["HistRet"] += 0.01
        changed = v25m.hash_memory_enriched_pool(
            "base", pool.long_tickers, pool.short_tickers, rows
        )

        self.assertNotEqual(original, changed)
        self.assertNotIn("HistScore", v25m.V25_MEMORY_COLUMNS)


if __name__ == "__main__":
    unittest.main()
