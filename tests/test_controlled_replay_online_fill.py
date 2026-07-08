import argparse
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts import fill_llm_reversal_controlled_replay_scores as fill


def _candidate_grid() -> pd.DataFrame:
    rows = []
    for date in ["2020-01-03", "2020-01-10"]:
        for side, tickers in (("LONG", ["AAA", "BBB"]), ("SHORT", ["CCC", "DDD"])):
            for idx, ticker in enumerate(tickers, start=1):
                rows.append(
                    {
                        "rebalance_date": date,
                        "lookback_date": "2019-12-27",
                        "entry_date": date,
                        "exit_date": "2020-01-17",
                        "ticker": ticker,
                        "pool_side": side,
                        "side_rank_by_trail": idx,
                        "trail_ret": -0.05 if side == "LONG" else 0.05,
                        "deterministic_signal": 0.05 if side == "LONG" else -0.05,
                        "feature_MRL": float(100 - idx),
                        "feature_MRS": float(idx),
                    }
                )
    return pd.DataFrame(rows)


def _fake_pools_from_grid(candidate_grid: pd.DataFrame) -> list[SimpleNamespace]:
    pools = []
    grid = candidate_grid.copy()
    grid["rebalance_date"] = pd.to_datetime(grid["rebalance_date"]).dt.strftime("%Y-%m-%d")
    for date, date_rows in grid.groupby("rebalance_date"):
        pools.append(
            SimpleNamespace(
                signal_date=pd.Timestamp(date),
                long_tickers=date_rows[date_rows["pool_side"] == "LONG"]["ticker"].tolist(),
                short_tickers=date_rows[date_rows["pool_side"] == "SHORT"]["ticker"].tolist(),
            )
        )
    return pools


class ControlledReplayOnlineFillTests(unittest.TestCase):
    def test_candidate_grid_subset_uses_earliest_dates(self):
        grid = fill.candidate_grid_for_requested_dates(_candidate_grid(), max_dates=1)

        self.assertEqual(sorted(grid["rebalance_date"].unique()), ["2020-01-03"])

    def test_rebuilt_pool_validation_accepts_exact_grid(self):
        grid = _candidate_grid()
        pools = _fake_pools_from_grid(grid)

        audit = fill.validate_pools_against_grid(pools, grid)

        self.assertEqual(audit["status"], "valid")
        self.assertEqual(audit["row_count"], len(grid))

    def test_rebuilt_pool_validation_rejects_mismatch(self):
        grid = _candidate_grid()
        pools = _fake_pools_from_grid(grid)
        pools[0].long_tickers[0] = "ZZZ"

        with self.assertRaisesRegex(ValueError, "missing=1 unexpected=1"):
            fill.validate_pools_against_grid(pools, grid)

    def test_cache_path_defaults_to_pool60_direct_score_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                cache_path=None,
                cache_dir=Path(tmpdir),
                start="2009-01-01",
                end="2014-01-01",
                model="deepseek-reasoner",
                pool_per_side=60,
            )

            path = fill.cache_path_for_args(args)

        self.assertEqual(
            path.name,
            "llm_reversal_signal_v1_2009-01-01_2014-01-01_pool60_mdl-deepseek-reasoner.json",
        )

    def test_run_fill_requires_confirmation_before_pending_calls(self):
        grid = _candidate_grid()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            candidate_grid = tmp / "candidate_grid.csv"
            grid.to_csv(candidate_grid, index=False)
            args = argparse.Namespace(
                start="2009-01-01",
                end="2014-01-01",
                candidate_grid=candidate_grid,
                output_dir=tmp,
                cache_dir=tmp,
                cache_path=tmp / "direct.json",
                membership_path=tmp / "membership.parquet",
                pool_per_side=2,
                positions_per_side=1,
                model="deepseek-reasoner",
                temperature=0.6,
                rate_limit=0.0,
                prompt_style=fill.DEFAULT_PROMPT_STYLE,
                max_dates=None,
                dry_run=False,
                confirm_online_llm_calls=False,
                skip_refresh_offline_audit=True,
            )
            coverage = {
                "pending_llm_call_count": 1,
                "cached_count": 0,
                "missing_count": 1,
                "invalid_count": 0,
            }

            with mock.patch.object(fill, "build_matching_pools", return_value=(["pool"], {"status": "valid"})):
                with mock.patch.object(fill, "summarize_cache_coverage", return_value=coverage):
                    with self.assertRaisesRegex(RuntimeError, "confirm-online-llm-calls"):
                        fill.run_fill(args)


if __name__ == "__main__":
    unittest.main()
