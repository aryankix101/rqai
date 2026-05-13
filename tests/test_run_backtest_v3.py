import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest_v3 as rb


def _make_synthetic_histories(num_tickers: int = 50) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("1999-10-01", "2000-03-31")
    histories: dict[str, pd.DataFrame] = {}

    for idx in range(num_tickers):
        ticker = "AAPL" if idx == 0 else f"T{idx:03d}"
        day_idx = np.arange(len(dates), dtype=float)
        base = 15.0 + idx * 0.2
        seasonal = 0.03 * np.sin((day_idx + idx) / 4.0)
        drift = 0.0008 * day_idx
        adj_close = base * (1.0 + drift + seasonal)
        adj_open = adj_close * (0.998 + 0.0001 * (idx % 3))
        raw_close = adj_close * (1.0 + 0.0005 * ((idx % 5) - 2))
        volume = np.full(len(dates), 700_000 + idx * 2_000, dtype=float)

        histories[ticker] = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "asset_id": ticker,
                "source_column": ticker,
                "is_historical_suffix": False,
                "collision_count": 1,
                "open": raw_close,
                "high": raw_close * 1.01,
                "low": raw_close * 0.99,
                "close": raw_close,
                "volume": volume,
                "adj_open": adj_open,
                "adj_high": adj_close * 1.01,
                "adj_low": adj_close * 0.99,
                "adj_close": adj_close,
            }
        )

    return histories


def _write_membership_csv(path: Path, tickers: list[str]) -> None:
    rows = []
    for date in ["1999-12-31", "2000-01-21", "2000-02-11", "2000-02-25"]:
        for ticker in tickers:
            symbol = f"{ticker}-200012" if ticker == "AAPL" and date == "1999-12-31" else ticker
            rows.append({"date": date, "symbol": symbol, "in_r1000": 1})
    pd.DataFrame(rows).to_csv(path, index=False)


class RunBacktestV3Tests(unittest.TestCase):
    def test_parse_cost_bps_levels_adds_gross_level(self):
        self.assertEqual(rb.parse_cost_bps_levels("2.5,5"), (0.0, 2.5, 5.0))

    def test_load_membership_lookup_normalizes_and_falls_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            membership_path = Path(tmpdir) / "membership.csv"
            pd.DataFrame(
                [
                    {"date": "1999-12-31", "symbol": "AAA-200012", "in_r1000": 1},
                    {"date": "1999-12-31", "symbol": "BBB", "in_r1000": 1},
                    {"date": "2000-01-14", "symbol": "CCC", "in_r1000": 1},
                ]
            ).to_csv(membership_path, index=False)

            lookup = rb.load_membership_lookup(membership_path)

            self.assertEqual(lookup.universe_on_or_before(pd.Timestamp("1999-12-31")), {"AAA", "BBB"})
            self.assertEqual(lookup.universe_on_or_before(pd.Timestamp("2000-01-07")), {"AAA", "BBB"})
            self.assertEqual(lookup.universe_on_or_before(pd.Timestamp("2000-01-14")), {"CCC"})

    def test_run_benchmark_respects_window_and_writes_summary_outputs(self):
        histories = _make_synthetic_histories()
        tickers = sorted(histories.keys())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            membership_path = tmpdir_path / "membership.csv"
            _write_membership_csv(membership_path, tickers)

            config = rb.BenchmarkConfig(
                start=pd.Timestamp("2000-01-01"),
                end=pd.Timestamp("2000-02-29"),
                output_dir=tmpdir_path / "outputs",
                membership_path=membership_path,
            )

            artifacts = rb.run_benchmark(config, histories=histories)
            rb.save_artifacts(artifacts, config.output_dir)

            self.assertFalse(artifacts.weekly_returns.empty)
            self.assertGreaterEqual(artifacts.weekly_returns.index.min(), pd.Timestamp("2000-01-01"))
            self.assertLessEqual(artifacts.weekly_returns.index.max(), pd.Timestamp("2000-02-29"))
            self.assertIn(pd.Timestamp("2000-01-07"), artifacts.weekly_returns.index)
            self.assertIn("ls_c2c_15", artifacts.weekly_returns.columns)
            self.assertIn("ls_o2o_15", artifacts.weekly_returns.columns)

            strategy_variants = set(artifacts.strategy_summary["variant_name"].tolist())
            self.assertIn("LS_O2O_15bps", strategy_variants)
            self.assertIn("LS_C2C_Gross", strategy_variants)

            expected_files = {
                "weekly_returns.csv",
                "signal_checks.csv",
                "portfolio_selections.csv",
                "headline_metrics.csv",
                "yearly_returns.csv",
                "strategy_summary.csv",
                "cumulative_returns.png",
                "decile_profile.png",
                "strategy_comparison.png",
            }
            self.assertEqual(expected_files, {path.name for path in config.output_dir.iterdir()})

    def test_run_benchmark_supports_trading_day_cadence_mode(self):
        histories = _make_synthetic_histories()
        tickers = sorted(histories.keys())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            membership_path = tmpdir_path / "membership.csv"
            _write_membership_csv(membership_path, tickers)

            config = rb.BenchmarkConfig(
                start=pd.Timestamp("2000-01-01"),
                end=pd.Timestamp("2000-02-29"),
                output_dir=tmpdir_path / "outputs_10d",
                membership_path=membership_path,
                cadence_mode="trading_days",
                signal_lookback_days=5,
                rebalance_days=10,
                holding_days=10,
            )

            artifacts = rb.run_benchmark(config, histories=histories)

            self.assertFalse(artifacts.weekly_returns.empty)
            self.assertIn("ls_c2c_5", artifacts.weekly_returns.columns)
            self.assertGreaterEqual(artifacts.weekly_returns.index.min(), pd.Timestamp("2000-01-01"))
            self.assertLessEqual(artifacts.weekly_returns.index.max(), pd.Timestamp("2000-02-29"))
            self.assertLess(len(artifacts.weekly_returns), 8)

    def test_run_benchmark_supports_fixed_book_size(self):
        histories = _make_synthetic_histories(num_tickers=120)
        tickers = sorted(histories.keys())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            membership_path = tmpdir_path / "membership.csv"
            _write_membership_csv(membership_path, tickers)

            config = rb.BenchmarkConfig(
                start=pd.Timestamp("2000-01-01"),
                end=pd.Timestamp("2000-02-29"),
                output_dir=tmpdir_path / "outputs_fixed_book",
                membership_path=membership_path,
                cadence_mode="trading_days",
                signal_lookback_days=5,
                rebalance_days=10,
                holding_days=10,
                positions_per_side=10,
            )

            artifacts = rb.run_benchmark(config, histories=histories)

            self.assertFalse(artifacts.weekly_returns.empty)
            self.assertTrue((artifacts.weekly_returns["n_long"] == 10).all())
            self.assertTrue((artifacts.weekly_returns["n_short"] == 10).all())
            self.assertIn("ls_c2c_5", artifacts.weekly_returns.columns)


if __name__ == "__main__":
    unittest.main()
