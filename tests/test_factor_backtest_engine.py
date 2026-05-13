import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import factor_backtest_engine as fbe


class CrossSectionalSnapshotTests(unittest.TestCase):
    def test_explicit_llm_diagnostics_default_to_full_universe(self):
        date = pd.Timestamp("2024-01-05")
        common = {
            "date": date,
            "fwd_10d_ret": 5.0,
            "llm_side": "LONG",
            "llm_weight_raw": 0.5,
            "llm_weight_exec": 0.5,
            "llm_source": "llm",
            "llm_execution_mode": "equal_weight",
        }
        all_data = {
            "AAA": pd.DataFrame([{**common, "direction_score": 1.0, "base_direction_score": 1.0, "llm_selected": 0}]),
            "BBB": pd.DataFrame([{**common, "direction_score": 2.0, "base_direction_score": 2.0, "llm_selected": 1}]),
        }

        full_snapshot = fbe._cross_sectional_snapshot(
            all_data,
            date,
            "direction_score",
            require_exact_dates=True,
            selected_only=False,
        )
        selected_snapshot = fbe._cross_sectional_snapshot(
            all_data,
            date,
            "direction_score",
            require_exact_dates=True,
            selected_only=True,
        )

        self.assertEqual(set(full_snapshot["ticker"]), {"AAA", "BBB"})
        self.assertEqual(set(selected_snapshot["ticker"]), {"BBB"})

    def test_membership_gating_uses_asof_snapshot_and_symbol_normalization(self):
        date = pd.Timestamp("2024-01-05")
        all_data = {
            "AAA": pd.DataFrame([{"date": date, "direction_score": 1.0, "fwd_10d_ret": 3.0}]),
            "BBB": pd.DataFrame([{"date": date, "direction_score": 2.0, "fwd_10d_ret": 4.0}]),
            "CCC": pd.DataFrame([{"date": date, "direction_score": 3.0, "fwd_10d_ret": 5.0}]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            membership_path = Path(tmpdir) / "membership.csv"
            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        [
                            "2024-01-04",
                            "2024-01-04",
                            "2024-01-04",
                            "2024-01-06",
                            "2024-01-06",
                            "2024-01-06",
                        ]
                    ),
                    "symbol": ["AAA-200012", "BBB", "CCC", "AAA-200012", "BBB", "CCC"],
                    "in_r1000": [1, 0, 1, 1, 1, 1],
                }
            ).to_csv(membership_path, index=False)
            lookup = fbe.load_russell_membership_lookup(membership_path)

        snapshot = fbe._cross_sectional_snapshot(
            all_data,
            date,
            "direction_score",
            require_exact_dates=True,
            selected_only=False,
            membership_lookup=lookup,
        )

        self.assertEqual(set(snapshot["ticker"]), {"AAA", "CCC"})


class CandidateSnapshotTests(unittest.TestCase):
    def test_candidate_snapshot_excludes_non_members(self):
        signal_date = pd.Timestamp("2024-01-05")
        trade_date = pd.Timestamp("2024-01-05")
        aaa_df = pd.DataFrame(
            {
                "date": [signal_date],
                "adj_close": [10.0],
                "direction_score": [5.0],
                "magnitude_score": [60.0],
            }
        )
        bbb_df = pd.DataFrame(
            {
                "date": [signal_date],
                "adj_close": [11.0],
                "direction_score": [8.0],
                "magnitude_score": [70.0],
            }
        )
        prepared = {
            "AAA": fbe._prepare_ticker_data(aaa_df, "direction_score"),
            "BBB": fbe._prepare_ticker_data(bbb_df, "direction_score"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            membership_path = Path(tmpdir) / "membership.csv"
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "symbol": ["AAA-201001"],
                    "in_r1000": [1],
                }
            ).to_csv(membership_path, index=False)
            lookup = fbe.load_russell_membership_lookup(membership_path)

        snapshot = fbe._candidate_snapshot(
            "direction_score",
            prepared,
            signal_date,
            trade_date,
            require_exact_dates=True,
            membership_lookup=lookup,
        )

        self.assertEqual(set(snapshot["date_scores"]), {"AAA"})
        self.assertEqual(snapshot["long_candidates"], ["AAA"])
        self.assertEqual(snapshot["short_candidates"], ["AAA"])


class ExplicitLlmSnapshotTests(unittest.TestCase):
    def _explicit_row(self, date, side, price, weight=0.02):
        return {
            "date": pd.Timestamp(date),
            "adj_close": price,
            "direction_score": 1.0 if side == "LONG" else -1.0,
            "base_direction_score": 1.0 if side == "LONG" else -1.0,
            "magnitude_score": 50.0,
            "base_magnitude_score": 50.0,
            "llm_selected": 1,
            "llm_side": side,
            "llm_weight_raw": weight,
            "llm_weight_exec": weight,
            "llm_source": "llm",
            "llm_execution_mode": "equal_weight",
        }

    def test_cash_missing_preserves_cache_weights_and_audits_drop(self):
        signal_date = pd.Timestamp("2024-01-05")
        all_data = {
            "AAA": pd.DataFrame([self._explicit_row(signal_date, "LONG", 10.0, 0.02)]),
            "LOW": pd.DataFrame([self._explicit_row(signal_date, "LONG", 4.0, 0.02)]),
            "BBB": pd.DataFrame([self._explicit_row(signal_date, "SHORT", 20.0, 0.02)]),
        }
        prepared = {
            ticker: fbe._prepare_ticker_data(df, "direction_score")
            for ticker, df in all_data.items()
        }

        snapshot = fbe._explicit_llm_snapshot(
            all_data,
            "direction_score",
            signal_date,
            signal_date,
            prepared,
            available_signal_dates=[signal_date],
            llm_missing_policy="cash_missing",
        )

        self.assertEqual(snapshot["raw_long_count"], 2)
        self.assertEqual(snapshot["raw_short_count"], 1)
        self.assertEqual(snapshot["target_weights"], {"AAA": 0.02, "BBB": -0.02})
        reasons = {(row["ticker"], row["drop_reason"]) for row in snapshot["audit_rows"]}
        self.assertIn(("LOW", "below_min_price"), reasons)
        self.assertIn(("AAA", "survived_to_execution"), reasons)

    def test_fail_if_incomplete_raises_when_selected_name_drops(self):
        signal_date = pd.Timestamp("2024-01-05")
        all_data = {
            "AAA": pd.DataFrame([self._explicit_row(signal_date, "LONG", 10.0)]),
            "LOW": pd.DataFrame([self._explicit_row(signal_date, "LONG", 4.0)]),
            "BBB": pd.DataFrame([self._explicit_row(signal_date, "SHORT", 20.0)]),
        }
        prepared = {
            ticker: fbe._prepare_ticker_data(df, "direction_score")
            for ticker, df in all_data.items()
        }

        with self.assertRaisesRegex(RuntimeError, "Explicit LLM book incomplete"):
            fbe._explicit_llm_snapshot(
                all_data,
                "direction_score",
                signal_date,
                signal_date,
                prepared,
                available_signal_dates=[signal_date],
                llm_missing_policy="fail_if_incomplete",
            )


class ReviewEligibilityTests(unittest.TestCase):
    def test_target_weight_filter_forces_out_inactive_or_untradable_names(self):
        signal_date = pd.Timestamp("2024-01-05")
        trade_date = pd.Timestamp("2024-01-05")
        prepared = {
            "AAA": fbe._prepare_ticker_data(
                pd.DataFrame({"date": [trade_date], "adj_close": [10.0], "direction_score": [1.0]}),
                "direction_score",
            ),
            "DEAD": fbe._prepare_ticker_data(
                pd.DataFrame({"date": [trade_date], "adj_close": [12.0], "direction_score": [1.0]}),
                "direction_score",
            ),
            "LOW": fbe._prepare_ticker_data(
                pd.DataFrame({"date": [trade_date], "adj_close": [4.0], "direction_score": [1.0]}),
                "direction_score",
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            membership_path = Path(tmpdir) / "membership.csv"
            pd.DataFrame(
                {
                    "date": [signal_date, signal_date],
                    "symbol": ["AAA", "LOW"],
                    "in_r1000": [1, 1],
                }
            ).to_csv(membership_path, index=False)
            lookup = fbe.load_russell_membership_lookup(membership_path)

        filtered = fbe._filter_target_weights_for_review_eligibility(
            {"AAA": 0.5, "DEAD": -0.003, "LOW": -0.5},
            prepared,
            lookup,
            signal_date,
            trade_date,
        )

        self.assertEqual(filtered, {"AAA": 0.5})


class StaticPeriodicPathTests(unittest.TestCase):
    def test_build_factor_portfolio_periodic_static_collapses_to_review_periods(self):
        dates = pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
            ]
        )
        spy_df = pd.DataFrame(
            {
                "date": dates,
                "adj_close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "vix_percentile_252d": [0.5] * len(dates),
            }
        )
        regime_map = {pd.Timestamp(date): "normal" for date in dates}

        all_data = {
            "AAA": pd.DataFrame(
                {
                    "date": dates,
                    "adj_close": [10.0, 10.4, 10.8, 11.0, 11.4],
                    "direction_score": [5.0, 5.0, 5.0, 5.0, 5.0],
                    "magnitude_score": [60.0, 60.0, 60.0, 60.0, 60.0],
                }
            ),
            "BBB": pd.DataFrame(
                {
                    "date": dates,
                    "adj_close": [20.0, 19.6, 19.1, 18.9, 18.5],
                    "direction_score": [-5.0, -5.0, -5.0, -5.0, -5.0],
                    "magnitude_score": [55.0, 55.0, 55.0, 55.0, 55.0],
                }
            ),
        }
        scheduled = frozenset(pd.to_datetime([dates[0], dates[2], dates[4]]))
        config = fbe.PortfolioConstructionConfig(
            review_policy=fbe.ReviewPolicy(
                mode="static",
                static_review_days=2,
                regime_intervals=fbe.DEFAULT_REGIME_REVIEW_INTERVALS.copy(),
                scheduled_signal_dates=scheduled,
            ),
            hysteresis_policy=fbe.HysteresisPolicy(
                mode="none",
                top_n=1,
                bottom_n=1,
            ),
            top_n=1,
            bottom_n=1,
            weighting_mode="equal",
        )

        result = fbe.build_factor_portfolio(
            all_data,
            spy_df,
            rebalance_dates=list(scheduled),
            score_type="direction_score",
            regime_map=regime_map,
            config=config,
            trading_dates=list(dates),
            candidate_pool_config=fbe.CandidatePoolConfig(mode="raw"),
            periodic_static=True,
        )
        pnl_df = result["pnl_df"]

        self.assertLess(len(pnl_df), len(dates))
        self.assertIn("period_length_days", pnl_df.columns)
        self.assertTrue((pnl_df["return_granularity"] == "periodic_static").all())
        self.assertGreater(pnl_df["period_length_days"].max(), 1)


class ComputeMetricsTests(unittest.TestCase):
    def test_compute_metrics_defaults_missing_period_length_days_to_one(self):
        pnl_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "daily_ret_bps": [10.0, -5.0],
                "gross_daily_ret_bps": [12.0, -3.0],
                "long_pnl_bps": [7.0, -1.0],
                "short_pnl_bps": [3.0, -4.0],
                "num_longs": [30, 30],
                "num_shorts": [30, 30],
                "cost_bps_applied": [2.0, 2.0],
            }
        )
        regime_map = {
            pd.Timestamp("2024-01-02"): "normal",
            pd.Timestamp("2024-01-03"): "normal",
        }

        result_df, metrics = fbe.compute_metrics(
            pnl_df,
            regime_map=regime_map,
            all_data={},
            review_signal_dates=[],
            score_type="direction_score",
        )

        self.assertIn("dailyized_ret", result_df.columns)
        self.assertEqual(metrics["overall"]["total_period_days"], 2)


class MainCliTests(unittest.TestCase):
    def test_main_preserves_weighting_mode_in_run_config(self):
        spy_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "adj_close": [100.0, 101.0, 102.0],
                "vix_percentile_252d": [0.5, 0.5, 0.5],
            }
        )
        composite_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "adj_close": [10.0, 10.1, 10.2],
                "direction_score": [1.0, 1.5, 2.0],
                "magnitude_score": [50.0, 55.0, 60.0],
            }
        )
        seen_configs = []

        def fake_run_backtest(*args, **kwargs):
            seen_configs.append(
                (
                    kwargs["config"].weighting_mode,
                    kwargs["config"].book_sizing_mode,
                )
            )
            return pd.DataFrame(), {"overall": {}}, pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(fbe, "load_all_composites", return_value={"AAA": composite_df}),
                patch.object(fbe, "load_spy_data", return_value=spy_df),
                patch.object(fbe, "run_backtest", side_effect=fake_run_backtest),
                patch.object(fbe, "OUTPUT_DIR_BASE", Path(tmpdir)),
                patch(
                    "sys.argv",
                    [
                        "factor_backtest_engine.py",
                        "10",
                        "--weighting-mode",
                        "rank",
                        "--book-sizing-mode",
                        "decile",
                        "--skip-plots",
                    ],
                ),
            ):
                fbe.main()
                manifests = list(Path(tmpdir).glob("**/run_config.json"))
                self.assertEqual(len(manifests), 1)
                manifest_path = manifests[0]
                self.assertTrue(manifest_path.exists())
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["weighting_mode"], "rank")
                self.assertEqual(manifest["book_sizing_mode"], "decile")

        self.assertEqual(seen_configs, [("rank", "decile"), ("rank", "decile")])

    def test_main_supports_direction_only_and_custom_book_size(self):
        spy_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "adj_close": [100.0, 101.0, 102.0],
                "vix_percentile_252d": [0.5, 0.5, 0.5],
            }
        )
        composite_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "adj_close": [10.0, 10.1, 10.2],
                "direction_score": [1.0, 1.5, 2.0],
                "magnitude_score": [50.0, 55.0, 60.0],
            }
        )
        seen = []

        def fake_run_backtest(*args, **kwargs):
            seen.append((args[0], kwargs["config"].top_n, kwargs["config"].bottom_n))
            return pd.DataFrame(), {"overall": {}}, pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(fbe, "load_all_composites", return_value={"AAA": composite_df}),
                patch.object(fbe, "load_spy_data", return_value=spy_df),
                patch.object(fbe, "run_backtest", side_effect=fake_run_backtest),
                patch.object(fbe, "OUTPUT_DIR_BASE", Path(tmpdir)),
                patch(
                    "sys.argv",
                    [
                        "factor_backtest_engine.py",
                        "10",
                        "--score-types",
                        "direction_score",
                        "--top-n",
                        "60",
                        "--bottom-n",
                        "60",
                        "--skip-plots",
                    ],
                ),
            ):
                fbe.main()

        self.assertEqual(seen, [("direction_score", 60, 60)])

    def test_main_records_membership_gating_in_run_config(self):
        spy_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "adj_close": [100.0, 101.0, 102.0],
                "vix_percentile_252d": [0.5, 0.5, 0.5],
            }
        )
        composite_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "adj_close": [10.0, 10.1, 10.2],
                "direction_score": [1.0, 1.5, 2.0],
                "magnitude_score": [50.0, 55.0, 60.0],
                "dataset_label": ["russell_ohlcv"] * 3,
            }
        )
        membership_lookup = fbe.RussellMembershipLookup(
            dates_ns=np.array([pd.Timestamp("2024-01-02").value], dtype=np.int64),
            universes=(frozenset({"AAA"}),),
            path="/tmp/r1000_membership.parquet",
        )

        def fake_run_backtest(*args, **kwargs):
            return pd.DataFrame(), {"overall": {}}, pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(fbe, "load_all_composites", return_value={"AAA": composite_df}),
                patch.object(fbe, "load_spy_data", return_value=spy_df),
                patch.object(fbe, "load_russell_membership_lookup", return_value=membership_lookup),
                patch.object(fbe, "run_backtest", side_effect=fake_run_backtest),
                patch.object(fbe, "OUTPUT_DIR_BASE", Path(tmpdir)),
                patch("sys.argv", ["factor_backtest_engine.py", "10", "--skip-plots"]),
            ):
                fbe.main()
                manifest_path = Path(tmpdir) / "10d_horizon" / "run_config.json"
                manifest = json.loads(manifest_path.read_text())

        self.assertTrue(manifest["membership_gating_enabled"])
        self.assertEqual(manifest["membership_path"], membership_lookup.path)

    def test_main_records_custom_cost_bps_in_run_config(self):
        spy_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "adj_close": [100.0, 101.0, 102.0],
                "vix_percentile_252d": [0.5, 0.5, 0.5],
            }
        )
        composite_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "adj_close": [10.0, 10.1, 10.2],
                "direction_score": [1.0, 1.5, 2.0],
                "magnitude_score": [50.0, 55.0, 60.0],
            }
        )

        def fake_run_backtest(*args, **kwargs):
            return pd.DataFrame(), {"overall": {}}, pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(fbe, "load_all_composites", return_value={"AAA": composite_df}),
                patch.object(fbe, "load_spy_data", return_value=spy_df),
                patch.object(fbe, "run_backtest", side_effect=fake_run_backtest),
                patch.object(fbe, "OUTPUT_DIR_BASE", Path(tmpdir)),
                patch("sys.argv", ["factor_backtest_engine.py", "10", "--cost-bps", "5", "--skip-plots"]),
            ):
                fbe.main()
                manifest_path = Path(tmpdir) / "10d_horizon" / "run_config.json"
                manifest = json.loads(manifest_path.read_text())

        self.assertEqual(manifest["cost_bps"], 5.0)
        self.assertAlmostEqual(manifest["cost_rate"], 0.0005)

    def test_main_records_periodic_static_in_run_config(self):
        spy_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "adj_close": [100.0, 101.0, 102.0],
                "vix_percentile_252d": [0.5, 0.5, 0.5],
            }
        )
        composite_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "adj_close": [10.0, 10.1, 10.2],
                "direction_score": [1.0, 1.5, 2.0],
                "magnitude_score": [50.0, 55.0, 60.0],
            }
        )
        seen_flags = []

        def fake_run_backtest(*args, **kwargs):
            seen_flags.append(kwargs["periodic_static"])
            return pd.DataFrame(), {"overall": {}}, pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(fbe, "load_all_composites", return_value={"AAA": composite_df}),
                patch.object(fbe, "load_spy_data", return_value=spy_df),
                patch.object(fbe, "run_backtest", side_effect=fake_run_backtest),
                patch.object(fbe, "OUTPUT_DIR_BASE", Path(tmpdir)),
                patch(
                    "sys.argv",
                    [
                        "factor_backtest_engine.py",
                        "10",
                        "--review-mode",
                        "static",
                        "--periodic-static",
                        "--skip-plots",
                    ],
                ),
            ):
                fbe.main()
                manifest_path = Path(tmpdir) / "10d_horizon" / "run_config.json"
                manifest = json.loads(manifest_path.read_text())

        self.assertTrue(manifest["periodic_static"])
        self.assertEqual(seen_flags, [True, True])

    def test_main_rejects_unknown_flags(self):
        with patch("sys.argv", ["factor_backtest_engine.py", "10", "--bogus-flag", "value"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                fbe.main()

        self.assertIn("Unsupported argument(s): --bogus-flag", buf.getvalue())


class CostSensitivityTests(unittest.TestCase):
    def test_cagr_is_nan_when_terminal_wealth_is_non_positive(self):
        cum = pd.Series([1.0, 0.8, -0.1])
        self.assertTrue(np.isnan(fbe._cagr(cum)))

    def test_slide6_cost_sensitivity_handles_non_positive_terminal_paths(self):
        pnl_df = pd.DataFrame(
            {
                "daily_ret_bps": [-15000.0, 100.0, 50.0],
                "cost_bps_applied": [0.0, 5.0, 5.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            fbe._slide6_cost_sensitivity(pnl_df, output_dir)
            self.assertTrue((output_dir / "slide_06_cost_sensitivity.png").exists())


if __name__ == "__main__":
    unittest.main()
