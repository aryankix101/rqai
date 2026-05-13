import unittest
import tempfile
from pathlib import Path

from scripts.inspect_llm_cache import (
    _build_validity_audit,
    _load_selection_detail,
    _raw_cache_selection_detail,
    _summarize_baseline_comparison,
    _summarize_survival,
)


class LlmCacheInspectionTests(unittest.TestCase):
    def test_raw_and_realized_counts_are_reported_separately(self):
        portfolios = [
            (
                "2024-01-05",
                {
                    "long_basket": [
                        {"ticker": "L1", "weight_exec": 0.02},
                        {"ticker": "L2", "weight_exec": 0.02},
                    ],
                    "short_basket": [
                        {"ticker": "S1", "weight_exec": 0.02},
                        {"ticker": "S2", "weight_exec": 0.02},
                    ],
                },
            )
        ]
        raw_detail = _raw_cache_selection_detail(portfolios)
        realized_detail = {
            ("2024-01-05", "LONG", "L1"): {
                "date": "2024-01-05",
                "side": "LONG",
                "ticker": "L1",
                "weight": 0.5,
                "realized_return_pct": 1.0,
            },
            ("2024-01-05", "SHORT", "S1"): {
                "date": "2024-01-05",
                "side": "SHORT",
                "ticker": "S1",
                "weight": -0.5,
                "realized_return_pct": -1.0,
            },
        }

        survival_rows, survival_summary = _summarize_survival(raw_detail, realized_detail)

        long_summary = next(
            row for row in survival_summary if row["date"] == "2024-01-05" and row["side"] == "LONG"
        )
        short_summary = next(
            row for row in survival_summary if row["date"] == "2024-01-05" and row["side"] == "SHORT"
        )
        self.assertEqual(long_summary["raw_selected_count"], 2)
        self.assertEqual(long_summary["realized_survivor_count"], 1)
        self.assertEqual(short_summary["raw_selected_count"], 2)
        self.assertEqual(short_summary["realized_survivor_count"], 1)
        missing = [row for row in survival_rows if not row["survived_to_realized"]]
        self.assertEqual({row["drop_reason"] for row in missing}, {"missing_from_realized_attribution_no_audit"})

    def test_selection_loader_does_not_double_sign_factor_engine_shorts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "portfolio_selections.csv"
            path.write_text(
                "rebalance_date,signal_date,ticker,position_type,rank,score,weight,realized_return_pct\n"
                "2024-01-05,2024-01-05,AAA,SHORT,1,0,-0.02,5.0\n"
            )

            detail = _load_selection_detail(path)

        row = detail[("2024-01-05", "SHORT", "AAA")]
        self.assertEqual(row["return_format"], "signed_position_pnl")
        self.assertEqual(row["signed_return_pct"], 5.0)
        self.assertAlmostEqual(row["contribution_pct"], 0.1)

    def test_selection_loader_signs_raw_deterministic_shorts_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "portfolio_selections.csv"
            path.write_text(
                "rebalance_date,signal_date,ticker,position_type,rank,score,weight,realized_return_pct,realized_c2c_return_pct\n"
                "2024-01-05,2024-01-05,AAA,SHORT,1,0,0.02,5.0,5.0\n"
            )

            detail = _load_selection_detail(path)

        row = detail[("2024-01-05", "SHORT", "AAA")]
        self.assertEqual(row["return_format"], "raw_underlying_return")
        self.assertEqual(row["signed_return_pct"], -5.0)
        self.assertAlmostEqual(row["contribution_pct"], -0.1)

    def test_baseline_comparison_only_uses_supplied_cache_dates(self):
        llm_detail = {
            ("2024-01-05", "LONG", "L1"): {
                "date": "2024-01-05",
                "side": "LONG",
                "ticker": "L1",
                "weight": 0.5,
                "realized_return_pct": 1.0,
                "signed_return_pct": 1.0,
                "contribution_pct": 0.5,
            }
        }
        baseline_detail = {
            ("2024-01-05", "LONG", "L1"): {
                "date": "2024-01-05",
                "side": "LONG",
                "ticker": "L1",
                "weight": 0.5,
                "realized_return_pct": 1.0,
                "signed_return_pct": 1.0,
                "contribution_pct": 0.5,
            }
        }

        rows, summary = _summarize_baseline_comparison(llm_detail, baseline_detail)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(summary), 1)
        self.assertEqual(rows[0]["decision"], "kept")

    def test_fast_validity_audit_separates_raw_from_realized(self):
        raw_detail = {
            ("2024-01-05", "LONG", "L1"): {
                "date": "2024-01-05",
                "side": "LONG",
                "ticker": "L1",
                "rank": 1,
                "weight": 0.02,
            },
            ("2024-01-05", "LONG", "L2"): {
                "date": "2024-01-05",
                "side": "LONG",
                "ticker": "L2",
                "rank": 2,
                "weight": 0.02,
            },
        }
        realized_detail = {
            ("2024-01-05", "LONG", "L1"): {
                "date": "2024-01-05",
                "side": "LONG",
                "ticker": "L1",
                "weight": 0.5,
                "position_pnl_return_pct": 1.25,
            }
        }
        baseline_detail = {
            ("2024-01-05", "LONG", "L2"): {
                "date": "2024-01-05",
                "side": "LONG",
                "ticker": "L2",
                "rank": 7,
                "position_pnl_return_pct": 2.5,
            }
        }

        rows = _build_validity_audit(
            raw_detail,
            realized_detail,
            baseline_detail,
            review_log_path=None,
            composite_dir=None,
            membership_path=None,
        )

        by_ticker = {row["ticker"]: row for row in rows}
        self.assertTrue(by_ticker["L1"]["written_to_realized_attribution"])
        self.assertEqual(by_ticker["L1"]["likely_drop_reason"], "survived_to_realized")
        self.assertFalse(by_ticker["L2"]["written_to_realized_attribution"])
        self.assertTrue(by_ticker["L2"]["in_det_top50"])
        self.assertEqual(
            by_ticker["L2"]["likely_drop_reason"],
            "cache_valid_but_missing_from_realized_attribution",
        )
        self.assertEqual(by_ticker["L2"]["composite_file_exists"], "cache_implied_true")


if __name__ == "__main__":
    unittest.main()
