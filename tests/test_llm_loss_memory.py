import unittest

import pandas as pd

from llm_loss_memory import LossMemoryStore


class LossMemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.composite_dfs = {
            "AAA": pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2024-01-05"),
                        "adj_close": 100.0,
                        "direction_score": 40.0,
                        "magnitude_score": 70.0,
                        "mom_3m": 0.30,
                        "rsi_5d": 72.0,
                        "adx_14d": 30.0,
                        "vol_percentile_504d": 0.80,
                        "vix_level": 18.0,
                        "vix_percentile_252d": 0.40,
                        "vol_imbalance_20d": 0.20,
                        "m2_m1_ratio": 1.50,
                        "atr_14d_pct": 0.03,
                        "realized_vol_10d": 0.25,
                        "vol_adjusted_trend_ma100": 0.40,
                        "range_pos_55d": 0.90,
                        "pct_from_52w_high": 0.05,
                        "range_pct": 0.02,
                        "autocorr_5d": 0.10,
                        "vel_10d_over_vol": 0.80,
                        "days_since_52w_high": 4.0,
                        "vol_adjusted_trend_ma100_slope": 0.15,
                        "pct_from_ma100": 0.09,
                    },
                    {
                        "date": pd.Timestamp("2024-01-12"),
                        "adj_close": 92.0,
                        "direction_score": 5.0,
                        "magnitude_score": 65.0,
                    },
                ]
            ),
            "BBB": pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2024-01-12"),
                        "adj_close": 50.0,
                        "direction_score": 42.0,
                        "magnitude_score": 71.0,
                        "mom_3m": 0.31,
                        "rsi_5d": 73.0,
                        "adx_14d": 31.0,
                        "vol_percentile_504d": 0.81,
                        "vix_level": 18.0,
                        "vix_percentile_252d": 0.40,
                        "vol_imbalance_20d": 0.19,
                        "m2_m1_ratio": 1.49,
                        "atr_14d_pct": 0.03,
                        "realized_vol_10d": 0.24,
                        "vol_adjusted_trend_ma100": 0.41,
                        "range_pos_55d": 0.88,
                        "pct_from_52w_high": 0.06,
                        "range_pct": 0.02,
                        "autocorr_5d": 0.12,
                        "vel_10d_over_vol": 0.79,
                        "days_since_52w_high": 5.0,
                        "vol_adjusted_trend_ma100_slope": 0.14,
                        "pct_from_ma100": 0.08,
                    }
                ]
            ),
        }

    def test_finalize_portfolio_records_realized_loss_case(self):
        store = LossMemoryStore(top_k=2, max_targets_per_side=2)
        added = store.finalize_portfolio(
            entry_portfolio={
                "rebalance_date": "2024-01-05",
                "regime": "normal",
                "long_basket": [
                    {"ticker": "AAA", "weight_exec": 0.5, "rationale": "Momentum"}
                ],
                "short_basket": [],
            },
            exit_date=pd.Timestamp("2024-01-12"),
            composite_dfs=self.composite_dfs,
        )
        self.assertEqual(added, 1)
        self.assertEqual(store.total_cases, 1)

        payload = store.build_prompt_context(
            universe={
                "BBB": {
                    "direction_score": 42.0,
                    "magnitude_score": 71.0,
                    "indicators": {
                        "mom_3m": 0.31,
                        "rsi_5d": 73.0,
                        "adx_14d": 31.0,
                        "vol_percentile_504d": 0.81,
                        "vix_level": 18.0,
                        "vix_percentile_252d": 0.40,
                        "vol_imbalance_20d": 0.19,
                        "m2_m1_ratio": 1.49,
                        "atr_14d_pct": 0.03,
                        "realized_vol_10d": 0.24,
                        "vol_adjusted_trend_ma100": 0.41,
                        "range_pos_55d": 0.88,
                        "pct_from_52w_high": 0.06,
                        "range_pct": 0.02,
                        "autocorr_5d": 0.12,
                        "vel_10d_over_vol": 0.79,
                        "days_since_52w_high": 5.0,
                        "vol_adjusted_trend_ma100_slope": 0.14,
                        "pct_from_ma100": 0.08,
                    },
                }
            },
            previous_portfolio=None,
            regime="normal",
            as_of_date=pd.Timestamp("2024-01-12"),
            top_n_magnitude=10,
        )

        self.assertEqual(payload["target_count"], 1)
        self.assertEqual(payload["case_count"], 1)
        self.assertIn("BBB", payload["text"])
        self.assertIn("AAA 2024-01-05→2024-01-12", payload["text"])

    def test_prompt_context_uses_only_realized_cases(self):
        store = LossMemoryStore(top_k=2, max_targets_per_side=2)
        payload = store.build_prompt_context(
            universe={
                "BBB": {
                    "direction_score": 42.0,
                    "magnitude_score": 71.0,
                    "indicators": {
                        "mom_3m": 0.31,
                        "rsi_5d": 73.0,
                        "adx_14d": 31.0,
                    },
                }
            },
            previous_portfolio=None,
            regime="normal",
            as_of_date=pd.Timestamp("2024-01-12"),
            top_n_magnitude=10,
        )
        self.assertEqual(payload["text"], "")
        self.assertEqual(payload["case_count"], 0)


if __name__ == "__main__":
    unittest.main()
