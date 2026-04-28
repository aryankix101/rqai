import unittest

import pandas as pd

from data_source_config import DATASET_VERSION
from llm_ab_comparison import compute_metrics, summarize_backtest_health
from llm_portfolio_constructor import (
    COMPOSITE_SCORES_DIR,
    DEFAULT_LLM_STRATEGY,
    DEFAULT_MODEL,
    PROMPT_FORMAT_VERSION,
    _cache_is_compatible,
    _resolve_llm_strategy_module,
    build_cache_filename,
    load_signal_data_preloaded,
    prompt_style_label,
)
from technical_llm import (
    _build_candidate_lookup,
    _build_weighted_portfolio,
    _format_universe_for_prompt,
    build_turnover_overlay_system_prompt,
    build_portfolio_system_prompt,
    derive_portfolio_transition,
)


class PromptModeTests(unittest.TestCase):
    def test_signal_only_prompt_uses_alpha_discovery_language(self):
        prompt = build_portfolio_system_prompt(turnover_aware=False)
        self.assertIn("ALPHA DISCOVERY STAGE", prompt)
        self.assertIn("This is the only decision stage", prompt)
        self.assertNotIn("TURNOVER OVERLAY STAGE", prompt)

        universe = {
            "AAA": {"direction_score": 12.0, "magnitude_score": 80.0, "indicators": {}},
            "BBB": {"direction_score": -14.0, "magnitude_score": 75.0, "indicators": {}},
        }
        previous_portfolio = {
            "long_basket": [{"ticker": "OLDL"}],
            "short_basket": [{"ticker": "OLDS"}],
        }
        prompt_body = _format_universe_for_prompt(
            universe=universe,
            previous_portfolio=previous_portfolio,
            regime="normal",
            rebalance_id="w_00001",
            turnover_aware=False,
        )
        self.assertIn("PIPELINE MODE: single-stage alpha discovery", prompt_body)
        self.assertIn("TURNOVER OBJECTIVE: ignore trading costs", prompt_body)
        self.assertNotIn("PREVIOUS LONG POSITIONS", prompt_body)
        self.assertNotIn("PREVIOUS SHORT POSITIONS", prompt_body)
        self.assertNotIn("AMBIGUOUS CANDIDATES", prompt_body)
        self.assertNotIn("AMBIGUOUS pool", prompt_body)
        self.assertIn("MID", prompt_body)
        self.assertIn("soft penalty only", prompt_body)
        self.assertIn("Do not assign weights", prompt_body)

    def test_turnover_pipeline_splits_signal_stage_and_overlay_stage(self):
        prompt = build_portfolio_system_prompt(turnover_aware=True)
        self.assertIn("ALPHA DISCOVERY STAGE", prompt)
        self.assertIn("This is stage 1 of 2", prompt)
        self.assertIn("a later implementation overlay", prompt)

        overlay_prompt = build_turnover_overlay_system_prompt()
        self.assertIn("TURNOVER OVERLAY STAGE", overlay_prompt)
        self.assertIn("NOT discovering alpha from scratch", overlay_prompt)

        universe = {
            "AAA": {"direction_score": 12.0, "magnitude_score": 80.0, "indicators": {}},
            "BBB": {"direction_score": -14.0, "magnitude_score": 75.0, "indicators": {}},
        }
        previous_portfolio = {
            "long_basket": [{"ticker": "OLDL"}],
            "short_basket": [{"ticker": "OLDS"}],
        }
        prompt_body = _format_universe_for_prompt(
            universe=universe,
            previous_portfolio=previous_portfolio,
            regime="normal",
            rebalance_id="w_00001",
            turnover_aware=True,
        )
        self.assertIn("PIPELINE MODE: stage 1 of 2", prompt_body)
        self.assertIn("implementation will be handled later", prompt_body)
        self.assertNotIn("PREVIOUS LONG POSITIONS", prompt_body)

    def test_prompt_uses_percentile_backed_indicator_values(self):
        composite_dfs = {
            "AAA": pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2023-12-29"),
                        "direction_score": 8.0,
                        "magnitude_score": 70.0,
                        "adj_close": 20.0,
                    },
                    {
                        "date": pd.Timestamp("2024-01-01"),
                        "direction_score": 9.0,
                        "magnitude_score": 72.0,
                        "adj_close": 21.0,
                    },
                    {
                        "date": pd.Timestamp("2024-01-02"),
                        "direction_score": 10.0,
                        "magnitude_score": 74.0,
                        "adj_close": 22.0,
                    },
                    {
                        "date": pd.Timestamp("2024-01-03"),
                        "direction_score": 11.0,
                        "magnitude_score": 76.0,
                        "adj_close": 23.0,
                    },
                    {
                        "date": pd.Timestamp("2024-01-04"),
                        "direction_score": 11.5,
                        "magnitude_score": 78.0,
                        "adj_close": 24.0,
                    },
                    {
                        "date": pd.Timestamp("2024-01-05"),
                        "direction_score": 12.0,
                        "magnitude_score": 80.0,
                        "adj_close": 25.0,
                        "mom_3m_percentile": 0.731,
                        "rsi_5d_percentile": 0.612,
                        "adx_14d_percentile": 0.587,
                        "vol_percentile_504d_percentile": 0.444,
                        "vix_level_percentile": 0.333,
                        "vix_percentile_252d_percentile": 0.222,
                        "days_since_52w_high_percentile": 0.654,
                        "pct_from_ma100_percentile": 0.777,
                    }
                ]
            ),
            "BBB": pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2023-12-29"),
                        "direction_score": -10.0,
                        "magnitude_score": 65.0,
                        "adj_close": 35.0,
                    },
                    {
                        "date": pd.Timestamp("2024-01-01"),
                        "direction_score": -11.0,
                        "magnitude_score": 67.0,
                        "adj_close": 34.0,
                    },
                    {
                        "date": pd.Timestamp("2024-01-02"),
                        "direction_score": -12.0,
                        "magnitude_score": 69.0,
                        "adj_close": 33.0,
                    },
                    {
                        "date": pd.Timestamp("2024-01-03"),
                        "direction_score": -13.0,
                        "magnitude_score": 71.0,
                        "adj_close": 32.0,
                    },
                    {
                        "date": pd.Timestamp("2024-01-04"),
                        "direction_score": -13.5,
                        "magnitude_score": 73.0,
                        "adj_close": 31.0,
                    },
                    {
                        "date": pd.Timestamp("2024-01-05"),
                        "direction_score": -14.0,
                        "magnitude_score": 75.0,
                        "adj_close": 30.0,
                    }
                ]
            ),
        }

        universe = load_signal_data_preloaded(composite_dfs, pd.Timestamp("2024-01-05"))
        self.assertAlmostEqual(universe["AAA"]["indicators"]["mom_3m"], 0.731)
        self.assertAlmostEqual(universe["AAA"]["indicators"]["rsi_5d"], 0.612)
        self.assertAlmostEqual(universe["AAA"]["indicators"]["pct_from_ma100"], 0.777)
        self.assertAlmostEqual(universe["AAA"]["trail_return_5d"], 0.25)
        self.assertAlmostEqual(universe["AAA"]["mean_reversion_signal"], -0.25)
        self.assertAlmostEqual(universe["BBB"]["trail_return_5d"], -0.1428571428571429)
        self.assertAlmostEqual(universe["BBB"]["mean_reversion_signal"], 0.1428571428571429)

        prompt_body = _format_universe_for_prompt(
            universe=universe,
            previous_portfolio=None,
            regime="normal",
            rebalance_id="w_00001",
            turnover_aware=False,
        )
        self.assertIn("percentile-normalized", prompt_body)
        self.assertIn("0.731", prompt_body)
        self.assertIn("0.654", prompt_body)

    def test_mean_reversion_loader_does_not_require_dir_or_mag_scores(self):
        composite_dfs = {
            "AAA": pd.DataFrame(
                [
                    {"date": pd.Timestamp("2024-01-01"), "adj_close": 10.0},
                    {"date": pd.Timestamp("2024-01-02"), "adj_close": 10.5},
                    {"date": pd.Timestamp("2024-01-03"), "adj_close": 11.0},
                    {"date": pd.Timestamp("2024-01-04"), "adj_close": 11.5},
                    {"date": pd.Timestamp("2024-01-05"), "adj_close": 12.0},
                    {
                        "date": pd.Timestamp("2024-01-08"),
                        "adj_close": 9.0,
                        "mom_3m_percentile": 0.25,
                    },
                ]
            )
        }

        deterministic_universe = load_signal_data_preloaded(
            composite_dfs,
            pd.Timestamp("2024-01-08"),
            strategy_style="deterministic",
        )
        mean_reversion_universe = load_signal_data_preloaded(
            composite_dfs,
            pd.Timestamp("2024-01-08"),
            strategy_style="mean_reversion",
        )

        self.assertEqual(deterministic_universe, {})
        self.assertIn("AAA", mean_reversion_universe)
        self.assertEqual(mean_reversion_universe["AAA"]["direction_score"], 0.0)
        self.assertEqual(mean_reversion_universe["AAA"]["magnitude_score"], 0.0)
        self.assertAlmostEqual(mean_reversion_universe["AAA"]["trail_return_5d"], -0.1)
        self.assertAlmostEqual(mean_reversion_universe["AAA"]["mean_reversion_signal"], 0.1)

    def test_deterministic_weight_builder_normalizes_candidate_utilities(self):
        signal_stage_output = {
            "long_candidates": [
                {"ticker": "AAA", "edge_score": 95.0, "confidence": 0.90, "utility": 85.5, "rank": 1, "rationale": "A"},
                {"ticker": "BBB", "edge_score": 80.0, "confidence": 0.80, "utility": 64.0, "rank": 2, "rationale": "B"},
                {"ticker": "CCC", "edge_score": 70.0, "confidence": 0.70, "utility": 49.0, "rank": 3, "rationale": "C"},
                {"ticker": "DDD", "edge_score": 60.0, "confidence": 0.70, "utility": 42.0, "rank": 4, "rationale": "D"},
                {"ticker": "EEE", "edge_score": 55.0, "confidence": 0.60, "utility": 33.0, "rank": 5, "rationale": "E"},
            ],
            "short_candidates": [
                {"ticker": "FFF", "edge_score": 92.0, "confidence": 0.88, "utility": 80.96, "rank": 1, "rationale": "F"},
                {"ticker": "GGG", "edge_score": 83.0, "confidence": 0.84, "utility": 69.72, "rank": 2, "rationale": "G"},
                {"ticker": "HHH", "edge_score": 74.0, "confidence": 0.73, "utility": 54.02, "rank": 3, "rationale": "H"},
                {"ticker": "III", "edge_score": 66.0, "confidence": 0.65, "utility": 42.90, "rank": 4, "rationale": "I"},
                {"ticker": "JJJ", "edge_score": 58.0, "confidence": 0.60, "utility": 34.80, "rank": 5, "rationale": "J"},
            ],
        }
        candidate_lookup = _build_candidate_lookup(signal_stage_output)
        selection = {
            "long_basket": [{"ticker": entry["ticker"], "rationale": entry["rationale"]} for entry in signal_stage_output["long_candidates"]],
            "short_basket": [{"ticker": entry["ticker"], "rationale": entry["rationale"]} for entry in signal_stage_output["short_candidates"]],
        }
        universe = {
            ticker: {"direction_score": 20.0 if ticker < "F" else -20.0, "magnitude_score": 80.0, "indicators": {}}
            for ticker in ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]
        }

        long_basket, short_basket = _build_weighted_portfolio(
            selection,
            candidate_lookup,
            universe,
            previous_portfolio=None,
            use_equal_weight=False,
        )

        self.assertAlmostEqual(sum(entry["weight_exec"] for entry in long_basket), 1.0)
        self.assertAlmostEqual(sum(entry["weight_exec"] for entry in short_basket), 1.0)
        self.assertGreater(long_basket[0]["weight_raw"], long_basket[-1]["weight_raw"])
        self.assertGreater(short_basket[0]["weight_raw"], short_basket[-1]["weight_raw"])

    def test_derive_portfolio_transition_treats_flips_as_drops(self):
        previous_portfolio = {
            "long_basket": [{"ticker": "AAA"}],
            "short_basket": [{"ticker": "BBB"}],
        }
        long_basket = [{"ticker": "CCC"}]
        short_basket = [{"ticker": "AAA"}, {"ticker": "BBB"}]

        kept, dropped = derive_portfolio_transition(previous_portfolio, long_basket, short_basket)

        self.assertEqual(kept, ["BBB"])
        self.assertEqual(dropped, ["AAA"])


class CacheCompatibilityTests(unittest.TestCase):
    def test_cache_compatibility_requires_matching_prompt_style(self):
        cache = {
            "metadata": {
                "schema_version": 3,
                "execution_mode": "equal_weight",
                "turnover_aware": False,
                "strategy_style": DEFAULT_LLM_STRATEGY,
                "model": DEFAULT_MODEL,
                "temperature": 0.15,
                "top_n_magnitude": 60,
                "strategy_signal_source": "technical_composite_v1",
                "prompt_style": prompt_style_label(False),
                "prompt_format_version": PROMPT_FORMAT_VERSION,
                "target_positions_per_side": None,
                "log_llm_trace": False,
                "loss_memory_enabled": False,
                "loss_memory_top_k": 3,
                "loss_memory_max_targets_per_side": 4,
                "llm_source_policy": "strict_llm_only",
                "source_dataset": DATASET_VERSION,
                "composite_scores_dir": str(COMPOSITE_SCORES_DIR),
            },
            "portfolios": {
                "2024-01-05": {
                    "llm_source": "llm",
                    "long_basket": [{"ticker": "AAA", "llm_source": "llm", "weight_raw": 0.5, "weight_exec": 0.5}],
                    "short_basket": [{"ticker": "BBB", "llm_source": "llm", "weight_raw": 0.5, "weight_exec": 0.5}],
                }
            },
        }

        self.assertTrue(
            _cache_is_compatible(
                cache,
                "equal_weight",
                turnover_aware=False,
                strategy_style=DEFAULT_LLM_STRATEGY,
                model=DEFAULT_MODEL,
                temperature=0.15,
                top_n_magnitude=60,
                target_positions_per_side=None,
                enable_loss_memory=False,
                loss_memory_top_k=3,
                loss_memory_max_targets_per_side=4,
                log_llm_trace=False,
            )
        )
        self.assertFalse(
            _cache_is_compatible(
                cache,
                "equal_weight",
                turnover_aware=True,
                strategy_style=DEFAULT_LLM_STRATEGY,
                model=DEFAULT_MODEL,
                temperature=0.15,
                top_n_magnitude=60,
                target_positions_per_side=None,
                enable_loss_memory=False,
                loss_memory_top_k=3,
                loss_memory_max_targets_per_side=4,
                log_llm_trace=False,
            )
        )
        self.assertEqual(
            build_cache_filename("2019-01-01", "2019-12-31", 5, True, False),
            f"llm_portfolio_deterministic_2019-01-01_2019-12-31_5d_equal_weight_signal_only_{DATASET_VERSION}.json",
        )

    def test_cache_compatibility_requires_matching_strategy_style(self):
        cache = {
            "metadata": {
                "schema_version": 3,
                "execution_mode": "equal_weight",
                "turnover_aware": False,
                "strategy_style": "mean_reversion",
                "model": DEFAULT_MODEL,
                "temperature": 0.15,
                "top_n_magnitude": 60,
                "strategy_signal_source": "trail_return_5d_rank_v1",
                "prompt_style": prompt_style_label(False),
                "prompt_format_version": PROMPT_FORMAT_VERSION,
                "target_positions_per_side": None,
                "log_llm_trace": False,
                "loss_memory_enabled": False,
                "loss_memory_top_k": 3,
                "loss_memory_max_targets_per_side": 4,
                "llm_source_policy": "strict_llm_only",
                "source_dataset": DATASET_VERSION,
                "composite_scores_dir": str(COMPOSITE_SCORES_DIR),
            },
            "portfolios": {
                "2024-01-05": {
                    "llm_source": "llm",
                    "long_basket": [{"ticker": "AAA", "llm_source": "llm", "weight_raw": 0.5, "weight_exec": 0.5}],
                    "short_basket": [{"ticker": "BBB", "llm_source": "llm", "weight_raw": 0.5, "weight_exec": 0.5}],
                }
            },
        }

        self.assertFalse(
            _cache_is_compatible(
                cache,
                "equal_weight",
                turnover_aware=False,
                strategy_style="deterministic",
                model=DEFAULT_MODEL,
                temperature=0.15,
                top_n_magnitude=60,
                target_positions_per_side=None,
                enable_loss_memory=False,
                loss_memory_top_k=3,
                loss_memory_max_targets_per_side=4,
                log_llm_trace=False,
            )
        )

    def test_cache_compatibility_requires_matching_book_size_and_trace_mode(self):
        cache = {
            "metadata": {
                "schema_version": 3,
                "execution_mode": "equal_weight",
                "turnover_aware": False,
                "strategy_style": DEFAULT_LLM_STRATEGY,
                "model": DEFAULT_MODEL,
                "temperature": 0.15,
                "top_n_magnitude": 60,
                "strategy_signal_source": "technical_composite_v1",
                "prompt_style": prompt_style_label(False),
                "prompt_format_version": PROMPT_FORMAT_VERSION,
                "target_positions_per_side": 100,
                "log_llm_trace": True,
                "loss_memory_enabled": False,
                "loss_memory_top_k": 3,
                "loss_memory_max_targets_per_side": 4,
                "llm_source_policy": "strict_llm_only",
                "source_dataset": DATASET_VERSION,
                "composite_scores_dir": str(COMPOSITE_SCORES_DIR),
            },
            "portfolios": {
                "2024-01-05": {
                    "llm_source": "llm",
                    "long_basket": [{"ticker": "AAA", "llm_source": "llm", "weight_raw": 0.5, "weight_exec": 0.5}],
                    "short_basket": [{"ticker": "BBB", "llm_source": "llm", "weight_raw": 0.5, "weight_exec": 0.5}],
                }
            },
        }

        self.assertFalse(
            _cache_is_compatible(
                cache,
                "equal_weight",
                turnover_aware=False,
                strategy_style=DEFAULT_LLM_STRATEGY,
                model=DEFAULT_MODEL,
                temperature=0.15,
                top_n_magnitude=60,
                target_positions_per_side=None,
                enable_loss_memory=False,
                loss_memory_top_k=3,
                loss_memory_max_targets_per_side=4,
                log_llm_trace=True,
            )
        )
        self.assertFalse(
            _cache_is_compatible(
                cache,
                "equal_weight",
                turnover_aware=False,
                strategy_style=DEFAULT_LLM_STRATEGY,
                model=DEFAULT_MODEL,
                temperature=0.15,
                top_n_magnitude=60,
                target_positions_per_side=100,
                enable_loss_memory=False,
                loss_memory_top_k=3,
                loss_memory_max_targets_per_side=4,
                log_llm_trace=False,
            )
        )

    def test_cache_compatibility_requires_matching_model_temperature_and_topn(self):
        cache = {
            "metadata": {
                "schema_version": 3,
                "execution_mode": "equal_weight",
                "turnover_aware": False,
                "strategy_style": DEFAULT_LLM_STRATEGY,
                "model": DEFAULT_MODEL,
                "temperature": 0.15,
                "top_n_magnitude": 60,
                "strategy_signal_source": "technical_composite_v1",
                "prompt_style": prompt_style_label(False),
                "prompt_format_version": PROMPT_FORMAT_VERSION,
                "target_positions_per_side": None,
                "log_llm_trace": False,
                "loss_memory_enabled": False,
                "loss_memory_top_k": 3,
                "loss_memory_max_targets_per_side": 4,
                "llm_source_policy": "strict_llm_only",
                "source_dataset": DATASET_VERSION,
                "composite_scores_dir": str(COMPOSITE_SCORES_DIR),
            },
            "portfolios": {
                "2024-01-05": {
                    "llm_source": "llm",
                    "long_basket": [{"ticker": "AAA", "llm_source": "llm", "weight_raw": 0.5, "weight_exec": 0.5}],
                    "short_basket": [{"ticker": "BBB", "llm_source": "llm", "weight_raw": 0.5, "weight_exec": 0.5}],
                }
            },
        }

        self.assertFalse(
            _cache_is_compatible(
                cache,
                "equal_weight",
                turnover_aware=False,
                strategy_style=DEFAULT_LLM_STRATEGY,
                model="different-model",
                temperature=0.15,
                top_n_magnitude=60,
                target_positions_per_side=None,
                enable_loss_memory=False,
                loss_memory_top_k=3,
                loss_memory_max_targets_per_side=4,
                log_llm_trace=False,
            )
        )
        self.assertFalse(
            _cache_is_compatible(
                cache,
                "equal_weight",
                turnover_aware=False,
                strategy_style=DEFAULT_LLM_STRATEGY,
                model=DEFAULT_MODEL,
                temperature=0.30,
                top_n_magnitude=60,
                target_positions_per_side=None,
                enable_loss_memory=False,
                loss_memory_top_k=3,
                loss_memory_max_targets_per_side=4,
                log_llm_trace=False,
            )
        )
        self.assertFalse(
            _cache_is_compatible(
                cache,
                "equal_weight",
                turnover_aware=False,
                strategy_style=DEFAULT_LLM_STRATEGY,
                model=DEFAULT_MODEL,
                temperature=0.15,
                top_n_magnitude=80,
                target_positions_per_side=None,
                enable_loss_memory=False,
                loss_memory_top_k=3,
                loss_memory_max_targets_per_side=4,
                log_llm_trace=False,
            )
        )

    def test_cache_filename_separates_model_temperature_and_topn_variants(self):
        baseline = build_cache_filename(
            "2019-01-01",
            "2019-12-31",
            5,
            True,
            False,
            model=DEFAULT_MODEL,
            temperature=0.15,
            top_n_magnitude=60,
        )
        variant = build_cache_filename(
            "2019-01-01",
            "2019-12-31",
            5,
            True,
            False,
            model="other-model",
            temperature=0.30,
            top_n_magnitude=80,
        )

        self.assertNotEqual(baseline, variant)
        self.assertIn("mdl-", baseline)
        self.assertIn("top60", baseline)
        self.assertIn("temp0p15", baseline)

    def test_strategy_resolver_returns_expected_modules(self):
        self.assertEqual(_resolve_llm_strategy_module("deterministic").__name__, "technical_llm")
        self.assertEqual(_resolve_llm_strategy_module("mean_reversion").__name__, "mean_reversion_llm")


class ComparisonMetricTests(unittest.TestCase):
    def test_compute_metrics_tracks_gross_net_and_cost_drag(self):
        pnl_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02")],
                "daily_ret_bps": [100.0],
                "gross_daily_ret_bps": [200.0],
                "cost_bps_applied": [100.0],
                "long_pnl_bps": [50.0],
                "short_pnl_bps": [50.0],
            }
        )

        metrics = compute_metrics(pnl_df, "LLM")

        self.assertAlmostEqual(metrics["net_total_return_pct"], 1.0)
        self.assertAlmostEqual(metrics["gross_total_return_pct"], 2.0)
        self.assertAlmostEqual(metrics["cost_drag_return_pct"], 1.0)

    def test_summarize_backtest_health_flags_flatline_no_book_output(self):
        pnl_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
                "daily_ret_bps": [0.0, 0.0],
                "gross_daily_ret_bps": [0.0, 0.0],
                "long_pnl_bps": [0.0, 0.0],
                "short_pnl_bps": [0.0, 0.0],
                "cost_bps_applied": [0.0, 0.0],
                "num_longs": [0, 0],
                "num_shorts": [0, 0],
            }
        )
        review_log_df = pd.DataFrame(
            {
                "executed": [False, False],
                "candidate_longs": [0, 0],
                "candidate_shorts": [0, 0],
            }
        )

        health = summarize_backtest_health(pnl_df, review_log_df=review_log_df)

        self.assertTrue(health["flatline_pnl"])
        self.assertFalse(health["has_positions"])
        self.assertFalse(health["has_candidates"])
        self.assertFalse(health["has_executions"])
        self.assertTrue(health["is_degenerate"])

    def test_summarize_backtest_health_accepts_live_book_activity(self):
        pnl_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
                "daily_ret_bps": [0.0, 12.0],
                "gross_daily_ret_bps": [0.0, 15.0],
                "long_pnl_bps": [0.0, 8.0],
                "short_pnl_bps": [0.0, 4.0],
                "cost_bps_applied": [0.0, 3.0],
                "num_longs": [30, 30],
                "num_shorts": [30, 30],
            }
        )
        review_log_df = pd.DataFrame(
            {
                "executed": [True],
                "candidate_longs": [30],
                "candidate_shorts": [30],
            }
        )

        health = summarize_backtest_health(pnl_df, review_log_df=review_log_df)

        self.assertTrue(health["has_positions"])
        self.assertTrue(health["has_candidates"])
        self.assertTrue(health["has_executions"])
        self.assertFalse(health["is_degenerate"])


if __name__ == "__main__":
    unittest.main()
