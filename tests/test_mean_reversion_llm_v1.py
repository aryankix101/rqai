import tempfile
import unittest
import json
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

import mean_reversion_llm_v1 as v1
import run_backtest_v3 as rb


def _make_histories(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str = "2020-06-30",
    signal_date: str | None = None,
    shocks: dict[str, float] | None = None,
    low_price: set[str] | None = None,
    low_adv: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(start, end)
    signal_ts = pd.Timestamp(signal_date) if signal_date else None
    shocks = shocks or {}
    low_price = low_price or set()
    low_adv = low_adv or set()
    histories = {}

    for idx, ticker in enumerate(tickers):
        day_idx = np.arange(len(dates), dtype=float)
        base = 20.0 + idx
        close = base * (1.0 + 0.0004 * day_idx + 0.03 * np.sin((day_idx + idx) / 7.0))
        if signal_ts is not None and ticker in shocks:
            close[dates >= signal_ts] *= 1.0 + shocks[ticker]
        if ticker in low_price and signal_ts is not None:
            close[dates == signal_ts] = 4.0
        volume = np.full(len(dates), 600_000 + idx * 10_000, dtype=float)
        if ticker in low_adv:
            volume[:] = 10_000.0
        raw_close = close.copy()
        histories[ticker] = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "open": raw_close * 0.999,
                "high": raw_close * 1.02,
                "low": raw_close * 0.98,
                "close": raw_close,
                "volume": volume,
                "adj_open": close * 0.999,
                "adj_high": close * 1.02,
                "adj_low": close * 0.98,
                "adj_close": close,
            }
        )
    return histories


def _membership(tickers: list[str]) -> rb.MembershipLookup:
    date = pd.Timestamp("2018-12-31")
    return rb.MembershipLookup(dates=(date,), symbols_by_date={date: set(tickers)})


def _config(start="2020-01-01", end="2020-06-30", tmpdir: str | None = None) -> rb.BenchmarkConfig:
    return rb.BenchmarkConfig(
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        output_dir=Path(tmpdir or "."),
        membership_path=Path("unused.csv"),
        cost_bps_levels=(0.0, 2.5, 5.0),
        min_universe_size=10,
    )


class MeanReversionLLMV1Tests(unittest.TestCase):
    def test_candidate_pool_applies_filters_and_selects_extreme_trail5(self):
        signal_date = "2020-03-20"
        tickers = [f"T{i:02d}" for i in range(12)]
        shocks = {
            "T00": -0.30,
            "T01": -0.25,
            "T02": -0.20,
            "T03": 0.30,
            "T04": 0.25,
            "T05": 0.20,
        }
        histories = _make_histories(
            tickers,
            signal_date=signal_date,
            shocks=shocks,
            low_price={"T06"},
            low_adv={"T07"},
        )
        market = rb.build_market_frames(histories)
        features = v1.build_feature_frames(histories)
        pool = v1.build_candidate_pool(
            market,
            features,
            _membership(tickers),
            spy_df=None,
            signal_date=pd.Timestamp(signal_date),
            pool_per_side=2,
            config=_config(),
        )

        self.assertIsNotNone(pool)
        assert pool is not None
        self.assertNotIn("T06", pool.valid_tickers)
        self.assertNotIn("T07", pool.valid_tickers)
        self.assertEqual(pool.long_tickers, ["T00", "T01"])
        self.assertEqual(pool.short_tickers, ["T03", "T04"])

    def test_prompt_uses_exact_v1_feature_set_and_excludes_old_fields(self):
        signal_date = "2020-03-20"
        tickers = [f"T{i:02d}" for i in range(8)]
        shocks = {"T00": -0.30, "T01": -0.20, "T02": 0.30, "T03": 0.20}
        histories = _make_histories(tickers, signal_date=signal_date, shocks=shocks)
        market = rb.build_market_frames(histories)
        features = v1.build_feature_frames(histories)
        pool = v1.build_candidate_pool(
            market,
            features,
            _membership(tickers),
            spy_df=None,
            signal_date=pd.Timestamp(signal_date),
            pool_per_side=2,
            config=_config(),
        )
        assert pool is not None

        prompt = v1.format_user_prompt(pool)
        self.assertIn(
            "Columns: Ticker  MRL  MRS  Trail5  MRSig  RSI5  Rng55  Vel/V  %MA100  52WH  Mom3m  AutoC  ADX  VolPctl",
            prompt,
        )
        forbidden = [
            "direction_score",
            "magnitude_score",
            "VixPctl",
            "VolImb",
            "M2/M1",
            "ATR",
            "RV10",
            "VAT_MA",
            "rationale",
            "rank",
        ]
        for token in forbidden:
            self.assertNotIn(token, prompt)

    def test_raw_confidence_prompt_style_has_separate_version_and_cache(self):
        self.assertEqual(
            v1.prompt_version_for_style(v1.PROMPT_STYLE_RAW_CONFIDENCE),
            v1.RAW_CONFIDENCE_PROMPT_VERSION,
        )
        system_prompt = v1.system_prompt_for_style(v1.PROMPT_STYLE_RAW_CONFIDENCE).lower()
        self.assertIn("raw confidence", system_prompt)
        self.assertIn("not a position size", system_prompt)
        self.assertIn("not a probability", system_prompt)

        args = v1.parse_args(
            [
                "--prompt-style",
                v1.PROMPT_STYLE_RAW_CONFIDENCE,
                "--start",
                "2009-01-01",
                "--end",
                "2014-01-01",
                "--pool-per-side",
                "60",
            ]
        )
        self.assertIn("raw_confidence", str(v1.cache_path_for_args(args)))
        self.assertIn("pool60", str(v1.cache_path_for_args(args)))

    def test_rank_confidence_prompt_style_has_separate_version_and_cache(self):
        self.assertEqual(
            v1.prompt_version_for_style(v1.PROMPT_STYLE_RANK_CONFIDENCE),
            v1.RANK_CONFIDENCE_PROMPT_VERSION,
        )
        system_prompt = v1.system_prompt_for_style(v1.PROMPT_STYLE_RANK_CONFIDENCE).lower()
        self.assertIn("rank + confidence", system_prompt)
        self.assertIn("rank 1", system_prompt)
        self.assertIn("confidence_score", system_prompt)

        args = v1.parse_args(
            [
                "--prompt-style",
                v1.PROMPT_STYLE_RANK_CONFIDENCE,
                "--start",
                "2009-01-01",
                "--end",
                "2014-01-01",
                "--pool-per-side",
                "60",
            ]
        )
        self.assertIn("rank_confidence", str(v1.cache_path_for_args(args)))
        self.assertIn("pool60", str(v1.cache_path_for_args(args)))

    def test_validate_rank_confidence_scores_normalizes_old_style_signal(self):
        cleaned = v1.validate_llm_scores(
            {
                "long_rankings": [
                    {"ticker": "BBB", "rank": 2, "confidence_score": 60},
                    {"ticker": "aaa", "rank": 1, "confidence_score": 80},
                ],
                "short_rankings": [
                    {"ticker": "DDD", "rank": 2, "confidence_score": 50},
                    {"ticker": "CCC", "rank": 1, "confidence_score": 90},
                ],
            },
            ["AAA", "BBB"],
            ["CCC", "DDD"],
            prompt_style=v1.PROMPT_STYLE_RANK_CONFIDENCE,
        )

        self.assertEqual(
            cleaned["long_scores"],
            [
                {"ticker": "AAA", "rank": 1, "confidence_score": 80, "score": 80},
                {"ticker": "BBB", "rank": 2, "confidence_score": 60, "score": 30},
            ],
        )
        self.assertEqual(
            cleaned["short_scores"],
            [
                {"ticker": "CCC", "rank": 1, "confidence_score": 90, "score": 90},
                {"ticker": "DDD", "rank": 2, "confidence_score": 50, "score": 25},
            ],
        )

        bad_rank = {
            "long_rankings": [
                {"ticker": "AAA", "rank": 1, "confidence_score": 80},
                {"ticker": "BBB", "rank": 1, "confidence_score": 60},
            ],
            "short_rankings": [
                {"ticker": "CCC", "rank": 1, "confidence_score": 90},
                {"ticker": "DDD", "rank": 2, "confidence_score": 50},
            ],
        }
        with self.assertRaises(ValueError):
            v1.validate_llm_scores(
                bad_rank,
                ["AAA", "BBB"],
                ["CCC", "DDD"],
                prompt_style=v1.PROMPT_STYLE_RANK_CONFIDENCE,
            )

    def test_validate_rank_confidence_scores_accepts_normalized_cache_entry(self):
        cleaned = v1.validate_llm_scores(
            {
                "long_scores": [
                    {"ticker": "AAA", "rank": 1, "confidence_score": 80, "score": 80},
                    {"ticker": "BBB", "rank": 2, "confidence_score": 60, "score": 30},
                ],
                "short_scores": [
                    {"ticker": "CCC", "rank": 1, "confidence_score": 90, "score": 90},
                    {"ticker": "DDD", "rank": 2, "confidence_score": 50, "score": 25},
                ],
            },
            ["AAA", "BBB"],
            ["CCC", "DDD"],
            prompt_style=v1.PROMPT_STYLE_RANK_CONFIDENCE,
        )

        self.assertEqual(
            cleaned["long_scores"],
            [
                {"ticker": "AAA", "rank": 1, "confidence_score": 80, "score": 80},
                {"ticker": "BBB", "rank": 2, "confidence_score": 60, "score": 30},
            ],
        )

    def test_validate_llm_scores_rounds_floats_and_rejects_bad_outputs(self):
        cleaned = v1.validate_llm_scores(
            {
                "long_scores": [{"ticker": "aaa", "score": 72.6}, {"ticker": "BBB", "score": 12.2}],
                "short_scores": [{"ticker": "CCC", "score": 99.4}, {"ticker": "DDD", "score": 0}],
            },
            ["AAA", "BBB"],
            ["CCC", "DDD"],
        )
        self.assertEqual(cleaned["long_scores"][0], {"ticker": "AAA", "score": 73})
        self.assertEqual(cleaned["short_scores"][0], {"ticker": "CCC", "score": 99})

        bad_cases = [
            {"long_scores": [{"ticker": "AAA", "score": 1}], "short_scores": [{"ticker": "CCC", "score": 2}]},
            {
                "long_scores": [{"ticker": "AAA", "score": 1}, {"ticker": "AAA", "score": 2}],
                "short_scores": [{"ticker": "CCC", "score": 2}, {"ticker": "DDD", "score": 3}],
            },
            {
                "long_scores": [{"ticker": "AAA", "score": -1}, {"ticker": "BBB", "score": 2}],
                "short_scores": [{"ticker": "CCC", "score": 2}, {"ticker": "DDD", "score": 3}],
            },
            {
                "long_scores": [{"ticker": "CCC", "score": 1}, {"ticker": "BBB", "score": 2}],
                "short_scores": [{"ticker": "AAA", "score": 2}, {"ticker": "DDD", "score": 3}],
            },
            {
                "long_scores": [{"ticker": "AAA", "score": "72"}, {"ticker": "BBB", "score": 2}],
                "short_scores": [{"ticker": "CCC", "score": 2}, {"ticker": "DDD", "score": 3}],
            },
            {
                "long_scores": [{"ticker": "AAA", "score": True}, {"ticker": "BBB", "score": 2}],
                "short_scores": [{"ticker": "CCC", "score": 2}, {"ticker": "DDD", "score": 3}],
            },
        ]
        for bad in bad_cases:
            with self.assertRaises(ValueError):
                v1.validate_llm_scores(bad, ["AAA", "BBB"], ["CCC", "DDD"])

    def test_signed_scores_are_positive_for_longs_negative_for_shorts(self):
        signed = v1.signed_scores_from_llm(
            {
                "long_scores": [{"ticker": "AAA", "score": 80}],
                "short_scores": [{"ticker": "BBB", "score": 70}],
            }
        )
        self.assertEqual(signed, {"AAA": 80.0, "BBB": -70.0})
        self.assertNotIn("CCC", signed)

    def test_ensure_llm_score_recalls_invalid_rank_confidence_cache(self):
        pool = v1.CandidatePool(
            signal_date=pd.Timestamp("2020-01-03"),
            lookback_date=pd.Timestamp("2019-12-20"),
            exit_date=pd.Timestamp("2020-01-10"),
            entry_open_date=pd.Timestamp("2020-01-06"),
            exit_open_date=pd.Timestamp("2020-01-13"),
            valid_tickers=["AAA", "BBB", "CCC", "DDD"],
            long_tickers=["AAA", "BBB"],
            short_tickers=["CCC", "DDD"],
            trail_by_ticker={},
            signal_by_ticker={},
            feature_rows={},
            universe_hash="hash1",
            regime="mixed",
        )
        model = "test-model"
        prompt_style = v1.PROMPT_STYLE_RANK_CONFIDENCE
        prompt_version = v1.prompt_version_for_style(prompt_style)
        date_key = str(pool.signal_date.date())
        cache = {
            "metadata": {},
            "signals": {
                date_key: {
                    "cache_key": v1._cache_key(pool.signal_date, prompt_version, model, pool.universe_hash),
                    "long_scores": [{"ticker": "AAA", "score": 80}],
                    "short_scores": [{"ticker": "CCC", "score": 90}],
                }
            },
            "failures": {},
        }
        calls = 0

        async def fake_call_llm_scores(*args, **kwargs):
            nonlocal calls
            calls += 1
            return {
                "long_scores": [
                    {"ticker": "AAA", "rank": 1, "confidence_score": 80, "score": 80},
                    {"ticker": "BBB", "rank": 2, "confidence_score": 60, "score": 30},
                ],
                "short_scores": [
                    {"ticker": "CCC", "rank": 1, "confidence_score": 90, "score": 90},
                    {"ticker": "DDD", "rank": 2, "confidence_score": 50, "score": 25},
                ],
                "parse_status": "ok",
                "retry_count": 0,
            }

        original_call = v1.call_llm_scores
        v1.call_llm_scores = fake_call_llm_scores
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = asyncio.run(
                    v1.ensure_llm_score_for_pool(
                        client=None,
                        pool=pool,
                        cache=cache,
                        cache_path=Path(tmpdir) / "scores.json",
                        model=model,
                        temperature=0.0,
                        prompt_style=prompt_style,
                    )
                )
        finally:
            v1.call_llm_scores = original_call

        self.assertEqual(calls, 1)
        self.assertEqual(result["long_scores"][0]["ticker"], "AAA")
        self.assertNotIn(date_key, cache["failures"])

    def test_small_fixture_writes_three_run_artifacts_and_comparison(self):
        tickers = [f"T{i:02d}" for i in range(30)]
        histories = _make_histories(tickers)
        market = rb.build_market_frames(histories)
        features = v1.build_feature_frames(histories)
        membership = _membership(tickers)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _config("2019-04-01", "2020-06-01", tmpdir)
            baseline_universe = v1.baseline_universe_bundle(market, membership, config)
            baseline_pool, pools = v1.baseline_pool_bundle(
                market,
                features,
                membership,
                spy_df=None,
                config=config,
                pool_per_side=5,
                positions_per_side=2,
            )
            llm_scores = {}
            for pool in pools:
                llm_scores[str(pool.signal_date.date())] = {
                    "long_scores": [
                        {"ticker": ticker, "score": 100 - idx}
                        for idx, ticker in enumerate(pool.long_tickers)
                    ],
                    "short_scores": [
                        {"ticker": ticker, "score": 100 - idx}
                        for idx, ticker in enumerate(pool.short_tickers)
                    ],
                }
            weekly, checks, selections, _ = v1.run_pool_strategy(
                market,
                features,
                membership,
                spy_df=None,
                config=config,
                pool_per_side=5,
                positions_per_side=2,
                score_provider="llm",
                llm_scores_by_date=llm_scores,
            )
            llm_pool = v1.build_bundle(weekly, checks, selections, config)
            bundles = {
                "Baseline-Universe": baseline_universe,
                "Baseline-Pool": baseline_pool,
                "LLM-Pool v1": llm_pool,
            }
            out = Path(tmpdir)
            for name, bundle in bundles.items():
                v1.save_bundle(bundle, out / name.replace(" ", "_").lower())
            comparison = v1.build_comparison_scorecard(bundles)
            comparison.to_csv(out / "comparison_scorecard.csv", index=False)

            self.assertTrue((out / "comparison_scorecard.csv").exists())
            self.assertEqual(set(comparison["run"]), set(bundles))
            self.assertTrue((out / "baseline-universe" / "weekly_returns.csv").exists())
            self.assertTrue((out / "llm-pool_v1" / "portfolio_selections.csv").exists())

    def test_copy_like_correlation_summary_is_json_serializable(self):
        corr_df = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2009-02-20"),
                    "pool_side": "LONG",
                    "score_trail5_spearman": -0.98,
                },
                {
                    "date": pd.Timestamp("2009-02-20"),
                    "pool_side": "SHORT",
                    "score_trail5_spearman": 0.25,
                },
            ]
        )

        summary = v1.copy_like_correlation_summary(corr_df)
        json.dumps(summary)

        self.assertEqual(summary["copy_like_correlation_periods"], 1)
        self.assertEqual(summary["copy_like_correlation_sample"][0]["date"], "2009-02-20")


if __name__ == "__main__":
    unittest.main()
