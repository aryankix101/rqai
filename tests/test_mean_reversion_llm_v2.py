"""Test the V2 prompt contract, cache flow, replay logic, and report outputs.

These regression tests keep scorer behavior, accounting, experimental guards,
and offline artifact generation stable as the V2 runner evolves.
"""

import asyncio
import ast
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import mean_reversion_llm_v1 as v1
import mean_reversion_llm_v2 as v2
import run_backtest_v3 as rb
import technical_llm
from tests.test_mean_reversion_llm_v1 import _make_histories, _membership


def _v2_config(start: str, end: str, tmpdir: str) -> rb.BenchmarkConfig:
    return rb.BenchmarkConfig(
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        output_dir=Path(tmpdir),
        membership_path=Path("unused.csv"),
        cost_bps_levels=(0.0, 2.5, 5.0),
        cadence_mode="trading_days",
        signal_lookback_days=5,
        rebalance_days=5,
        holding_days=5,
        min_universe_size=10,
    )


def _score_pool() -> v1.CandidatePool:
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    return v1.CandidatePool(
        signal_date=pd.Timestamp("2020-01-10"),
        lookback_date=pd.Timestamp("2020-01-03"),
        exit_date=pd.Timestamp("2020-01-17"),
        entry_open_date=pd.Timestamp("2020-01-13"),
        exit_open_date=pd.Timestamp("2020-01-21"),
        valid_tickers=tickers,
        long_tickers=["AAA", "BBB"],
        short_tickers=["CCC", "DDD"],
        trail_by_ticker={"AAA": -0.1, "BBB": -0.08, "CCC": 0.09, "DDD": 0.07},
        signal_by_ticker={"AAA": 0.1, "BBB": 0.08, "CCC": -0.09, "DDD": -0.07},
        feature_rows={
            ticker: {
                "Ticker": ticker,
                "MRL": 100.0,
                "MRS": 0.0,
                "Trail5": -10.0,
                "MRSig": 10.0,
                "RSI5": 20.0,
                "Rng55": 0.2,
                "Vel/V": -0.5,
                "%MA100": -5.0,
                "52WH": -20.0,
                "Mom3m": 1.0,
                "AutoC": -0.1,
                "ADX": 30.0,
                "VolPctl": 80.0,
            }
            for ticker in tickers
        },
        universe_hash="safe-thinking-test-hash",
        regime="normal",
        market_context={
            "spy_vs_200d_ma": "below",
            "spy_distance_from_200d_ma_pct": -19.7,
            "vix_level": 38.6,
            "vix_percentile_252d": 73.1,
        },
    )


def _write_experimental_prompt_spec(
    root: Path,
    candidate_id: str = "g0-c01",
    *,
    anonymize_tickers: bool = True,
) -> Path:
    parent = v2.V2_MINIMAL_CHANGE_CONTROL_PROMPT
    old_text = "Use the 0-100 range only when the supplied cross-section supports it. Do not manufacture separation."
    new_text = "Use the 0-100 range only when the supplied cross-section supports it. Preserve genuine separation."
    prompt_text = parent.replace(old_text, new_text, 1)
    payload = {
        "candidate_id": candidate_id,
        "generation": "G0",
        "parent_id": "v2-minimal-change-control",
        "mutation_mode": "substitutive",
        "patch": {"operation": "replace_once", "old_text": old_text, "new_text": new_text},
        "hypothesis": "Clarifying genuine dispersion may improve ranking quality.",
        "mechanism": "Discourage artificial compression without forcing extremes.",
        "expected_failure": "The wording may have no measurable effect.",
        "prior_experiment_ids": ["v2-minimal-change-control"],
        "prompt_text": prompt_text,
        "prompt_hash": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "output_contract": "candidate_declared_numeric_score",
        "score_min": 0,
        "score_max": 100,
        "score_direction": "higher_is_better",
        "anonymize_tickers": anonymize_tickers,
    }
    path = root / f"{candidate_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_signal_manifest(root: Path, dates: list[str], name: str = "dates.json") -> Path:
    canonical_hash = hashlib.sha256(
        json.dumps(dates, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path = root / name
    path.write_text(
        json.dumps({"signal_dates": dates, "date_manifest_hash": canonical_hash}, indent=2),
        encoding="utf-8",
    )
    return path


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class MeanReversionLLMV2Tests(unittest.TestCase):
    def test_v2_uses_single_hash_versioned_prompt_contract(self):
        self.assertEqual(v2.V2_PROMPT_STYLE, v1.PROMPT_STYLE_V1_SCORE)
        self.assertIn("FEATURE GLOSSARY", v2.V2_SYSTEM_PROMPT)
        self.assertEqual(v2.prompt_text_for_variant("v2"), v2.V2_SYSTEM_PROMPT)
        self.assertIn(v2.prompt_hash_for_variant("v2"), v2.prompt_version_for_variant("v2"))
        self.assertEqual(
            v2.PROMPT_VARIANTS,
            (
                "v2",
                "v2-frozen",
                "v2.2",
                "v2.3",
                "v2.4",
                "v2.5",
                "v2.6",
                "v2-minimal-change-control",
                "v2-no-regime",
            ),
        )

    def test_v2_no_regime_removes_only_regime_glossary_and_user_line(self):
        pool = _score_pool()
        standard = v1.format_user_prompt(pool)
        ablated = v2.format_v2_no_regime_user_prompt(pool)
        args = v2.parse_args(
            [
                "--prompt-variant", "v2-no-regime",
                "--temperature", "1",
                "--experimental-override",
            ]
        )

        expected_system_prompt = v2.V2_SYSTEM_PROMPT.replace(
            v2.V2_NO_REGIME_GLOSSARY_BLOCK,
            "",
            1,
        )
        self.assertEqual(v2.prompt_text_for_variant("v2-no-regime"), expected_system_prompt)
        self.assertNotIn("Market Regime\n- normal:", expected_system_prompt)
        self.assertIn("supportive regime", expected_system_prompt)
        self.assertNotEqual(
            v2.prompt_hash_for_variant("v2-no-regime"),
            v2.prompt_hash_for_variant("v2"),
        )
        self.assertEqual(
            ablated,
            standard.replace(f"MARKET REGIME: {pool.regime}\n\n", "", 1),
        )
        self.assertNotIn("MARKET REGIME:", ablated)
        self.assertNotIn("SPY vs 200-day MA", ablated)
        self.assertNotIn("VIX level", ablated)
        self.assertEqual(v2.output_contract_for_variant("v2-no-regime"), "score_0_100")
        self.assertIn("MARKET REGIME line omitted", v2.prompt_source_for_variant("v2-no-regime"))
        self.assertIn(
            f"_v2nr-{v2.prompt_hash_for_variant('v2-no-regime')}_anon.json",
            v2.cache_path_for_args(args).name,
        )
        self.assertFalse(v2.should_collect_fresh_v2_control("v2-no-regime", True))
        self.assertTrue(v2.should_collect_fresh_v2_control("v2.5", True))

    def test_v26_and_minimal_change_prompts_have_distinct_provenance(self):
        v26_args = v2.parse_args(["--prompt-variant", "v2.6", "--experimental-override"])
        minimal_args = v2.parse_args(
            ["--prompt-variant", "v2-minimal-change-control", "--experimental-override"]
        )

        self.assertEqual(v2.prompt_text_for_variant("v2.6"), v2.V26_SYSTEM_PROMPT)
        self.assertEqual(
            v2.prompt_text_for_variant("v2-minimal-change-control"),
            v2.V2_MINIMAL_CHANGE_CONTROL_PROMPT,
        )
        self.assertEqual(v2.output_contract_for_variant("v2.6"), "score_0_100")
        self.assertEqual(v2.output_contract_for_variant("v2-minimal-change-control"), "score_0_100")
        self.assertIn("_v26-", v2.cache_path_for_args(v26_args).name)
        self.assertIn("_v2mc-", v2.cache_path_for_args(minimal_args).name)
        self.assertNotEqual(
            v2.prompt_hash_for_variant("v2.6"),
            v2.prompt_hash_for_variant("v2-minimal-change-control"),
        )

    def test_frozen_v2_reuses_exact_original_score_prompt_and_separate_cache(self):
        args = v2.parse_args(
            ["--prompt-variant", "v2-frozen", "--anonymize-tickers", "--model", "deepseek-v4-flash"]
        )
        prompt_hash = v2.prompt_hash_for_variant("v2-frozen")

        self.assertIn("the 150 biggest 5-day losers", v2.prompt_text_for_variant("v2-frozen"))
        self.assertIn("historical v1 prompt", v2.prompt_source_for_variant("v2-frozen"))
        self.assertNotEqual(v2.prompt_text_for_variant("v2-frozen"), v2.V2_SYSTEM_PROMPT)
        self.assertEqual(v2.output_contract_for_variant("v2-frozen"), "score_0_100")
        self.assertTrue(str(v2.cache_path_for_args(args)).endswith(f"_v2f-{prompt_hash}_anon.json"))

    def test_v2_defaults_are_gross_only(self):
        args = v2.parse_args([])

        self.assertEqual(v2.DEFAULT_COST_BPS_LEVELS, (0.0,))
        self.assertEqual(tuple(args.cost_bps_levels), (0.0,))
        self.assertEqual(args.schedule, "anchors")
        self.assertEqual(tuple(args.pool_sizes), (60,))
        self.assertEqual(tuple(args.book_sizes), (30,))
        self.assertEqual(args.llm_pool_size, 60)
        self.assertEqual(args.model, "deepseek-v4-flash")
        self.assertEqual(args.thinking_mode, "disabled")
        self.assertFalse(args.v25_paired_control)

    def test_experimental_prompt_spec_materializes_one_score_only_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_experimental_prompt_spec(Path(tmpdir))
            spec = v2.load_experimental_prompt_spec(path)
            args = v2.parse_args(
                [
                    "--experimental-prompt-spec", str(path),
                    "--temperature", "1",
                    "--anchor-offsets", "3",
                    "--experimental-override",
                ]
            )

            self.assertEqual(spec.candidate_id, "g0-c01")
            self.assertEqual(spec.generation, "G0")
            self.assertEqual(spec.output_contract, "candidate_declared_numeric_score")
            self.assertEqual((spec.score_min, spec.score_max), (0.0, 100.0))
            self.assertEqual(spec.score_direction, "higher_is_better")
            self.assertTrue(spec.anonymize_tickers)
            self.assertEqual(spec.prompt_hash, hashlib.sha256(spec.prompt_text.encode()).hexdigest()[:12])
            self.assertIn(f"_exp-g0-c01-{spec.prompt_hash}_", v2.cache_path_for_args(args).name)
            self.assertEqual(v2.prompt_hash_for_variant("v2"), v2.INCUMBENT_PROMPT_HASH)

            visible_path = _write_experimental_prompt_spec(
                Path(tmpdir), candidate_id="g0-visible", anonymize_tickers=False
            )
            visible_spec = v2.load_experimental_prompt_spec(visible_path)
            self.assertFalse(visible_spec.anonymize_tickers)

    def test_experimental_prompt_spec_rejects_output_contract_mutation_and_unsafe_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = _write_experimental_prompt_spec(root)
            payload = json.loads(path.read_text(encoding="utf-8"))
            parent = v2.V2_MINIMAL_CHANGE_CONTROL_PROMPT
            old_text = '"long_scores":  [{"ticker": "SEC_0001", "score": 72}'
            new_text = '"rankings":  [{"ticker": "SEC_0001", "score": 72}'
            prompt = parent.replace(old_text, new_text, 1)
            payload["patch"] = {"operation": "replace_once", "old_text": old_text, "new_text": new_text}
            payload["prompt_text"] = prompt
            payload["prompt_hash"] = hashlib.sha256(prompt.encode()).hexdigest()
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "long_scores and short_scores"):
                v2.load_experimental_prompt_spec(path)

            path = _write_experimental_prompt_spec(root, candidate_id="safe")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["candidate_id"] = ".."
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "path component"):
                v2.load_experimental_prompt_spec(path)

    def test_autoresearch_native_score_contract_is_canonicalized(self):
        result = v2.validate_and_canonicalize_autoresearch_scores(
            {
                "long_scores": [{"ticker": "L1", "score": -100}, {"ticker": "L2", "score": 100}],
                "short_scores": [{"ticker": "S1", "score": 0}],
            },
            expected_longs=["L1", "L2"],
            expected_shorts=["S1"],
            score_min=-100,
            score_max=100,
            score_direction="lower_is_better",
        )

        self.assertEqual([row["score"] for row in result["long_scores"]], [100.0, 0.0])
        self.assertEqual(result["short_scores"][0]["score"], 50.0)
        self.assertEqual(result["native_score_contract"]["score_direction"], "lower_is_better")

    def test_signal_manifest_is_exact_sorted_and_firewalls_validation_and_holdout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            development = v2.load_signal_date_manifest(
                _write_signal_manifest(root, ["2013-01-04", "2013-01-11"])
            )
            config = _v2_config("2009-01-01", "2014-01-01", tmpdir)
            self.assertEqual(v2.validate_autoresearch_manifest_scope(development, config), "development")

            validation = v2.load_signal_date_manifest(
                _write_signal_manifest(root, ["2015-01-09"], "validation.json")
            )
            validation_config = _v2_config("2015-01-01", "2019-12-31", tmpdir)
            self.assertEqual(v2.validate_autoresearch_manifest_scope(validation, validation_config), "validation")

            holdout = v2.load_signal_date_manifest(
                _write_signal_manifest(root, ["2020-01-03"], "holdout.json")
            )
            holdout_config = _v2_config("2020-01-01", "2020-12-31", tmpdir)
            with self.assertRaisesRegex(ValueError, "2020 or later"):
                v2.validate_autoresearch_manifest_scope(holdout, holdout_config)
            with self.assertRaisesRegex(ValueError, "signal or exit dates"):
                v2.validate_autoresearch_pool_dates([_score_pool()], "validation")

            for bad_date, bad_start, bad_end in (
                ("2008-12-31", "2008-01-01", "2014-01-01"),
                ("2014-01-02", "2009-01-01", "2014-01-02"),
            ):
                bad_manifest = v2.load_signal_date_manifest(
                    _write_signal_manifest(root, [bad_date], f"bad-{bad_date}.json")
                )
                with self.assertRaisesRegex(ValueError, "2009-01-01 through 2014-01-01"):
                    v2.validate_autoresearch_manifest_scope(
                        bad_manifest,
                        _v2_config(bad_start, bad_end, tmpdir),
                    )

            exit_crosses_holdout = replace(
                _score_pool(),
                signal_date=pd.Timestamp("2019-12-24"),
                lookback_date=pd.Timestamp("2019-12-17"),
                exit_date=pd.Timestamp("2020-01-02"),
            )
            with self.assertRaisesRegex(ValueError, "2020 or later"):
                v2.validate_autoresearch_pool_dates(
                    [exit_crosses_holdout], "validation"
                )

            unsorted = root / "unsorted.json"
            unsorted.write_text(json.dumps(["2013-01-11", "2013-01-04"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sorted"):
                v2.load_signal_date_manifest(unsorted)

    def test_validation_freeze_binds_prompt_dates_code_harness_and_development_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = v2.load_experimental_prompt_spec(_write_experimental_prompt_spec(root))
            manifest = v2.load_signal_date_manifest(
                _write_signal_manifest(root, ["2015-01-09"], "validation.json")
            )
            args = v2.parse_args(
                [
                    "--phase", "full",
                    "--start", "2015-01-01",
                    "--end", "2019-12-31",
                    "--temperature", "1",
                    "--anchor-offsets", "3",
                    "--experimental-override",
                ]
            )
            freeze_path = root / "freeze.json"
            development_path = root / "development_result.json"
            development_path.write_text(
                json.dumps({"candidate_id": spec.candidate_id, "status": "completed"}),
                encoding="utf-8",
            )
            lineage = [
                {
                    "candidate_id": spec.candidate_id,
                    "generation": "G0",
                    "parent_id": "v2-minimal-change-control",
                    "prompt_hash": spec.prompt_sha256,
                    "prompt_spec_hash": spec.spec_sha256,
                }
            ]
            freeze = {
                "approved": True,
                "campaign_id": "unit-freeze",
                "candidate_id": spec.candidate_id,
                "prompt_hash": spec.prompt_sha256,
                "prompt_spec_hash": spec.spec_sha256,
                "date_manifest_hash": manifest.date_manifest_hash,
                "date_manifest_file_hash": manifest.source_sha256,
                "lineage": lineage,
                "lineage_hash": hashlib.sha256(
                    json.dumps(
                        lineage,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                ).hexdigest(),
                "code_fingerprint": v2.autoresearch_code_fingerprint(),
                "supervisor_code_fingerprint": "c" * 64,
                "harness_fingerprint": v2.autoresearch_harness_fingerprint(args),
                "development_results_hash": hashlib.sha256(
                    development_path.read_bytes()
                ).hexdigest(),
                "development_results_path": str(development_path.resolve()),
            }
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            loaded = v2.load_and_validate_validation_freeze(freeze_path, spec, manifest, args)
            self.assertEqual(loaded["candidate_id"], spec.candidate_id)
            freeze["harness_fingerprint"] = "b" * 64
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "harness fingerprint"):
                v2.load_and_validate_validation_freeze(freeze_path, spec, manifest, args)

    def test_autoresearch_configuration_is_flash_temp_one_disabled_and_gross(self):
        args = v2.parse_args(
            [
                "--phase", "full",
                "--temperature", "1",
                "--anchor-offsets", "3",
                "--experimental-override",
            ]
        )
        v2.validate_autoresearch_args(args)
        visible = v2.parse_args(
            [
                "--phase", "full",
                "--temperature", "1",
                "--anchor-offsets", "3",
                "--no-anonymize-tickers",
                "--experimental-override",
            ]
        )
        v2.validate_autoresearch_args(visible)
        for field, value, message in (
            ("temperature", 0.7, "temperature"),
            ("thinking_mode", "enabled", "thinking mode"),
            ("model", "deepseek-v4-pro", "model"),
        ):
            bad = v2.parse_args(
                [
                    "--phase", "full",
                    "--temperature", "1",
                    "--anchor-offsets", "3",
                    "--experimental-override",
                ]
            )
            setattr(bad, field, value)
            with self.assertRaisesRegex(ValueError, message):
                v2.validate_autoresearch_args(bad)

    def test_progress_file_is_append_only_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            v2.append_progress_event(path, "runner_started", candidate_id="g0-c01")
            v2.append_progress_event(path, "cache_hit", signal_date="2013-01-04")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["event"] for row in rows], ["runner_started", "cache_hit"])
            self.assertTrue(all(row["artifact_type"] == "llm_reversal_signal_v2_progress" for row in rows))

    def test_nested_autoresearch_cache_can_expand_and_project_but_rejects_partial_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = v2.load_experimental_prompt_spec(_write_experimental_prompt_spec(root))
            small = v2.load_signal_date_manifest(_write_signal_manifest(root, ["2013-01-04"], "small.json"))
            large = v2.load_signal_date_manifest(
                _write_signal_manifest(root, ["2013-01-04", "2013-01-11"], "large.json")
            )
            cache_path = root / "cache.json"
            v1.save_cache(
                {
                    "metadata": {"experimental_prompt_spec_sha256": spec.spec_sha256},
                    "signals": {"2013-01-04|pool60": {"rebalance_date": "2013-01-04"}},
                },
                cache_path,
            )
            expanded = asyncio.run(
                v2.collect_llm_scores_for_v2_pools(
                    {}, cache_path, "deepseek-v4-flash", 1.0, 0.0,
                    thinking_mode="disabled",
                    anonymize_tickers=True,
                    experimental_prompt_spec=spec,
                    signal_date_manifest=large,
                    autoresearch_mode=True,
                )
            )
            self.assertIn("2013-01-04|pool60", expanded)
            expanded_cache = v1.load_cache(cache_path)
            expanded_cache["signals"]["2013-01-11|pool60"] = {"rebalance_date": "2013-01-11"}
            v1.save_cache(expanded_cache, cache_path)
            projected = asyncio.run(
                v2.collect_llm_scores_for_v2_pools(
                    {}, cache_path, "deepseek-v4-flash", 1.0, 0.0,
                    thinking_mode="disabled",
                    anonymize_tickers=True,
                    experimental_prompt_spec=spec,
                    signal_date_manifest=small,
                    autoresearch_mode=True,
                )
            )
            self.assertEqual(set(projected), {"2013-01-04|pool60"})
            rebound = v1.load_cache(cache_path)
            self.assertEqual(len(rebound["metadata"]["authorized_date_manifest_hashes"]), 2)

            partial = v2.load_signal_date_manifest(
                _write_signal_manifest(root, ["2013-01-04", "2013-01-18"], "partial.json")
            )
            with self.assertRaisesRegex(ValueError, "not nested"):
                asyncio.run(
                    v2.collect_llm_scores_for_v2_pools(
                        {}, cache_path, "deepseek-v4-flash", 1.0, 0.0,
                        thinking_mode="disabled",
                        anonymize_tickers=True,
                        experimental_prompt_spec=spec,
                        signal_date_manifest=partial,
                        autoresearch_mode=True,
                    )
                )

    def test_bounded_nonthinking_call_forces_temperature_one_and_records_usage(self):
        class FakeClient:
            def __init__(self):
                self.payload = None

            async def post(self, _url, json, headers):
                self.payload = json
                return _FakeResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": json_module.dumps(
                                        {
                                            "long_scores": [
                                                {"ticker": "AAA", "score": 90},
                                                {"ticker": "BBB", "score": 70},
                                            ],
                                            "short_scores": [
                                                {"ticker": "CCC", "score": 85},
                                                {"ticker": "DDD", "score": 65},
                                            ],
                                        }
                                    )
                                },
                            }
                        ],
                        "usage": {"completion_tokens": 100},
                    }
                )

        json_module = json
        client = FakeClient()
        usage = []
        result = asyncio.run(
            v2.call_llm_scores_bounded_nonthinking(
                client=client,
                pool=_score_pool(),
                model="deepseek-v4-flash",
                temperature=1.0,
                system_prompt=v2.V2_MINIMAL_CHANGE_CONTROL_PROMPT,
                prompt_identity="test",
                prompt_hash="abc123def456",
                max_output_tokens=4096,
                max_attempts=2,
                token_budget=v2.OutputTokenBudget(limit=10_000),
                call_budget=v2.LLMCallBudget(limit=10),
                usage_callback=usage.append,
            )
        )
        self.assertEqual(result["parse_status"], "ok")
        self.assertEqual(client.payload["temperature"], 1.0)
        self.assertEqual(client.payload["thinking"], {"type": "disabled"})
        self.assertEqual(client.payload["max_tokens"], 4096)
        self.assertEqual(usage[0]["output_tokens_charged"], 100)

    def test_format_only_diagnostic_requires_every_numeric_score(self):
        wrong_wrapper = {
            "result": {
                "long": [
                    {"symbol": "AAA", "rating": 90},
                    {"symbol": "BBB", "rating": 70},
                ],
                "short": [
                    {"symbol": "CCC", "rating": 85},
                    {"symbol": "DDD", "rating": 65},
                ],
            }
        }
        incomplete = {
            "long": [{"symbol": "AAA", "rating": 90}],
            "short": [{"symbol": "CCC", "rating": 85}],
        }

        self.assertTrue(
            v2._is_format_only_score_payload(
                wrong_wrapper,
                ["AAA", "BBB"],
                ["CCC", "DDD"],
            )
        )
        self.assertFalse(
            v2._is_format_only_score_payload(
                incomplete,
                ["AAA", "BBB"],
                ["CCC", "DDD"],
            )
        )

    def test_shared_autoresearch_budget_counts_control_and_treatment_caches_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            control = root / "control.json"
            treatment = root / "treatment.json"
            v1.save_cache(
                {
                    "llm_call_usage": [
                        {"status": "ok", "output_tokens_charged": 120},
                        {"status": "failed", "output_tokens_charged": 80},
                    ]
                },
                control,
            )
            v1.save_cache(
                {"llm_call_usage": [{"status": "ok", "output_tokens_charged": 100}]},
                treatment,
            )

            calls, tokens = v2._shared_autoresearch_budgets(
                [control, treatment, control],
                call_limit=4,
                output_token_limit=4_396,
            )
            self.assertEqual(calls.used, 3)
            self.assertEqual(tokens.used, 300)
            calls.reserve_request()
            tokens.reserve_request(4_096)
            with self.assertRaises(v2.LLMSafeStop):
                calls.reserve_request()

    def test_production_incumbent_is_frozen_to_anonymized_v2(self):
        args = v2.parse_args([])

        v2.validate_incumbent_args(args)
        self.assertTrue(args.anonymize_tickers)
        self.assertEqual(args.prompt_variant, "v2")
        self.assertEqual(args.temperature, 0.6)
        self.assertEqual(v2.prompt_hash_for_variant("v2"), v2.INCUMBENT_PROMPT_HASH)
        self.assertEqual(
            v2.incumbent_configuration_manifest()["scorer_source"],
            "mean_reversion_llm_v2.V2_SYSTEM_PROMPT",
        )

        visible_args = v2.parse_args(["--no-anonymize-tickers"])
        with self.assertRaisesRegex(ValueError, "tickers must be anonymized"):
            v2.validate_incumbent_args(visible_args)

    def test_v2_module_does_not_directly_import_legacy_or_generic_selector(self):
        tree = ast.parse(Path(v2.__file__).read_text(encoding="utf-8"))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }

        self.assertNotIn("mean_reversion_llm", imported_modules)
        self.assertNotIn("llm_portfolio_constructor", imported_modules)
        self.assertEqual(v2.prompt_text_for_variant("v2"), v2.V2_SYSTEM_PROMPT)

        v1_source = Path(v1.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from llm_portfolio_constructor import", v1_source)

    def test_deepseek_v4_flash_payload_uses_json_mode_and_explicit_non_thinking(self):
        payload = {"model": "deepseek-v4-flash", "messages": [], "stream": False}
        v2._harden_json_payload(payload, "deepseek-v4-flash", thinking_mode="disabled")

        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["max_tokens"], 8192)

    def test_deepseek_v4_flash_payload_can_enable_thinking(self):
        payload = {"model": "deepseek-v4-flash", "messages": [], "stream": False}
        v2._harden_json_payload(
            payload,
            "deepseek-v4-flash",
            thinking_mode="enabled",
            max_output_tokens=16_000,
        )

        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["max_tokens"], 16_000)
        self.assertFalse(technical_llm.model_supports_temperature("deepseek-v4-flash", thinking_mode="enabled"))

    def test_safe_thinking_cli_and_cache_path_capture_call_contract(self):
        args = v2.parse_args(
            [
                "--model", "deepseek-v4-flash",
                "--thinking-mode", "enabled",
                "--thinking-call-policy", "split-side",
                "--llm-max-output-tokens", "16000",
                "--run-output-token-budget", "4500000",
                "--llm-max-attempts", "2",
            ]
        )

        cache_name = v2.cache_path_for_args(args).name
        self.assertEqual(args.thinking_call_policy, "split-side")
        self.assertEqual(args.llm_max_output_tokens, 16_000)
        self.assertEqual(args.run_output_token_budget, 4_500_000)
        self.assertEqual(args.llm_max_attempts, 2)
        self.assertIn("mdl-deepseek-v4-flash", cache_name)
        self.assertIn("temp-0p6", cache_name)
        self.assertIn("think-enabled", cache_name)
        self.assertIn("call-split-side_mtok16000_a2", cache_name)

    def test_split_side_combines_into_normal_v2_score_schema(self):
        combined = v2.combine_split_side_score_results(
            {"scores": [{"ticker": "AAA", "score": 90}, {"ticker": "BBB", "score": 70}]},
            {"scores": [{"ticker": "CCC", "score": 80}, {"ticker": "DDD", "score": 60}]},
            ["AAA", "BBB"],
            ["CCC", "DDD"],
        )

        self.assertEqual([row["ticker"] for row in combined["long_scores"]], ["AAA", "BBB"])
        self.assertEqual([row["ticker"] for row in combined["short_scores"]], ["CCC", "DDD"])
        self.assertEqual(combined["llm_call_policy"], "split_by_side_thinking")
        with self.assertRaises(ValueError):
            v2.validate_split_side_scores(
                {"scores": [{"ticker": "AAA", "score": 90}]},
                ["AAA", "BBB"],
                "LONG",
            )

    def test_length_finish_reason_is_checkpointed_once_and_not_retried(self):
        class FakeClient:
            def __init__(self):
                self.calls = 0

            async def post(self, *args, **kwargs):
                self.calls += 1
                return _FakeResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {"content": "", "reasoning_content": "unfinished"},
                            }
                        ],
                        "usage": {"completion_tokens": 16_000},
                    }
                )

        client = FakeClient()
        budget = v2.OutputTokenBudget(limit=100_000)
        usage_rows = []
        with patch.object(technical_llm, "get_deepseek_api_key", return_value="test-key"):
            with self.assertRaises(v2.LLMSafeStop):
                asyncio.run(
                    v2.call_llm_score_side_thinking(
                        client=client,
                        pool=_score_pool(),
                        side="LONG",
                        model="deepseek-v4-flash",
                        prompt_variant="v2",
                        max_output_tokens=16_000,
                        max_attempts=2,
                        token_budget=budget,
                        usage_callback=usage_rows.append,
                    )
                )

        self.assertEqual(client.calls, 1)
        self.assertEqual(budget.used, 16_000)
        self.assertEqual(len(usage_rows), 1)
        self.assertEqual(usage_rows[0]["finish_reason"], "length")

    def test_cumulative_output_budget_stops_before_request(self):
        class FakeClient:
            def __init__(self):
                self.calls = 0

            async def post(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("provider request should not start")

        client = FakeClient()
        budget = v2.OutputTokenBudget(limit=20_000, used=10_000)
        with patch.object(technical_llm, "get_deepseek_api_key", return_value="test-key"):
            with self.assertRaises(v2.LLMSafeStop):
                asyncio.run(
                    v2.call_llm_score_side_thinking(
                        client=client,
                        pool=_score_pool(),
                        side="LONG",
                        model="deepseek-v4-flash",
                        prompt_variant="v2",
                        max_output_tokens=16_000,
                        max_attempts=2,
                        token_budget=budget,
                        usage_callback=lambda row: None,
                    )
                )

        self.assertEqual(client.calls, 0)
        self.assertEqual(budget.used, 10_000)

    def test_identical_length_failure_is_not_resent_after_restart(self):
        pool = _score_pool()
        storage_key = v2.llm_score_storage_key(pool, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "safe-thinking.json"
            v1.save_cache(
                {
                    "signals": {},
                    "partial_signals": {},
                    "llm_call_usage": [
                        {
                            "storage_key": storage_key,
                            "side": "LONG",
                            "attempt": 1,
                            "model": "deepseek-v4-flash",
                            "prompt_variant": "v2",
                            "prompt_hash": v2.prompt_hash_for_variant("v2"),
                            "max_output_tokens": 16_000,
                            "output_tokens_charged": 16_000,
                            "finish_reason": "length",
                        }
                    ],
                },
                cache_path,
            )
            calls = []

            class FakeAsyncClient:
                def __init__(self, *args, **kwargs):
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def post(self, *args, **kwargs):
                    calls.append(kwargs)
                    raise AssertionError("identical length-exhausted request must not be resent")

            with patch.object(v2.httpx, "AsyncClient", FakeAsyncClient):
                with self.assertRaises(v2.LLMSafeStop):
                    asyncio.run(
                        v2.collect_llm_scores_for_v2_pools(
                            {storage_key: pool},
                            cache_path=cache_path,
                            model="deepseek-v4-flash",
                            temperature=0.6,
                            rate_limit=0,
                            thinking_mode="enabled",
                            thinking_call_policy="split-side",
                            llm_max_output_tokens=16_000,
                            run_output_token_budget=100_000,
                            llm_max_attempts=2,
                        )
                    )

            self.assertEqual(calls, [])

    def test_one_date_split_side_smoke_checkpoints_long_before_short(self):
        pool = _score_pool()
        storage_key = v2.llm_score_storage_key(pool, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "safe-thinking.json"
            calls = []
            test_case = self

            class FakeAsyncClient:
                def __init__(self, *args, **kwargs):
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def post(self, url, **kwargs):
                    request_payload = kwargs["json"]
                    calls.append(request_payload)
                    user_prompt = request_payload["messages"][-1]["content"]
                    if "SHORT POOL" in user_prompt:
                        persisted = v1.load_cache(cache_path)
                        long_partial = persisted["partial_signals"][storage_key]["sides"]["long"]
                        test_case.assertEqual(len(long_partial["scores"]), 2)
                        scores = [{"ticker": "CCC", "score": 80}, {"ticker": "DDD", "score": 60}]
                        completion_tokens = 120
                    else:
                        scores = [{"ticker": "AAA", "score": 90}, {"ticker": "BBB", "score": 70}]
                        completion_tokens = 100
                    return _FakeResponse(
                        {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {
                                        "content": json.dumps({"scores": scores}),
                                        "reasoning_content": "mock reasoning",
                                    },
                                }
                            ],
                            "usage": {"completion_tokens": completion_tokens},
                        }
                    )

            with patch.object(v2.httpx, "AsyncClient", FakeAsyncClient), patch.object(
                technical_llm, "get_deepseek_api_key", return_value="test-key"
            ):
                scores = asyncio.run(
                    v2.collect_llm_scores_for_v2_pools(
                        {storage_key: pool},
                        cache_path=cache_path,
                        model="deepseek-v4-flash",
                        temperature=0.6,
                        rate_limit=0,
                        thinking_mode="enabled",
                        thinking_call_policy="split-side",
                        llm_max_output_tokens=16_000,
                        run_output_token_budget=100_000,
                        llm_max_attempts=2,
                    )
                )

            persisted = v1.load_cache(cache_path)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(persisted["llm_call_usage"]), 2)
            self.assertEqual(persisted["metadata"]["run_output_tokens_used"], 220)
            self.assertEqual(len(scores[storage_key]["long_scores"]), 2)
            self.assertEqual(len(scores[storage_key]["short_scores"]), 2)
            self.assertNotIn(storage_key, persisted["partial_signals"])

    def test_split_side_partial_cache_resumes_only_missing_side(self):
        pool = _score_pool()
        storage_key = v2.llm_score_storage_key(pool, 2)
        side_scores = {
            "long": [{"ticker": "AAA", "score": 90}, {"ticker": "BBB", "score": 70}],
            "short": [{"ticker": "CCC", "score": 80}, {"ticker": "DDD", "score": 60}],
        }

        for completed_side, missing_side in (("long", "short"), ("short", "long")):
            with self.subTest(completed_side=completed_side), tempfile.TemporaryDirectory() as tmpdir:
                cache_path = Path(tmpdir) / "safe-thinking.json"
                partial_identity = {
                    "storage_key": storage_key,
                    "model": "deepseek-v4-flash",
                    "thinking_mode": "enabled",
                    "llm_call_policy": "split_by_side_thinking",
                    "prompt_variant": "v2",
                    "prompt_hash": v2.prompt_hash_for_variant("v2"),
                    "universe_hash": pool.universe_hash,
                    "anonymize_tickers": False,
                }
                v1.save_cache(
                    {
                        "signals": {},
                        "partial_signals": {
                            storage_key: {
                                **partial_identity,
                                "sides": {
                                    completed_side: {
                                        "side": completed_side.upper(),
                                        "scores": side_scores[completed_side],
                                        "retry_count": 0,
                                        "llm_usage": {"provider_usage": {"completion_tokens": 90}},
                                    }
                                },
                            }
                        },
                        "llm_call_usage": [
                            {
                                "storage_key": storage_key,
                                "side": completed_side.upper(),
                                "attempt": 1,
                                "output_tokens_charged": 90,
                                "finish_reason": "stop",
                            }
                        ],
                    },
                    cache_path,
                )

                calls = []

                class FakeAsyncClient:
                    def __init__(self, *args, **kwargs):
                        pass

                    async def __aenter__(self):
                        return self

                    async def __aexit__(self, exc_type, exc, tb):
                        return False

                    async def post(self, url, json=None, headers=None):
                        calls.append(json)
                        return _FakeResponse(
                            {
                                "choices": [
                                    {
                                        "finish_reason": "stop",
                                        "message": {
                                            "content": __import__("json").dumps(
                                                {"scores": side_scores[missing_side]}
                                            ),
                                            "reasoning_content": "brief reasoning",
                                        },
                                    }
                                ],
                                "usage": {"completion_tokens": 80},
                            }
                        )

                with patch.object(v2.httpx, "AsyncClient", FakeAsyncClient), patch.object(
                    technical_llm, "get_deepseek_api_key", return_value="test-key"
                ):
                    scores = asyncio.run(
                        v2.collect_llm_scores_for_v2_pools(
                            {storage_key: pool},
                            cache_path=cache_path,
                            model="deepseek-v4-flash",
                            temperature=0.6,
                            rate_limit=0,
                            thinking_mode="enabled",
                            thinking_call_policy="split-side",
                            llm_max_output_tokens=16_000,
                            run_output_token_budget=100_000,
                            llm_max_attempts=2,
                        )
                    )

                persisted = v1.load_cache(cache_path)
                self.assertEqual(len(calls), 1)
                self.assertIn(storage_key, scores)
                self.assertNotIn(storage_key, persisted["partial_signals"])
                self.assertEqual(persisted["metadata"]["run_output_tokens_used"], 170)
                self.assertEqual(scores[storage_key]["llm_call_policy"], "split_by_side_thinking")

    def test_deepseek_reasoner_payload_skips_json_mode(self):
        payload = {"model": "deepseek-reasoner", "messages": [], "stream": False}
        v2._harden_json_payload(payload, "deepseek-reasoner")

        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["max_tokens"], 8192)

    def test_frozen_phase2_arguments_are_reproducible(self):
        args = v2.parse_args(
            [
                "--start", "2015-01-01", "--end", "2020-01-01",
                "--prompt-variant", "v2-frozen", "--anonymize-tickers",
                "--model", "deepseek-v4-flash", "--temperature", "0.6",
                "--pool-sizes", "60", "--book-sizes", "30",
                "--schedule", "anchors", "--anchor-offsets", "3",
            ]
        )
        self.assertEqual(args.prompt_variant, "v2-frozen")
        self.assertTrue(args.anonymize_tickers)
        self.assertEqual(args.model, "deepseek-v4-flash")
        self.assertEqual(args.temperature, 0.6)
        self.assertEqual(tuple(args.anchor_offsets), (3,))

    def test_variant_grid_replays_valid_book_sizes_only(self):
        default_variants = v2.build_variant_grid((60, 150), (30, 75), (0, 1))
        default_labels = {variant.label for variant in default_variants}
        self.assertIn("pool60_pos30_anchor0", default_labels)
        self.assertIn("pool150_pos75_anchor1", default_labels)
        self.assertNotIn("pool60_pos75_anchor0", default_labels)

        friday_variants = v2.build_variant_grid((60, 150), (30, 75), (0, 1), schedule="friday")
        friday_labels = {variant.label for variant in friday_variants}

        self.assertIn("pool60_pos30_friday", friday_labels)
        self.assertIn("pool150_pos30_friday", friday_labels)
        self.assertIn("pool150_pos75_friday", friday_labels)
        self.assertNotIn("pool60_pos75_friday", friday_labels)

        variants = v2.build_variant_grid((60, 150), (30, 75), (0, 1), schedule="anchors")
        labels = {variant.label for variant in variants}

        self.assertIn("pool60_pos30_anchor0", labels)
        self.assertIn("pool150_pos30_anchor0", labels)
        self.assertIn("pool150_pos75_anchor1", labels)
        self.assertNotIn("pool60_pos75_anchor0", labels)

    def test_llm_cache_phase_is_available(self):
        args = v2.parse_args(["--phase", "llm-cache"])

        self.assertEqual(args.phase, "llm-cache")

    def test_anonymize_tickers_flag_uses_separate_cache_path(self):
        args = v2.parse_args(["--anonymize-tickers", "--model", "deepseek-v4-flash"])
        prompt_hash = v2.prompt_hash_for_variant("v2")

        self.assertTrue(args.anonymize_tickers)
        self.assertTrue(str(v2.cache_path_for_args(args)).endswith(f"_v2-{prompt_hash}_anon.json"))

    def test_rationale_top_k_uses_separate_cache_path(self):
        args = v2.parse_args(["--anonymize-tickers", "--rationale-top-k-per-side", "30", "--model", "deepseek-v4-flash"])

        self.assertEqual(args.rationale_top_k_per_side, 30)
        self.assertTrue(str(v2.cache_path_for_args(args)).endswith("_anon_ratk30.json"))

    def test_structured_reasoning_uses_separate_cache_path(self):
        args = v2.parse_args(["--anonymize-tickers", "--structured-reasoning", "--model", "deepseek-v4-flash"])

        self.assertTrue(args.structured_reasoning)
        self.assertTrue(str(v2.cache_path_for_args(args)).endswith("_anon_sr.json"))

    def test_v22_prompt_variant_parses_and_uses_hashed_cache_path(self):
        args = v2.parse_args(["--prompt-variant", "v2.2", "--anonymize-tickers", "--model", "deepseek-v4-flash"])
        prompt_hash = v2.prompt_hash_for_variant("v2.2")

        self.assertEqual(args.prompt_variant, "v2.2")
        self.assertEqual(tuple(args.v22_lambdas), v2.DEFAULT_V22_LAMBDAS)
        self.assertIn("context_adjustment", v2.prompt_text_for_variant("v2.2"))
        self.assertIn(prompt_hash, v2.prompt_version_for_variant("v2.2"))
        self.assertTrue(str(v2.cache_path_for_args(args)).endswith(f"_v22-{prompt_hash}_anon.json"))

    def test_v23_prompt_variant_parses_and_uses_hashed_cache_path(self):
        args = v2.parse_args(["--prompt-variant", "v2.3", "--anonymize-tickers", "--model", "deepseek-v4-flash"])
        prompt_hash = v2.prompt_hash_for_variant("v2.3")

        self.assertEqual(args.prompt_variant, "v2.3")
        self.assertIn("long_ranking", v2.prompt_text_for_variant("v2.3"))
        self.assertEqual(v2.output_contract_for_variant("v2.3"), "rank_only_complete_ordering")
        self.assertIn(prompt_hash, v2.prompt_version_for_variant("v2.3"))
        self.assertTrue(str(v2.cache_path_for_args(args)).endswith(f"_v23-{prompt_hash}_anon.json"))

    def test_v24_uses_frozen_prompt_with_raw_market_context_in_user_input(self):
        args = v2.parse_args(["--prompt-variant", "v2.4", "--anonymize-tickers", "--model", "deepseek-v4-flash"])
        prompt_hash = v2.prompt_hash_for_variant("v2.4")
        system_prompt = v2.prompt_text_for_variant("v2.4")
        user_prompt = v2.format_v24_user_prompt(_score_pool())

        self.assertEqual(args.prompt_variant, "v2.4")
        self.assertEqual(system_prompt, v2.V2_FROZEN_SYSTEM_PROMPT)
        self.assertNotIn("FEATURE GLOSSARY", system_prompt)
        self.assertEqual(v2.output_contract_for_variant("v2.4"), "score_0_100")
        self.assertIn("SPY vs 200-day MA: below", user_prompt)
        self.assertIn("SPY distance from 200-day MA: -19.7%", user_prompt)
        self.assertIn("VIX level: 38.6", user_prompt)
        self.assertIn("VIX trailing-252-day percentile: 73.1", user_prompt)
        self.assertIn("Score every ticker in both pools", user_prompt)
        self.assertNotIn("MARKET REGIME", user_prompt)
        self.assertNotIn("normal", user_prompt)
        self.assertTrue(str(v2.cache_path_for_args(args)).endswith(f"_v24-{prompt_hash}_anon.json"))

    def test_v25_adds_only_balanced_memory_fields_and_uses_isolated_cache(self):
        args = v2.parse_args(
            [
                "--prompt-variant", "v2.5",
                "--anonymize-tickers",
                "--model", "deepseek-v4-flash",
                "--anchor-offsets", "3",
            ]
        )
        pool = _score_pool()
        for row in pool.feature_rows.values():
            row.update({"HistN": 20, "HistWin": 60.0, "HistRet": 1.25})
        prompt = v2.prompt_text_for_variant("v2.5")
        user_prompt = v2.format_v25_user_prompt(pool)

        self.assertTrue(prompt.startswith(v2.V2_SYSTEM_PROMPT))
        self.assertIn("HistN", prompt)
        self.assertIn("HistWin", user_prompt)
        self.assertIn("HistRet", user_prompt)
        self.assertNotIn("HistScore", prompt)
        self.assertEqual(v2.output_contract_for_variant("v2.5"), "score_0_100")
        self.assertIn(
            f"_v25-{v2.prompt_hash_for_variant('v2.5')}_mem-{v2.v25m.memory_spec_hash()}_anon.json",
            str(v2.cache_path_for_args(args)),
        )

    def test_v25_llm_call_preserves_v2_score_contract_with_memory_fields(self):
        captured = {}
        pool = _score_pool()
        for row in pool.feature_rows.values():
            row.update({"HistN": 20, "HistWin": 60.0, "HistRet": 1.25})

        class FakeClient:
            async def post(self, url, **kwargs):
                captured.update(kwargs["json"])
                return _FakeResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "long_scores": [
                                                {"ticker": "AAA", "score": 90},
                                                {"ticker": "BBB", "score": 70},
                                            ],
                                            "short_scores": [
                                                {"ticker": "CCC", "score": 80},
                                                {"ticker": "DDD", "score": 60},
                                            ],
                                        }
                                    )
                                },
                            }
                        ],
                        "usage": {"completion_tokens": 20},
                    }
                )

        with patch.object(technical_llm, "get_deepseek_api_key", return_value="test-key"), patch.object(
            technical_llm,
            "get_deepseek_url",
            return_value="https://example.test/chat",
        ):
            result = asyncio.run(
                v2.call_llm_scores_for_prompt_variant(
                    client=FakeClient(),
                    pool=pool,
                    model="deepseek-v4-flash",
                    temperature=0.6,
                    prompt_variant="v2.5",
                )
            )

        system_prompt = captured["messages"][0]["content"]
        user_prompt = captured["messages"][1]["content"]
        self.assertTrue(system_prompt.startswith(v2.V2_SYSTEM_PROMPT))
        self.assertIn("HistN", user_prompt)
        self.assertIn("HistWin", user_prompt)
        self.assertIn("HistRet", user_prompt)
        self.assertNotIn("HistScore", system_prompt + user_prompt)
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertEqual(result["long_scores"][0], {"ticker": "AAA", "score": 90})
        self.assertEqual(result["short_scores"][0], {"ticker": "CCC", "score": 80})

    def test_v25_anonymization_preserves_memory_but_hides_real_tickers(self):
        pool = _score_pool()
        for row in pool.feature_rows.values():
            row.update({"HistN": 20, "HistWin": 60.0, "HistRet": 1.25})

        anonymized, mapping = v2.anonymize_pool_for_llm(pool)
        user_prompt = v2.format_v25_user_prompt(anonymized)

        self.assertEqual(set(mapping.values()), set(pool.pool_tickers))
        self.assertNotIn("AAA", user_prompt)
        self.assertIn("SEC_0001", user_prompt)
        self.assertIn("20", user_prompt)
        self.assertIn("60", user_prompt)
        self.assertIn("1.25", user_prompt)

    def test_v25_main_rejects_unfrozen_harness(self):
        with self.assertRaises(ValueError):
            v2.main(["--prompt-variant", "v2.5", "--anchor-offsets", "3"])
        with self.assertRaises(ValueError):
            v2.main(
                [
                    "--prompt-variant", "v2.5",
                    "--anchor-offsets", "3",
                    "--anonymize-tickers",
                    "--temperature", "1.0",
                    "--phase", "llm-cache",
                ]
            )

    def test_raw_market_context_is_point_in_time_and_continuous(self):
        dates = pd.bdate_range("2019-01-01", periods=201)
        spy = pd.DataFrame(
            {
                "date": dates,
                "adj_close": [100.0] * 200 + [80.0],
                "vix_level": [20.0] * 200 + [38.6],
                "vix_percentile_252d": [50.0] * 200 + [73.1],
            }
        )

        context = v2.raw_market_context_from_spy(spy, dates[-1])

        self.assertEqual(context["spy_vs_200d_ma"], "below")
        self.assertAlmostEqual(context["spy_distance_from_200d_ma_pct"], -19.9199, places=3)
        self.assertEqual(context["vix_level"], 38.6)
        self.assertEqual(context["vix_percentile_252d"], 73.1)

    def test_v24_llm_call_uses_raw_context_with_frozen_score_contract(self):
        captured = {}

        class FakeClient:
            async def post(self, url, **kwargs):
                captured.update(kwargs["json"])
                return _FakeResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "long_scores": [
                                                {"ticker": "AAA", "score": 90},
                                                {"ticker": "BBB", "score": 70},
                                            ],
                                            "short_scores": [
                                                {"ticker": "CCC", "score": 80},
                                                {"ticker": "DDD", "score": 60},
                                            ],
                                        }
                                    )
                                },
                            }
                        ],
                        "usage": {"completion_tokens": 20},
                    }
                )

        with patch.object(technical_llm, "get_deepseek_api_key", return_value="test-key"), patch.object(
            technical_llm,
            "get_deepseek_url",
            return_value="https://example.test/chat",
        ):
            result = asyncio.run(
                v2.call_llm_scores_for_prompt_variant(
                    client=FakeClient(),
                    pool=_score_pool(),
                    model="deepseek-v4-flash",
                    temperature=0.6,
                    prompt_variant="v2.4",
                )
            )

        user_prompt = captured["messages"][1]["content"]
        self.assertIn("SPY distance from 200-day MA: -19.7%", user_prompt)
        self.assertNotIn("MARKET REGIME", user_prompt)
        self.assertEqual(result["long_scores"][0], {"ticker": "AAA", "score": 90})
        self.assertEqual(result["short_scores"][0], {"ticker": "CCC", "score": 80})

    def test_v21_prompt_variant_is_not_available(self):
        with self.assertRaises(SystemExit):
            v2.parse_args(["--prompt-variant", "v2.1"])

    def test_v2_prompt_hash_changes_when_prompt_text_changes(self):
        original = v2.V2_SYSTEM_PROMPT
        try:
            first_hash = v2.prompt_hash_for_variant("v2")
            first_path = str(v2.cache_path_for_args(v2.parse_args([])))
            v2.V2_SYSTEM_PROMPT = original + "\nHASH CHANGE TEST"
            second_hash = v2.prompt_hash_for_variant("v2")
            second_path = str(v2.cache_path_for_args(v2.parse_args([])))

            self.assertNotEqual(first_hash, second_hash)
            self.assertNotEqual(first_path, second_path)
        finally:
            v2.V2_SYSTEM_PROMPT = original

    def test_v2_prompt_version_does_not_cross_load_old_v1_cache_key(self):
        pool = v1.CandidatePool(
            signal_date=pd.Timestamp("2020-01-10"),
            lookback_date=pd.Timestamp("2020-01-03"),
            exit_date=pd.Timestamp("2020-01-17"),
            entry_open_date=pd.Timestamp("2020-01-13"),
            exit_open_date=pd.Timestamp("2020-01-21"),
            valid_tickers=["AAA", "BBB"],
            long_tickers=["AAA"],
            short_tickers=["BBB"],
            trail_by_ticker={"AAA": -0.1, "BBB": 0.1},
            signal_by_ticker={"AAA": 0.1, "BBB": -0.1},
            feature_rows={
                ticker: {
                    "Ticker": ticker,
                    "MRL": 100.0,
                    "MRS": 0.0,
                    "Trail5": -10.0,
                    "MRSig": 10.0,
                    "RSI5": 20.0,
                    "Rng55": 0.2,
                    "Vel/V": -0.5,
                    "%MA100": -5.0,
                    "52WH": -20.0,
                    "Mom3m": 1.0,
                    "AutoC": -0.1,
                    "ADX": 30.0,
                    "VolPctl": 80.0,
                }
                for ticker in ["AAA", "BBB"]
            },
            universe_hash="cache-hash",
            regime="normal",
        )
        entry = {
            "cache_key": v1._cache_key(pool.signal_date, v1.PROMPT_VERSION, "deepseek-chat", pool.universe_hash),
            "long_scores": [{"ticker": "AAA", "score": 70}],
            "short_scores": [{"ticker": "BBB", "score": 80}],
        }

        self.assertIsNone(
            v1._entry_from_cache(
                entry,
                pool,
                "deepseek-chat",
                prompt_version=v2.prompt_version_for_variant("v2"),
                prompt_style=v2.V2_PROMPT_STYLE,
            )
        )

    def test_structured_reasoning_cannot_combine_with_selected_rationale(self):
        with self.assertRaises(ValueError):
            v2.main(["--structured-reasoning", "--rationale-top-k-per-side", "30"])

    def test_v22_cannot_combine_with_reasoning_modes(self):
        with self.assertRaises(ValueError):
            v2.main(["--prompt-variant", "v2.2", "--structured-reasoning"])
        with self.assertRaises(ValueError):
            v2.main(["--prompt-variant", "v2.2", "--rationale-top-k-per-side", "30"])

    def test_rank_only_variants_cannot_combine_with_reasoning_modes(self):
        for prompt_variant in ("v2.3", "v2.4"):
            with self.subTest(prompt_variant=prompt_variant), self.assertRaises(ValueError):
                v2.main(["--prompt-variant", prompt_variant, "--structured-reasoning"])
            with self.subTest(prompt_variant=prompt_variant), self.assertRaises(ValueError):
                v2.main(["--prompt-variant", prompt_variant, "--rationale-top-k-per-side", "30"])

    def test_v23_ranking_validation_and_signed_scores(self):
        validated = v2.validate_llm_rankings(
            {
                "long_ranking": ["BBB", "AAA"],
                "short_ranking": [{"ticker": "DDD"}, {"ticker": "CCC"}],
            },
            expected_longs=["AAA", "BBB"],
            expected_shorts=["CCC", "DDD"],
        )
        signed = v2.signed_scores_from_llm_rankings(validated)

        self.assertEqual(validated["long_ranking"], ["BBB", "AAA"])
        self.assertEqual(validated["short_ranking"], ["DDD", "CCC"])
        self.assertEqual(validated["long_scores"][0]["rank"], 1)
        self.assertEqual(validated["long_scores"][0]["score"], 100.0)
        self.assertEqual(validated["long_scores"][1]["score"], 0.0)
        self.assertEqual(signed["BBB"], 2.0)
        self.assertEqual(signed["AAA"], 1.0)
        self.assertEqual(signed["DDD"], -2.0)
        self.assertEqual(signed["CCC"], -1.0)

    def test_v23_ranking_validation_rejects_missing_duplicate_or_wrong_side(self):
        with self.assertRaises(ValueError):
            v2.validate_llm_rankings(
                {"long_ranking": ["AAA", "AAA"], "short_ranking": ["CCC", "DDD"]},
                expected_longs=["AAA", "BBB"],
                expected_shorts=["CCC", "DDD"],
            )
        with self.assertRaises(ValueError):
            v2.validate_llm_rankings(
                {"long_ranking": ["AAA", "CCC"], "short_ranking": ["BBB", "DDD"]},
                expected_longs=["AAA", "BBB"],
                expected_shorts=["CCC", "DDD"],
            )

    def test_v22_adjustment_validation_and_lambda_math(self):
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        feature_rows = {
            ticker: {
                "Ticker": ticker,
                "MRL": 100.0,
                "MRS": 0.0,
                "Trail5": -10.0,
                "MRSig": 10.0,
                "RSI5": 20.0,
                "Rng55": 0.2,
                "Vel/V": -0.5,
                "%MA100": -5.0,
                "52WH": -20.0,
                "Mom3m": 1.0,
                "AutoC": -0.1,
                "ADX": 30.0,
                "VolPctl": 80.0,
            }
            for ticker in tickers
        }
        pool = v1.CandidatePool(
            signal_date=pd.Timestamp("2020-01-10"),
            lookback_date=pd.Timestamp("2020-01-03"),
            exit_date=pd.Timestamp("2020-01-17"),
            entry_open_date=pd.Timestamp("2020-01-13"),
            exit_open_date=pd.Timestamp("2020-01-21"),
            valid_tickers=tickers,
            long_tickers=["AAA", "BBB"],
            short_tickers=["CCC", "DDD"],
            trail_by_ticker={"AAA": -0.2, "BBB": -0.1, "CCC": 0.2, "DDD": 0.1},
            signal_by_ticker={"AAA": 0.2, "BBB": 0.1, "CCC": -0.2, "DDD": -0.1},
            feature_rows=feature_rows,
            universe_hash="cache-hash",
            regime="normal",
        )
        validated = v2.validate_llm_adjustments(
            {
                "long_adjustments": [
                    {"ticker": "AAA", "context_adjustment": -3},
                    {"ticker": "BBB", "context_adjustment": 2},
                ],
                "short_adjustments": [
                    {"ticker": "CCC", "context_adjustment": 1},
                    {"ticker": "DDD", "context_adjustment": -2},
                ],
            },
            pool.long_tickers,
            pool.short_tickers,
        )

        signed, base, adjustments, final = v2.v22_final_scores_for_pool(pool, validated, adjustment_lambda=4.0)

        self.assertEqual(adjustments["AAA"], -3)
        self.assertEqual(base["AAA"], 100.0)
        self.assertEqual(base["BBB"], 0.0)
        self.assertEqual(final["AAA"], 88.0)
        self.assertEqual(final["BBB"], 8.0)
        self.assertEqual(signed["CCC"], -100.0)
        self.assertEqual(final["DDD"], 0.0)

    def test_v22_ties_break_by_base_score(self):
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        signal, long_mask, short_mask = v2._book_masks_from_scores_with_base_tiebreak(
            pool_tickers=tickers,
            signed_scores={"AAA": 50.0, "BBB": 50.0, "CCC": -50.0, "DDD": -50.0},
            base_scores={"AAA": 40.0, "BBB": 80.0, "CCC": 40.0, "DDD": 80.0},
            long_tickers={"AAA", "BBB"},
            short_tickers={"CCC", "DDD"},
            positions_per_side=1,
        )

        self.assertEqual(signal.tolist(), [50.0, 50.0, -50.0, -50.0])
        self.assertEqual([tickers[idx] for idx in list(long_mask.nonzero()[0])], ["BBB"])
        self.assertEqual([tickers[idx] for idx in list(short_mask.nonzero()[0])], ["DDD"])

    def test_validate_attributions_filters_to_top_score_tickers_without_failing(self):
        raw = {
            "long_attributions": [
                {"ticker": "AAA", "primary_features": ["Trail5", "RSI5"], "rationale_label": "extreme oversold"},
                {"ticker": "ZZZ", "primary_features": ["Rng55"], "rationale_label": "not selected"},
            ],
            "short_attributions": [
                {"ticker": "CCC", "primary_features": ["MRSig"], "rationale_label": "sharp winner"},
                {"ticker": "DDD", "primary_features": ["ADX", "VolPctl"], "rationale_label": "strong trend high vol"},
            ],
        }
        validated = {
            "long_scores": [
                {"ticker": "AAA", "score": 90},
                {"ticker": "BBB", "score": 80},
                {"ticker": "EEE", "score": 70},
            ],
            "short_scores": [
                {"ticker": "CCC", "score": 95},
                {"ticker": "DDD", "score": 85},
                {"ticker": "FFF", "score": 75},
            ],
        }

        attributions = v2.validate_attributions(raw, validated, top_k_per_side=2)

        self.assertEqual(attributions["long_attributions"][0]["ticker"], "AAA")
        self.assertEqual(attributions["short_attributions"][1]["primary_features"], ["ADX", "VolPctl"])
        self.assertEqual(attributions["attribution_status"], "partial")
        self.assertEqual(attributions["attribution_extra_count"], 1)
        self.assertEqual(attributions["attribution_missing_count"], 1)

    def test_clean_structured_reasoning_metadata_enriches_all_scored_candidates(self):
        raw = {
            "long_scores": [
                {
                    "ticker": "AAA",
                    "score": 90,
                    "pf": [["Trail5", "supports"], ["RSI5", "supports"]],
                    "label": "oversold reversal",
                },
                {
                    "ticker": "BBB",
                    "score": 80,
                    "pf": [["BadFeature", "supports"]],
                    "label": "made up label",
                },
            ],
            "short_scores": [
                {
                    "ticker": "CCC",
                    "score": 85,
                    "pf": [["MRSig", "supports"]],
                    "label": "overbought reversal",
                }
            ],
        }
        validated = {
            "long_scores": [{"ticker": "AAA", "score": 90}, {"ticker": "BBB", "score": 80}],
            "short_scores": [{"ticker": "CCC", "score": 85}],
        }

        enriched = v2.clean_structured_reasoning_metadata(raw, validated)

        self.assertEqual(enriched["long_scores"][0]["primary_features"][0]["feature"], "Trail5")
        self.assertEqual(enriched["long_scores"][0]["setup_label"], "oversold reversal")
        self.assertEqual(enriched["long_scores"][1]["reasoning_status"], "partial")
        self.assertEqual(enriched["long_scores"][1]["setup_label"], "mixed signals")
        self.assertEqual(enriched["structured_reasoning_status"], "partial")

    def test_structured_side_prompt_is_one_side_only_and_compact_contract(self):
        pool = v1.CandidatePool(
            signal_date=pd.Timestamp("2020-01-10"),
            lookback_date=pd.Timestamp("2020-01-03"),
            exit_date=pd.Timestamp("2020-01-17"),
            entry_open_date=pd.Timestamp("2020-01-13"),
            exit_open_date=pd.Timestamp("2020-01-21"),
            valid_tickers=["AAA", "BBB", "CCC", "DDD"],
            long_tickers=["AAA", "BBB"],
            short_tickers=["CCC", "DDD"],
            trail_by_ticker={"AAA": -0.1, "BBB": -0.08, "CCC": 0.09, "DDD": 0.07},
            signal_by_ticker={"AAA": 0.1, "BBB": 0.08, "CCC": -0.09, "DDD": -0.07},
            feature_rows={
                ticker: {
                    "Ticker": ticker,
                    "MRL": 100.0,
                    "MRS": 0.0,
                    "Trail5": -10.0,
                    "MRSig": 10.0,
                    "RSI5": 20.0,
                    "Rng55": 0.2,
                    "Vel/V": -0.5,
                    "%MA100": -5.0,
                    "52WH": -20.0,
                    "Mom3m": 1.0,
                    "AutoC": -0.1,
                    "ADX": 30.0,
                    "VolPctl": 80.0,
                }
                for ticker in ["AAA", "BBB", "CCC", "DDD"]
            },
            universe_hash="hash",
            regime="normal",
        )

        prompt = v2.format_structured_side_user_prompt(pool, "LONG")
        system = v2.structured_reasoning_system_prompt()

        self.assertIn("LONG POOL", prompt)
        self.assertNotIn("SHORT POOL", prompt)
        self.assertIn('"scores"', system)
        self.assertIn('"pf"', system)
        self.assertNotIn('"long_scores"', system)

    def test_attribution_map_reads_structured_reasoning_from_score_rows(self):
        entry = {
            "long_scores": [
                {
                    "ticker": "AAA",
                    "score": 90,
                    "primary_features": [
                        {"feature": "Trail5", "effect": "supports"},
                        {"feature": "RSI5", "effect": "supports"},
                    ],
                    "setup_label": "oversold reversal",
                    "reasoning_status": "complete",
                    "anonymous_ticker": "SEC_0001",
                }
            ],
            "short_scores": [],
        }

        mapping = v2.attribution_map_from_entry(entry)

        self.assertEqual(mapping["AAA"]["attribution_features"], "Trail5|RSI5")
        self.assertEqual(mapping["AAA"]["attribution_effects"], "Trail5:supports|RSI5:supports")
        self.assertEqual(mapping["AAA"]["setup_label"], "oversold reversal")
        self.assertEqual(mapping["AAA"]["anonymous_ticker"], "SEC_0001")

    def test_anonymized_pool_hides_real_tickers_and_maps_scores_back(self):
        pool = v1.CandidatePool(
            signal_date=pd.Timestamp("2020-01-10"),
            lookback_date=pd.Timestamp("2020-01-03"),
            exit_date=pd.Timestamp("2020-01-17"),
            entry_open_date=pd.Timestamp("2020-01-13"),
            exit_open_date=pd.Timestamp("2020-01-21"),
            valid_tickers=["AAA", "BBB", "CCC", "DDD"],
            long_tickers=["AAA", "BBB"],
            short_tickers=["CCC", "DDD"],
            trail_by_ticker={"AAA": -0.1, "BBB": -0.08, "CCC": 0.09, "DDD": 0.07},
            signal_by_ticker={"AAA": 0.1, "BBB": 0.08, "CCC": -0.09, "DDD": -0.07},
            feature_rows={
                ticker: {
                    "Ticker": ticker,
                    "MRL": 100.0,
                    "MRS": 0.0,
                    "Trail5": -10.0,
                    "MRSig": 10.0,
                    "RSI5": 20.0,
                    "Rng55": 0.2,
                    "Vel/V": -0.5,
                    "%MA100": -5.0,
                    "52WH": -20.0,
                    "Mom3m": 1.0,
                    "AutoC": -0.1,
                    "ADX": 30.0,
                    "VolPctl": 80.0,
                }
                for ticker in ["AAA", "BBB", "CCC", "DDD"]
            },
            universe_hash="real-hash",
            regime="normal",
        )

        anonymized, anon_to_real = v2.anonymize_pool_for_llm(pool)
        prompt = v1.format_user_prompt(anonymized)

        self.assertNotIn("AAA", prompt)
        self.assertNotIn("BBB", prompt)
        self.assertIn("SEC_0001", prompt)
        self.assertIn("SEC_0004", prompt)
        self.assertEqual(anon_to_real["SEC_0001"], "AAA")
        self.assertEqual(anonymized.long_tickers, ["SEC_0001", "SEC_0002"])
        self.assertEqual(anonymized.short_tickers, ["SEC_0003", "SEC_0004"])
        self.assertNotEqual(anonymized.universe_hash, pool.universe_hash)

        remapped = v2.deanonymize_llm_scores(
            {
                "long_scores": [{"ticker": "SEC_0001", "score": 90}, {"ticker": "SEC_0002", "score": 70}],
                "short_scores": [{"ticker": "SEC_0003", "score": 80}, {"ticker": "SEC_0004", "score": 60}],
                "long_attributions": [
                    {
                        "ticker": "SEC_0001",
                        "primary_features": ["Trail5"],
                        "rationale_label": "extreme oversold",
                    }
                ],
                "short_attributions": [
                    {
                        "ticker": "SEC_0003",
                        "primary_features": ["MRSig"],
                        "rationale_label": "sharp winner",
                    }
                ],
            },
            anon_to_real,
        )

        self.assertEqual([row["ticker"] for row in remapped["long_scores"]], ["AAA", "BBB"])
        self.assertEqual(remapped["long_scores"][0]["anonymous_ticker"], "SEC_0001")
        self.assertEqual(remapped["long_attributions"][0]["ticker"], "AAA")
        self.assertEqual(remapped["long_attributions"][0]["anonymous_ticker"], "SEC_0001")

    def test_attribution_outcome_summary_groups_by_feature_and_side(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            selections = pd.DataFrame(
                [
                    {
                        "position_type": "LONG",
                        "attribution_features": "Trail5|RSI5",
                        "realized_c2c_return_pct": 4.0,
                        "realized_o2o_return_pct": 3.0,
                    },
                    {
                        "position_type": "LONG",
                        "attribution_features": "Trail5",
                        "realized_c2c_return_pct": -2.0,
                        "realized_o2o_return_pct": -1.0,
                    },
                    {
                        "position_type": "SHORT",
                        "attribution_features": "ADX",
                        "realized_c2c_return_pct": 5.0,
                        "realized_o2o_return_pct": 6.0,
                    },
                ]
            )

            summary = v2.save_attribution_outcome_summary(selections, Path(tmpdir))

            trail5 = summary[(summary["feature"] == "Trail5") & (summary["side"] == "LONG")].iloc[0]
            self.assertEqual(int(trail5["n"]), 2)
            self.assertEqual(float(trail5["avg_c2c_return_pct"]), 1.0)
            self.assertTrue((Path(tmpdir) / "attribution_outcome_summary.csv").exists())

    def test_v2_gross_comparison_prefers_anchor_average_zero_cost_c2c(self):
        scorecard = pd.DataFrame(
            [
                {
                    "pool_per_side": 60,
                    "positions_per_side": 30,
                    "anchor_offset": 0,
                    "is_anchor_average": False,
                    "scorer": "trail5",
                    "run": "Baseline-Pool",
                    "execution_model": "C2C",
                    "cost_bps": 0.0,
                    "sharpe": 1.0,
                    "cagr_pct": 10.0,
                    "total_return_pct": 50.0,
                    "max_drawdown_pct": -8.0,
                    "hit_rate_pct": 55.0,
                    "ic_mean": 0.02,
                    "n_periods": 10,
                },
                {
                    "pool_per_side": 60,
                    "positions_per_side": 30,
                    "anchor_offset": "average",
                    "is_anchor_average": True,
                    "scorer": "trail5",
                    "run": "Baseline-Pool",
                    "execution_model": "C2C",
                    "cost_bps": 0.0,
                    "sharpe": 1.2,
                    "cagr_pct": 12.0,
                    "total_return_pct": 60.0,
                    "max_drawdown_pct": -7.0,
                    "hit_rate_pct": 56.0,
                    "ic_mean": 0.03,
                    "n_periods": 10,
                },
                {
                    "pool_per_side": 60,
                    "positions_per_side": 30,
                    "anchor_offset": "average",
                    "is_anchor_average": True,
                    "scorer": "trail5",
                    "run": "Baseline-Pool",
                    "execution_model": "C2C",
                    "cost_bps": 5.0,
                    "sharpe": 0.4,
                    "cagr_pct": 4.0,
                    "total_return_pct": 15.0,
                    "max_drawdown_pct": -9.0,
                    "hit_rate_pct": 50.0,
                    "ic_mean": 0.03,
                    "n_periods": 10,
                },
            ]
        )

        comparison = v2.v2_gross_comparison_rows(scorecard)

        self.assertEqual(len(comparison), 2)
        self.assertEqual(set(comparison["family"]), {"V2"})
        self.assertEqual(set(comparison["anchor_offset"].astype(str)), {"average", "0"})
        self.assertEqual(set(comparison["cost_bps"]), {0.0})
        self.assertIn("Baseline-Pool Anchor Avg", set(comparison["run"]))
        self.assertIn("Baseline-Pool Best Offset", set(comparison["run"]))

        avg_row = comparison[comparison["anchor_offset"].astype(str) == "average"].iloc[0]
        best_row = comparison[comparison["anchor_offset"].astype(str) == "0"].iloc[0]
        self.assertEqual(float(avg_row["sharpe"]), 1.2)
        self.assertEqual(float(best_row["sharpe"]), 1.0)
        self.assertEqual(avg_row["harness"], "v2 multi-anchor replay harness")
        self.assertEqual(avg_row["prompt_schema"], "none: deterministic Trail5 baseline")
        self.assertEqual(avg_row["candidate_pool"], "60 biggest 5-day losers + 60 biggest 5-day winners")
        self.assertIn("ADV20", avg_row["liquidity_filters"])

    def test_v2_gross_comparison_labels_friday_schedule(self):
        scorecard = pd.DataFrame(
            [
                {
                    "pool_per_side": 150,
                    "positions_per_side": 75,
                    "anchor_offset": "Friday",
                    "is_anchor_average": False,
                    "score_pool_per_side": 150,
                    "scorer": "llm_score",
                    "run": "LLM-Pool v2",
                    "execution_model": "C2C",
                    "cost_bps": 0.0,
                    "sharpe": 1.1,
                    "cagr_pct": 17.0,
                    "total_return_pct": 100.0,
                    "max_drawdown_pct": -12.0,
                    "hit_rate_pct": 55.0,
                    "ic_mean": 0.03,
                    "n_periods": 251,
                }
            ]
        )

        comparison = v2.v2_gross_comparison_rows(scorecard)

        self.assertEqual(comparison.iloc[0]["basis"], "Friday gross C2C")
        self.assertEqual(comparison.iloc[0]["harness"], "v2 Friday replay harness")
        self.assertEqual(comparison.iloc[0]["cadence"], "Friday-only weekly schedule")
        self.assertEqual(comparison.iloc[0]["accounting"], "gross Friday-to-Friday C2C")
        self.assertEqual(comparison.iloc[0]["prompt_schema"], "V2 0-100 score prompt")
        self.assertEqual(comparison.iloc[0]["score_source"], "LLM scored 150/150; row pool 150/150")
        self.assertIn("pool 150", comparison.iloc[0]["replay_cache"])

    def test_v23_gross_comparison_labels_rank_only_schema(self):
        scorecard = pd.DataFrame(
            [
                {
                    "pool_per_side": 60,
                    "positions_per_side": 30,
                    "anchor_offset": 3,
                    "is_anchor_average": False,
                    "score_pool_per_side": 60,
                    "scorer": "llm_rank_order",
                    "run": "LLM-Pool v2.3",
                    "execution_model": "C2C",
                    "cost_bps": 0.0,
                    "sharpe": 1.1,
                    "cagr_pct": 17.0,
                    "total_return_pct": 100.0,
                    "max_drawdown_pct": -12.0,
                    "hit_rate_pct": 55.0,
                    "ic_mean": 0.03,
                    "n_periods": 251,
                }
            ]
        )

        comparison = v2.v2_gross_comparison_rows(scorecard)

        self.assertEqual(comparison.iloc[0]["prompt_schema"], "V2.3 rank-only ordering prompt")
        self.assertEqual(comparison.iloc[0]["score_source"], "LLM ranked 60/60; row pool 60/60")

    def test_v24_gross_comparison_labels_frozen_raw_market_context_schema(self):
        scorecard = pd.DataFrame(
            [
                {
                    "pool_per_side": 60,
                    "positions_per_side": 30,
                    "anchor_offset": 3,
                    "is_anchor_average": False,
                    "score_pool_per_side": 60,
                    "scorer": "llm_score",
                    "run": "LLM-Pool v2.4",
                    "execution_model": "C2C",
                    "cost_bps": 0.0,
                    "sharpe": 1.1,
                    "cagr_pct": 17.0,
                    "total_return_pct": 100.0,
                    "max_drawdown_pct": -12.0,
                    "hit_rate_pct": 55.0,
                    "ic_mean": 0.03,
                    "n_periods": 251,
                }
            ]
        )

        comparison = v2.v2_gross_comparison_rows(scorecard)

        self.assertEqual(
            comparison.iloc[0]["prompt_schema"],
            "V2.4 frozen 0-100 score prompt with raw SPY/VIX context",
        )
        self.assertEqual(comparison.iloc[0]["score_source"], "LLM scored 60/60; row pool 60/60")

    def test_v2_can_slice_broad_llm_scores_for_smaller_pool(self):
        tickers = [f"T{i:02d}" for i in range(16)]
        histories = _make_histories(tickers, start="2019-01-01", end="2019-06-30")
        market = rb.build_market_frames(histories)
        features = v1.build_feature_frames(histories)
        membership = _membership(tickers)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _v2_config("2019-03-01", "2019-06-01", tmpdir)
            context = v2.build_v2_pool_context(market)
            signal_idx = int(context.trading_dates.get_loc(pd.Timestamp("2019-03-15")))
            pool2 = v2.build_candidate_pool_for_index(
                market,
                features,
                membership,
                spy_df=None,
                signal_idx=signal_idx,
                pool_per_side=2,
                config=config,
                context=context,
            )
            pool4 = v2.build_candidate_pool_for_index(
                market,
                features,
                membership,
                spy_df=None,
                signal_idx=signal_idx,
                pool_per_side=4,
                config=config,
                context=context,
            )

        assert pool2 is not None
        assert pool4 is not None
        broad_entry = {
            "long_scores": [{"ticker": ticker, "score": 80} for ticker in pool4.long_tickers],
            "short_scores": [{"ticker": ticker, "score": 70} for ticker in pool4.short_tickers],
        }
        validated = v2.validate_llm_scores_for_v2_pool(
            {v2.llm_score_storage_key(pool2, 4): broad_entry},
            pool2,
            pool_per_side=2,
            llm_source_pool_per_side=4,
        )

        self.assertEqual({row["ticker"] for row in validated["long_scores"]}, {ticker.upper() for ticker in pool2.long_tickers})
        self.assertEqual({row["ticker"] for row in validated["short_scores"]}, {ticker.upper() for ticker in pool2.short_tickers})

    def test_v23_can_slice_broad_llm_ranking_for_smaller_pool(self):
        tickers = [f"T{i:02d}" for i in range(16)]
        histories = _make_histories(tickers, start="2019-01-01", end="2019-06-30")
        market = rb.build_market_frames(histories)
        features = v1.build_feature_frames(histories)
        membership = _membership(tickers)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _v2_config("2019-03-01", "2019-06-01", tmpdir)
            context = v2.build_v2_pool_context(market)
            signal_idx = int(context.trading_dates.get_loc(pd.Timestamp("2019-03-15")))
            pool2 = v2.build_candidate_pool_for_index(
                market,
                features,
                membership,
                spy_df=None,
                signal_idx=signal_idx,
                pool_per_side=2,
                config=config,
                context=context,
            )
            pool4 = v2.build_candidate_pool_for_index(
                market,
                features,
                membership,
                spy_df=None,
                signal_idx=signal_idx,
                pool_per_side=4,
                config=config,
                context=context,
            )

        assert pool2 is not None
        assert pool4 is not None
        broad_entry = v2.validate_llm_rankings(
            {
                "long_ranking": list(reversed(pool4.long_tickers)),
                "short_ranking": list(reversed(pool4.short_tickers)),
            },
            expected_longs=pool4.long_tickers,
            expected_shorts=pool4.short_tickers,
        )
        validated = v2.validate_llm_rankings_for_v2_pool(
            {v2.llm_score_storage_key(pool2, 4): broad_entry},
            pool2,
            pool_per_side=2,
            llm_source_pool_per_side=4,
        )

        self.assertEqual(set(validated["long_ranking"]), {ticker.upper() for ticker in pool2.long_tickers})
        self.assertEqual(set(validated["short_ranking"]), {ticker.upper() for ticker in pool2.short_tickers})
        self.assertEqual(len(validated["long_ranking"]), 2)
        self.assertEqual(len(validated["short_ranking"]), 2)

    def test_legacy_gross_comparison_prefers_anchor_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame(
                [
                    {
                        "run": "Anchor-Monday",
                        "anchor": "Monday",
                        "execution_model": "C2C",
                        "cost_bps": 0.0,
                        "n_periods": 10,
                        "sharpe": 0.5,
                        "cagr_pct": 5.0,
                        "total_return_pct": 8.0,
                        "max_drawdown_pct": -3.0,
                        "hit_rate_pct": 50.0,
                        "ic_mean": 0.01,
                    },
                    {
                        "run": "Anchor-Friday",
                        "anchor": "Friday",
                        "execution_model": "C2C",
                        "cost_bps": 0.0,
                        "n_periods": 10,
                        "sharpe": 0.9,
                        "cagr_pct": 9.0,
                        "total_return_pct": 15.0,
                        "max_drawdown_pct": -4.0,
                        "hit_rate_pct": 55.0,
                        "ic_mean": 0.02,
                    },
                    {
                        "run": "Anchor-Average metrics",
                        "anchor": "Average",
                        "execution_model": "C2C",
                        "cost_bps": 0.0,
                        "n_periods": 10,
                        "sharpe": 0.7,
                        "cagr_pct": 7.0,
                        "total_return_pct": 11.0,
                        "max_drawdown_pct": -5.0,
                        "hit_rate_pct": 53.0,
                        "ic_mean": 0.03,
                    },
                    {
                        "run": "Anchor-Friday",
                        "anchor": "Friday",
                        "execution_model": "C2C",
                        "cost_bps": 5.0,
                        "n_periods": 10,
                        "sharpe": 0.1,
                        "cagr_pct": 1.0,
                        "total_return_pct": 2.0,
                        "max_drawdown_pct": -6.0,
                        "hit_rate_pct": 45.0,
                        "ic_mean": 0.02,
                    },
                ]
            ).to_csv(root / "anchor_summary.csv", index=False)

            comparison = v2.legacy_gross_comparison_rows(root)

        self.assertEqual(set(comparison["anchor_offset"]), {"Friday", "Average"})
        self.assertEqual(set(comparison["cost_bps"]), {0.0})
        self.assertIn("Legacy Det Raw Friday", set(comparison["run"]))
        self.assertIn("Legacy Det Raw Anchor Average", set(comparison["run"]))
        self.assertEqual(set(comparison["harness"]), {"legacy raw anchor replay"})
        self.assertEqual(set(comparison["book"]), {"30/30 equal-weight long-short"})
        self.assertIn("price >= $5", comparison.iloc[0]["liquidity_filters"])

    def test_legacy_factor_rows_use_fixed_window_selection_accounting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            score_dir = root / "llm_raw_confidence_score"
            score_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"rebalance_date": "2020-01-03", "position_type": "LONG", "realized_return_pct": 10.0},
                    {"rebalance_date": "2020-01-03", "position_type": "SHORT", "realized_return_pct": 5.0},
                    {"rebalance_date": "2020-01-10", "position_type": "LONG", "realized_return_pct": 0.0},
                    {"rebalance_date": "2020-01-10", "position_type": "SHORT", "realized_return_pct": -10.0},
                ]
            ).to_csv(score_dir / "portfolio_selections.csv", index=False)
            pd.DataFrame(
                [
                    {"score_type": "llm_raw_confidence_score", "ic_mean": 0.12},
                ]
            ).to_csv(root / "compact_leaderboard.csv", index=False)
            pd.DataFrame(
                [
                    {"turnover_total": 2.0},
                    {"turnover_total": 1.0},
                ]
            ).to_csv(score_dir / "review_log.csv", index=False)

            comparison = v2.legacy_factor_gross_comparison_rows(root, include_deterministic=False)

        self.assertEqual(len(comparison), 1)
        row = comparison.iloc[0]
        self.assertEqual(row["execution_model"], "C2C")
        self.assertEqual(row["accounting"], "gross fixed 5-trading-day C2C")
        self.assertEqual(row["harness"], "legacy factor selections replay")
        self.assertEqual(row["n_periods"], 2)
        self.assertAlmostEqual(float(row["total_return_pct"]), 3.5)
        self.assertAlmostEqual(float(row["ic_mean"]), 0.12)
        self.assertAlmostEqual(float(row["avg_turnover_total"]), 1.5)

    def test_llm_cache_storage_key_separates_pool_sizes_on_same_date(self):
        tickers = [f"T{i:02d}" for i in range(12)]
        histories = _make_histories(tickers, start="2019-01-01", end="2019-06-30")
        market = rb.build_market_frames(histories)
        features = v1.build_feature_frames(histories)
        membership = _membership(tickers)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _v2_config("2019-03-01", "2019-06-01", tmpdir)
            context = v2.build_v2_pool_context(market)
            signal_idx = int(context.trading_dates.get_loc(pd.Timestamp("2019-03-15")))
            pool2 = v2.build_candidate_pool_for_index(
                market,
                features,
                membership,
                spy_df=None,
                signal_idx=signal_idx,
                pool_per_side=2,
                config=config,
                context=context,
            )
            pool3 = v2.build_candidate_pool_for_index(
                market,
                features,
                membership,
                spy_df=None,
                signal_idx=signal_idx,
                pool_per_side=3,
                config=config,
                context=context,
            )

        assert pool2 is not None
        assert pool3 is not None
        self.assertNotEqual(
            v2.llm_score_storage_key(pool2, 2),
            v2.llm_score_storage_key(pool3, 3),
        )

    def test_friday_deterministic_replay_matches_v1(self):
        tickers = [f"T{i:02d}" for i in range(24)]
        histories = _make_histories(tickers, start="2019-01-01", end="2019-09-30")
        market = rb.build_market_frames(histories)
        features = v1.build_feature_frames(histories)
        membership = _membership(tickers)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _v2_config("2019-04-01", "2019-08-31", tmpdir)
            v1_weekly, _, _, _ = v1.run_pool_strategy(
                market,
                features,
                membership,
                spy_df=None,
                config=config,
                pool_per_side=5,
                positions_per_side=2,
                score_provider="trail5",
            )

            v2_weekly, _, _, _ = v2.run_v2_pool_strategy(
                market,
                features,
                membership,
                spy_df=None,
                config=config,
                pool_per_side=5,
                positions_per_side=2,
                anchor_offset="Friday",
                score_provider="trail5",
            )

        columns = ["long_ret_c2c", "short_ret_c2c", "long_ret_o2o", "short_ret_o2o", "ic_c2c", "ic_o2o"]
        pd.testing.assert_index_equal(v1_weekly.index, v2_weekly.index)
        pd.testing.assert_frame_equal(v1_weekly[columns], v2_weekly[columns])

    def test_v2_score_diagnostics_accepts_v22_adjustment_provider(self):
        checks = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-10")],
                "pool_side": ["LONG", "LONG", "SHORT"],
                "score_provider": ["llm_adjustment", "llm_adjustment", "llm_adjustment"],
                "llm_score": [90.0, 70.0, 80.0],
                "trail_ret": [-0.10, -0.05, 0.12],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            v2.save_v2_score_diagnostics(checks, Path(tmpdir))
            self.assertTrue((Path(tmpdir) / "score_trail5_correlations.csv").exists())

    def test_v2_score_diagnostics_accepts_v23_rank_provider(self):
        checks = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-10")],
                "pool_side": ["LONG", "LONG", "SHORT"],
                "score_provider": ["llm_rank", "llm_rank", "llm_rank"],
                "llm_score": [100.0, 0.0, 100.0],
                "trail_ret": [-0.10, -0.05, 0.12],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            v2.save_v2_score_diagnostics(checks, Path(tmpdir))
            self.assertTrue((Path(tmpdir) / "score_trail5_correlations.csv").exists())


if __name__ == "__main__":
    unittest.main()
