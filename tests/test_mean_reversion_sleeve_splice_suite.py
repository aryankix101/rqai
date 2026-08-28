"""Test sleeve-splice prompt contracts, supervisors, and safety gates.

The cases ensure scorer legs remain resumable, paid execution requires explicit
acknowledgement, and completed caches are combined and analyzed offline.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mean_reversion_llm_v2 as runner
import mean_reversion_llm_v2_sleeve_splice_experiments as experiments
import run_mean_reversion_sleeve_splice_suite as suite_runner


class SleeveSplicePromptContractTests(unittest.TestCase):
    def test_frozen_original_and_b_are_the_canonical_tested_prompts(self) -> None:
        self.assertEqual(experiments.ORIGINAL_V2_PROMPT, runner.V2_SYSTEM_PROMPT)
        self.assertEqual(
            experiments.B_PROMPT,
            runner.G0_05_DEPRESCRIBED_CONTEXT_SYSTEM_PROMPT,
        )
        self.assertEqual(
            experiments.sha256_text(experiments.ORIGINAL_V2_PROMPT),
            experiments.EXPECTED_PROMPT_SHA256["original_v2"],
        )
        self.assertEqual(
            experiments.sha256_text(experiments.B_PROMPT),
            experiments.EXPECTED_PROMPT_SHA256["b_deprescribed"],
        )

    def test_experiment_2_is_exactly_one_line_different_from_b(self) -> None:
        expected = experiments.B_PROMPT.replace(
            experiments.MINIMAL_RANGE_RULE,
            experiments.ORIGINAL_RANGE_RULE,
            1,
        )
        self.assertEqual(experiments.B_CONTEXT_ORIGINAL_RANGE_PROMPT, expected)
        self.assertEqual(experiments.B_PROMPT.count(experiments.MINIMAL_RANGE_RULE), 1)
        self.assertNotIn(
            experiments.MINIMAL_RANGE_RULE,
            experiments.B_CONTEXT_ORIGINAL_RANGE_PROMPT,
        )
        self.assertNotIn(
            experiments.PROPORTIONAL_RANGE_LANGUAGE,
            experiments.B_CONTEXT_ORIGINAL_RANGE_PROMPT,
        )
        self.assertEqual(
            experiments.sha256_text(experiments.B_CONTEXT_ORIGINAL_RANGE_PROMPT),
            "95f008b9ac256f0a5044b714b7a937e44c8e55e1f0e3f47adaf6d2bf8120618a",
        )

    def test_experiment_3_is_explicitly_asymmetric_and_preserves_glossary(self) -> None:
        prompt = experiments.ASYMMETRIC_HYBRID_PROMPT
        for clause in (
            experiments.B_SHORT_FRAMING,
            "LONG and SHORT scores are side-relative.",
            "The two score-range rules intentionally differ.",
            "Do not compare scores across sides.",
            "Return every ticker exactly once on its original side.",
            "Return no rationale, weights, portfolio construction, or commentary.",
        ):
            self.assertIn(clause, prompt)
        self.assertNotIn(experiments.PROPORTIONAL_RANGE_LANGUAGE, prompt)
        self.assertEqual(
            runner._feature_glossary_block(prompt),
            runner._feature_glossary_block(runner.V2_SYSTEM_PROMPT),
        )

    def test_all_four_specs_round_trip_through_the_bounded_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = experiments.write_prompt_specs(Path(tmp))
            loaded = {arm: runner.load_experimental_prompt_spec(path) for arm, path in paths.items()}
        self.assertEqual(tuple(loaded), suite_runner.SCORER_ARMS)
        self.assertTrue(all(spec.side_scope == "both" for spec in loaded.values()))
        self.assertTrue(all(spec.anonymize_tickers for spec in loaded.values()))
        self.assertEqual(
            loaded["asymmetric_hybrid"].prompt_text,
            experiments.ASYMMETRIC_HYBRID_PROMPT,
        )


class SleeveSpliceConstructionTests(unittest.TestCase):
    @staticmethod
    def _entry(long_score: int, short_score: int, universe_hash: str = "same") -> dict:
        return {
            "rebalance_date": "2010-01-04",
            "universe_hash": universe_hash,
            "prompt_hash": "source",
            "cache_key": "cache",
            "long_scores": [{"ticker": "LONG", "score": long_score}],
            "short_scores": [{"ticker": "SHORT", "score": short_score}],
        }

    def test_splice_uses_only_original_longs_and_b_shorts(self) -> None:
        original = {"2010-01-04|pool60": self._entry(91, 12)}
        b_scores = {"2010-01-04|pool60": self._entry(7, 88)}
        result = experiments.splice_score_entries(original, b_scores)["2010-01-04|pool60"]
        self.assertEqual(result["long_scores"], [{"ticker": "LONG", "score": 91}])
        self.assertEqual(result["short_scores"], [{"ticker": "SHORT", "score": 88}])
        self.assertNotEqual(result["long_scores"], b_scores["2010-01-04|pool60"]["long_scores"])
        self.assertNotEqual(result["short_scores"], original["2010-01-04|pool60"]["short_scores"])

    def test_splice_rejects_different_inputs(self) -> None:
        original = {"2010-01-04|pool60": self._entry(91, 12, "first")}
        b_scores = {"2010-01-04|pool60": self._entry(7, 88, "second")}
        with self.assertRaisesRegex(ValueError, "universe mismatch"):
            experiments.splice_score_entries(original, b_scores)


class SleeveSpliceSupervisorSafetyTests(unittest.TestCase):
    def test_prepare_and_validate_never_start_subprocess_or_http_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            with mock.patch.object(
                suite_runner.subprocess,
                "run",
                side_effect=AssertionError("subprocess must not run"),
            ), mock.patch.object(
                runner.httpx,
                "AsyncClient",
                side_effect=AssertionError("HTTP client must not be created"),
            ):
                suite = suite_runner.prepare_suite(output, freeze_inputs=False)
                validation = suite_runner.validate_suite(output / "suite_manifest.json")
        self.assertEqual(suite["portfolio_experiments"], 3)
        self.assertEqual(suite["scorer_legs"], 4)
        self.assertEqual(validation["network_calls_made"], 0)

    def test_suite_freezes_four_legs_1004_nominal_calls_and_1100_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            suite = suite_runner.prepare_suite(output, freeze_inputs=False)
            cache_paths = [suite["arms"][arm]["cache"] for arm in suite_runner.SCORER_ARMS]
        self.assertEqual(tuple(suite["commands"]), suite_runner.SCORER_ARMS)
        self.assertEqual(suite["nominal_paid_calls"], 1004)
        self.assertEqual(suite["hard_aggregate_call_attempt_cap"], 1100)
        self.assertEqual(len(set(cache_paths)), 4)
        for command in suite["commands"].values():
            self.assertIn("--phase", command)
            self.assertEqual(command[command.index("--phase") + 1], "llm-cache")
            self.assertNotIn("--validation-freeze", command)

    def test_wrong_paid_confirmation_rejects_before_subprocess(self) -> None:
        with mock.patch.object(
            suite_runner.subprocess,
            "run",
            side_effect=AssertionError("subprocess must not run"),
        ):
            with self.assertRaisesRegex(suite_runner.SuiteError, "not authorized"):
                suite_runner.run_paid_suite(Path("does-not-matter.json"), confirmation="NO")

    def test_readiness_report_says_no_paid_calls_were_made(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            suite_runner.prepare_suite(output, freeze_inputs=False)
            report = json.loads((output / "readiness_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["paid_calls_made_during_setup"], 0)
        self.assertEqual(report["network_calls_made_during_setup"], 0)
        self.assertEqual(report["remaining_nominal_calls"], 1004)


if __name__ == "__main__":
    unittest.main()
