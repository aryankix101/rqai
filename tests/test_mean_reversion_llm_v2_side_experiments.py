"""Test dedicated-side prompt specifications and hybrid portfolio evaluation.

The suite protects side-scoped cache identity, fixed experiment contracts, and
offline combination of dedicated scorers with the completed joint baseline.
"""

import json
import tempfile
import unittest
from pathlib import Path

import mean_reversion_llm_v2 as runner
import mean_reversion_llm_v2_side_experiments as side_exp


class SidePromptContractTests(unittest.TestCase):
    def test_fixed_specs_round_trip_through_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = side_exp.write_fixed_prompt_specs(Path(tmp))
            long_spec = runner.load_experimental_prompt_spec(paths["long"])
            short_spec = runner.load_experimental_prompt_spec(paths["short"])
        self.assertEqual(long_spec.side_scope, "long")
        self.assertEqual(short_spec.side_scope, "short")
        self.assertEqual(long_spec.replicate_id, "r1")
        self.assertIn("Your sole job is identifying recent losers", long_spec.prompt_text)
        self.assertIn("Your sole job is identifying recent winners", short_spec.prompt_text)
        self.assertEqual(
            runner._feature_glossary_block(long_spec.prompt_text),
            runner._feature_glossary_block(runner.V2_MINIMAL_CHANGE_CONTROL_PROMPT),
        )

    def test_side_validator_accepts_exact_one_side(self) -> None:
        result = runner.validate_and_canonicalize_autoresearch_side_scores(
            {"scores": [{"ticker": "AAA", "score": 80}, {"ticker": "BBB", "score": 20}]},
            ["AAA", "BBB"],
            side_scope="long",
            score_min=0,
            score_max=100,
            score_direction="higher_is_better",
        )
        self.assertEqual(result["side_scope"], "long")
        self.assertEqual([row["score"] for row in result["scores"]], [80.0, 20.0])

    def test_side_validator_rejects_opposite_or_extra_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "only the scores"):
            runner.validate_and_canonicalize_autoresearch_side_scores(
                {"scores": [{"ticker": "AAA", "score": 80}], "short_scores": []},
                ["AAA"],
                side_scope="long",
                score_min=0,
                score_max=100,
                score_direction="higher_is_better",
            )
        with self.assertRaisesRegex(ValueError, "wrong-side"):
            runner.validate_and_canonicalize_autoresearch_side_scores(
                {"scores": [{"ticker": "ZZZ", "score": 80}]},
                ["AAA"],
                side_scope="short",
                score_min=0,
                score_max=100,
                score_direction="higher_is_better",
            )

    def test_replicate_changes_spec_identity_but_not_prompt_hash(self) -> None:
        first = side_exp.fixed_prompt_spec_payload("long", "r1")
        second = side_exp.fixed_prompt_spec_payload("long", "r2")
        self.assertEqual(first["prompt_hash"], second["prompt_hash"])
        self.assertNotEqual(first["replicate_id"], second["replicate_id"])

    def test_side_campaign_rejects_two_sided_format_repair_appendix(self) -> None:
        import mean_reversion_llm_v2_autoresearch as research

        parent = runner.dedicated_side_system_prompt("long")
        proposal = {
            "mutation_mode": "format_repair",
            "patch": {"operation": "replace_once", "old_text": "Return JSON", "new_text": "Return JSON exactly"},
        }
        with self.assertRaisesRegex(research.CampaignError, "do not permit"):
            research.campaign_materialize_prompt_spec(
                proposal,
                candidate_id="g1-test",
                generation="G1",
                parent_id="g0-test",
                parent_prompt=parent,
                parent_hash="unused",
                side_scope="long",
            )


if __name__ == "__main__":
    unittest.main()
