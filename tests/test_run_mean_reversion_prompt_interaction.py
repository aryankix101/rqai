"""Test the development-only prompt-interaction suite supervisor.

The cases protect blind-period exclusions, frozen-control reuse, resumability,
and analysis of the predeclared B/C/D interaction experiments.
"""

import json
import tempfile
import unittest
from pathlib import Path

import mean_reversion_research_diagnostics as diagnostics
import run_mean_reversion_prompt_interaction as interaction


class PromptInteractionSupervisorTests(unittest.TestCase):
    def test_predeclared_suite_is_development_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = diagnostics.prepare_prompt_interaction_suite(root)
            loaded = interaction.read_suite(root / "suite_manifest.json")

        self.assertTrue(loaded["development_only"])
        self.assertFalse(loaded["blind_period_touched"])
        self.assertEqual(set(loaded["commands"]), {"B", "C", "D"})
        self.assertIn("--paired-control", loaded["commands"]["B"])
        self.assertNotIn("--paired-control", loaded["commands"]["C"])
        self.assertNotIn("--paired-control", loaded["commands"]["D"])
        self.assertEqual(suite["purpose"], "predeclared_four_way_prompt_interaction")

    def test_suite_firewall_rejects_validation_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostics.prepare_prompt_interaction_suite(root)
            path = root / "suite_manifest.json"
            suite = json.loads(path.read_text())
            suite["commands"]["B"].extend(["--validation-freeze", "unsafe.json"])
            path.write_text(json.dumps(suite))

            with self.assertRaisesRegex(
                diagnostics.ResearchDiagnosticError, "validation"
            ):
                interaction.read_suite(path)


if __name__ == "__main__":
    unittest.main()
