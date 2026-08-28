"""Test top-ten autoresearch selection and experiment orchestration.

The suite verifies candidate ranking, manifest behavior, subprocess commands,
and safeguards around the staged V2 prompt-search workflow.
"""

import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import mean_reversion_llm_v2_autoresearch as autoresearch


class Top10ExponentialWeightingTests(unittest.TestCase):
    def test_exp10_weights_only_top_ten_and_normalizes(self):
        ranks = pd.Series(range(1, 13), dtype=float)
        dates = pd.Series(pd.Timestamp("2013-01-01"), index=ranks.index)

        weights = autoresearch.vector_weights_for_rank(dates, ranks, "exp10_h2")

        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertTrue((weights.iloc[:10] > 0.0).all())
        self.assertTrue((weights.iloc[10:] == 0.0).all())
        self.assertAlmostEqual(float(weights.iloc[0] / weights.iloc[1]), float(np.exp(0.5)))

    def test_existing_exp60_model_keeps_its_rank_limit(self):
        kind, value = autoresearch.parse_weight_model("exp60_h10")

        self.assertEqual(kind, "exp")
        self.assertEqual(value, (60, 10.0))

    def test_primary_anonymized_score_gets_top10_exponential_specs(self):
        specs = autoresearch.build_process_specs(
            ["score_v2_anon"],
            ["score_v2_anon"],
            {"score_v2_anon": "Anonymized V2"},
        )
        names = {spec.name for spec in specs}

        for half_life in autoresearch.TOP10_EXP_HALF_LIVES:
            self.assertIn(f"score_v2_anon__exp10_h{half_life}", names)

    def test_short_rank_audit_uses_signed_short_returns_and_search_adjustment(self):
        rows = []
        for date_idx, date in enumerate(pd.date_range("2013-01-01", periods=8, freq="7D")):
            for rank in range(1, 11):
                forward_return = {1: -0.03, 2: 0.00, 3: 0.03}.get(rank, 0.01)
                rows.append(
                    {
                        "date": date,
                        "pool_side": "SHORT",
                        "ticker": f"SEC_{date_idx:02d}_{rank:02d}",
                        "score_v2_anon": 101 - rank,
                        "fwd_c2c": forward_return,
                    }
                )
        panel = pd.DataFrame(rows)

        slices, null = autoresearch.build_short_rank_audit(panel, n_sims=50, seed=11)

        rank1 = slices[slices["construction"].eq("Rank 1 alone")].iloc[0]
        rank3 = slices[slices["construction"].eq("Rank 3 alone")].iloc[0]
        self.assertGreater(rank1["avg_weekly_side_pnl_pct"], 0.0)
        self.assertLess(rank3["avg_weekly_side_pnl_pct"], 0.0)
        self.assertEqual(int(null.iloc[0]["n_concentration_levels"]), 6)
        self.assertEqual(int(null.iloc[0]["n_randomizations"]), 50)
        self.assertIn(null.iloc[0]["observed_best_model"], {"eq10", "exp10_h10", "exp10_h5", "exp10_h3", "exp10_h2", "exp10_h1"})


class PromptAutoresearchCampaignTests(unittest.TestCase):
    @staticmethod
    def _proposal(suffix: str, *, parent_id: str | None = None) -> dict:
        old = (
            "The strongest reversion setups combine an\n"
            "extreme recent move with exhaustion signals and a supportive regime."
        )
        proposal = {
            "mutation_mode": "additive",
            "patch": {
                "operation": "replace_once",
                "old_text": old,
                "new_text": old
                + f"\nPrefer internally consistent cross-sectional evidence ({suffix}).",
            },
            "hypothesis": f"Hypothesis {suffix}",
            "mechanism": "Improves relative ordering.",
            "expected_failure": "May compress scores.",
            "prior_experiment_ids": ["v2-minimal-change-control"],
            "score_min": 0,
            "score_max": 100,
            "score_direction": "higher_is_better",
            "anonymize_tickers": True,
        }
        if parent_id is not None:
            proposal["parent_id"] = parent_id
        return proposal

    @staticmethod
    def _write_frozen_development_manifests(
        campaign_dir: Path,
        state: dict,
    ) -> None:
        all_dates = [
            str(value.date())
            for value in pd.bdate_range("2009-01-06", periods=251)
        ]
        hashes = {}
        file_hashes = {}
        for stage, size in autoresearch.CAMPAIGN_STAGE_SIZES.items():
            dates = all_dates[:size]
            path = campaign_dir / f"manifests/{stage}.json"
            autoresearch.campaign_atomic_json(
                path,
                {
                    "schema_version": 1,
                    "campaign_id": campaign_dir.name,
                    "stage": stage,
                    "purpose": "development",
                    "dates": dates,
                    "date_count": size,
                    "manifest_hash": autoresearch.campaign_manifest_hash(dates),
                    "generated_before_candidates": True,
                },
            )
            hashes[stage] = autoresearch.campaign_manifest_hash(dates)
            file_hashes[stage] = autoresearch.campaign_file_hash(path)
        state["development"]["manifest_hashes"] = hashes
        state["development"]["manifest_file_hashes"] = file_hashes

    def test_researcher_payload_is_always_temp_one_non_thinking_pro(self):
        payload = autoresearch.campaign_researcher_payload("propose")

        self.assertEqual(payload["model"], "deepseek-v4-pro")
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(
            payload["max_tokens"], autoresearch.CAMPAIGN_PRO_MAX_OUTPUT_TOKENS
        )

    def test_stop_sentinel_prevents_pro_call_and_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign_dir = Path(tmp) / "stop-before-pro"
            campaign_dir.mkdir()
            state = autoresearch.campaign_initial_state(
                campaign_dir.name, autoresearch.CAMPAIGN_RUNNER
            )
            (campaign_dir / "STOP").touch()

            def must_not_call(_prompt):
                raise AssertionError("researcher was called after stop")

            with self.assertRaisesRegex(autoresearch.CampaignStop, "Stop requested"):
                autoresearch.campaign_use_researcher(
                    campaign_dir, state, "prompt", must_not_call
                )
            self.assertEqual(state["usage"]["development"]["pro_calls"], 0)

            (campaign_dir / "STOP").unlink()
            with autoresearch.campaign_lock(campaign_dir):
                with self.assertRaisesRegex(
                    autoresearch.CampaignError, "already owns"
                ):
                    with autoresearch.campaign_lock(campaign_dir):
                        self.fail("nested lock unexpectedly acquired")

    def test_format_repair_parent_requires_wrapper_only_terminal_failure(self):
        wrapper_failure = {
            "candidate_id": "G0_01",
            "status": "failed",
            "gate": {"pass": False, "reason": "runner_or_artifact_failure"},
            "reliability": {
                "format_only_dates": ["redacted-date"],
                "format_only_retries": 2,
                "non_format_schema_retries": 0,
                "transient_retries": 0,
            },
        }
        recovered_but_low_dispersion = {
            **wrapper_failure,
            "candidate_id": "G0_02",
            "status": "completed",
            "gate": {"pass": False, "reason": "dispersion_and_schema"},
        }
        transient_failure = {
            **wrapper_failure,
            "candidate_id": "G0_03",
            "reliability": {
                **wrapper_failure["reliability"],
                "transient_retries": 1,
            },
        }

        self.assertEqual(
            autoresearch.campaign_format_repair_parent_ids(
                [recovered_but_low_dispersion, transient_failure, wrapper_failure]
            ),
            ["G0_01"],
        )

    def test_detached_startup_failure_is_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign_dir = Path(tmp) / "startup-recovery"
            campaign_dir.mkdir()
            state = autoresearch.campaign_initial_state(
                campaign_dir.name, autoresearch.CAMPAIGN_RUNNER
            )
            state.update({"status": "starting", "phase": "proposal"})
            autoresearch.campaign_save(campaign_dir, state)

            autoresearch.campaign_record_startup_failure(
                campaign_dir, autoresearch.CampaignError("runner lock busy")
            )

            recovered = autoresearch.campaign_load(campaign_dir)
            self.assertEqual(recovered["status"], "error")
            self.assertIsNone(recovered["supervisor_pid"])
            self.assertIsNone(recovered["terminal_reason"])
            self.assertIn("runner lock busy", recovered["error"])

    def test_runner_subprocess_uses_stable_detached_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign_dir = Path(tmp) / "stable-streams"
            campaign_dir.mkdir()
            state = autoresearch.campaign_initial_state(
                campaign_dir.name, autoresearch.CAMPAIGN_RUNNER
            )
            autoresearch.campaign_save(campaign_dir, state)
            process = mock.Mock(pid=1234, returncode=0)
            process.poll.return_value = 0
            popen = mock.Mock(return_value=process)

            autoresearch.campaign_run_subprocess(
                campaign_dir,
                state,
                [sys.executable, "mean_reversion_llm_v2.py"],
                campaign_dir / "candidate-stage",
                popen=popen,
            )

            kwargs = popen.call_args.kwargs
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
            self.assertTrue(kwargs["close_fds"])

    def test_nested_manifests_and_holdout_firewall(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign_dir = Path(tmp) / "campaign"
            source = Path(tmp) / "dates.csv"
            pd.DataFrame({"date": pd.bdate_range("2009-01-06", periods=251)}).to_csv(source, index=False)

            manifests = autoresearch.campaign_build_nested_manifests(campaign_dir, source)

            smoke = set(manifests["smoke"]["dates"])
            mini = set(manifests["mini"]["dates"])
            medium = set(manifests["medium"]["dates"])
            full = set(manifests["full"]["dates"])
            self.assertEqual([len(smoke), len(mini), len(medium), len(full)], [5, 24, 80, 251])
            self.assertTrue(smoke < mini < medium < full)
            with self.assertRaises(autoresearch.CampaignFirewallError):
                autoresearch.campaign_assert_dates(["2020-01-02"])

    def test_catalog_deduplicates_complete_runs_excludes_incomplete_and_never_reads_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            development = root / "prompt_trials_2009_2014"
            dates = (
                pd.read_csv(autoresearch.CAMPAIGN_DATE_SOURCE, usecols=["date"])["date"]
                .drop_duplicates()
                .astype(str)
                .tolist()
            )
            config = {
                "start": "2009-01-01",
                "end": "2014-01-01",
                "prompt_hash": "a" * 64,
                "prompt_variant": "same-prompt",
                "model": autoresearch.CAMPAIGN_SCORER_MODEL,
                "temperature": 1.0,
                "thinking_mode": "disabled",
                "pool_sizes": [60],
                "book_sizes": [30],
                "llm_pool_size": 60,
                "anchor_offsets": [3],
                "schedule": "anchors",
                "anonymize_tickers": True,
                "cost_bps_levels": [0.0],
                "output_contract": autoresearch.CAMPAIGN_OUTPUT_CONTRACT,
            }
            for name in ("a_complete", "b_duplicate"):
                run_dir = development / name
                autoresearch.campaign_atomic_json(run_dir / "run_config.json", config)
                run_dates = dates[:24] if name == "a_complete" else dates
                pd.DataFrame({"date": run_dates, "capital_c2c": 0.0}).to_csv(
                    run_dir / "weekly_returns.csv", index=False
                )

            score_rows = [
                {"ticker": f"SEC_{index:04d}", "score": 50 + (index % 11)}
                for index in range(60)
            ]
            autoresearch.campaign_atomic_json(
                development / "cache/complete.json",
                {
                    "metadata": {
                        "prompt_hash": config["prompt_hash"],
                        "model": config["model"],
                        "temperature": config["temperature"],
                        "thinking_mode": config["thinking_mode"],
                        "anonymize_tickers": config["anonymize_tickers"],
                        "output_contract": config["output_contract"],
                    },
                    "signals": {
                        f"{date}|pool60": {
                            "long_scores": score_rows,
                            "short_scores": score_rows,
                        }
                        for date in dates
                    },
                    "partial_signals": {},
                },
            )

            incomplete_dir = development / "c_incomplete"
            incomplete_config = dict(config)
            incomplete_config.update(
                {"prompt_hash": "b" * 64, "prompt_variant": "incomplete-prompt"}
            )
            autoresearch.campaign_atomic_json(
                incomplete_dir / "run_config.json", incomplete_config
            )
            pd.DataFrame(
                {"date": dates[:24], "capital_c2c": 0.0}
            ).to_csv(incomplete_dir / "weekly_returns.csv", index=False)

            holdout = root / "forbidden_2009_2014_blended_2020_2024"
            holdout.mkdir(parents=True)
            # Invalid JSON makes this a tripwire: cataloging would fail if the
            # holdout tree were opened rather than rejected at the top level.
            (holdout / "run_config.json").write_text("{not-json", encoding="utf-8")

            catalog = autoresearch.campaign_catalog(root)

            self.assertEqual(catalog["summary"]["entries"], 3)
            self.assertEqual(catalog["summary"]["complete_unique"], 1)
            eligible = [row for row in catalog["entries"] if row["leaderboard_eligible"]]
            duplicates = [row for row in catalog["entries"] if row["duplicate"]]
            incomplete = [row for row in catalog["entries"] if not row["complete"]]
            self.assertEqual(len(eligible), 1)
            self.assertEqual(len(duplicates), 1)
            self.assertEqual(len(incomplete), 2)
            self.assertTrue(eligible[0]["source_path"].endswith("b_duplicate"))
            self.assertNotIn("2020_2024", json.dumps(catalog, sort_keys=True))

    def test_sanitized_feedback_contains_only_qualitative_development_labels(self):
        result = {
            "candidate_id": "G0_01",
            "parent_id": "v2-minimal-change-control",
            "stage": "mini",
            "gate": {"pass": True},
            "metrics": {
                "paired_annualized_lift_pct": 91.234567,
                "long_paired_lift_pct": 13.456789,
                "short_paired_lift_pct": -7.654321,
                "tail_improvement_pct": -2.345678,
                "sharpe": 8.765432,
            },
            "dates": ["2009-07-14"],
            "raw_returns": [0.123456789],
            "smoke": {"low_dispersion_warning": True},
        }

        feedback = autoresearch.campaign_sanitized_feedback(result)
        serialized = json.dumps(feedback, sort_keys=True)

        self.assertEqual(
            set(feedback),
            {
                "candidate",
                "parent",
                "stage",
                "result_class",
                "qualitative_labels",
                "permitted_action",
            },
        )
        self.assertEqual(feedback["result_class"], "promising")
        self.assertEqual(
            feedback["qualitative_labels"],
            [
                "long_side_improvement",
                "short_side_degradation",
                "tail_risk_degradation",
                "score_compression",
            ],
        )
        for forbidden in (
            "91.234567",
            "13.456789",
            "7.654321",
            "2.345678",
            "8.765432",
            "2009-07-14",
            "0.123456789",
            "raw_returns",
            "metrics",
            "dates",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_ranking_uses_reliability_for_metric_ties_then_candidate_id(self):
        tied_metrics = {
            "paired_annualized_lift_pct": 2.0,
            "sharpe_improvement": 0.2,
            "tail_improvement_pct": 0.1,
            "long_paired_lift_pct": 1.0,
            "short_paired_lift_pct": 1.0,
        }
        incomplete_a = {
            "candidate_id": "A_candidate",
            "metrics": tied_metrics,
            "reliability": {
                "complete_coverage": False,
                "schema_retries": 0,
                "transient_retries": 0,
            },
        }
        reliable_z = {
            "candidate_id": "Z_candidate",
            "metrics": tied_metrics,
            "reliability": {
                "complete_coverage": True,
                "schema_retries": 0,
                "transient_retries": 0,
            },
        }

        self.assertEqual(
            autoresearch.campaign_rank_results(
                [incomplete_a, reliable_z], "mini"
            )[0],
            "Z_candidate",
        )

        complete_a = dict(incomplete_a)
        complete_a["reliability"] = dict(reliable_z["reliability"])
        forward = autoresearch.campaign_rank_results(
            [reliable_z, complete_a], "mini"
        )
        reverse = autoresearch.campaign_rank_results(
            [complete_a, reliable_z], "mini"
        )
        self.assertEqual(forward, ["A_candidate", "Z_candidate"])
        self.assertEqual(reverse, forward)

    def test_full_ranking_prefers_robust_g002_over_tiny_headline_gap(self):
        def result(candidate_id, lift, stability, long_lift, short_lift, lower):
            return {
                "candidate_id": candidate_id,
                "metrics": {
                    "paired_annualized_lift_pct": lift,
                    "sharpe_improvement": 0.1,
                    "chronological_block_stability_pct": stability,
                    "long_paired_lift_pct": long_lift,
                    "short_paired_lift_pct": short_lift,
                    "trimmed_5pct_annualized_lift_pct": -2.0,
                    "max_drawdown_pct": -7.0,
                },
                "bootstrap_vs_incumbent": {"lower_95_pct": lower},
                "controls": {"fresh_incumbent": {"cagr_pct": 13.5}},
            }

        g001 = result("G0_01", 1.16, 50.0, -0.12, 1.28, -1.87)
        g002 = result("G0_02", 1.06, 75.0, 0.02, 1.03, -2.07)

        ranked = sorted([g001, g002], key=autoresearch.campaign_full_sort_key)

        self.assertEqual(ranked[0]["candidate_id"], "G0_02")

    def test_mini_ranking_uses_signs_not_noisy_lift_magnitude(self):
        def result(candidate_id, lift, long_lift, short_lift):
            return {
                "candidate_id": candidate_id,
                "metrics": {
                    "paired_annualized_lift_pct": lift,
                    "long_paired_lift_pct": long_lift,
                    "short_paired_lift_pct": short_lift,
                    "chronological_block_stability_pct": 50.0,
                    "tail_improvement_pct": 0.0,
                },
                "reliability": {
                    "complete_coverage": True,
                    "schema_retries": 0,
                    "transient_retries": 0,
                },
            }

        noisy = result("A_noisy", 12.0, 12.5, -0.5)
        balanced = result("Z_balanced", 0.5, 0.25, 0.25)

        self.assertEqual(
            autoresearch.campaign_rank_results([noisy, balanced], "mini")[0],
            "Z_balanced",
        )

    def test_mini_prunes_when_moving_block_upper_bound_is_nonpositive(self):
        dates = [
            str(value.date())
            for value in pd.bdate_range("2009-01-06", periods=24)
        ]
        manifest = {"dates": dates, "date_count": len(dates)}

        def returns(value: float) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "capital_c2c": [value] * len(dates),
                    "long_contribution": [value / 2.0] * len(dates),
                    "short_contribution": [value / 2.0] * len(dates),
                }
            )

        frames = {
            "treatment": returns(-0.001),
            "control": returns(0.0),
            "trail5": returns(0.0),
        }
        with mock.patch.object(
            autoresearch,
            "campaign_resolve_artifact_dir",
            side_effect=lambda _output, role: Path(role),
        ), mock.patch.object(
            autoresearch,
            "campaign_load_returns",
            side_effect=lambda run_dir, _dates, **_kwargs: frames[
                Path(run_dir).name
            ],
        ):
            result = autoresearch.campaign_evaluate_stage(
                Path("unused"), manifest, "mini"
            )

        bootstrap = result["bootstrap_vs_incumbent"]
        self.assertEqual(bootstrap["n_boot"], 5_000)
        self.assertEqual(bootstrap["block_size"], 4)
        self.assertLessEqual(bootstrap["upper_90_pct"], 0.0)
        self.assertEqual(
            result["gate"],
            {"pass": False, "reason": "one_sided_90pct_upper_bound"},
        )

    def test_validation_holm_terminal_classes_and_fresh_control_drawdown_gate(self):
        result = {
            "metrics": {
                "n_periods": 250,
                "cagr_pct": 18.0,
                "sharpe": 1.8,
                "max_drawdown_pct": -10.0,
            },
            "controls": {
                "fresh_incumbent": {
                    "cagr_pct": 12.0,
                    "sharpe": 1.2,
                    "max_drawdown_pct": -9.0,
                },
                # Trail5 drawdown is intentionally much shallower: the frozen
                # 2pp drawdown gate is against the fresh control only.
                "trail5": {
                    "cagr_pct": 10.0,
                    "sharpe": 1.0,
                    "max_drawdown_pct": -2.0,
                },
            },
            "bootstrap_vs_incumbent": {
                "mean_annualized_delta_pct": 4.0,
                "p_not_positive": 0.01,
                "lower_one_sided_97_5_pct": 0.5,
                "lower_one_sided_95_pct": 0.8,
            },
            "bootstrap_vs_trail5": {
                "mean_annualized_delta_pct": 5.0,
                "p_not_positive": 0.02,
                "lower_one_sided_97_5_pct": 0.4,
                "lower_one_sided_95_pct": 0.7,
            },
        }

        confirmed = autoresearch.campaign_validation_decision(
            result, expected_periods=250
        )
        self.assertEqual(confirmed["terminal_status"], "statistically_confirmed")
        self.assertEqual(confirmed["holm_adjusted_p_not_positive"], [0.02, 0.02])
        self.assertTrue(confirmed["pass"])

        inconclusive = json.loads(json.dumps(result))
        inconclusive["bootstrap_vs_incumbent"]["p_not_positive"] = 0.2
        inconclusive["bootstrap_vs_incumbent"]["lower_one_sided_95_pct"] = -0.1
        inconclusive["bootstrap_vs_incumbent"]["lower_one_sided_97_5_pct"] = -0.2
        decision = autoresearch.campaign_validation_decision(
            inconclusive, expected_periods=250
        )
        self.assertEqual(decision["terminal_status"], "validated_winner")

        failed = json.loads(json.dumps(result))
        failed["metrics"]["sharpe"] = 0.9
        self.assertEqual(
            autoresearch.campaign_validation_decision(
                failed, expected_periods=250
            )["terminal_status"],
            "failed_validation",
        )

    def test_frozen_prompt_and_manifest_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign_dir = Path(tmp) / "tamper_campaign"
            campaign_dir.mkdir()
            state = autoresearch.campaign_initial_state(
                campaign_dir.name, autoresearch.CAMPAIGN_RUNNER
            )
            self._write_frozen_development_manifests(campaign_dir, state)

            parent, parent_hash = autoresearch.campaign_incumbent_prompt()
            spec = autoresearch.campaign_materialize_prompt_spec(
                self._proposal("tamper-proof"),
                candidate_id="G0_01",
                generation="G0",
                parent_id="v2-minimal-change-control",
                parent_prompt=parent,
                parent_hash=parent_hash,
            )
            spec_path = campaign_dir / "candidates/G0_01/prompt_spec.json"
            autoresearch.campaign_atomic_json(spec_path, spec)
            state["candidates"]["G0_01"] = {
                "candidate_id": "G0_01",
                "generation": "G0",
                "parent_id": "v2-minimal-change-control",
                "prompt_hash": spec["prompt_hash"],
                "anonymize_tickers": spec["anonymize_tickers"],
                "prompt_spec_hash": autoresearch.campaign_file_hash(spec_path),
            }

            loaded, _ = autoresearch.campaign_verify_prompt_spec(
                campaign_dir, state, "G0_01"
            )
            self.assertEqual(loaded["prompt_hash"], spec["prompt_hash"])
            spec["hypothesis"] = "tampered after static validation"
            autoresearch.campaign_atomic_json(spec_path, spec)
            with self.assertRaises(autoresearch.CampaignFirewallError):
                autoresearch.campaign_verify_prompt_spec(
                    campaign_dir, state, "G0_01"
                )

            manifest_path = campaign_dir / "manifests/mini.json"
            autoresearch.campaign_read_manifest(
                manifest_path,
                expected_hash=state["development"]["manifest_hashes"]["mini"],
                expected_file_hash=state["development"]["manifest_file_hashes"][
                    "mini"
                ],
            )
            manifest = autoresearch.campaign_read_json(manifest_path)
            manifest["dates"][0] = "2009-01-07"
            autoresearch.campaign_atomic_json(manifest_path, manifest)
            with self.assertRaises(autoresearch.CampaignFirewallError):
                autoresearch.campaign_read_manifest(
                    manifest_path,
                    expected_hash=state["development"]["manifest_hashes"][
                        "mini"
                    ],
                    expected_file_hash=state["development"][
                        "manifest_file_hashes"
                    ]["mini"],
                )

    def test_manual_validation_approval_freezes_all_development_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign_dir = Path(tmp) / "approval-campaign"
            campaign_dir.mkdir()
            state = autoresearch.campaign_initial_state(
                campaign_dir.name, autoresearch.CAMPAIGN_RUNNER
            )
            self._write_frozen_development_manifests(campaign_dir, state)
            autoresearch.campaign_atomic_json(
                campaign_dir / "catalog.json", {"entries": [], "summary": {}}
            )

            parent, parent_hash = autoresearch.campaign_incumbent_prompt()
            spec = autoresearch.campaign_materialize_prompt_spec(
                self._proposal("approval"),
                candidate_id="G0_01",
                generation="G0",
                parent_id="v2-minimal-change-control",
                parent_prompt=parent,
                parent_hash=parent_hash,
            )
            spec_path = campaign_dir / "candidates/G0_01/prompt_spec.json"
            autoresearch.campaign_atomic_json(spec_path, spec)
            spec_file_hash = autoresearch.campaign_file_hash(spec_path)
            state["candidates"]["G0_01"] = {
                "candidate_id": "G0_01",
                "generation": "G0",
                "parent_id": "v2-minimal-change-control",
                "prompt_hash": spec["prompt_hash"],
                "anonymize_tickers": spec["anonymize_tickers"],
                "prompt_spec_hash": spec_file_hash,
                "status": "full_pass",
            }
            frozen_inputs = {
                "candidate_id": "G0_01",
                "parent_id": "v2-minimal-change-control",
                "prompt_hash": spec["prompt_hash"],
                "prompt_spec_hash": spec_file_hash,
                "manifest_hash": state["development"]["manifest_hashes"]["full"],
                "code_fingerprint": state["code_fingerprint"],
                "supervisor_code_fingerprint": state["supervisor_code_fingerprint"],
                "harness_hash": state["harness_hash"],
                "anonymize_tickers": spec["anonymize_tickers"],
                "control_artifact": autoresearch.campaign_control_artifact(
                    anonymize_tickers=spec["anonymize_tickers"],
                    validation=False,
                ),
                "trail5_artifact": autoresearch.campaign_trail5_artifact(
                    validation=False
                ),
            }
            full_path = autoresearch.campaign_stage_result_path(
                campaign_dir, "G0_01", "full"
            )
            autoresearch.campaign_atomic_json(
                full_path,
                {
                    "candidate_id": "G0_01",
                    "parent_id": "v2-minimal-change-control",
                    "stage": "full",
                    "status": "completed",
                    "gate": {"pass": True},
                    "frozen_inputs": frozen_inputs,
                },
            )
            state.update(
                {
                    "winner_id": "G0_01",
                    "status": "awaiting_validation_approval",
                    "phase": "approval",
                }
            )
            autoresearch.campaign_save(campaign_dir, state)

            def fake_validation_manifest(target: Path):
                dates = ["2015-01-06"]
                exits = ["2015-01-13"]
                manifest = {
                    "schema_version": 1,
                    "campaign_id": target.name,
                    "stage": "validation",
                    "purpose": "validation",
                    "dates": dates,
                    "exit_dates": exits,
                    "date_count": 1,
                    "manifest_hash": autoresearch.campaign_manifest_hash(dates),
                    "exit_date_hash": autoresearch.campaign_manifest_hash(exits),
                }
                autoresearch.campaign_atomic_json(
                    target / "manifests/validation.json", manifest
                )
                return manifest

            with mock.patch.object(
                autoresearch,
                "campaign_build_validation_manifest",
                side_effect=fake_validation_manifest,
            ):
                freeze_path = autoresearch.campaign_approve_validation(
                    campaign_dir, approver="unit-test"
                )

            freeze = autoresearch.campaign_read_json(freeze_path)
            approved = autoresearch.campaign_load(campaign_dir)
            self.assertEqual(approved["status"], "validation_approved")
            self.assertEqual(freeze["prompt_spec_hash"], spec_file_hash)
            self.assertEqual(
                freeze["development_results_hash"],
                autoresearch.campaign_file_hash(full_path),
            )
            self.assertEqual(freeze["temperature"], 1.0)
            self.assertEqual(freeze["thinking_mode"], "disabled")
            self.assertEqual(freeze["lineage"][0]["candidate_id"], "G0_01")

            full_result = autoresearch.campaign_read_json(full_path)
            full_result["tampered"] = True
            autoresearch.campaign_atomic_json(full_path, full_result)
            with self.assertRaisesRegex(
                autoresearch.CampaignFirewallError,
                "fingerprint mismatch",
            ):
                autoresearch.campaign_run_validation(campaign_dir)

    def test_validation_attempt_failures_checkpoint_and_resume_without_sticky_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign_dir = Path(tmp) / "validation-resume"
            campaign_dir.mkdir()
            state = autoresearch.campaign_initial_state(
                campaign_dir.name, autoresearch.CAMPAIGN_RUNNER
            )
            self._write_frozen_development_manifests(campaign_dir, state)
            parent, parent_hash = autoresearch.campaign_incumbent_prompt()
            spec = autoresearch.campaign_materialize_prompt_spec(
                self._proposal("validation-resume"),
                candidate_id="G0_01",
                generation="G0",
                parent_id="v2-minimal-change-control",
                parent_prompt=parent,
                parent_hash=parent_hash,
            )
            spec_path = campaign_dir / "candidates/G0_01/prompt_spec.json"
            autoresearch.campaign_atomic_json(spec_path, spec)
            state["candidates"]["G0_01"] = {
                "candidate_id": "G0_01",
                "generation": "G0",
                "parent_id": "v2-minimal-change-control",
                "prompt_hash": spec["prompt_hash"],
                "anonymize_tickers": spec["anonymize_tickers"],
                "prompt_spec_hash": autoresearch.campaign_file_hash(spec_path),
            }
            validation_manifest = {
                "schema_version": 1,
                "campaign_id": campaign_dir.name,
                "stage": "validation",
                "purpose": "validation",
                "dates": ["2015-01-06"],
                "exit_dates": ["2015-01-13"],
                "date_count": 1,
                "manifest_hash": autoresearch.campaign_manifest_hash(
                    ["2015-01-06"]
                ),
                "exit_date_hash": autoresearch.campaign_manifest_hash(
                    ["2015-01-13"]
                ),
            }
            validation_path = campaign_dir / "manifests/validation.json"
            autoresearch.campaign_atomic_json(validation_path, validation_manifest)
            state["validation"].update(
                {
                    "approved": True,
                    "manifest_hash": validation_manifest["manifest_hash"],
                    "manifest_file_hash": autoresearch.campaign_file_hash(
                        validation_path
                    ),
                }
            )

            def failed_executor(*_args, **_kwargs):
                raise RuntimeError("simulated provider outage")

            for attempt in range(3):
                with self.assertRaisesRegex(
                    autoresearch.CampaignStop,
                    "failure checkpointed|Three consecutive",
                ):
                    autoresearch.campaign_execute_stage(
                        campaign_dir,
                        state,
                        "G0_01",
                        "validation",
                        validation=True,
                        executor=failed_executor,
                    )
                self.assertEqual(state["consecutive_provider_failures"], attempt + 1)
                self.assertFalse((campaign_dir / "validation/result.json").exists())

            attempts = list(
                (campaign_dir / "validation/attempt_failures").glob("*.json")
            )
            self.assertEqual(len(attempts), 3)

            completed_result = {
                "status": "completed",
                "stage": "validation",
                "metrics": {"n_periods": 1},
                "controls": {},
                "bootstrap_vs_incumbent": {},
                "bootstrap_vs_trail5": {},
                "gate": {"pass": True},
            }
            with mock.patch.object(
                autoresearch,
                "campaign_evaluate_stage",
                return_value=completed_result,
            ):
                resumed = autoresearch.campaign_execute_stage(
                    campaign_dir,
                    state,
                    "G0_01",
                    "validation",
                    validation=True,
                    executor=lambda *_args, **_kwargs: None,
                )

            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(state["consecutive_provider_failures"], 0)
            self.assertTrue((campaign_dir / "validation/result.json").exists())

    def test_campaign_judge_is_gross_only_and_has_no_banned_alpha_or_cost_rules(self):
        harness = autoresearch.campaign_harness()
        self.assertEqual(harness["return_basis"], "gross_fixed_horizon_c2c")
        self.assertEqual(harness["cost_bps_levels"], [0.0])
        self.assertEqual(harness["ticker_visibility"], "candidate_selected")
        self.assertEqual(harness["allowed_anonymize_tickers"], [True, False])

        judge_source = "\n".join(
            inspect.getsource(function)
            for function in (
                autoresearch.campaign_evaluate_stage,
                autoresearch.campaign_rank_results,
                autoresearch.campaign_full_sort_key,
                autoresearch.campaign_sanitized_feedback,
            )
        ).lower()
        for banned in (
            "newey",
            "spy",
            "regression",
            "basis-point sensitivity",
            "bps_sensitivity",
            "cost_sensitivity",
            "estimated_cost",
        ):
            self.assertNotIn(banned, judge_source)
        self.assertIn("paired_annualized_lift_pct", judge_source)
        self.assertIn("capital_c2c", inspect.getsource(autoresearch.campaign_evaluate_stage))

    def test_prompt_spec_preserves_glossary_and_declares_score_contract(self):
        parent, parent_hash = autoresearch.campaign_incumbent_prompt()
        old = "The strongest reversion setups combine an\nextreme recent move with exhaustion signals and a supportive regime."
        proposal = {
            "mutation_mode": "additive",
            "patch": {"operation": "replace_once", "old_text": old, "new_text": old + "\nPrefer cross-sectional evidence that is internally consistent."},
            "hypothesis": "Consistency improves ordering.",
            "mechanism": "Reduces contradictory score assignments.",
            "expected_failure": "Scores may compress.",
            "prior_experiment_ids": ["v2-minimal-change-control"],
            "score_min": -100,
            "score_max": 100,
            "score_direction": "lower_is_better",
            "anonymize_tickers": False,
        }

        spec = autoresearch.campaign_materialize_prompt_spec(
            proposal,
            candidate_id="G0_01",
            generation="G0",
            parent_id="v2-minimal-change-control",
            parent_prompt=parent,
            parent_hash=parent_hash,
        )

        self.assertEqual(spec["temperature"], 1.0)
        self.assertEqual(spec["thinking_mode"], "disabled")
        self.assertEqual(spec["output_contract"], "candidate_declared_numeric_score")
        self.assertEqual(spec["score_min"], -100)
        self.assertEqual(spec["score_max"], 100)
        self.assertEqual(spec["score_direction"], "lower_is_better")
        self.assertFalse(spec["anonymize_tickers"])
        self.assertEqual(spec["prompt_hash"], autoresearch.campaign_sha256(spec["prompt_text"]))

    def test_visible_candidate_uses_visible_runner_flag_without_control_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign_dir = Path(tmp) / "visible-campaign"
            campaign_dir.mkdir()
            state = autoresearch.campaign_initial_state(
                campaign_dir.name, autoresearch.CAMPAIGN_RUNNER
            )
            self._write_frozen_development_manifests(campaign_dir, state)
            parent, parent_hash = autoresearch.campaign_incumbent_prompt()
            proposal = self._proposal("visible")
            proposal["anonymize_tickers"] = False
            spec = autoresearch.campaign_materialize_prompt_spec(
                proposal,
                candidate_id="G0_01",
                generation="G0",
                parent_id="v2-minimal-change-control",
                parent_prompt=parent,
                parent_hash=parent_hash,
            )
            spec_path = campaign_dir / "candidates/G0_01/prompt_spec.json"
            autoresearch.campaign_atomic_json(spec_path, spec)

            command, _ = autoresearch.campaign_runner_command(
                state, campaign_dir, spec_path, "smoke"
            )

            self.assertIn("--no-anonymize-tickers", command)
            self.assertNotIn("--anonymize-tickers", command)
            self.assertNotIn("--paired-control", command)
            self.assertNotIn("--paired-control-cache", command)
            control = autoresearch.campaign_control_artifact(
                anonymize_tickers=False, validation=False
            )
            self.assertTrue(control["visibility_matched"])
            self.assertIn("v2_minimal_change_visible", control["run_dir"])

    def test_atomic_prompt_mutation_has_no_changed_token_or_character_delta_cap(self):
        parent, parent_hash = autoresearch.campaign_incumbent_prompt()
        old = (
            "The strongest reversion setups combine an\n"
            "extreme recent move with exhaustion signals and a supportive regime."
        )
        addition = " ".join(["Prefer consistent evidence."] * 45)
        proposal = self._proposal("long atomic elaboration")
        proposal["patch"]["new_text"] = old + "\n" + addition

        spec = autoresearch.campaign_materialize_prompt_spec(
            proposal,
            candidate_id="G0_01",
            generation="G0",
            parent_id="v2-minimal-change-control",
            parent_prompt=parent,
            parent_hash=parent_hash,
        )

        self.assertGreater(spec["compliance"]["changed_tokens"], 80)

    def test_rejected_proposal_audit_is_sanitized_and_append_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign_dir = Path(temporary) / "campaign"
            campaign_dir.mkdir()
            proposal = self._proposal("unsafe audit")
            proposal["patch"]["new_text"] += " Treat AAPL as of 2024-01-01 as the archetype."

            autoresearch.campaign_record_rejected_proposal(
                campaign_dir,
                proposal,
                candidate_id="G0_01",
                generation="G0",
                parent_id="v2-minimal-change-control",
                reason="Mutation introduces an identifier",
            )
            autoresearch.campaign_record_rejected_proposal(
                campaign_dir,
                {**proposal, "hypothesis": "Second rejected idea"},
                candidate_id="G0_01",
                generation="G0",
                parent_id="v2-minimal-change-control",
                reason="Mutation introduces dated knowledge",
            )

            records = [
                json.loads(line)
                for line in (campaign_dir / "rejected_proposals.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertNotEqual(records[0]["rejection_id"], records[1]["rejection_id"])
            serialized = json.dumps(records)
            self.assertNotIn("AAPL", serialized)
            self.assertNotIn("2024-01-01", serialized)
            self.assertIn("[REDACTED]", serialized)
            feedback = autoresearch.campaign_recent_rejection_feedback(
                campaign_dir,
                generation="G0",
            )
            self.assertEqual(len(feedback), 2)
            self.assertEqual(feedback[0]["returned_mutation_mode"], "additive")
            self.assertEqual(feedback[0]["returned_patch_operation"], "replace_once")
            self.assertNotIn("new_text", json.dumps(feedback))

    def test_g0_researcher_prompt_enumerates_exact_mutation_schema(self):
        parent, parent_hash = autoresearch.campaign_incumbent_prompt()
        prompt = autoresearch.campaign_researcher_prompt_g0(
            {"incumbent": {"prompt_text": parent, "prompt_hash": parent_hash}},
            {"entries": []},
        )

        self.assertIn("'additive', 'substitutive', or 'rewrite'", prompt)
        self.assertIn("operation:'replace_once'", prompt)
        self.assertIn("FEATURE GLOSSARY block must remain byte-for-byte identical", prompt)
        self.assertIn("score_direction", prompt)
        self.assertIn("anonymize_tickers", prompt)
        self.assertIn("real ticker identifiers", prompt)

    def test_static_prompt_guard_is_mechanical_and_freezes_glossary(self):
        parent, parent_hash = autoresearch.campaign_incumbent_prompt()
        old = (
            "The strongest reversion setups combine an\n"
            "extreme recent move with exhaustion signals and a supportive regime."
        )

        multi_idea = self._proposal("unused")
        multi_idea["mutation_mode"] = "rewrite"
        multi_idea["patch"]["new_text"] = old.replace(
            "strongest", "best"
        ).replace("supportive", "favorable")
        rewritten = autoresearch.campaign_materialize_prompt_spec(
            multi_idea,
            candidate_id="G0_01",
            generation="G0",
            parent_id="v2-minimal-change-control",
            parent_prompt=parent,
            parent_hash=parent_hash,
        )
        self.assertEqual(rewritten["mutation_mode"], "rewrite")

        full_rewrite = self._proposal("full rewrite")
        full_rewrite["mutation_mode"] = "rewrite"
        full_rewrite["patch"] = {
            "operation": "replace_once",
            "old_text": parent,
            "new_text": parent.replace(
                "Mean reversion is the tendency", "Rank candidates by whether"
            ).replace(
                "Do not manufacture separation.", "Use whatever separation the evidence supports."
            ),
        }
        full_spec = autoresearch.campaign_materialize_prompt_spec(
            full_rewrite,
            candidate_id="G0_02",
            generation="G0",
            parent_id="v2-minimal-change-control",
            parent_prompt=parent,
            parent_hash=parent_hash,
        )
        self.assertGreater(full_spec["compliance"]["change_blocks"], 1)

        ticker_leak = self._proposal("unused")
        ticker_leak["patch"]["new_text"] = old + "\nTreat AAPL as the archetype."
        with self.assertRaisesRegex(autoresearch.CampaignError, "ticker|identifier"):
            autoresearch.campaign_materialize_prompt_spec(
                ticker_leak,
                candidate_id="G0_01",
                generation="G0",
                parent_id="v2-minimal-change-control",
                parent_prompt=parent,
                parent_hash=parent_hash,
            )

        generic = self._proposal("unused")
        generic["patch"]["new_text"] = old + "\nBe careful and consider all factors."
        accepted = autoresearch.campaign_materialize_prompt_spec(
            generic,
            candidate_id="G0_03",
            generation="G0",
            parent_id="v2-minimal-change-control",
            parent_prompt=parent,
            parent_hash=parent_hash,
        )
        self.assertIn("Be careful", accepted["prompt_text"])

        glossary_edit = self._proposal("unused")
        glossary_edit["mutation_mode"] = "rewrite"
        glossary_edit["patch"] = {
            "operation": "replace_once",
            "old_text": "ADX\n- 14-day Average Directional Index, trend strength only.",
            "new_text": "ADX\n- A different and inaccurate definition.",
        }
        with self.assertRaisesRegex(autoresearch.CampaignError, "glossary"):
            autoresearch.campaign_materialize_prompt_spec(
                glossary_edit,
                candidate_id="G0_04",
                generation="G0",
                parent_id="v2-minimal-change-control",
                parent_prompt=parent,
                parent_hash=parent_hash,
            )

        g0_format = self._proposal("unused")
        g0_format["mutation_mode"] = "format_repair"
        g0_format["patch"]["new_text"] = old + "\nReturn a strict JSON response."
        with self.assertRaisesRegex(autoresearch.CampaignError, "G1 child"):
            autoresearch.campaign_materialize_prompt_spec(
                g0_format,
                candidate_id="G0_01",
                generation="G0",
                parent_id="v2-minimal-change-control",
                parent_prompt=parent,
                parent_hash=parent_hash,
            )

    def test_progress_usage_counts_real_runner_events_and_format_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp) / "events.jsonl"
            rows = [
                {"event": "llm_call_started", "signal_date": "2009-01-06"},
                {"event": "llm_call_failed", "signal_date": "2009-01-06", "schema_error": True, "transient_error": False, "format_only_repair_eligible": True, "output_tokens": 12},
                {"event": "llm_call_started", "signal_date": "2009-01-06"},
                {"event": "llm_call_succeeded", "signal_date": "2009-01-06", "output_tokens": 20},
            ]
            progress.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            usage = autoresearch.campaign_progress_usage(progress)

            self.assertEqual(usage["flash_calls"], 2)
            self.assertEqual(usage["flash_output_tokens"], 32)
            self.assertEqual(usage["schema_retry_dates"], ["2009-01-06"])
            self.assertEqual(usage["format_only_dates"], ["2009-01-06"])

    def test_campaign_terminates_with_no_winner_when_smoke_prunes_every_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign_dir = Path(tmp) / "no_winner_campaign"
            campaign_dir.mkdir()
            state = autoresearch.campaign_initial_state(
                campaign_dir.name, autoresearch.CAMPAIGN_RUNNER
            )
            self._write_frozen_development_manifests(campaign_dir, state)
            autoresearch.campaign_atomic_json(
                campaign_dir / "catalog.json",
                {"entries": [], "summary": {"entries": 0}},
            )
            autoresearch.campaign_save(campaign_dir, state)
            calls = []

            def fake_researcher(_prompt):
                return (
                    [self._proposal(f"loser-{idx}") for idx in range(6)],
                    {"calls": 1, "output_tokens": 20},
                )

            def fake_executor(_campaign_dir, _state, command, output_dir):
                calls.append(command)
                output_dir.mkdir(parents=True, exist_ok=True)

            failed_smoke = {
                "status": "completed",
                "stage": "smoke",
                "metrics": {"n_periods": 5},
                "smoke": {
                    "pass": False,
                    "low_dispersion_warning": False,
                },
                "gate": {"pass": False, "reason": "dispersion_and_schema"},
            }
            with mock.patch.object(
                autoresearch,
                "campaign_evaluate_stage",
                return_value=failed_smoke,
            ):
                completed = autoresearch.campaign_run(
                    campaign_dir,
                    researcher=fake_researcher,
                    executor=fake_executor,
                )

            self.assertEqual(completed["status"], "no_winner")
            self.assertEqual(completed["terminal_reason"], "no_smoke_survivors")
            self.assertEqual(len(calls), 6)
            self.assertTrue(
                all(
                    Path(command[command.index("--output-dir") + 1]).name
                    == "smoke"
                    for command in calls
                )
            )

    def test_fake_end_to_end_campaign_uses_only_v2_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign_dir = Path(tmp) / "fake_campaign"
            campaign_dir.mkdir()
            state = autoresearch.campaign_initial_state("fake_campaign", autoresearch.CAMPAIGN_RUNNER)
            self._write_frozen_development_manifests(campaign_dir, state)
            autoresearch.campaign_atomic_json(campaign_dir / "catalog.json", {"entries": [], "summary": {"entries": 0}})
            autoresearch.campaign_save(campaign_dir, state)
            calls = []

            def proposal(suffix, parent_id=None):
                return self._proposal(suffix, parent_id=parent_id)

            researcher_round = {"value": 0}

            def fake_researcher(_prompt):
                researcher_round["value"] += 1
                if researcher_round["value"] == 1:
                    return [proposal(f"g0-{idx}") for idx in range(3)], {"calls": 1, "output_tokens": 50}
                if researcher_round["value"] == 2:
                    return [proposal(f"g0-{idx}") for idx in range(3, 6)], {"calls": 1, "output_tokens": 50}
                return [proposal("g1-1", "G0_01"), proposal("g1-2", "G0_02")], {"calls": 1, "output_tokens": 50}

            def fake_executor(_campaign_dir, _state, command, output_dir):
                calls.append(command)
                output_dir.mkdir(parents=True, exist_ok=True)

            def fake_evaluate(_output_dir, manifest, stage, **_kwargs):
                metrics = {
                    "n_periods": manifest["date_count"],
                    "annual_return_pct": 20.0,
                    "sharpe": 2.0,
                    "paired_annualized_lift_pct": 5.0,
                    "paired_lift_vs_trail5_pct": 6.0,
                    "sharpe_improvement": 1.0,
                    "tail_improvement_pct": 1.0,
                    "paired_hit_rate_pct": 60.0,
                    "long_paired_lift_pct": 3.0,
                    "short_paired_lift_pct": 2.0,
                    "chronological_block_stability_pct": 100.0,
                    "cagr_pct": 20.0,
                    "max_drawdown_pct": -5.0,
                }
                result = {
                    "status": "completed",
                    "stage": stage,
                    "metrics": metrics,
                    "controls": {"fresh_incumbent": {"cagr_pct": 10.0, "sharpe": 1.0, "max_drawdown_pct": -6.0}, "trail5": {"cagr_pct": 8.0, "sharpe": 0.8, "max_drawdown_pct": -7.0}},
                    "bootstrap_vs_incumbent": {"mean_annualized_delta_pct": 5.0, "upper_90_pct": 6.0, "lower_95_pct": 1.0, "p_not_positive": 0.01},
                    "bootstrap_vs_trail5": {"mean_annualized_delta_pct": 6.0, "upper_90_pct": 7.0, "lower_95_pct": 2.0, "p_not_positive": 0.01},
                    "gate": {"pass": True},
                }
                if stage == "smoke":
                    result["smoke"] = {"pass": True, "low_dispersion_warning": False}
                if stage == "full":
                    result["gate"]["label"] = "development_significant"
                return result

            with mock.patch.object(autoresearch, "campaign_evaluate_stage", side_effect=fake_evaluate):
                completed = autoresearch.campaign_run(
                    campaign_dir,
                    researcher=fake_researcher,
                    executor=fake_executor,
                )

            self.assertEqual(completed["status"], "awaiting_validation_approval")
            self.assertEqual(len(completed["candidates"]), 8)
            self.assertTrue(calls)
            full_calls = 0
            for command in calls:
                self.assertEqual(Path(command[1]).resolve(), autoresearch.CAMPAIGN_RUNNER.resolve())
                self.assertNotIn("--paired-control", command)
                self.assertNotIn("--paired-control-cache", command)
                self.assertEqual(command[command.index("--temperature") + 1], "1")
                self.assertEqual(command[command.index("--thinking-mode") + 1], "disabled")
                self.assertNotIn("phase2", " ".join(command).lower())
                output_dir = Path(command[command.index("--output-dir") + 1])
                full_calls += int(output_dir.name == "full")
            self.assertEqual(full_calls, autoresearch.CAMPAIGN_MAX_FULL_RUNS)


if __name__ == "__main__":
    unittest.main()
