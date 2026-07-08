import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_llm_reversal_controlled_replay_offline as replay


def _candidate_grid() -> pd.DataFrame:
    rows = []
    for date in ["2020-01-03", "2020-01-10"]:
        for side, tickers in (("LONG", ["AAA", "BBB"]), ("SHORT", ["CCC", "DDD"])):
            for idx, ticker in enumerate(tickers, start=1):
                rows.append(
                    {
                        "rebalance_date": date,
                        "lookback_date": "2019-12-27",
                        "entry_date": date,
                        "exit_date": "2020-01-17",
                        "ticker": ticker,
                        "pool_side": side,
                        "side_rank_by_trail": idx,
                        "trail_ret": -0.05 if side == "LONG" else 0.05,
                        "deterministic_signal": 0.05 if side == "LONG" else -0.05,
                        "fwd_c2c": 0.02 if ticker in {"AAA", "CCC"} else -0.01,
                        "fwd_o2o": 0.01 if ticker in {"AAA", "CCC"} else -0.02,
                        "feature_MRL": float(100 - idx),
                        "feature_MRS": float(idx),
                    }
                )
    return pd.DataFrame(rows)


def _write_signal_checks(path: Path, candidate_grid: pd.DataFrame, score: int = 80) -> None:
    artifact = candidate_grid[["rebalance_date", "ticker", "pool_side"]].copy()
    artifact = artifact.rename(columns={"rebalance_date": "date"})
    artifact["score_provider"] = "llm"
    artifact["llm_score"] = score
    artifact["signal"] = np.where(artifact["pool_side"] == "LONG", score, -score)
    artifact.to_csv(path, index=False)


def _write_direct_cache(
    path: Path,
    candidate_grid: pd.DataFrame,
    score: int = 55,
    pool_per_side: int = 2,
    prompt_style: str = replay.DIRECT_SCORE_PROMPT_STYLE,
    prompt_version: str = replay.DIRECT_SCORE_PROMPT_VERSION,
) -> None:
    signals = {}
    grid = candidate_grid.copy()
    grid["rebalance_date"] = pd.to_datetime(grid["rebalance_date"]).dt.strftime("%Y-%m-%d")
    for date, date_rows in grid.groupby("rebalance_date"):
        long_rows = date_rows[date_rows["pool_side"] == "LONG"]
        short_rows = date_rows[date_rows["pool_side"] == "SHORT"]
        signals[date] = {
            "rebalance_date": date,
            "prompt_style": prompt_style,
            "prompt_version": prompt_version,
            "pool_per_side": pool_per_side,
            "long_scores": [{"ticker": ticker, "score": score} for ticker in long_rows["ticker"]],
            "short_scores": [{"ticker": ticker, "score": score} for ticker in short_rows["ticker"]],
        }
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "artifact_type": "llm_reversal_signal_v1_scores",
                    "prompt_style": prompt_style,
                    "prompt_version": prompt_version,
                },
                "signals": signals,
            }
        )
    )


class ControlledReplayOfflineTests(unittest.TestCase):
    def test_candidate_pool_hash_excludes_features_but_feature_hash_includes_them(self):
        base = _candidate_grid()
        changed = base.copy()
        changed["feature_MRL"] = changed["feature_MRL"] + 1.0

        self.assertEqual(replay.candidate_pool_hash(base), replay.candidate_pool_hash(changed))
        self.assertNotEqual(
            replay.feature_payload_hash(base, ["feature_MRL", "feature_MRS"]),
            replay.feature_payload_hash(changed, ["feature_MRL", "feature_MRS"]),
        )

    def test_feature_payload_hash_is_stable_after_csv_roundtrip(self):
        grid = _candidate_grid()
        expected = replay.feature_payload_hash(grid, ["feature_MRL", "feature_MRS"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "grid.csv"
            grid.to_csv(path, index=False)
            restored = pd.read_csv(path)

        self.assertEqual(expected, replay.feature_payload_hash(restored, ["feature_MRL", "feature_MRS"]))

    def test_direct_score_150_cache_is_rejected_for_60_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "direct.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "signals": {
                            "2020-01-03": {
                                "pool_per_side": 150,
                                "long_scores": [{"ticker": "AAA", "score": 1}],
                                "short_scores": [{"ticker": "CCC", "score": 1}],
                            }
                        }
                    }
                )
            )

            audit = replay.inspect_direct_score_cache(cache_path, expected_pool_per_side=60)

        self.assertEqual(audit["status"], "missing_offline_score")
        self.assertTrue(audit["rejected_cache"])
        self.assertIn("150", audit["reason"])

    def test_signal_check_artifact_must_match_frozen_grid(self):
        grid = _candidate_grid()
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "signal_checks.csv"
            bad = grid[["rebalance_date", "ticker", "pool_side"]].copy()
            bad = bad.rename(columns={"rebalance_date": "date"})
            bad.loc[0, "ticker"] = "ZZZ"
            bad["llm_score"] = 50
            bad.to_csv(artifact_path, index=False)

            scores, audit = replay.load_signal_checks_artifact(
                artifact_path,
                replay.SCORER_RAW_CONFIDENCE,
                grid,
            )

        self.assertTrue(scores.empty)
        self.assertEqual(audit["status"], "incompatible_cache_or_artifact")
        self.assertEqual(audit["missing_rows"], 1)
        self.assertEqual(audit["unexpected_rows"], 1)

    def test_direct_score_cache_loads_only_when_it_matches_frozen_grid(self):
        grid = _candidate_grid()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "direct.json"
            _write_direct_cache(cache_path, grid, score=55)

            scores, audit = replay.load_direct_score_cache(
                cache_path,
                candidate_grid=grid,
                expected_pool_per_side=2,
            )

        self.assertFalse(scores.empty)
        self.assertEqual(audit["status"], "valid")
        self.assertEqual(len(scores), len(grid))
        self.assertIn("v1_direct_score", scores.columns)

    def test_direct_score_cache_wrong_prompt_is_rejected(self):
        grid = _candidate_grid()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "direct.json"
            _write_direct_cache(cache_path, grid, prompt_style="raw_confidence")

            scores, audit = replay.load_direct_score_cache(
                cache_path,
                candidate_grid=grid,
                expected_pool_per_side=2,
            )

        self.assertTrue(scores.empty)
        self.assertEqual(audit["status"], "incompatible_cache_or_artifact")
        self.assertIn("wrong prompt_style", audit["reason"])

    def test_offline_runner_has_no_online_imports(self):
        source = Path(replay.__file__).read_text()
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        forbidden = {"httpx", "technical_llm", "mean_reversion_llm", "mean_reversion_llm_v1"}
        self.assertFalse(imported & forbidden)

    def test_score_matrix_uses_only_local_artifacts_and_marks_direct_missing(self):
        grid = _candidate_grid()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_path = tmp / "raw.csv"
            rank_path = tmp / "rank.csv"
            direct_path = tmp / "missing_direct.json"
            _write_signal_checks(raw_path, grid, score=70)
            _write_signal_checks(rank_path, grid, score=60)

            matrix, coverage = replay.build_score_matrix(
                grid,
                raw_artifact=raw_path,
                rank_artifact=rank_path,
                direct_cache=direct_path,
                pool_per_side=2,
            )

        self.assertEqual(len(matrix), len(grid))
        self.assertIn("deterministic_mr_score", matrix.columns)
        self.assertIn("v1_raw_confidence_score", matrix.columns)
        self.assertIn("v1_rank_confidence_score", matrix.columns)
        self.assertIn("v1_direct_score", matrix.columns)
        self.assertEqual(coverage["summary"]["status"], "PARTIAL_OFFLINE_REPLAY")
        self.assertEqual(coverage["summary"]["llm_calls_attempted"], 0)

    def test_score_matrix_becomes_valid_when_all_local_scorers_match(self):
        grid = _candidate_grid()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_path = tmp / "raw.csv"
            rank_path = tmp / "rank.csv"
            direct_path = tmp / "direct.json"
            _write_signal_checks(raw_path, grid, score=70)
            _write_signal_checks(rank_path, grid, score=60)
            _write_direct_cache(direct_path, grid, score=55)

            matrix, coverage = replay.build_score_matrix(
                grid,
                raw_artifact=raw_path,
                rank_artifact=rank_path,
                direct_cache=direct_path,
                pool_per_side=2,
            )

        self.assertEqual(len(matrix), len(grid))
        self.assertEqual(coverage["summary"]["status"], "VALID_CONTROLLED_REPLAY")
        self.assertEqual(int(matrix["v1_direct_score"].isna().sum()), 0)

    def test_performance_scorecard_selects_fixed_book_from_scores(self):
        grid = _candidate_grid()
        score_matrix = grid[["rebalance_date", "ticker", "pool_side"]].copy()
        score_matrix["v1_raw_confidence_score"] = np.where(
            score_matrix["ticker"].isin(["BBB", "DDD"]),
            90,
            10,
        )

        perf = replay.build_performance_for_scorer(
            candidate_grid=grid,
            score_matrix=score_matrix,
            scorer=replay.SCORER_RAW_CONFIDENCE,
            positions_per_side=1,
            cost_bps_levels=(0.0, 2.5),
            periods_per_year=52.0,
        )

        self.assertEqual(len(perf.weekly_returns), 2)
        self.assertIn("ls_c2c_2p5", perf.weekly_returns.columns)
        first_date = perf.portfolio_selections[perf.portfolio_selections["rebalance_date"] == "2020-01-03"]
        self.assertEqual(set(first_date["ticker"]), {"BBB", "DDD"})
        self.assertFalse(perf.strategy_summary.empty)

    def test_save_performance_scorecard_marks_partial_status(self):
        grid = _candidate_grid()
        score_matrix = grid[["rebalance_date", "ticker", "pool_side"]].copy()
        score_matrix["deterministic_mr_score"] = grid["deterministic_signal"]
        perf = replay.build_performance_for_scorer(
            candidate_grid=grid,
            score_matrix=score_matrix,
            scorer=replay.SCORER_DETERMINISTIC,
            positions_per_side=1,
            cost_bps_levels=(0.0,),
            periods_per_year=52.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            scorecard = replay.save_controlled_performance_outputs(
                Path(tmpdir),
                {replay.SCORER_DETERMINISTIC: perf},
                replay_status="PARTIAL_OFFLINE_REPLAY",
            )

        self.assertEqual(scorecard["replay_status"].iloc[0], "PARTIAL_OFFLINE_REPLAY")
        self.assertFalse(bool(scorecard["all_required_scorers_valid"].iloc[0]))

    def test_reuse_candidate_grid_refreshes_coverage_without_data_load(self):
        grid = _candidate_grid()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_dir = tmp / "out"
            output_dir.mkdir()
            raw_path = tmp / "raw.csv"
            rank_path = tmp / "rank.csv"
            grid.to_csv(output_dir / "candidate_grid.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "rebalance_date": "2020-01-03",
                        "audit_status": "ok",
                        "candidate_pool_hash": replay.candidate_pool_hash(grid),
                    }
                ]
            ).to_csv(output_dir / "candidate_pool_audit.csv", index=False)
            _write_signal_checks(raw_path, grid, score=70)
            _write_signal_checks(rank_path, grid, score=60)

            args = replay.parse_args(
                [
                    "--reuse-candidate-grid",
                    "--output-dir",
                    str(output_dir),
                    "--raw-confidence-artifact",
                    str(raw_path),
                    "--rank-confidence-artifact",
                    str(rank_path),
                    "--direct-score-cache",
                    str(tmp / "missing_direct.json"),
                    "--pool-per-side",
                    "2",
                    "--positions-per-side",
                    "1",
                ]
            )
            coverage = replay.run_offline_replay(args)

        self.assertEqual(coverage["summary"]["status"], "PARTIAL_OFFLINE_REPLAY")
        self.assertEqual(coverage["candidate_grid"]["row_count"], len(grid))


if __name__ == "__main__":
    unittest.main()
