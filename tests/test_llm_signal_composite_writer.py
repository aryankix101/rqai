import unittest
import tempfile
from pathlib import Path

import pandas as pd

import llm_signal_composite_writer as writer
from data_source_config import DATASET_VERSION
from llm_signal_composite_writer import (
    _add_deterministic_mean_reversion_scores,
    transform_signal_rankings,
    write_llm_signal_composites,
)


class LlmSignalCompositeWriterTests(unittest.TestCase):
    def test_transform_signal_rankings_builds_signed_score_variants(self):
        signals = {
            "2024-01-05": {
                "long_rankings": [
                    {"ticker": "AAA", "rank": 1, "confidence_score": 80},
                    {"ticker": "BBB", "rank": 2, "confidence_score": 50},
                ],
                "short_rankings": [
                    {"ticker": "CCC", "rank": 1, "confidence_score": 90},
                    {"ticker": "DDD", "rank": 2, "confidence_score": 60},
                ],
            }
        }

        transformed = transform_signal_rankings(signals)

        self.assertEqual(transformed["AAA"]["2024-01-05"]["llm_raw_confidence_score"], 80.0)
        self.assertEqual(transformed["BBB"]["2024-01-05"]["llm_rank_score"], 50.0)
        self.assertEqual(transformed["AAA"]["2024-01-05"]["llm_rank_confidence_score"], 80.0)
        self.assertEqual(transformed["CCC"]["2024-01-05"]["llm_raw_confidence_score"], -90.0)
        self.assertEqual(transformed["CCC"]["2024-01-05"]["llm_rank_score"], -100.0)
        self.assertEqual(transformed["DDD"]["2024-01-05"]["llm_rank_confidence_score"], -30.0)

    def test_deterministic_mean_reversion_scores_use_cross_sectional_rank(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        frames = {
            "AAA": pd.DataFrame({"date": dates, "adj_close": [10, 10, 10, 10, 10, 8]}),
            "BBB": pd.DataFrame({"date": dates, "adj_close": [10, 10, 10, 10, 10, 10]}),
            "CCC": pd.DataFrame({"date": dates, "adj_close": [10, 10, 10, 10, 10, 12]}),
        }

        _add_deterministic_mean_reversion_scores(frames)

        last_scores = {
            ticker: float(df.iloc[-1]["mean_reversion_rank_score"])
            for ticker, df in frames.items()
        }
        self.assertEqual(last_scores["AAA"], 100.0)
        self.assertEqual(last_scores["BBB"], 0.0)
        self.assertEqual(last_scores["CCC"], -100.0)
        self.assertAlmostEqual(frames["AAA"].iloc[-1]["mean_reversion_raw_score"], 20.0)
        self.assertAlmostEqual(frames["CCC"].iloc[-1]["mean_reversion_raw_score"], -20.0)

    def test_writer_computes_deterministic_baseline_before_window_trim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_dir = tmp_path / "source"
            output_dir = tmp_path / "out"
            source_dir.mkdir()

            dates = pd.date_range("2024-01-01", periods=6, freq="B")
            for ticker, last_price in {"AAA": 8.0, "BBB": 12.0}.items():
                pd.DataFrame(
                    {
                        "date": dates,
                        "adj_close": [10.0, 10.0, 10.0, 10.0, 10.0, last_price],
                        "direction_score": [0.0] * len(dates),
                        "magnitude_score": [0.0] * len(dates),
                    }
                ).to_csv(source_dir / f"composite_score_{ticker}.csv", index=False)

            cache_path = tmp_path / "signal_cache.json"
            cache_path.write_text(
                """{
  "metadata": {
    "artifact_type": "llm_mean_reversion_signal_rankings",
    "schema_version": 1,
    "source_dataset": "__DATASET__"
  },
  "signals": {
    "2024-01-08": {
      "long_rankings": [{"ticker": "AAA", "rank": 1, "confidence_score": 80}],
      "short_rankings": [{"ticker": "BBB", "rank": 1, "confidence_score": 70}]
    }
  }
}""".replace("__DATASET__", DATASET_VERSION),
                encoding="utf-8",
            )

            original_dir = writer.COMPOSITE_SCORES_DIR
            try:
                writer.COMPOSITE_SCORES_DIR = source_dir
                write_llm_signal_composites(str(cache_path), str(output_dir))
            finally:
                writer.COMPOSITE_SCORES_DIR = original_dir

            aaa = pd.read_csv(output_dir / "composite_score_AAA.csv")
            bbb = pd.read_csv(output_dir / "composite_score_BBB.csv")
            self.assertEqual(len(aaa), 1)
            self.assertAlmostEqual(float(aaa.iloc[0]["mean_reversion_raw_score"]), 20.0)
            self.assertAlmostEqual(float(bbb.iloc[0]["mean_reversion_raw_score"]), -20.0)
            self.assertAlmostEqual(float(aaa.iloc[0]["mean_reversion_rank_score"]), 100.0)
            self.assertAlmostEqual(float(bbb.iloc[0]["mean_reversion_rank_score"]), -100.0)


if __name__ == "__main__":
    unittest.main()
