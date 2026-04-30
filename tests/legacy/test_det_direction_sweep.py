import unittest
from types import SimpleNamespace

from archive.legacy_experiments import det_direction_sweep as dds


class DetDirectionSweepTests(unittest.TestCase):
    def test_expand_specs_dedupes_raw_pool_threshold_variants(self):
        args = SimpleNamespace(
            forward_days=10,
            forward_days_list=[10],
            review_modes=["adaptive"],
            static_review_days=None,
            hysteresis_modes=["combined"],
            weighting_modes=["equal"],
            top_ns=[30],
            long_retain_buffers=[20],
            short_retain_buffers=[5],
            replacement_gaps=[5],
            review_normal_days=[20],
            review_grind_up_days=[60],
            review_slow_bleed_days=[20],
            review_squeeze_days=[5],
            review_crisis_days=[5],
            review_unknown_days=[20],
            candidate_pool_modes=["raw", "llm_mirror"],
            direction_thresholds=[5.0, 8.0],
            top_n_magnitudes=[60, 100],
            max_runs=None,
        )

        specs = dds.expand_specs(args)
        raw_specs = [spec for spec in specs if spec.candidate_pool_mode == "raw"]
        mirror_specs = [spec for spec in specs if spec.candidate_pool_mode == "llm_mirror"]

        self.assertEqual(len(raw_specs), 1)
        self.assertEqual(len(mirror_specs), 4)

    def test_build_variant_label_includes_top_n_and_pool_details(self):
        spec = dds.SweepSpec(
            forward_days=15,
            review_mode="adaptive",
            static_review_days=None,
            hysteresis_mode="combined",
            weighting_mode="rank",
            top_n=100,
            bottom_n=100,
            long_retain_buffer=20,
            short_retain_buffer=5,
            replacement_gap=5,
            review_normal_days=20,
            review_grind_up_days=60,
            review_slow_bleed_days=20,
            review_squeeze_days=5,
            review_crisis_days=5,
            review_unknown_days=20,
            candidate_pool_mode="llm_mirror",
            direction_threshold=8.0,
            top_n_magnitude=200,
        )
        label = dds.build_variant_label("det_dir_sweep", spec)
        self.assertIn("fd_15", label)
        self.assertIn("top_100", label)
        self.assertIn("pool_llm_mirror", label)
        self.assertIn("m_200", label)
        self.assertIn("d_8p0", label)

    def test_expand_specs_supports_multiple_horizons(self):
        args = SimpleNamespace(
            forward_days=10,
            forward_days_list=[10, 15],
            review_modes=["adaptive"],
            static_review_days=None,
            hysteresis_modes=["combined"],
            weighting_modes=["equal"],
            top_ns=[30],
            long_retain_buffers=[20],
            short_retain_buffers=[5],
            replacement_gaps=[5],
            review_normal_days=[20],
            review_grind_up_days=[60],
            review_slow_bleed_days=[20],
            review_squeeze_days=[5],
            review_crisis_days=[5],
            review_unknown_days=[20],
            candidate_pool_modes=["raw"],
            direction_thresholds=[5.0],
            top_n_magnitudes=[60],
            max_runs=None,
        )
        specs = dds.expand_specs(args)
        self.assertEqual({spec.forward_days for spec in specs}, {10, 15})
        self.assertEqual(len(specs), 2)


if __name__ == "__main__":
    unittest.main()
