from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "benchmark_oscar_history_maxnreg.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_oscar_history_maxnreg",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkOscarHistoryMaxnregTest(unittest.TestCase):
    def test_variant_matrix_isolates_h4_maxnreg_effect(self) -> None:
        self.assertEqual(
            BENCHMARK.VARIANTS,
            [
                {
                    "name": "history_h8_t16_w8_reference",
                    "block_h": 8,
                    "block_t": 16,
                    "num_warps": 8,
                    "manual_history_value_reduce": False,
                    "maxnreg": None,
                },
                {
                    "name": "history_h4_t16_w8_uncapped_control",
                    "block_h": 4,
                    "block_t": 16,
                    "num_warps": 8,
                    "manual_history_value_reduce": False,
                    "maxnreg": None,
                },
                {
                    "name": "history_h4_t16_w8_maxnreg128_candidate",
                    "block_h": 4,
                    "block_t": 16,
                    "num_warps": 8,
                    "manual_history_value_reduce": False,
                    "maxnreg": 128,
                },
            ],
        )

    def test_promotion_requires_candidate_to_beat_both_controls(self) -> None:
        self.assertTrue(BENCHMARK.promotion_eligible(9.0, [10.0, 9.5]))
        self.assertFalse(BENCHMARK.promotion_eligible(9.5, [10.0, 9.5]))
        self.assertFalse(BENCHMARK.promotion_eligible(9.6, [10.0, 9.5]))

    def test_launch_options_only_emit_explicit_maxnreg(self) -> None:
        self.assertEqual(
            BENCHMARK.launch_options(BENCHMARK.VARIANTS[1]),
            {"num_warps": 8, "num_stages": 1},
        )
        self.assertEqual(
            BENCHMARK.launch_options(BENCHMARK.VARIANTS[2]),
            {"num_warps": 8, "num_stages": 1, "maxnreg": 128},
        )


if __name__ == "__main__":
    unittest.main()
