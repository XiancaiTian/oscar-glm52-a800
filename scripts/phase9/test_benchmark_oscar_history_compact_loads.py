from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "benchmark_oscar_history_compact_loads.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_oscar_history_compact_loads",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkOscarHistoryCompactLoadsTest(unittest.TestCase):
    def test_variant_matrix_changes_only_compact_load_flag(self) -> None:
        self.assertEqual(
            BENCHMARK.VARIANTS,
            [
                {
                    "name": "history_h8_t16_w8_reference",
                    "block_h": 8,
                    "block_t": 16,
                    "num_warps": 8,
                    "manual_history_value_reduce": False,
                    "compact_history_loads": False,
                },
                {
                    "name": "history_h8_t16_w8_compact_loads_candidate",
                    "block_h": 8,
                    "block_t": 16,
                    "num_warps": 8,
                    "manual_history_value_reduce": False,
                    "compact_history_loads": True,
                },
            ],
        )

    def test_speed_gate_requires_strict_candidate_improvement(self) -> None:
        self.assertTrue(BENCHMARK.promotion_eligible(9.9, [10.0]))
        self.assertFalse(BENCHMARK.promotion_eligible(10.0, [10.0]))
        self.assertFalse(BENCHMARK.promotion_eligible(10.1, [10.0]))

    def test_shared_launcher_forwards_compact_flag(self) -> None:
        source = inspect.getsource(BENCHMARK.history_bench.launch_history)

        self.assertIn(
            'compact_history_loads=variant.get("compact_history_loads", False)',
            source,
        )


if __name__ == "__main__":
    unittest.main()
