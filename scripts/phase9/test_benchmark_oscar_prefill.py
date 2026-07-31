from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "benchmark_oscar_prefill.py"
SPEC = importlib.util.spec_from_file_location("benchmark_oscar_prefill", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkOscarPrefillTest(unittest.TestCase):
    def test_selected_tokens_are_causal_and_tail_is_invalid(self) -> None:
        selected = BENCHMARK.make_selected_tokens(
            torch,
            seed=42,
            device=torch.device("cpu"),
            seq_len=1024,
        )

        self.assertEqual(tuple(selected.shape), (1024, 2048))
        for position in (0, 1, 63, 64, 767, 1023):
            valid = selected[position, : position + 1]
            invalid = selected[position, position + 1 :]
            self.assertEqual(
                sorted(valid.tolist()),
                list(range(position + 1)),
            )
            self.assertTrue((invalid == -1).all())

    def test_config_matrix_preserves_formal_baseline(self) -> None:
        configs = BENCHMARK.build_configs(1024)
        names = [config["name"] for config in configs]
        self.assertEqual(names[0], BENCHMARK.BASELINE_CONFIG)
        self.assertEqual(
            configs[0],
            {
                "name": "full_topk_split16",
                "topk_width": 2048,
                "num_splits": 16,
            },
        )
        self.assertEqual(
            [config["num_splits"] for config in configs[2:]],
            [16, 8, 4, 2, 1],
        )

    def test_full_width_shape_uses_only_unique_configs(self) -> None:
        self.assertEqual(
            BENCHMARK.build_configs(2048),
            [
                {
                    "name": "full_topk_split16",
                    "topk_width": 2048,
                    "num_splits": 16,
                },
                {
                    "name": "full_topk_split1",
                    "topk_width": 2048,
                    "num_splits": 1,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
