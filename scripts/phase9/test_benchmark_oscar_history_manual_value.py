from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "benchmark_oscar_history_manual_value.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_oscar_history_manual_value",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkOscarHistoryManualValueTest(unittest.TestCase):
    def test_default_shape_matches_final_32k_prefill_chunk(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["benchmark_oscar_history_manual_value.py", "--output", "result.json"],
        ):
            args = BENCHMARK.parse_args()

        self.assertEqual(args.final_seq_len, 32_768)
        self.assertEqual(args.query_tokens, 2_048)
        self.assertEqual(args.warmup, 2)
        self.assertEqual(args.repeats, 5)
        self.assertEqual(args.iterations, 1)

    def test_variant_matrix_freezes_reference_and_candidate(self) -> None:
        self.assertEqual(
            BENCHMARK.VARIANTS,
            [
                {
                    "name": "history_h8_t16_w8_dot_reference",
                    "block_h": 8,
                    "block_t": 16,
                    "num_warps": 8,
                    "manual_history_value_reduce": False,
                },
                {
                    "name": "history_h4_t8_w4_manual_candidate",
                    "block_h": 4,
                    "block_t": 8,
                    "num_warps": 4,
                    "manual_history_value_reduce": True,
                },
            ],
        )

    def test_history_tokens_are_valid_and_exclude_bf16_regions(self) -> None:
        selected = BENCHMARK.make_history_selected_tokens(
            torch,
            query_tokens=4,
            final_seq_len=32_768,
            seed=42,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(selected.shape), (4, 2_048))
        self.assertTrue((selected >= BENCHMARK.PREFIX_TOKENS).all())
        self.assertTrue(
            (selected < 32_768 - BENCHMARK.RECENT_TOKENS).all(),
        )
        for row in selected:
            self.assertEqual(row.unique().numel(), BENCHMARK.TOPK)

    def test_speed_gate_requires_strict_candidate_improvement(self) -> None:
        self.assertTrue(BENCHMARK.candidate_is_faster(9.9, 10.0))
        self.assertFalse(BENCHMARK.candidate_is_faster(10.0, 10.0))
        self.assertFalse(BENCHMARK.candidate_is_faster(10.1, 10.0))

    def test_parse_args_rejects_misaligned_history(self) -> None:
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "benchmark_oscar_history_manual_value.py",
                    "--output",
                    "result.json",
                    "--final-seq-len",
                    "32767",
                ],
            ),
            self.assertRaises(SystemExit),
        ):
            BENCHMARK.parse_args()


if __name__ == "__main__":
    unittest.main()
