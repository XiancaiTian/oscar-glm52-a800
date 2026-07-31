from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).parent
MODULE_PATH = SCRIPT_DIR / "benchmark_topk_prefill_sort.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_topk_prefill_sort", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkTopkPrefillSortTest(unittest.TestCase):
    def test_parse_args_defaults_to_final_32k_chunk(self) -> None:
        args = BENCHMARK.parse_args(["--output", "result.json"])

        self.assertEqual(args.query_tokens, 2048)
        self.assertEqual(args.final_seq_len, 32768)
        self.assertEqual(args.topk, 2048)
        self.assertEqual(args.warmup, 5)
        self.assertEqual(args.repeats, 7)
        self.assertEqual(args.iterations, 20)
        self.assertEqual(args.seed, 42)

    def test_parse_args_rejects_invalid_geometry(self) -> None:
        with self.assertRaises(SystemExit):
            BENCHMARK.parse_args(
                [
                    "--output",
                    "result.json",
                    "--query-tokens",
                    "4096",
                    "--final-seq-len",
                    "2048",
                ]
            )

    def test_make_row_bounds_matches_final_chunk(self) -> None:
        row_starts, row_ends = BENCHMARK.make_row_bounds(
            torch,
            query_tokens=4,
            final_seq_len=16,
            device=torch.device("cpu"),
        )

        self.assertEqual(row_starts.tolist(), [0, 0, 0, 0])
        self.assertEqual(row_ends.tolist(), [13, 14, 15, 16])
        self.assertEqual(row_starts.dtype, torch.int32)
        self.assertEqual(row_ends.dtype, torch.int32)

    def test_summarize_samples_reports_per_call_values(self) -> None:
        summary = BENCHMARK.summarize_samples([4.0, 6.0, 5.0], iterations=2)

        self.assertEqual(summary["samples_ms"], [2.0, 3.0, 2.5])
        self.assertEqual(summary["median_ms"], 2.5)
        self.assertEqual(summary["mean_ms"], 2.5)
        self.assertEqual(summary["min_ms"], 2.0)
        self.assertEqual(summary["max_ms"], 3.0)

    def test_validate_index_outputs_accepts_same_sorted_sets(self) -> None:
        raw = torch.tensor([[3, 0, 2, 1], [7, 5, 6, 4]], dtype=torch.int32)
        sorted_indices = torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            dtype=torch.int32,
        )

        result = BENCHMARK.validate_index_outputs(torch, raw, sorted_indices)

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["same_selected_set_per_row"])
        self.assertTrue(result["sorted_indices_monotonic_per_row"])
        self.assertEqual(result["invalid_index_count"], 0)

    def test_validate_index_outputs_rejects_changed_set(self) -> None:
        raw = torch.tensor([[3, 0, 2, 1]], dtype=torch.int32)
        sorted_indices = torch.tensor([[0, 1, 2, 4]], dtype=torch.int32)

        with self.assertRaisesRegex(AssertionError, "selected index set changed"):
            BENCHMARK.validate_index_outputs(torch, raw, sorted_indices)


if __name__ == "__main__":
    unittest.main()
