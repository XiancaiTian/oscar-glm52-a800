from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "benchmark_oscar_prefill.py"
SPEC = importlib.util.spec_from_file_location("benchmark_oscar_prefill", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkOscarPrefillTest(unittest.TestCase):
    def test_parse_args_can_request_sorted_selected_indices(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "benchmark_oscar_prefill.py",
                "--output",
                "result.json",
                "--include-sorted-selected-indices",
            ],
        ):
            args = BENCHMARK.parse_args()

        self.assertTrue(args.include_sorted_selected_indices)

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

    def test_selected_tokens_support_later_prefill_chunk(self) -> None:
        query_tokens = 4
        final_seq_len = 4096
        selected = BENCHMARK.make_selected_tokens(
            torch,
            seed=42,
            device=torch.device("cpu"),
            seq_len=query_tokens,
            final_seq_len=final_seq_len,
        )

        self.assertEqual(tuple(selected.shape), (query_tokens, 2048))
        for row, query_position in zip(
            selected,
            range(final_seq_len - query_tokens, final_seq_len),
            strict=True,
        ):
            self.assertFalse((row == -1).any())
            self.assertEqual(row.unique().numel(), 2048)
            self.assertTrue((row <= query_position).all())
        self.assertTrue((selected > query_tokens).any())
        coverage = BENCHMARK.summarize_selected_tiles(
            torch,
            selected,
            final_seq_len=final_seq_len,
        )
        self.assertEqual(coverage["valid_selected_tokens"], query_tokens * 2048)
        self.assertEqual(
            coverage["total_tiles"],
            query_tokens * 2048 // 16,
        )
        self.assertGreater(coverage["all_history_tiles"], 0)

    def test_selected_tile_coverage_distinguishes_history_opportunities(self) -> None:
        selected = torch.tensor(
            [
                list(range(16)),
                list(range(100, 116)),
                list(range(8)) + list(range(100, 108)),
                [0, 1] + [-1] * 14,
            ],
            dtype=torch.int32,
        )

        coverage = BENCHMARK.summarize_selected_tiles(
            torch,
            selected,
            final_seq_len=1000,
        )

        self.assertEqual(coverage["active_tiles"], 4)
        self.assertEqual(coverage["tiles_with_history"], 2)
        self.assertEqual(coverage["tiles_without_history"], 2)
        self.assertEqual(coverage["history_only_tiles"], 1)
        self.assertEqual(coverage["mixed_precision_tiles"], 1)
        self.assertEqual(coverage["all_bf16_tiles"], 1)

    def test_sorted_selected_tokens_match_native_prefill_shortcut(self) -> None:
        selected = BENCHMARK.make_selected_tokens(
            torch,
            seed=42,
            device=torch.device("cpu"),
            seq_len=1024,
        )
        query_positions = torch.arange(1024, dtype=torch.int32)

        sorted_selected = BENCHMARK.sort_selected_tokens_like_prefill_topk(
            torch,
            selected,
            query_positions=query_positions,
        )

        self.assertTrue(torch.equal(sorted_selected, selected))

    def test_sorted_selected_tokens_reorder_only_long_rows(self) -> None:
        selected = torch.tensor(
            [
                [3, 0, 2, 1, -1, -1],
                [9, 4, 7, 5, 8, 6],
            ],
            dtype=torch.int32,
        )
        query_positions = torch.tensor([3, 9], dtype=torch.int32)

        sorted_selected = BENCHMARK.sort_selected_tokens_like_prefill_topk(
            torch,
            selected,
            query_positions=query_positions,
        )

        self.assertEqual(sorted_selected[0].tolist(), selected[0].tolist())
        self.assertEqual(sorted_selected[1].tolist(), [4, 5, 6, 7, 8, 9])
        self.assertEqual(
            sorted(selected[1].tolist()),
            sorted_selected[1].tolist(),
        )

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

    def test_sorted_config_adds_only_matching_split1_candidate(self) -> None:
        configs = BENCHMARK.build_configs(
            2048,
            include_sorted_selected_indices=True,
        )

        self.assertEqual(
            configs,
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
                {
                    "name": "full_topk_split1_sorted_indices",
                    "topk_width": 2048,
                    "num_splits": 1,
                    "selected_order": "token_index",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
