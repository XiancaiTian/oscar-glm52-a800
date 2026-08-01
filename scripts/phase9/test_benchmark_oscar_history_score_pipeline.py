from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "benchmark_oscar_history_score_pipeline.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_oscar_history_score_pipeline",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkOscarHistoryScorePipelineTest(unittest.TestCase):
    def test_default_shape_matches_final_32k_prefill_chunk(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["benchmark_oscar_history_score_pipeline.py", "--output", "result.json"],
        ):
            args = BENCHMARK.parse_args()

        self.assertEqual(args.final_seq_len, 32_768)
        self.assertEqual(args.query_tokens, 2_048)
        self.assertEqual(args.warmup, 2)
        self.assertEqual(args.repeats, 5)
        self.assertEqual(args.iterations, 1)

    def test_candidate_freezes_strict_pipeline_geometry(self) -> None:
        self.assertEqual(
            BENCHMARK.CANDIDATE,
            {
                "name": "history_score_h4_lse_value_h2_pipeline_candidate",
                "score_block_h": 4,
                "score_block_t": 16,
                "score_num_warps": 4,
                "lse_block_tiles": 128,
                "lse_num_warps": 4,
                "value_block_h": 2,
                "value_block_t": 16,
                "value_block_dv": 128,
                "value_num_warps": 4,
            },
        )

    def test_history_inputs_reuse_frozen_all_history_builder(self) -> None:
        selected = BENCHMARK.manual_bench.make_history_selected_tokens(
            torch,
            query_tokens=4,
            final_seq_len=32_768,
            seed=42,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(selected.shape), (4, 2_048))
        self.assertTrue((selected >= BENCHMARK.PREFIX_TOKENS).all())
        self.assertTrue((selected < 32_768 - BENCHMARK.RECENT_TOKENS).all())

    def test_speed_gate_requires_strict_total_pipeline_improvement(self) -> None:
        self.assertTrue(BENCHMARK.candidate_is_faster(9.9, 10.0))
        self.assertFalse(BENCHMARK.candidate_is_faster(10.0, 10.0))
        self.assertFalse(BENCHMARK.candidate_is_faster(10.1, 10.0))

    def test_parse_args_rejects_misaligned_history(self) -> None:
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "benchmark_oscar_history_score_pipeline.py",
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
