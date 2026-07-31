from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


SCRIPT_DIR = Path(__file__).parent
MODULE_PATH = SCRIPT_DIR / "benchmark_oscar_rotation.py"
SPEC = importlib.util.spec_from_file_location("benchmark_oscar_rotation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkOscarRotationTest(unittest.TestCase):
    def test_parse_args_defaults_to_formal_prefill_geometry(self) -> None:
        args = BENCHMARK.parse_args(["--output", "result.json"])

        self.assertEqual(args.rows, 2048)
        self.assertEqual(args.latent_rank, 512)
        self.assertEqual(args.rotation_layer, "0")
        self.assertEqual(args.accuracy_layers, ["0", "25", "51", "77"])
        self.assertEqual(args.warmup, 20)
        self.assertEqual(args.repeats, 7)
        self.assertEqual(args.iterations, 20)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.atol, 0.35)
        self.assertEqual(args.rtol, 0.02)
        self.assertEqual(args.clip_ratio, 0.96)

    def test_parse_args_rejects_invalid_geometry(self) -> None:
        with self.assertRaises(SystemExit):
            BENCHMARK.parse_args(["--output", "result.json", "--rows", "0"])
        with self.assertRaises(SystemExit):
            BENCHMARK.parse_args(["--output", "result.json", "--latent-rank", "510"])

    def test_rotation_kernel_parameters_match_production(self) -> None:
        self.assertEqual(
            BENCHMARK.rotation_kernel_parameters(),
            {
                "block_m": 16,
                "block_n": 64,
                "block_k": 32,
                "num_warps": 4,
                "num_stages": 2,
            },
        )

    def test_benchmark_kernel_preserves_production_ieee_contract(self) -> None:
        project_root = SCRIPT_DIR.parents[1]
        production_path = (
            project_root
            / "glm52_oscar_vllm/vllm/v1/attention/ops/triton_oscar_mla_store.py"
        )
        production = production_path.read_text(encoding="utf-8")
        benchmark = MODULE_PATH.read_text(encoding="utf-8")

        for fragment in (
            "block_m = 16",
            "block_n = 64",
            "block_k = 32",
            "num_warps=4",
            "num_stages=2",
            'input_precision="ieee"',
        ):
            self.assertIn(fragment, production)
        self.assertEqual(benchmark.count('input_precision="ieee"'), 1)
        self.assertEqual(benchmark.count('input_precision="tf32"'), 1)

    def test_summarize_samples_reports_per_call_values(self) -> None:
        summary = BENCHMARK.summarize_samples([8.0, 12.0, 10.0], iterations=4)

        self.assertEqual(summary["samples_ms"], [2.0, 3.0, 2.5])
        self.assertEqual(summary["median_ms"], 2.5)
        self.assertEqual(summary["mean_ms"], 2.5)
        self.assertEqual(summary["min_ms"], 2.0)
        self.assertEqual(summary["max_ms"], 3.0)

    def test_validate_rotation_outputs_accepts_values_within_tolerance(self) -> None:
        ieee = torch.tensor([[1.0, 2.0], [-1.0, 0.0]])
        candidate = torch.tensor([[1.1, 1.9], [-0.95, 0.0]])

        result = BENCHMARK.validate_rotation_outputs(
            torch,
            ieee,
            candidate,
            atol=0.2,
            rtol=0.0,
        )

        self.assertEqual(result["status"], "passed")
        self.assertAlmostEqual(result["max_abs_error"], 0.1, places=6)
        self.assertGreater(result["mean_abs_error"], 0.0)
        self.assertEqual(result["mismatched_values"], 0)

    def test_validate_rotation_outputs_rejects_values_outside_tolerance(self) -> None:
        ieee = torch.tensor([[1.0, 2.0]])
        candidate = torch.tensor([[2.0, 2.0]])

        with self.assertRaisesRegex(AssertionError, "rotation candidate"):
            BENCHMARK.validate_rotation_outputs(
                torch,
                ieee,
                candidate,
                atol=0.1,
                rtol=0.0,
            )

    def test_compare_timings_reports_tf32_speedup(self) -> None:
        result = BENCHMARK.compare_timings(
            {"cuda": {"median_ms": 4.0}, "wall": {"median_ms": 5.0}},
            {"cuda": {"median_ms": 2.0}, "wall": {"median_ms": 2.5}},
        )

        self.assertEqual(result["median_cuda_ms"], -2.0)
        self.assertEqual(result["median_cuda_percent"], -50.0)
        self.assertEqual(result["cuda_speedup"], 2.0)
        self.assertEqual(result["median_wall_ms"], -2.5)


if __name__ == "__main__":
    unittest.main()
