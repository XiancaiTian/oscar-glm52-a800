from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RUNTIME_ROOT = Path("/nfs/AE/txc/oscar-glm")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


matrix = load_module(
    "phase9_matrix",
    SCRIPT_DIR / "run_performance_matrix.py",
)
comparison = load_module(
    "phase9_comparison",
    SCRIPT_DIR / "compare_performance.py",
)


class Stage9ToolsTest(unittest.TestCase):
    def test_frozen_matrix_contract(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs/phase9/performance_matrix.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["status"], "ready")
        self.assertEqual(config["matrix"]["input_lengths"], [1024, 8192, 32768])
        self.assertEqual(config["matrix"]["batch_sizes"], [1, 4, 8])
        self.assertEqual(config["matrix"]["rounds"], 3)
        self.assertEqual(
            config["context_128k"]["input_length"]
            + config["context_128k"]["output_length"],
            131072,
        )
        self.assertEqual(matrix.BENCHMARK_HELP_ARGUMENT, "--help=all")

    def test_profiler_config_is_canonical(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "build_profiler_config.py"),
                "--profile-dir",
                "/dev/shm/stage9-profile-test",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["profiler"], "torch")
        self.assertEqual(
            payload["torch_profiler_dir"],
            "/dev/shm/stage9-profile-test",
        )
        self.assertFalse(payload["torch_profiler_with_stack"])
        self.assertTrue(payload["torch_profiler_use_gzip"])

    def test_profiler_table_parser(self) -> None:
        table = """\
Name  Self CPU %  Self CPU  CPU total %  CPU total  CPU time avg  Self CUDA  Self CUDA %  CUDA total  CUDA time avg  # of Calls
kernel_a  0.00%  0us  0.00%  0us  0us  2.000ms  50.00%  2.000ms  2.000ms  1
kernel_b  0.00%  0us  0.00%  0us  0us  1.000ms  25.00%  1.000ms  1.000ms  1
Self CUDA time total: 4.000ms
"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profiler_out_0.txt"
            path.write_text(table, encoding="utf-8")
            self.assertEqual(matrix.parse_cuda_total(path), 4.0)
            rows = comparison.top_cuda_rows(path)
        self.assertEqual([row["name"] for row in rows], ["kernel_a", "kernel_b"])
        self.assertEqual(rows[0]["self_cuda_time_ms"], 2.0)

    def test_server_metrics_parser(self) -> None:
        payload = """\
vllm:num_requests_running{engine="0",model_name="test"} 5.0
vllm:num_requests_waiting{engine="0",model_name="test"} 3.0
vllm:kv_cache_usage_perc{engine="0",model_name="test"} 0.75
vllm:num_preemptions_total{engine="0",model_name="test"} 2.0
"""
        metrics = matrix.parse_server_metrics(payload)
        self.assertEqual(metrics["num_requests_running"], 5.0)
        self.assertEqual(metrics["num_requests_waiting"], 3.0)
        self.assertEqual(metrics["kv_cache_usage_perc"], 0.75)
        self.assertEqual(metrics["num_preemptions_total"], 2.0)
        with self.assertRaisesRegex(ValueError, "missing vLLM server metrics"):
            matrix.parse_server_metrics('vllm:num_requests_running{engine="0"} 1.0\n')

    def test_artifact_paths_are_scoped(self) -> None:
        self.assertTrue(
            matrix.is_scoped_artifact_path(
                RUNTIME_ROOT / "artifacts/phase9/run",
                RUNTIME_ROOT,
            )
        )
        self.assertTrue(
            matrix.is_scoped_artifact_path(
                Path("/dev/shm/oscar-glm-stage9/run"),
                RUNTIME_ROOT,
            )
        )
        self.assertFalse(
            matrix.is_scoped_artifact_path(
                Path("/nfs/AE/zhanghong/workflow/vllm_a"),
                RUNTIME_ROOT,
            )
        )

    def test_benchmark_command_fixes_workload(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(
                variant="candidate",
                server_run_dir=root / "server",
                output_dir=root / "output",
                profile_dir=root / "profile",
                base_url="http://127.0.0.1:18084",
                runtime_project_root=RUNTIME_ROOT,
                include_128k=True,
                formal=True,
            )
            runner = matrix.MatrixRunner(args)
            command = runner.benchmark_command(
                input_length=32768,
                output_length=128,
                batch_size=8,
                num_prompts=24,
                num_warmups=8,
                result_dir=root / "result",
                profile=True,
            )
        joined = " ".join(command)
        for expected in (
            "--random-input-len 32768",
            "--random-output-len 128",
            "--random-range-ratio 0",
            "--num-prompts 24",
            "--num-warmups 8",
            "--max-concurrency 8",
            "--temperature 0",
            "--ignore-eos",
            "--profile",
        ):
            self.assertIn(expected, joined)

    def test_result_validation_requires_exact_completed_workload(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(
                variant="baseline",
                server_run_dir=root / "server",
                output_dir=root / "output",
                profile_dir=root / "profile",
                base_url="http://127.0.0.1:18083",
                runtime_project_root=RUNTIME_ROOT,
                include_128k=False,
                formal=True,
            )
            runner = matrix.MatrixRunner(args)
            result_path = root / "result.json"
            payload = {
                "completed": 4,
                "failed": 0,
                "input_lens": [8192] * 4,
                "output_lens": [128] * 4,
                "errors": [""] * 4,
                "max_concurrent_requests": 4,
                "request_throughput": 1.0,
                "output_throughput": 128.0,
                "total_token_throughput": 8320.0,
                "mean_ttft_ms": 10.0,
                "median_ttft_ms": 9.0,
                "mean_tpot_ms": 2.0,
                "median_tpot_ms": 1.9,
            }
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            validated = runner.validate_result(
                result_path,
                input_length=8192,
                output_length=128,
                num_prompts=4,
                batch_size=4,
            )
            self.assertEqual(validated["completed"], 4)

            payload["input_lens"][0] = 8191
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input length mismatch"):
                runner.validate_result(
                    result_path,
                    input_length=8192,
                    output_length=128,
                    num_prompts=4,
                    batch_size=4,
                )

    def test_profiler_capture_requires_all_rank_tables_and_traces(self) -> None:
        table = """\
Name  Self CPU %  Self CPU  CPU total %  CPU total  CPU time avg  Self CUDA  Self CUDA %  CUDA total  CUDA time avg  # of Calls
kernel_a  0.00%  0us  0.00%  0us  0us  2.000ms  50.00%  2.000ms  2.000ms  1
Self CUDA time total: 4.000ms
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(
                variant="candidate",
                server_run_dir=root / "server",
                output_dir=root / "output",
                profile_dir=root / "profile",
                base_url="http://127.0.0.1:18084",
                runtime_project_root=RUNTIME_ROOT,
                include_128k=True,
                formal=True,
            )
            runner = matrix.MatrixRunner(args)
            runner.profile_dir.mkdir()
            for rank in range(8):
                (runner.profile_dir / f"profiler_out_{rank}.txt").write_text(
                    table,
                    encoding="utf-8",
                )
                (runner.profile_dir / f"rank_{rank}.pt.trace.json.gz").write_bytes(
                    f"rank-{rank}".encode()
                )
            (root / "captured").mkdir()
            captured = runner.capture_profiler(
                before_files=set(),
                started_ns=0,
                output_dir=root / "captured",
            )
        self.assertEqual(len(captured["tables"]), 8)
        self.assertEqual(len(captured["trace_files"]), 8)
        self.assertEqual(captured["kernel_time_ms_critical_rank"], 4.0)

    def test_relative_metric_directions(self) -> None:
        self.assertAlmostEqual(comparison.relative_increase(10.0, 12.0), 0.2)
        self.assertAlmostEqual(comparison.relative_drop(10.0, 8.0), 0.2)


if __name__ == "__main__":
    unittest.main()
