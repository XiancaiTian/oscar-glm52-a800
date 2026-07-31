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
            config["runtime_container"]["image_id"],
            "sha256:0e13b724a2b89f3d698a3a130f13f27d8f8ef1c3acf96fbf38920306f79d50d5",
        )
        self.assertEqual(
            config["runtime_container"]["base_image_id"],
            config["candidate"]["config_digest"],
        )
        self.assertEqual(
            config["context_128k"]["input_length"]
            + config["context_128k"]["output_length"],
            131072,
        )
        self.assertEqual(matrix.BENCHMARK_HELP_ARGUMENT, "--help=all")

    def test_single_cell_probe_is_an_exact_matrix_subset(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs/phase9/performance_matrix.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            matrix.selected_matrix_cells(config, [1024, 1]),
            [(1024, 1)],
        )
        self.assertEqual(len(matrix.selected_matrix_cells(config, None)), 9)
        with self.assertRaisesRegex(ValueError, "outside the frozen matrix"):
            matrix.selected_matrix_cells(config, [2048, 1])

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
        rejected = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "build_profiler_config.py"),
                "--profile-dir",
                "/nfs/AE/zhanghong/workflow/vllm_a/profile",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(rejected.returncode, 1)
        self.assertIn("must be under project artifacts", rejected.stderr)

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

    def test_result_validation_uses_configured_concurrency(self) -> None:
        result = {
            "completed": 3,
            "failed": 0,
            "input_lens": [1024, 1024, 1024],
            "output_lens": [128, 128, 128],
            "errors": ["", "", ""],
            "max_concurrency": 1,
            "max_concurrent_requests": 2,
            "request_throughput": 1.0,
            "output_throughput": 1.0,
            "total_token_throughput": 1.0,
            "mean_ttft_ms": 1.0,
            "median_ttft_ms": 1.0,
            "mean_tpot_ms": 1.0,
            "median_tpot_ms": 1.0,
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            runner = matrix.MatrixRunner.__new__(matrix.MatrixRunner)
            validated = runner.validate_result(
                path,
                input_length=1024,
                output_length=128,
                num_prompts=3,
                batch_size=1,
            )
            self.assertEqual(validated["max_concurrent_requests"], 2)
            result["max_concurrency"] = 2
            path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "configured concurrency mismatch",
            ):
                runner.validate_result(
                    path,
                    input_length=1024,
                    output_length=128,
                    num_prompts=3,
                    batch_size=1,
                )

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
        self.assertFalse(
            comparison.is_scoped_artifact_path(
                Path("/nfs/AE/zhanghong/workflow/vllm_a/result.json"),
                RUNTIME_ROOT,
            )
        )

    def test_comparison_rejects_changed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "performance.json"
            config_path.write_text('{"status": "ready"}\n', encoding="utf-8")
            summary = {
                "performance_config": str(config_path),
                "performance_config_sha256": matrix.sha256_file(config_path),
            }
            path, config = comparison.verified_performance_config(summary)
            self.assertEqual(path, config_path)
            self.assertEqual(config["status"], "ready")
            config_path.write_text('{"status": "changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "performance configuration hash mismatch",
            ):
                comparison.verified_performance_config(summary)

    def test_comparison_requires_identical_runtime_provenance(self) -> None:
        model = {
            "filename_size_mtime_ns_manifest_sha256": "model-manifest",
            "metadata_sha256": {"config.json": "config"},
            "safetensors_count": 141,
        }
        config = {
            "source": {"commit": "source-commit"},
            "model": {
                "filename_size_mtime_ns_manifest_sha256": "model-manifest",
            },
        }

        def summary(variant: str) -> dict:
            return {
                "performance_config_sha256": "performance-config",
                "preflight": {
                    "variant": variant,
                    "main_commit": "main-commit",
                    "source_commit": "source-commit",
                    "frozen_runtime_inputs": {
                        "performance_config_sha256": "performance-config",
                        "main_commit": "main-commit",
                        "source_commit": "source-commit",
                        "model": model,
                    },
                },
            }

        baseline = comparison.verified_provenance(
            summary("baseline"),
            config,
            "baseline",
        )
        candidate_summary = summary("candidate")
        candidate = comparison.verified_provenance(
            candidate_summary,
            config,
            "candidate",
        )
        self.assertEqual(baseline, candidate)

        candidate_summary["preflight"]["frozen_runtime_inputs"]["main_commit"] = (
            "changed-main"
        )
        with self.assertRaisesRegex(
            ValueError,
            "frozen runtime input mismatch for main_commit",
        ):
            comparison.verified_provenance(
                candidate_summary,
                config,
                "candidate",
            )

    def test_comparison_rehashes_all_rank_profiler_evidence(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            root = Path(temp)
            tables = []
            traces = []
            for rank in range(8):
                table = root / f"profiler_out_{rank}.txt"
                table.write_text(
                    f"Self CUDA time total: {rank + 1}.000ms\n",
                    encoding="utf-8",
                )
                trace = root / f"worker_rank{rank}.pt.trace.json.gz"
                trace.write_bytes(f"trace-{rank}".encode())
                tables.append(
                    {
                        "rank": rank,
                        "path": str(table),
                        "sha256": matrix.sha256_file(table),
                        "self_cuda_time_total_ms": float(rank + 1),
                    }
                )
                traces.append(
                    {
                        "rank": rank,
                        "path": str(trace),
                        "bytes": trace.stat().st_size,
                        "sha256": matrix.sha256_file(trace),
                    }
                )
            frontend = root / "host.async_llm.123.pt.trace.json.gz"
            frontend.write_bytes(b"frontend")
            cell = {
                "profile": {
                    "profiler": {
                        "tables": tables,
                        "trace_files": traces,
                        "frontend_trace_files": [
                            {
                                "path": str(frontend),
                                "bytes": frontend.stat().st_size,
                                "sha256": matrix.sha256_file(frontend),
                            }
                        ],
                        "critical_rank": 7,
                        "kernel_time_ms_critical_rank": 8.0,
                    }
                }
            }
            critical = comparison.verified_profiler_evidence(cell, PROJECT_ROOT)
            self.assertEqual(critical, root / "profiler_out_7.txt")
            (root / "worker_rank3.pt.trace.json.gz").write_bytes(b"changed")
            with self.assertRaisesRegex(
                ValueError,
                "profiler trace (size|hash) mismatch",
            ):
                comparison.verified_profiler_evidence(cell, PROJECT_ROOT)

    def test_model_identity_detects_metadata_and_shard_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            for name in (
                "config.json",
                "generation_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "model.safetensors.index.json",
            ):
                (model_dir / name).write_text(name, encoding="utf-8")
            shard = model_dir / "model-00001-of-00001.safetensors"
            shard.write_bytes(b"weights")
            before = matrix.model_identity(model_dir)
            self.assertEqual(before["safetensors_count"], 1)

            (model_dir / "tokenizer.json").write_text("changed", encoding="utf-8")
            metadata_changed = matrix.model_identity(model_dir)
            self.assertNotEqual(before, metadata_changed)

            shard.write_bytes(b"changed weights")
            shard_changed = matrix.model_identity(model_dir)
            self.assertNotEqual(metadata_changed, shard_changed)

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
                "max_concurrency": 4,
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
                trace_name = (
                    f"dp0_pp0_tp{rank}_dcp0_ep0_rank{rank}.123456789.pt.trace.json.gz"
                )
                (runner.profile_dir / trace_name).write_bytes(f"rank-{rank}".encode())
            frontend = (
                runner.profile_dir / "container.async_llm.123456789.pt.trace.json.gz"
            )
            frontend.write_bytes(b"frontend")
            (root / "captured").mkdir()
            captured = runner.capture_profiler(
                before_files=set(),
                started_ns=0,
                output_dir=root / "captured",
            )
        self.assertEqual(len(captured["tables"]), 8)
        self.assertEqual(len(captured["trace_files"]), 8)
        self.assertEqual(len(captured["frontend_trace_files"]), 1)
        self.assertEqual(
            sorted(item["rank"] for item in captured["trace_files"]),
            list(range(8)),
        )
        self.assertEqual(captured["kernel_time_ms_critical_rank"], 4.0)

    def test_profiler_capture_rejects_duplicate_trace_rank(self) -> None:
        table = "Self CUDA time total: 4.000ms\n"
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
                trace_name = f"cycle_{rank}_rank0.123456789.pt.trace.json.gz"
                (runner.profile_dir / trace_name).write_bytes(b"trace")
            (root / "captured").mkdir()
            with self.assertRaisesRegex(
                ValueError,
                "expected profiler traces for TP ranks 0-7",
            ):
                runner.capture_profiler(
                    before_files=set(),
                    started_ns=0,
                    output_dir=root / "captured",
                )

    def test_relative_metric_directions(self) -> None:
        self.assertAlmostEqual(comparison.relative_increase(10.0, 12.0), 0.2)
        self.assertAlmostEqual(comparison.relative_drop(10.0, 8.0), 0.2)


if __name__ == "__main__":
    unittest.main()
