#!/usr/bin/env python3
"""Run the frozen Stage 9 serving matrix against one TP=8 server."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import threading
import time
from typing import Any

import requests


CUDA_TOTAL_RE = re.compile(
    r"Self CUDA time total:\s*([0-9]+(?:\.[0-9]+)?)\s*(ns|us|ms|s)"
)
RANK_TABLE_RE = re.compile(r"profiler_out_([0-9]+)\.txt$")
TRACE_RANK_RE = re.compile(r"(?:^|_)rank([0-9]+)(?:[._]|$)")
SERVER_METRIC_RE = re.compile(
    r"^vllm:(num_requests_running|num_requests_waiting|"
    r"kv_cache_usage_perc|num_preemptions_total)\{[^}]*\}\s+"
    r"([-+0-9.eE]+)$",
    re.MULTILINE,
)
REQUIRED_SERVER_METRICS = {
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_perc",
    "num_preemptions_total",
}
BENCHMARK_HELP_ARGUMENT = "--help=all"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(model_dir: Path) -> dict[str, Any]:
    metadata_names = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    )
    shards = sorted(model_dir.glob("*.safetensors"))
    shard_rows = []
    for path in shards:
        stat = path.stat()
        shard_rows.append(f"{path.name} {stat.st_size} {stat.st_mtime_ns}\n")
    return {
        "metadata_sha256": {
            name: sha256_file(model_dir / name) for name in metadata_names
        },
        "safetensors_count": len(shards),
        "filename_size_mtime_ns_manifest_sha256": hashlib.sha256(
            "".join(shard_rows).encode()
        ).hexdigest(),
    }


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def is_scoped_artifact_path(path: Path, runtime_root: Path) -> bool:
    roots = (runtime_root / "artifacts", Path("/dev/shm"))
    return any(path == root or path.is_relative_to(root) for root in roots)


def duration_ms(value: float, unit: str) -> float:
    factors = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
    return value * factors[unit]


def parse_cuda_total(path: Path) -> float:
    match = CUDA_TOTAL_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"missing CUDA total in profiler table: {path}")
    return duration_ms(float(match.group(1)), match.group(2))


def parse_server_metrics(payload: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for name, raw_value in SERVER_METRIC_RE.findall(payload):
        values.setdefault(name, []).append(float(raw_value))
    missing = REQUIRED_SERVER_METRICS - set(values)
    if missing:
        raise ValueError(f"missing vLLM server metrics: {sorted(missing)}")
    return {
        "num_requests_running": sum(values["num_requests_running"]),
        "num_requests_waiting": sum(values["num_requests_waiting"]),
        "kv_cache_usage_perc": max(values["kv_cache_usage_perc"]),
        "num_preemptions_total": sum(values["num_preemptions_total"]),
    }


@dataclass
class GpuSample:
    timestamp_unix: float
    rows: list[dict[str, int]]
    server: dict[str, float]


class GpuSampler:
    def __init__(self, interval_seconds: float, metrics_url: str) -> None:
        self.interval_seconds = interval_seconds
        self.metrics_url = metrics_url
        self.samples: list[GpuSample] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()

    def _sample(self) -> None:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            self.errors.append(completed.stderr.strip())
            return
        rows = []
        for line in completed.stdout.splitlines():
            fields = [int(item.strip()) for item in line.split(",")]
            if len(fields) != 4:
                self.errors.append(f"invalid nvidia-smi row: {line}")
                return
            rows.append(
                {
                    "index": fields[0],
                    "memory_used_mib": fields[1],
                    "memory_total_mib": fields[2],
                    "utilization_gpu_percent": fields[3],
                }
            )
        if len(rows) != 8 or [row["index"] for row in rows] != list(range(8)):
            self.errors.append(f"expected GPUs 0-7, got {rows}")
            return
        try:
            response = requests.get(self.metrics_url, timeout=5)
            response.raise_for_status()
            server = parse_server_metrics(response.text)
        except (requests.RequestException, ValueError) as error:
            self.errors.append(f"server metrics sampling failed: {error}")
            return
        self.samples.append(GpuSample(time.time(), rows, server))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def write(self, path: Path) -> dict[str, Any]:
        with path.open("w", encoding="utf-8") as output:
            for sample in self.samples:
                output.write(
                    json.dumps(
                        {
                            "timestamp_unix": sample.timestamp_unix,
                            "gpus": sample.rows,
                            "server": sample.server,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        if not self.samples:
            raise ValueError("runtime sampler produced no samples")
        peaks = {
            str(index): max(
                sample.rows[index]["memory_used_mib"] for sample in self.samples
            )
            for index in range(8)
        }
        server = {
            "max_requests_running": max(
                sample.server["num_requests_running"] for sample in self.samples
            ),
            "max_requests_waiting": max(
                sample.server["num_requests_waiting"] for sample in self.samples
            ),
            "max_kv_cache_usage_perc": max(
                sample.server["kv_cache_usage_perc"] for sample in self.samples
            ),
            "preemptions_delta": (
                self.samples[-1].server["num_preemptions_total"]
                - self.samples[0].server["num_preemptions_total"]
            ),
        }
        return {
            "samples": len(self.samples),
            "errors": self.errors,
            "peak_memory_mib_by_gpu": peaks,
            "peak_memory_mib_max": max(peaks.values()),
            "peak_memory_mib_sum": sum(peaks.values()),
            "server": server,
            "samples_sha256": sha256_file(path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--server-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--runtime-project-root", type=Path)
    parser.add_argument("--include-128k", action="store_true")
    parser.add_argument("--formal", action="store_true")
    return parser.parse_args()


class MatrixRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = Path(__file__).resolve().parents[2]
        self.runtime_root = (args.runtime_project_root or self.project_root).resolve()
        self.config_path = self.project_root / "configs/phase9/performance_matrix.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.output_dir = args.output_dir.resolve()
        self.profile_dir = args.profile_dir.resolve()
        self.server_run_dir = args.server_run_dir.resolve()
        self.base_url = args.base_url.rstrip("/")
        self.metrics_url = f"{self.base_url}/metrics"
        self.rootfs = self.runtime_root / "artifacts/phase0-candidate-bundle/rootfs"
        self.python = self.rootfs / "usr/bin/python3.12"
        self.source_dir = self.runtime_root / "glm52_oscar_vllm"
        venv = self.rootfs / "opt/fp8_speed_up_v4_venv"
        self.client_env = os.environ.copy()
        self.client_env.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "GLM52_CANDIDATE_ROOTFS": str(self.rootfs),
                "PYTHONHOME": str(self.rootfs / "usr"),
                "VIRTUAL_ENV": str(venv),
                "PYTHONPATH": ":".join(
                    [
                        str(self.source_dir),
                        str(venv / "lib/python3.12/site-packages"),
                        str(self.rootfs / "usr/local/lib/python3.12/dist-packages"),
                        str(self.rootfs / "usr/lib/python3/dist-packages"),
                    ]
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1",
                "XDG_CACHE_HOME": str(
                    self.runtime_root / "artifacts/phase9/client-cache"
                ),
            }
        )
        self.progress_log = self.output_dir / "progress_10min.log"

    def assert_runtime_inputs_unchanged(self) -> None:
        current = {
            "performance_config_sha256": sha256_file(self.config_path),
            "main_commit": git(self.runtime_root, "rev-parse", "HEAD"),
            "source_commit": git(self.source_dir, "rev-parse", "HEAD"),
            "model": model_identity(Path(self.config["model"]["path"])),
        }
        if current != self.frozen_runtime_inputs:
            raise RuntimeError(
                f"Stage 9 runtime inputs changed: "
                f"{current} != {self.frozen_runtime_inputs}"
            )
        for repo in (self.runtime_root, self.source_dir):
            if git(repo, "status", "--porcelain", "--untracked-files=all"):
                raise RuntimeError(f"repository became dirty: {repo}")

    def validate_preflight(self) -> dict[str, Any]:
        if not self.args.formal:
            raise ValueError("Stage 9 matrix requires --formal")
        if self.output_dir.exists():
            raise FileExistsError(f"output directory exists: {self.output_dir}")
        for label, path in (
            ("output directory", self.output_dir),
            ("profile directory", self.profile_dir),
            ("server run directory", self.server_run_dir),
        ):
            if not is_scoped_artifact_path(path, self.runtime_root):
                raise ValueError(
                    f"{label} must be under runtime artifacts/ or /dev/shm/: {path}"
                )
        self.output_dir.mkdir(parents=True)
        if self.config["status"] != "ready":
            raise ValueError("performance config is not ready")
        if self.args.variant == "baseline" and self.args.include_128k:
            raise ValueError("128K validation is reserved for the OSCAR candidate")

        runtime_manifest_path = self.server_run_dir / "runtime_manifest.json"
        parsed_args_path = self.server_run_dir / "parsed_server_args.json"
        runtime_manifest = json.loads(runtime_manifest_path.read_text())
        parsed_args = json.loads(parsed_args_path.read_text())
        expected_dtype = "auto" if self.args.variant == "baseline" else "oscar_mla_int2"
        expected = {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "max_model_len": self.config["server"]["max_model_len"],
            "max_num_seqs": self.config["server"]["max_num_seqs"],
            "kv_cache_dtype": expected_dtype,
            "enable_prefix_caching": False,
            "enforce_eager": True,
            "async_scheduling": False,
            "profiler": "torch",
            "torch_profiler_dir": str(self.profile_dir),
        }
        for name, value in expected.items():
            if parsed_args.get(name) != value:
                raise ValueError(
                    f"server argument mismatch for {name}: "
                    f"{parsed_args.get(name)!r} != {value!r}"
                )

        main_head = git(self.runtime_root, "rev-parse", "HEAD")
        source_head = git(self.source_dir, "rev-parse", "HEAD")
        if runtime_manifest["main_commit"] != main_head:
            raise ValueError("server/main commit mismatch")
        if runtime_manifest["source_repository_commit"] != source_head:
            raise ValueError("server/source commit mismatch")
        if source_head != self.config["source"]["commit"]:
            raise ValueError("unexpected source commit")
        for repo in (self.runtime_root, self.source_dir):
            if git(repo, "status", "--porcelain", "--untracked-files=all"):
                raise ValueError(f"repository is not clean: {repo}")
            branch = git(repo, "symbolic-ref", "--short", "HEAD")
            remote = git(repo, "remote", "get-url", "origin")
            remote_head = subprocess.check_output(
                ["git", "ls-remote", "--heads", remote, f"refs/heads/{branch}"],
                text=True,
            ).split()[0]
            if remote_head != git(repo, "rev-parse", "HEAD"):
                raise ValueError(f"repository is not published: {repo}")
        current_model_identity = model_identity(Path(self.config["model"]["path"]))
        if (
            current_model_identity["filename_size_mtime_ns_manifest_sha256"]
            != self.config["model"]["filename_size_mtime_ns_manifest_sha256"]
        ):
            raise ValueError("model filename/size/mtime identity mismatch")
        self.frozen_runtime_inputs = {
            "performance_config_sha256": sha256_file(self.config_path),
            "main_commit": main_head,
            "source_commit": source_head,
            "model": current_model_identity,
        }

        pid = int((self.server_run_dir / "server.pid").read_text())
        os.kill(pid, 0)
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        models = requests.get(f"{self.base_url}/v1/models", timeout=10)
        models.raise_for_status()
        model_ids = [item["id"] for item in models.json()["data"]]
        if self.config["model"]["served_name"] not in model_ids:
            raise ValueError(f"served model is missing: {model_ids}")

        cli_help = subprocess.run(
            [
                str(self.python),
                "-m",
                "vllm.entrypoints.cli.main",
                "bench",
                "serve",
                BENCHMARK_HELP_ARGUMENT,
            ],
            cwd=self.source_dir,
            env=self.client_env,
            check=False,
            capture_output=True,
            text=True,
        )
        help_text = cli_help.stdout + cli_help.stderr
        required_flags = {
            "--num-warmups",
            "--max-concurrency",
            "--random-input-len",
            "--random-output-len",
            "--random-range-ratio",
            "--profile",
            "--save-detailed",
        }
        missing = sorted(flag for flag in required_flags if flag not in help_text)
        if cli_help.returncode != 0 or missing:
            raise ValueError(
                f"benchmark CLI validation failed: status={cli_help.returncode}, "
                f"missing={missing}"
            )
        (self.output_dir / "benchmark_cli_help.txt").write_text(
            help_text,
            encoding="utf-8",
        )
        preflight = {
            "status": "passed",
            "variant": self.args.variant,
            "runtime_manifest": str(runtime_manifest_path),
            "runtime_manifest_sha256": sha256_file(runtime_manifest_path),
            "parsed_server_args": str(parsed_args_path),
            "parsed_server_args_sha256": sha256_file(parsed_args_path),
            "main_commit": main_head,
            "source_commit": source_head,
            "frozen_runtime_inputs": self.frozen_runtime_inputs,
            "server_pid": pid,
            "served_models": model_ids,
            "benchmark_cli_help_sha256": sha256_file(
                self.output_dir / "benchmark_cli_help.txt"
            ),
        }
        json_dump(self.output_dir / "preflight.json", preflight)
        return preflight

    def benchmark_command(
        self,
        *,
        input_length: int,
        output_length: int,
        batch_size: int,
        num_prompts: int,
        num_warmups: int,
        result_dir: Path,
        profile: bool,
    ) -> list[str]:
        command = [
            str(self.python),
            "-m",
            "vllm.entrypoints.cli.main",
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--base-url",
            self.base_url,
            "--endpoint",
            "/v1/completions",
            "--model",
            self.config["model"]["path"],
            "--served-model-name",
            self.config["model"]["served_name"],
            "--tokenizer",
            self.config["model"]["path"],
            "--trust-remote-code",
            "--dataset-name",
            "random",
            "--random-input-len",
            str(input_length),
            "--random-output-len",
            str(output_length),
            "--random-range-ratio",
            "0",
            "--num-prompts",
            str(num_prompts),
            "--num-warmups",
            str(num_warmups),
            "--max-concurrency",
            str(batch_size),
            "--request-rate",
            "inf",
            "--seed",
            str(self.config["matrix"]["seed"]),
            "--temperature",
            "0",
            "--ignore-eos",
            "--disable-tqdm",
            "--percentile-metrics",
            "ttft,tpot",
            "--metric-percentiles",
            "50,90,99",
            "--save-result",
            "--save-detailed",
            "--result-dir",
            str(result_dir),
            "--result-filename",
            "result.json",
            "--metadata",
            f"variant={self.args.variant}",
            f"input_length={input_length}",
            f"batch_size={batch_size}",
            f"profile={str(profile).lower()}",
        ]
        if profile:
            command.append("--profile")
        return command

    def run_command(
        self,
        command: list[str],
        run_dir: Path,
        label: str,
    ) -> dict[str, Any]:
        self.assert_runtime_inputs_unchanged()
        run_dir.mkdir(parents=True)
        (run_dir / "command.txt").write_text(
            shlex.join(command) + "\n",
            encoding="utf-8",
        )
        sampler = GpuSampler(
            self.config["matrix"]["gpu_sample_interval_seconds"],
            self.metrics_url,
        )
        started = time.time()
        next_progress = 600
        sampler.start()
        try:
            with (run_dir / "runner.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.source_dir,
                    env=self.client_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                while True:
                    try:
                        process.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        elapsed = int(time.time() - started)
                        if elapsed >= next_progress:
                            line = (
                                f"{utc_now()} label={label} pid={process.pid} "
                                f"elapsed_seconds={elapsed}"
                            )
                            print(line, flush=True)
                            with self.progress_log.open(
                                "a", encoding="utf-8"
                            ) as progress:
                                progress.write(line + "\n")
                            next_progress += 600
        finally:
            sampler.stop()
        elapsed = time.time() - started
        gpu = sampler.write(run_dir / "gpu_samples.jsonl")
        self.assert_runtime_inputs_unchanged()
        if process.returncode != 0:
            raise RuntimeError(
                f"benchmark failed for {label}: status={process.returncode}"
            )
        if gpu["errors"]:
            raise RuntimeError(f"GPU sampling failed for {label}: {gpu['errors']}")
        return {
            "elapsed_seconds": elapsed,
            "gpu": gpu,
            "command_sha256": sha256_file(run_dir / "command.txt"),
            "runner_log_sha256": sha256_file(run_dir / "runner.log"),
        }

    def validate_result(
        self,
        result_path: Path,
        *,
        input_length: int,
        output_length: int,
        num_prompts: int,
        batch_size: int,
    ) -> dict[str, Any]:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["completed"] != num_prompts or result["failed"] != 0:
            raise ValueError(f"incomplete benchmark result: {result_path}")
        if result["input_lens"] != [input_length] * num_prompts:
            raise ValueError(f"input length mismatch: {result_path}")
        if result["output_lens"] != [output_length] * num_prompts:
            raise ValueError(f"output length mismatch: {result_path}")
        if any(result["errors"]):
            raise ValueError(f"request errors present: {result_path}")
        if result["max_concurrent_requests"] > batch_size:
            raise ValueError(f"concurrency exceeded: {result_path}")
        required = (
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
            "mean_ttft_ms",
            "median_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
        )
        for name in required:
            value = result[name]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid metric {name}: {value}")
        return result

    def capture_profiler(
        self,
        *,
        before_files: set[str],
        started_ns: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        tables = []
        for path in sorted(self.profile_dir.glob("profiler_out_*.txt")):
            match = RANK_TABLE_RE.search(path.name)
            if match is None or path.stat().st_mtime_ns < started_ns:
                continue
            target = output_dir / path.name
            target.write_bytes(path.read_bytes())
            tables.append(
                {
                    "rank": int(match.group(1)),
                    "path": str(target),
                    "sha256": sha256_file(target),
                    "self_cuda_time_total_ms": parse_cuda_total(target),
                }
            )
        if sorted(item["rank"] for item in tables) != list(range(8)):
            raise ValueError(f"expected profiler tables for ranks 0-7: {tables}")
        after_paths = {
            str(path.relative_to(self.profile_dir))
            for path in self.profile_dir.rglob("*")
            if path.is_file()
        }
        trace_files = []
        for relative in sorted(after_paths - before_files):
            path = self.profile_dir / relative
            if path.name.startswith("profiler_out_"):
                continue
            if path.stat().st_size <= 0:
                raise ValueError(f"empty profiler trace: {path}")
            match = TRACE_RANK_RE.search(path.name)
            if match is None:
                raise ValueError(f"profiler trace has no rank identity: {path}")
            trace_files.append(
                {
                    "rank": int(match.group(1)),
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        trace_ranks = sorted({item["rank"] for item in trace_files})
        if trace_ranks != list(range(8)):
            raise ValueError(
                f"expected profiler traces for TP ranks 0-7, got ranks {trace_ranks}"
            )
        critical = max(tables, key=lambda item: item["self_cuda_time_total_ms"])
        return {
            "tables": tables,
            "critical_rank": critical["rank"],
            "kernel_time_ms_critical_rank": critical["self_cuda_time_total_ms"],
            "trace_files": trace_files,
        }

    def run_cell(self, input_length: int, batch_size: int) -> dict[str, Any]:
        matrix = self.config["matrix"]
        cell_name = f"input_{input_length}_batch_{batch_size}"
        cell_dir = self.output_dir / "matrix" / cell_name
        rounds = []
        for round_index in range(1, matrix["rounds"] + 1):
            round_dir = cell_dir / f"round_{round_index}"
            num_prompts = batch_size * matrix["prompts_per_batch"]
            command = self.benchmark_command(
                input_length=input_length,
                output_length=matrix["output_length"],
                batch_size=batch_size,
                num_prompts=num_prompts,
                num_warmups=batch_size * matrix["warmup_requests_per_batch"],
                result_dir=round_dir,
                profile=False,
            )
            execution = self.run_command(
                command,
                round_dir,
                f"{cell_name}/round_{round_index}",
            )
            result_path = round_dir / "result.json"
            result = self.validate_result(
                result_path,
                input_length=input_length,
                output_length=matrix["output_length"],
                num_prompts=num_prompts,
                batch_size=batch_size,
            )
            validation = {
                "status": "passed",
                "result_sha256": sha256_file(result_path),
                **execution,
            }
            json_dump(round_dir / "validation.json", validation)
            rounds.append(
                {
                    "round": round_index,
                    "result": str(result_path),
                    "validation": str(round_dir / "validation.json"),
                    "metrics": {
                        name: result[name]
                        for name in (
                            "duration",
                            "request_throughput",
                            "output_throughput",
                            "total_token_throughput",
                            "mean_ttft_ms",
                            "median_ttft_ms",
                            "mean_tpot_ms",
                            "median_tpot_ms",
                            "max_concurrent_requests",
                        )
                    },
                    "peak_memory_mib_max": execution["gpu"]["peak_memory_mib_max"],
                    "peak_memory_mib_sum": execution["gpu"]["peak_memory_mib_sum"],
                    "server": execution["gpu"]["server"],
                }
            )

        metric_names = (
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
            "mean_ttft_ms",
            "median_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
        )
        medians = {
            name: statistics.median(item["metrics"][name] for item in rounds)
            for name in metric_names
        }
        variation = {
            name: (
                max(item["metrics"][name] for item in rounds)
                - min(item["metrics"][name] for item in rounds)
            )
            / medians[name]
            for name in metric_names
        }

        profile = self.config["profiler"]
        profile_dir = cell_dir / "profile"
        before = {
            str(path.relative_to(self.profile_dir))
            for path in self.profile_dir.rglob("*")
            if path.is_file()
        }
        started_ns = time.time_ns()
        profile_prompts = batch_size * profile["prompts_per_batch"]
        command = self.benchmark_command(
            input_length=input_length,
            output_length=matrix["output_length"],
            batch_size=batch_size,
            num_prompts=profile_prompts,
            num_warmups=batch_size * profile["warmup_requests_per_batch"],
            result_dir=profile_dir,
            profile=True,
        )
        profile_execution = self.run_command(
            command,
            profile_dir,
            f"{cell_name}/profile",
        )
        profile_result_path = profile_dir / "result.json"
        self.validate_result(
            profile_result_path,
            input_length=input_length,
            output_length=matrix["output_length"],
            num_prompts=profile_prompts,
            batch_size=batch_size,
        )
        profiler = self.capture_profiler(
            before_files=before,
            started_ns=started_ns,
            output_dir=profile_dir,
        )
        profile_validation = {
            "status": "passed",
            "result_sha256": sha256_file(profile_result_path),
            **profile_execution,
            "profiler": profiler,
        }
        json_dump(profile_dir / "validation.json", profile_validation)

        summary = {
            "status": "passed",
            "input_length": input_length,
            "output_length": matrix["output_length"],
            "batch_size": batch_size,
            "rounds": rounds,
            "median_metrics": medians,
            "relative_range_by_metric": variation,
            "peak_memory_mib_max": max(item["peak_memory_mib_max"] for item in rounds),
            "peak_memory_mib_sum": max(item["peak_memory_mib_sum"] for item in rounds),
            "server_scheduling": {
                "max_requests_running": max(
                    item["server"]["max_requests_running"] for item in rounds
                ),
                "max_requests_waiting": max(
                    item["server"]["max_requests_waiting"] for item in rounds
                ),
                "max_kv_cache_usage_perc": max(
                    item["server"]["max_kv_cache_usage_perc"] for item in rounds
                ),
                "preemptions_delta": sum(
                    item["server"]["preemptions_delta"] for item in rounds
                ),
                "reached_client_concurrency": (
                    max(item["server"]["max_requests_running"] for item in rounds)
                    >= batch_size
                ),
                "capacity_limited": (
                    max(item["server"]["max_requests_waiting"] for item in rounds) > 0
                    or sum(item["server"]["preemptions_delta"] for item in rounds) > 0
                    or max(item["server"]["max_requests_running"] for item in rounds)
                    < batch_size
                ),
            },
            "profile": profile_validation,
        }
        json_dump(cell_dir / "summary.json", summary)
        return summary

    def run_128k(self) -> dict[str, Any]:
        context = self.config["context_128k"]
        output_dir = self.output_dir / "context_128k"
        command = self.benchmark_command(
            input_length=context["input_length"],
            output_length=context["output_length"],
            batch_size=context["batch_size"],
            num_prompts=1,
            num_warmups=context["warmup_requests"],
            result_dir=output_dir,
            profile=False,
        )
        execution = self.run_command(command, output_dir, "context_128k")
        result_path = output_dir / "result.json"
        result = self.validate_result(
            result_path,
            input_length=context["input_length"],
            output_length=context["output_length"],
            num_prompts=1,
            batch_size=1,
        )
        validation = {
            "status": "passed",
            "input_length": result["input_lens"][0],
            "output_length": result["output_lens"][0],
            "total_length": result["input_lens"][0] + result["output_lens"][0],
            "expected_total_length": context["total_length"],
            "result_sha256": sha256_file(result_path),
            **execution,
        }
        if validation["total_length"] != context["total_length"]:
            raise ValueError(f"128K total length mismatch: {validation}")
        json_dump(output_dir / "validation.json", validation)
        return validation

    def run(self) -> int:
        started = time.time()
        preflight = self.validate_preflight()
        cells = []
        for input_length in self.config["matrix"]["input_lengths"]:
            for batch_size in self.config["matrix"]["batch_sizes"]:
                cells.append(self.run_cell(input_length, batch_size))
        context_128k = self.run_128k() if self.args.include_128k else None
        summary = {
            "format_version": 1,
            "status": "passed",
            "variant": self.args.variant,
            "started_at_unix": started,
            "ended_at_unix": time.time(),
            "duration_seconds": time.time() - started,
            "performance_config": str(self.config_path),
            "performance_config_sha256": sha256_file(self.config_path),
            "preflight": preflight,
            "cells": cells,
            "context_128k": context_128k,
        }
        json_dump(self.output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0


def main() -> int:
    return MatrixRunner(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
