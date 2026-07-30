#!/usr/bin/env python3
"""Compare Stage 9 baseline and OSCAR performance summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


DURATION_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(ns|us|ms|s)$")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_scoped_artifact_path(path: Path, project_root: Path) -> bool:
    roots = (project_root / "artifacts", Path("/dev/shm"))
    return any(path == root or path.is_relative_to(root) for root in roots)


def verified_performance_config(summary: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(summary["performance_config"]).resolve()
    actual_sha256 = sha256_file(path)
    expected_sha256 = summary["performance_config_sha256"]
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"performance configuration hash mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return path, read_json(path)


def verified_provenance(
    summary: dict[str, Any],
    config: dict[str, Any],
    expected_variant: str,
) -> dict[str, Any]:
    preflight = summary["preflight"]
    frozen = preflight["frozen_runtime_inputs"]
    expected = {
        "performance_config_sha256": summary["performance_config_sha256"],
        "main_commit": preflight["main_commit"],
        "source_commit": config["source"]["commit"],
    }
    if preflight["variant"] != expected_variant:
        raise ValueError(
            f"preflight variant mismatch: {preflight['variant']} != {expected_variant}"
        )
    for name, value in expected.items():
        if frozen.get(name) != value:
            raise ValueError(
                f"frozen runtime input mismatch for {name}: "
                f"{frozen.get(name)} != {value}"
            )
    if preflight["source_commit"] != config["source"]["commit"]:
        raise ValueError("preflight source commit mismatch")
    model_identity = frozen["model"]
    expected_model_manifest = config["model"]["filename_size_mtime_ns_manifest_sha256"]
    if (
        model_identity["filename_size_mtime_ns_manifest_sha256"]
        != expected_model_manifest
    ):
        raise ValueError("preflight model identity mismatch")
    return frozen


def duration_ms(value: str) -> float:
    match = DURATION_RE.fullmatch(value)
    if match is None:
        raise ValueError(value)
    factors = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
    return float(match.group(1)) * factors[match.group(2)]


def top_cuda_rows(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Name" in line and "Self CUDA" in line
        ),
        None,
    )
    if header_index is None:
        return []
    headers = re.split(r"\s{2,}", lines[header_index].strip())
    try:
        cuda_index = headers.index("Self CUDA")
    except ValueError:
        return []
    rows = []
    for line in lines[header_index + 1 :]:
        if line.startswith("Self CUDA time total:"):
            break
        if not line.strip() or set(line.strip()) <= {"-"}:
            continue
        fields = re.split(r"\s{2,}", line.strip())
        if len(fields) <= cuda_index:
            continue
        try:
            cuda_ms = duration_ms(fields[cuda_index])
        except ValueError:
            continue
        rows.append(
            {
                "name": fields[0],
                "self_cuda_time_ms": cuda_ms,
                "row": line.rstrip(),
            }
        )
    return sorted(
        rows,
        key=lambda item: item["self_cuda_time_ms"],
        reverse=True,
    )[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def cell_key(cell: dict[str, Any]) -> tuple[int, int]:
    return cell["input_length"], cell["batch_size"]


def relative_increase(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError(f"baseline metric must be positive: {baseline}")
    return candidate / baseline - 1.0


def relative_drop(baseline: float, candidate: float) -> float:
    return -relative_increase(baseline, candidate)


def verified_profiler_evidence(
    cell: dict[str, Any],
    project_root: Path,
) -> Path:
    profiler = cell["profile"]["profiler"]
    tables = sorted(profiler["tables"], key=lambda item: item["rank"])
    if [item["rank"] for item in tables] != list(range(8)):
        raise ValueError("profiler tables do not cover TP ranks 0-7")
    for item in tables:
        path = Path(item["path"]).resolve()
        if not is_scoped_artifact_path(path, project_root):
            raise ValueError(f"profiler table is outside artifact roots: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != item["sha256"]:
            raise ValueError(
                f"profiler table hash mismatch: {actual_sha256} != {item['sha256']}"
            )
        actual_cuda_ms = duration_ms_from_table(path)
        if not math.isclose(
            actual_cuda_ms,
            item["self_cuda_time_total_ms"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("profiler table CUDA total mismatch")

    traces = profiler["trace_files"]
    if sorted(item["rank"] for item in traces) != list(range(8)):
        raise ValueError("profiler traces do not cover TP ranks 0-7")
    for item in traces:
        path = Path(item["path"]).resolve()
        if not is_scoped_artifact_path(path, project_root):
            raise ValueError(f"profiler trace is outside artifact roots: {path}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError("profiler trace size mismatch")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != item["sha256"]:
            raise ValueError(
                f"profiler trace hash mismatch: {actual_sha256} != {item['sha256']}"
            )

    frontend_traces = profiler["frontend_trace_files"]
    if len(frontend_traces) != 1:
        raise ValueError("profiler evidence must contain one frontend trace")
    for item in frontend_traces:
        path = Path(item["path"]).resolve()
        if not is_scoped_artifact_path(path, project_root):
            raise ValueError(f"frontend trace is outside artifact roots: {path}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError("frontend profiler trace size mismatch")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != item["sha256"]:
            raise ValueError(
                "frontend profiler trace hash mismatch: "
                f"{actual_sha256} != {item['sha256']}"
            )

    expected_critical = max(
        tables,
        key=lambda item: item["self_cuda_time_total_ms"],
    )
    if profiler["critical_rank"] != expected_critical["rank"]:
        raise ValueError("profiler critical rank mismatch")
    if not math.isclose(
        profiler["kernel_time_ms_critical_rank"],
        expected_critical["self_cuda_time_total_ms"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("profiler critical CUDA total mismatch")
    return Path(expected_critical["path"]).resolve()


def duration_ms_from_table(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"Self CUDA time total:\s*([0-9]+(?:\.[0-9]+)?)\s*(ns|us|ms|s)",
        text,
    )
    if match is None:
        raise ValueError(f"missing CUDA total in profiler table: {path}")
    return duration_ms("".join(match.groups()))


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_path = args.output.resolve()
    if not is_scoped_artifact_path(output_path, project_root):
        raise SystemExit(
            f"output must be under project artifacts/ or /dev/shm/: {output_path}"
        )
    if output_path.exists():
        raise SystemExit(f"output already exists: {output_path}")
    baseline_path = args.baseline.resolve()
    candidate_path = args.candidate.resolve()
    for label, path in (
        ("baseline summary", baseline_path),
        ("candidate summary", candidate_path),
    ):
        if not is_scoped_artifact_path(path, project_root):
            raise SystemExit(
                f"{label} must be under project artifacts/ or /dev/shm/: {path}"
            )
    baseline = read_json(baseline_path)
    candidate = read_json(candidate_path)
    if baseline["status"] != "passed" or baseline["variant"] != "baseline":
        raise SystemExit("invalid baseline summary")
    if candidate["status"] != "passed" or candidate["variant"] != "candidate":
        raise SystemExit("invalid candidate summary")
    if baseline["performance_config_sha256"] != candidate["performance_config_sha256"]:
        raise SystemExit("performance configuration mismatch")
    _, baseline_config = verified_performance_config(baseline)
    _, config = verified_performance_config(candidate)
    if baseline_config != config:
        raise SystemExit("performance configuration content mismatch")
    baseline_runtime = verified_provenance(baseline, config, "baseline")
    candidate_runtime = verified_provenance(candidate, config, "candidate")
    if baseline_runtime != candidate_runtime:
        raise SystemExit("baseline/candidate runtime provenance mismatch")
    thresholds = config["regression_thresholds"]
    baseline_cells = {cell_key(cell): cell for cell in baseline["cells"]}
    candidate_cells = {cell_key(cell): cell for cell in candidate["cells"]}
    expected = {
        (input_length, batch_size)
        for input_length in config["matrix"]["input_lengths"]
        for batch_size in config["matrix"]["batch_sizes"]
    }
    if set(baseline_cells) != expected or set(candidate_cells) != expected:
        raise SystemExit("matrix cell set mismatch")

    comparisons = []
    all_regressions = []
    for key in sorted(expected):
        native = baseline_cells[key]
        oscar = candidate_cells[key]
        native_table = verified_profiler_evidence(native, project_root)
        oscar_table = verified_profiler_evidence(oscar, project_root)
        latency = {
            name: relative_increase(
                native["median_metrics"][name],
                oscar["median_metrics"][name],
            )
            for name in ("mean_ttft_ms", "mean_tpot_ms")
        }
        throughput = {
            name: relative_drop(
                native["median_metrics"][name],
                oscar["median_metrics"][name],
            )
            for name in (
                "request_throughput",
                "output_throughput",
                "total_token_throughput",
            )
        }
        memory = {
            "peak_memory_mib_max": relative_increase(
                native["peak_memory_mib_max"],
                oscar["peak_memory_mib_max"],
            ),
            "peak_memory_mib_sum": relative_increase(
                native["peak_memory_mib_sum"],
                oscar["peak_memory_mib_sum"],
            ),
        }
        kernel = relative_increase(
            native["profile"]["profiler"]["kernel_time_ms_critical_rank"],
            oscar["profile"]["profiler"]["kernel_time_ms_critical_rank"],
        )
        regressions = []
        for name, value in latency.items():
            if value > thresholds["latency_max_increase"]:
                regressions.append({"metric": name, "relative_regression": value})
        for name, value in throughput.items():
            if value > thresholds["throughput_max_drop"]:
                regressions.append({"metric": name, "relative_regression": value})
        if memory["peak_memory_mib_max"] > thresholds["peak_memory_max_increase"]:
            regressions.append(
                {
                    "metric": "peak_memory_mib_max",
                    "relative_regression": memory["peak_memory_mib_max"],
                }
            )
        if kernel > thresholds["kernel_time_max_increase"]:
            regressions.append(
                {
                    "metric": "kernel_time_ms_critical_rank",
                    "relative_regression": kernel,
                }
            )
        all_regressions.extend(
            {"input_length": key[0], "batch_size": key[1], **item}
            for item in regressions
        )
        comparisons.append(
            {
                "input_length": key[0],
                "batch_size": key[1],
                "baseline": {
                    "metrics": native["median_metrics"],
                    "peak_memory_mib_max": native["peak_memory_mib_max"],
                    "peak_memory_mib_sum": native["peak_memory_mib_sum"],
                    "kernel_time_ms_critical_rank": native["profile"]["profiler"][
                        "kernel_time_ms_critical_rank"
                    ],
                    "critical_rank": native["profile"]["profiler"]["critical_rank"],
                    "server_scheduling": native["server_scheduling"],
                },
                "candidate": {
                    "metrics": oscar["median_metrics"],
                    "peak_memory_mib_max": oscar["peak_memory_mib_max"],
                    "peak_memory_mib_sum": oscar["peak_memory_mib_sum"],
                    "kernel_time_ms_critical_rank": oscar["profile"]["profiler"][
                        "kernel_time_ms_critical_rank"
                    ],
                    "critical_rank": oscar["profile"]["profiler"]["critical_rank"],
                    "server_scheduling": oscar["server_scheduling"],
                },
                "relative_regression": {
                    "latency": latency,
                    "throughput": throughput,
                    "memory": memory,
                    "kernel_time": kernel,
                },
                "regressions_over_threshold": regressions,
                "profiling": {
                    "baseline_table": str(native_table),
                    "candidate_table": str(oscar_table),
                    "baseline_top_cuda": top_cuda_rows(native_table),
                    "candidate_top_cuda": top_cuda_rows(oscar_table),
                },
            }
        )

    context = candidate.get("context_128k")
    if not context or context["status"] != "passed":
        raise SystemExit("candidate 128K evidence is missing")
    result = {
        "format_version": 1,
        "status": ("passed" if not all_regressions else "passed_with_regressions"),
        "baseline_summary": str(baseline_path),
        "baseline_summary_sha256": sha256_file(baseline_path),
        "candidate_summary": str(candidate_path),
        "candidate_summary_sha256": sha256_file(candidate_path),
        "performance_config_sha256": candidate["performance_config_sha256"],
        "thresholds": thresholds,
        "cells": comparisons,
        "regressions_over_threshold": all_regressions,
        "requires_written_attribution": bool(all_regressions),
        "context_128k": context,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
