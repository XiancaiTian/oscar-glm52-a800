#!/usr/bin/env python3
"""Isolate the first prefill window in Stage 9 PyTorch profiler traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import tempfile
import time
from typing import Any

import ijson


FORMAT_VERSION = 1
EXECUTE_PREFIX = "execute_context_"
PREFILL_PATTERN = re.compile(
    r"^execute_context_(?P<context>\d+)\((?P<tokens>\d+)\)"
    r"_generation_0\(0\)$"
)
GENERATION_PATTERN = re.compile(r"_generation_(?P<generation>\d+)\(")
RANK_PATTERN = re.compile(r"_rank(?P<rank>\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top-kernels", type=int, default=30)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.top_kernels <= 0:
        parser.error("--top-kernels must be positive")
    if len(set(args.trace)) != len(args.trace):
        parser.error("--trace paths must be unique")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o644)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _rank_from_path(path: Path) -> int:
    match = RANK_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse rank from trace filename: {path.name}")
    return int(match.group("rank"))


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def analyze_trace(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    rank = _rank_from_path(path)
    prefill: tuple[str, float, float, int] | None = None
    execute_contexts: list[tuple[str, float]] = []
    kernel_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    annotation_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    event_count = 0
    kernel_before_prefill_annotation = 0

    with gzip.open(path, "rb") as handle:
        for event in ijson.items(handle, "traceEvents.item"):
            event_count += 1
            category = str(event.get("cat", ""))
            name = str(event.get("name", ""))
            timestamp = float(event.get("ts", -1))
            duration = float(event.get("dur", 0))

            if category == "user_annotation" and name.startswith(EXECUTE_PREFIX):
                execute_contexts.append((name, duration))
                match = PREFILL_PATTERN.match(name)
                if (
                    match is not None
                    and int(match.group("context")) > 0
                    and int(match.group("tokens")) > 0
                ):
                    if prefill is not None:
                        raise ValueError(f"multiple prefill windows in trace: {path}")
                    prefill = (
                        name,
                        timestamp,
                        duration,
                        int(match.group("tokens")),
                    )

            if category == "kernel" and prefill is None:
                kernel_before_prefill_annotation += 1
                continue
            if prefill is None:
                continue
            _, prefill_start, prefill_duration, _ = prefill
            if not prefill_start <= timestamp < prefill_start + prefill_duration:
                continue
            if category == "kernel":
                kernel_stats[name][0] += 1
                kernel_stats[name][1] += duration
            elif category == "user_annotation" and name != prefill[0]:
                annotation_stats[name][0] += 1
                annotation_stats[name][1] += duration

    if prefill is None:
        raise ValueError(f"trace does not contain a prefill window: {path}")
    if kernel_before_prefill_annotation:
        raise ValueError(
            "trace contains kernel events before its prefill annotation: "
            f"{path} count={kernel_before_prefill_annotation}"
        )
    prefill_name, _, prefill_duration, prefill_tokens = prefill
    kernel_total_us = sum(row[1] for row in kernel_stats.values())
    generation_durations_ms = []
    for name, duration in execute_contexts:
        match = GENERATION_PATTERN.search(name)
        if match is not None and int(match.group("generation")) > 0:
            generation_durations_ms.append(duration / 1000.0)
    if not generation_durations_ms:
        raise ValueError(f"trace contains no generation windows: {path}")

    return {
        "rank": rank,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "event_count": event_count,
        "execute_context_count": len(execute_contexts),
        "prefill": {
            "name": prefill_name,
            "tokens": prefill_tokens,
            "duration_ms": prefill_duration / 1000.0,
            "kernel_total_ms": kernel_total_us / 1000.0,
            "kernel_coverage": kernel_total_us / prefill_duration,
            "kernels": {
                name: {
                    "calls": int(row[0]),
                    "total_ms": row[1] / 1000.0,
                    "share_of_prefill": row[1] / prefill_duration,
                    "share_of_kernel_total": (
                        row[1] / kernel_total_us if kernel_total_us else 0.0
                    ),
                }
                for name, row in kernel_stats.items()
            },
            "nested_annotations": {
                name: {
                    "calls": int(row[0]),
                    "total_ms": row[1] / 1000.0,
                }
                for name, row in annotation_stats.items()
            },
        },
        "generation_duration_ms": _summarize(generation_durations_ms),
    }


def aggregate(traces: list[dict[str, Any]], top_kernels: int) -> dict[str, Any]:
    kernel_names = sorted(
        {name for trace in traces for name in trace["prefill"]["kernels"].keys()}
    )
    kernels = []
    for name in kernel_names:
        totals = [
            trace["prefill"]["kernels"].get(name, {}).get("total_ms", 0.0)
            for trace in traces
        ]
        calls = [
            float(trace["prefill"]["kernels"].get(name, {}).get("calls", 0))
            for trace in traces
        ]
        kernels.append(
            {
                "name": name,
                "total_ms": _summarize(totals),
                "calls": _summarize(calls),
            }
        )
    kernels.sort(key=lambda row: row["total_ms"]["median"], reverse=True)
    return {
        "rank_count": len(traces),
        "ranks": [trace["rank"] for trace in traces],
        "execute_context_count": _summarize(
            [float(trace["execute_context_count"]) for trace in traces]
        ),
        "prefill_duration_ms": _summarize(
            [trace["prefill"]["duration_ms"] for trace in traces]
        ),
        "prefill_kernel_total_ms": _summarize(
            [trace["prefill"]["kernel_total_ms"] for trace in traces]
        ),
        "prefill_kernel_coverage": _summarize(
            [trace["prefill"]["kernel_coverage"] for trace in traces]
        ),
        "top_kernels": kernels[:top_kernels],
    }


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    traces: list[dict[str, Any]] = []
    worker_count = min(args.workers, len(args.trace))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(analyze_trace, str(path)): path for path in args.trace
        }
        for future in as_completed(futures):
            result = future.result()
            traces.append(result)
            print(
                f"rank={result['rank']} "
                f"prefill_ms={result['prefill']['duration_ms']:.3f} "
                f"kernel_ms={result['prefill']['kernel_total_ms']:.3f}",
                flush=True,
            )
    traces.sort(key=lambda row: row["rank"])
    ranks = [trace["rank"] for trace in traces]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"duplicate ranks in traces: {ranks}")

    result = {
        "format_version": FORMAT_VERSION,
        "status": "passed",
        "scope": "stage9_first_prefill_trace_analysis",
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "python": platform.python_version(),
            "ijson": ijson.__version__,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "aggregate": aggregate(traces, args.top_kernels),
        "traces": traces,
    }
    atomic_write_json(args.output, result)
    print(f"result: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
