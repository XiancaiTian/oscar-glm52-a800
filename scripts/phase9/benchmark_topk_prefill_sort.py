#!/usr/bin/env python3
"""Benchmark native prefill top-k with and without selected-index sorting."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
PREFILL_SORT_ENV = "VLLM_TOPK_PREFILL_SORT_INDICES"
TOPK_ENV_CACHE = "VLLM_TOPK_ENV_CACHE"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-tokens", type=int, default=2048)
    parser.add_argument("--final-seq-len", type=int, default=32768)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for name in (
        "query_tokens",
        "final_seq_len",
        "topk",
        "warmup",
        "repeats",
        "iterations",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.query_tokens > args.final_seq_len:
        parser.error("--query-tokens must be no greater than --final-seq-len")
    if args.topk > min(args.final_seq_len, 2048):
        parser.error("--topk must be no greater than min(final sequence length, 2048)")
    return args


def make_row_bounds(
    torch: Any,
    *,
    query_tokens: int,
    final_seq_len: int,
    device: Any,
) -> tuple[Any, Any]:
    query_start = final_seq_len - query_tokens
    row_starts = torch.zeros(query_tokens, dtype=torch.int32, device=device)
    row_ends = torch.arange(
        query_start + 1,
        final_seq_len + 1,
        dtype=torch.int32,
        device=device,
    )
    return row_starts, row_ends


def summarize_samples(samples_ms: list[float], *, iterations: int) -> dict[str, Any]:
    per_call = [sample / iterations for sample in samples_ms]
    return {
        "samples_ms": per_call,
        "median_ms": statistics.median(per_call),
        "mean_ms": statistics.fmean(per_call),
        "min_ms": min(per_call),
        "max_ms": max(per_call),
    }


def validate_index_outputs(torch: Any, raw: Any, sorted_indices: Any) -> dict[str, Any]:
    raw_sorted = torch.sort(raw, dim=1).values
    same_set = bool(torch.equal(raw_sorted, sorted_indices))
    if not same_set:
        raise AssertionError("selected index set changed after native sorting")
    monotonic = bool((sorted_indices[:, 1:] >= sorted_indices[:, :-1]).all().item())
    if not monotonic:
        raise AssertionError("sorted native indices are not monotonic")
    invalid_count = int((sorted_indices < 0).sum().item())
    if invalid_count:
        raise AssertionError("unexpected invalid selected indices in final 32K chunk")
    return {
        "status": "passed",
        "same_selected_set_per_row": same_set,
        "sorted_indices_monotonic_per_row": monotonic,
        "invalid_index_count": invalid_count,
    }


def call_topk(torch: Any, inputs: dict[str, Any], indices: Any) -> None:
    logits = inputs["logits"]
    torch.ops._C.top_k_per_row_prefill(
        logits,
        inputs["row_starts"],
        inputs["row_ends"],
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        indices.shape[1],
    )


def benchmark_mode(
    torch: Any,
    inputs: dict[str, Any],
    indices: Any,
    *,
    sort_indices: bool,
    warmup: int,
    repeats: int,
    iterations: int,
) -> dict[str, Any]:
    os.environ[PREFILL_SORT_ENV] = "1" if sort_indices else "0"
    for _ in range(warmup):
        call_topk(torch, inputs, indices)
    torch.cuda.synchronize()

    cuda_samples_ms: list[float] = []
    wall_samples_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        for _ in range(iterations):
            call_topk(torch, inputs, indices)
        end.record()
        end.synchronize()
        wall_samples_ms.append((time.perf_counter() - wall_start) * 1000.0)
        cuda_samples_ms.append(float(start.elapsed_time(end)))

    return {
        "sort_indices": sort_indices,
        "warmup_calls": warmup,
        "measured_samples": repeats,
        "iterations_per_sample": iterations,
        "cuda": summarize_samples(cuda_samples_ms, iterations=iterations),
        "wall": summarize_samples(wall_samples_ms, iterations=iterations),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ[TOPK_ENV_CACHE] = "0"

    torch = importlib.import_module("torch")
    importlib.import_module("vllm._C")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    logits = torch.randn(
        (args.query_tokens, args.final_seq_len),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    row_starts, row_ends = make_row_bounds(
        torch,
        query_tokens=args.query_tokens,
        final_seq_len=args.final_seq_len,
        device=device,
    )
    inputs = {
        "logits": logits,
        "row_starts": row_starts,
        "row_ends": row_ends,
    }
    raw_indices = torch.empty(
        (args.query_tokens, args.topk),
        dtype=torch.int32,
        device=device,
    )
    sorted_indices = torch.empty_like(raw_indices)

    raw_timing = benchmark_mode(
        torch,
        inputs,
        raw_indices,
        sort_indices=False,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    sorted_timing = benchmark_mode(
        torch,
        inputs,
        sorted_indices,
        sort_indices=True,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    correctness = validate_index_outputs(torch, raw_indices, sorted_indices)

    raw_cuda = raw_timing["cuda"]["median_ms"]
    sorted_cuda = sorted_timing["cuda"]["median_ms"]
    raw_wall = raw_timing["wall"]["median_ms"]
    sorted_wall = sorted_timing["wall"]["median_ms"]
    payload = {
        "format_version": FORMAT_VERSION,
        "status": "passed",
        "system": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "environment": {
            TOPK_ENV_CACHE: os.environ[TOPK_ENV_CACHE],
            "comparison_variable": PREFILL_SORT_ENV,
        },
        "workload": {
            "query_tokens": args.query_tokens,
            "query_start_position": args.final_seq_len - args.query_tokens,
            "final_sequence_length": args.final_seq_len,
            "row_end_min": int(row_ends[0].item()),
            "row_end_max": int(row_ends[-1].item()),
            "topk_width": args.topk,
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "seed": args.seed,
        },
        "unsorted": raw_timing,
        "sorted": sorted_timing,
        "sorted_minus_unsorted": {
            "median_cuda_ms": sorted_cuda - raw_cuda,
            "median_cuda_percent": (sorted_cuda / raw_cuda - 1.0) * 100.0,
            "median_wall_ms": sorted_wall - raw_wall,
            "median_wall_percent": (sorted_wall / raw_wall - 1.0) * 100.0,
        },
        "correctness": correctness,
        "interpretation_boundary": (
            "This benchmark measures only native top_k_per_row_prefill on one "
            "synthetic final 2K chunk. It does not measure stage1, TTFT, TPOT, "
            "throughput, or formal DSA logits."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
