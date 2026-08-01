#!/usr/bin/env python3
"""Screen standalone history manual value reduction at the 32K final chunk."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import benchmark_oscar_mixed_splits as decode_bench
import compile_oscar_prefill_cache_split as resource_screen

FORMAT_VERSION = 1
EXPECTED_SOURCE_COMMIT = "67a0e47ff72f10a322de17b81c4134984e017bd6"
TOPK = decode_bench.TOPK
NUM_HEADS = decode_bench.NUM_HEADS
LATENT_RANK = decode_bench.LATENT_RANK
ROPE_HEAD_SIZE = decode_bench.ROPE_HEAD_SIZE
PREFIX_TOKENS = decode_bench.PREFIX_TOKENS
RECENT_TOKENS = decode_bench.RECENT_TOKENS
BLOCK_SIZE = decode_bench.BLOCK_SIZE
GROUP_SIZE = decode_bench.GROUP_SIZE
ATTENTION_SCALE = decode_bench.ATTENTION_SCALE
REFERENCE_NAME = "history_h8_t16_w8_dot_reference"
CANDIDATE_NAME = "history_h4_t8_w4_manual_candidate"
VARIANTS = [
    {
        "name": REFERENCE_NAME,
        "block_h": 8,
        "block_t": 16,
        "num_warps": 8,
        "manual_history_value_reduce": False,
    },
    {
        "name": CANDIDATE_NAME,
        "block_h": 4,
        "block_t": 8,
        "num_warps": 4,
        "manual_history_value_reduce": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-seq-len", type=int, default=32_768)
    parser.add_argument("--query-tokens", type=int, default=2_048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-source-commit",
        default=EXPECTED_SOURCE_COMMIT,
    )
    args = parser.parse_args()
    for name in ("final_seq_len", "query_tokens", "warmup", "repeats", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.query_tokens > args.final_seq_len:
        parser.error("--query-tokens must not exceed --final-seq-len")
    history_tokens = args.final_seq_len - PREFIX_TOKENS - RECENT_TOKENS
    if history_tokens <= 0 or history_tokens % BLOCK_SIZE:
        parser.error(
            "--final-seq-len history region must be positive and block aligned"
        )
    earliest_query_position = args.final_seq_len - args.query_tokens
    if earliest_query_position + 1 - PREFIX_TOKENS < TOPK:
        parser.error(
            "earliest query position must contain at least TOPK history tokens"
        )
    if args.expected_source_commit != EXPECTED_SOURCE_COMMIT:
        parser.error("--expected-source-commit does not match the frozen source")
    return args


def candidate_is_faster(candidate_ms: float, reference_ms: float) -> bool:
    return candidate_ms < reference_ms


def make_history_selected_tokens(
    torch: Any,
    *,
    query_tokens: int,
    final_seq_len: int,
    seed: int,
    device: Any,
) -> Any:
    earliest_query_position = final_seq_len - query_tokens
    history_end = min(final_seq_len - RECENT_TOKENS, earliest_query_position + 1)
    history_width = history_end - PREFIX_TOKENS
    if history_width < TOPK:
        raise ValueError("history region is smaller than TOPK")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected_row = (
        torch.randperm(history_width, generator=generator, dtype=torch.int32)[:TOPK]
        + PREFIX_TOKENS
    )
    return selected_row.unsqueeze(0).expand(query_tokens, -1).contiguous().to(device)


def make_inputs(
    torch: Any,
    *,
    query_tokens: int,
    final_seq_len: int,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    history_tokens = final_seq_len - PREFIX_TOKENS - RECENT_TOKENS
    history_pages = history_tokens // BLOCK_SIZE
    rope_pages = final_seq_len // BLOCK_SIZE
    return {
        "query_rotated": torch.randn(
            query_tokens,
            NUM_HEADS,
            LATENT_RANK,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        "query_rope": torch.randn(
            query_tokens,
            NUM_HEADS,
            ROPE_HEAD_SIZE,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "selected_tokens": make_history_selected_tokens(
            torch,
            query_tokens=query_tokens,
            final_seq_len=final_seq_len,
            seed=seed,
            device=device,
        ),
        "query_request_indices": torch.zeros(
            query_tokens,
            dtype=torch.int32,
            device=device,
        ),
        "query_positions": torch.arange(
            final_seq_len - query_tokens,
            final_seq_len,
            dtype=torch.int32,
            device=device,
        ),
        "rope": torch.randn(
            rope_pages,
            BLOCK_SIZE,
            ROPE_HEAD_SIZE,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "rope_block_table": torch.arange(
            rope_pages,
            dtype=torch.int32,
            device=device,
        ).unsqueeze(0),
        "history_data": torch.randint(
            0,
            256,
            (history_pages, BLOCK_SIZE, LATENT_RANK // 4),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        ),
        "history_scale": torch.rand(
            history_pages,
            BLOCK_SIZE,
            LATENT_RANK // GROUP_SIZE,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        "history_zero": torch.rand(
            history_pages,
            BLOCK_SIZE,
            LATENT_RANK // GROUP_SIZE,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        "history_page_table": torch.arange(
            history_pages,
            dtype=torch.int32,
            device=device,
        ).unsqueeze(0),
        "hp_rows": torch.zeros(1, dtype=torch.int32, device=device),
        "seq_lens": torch.tensor([final_seq_len], dtype=torch.int32, device=device),
    }


def make_output_buffers(torch: Any, query_tokens: int, device: Any) -> tuple[Any, Any]:
    output = torch.empty(
        query_tokens,
        NUM_HEADS,
        LATENT_RANK,
        dtype=torch.float32,
        device=device,
    )
    lse = torch.empty(
        query_tokens,
        NUM_HEADS,
        dtype=torch.float32,
        device=device,
    )
    return output, lse


def launch_history(
    triton: Any,
    inputs: dict[str, Any],
    output: Any,
    lse: Any,
    variant: dict[str, Any],
) -> None:
    query_rotated = inputs["query_rotated"]
    query_rope = inputs["query_rope"]
    selected_tokens = inputs["selected_tokens"]
    query_request_indices = inputs["query_request_indices"]
    query_positions = inputs["query_positions"]
    rope = inputs["rope"]
    rope_block_table = inputs["rope_block_table"]
    history_data = inputs["history_data"]
    history_scale = inputs["history_scale"]
    history_zero = inputs["history_zero"]
    history_page_table = inputs["history_page_table"]
    hp_rows = inputs["hp_rows"]
    seq_lens = inputs["seq_lens"]
    grid = (query_rotated.shape[0], triton.cdiv(NUM_HEADS, variant["block_h"]))
    resource_screen._history_prefill_stage1[grid](
        query_rotated,
        query_rope,
        selected_tokens,
        query_request_indices,
        query_positions,
        rope,
        rope_block_table,
        history_data,
        history_scale,
        history_zero,
        history_page_table,
        hp_rows,
        seq_lens,
        output,
        lse,
        stride_query_rotated_b=query_rotated.stride(0),
        stride_query_rotated_h=query_rotated.stride(1),
        stride_query_rotated_d=query_rotated.stride(2),
        stride_query_rope_b=query_rope.stride(0),
        stride_query_rope_h=query_rope.stride(1),
        stride_query_rope_d=query_rope.stride(2),
        stride_selected_b=selected_tokens.stride(0),
        stride_selected_k=selected_tokens.stride(1),
        stride_query_request=query_request_indices.stride(0),
        stride_query_position=query_positions.stride(0),
        stride_rope_block=rope.stride(0),
        stride_rope_token=rope.stride(1),
        stride_rope_d=rope.stride(2),
        stride_rope_block_table_b=rope_block_table.stride(0),
        stride_rope_block_table_page=rope_block_table.stride(1),
        stride_data_page=history_data.stride(0),
        stride_data_token=history_data.stride(1),
        stride_data_byte=history_data.stride(2),
        stride_scale_page=history_scale.stride(0),
        stride_scale_token=history_scale.stride(1),
        stride_scale_group=history_scale.stride(2),
        stride_zero_page=history_zero.stride(0),
        stride_zero_token=history_zero.stride(1),
        stride_zero_group=history_zero.stride(2),
        stride_page_table_b=history_page_table.stride(0),
        stride_page_table_page=history_page_table.stride(1),
        stride_hp_rows=hp_rows.stride(0),
        stride_seq_lens=seq_lens.stride(0),
        stride_mid_b=output.stride(0),
        stride_mid_h=output.stride(1),
        stride_mid_d=output.stride(2),
        stride_lse_b=lse.stride(0),
        stride_lse_h=lse.stride(1),
        topk=TOPK,
        prefix_tokens=PREFIX_TOKENS,
        recent_tokens=RECENT_TOKENS,
        rope_block_size=BLOCK_SIZE,
        rope_head_size=ROPE_HEAD_SIZE,
        history_block_size=BLOCK_SIZE,
        latent_rank=LATENT_RANK,
        group_size=GROUP_SIZE,
        attention_scale=ATTENTION_SCALE,
        num_requests=1,
        num_heads=NUM_HEADS,
        block_h=variant["block_h"],
        block_t=variant["block_t"],
        block_d=LATENT_RANK,
        block_r=ROPE_HEAD_SIZE,
        reload_history_for_value=False,
        manual_history_value_reduce=variant["manual_history_value_reduce"],
        num_warps=variant["num_warps"],
        num_stages=1,
    )


def main() -> int:
    args = parse_args()

    import torch
    import triton

    runtime_identity = {
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    expected_runtime_identity = {
        "python_executable": decode_bench.EXPECTED_PYTHON,
        "torch": decode_bench.EXPECTED_TORCH,
        "cuda_runtime": decode_bench.EXPECTED_CUDA_RUNTIME,
    }
    if runtime_identity != expected_runtime_identity:
        raise RuntimeError(
            "benchmark runtime identity mismatch: "
            f"actual={runtime_identity}, expected={expected_runtime_identity}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one visible CUDA GPU")
    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    inputs = make_inputs(
        torch,
        query_tokens=args.query_tokens,
        final_seq_len=args.final_seq_len,
        seed=args.seed,
        device=device,
    )
    buffers = {
        variant["name"]: make_output_buffers(torch, args.query_tokens, device)
        for variant in VARIANTS
    }

    for variant in VARIANTS:
        output, lse = buffers[variant["name"]]
        launch_history(triton, inputs, output, lse, variant)
    torch.cuda.synchronize(device)
    reference_output, reference_lse = buffers[REFERENCE_NAME]
    candidate_output, candidate_lse = buffers[CANDIDATE_NAME]
    output_error = decode_bench.tensor_error(
        torch,
        candidate_output,
        reference_output,
    )
    lse_error = decode_bench.tensor_error(torch, candidate_lse, reference_lse)
    correctness_passed = bool(
        torch.isfinite(candidate_output).all()
        and torch.isfinite(candidate_lse).all()
        and torch.allclose(
            candidate_output,
            reference_output,
            atol=decode_bench.OUTPUT_ATOL,
            rtol=decode_bench.OUTPUT_RTOL,
        )
        and torch.allclose(
            candidate_lse,
            reference_lse,
            atol=decode_bench.LSE_ATOL,
            rtol=decode_bench.LSE_RTOL,
        )
    )
    correctness = {
        "passed": correctness_passed,
        "output": output_error,
        "lse": lse_error,
        "output_atol": decode_bench.OUTPUT_ATOL,
        "output_rtol": decode_bench.OUTPUT_RTOL,
        "lse_atol": decode_bench.LSE_ATOL,
        "lse_rtol": decode_bench.LSE_RTOL,
    }
    if not correctness_passed:
        decode_bench.atomic_write_json(
            args.output,
            {
                "format_version": FORMAT_VERSION,
                "status": "correctness_failed",
                "scope": "oscar_history_manual_value_microbenchmark",
                "fixed_gpu_count": 1,
                "shape": {
                    "batch_size": 1,
                    "final_sequence_length": args.final_seq_len,
                    "query_tokens": args.query_tokens,
                    "topk": TOPK,
                },
                "correctness": correctness,
            },
        )
        raise RuntimeError(f"manual value correctness failed: {correctness}")

    for variant in VARIANTS:
        output, lse = buffers[variant["name"]]
        for _ in range(args.warmup):
            launch_history(triton, inputs, output, lse, variant)
        torch.cuda.synchronize(device)
        print(f"warmup complete: {variant['name']}", flush=True)

    measurements = {
        variant["name"]: {"cuda_ms": [], "wall_ms": []} for variant in VARIANTS
    }
    benchmark_started = time.monotonic()
    last_heartbeat = benchmark_started
    for repeat in range(args.repeats):
        order = VARIANTS if repeat % 2 == 0 else list(reversed(VARIANTS))
        for variant in order:
            output, lse = buffers[variant["name"]]
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_started = time.perf_counter()
            start_event.record()
            for _ in range(args.iterations):
                launch_history(triton, inputs, output, lse, variant)
            end_event.record()
            end_event.synchronize()
            cuda_ms = start_event.elapsed_time(end_event) / args.iterations
            wall_ms = (time.perf_counter() - wall_started) * 1000.0 / args.iterations
            measurements[variant["name"]]["cuda_ms"].append(cuda_ms)
            measurements[variant["name"]]["wall_ms"].append(wall_ms)
            print(
                f"repeat={repeat + 1}/{args.repeats} "
                f"variant={variant['name']} cuda_ms={cuda_ms:.6f} "
                f"wall_ms={wall_ms:.6f}",
                flush=True,
            )
            now = time.monotonic()
            if now - last_heartbeat >= 600:
                elapsed_seconds = now - benchmark_started
                print(
                    f"10-minute progress: elapsed_seconds={elapsed_seconds:.1f}",
                    flush=True,
                )
                last_heartbeat = now

    results = {
        variant_name: {
            metric: decode_bench.summarize(samples)
            for metric, samples in variant_measurements.items()
        }
        for variant_name, variant_measurements in measurements.items()
    }
    reference_ms = results[REFERENCE_NAME]["cuda_ms"]["median"]
    candidate_ms = results[CANDIDATE_NAME]["cuda_ms"]["median"]
    faster = candidate_is_faster(candidate_ms, reference_ms)
    properties = torch.cuda.get_device_properties(device)
    result = {
        "format_version": FORMAT_VERSION,
        "status": "passed",
        "scope": "oscar_history_manual_value_microbenchmark",
        "fixed_gpu_count": 1,
        "expected_source_commit": args.expected_source_commit,
        "shape": {
            "batch_size": 1,
            "final_sequence_length": args.final_seq_len,
            "query_tokens": args.query_tokens,
            "topk": TOPK,
            "local_attention_heads": NUM_HEADS,
            "latent_rank": LATENT_RANK,
            "rope_head_size": ROPE_HEAD_SIZE,
            "prefix_tokens": PREFIX_TOKENS,
            "history_tokens": args.final_seq_len - PREFIX_TOKENS - RECENT_TOKENS,
            "recent_tokens": RECENT_TOKENS,
        },
        "measurement": {
            "warmup_per_variant": args.warmup,
            "repeats": args.repeats,
            "iterations_per_repeat": args.iterations,
            "seed": args.seed,
            "elapsed_seconds": time.monotonic() - benchmark_started,
            "variants": VARIANTS,
        },
        "correctness": correctness,
        "results_by_variant": results,
        "comparison": {
            "reference": REFERENCE_NAME,
            "candidate": CANDIDATE_NAME,
            "reference_median_cuda_ms": reference_ms,
            "candidate_median_cuda_ms": candidate_ms,
            "candidate_delta_percent": (candidate_ms / reference_ms - 1.0) * 100.0,
            "candidate_speedup": reference_ms / candidate_ms,
            "candidate_is_faster": faster,
            "promotion_eligible": correctness_passed and faster,
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "triton": triton.__version__,
            "gpu_name": properties.name,
            "gpu_compute_capability": [properties.major, properties.minor],
            "gpu_total_memory_bytes": properties.total_memory,
            "benchmark_script_sha256": decode_bench.sha256_file(Path(__file__)),
            "resource_screen_script_sha256": decode_bench.sha256_file(
                Path(resource_screen.__file__)
            ),
        },
    }
    decode_bench.atomic_write_json(args.output, result)
    print(
        f"candidate={CANDIDATE_NAME} median_cuda_ms={candidate_ms:.6f} "
        f"reference_cuda_ms={reference_ms:.6f} "
        f"delta_percent={result['comparison']['candidate_delta_percent']:.6f} "
        f"promotion_eligible={result['comparison']['promotion_eligible']}",
        flush=True,
    )
    print(f"result: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
