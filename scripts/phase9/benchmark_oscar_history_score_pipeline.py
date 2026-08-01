#!/usr/bin/env python3
"""Screen the standalone history score/LSE/value pipeline on a 32K final chunk."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import benchmark_oscar_history_manual_value as manual_bench
import benchmark_oscar_mixed_splits as decode_bench
import compile_oscar_history_score_pipeline as pipeline

FORMAT_VERSION = 1
EXPECTED_SOURCE_COMMIT = pipeline.EXPECTED_SOURCE_COMMIT
TOPK = manual_bench.TOPK
NUM_HEADS = manual_bench.NUM_HEADS
LATENT_RANK = manual_bench.LATENT_RANK
ROPE_HEAD_SIZE = manual_bench.ROPE_HEAD_SIZE
PREFIX_TOKENS = manual_bench.PREFIX_TOKENS
RECENT_TOKENS = manual_bench.RECENT_TOKENS
BLOCK_SIZE = manual_bench.BLOCK_SIZE
GROUP_SIZE = manual_bench.GROUP_SIZE
ATTENTION_SCALE = manual_bench.ATTENTION_SCALE
REFERENCE_NAME = "history_h8_t16_w8_dot_reference"
CANDIDATE_NAME = "history_score_h4_lse_value_h2_pipeline_candidate"
REFERENCE = manual_bench.VARIANTS[0]
CANDIDATE = {
    "name": CANDIDATE_NAME,
    "score_block_h": 4,
    "score_block_t": 16,
    "score_num_warps": 4,
    "lse_block_tiles": 128,
    "lse_num_warps": 4,
    "value_block_h": 2,
    "value_block_t": 16,
    "value_block_dv": 128,
    "value_num_warps": 4,
}
VARIANT_NAMES = (REFERENCE_NAME, CANDIDATE_NAME)


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


def make_pipeline_buffers(torch: Any, query_tokens: int, device: Any) -> dict[str, Any]:
    num_tiles = TOPK // CANDIDATE["score_block_t"]
    return {
        "scores": torch.empty(
            query_tokens,
            NUM_HEADS,
            TOPK,
            dtype=torch.float32,
            device=device,
        ),
        "tile_lse": torch.empty(
            query_tokens,
            NUM_HEADS,
            num_tiles,
            dtype=torch.float32,
            device=device,
        ),
        "output": torch.empty(
            query_tokens,
            NUM_HEADS,
            LATENT_RANK,
            dtype=torch.float32,
            device=device,
        ),
        "lse": torch.empty(
            query_tokens,
            NUM_HEADS,
            dtype=torch.float32,
            device=device,
        ),
    }


def launch_pipeline(
    triton: Any,
    inputs: dict[str, Any],
    buffers: dict[str, Any],
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
    scores = buffers["scores"]
    tile_lse = buffers["tile_lse"]
    final_lse = buffers["lse"]
    output = buffers["output"]
    query_tokens = query_rotated.shape[0]
    num_tiles = TOPK // CANDIDATE["score_block_t"]

    score_grid = (
        query_tokens,
        triton.cdiv(NUM_HEADS, CANDIDATE["score_block_h"]),
        num_tiles,
    )
    pipeline._history_score_kernel[score_grid](
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
        scores,
        tile_lse,
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
        stride_score_b=scores.stride(0),
        stride_score_h=scores.stride(1),
        stride_score_k=scores.stride(2),
        stride_tile_lse_b=tile_lse.stride(0),
        stride_tile_lse_h=tile_lse.stride(1),
        stride_tile_lse_tile=tile_lse.stride(2),
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
        block_h=CANDIDATE["score_block_h"],
        block_t=CANDIDATE["score_block_t"],
        block_d=LATENT_RANK,
        block_r=ROPE_HEAD_SIZE,
        num_warps=CANDIDATE["score_num_warps"],
        num_stages=1,
    )

    lse_grid = (query_tokens, NUM_HEADS)
    pipeline._history_lse_kernel[lse_grid](
        tile_lse,
        final_lse,
        stride_tile_lse_b=tile_lse.stride(0),
        stride_tile_lse_h=tile_lse.stride(1),
        stride_tile_lse_tile=tile_lse.stride(2),
        stride_final_lse_b=final_lse.stride(0),
        stride_final_lse_h=final_lse.stride(1),
        num_heads=NUM_HEADS,
        num_tiles=num_tiles,
        block_tiles=CANDIDATE["lse_block_tiles"],
        num_warps=CANDIDATE["lse_num_warps"],
        num_stages=1,
    )

    value_grid = (
        query_tokens,
        triton.cdiv(NUM_HEADS, CANDIDATE["value_block_h"]),
        triton.cdiv(LATENT_RANK, CANDIDATE["value_block_dv"]),
    )
    pipeline._history_value_kernel[value_grid](
        selected_tokens,
        query_request_indices,
        query_positions,
        history_data,
        history_scale,
        history_zero,
        history_page_table,
        hp_rows,
        seq_lens,
        scores,
        final_lse,
        output,
        stride_selected_b=selected_tokens.stride(0),
        stride_selected_k=selected_tokens.stride(1),
        stride_query_request=query_request_indices.stride(0),
        stride_query_position=query_positions.stride(0),
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
        stride_score_b=scores.stride(0),
        stride_score_h=scores.stride(1),
        stride_score_k=scores.stride(2),
        stride_final_lse_b=final_lse.stride(0),
        stride_final_lse_h=final_lse.stride(1),
        stride_output_b=output.stride(0),
        stride_output_h=output.stride(1),
        stride_output_d=output.stride(2),
        topk=TOPK,
        prefix_tokens=PREFIX_TOKENS,
        recent_tokens=RECENT_TOKENS,
        history_block_size=BLOCK_SIZE,
        latent_rank=LATENT_RANK,
        group_size=GROUP_SIZE,
        num_requests=1,
        num_heads=NUM_HEADS,
        block_h=CANDIDATE["value_block_h"],
        block_t=CANDIDATE["value_block_t"],
        block_dv=CANDIDATE["value_block_dv"],
        num_warps=CANDIDATE["value_num_warps"],
        num_stages=1,
    )


def launch_variant(
    triton: Any,
    inputs: dict[str, Any],
    reference_buffers: tuple[Any, Any],
    candidate_buffers: dict[str, Any],
    name: str,
) -> None:
    if name == REFERENCE_NAME:
        manual_bench.launch_history(
            triton,
            inputs,
            reference_buffers[0],
            reference_buffers[1],
            REFERENCE,
        )
        return
    if name == CANDIDATE_NAME:
        launch_pipeline(triton, inputs, candidate_buffers)
        return
    raise ValueError(f"unknown variant: {name}")


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
    inputs = manual_bench.make_inputs(
        torch,
        query_tokens=args.query_tokens,
        final_seq_len=args.final_seq_len,
        seed=args.seed,
        device=device,
    )
    reference_buffers = manual_bench.make_output_buffers(
        torch,
        args.query_tokens,
        device,
    )
    candidate_buffers = make_pipeline_buffers(torch, args.query_tokens, device)

    for name in VARIANT_NAMES:
        launch_variant(
            triton,
            inputs,
            reference_buffers,
            candidate_buffers,
            name,
        )
    torch.cuda.synchronize(device)
    reference_output, reference_lse = reference_buffers
    candidate_output = candidate_buffers["output"]
    candidate_lse = candidate_buffers["lse"]
    output_error = decode_bench.tensor_error(torch, candidate_output, reference_output)
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
                "scope": "oscar_history_score_pipeline_microbenchmark",
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
        raise RuntimeError(f"score pipeline correctness failed: {correctness}")

    for name in VARIANT_NAMES:
        for _ in range(args.warmup):
            launch_variant(
                triton,
                inputs,
                reference_buffers,
                candidate_buffers,
                name,
            )
        torch.cuda.synchronize(device)
        print(f"warmup complete: {name}", flush=True)

    measurements = {name: {"cuda_ms": [], "wall_ms": []} for name in VARIANT_NAMES}
    benchmark_started = time.monotonic()
    last_heartbeat = benchmark_started
    for repeat in range(args.repeats):
        order = VARIANT_NAMES if repeat % 2 == 0 else tuple(reversed(VARIANT_NAMES))
        for name in order:
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_started = time.perf_counter()
            start_event.record()
            for _ in range(args.iterations):
                launch_variant(
                    triton,
                    inputs,
                    reference_buffers,
                    candidate_buffers,
                    name,
                )
            end_event.record()
            end_event.synchronize()
            cuda_ms = start_event.elapsed_time(end_event) / args.iterations
            wall_ms = (time.perf_counter() - wall_started) * 1000.0 / args.iterations
            measurements[name]["cuda_ms"].append(cuda_ms)
            measurements[name]["wall_ms"].append(wall_ms)
            print(
                f"repeat={repeat + 1}/{args.repeats} variant={name} "
                f"cuda_ms={cuda_ms:.6f} wall_ms={wall_ms:.6f}",
                flush=True,
            )
            now = time.monotonic()
            if now - last_heartbeat >= 600:
                print(
                    f"10-minute progress: elapsed_seconds={now - benchmark_started:.1f}",
                    flush=True,
                )
                last_heartbeat = now

    results = {
        name: {
            metric: decode_bench.summarize(samples)
            for metric, samples in variant_measurements.items()
        }
        for name, variant_measurements in measurements.items()
    }
    reference_ms = results[REFERENCE_NAME]["cuda_ms"]["median"]
    candidate_ms = results[CANDIDATE_NAME]["cuda_ms"]["median"]
    faster = candidate_is_faster(candidate_ms, reference_ms)
    properties = torch.cuda.get_device_properties(device)
    result = {
        "format_version": FORMAT_VERSION,
        "status": "passed",
        "scope": "oscar_history_score_pipeline_microbenchmark",
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
        "scratch": pipeline.scratch_layout_bytes(
            query_tokens=args.query_tokens,
            num_heads=NUM_HEADS,
            topk=TOPK,
            block_t=CANDIDATE["score_block_t"],
        ),
        "measurement": {
            "warmup_per_variant": args.warmup,
            "repeats": args.repeats,
            "iterations_per_repeat": args.iterations,
            "seed": args.seed,
            "elapsed_seconds": time.monotonic() - benchmark_started,
            "reference": REFERENCE,
            "candidate": CANDIDATE,
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
            "pipeline_script_sha256": decode_bench.sha256_file(Path(pipeline.__file__)),
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
