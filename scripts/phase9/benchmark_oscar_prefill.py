#!/usr/bin/env python3
"""Benchmark OSCAR prefill split counts and safe top-k tail cropping."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
import sys
import time
from typing import Any

import benchmark_oscar_mixed_splits as decode_bench


FORMAT_VERSION = 1
CONFIGS = [
    {"name": "full_topk_split16", "topk_width": 2048, "num_splits": 16},
    {"name": "full_topk_split1", "topk_width": 2048, "num_splits": 1},
    {"name": "cropped_topk_split16", "topk_width": 1024, "num_splits": 16},
    {"name": "cropped_topk_split8", "topk_width": 1024, "num_splits": 8},
    {"name": "cropped_topk_split4", "topk_width": 1024, "num_splits": 4},
    {"name": "cropped_topk_split2", "topk_width": 1024, "num_splits": 2},
    {"name": "cropped_topk_split1", "topk_width": 1024, "num_splits": 1},
]
BASELINE_CONFIG = "full_topk_split16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rotation-artifact",
        type=Path,
        default=Path("/opt/oscar_artifacts/rotation_fit_v2/rotations.pt"),
    )
    parser.add_argument("--rotation-layer", default="0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("warmup", "repeats", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    return args


def make_selected_tokens(torch: Any, seed: int, device: Any) -> Any:
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(
        decode_bench.SEQ_LEN,
        generator=cpu_generator,
        dtype=torch.int32,
    )
    selected = torch.full(
        (decode_bench.SEQ_LEN, decode_bench.TOPK),
        -1,
        dtype=torch.int32,
    )
    for query_position in range(decode_bench.SEQ_LEN):
        causal = permutation[permutation <= query_position]
        selected[query_position, : causal.numel()] = causal
    return selected.to(device=device)


def make_inputs(torch: Any, *, rotation: Any, seed: int, device: Any) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    history_tokens = (
        decode_bench.SEQ_LEN - decode_bench.PREFIX_TOKENS - decode_bench.RECENT_TOKENS
    )
    history_pages = history_tokens // decode_bench.BLOCK_SIZE
    rope_pages = decode_bench.SEQ_LEN // decode_bench.BLOCK_SIZE
    return {
        "query": torch.randn(
            decode_bench.SEQ_LEN,
            decode_bench.NUM_HEADS,
            decode_bench.LATENT_RANK,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "query_rope": torch.randn(
            decode_bench.SEQ_LEN,
            decode_bench.NUM_HEADS,
            decode_bench.ROPE_HEAD_SIZE,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "selected_tokens": make_selected_tokens(torch, seed, device),
        "query_request_indices": torch.zeros(
            decode_bench.SEQ_LEN,
            dtype=torch.int32,
            device=device,
        ),
        "query_positions": torch.arange(
            decode_bench.SEQ_LEN,
            dtype=torch.int32,
            device=device,
        ),
        "prefix": torch.randn(
            1,
            decode_bench.PREFIX_TOKENS,
            decode_bench.LATENT_RANK,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "recent": torch.randn(
            1,
            decode_bench.RECENT_TOKENS,
            decode_bench.LATENT_RANK,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "rope": torch.randn(
            rope_pages,
            decode_bench.BLOCK_SIZE,
            decode_bench.ROPE_HEAD_SIZE,
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
            (
                history_pages,
                decode_bench.BLOCK_SIZE,
                decode_bench.LATENT_RANK // 4,
            ),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        ),
        "history_scale": (
            torch.rand(
                history_pages,
                decode_bench.BLOCK_SIZE,
                decode_bench.LATENT_RANK // decode_bench.GROUP_SIZE,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.5
            + 0.5
        ),
        "history_zero": (
            torch.rand(
                history_pages,
                decode_bench.BLOCK_SIZE,
                decode_bench.LATENT_RANK // decode_bench.GROUP_SIZE,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 3.0
        ),
        "history_page_table": torch.arange(
            history_pages,
            dtype=torch.int32,
            device=device,
        ).unsqueeze(0),
        "hp_rows": torch.zeros(1, dtype=torch.int32, device=device),
        "seq_lens": torch.tensor(
            [decode_bench.SEQ_LEN],
            dtype=torch.int32,
            device=device,
        ),
        "rotation": rotation,
    }


def call_attention(
    function: Any,
    inputs: dict[str, Any],
    config: dict[str, Any],
) -> tuple[Any, Any]:
    selected = inputs["selected_tokens"][:, : config["topk_width"]]
    return function(
        inputs["query"],
        inputs["query_rope"],
        selected,
        inputs["query_request_indices"],
        inputs["query_positions"],
        inputs["prefix"],
        inputs["recent"],
        inputs["rope"],
        inputs["rope_block_table"],
        inputs["history_data"],
        inputs["history_scale"],
        inputs["history_zero"],
        inputs["history_page_table"],
        inputs["hp_rows"],
        inputs["seq_lens"],
        inputs["rotation"],
        attention_scale=decode_bench.ATTENTION_SCALE,
        num_splits=config["num_splits"],
    )


def main() -> int:
    args = parse_args()

    import torch

    from vllm.v1.attention.ops import triton_oscar_mla_decode

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
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    rotation = decode_bench.load_rotation(
        torch,
        args.rotation_artifact,
        args.rotation_layer,
        device,
    )
    inputs = make_inputs(torch, rotation=rotation, seed=args.seed, device=device)
    function = triton_oscar_mla_decode.oscar_mla_sparse_prefill

    outputs: dict[str, tuple[Any, Any]] = {}
    for config in CONFIGS:
        outputs[config["name"]] = call_attention(function, inputs, config)
    torch.cuda.synchronize(device)
    reference_output, reference_lse = outputs[BASELINE_CONFIG]
    correctness: dict[str, dict[str, Any]] = {}
    for name, (output, lse) in outputs.items():
        output_error = decode_bench.tensor_error(
            torch,
            output,
            reference_output,
        )
        lse_error = decode_bench.tensor_error(torch, lse, reference_lse)
        correct = bool(
            torch.isfinite(output).all()
            and torch.isfinite(lse).all()
            and torch.allclose(
                output,
                reference_output,
                atol=decode_bench.OUTPUT_ATOL,
                rtol=decode_bench.OUTPUT_RTOL,
            )
            and torch.allclose(
                lse,
                reference_lse,
                atol=decode_bench.LSE_ATOL,
                rtol=decode_bench.LSE_RTOL,
            )
        )
        correctness[name] = {
            "correct": correct,
            "output": output_error,
            "lse": lse_error,
        }
    if not all(row["correct"] for row in correctness.values()):
        raise RuntimeError(f"configuration correctness check failed: {correctness}")
    del outputs
    torch.cuda.empty_cache()

    for config in CONFIGS:
        for _ in range(args.warmup):
            call_attention(function, inputs, config)
        torch.cuda.synchronize(device)
        print(f"warmup complete: {config['name']}", flush=True)

    measurements: dict[str, dict[str, list[float]]] = {
        config["name"]: {
            "cuda_ms": [],
            "wall_ms": [],
            "peak_delta_allocated_mib": [],
        }
        for config in CONFIGS
    }
    benchmark_started = time.monotonic()
    last_heartbeat = benchmark_started
    for repeat in range(args.repeats):
        order = CONFIGS if repeat % 2 == 0 else list(reversed(CONFIGS))
        for config in order:
            torch.cuda.synchronize(device)
            baseline_allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start_event.record()
            for _ in range(args.iterations):
                call_attention(function, inputs, config)
            end_event.record()
            end_event.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000.0 / args.iterations
            cuda_ms = start_event.elapsed_time(end_event) / args.iterations
            peak_delta_mib = (
                torch.cuda.max_memory_allocated(device) - baseline_allocated
            ) / (1024**2)
            name = config["name"]
            measurements[name]["cuda_ms"].append(cuda_ms)
            measurements[name]["wall_ms"].append(wall_ms)
            measurements[name]["peak_delta_allocated_mib"].append(peak_delta_mib)
            print(
                f"repeat={repeat + 1}/{args.repeats} config={name} "
                f"cuda_ms={cuda_ms:.3f} wall_ms={wall_ms:.3f}",
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
            for metric, samples in metrics.items()
        }
        for name, metrics in measurements.items()
    }
    baseline_ms = results[BASELINE_CONFIG]["cuda_ms"]["median"]
    for name, row in results.items():
        row["speedup_vs_full_topk_split16"] = baseline_ms / row["cuda_ms"]["median"]
    winner = min(
        CONFIGS,
        key=lambda config: results[config["name"]]["cuda_ms"]["median"],
    )
    properties = torch.cuda.get_device_properties(device)
    result = {
        "format_version": FORMAT_VERSION,
        "status": "passed",
        "scope": "oscar_prefill_split_topk_microbenchmark",
        "fixed_gpu_count": 1,
        "shape": {
            "query_tokens": decode_bench.SEQ_LEN,
            "final_sequence_length": decode_bench.SEQ_LEN,
            "full_topk_width": decode_bench.TOPK,
            "cropped_topk_width": decode_bench.SEQ_LEN,
            "local_attention_heads": decode_bench.NUM_HEADS,
            "latent_rank": decode_bench.LATENT_RANK,
            "rope_head_size": decode_bench.ROPE_HEAD_SIZE,
            "prefix_tokens": decode_bench.PREFIX_TOKENS,
            "history_tokens": (
                decode_bench.SEQ_LEN
                - decode_bench.PREFIX_TOKENS
                - decode_bench.RECENT_TOKENS
            ),
            "recent_tokens": decode_bench.RECENT_TOKENS,
        },
        "measurement": {
            "configs": CONFIGS,
            "baseline_config": BASELINE_CONFIG,
            "warmup_per_config": args.warmup,
            "repeats": args.repeats,
            "iterations_per_repeat": args.iterations,
            "seed": args.seed,
            "elapsed_seconds": time.monotonic() - benchmark_started,
        },
        "correctness": {
            "output_atol": decode_bench.OUTPUT_ATOL,
            "output_rtol": decode_bench.OUTPUT_RTOL,
            "lse_atol": decode_bench.LSE_ATOL,
            "lse_rtol": decode_bench.LSE_RTOL,
            "by_config": correctness,
        },
        "results_by_config": results,
        "winner": {
            "name": winner["name"],
            "topk_width": winner["topk_width"],
            "num_splits": winner["num_splits"],
            "median_cuda_ms": results[winner["name"]]["cuda_ms"]["median"],
            "speedup_vs_full_topk_split16": results[winner["name"]][
                "speedup_vs_full_topk_split16"
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name,
            "gpu_compute_capability": [properties.major, properties.minor],
            "gpu_total_memory_bytes": properties.total_memory,
            "rotation_layer": args.rotation_layer,
            "rotation_artifact": str(args.rotation_artifact),
            "rotation_artifact_sha256": decode_bench.sha256_file(
                args.rotation_artifact
            ),
            "benchmark_script_sha256": decode_bench.sha256_file(Path(__file__)),
            "decode_benchmark_helper_sha256": decode_bench.sha256_file(
                Path(decode_bench.__file__)
            ),
            "kernel_source": str(Path(triton_oscar_mla_decode.__file__).resolve()),
            "kernel_source_sha256": decode_bench.sha256_file(
                Path(triton_oscar_mla_decode.__file__).resolve()
            ),
        },
    }
    decode_bench.atomic_write_json(args.output, result)
    print(
        f"winner: config={winner['name']} "
        f"median_cuda_ms={result['winner']['median_cuda_ms']:.3f} "
        f"speedup={result['winner']['speedup_vs_full_topk_split16']:.3f}x",
        flush=True,
    )
    print(f"result: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
