#!/usr/bin/env python3
"""Benchmark OSCAR mixed-attention split counts at the Stage 9 1K/batch1 shape."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time
from typing import Any


FORMAT_VERSION = 1
SEQ_LEN = 1024
TOPK = 2048
NUM_HEADS = 8
LATENT_RANK = 512
ROPE_HEAD_SIZE = 64
PREFIX_TOKENS = 64
RECENT_TOKENS = 256
BLOCK_SIZE = 16
GROUP_SIZE = 128
ATTENTION_SCALE = 576**-0.5
OUTPUT_ATOL = 2e-3
OUTPUT_RTOL = 2e-3
LSE_ATOL = 2e-3
LSE_RTOL = 2e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rotation-artifact",
        type=Path,
        default=Path("/opt/oscar_artifacts/rotation_fit_v2/rotations.pt"),
    )
    parser.add_argument("--rotation-layer", default="0")
    parser.add_argument("--splits", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(set(args.splits)) != len(args.splits):
        parser.error("--splits must not contain duplicates")
    if not args.splits or any(value < 1 or value > 32 for value in args.splits):
        parser.error("--splits values must be in [1, 32]")
    for name in ("warmup", "repeats", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
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


def load_rotation(torch: Any, path: Path, layer: str, device: Any) -> Any:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("rotations"), dict):
        raise ValueError("rotation artifact must contain a rotations mapping")
    rotations = payload["rotations"]
    if layer not in rotations:
        raise ValueError(f"rotation artifact does not contain layer {layer}")
    rotation = rotations[layer]
    if tuple(rotation.shape) != (LATENT_RANK, LATENT_RANK):
        raise ValueError(
            f"layer {layer} rotation shape is {tuple(rotation.shape)}, "
            f"expected {(LATENT_RANK, LATENT_RANK)}"
        )
    return rotation.contiguous().to(device=device, dtype=torch.float32)


def make_inputs(torch: Any, *, rotation: Any, seed: int, device: Any) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    history_tokens = SEQ_LEN - PREFIX_TOKENS - RECENT_TOKENS
    history_pages = history_tokens // BLOCK_SIZE
    rope_pages = SEQ_LEN // BLOCK_SIZE

    selected = torch.full((1, TOPK), -1, dtype=torch.int32, device=device)
    selected[0, :SEQ_LEN] = torch.randperm(
        SEQ_LEN,
        dtype=torch.int64,
        device=device,
        generator=generator,
    ).to(torch.int32)
    return {
        "query": torch.randn(
            1,
            NUM_HEADS,
            LATENT_RANK,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "query_rope": torch.randn(
            1,
            NUM_HEADS,
            ROPE_HEAD_SIZE,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "selected_tokens": selected,
        "prefix": torch.randn(
            1,
            PREFIX_TOKENS,
            LATENT_RANK,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "recent": torch.randn(
            1,
            RECENT_TOKENS,
            LATENT_RANK,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
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
        "history_scale": (
            torch.rand(
                history_pages,
                BLOCK_SIZE,
                LATENT_RANK // GROUP_SIZE,
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
                BLOCK_SIZE,
                LATENT_RANK // GROUP_SIZE,
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
        "seq_lens": torch.tensor([SEQ_LEN], dtype=torch.int32, device=device),
        "rotation": rotation,
    }


def call_attention(
    function: Any, inputs: dict[str, Any], split: int
) -> tuple[Any, Any]:
    return function(
        inputs["query"],
        inputs["query_rope"],
        inputs["selected_tokens"],
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
        attention_scale=ATTENTION_SCALE,
        num_splits=split,
    )


def tensor_error(torch: Any, actual: Any, expected: Any) -> dict[str, float]:
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(1e-6)
    return {
        "max_abs": float(absolute.max().item()),
        "max_rel": float(relative.max().item()),
    }


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "samples": values,
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> int:
    args = parse_args()

    import torch

    from vllm.v1.attention.ops import triton_oscar_mla_decode

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one visible CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    rotation = load_rotation(
        torch,
        args.rotation_artifact,
        args.rotation_layer,
        device,
    )
    inputs = make_inputs(torch, rotation=rotation, seed=args.seed, device=device)
    function = triton_oscar_mla_decode.oscar_mla_sparse_decode

    reference_split = 16 if 16 in args.splits else args.splits[0]
    outputs: dict[int, tuple[Any, Any]] = {}
    for split in args.splits:
        outputs[split] = call_attention(function, inputs, split)
    torch.cuda.synchronize(device)

    reference_output, reference_lse = outputs[reference_split]
    correctness: dict[int, dict[str, Any]] = {}
    for split, (output, lse) in outputs.items():
        output_error = tensor_error(torch, output, reference_output)
        lse_error = tensor_error(torch, lse, reference_lse)
        correct = bool(
            torch.isfinite(output).all()
            and torch.isfinite(lse).all()
            and torch.allclose(
                output,
                reference_output,
                atol=OUTPUT_ATOL,
                rtol=OUTPUT_RTOL,
            )
            and torch.allclose(
                lse,
                reference_lse,
                atol=LSE_ATOL,
                rtol=LSE_RTOL,
            )
        )
        correctness[split] = {
            "correct": correct,
            "output": output_error,
            "lse": lse_error,
        }
    if not all(row["correct"] for row in correctness.values()):
        raise RuntimeError(f"split correctness check failed: {correctness}")

    for split in args.splits:
        for _ in range(args.warmup):
            call_attention(function, inputs, split)
        torch.cuda.synchronize(device)
        print(f"warmup complete: split={split}", flush=True)

    measurements: dict[int, dict[str, list[float]]] = {
        split: {"cuda_ms": [], "wall_ms": [], "peak_allocated_mib": []}
        for split in args.splits
    }
    benchmark_started = time.monotonic()
    last_heartbeat = benchmark_started
    for repeat in range(args.repeats):
        order = args.splits if repeat % 2 == 0 else list(reversed(args.splits))
        for split in order:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start_event.record()
            for _ in range(args.iterations):
                call_attention(function, inputs, split)
            end_event.record()
            end_event.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000.0 / args.iterations
            cuda_ms = start_event.elapsed_time(end_event) / args.iterations
            peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
            measurements[split]["cuda_ms"].append(cuda_ms)
            measurements[split]["wall_ms"].append(wall_ms)
            measurements[split]["peak_allocated_mib"].append(peak_mib)
            print(
                f"repeat={repeat + 1}/{args.repeats} split={split} "
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
        split: {name: summarize(values) for name, values in split_measurements.items()}
        for split, split_measurements in measurements.items()
    }
    winner = min(
        args.splits,
        key=lambda split: results[split]["cuda_ms"]["median"],
    )
    properties = torch.cuda.get_device_properties(device)
    result = {
        "format_version": FORMAT_VERSION,
        "status": "passed",
        "scope": "oscar_mixed_attention_split_microbenchmark",
        "fixed_gpu_count": 1,
        "shape": {
            "sequence_length": SEQ_LEN,
            "topk_slots": TOPK,
            "valid_selected_tokens": SEQ_LEN,
            "batch_size": 1,
            "local_attention_heads": NUM_HEADS,
            "latent_rank": LATENT_RANK,
            "rope_head_size": ROPE_HEAD_SIZE,
            "prefix_tokens": PREFIX_TOKENS,
            "history_tokens": SEQ_LEN - PREFIX_TOKENS - RECENT_TOKENS,
            "recent_tokens": RECENT_TOKENS,
            "block_size": BLOCK_SIZE,
            "group_size": GROUP_SIZE,
        },
        "measurement": {
            "splits": args.splits,
            "reference_split": reference_split,
            "warmup_per_split": args.warmup,
            "repeats": args.repeats,
            "iterations_per_repeat": args.iterations,
            "seed": args.seed,
            "attention_scale": ATTENTION_SCALE,
            "elapsed_seconds": time.monotonic() - benchmark_started,
        },
        "correctness": {
            "output_atol": OUTPUT_ATOL,
            "output_rtol": OUTPUT_RTOL,
            "lse_atol": LSE_ATOL,
            "lse_rtol": LSE_RTOL,
            "by_split": correctness,
        },
        "results_by_split": results,
        "winner": {
            "num_splits": winner,
            "median_cuda_ms": results[winner]["cuda_ms"]["median"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name,
            "gpu_compute_capability": [
                properties.major,
                properties.minor,
            ],
            "gpu_total_memory_bytes": properties.total_memory,
            "rotation_layer": args.rotation_layer,
            "rotation_artifact": str(args.rotation_artifact),
            "rotation_artifact_sha256": sha256_file(args.rotation_artifact),
            "benchmark_script_sha256": sha256_file(Path(__file__)),
            "kernel_source": str(Path(triton_oscar_mla_decode.__file__).resolve()),
            "kernel_source_sha256": sha256_file(
                Path(triton_oscar_mla_decode.__file__).resolve()
            ),
        },
    }
    atomic_write_json(args.output, result)
    print(
        f"winner: split={winner} "
        f"median_cuda_ms={results[winner]['cuda_ms']['median']:.6f}",
        flush=True,
    )
    print(f"result: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
