#!/usr/bin/env python3
"""Screen IEEE and TF32 OSCAR latent rotation at the 32K prefill shape."""

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
from collections.abc import Sequence
from typing import Any


FORMAT_VERSION = 1
DEFAULT_ROTATION_ARTIFACT = Path("/opt/oscar_artifacts/rotation_fit_v2/rotations.pt")
DEFAULT_ACCURACY_LAYERS = ["0", "25", "51", "77"]
GROUP_SIZE = 128
HISTORY_BLOCK_SIZE = 16


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("tf32", "ieee-sweep"), default="tf32")
    parser.add_argument(
        "--rotation-artifact",
        type=Path,
        default=DEFAULT_ROTATION_ARTIFACT,
    )
    parser.add_argument("--rotation-layer", default="0")
    parser.add_argument(
        "--accuracy-layers",
        nargs="+",
        default=DEFAULT_ACCURACY_LAYERS.copy(),
    )
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--latent-rank", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=0.35)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--clip-ratio", type=float, default=0.96)
    args = parser.parse_args(argv)
    for name in ("rows", "warmup", "repeats", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.latent_rank <= 0 or args.latent_rank % 64:
        parser.error("--latent-rank must be a positive multiple of 64")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    if not 0 < args.clip_ratio <= 1:
        parser.error("--clip-ratio must be in (0, 1]")
    if len(set(args.accuracy_layers)) != len(args.accuracy_layers):
        parser.error("--accuracy-layers must not contain duplicates")
    return args


def rotation_kernel_parameters() -> dict[str, int]:
    return {
        "block_m": 16,
        "block_n": 64,
        "block_k": 32,
        "num_warps": 4,
        "num_stages": 2,
    }


def build_ieee_sweep_configs() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "block_m": block_m,
            "block_n": block_n,
            "block_k": 32,
            "num_warps": num_warps,
            "num_stages": 2,
        }
        for name, block_m, block_n, num_warps in (
            ("m16_n64_w4", 16, 64, 4),
            ("m16_n64_w8", 16, 64, 8),
            ("m32_n64_w4", 32, 64, 4),
            ("m32_n64_w8", 32, 64, 8),
            ("m16_n128_w4", 16, 128, 4),
            ("m16_n128_w8", 16, 128, 8),
        )
    ]


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


def load_rotations(
    torch: Any,
    path: Path,
    layers: Sequence[str],
    *,
    latent_rank: int,
    device: Any,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("rotations"), dict):
        raise ValueError("rotation artifact must contain a rotations mapping")
    artifact_rotations = payload["rotations"]
    rotations = {}
    for layer in layers:
        if layer not in artifact_rotations:
            raise ValueError(f"rotation artifact does not contain layer {layer}")
        rotation = artifact_rotations[layer]
        if tuple(rotation.shape) != (latent_rank, latent_rank):
            raise ValueError(
                f"layer {layer} rotation shape is {tuple(rotation.shape)}, "
                f"expected {(latent_rank, latent_rank)}"
            )
        rotations[layer] = rotation.contiguous().to(
            device=device,
            dtype=torch.float32,
        )
    return rotations


def build_rotation_kernel(triton: Any, tl: Any) -> Any:
    @triton.jit
    def rotation_kernel(
        latent_ptr,
        rotation_ptr,
        output_ptr,
        num_rows,
        latent_rank: tl.constexpr,
        stride_latent_row: tl.constexpr,
        stride_latent_dim: tl.constexpr,
        stride_rotation_row: tl.constexpr,
        stride_rotation_col: tl.constexpr,
        stride_output_row: tl.constexpr,
        stride_output_dim: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        use_tf32: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(latent_rank, block_n)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        rows = pid_m * block_m + tl.arange(0, block_m)
        cols = pid_n * block_n + tl.arange(0, block_n)
        ks = tl.arange(0, block_k)
        latent_ptrs = (
            latent_ptr
            + rows[:, None] * stride_latent_row
            + ks[None, :] * stride_latent_dim
        )
        rotation_ptrs = (
            rotation_ptr
            + ks[:, None] * stride_rotation_row
            + cols[None, :] * stride_rotation_col
        )
        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

        for k_start in range(0, latent_rank, block_k):
            k_mask = k_start + ks < latent_rank
            latent = tl.load(
                latent_ptrs,
                mask=(rows[:, None] < num_rows) & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            rotation = tl.load(
                rotation_ptrs,
                mask=k_mask[:, None] & (cols[None, :] < latent_rank),
                other=0.0,
            ).to(tl.float32)
            if use_tf32:
                accumulator = tl.dot(
                    latent,
                    rotation,
                    accumulator,
                    input_precision="tf32",
                )
            else:
                accumulator = tl.dot(
                    latent,
                    rotation,
                    accumulator,
                    input_precision="ieee",
                )
            latent_ptrs += block_k * stride_latent_dim
            rotation_ptrs += block_k * stride_rotation_row

        output_ptrs = (
            output_ptr
            + rows[:, None] * stride_output_row
            + cols[None, :] * stride_output_dim
        )
        tl.store(
            output_ptrs,
            accumulator,
            mask=(rows[:, None] < num_rows) & (cols[None, :] < latent_rank),
        )

    return rotation_kernel


def launch_rotation(
    triton: Any,
    kernel: Any,
    latent: Any,
    rotation: Any,
    output: Any,
    *,
    use_tf32: bool,
    config: dict[str, Any] | None = None,
) -> None:
    params = rotation_kernel_parameters() if config is None else config
    num_rows, latent_rank = latent.shape
    grid = (
        triton.cdiv(num_rows, params["block_m"])
        * triton.cdiv(latent_rank, params["block_n"]),
    )
    kernel[grid](
        latent,
        rotation,
        output,
        num_rows,
        latent_rank=latent_rank,
        stride_latent_row=latent.stride(0),
        stride_latent_dim=latent.stride(1),
        stride_rotation_row=rotation.stride(0),
        stride_rotation_col=rotation.stride(1),
        stride_output_row=output.stride(0),
        stride_output_dim=output.stride(1),
        block_m=params["block_m"],
        block_n=params["block_n"],
        block_k=params["block_k"],
        use_tf32=use_tf32,
        num_warps=params["num_warps"],
        num_stages=params["num_stages"],
    )


def summarize_samples(samples_ms: list[float], *, iterations: int) -> dict[str, Any]:
    per_call = [sample / iterations for sample in samples_ms]
    return {
        "samples_ms": per_call,
        "median_ms": statistics.median(per_call),
        "mean_ms": statistics.fmean(per_call),
        "min_ms": min(per_call),
        "max_ms": max(per_call),
    }


def validate_rotation_outputs(
    torch: Any,
    ieee: Any,
    candidate: Any,
    *,
    atol: float,
    rtol: float,
    label: str = "rotation candidate",
) -> dict[str, Any]:
    difference = (candidate - ieee).abs()
    close = torch.isclose(candidate, ieee, atol=atol, rtol=rtol)
    mismatched = int((~close).sum().item())
    result = {
        "status": "passed" if mismatched == 0 else "failed",
        "atol": atol,
        "rtol": rtol,
        "values": int(ieee.numel()),
        "mismatched_values": mismatched,
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "max_relative_error": float(
            (difference / ieee.abs().clamp_min(1e-12)).max().item()
        ),
    }
    if mismatched:
        raise AssertionError(
            f"{label} exceeds atol={atol}, rtol={rtol}: "
            f"mismatched_values={mismatched}, "
            f"max_abs_error={result['max_abs_error']}"
        )
    return result


def benchmark_mode(
    torch: Any,
    triton: Any,
    kernel: Any,
    latent: Any,
    rotation: Any,
    output: Any,
    *,
    use_tf32: bool,
    warmup: int,
    repeats: int,
    iterations: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for _ in range(warmup):
        launch_rotation(
            triton,
            kernel,
            latent,
            rotation,
            output,
            use_tf32=use_tf32,
            config=config,
        )
    torch.cuda.synchronize()

    cuda_samples_ms = []
    wall_samples_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        for _ in range(iterations):
            launch_rotation(
                triton,
                kernel,
                latent,
                rotation,
                output,
                use_tf32=use_tf32,
                config=config,
            )
        end.record()
        end.synchronize()
        cuda_samples_ms.append(float(start.elapsed_time(end)))
        wall_samples_ms.append((time.perf_counter() - wall_start) * 1000.0)
    return {
        "input_precision": "tf32" if use_tf32 else "ieee",
        "warmup_calls": warmup,
        "measured_samples": repeats,
        "iterations_per_sample": iterations,
        "cuda": summarize_samples(cuda_samples_ms, iterations=iterations),
        "wall": summarize_samples(wall_samples_ms, iterations=iterations),
    }


def compare_timings(
    ieee: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float]:
    ieee_cuda = ieee["cuda"]["median_ms"]
    candidate_cuda = candidate["cuda"]["median_ms"]
    ieee_wall = ieee["wall"]["median_ms"]
    candidate_wall = candidate["wall"]["median_ms"]
    return {
        "median_cuda_ms": candidate_cuda - ieee_cuda,
        "median_cuda_percent": (candidate_cuda / ieee_cuda - 1.0) * 100.0,
        "cuda_speedup": ieee_cuda / candidate_cuda,
        "median_wall_ms": candidate_wall - ieee_wall,
        "median_wall_percent": (candidate_wall / ieee_wall - 1.0) * 100.0,
        "wall_speedup": ieee_wall / candidate_wall,
    }


def select_best_ieee_config(results: dict[str, dict[str, Any]]) -> str:
    passed = {
        name: result for name, result in results.items() if result["status"] == "passed"
    }
    if not passed:
        raise ValueError("IEEE sweep has no passed config")
    return min(
        passed,
        key=lambda name: passed[name]["timing"]["cuda"]["median_ms"],
    )


def make_history_tensors(
    torch: Any,
    *,
    rows: int,
    latent_rank: int,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    pages = (rows + HISTORY_BLOCK_SIZE - 1) // HISTORY_BLOCK_SIZE
    data = torch.zeros(
        pages,
        HISTORY_BLOCK_SIZE,
        latent_rank // 4,
        dtype=torch.uint8,
        device=device,
    )
    scale = torch.zeros(
        pages,
        HISTORY_BLOCK_SIZE,
        latent_rank // GROUP_SIZE,
        dtype=torch.float32,
        device=device,
    )
    zero = torch.zeros_like(scale)
    row_ids = torch.arange(rows, dtype=torch.int32, device=device)
    page_ids = row_ids // HISTORY_BLOCK_SIZE
    page_offsets = row_ids % HISTORY_BLOCK_SIZE
    return data, scale, zero, page_ids, page_offsets


def quantize_and_restore(
    torch: Any,
    quantize: Any,
    dequantize: Any,
    rotated: Any,
    *,
    clip_ratio: float,
) -> tuple[Any, Any, Any, Any]:
    data, scale, zero, page_ids, page_offsets = make_history_tensors(
        torch,
        rows=rotated.shape[0],
        latent_rank=rotated.shape[1],
        device=rotated.device,
    )
    quantize(
        rotated,
        data,
        scale,
        zero,
        page_ids,
        page_offsets,
        clip_ratio=clip_ratio,
    )
    restored = dequantize(
        data,
        scale,
        zero,
        page_ids,
        page_offsets,
    )
    return restored, data, scale, zero


def run_ieee_sweep(
    torch: Any,
    triton: Any,
    kernel: Any,
    production_rotate: Any,
    rotations: dict[str, Any],
    args: argparse.Namespace,
    device: Any,
) -> dict[str, Any]:
    configs = build_ieee_sweep_configs()
    accuracy_inputs = {}
    for index, layer in enumerate(args.accuracy_layers):
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        latent = torch.randn(
            args.rows,
            args.latent_rank,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        accuracy_inputs[layer] = {
            "latent": latent,
            "production": production_rotate(latent, rotations[layer]),
        }

    timing_generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    timing_latent = torch.randn(
        args.rows,
        args.latent_rank,
        dtype=torch.bfloat16,
        device=device,
        generator=timing_generator,
    )
    timing_rotation = rotations[args.rotation_layer]
    results = {}
    for config in configs:
        name = config["name"]
        try:
            accuracy = {}
            for layer, inputs in accuracy_inputs.items():
                output = torch.empty_like(inputs["production"])
                launch_rotation(
                    triton,
                    kernel,
                    inputs["latent"],
                    rotations[layer],
                    output,
                    use_tf32=False,
                    config=config,
                )
                accuracy[layer] = validate_rotation_outputs(
                    torch,
                    inputs["production"],
                    output,
                    atol=0.0,
                    rtol=0.0,
                    label=f"IEEE sweep config {name} layer {layer}",
                )
            timing_output = torch.empty(
                (args.rows, args.latent_rank),
                dtype=torch.float32,
                device=device,
            )
            timing = benchmark_mode(
                torch,
                triton,
                kernel,
                timing_latent,
                timing_rotation,
                timing_output,
                use_tf32=False,
                warmup=args.warmup,
                repeats=args.repeats,
                iterations=args.iterations,
                config=config,
            )
            results[name] = {
                "status": "passed",
                "config": config,
                "accuracy": accuracy,
                "timing": timing,
            }
        except Exception as error:
            results[name] = {
                "status": "compile_or_runtime_failed",
                "config": config,
                "error_type": type(error).__name__,
                "error": str(error),
            }

    baseline_name = configs[0]["name"]
    if results[baseline_name]["status"] != "passed":
        raise RuntimeError(
            "production IEEE baseline failed: " + results[baseline_name]["error"]
        )
    baseline_timing = results[baseline_name]["timing"]
    for name, result in results.items():
        if result["status"] == "passed":
            result["relative_to_production"] = compare_timings(
                baseline_timing,
                result["timing"],
            )
    best_name = select_best_ieee_config(results)
    return {
        "format_version": FORMAT_VERSION,
        "status": "passed",
        "mode": "ieee-sweep",
        "system": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "identity": {
            "script_sha256": sha256_file(Path(__file__)),
            "rotation_artifact": str(args.rotation_artifact),
            "rotation_artifact_sha256": sha256_file(args.rotation_artifact),
            "timing_layer": args.rotation_layer,
            "accuracy_layers": args.accuracy_layers,
        },
        "workload": {
            "rows": args.rows,
            "latent_rank": args.latent_rank,
            "latent_dtype": "torch.bfloat16",
            "rotation_dtype": "torch.float32",
            "output_dtype": "torch.float32",
            "seed": args.seed,
            "input_precision": "ieee",
            "fixed_block_k": 32,
            "fixed_num_stages": 2,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "iterations": args.iterations,
        },
        "baseline": baseline_name,
        "best_config": best_name,
        "results": results,
        "interpretation_boundary": (
            "Synthetic BF16 latent rows and four real fitted rotation matrices. "
            "Every passed config is bitwise equal to production for those layers. "
            "Timing isolates one 2048x512 IEEE rotation kernel on one GPU; it "
            "does not measure TTFT, TPOT, throughput, GSM8K, or all 78 layers."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from vllm.triton_utils import tl, triton
    from vllm.v1.attention.ops.triton_oscar_mla_store import (
        oscar_mla_dequantize_history,
        oscar_mla_quantize_store_history,
        oscar_mla_rotate,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.latent_rank % GROUP_SIZE:
        raise ValueError("latent rank must be divisible by the INT2 group size")

    device = torch.device("cuda:0")
    requested_layers = list(dict.fromkeys([args.rotation_layer, *args.accuracy_layers]))
    rotations = load_rotations(
        torch,
        args.rotation_artifact,
        requested_layers,
        latent_rank=args.latent_rank,
        device=device,
    )
    kernel = build_rotation_kernel(triton, tl)
    if args.mode == "ieee-sweep":
        payload = run_ieee_sweep(
            torch,
            triton,
            kernel,
            oscar_mla_rotate,
            rotations,
            args,
            device,
        )
        atomic_write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    accuracy = {}
    for index, layer in enumerate(args.accuracy_layers):
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        latent = torch.randn(
            args.rows,
            args.latent_rank,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        rotation = rotations[layer]
        production = oscar_mla_rotate(latent, rotation)
        local_ieee = torch.empty_like(production)
        candidate = torch.empty_like(production)
        launch_rotation(
            triton,
            kernel,
            latent,
            rotation,
            local_ieee,
            use_tf32=False,
        )
        launch_rotation(
            triton,
            kernel,
            latent,
            rotation,
            candidate,
            use_tf32=True,
        )
        torch.cuda.synchronize()
        ieee_contract = validate_rotation_outputs(
            torch,
            production,
            local_ieee,
            atol=0.0,
            rtol=0.0,
            label="benchmark IEEE implementation",
        )
        rotation_correctness = validate_rotation_outputs(
            torch,
            production,
            candidate,
            atol=args.atol,
            rtol=args.rtol,
        )
        ieee_restored, ieee_data, ieee_scale, ieee_zero = quantize_and_restore(
            torch,
            oscar_mla_quantize_store_history,
            oscar_mla_dequantize_history,
            production,
            clip_ratio=args.clip_ratio,
        )
        candidate_restored, candidate_data, candidate_scale, candidate_zero = (
            quantize_and_restore(
                torch,
                oscar_mla_quantize_store_history,
                oscar_mla_dequantize_history,
                candidate,
                clip_ratio=args.clip_ratio,
            )
        )
        restored_correctness = validate_rotation_outputs(
            torch,
            ieee_restored,
            candidate_restored,
            atol=args.atol,
            rtol=args.rtol,
            label="INT2-restored TF32 candidate",
        )
        accuracy[layer] = {
            "benchmark_ieee_matches_production": ieee_contract,
            "tf32_rotation": rotation_correctness,
            "tf32_int2_restored": restored_correctness,
            "packed_byte_equal_fraction": float(
                (candidate_data == ieee_data).float().mean().item()
            ),
            "scale_max_abs_error": float(
                (candidate_scale - ieee_scale).abs().max().item()
            ),
            "zero_max_abs_error": float(
                (candidate_zero - ieee_zero).abs().max().item()
            ),
        }

    timing_generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    timing_latent = torch.randn(
        args.rows,
        args.latent_rank,
        dtype=torch.bfloat16,
        device=device,
        generator=timing_generator,
    )
    timing_rotation = rotations[args.rotation_layer]
    ieee_output = torch.empty(
        (args.rows, args.latent_rank),
        dtype=torch.float32,
        device=device,
    )
    candidate_output = torch.empty_like(ieee_output)
    ieee_timing = benchmark_mode(
        torch,
        triton,
        kernel,
        timing_latent,
        timing_rotation,
        ieee_output,
        use_tf32=False,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    candidate_timing = benchmark_mode(
        torch,
        triton,
        kernel,
        timing_latent,
        timing_rotation,
        candidate_output,
        use_tf32=True,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )

    payload = {
        "format_version": FORMAT_VERSION,
        "status": "passed",
        "mode": "tf32",
        "system": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "identity": {
            "script_sha256": sha256_file(Path(__file__)),
            "rotation_artifact": str(args.rotation_artifact),
            "rotation_artifact_sha256": sha256_file(args.rotation_artifact),
            "timing_layer": args.rotation_layer,
            "accuracy_layers": args.accuracy_layers,
        },
        "workload": {
            "rows": args.rows,
            "latent_rank": args.latent_rank,
            "latent_dtype": "torch.bfloat16",
            "rotation_dtype": "torch.float32",
            "output_dtype": "torch.float32",
            "seed": args.seed,
            "clip_ratio": args.clip_ratio,
            "kernel_parameters": rotation_kernel_parameters(),
        },
        "accuracy": accuracy,
        "ieee": ieee_timing,
        "tf32": candidate_timing,
        "tf32_minus_ieee": compare_timings(ieee_timing, candidate_timing),
        "interpretation_boundary": (
            "Synthetic BF16 latent rows and four real fitted rotation matrices. "
            "Timing isolates one 2048x512 rotation kernel on one GPU; it does "
            "not measure TTFT, TPOT, throughput, GSM8K, or all 78 layers."
        ),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
