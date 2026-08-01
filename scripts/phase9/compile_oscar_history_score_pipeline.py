#!/usr/bin/env python3
"""Offline-compile a score/LSE/value history prefill pipeline for SM80."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
import compile_oscar_prefill_cache_split as base
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from vllm.v1.attention.ops import triton_oscar_mla_decode

FORMAT_VERSION = 1
TARGET = GPUTarget("cuda", 80, 32)
EXPECTED_SOURCE_COMMIT = "67a0e47ff72f10a322de17b81c4134984e017bd6"
REFERENCE_RESOURCES = {
    "name": "history_h8_t16_w8",
    "shared_bytes": 84_992,
    "registers_per_thread": 199,
    "stack_bytes_per_thread": 0,
}


@dataclass(frozen=True)
class Variant:
    name: str
    kernel_mode: str
    block_h: int
    block_t: int
    block_dv: int
    num_warps: int


VARIANTS = [
    Variant("score_h8_t16_w8", "score", 8, 16, 128, 8),
    Variant("score_h4_t16_w8", "score", 4, 16, 128, 8),
    Variant("lse_tiles128_w4", "lse", 1, 128, 128, 4),
    Variant("value_h2_d128_t16_w4", "value", 2, 16, 128, 4),
    Variant("value_h1_d128_t16_w4", "value", 1, 16, 128, 4),
]


@triton.jit
def _history_score_kernel(
    query_rotated_ptr,
    query_rope_ptr,
    selected_tokens_ptr,
    query_request_indices_ptr,
    query_positions_ptr,
    rope_ptr,
    rope_block_table_ptr,
    history_data_ptr,
    history_scale_ptr,
    history_zero_ptr,
    history_page_table_ptr,
    hp_rows_ptr,
    seq_lens_ptr,
    score_scratch_ptr,
    tile_lse_ptr,
    stride_query_rotated_b: tl.constexpr,
    stride_query_rotated_h: tl.constexpr,
    stride_query_rotated_d: tl.constexpr,
    stride_query_rope_b: tl.constexpr,
    stride_query_rope_h: tl.constexpr,
    stride_query_rope_d: tl.constexpr,
    stride_selected_b: tl.constexpr,
    stride_selected_k: tl.constexpr,
    stride_query_request: tl.constexpr,
    stride_query_position: tl.constexpr,
    stride_rope_block: tl.constexpr,
    stride_rope_token: tl.constexpr,
    stride_rope_d: tl.constexpr,
    stride_rope_block_table_b: tl.constexpr,
    stride_rope_block_table_page: tl.constexpr,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    stride_page_table_b: tl.constexpr,
    stride_page_table_page: tl.constexpr,
    stride_hp_rows: tl.constexpr,
    stride_seq_lens: tl.constexpr,
    stride_score_b: tl.constexpr,
    stride_score_h: tl.constexpr,
    stride_score_k: tl.constexpr,
    stride_tile_lse_b: tl.constexpr,
    stride_tile_lse_h: tl.constexpr,
    stride_tile_lse_tile: tl.constexpr,
    topk: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    rope_block_size: tl.constexpr,
    rope_head_size: tl.constexpr,
    history_block_size: tl.constexpr,
    latent_rank: tl.constexpr,
    group_size: tl.constexpr,
    attention_scale: tl.constexpr,
    num_requests: tl.constexpr,
    num_heads: tl.constexpr,
    block_h: tl.constexpr,
    block_t: tl.constexpr,
    block_d: tl.constexpr,
    block_r: tl.constexpr,
):
    query_row = tl.program_id(0)
    head_group = tl.program_id(1)
    tile_index = tl.program_id(2)
    heads = head_group * block_h + tl.arange(0, block_h)
    head_mask = heads < num_heads
    token_offsets = tl.arange(0, block_t)
    selected_offsets = tile_index * block_t + token_offsets
    selected_mask = selected_offsets < topk

    request = tl.load(query_request_indices_ptr + query_row * stride_query_request)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.where(request_valid, request, 0)
    query_position = tl.load(query_positions_ptr + query_row * stride_query_position)
    hp_row = tl.load(hp_rows_ptr + safe_request * stride_hp_rows)
    seq_len = tl.load(seq_lens_ptr + safe_request * stride_seq_lens)
    causal_seq_len = tl.minimum(seq_len, query_position + 1)
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)
    tokens = tl.load(
        selected_tokens_ptr
        + query_row * stride_selected_b
        + selected_offsets * stride_selected_k,
        mask=selected_mask,
        other=-1,
    )
    valid = (
        selected_mask
        & request_valid
        & (query_position >= 0)
        & (tokens >= 0)
        & (tokens < causal_seq_len)
        & (hp_row >= 0)
    )
    is_history = valid & (tokens >= prefix_tokens) & (tokens < recent_start)

    dims = tl.arange(0, block_d)
    dim_mask = dims < latent_rank
    query_rotated = tl.load(
        query_rotated_ptr
        + query_row * stride_query_rotated_b
        + heads[:, None] * stride_query_rotated_h
        + dims[None, :] * stride_query_rotated_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    history_indices = tokens - prefix_tokens
    logical_pages = history_indices // history_block_size
    page_offsets = history_indices % history_block_size
    physical_pages = tl.load(
        history_page_table_ptr
        + safe_request * stride_page_table_b
        + logical_pages * stride_page_table_page,
        mask=is_history,
        other=0,
    )
    byte_offsets = dims // 4
    shifts = (dims % 4) * 2
    data_base = physical_pages * stride_data_page + page_offsets * stride_data_token
    packed = tl.load(
        history_data_ptr
        + data_base[None, :]
        + byte_offsets[:, None] * stride_data_byte,
        mask=dim_mask[:, None] & is_history[None, :],
        other=0,
    ).to(tl.int32)
    quantized = ((packed >> shifts[:, None]) & 0x3).to(tl.float32)
    groups = dims // group_size
    scale = tl.load(
        history_scale_ptr
        + physical_pages[None, :] * stride_scale_page
        + page_offsets[None, :] * stride_scale_token
        + groups[:, None] * stride_scale_group,
        mask=dim_mask[:, None] & is_history[None, :],
        other=0.0,
    ).to(tl.float32)
    zero = tl.load(
        history_zero_ptr
        + physical_pages[None, :] * stride_zero_page
        + page_offsets[None, :] * stride_zero_token
        + groups[:, None] * stride_zero_group,
        mask=dim_mask[:, None] & is_history[None, :],
        other=0.0,
    ).to(tl.float32)
    history_values = (quantized - zero) * scale

    rope_dims = tl.arange(0, block_r)
    rope_dim_mask = rope_dims < rope_head_size
    query_rope = tl.load(
        query_rope_ptr
        + query_row * stride_query_rope_b
        + heads[:, None] * stride_query_rope_h
        + rope_dims[None, :] * stride_query_rope_d,
        mask=head_mask[:, None] & rope_dim_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    rope_logical_pages = tokens // rope_block_size
    rope_page_offsets = tokens % rope_block_size
    rope_physical_pages = tl.load(
        rope_block_table_ptr
        + safe_request * stride_rope_block_table_b
        + rope_logical_pages * stride_rope_block_table_page,
        mask=is_history,
        other=0,
    )
    rope_values = tl.load(
        rope_ptr
        + rope_physical_pages[None, :] * stride_rope_block
        + rope_page_offsets[None, :] * stride_rope_token
        + rope_dims[:, None] * stride_rope_d,
        mask=rope_dim_mask[:, None] & is_history[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    scores = tl.dot(
        query_rotated,
        history_values,
        input_precision="tf32",
    ) + tl.dot(query_rope, rope_values)
    scores *= attention_scale
    score_mask = head_mask[:, None] & is_history[None, :]
    scores = tl.where(score_mask, scores, -float("inf"))
    score_base = (
        query_row * stride_score_b
        + heads[:, None] * stride_score_h
        + selected_offsets[None, :] * stride_score_k
    )
    tl.store(score_scratch_ptr + score_base, scores, mask=head_mask[:, None])

    tile_max = tl.max(scores, axis=1)
    safe_tile_max = tl.where(tile_max == -float("inf"), 0.0, tile_max)
    tile_sum = tl.sum(
        tl.where(score_mask, tl.exp(scores - safe_tile_max[:, None]), 0.0),
        axis=1,
    )
    tile_lse = tl.where(
        head_mask & (tile_sum > 0.0),
        safe_tile_max + tl.log(tile_sum),
        -float("inf"),
    )
    tile_lse_base = (
        query_row * stride_tile_lse_b
        + heads * stride_tile_lse_h
        + tile_index * stride_tile_lse_tile
    )
    tl.store(tile_lse_ptr + tile_lse_base, tile_lse, mask=head_mask)


@triton.jit
def _history_lse_kernel(
    tile_lse_ptr,
    final_lse_ptr,
    stride_tile_lse_b: tl.constexpr,
    stride_tile_lse_h: tl.constexpr,
    stride_tile_lse_tile: tl.constexpr,
    stride_final_lse_b: tl.constexpr,
    stride_final_lse_h: tl.constexpr,
    num_heads: tl.constexpr,
    num_tiles: tl.constexpr,
    block_tiles: tl.constexpr,
):
    query_row = tl.program_id(0)
    head = tl.program_id(1)
    tiles = tl.arange(0, block_tiles)
    tile_mask = tiles < num_tiles
    tile_lse = tl.load(
        tile_lse_ptr
        + query_row * stride_tile_lse_b
        + head * stride_tile_lse_h
        + tiles * stride_tile_lse_tile,
        mask=tile_mask,
        other=-float("inf"),
    )
    maximum = tl.max(tile_lse, axis=0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    total = tl.sum(
        tl.where(tile_mask, tl.exp(tile_lse - safe_maximum), 0.0),
        axis=0,
    )
    final_lse = tl.where(
        total > 0.0,
        safe_maximum + tl.log(total),
        -float("inf"),
    )
    tl.store(
        final_lse_ptr + query_row * stride_final_lse_b + head * stride_final_lse_h,
        final_lse,
        mask=head < num_heads,
    )


@triton.jit
def _history_value_kernel(
    selected_tokens_ptr,
    query_request_indices_ptr,
    query_positions_ptr,
    history_data_ptr,
    history_scale_ptr,
    history_zero_ptr,
    history_page_table_ptr,
    hp_rows_ptr,
    seq_lens_ptr,
    score_scratch_ptr,
    final_lse_ptr,
    output_ptr,
    stride_selected_b: tl.constexpr,
    stride_selected_k: tl.constexpr,
    stride_query_request: tl.constexpr,
    stride_query_position: tl.constexpr,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    stride_page_table_b: tl.constexpr,
    stride_page_table_page: tl.constexpr,
    stride_hp_rows: tl.constexpr,
    stride_seq_lens: tl.constexpr,
    stride_score_b: tl.constexpr,
    stride_score_h: tl.constexpr,
    stride_score_k: tl.constexpr,
    stride_final_lse_b: tl.constexpr,
    stride_final_lse_h: tl.constexpr,
    stride_output_b: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    topk: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    history_block_size: tl.constexpr,
    latent_rank: tl.constexpr,
    group_size: tl.constexpr,
    num_requests: tl.constexpr,
    num_heads: tl.constexpr,
    block_h: tl.constexpr,
    block_t: tl.constexpr,
    block_dv: tl.constexpr,
):
    query_row = tl.program_id(0)
    head_group = tl.program_id(1)
    dim_group = tl.program_id(2)
    heads = head_group * block_h + tl.arange(0, block_h)
    head_mask = heads < num_heads
    dims = dim_group * block_dv + tl.arange(0, block_dv)
    dim_mask = dims < latent_rank

    request = tl.load(query_request_indices_ptr + query_row * stride_query_request)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.where(request_valid, request, 0)
    query_position = tl.load(query_positions_ptr + query_row * stride_query_position)
    hp_row = tl.load(hp_rows_ptr + safe_request * stride_hp_rows)
    seq_len = tl.load(seq_lens_ptr + safe_request * stride_seq_lens)
    causal_seq_len = tl.minimum(seq_len, query_position + 1)
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)
    final_lse = tl.load(
        final_lse_ptr + query_row * stride_final_lse_b + heads * stride_final_lse_h,
        mask=head_mask,
        other=-float("inf"),
    )
    lse_valid = final_lse != -float("inf")
    safe_lse = tl.where(lse_valid, final_lse, 0.0)
    token_offsets = tl.arange(0, block_t)
    history_acc = tl.zeros((block_h, block_dv), dtype=tl.float32)

    for tile_start in tl.range(0, topk, block_t):
        selected_offsets = tile_start + token_offsets
        selected_mask = selected_offsets < topk
        tokens = tl.load(
            selected_tokens_ptr
            + query_row * stride_selected_b
            + selected_offsets * stride_selected_k,
            mask=selected_mask,
            other=-1,
        )
        valid = (
            selected_mask
            & request_valid
            & (query_position >= 0)
            & (tokens >= 0)
            & (tokens < causal_seq_len)
            & (hp_row >= 0)
        )
        is_history = valid & (tokens >= prefix_tokens) & (tokens < recent_start)
        scores = tl.load(
            score_scratch_ptr
            + query_row * stride_score_b
            + heads[:, None] * stride_score_h
            + selected_offsets[None, :] * stride_score_k,
            mask=head_mask[:, None] & selected_mask[None, :],
            other=-float("inf"),
        )
        score_mask = head_mask[:, None] & is_history[None, :] & lse_valid[:, None]
        probabilities = tl.where(
            score_mask,
            tl.exp(scores - safe_lse[:, None]),
            0.0,
        )

        history_indices = tokens - prefix_tokens
        logical_pages = history_indices // history_block_size
        page_offsets = history_indices % history_block_size
        physical_pages = tl.load(
            history_page_table_ptr
            + safe_request * stride_page_table_b
            + logical_pages * stride_page_table_page,
            mask=is_history,
            other=0,
        )
        byte_offsets = dims // 4
        shifts = (dims % 4) * 2
        data_base = physical_pages * stride_data_page + page_offsets * stride_data_token
        packed = tl.load(
            history_data_ptr
            + data_base[None, :]
            + byte_offsets[:, None] * stride_data_byte,
            mask=dim_mask[:, None] & is_history[None, :],
            other=0,
        ).to(tl.int32)
        quantized = ((packed >> shifts[:, None]) & 0x3).to(tl.float32)
        groups = dims // group_size
        scale = tl.load(
            history_scale_ptr
            + physical_pages[None, :] * stride_scale_page
            + page_offsets[None, :] * stride_scale_token
            + groups[:, None] * stride_scale_group,
            mask=dim_mask[:, None] & is_history[None, :],
            other=0.0,
        ).to(tl.float32)
        zero = tl.load(
            history_zero_ptr
            + physical_pages[None, :] * stride_zero_page
            + page_offsets[None, :] * stride_zero_token
            + groups[:, None] * stride_zero_group,
            mask=dim_mask[:, None] & is_history[None, :],
            other=0.0,
        ).to(tl.float32)
        history_values = (quantized - zero) * scale
        history_acc += tl.dot(
            probabilities,
            tl.trans(history_values),
            input_precision="tf32",
        )

    output_base = (
        query_row * stride_output_b
        + heads[:, None] * stride_output_h
        + dims[None, :] * stride_output_d
    )
    tl.store(
        output_ptr + output_base,
        history_acc,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


SCORE_POINTER_SIGNATURE = {
    "query_rotated_ptr": "*fp32",
    "query_rope_ptr": "*bf16",
    "selected_tokens_ptr": "*i32",
    "query_request_indices_ptr": "*i32",
    "query_positions_ptr": "*i32",
    "rope_ptr": "*bf16",
    "rope_block_table_ptr": "*i32",
    "history_data_ptr": "*u8",
    "history_scale_ptr": "*fp32",
    "history_zero_ptr": "*fp32",
    "history_page_table_ptr": "*i32",
    "hp_rows_ptr": "*i32",
    "seq_lens_ptr": "*i32",
    "score_scratch_ptr": "*fp32",
    "tile_lse_ptr": "*fp32",
}
LSE_POINTER_SIGNATURE = {
    "tile_lse_ptr": "*fp32",
    "final_lse_ptr": "*fp32",
}
VALUE_POINTER_SIGNATURE = {
    "selected_tokens_ptr": "*i32",
    "query_request_indices_ptr": "*i32",
    "query_positions_ptr": "*i32",
    "history_data_ptr": "*u8",
    "history_scale_ptr": "*fp32",
    "history_zero_ptr": "*fp32",
    "history_page_table_ptr": "*i32",
    "hp_rows_ptr": "*i32",
    "seq_lens_ptr": "*i32",
    "score_scratch_ptr": "*fp32",
    "final_lse_ptr": "*fp32",
    "output_ptr": "*fp32",
}

BASE_CONSTANTS = {
    **base.BASE_CONSTANTS,
    "stride_score_b": 16_384,
    "stride_score_h": 2_048,
    "stride_score_k": 1,
    "stride_tile_lse_b": 1_024,
    "stride_tile_lse_h": 128,
    "stride_tile_lse_tile": 1,
    "stride_final_lse_b": 8,
    "stride_final_lse_h": 1,
    "stride_output_b": 4_096,
    "stride_output_h": 512,
    "stride_output_d": 1,
    "num_tiles": 128,
    "block_tiles": 128,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument(
        "--expected-source-commit",
        default=EXPECTED_SOURCE_COMMIT,
    )
    return parser.parse_args()


def scratch_layout_bytes(
    *,
    query_tokens: int,
    num_heads: int,
    topk: int,
    block_t: int,
) -> dict[str, int]:
    num_tiles = triton.cdiv(topk, block_t)
    scores = query_tokens * num_heads * topk * 4
    tile_lse = query_tokens * num_heads * num_tiles * 4
    final_lse = query_tokens * num_heads * 4
    return {
        "scores": scores,
        "tile_lse": tile_lse,
        "final_lse": final_lse,
        "total": scores + tile_lse + final_lse,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_pipeline_gate(results: list[dict[str, Any]]) -> dict[str, Any]:
    stages = ("score", "lse", "value")
    strict_by_stage = {
        stage: [
            row["name"]
            for row in results
            if row.get("kernel_mode") == stage
            and row.get("status") == "compiled"
            and row.get("strict_promotion_candidate")
        ]
        for stage in stages
    }
    missing = [stage for stage in stages if not strict_by_stage[stage]]
    return {
        "strict_candidates_by_stage": strict_by_stage,
        "missing_strict_stages": missing,
        "pipeline_strict_promotion_feasible": not missing,
    }


def kernel_spec(mode: str) -> tuple[Any, dict[str, str]]:
    if mode == "score":
        return _history_score_kernel, SCORE_POINTER_SIGNATURE
    if mode == "lse":
        return _history_lse_kernel, LSE_POINTER_SIGNATURE
    if mode == "value":
        return _history_value_kernel, VALUE_POINTER_SIGNATURE
    raise ValueError(f"unsupported kernel mode: {mode}")


def constants_for(fn: Any, variant: Variant) -> dict[str, Any]:
    constants = {
        name: value for name, value in BASE_CONSTANTS.items() if name in fn.arg_names
    }
    for name in ("block_h", "block_t", "block_dv"):
        if name in fn.arg_names:
            constants[name] = getattr(variant, name)
    return constants


def compile_variant(
    variant: Variant,
    *,
    output: Path,
    cuobjdump: Path,
) -> dict[str, Any]:
    fn, signature = kernel_spec(variant.kernel_mode)
    cache_dir = output / "cache" / variant.name
    cache_dir.mkdir(parents=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    row: dict[str, Any] = asdict(variant)
    row["num_stages"] = 1
    started = time.monotonic()
    try:
        compiled = triton.compile(
            ASTSource(
                fn=fn,
                signature=signature,
                constexprs=constants_for(fn, variant),
            ),
            target=TARGET,
            options={"num_warps": variant.num_warps, "num_stages": 1},
        )
        cubin = bytes(compiled.asm["cubin"])
        cubin_path = output / f"{variant.name}.cubin"
        cubin_path.write_bytes(cubin)
        resource = subprocess.run(
            [str(cuobjdump), "--dump-resource-usage", str(cubin_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        resource_text = resource.stdout + resource.stderr
        resource_path = output / f"{variant.name}.resource.txt"
        resource_path.write_text(resource_text, encoding="utf-8")
        parsed = base.parse_resource_usage(resource_text)
        shared_bytes = int(compiled.metadata.shared)
        row.update(
            status="compiled",
            shared_bytes=shared_bytes,
            cubin_bytes=len(cubin),
            cubin_sha256=hashlib.sha256(cubin).hexdigest(),
            resource_usage_sha256=sha256_file(resource_path),
            **parsed,
        )
        row.update(
            base.classify_resources(
                shared_bytes=shared_bytes,
                registers_per_thread=parsed["registers_per_thread"],
                stack_bytes_per_thread=parsed["stack_bytes_per_thread"],
                num_warps=variant.num_warps,
            )
        )
    except Exception as exc:
        row.update(
            status="compile_rejected",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    row["elapsed_seconds"] = time.monotonic() - started
    (output / f"{variant.name}.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("VARIANT", json.dumps(row, sort_keys=True), flush=True)
    return row


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    if args.expected_source_commit != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError("expected source commit does not match the frozen source")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA must not be initialized during offline compilation")
    if not args.cuobjdump.is_file():
        raise FileNotFoundError(args.cuobjdump)
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"output directory must be empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if triton_oscar_mla_decode._prefill_head_block_size(8) != 8:
        raise RuntimeError("runtime does not contain the expected h8 prefill candidate")

    results = [
        compile_variant(variant, output=args.output, cuobjdump=args.cuobjdump)
        for variant in VARIANTS
    ]
    source_path = Path(triton_oscar_mla_decode.__file__).resolve()
    summary = {
        "format_version": FORMAT_VERSION,
        "status": "passed",
        "scope": "oscar_history_score_lse_value_offline_resource_screen",
        "mode": "cpu_only_offline_sm80_compile",
        "expected_source_commit": args.expected_source_commit,
        "runtime": {
            "python": os.sys.version.split()[0],
            "python_executable": os.sys.executable,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_initialized": torch.cuda.is_initialized(),
        },
        "target": {"backend": "cuda", "arch": 80, "warp_size": 32},
        "limits": {
            "sm_shared_bytes": base.SM_SHARED_LIMIT_BYTES,
            "sm_registers": base.SM_REGISTER_LIMIT,
            "two_block_shared_limit_bytes": base.SM_SHARED_LIMIT_BYTES // 2,
        },
        "reference_resources": REFERENCE_RESOURCES,
        "scratch": scratch_layout_bytes(
            query_tokens=2_048,
            num_heads=8,
            topk=2_048,
            block_t=16,
        ),
        "pipeline_gate": summarize_pipeline_gate(results),
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "tool_path": str(Path(__file__).resolve()),
            "tool_sha256": sha256_file(Path(__file__).resolve()),
        },
        "compiled_count": sum(row["status"] == "compiled" for row in results),
        "rejected_count": sum(row["status"] != "compiled" for row in results),
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
