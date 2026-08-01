#!/usr/bin/env python3
"""Offline-compile cache-specialized grouped OSCAR prefill kernels for SM80."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from vllm.v1.attention.ops import triton_oscar_mla_decode

FORMAT_VERSION = 6
TARGET = GPUTarget("cuda", 80, 32)
SM_SHARED_LIMIT_BYTES = 166_912
SM_REGISTER_LIMIT = 65_536
EXPECTED_BASELINE_SHARED_BYTES = 109_568


@dataclass(frozen=True)
class Variant:
    name: str
    kernel_mode: str
    block_h: int
    block_t: int
    num_warps: int
    reload_history_for_value: bool = False
    manual_history_value_reduce: bool = False


VARIANTS = [
    Variant("mixed_h8_t16_w8", "mixed", 8, 16, 8),
    Variant("history_h8_t16_w8", "history", 8, 16, 8),
    Variant("history_h8_t16_w4", "history", 8, 16, 4),
    Variant("history_h4_t16_w8", "history", 4, 16, 8),
    Variant("history_h4_t16_w4", "history", 4, 16, 4),
    Variant("history_h2_t16_w8", "history", 2, 16, 8),
    Variant("history_h2_t16_w4", "history", 2, 16, 4),
    Variant("history_h1_t16_w8", "history", 1, 16, 8),
    Variant("history_h1_t16_w4", "history", 1, 16, 4),
    Variant("history_h4_t8_w4", "history", 4, 8, 4),
    Variant("history_h2_t8_w4", "history", 2, 8, 4),
    Variant("history_h1_t8_w4", "history", 1, 8, 4),
    Variant(
        "history_manual_value_h4_t16_w4",
        "history",
        4,
        16,
        4,
        manual_history_value_reduce=True,
    ),
    Variant(
        "history_manual_value_h2_t16_w4",
        "history",
        2,
        16,
        4,
        manual_history_value_reduce=True,
    ),
    Variant(
        "history_manual_value_h1_t16_w4",
        "history",
        1,
        16,
        4,
        manual_history_value_reduce=True,
    ),
    Variant(
        "history_manual_value_h4_t8_w4",
        "history",
        4,
        8,
        4,
        manual_history_value_reduce=True,
    ),
    Variant(
        "history_manual_value_h2_t8_w4",
        "history",
        2,
        8,
        4,
        manual_history_value_reduce=True,
    ),
    Variant(
        "history_manual_value_h1_t8_w4",
        "history",
        1,
        8,
        4,
        manual_history_value_reduce=True,
    ),
    Variant("history_h8_t32_w8", "history", 8, 32, 8),
    Variant("history_h4_t32_w8", "history", 4, 32, 8),
    Variant("history_reload_h8_t16_w8", "history", 8, 16, 8, True),
    Variant("history_reload_h4_t16_w8", "history", 4, 16, 8, True),
    Variant("history_reload_h4_t16_w4", "history", 4, 16, 4, True),
    Variant("history_reload_h2_t16_w4", "history", 2, 16, 4, True),
    Variant("history_reload_h1_t16_w4", "history", 1, 16, 4, True),
    Variant("bf16_h8_t16_w8", "bf16", 8, 16, 8),
    Variant("bf16_h8_t16_w4", "bf16", 8, 16, 4),
    Variant("bf16_h4_t16_w8", "bf16", 4, 16, 8),
    Variant("bf16_h4_t16_w4", "bf16", 4, 16, 4),
]


@triton.jit
def _history_prefill_stage1(
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
    mid_history_ptr,
    mid_lse_ptr,
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
    stride_mid_b: tl.constexpr,
    stride_mid_h: tl.constexpr,
    stride_mid_d: tl.constexpr,
    stride_lse_b: tl.constexpr,
    stride_lse_h: tl.constexpr,
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
    reload_history_for_value: tl.constexpr,
    manual_history_value_reduce: tl.constexpr,
):
    query_row = tl.program_id(0)
    head_group = tl.program_id(1)
    heads = head_group * block_h + tl.arange(0, block_h)
    head_mask = heads < num_heads
    request = tl.load(query_request_indices_ptr + query_row * stride_query_request)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.where(request_valid, request, 0)
    query_position = tl.load(query_positions_ptr + query_row * stride_query_position)

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
    hp_row = tl.load(hp_rows_ptr + safe_request * stride_hp_rows)
    seq_len = tl.load(seq_lens_ptr + safe_request * stride_seq_lens)
    causal_seq_len = tl.minimum(seq_len, query_position + 1)
    effective_topk = tl.minimum(topk, causal_seq_len)
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)

    token_offsets = tl.arange(0, block_t)
    m_prev = tl.full((block_h,), -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros((block_h,), dtype=tl.float32)
    history_acc = tl.zeros((block_h, block_d), dtype=tl.float32)

    for tile_start in tl.range(0, effective_topk, block_t):
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
        has_history = tl.sum(is_history.to(tl.int32), axis=0) > 0
        if has_history:
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
            data_base = (
                physical_pages * stride_data_page + page_offsets * stride_data_token
            )
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
            m_new = tl.maximum(tl.max(scores, axis=1), m_prev)
            previous_scale = tl.exp(m_prev - m_new)
            probabilities = tl.exp(scores - m_new[:, None])
            probabilities = tl.where(score_mask, probabilities, 0.0)
            if reload_history_for_value:
                value_packed = tl.load(
                    history_data_ptr
                    + data_base[None, :]
                    + byte_offsets[:, None] * stride_data_byte,
                    mask=dim_mask[:, None] & is_history[None, :],
                    other=0,
                ).to(tl.int32)
                value_quantized = ((value_packed >> shifts[:, None]) & 0x3).to(
                    tl.float32
                )
                value_scale = tl.load(
                    history_scale_ptr
                    + physical_pages[None, :] * stride_scale_page
                    + page_offsets[None, :] * stride_scale_token
                    + groups[:, None] * stride_scale_group,
                    mask=dim_mask[:, None] & is_history[None, :],
                    other=0.0,
                ).to(tl.float32)
                value_zero = tl.load(
                    history_zero_ptr
                    + physical_pages[None, :] * stride_zero_page
                    + page_offsets[None, :] * stride_zero_token
                    + groups[:, None] * stride_zero_group,
                    mask=dim_mask[:, None] & is_history[None, :],
                    other=0.0,
                ).to(tl.float32)
                history_values_for_value = (value_quantized - value_zero) * value_scale
            else:
                history_values_for_value = history_values
            if manual_history_value_reduce:
                value_contribution = tl.sum(
                    probabilities[:, None, :] * history_values_for_value[None, :, :],
                    axis=2,
                )
            else:
                value_contribution = tl.dot(
                    probabilities,
                    tl.trans(history_values_for_value),
                    input_precision="tf32",
                )
            history_acc = history_acc * previous_scale[:, None] + value_contribution
            l_prev = l_prev * previous_scale + tl.sum(probabilities, axis=1)
            m_prev = m_new

    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    output_mask = head_mask[:, None] & dim_mask[None, :]
    output_base = (
        query_row * stride_mid_b
        + heads[:, None] * stride_mid_h
        + dims[None, :] * stride_mid_d
    )
    tl.store(
        mid_history_ptr + output_base,
        history_acc / safe_l[:, None],
        mask=output_mask,
    )
    local_lse = tl.where(l_prev > 0.0, m_prev + tl.log(safe_l), -float("inf"))
    tl.store(
        mid_lse_ptr + query_row * stride_lse_b + heads * stride_lse_h,
        local_lse,
        mask=head_mask,
    )


@triton.jit
def _bf16_prefill_stage1(
    query_ptr,
    query_rope_ptr,
    selected_tokens_ptr,
    query_request_indices_ptr,
    query_positions_ptr,
    prefix_ptr,
    recent_ptr,
    rope_ptr,
    rope_block_table_ptr,
    hp_rows_ptr,
    seq_lens_ptr,
    mid_bf16_ptr,
    mid_lse_ptr,
    stride_query_b: tl.constexpr,
    stride_query_h: tl.constexpr,
    stride_query_d: tl.constexpr,
    stride_query_rope_b: tl.constexpr,
    stride_query_rope_h: tl.constexpr,
    stride_query_rope_d: tl.constexpr,
    stride_selected_b: tl.constexpr,
    stride_selected_k: tl.constexpr,
    stride_query_request: tl.constexpr,
    stride_query_position: tl.constexpr,
    stride_prefix_row: tl.constexpr,
    stride_prefix_token: tl.constexpr,
    stride_prefix_d: tl.constexpr,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_d: tl.constexpr,
    stride_rope_block: tl.constexpr,
    stride_rope_token: tl.constexpr,
    stride_rope_d: tl.constexpr,
    stride_rope_block_table_b: tl.constexpr,
    stride_rope_block_table_page: tl.constexpr,
    stride_hp_rows: tl.constexpr,
    stride_seq_lens: tl.constexpr,
    stride_mid_b: tl.constexpr,
    stride_mid_h: tl.constexpr,
    stride_mid_d: tl.constexpr,
    stride_lse_b: tl.constexpr,
    stride_lse_h: tl.constexpr,
    topk: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    rope_block_size: tl.constexpr,
    rope_head_size: tl.constexpr,
    latent_rank: tl.constexpr,
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
    heads = head_group * block_h + tl.arange(0, block_h)
    head_mask = heads < num_heads
    request = tl.load(query_request_indices_ptr + query_row * stride_query_request)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.where(request_valid, request, 0)
    query_position = tl.load(query_positions_ptr + query_row * stride_query_position)

    dims = tl.arange(0, block_d)
    dim_mask = dims < latent_rank
    query = tl.load(
        query_ptr
        + query_row * stride_query_b
        + heads[:, None] * stride_query_h
        + dims[None, :] * stride_query_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
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
    hp_row = tl.load(hp_rows_ptr + safe_request * stride_hp_rows)
    seq_len = tl.load(seq_lens_ptr + safe_request * stride_seq_lens)
    causal_seq_len = tl.minimum(seq_len, query_position + 1)
    effective_topk = tl.minimum(topk, causal_seq_len)
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)

    token_offsets = tl.arange(0, block_t)
    m_prev = tl.full((block_h,), -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros((block_h,), dtype=tl.float32)
    bf16_acc = tl.zeros((block_h, block_d), dtype=tl.float32)

    for tile_start in tl.range(0, effective_topk, block_t):
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
        is_prefix = valid & (tokens < prefix_tokens)
        is_recent = valid & (tokens >= recent_start)
        is_bf16 = is_prefix | is_recent
        has_bf16 = tl.sum(is_bf16.to(tl.int32), axis=0) > 0
        if has_bf16:
            prefix_base = hp_row * stride_prefix_row + tokens * stride_prefix_token
            prefix_values = tl.load(
                prefix_ptr + prefix_base[None, :] + dims[:, None] * stride_prefix_d,
                mask=dim_mask[:, None] & is_prefix[None, :],
                other=0.0,
            )
            recent_indices = (tokens - prefix_tokens) % recent_tokens
            recent_base = (
                hp_row * stride_recent_row + recent_indices * stride_recent_token
            )
            recent_values = tl.load(
                recent_ptr + recent_base[None, :] + dims[:, None] * stride_recent_d,
                mask=dim_mask[:, None] & is_recent[None, :],
                other=0.0,
            )
            bf16_values = tl.where(
                is_prefix[None, :],
                prefix_values,
                recent_values,
            ).to(tl.bfloat16)

            rope_logical_pages = tokens // rope_block_size
            rope_page_offsets = tokens % rope_block_size
            rope_physical_pages = tl.load(
                rope_block_table_ptr
                + safe_request * stride_rope_block_table_b
                + rope_logical_pages * stride_rope_block_table_page,
                mask=is_bf16,
                other=0,
            )
            rope_values = tl.load(
                rope_ptr
                + rope_physical_pages[None, :] * stride_rope_block
                + rope_page_offsets[None, :] * stride_rope_token
                + rope_dims[:, None] * stride_rope_d,
                mask=rope_dim_mask[:, None] & is_bf16[None, :],
                other=0.0,
            ).to(tl.bfloat16)

            scores = tl.dot(query, bf16_values) + tl.dot(query_rope, rope_values)
            scores *= attention_scale
            score_mask = head_mask[:, None] & is_bf16[None, :]
            scores = tl.where(score_mask, scores, -float("inf"))
            m_new = tl.maximum(tl.max(scores, axis=1), m_prev)
            previous_scale = tl.exp(m_prev - m_new)
            probabilities = tl.exp(scores - m_new[:, None])
            probabilities = tl.where(score_mask, probabilities, 0.0)
            bf16_acc = bf16_acc * previous_scale[:, None] + tl.dot(
                probabilities,
                tl.trans(bf16_values.to(tl.float32)),
                input_precision="tf32",
            )
            l_prev = l_prev * previous_scale + tl.sum(probabilities, axis=1)
            m_prev = m_new

    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    output_mask = head_mask[:, None] & dim_mask[None, :]
    output_base = (
        query_row * stride_mid_b
        + heads[:, None] * stride_mid_h
        + dims[None, :] * stride_mid_d
    )
    tl.store(
        mid_bf16_ptr + output_base,
        bf16_acc / safe_l[:, None],
        mask=output_mask,
    )
    local_lse = tl.where(l_prev > 0.0, m_prev + tl.log(safe_l), -float("inf"))
    tl.store(
        mid_lse_ptr + query_row * stride_lse_b + heads * stride_lse_h,
        local_lse,
        mask=head_mask,
    )


MIXED_POINTER_SIGNATURE = {
    "query_ptr": "*bf16",
    "query_rotated_ptr": "*fp32",
    "query_rope_ptr": "*bf16",
    "selected_tokens_ptr": "*i32",
    "query_request_indices_ptr": "*i32",
    "query_positions_ptr": "*i32",
    "prefix_ptr": "*bf16",
    "recent_ptr": "*bf16",
    "rope_ptr": "*bf16",
    "rope_block_table_ptr": "*i32",
    "history_data_ptr": "*u8",
    "history_scale_ptr": "*fp32",
    "history_zero_ptr": "*fp32",
    "history_page_table_ptr": "*i32",
    "hp_rows_ptr": "*i32",
    "seq_lens_ptr": "*i32",
    "mid_bf16_ptr": "*fp32",
    "mid_history_ptr": "*fp32",
    "mid_lse_ptr": "*fp32",
}

HISTORY_POINTER_SIGNATURE = {
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
    "mid_history_ptr": "*fp32",
    "mid_lse_ptr": "*fp32",
}

BF16_POINTER_SIGNATURE = {
    "query_ptr": "*bf16",
    "query_rope_ptr": "*bf16",
    "selected_tokens_ptr": "*i32",
    "query_request_indices_ptr": "*i32",
    "query_positions_ptr": "*i32",
    "prefix_ptr": "*bf16",
    "recent_ptr": "*bf16",
    "rope_ptr": "*bf16",
    "rope_block_table_ptr": "*i32",
    "hp_rows_ptr": "*i32",
    "seq_lens_ptr": "*i32",
    "mid_bf16_ptr": "*fp32",
    "mid_lse_ptr": "*fp32",
}

BASE_CONSTANTS = {
    "stride_query_b": 4096,
    "stride_query_h": 512,
    "stride_query_d": 1,
    "stride_query_rotated_b": 4096,
    "stride_query_rotated_h": 512,
    "stride_query_rotated_d": 1,
    "stride_query_rope_b": 512,
    "stride_query_rope_h": 64,
    "stride_query_rope_d": 1,
    "stride_selected_b": 2048,
    "stride_selected_k": 1,
    "stride_query_request": 1,
    "stride_query_position": 1,
    "stride_prefix_row": 32768,
    "stride_prefix_token": 512,
    "stride_prefix_d": 1,
    "stride_recent_row": 131072,
    "stride_recent_token": 512,
    "stride_recent_d": 1,
    "stride_rope_block": 1024,
    "stride_rope_token": 64,
    "stride_rope_d": 1,
    "stride_rope_block_table_b": 128,
    "stride_rope_block_table_page": 1,
    "stride_data_page": 2048,
    "stride_data_token": 128,
    "stride_data_byte": 1,
    "stride_scale_page": 64,
    "stride_scale_token": 4,
    "stride_scale_group": 1,
    "stride_zero_page": 64,
    "stride_zero_token": 4,
    "stride_zero_group": 1,
    "stride_page_table_b": 108,
    "stride_page_table_page": 1,
    "stride_hp_rows": 1,
    "stride_seq_lens": 1,
    "stride_mid_b": 4096,
    "stride_mid_h": 512,
    "stride_mid_split": 512,
    "stride_mid_d": 1,
    "stride_lse_b": 8,
    "stride_lse_h": 1,
    "stride_lse_split": 1,
    "topk": 2048,
    "prefix_tokens": 64,
    "recent_tokens": 256,
    "rope_block_size": 16,
    "rope_head_size": 64,
    "history_block_size": 16,
    "latent_rank": 512,
    "group_size": 128,
    "attention_scale": 576**-0.5,
    "num_requests": 1,
    "num_heads": 8,
    "block_d": 512,
    "block_r": 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_resources(
    *,
    shared_bytes: int,
    registers_per_thread: int,
    stack_bytes_per_thread: int,
    num_warps: int,
) -> dict[str, Any]:
    threads_per_block = num_warps * 32
    shared_allows_two = shared_bytes * 2 <= SM_SHARED_LIMIT_BYTES
    registers_per_block = registers_per_thread * threads_per_block
    registers_allow_two = registers_per_block * 2 <= SM_REGISTER_LIMIT
    dual_block_feasible = shared_allows_two and registers_allow_two
    stack_free = stack_bytes_per_thread == 0
    return {
        "threads_per_block": threads_per_block,
        "registers_per_block": registers_per_block,
        "shared_allows_two_blocks": shared_allows_two,
        "registers_allow_two_blocks": registers_allow_two,
        "stack_free": stack_free,
        "dual_block_resource_feasible": dual_block_feasible,
        "strict_promotion_candidate": dual_block_feasible and stack_free,
    }


def summarize_split_gate(results: list[dict[str, Any]]) -> dict[str, Any]:
    compiled = [row for row in results if row.get("status") == "compiled"]
    history_dual = [
        row["name"]
        for row in compiled
        if row["kernel_mode"] == "history" and row["dual_block_resource_feasible"]
    ]
    history_strict = [
        row["name"]
        for row in compiled
        if row["kernel_mode"] == "history" and row["strict_promotion_candidate"]
    ]
    bf16_dual = [
        row["name"]
        for row in compiled
        if row["kernel_mode"] == "bf16" and row["dual_block_resource_feasible"]
    ]
    bf16_strict = [
        row["name"]
        for row in compiled
        if row["kernel_mode"] == "bf16" and row["strict_promotion_candidate"]
    ]
    return {
        "history_dual_block_candidates": history_dual,
        "history_strict_candidates": history_strict,
        "bf16_dual_block_candidates": bf16_dual,
        "bf16_strict_candidates": bf16_strict,
        "cache_split_dual_block_feasible": bool(history_dual and bf16_dual),
        "cache_split_strict_promotion_feasible": bool(history_strict and bf16_strict),
    }


def summarize_reload_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    compiled_by_name = {
        row["name"]: row for row in results if row.get("status") == "compiled"
    }
    pairs = []
    for reload_name in sorted(
        name for name in compiled_by_name if name.startswith("history_reload_")
    ):
        baseline_name = reload_name.replace("history_reload_", "history_", 1)
        if baseline_name not in compiled_by_name:
            raise RuntimeError(f"missing reload baseline: {baseline_name}")
        baseline = compiled_by_name[baseline_name]
        reload = compiled_by_name[reload_name]
        binary_identical = reload["cubin_sha256"] == baseline["cubin_sha256"]
        resource_identical = all(
            reload[key] == baseline[key]
            for key in (
                "resource_usage_sha256",
                "shared_bytes",
                "registers_per_thread",
                "stack_bytes_per_thread",
            )
        )
        pairs.append(
            {
                "baseline_name": baseline_name,
                "reload_name": reload_name,
                "binary_identical": binary_identical,
                "resource_identical": resource_identical,
                "shared_bytes_delta": reload["shared_bytes"] - baseline["shared_bytes"],
                "registers_per_thread_delta": reload["registers_per_thread"]
                - baseline["registers_per_thread"],
                "stack_bytes_per_thread_delta": reload["stack_bytes_per_thread"]
                - baseline["stack_bytes_per_thread"],
            }
        )
    all_identical = bool(pairs) and all(
        pair["binary_identical"] and pair["resource_identical"] for pair in pairs
    )
    return {
        "pairs": pairs,
        "all_pairs_binary_and_resource_identical": all_identical,
        "reload_changed_any_candidate": any(
            not pair["binary_identical"] or not pair["resource_identical"]
            for pair in pairs
        ),
    }


def parse_resource_usage(output: str) -> dict[str, int]:
    match = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+)", output)
    if match is None:
        raise RuntimeError(f"unable to parse cuobjdump resource usage: {output}")
    return {
        "registers_per_thread": int(match.group(1)),
        "stack_bytes_per_thread": int(match.group(2)),
        "static_shared_bytes": int(match.group(3)),
    }


def kernel_spec(mode: str) -> tuple[Any, dict[str, str]]:
    if mode == "mixed":
        return (
            triton_oscar_mla_decode._mixed_sparse_prefill_stage1,
            MIXED_POINTER_SIGNATURE,
        )
    if mode == "history":
        return _history_prefill_stage1, HISTORY_POINTER_SIGNATURE
    if mode == "bf16":
        return _bf16_prefill_stage1, BF16_POINTER_SIGNATURE
    raise ValueError(f"unsupported kernel mode: {mode}")


def constants_for(fn: Any, variant: Variant) -> dict[str, Any]:
    constants = {
        name: value for name, value in BASE_CONSTANTS.items() if name in fn.arg_names
    }
    constants.update(block_h=variant.block_h, block_t=variant.block_t)
    if "reload_history_for_value" in fn.arg_names:
        constants["reload_history_for_value"] = variant.reload_history_for_value
    if "manual_history_value_reduce" in fn.arg_names:
        constants["manual_history_value_reduce"] = variant.manual_history_value_reduce
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
        source = ASTSource(
            fn=fn,
            signature=signature,
            constexprs=constants_for(fn, variant),
        )
        compiled = triton.compile(
            source,
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
        parsed = parse_resource_usage(resource_text)
        shared_bytes = int(compiled.metadata.shared)
        row.update(
            status="compiled",
            shared_bytes=shared_bytes,
            cubin_bytes=len(cubin),
            cubin_sha256=sha256_bytes(cubin),
            resource_usage_sha256=sha256_file(resource_path),
            **parsed,
        )
        row.update(
            classify_resources(
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
    if not hasattr(
        triton_oscar_mla_decode._mixed_sparse_prefill_stage1,
        "arg_names",
    ):
        raise RuntimeError("real Triton JITFunction was not imported")

    results = [
        compile_variant(variant, output=args.output, cuobjdump=args.cuobjdump)
        for variant in VARIANTS
    ]
    baseline = results[0]
    if (
        baseline.get("status") != "compiled"
        or baseline.get("shared_bytes") != EXPECTED_BASELINE_SHARED_BYTES
    ):
        raise RuntimeError(f"baseline reproduction failed: {baseline}")

    compiled_rows = [row for row in results if row["status"] == "compiled"]
    split_gate = summarize_split_gate(results)
    reload_comparison = summarize_reload_comparison(results)
    source_path = Path(triton_oscar_mla_decode.__file__).resolve()
    summary = {
        "format_version": FORMAT_VERSION,
        "status": "passed",
        "scope": "oscar_prefill_cache_split_offline_resource_screen",
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
            "sm_shared_bytes": SM_SHARED_LIMIT_BYTES,
            "sm_registers": SM_REGISTER_LIMIT,
            "two_block_shared_limit_bytes": SM_SHARED_LIMIT_BYTES // 2,
        },
        "baseline": {
            "name": baseline["name"],
            "expected_shared_bytes": EXPECTED_BASELINE_SHARED_BYTES,
            "reproduced": True,
        },
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "tool_path": str(Path(__file__).resolve()),
            "tool_sha256": sha256_file(Path(__file__).resolve()),
        },
        "compiled_count": len(compiled_rows),
        "rejected_count": len(results) - len(compiled_rows),
        "split_gate": split_gate,
        "reload_comparison": reload_comparison,
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
