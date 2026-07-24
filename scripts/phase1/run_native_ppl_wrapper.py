#!/usr/bin/env python3
"""Run the frozen WikiText-2 runner with the stage-1 native TP=8 config."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

EXPECTED_RUNNER_SHA256 = (
    "eec6b1a4be99a068f80b2e9c0d684392f1bf1a669cae4fbc881581d6baa25668"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--suite-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    actual_sha = sha256_file(args.runner)
    if actual_sha != EXPECTED_RUNNER_SHA256:
        raise SystemExit(
            f"perplexity runner changed: expected {EXPECTED_RUNNER_SHA256}, "
            f"got {actual_sha}"
        )

    spec = importlib.util.spec_from_file_location("frozen_ppl_runner", args.runner)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load runner: {args.runner}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def build_native_llm(
        *,
        model_path: str,
        dtype: str,
        trust_remote_code: bool,
        max_model_len: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        kv_cache_dtype: str | None,
    ) -> Any:
        from vllm import LLM

        if tensor_parallel_size != 8:
            raise ValueError("stage-1 perplexity requires tensor_parallel_size=8")
        if not enforce_eager:
            raise ValueError("stage-1 perplexity requires eager execution")
        return LLM(
            model=model_path,
            tokenizer=model_path,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            kv_cache_dtype=kv_cache_dtype or "auto",
            attention_config={"backend": "TRITON_MLA_SPARSE"},
            enable_prefix_caching=False,
            disable_custom_all_reduce=True,
            all2all_backend="deepep_low_latency",
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            safetensors_load_strategy="lazy",
            seed=42,
        )

    module.build_llm = build_native_llm
    sys.argv = [
        str(args.runner),
        "--suite-dir",
        str(args.suite_dir),
        "--output-dir",
        str(args.output_dir),
        "--model-path",
        args.model_path,
        "--benchmarks",
        "WikiText-2",
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--max-length",
        "2048",
        "--stride",
        "512",
        "--batch-size",
        "8",
        "--tensor-parallel-size",
        "8",
        "--gpu-memory-utilization",
        "0.92",
        "--enforce-eager",
        "--kv-cache-dtype",
        "auto",
        "--seed",
        "42",
    ]
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
