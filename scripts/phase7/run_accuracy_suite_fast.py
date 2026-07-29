#!/usr/bin/env python3
"""Run official_v5 scoring with crash-safe incremental prediction checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType
from typing import Any


CHECKPOINT_FORMAT_VERSION = 1


def load_frozen_runner(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("official_v5_frozen_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
    )


def fast_protocol_fingerprint(
    frozen: ModuleType,
    *,
    suite_dir: Path,
    config: dict[str, Any],
    model: str,
    manifest: list[dict[str, Any]],
    benchmark_budgets: dict[str, dict[str, int]],
) -> str:
    frozen_fingerprint = frozen.protocol_fingerprint(
        suite_dir=suite_dir,
        config=config,
        model=model,
        manifest=manifest,
        benchmark_budgets=benchmark_budgets,
    )
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "fast_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_protocol_fingerprint": frozen_fingerprint,
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def save_checkpoint(
    checkpoint_dir: Path,
    *,
    index: int,
    result: dict[str, Any],
) -> None:
    atomic_write_json(checkpoint_dir / f"{index:06d}.json", result)


def load_cached_results(
    *,
    manifest: list[dict[str, Any]],
    predictions_path: Path,
    checkpoint_dir: Path,
    protocol_fingerprint: str,
) -> dict[int, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    if predictions_path.exists():
        with predictions_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    candidates[row["id"]] = row
    if checkpoint_dir.exists():
        for path in sorted(checkpoint_dir.glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            candidates[row["id"]] = row

    manifest_index = {row["id"]: index for index, row in enumerate(manifest)}
    cached: dict[int, dict[str, Any]] = {}
    for sample_id, row in candidates.items():
        if sample_id not in manifest_index:
            raise RuntimeError(f"resume cache contains unknown sample: {sample_id}")
        if row.get("protocol_fingerprint") != protocol_fingerprint:
            raise RuntimeError(
                "resume cache protocol fingerprint does not match this run"
            )
        if row.get("evaluator_status") == "scored":
            cached[manifest_index[sample_id]] = row
    return cached


def load_prior_active_duration(
    *,
    state_path: Path,
    protocol_fingerprint: str,
) -> float:
    if not state_path.exists():
        return 0.0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("protocol_fingerprint") != protocol_fingerprint:
        raise RuntimeError("runner state protocol fingerprint does not match this run")
    return float(state.get("active_duration_seconds", 0.0))


def normalize_fast_budgets(
    *,
    computed: dict[str, dict[str, int]],
    config: dict[str, Any],
) -> dict[str, dict[str, int]]:
    screening = config["screening_protocol"]
    budget = computed["GSM8K"]
    server_max_model_len = int(screening["server_max_model_len"])
    benchmark_max_prompt_tokens = int(screening["benchmark_max_prompt_tokens"])
    observed_max_prompt_tokens = budget["max_prompt_tokens"]
    if budget["server_max_model_len"] != server_max_model_len:
        raise RuntimeError(
            "tokenizer server max_model_len does not match fast protocol"
        )
    if observed_max_prompt_tokens > benchmark_max_prompt_tokens:
        raise RuntimeError(
            "selected GSM8K prompt exceeds the frozen full-benchmark maximum"
        )
    return {
        "GSM8K": {
            "max_prompt_tokens": benchmark_max_prompt_tokens,
            "observed_max_prompt_tokens": observed_max_prompt_tokens,
            "server_max_model_len": server_max_model_len,
            "fixed_output_limit": (server_max_model_len - benchmark_max_prompt_tokens),
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-runner", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--code-eval-isolated", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be positive")

    frozen = load_frozen_runner(args.frozen_runner)
    manifest = [
        row
        for row in frozen.read_jsonl(args.suite_dir / "manifest.jsonl")
        if row["benchmark"] == "GSM8K"
    ]
    config = json.loads(
        (args.suite_dir / "eval_config.json").read_text(encoding="utf-8")
    )
    protocol_version = config.get("protocol_version")
    for sample in manifest:
        sample["_protocol_version"] = protocol_version
    frozen.validate_official_v5_environment(
        manifest,
        code_eval_isolated=args.code_eval_isolated,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timeouts = config["timeouts_seconds"]
    decoding = config["decoding"]
    run_started_at = time.time()
    tokenizations: dict[int, dict[str, int]] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                frozen.tokenize_prompt,
                base_url=args.base_url,
                model=args.model,
                sample=sample,
                decoding=decoding,
                timeout=timeouts["math_reasoning"],
            ): index
            for index, sample in enumerate(manifest)
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            tokenizations[index] = future.result()
            if completed_count % 20 == 0 or completed_count == len(manifest):
                print(
                    f"tokenized {completed_count}/{len(manifest)}",
                    flush=True,
                )

    benchmark_budgets = normalize_fast_budgets(
        computed=frozen.compute_benchmark_budgets(
            manifest,
            tokenizations,
        ),
        config=config,
    )
    protocol_fingerprint = fast_protocol_fingerprint(
        frozen,
        suite_dir=args.suite_dir,
        config=config,
        model=args.model,
        manifest=manifest,
        benchmark_budgets=benchmark_budgets,
    )
    predictions_path = args.output_dir / "predictions.jsonl"
    checkpoint_dir = args.output_dir / "prediction_checkpoints"
    state_path = args.output_dir / "fast_runner_state.json"
    cached = (
        load_cached_results(
            manifest=manifest,
            predictions_path=predictions_path,
            checkpoint_dir=checkpoint_dir,
            protocol_fingerprint=protocol_fingerprint,
        )
        if args.resume
        else {}
    )
    prior_active_duration = (
        load_prior_active_duration(
            state_path=state_path,
            protocol_fingerprint=protocol_fingerprint,
        )
        if args.resume
        else 0.0
    )
    ordered = dict(cached)
    atomic_write_json(
        state_path,
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "manifest_total": len(manifest),
            "completed": len(ordered),
            "resumed": len(cached),
            "active_duration_seconds": (
                prior_active_duration + time.time() - run_started_at
            ),
            "status": "running",
        },
    )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                frozen.request_completion,
                base_url=args.base_url,
                model=args.model,
                sample=sample,
                decoding=decoding,
                timeout=timeouts["math_reasoning"],
                input_tokens=tokenizations[index]["input_tokens"],
                server_max_model_len=tokenizations[index]["server_max_model_len"],
                fixed_output_limit=benchmark_budgets["GSM8K"]["fixed_output_limit"],
            ): index
            for index, sample in enumerate(manifest)
            if index not in cached
        }
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            result["protocol_fingerprint"] = protocol_fingerprint
            ordered[index] = result
            save_checkpoint(checkpoint_dir, index=index, result=result)
            completed = len(ordered)
            if completed % args.checkpoint_every == 0:
                atomic_write_jsonl(
                    predictions_path,
                    [ordered[item] for item in sorted(ordered)],
                )
            atomic_write_json(
                state_path,
                {
                    "format_version": CHECKPOINT_FORMAT_VERSION,
                    "protocol_fingerprint": protocol_fingerprint,
                    "manifest_total": len(manifest),
                    "completed": completed,
                    "resumed": len(cached),
                    "active_duration_seconds": (
                        prior_active_duration + time.time() - run_started_at
                    ),
                    "status": "running",
                },
            )
            if completed % 20 == 0 or completed == len(manifest):
                print(f"completed {completed}/{len(manifest)}", flush=True)

    results = [ordered[index] for index in range(len(manifest))]
    atomic_write_jsonl(predictions_path, results)
    by_benchmark = frozen.summarize(
        results,
        "benchmark",
        native_metrics_only=True,
    )
    by_task_type = frozen.summarize(
        results,
        "task_type",
        native_metrics_only=True,
    )
    frozen.write_json(args.output_dir / "summary_by_benchmark.json", by_benchmark)
    frozen.write_json(args.output_dir / "summary_by_task_type.json", by_task_type)
    frozen.write_csv(args.output_dir / "summary_by_benchmark.csv", by_benchmark)
    frozen.write_csv(args.output_dir / "summary_by_task_type.csv", by_task_type)
    failed = [
        row
        for row in results
        if row["evaluator_status"] != "scored"
        or row.get("score") != 1.0
        or row.get("truncated")
    ]
    atomic_write_jsonl(args.output_dir / "failed_cases.jsonl", failed)

    scored = [
        row
        for row in results
        if row["evaluator_status"] == "scored" and row["score"] is not None
    ]
    invalid_count = sum(row["evaluator_status"] != "scored" for row in results)
    completion_tokens = [
        int(row["token_usage"]["completion_tokens"])
        for row in results
        if (row.get("token_usage") or {}).get("completion_tokens") is not None
    ]
    duration_seconds = prior_active_duration + time.time() - run_started_at
    summary = {
        "total": len(results),
        "scored": len(scored),
        "truncated_count": sum(bool(row.get("truncated")) for row in results),
        "truncation_rate": (
            sum(bool(row.get("truncated")) for row in results) / len(results)
            if results
            else None
        ),
        "valid": invalid_count == 0,
        "protocol_version": protocol_version,
        "screening_protocol": "official_v5_fast_screen",
        "protocol_fingerprint": protocol_fingerprint,
        "benchmark_token_budgets": benchmark_budgets,
        "status_counts": dict(Counter(row["evaluator_status"] for row in results)),
        "started_at_unix": run_started_at,
        "ended_at_unix": time.time(),
        "duration_seconds": duration_seconds,
        "requests_per_hour": (
            len(results) * 3600 / duration_seconds if duration_seconds else None
        ),
        "completion_tokens_total": sum(completion_tokens),
        "completion_tokens_mean": (
            sum(completion_tokens) / len(completion_tokens)
            if completion_tokens
            else None
        ),
        "native_metrics": frozen.compute_native_metrics(results),
        "code_eval_environment": frozen.code_eval_environment(),
    }
    frozen.write_json(args.output_dir / "summary.json", summary)
    atomic_write_json(
        state_path,
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "manifest_total": len(manifest),
            "completed": len(results),
            "resumed": len(cached),
            "active_duration_seconds": duration_seconds,
            "status": "completed",
        },
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 1 if invalid_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
