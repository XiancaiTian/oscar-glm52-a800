#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("MAX_TOKENS", "256")
os.environ.setdefault("REQUIRED_FAST_PER_PREFILL", "1")
os.environ.setdefault("REQUIRED_REQUESTS_PER_PREFILL", "2")
os.environ.setdefault("MAX_ATTEMPTS_PER_TARGET", "12")
os.environ.setdefault("TTFT_HARD_MAX_PROMPT_TOKENS", "50000")
os.environ.setdefault("SHORT_TTFT_MAX_MS", "10000")

import messages_full_bucket_warmup_197k_probe as warm  # noqa: E402


warm.MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))


STAGE_CONFIG = {
    "stage_16k": ("warmup_shapes_stage_16k_v2.txt", 16384),
    "stage_8k": ("warmup_shapes_stage_8k_v2.txt", 8192),
    "stage_60k": ("warmup_shapes_stage_60k_v2.txt", 61440),
    "stage_90k": ("warmup_shapes_stage_90k_v2.txt", 92160),
}

TTFT_HARD_MAX_PROMPT_TOKENS = int(os.environ.get("TTFT_HARD_MAX_PROMPT_TOKENS", "50000"))
SHORT_TTFT_MAX_MS = float(os.environ.get("SHORT_TTFT_MAX_MS", "10000"))


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_shapes(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def sha256sum(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(values: list[Any]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float))]


def stats(values: list[Any]) -> dict[str, Any]:
    vals = finite(values)
    if not vals:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": len(vals),
        "min": round(min(vals), 3),
        "mean": round(statistics.mean(vals), 3),
        "median": round(statistics.median(vals), 3),
        "max": round(max(vals), 3),
    }


def normalize_prefix_cache_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return "enabled"
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return "disabled"
    return normalized


def row_ok(row: dict[str, Any]) -> bool:
    return (
        row.get("status") == 200
        and row.get("error") is None
        and isinstance(row.get("output_tokens"), int)
        and row["output_tokens"] > 0
        and isinstance(row.get("ttft_first_delta_ms"), (int, float))
        and isinstance(row.get("prefill_rpc_ms"), (int, float))
        and isinstance(row.get("decode_first_ms"), (int, float))
        and isinstance(row.get("proxy_timing"), dict)
        and row["proxy_timing"].get("api") == "/messages"
        and str(row.get("prefill") or "").strip() in warm.PREFILLS
        and str(row.get("decode") or "").strip() == warm.DECODE
    )


def row_fast(row: dict[str, Any]) -> bool:
    return row_ok(row) and float(row["ttft_first_delta_ms"]) < SHORT_TTFT_MAX_MS


def target_requires_fast(target: int) -> bool:
    return target <= TTFT_HARD_MAX_PROMPT_TOKENS


def coverage_satisfied(
    target: int,
    current: dict[str, dict[str, int]],
    required_requests: int,
    required_fast: int,
) -> bool:
    if target_requires_fast(target):
        return all(
            cov["requests"] >= required_requests and cov["fast"] >= required_fast
            for cov in current.values()
        )
    return all(cov["requests"] >= required_requests for cov in current.values())


def parse_targets(stage: str, padding_multiple: int, shapes: list[int]) -> list[int]:
    raw = os.environ.get("DECODE_WARMUP_TARGETS", "").strip()
    if raw:
        targets = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        # Short prompts exercise the same first decode path that caused the
        # first 68-sample request to spend seconds in decode cold setup.
        targets = [min(2048, padding_multiple)]
    if not targets:
        raise RuntimeError(f"{stage} decode warmup target list is empty")
    max_shape = max(shapes)
    for target in targets:
        if target <= 0 or target > max_shape:
            raise ValueError(f"invalid decode warmup target {target}; max_shape={max_shape}")
    return targets


def main() -> int:
    stage = os.environ.get("STAGE_NAME") or os.environ.get("V2_STAGE_NAME") or "stage_90k"
    if stage not in STAGE_CONFIG:
        raise SystemExit(f"unknown stage {stage!r}; expected one of {sorted(STAGE_CONFIG)}")

    shape_file_name, padding_multiple = STAGE_CONFIG[stage]
    shape_list = TASK_ROOT / "configs" / shape_file_name
    shapes = read_shapes(shape_list)
    shape_sha = sha256sum(shape_list)
    targets = parse_targets(stage, padding_multiple, shapes)
    prefix_cache_mode = normalize_prefix_cache_mode(os.environ.get("PREFIX_CACHE_MODE", "disabled"))
    allow_prefix_cache_enabled_final = os.environ.get("ALLOW_PREFIX_CACHE_ENABLED_FINAL", "0") in {
        "1",
        "true",
        "TRUE",
    }
    deploy_manifest_path, deploy_manifest = warm.latest_deploy_manifest()
    manifest_prefix_enabled = deploy_manifest.get("enable_prefix_caching")
    required_fast = int(os.environ.get("REQUIRED_FAST_PER_PREFILL", "1"))
    required_requests = int(os.environ.get("REQUIRED_REQUESTS_PER_PREFILL", "2"))
    max_attempts = int(os.environ.get("MAX_ATTEMPTS_PER_TARGET", "12"))
    run_id = os.environ.get("RUN_ID", f"v2_{stage}_decode_first_warmup_{utc_stamp()}")

    raw_dir = TASK_ROOT / "reports/raw" / run_id
    api_dir = TASK_ROOT / "reports/api"
    log_dir = TASK_ROOT / "logs/api"
    accept_dir = TASK_ROOT / "reports/acceptance"
    for path in (raw_dir, api_dir, log_dir, accept_dir):
        path.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{run_id}.log"
    proxy_log = warm.latest_proxy_log()
    bucket = warm.bucket_config()
    first_header = f"{run_id}_FILLER_PROBE_PAYLOAD:"
    fillers = warm.one_token_fillers(first_header)

    results: list[dict[str, Any]] = []
    coverage: dict[int, dict[str, dict[str, int]]] = {}
    filler_idx = 0

    with log_path.open("w", encoding="utf-8") as log:

        def log_print(line: str) -> None:
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        log_print(f"run_id={run_id}")
        log_print(f"stage={stage}")
        log_print(f"padding_multiple={padding_multiple}")
        log_print(f"shape_list={shape_list}")
        log_print(f"shape_list_sha256={shape_sha}")
        log_print(f"targets={targets}")
        log_print(f"max_tokens={warm.MAX_TOKENS}")
        log_print(f"required_fast_per_prefill={required_fast}")
        log_print(f"required_requests_per_prefill={required_requests}")
        log_print(f"max_attempts_per_target={max_attempts}")
        log_print(f"ttft_hard_max_prompt_tokens={TTFT_HARD_MAX_PROMPT_TOKENS}")
        log_print(f"short_ttft_max_ms={SHORT_TTFT_MAX_MS}")
        log_print(f"prefix_cache_mode={prefix_cache_mode}")
        log_print(f"deploy_manifest={deploy_manifest_path}")
        log_print(f"manifest_enable_prefix_caching={manifest_prefix_enabled}")
        log_print(f"endpoint={warm.ENDPOINT}")
        log_print(f"proxy_log={proxy_log}")
        log_print(f"prefills={sorted(warm.PREFILLS)} decode={warm.DECODE}")
        log_print(f"bucket_config={json.dumps(bucket, ensure_ascii=False)}")

        for target in targets:
            current = {
                prefill: {"requests": 0, "fast": 0}
                for prefill in sorted(warm.PREFILLS)
            }
            attempts = 0
            while not coverage_satisfied(
                target,
                current,
                required_requests,
                required_fast,
            ):
                prepared = []
                for _ in range(2):
                    attempts += 1
                    filler = fillers[filler_idx % len(fillers)]
                    filler_idx += 1
                    label = f"{stage}_decode_warm_{target:06d}_{attempts:02d}"
                    prepared.append(
                        warm.prepare_sample(label, target, filler, "v2_decode_warmup", raw_dir)
                    )
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(prepared)) as executor:
                    futs = [
                        executor.submit(warm.stream_prepared_sample, sample, raw_dir, proxy_log)
                        for sample in prepared
                    ]
                    for fut in concurrent.futures.as_completed(futs):
                        row = fut.result()
                        row["stage"] = stage
                        row["shape_padding_multiple"] = padding_multiple
                        row["shape_list"] = str(shape_list)
                        row["shape_list_sha256"] = shape_sha
                        row["max_tokens"] = warm.MAX_TOKENS
                        row["prefix_cache_mode"] = prefix_cache_mode
                        row["request_ok"] = row_ok(row)
                        row["fast"] = row_fast(row)
                        row["ok"] = row["request_ok"]
                        results.append(row)
                        prefill = str(row.get("prefill") or "").strip()
                        if row["ok"] and prefill in current:
                            current[prefill]["requests"] += 1
                            if row["fast"]:
                                current[prefill]["fast"] += 1
                        log_print(warm.row_line(row) + f" ok={row['ok']} fast={row['fast']}")
                log_print(
                    f"decode_warmup_progress stage={stage} target={target} "
                    f"coverage={current} attempts={attempts}"
                )
                if attempts >= max_attempts:
                    break
                if not coverage_satisfied(
                    target,
                    current,
                    required_requests,
                    required_fast,
                ):
                    time.sleep(1)
            coverage[target] = current

    hard_gate_targets = [target for target in targets if target_requires_fast(target)]
    fast_coverage = {
        target: coverage[target]
        for target in hard_gate_targets
        if target in coverage
    }
    checks = {
        "stage_known": stage in STAGE_CONFIG,
        "max_tokens_256": warm.MAX_TOKENS == 256,
        "targets_non_empty": bool(targets),
        "all_requests_completed": all(row.get("request_ok") is True for row in results),
        "all_prefills_have_required_requests": all(
            cov["requests"] >= required_requests
            for target_cov in coverage.values()
            for cov in target_cov.values()
        ),
        "hard_gate_targets_have_fast_hit": all(
            cov["fast"] >= required_fast
            for target_cov in fast_coverage.values()
            for cov in target_cov.values()
        ),
        "prefix_cache_mode_valid": prefix_cache_mode in {"disabled", "enabled"},
        "prefix_cache_disabled_for_experiment": (
            True if allow_prefix_cache_enabled_final else prefix_cache_mode == "disabled"
        ),
        "prefix_cache_enabled_for_final": (
            prefix_cache_mode == "enabled" if allow_prefix_cache_enabled_final else True
        ),
        "prefix_cache_manifest_matches_gate": (
            manifest_prefix_enabled is True
            if allow_prefix_cache_enabled_final
            else manifest_prefix_enabled is False
        ),
        "all_proxy_timing_recorded": all(
            isinstance(row.get("proxy_timing"), dict) for row in results
        ),
        "all_content_unique": len({row.get("content_sha256") for row in results}) == len(results),
    }
    pass_value = all(value is True for value in checks.values())
    summary = {
        "run_id": run_id,
        "generated_at": utc_now(),
        "stage": stage,
        "shape_padding_multiple": padding_multiple,
        "shape_list": str(shape_list),
        "shape_list_sha256": shape_sha,
        "targets": targets,
        "ttft_hard_max_prompt_tokens": TTFT_HARD_MAX_PROMPT_TOKENS,
        "short_ttft_max_ms": SHORT_TTFT_MAX_MS,
        "hard_gate_targets": hard_gate_targets,
        "max_tokens": warm.MAX_TOKENS,
        "required_fast_per_prefill": required_fast,
        "required_requests_per_prefill": required_requests,
        "prefix_cache_mode": prefix_cache_mode,
        "allow_prefix_cache_enabled_final": allow_prefix_cache_enabled_final,
        "deploy_manifest": str(deploy_manifest_path) if deploy_manifest_path else None,
        "manifest_enable_prefix_caching": manifest_prefix_enabled,
        "prefills": sorted(warm.PREFILLS),
        "decode": warm.DECODE,
        "endpoint": warm.ENDPOINT,
        "proxy_log": str(proxy_log),
        "prefill_shape_bucket": bucket,
        "coverage": coverage,
        "stats": {
            "ttft_ms": stats([row.get("ttft_first_delta_ms") for row in results]),
            "prefill_rpc_ms": stats([row.get("prefill_rpc_ms") for row in results]),
            "decode_first_ms": stats([row.get("decode_first_ms") for row in results]),
            "output_tokens": stats([row.get("output_tokens") for row in results]),
        },
        "checks": checks,
        "pass": pass_value,
        "results": results,
        "raw_dir": str(raw_dir),
        "log_path": str(log_path),
    }
    write_json(raw_dir / "summary.json", summary)
    write_json(api_dir / f"{run_id}_summary.json", summary)
    acceptance = {
        "created_at": utc_now(),
        "stage": stage,
        "gate": "v2_decode_first_warmup",
        "pass": pass_value,
        "summary_json": str(raw_dir / "summary.json"),
        "shape_list": str(shape_list),
        "shape_list_sha256": shape_sha,
        "max_tokens": warm.MAX_TOKENS,
        "ttft_hard_max_prompt_tokens": TTFT_HARD_MAX_PROMPT_TOKENS,
        "short_ttft_max_ms": SHORT_TTFT_MAX_MS,
        "hard_gate_targets": hard_gate_targets,
        "prefix_cache_mode": prefix_cache_mode,
        "allow_prefix_cache_enabled_final": allow_prefix_cache_enabled_final,
        "deploy_manifest": str(deploy_manifest_path) if deploy_manifest_path else None,
        "manifest_enable_prefix_caching": manifest_prefix_enabled,
        "checks": checks,
        "coverage": coverage,
        "stats": summary["stats"],
    }
    accept_path = accept_dir / f"{run_id}_acceptance.json"
    write_json(accept_path, acceptance)
    report_path = api_dir / f"{run_id}_report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# {stage} Decode First Warmup",
                "",
                f"- pass: `{pass_value}`",
                f"- max_tokens: `{warm.MAX_TOKENS}`",
                f"- targets: `{targets}`",
                f"- acceptance: `{accept_path}`",
                f"- raw_summary: `{raw_dir / 'summary.json'}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(acceptance, ensure_ascii=False, indent=2), flush=True)
    return 0 if pass_value else 1


if __name__ == "__main__":
    raise SystemExit(main())
