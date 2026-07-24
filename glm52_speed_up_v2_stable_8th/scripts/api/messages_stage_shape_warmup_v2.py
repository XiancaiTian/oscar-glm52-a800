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

os.environ.setdefault("MAX_TOKENS", "1")
os.environ.setdefault("REQUIRED_PER_PREFILL", "3")
os.environ.setdefault("MAX_ATTEMPTS_PER_LENGTH", "18")
os.environ.setdefault("FORMAL_TARGETS", "")
os.environ.setdefault("TTFT_HARD_MAX_PROMPT_TOKENS", "50000")
os.environ.setdefault("SHORT_TTFT_MAX_MS", "10000")

import messages_full_bucket_warmup_197k_probe as warm  # noqa: E402


STAGE_CONFIG = {
    "stage_16k": ("warmup_shapes_stage_16k_v2.txt", 16384, 11),
    "stage_8k": ("warmup_shapes_stage_8k_v2.txt", 8192, 22),
    "stage_60k": ("warmup_shapes_stage_60k_v2.txt", 61440, 3),
    "stage_90k": ("warmup_shapes_stage_90k_v2.txt", 92160, 3),
}

STAGE_EXPLICIT_SHAPES = {
    "stage_90k": [92160, 184320, 200704],
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
    values = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if values != sorted(values):
        raise ValueError(f"shape list is not sorted: {path}")
    return values


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


def message_start_cache_read_tokens(row: dict[str, Any]) -> int | None:
    value = row.get("message_start_cache_read_input_tokens")
    if isinstance(value, int):
        return value
    usage = row.get("message_start_usage")
    if isinstance(usage, dict) and isinstance(usage.get("cache_read_input_tokens"), int):
        return int(usage["cache_read_input_tokens"])
    return None


def row_fast(row: dict[str, Any]) -> bool:
    return (
        row.get("status") == 200
        and row.get("error") is None
        and row.get("output_tokens") == 1
        and isinstance(row.get("ttft_first_delta_ms"), (int, float))
        and row["ttft_first_delta_ms"] < SHORT_TTFT_MAX_MS
        and isinstance(row.get("prefill_rpc_ms"), (int, float))
        and isinstance(row.get("decode_first_ms"), (int, float))
        and isinstance(row.get("proxy_timing"), dict)
        and row["proxy_timing"].get("api") == "/messages"
        and str(row.get("prefill") or "").strip() in warm.PREFILLS
        and str(row.get("decode") or "").strip() == warm.DECODE
    )


def request_ok(row: dict[str, Any]) -> bool:
    return (
        row.get("status") == 200
        and row.get("error") is None
        and row.get("output_tokens") == 1
        and isinstance(row.get("ttft_first_delta_ms"), (int, float))
        and isinstance(row.get("prefill_rpc_ms"), (int, float))
        and isinstance(row.get("decode_first_ms"), (int, float))
        and isinstance(row.get("proxy_timing"), dict)
        and row["proxy_timing"].get("api") == "/messages"
        and str(row.get("prefill") or "").strip() in warm.PREFILLS
        and str(row.get("decode") or "").strip() == warm.DECODE
    )


def main() -> int:
    stage = os.environ.get("STAGE_NAME") or os.environ.get("V2_STAGE_NAME") or "stage_90k"
    if stage not in STAGE_CONFIG:
        raise SystemExit(f"unknown stage {stage!r}; expected one of {sorted(STAGE_CONFIG)}")

    shape_file_name, padding_multiple, expected_count = STAGE_CONFIG[stage]
    shape_list = TASK_ROOT / "configs" / shape_file_name
    shapes = read_shapes(shape_list)
    shape_sha = sha256sum(shape_list)
    expected_shapes = STAGE_EXPLICIT_SHAPES.get(stage)
    if len(shapes) != expected_count:
        raise RuntimeError(f"{stage} shape count {len(shapes)} != {expected_count}")

    required_per_prefill = int(os.environ.get("REQUIRED_PER_PREFILL", "3"))
    max_attempts = int(os.environ.get("MAX_ATTEMPTS_PER_LENGTH", "18"))
    prefix_cache_mode = normalize_prefix_cache_mode(os.environ.get("PREFIX_CACHE_MODE", "disabled"))
    run_id = os.environ.get("RUN_ID", f"v2_{stage}_shape_warmup_{utc_stamp()}")
    raw_dir = TASK_ROOT / "reports/raw" / run_id
    api_dir = TASK_ROOT / "reports/api"
    log_dir = TASK_ROOT / "logs/api"
    accept_dir = TASK_ROOT / "reports/acceptance"
    for p in (raw_dir, api_dir, log_dir, accept_dir):
        p.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{run_id}.log"
    proxy_log = warm.latest_proxy_log()
    bucket = warm.bucket_config()
    first_header = f"{run_id}_FILLER_PROBE_PAYLOAD:"
    fillers = warm.one_token_fillers(first_header)

    results: list[dict[str, Any]] = []
    coverage: dict[int, dict[str, int]] = {}
    cross_shape_transitions: dict[str, int] = {}
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
        log_print(f"endpoint={warm.ENDPOINT}")
        log_print(f"proxy_log={proxy_log}")
        log_print(f"prefills={sorted(warm.PREFILLS)} decode={warm.DECODE}")
        log_print(f"bucket_config={json.dumps(bucket, ensure_ascii=False)}")
        log_print(f"required_per_prefill={required_per_prefill} max_attempts={max_attempts}")
        log_print(f"prefix_cache_mode={prefix_cache_mode}")

        previous_target: int | None = None
        for shape_index, target in enumerate(shapes, start=1):
            current = {prefill: 0 for prefill in warm.PREFILLS}
            attempts = 0
            while not all(count >= required_per_prefill for count in current.values()):
                missing = [p for p, count in current.items() if count < required_per_prefill]
                batch_size = min(2, max(1, len(missing)))
                prepared = []
                transition_from = previous_target if attempts == 0 and previous_target != target else None
                for _ in range(batch_size):
                    attempts += 1
                    filler = fillers[filler_idx % len(fillers)]
                    filler_idx += 1
                    label = f"{stage}_warm_{target:06d}_{attempts:02d}"
                    sample = warm.prepare_sample(label, target, filler, "v7_stage_warmup", raw_dir)
                    prepared.append(sample)
                    if transition_from is not None:
                        cross_shape_transitions[sample.label] = transition_from
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
                        row["prefix_cache_mode"] = prefix_cache_mode
                        if row.get("label") in cross_shape_transitions:
                            row["cross_shape_transition_from"] = cross_shape_transitions[row["label"]]
                            row["cross_shape_transition_to"] = target
                            row["cross_shape_cache_read_input_tokens"] = message_start_cache_read_tokens(row)
                        row["request_ok"] = request_ok(row)
                        row["fast"] = row_fast(row)
                        row["ok"] = row["request_ok"]
                        results.append(row)
                        prefill = str(row.get("prefill") or "").strip()
                        if row["ok"] and prefill in current:
                            current[prefill] += 1
                        log_print(warm.row_line(row) + f" ok={row['ok']}")
                log_print(
                    f"warmup_progress stage={stage} index={shape_index}/{len(shapes)} "
                    f"target={target} coverage={current} attempts={attempts}"
                )
                if attempts >= max_attempts:
                    break
                if not all(count >= required_per_prefill for count in current.values()):
                    time.sleep(1)
            coverage[target] = current
            previous_target = target

    warmed = {
        target
        for target, cov in coverage.items()
        if all(count >= required_per_prefill for count in cov.values())
    }
    short_shapes = [shape for shape in shapes if shape <= TTFT_HARD_MAX_PROMPT_TOKENS]
    short_shape_fast_coverage = {
        shape: {
            prefill: sum(
                1
                for row in results
                if row.get("target_tokens") == shape
                and str(row.get("prefill") or "").strip() == prefill
                and row.get("fast") is True
            )
            for prefill in sorted(warm.PREFILLS)
        }
        for shape in short_shapes
    }
    cross_shape_rows = [
        row for row in results if row.get("cross_shape_transition_from") is not None
    ]
    cross_shape_cache_evidence_present = all(
        isinstance(row.get("cross_shape_cache_read_input_tokens"), int)
        for row in cross_shape_rows
    )
    cross_shape_cache_hit_rows = [
        row
        for row in cross_shape_rows
        if row.get("cross_shape_cache_read_input_tokens") not in (0, None)
    ]
    cross_shape_cache_read_tokens_total = sum(
        int(row.get("cross_shape_cache_read_input_tokens") or 0)
        for row in cross_shape_rows
        if isinstance(row.get("cross_shape_cache_read_input_tokens"), int)
    )
    cross_shape_cache_hit_rate = (
        len(cross_shape_cache_hit_rows) / len(cross_shape_rows) if cross_shape_rows else 0.0
    )
    prefix_cache_enabled = prefix_cache_mode == "enabled"
    checks = {
        "stage_known": stage in STAGE_CONFIG,
        "prefix_cache_mode_valid": prefix_cache_mode in {"disabled", "enabled"},
        "shape_count_matches": len(shapes) == expected_count,
        "shape_list_sorted": shapes == sorted(shapes),
        "shape_list_expected": (
            True if expected_shapes is None else shapes == expected_shapes
        ),
        "shape_step_matches": (
            True
            if expected_shapes is not None and shapes == expected_shapes
            else all((b - a) == padding_multiple for a, b in zip(shapes, shapes[1:]))
        ),
        "all_shapes_each_prefill_ge_3": len(warmed) == len(shapes),
        "short_shapes_each_prefill_has_lt_10s": all(
            all(count >= 1 for count in per_prefill.values())
            for per_prefill in short_shape_fast_coverage.values()
        ),
        "all_requests_completed": all(row.get("request_ok") is True for row in results),
        "fast_hit_evidence_recorded": all("fast" in row for row in results),
        "all_proxy_timing_recorded": all(
            isinstance(row.get("proxy_timing"), dict) for row in results
        ),
        "all_content_unique": len({row.get("content_sha256") for row in results}) == len(results),
        "enabled_cross_shape_transition_rows_present": (
            True if not prefix_cache_enabled else len(cross_shape_rows) >= max(0, len(shapes) - 1)
        ),
        "enabled_cross_shape_cache_evidence_present": (
            True if not prefix_cache_enabled else cross_shape_cache_evidence_present
        ),
        "enabled_cross_shape_cache_hit_rate_zero": (
            True
            if not prefix_cache_enabled
            else cross_shape_cache_read_tokens_total == 0 and not cross_shape_cache_hit_rows
        ),
    }
    pass_value = all(value is True for value in checks.values())
    summary = {
        "run_id": run_id,
        "generated_at": utc_now(),
        "stage": stage,
        "shape_padding_multiple": padding_multiple,
        "shape_list": str(shape_list),
        "shape_list_sha256": shape_sha,
        "shape_count": len(shapes),
        "shapes": shapes,
        "required_per_prefill": required_per_prefill,
        "ttft_hard_gate": {
            "max_prompt_tokens": TTFT_HARD_MAX_PROMPT_TOKENS,
            "max_ttft_ms": SHORT_TTFT_MAX_MS,
            "short_shapes": short_shapes,
        },
        "prefix_cache_mode": prefix_cache_mode,
        "prefix_cache_disabled_for_experiment": prefix_cache_mode == "disabled",
        "prefix_cache_enabled_for_experiment": prefix_cache_mode == "enabled",
        "prefills": sorted(warm.PREFILLS),
        "decode": warm.DECODE,
        "endpoint": warm.ENDPOINT,
        "proxy_log": str(proxy_log),
        "prefill_shape_bucket": bucket,
        "coverage": coverage,
        "incomplete_shapes": [shape for shape in shapes if shape not in warmed],
        "fast_counts": {
            str(target): {
                prefill: sum(
                    1
                    for row in results
                    if row.get("target_tokens") == target
                    and str(row.get("prefill") or "").strip() == prefill
                    and row.get("fast") is True
                )
                for prefill in sorted(warm.PREFILLS)
            }
            for target in shapes
        },
        "short_shape_fast_coverage": short_shape_fast_coverage,
        "cross_shape_cache": {
            "required": prefix_cache_enabled,
            "rows": len(cross_shape_rows),
            "hit_rows": len(cross_shape_cache_hit_rows),
            "cache_read_tokens_total": cross_shape_cache_read_tokens_total,
            "hit_rate": round(cross_shape_cache_hit_rate, 6),
            "evidence_present": cross_shape_cache_evidence_present,
            "violations": cross_shape_cache_hit_rows,
        },
        "stats": {
            "ttft_ms": stats([row.get("ttft_first_delta_ms") for row in results]),
            "prefill_ms": stats([row.get("prefill_ms") for row in results]),
            "prefill_rpc_ms": stats([row.get("prefill_rpc_ms") for row in results]),
            "decode_first_ms": stats([row.get("decode_first_ms") for row in results]),
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
        "gate": "v7_stage_shape_warmup",
        "pass": pass_value,
        "summary_json": str(raw_dir / "summary.json"),
        "shape_list": str(shape_list),
        "shape_list_sha256": shape_sha,
        "shape_count": len(shapes),
        "required_per_prefill": required_per_prefill,
        "prefix_cache_mode": prefix_cache_mode,
        "prefix_cache_disabled_for_experiment": prefix_cache_mode == "disabled",
        "prefix_cache_enabled_for_experiment": prefix_cache_mode == "enabled",
        "cross_shape_cache": summary["cross_shape_cache"],
        "checks": checks,
        "incomplete_shapes": summary["incomplete_shapes"],
        "coverage": coverage,
        "stats": summary["stats"],
    }
    accept_path = accept_dir / f"{run_id}_acceptance.json"
    write_json(accept_path, acceptance)
    report_path = api_dir / f"{run_id}_report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# {stage} Shape Warmup",
                "",
                f"- pass: `{pass_value}`",
                f"- shape_list: `{shape_list}`",
                f"- shape_list_sha256: `{shape_sha}`",
                f"- required_per_prefill: `{required_per_prefill}`",
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
