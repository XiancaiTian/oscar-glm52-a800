#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault(
    "SAMPLE_FILE",
    str(TASK_ROOT / "ccw_data_v1/samples_merged_eval11_calib57_seed20260701.jsonl"),
)
os.environ.setdefault("MAX_TOKENS", "256")
os.environ.setdefault("CONCURRENCY", "2")
os.environ.setdefault("SHORT_TTFT_MAX_MS", "10000")
os.environ.setdefault("TTFT_HARD_MAX_PROMPT_TOKENS", "50000")
os.environ.setdefault("SUPPLEMENT_PREFILL_COVERAGE", "1")
os.environ.setdefault("SUPPLEMENT_MAX_ATTEMPTS_PER_SHAPE", "8")

import messages_ccw_bucket2048_ttft_tpot as ccw  # noqa: E402


STAGE_MULTIPLE = {
    "stage_16k": 16384,
    "stage_8k": 8192,
    "stage_60k": 61440,
    "stage_90k": 92160,
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finite(values: list[Any]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def stats(values: list[Any]) -> dict[str, Any]:
    vals = finite(values)
    if not vals:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None, "p90": None}
    ordered = sorted(vals)
    p90_index = int(math.ceil(0.9 * len(ordered))) - 1
    return {
        "count": len(vals),
        "min": round(min(vals), 3),
        "mean": round(statistics.mean(vals), 3),
        "median": round(statistics.median(vals), 3),
        "max": round(max(vals), 3),
        "p90": round(ordered[max(0, min(p90_index, len(ordered) - 1))], 3),
    }


def normalize_prefix_cache_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return "enabled"
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return "disabled"
    return normalized


def padded_shape(tokens: int | None, multiple: int) -> int | None:
    if not isinstance(tokens, int) or tokens <= 0:
        return None
    return int(math.ceil(tokens / multiple) * multiple)


def with_nonce(messages: list[dict[str, str]], *, stage: str, index: int) -> list[dict[str, str]]:
    nonce = f"[v2_prefix_bypass stage={stage} index={index} ts={utc_stamp()} rand={os.urandom(8).hex()}]\n"
    out = [dict(m) for m in messages]
    if not out:
        return [{"role": "user", "content": nonce}]
    if out[0].get("role") != "user":
        out.insert(0, {"role": "user", "content": nonce})
    else:
        out[0]["content"] = nonce + str(out[0].get("content") or "")
    return out


def load_samples_with_nonce(stage: str) -> list[ccw.BenchSample]:
    rows: list[ccw.BenchSample] = []
    with ccw.SAMPLE_FILE.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, 1):
            raw = json.loads(line)
            base_messages, note = ccw.transform_messages(raw.get("messages") or [])
            messages = with_nonce(base_messages, stage=stage, index=index)
            token_count, error = ccw.count_tokens(messages)
            if error:
                raise RuntimeError(f"sample {index} {raw.get('sample_id')} {error}")
            label = f"sample_{index:02d}_{raw.get('sample_id', 'unknown')}"
            rows.append(
                ccw.BenchSample(
                    index=index,
                    label=label,
                    sample_id=str(raw.get("sample_id", "")),
                    input_length_bucket=raw.get("input_length_bucket"),
                    original_glm52_input_tokens=raw.get("glm52_input_tokens"),
                    original_input_length=raw.get("input_length"),
                    transformed_messages=messages,
                    transformed_prompt_tokens=token_count,
                    transformed_text_chars=ccw.text_size(messages),
                    transform_note=note + ";v2_nonce_prepended",
                    prompt_sha256=ccw.prompt_hash(messages),
                    raw_sample=raw,
                )
            )
    return rows


def usage_prompt_tokens(row: dict[str, Any]) -> tuple[int | None, str]:
    usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
    if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] > 0:
        return int(usage["prompt_tokens"]), "usage.prompt_tokens"
    if isinstance(usage.get("input_tokens"), int) and usage["input_tokens"] > 0:
        return int(usage["input_tokens"]), "usage.input_tokens"
    service_tokens = row.get("service_count_tokens")
    if isinstance(service_tokens, int) and service_tokens > 0:
        return service_tokens, "service_count_tokens_fallback_usage_zero"
    return None, "missing"


TTFT_HARD_MAX_PROMPT_TOKENS = int(os.environ.get("TTFT_HARD_MAX_PROMPT_TOKENS", "50000"))
SHORT_TTFT_MAX_MS = float(os.environ.get("SHORT_TTFT_MAX_MS", "10000"))


def hard_gate_prompt_tokens(row: dict[str, Any]) -> int | None:
    usage_tokens = row.get("usage_prompt_tokens_observed")
    if isinstance(usage_tokens, int):
        return usage_tokens
    service_tokens = row.get("service_count_tokens")
    if isinstance(service_tokens, int):
        return service_tokens
    return None


def is_ttft_hard_gate_row(row: dict[str, Any]) -> bool:
    tokens = hard_gate_prompt_tokens(row)
    return isinstance(tokens, int) and tokens <= TTFT_HARD_MAX_PROMPT_TOKENS


def classify(row: dict[str, Any], multiple: int) -> list[str]:
    reasons: list[str] = []
    if row.get("status") != 200:
        reasons.append(f"http_status={row.get('status')}")
    if row.get("error"):
        reasons.append(f"client_error={row.get('error')}")
    if not isinstance(row.get("usage_prompt_tokens_observed"), int):
        reasons.append("missing_usage_prompt_tokens")
    if not isinstance(row.get("ttft_ms"), (int, float)):
        reasons.append("missing_ttft")
    elif is_ttft_hard_gate_row(row) and row["ttft_ms"] >= SHORT_TTFT_MAX_MS:
        reasons.append(f"short_prompt_ttft_ge_{int(SHORT_TTFT_MAX_MS)}ms")
    if not isinstance(row.get("prefill_rpc_ms"), (int, float)):
        reasons.append("missing_real_prefill")
    if not isinstance(row.get("decode_first_ms"), (int, float)):
        reasons.append("missing_decode_first")
    if not isinstance(row.get("proxy_timing"), dict):
        reasons.append("missing_proxy_timing")
    if str(row.get("prefill") or "").strip() not in ccw.PREFILLS:
        reasons.append(f"unexpected_prefill={row.get('prefill')}")
    if str(row.get("decode") or "").strip() != ccw.DECODE:
        reasons.append(f"unexpected_decode={row.get('decode')}")
    if not isinstance(row.get("output_tokens"), int) or row["output_tokens"] <= 0:
        reasons.append("missing_output_tokens")
    if isinstance(row.get("output_tokens"), int) and row["output_tokens"] > 1:
        if not isinstance(row.get("client_tpot_ms"), (int, float)):
            reasons.append("missing_tpot_for_multi_token_output")
    if padded_shape(row.get("service_count_tokens"), multiple) is None:
        reasons.append("missing_service_count_tokens")
    return reasons


def annotate_row(
    row: dict[str, Any],
    sample: ccw.BenchSample,
    *,
    stage: str,
    multiple: int,
    prefix_cache_mode: str,
    coverage_supplement: bool,
) -> dict[str, Any]:
    service_tokens = sample.transformed_prompt_tokens
    row["stage"] = stage
    row["service_count_tokens"] = service_tokens
    usage_tokens, usage_source = usage_prompt_tokens(row)
    row["usage_prompt_tokens_observed"] = usage_tokens
    row["usage_prompt_tokens_source"] = usage_source
    row["padded_shape"] = padded_shape(service_tokens, multiple)
    row["prefix_cache_mode"] = prefix_cache_mode
    row["coverage_supplement"] = coverage_supplement
    row["abnormal_reasons"] = classify(row, multiple)
    row["abnormal"] = bool(row["abnormal_reasons"])
    return row


def shape_prefill_map(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    shape_prefills: dict[str, set[str]] = {}
    for row in rows:
        shape = str(row.get("padded_shape"))
        prefill = str(row.get("prefill") or "").strip()
        if shape != "None" and prefill in ccw.PREFILLS:
            shape_prefills.setdefault(shape, set()).add(prefill)
    return shape_prefills


def missing_prefills_by_shape(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        shape: sorted(ccw.PREFILLS - prefills)
        for shape, prefills in shape_prefill_map(rows).items()
        if prefills != ccw.PREFILLS
    }


def make_supplement_sample(base: ccw.BenchSample, *, stage: str, attempt: int) -> ccw.BenchSample:
    base_messages, note = ccw.transform_messages(base.raw_sample.get("messages") or [])
    messages = with_nonce(base_messages, stage=stage, index=base.index * 1000 + attempt)
    token_count, error = ccw.count_tokens(messages)
    if error:
        raise RuntimeError(f"supplement sample {base.index} {base.sample_id} {error}")
    return ccw.BenchSample(
        index=base.index,
        label=f"supp_pfill_{base.index:02d}_{attempt:02d}_{base.sample_id}",
        sample_id=base.sample_id,
        input_length_bucket=base.input_length_bucket,
        original_glm52_input_tokens=base.original_glm52_input_tokens,
        original_input_length=base.original_input_length,
        transformed_messages=messages,
        transformed_prompt_tokens=token_count,
        transformed_text_chars=ccw.text_size(messages),
        transform_note=note + f";v2_nonce_prefill_coverage_attempt={attempt}",
        prompt_sha256=ccw.prompt_hash(messages),
        raw_sample=base.raw_sample,
    )


def main() -> int:
    stage = os.environ.get("STAGE_NAME") or os.environ.get("V2_STAGE_NAME") or "stage_90k"
    if stage not in STAGE_MULTIPLE:
        raise SystemExit(f"unknown stage {stage!r}; expected one of {sorted(STAGE_MULTIPLE)}")
    multiple = STAGE_MULTIPLE[stage]
    prefix_cache_mode = normalize_prefix_cache_mode(os.environ.get("PREFIX_CACHE_MODE", "disabled"))
    allow_prefix_cache_enabled_final = os.environ.get("ALLOW_PREFIX_CACHE_ENABLED_FINAL", "0") in {
        "1",
        "true",
        "TRUE",
    }
    deploy_manifest_path, deploy_manifest = ccw.latest_deploy_manifest()
    manifest_prefix_enabled = deploy_manifest.get("enable_prefix_caching")
    run_id = os.environ.get("RUN_ID", f"v2_{stage}_final_68_messages_{utc_stamp()}")
    raw_dir = TASK_ROOT / "reports/raw" / run_id
    api_dir = TASK_ROOT / "reports/api"
    accept_dir = TASK_ROOT / "reports/acceptance"
    log_dir = TASK_ROOT / "logs/api"
    for p in (raw_dir, api_dir, accept_dir, log_dir):
        p.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{run_id}.log"
    proxy_log = ccw.latest_proxy_log()
    samples = load_samples_with_nonce(stage)
    samples_by_index = {sample.index: sample for sample in samples}
    text_chars = [sample.transformed_text_chars for sample in samples]
    if len(set(text_chars)) != len(text_chars):
        raise RuntimeError("text_chars are not unique after nonce; proxy timing match unsafe")

    results: list[dict[str, Any]] = []
    supplement_results: list[dict[str, Any]] = []
    with log_path.open("w", encoding="utf-8") as log:
        def log_print(line: str) -> None:
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        log_print(f"run_id={run_id}")
        log_print(f"stage={stage}")
        log_print(f"padding_multiple={multiple}")
        log_print(f"sample_file={ccw.SAMPLE_FILE}")
        log_print(f"sample_count={len(samples)} concurrency={ccw.CONCURRENCY} max_tokens={ccw.MAX_TOKENS}")
        log_print(f"prefix_cache_mode={prefix_cache_mode}")
        log_print(f"endpoint={ccw.ENDPOINT}")
        log_print(f"proxy_log={proxy_log}")
        log_print(f"bucket_config={json.dumps(ccw.bucket_config(), ensure_ascii=False)}")
        log_print(
            "prefill_coverage_supplement="
            f"enabled:{os.environ.get('SUPPLEMENT_PREFILL_COVERAGE')}"
            f" max_attempts_per_shape:{os.environ.get('SUPPLEMENT_MAX_ATTEMPTS_PER_SHAPE')}"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=ccw.CONCURRENCY) as executor:
            futs = {
                executor.submit(ccw.stream_sample, sample, raw_dir, proxy_log): sample
                for sample in samples
            }
            completed = 0
            for fut in concurrent.futures.as_completed(futs):
                row = fut.result()
                completed += 1
                sample = futs[fut]
                row = annotate_row(
                    row,
                    sample,
                    stage=stage,
                    multiple=multiple,
                    prefix_cache_mode=prefix_cache_mode,
                    coverage_supplement=False,
                )
                results.append(row)
                log_print(
                    f"progress {completed}/{len(samples)} index={row['index']} "
                    f"sample_id={row['sample_id']} service_tokens={row['service_count_tokens']} "
                    f"usage_tokens={row['usage_prompt_tokens_observed']} "
                    f"source={row['usage_prompt_tokens_source']} padded={row['padded_shape']} "
                    f"prefill={row['prefill']} ttft={row['ttft_ms']} abnormal={row['abnormal']} "
                    f"reasons={';'.join(row['abnormal_reasons']) or '-'}"
                )

        supplement_enabled = os.environ.get("SUPPLEMENT_PREFILL_COVERAGE", "1") in {
            "1",
            "true",
            "TRUE",
        }
        max_supplement_attempts = int(os.environ.get("SUPPLEMENT_MAX_ATTEMPTS_PER_SHAPE", "8"))
        if supplement_enabled:
            main_rows = sorted(results, key=lambda r: r["index"])
            missing = missing_prefills_by_shape(main_rows)
            log_print(
                "prefill_coverage_after_68 "
                f"shape_prefills={json.dumps({k: sorted(v) for k, v in shape_prefill_map(main_rows).items()}, ensure_ascii=False)} "
                f"missing={json.dumps(missing, ensure_ascii=False)}"
            )
            for shape, missing_prefills in list(missing.items()):
                base_row = next((r for r in main_rows if str(r.get("padded_shape")) == shape), None)
                if not base_row:
                    continue
                base_sample = samples_by_index[int(base_row["index"])]
                for attempt in range(1, max_supplement_attempts + 1):
                    current_missing = missing_prefills_by_shape(main_rows + supplement_results).get(shape, [])
                    if not current_missing:
                        break
                    sample = make_supplement_sample(base_sample, stage=stage, attempt=attempt)
                    row = ccw.stream_sample(sample, raw_dir, proxy_log)
                    row = annotate_row(
                        row,
                        sample,
                        stage=stage,
                        multiple=multiple,
                        prefix_cache_mode=prefix_cache_mode,
                        coverage_supplement=True,
                    )
                    supplement_results.append(row)
                    log_print(
                        f"coverage_supplement shape={shape} attempt={attempt}/{max_supplement_attempts} "
                        f"index={row['index']} service_tokens={row['service_count_tokens']} "
                        f"padded={row['padded_shape']} prefill={row['prefill']} "
                        f"missing_before={current_missing} ttft={row['ttft_ms']} "
                        f"ok={not row['abnormal']} reasons={';'.join(row['abnormal_reasons']) or '-'}"
                    )

    rows = sorted(results, key=lambda r: r["index"])
    coverage_rows = rows + supplement_results
    abnormal = [r for r in rows if r.get("abnormal")]
    hard_gate_rows = [r for r in rows if is_ttft_hard_gate_row(r)]
    non_hard_gate_rows = [r for r in rows if not is_ttft_hard_gate_row(r)]
    long_ttft_rows = [
        r
        for r in non_hard_gate_rows
        if isinstance(r.get("ttft_ms"), (int, float)) and r["ttft_ms"] >= SHORT_TTFT_MAX_MS
    ]
    shape_prefills = shape_prefill_map(coverage_rows)
    shape_prefills_json = {k: sorted(v) for k, v in shape_prefills.items()}
    shapes_missing_prefill = missing_prefills_by_shape(coverage_rows)
    checks = {
        "sample_count_68": len(rows) == 68,
        "concurrency_2": ccw.CONCURRENCY == 2,
        "max_tokens_256": ccw.MAX_TOKENS == 256,
        "all_http_200": all(r.get("status") == 200 for r in rows),
        "all_ttft_recorded": all(isinstance(r.get("ttft_ms"), (int, float)) for r in rows),
        "all_usage_prompt_tokens_recorded": all(
            isinstance(r.get("usage_prompt_tokens_observed"), int) for r in rows
        ),
        "short_prompt_ttft_lt_10s": all(
            isinstance(r.get("ttft_ms"), (int, float)) and r["ttft_ms"] < SHORT_TTFT_MAX_MS
            for r in hard_gate_rows
        ),
        "all_real_prefill_recorded": all(isinstance(r.get("prefill_rpc_ms"), (int, float)) for r in rows),
        "all_decode_first_recorded": all(isinstance(r.get("decode_first_ms"), (int, float)) for r in rows),
        "all_proxy_timing_recorded": all(isinstance(r.get("proxy_timing"), dict) for r in rows),
        "all_output_tokens_positive": all(isinstance(r.get("output_tokens"), int) and r["output_tokens"] > 0 for r in rows),
        "all_service_count_tokens_recorded": all(isinstance(r.get("service_count_tokens"), int) for r in rows),
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
        "all_coverage_supplement_requests_ok": all(
            r.get("status") == 200
            and not r.get("error")
            and isinstance(r.get("proxy_timing"), dict)
            for r in supplement_results
        ),
        "all_observed_shapes_cover_p0_p1": not shapes_missing_prefill,
        "no_abnormal_samples": len(abnormal) == 0,
    }
    pass_value = all(v is True for v in checks.values())
    summary = {
        "run_id": run_id,
        "generated_at": utc_now(),
        "stage": stage,
        "shape_padding_multiple": multiple,
        "endpoint": ccw.ENDPOINT,
        "count_endpoint": ccw.COUNT_ENDPOINT,
        "model": ccw.MODEL,
        "sample_file": str(ccw.SAMPLE_FILE),
        "sample_count": len(rows),
        "concurrency": ccw.CONCURRENCY,
        "max_tokens": ccw.MAX_TOKENS,
        "prefix_cache_mode": prefix_cache_mode,
        "allow_prefix_cache_enabled_final": allow_prefix_cache_enabled_final,
        "deploy_manifest": str(deploy_manifest_path) if deploy_manifest_path else None,
        "manifest_enable_prefix_caching": manifest_prefix_enabled,
        "input_tokens_source": "service /v1/messages/count_tokens; response usage saved separately",
        "ttft_hard_gate": {
            "prompt_tokens_source": "usage.prompt_tokens",
            "max_prompt_tokens": TTFT_HARD_MAX_PROMPT_TOKENS,
            "max_ttft_ms": SHORT_TTFT_MAX_MS,
            "hard_gate_sample_count": len(hard_gate_rows),
            "non_hard_gate_sample_count": len(non_hard_gate_rows),
        },
        "prefills": sorted(ccw.PREFILLS),
        "decode": ccw.DECODE,
        "shape_prefills": shape_prefills_json,
        "shapes_missing_prefill": shapes_missing_prefill,
        "coverage_supplement": {
            "enabled": supplement_enabled,
            "max_attempts_per_shape": max_supplement_attempts,
            "request_count": len(supplement_results),
            "results": supplement_results,
        },
        "stats": {
            "ttft_ms": stats([r.get("ttft_ms") for r in rows]),
            "ttft_ms_hard_gate_prompt_le_50k": stats(
                [r.get("ttft_ms") for r in hard_gate_rows]
            ),
            "ttft_ms_prompt_gt_50k_record_only": stats(
                [r.get("ttft_ms") for r in non_hard_gate_rows]
            ),
            "tpot_ms": stats([r.get("client_tpot_ms") for r in rows]),
            "prefill_ms": stats([r.get("prefill_ms") for r in rows]),
            "prefill_rpc_ms": stats([r.get("prefill_rpc_ms") for r in rows]),
            "decode_first_ms": stats([r.get("decode_first_ms") for r in rows]),
            "output_tokens": stats([r.get("output_tokens") for r in rows]),
        },
        "abnormal_count": len(abnormal),
        "abnormal_indices": [r["index"] for r in abnormal],
        "record_only_long_ttft_ge_10s_count": len(long_ttft_rows),
        "record_only_long_ttft_ge_10s_indices": [r["index"] for r in long_ttft_rows],
        "abnormal_rows": [
            {
                "index": r["index"],
                "sample_id": r["sample_id"],
                "service_count_tokens": r.get("service_count_tokens"),
                "padded_shape": r.get("padded_shape"),
                "prefill": r.get("prefill"),
                "ttft_ms": r.get("ttft_ms"),
                "prefill_rpc_ms": r.get("prefill_rpc_ms"),
                "decode_first_ms": r.get("decode_first_ms"),
                "abnormal_reasons": r.get("abnormal_reasons"),
            }
            for r in abnormal
        ],
        "checks": checks,
        "pass": pass_value,
        "results": rows,
        "coverage_results": coverage_rows,
        "raw_dir": str(raw_dir),
        "log_path": str(log_path),
        "proxy_log": str(proxy_log),
    }
    write_json(raw_dir / "summary.json", summary)
    write_json(api_dir / f"{run_id}_summary.json", summary)
    acceptance = {
        "created_at": utc_now(),
        "stage": stage,
        "gate": "v2_final_68_messages_ttft_tpot",
        "pass": pass_value,
        "summary_json": str(raw_dir / "summary.json"),
        "checks": checks,
        "abnormal_count": len(abnormal),
        "abnormal_indices": [r["index"] for r in abnormal],
        "stats": summary["stats"],
        "prefix_cache_mode": prefix_cache_mode,
        "shape_prefills": shape_prefills_json,
        "shapes_missing_prefill": shapes_missing_prefill,
        "coverage_supplement": {
            "enabled": supplement_enabled,
            "request_count": len(supplement_results),
            "max_attempts_per_shape": max_supplement_attempts,
        },
        "ttft_hard_gate": summary["ttft_hard_gate"],
    }
    accept_path = accept_dir / f"{run_id}_acceptance.json"
    write_json(accept_path, acceptance)
    report_path = api_dir / f"{run_id}_report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# {stage} 68 Messages TTFT/TPOT",
                "",
                f"- pass: `{pass_value}`",
                f"- abnormal_count: `{len(abnormal)}`",
                f"- max_ttft_ms: `{summary['stats']['ttft_ms']['max']}`",
                f"- max_ttft_ms_hard_gate_prompt_le_50k: `{summary['stats']['ttft_ms_hard_gate_prompt_le_50k']['max']}`",
                f"- record_only_long_ttft_ge_10s_count: `{len(long_ttft_rows)}`",
                f"- coverage_supplement_request_count: `{len(supplement_results)}`",
                f"- prefix_cache_mode: `{prefix_cache_mode}`",
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
