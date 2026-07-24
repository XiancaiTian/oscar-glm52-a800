#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ROOT = Path(os.environ.get("TASK_ROOT", str(Path(__file__).resolve().parents[2])))
SAMPLE_FILE = Path(
    os.environ.get(
        "SAMPLE_FILE",
        str(TASK_ROOT / "ccw_data_v1/samples_bucket2048_k2_seed20260701.jsonl"),
    )
)
ENDPOINT = os.environ.get("MESSAGES_ENDPOINT", "http://127.0.0.1:19181/v1/messages")
COUNT_ENDPOINT = os.environ.get(
    "MESSAGES_COUNT_ENDPOINT", "http://127.0.0.1:19181/v1/messages/count_tokens"
)
MODEL = os.environ.get("MODEL", os.environ.get("MODEL_ID", "GLM-5.2-FP8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "2"))
REQUEST_TIMEOUT_S = int(os.environ.get("REQUEST_TIMEOUT_S", "2400"))
PROXY_POLL_TIMEOUT_S = int(os.environ.get("PROXY_POLL_TIMEOUT_S", "120"))
SHORT_PROMPT_TOKEN_MAX = int(os.environ.get("SHORT_PROMPT_TOKEN_MAX", "50000"))
SHORT_TTFT_MAX_MS = float(os.environ.get("SHORT_TTFT_MAX_MS", "10000"))
GLOBAL_TTFT_ABNORMAL_MS = float(os.environ.get("GLOBAL_TTFT_ABNORMAL_MS", "60000"))
PREFILL_QUEUE_ABNORMAL_MS = float(os.environ.get("PREFILL_QUEUE_ABNORMAL_MS", "1000"))
PREFILLS = {
    x.strip()
    for x in os.environ.get(
        "PREFILLS", "127.0.0.1:19182"
    ).split(",")
    if x.strip()
}
DECODE = os.environ.get("DECODE", "127.0.0.1:19183")


@dataclass(frozen=True)
class BenchSample:
    index: int
    label: str
    sample_id: str
    input_length_bucket: str | None
    original_glm52_input_tokens: int | None
    original_input_length: int | None
    transformed_messages: list[dict[str, str]]
    transformed_prompt_tokens: int | None
    transformed_text_chars: int
    transform_note: str
    prompt_sha256: str
    raw_sample: dict[str, Any]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge_usage(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if (
            key in {"input_tokens", "prompt_tokens"}
            and value == 0
            and isinstance(target.get(key), int)
            and target[key] > 0
        ):
            continue
        target[key] = value


def latest_proxy_log() -> Path:
    log_root = (
        TASK_ROOT
        / "logs/deploy/stage62_glm52_opt_indexshare_rdma_mtp_skipshare_gmem092_2p1d_v2_stable"
    )
    logs = sorted(log_root.glob("*_proxy*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise RuntimeError(f"no proxy log found under {log_root}")
    return logs[-1]


def latest_deploy_manifest() -> tuple[Path | None, dict[str, Any]]:
    manifests = sorted(
        (TASK_ROOT / "reports/raw").glob("*_deploy_manifest.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not manifests:
        return None, {}
    path = manifests[-1]
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path, {}


def bucket_config() -> dict[str, Any]:
    manifest_path, manifest = latest_deploy_manifest()
    return {
        "enabled": str(manifest.get("prefill_shape_bucket", "")),
        "multiple": int(manifest.get("prefill_shape_bucket_multiple", 0) or 0),
        "max": int(manifest.get("prefill_shape_bucket_max", 0) or 0),
        "pad_metadata": str(manifest.get("prefill_shape_bucket_pad_metadata", "")),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_run_id": manifest.get("run_id"),
    }


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def transform_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, str]], str]:
    transformed: list[dict[str, str]] = []
    converted_roles: list[str] = []
    for msg in messages:
        original_role = str(msg.get("role", "user"))
        text = content_to_text(msg.get("content"))
        extras: dict[str, Any] = {}
        for key in ("tool_calls", "function_call", "name", "tool_call_id"):
            if key in msg:
                extras[key] = msg[key]
        if extras:
            extra_text = json.dumps(extras, ensure_ascii=False, sort_keys=True)
            text = f"{text}\n\n[message_extra]\n{extra_text}" if text else extra_text

        if original_role in {"user", "assistant"}:
            role = original_role
        else:
            role = "user"
            converted_roles.append(original_role)
            text = f"[{original_role}]\n{text}"

        if transformed and transformed[-1]["role"] == role:
            transformed[-1]["content"] += "\n\n" + text
        else:
            transformed.append({"role": role, "content": text})

    if not transformed:
        transformed = [{"role": "user", "content": ""}]
    if transformed[0]["role"] == "assistant":
        transformed.insert(0, {"role": "user", "content": ""})
    note = "no_role_conversion"
    if converted_roles:
        unique_roles = ",".join(sorted(set(converted_roles)))
        note = f"converted_to_user_roles={unique_roles};merged_adjacent_same_role"
    return transformed, note


def text_size(messages: list[dict[str, str]]) -> int:
    return sum(len(msg.get("content", "")) for msg in messages)


def prompt_hash(messages: list[dict[str, str]]) -> str:
    data = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def post_json(url: str, payload: dict[str, Any], timeout: int = 600) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"raw": text}
        return exc.code, parsed


def count_tokens(messages: list[dict[str, str]]) -> tuple[int | None, str | None]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
    }
    status, body = post_json(COUNT_ENDPOINT, payload, timeout=REQUEST_TIMEOUT_S)
    if status != 200:
        return None, f"count_tokens status={status} body={body}"
    value = body.get("input_tokens")
    if not isinstance(value, int):
        return None, f"count_tokens missing input_tokens body={body}"
    return value, None


def load_samples() -> list[BenchSample]:
    samples: list[BenchSample] = []
    with SAMPLE_FILE.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, 1):
            raw = json.loads(line)
            messages, note = transform_messages(raw.get("messages") or [])
            token_count, error = count_tokens(messages)
            if error:
                raise RuntimeError(f"sample {index} {raw.get('sample_id')} {error}")
            label = f"sample_{index:02d}_{raw.get('sample_id', 'unknown')}"
            samples.append(
                BenchSample(
                    index=index,
                    label=label,
                    sample_id=str(raw.get("sample_id", "")),
                    input_length_bucket=raw.get("input_length_bucket"),
                    original_glm52_input_tokens=raw.get("glm52_input_tokens"),
                    original_input_length=raw.get("input_length"),
                    transformed_messages=messages,
                    transformed_prompt_tokens=token_count,
                    transformed_text_chars=text_size(messages),
                    transform_note=note,
                    prompt_sha256=prompt_hash(messages),
                    raw_sample=raw,
                )
            )
    return samples


def proxy_log_size(proxy_log: Path) -> int:
    return proxy_log.stat().st_size if proxy_log.exists() else 0


def parse_proxy_pairs(line: str) -> dict[str, Any] | None:
    if "PROXY_TIMING " not in line and "PROXY_DONE " not in line:
        return None
    pairs = dict(re.findall(r"(\w+)=([^ ]+)", line))
    if not pairs:
        return None
    out: dict[str, Any] = {"raw": line.rstrip("\n")}
    if "PROXY_TIMING " in line:
        out["kind"] = "timing"
    elif "PROXY_DONE " in line:
        out["kind"] = "done"
    for key, value in pairs.items():
        value = value.strip()
        if key in {
            "prefill_ms",
            "prefill_queue_ms",
            "prefill_rpc_ms",
            "decode_first_ms",
            "ttft_proxy_ms",
            "total_ms",
        }:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
        elif key in {"text_chars"}:
            try:
                out[key] = int(value)
            except ValueError:
                out[key] = value
        else:
            out[key] = value
    return out


def poll_proxy_records(
    proxy_log: Path, offset: int, text_chars: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    deadline = time.time() + PROXY_POLL_TIMEOUT_S
    timing: dict[str, Any] | None = None
    done: dict[str, Any] | None = None
    while time.time() < deadline:
        if not proxy_log.exists():
            time.sleep(1)
            continue
        with proxy_log.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            lines = f.readlines()
        for line in lines:
            parsed = parse_proxy_pairs(line)
            if not parsed:
                continue
            if parsed.get("api") != "/messages":
                continue
            if parsed.get("kind") == "timing":
                if str(parsed.get("max_tokens")) != str(MAX_TOKENS):
                    continue
                if parsed.get("text_chars") != text_chars:
                    continue
                timing = parsed
            elif parsed.get("kind") == "done":
                if timing and parsed.get("request_id") == timing.get("request_id"):
                    done = parsed
        if timing and done:
            return timing, done
        time.sleep(1)
    return timing, done


def round_ms(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def extract_output_tokens(usage: dict[str, Any], fallback: int) -> int | None:
    if not isinstance(usage, dict):
        return fallback if fallback else None
    for key in ("output_tokens", "completion_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return fallback if fallback else None


def classify_abnormal(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row.get("status") != 200:
        reasons.append(f"http_status={row.get('status')}")
    if row.get("error"):
        reasons.append(f"client_error={row.get('error')}")
    if not isinstance(row.get("ttft_ms"), (int, float)):
        reasons.append("missing_ttft")
    if not isinstance(row.get("client_tpot_ms"), (int, float)):
        reasons.append("missing_tpot")
    if not isinstance(row.get("prefill_rpc_ms"), (int, float)):
        reasons.append("missing_real_prefill")
    if not isinstance(row.get("decode_first_ms"), (int, float)):
        reasons.append("missing_decode_first")

    prompt_tokens = row.get("transformed_prompt_tokens")
    ttft_ms = row.get("ttft_ms")
    if isinstance(prompt_tokens, int) and isinstance(ttft_ms, (int, float)):
        if prompt_tokens <= SHORT_PROMPT_TOKEN_MAX and ttft_ms >= SHORT_TTFT_MAX_MS:
            reasons.append(
                f"ttft_ge_{SHORT_TTFT_MAX_MS:g}ms_for_le_{SHORT_PROMPT_TOKEN_MAX}_tokens"
            )
        if ttft_ms >= GLOBAL_TTFT_ABNORMAL_MS:
            reasons.append(f"ttft_ge_{GLOBAL_TTFT_ABNORMAL_MS:g}ms")

    queue_ms = row.get("prefill_queue_ms")
    if isinstance(queue_ms, (int, float)) and queue_ms >= PREFILL_QUEUE_ABNORMAL_MS:
        reasons.append(f"prefill_queue_ge_{PREFILL_QUEUE_ABNORMAL_MS:g}ms")
    return reasons


def stream_sample(sample: BenchSample, raw_dir: Path, proxy_log: Path) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": sample.transformed_messages,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    request_path = raw_dir / f"{sample.label}_request.json"
    response_path = raw_dir / f"{sample.label}_stream_events.jsonl"
    write_json(
        request_path,
        {
            "payload": payload,
            "sample_id": sample.sample_id,
            "transform_note": sample.transform_note,
            "original_glm52_input_tokens": sample.original_glm52_input_tokens,
            "transformed_prompt_tokens": sample.transformed_prompt_tokens,
            "prompt_sha256": sample.prompt_sha256,
        },
    )

    offset = proxy_log_size(proxy_log)
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )

    started = time.perf_counter()
    status: int | None = None
    error: str | None = None
    first_line_ms: float | None = None
    first_data_ms: float | None = None
    message_start_ms: float | None = None
    first_token_delta_ms: float | None = None
    last_token_delta_ms: float | None = None
    message_stop_ms: float | None = None
    first_delta_kind: str | None = None
    event_count = 0
    token_delta_events = 0
    usage: dict[str, Any] = {}
    message_start_usage: dict[str, Any] = {}
    final_usage: dict[str, Any] = {}
    output_text_parts: list[str] = []

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp, response_path.open(
            "w", encoding="utf-8"
        ) as out:
            status = resp.status
            current_event: str | None = None
            for raw_line in resp:
                now_ms = (time.perf_counter() - started) * 1000
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line and first_line_ms is None:
                    first_line_ms = now_ms
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                raw_data = line.split(":", 1)[1].strip()
                if raw_data == "[DONE]":
                    out.write(json.dumps({"elapsed_ms": round_ms(now_ms), "done": True}) + "\n")
                    break
                try:
                    parsed = json.loads(raw_data)
                except Exception:
                    parsed = {"raw": raw_data}
                event_count += 1
                out.write(
                    json.dumps(
                        {"elapsed_ms": round_ms(now_ms), "event": current_event, "data": parsed},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if first_data_ms is None:
                    first_data_ms = now_ms

                event_type = parsed.get("type") if isinstance(parsed, dict) else None
                if event_type == "message_start" and message_start_ms is None:
                    message_start_ms = now_ms
                if event_type == "message_start":
                    message = parsed.get("message") if isinstance(parsed, dict) else None
                    if isinstance(message, dict) and isinstance(message.get("usage"), dict):
                        message_start_usage.update(message["usage"])
                        merge_usage(usage, message["usage"])
                if isinstance(parsed, dict) and isinstance(parsed.get("usage"), dict):
                    if event_type != "message_start":
                        final_usage.update(parsed["usage"])
                    merge_usage(usage, parsed["usage"])
                if event_type == "message_stop":
                    message_stop_ms = now_ms
                    break
                if event_type == "content_block_delta":
                    delta = parsed.get("delta") if isinstance(parsed, dict) else {}
                    if isinstance(delta, dict):
                        value = delta.get("text") or delta.get("thinking") or delta.get("partial_json")
                        delta_type = str(delta.get("type") or "content_block_delta")
                        if value not in (None, "", [], {}):
                            token_delta_events += 1
                            output_text_parts.append(str(value))
                            if first_token_delta_ms is None:
                                first_token_delta_ms = now_ms
                                first_delta_kind = delta_type
                            last_token_delta_ms = now_ms

                choices = parsed.get("choices") if isinstance(parsed, dict) else None
                if isinstance(choices, list) and choices:
                    choice = choices[0]
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    if isinstance(delta, dict):
                        for key in (
                            "content",
                            "reasoning",
                            "reasoning_content",
                            "tool_calls",
                            "function_call",
                        ):
                            value = delta.get(key)
                            if value not in (None, "", [], {}):
                                token_delta_events += 1
                                output_text_parts.append(str(value))
                                if first_token_delta_ms is None:
                                    first_token_delta_ms = now_ms
                                    first_delta_kind = key
                                last_token_delta_ms = now_ms
                                break
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)

    elapsed_ms = (time.perf_counter() - started) * 1000
    proxy_timing, proxy_done = poll_proxy_records(
        proxy_log, offset, sample.transformed_text_chars
    )
    output_tokens = extract_output_tokens(usage, token_delta_events)
    client_tpot_ms: float | None = None
    client_tpot_e2e_ms: float | None = None
    if (
        output_tokens is not None
        and output_tokens > 1
        and first_token_delta_ms is not None
        and last_token_delta_ms is not None
    ):
        client_tpot_ms = (last_token_delta_ms - first_token_delta_ms) / (output_tokens - 1)
        client_tpot_e2e_ms = (elapsed_ms - first_token_delta_ms) / (output_tokens - 1)

    ttft_ms = first_token_delta_ms
    ttft_kind = first_delta_kind
    if ttft_ms is None:
        ttft_ms = message_start_ms if message_start_ms is not None else first_line_ms
        if ttft_ms is not None:
            ttft_kind = "message_start_fallback" if message_start_ms is not None else "first_line_fallback"

    prefill_queue_ms = (
        proxy_timing.get("prefill_queue_ms") if isinstance(proxy_timing, dict) else None
    )
    prefill_rpc_ms = (
        proxy_timing.get("prefill_rpc_ms") if isinstance(proxy_timing, dict) else None
    )
    if prefill_rpc_ms is None and isinstance(proxy_timing, dict):
        prefill_rpc_ms = proxy_timing.get("prefill_ms")
    row = {
        "index": sample.index,
        "label": sample.label,
        "sample_id": sample.sample_id,
        "status": status,
        "error": error,
        "input_length_bucket": sample.input_length_bucket,
        "original_glm52_input_tokens": sample.original_glm52_input_tokens,
        "original_input_length": sample.original_input_length,
        "transformed_prompt_tokens": sample.transformed_prompt_tokens,
        "transformed_text_chars": sample.transformed_text_chars,
        "transform_note": sample.transform_note,
        "prompt_sha256": sample.prompt_sha256,
        "max_tokens": MAX_TOKENS,
        "elapsed_ms": round_ms(elapsed_ms),
        "first_line_ms": round_ms(first_line_ms),
        "first_data_ms": round_ms(first_data_ms),
        "message_start_ms": round_ms(message_start_ms),
        "ttft_ms": round_ms(ttft_ms),
        "ttft_kind": ttft_kind,
        "last_token_delta_ms": round_ms(last_token_delta_ms),
        "message_stop_ms": round_ms(message_stop_ms),
        "client_tpot_ms": round_ms(client_tpot_ms),
        "client_tpot_e2e_ms": round_ms(client_tpot_e2e_ms),
        "output_tokens": output_tokens,
        "token_delta_events": token_delta_events,
        "event_count": event_count,
        "usage": usage,
        "message_start_usage": message_start_usage,
        "final_usage": final_usage,
        "message_start_cache_read_input_tokens": message_start_usage.get("cache_read_input_tokens"),
        "message_start_cache_creation_input_tokens": message_start_usage.get("cache_creation_input_tokens"),
        "final_cache_read_input_tokens": final_usage.get("cache_read_input_tokens"),
        "final_cache_creation_input_tokens": final_usage.get("cache_creation_input_tokens"),
        "output_text_sha256": hashlib.sha256("".join(output_text_parts).encode("utf-8")).hexdigest(),
        "output_text_chars": sum(len(part) for part in output_text_parts),
        "proxy_timing": proxy_timing,
        "proxy_done": proxy_done,
        "prefill": proxy_timing.get("prefill") if isinstance(proxy_timing, dict) else None,
        "decode": proxy_timing.get("decode") if isinstance(proxy_timing, dict) else None,
        "prefill_ms": proxy_timing.get("prefill_ms") if isinstance(proxy_timing, dict) else None,
        "prefill_queue_ms": prefill_queue_ms,
        "prefill_rpc_ms": prefill_rpc_ms,
        "real_prefill_ms": prefill_rpc_ms,
        "decode_first_ms": proxy_timing.get("decode_first_ms") if isinstance(proxy_timing, dict) else None,
        "ttft_proxy_ms": proxy_timing.get("ttft_proxy_ms") if isinstance(proxy_timing, dict) else None,
        "proxy_total_ms": proxy_done.get("total_ms") if isinstance(proxy_done, dict) else None,
        "request_path": str(request_path),
        "response_path": str(response_path),
    }
    row["abnormal_reasons"] = classify_abnormal(row)
    row["abnormal"] = bool(row["abnormal_reasons"])
    return row


def finite(values: list[Any]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float))]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def stats(values: list[Any]) -> dict[str, float | int | None]:
    vals = finite(values)
    return {
        "count": len(vals),
        "min": round(min(vals), 3) if vals else None,
        "max": round(max(vals), 3) if vals else None,
        "mean": round(statistics.mean(vals), 3) if vals else None,
        "median": round(statistics.median(vals), 3) if vals else None,
        "p90": percentile(vals, 0.9),
    }


def row_line(row: dict[str, Any]) -> str:
    return (
        f"result index={row['index']} sample_id={row['sample_id']} status={row['status']} "
        f"prompt_tokens={row['transformed_prompt_tokens']} output_tokens={row['output_tokens']} "
        f"ttft_ms={row['ttft_ms']} tpot_ms={row['client_tpot_ms']} "
        f"elapsed_ms={row['elapsed_ms']} prefill={row['prefill']} "
        f"prefill_ms={row['prefill_ms']} real_prefill_ms={row['real_prefill_ms']} "
        f"prefill_queue_ms={row['prefill_queue_ms']} decode_first_ms={row['decode_first_ms']} "
        f"abnormal={row['abnormal']} reasons={';'.join(row['abnormal_reasons']) or '-'} "
        f"error={row['error']}"
    )


def write_report(
    report_path: Path,
    summary: dict[str, Any],
    acceptance_path: Path,
    api_summary_path: Path,
) -> None:
    rows = sorted(summary["results"], key=lambda r: r["index"])
    lines = [
        "# CCW /v1/messages TTFT TPOT",
        "",
        f"- generated_at_utc: `{summary['generated_at']}`",
        f"- endpoint: `{summary['endpoint']}`",
        f"- model: `{summary['model']}`",
        f"- sample_file: `{summary['sample_file']}`",
        f"- sample_count: `{summary['sample_count']}`",
        f"- max_tokens: `{summary['max_tokens']}`",
        f"- concurrency: `{summary['concurrency']}`",
        f"- prompt transform: `system/tool/other roles converted to user text; adjacent same-role messages merged`",
        f"- pass: `{summary['pass']}`",
        f"- acceptance: `{acceptance_path}`",
        "",
        "## Aggregate",
        "",
        "| metric | count | min ms | median ms | mean ms | p90 ms | max ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("ttft_ms", "TTFT"),
        ("client_tpot_ms", "TPOT"),
        ("elapsed_ms", "E2E"),
        ("prefill_ms", "Prefill total"),
        ("prefill_rpc_ms", "Real Prefill"),
        ("prefill_queue_ms", "Prefill queue"),
        ("decode_first_ms", "Decode first"),
    ]:
        s = summary["stats"][key]
        lines.append(
            f"| {label} | {s['count']} | {s['min']} | {s['median']} | "
            f"{s['mean']} | {s['p90']} | {s['max']} |"
        )
    lines.extend(
        [
            "",
            "## Samples",
            "",
            "| # | sample_id | prompt tokens | out tok | TTFT ms | TPOT ms | E2E ms | prefill | real prefill ms | queue ms | decode first ms | abnormal | transform |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['index']} | `{row['sample_id']}` | {row['transformed_prompt_tokens']} | "
            f"{row['output_tokens']} | {row['ttft_ms']} | {row['client_tpot_ms']} | "
            f"{row['elapsed_ms']} | `{row['prefill']}` | {row['real_prefill_ms']} | "
            f"{row['prefill_queue_ms']} | {row['decode_first_ms']} | "
            f"`{'; '.join(row['abnormal_reasons']) or '-'}` | `{row['transform_note']}` |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| check | result |",
            "| --- | ---: |",
        ]
    )
    for name, ok in summary["checks"].items():
        lines.append(f"| `{name}` | `{ok}` |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- raw summary: `{summary['raw_summary_path']}`",
            f"- API summary: `{api_summary_path}`",
            f"- log: `{summary['log_path']}`",
            f"- proxy log: `{summary['proxy_log']}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    sample_stem = SAMPLE_FILE.stem
    run_id = os.environ.get("RUN_ID", f"messages_ccw_{sample_stem}_ttft_tpot_{utc_stamp()}")
    raw_dir = TASK_ROOT / "reports/raw" / run_id
    api_dir = TASK_ROOT / "reports/api"
    accept_dir = TASK_ROOT / "reports/acceptance"
    log_dir = TASK_ROOT / "logs/api"
    raw_dir.mkdir(parents=True, exist_ok=True)
    api_dir.mkdir(parents=True, exist_ok=True)
    accept_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"
    proxy_log = latest_proxy_log()
    samples = load_samples()
    text_chars = [sample.transformed_text_chars for sample in samples]
    if len(set(text_chars)) != len(text_chars):
        raise RuntimeError("transformed text_chars are not unique; proxy timing matching is unsafe")

    results: list[dict[str, Any]] = []
    with log_path.open("w", encoding="utf-8") as log:

        def log_print(line: str) -> None:
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        log_print(f"run_id={run_id}")
        log_print(f"endpoint={ENDPOINT}")
        log_print(f"count_endpoint={COUNT_ENDPOINT}")
        log_print(f"model={MODEL}")
        log_print(f"sample_file={SAMPLE_FILE}")
        log_print(f"sample_count={len(samples)} max_tokens={MAX_TOKENS} concurrency={CONCURRENCY}")
        log_print(f"proxy_log={proxy_log}")
        log_print(f"bucket_config={json.dumps(bucket_config(), ensure_ascii=False)}")
        for sample in samples:
            write_json(
                raw_dir / f"{sample.label}_transform_meta.json",
                {
                    "sample_id": sample.sample_id,
                    "index": sample.index,
                    "original_glm52_input_tokens": sample.original_glm52_input_tokens,
                    "transformed_prompt_tokens": sample.transformed_prompt_tokens,
                    "transformed_text_chars": sample.transformed_text_chars,
                    "transform_note": sample.transform_note,
                    "prompt_sha256": sample.prompt_sha256,
                    "messages": sample.transformed_messages,
                },
            )
            log_print(
                "prepared "
                f"index={sample.index} sample_id={sample.sample_id} "
                f"orig_tokens={sample.original_glm52_input_tokens} "
                f"transformed_tokens={sample.transformed_prompt_tokens} "
                f"text_chars={sample.transformed_text_chars} note={sample.transform_note}"
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            future_to_sample = {
                executor.submit(stream_sample, sample, raw_dir, proxy_log): sample
                for sample in samples
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_sample):
                row = future.result()
                results.append(row)
                completed += 1
                log_print(f"progress {completed}/{len(samples)} {row_line(row)}")

    rows = sorted(results, key=lambda r: r["index"])
    abnormal_rows = [row for row in rows if row.get("abnormal")]
    checks = {
        "sample_count_matches_file": len(rows) == len(samples),
        "concurrency_2": CONCURRENCY == 2,
        "max_tokens_256": MAX_TOKENS == 256,
        "all_http_200": all(row.get("status") == 200 for row in rows),
        "all_output_tokens_le_256": all(
            isinstance(row.get("output_tokens"), int) and row["output_tokens"] <= MAX_TOKENS
            for row in rows
        ),
        "all_ttft_recorded": all(isinstance(row.get("ttft_ms"), (int, float)) for row in rows),
        "all_tpot_recorded": all(
            isinstance(row.get("client_tpot_ms"), (int, float)) for row in rows
        ),
        "all_proxy_timing_recorded": all(
            isinstance(row.get("proxy_timing"), dict) for row in rows
        ),
        "all_messages_full_prefill_decode_path": all(
            isinstance(row.get("proxy_timing"), dict)
            and row["proxy_timing"].get("api") == "/messages"
            and str(row.get("prefill") or "").strip() in PREFILLS
            and str(row.get("decode") or "").strip() == DECODE
            and isinstance(row.get("prefill_rpc_ms"), (int, float))
            and isinstance(row.get("decode_first_ms"), (int, float))
            and isinstance(row.get("ttft_proxy_ms"), (int, float))
            for row in rows
        ),
        "short_le_50k_ttft_lt_10s": all(
            not isinstance(row.get("transformed_prompt_tokens"), int)
            or row["transformed_prompt_tokens"] > SHORT_PROMPT_TOKEN_MAX
            or (
                isinstance(row.get("ttft_ms"), (int, float))
                and row["ttft_ms"] < SHORT_TTFT_MAX_MS
            )
            for row in rows
        ),
        "no_abnormal_samples": len(abnormal_rows) == 0,
    }
    pass_value = all(value is True for value in checks.values())
    summary = {
        "run_id": run_id,
        "generated_at": utc_now(),
        "endpoint": ENDPOINT,
        "count_endpoint": COUNT_ENDPOINT,
        "model": MODEL,
        "sample_file": str(SAMPLE_FILE),
        "sample_count": len(rows),
        "max_tokens": MAX_TOKENS,
        "concurrency": CONCURRENCY,
        "prefills": sorted(PREFILLS),
        "decode": DECODE,
        "prefill_shape_bucket": bucket_config(),
        "stats": {
            "ttft_ms": stats([row.get("ttft_ms") for row in rows]),
            "client_tpot_ms": stats([row.get("client_tpot_ms") for row in rows]),
            "client_tpot_e2e_ms": stats([row.get("client_tpot_e2e_ms") for row in rows]),
            "elapsed_ms": stats([row.get("elapsed_ms") for row in rows]),
            "prefill_ms": stats([row.get("prefill_ms") for row in rows]),
            "prefill_rpc_ms": stats([row.get("prefill_rpc_ms") for row in rows]),
            "prefill_queue_ms": stats([row.get("prefill_queue_ms") for row in rows]),
            "decode_first_ms": stats([row.get("decode_first_ms") for row in rows]),
            "output_tokens": stats([row.get("output_tokens") for row in rows]),
        },
        "abnormal_count": len(abnormal_rows),
        "abnormal_indices": [row["index"] for row in abnormal_rows],
        "abnormal_rows": [
            {
                "index": row["index"],
                "sample_id": row["sample_id"],
                "transformed_prompt_tokens": row["transformed_prompt_tokens"],
                "ttft_ms": row["ttft_ms"],
                "prefill": row["prefill"],
                "prefill_ms": row["prefill_ms"],
                "real_prefill_ms": row["real_prefill_ms"],
                "prefill_queue_ms": row["prefill_queue_ms"],
                "decode_first_ms": row["decode_first_ms"],
                "client_tpot_ms": row["client_tpot_ms"],
                "abnormal_reasons": row["abnormal_reasons"],
            }
            for row in abnormal_rows
        ],
        "checks": checks,
        "pass": pass_value,
        "results": rows,
        "raw_dir": str(raw_dir),
        "raw_summary_path": str(raw_dir / "summary.json"),
        "log_path": str(log_path),
        "proxy_log": str(proxy_log),
        "tpot_definition": (
            "client_tpot_ms=(last streamed token delta time - first streamed token delta time)"
            "/(output_tokens-1); client_tpot_e2e_ms also includes post-last-token stream tail."
        ),
        "transform_policy": (
            "For /v1/messages compatibility, roles other than user/assistant are represented "
            "as user text with a role tag, and adjacent same-role messages are merged."
        ),
    }
    write_json(raw_dir / "summary.json", summary)
    api_summary_path = api_dir / f"{run_id}_summary.json"
    write_json(api_summary_path, summary)

    acceptance = {
        "created_at": utc_now(),
        "gate": "messages_ccw_ttft_tpot",
        "run_id": run_id,
        "pass": pass_value,
        "summary_json": str(raw_dir / "summary.json"),
        "checks": checks,
        "stats": summary["stats"],
        "abnormal_count": summary["abnormal_count"],
        "abnormal_indices": summary["abnormal_indices"],
        "abnormal_rows": summary["abnormal_rows"],
    }
    acceptance_path = accept_dir / f"{run_id}_acceptance.json"
    write_json(acceptance_path, acceptance)
    report_path = api_dir / f"{run_id}_report.md"
    write_report(report_path, summary, acceptance_path, api_summary_path)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2), flush=True)
    return 0 if pass_value else 1


if __name__ == "__main__":
    raise SystemExit(main())
