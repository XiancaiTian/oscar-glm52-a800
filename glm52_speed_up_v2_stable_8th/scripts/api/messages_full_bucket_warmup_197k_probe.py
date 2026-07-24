#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import ast
import hashlib
import json
import math
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
ENDPOINT = os.environ.get("MESSAGES_ENDPOINT", "http://127.0.0.1:19181/v1/messages")
COUNT_ENDPOINT = os.environ.get(
    "MESSAGES_COUNT_ENDPOINT", "http://127.0.0.1:19181/v1/messages/count_tokens"
)
MODEL = os.environ.get("MODEL", os.environ.get("MODEL_ID", "GLM-5.2-FP8"))
MAX_TOKENS = 1
STEP_TOKENS = int(os.environ.get("WARMUP_STEP_TOKENS", "2048"))
MAX_WARMUP_TOKENS = int(os.environ.get("MAX_WARMUP_TOKENS", "196608"))
REQUIRED_PER_PREFILL = int(os.environ.get("REQUIRED_PER_PREFILL", "1"))
MAX_ATTEMPTS_PER_LENGTH = int(os.environ.get("MAX_ATTEMPTS_PER_LENGTH", "6"))
REQUEST_TIMEOUT_S = int(os.environ.get("REQUEST_TIMEOUT_S", "2400"))
PROXY_POLL_TIMEOUT_S = int(os.environ.get("PROXY_POLL_TIMEOUT_S", "120"))
ALLOW_EMPTY_WARMUP = os.environ.get("ALLOW_EMPTY_WARMUP", "0") == "1"
WARMUP_TARGETS_ENV = os.environ.get("WARMUP_TARGETS", "").strip()
WARMUP_INCLUDE_TAIL_REP = (
    os.environ.get("WARMUP_INCLUDE_TAIL_REPRESENTATIVE", "0") == "1"
)
WARMUP_TAIL_REP_TOKENS = int(
    os.environ.get("WARMUP_TAIL_REPRESENTATIVE_TOKENS", "8")
)
FORMAL_TARGETS = tuple(
    int(x.strip())
    for x in os.environ.get("FORMAL_TARGETS", "40964,40965,40966").split(",")
    if x.strip()
)
FORMAL_TTFT_MAX_MS = float(os.environ.get("FORMAL_TTFT_MAX_MS", "10000"))
FORMAL_TTFT_SPREAD_LIMIT_MS = float(os.environ.get("FORMAL_TTFT_SPREAD_LIMIT_MS", "1000"))
FORMAL_TTFT_HARD_MAX_PROMPT_TOKENS = int(
    os.environ.get("FORMAL_TTFT_HARD_MAX_PROMPT_TOKENS", "50000")
)
PREFILLS = {
    x.strip()
    for x in os.environ.get(
        "PREFILLS", "127.0.0.1:19182"
    ).split(",")
    if x.strip()
}
DECODE = os.environ.get("DECODE", "127.0.0.1:19183")
TARGET_PREFILL = os.environ.get("TARGET_PREFILL", "").strip()
PREFILL_PROXY_INTERNAL_TOKEN = os.environ.get(
    "PREFILL_PROXY_INTERNAL_TOKEN", ""
).strip()
FILLER_CANDIDATES = [
    " a",
    " the",
    " warm",
    " token",
    " padding",
    " bucket",
    " stable",
    " quick",
]
RESUME_PROGRESS_LOG = os.environ.get("RESUME_PROGRESS_LOG", "")


@dataclass
class PreparedSample:
    label: str
    phase: str
    target_tokens: int
    content: str
    actual_tokens: int
    content_sha256: str
    filler: str
    content_chars: int
    request_meta_path: Path


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def max_label(tokens: int) -> str:
    return str(tokens).lower()


def build_warmup_targets() -> list[int]:
    if WARMUP_TARGETS_ENV:
        return [int(x.strip()) for x in WARMUP_TARGETS_ENV.split(",") if x.strip()]

    base_targets = list(range(STEP_TOKENS, MAX_WARMUP_TOKENS + 1, STEP_TOKENS))
    if not WARMUP_INCLUDE_TAIL_REP:
        return base_targets

    targets: list[int] = []
    seen: set[int] = set()
    for bucket_target in base_targets:
        if bucket_target not in seen:
            targets.append(bucket_target)
            seen.add(bucket_target)
        tail_target = bucket_target - STEP_TOKENS + WARMUP_TAIL_REP_TOKENS
        if 0 < tail_target < bucket_target and tail_target not in seen:
            targets.append(tail_target)
            seen.add(tail_target)
    return targets


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


def count_tokens(content: str) -> int:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": MAX_TOKENS,
    }
    status, body = post_json(COUNT_ENDPOINT, payload, timeout=REQUEST_TIMEOUT_S)
    if status != 200 or not isinstance(body.get("input_tokens"), int):
        raise RuntimeError(f"count_tokens failed status={status} body={body}")
    return int(body["input_tokens"])


def one_token_fillers(header: str) -> list[str]:
    base_tokens = count_tokens(header)
    valid: list[str] = []
    seen_lens: set[int] = set()
    for filler in FILLER_CANDIDATES:
        try:
            delta = count_tokens(header + filler) - base_tokens
        except Exception:
            continue
        if delta == 1 and len(filler) not in seen_lens:
            valid.append(filler)
            seen_lens.add(len(filler))
    if len(valid) < 3:
        raise RuntimeError(f"not enough one-token fillers with distinct lengths: {valid}")
    return valid


def make_exact_content(label: str, target_tokens: int, filler: str) -> tuple[str, int]:
    header = (
        f"{label}_BEGIN_UNIQUE_{utc_stamp()}\n"
        "This /v1/messages full-bucket warmup sample is intentionally unique. "
        "The label, timestamp, and filler differ so prefix-cache cannot be reused.\n"
        f"{label}_PAYLOAD:"
    )
    base_tokens = count_tokens(header)
    if base_tokens >= target_tokens:
        raise ValueError(f"header too long for {label}: {base_tokens} >= {target_tokens}")
    if count_tokens(header + filler) - base_tokens != 1:
        raise ValueError(f"filler {filler!r} is not one token for {label}")

    repeats = target_tokens - base_tokens
    content = header + filler * repeats
    actual = count_tokens(content)
    for _ in range(5):
        if actual == target_tokens:
            return content, actual
        repeats += target_tokens - actual
        if repeats < 0:
            break
        content = header + filler * repeats
        actual = count_tokens(content)
    raise ValueError(f"target mismatch for {label}: target={target_tokens} actual={actual}")


def proxy_log_size(proxy_log: Path) -> int:
    return proxy_log.stat().st_size if proxy_log.exists() else 0


def parse_proxy_timing(line: str) -> dict[str, Any] | None:
    if "PROXY_TIMING " not in line:
        return None
    pairs = dict(re.findall(r"(\w+)=([^ ]+)", line))
    if not pairs:
        return None
    out: dict[str, Any] = {"raw": line.rstrip("\n")}
    for key, value in pairs.items():
        if key in {
            "prefill_ms",
            "prefill_queue_ms",
            "prefill_rpc_ms",
            "decode_first_ms",
            "ttft_proxy_ms",
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


def poll_proxy_timing(proxy_log: Path, offset: int, text_chars: int) -> dict[str, Any] | None:
    deadline = time.time() + PROXY_POLL_TIMEOUT_S
    while time.time() < deadline:
        if not proxy_log.exists():
            time.sleep(1)
            continue
        with proxy_log.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            lines = f.readlines()
        for line in lines:
            parsed = parse_proxy_timing(line)
            if not parsed:
                continue
            if parsed.get("api") != "/messages":
                continue
            if str(parsed.get("max_tokens")) != str(MAX_TOKENS):
                continue
            if parsed.get("text_chars") != text_chars:
                continue
            return parsed
        time.sleep(1)
    return None


def prepare_sample(
    label: str,
    target_tokens: int,
    filler: str,
    phase: str,
    raw_dir: Path,
) -> PreparedSample:
    content, actual_tokens = make_exact_content(label, target_tokens, filler)
    content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    meta_path = raw_dir / f"{label}_request_meta.json"
    write_json(
        meta_path,
        {
            "label": label,
            "phase": phase,
            "target_tokens": target_tokens,
            "actual_tokens": actual_tokens,
            "content_chars": len(content),
            "content_sha256": content_sha,
            "filler": filler,
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "stream": True,
        },
    )
    return PreparedSample(
        label=label,
        phase=phase,
        target_tokens=target_tokens,
        content=content,
        actual_tokens=actual_tokens,
        content_sha256=content_sha,
        filler=filler,
        content_chars=len(content),
        request_meta_path=meta_path,
    )


def stream_prepared_sample(sample: PreparedSample, raw_dir: Path, proxy_log: Path) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": sample.content}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    if TARGET_PREFILL:
        payload["_prefill_target"] = TARGET_PREFILL
    response_path = raw_dir / f"{sample.label}_stream_events.jsonl"
    offset = proxy_log_size(proxy_log)
    headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    if PREFILL_PROXY_INTERNAL_TOKEN:
        headers["X-Prefill-Warmup-Token"] = PREFILL_PROXY_INTERNAL_TOKEN
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    status: int | None = None
    error: str | None = None
    first_line_ms: float | None = None
    message_start_ms: float | None = None
    first_delta_ms: float | None = None
    first_delta_kind: str | None = None
    event_count = 0
    usage: dict[str, Any] = {}
    message_start_usage: dict[str, Any] = {}
    final_usage: dict[str, Any] = {}

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp, response_path.open(
            "w", encoding="utf-8"
        ) as out:
            status = resp.status
            current_event = None
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
                try:
                    parsed = json.loads(raw_data)
                except Exception:
                    parsed = {"raw": raw_data}
                event_count += 1
                out.write(
                    json.dumps(
                        {"elapsed_ms": round(now_ms, 3), "event": current_event, "data": parsed},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                event_type = parsed.get("type") if isinstance(parsed, dict) else None
                if event_type == "message_start" and message_start_ms is None:
                    message_start_ms = now_ms
                if event_type == "content_block_delta":
                    delta = parsed.get("delta") if isinstance(parsed, dict) else {}
                    delta_type = delta.get("type") if isinstance(delta, dict) else None
                    if first_delta_ms is None:
                        first_delta_ms = now_ms
                        first_delta_kind = delta_type
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
                    break
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)

    elapsed_ms = (time.perf_counter() - started) * 1000
    proxy_timing = poll_proxy_timing(proxy_log, offset, sample.content_chars)
    ttft_ms = first_delta_ms
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
    return {
        "label": sample.label,
        "phase": sample.phase,
        "target_tokens": sample.target_tokens,
        "actual_count_tokens": sample.actual_tokens,
        "content_chars": sample.content_chars,
        "content_sha256": sample.content_sha256,
        "filler": sample.filler,
        "status": status,
        "error": error,
        "elapsed_ms": round(elapsed_ms, 3),
        "first_line_ms": round(first_line_ms, 3) if first_line_ms is not None else None,
        "message_start_ms": round(message_start_ms, 3) if message_start_ms is not None else None,
        "ttft_first_delta_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
        "first_delta_kind": ttft_kind,
        "raw_first_delta_ms": round(first_delta_ms, 3) if first_delta_ms is not None else None,
        "event_count": event_count,
        "usage": usage,
        "message_start_usage": message_start_usage,
        "final_usage": final_usage,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "message_start_cache_read_input_tokens": message_start_usage.get("cache_read_input_tokens"),
        "message_start_cache_creation_input_tokens": message_start_usage.get("cache_creation_input_tokens"),
        "final_cache_read_input_tokens": final_usage.get("cache_read_input_tokens"),
        "final_cache_creation_input_tokens": final_usage.get("cache_creation_input_tokens"),
        "proxy_timing": proxy_timing,
        "prefill": proxy_timing.get("prefill") if isinstance(proxy_timing, dict) else None,
        "decode": proxy_timing.get("decode") if isinstance(proxy_timing, dict) else None,
        "prefill_ms": proxy_timing.get("prefill_ms") if isinstance(proxy_timing, dict) else None,
        "prefill_queue_ms": prefill_queue_ms,
        "prefill_rpc_ms": prefill_rpc_ms,
        "real_prefill_ms": prefill_rpc_ms,
        "decode_first_ms": proxy_timing.get("decode_first_ms") if isinstance(proxy_timing, dict) else None,
        "ttft_proxy_ms": proxy_timing.get("ttft_proxy_ms") if isinstance(proxy_timing, dict) else None,
        "request_meta_path": str(sample.request_meta_path),
        "response_path": str(response_path),
    }


def finite_ttfts(rows: list[dict[str, Any]]) -> list[float]:
    return [
        float(row["ttft_first_delta_ms"])
        for row in rows
        if isinstance(row.get("ttft_first_delta_ms"), (int, float))
    ]


def canonical_bucket(tokens: int) -> int:
    return int(math.ceil(tokens / STEP_TOKENS) * STEP_TOKENS)


def load_completed_coverage(path_text: str) -> dict[int, dict[str, int]]:
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"resume progress log not found: {path}")
    completed: dict[int, dict[str, int]] = {}
    progress_pattern = re.compile(r"warmup_progress .*target=(\d+) coverage=(\{.*\}) attempts=")
    skip_pattern = re.compile(r"warmup_skip_completed .*target=(\d+) coverage=(\{.*\})")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = progress_pattern.search(line) or skip_pattern.search(line)
        if not match:
            continue
        target = int(match.group(1))
        try:
            raw_coverage = ast.literal_eval(match.group(2))
        except Exception:
            continue
        if not isinstance(raw_coverage, dict):
            continue
        coverage = {
            str(prefill): int(raw_coverage.get(prefill, 0) or 0)
            for prefill in PREFILLS
        }
        if all(count >= REQUIRED_PER_PREFILL for count in coverage.values()):
            completed[target] = coverage
    return completed


def row_line(row: dict[str, Any]) -> str:
    return (
        "result "
        f"label={row['label']} phase={row['phase']} target={row['target_tokens']} "
        f"status={row['status']} prefill={row['prefill']} decode={row['decode']} "
        f"ttft={row['ttft_first_delta_ms']} proxy_ttft={row['ttft_proxy_ms']} "
        f"prefill_ms={row['prefill_ms']} real_prefill_ms={row['real_prefill_ms']} "
        f"prefill_queue_ms={row['prefill_queue_ms']} decode_first_ms={row['decode_first_ms']} "
        f"output_tokens={row['output_tokens']} chars={row['content_chars']} "
        f"error={row['error']}"
    )


def main() -> int:
    run_id = os.environ.get(
        "RUN_ID",
        f"messages_full_bucket_warmup_{STEP_TOKENS}_{max_label(MAX_WARMUP_TOKENS)}_{utc_stamp()}",
    )
    raw_dir = TASK_ROOT / "reports/raw" / run_id
    api_dir = TASK_ROOT / "reports/api"
    log_dir = TASK_ROOT / "logs/api"
    accept_dir = TASK_ROOT / "reports/acceptance"
    for path in [raw_dir, api_dir, log_dir, accept_dir]:
        path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"
    proxy_log = latest_proxy_log()
    bucket = bucket_config()
    warmup_targets = build_warmup_targets()
    if not warmup_targets and not ALLOW_EMPTY_WARMUP:
        raise RuntimeError("warmup target list is empty")
    resumed_coverage = load_completed_coverage(RESUME_PROGRESS_LOG)

    first_header = f"{run_id}_FILLER_PROBE_PAYLOAD:"
    fillers = one_token_fillers(first_header)
    results: list[dict[str, Any]] = []
    warmup_coverage: dict[int, dict[str, int]] = {}
    filler_idx = 0

    with log_path.open("w", encoding="utf-8") as log:
        def log_print(line: str) -> None:
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        log_print(f"run_id={run_id}")
        log_print(f"endpoint={ENDPOINT}")
        log_print(f"count_endpoint={COUNT_ENDPOINT}")
        log_print(f"model={MODEL}")
        log_print(f"proxy_log={proxy_log}")
        log_print(f"prefills={sorted(PREFILLS)} decode={DECODE}")
        log_print(f"bucket_config={json.dumps(bucket, ensure_ascii=False)}")
        if warmup_targets:
            log_print(
                f"warmup_targets=count:{len(warmup_targets)} "
                f"first:{warmup_targets[0]} last:{warmup_targets[-1]} "
                f"step:{STEP_TOKENS} explicit={bool(WARMUP_TARGETS_ENV)}"
            )
        else:
            log_print("warmup_targets=count:0 explicit=false")
        log_print(f"formal_targets={FORMAL_TARGETS}")
        log_print(
            "tail_representative="
            f"enabled:{WARMUP_INCLUDE_TAIL_REP} tokens:{WARMUP_TAIL_REP_TOKENS}"
        )
        log_print(f"one_token_fillers={fillers}")
        log_print(f"resume_progress_log={RESUME_PROGRESS_LOG or '<none>'}")
        log_print(
            "resume_completed_targets="
            f"{len([t for t in warmup_targets if t in resumed_coverage])}"
        )

        for target_index, target in enumerate(warmup_targets, start=1):
            if target in resumed_coverage:
                warmup_coverage[target] = resumed_coverage[target]
                log_print(
                    "warmup_skip_completed "
                    f"index={target_index}/{len(warmup_targets)} target={target} "
                    f"coverage={resumed_coverage[target]}"
                )
                continue
            coverage = {prefill: 0 for prefill in PREFILLS}
            attempts = 0
            while not all(count >= REQUIRED_PER_PREFILL for count in coverage.values()):
                missing = [p for p, count in coverage.items() if count < REQUIRED_PER_PREFILL]
                batch_size = 2 if attempts == 0 else max(1, min(2, len(missing)))
                prepared: list[PreparedSample] = []
                for _ in range(batch_size):
                    attempts += 1
                    filler = fillers[filler_idx % len(fillers)]
                    filler_idx += 1
                    label = f"warm_{target:06d}_{attempts:02d}"
                    prepared.append(
                        prepare_sample(label, target, filler, "warmup_full_bucket", raw_dir)
                    )
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(prepared)) as executor:
                    futures = [
                        executor.submit(stream_prepared_sample, sample, raw_dir, proxy_log)
                        for sample in prepared
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        row = future.result()
                        results.append(row)
                        if str(row.get("prefill")) in coverage:
                            coverage[str(row["prefill"])] += 1
                        log_print(row_line(row))
                log_print(
                    "warmup_progress "
                    f"index={target_index}/{len(warmup_targets)} target={target} "
                    f"coverage={coverage} attempts={attempts}"
                )
                if attempts >= MAX_ATTEMPTS_PER_LENGTH:
                    log_print(
                        "warmup_target_incomplete "
                        f"target={target} coverage={coverage} attempts={attempts}"
                    )
                    break
                if not all(count >= REQUIRED_PER_PREFILL for count in coverage.values()):
                    time.sleep(2)
            warmup_coverage[target] = coverage

        log_print("phase=formal_40964_40965_40966_after_all_bucket_warmup")
        for target in FORMAL_TARGETS:
            filler = fillers[filler_idx % len(fillers)]
            filler_idx += 1
            label = f"formal_{target}"
            sample = prepare_sample(label, target, filler, "formal_post_full_bucket_warmup", raw_dir)
            row = stream_prepared_sample(sample, raw_dir, proxy_log)
            results.append(row)
            log_print(row_line(row))
            time.sleep(2)

    warmup_rows = [r for r in results if r["phase"] == "warmup_full_bucket"]
    formal_rows = [r for r in results if r["phase"] == "formal_post_full_bucket_warmup"]
    formal_hard_gate_rows = [
        r
        for r in formal_rows
        if isinstance(r.get("target_tokens"), int)
        and r["target_tokens"] <= FORMAL_TTFT_HARD_MAX_PROMPT_TOKENS
    ]
    formal_record_only_rows = [
        r for r in formal_rows if r not in formal_hard_gate_rows
    ]
    formal_ttfts = finite_ttfts(formal_rows)
    formal_spread_ms = (
        round(max(formal_ttfts) - min(formal_ttfts), 3)
        if len(formal_ttfts) == len(FORMAL_TARGETS)
        else None
    )
    formal_hard_gate_ttfts = finite_ttfts(formal_hard_gate_rows)
    formal_hard_gate_spread_ms = (
        round(max(formal_hard_gate_ttfts) - min(formal_hard_gate_ttfts), 3)
        if len(formal_hard_gate_ttfts) == len(formal_hard_gate_rows)
        and len(formal_hard_gate_ttfts) > 1
        else None
    )
    warmed_targets = {
        target
        for target, coverage in warmup_coverage.items()
        if all(count >= REQUIRED_PER_PREFILL for count in coverage.values())
    }
    formal_buckets = {target: canonical_bucket(target) for target in FORMAL_TARGETS}
    all_rows = warmup_rows + formal_rows
    checks = {
        "warmup_target_count": len(warmup_targets),
        "warmup_all_lengths_each_prefill": len(warmed_targets) == len(warmup_targets),
        "all_http_200": all(r.get("status") == 200 for r in all_rows),
        "all_target_token_counts_exact": all(
            r.get("actual_count_tokens") == r.get("target_tokens") for r in all_rows
        ),
        "all_output_tokens_1": all(r.get("output_tokens") == 1 for r in all_rows),
        "all_proxy_timing_recorded": all(isinstance(r.get("proxy_timing"), dict) for r in all_rows),
        "all_messages_full_prefill_decode_path": all(
            isinstance(r.get("proxy_timing"), dict)
            and r["proxy_timing"].get("api") == "/messages"
            and str(r.get("prefill") or "").strip() in PREFILLS
            and str(r.get("decode") or "").strip() == DECODE
            and isinstance(r.get("prefill_rpc_ms"), (int, float))
            and isinstance(r.get("decode_first_ms"), (int, float))
            and isinstance(r.get("ttft_proxy_ms"), (int, float))
            for r in all_rows
        ),
        "sample_contents_unique": len({r.get("content_sha256") for r in all_rows})
        == len(all_rows),
        "formal_targets_present": {r.get("target_tokens") for r in formal_rows}
        == set(FORMAL_TARGETS),
        "formal_exact_lengths_not_in_warmup": not any(t in warmed_targets for t in FORMAL_TARGETS),
        "formal_canonical_buckets_warmed": all(
            bucket_target in warmed_targets for bucket_target in formal_buckets.values()
        ),
        "formal_short_prompt_ttft_lt_10s": len(formal_hard_gate_ttfts)
        == len(formal_hard_gate_rows)
        and all(ttft < FORMAL_TTFT_MAX_MS for ttft in formal_hard_gate_ttfts),
        "formal_short_prompt_ttft_spread_le_1s": (
            len(formal_hard_gate_rows) <= 1
            or (
                formal_hard_gate_spread_ms is not None
                and formal_hard_gate_spread_ms <= FORMAL_TTFT_SPREAD_LIMIT_MS
            )
        ),
    }
    pass_value = all(value is True or isinstance(value, int) for value in checks.values()) and (
        checks["warmup_all_lengths_each_prefill"] is True
        and checks["all_http_200"] is True
        and checks["all_target_token_counts_exact"] is True
        and checks["all_output_tokens_1"] is True
        and checks["all_proxy_timing_recorded"] is True
        and checks["all_messages_full_prefill_decode_path"] is True
        and checks["sample_contents_unique"] is True
        and checks["formal_targets_present"] is True
        and checks["formal_exact_lengths_not_in_warmup"] is True
        and checks["formal_canonical_buckets_warmed"] is True
        and checks["formal_short_prompt_ttft_lt_10s"] is True
        and checks["formal_short_prompt_ttft_spread_le_1s"] is True
    )
    summary = {
        "run_id": run_id,
        "generated_at": utc_now(),
        "endpoint": ENDPOINT,
        "count_endpoint": COUNT_ENDPOINT,
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "prefills": sorted(PREFILLS),
        "decode": DECODE,
        "warmup": {
            "step_tokens": STEP_TOKENS,
            "max_warmup_tokens": MAX_WARMUP_TOKENS,
            "include_tail_representative": WARMUP_INCLUDE_TAIL_REP,
            "tail_representative_tokens": WARMUP_TAIL_REP_TOKENS,
            "explicit_targets": bool(WARMUP_TARGETS_ENV),
            "targets": warmup_targets,
            "target_count": len(warmup_targets),
            "required_per_prefill": REQUIRED_PER_PREFILL,
            "coverage": warmup_coverage,
            "warmed_targets": sorted(warmed_targets),
            "incomplete_targets": [
                target for target in warmup_targets if target not in warmed_targets
            ],
            "resumed_from_log": RESUME_PROGRESS_LOG or None,
            "skipped_completed_targets": sorted(
                target for target in warmup_targets if target in resumed_coverage
            ),
        },
        "formal": {
            "targets": list(FORMAL_TARGETS),
            "canonical_buckets": formal_buckets,
            "ttft_ms": {
                str(row.get("target_tokens")): row.get("ttft_first_delta_ms")
                for row in formal_rows
            },
            "proxy_ttft_ms": {
                str(row.get("target_tokens")): row.get("ttft_proxy_ms")
                for row in formal_rows
            },
            "prefill_ms": {
                str(row.get("target_tokens")): row.get("prefill_ms")
                for row in formal_rows
            },
            "prefill_rpc_ms": {
                str(row.get("target_tokens")): row.get("prefill_rpc_ms")
                for row in formal_rows
            },
            "prefill_queue_ms": {
                str(row.get("target_tokens")): row.get("prefill_queue_ms")
                for row in formal_rows
            },
            "decode_first_ms": {
                str(row.get("target_tokens")): row.get("decode_first_ms")
                for row in formal_rows
            },
            "spread_ms": formal_spread_ms,
            "hard_gate_max_prompt_tokens": FORMAL_TTFT_HARD_MAX_PROMPT_TOKENS,
            "hard_gate_targets": [
                row.get("target_tokens") for row in formal_hard_gate_rows
            ],
            "record_only_targets": [
                row.get("target_tokens") for row in formal_record_only_rows
            ],
            "hard_gate_spread_ms": formal_hard_gate_spread_ms,
            "mean_ttft_ms": (
                round(statistics.mean(formal_ttfts), 3) if formal_ttfts else None
            ),
            "mean_hard_gate_ttft_ms": (
                round(statistics.mean(formal_hard_gate_ttfts), 3)
                if formal_hard_gate_ttfts
                else None
            ),
        },
        "thresholds": {
            "formal_ttft_max_ms": FORMAL_TTFT_MAX_MS,
            "formal_ttft_spread_limit_ms": FORMAL_TTFT_SPREAD_LIMIT_MS,
            "formal_ttft_hard_max_prompt_tokens": FORMAL_TTFT_HARD_MAX_PROMPT_TOKENS,
        },
        "prefill_shape_bucket": bucket,
        "checks": checks,
        "pass": pass_value,
        "warmup_results": warmup_rows,
        "formal_results": formal_rows,
        "raw_dir": str(raw_dir),
        "log_path": str(log_path),
        "proxy_log": str(proxy_log),
    }
    write_json(raw_dir / "summary.json", summary)
    write_json(api_dir / f"{run_id}_summary.json", summary)

    acceptance = {
        "created_at": utc_now(),
        "gate": f"messages_full_bucket_warmup_{STEP_TOKENS}_to_{max_label(MAX_WARMUP_TOKENS)}",
        "run_id": run_id,
        "pass": pass_value,
        "summary_json": str(raw_dir / "summary.json"),
        "checks": checks,
        "thresholds": summary["thresholds"],
        "formal": summary["formal"],
        "warmup_target_count": len(warmup_targets),
        "incomplete_targets": summary["warmup"]["incomplete_targets"],
    }
    accept_path = accept_dir / f"{run_id}_acceptance.json"
    write_json(accept_path, acceptance)

    lines = [
        f"# Messages Full Bucket Warmup {STEP_TOKENS} To {MAX_WARMUP_TOKENS}",
        "",
        f"- generated_at_utc: `{summary['generated_at']}`",
        f"- endpoint: `{ENDPOINT}`",
        f"- model: `{MODEL}`",
        f"- max_tokens: `{MAX_TOKENS}`",
        (
            f"- warmup range: `{warmup_targets[0]}..{warmup_targets[-1]}` "
            f"step `{STEP_TOKENS}`"
            if warmup_targets
            else "- warmup range: `<empty>`"
        ),
        f"- warmup target count: `{len(warmup_targets)}`",
        f"- tail representative: `{WARMUP_INCLUDE_TAIL_REP}` tokens `{WARMUP_TAIL_REP_TOKENS}`",
        f"- required per Prefill: `{REQUIRED_PER_PREFILL}`",
        f"- formal targets: `{', '.join(str(x) for x in FORMAL_TARGETS)}`",
        f"- formal canonical buckets: `{formal_buckets}`",
        f"- pass: `{pass_value}`",
        f"- acceptance: `{accept_path}`",
        "",
        "## Formal TTFT",
        "",
        "| target | canonical bucket | prefill | decode | client TTFT ms | proxy TTFT ms | prefill ms | decode first ms | output tokens |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in formal_rows:
        target = int(row["target_tokens"])
        lines.append(
            f"| {target} | {formal_buckets[target]} | `{row['prefill']}` | `{row['decode']}` | "
            f"{row['ttft_first_delta_ms']} | {row['ttft_proxy_ms']} | "
            f"{row['prefill_ms']} | {row['decode_first_ms']} | {row['output_tokens']} |"
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
    for name, ok in checks.items():
        lines.append(f"| `{name}` | `{ok}` |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- raw summary: `{raw_dir / 'summary.json'}`",
            f"- API summary: `{api_dir / f'{run_id}_summary.json'}`",
            f"- log: `{log_path}`",
            f"- proxy log: `{proxy_log}`",
        ]
    )
    report_path = api_dir / f"{run_id}_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(acceptance, ensure_ascii=False, indent=2), flush=True)
    return 0 if pass_value else 1


if __name__ == "__main__":
    raise SystemExit(main())
