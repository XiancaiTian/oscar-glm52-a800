#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ROOT = Path(
    os.environ.get(
        "TASK_ROOT", str(Path(__file__).resolve().parents[2])
    )
)
SCHEME = os.environ.get(
    "SCHEME", "stage62_glm52_opt_indexshare_rdma_mtp_skipshare_gmem092_2p1d_v2_stable"
)
LOG_ROOT = TASK_ROOT / "logs" / "deploy" / SCHEME
PROXY_ROOT = os.environ.get("PROXY_ROOT", "http://127.0.0.1:19181").rstrip("/")
BASE_URL = f"{PROXY_ROOT}/v1"
MESSAGES_ENDPOINT = os.environ.get("MESSAGES_ENDPOINT", f"{BASE_URL}/messages")
COUNT_ENDPOINT = os.environ.get(
    "MESSAGES_COUNT_ENDPOINT", f"{BASE_URL}/messages/count_tokens"
)
MODEL = os.environ.get("MODEL", os.environ.get("MODEL_ID", "GLM-5.2-FP8"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "202752"))
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "198000"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32000"))
OVERFLOW_TOKENS = int(os.environ.get("OVERFLOW_TOKENS", "512"))
REQUEST_TIMEOUT_S = int(os.environ.get("REQUEST_TIMEOUT_S", "1800"))
COUNT_TIMEOUT_S = int(os.environ.get("COUNT_TIMEOUT_S", "900"))
WAIT_RELEASE_S = int(os.environ.get("WAIT_RELEASE_S", "540"))
WAIT_NO_PREFILL_S = int(os.environ.get("WAIT_NO_PREFILL_S", "60"))
POLL_S = int(os.environ.get("POLL_S", "15"))
MODE = os.environ.get("MODE", "baseline").strip().lower()
RUN_ID = os.environ.get(
    "RUN_ID", f"prefill_overlimit_{MODE}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
)
FILLER_UNIT = os.environ.get(
    "FILLER_UNIT",
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.\n",
)
MAX_RESPONSE_BYTES = int(os.environ.get("MAX_RESPONSE_BYTES", "2097152"))

RAW_DIR = TASK_ROOT / "reports" / "raw" / RUN_ID
ACCEPTANCE = TASK_ROOT / "reports" / "acceptance" / f"{RUN_ID}_acceptance.json"
API_DIR = TASK_ROOT / "reports" / "api"
LOG_PATH = TASK_ROOT / "logs" / "api" / f"{RUN_ID}.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{utc_now()} {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def post_json(url: str, payload: dict[str, Any], timeout: int) -> tuple[int, Any, str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = {"raw": text}
            return resp.status, parsed, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"raw": text}
        return exc.code, parsed, text


def count_tokens(content: str, count_index: int) -> int:
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": content}],
    }
    status, body, text = post_json(COUNT_ENDPOINT, payload, COUNT_TIMEOUT_S)
    write_json(
        RAW_DIR / "count_tokens" / f"{count_index:03d}.json",
        {
            "status": status,
            "body": body,
            "text_chars": len(content),
            "payload_sha256": hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
    )
    if status != 200:
        match = re.search(r"value=(\d+)", text) or re.search(
            r"at least (\d+) input tokens", text
        )
        if match:
            return int(match.group(1))
        raise RuntimeError(
            f"count_tokens failed status={status} body={body} text={text[:512]!r}"
        )
    if not isinstance(body, dict) or not isinstance(body.get("input_tokens"), int):
        raise RuntimeError(
            f"count_tokens failed status={status} body={body} text={text[:512]!r}"
        )
    return int(body["input_tokens"])


def build_prompt(target_input_tokens: int) -> tuple[str, int, list[dict[str, int]]]:
    probes: list[dict[str, int]] = []
    count_index = 0
    lo_units = 0
    hi_units = 1
    while True:
        count_index += 1
        tokens = count_tokens(FILLER_UNIT * hi_units, count_index)
        probes.append({"units": hi_units, "input_tokens": tokens})
        log(f"count_probe units={hi_units} input_tokens={tokens}")
        if tokens >= target_input_tokens:
            break
        lo_units = hi_units
        hi_units *= 2
        if hi_units > 1_000_000:
            raise RuntimeError("prompt search exceeded 1,000,000 filler units")

    best_units = hi_units
    best_tokens = tokens
    while lo_units + 1 < hi_units:
        mid_units = (lo_units + hi_units) // 2
        count_index += 1
        tokens = count_tokens(FILLER_UNIT * mid_units, count_index)
        probes.append({"units": mid_units, "input_tokens": tokens})
        log(f"count_probe units={mid_units} input_tokens={tokens}")
        if tokens >= target_input_tokens:
            hi_units = mid_units
            best_units = mid_units
            best_tokens = tokens
        else:
            lo_units = mid_units

    return FILLER_UNIT * best_units, best_tokens, probes


def latest_logs() -> list[Path]:
    if not LOG_ROOT.exists():
        return []
    patterns = [
        "*proxy*.log",
        "*decode*.log",
        "*prefill_p0_rank0.log",
        "*prefill_p0_rank1.log",
        "*prefill_p1_rank0.log",
        "*prefill_p1_rank1.log",
    ]
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(LOG_ROOT.glob(pattern))
    return sorted(paths, key=lambda p: (p.stat().st_mtime, str(p)))


def log_offsets(paths: list[Path]) -> dict[str, int]:
    return {str(path): path.stat().st_size for path in paths if path.exists()}


def read_since_offsets(offsets: dict[str, int]) -> dict[str, str]:
    paths = {Path(path) for path in offsets}
    paths.update(latest_logs())
    out: dict[str, str] = {}
    for path in sorted(paths, key=str):
        if not path.exists():
            continue
        offset = offsets.get(str(path), 0)
        try:
            with path.open("rb") as f:
                f.seek(offset)
                data = f.read()
        except Exception as exc:
            out[str(path)] = f"<read_error {type(exc).__name__}: {exc}>"
            continue
        if data:
            out[str(path)] = data.decode("utf-8", errors="replace")
    return out


def flatten_log_text(excerpts: dict[str, str]) -> str:
    return "\n".join(
        f"===== {path} =====\n{text}" for path, text in sorted(excerpts.items())
    )


def extract_request_ids(text: str, response_body: Any) -> list[str]:
    ids = re.findall(r"request_id=([0-9a-fA-F-]{16,})", text)
    if isinstance(response_body, dict):
        details = response_body.get("details")
        if isinstance(details, dict):
            request_id = details.get("request_id")
            if isinstance(request_id, str):
                ids.append(request_id)
    seen: set[str] = set()
    ordered = []
    for request_id in ids:
        if request_id not in seen:
            seen.add(request_id)
            ordered.append(request_id)
    return ordered


def post_messages_stream(payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        MESSAGES_ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    started = time.monotonic()
    chunks: list[bytes] = []
    result: dict[str, Any] = {
        "started_at": utc_now(),
        "status": None,
        "headers": {},
        "exception": None,
        "elapsed_s": None,
        "body_truncated": False,
    }
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            result["status"] = resp.status
            result["headers"] = dict(resp.headers.items())
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                if sum(len(x) for x in chunks) + len(chunk) <= MAX_RESPONSE_BYTES:
                    chunks.append(chunk)
                else:
                    result["body_truncated"] = True
    except urllib.error.HTTPError as exc:
        result["status"] = exc.code
        result["headers"] = dict(exc.headers.items())
        body = exc.read(MAX_RESPONSE_BYTES + 1)
        result["body_truncated"] = len(body) > MAX_RESPONSE_BYTES
        chunks.append(body[:MAX_RESPONSE_BYTES])
    except (urllib.error.URLError, socket.timeout, TimeoutError, Exception) as exc:
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    body_bytes = b"".join(chunks)
    body_text = body_bytes.decode("utf-8", errors="replace")
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    result["body_text"] = body_text
    try:
        result["body_json"] = json.loads(body_text)
    except Exception:
        result["body_json"] = None
    return result


def wait_and_collect(
    before_offsets: dict[str, int],
    response_body: Any,
) -> tuple[dict[str, str], dict[str, Any]]:
    wait_s = WAIT_RELEASE_S if MODE == "baseline" else WAIT_NO_PREFILL_S
    deadline = time.monotonic() + wait_s
    observed: dict[str, Any] = {
        "wait_s": wait_s,
        "release_seen": False,
        "context_reject_seen": False,
        "stream_error_seen": False,
        "decode_length_error_seen": False,
        "request_ids": [],
    }
    last_excerpts: dict[str, str] = {}
    while True:
        excerpts = read_since_offsets(before_offsets)
        text = flatten_log_text(excerpts)
        observed["request_ids"] = extract_request_ids(text, response_body)
        observed["release_seen"] = "retrieved by 0 decode worker" in text
        observed["context_reject_seen"] = "PROXY_CONTEXT_LENGTH_REJECT" in text
        observed["stream_error_seen"] = "PROXY_STREAM_ERROR" in text
        observed["decode_length_error_seen"] = (
            "maximum context length" in text
            or "max context length" in text
            or "VLLMValidationError" in text
        )
        last_excerpts = excerpts
        if MODE == "baseline" and observed["release_seen"]:
            break
        if MODE != "baseline" and time.monotonic() >= deadline:
            break
        if MODE == "baseline" and time.monotonic() >= deadline:
            break
        time.sleep(POLL_S)
    return last_excerpts, observed


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    API_DIR.mkdir(parents=True, exist_ok=True)
    target_input_tokens = MAX_MODEL_LEN - MAX_TOKENS + OVERFLOW_TOKENS
    if target_input_tokens <= 0:
        raise RuntimeError("target_input_tokens must be positive")
    if target_input_tokens >= MAX_INPUT_TOKENS:
        raise RuntimeError(
            "target_input_tokens must stay below MAX_INPUT_TOKENS for this gate: "
            f"target_input_tokens={target_input_tokens} "
            f"max_input_tokens={MAX_INPUT_TOKENS}"
        )
    if target_input_tokens + 1 > MAX_MODEL_LEN:
        raise RuntimeError(
            "target input would make the Prefill-side max_tokens=1 request invalid"
        )
    log(
        "run_start "
        f"mode={MODE} run_id={RUN_ID} target_input_tokens={target_input_tokens} "
        f"max_tokens={MAX_TOKENS} max_model_len={MAX_MODEL_LEN}"
    )

    content, input_tokens, probes = build_prompt(target_input_tokens)
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "messages": [{"role": "user", "content": content}],
        "stream_options": {"include_usage": True},
    }
    request_meta = {
        "created_at": utc_now(),
        "mode": MODE,
        "endpoint": MESSAGES_ENDPOINT,
        "count_endpoint": COUNT_ENDPOINT,
        "model": MODEL,
        "input_tokens": input_tokens,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_tokens": MAX_TOKENS,
        "total_tokens": input_tokens + MAX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "overflow_tokens": input_tokens + MAX_TOKENS - MAX_MODEL_LEN,
        "text_chars": len(content),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "count_probes": probes,
    }
    write_json(RAW_DIR / "request_meta.json", request_meta)
    if input_tokens >= MAX_INPUT_TOKENS:
        raise RuntimeError(
            "calibrated input_tokens must stay below MAX_INPUT_TOKENS: "
            f"input_tokens={input_tokens} max_input_tokens={MAX_INPUT_TOKENS}"
        )
    write_json(
        RAW_DIR / "request_redacted.json",
        {
            **payload,
            "messages": [
                {
                    "role": "user",
                    "content": f"<redacted {len(content)} chars sha256={request_meta['content_sha256']}>",
                }
            ],
        },
    )
    before = log_offsets(latest_logs())
    log(
        "request_send "
        f"input_tokens={input_tokens} max_tokens={MAX_TOKENS} "
        f"total={input_tokens + MAX_TOKENS}"
    )
    response = post_messages_stream(payload)
    write_json(RAW_DIR / "response.json", response)
    log(
        "request_done "
        f"status={response.get('status')} exception={response.get('exception')} "
        f"elapsed_s={response.get('elapsed_s')}"
    )

    response_body = response.get("body_json") or {}
    excerpts, observed = wait_and_collect(before, response_body)
    log_text = flatten_log_text(excerpts)
    (RAW_DIR / "log_excerpt.txt").write_text(log_text, encoding="utf-8")
    write_json(RAW_DIR / "observed.json", observed)

    body_json = response.get("body_json")
    error_code = None
    if isinstance(body_json, dict):
        error = body_json.get("error")
        if isinstance(error, dict):
            error_code = error.get("code")
    baseline_pass = (
        MODE == "baseline"
        and request_meta["input_tokens"] < MAX_INPUT_TOKENS
        and request_meta["total_tokens"] > MAX_MODEL_LEN
        and not (response.get("status") == 400 and error_code == "context_length_exceeded")
        and observed["stream_error_seen"]
        and observed["release_seen"]
    )
    fixed_pass = (
        MODE != "baseline"
        and request_meta["input_tokens"] < MAX_INPUT_TOKENS
        and request_meta["total_tokens"] > MAX_MODEL_LEN
        and response.get("status") == 400
        and error_code == "context_length_exceeded"
        and observed["context_reject_seen"]
        and not observed["stream_error_seen"]
        and not observed["release_seen"]
    )
    acceptance = {
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "gate": "prefill_overlimit_root_cause"
        if MODE == "baseline"
        else "prefill_overlimit_fix",
        "mode": MODE,
        "pass": bool(baseline_pass or fixed_pass),
        "request": request_meta,
        "response_status": response.get("status"),
        "response_exception": response.get("exception"),
        "response_error_code": error_code,
        "observed": observed,
        "raw_dir": str(RAW_DIR),
        "log": str(LOG_PATH),
        "acceptance": str(ACCEPTANCE),
    }
    write_json(ACCEPTANCE, acceptance)
    write_json(API_DIR / f"{RUN_ID}_summary.json", acceptance)
    log(f"acceptance pass={acceptance['pass']} path={ACCEPTANCE}")
    return 0 if acceptance["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
