#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parents[1]
BASE_URL = os.environ.get("OPENAI_BASE", "http://127.0.0.1:19181/v1").rstrip("/")
MODEL = os.environ.get("MODEL", os.environ.get("MODEL_ID", "GLM-5.2-FP8"))


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def post_json(path: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body_text = resp.read().decode("utf-8", errors="replace")
            try:
                body: Any = json.loads(body_text)
            except json.JSONDecodeError:
                body = body_text
            return {
                "ok": 200 <= resp.status < 300,
                "status": resp.status,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "body": body,
            }
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            body = body_text
        return {
            "ok": False,
            "status": exc.code,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "body": body,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": repr(exc),
        }


def tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "taskId": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed"],
            },
            "note": {"type": "string"},
        },
        "required": ["taskId", "status"],
        "additionalProperties": False,
    }


def result_has_anthropic_tool_use(result: dict[str, Any]) -> bool:
    body = result.get("body")
    if not isinstance(body, dict):
        return False
    for block in body.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return block.get("name") == "TaskUpdate" and isinstance(block.get("input"), dict)
    return False


def result_has_openai_tool_call(result: dict[str, Any]) -> bool:
    body = result.get("body")
    if not isinstance(body, dict):
        return False
    choices = body.get("choices") or []
    if not choices:
        return False
    message = choices[0].get("message") or {}
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") == "TaskUpdate":
            return True
    return False


def main() -> int:
    run_id = f"tool_choice_regression_{utc_stamp()}"
    raw_dir = TASK_ROOT / "reports/raw" / run_id
    api_dir = TASK_ROOT / "reports/api"
    accept_dir = TASK_ROOT / "reports/acceptance"
    log_dir = TASK_ROOT / "logs/api"
    for path in (raw_dir, api_dir, accept_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    prompt = (
        "Call the TaskUpdate tool exactly once with taskId=\"quant-smoke-001\", "
        "status=\"in_progress\", note=\"forced named tool_choice regression\". "
        "Do not answer in prose."
    )
    schema = tool_schema()
    messages_forced_payload = {
        "model": MODEL,
        "max_tokens": 256,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "name": "TaskUpdate",
                "description": "Update the state of a task in the task manager.",
                "input_schema": schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": "TaskUpdate"},
    }
    chat_forced_payload = {
        "model": MODEL,
        "max_tokens": 256,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "TaskUpdate",
                    "description": "Update the state of a task in the task manager.",
                    "parameters": schema,
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "TaskUpdate"}},
    }
    messages_auto_payload = dict(messages_forced_payload)
    messages_auto_payload.pop("tool_choice")
    chat_auto_payload = dict(chat_forced_payload)
    chat_auto_payload.pop("tool_choice")

    cases = {
        "messages_forced": ("/messages", messages_forced_payload, result_has_anthropic_tool_use),
        "chat_forced": ("/chat/completions", chat_forced_payload, result_has_openai_tool_call),
        "messages_auto": ("/messages", messages_auto_payload, result_has_anthropic_tool_use),
        "chat_auto": ("/chat/completions", chat_auto_payload, result_has_openai_tool_call),
    }

    results: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "base_url": BASE_URL,
        "model": MODEL,
        "payloads": {
            "messages_forced": messages_forced_payload,
            "chat_forced": chat_forced_payload,
            "messages_auto": messages_auto_payload,
            "chat_auto": chat_auto_payload,
        },
        "results": {},
        "checks": {},
    }
    for name, (path, payload, parser) in cases.items():
        result = post_json(path, payload)
        results["results"][name] = result
        results["checks"][f"{name}_http_2xx"] = result.get("ok") is True
        results["checks"][f"{name}_not_500"] = result.get("status") != 500
        results["checks"][f"{name}_tool_call_present"] = parser(result)

    # The historical service regression was HTTP 500 for forced named
    # tool_choice. Auto tool calls remain a stricter parser capability check.
    required_checks = [
        "messages_forced_http_2xx",
        "messages_forced_not_500",
        "chat_forced_http_2xx",
        "chat_forced_not_500",
        "messages_auto_http_2xx",
        "messages_auto_tool_call_present",
        "chat_auto_http_2xx",
        "chat_auto_tool_call_present",
    ]
    results["required_checks"] = required_checks
    results["pass"] = all(results["checks"].get(name) is True for name in required_checks)
    results["raw_dir"] = str(raw_dir)

    raw_path = raw_dir / "tool_choice_regression_raw.json"
    summary_path = api_dir / f"{run_id}_summary.json"
    accept_path = accept_dir / f"{run_id}_acceptance.json"
    for path in (raw_path, summary_path, accept_path):
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log_path = log_dir / f"{run_id}.log"
    log_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
