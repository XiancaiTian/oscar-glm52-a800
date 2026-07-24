#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


TASK_ROOT = Path(__file__).resolve().parents[2]
MCP_URL = os.environ.get("MCP_URL", "http://192.168.18.107:8000/mcp")
BASE_URL = os.environ.get(
    "MCP_TARGET_BASE_URL", os.environ.get("OPENAI_BASE", "http://127.0.0.1:19181/v1")
)
MODEL_ID = os.environ.get("MCP_TARGET_MODEL", os.environ.get("MODEL_ID", "GLM-5.2-FP8"))
API_KEY = os.environ.get("MCP_TARGET_API_KEY", "EMPTY")
POLL_SECONDS = float(os.environ.get("MCP_POLL_SECONDS", "10"))
TIMEOUT_SECONDS = float(os.environ.get("MCP_TIMEOUT_SECONDS", "5400"))


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_sse(text: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for block in text.strip().split("\n\n"):
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            continue
        joined = "\n".join(data_lines)
        try:
            messages.append(json.loads(joined))
        except json.JSONDecodeError:
            messages.append({"raw_data": joined})
    return messages


def recursive_find(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                return v
        for v in obj.values():
            found = recursive_find(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = recursive_find(item, key)
            if found is not None:
                return found
    return None


def flatten_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return "\n".join(flatten_text(v) for v in obj.values())
    if isinstance(obj, list):
        return "\n".join(flatten_text(v) for v in obj)
    return ""


def parse_tool_payload(result: dict[str, Any]) -> Any:
    payload = result.get("structuredContent")
    if payload is not None:
        return payload
    content = result.get("content")
    if isinstance(content, list):
        texts = [item.get("text") for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)]
        if texts:
            joined = "\n".join(texts)
            try:
                return json.loads(joined)
            except json.JSONDecodeError:
                return {"text": joined}
    return result


def value_pass(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"pass", "passed", "success", "ok", "accepted"}
    if isinstance(value, dict):
        for key in ("status", "verdict", "result"):
            if key in value:
                return value_pass(value[key])
        text = json.dumps(value, ensure_ascii=False).lower()
        return bool(re.search(r"\b(pass|passed|success|ok|accepted)\b", text))
    return False


def layer_value(payload: Any, layer: str) -> Any:
    if isinstance(payload, dict):
        for key in (layer, layer.lower(), f"{layer}_result", f"{layer.lower()}_result"):
            if key in payload:
                return payload[key]
        for key, value in payload.items():
            if isinstance(key, str) and key.lower() == layer.lower():
                return value
        for value in payload.values():
            found = layer_value(value, layer)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                name = str(item.get("layer") or item.get("name") or item.get("level") or "").upper()
                if name == layer:
                    return item
            found = layer_value(item, layer)
            if found is not None:
                return found
    return None


def layer_pass(payload: Any, layer: str) -> bool:
    value = layer_value(payload, layer)
    if value is not None:
        return value_pass(value)
    text = flatten_text(payload).lower()
    return bool(re.search(rf"\b{layer.lower()}\b[^\\n]{{0,120}}\b(pass|passed|success|accepted)\b", text))


def l2_pass_or_accepted(payload: Any) -> bool:
    if layer_pass(payload, "L2"):
        return True
    l2 = layer_value(payload, "L2")
    text = flatten_text(l2 if l2 is not None else payload)
    return bool(
        re.search(r"litellm_streaming", text, re.I)
        and re.search(r"(reasoning-only|no content_block_delta|输出长度限制|stop_reason.+max_tokens|max_tokens)", text, re.I | re.S)
        and not re.search(r"(timeout|超时|非预期)", text, re.I)
    )


class MCPClient:
    def __init__(self, url: str):
        self.url = url
        self.session_id: str | None = None
        self.next_id = 1
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    def post(self, payload: dict[str, Any], timeout: float = 60) -> list[dict[str, Any]]:
        headers = dict(self.headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        response = requests.post(self.url, headers=headers, json=payload, timeout=timeout)
        if response.headers.get("mcp-session-id"):
            self.session_id = response.headers["mcp-session-id"]
        response.raise_for_status()
        if response.status_code == 202 or not response.text.strip():
            return []
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return parse_sse(response.text)
        return [response.json()]

    def request(self, method: str, params: dict[str, Any] | None = None, timeout: float = 60) -> dict[str, Any]:
        msg_id = self.next_id
        self.next_id += 1
        messages = self.post(
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}},
            timeout=timeout,
        )
        for message in messages:
            if message.get("id") == msg_id:
                if "error" in message:
                    raise RuntimeError(json.dumps(message["error"], ensure_ascii=False))
                return message.get("result") or {}
        raise RuntimeError(f"no response for id={msg_id}: {messages!r}")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self.post({"jsonrpc": "2.0", "method": method, "params": params or {}}, timeout=60)

    def initialize(self) -> dict[str, Any]:
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "codex-direct", "version": "1.0"},
            },
        )
        self.notify("notifications/initialized")
        return result

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)


def main() -> int:
    run_id = os.environ.get("RUN_ID", f"task2_mcp_l123_direct_{utc_stamp()}")
    raw_dir = TASK_ROOT / "reports" / "raw" / run_id
    api_dir = TASK_ROOT / "reports" / "api"
    acc_dir = TASK_ROOT / "reports" / "acceptance"
    log_dir = TASK_ROOT / "logs" / "api"
    for path in (raw_dir, api_dir, acc_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{run_id}.log"
    cmd_path = log_dir / f"{run_id}.cmd"
    summary_path = api_dir / f"{run_id}_summary.json"
    acceptance_path = acc_dir / f"{run_id}_acceptance.json"
    cmd_path.write_text(
        f"RUN_ID={run_id} MCP_URL={MCP_URL} MCP_TARGET_BASE_URL={BASE_URL} {__file__}\n",
        encoding="utf-8",
    )

    events: list[dict[str, Any]] = []

    def log_event(name: str, data: Any) -> None:
        event = {"at": utc_now(), "event": name, "data": data}
        events.append(event)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)

    client = MCPClient(MCP_URL)
    initialize_result = client.initialize()
    write_json(raw_dir / "initialize.json", initialize_result)
    log_event("initialize", initialize_result)

    tools_result = client.request("tools/list")
    write_json(raw_dir / "tools_list.json", tools_result)
    tool_names = [tool.get("name") for tool in tools_result.get("tools", []) if isinstance(tool, dict)]
    log_event("tools_list", tool_names)

    validate_args = {
        "base_url": BASE_URL,
        "model_id": MODEL_ID,
        "protocol": "openai",
        "api_key": API_KEY,
        "supports_vlm": False,
        "scope": "full",
    }
    validate_result = client.call_tool("validate_model", validate_args, timeout=120)
    validate_payload = parse_tool_payload(validate_result)
    write_json(raw_dir / "validate_model_result.json", validate_result)
    log_event("validate_model", validate_payload)
    mcp_run_id = recursive_find(validate_payload, "run_id")
    if not isinstance(mcp_run_id, str):
        raise RuntimeError(f"validate_model did not return run_id: {validate_payload!r}")

    started = time.time()
    poll_index = 0
    final_payload: Any = None
    status_value = "unknown"
    while time.time() - started < TIMEOUT_SECONDS:
        poll_index += 1
        status_result = client.call_tool("get_run_status", {"run_id": mcp_run_id}, timeout=120)
        status_payload = parse_tool_payload(status_result)
        write_json(raw_dir / f"status_{poll_index:04d}.json", status_payload)
        status_value = str(recursive_find(status_payload, "status") or "").lower()
        log_event("get_run_status", {"poll": poll_index, "status": status_value, "run_id": mcp_run_id})
        if status_value in {"completed", "complete", "failed", "error", "cancelled", "canceled"}:
            final_payload = status_payload
            break
        time.sleep(POLL_SECONDS)

    if final_payload is None:
        final_payload = status_payload if "status_payload" in locals() else {}
        status_value = "timeout"

    write_json(raw_dir / "final_status.json", final_payload)
    final_text = flatten_text(final_payload)
    checks = {
        "mcp_initialize_ok": bool(initialize_result.get("serverInfo")),
        "tools_available": {"validate_model", "get_run_status"}.issubset(set(tool_names)),
        "validate_model_run_id": isinstance(mcp_run_id, str) and bool(mcp_run_id),
        "run_completed": status_value in {"completed", "complete"},
        "l1_reported_pass": layer_pass(final_payload, "L1"),
        "l2_reported_pass_or_accepted": l2_pass_or_accepted(final_payload),
        "l3_reported_pass": layer_pass(final_payload, "L3"),
        "vlm_marked_unsupported": (
            validate_args["supports_vlm"] is False
            or
            "unsupported" in final_text.lower()
            or "supports_vlm" in final_text
            or "不支持" in final_text
        ),
        "target_and_model_present": BASE_URL in final_text and MODEL_ID in final_text,
    }
    result = {
        "created_at": utc_now(),
        "run_id": run_id,
        "mcp_run_id": mcp_run_id,
        "gate": "task2_mcp_l123_direct",
        "mcp_url": MCP_URL,
        "target": BASE_URL,
        "model": MODEL_ID,
        "status": status_value,
        "poll_count": poll_index,
        "checks": checks,
        "pass": all(checks.values()),
        "raw_dir": str(raw_dir),
        "final_status": str(raw_dir / "final_status.json"),
    }
    write_json(summary_path, result)
    write_json(acceptance_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
