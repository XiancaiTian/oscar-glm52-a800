#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ROOT = Path(os.environ.get("TASK_ROOT", str(Path(__file__).resolve().parents[2])))
PROXY_ROOT = os.environ.get("PROXY_ROOT", "http://127.0.0.1:19181").rstrip("/")
DECODE_ROOT = os.environ.get("DECODE_ROOT", "http://127.0.0.1:19183").rstrip("/")
MODEL_ID = os.environ.get("MODEL_ID", "GLM-5.2-FP8")
ALLOWED_HOSTS = {
    host.strip()
    for host in os.environ.get("ALLOWED_HOSTS", "").split(",")
    if host.strip()
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def request(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = None
    headers = {"Authorization": "Bearer EMPTY"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = resp.status
            headers_out = dict(resp.headers.items())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        status = exc.code
        headers_out = dict(exc.headers.items()) if exc.headers else {}
    except Exception as exc:  # noqa: BLE001
        return {
            "method": method,
            "url": url,
            "status": None,
            "ok": False,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "error": repr(exc),
            "body_preview": "",
        }
    parsed = None
    try:
        parsed = json.loads(body)
    except Exception:
        parsed = None
    return {
        "method": method,
        "url": url,
        "status": status,
        "ok": 200 <= status < 300,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "headers": headers_out,
        "body_preview": body[:4000],
        "json": parsed,
    }


def model_seen(result: dict[str, Any]) -> bool:
    body = result.get("json")
    if not isinstance(body, dict):
        return False
    data = body.get("data")
    if not isinstance(data, list):
        return False
    return any(isinstance(item, dict) and item.get("id") == MODEL_ID for item in data)


def has_messages_usage(result: dict[str, Any]) -> bool:
    body = result.get("json")
    if not isinstance(body, dict):
        return False
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return False
    fields = [
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ]
    return all(isinstance(usage.get(field), int) for field in fields)


def non_empty_success_or_structured_error(result: dict[str, Any]) -> bool:
    if result.get("status") == 200:
        body = result.get("json")
        if not isinstance(body, dict):
            return False
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            text = first.get("text")
            message = first.get("message") if isinstance(first.get("message"), dict) else {}
            return bool(
                text
                or message.get("content")
                or message.get("reasoning")
                or message.get("reasoning_content")
                or message.get("tool_calls")
            )
        return bool(body.get("content") or body.get("usage"))
    body = result.get("json")
    return isinstance(body, dict) and isinstance(body.get("error"), dict)


def safe_post_payload(path: str) -> dict[str, Any]:
    if path.endswith("/v1/completions"):
        return {
            "model": MODEL_ID,
            "prompt": "Reply with OK.",
            "max_tokens": 4,
            "temperature": 0,
        }
    if path.endswith("/v1/chat/completions"):
        return {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 4,
            "temperature": 0,
            "chat_template_kwargs": {"thinking": True, "enable_thinking": True},
        }
    if path.endswith("/v1/messages"):
        return {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 8,
            "temperature": 0,
        }
    raise ValueError(path)


def enumerate_native_routes(openapi: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = openapi.get("paths") if isinstance(openapi, dict) else {}
    if not isinstance(paths, dict):
        return rows
    for path, methods in sorted(paths.items()):
        if not isinstance(path, str) or not isinstance(methods, dict):
            continue
        for method in sorted(methods):
            if method.upper() not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                continue
            rows.append({"path": path, "method": method.upper()})
    return rows


def route_exists(openapi: dict[str, Any], path: str, method: str) -> bool:
    paths = openapi.get("paths") if isinstance(openapi, dict) else {}
    methods = paths.get(path) if isinstance(paths, dict) else None
    return isinstance(methods, dict) and method.lower() in methods


def main() -> int:
    run_id = f"task2_api_route_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    raw_dir = TASK_ROOT / "reports" / "raw" / run_id
    api_dir = TASK_ROOT / "reports" / "api"
    acc_path = TASK_ROOT / "reports" / "acceptance" / f"{run_id}_acceptance.json"
    log_path = TASK_ROOT / "logs" / "api" / f"{run_id}.log"
    cmd_path = TASK_ROOT / "logs" / "api" / f"{run_id}.cmd"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_path.write_text(f"{__file__}\n", encoding="utf-8")

    proxy_gets = [
        "/v1/models",
        "/metrics",
        "/health",
        "/healthcheck",
        "/docs",
        "/openapi.json",
    ]
    decode_gets = ["/openapi.json", "/v1/models", "/metrics", "/health", "/docs"]
    post_paths = ["/v1/completions", "/v1/chat/completions", "/v1/messages"]

    raw: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": utc_now(),
        "proxy": {},
        "decode": {},
        "node_api": {},
        "native_route_matrix": [],
    }

    for path in proxy_gets:
        raw["proxy"][f"GET {path}"] = request("GET", PROXY_ROOT + path, timeout=60)
    for path in post_paths:
        raw["proxy"][f"POST {path}"] = request(
            "POST", PROXY_ROOT + path, safe_post_payload(path), timeout=180
        )
    for path in decode_gets:
        raw["decode"][f"GET {path}"] = request("GET", DECODE_ROOT + path, timeout=60)
    for path in post_paths:
        raw["decode"][f"POST {path}"] = request(
            "POST", DECODE_ROOT + path, safe_post_payload(path), timeout=180
        )

    prefill_port = os.environ.get("PREFILL_PORT", "19182")
    decode_port = os.environ.get("DECODE_PORT", "19183")
    proxy_port = os.environ.get("PROXY_PORT", "19181")
    p0_head = os.environ.get("PREFILL0_HEAD", os.environ.get("PREFILL_HEAD", "127.0.0.1"))
    p1_head = os.environ.get("PREFILL1_HEAD", p0_head)
    decode_head = os.environ.get("DECODE_HEAD", "127.0.0.1")
    nodes = {
        f"P0_{p0_head}_models": f"http://{p0_head}:{prefill_port}/v1/models",
        f"P1_{p1_head}_models": f"http://{p1_head}:{prefill_port}/v1/models",
        f"D0_{decode_head}_models": f"http://{decode_head}:{decode_port}/v1/models",
        f"Proxy_{decode_head}_healthcheck": f"http://{decode_head}:{proxy_port}/healthcheck",
    }
    for name, url in nodes.items():
        raw["node_api"][name] = request("GET", url, timeout=60)

    native_openapi = raw["decode"]["GET /openapi.json"].get("json")
    proxy_openapi = raw["proxy"]["GET /openapi.json"].get("json")
    native_routes = enumerate_native_routes(native_openapi if isinstance(native_openapi, dict) else {})
    for row in native_routes:
        path = row["path"]
        method = row["method"]
        exists = route_exists(proxy_openapi if isinstance(proxy_openapi, dict) else {}, path, method)
        executed = False
        status = None
        ok = exists
        if method == "GET" and path not in {"/shutdown", "/reset"}:
            executed = True
            checked = request("GET", PROXY_ROOT + path, timeout=30)
            status = checked.get("status")
            ok = exists and bool(checked.get("status") and int(checked["status"]) < 500)
        raw["native_route_matrix"].append(
            {
                "method": method,
                "path": path,
                "proxy_exposes_route": exists,
                "executed_on_proxy": executed,
                "proxy_status": status,
                "pass": ok,
            }
        )

    checks = {
        "proxy_models_200_and_model_id": raw["proxy"]["GET /v1/models"].get("status") == 200
        and model_seen(raw["proxy"]["GET /v1/models"]),
        "proxy_metrics_200_prometheus": raw["proxy"]["GET /metrics"].get("status") == 200
        and "# HELP" in str(raw["proxy"]["GET /metrics"].get("body_preview", "")),
        "proxy_health_ok": raw["proxy"]["GET /health"].get("status") == 200,
        "proxy_healthcheck_200": raw["proxy"]["GET /healthcheck"].get("status") == 200,
        "proxy_docs_200": raw["proxy"]["GET /docs"].get("status") == 200,
        "proxy_openapi_200": raw["proxy"]["GET /openapi.json"].get("status") == 200,
        "proxy_completions_no_unstructured_500": raw["proxy"]["POST /v1/completions"].get("status") != 500
        and non_empty_success_or_structured_error(raw["proxy"]["POST /v1/completions"]),
        "proxy_chat_no_empty_success": raw["proxy"]["POST /v1/chat/completions"].get("status") != 500
        and non_empty_success_or_structured_error(raw["proxy"]["POST /v1/chat/completions"]),
        "proxy_messages_usage_four_fields": raw["proxy"]["POST /v1/messages"].get("status") == 200
        and has_messages_usage(raw["proxy"]["POST /v1/messages"]),
        "decode_openapi_200": raw["decode"]["GET /openapi.json"].get("status") == 200,
        "decode_models_200": raw["decode"]["GET /v1/models"].get("status") == 200,
        "decode_metrics_200": raw["decode"]["GET /metrics"].get("status") == 200,
        "decode_health_200": raw["decode"]["GET /health"].get("status") == 200,
        "decode_docs_200": raw["decode"]["GET /docs"].get("status") == 200,
        "decode_completions_no_unstructured_500": raw["decode"]["POST /v1/completions"].get("status") != 500
        and non_empty_success_or_structured_error(raw["decode"]["POST /v1/completions"]),
        "decode_chat_no_empty_success": raw["decode"]["POST /v1/chat/completions"].get("status") != 500
        and non_empty_success_or_structured_error(raw["decode"]["POST /v1/chat/completions"]),
        "decode_messages_usage_four_fields": raw["decode"]["POST /v1/messages"].get("status") == 200
        and has_messages_usage(raw["decode"]["POST /v1/messages"]),
        "node_p0_models_200": raw["node_api"]["P0_63_models"].get("status") == 200,
        "node_p1_models_200": raw["node_api"]["P1_68_models"].get("status") == 200,
        "node_d0_models_200": raw["node_api"]["D0_76_models"].get("status") == 200,
        "node_proxy_healthcheck_200": raw["node_api"]["Proxy_76_healthcheck"].get("status") == 200,
        "native_routes_all_exposed": all(row["proxy_exposes_route"] for row in raw["native_route_matrix"]),
        "native_get_routes_no_5xx": all(row["pass"] for row in raw["native_route_matrix"] if row["executed_on_proxy"]),
    }
    acceptance = {
        "run_id": run_id,
        "generated_at": utc_now(),
        "pass": all(checks.values()),
        "checks": checks,
        "raw": str(raw_dir / "raw.json"),
        "summary": str(api_dir / f"{run_id}_summary.json"),
        "log": str(log_path),
        "route_count": len(raw["native_route_matrix"]),
    }
    summary = {
        "run_id": run_id,
        "generated_at": utc_now(),
        "checks": checks,
        "pass": acceptance["pass"],
        "proxy_statuses": {k: v.get("status") for k, v in raw["proxy"].items()},
        "decode_statuses": {k: v.get("status") for k, v in raw["decode"].items()},
        "node_statuses": {k: v.get("status") for k, v in raw["node_api"].items()},
        "native_route_matrix": raw["native_route_matrix"],
    }
    write_json(raw_dir / "raw.json", raw)
    write_json(api_dir / f"{run_id}_summary.json", summary)
    write_json(acc_path, acceptance)
    log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    return 0 if acceptance["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
