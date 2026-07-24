#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import base64
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ROOT = Path(os.environ.get("TASK_ROOT", str(Path(__file__).resolve().parents[2])))
BASE = TASK_ROOT.parent
PROXY_ROOT = os.environ.get("PROXY_ROOT", "http://127.0.0.1:19181").rstrip("/")
ENDPOINT = os.environ.get("CHAT_COMPLETIONS_ENDPOINT", f"{PROXY_ROOT}/v1/chat/completions")
HEALTHCHECK = os.environ.get("PROXY_HEALTHCHECK", f"{PROXY_ROOT}/healthcheck")
MODEL_ID = os.environ.get("MODEL_ID", "GLM-5.2-FP8")
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("ALLOWED_HOSTS", "").split(",")
    if host.strip()
]
FORBIDDEN_HOSTS = [
    host.strip()
    for host in os.environ.get("FORBIDDEN_HOSTS_CSV", "").split(",")
    if host.strip()
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def latest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def get_sshpass() -> str:
    if os.environ.get("SSHPASS"):
        return os.environ["SSHPASS"]
    if os.environ.get("ZHANGHONG_PASS"):
        return os.environ["ZHANGHONG_PASS"]
    info = BASE / "server_info.md"
    if not info.exists():
        return ""
    for line in info.read_text(encoding="utf-8", errors="replace").splitlines():
        if "passwd:" not in line:
            continue
        match = re.search(r"passwd:\s*`?([^`\s]+)`?", line)
        if match and match.group(1) != "zh":
            return match.group(1)
    return ""


def remote_snapshot(raw_dir: Path, label: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["SSHPASS"] = get_sshpass()
    rows: dict[str, Any] = {}
    inspect_cmd = "\n".join(
        [
            "set -euo pipefail",
            "docker ps --filter name=glm52_v2_stable --format '{{.Names}}|{{.Status}}|{{.Image}}|{{.ID}}'",
            "echo '--- inspect ---'",
            "ids=\"$(docker ps --filter name=glm52_v2_stable -q | tr '\\n' ' ')\"",
            "if [[ -n \"${ids// }\" ]]; then",
            "  docker inspect --format '{{.Name}}|{{.State.StartedAt}}|{{.RestartCount}}|{{.Image}}' ${ids}",
            "fi",
        ]
    )
    inspect_b64 = base64.b64encode(inspect_cmd.encode("utf-8")).decode("ascii")
    for host in ALLOWED_HOSTS:
        proc = subprocess.run(
            [
                "sshpass",
                "-e",
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                "LogLevel=ERROR",
                f"zhanghong@{host}",
                f"bash -lc 'printf %s {inspect_b64} | base64 -d | bash'",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        output = proc.stdout + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else "")
        rows[host] = {
            "return_code": proc.returncode,
            "output_path": str(raw_dir / f"containers_{label}_{host.replace('.', '_')}.txt"),
            "inspect_lines": [
                line
                for line in proc.stdout.splitlines()
                if line.startswith("/") and "|20" in line
            ],
        }
        (raw_dir / f"containers_{label}_{host.replace('.', '_')}.txt").write_text(
            output, encoding="utf-8"
        )
    return rows


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 240) -> dict[str, Any]:
    headers = {"Authorization": "Bearer EMPTY"}
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except Exception as exc:  # noqa: BLE001
        return {
            "status": None,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "error": repr(exc),
            "text": "",
        }
    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    return {
        "status": status,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "json": parsed,
        "text": text[:6000],
    }


def has_choice(result: dict[str, Any]) -> bool:
    body = result.get("json")
    if not isinstance(body, dict):
        return False
    choices = body.get("choices")
    return isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], dict)


def case_payloads() -> list[tuple[str, dict[str, Any]]]:
    base = {
        "model": MODEL_ID,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": True, "enable_thinking": True},
    }
    return [
        (
            "json_schema",
            {
                **base,
                "messages": [{"role": "user", "content": "Return a JSON object with answer set to ok."}],
                "max_tokens": 16,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_schema",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        ),
        (
            "regex",
            {
                **base,
                "messages": [{"role": "user", "content": "Answer yes or no."}],
                "max_tokens": 4,
                "guided_regex": "^(yes|no)$",
            },
        ),
        (
            "guided_grammar",
            {
                **base,
                "messages": [{"role": "user", "content": "Output exactly ok."}],
                "max_tokens": 4,
                "guided_grammar": 'root ::= "ok"',
            },
        ),
        (
            "tool_required_with_response_format",
            {
                **base,
                "messages": [
                    {"role": "user", "content": "Use the emit_answer tool once with answer set to ok."}
                ],
                "max_tokens": 24,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "emit_answer",
                            "description": "Emit a short answer.",
                            "parameters": {
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                            },
                        },
                    }
                ],
                "tool_choice": "required",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tool_guard",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        ),
    ]


def compare_snapshots(before: dict[str, Any], after: dict[str, Any]) -> bool:
    for host in ALLOWED_HOSTS:
        left = before.get(host, {})
        right = after.get(host, {})
        if left.get("return_code") != 0 or right.get("return_code") != 0:
            return False
        if sorted(left.get("inspect_lines") or []) != sorted(right.get("inspect_lines") or []):
            return False
    return True


def main() -> int:
    stamp = ts()
    run_id = f"task2_structured_tool_grammar_{stamp}"
    patch_run_id = f"task2_patch_revalidation_{stamp}"
    raw_dir = TASK_ROOT / "reports/raw" / run_id
    api_dir = TASK_ROOT / "reports/api"
    acc_dir = TASK_ROOT / "reports/acceptance"
    log_dir = TASK_ROOT / "logs/api"
    raw_dir.mkdir(parents=True, exist_ok=True)
    api_dir.mkdir(parents=True, exist_ok=True)
    acc_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    deploy_manifest_path = latest(sorted((TASK_ROOT / "reports/raw").glob("*_deploy_manifest.json")))
    deploy_manifest = read_json(deploy_manifest_path) or {}
    messages_path = latest(
        sorted(acc_dir.glob("task2_messages_usage_patch*_*.json"))
        + sorted(acc_dir.glob("task2_messages_cache_usage_*_acceptance.json"))
    )
    messages_acc = read_json(messages_path) or {}

    before = remote_snapshot(raw_dir, "before")
    cases: list[dict[str, Any]] = []
    for name, payload in case_payloads():
        payload_path = raw_dir / f"{name}_request.json"
        response_path = raw_dir / f"{name}_response.json"
        write_json(payload_path, payload)
        result = request_json(ENDPOINT, payload, timeout=300)
        health = request_json(HEALTHCHECK, timeout=60)
        result["case"] = name
        result["payload_path"] = str(payload_path)
        result["response_path"] = str(response_path)
        result["health_after"] = {
            "status": health.get("status"),
            "text": health.get("text", ""),
        }
        write_json(response_path, result)
        cases.append(result)
    after = remote_snapshot(raw_dir, "after")

    checks: dict[str, bool] = {
        "allowed_hosts_only": not any(host in " ".join(ALLOWED_HOSTS) for host in FORBIDDEN_HOSTS),
        "deploy_manifest_prefix_cache_enabled": deploy_manifest.get("enable_prefix_caching") is True,
        "deploy_manifest_force_include_usage_enabled": deploy_manifest.get("enable_force_include_usage") is True,
        "deploy_manifest_mtp_speculative_enabled": "speculative-config" in str(deploy_manifest.get("extra_serve_args", "")),
        "messages_usage_patch_current_pass": messages_acc.get("pass") is True,
        "all_cases_no_server_crash": True,
        "no_container_restart": compare_snapshots(before, after),
    }
    for case in cases:
        name = str(case["case"])
        checks[f"{name}_http_200"] = case.get("status") == 200
        checks[f"{name}_has_choice"] = has_choice(case)
        checks[f"{name}_health_after_ok"] = case.get("health_after", {}).get("status") == 200
        if case.get("status") is None or int(case.get("status")) >= 500:
            checks["all_cases_no_server_crash"] = False

    result = {
        "run_id": run_id,
        "patch_run_id": patch_run_id,
        "generated_at": utc_now(),
        "endpoint": ENDPOINT,
        "model": MODEL_ID,
        "pass": all(checks.values()),
        "checks": checks,
        "cases": cases,
        "deploy_manifest": str(deploy_manifest_path) if deploy_manifest_path else "",
        "messages_usage_patch_acceptance": str(messages_path) if messages_path else "",
        "container_snapshot_before": before,
        "container_snapshot_after": after,
        "raw_dir": str(raw_dir),
    }
    write_json(raw_dir / "raw.json", result)
    write_json(api_dir / f"{run_id}_summary.json", result)
    write_json(acc_dir / f"{run_id}_acceptance.json", result)
    write_json(acc_dir / f"{patch_run_id}.json", result)
    (log_dir / f"{run_id}.log").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "pass": result["pass"],
                "checks": checks,
                "raw_dir": str(raw_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
