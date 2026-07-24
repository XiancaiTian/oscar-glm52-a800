#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parents[2]
PROXY_ROOT = os.environ.get("PROXY_ROOT", "http://127.0.0.1:19181").rstrip("/")
RUN_ID = os.environ.get(
    "RUN_ID", f"task2_metric_ttft_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
)
METRICS_URL = os.environ.get("METRICS_URL", f"{PROXY_ROOT}/metrics")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric_value(text: str, name: str, suffix: str = "") -> float | None:
    full = re.escape(name + suffix)
    pattern = re.compile(rf"^{full}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$", re.M)
    match = pattern.search(text)
    if not match:
        return None
    return float(match.group(1))


def main() -> int:
    raw_dir = TASK_ROOT / "reports" / "raw" / RUN_ID
    api_dir = TASK_ROOT / "reports" / "api"
    acc_dir = TASK_ROOT / "reports" / "acceptance"
    log_dir = TASK_ROOT / "logs" / "api"
    for path in (raw_dir, api_dir, acc_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(METRICS_URL, headers={"Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        status = resp.status
        text = resp.read().decode("utf-8", errors="replace")

    (raw_dir / "metrics.prom").write_text(text, encoding="utf-8")
    ttft_name = "vllm:time_to_first_token_seconds"
    prefix_queries = metric_value(text, "vllm:prefix_cache_queries_total")
    prefix_hits = metric_value(text, "vllm:prefix_cache_hits_total")
    external_queries = metric_value(text, "vllm:external_prefix_cache_queries_total")
    external_hits = metric_value(text, "vllm:external_prefix_cache_hits_total")
    ttft_count = metric_value(text, ttft_name, "_count")
    ttft_sum = metric_value(text, ttft_name, "_sum")
    checks = {
        "metrics_http_200": status == 200,
        "ttft_metric_present": f"# HELP {ttft_name}" in text and f"# TYPE {ttft_name}" in text,
        "ttft_count_positive": isinstance(ttft_count, float) and ttft_count > 0,
        "ttft_sum_positive": isinstance(ttft_sum, float) and ttft_sum > 0,
        "prefix_cache_metrics_present": prefix_queries is not None and external_queries is not None,
    }
    result = {
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "gate": "task2_metric_ttft",
        "endpoint": METRICS_URL,
        "status": status,
        "ttft_metric_name": ttft_name,
        "ttft_count": ttft_count,
        "ttft_sum": ttft_sum,
        "prefix_cache_queries_total": prefix_queries,
        "prefix_cache_hits_total": prefix_hits,
        "external_prefix_cache_queries_total": external_queries,
        "external_prefix_cache_hits_total": external_hits,
        "checks": checks,
        "pass": all(checks.values()),
        "raw_metrics": str(raw_dir / "metrics.prom"),
    }
    write_json(api_dir / f"{RUN_ID}_summary.json", result)
    write_json(acc_dir / f"{RUN_ID}_acceptance.json", result)
    (log_dir / f"{RUN_ID}.log").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
