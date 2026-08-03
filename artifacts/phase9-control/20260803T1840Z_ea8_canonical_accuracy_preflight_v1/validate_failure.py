#!/usr/bin/env python3
"""Validate the fail-closed canonical preflight boundary."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def gpu_rows(name: str) -> list[list[str]]:
    return [[value.strip() for value in row] for row in csv.reader(text(name).splitlines())]


checks = []


def check(name: str, actual: object, expected: object) -> None:
    checks.append({"name": name, "status": "passed" if actual == expected else "failed", "actual": actual, "expected": expected})


first = datetime.fromisoformat(text("gpu_first_at_utc.txt").strip().replace("Z", "+00:00"))
second = datetime.fromisoformat(text("gpu_second_at_utc.txt").strip().replace("Z", "+00:00"))
check("idle_interval_seconds", (second - first).total_seconds(), 82.0)
for name in ("gpu_first.csv", "gpu_second.csv", "post_gpu.csv"):
    rows = gpu_rows(name)
    check(f"{name}.indices", [int(row[0]) for row in rows], list(range(8)))
    check(f"{name}.memory_used_mib", [int(row[2]) for row in rows], [0] * 8)
    check(f"{name}.utilization_pct", [int(row[3]) for row in rows], [0] * 8)
for name in ("gpu_first_compute.csv", "gpu_second_compute.csv", "post_compute.csv"):
    check(f"{name}.empty", text(name), "")

static_identity = json.loads(text("static_identity.json"))
check("static_identity.status", static_identity["status"], "passed")
check("static_identity.passed", static_identity["passed"], 8)
check("preflight.exit", int(text("preflight.exit").strip()), 32)
expected_error = (
    "mount: /dev/shm/oscar-glm-ea8-fast256-launch-20260803T1155Z/artifacts/phase6/"
    "20260803T1035Z_candidate_ea8ae6b77_splitk_stride_fix_v2/overlay_rootfs/opt/"
    "vllm_glm52_v1: mount point does not exist.\n"
)
check("preflight.stderr", text("preflight.stderr.log"), expected_error)
check("preflight.stdout.empty", text("preflight.stdout.log"), "")
check("post_container.empty", text("post_container.txt"), "")
check("output_files.empty", text("output_files.txt"), "")
check("model_load_markers", any(token in (text("preflight.stdout.log") + text("preflight.stderr.log")) for token in ("Loading model", "GSM8K", "accuracy")), False)

result = {
    "format_version": 1,
    "status": "passed" if all(item["status"] == "passed" for item in checks) else "failed",
    "passed": sum(item["status"] == "passed" for item in checks),
    "total": len(checks),
    "classification": "fail_closed_missing_ignored_overlay_mountpoint",
    "gpu_loaded": False,
    "accuracy_started": False,
    "retry_change": "materialize only the missing ignored overlay mountpoint in the clean clone",
    "checks": checks,
}
(ROOT / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

manifest_names = [
    "validate_failure.py",
    "validation.json",
    "static_identity.json",
    "gpu_first_at_utc.txt",
    "gpu_first.csv",
    "gpu_first_compute.csv",
    "gpu_second_at_utc.txt",
    "gpu_second.csv",
    "gpu_second_compute.csv",
    "preflight.stdout.log",
    "preflight.stderr.log",
    "preflight.exit",
    "preflight_completed_at_utc.txt",
    "post_at_utc.txt",
    "post_gpu.csv",
    "post_compute.csv",
    "post_container.txt",
    "output_files.txt",
]
with (ROOT / "evidence_manifest.sha256").open("w", encoding="utf-8") as handle:
    for name in manifest_names:
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        handle.write(f"{digest}  {name}\n")

print(json.dumps({"status": result["status"], "passed": result["passed"], "total": result["total"]}, sort_keys=True))
raise SystemExit(result["status"] != "passed")
