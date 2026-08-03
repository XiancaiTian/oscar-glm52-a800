#!/usr/bin/env python3
"""Validate the second fail-closed canonical preflight boundary."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def gpu(name: str) -> list[list[str]]:
    return [[field.strip() for field in row] for row in csv.reader(read(name).splitlines())]


checks = []


def check(name: str, actual: object, expected: object) -> None:
    checks.append({"name": name, "status": "passed" if actual == expected else "failed", "actual": actual, "expected": expected})


first = datetime.fromisoformat(read("gpu_first_at_utc.txt").strip().replace("Z", "+00:00"))
second = datetime.fromisoformat(read("gpu_second_at_utc.txt").strip().replace("Z", "+00:00"))
check("idle_interval_seconds", (second - first).total_seconds(), 74.0)
for name in ("gpu_first.csv", "gpu_second.csv", "post_gpu.csv"):
    rows = gpu(name)
    check(f"{name}.indices", [int(row[0]) for row in rows], list(range(8)))
    check(f"{name}.memory_used_mib", [int(row[2]) for row in rows], [0] * 8)
    check(f"{name}.utilization_pct", [int(row[3]) for row in rows], [0] * 8)
for name in ("gpu_first_compute.csv", "gpu_second_compute.csv", "post_compute.csv"):
    check(f"{name}.empty", read(name), "")
check("preflight.exit", int(read("preflight.exit").strip()), 32)
check("preflight.same_missing_mountpoint_error", "mount point does not exist" in read("preflight.stderr.log"), True)
check("preflight.stdout.empty", read("preflight.stdout.log"), "")
check("host_path_is_external_symlink", "host_link=/nfs/AE/txc/oscar-glm/artifacts/phase6/" in read("symlink_diagnosis.txt"), True)
check("container_mountpoint_not_visible", "mountpoint_visible=1" in read("bind_visibility_probe.log"), True)
check("post_container.empty", read("post_container.txt"), "")
check("output_files.empty", read("output_files.txt"), "")
all_logs = read("preflight.stdout.log") + read("preflight.stderr.log")
check("model_or_accuracy_started", any(token in all_logs for token in ("Loading model", "GSM8K", "accuracy")), False)

result = {
    "format_version": 1,
    "status": "passed" if all(item["status"] == "passed" for item in checks) else "failed",
    "passed": sum(item["status"] == "passed" for item in checks),
    "total": len(checks),
    "classification": "fail_closed_clone_symlink_target_outside_container_bind",
    "gpu_loaded": False,
    "accuracy_started": False,
    "retry_change": "replace only the ignored temporary-clone symlink with a real empty directory tree",
    "checks": checks,
}
(ROOT / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

names = [
    "validate_failure.py", "validation.json", "repair_state.txt",
    "gpu_first_at_utc.txt", "gpu_first.csv", "gpu_first_compute.csv",
    "gpu_second_at_utc.txt", "gpu_second.csv", "gpu_second_compute.csv",
    "preflight.stdout.log", "preflight.stderr.log", "preflight.exit",
    "preflight_completed_at_utc.txt", "symlink_diagnosis.txt",
    "bind_visibility_probe.log", "post_at_utc.txt", "post_gpu.csv",
    "post_compute.csv", "post_container.txt", "output_files.txt",
]
with (ROOT / "evidence_manifest.sha256").open("w", encoding="utf-8") as handle:
    for name in names:
        handle.write(f"{hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}  {name}\n")

print(json.dumps({"status": result["status"], "passed": result["passed"], "total": result["total"]}, sort_keys=True))
raise SystemExit(result["status"] != "passed")
