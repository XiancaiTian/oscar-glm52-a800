#!/usr/bin/env python3
"""Validate the executable-mode TDD red boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


checks = []


def check(name: str, actual: object, expected: object) -> None:
    checks.append({"name": name, "status": "passed" if actual == expected else "failed", "actual": actual, "expected": expected})


check("host_python38.exit", int(read("red.exit").strip()), 1)
check("host_python38.invalid_import_boundary", "cannot import name 'UTC'" in read("red.stderr.log"), True)
check("frozen_python312.exit", int(read("red_v2.exit").strip()), 1)
check("frozen_python312.invalid_import_boundary", "No module named 'requests'" in read("red_v2.stderr.log"), True)
check("control_image_target.exit", int(read("red_v3.exit").strip()), 1)
valid_red = read("red_v3.stderr.log")
check("control_image_target.assertion", "AssertionError: '100644' != '100755'" in valid_red, True)
check("control_image_target.single_failure", "Ran 1 test" in valid_red and "FAILED (failures=1)" in valid_red, True)
check("control_image_target.no_import_error", "ImportError" in valid_red or "ModuleNotFoundError" in valid_red, False)
check("git_mode", read("red_git_mode.txt").split()[0], "100644")
check("filesystem_mode", "filesystem_mode=755" in read("red_filesystem_mode.txt"), True)

test_path = PROJECT / "scripts/phase9/test_phase9_tools.py"
result = {
    "format_version": 1,
    "status": "passed" if all(item["status"] == "passed" for item in checks) else "failed",
    "passed": sum(item["status"] == "passed" for item in checks),
    "total": len(checks),
    "classification": "valid_tdd_red_candidate_wrapper_git_mode",
    "production_changed": False,
    "target_test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
    "checks": checks,
}
(ROOT / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

names = [
    "validate_red.py", "validation.json", "red.stdout.log", "red.stderr.log",
    "red.exit", "red_git_mode.txt", "red_filesystem_mode.txt", "red_completed_at_utc.txt",
    "python312.version.txt", "red_v2.stdout.log", "red_v2.stderr.log", "red_v2.exit",
    "red_v2_completed_at_utc.txt", "red_v3.stdout.log", "red_v3.stderr.log", "red_v3.exit",
    "red_v3_completed_at_utc.txt",
]
with (ROOT / "evidence_manifest.sha256").open("w", encoding="utf-8") as handle:
    for name in names:
        handle.write(f"{hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}  {name}\n")

print(json.dumps({"status": result["status"], "passed": result["passed"], "total": result["total"]}, sort_keys=True))
raise SystemExit(result["status"] != "passed")
