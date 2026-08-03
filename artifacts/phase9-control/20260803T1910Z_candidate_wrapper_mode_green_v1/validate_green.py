#!/usr/bin/env python3
"""Validate the candidate wrapper executable-mode green boundary."""

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


before = read("mode_before.txt").split()
after = read("mode_after.txt").split()
check("mode_before", before[0], "100644")
check("mode_after", after[0], "100755")
check("blob_unchanged", before[1], after[1])
check("text_sha_unchanged", read("text_sha_before.txt").split()[0], read("text_sha_after.txt").split()[0])
check("staged_summary", read("staged_summary.txt"), " mode change 100644 => 100755 scripts/phase9/run_candidate_tp8.sh\n")
check("staged_numstat", read("staged_numstat.txt"), "0\t0\tscripts/phase9/run_candidate_tp8.sh\n")
check("target.exit", int(read("target.exit").strip()), 0)
check("target.result", "Ran 1 test" in read("target.stderr.log") and read("target.stderr.log").rstrip().endswith("OK"), True)
check("full.exit", int(read("full.exit").strip()), 0)
check("full.result", "Ran 25 tests" in read("full.stderr.log") and read("full.stderr.log").rstrip().endswith("OK"), True)
check("gpu_markers_absent", "CUDA" in (read("target.stderr.log") + read("full.stderr.log")), False)

wrapper = PROJECT / "scripts/phase9/run_candidate_tp8.sh"
test_file = PROJECT / "scripts/phase9/test_phase9_tools.py"
result = {
    "format_version": 1,
    "status": "passed" if all(item["status"] == "passed" for item in checks) else "failed",
    "passed": sum(item["status"] == "passed" for item in checks),
    "total": len(checks),
    "classification": "candidate_wrapper_git_mode_green",
    "gpu_used": False,
    "wrapper_text_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
    "target_test_sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(),
    "checks": checks,
}
(ROOT / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

names = [
    "validate_green.py", "validation.json", "text_sha_before.txt", "text_sha_after.txt",
    "mode_before.txt", "mode_after.txt", "staged_summary.txt", "staged_numstat.txt",
    "target.stdout.log", "target.stderr.log", "target.exit", "full.stdout.log",
    "full.stderr.log", "full.exit", "completed_at_utc.txt",
]
with (ROOT / "evidence_manifest.sha256").open("w", encoding="utf-8") as handle:
    for name in names:
        handle.write(f"{hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}  {name}\n")

print(json.dumps({"status": result["status"], "passed": result["passed"], "total": result["total"]}, sort_keys=True))
raise SystemExit(result["status"] != "passed")
