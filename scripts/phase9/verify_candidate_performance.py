#!/usr/bin/env python3
"""Fail-closed Stage 9 verifier for the OSCAR candidate server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    actual: Any,
    expected: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "status": "passed" if actual == expected else "failed",
            "actual": actual,
            "expected": expected,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    runtime_root = Path(
        os.environ.get("OSCAR_RUNTIME_PROJECT_ROOT", project_root)
    ).resolve()
    manifest = read_json(args.manifest.resolve())
    performance = read_json(project_root / "configs/phase9/performance_matrix.json")
    checks: list[dict[str, Any]] = []
    frozen = performance["frozen_evaluator_snapshot"]
    frozen_root = runtime_root / frozen["path"]
    frozen_suite = frozen_root / frozen["accuracy_suite_relative_path"]
    add_check(
        checks,
        "frozen_evaluator.suite_dir",
        str(args.suite_dir.resolve()),
        str(frozen_suite.resolve()),
    )

    with tempfile.TemporaryDirectory(prefix="phase9-candidate-preflight-") as temp:
        temp_root = Path(temp)

        native_manifest = read_json(
            runtime_root / "configs/phase1/native_baseline.json"
        )
        native_manifest["paths"]["official_v4_suite"] = str(frozen_suite)
        native_manifest_path = temp_root / "native_baseline.json"
        native_manifest_path.write_text(
            json.dumps(native_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        stage5_manifest = read_json(runtime_root / "configs/phase5/oscar_tp8.json")
        stage5_manifest["base_manifest"] = str(native_manifest_path)
        stage5_manifest["base_manifest_sha256"] = sha256_file(native_manifest_path)
        stage5_manifest_path = temp_root / "oscar_tp8.json"
        stage5_manifest_path.write_text(
            json.dumps(stage5_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        phase7_manifest = json.loads(json.dumps(manifest))
        phase7_manifest["stage5_manifest"]["path"] = str(stage5_manifest_path)
        phase7_manifest["stage5_manifest"]["sha256"] = sha256_file(stage5_manifest_path)
        phase7_manifest_path = temp_root / "oscar_evaluation.json"
        phase7_manifest_path.write_text(
            json.dumps(phase7_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        phase7_output = temp_root / "phase7.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(runtime_root / "scripts/phase7/verify_candidate_evaluation.py"),
                "--manifest",
                str(phase7_manifest_path),
                "--output",
                str(phase7_output),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
        )
        phase7_result = read_json(phase7_output)
    add_check(checks, "phase7_preflight.exit_status", completed.returncode, 0)
    add_check(checks, "phase7_preflight.status", phase7_result["status"], "passed")
    checks.extend(
        {
            **item,
            "name": f"phase7.{item['name']}",
        }
        for item in phase7_result["checks"]
    )

    add_check(checks, "performance.status", performance["status"], "ready")
    add_check(
        checks,
        "performance.source.commit",
        performance["source"]["commit"],
        manifest["source"]["commit"],
    )
    for name in ("tag", "manifest_digest", "config_digest", "layer_digest"):
        add_check(
            checks,
            f"performance.candidate.{name}",
            performance["candidate"][name],
            manifest["candidate"][name],
        )
    add_check(
        checks,
        "performance.server.tensor_parallel_size",
        performance["server"]["tensor_parallel_size"],
        8,
    )
    add_check(
        checks,
        "performance.server.max_model_len",
        performance["server"]["max_model_len"],
        131072,
    )
    add_check(
        checks,
        "performance.matrix.input_lengths",
        performance["matrix"]["input_lengths"],
        [1024, 8192, 32768],
    )
    add_check(
        checks,
        "performance.matrix.batch_sizes",
        performance["matrix"]["batch_sizes"],
        [1, 4, 8],
    )
    add_check(
        checks,
        "performance.context_128k.total_length",
        performance["context_128k"]["input_length"]
        + performance["context_128k"]["output_length"],
        performance["context_128k"]["total_length"],
    )
    add_check(
        checks,
        "performance.context_128k.server_fit",
        performance["context_128k"]["total_length"],
        performance["server"]["max_model_len"],
    )

    frozen_files = {
        "accuracy_suite_identity_sha256": frozen_suite / "identity.json",
        "accuracy_manifest_sha256": (
            frozen_root / "accuracy_v4_fc374ff4_4aec8ee8/manifest.jsonl"
        ),
        "accuracy_runner_sha256": (
            frozen_root / "accuracy_v4_fc374ff4_4aec8ee8/run_accuracy_suite.py"
        ),
        "ppl_manifest_sha256": frozen_root / "manifest.jsonl",
        "ppl_runner_sha256": frozen_root / "run_vllm_perplexity_suite.py",
    }
    for name, path in frozen_files.items():
        add_check(
            checks,
            f"frozen_evaluator.{name}",
            sha256_file(path),
            frozen[name],
        )

    result = {
        "format_version": 1,
        "status": (
            "passed" if all(item["status"] == "passed" for item in checks) else "failed"
        ),
        "manifest": str(args.manifest.resolve()),
        "performance_config": str(
            project_root / "configs/phase9/performance_matrix.json"
        ),
        "runtime_project_root": str(runtime_root),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
