#!/usr/bin/env python3
"""Fail-closed Stage 9 verifier for the native performance server."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_phase1_verifier(project_root: Path):
    path = project_root / "scripts/phase1/verify_native_baseline.py"
    spec = importlib.util.spec_from_file_location("phase1_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Stage 1 verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    runtime_root = Path(
        os.environ.get("OSCAR_RUNTIME_PROJECT_ROOT", project_root)
    ).resolve()
    phase1 = load_phase1_verifier(project_root)
    manifest = read_json(args.manifest.resolve())
    performance = read_json(project_root / "configs/phase9/performance_matrix.json")
    checks = phase1.Checks()

    phase1.verify_oci(checks, manifest)
    phase1.verify_source(checks, manifest)
    phase1.verify_model(checks, manifest)

    checks.equal("performance.status", performance["status"], "ready")
    checks.equal(
        "performance.model.path",
        performance["model"]["path"],
        manifest["paths"]["model"],
    )
    checks.equal(
        "performance.source.commit",
        performance["source"]["commit"],
        manifest["source"]["repository_commit"],
    )
    checks.equal(
        "performance.server.tensor_parallel_size",
        performance["server"]["tensor_parallel_size"],
        8,
    )
    checks.equal(
        "performance.server.max_model_len",
        performance["server"]["max_model_len"],
        131072,
    )
    checks.equal(
        "performance.matrix.input_lengths",
        performance["matrix"]["input_lengths"],
        [1024, 8192, 32768],
    )
    checks.equal(
        "performance.matrix.batch_sizes",
        performance["matrix"]["batch_sizes"],
        [1, 4, 8],
    )
    checks.equal(
        "performance.matrix.rounds",
        performance["matrix"]["rounds"],
        3,
    )
    checks.equal(
        "performance.context_128k.total_length",
        performance["context_128k"]["input_length"]
        + performance["context_128k"]["output_length"],
        performance["context_128k"]["total_length"],
    )
    checks.equal(
        "performance.context_128k.server_fit",
        performance["context_128k"]["total_length"],
        performance["server"]["max_model_len"],
    )

    frozen = performance["frozen_evaluator_snapshot"]
    frozen_root = runtime_root / frozen["path"]
    frozen_suite = frozen_root / frozen["accuracy_suite_relative_path"]
    suite_manifest = json.loads(json.dumps(manifest))
    suite_manifest["paths"]["official_v4_suite"] = str(frozen_suite)
    phase1.verify_suite(checks, suite_manifest)
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
        checks.equal(
            f"frozen_evaluator.{name}",
            sha256_file(path),
            frozen[name],
        )

    result = {
        "format_version": 1,
        "status": "passed" if checks.passed else "failed",
        "manifest": str(args.manifest.resolve()),
        "performance_config": str(
            project_root / "configs/phase9/performance_matrix.json"
        ),
        "runtime_project_root": str(runtime_root),
        "checks": checks.items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if checks.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
