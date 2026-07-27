#!/usr/bin/env python3
"""Fail-closed verifier for the Stage 5 OSCAR TP=8 runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def load_phase1_verifier(project_root: Path):
    path = project_root / "scripts/phase1/verify_native_baseline.py"
    spec = importlib.util.spec_from_file_location("phase1_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load phase 1 verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    add_check(checks, "manifest.status", manifest["status"], "ready")
    if manifest["status"] != "ready":
        result = {
            "format_version": 1,
            "status": "failed",
            "manifest": str(args.manifest.resolve()),
            "checks": checks,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    base_manifest_path = project_root / manifest["base_manifest"]
    add_check(
        checks,
        "base_manifest.sha256",
        sha256_file(base_manifest_path),
        manifest["base_manifest_sha256"],
    )
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    source_repo = Path(base_manifest["paths"]["source_repository"])
    source = manifest["source"]
    add_check(
        checks,
        "source.commit",
        git(source_repo, "rev-parse", "HEAD"),
        source["commit"],
    )
    add_check(
        checks,
        "source.tree",
        git(source_repo, "rev-parse", "HEAD^{tree}"),
        source["tree"],
    )
    add_check(
        checks,
        "source.branch",
        git(source_repo, "symbolic-ref", "--short", "HEAD"),
        source["branch"],
    )

    phase1 = load_phase1_verifier(project_root)
    native_manifest = json.loads(json.dumps(base_manifest))
    native_manifest["source"]["repository_commit"] = source["commit"]
    native_manifest["source"]["repository_tree"] = source["tree"]
    native_checks = phase1.Checks()
    phase1.verify_oci(native_checks, native_manifest)
    phase1.verify_source(native_checks, native_manifest)
    phase1.verify_model(native_checks, native_manifest)
    phase1.verify_suite(native_checks, native_manifest)
    checks.extend(native_checks.items)

    artifact = manifest["rotation_artifact"]
    artifact_dir = project_root / artifact["path"]
    artifact_manifest_path = artifact_dir / "manifest.json"
    rotations_path = artifact_dir / "rotations.pt"
    expectation_path = project_root / artifact["runtime_expectation"]
    add_check(
        checks,
        "rotation_artifact.manifest_sha256",
        sha256_file(artifact_manifest_path),
        artifact["manifest_sha256"],
    )
    add_check(
        checks,
        "rotation_artifact.rotations_sha256",
        sha256_file(rotations_path),
        artifact["rotations_sha256"],
    )

    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    expectation = json.loads(expectation_path.read_text(encoding="utf-8"))
    metadata = artifact_manifest["metadata"]
    for key, expected in expectation.items():
        add_check(checks, f"rotation_artifact.metadata.{key}", metadata[key], expected)
    add_check(
        checks,
        "rotation_artifact.layer_ids",
        artifact_manifest["layer_ids"],
        list(range(expectation["num_layers"])),
    )
    add_check(
        checks,
        "rotation_artifact.internal_rotations_sha256",
        artifact_manifest["rotations_sha256"],
        artifact["rotations_sha256"],
    )

    server = manifest["server"]
    expected_server = {
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 1,
        "attention_backend": "TRITON_MLA_SPARSE",
        "kv_cache_dtype": "oscar_mla_int2",
        "max_model_len": 32768,
        "enable_prefix_caching": False,
        "enable_speculative_decoding": False,
        "enable_cuda_graph": False,
        "async_scheduling": False,
    }
    for key, expected in expected_server.items():
        add_check(checks, f"server.{key}", server[key], expected)

    passed = all(item["status"] == "passed" for item in checks)
    result = {
        "format_version": 1,
        "status": "passed" if passed else "failed",
        "manifest": str(args.manifest.resolve()),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
