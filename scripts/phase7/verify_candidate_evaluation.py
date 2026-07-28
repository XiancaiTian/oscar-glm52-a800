#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any


RUNTIME_NATIVE_LINKS = (
    "vllm/_C.abi3.so",
    "vllm/_C_stable_libtorch.abi3.so",
    "vllm/_moe_C.abi3.so",
    "vllm/cumem_allocator.abi3.so",
    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
    "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_hash(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def blob_path(layout: Path, digest: str) -> Path:
    algorithm, value = digest.split(":", 1)
    if algorithm != "sha256":
        raise ValueError(f"unsupported digest: {digest}")
    return layout / "blobs" / "sha256" / value


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


def load_phase6_verifier(project_root: Path):
    path = project_root / "scripts/phase6/verify_candidate_oci.py"
    spec = importlib.util.spec_from_file_location("phase6_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Stage 6 verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_runtime_source(
    checks: list[dict[str, Any]],
    source_root: Path,
    source_repo: Path,
    commit: str,
    expected_files: int,
    base_source_root: Path,
) -> None:
    phase6 = load_phase6_verifier(Path(__file__).resolve().parents[2])
    records = phase6.parse_git_tree(source_repo, commit)
    add_check(checks, "candidate_source.git_files", len(records), expected_files)
    tracked: set[str] = set()
    mismatches: list[str] = []
    for mode, object_type, object_id, relative in records:
        tracked.add(relative)
        path = source_root / relative
        if object_type != "blob":
            mismatches.append(f"{relative}:object_type={object_type}")
            continue
        if mode == "120000":
            if not path.is_symlink():
                mismatches.append(f"{relative}:not_symlink")
                continue
            data = os.readlink(path).encode()
        else:
            if not path.is_file() or path.is_symlink():
                mismatches.append(f"{relative}:not_regular")
                continue
            data = path.read_bytes()
            executable = bool(path.stat().st_mode & stat.S_IXUSR)
            if executable != (mode == "100755"):
                mismatches.append(f"{relative}:mode")
        if git_blob_hash(data) != object_id:
            mismatches.append(f"{relative}:blob")
    actual = {
        str(path.relative_to(source_root))
        for path in source_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    expected_extras = set(RUNTIME_NATIVE_LINKS)
    extras = actual - tracked
    missing = tracked - actual
    add_check(
        checks,
        "candidate_source.git_tree_match",
        {
            "mismatches": mismatches[:20],
            "missing": sorted(missing)[:20],
            "extras": sorted(extras),
        },
        {
            "mismatches": [],
            "missing": [],
            "extras": sorted(expected_extras),
        },
    )
    for relative in RUNTIME_NATIVE_LINKS:
        runtime_path = source_root / relative
        base_path = base_source_root / relative
        add_check(
            checks,
            f"candidate_source.lower_native_link.{relative}",
            {
                "is_symlink": runtime_path.is_symlink(),
                "target": os.path.realpath(runtime_path),
                "sha256": sha256_file(runtime_path),
            },
            {
                "is_symlink": True,
                "target": str(base_path.resolve()),
                "sha256": sha256_file(base_path),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    manifest = read_json(args.manifest.resolve())
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

    stage5_manifest_path = project_root / manifest["stage5_manifest"]["path"]
    frozen_suite_dir = (
        args.suite_dir.resolve()
        if args.suite_dir is not None
        else (
            project_root
            / "artifacts/phase7/frozen_evaluator_v4_20260728"
            / "accuracy_v4_fc374ff4_4aec8ee8/suite"
        )
    )
    add_check(
        checks,
        "stage5_manifest.sha256",
        sha256_file(stage5_manifest_path),
        manifest["stage5_manifest"]["sha256"],
    )
    with tempfile.TemporaryDirectory(prefix="phase7-stage5-preflight-") as temp:
        stage5_output = Path(temp) / "preflight.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts/phase5/verify_oscar_tp8.py"),
                "--manifest",
                str(stage5_manifest_path),
                "--output",
                str(stage5_output),
                "--suite-dir",
                str(frozen_suite_dir),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
        )
        stage5_result = read_json(stage5_output)
    add_check(checks, "stage5_preflight.exit_status", completed.returncode, 0)
    add_check(checks, "stage5_preflight.status", stage5_result["status"], "passed")

    candidate = manifest["candidate"]
    layout = project_root / candidate["layout"]
    index = read_json(layout / "index.json")
    descriptors = [
        item
        for item in index["manifests"]
        if item.get("annotations", {}).get("org.opencontainers.image.ref.name")
        == candidate["tag"]
    ]
    add_check(checks, "candidate.tag_count", len(descriptors), 1)
    descriptor = descriptors[0]
    add_check(
        checks,
        "candidate.manifest_digest",
        descriptor["digest"],
        candidate["manifest_digest"],
    )
    add_check(
        checks,
        "candidate.manifest_blob_sha256",
        sha256_file(blob_path(layout, descriptor["digest"])),
        candidate["manifest_digest"].removeprefix("sha256:"),
    )
    image_manifest = read_json(blob_path(layout, descriptor["digest"]))
    add_check(
        checks,
        "candidate.config_digest",
        image_manifest["config"]["digest"],
        candidate["config_digest"],
    )
    add_check(
        checks,
        "candidate.config_blob_sha256",
        sha256_file(blob_path(layout, image_manifest["config"]["digest"])),
        candidate["config_digest"].removeprefix("sha256:"),
    )
    add_check(
        checks,
        "candidate.layer_digest",
        image_manifest["layers"][-1]["digest"],
        candidate["layer_digest"],
    )
    add_check(
        checks,
        "candidate.layer_blob_sha256",
        sha256_file(blob_path(layout, image_manifest["layers"][-1]["digest"])),
        candidate["layer_digest"].removeprefix("sha256:"),
    )

    for name in ("build_report", "verification", "runtime_import"):
        evidence = candidate[name]
        evidence_path = project_root / evidence["path"]
        add_check(
            checks,
            f"candidate.{name}.sha256",
            sha256_file(evidence_path),
            evidence["sha256"],
        )
        add_check(
            checks,
            f"candidate.{name}.status",
            read_json(evidence_path)["status"],
            "passed" if name != "build_report" else "built",
        )

    overlay_root = project_root / candidate["overlay_rootfs"]
    source_root = overlay_root / "opt" / "vllm_glm52_v1"
    source = manifest["source"]
    source_repo = project_root / source["path"]
    base_source_root = (
        project_root / "artifacts/phase0-candidate-bundle/rootfs/opt/vllm_glm52_v1"
    )
    verify_runtime_source(
        checks,
        source_root,
        source_repo,
        source["commit"],
        source["files"],
        base_source_root,
    )

    artifact_root = overlay_root / manifest["rotation_artifact"]["relative_path"]
    actual_artifact_files = {
        str(path.relative_to(artifact_root))
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    add_check(
        checks,
        "candidate.rotation_artifact.files",
        sorted(actual_artifact_files),
        sorted(manifest["rotation_artifact"]["sha256"]),
    )
    for filename, expected in manifest["rotation_artifact"]["sha256"].items():
        add_check(
            checks,
            f"candidate.rotation_artifact.{filename}.sha256",
            sha256_file(artifact_root / filename),
            expected,
        )
    expectation = manifest["runtime_expectation"]
    expectation_path = overlay_root / expectation["relative_path"]
    add_check(
        checks,
        "candidate.runtime_expectation.sha256",
        sha256_file(expectation_path),
        expectation["sha256"],
    )

    baseline = manifest["baseline"]
    for group in ("accuracy", "ppl"):
        for name, value in baseline[group].items():
            if not name.endswith("_sha256"):
                continue
            path_name = name.removesuffix("_sha256")
            path = project_root / baseline[group][path_name]
            add_check(
                checks,
                f"baseline.{group}.{path_name}.sha256",
                sha256_file(path),
                value,
            )

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
    for name, expected in expected_server.items():
        add_check(
            checks,
            f"server.{name}",
            manifest["server"][name],
            expected,
        )

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
    raise SystemExit(main())
