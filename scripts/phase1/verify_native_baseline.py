#!/usr/bin/env python3
"""Fail-closed static verifier for the GLM-5.2 native baseline inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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


def sha256_head_tail(path: Path, size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(size))
        handle.seek(-size, 2)
        digest.update(handle.read(size))
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def equal(self, name: str, actual: Any, expected: Any) -> None:
        self.items.append(
            {
                "name": name,
                "status": "passed" if actual == expected else "failed",
                "actual": actual,
                "expected": expected,
            }
        )

    def true(self, name: str, condition: bool, detail: Any) -> None:
        self.items.append(
            {
                "name": name,
                "status": "passed" if condition else "failed",
                "actual": detail,
                "expected": True,
            }
        )

    @property
    def passed(self) -> bool:
        return all(item["status"] == "passed" for item in self.items)


def verify_oci(checks: Checks, manifest: dict[str, Any]) -> None:
    layout = Path(manifest["paths"]["candidate_oci_layout"])
    index = json.loads((layout / "index.json").read_text(encoding="utf-8"))
    candidates = [
        item
        for item in index["manifests"]
        if item.get("annotations", {}).get("org.opencontainers.image.ref.name")
        == "phase0-baseline"
    ]
    checks.equal("oci.phase0_baseline_tag_count", len(candidates), 1)
    if len(candidates) != 1:
        return
    descriptor = candidates[0]
    checks.equal(
        "oci.manifest_digest",
        descriptor["digest"],
        manifest["source"]["candidate_manifest_digest"],
    )
    manifest_blob = layout / "blobs/sha256" / descriptor["digest"].split(":", 1)[1]
    oci_manifest = json.loads(manifest_blob.read_text(encoding="utf-8"))
    checks.equal(
        "oci.config_digest",
        oci_manifest["config"]["digest"],
        manifest["source"]["candidate_config_digest"],
    )
    checks.equal(
        "oci.source_layer_digest",
        oci_manifest["layers"][-1]["digest"],
        manifest["source"]["source_layer_digest"],
    )
    config_blob = (
        layout
        / "blobs/sha256"
        / oci_manifest["config"]["digest"].split(":", 1)[1]
    )
    image_config = json.loads(config_blob.read_text(encoding="utf-8"))
    checks.equal(
        "oci.runtime_source_commit",
        image_config["config"]["Labels"]["org.opencontainers.image.revision"],
        manifest["source"]["runtime_source_commit"],
    )
    checks.equal(
        "oci.runtime_source_tree",
        image_config["config"]["Labels"]["ai.intellif.glm52.source-tree"],
        manifest["source"]["runtime_source_tree"],
    )


def verify_source(checks: Checks, manifest: dict[str, Any]) -> None:
    repo = Path(manifest["paths"]["source_repository"])
    rootfs_source = (
        Path(manifest["paths"]["candidate_rootfs"]) / "opt/vllm_glm52_v1"
    )
    checks.equal(
        "source.repository_commit",
        git(repo, "rev-parse", "HEAD"),
        manifest["source"]["repository_commit"],
    )
    checks.equal(
        "source.repository_tree",
        git(repo, "rev-parse", "HEAD^{tree}"),
        manifest["source"]["repository_tree"],
    )
    checks.equal(
        "source.runtime_source_tree",
        git(repo, "rev-parse", f"{manifest['source']['runtime_source_commit']}^{{tree}}"),
        manifest["source"]["runtime_source_tree"],
    )

    tree_output = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "ls-tree",
            "-r",
            "-z",
            manifest["source"]["runtime_source_commit"],
        ]
    )
    entries = [item for item in tree_output.split(b"\0") if item]
    mismatches: list[str] = []
    for entry in entries:
        metadata, raw_relative = entry.split(b"\t", 1)
        mode, object_type, expected_blob = metadata.decode().split()
        relative = os.fsdecode(raw_relative)
        rootfs_file = rootfs_source / relative
        if object_type != "blob" or not (
            rootfs_file.exists() or rootfs_file.is_symlink()
        ):
            mismatches.append(relative)
        else:
            if mode == "120000":
                data = os.fsencode(os.readlink(rootfs_file))
                mode_matches = rootfs_file.is_symlink()
            else:
                data = rootfs_file.read_bytes()
                executable = bool(rootfs_file.stat().st_mode & 0o111)
                mode_matches = executable == (mode == "100755")
            blob_header = f"blob {len(data)}\0".encode()
            actual_blob = hashlib.sha1(blob_header + data).hexdigest()
            if actual_blob != expected_blob or not mode_matches:
                mismatches.append(relative)
        if len(mismatches) == 20:
            break
    checks.true(
        "source.rootfs_runtime_tree_match",
        not mismatches,
        {
            "tracked_files": len(entries),
            "first_mismatches": mismatches,
        },
    )
    checks.equal(
        "source.runtime_source_files",
        len(entries),
        manifest["source"]["runtime_source_files"],
    )

    native_manifest = repo / "recovery/native_extensions.sha256"
    native_mismatches: list[str] = []
    native_count = 0
    for line in native_manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split(maxsplit=1)
        native_count += 1
        target = (rootfs_source / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            native_mismatches.append(relative)
    checks.true(
        "source.native_extensions_match",
        not native_mismatches,
        {"files": native_count, "mismatches": native_mismatches},
    )


def verify_model(checks: Checks, manifest: dict[str, Any]) -> None:
    model_dir = Path(manifest["paths"]["model"])
    expected = manifest["model"]
    file_hashes = {
        "config.json": expected["config_sha256"],
        "generation_config.json": expected["generation_config_sha256"],
        "tokenizer_config.json": expected["tokenizer_config_sha256"],
        "tokenizer.json": expected["tokenizer_sha256"],
        "model.safetensors.index.json": expected["index_sha256"],
    }
    for name, expected_hash in file_hashes.items():
        checks.equal(f"model.{name}.sha256", sha256_file(model_dir / name), expected_hash)

    shards = sorted(model_dir.glob("*.safetensors"))
    total_bytes = sum(item.stat().st_size for item in shards)
    checks.equal("model.safetensors_count", len(shards), expected["safetensors_count"])
    checks.equal("model.safetensors_total_bytes", total_bytes, expected["safetensors_total_bytes"])

    filename_size = "".join(
        f"{item.name} {item.stat().st_size}\n" for item in shards
    ).encode()
    checks.equal(
        "model.filename_size_manifest_sha256",
        hashlib.sha256(filename_size).hexdigest(),
        expected["filename_size_manifest_sha256"],
    )
    filename_size_mtime = "".join(
        f"{item.name} {item.stat().st_size} {item.stat().st_mtime_ns}\n"
        for item in shards
    ).encode()
    checks.equal(
        "model.filename_size_mtime_ns_manifest_sha256",
        hashlib.sha256(filename_size_mtime).hexdigest(),
        expected["filename_size_mtime_ns_manifest_sha256"],
    )

    for label in ("first_shard", "last_shard"):
        shard_expected = expected[label]
        shard = model_dir / shard_expected["name"]
        checks.equal(f"model.{label}.bytes", shard.stat().st_size, shard_expected["bytes"])
        checks.equal(
            f"model.{label}.head_tail_1mib_sha256",
            sha256_head_tail(shard),
            shard_expected["head_tail_1mib_sha256"],
        )

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures", [])
    checks.true(
        "model.architecture",
        expected["architecture"] in architectures,
        architectures,
    )
    for key, expected_value in expected["geometry"].items():
        checks.equal(f"model.geometry.{key}", config.get(key), expected_value)

    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    expert_tokens: set[str] = set()
    for name in index["weight_map"]:
        match = re.search(r"experts\.[0-9]+", name)
        if match:
            expert_tokens.add(match.group(0))
    expert_mapping = "".join(
        f"{token}\n" for token in sorted(expert_tokens)
    ).encode()
    checks.equal(
        "model.expert_mapping_sha256",
        hashlib.sha256(expert_mapping).hexdigest(),
        expected["expert_mapping_sha256"],
    )
    referenced_shards = sorted(set(index["weight_map"].values()))
    checks.equal("model.index_weight_entries", len(index["weight_map"]), 72117)
    checks.equal(
        "model.index_referenced_shards",
        len(referenced_shards),
        expected["safetensors_count"],
    )
    checks.equal(
        "model.index_total_size",
        index["metadata"]["total_size"],
        expected["safetensors_total_bytes"],
    )


def verify_suite(checks: Checks, manifest: dict[str, Any]) -> None:
    suite_dir = Path(manifest["paths"]["official_v4_suite"])
    expected = manifest["official_v4"]
    for name, key in (
        ("eval_config.json", "eval_config_sha256"),
        ("manifest.jsonl", "manifest_sha256"),
        ("suite_meta.json", "suite_meta_sha256"),
    ):
        checks.equal(f"suite.{name}.sha256", sha256_file(suite_dir / name), expected[key])

    accuracy_rows = 0
    for name, sample_expected in expected["samples"].items():
        sample = suite_dir / "samples" / name
        rows = sum(1 for _ in sample.open("rb"))
        checks.equal(f"suite.samples.{name}.rows", rows, sample_expected["rows"])
        checks.equal(
            f"suite.samples.{name}.sha256",
            sha256_file(sample),
            sample_expected["sha256"],
        )
        if name != "wikitext2_perplexity.jsonl":
            accuracy_rows += rows
    checks.equal("suite.accuracy_total_rows", accuracy_rows, 2360)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = Checks()
    verify_oci(checks, manifest)
    verify_source(checks, manifest)
    verify_model(checks, manifest)
    verify_suite(checks, manifest)

    result = {
        "format_version": 1,
        "status": "passed" if checks.passed else "failed",
        "manifest": str(args.manifest.resolve()),
        "checks": checks.items,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if checks.passed else 1


if __name__ == "__main__":
    sys.exit(main())
