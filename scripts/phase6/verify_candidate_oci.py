#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
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


def verify_blob(layout: Path, descriptor: dict[str, Any]) -> None:
    path = blob_path(layout, descriptor["digest"])
    if path.stat().st_size != descriptor["size"]:
        raise ValueError(f"blob size mismatch: {path}")
    if sha256_file(path) != descriptor["digest"].removeprefix("sha256:"):
        raise ValueError(f"blob digest mismatch: {path}")


def parse_git_tree(repo: Path, commit: str) -> list[tuple[str, str, str, str]]:
    raw = subprocess.run(
        ["git", "ls-tree", "-rz", "-r", "--full-tree", commit],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    records = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        metadata, name = item.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode().split()
        records.append((mode, object_type, object_id, os.fsdecode(name)))
    return records


def verify_source_tree(root: Path, repo: Path, commit: str) -> int:
    expected_paths: set[str] = set()
    records = parse_git_tree(repo, commit)
    for mode, object_type, object_id, relative in records:
        if object_type != "blob":
            raise ValueError(f"unsupported Git object in source tree: {relative}")
        expected_paths.add(relative)
        path = root / relative
        if mode == "120000":
            if not path.is_symlink():
                raise ValueError(f"expected symlink: {relative}")
            data = os.readlink(path).encode()
        else:
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"expected regular file: {relative}")
            data = path.read_bytes()
            executable = bool(path.stat().st_mode & stat.S_IXUSR)
            if executable != (mode == "100755"):
                raise ValueError(f"executable mode mismatch: {relative}")
        if git_blob_hash(data) != object_id:
            raise ValueError(f"Git blob mismatch: {relative}")
    actual_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_paths != expected_paths:
        extra = sorted(actual_paths - expected_paths)[:10]
        missing = sorted(expected_paths - actual_paths)[:10]
        raise ValueError(f"source paths differ: extra={extra}, missing={missing}")
    return len(records)


def verify_native_extensions(
    base_rootfs: Path, source_repo: Path, manifest_path: str
) -> int:
    manifest = source_repo / manifest_path
    source_root = base_rootfs / "opt" / "vllm_glm52_v1"
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split(maxsplit=1)
        actual = sha256_file((source_root / relative).resolve())
        if actual != expected:
            raise ValueError(f"native extension mismatch: {relative}")
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    manifest = read_json(args.manifest.resolve())
    build = read_json(args.build_report.resolve())
    layout = args.layout.resolve()
    extract_root = args.extract_root.resolve()
    output = args.output.resolve()
    source_repo = project_root / manifest["source"]["path"]
    base_layout = project_root / manifest["base"]["layout"]
    base_rootfs = project_root / manifest["base"]["unpacked_rootfs"]

    index = read_json(layout / "index.json")
    matches = [
        item
        for item in index["manifests"]
        if item.get("annotations", {}).get("org.opencontainers.image.ref.name")
        == manifest["output_tag"]
    ]
    if len(matches) != 1:
        raise ValueError(f"candidate tag count is {len(matches)}, expected 1")
    descriptor = matches[0]
    if descriptor["digest"] != build["candidate"]["manifest_digest"]:
        raise ValueError("candidate manifest differs from build report")
    verify_blob(layout, descriptor)
    candidate_manifest = read_json(blob_path(layout, descriptor["digest"]))
    verify_blob(layout, candidate_manifest["config"])
    candidate_config = read_json(
        blob_path(layout, candidate_manifest["config"]["digest"])
    )

    base_index = read_json(base_layout / "index.json")
    base_descriptor = next(
        item
        for item in base_index["manifests"]
        if item.get("annotations", {}).get("org.opencontainers.image.ref.name")
        == manifest["base"]["tag"]
    )
    base_manifest = read_json(blob_path(base_layout, base_descriptor["digest"]))
    if candidate_manifest["layers"][:-1] != base_manifest["layers"]:
        raise ValueError("candidate base layers differ from phase 0 baseline")
    candidate_layer = candidate_manifest["layers"][-1]
    if candidate_layer["digest"] != build["candidate_layer"]["digest"]:
        raise ValueError("candidate layer digest differs from build report")
    verify_blob(layout, candidate_layer)

    labels = candidate_config["config"]["Labels"]
    expected_labels = {
        "org.opencontainers.image.revision": manifest["source"]["commit"],
        "ai.intellif.glm52.source-tree": manifest["source"]["tree"],
        "ai.intellif.glm52.rotation-manifest-sha256": (
            manifest["rotation_artifact"]["sha256"]["manifest.json"]
        ),
        "ai.intellif.glm52.rotations-sha256": (
            manifest["rotation_artifact"]["sha256"]["rotations.pt"]
        ),
        "ai.intellif.glm52.dockerfile-sha256": (manifest["dockerfile"]["sha256"]),
    }
    for name, expected in expected_labels.items():
        if labels.get(name) != expected:
            raise ValueError(f"candidate label mismatch: {name}")
    environments = set(candidate_config["config"]["Env"])
    required_environments = {
        "PYTHONPATH=/opt/vllm_glm52_v1",
        ("VLLM_OSCAR_MLA_ROTATION_ARTIFACT=/opt/oscar_artifacts/rotation_fit_v2"),
    }
    if not required_environments <= environments:
        raise ValueError("candidate runtime environment is incomplete")

    if extract_root.exists():
        raise FileExistsError(f"extract root already exists: {extract_root}")
    extract_root.mkdir(parents=True)
    with tarfile.open(blob_path(layout, candidate_layer["digest"]), "r:gz") as tar:
        unsafe = [
            item.name
            for item in tar.getmembers()
            if item.name.startswith("/") or ".." in Path(item.name).parts
        ]
        if unsafe:
            raise ValueError(f"unsafe candidate layer members: {unsafe[:10]}")
        tar.extractall(extract_root, filter="data")

    source_root = extract_root / "opt" / "vllm_glm52_v1"
    source_files = verify_source_tree(
        source_root, source_repo, manifest["source"]["commit"]
    )
    artifact_root = extract_root / "opt" / "oscar_artifacts" / "rotation_fit_v2"
    for filename, expected in manifest["rotation_artifact"]["sha256"].items():
        actual = sha256_file(artifact_root / filename)
        if actual != expected:
            raise ValueError(f"extracted artifact mismatch: {filename}")
    native_extensions = verify_native_extensions(
        base_rootfs, source_repo, manifest["native_extensions"]["manifest"]
    )

    result = {
        "format_version": 1,
        "status": "passed",
        "candidate": build["candidate"],
        "candidate_layer": build["candidate_layer"],
        "base_layers_exact_match": True,
        "source": {
            "commit": manifest["source"]["commit"],
            "tree": manifest["source"]["tree"],
            "files_verified": source_files,
            "exact_git_tree_match": True,
        },
        "rotation_artifact": {
            "files_verified": len(manifest["rotation_artifact"]["sha256"]),
            "sha256": manifest["rotation_artifact"]["sha256"],
        },
        "native_extensions": {
            "files_verified": native_extensions,
            "base_layer_sha256_match": True,
            "overwritten_by_candidate_layer": False,
        },
        "runtime_environment": sorted(required_environments),
        "extract_root": str(extract_root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
