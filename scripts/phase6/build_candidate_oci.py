#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = True,
) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return completed.stdout.strip() if capture else ""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def write_blob(blobs: Path, value: Any) -> tuple[str, int]:
    raw = canonical_json(value)
    digest = hashlib.sha256(raw).hexdigest()
    (blobs / digest).write_bytes(raw)
    return f"sha256:{digest}", len(raw)


def blob_path(layout: Path, digest: str) -> Path:
    algorithm, value = digest.split(":", 1)
    if algorithm != "sha256":
        raise ValueError(f"unsupported digest: {digest}")
    return layout / "blobs" / "sha256" / value


def read_blob(layout: Path, digest: str) -> Any:
    return read_json(blob_path(layout, digest))


def descriptor_for_tag(index: dict[str, Any], tag: str) -> dict[str, Any]:
    matches = [
        item
        for item in index["manifests"]
        if item.get("annotations", {}).get("org.opencontainers.image.ref.name") == tag
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one OCI descriptor for {tag}, got {len(matches)}")
    return matches[0]


def require_clean_published(repo: Path, expected_branch: str) -> str:
    status = run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
    )
    if status:
        raise ValueError(f"repository is not clean: {repo}\n{status}")
    branch = run(["git", "symbolic-ref", "--quiet", "--short", "HEAD"], cwd=repo)
    if branch != expected_branch:
        raise ValueError(
            f"unexpected branch for {repo}: {branch}; expected {expected_branch}"
        )
    head = run(["git", "rev-parse", "HEAD"], cwd=repo)
    upstream = run(["git", "rev-parse", "@{upstream}"], cwd=repo)
    if head != upstream:
        raise ValueError(f"HEAD is not published for {repo}: {head} != {upstream}")
    return head


def clone_layout_with_hardlinks(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output OCI layout already exists: {output}")
    output.mkdir(parents=True)
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        output_path = output / relative
        if source_path.is_dir():
            output_path.mkdir()
        elif relative in (Path("index.json"), Path("oci-layout")):
            shutil.copy2(source_path, output_path)
        else:
            os.link(source_path, output_path)


def export_payload(
    source_repo: Path,
    source_commit: str,
    artifact_dir: Path,
    runtime_expectation: Path,
    payload_root: Path,
) -> tuple[int, int]:
    source_target = payload_root / "opt" / "vllm_glm52_v1"
    artifact_target = payload_root / "opt" / "oscar_artifacts" / "rotation_fit_v2"
    expectation_target = (
        payload_root / "opt" / "oscar_artifacts" / "oscar_runtime_expectation.json"
    )
    source_target.mkdir(parents=True)
    artifact_target.parent.mkdir(parents=True)

    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", source_commit],
        cwd=source_repo,
        stdout=subprocess.PIPE,
    )
    if archive.stdout is None:
        raise RuntimeError("git archive stdout was not created")
    extract = subprocess.run(
        ["tar", "-xf", "-", "-C", str(source_target)],
        stdin=archive.stdout,
        check=False,
    )
    archive.stdout.close()
    archive_status = archive.wait()
    if archive_status != 0 or extract.returncode != 0:
        raise RuntimeError(
            f"source export failed: git={archive_status}, tar={extract.returncode}"
        )

    shutil.copytree(artifact_dir, artifact_target)
    shutil.copy2(runtime_expectation, expectation_target)
    source_files = sum(path.is_file() for path in source_target.rglob("*"))
    artifact_files = sum(path.is_file() for path in artifact_target.rglob("*"))
    return source_files, artifact_files


def build_layer(
    payload_root: Path,
    work_dir: Path,
    source_epoch: int,
) -> tuple[Path, str, str, int, int]:
    layer_tar = work_dir / "candidate-layer.tar"
    layer_gzip = work_dir / "candidate-layer.tar.gz"
    run(
        [
            "tar",
            "--sort=name",
            f"--mtime=@{source_epoch}",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "--format=pax",
            "--pax-option=delete=atime,delete=ctime",
            "-C",
            str(payload_root),
            "-cf",
            str(layer_tar),
            "opt/vllm_glm52_v1",
            "opt/oscar_artifacts/rotation_fit_v2",
            "opt/oscar_artifacts/oscar_runtime_expectation.json",
        ],
        capture=False,
    )
    with layer_tar.open("rb") as source, layer_gzip.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=raw_output,
            mtime=0,
        ) as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
    with tarfile.open(layer_gzip, mode="r:gz") as archive:
        names = archive.getnames()
    if any(
        name.endswith(".so") or "/.wh." in name or Path(name).name.startswith(".wh.")
        for name in names
    ):
        raise ValueError("candidate layer contains a native extension or whiteout")
    return (
        layer_gzip,
        f"sha256:{sha256_file(layer_gzip)}",
        f"sha256:{sha256_file(layer_tar)}",
        layer_gzip.stat().st_size,
        len(names),
    )


def update_environment(environment: list[str], name: str, value: str) -> None:
    prefix = f"{name}="
    environment[:] = [item for item in environment if not item.startswith(prefix)]
    environment.append(f"{name}={value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-layout", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    manifest = read_json(args.manifest.resolve())
    if manifest["status"] != "ready":
        raise ValueError(f"candidate input manifest is not ready: {manifest['status']}")

    base_layout = project_root / manifest["base"]["layout"]
    source_repo = project_root / manifest["source"]["path"]
    artifact_dir = project_root / manifest["rotation_artifact"]["path"]
    runtime_expectation = project_root / manifest["runtime_expectation"]["path"]
    dockerfile = project_root / manifest["dockerfile"]["path"]
    output_layout = args.output_layout.resolve()
    output_report = args.output_report.resolve()

    main_commit = require_clean_published(
        project_root, manifest["main_repository"]["branch"]
    )
    source_commit = require_clean_published(source_repo, manifest["source"]["branch"])
    if source_commit != manifest["source"]["commit"]:
        raise ValueError(f"source commit mismatch: {source_commit}")
    source_tree = run(["git", "rev-parse", "HEAD^{tree}"], cwd=source_repo)
    if source_tree != manifest["source"]["tree"]:
        raise ValueError(f"source tree mismatch: {source_tree}")

    artifact_hashes = manifest["rotation_artifact"]["sha256"]
    for filename, expected in artifact_hashes.items():
        actual = sha256_file(artifact_dir / filename)
        if actual != expected:
            raise ValueError(f"artifact hash mismatch for {filename}: {actual}")
    expectation_hash = sha256_file(runtime_expectation)
    if expectation_hash != manifest["runtime_expectation"]["sha256"]:
        raise ValueError(f"runtime expectation hash mismatch: {expectation_hash}")
    actual_dockerfile_hash = sha256_file(dockerfile)
    if actual_dockerfile_hash != manifest["dockerfile"]["sha256"]:
        raise ValueError(f"Dockerfile hash mismatch: {actual_dockerfile_hash}")

    base_index = read_json(base_layout / "index.json")
    base_descriptor = descriptor_for_tag(base_index, manifest["base"]["tag"])
    if base_descriptor["digest"] != manifest["base"]["manifest_digest"]:
        raise ValueError(f"base manifest mismatch: {base_descriptor['digest']}")
    base_manifest = read_blob(base_layout, base_descriptor["digest"])
    base_config = read_blob(base_layout, base_manifest["config"]["digest"])

    source_epoch = int(
        run(["git", "show", "-s", "--format=%ct", source_commit], cwd=source_repo)
    )
    created_at = (
        dt.datetime.fromtimestamp(source_epoch, tz=dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    with tempfile.TemporaryDirectory(prefix="glm52-phase6-", dir="/tmp") as temp:
        work_dir = Path(temp)
        payload_root = work_dir / "rootfs"
        source_files, artifact_files = export_payload(
            source_repo,
            source_commit,
            artifact_dir,
            runtime_expectation,
            payload_root,
        )
        layer_path, layer_digest, layer_diff_id, layer_size, layer_members = (
            build_layer(payload_root, work_dir, source_epoch)
        )

        clone_layout_with_hardlinks(base_layout, output_layout)
        blobs = output_layout / "blobs" / "sha256"
        layer_blob = blob_path(output_layout, layer_digest)
        shutil.copyfile(layer_path, layer_blob)
        if sha256_file(layer_blob) != layer_digest.removeprefix("sha256:"):
            raise RuntimeError("copied candidate layer digest mismatch")

        config = json.loads(json.dumps(base_config))
        config["created"] = created_at
        image_config = config.setdefault("config", {})
        labels = image_config.setdefault("Labels", {})
        labels.update(
            {
                "ai.intellif.glm52.base-manifest": base_descriptor["digest"],
                "ai.intellif.glm52.candidate-layer-digest": layer_digest,
                "ai.intellif.glm52.dockerfile-sha256": actual_dockerfile_hash,
                "ai.intellif.glm52.rotation-manifest-sha256": (
                    artifact_hashes["manifest.json"]
                ),
                "ai.intellif.glm52.rotations-sha256": (artifact_hashes["rotations.pt"]),
                "ai.intellif.glm52.runtime-expectation-sha256": expectation_hash,
                "ai.intellif.glm52.source-tree": source_tree,
                "org.opencontainers.image.base.digest": base_descriptor["digest"],
                "org.opencontainers.image.created": created_at,
                "org.opencontainers.image.revision": source_commit,
                "org.opencontainers.image.title": ("GLM-5.2 OSCAR A800 candidate"),
            }
        )
        environment = image_config.setdefault("Env", [])
        update_environment(environment, "PYTHONPATH", "/opt/vllm_glm52_v1")
        update_environment(
            environment,
            "VLLM_OSCAR_MLA_ROTATION_ARTIFACT",
            "/opt/oscar_artifacts/rotation_fit_v2",
        )
        update_environment(
            environment,
            "VLLM_OSCAR_MLA_RUNTIME_EXPECTATION",
            "/opt/oscar_artifacts/oscar_runtime_expectation.json",
        )
        image_config["WorkingDir"] = "/opt/vllm_glm52_v1"
        image_config["Entrypoint"] = ["/bin/bash"]
        image_config["Cmd"] = ["-lc", "sleep infinity"]
        config.setdefault("rootfs", {}).setdefault("diff_ids", []).append(layer_diff_id)
        config.setdefault("history", []).append(
            {
                "created": created_at,
                "created_by": (
                    "COPY verified source and rotation artifact into candidate"
                ),
                "comment": "stage 6 reproducible OSCAR candidate layer",
            }
        )
        config_digest, config_size = write_blob(blobs, config)

        candidate_manifest = json.loads(json.dumps(base_manifest))
        candidate_manifest["config"] = {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "digest": config_digest,
            "size": config_size,
        }
        candidate_manifest.setdefault("layers", []).append(
            {
                "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                "digest": layer_digest,
                "size": layer_size,
            }
        )
        candidate_manifest.setdefault("annotations", {}).update(
            {
                "org.opencontainers.image.created": created_at,
                "org.opencontainers.image.revision": source_commit,
                "org.opencontainers.image.title": ("GLM-5.2 OSCAR A800 candidate"),
            }
        )
        candidate_digest, candidate_size = write_blob(blobs, candidate_manifest)
        output_index = {
            "schemaVersion": 2,
            "manifests": [
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": candidate_digest,
                    "size": candidate_size,
                    "annotations": {
                        "org.opencontainers.image.ref.name": manifest["output_tag"]
                    },
                }
            ],
        }
        (output_layout / "index.json").write_bytes(canonical_json(output_index))

    report = {
        "format_version": 1,
        "status": "built",
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": sha256_file(args.manifest.resolve()),
        "main_repository_commit": main_commit,
        "source": {
            "commit": source_commit,
            "tree": source_tree,
            "tracked_files": source_files,
        },
        "dockerfile": {
            "path": str(dockerfile),
            "sha256": actual_dockerfile_hash,
        },
        "rotation_artifact": {
            "path": str(artifact_dir),
            "files": artifact_files,
            "sha256": artifact_hashes,
        },
        "runtime_expectation": {
            "path": str(runtime_expectation),
            "target": manifest["runtime_expectation"]["target"],
            "sha256": expectation_hash,
        },
        "base": {
            "manifest_digest": base_descriptor["digest"],
            "config_digest": base_manifest["config"]["digest"],
            "layers": len(base_manifest["layers"]),
        },
        "candidate": {
            "tag": manifest["output_tag"],
            "image_id": config_digest,
            "manifest_digest": candidate_digest,
            "config_digest": config_digest,
            "layers": len(candidate_manifest["layers"]),
            "created": created_at,
        },
        "candidate_layer": {
            "digest": layer_digest,
            "diff_id": layer_diff_id,
            "size": layer_size,
            "members": layer_members,
            "contains_native_extensions": False,
            "contains_whiteouts": False,
        },
        "layout": str(output_layout),
        "layout_copy": "hardlinked immutable base blobs plus new candidate blobs",
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
