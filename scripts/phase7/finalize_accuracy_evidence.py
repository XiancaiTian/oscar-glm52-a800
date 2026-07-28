#!/usr/bin/env python3
"""Finalize an older in-flight accuracy result without mutating its evidence."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


EXPECTED_ENVIRONMENT = {
    "runner_sha256": "fc374ff4c4715e37d515d37aa794b3e649dc1710034d3355c69c21efa1a8aeff",
    "suite_manifest_sha256": (
        "4aec8ee85bee5eb73ce99c2009fcaedc79804bde1433f855fb77276ffccacfa5"
    ),
    "source_eval_config_sha256": (
        "660a79fb1e2d5fde60816572b3c2560dc5bc54e4461624382df8b6c19d06a270"
    ),
    "runtime_code_timeout_seconds": "3600",
    "runtime_instruction_following_timeout_seconds": "1800",
    "runtime_math_timeout_seconds": "1800",
}
EXPECTED_INFLIGHT_FILE_SHA256 = {
    "runner_command.txt": (
        "2aa98cf3ff1ac6d97307c9076b8e708c232b5dc5400a98c01bdb6c516ba6f522"
    ),
    "runner_environment.txt": (
        "c42bbfbd9dd7f834ac1fe8d5462b23d4f52085f2e7d79adab9386346792fc2ee"
    ),
    "runtime_suite/eval_config.json": (
        "61845910723026eacd7273628cfdc12200425284b2bfd3f0fe71b35621766b8b"
    ),
}
REQUIRED_FILES = (
    "predictions.jsonl",
    "summary.json",
    "runner_command.txt",
    "runner_environment.txt",
    "runtime_suite/eval_config.json",
    "validation.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_key_value_file(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            result[name] = value
    return result


def is_scoped_artifact_path(path: Path, project_root: Path) -> bool:
    roots = (project_root / "artifacts", Path("/dev/shm"))
    return any(path.is_relative_to(root) for root in roots)


def clone_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
            raise
        shutil.copy2(source, target)


def finalize(
    source_dir: Path,
    output_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    for label, path in (("source", source_dir), ("output", output_dir)):
        if not is_scoped_artifact_path(path, project_root):
            raise ValueError(f"{label} must be under project artifacts/ or /dev/shm/")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if output_dir.is_relative_to(source_dir) or source_dir.is_relative_to(output_dir):
        raise ValueError("source and output directories must not contain each other")

    for relative in REQUIRED_FILES:
        path = source_dir / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required regular file is missing: {path}")
    for relative, expected in EXPECTED_INFLIGHT_FILE_SHA256.items():
        if sha256_file(source_dir / relative) != expected:
            raise ValueError(f"in-flight evidence changed: {relative}")

    runner_validation = read_json(source_dir / "validation.json")
    summary = read_json(source_dir / "summary.json")
    predictions = source_dir / "predictions.jsonl"
    with predictions.open(encoding="utf-8") as source:
        prediction_rows = sum(1 for line in source if line)
    expected_result = {
        "status": "passed",
        "total": 2360,
        "scored": 2360,
        "predictions_rows": 2360,
    }
    for name, value in expected_result.items():
        if runner_validation.get(name) != value:
            raise ValueError(f"invalid runner validation for {name}")
    if (
        summary.get("total") != 2360
        or summary.get("scored") != 2360
        or summary.get("accuracy") != runner_validation.get("accuracy")
        or prediction_rows != 2360
    ):
        raise ValueError("summary/prediction completeness mismatch")
    if sha256_file(predictions) != runner_validation.get("predictions_sha256"):
        raise ValueError("runner prediction hash mismatch")

    environment = read_key_value_file(source_dir / "runner_environment.txt")
    for name, expected in EXPECTED_ENVIRONMENT.items():
        if environment.get(name) != expected:
            raise ValueError(f"runner environment mismatch for {name}")
    runtime_config = source_dir / "runtime_suite/eval_config.json"
    if sha256_file(runtime_config) != environment.get("runtime_eval_config_sha256"):
        raise ValueError("runtime evaluation configuration hash mismatch")

    files = [path for path in source_dir.rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in files):
        raise ValueError("source evidence contains a symlink")
    output_dir.mkdir(parents=True)
    for path in files:
        relative = path.relative_to(source_dir)
        if relative == Path("validation.json"):
            relative = Path("runner_validation.json")
        clone_file(path, output_dir / relative)

    result = {
        **runner_validation,
        "finalized": True,
        "source_output_dir": str(source_dir),
        "runner_validation_sha256": sha256_file(output_dir / "runner_validation.json"),
        "summary_sha256": sha256_file(output_dir / "summary.json"),
        "runner_command_sha256": sha256_file(output_dir / "runner_command.txt"),
        "runner_environment_sha256": sha256_file(output_dir / "runner_environment.txt"),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    result = finalize(args.source_output_dir, args.output_dir, project_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
