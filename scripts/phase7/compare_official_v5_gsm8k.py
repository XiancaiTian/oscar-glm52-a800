#!/usr/bin/env python3
"""Compare native and OSCAR official_v5 GSM8K results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require_safe_path(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    allowed = ((project_root / "artifacts").resolve(), Path("/dev/shm"))
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed):
        raise ValueError(f"path is outside project artifacts/ and /dev/shm: {resolved}")
    return resolved


def load_result(
    directory: Path,
    role: str,
    expected_total: int,
) -> dict[str, Any]:
    validation = read_json(directory / "validation.json")
    summary = read_json(directory / "summary.json")
    by_benchmark = read_json(directory / "summary_by_benchmark.json")
    predictions = read_jsonl(directory / "predictions.jsonl")
    runtime_manifest_path = directory.parents[1] / "runtime_manifest.json"
    runtime_manifest = read_json(runtime_manifest_path)

    expected_validation = {
        "status": "passed",
        "evaluation_role": role,
        "protocol_version": "official_v5",
        "scope": "current_stage_gsm8k",
        "total": expected_total,
        "scored": expected_total,
        "request_failures": 0,
        "reasoning_effort": "high",
    }
    for name, expected in expected_validation.items():
        if validation.get(name) != expected:
            raise ValueError(f"{role} validation mismatch: {name}")
    file_hashes = {
        "summary_sha256": directory / "summary.json",
        "summary_by_benchmark_sha256": directory / "summary_by_benchmark.json",
        "predictions_sha256": directory / "predictions.jsonl",
        "runner_command_sha256": directory / "runner_command.txt",
        "runner_environment_sha256": directory / "runner_environment.txt",
        "runtime_manifest_sha256": runtime_manifest_path,
        "runtime_suite_manifest_sha256": (
            directory / "runtime_suite" / "manifest.jsonl"
        ),
        "runtime_suite_eval_config_sha256": (
            directory / "runtime_suite" / "eval_config.json"
        ),
    }
    for name, path in file_hashes.items():
        if validation.get(name) != sha256_file(path):
            raise ValueError(f"{role} evidence hash mismatch: {name}")
    if (
        summary.get("valid") is not True
        or summary.get("protocol_version") != "official_v5"
        or summary.get("total") != expected_total
        or summary.get("scored") != expected_total
        or summary.get("status_counts") != {"scored": expected_total}
    ):
        raise ValueError(f"{role} summary is incomplete")
    if len(by_benchmark) != 1 or by_benchmark[0].get("benchmark") != "GSM8K":
        raise ValueError(f"{role} benchmark summary is not GSM8K-only")
    if len(predictions) != expected_total:
        raise ValueError(f"{role} prediction count is not {expected_total}")
    ids = [row["id"] for row in predictions]
    if len(set(ids)) != expected_total:
        raise ValueError(f"{role} prediction IDs are not unique")
    if any(
        row.get("benchmark") != "GSM8K"
        or row.get("evaluator_status") != "scored"
        or row.get("score") not in {0.0, 1.0}
        for row in predictions
    ):
        raise ValueError(f"{role} predictions contain invalid rows")
    accuracy = sum(row["score"] for row in predictions) / expected_total
    native_accuracy = summary["native_metrics"]["GSM8K"]["accuracy"]
    if abs(accuracy - native_accuracy) > 1e-15:
        raise ValueError(f"{role} accuracy does not match predictions")
    if abs(accuracy - validation["accuracy"]) > 1e-15:
        raise ValueError(f"{role} validation accuracy mismatch")
    fingerprints = {row.get("protocol_fingerprint") for row in predictions}
    if fingerprints != {summary["protocol_fingerprint"]}:
        raise ValueError(f"{role} protocol fingerprint mismatch")
    return {
        "validation": validation,
        "summary": summary,
        "predictions": predictions,
        "runtime_manifest": runtime_manifest,
        "accuracy": accuracy,
    }


def main_diff_is_allowed(paths: list[str]) -> bool:
    return all(
        path in {"progress.md", "findings.md", "task_plan.md"}
        or path.startswith("docs/")
        for path in paths
    )


def verify_main_commit_compatibility(
    project_root: Path,
    baseline_commit: str,
    candidate_commit: str,
) -> list[str]:
    if baseline_commit == candidate_commit:
        return []
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "merge-base",
            "--is-ancestor",
            baseline_commit,
            candidate_commit,
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("candidate main commit does not descend from baseline")
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            baseline_commit,
            candidate_commit,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    if not main_diff_is_allowed(changed):
        raise ValueError(f"runtime-affecting main commit differences: {changed}")
    return changed


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    max_drop: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for name in (
        "source_repository_commit",
        "model_filename_size_mtime_ns_manifest_sha256",
        "evaluation_protocol",
        "evaluation_scope",
        "evaluation_manifest_sha256",
    ):
        if baseline["runtime_manifest"].get(name) != candidate["runtime_manifest"].get(
            name
        ):
            raise ValueError(f"runtime provenance mismatch: {name}")
    if (
        baseline["summary"]["protocol_fingerprint"]
        != candidate["summary"]["protocol_fingerprint"]
    ):
        raise ValueError("baseline and candidate protocol fingerprints differ")

    baseline_by_id = {row["id"]: row for row in baseline["predictions"]}
    candidate_by_id = {row["id"]: row for row in candidate["predictions"]}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("baseline and candidate sample IDs differ")

    counts = {
        "both_correct": 0,
        "baseline_only_correct": 0,
        "candidate_only_correct": 0,
        "both_incorrect": 0,
    }
    differences = []
    for sample_id in sorted(baseline_by_id):
        native = baseline_by_id[sample_id]
        oscar = candidate_by_id[sample_id]
        for name in ("prompt_hash", "gold", "task_type"):
            if native.get(name) != oscar.get(name):
                raise ValueError(f"sample identity mismatch for {sample_id}: {name}")
        native_correct = native["score"] == 1.0
        oscar_correct = oscar["score"] == 1.0
        if native_correct and oscar_correct:
            category = "both_correct"
        elif native_correct:
            category = "baseline_only_correct"
        elif oscar_correct:
            category = "candidate_only_correct"
        else:
            category = "both_incorrect"
        counts[category] += 1
        if native_correct != oscar_correct:
            differences.append(
                {
                    "id": sample_id,
                    "category": category,
                    "baseline_score": native["score"],
                    "candidate_score": oscar["score"],
                    "baseline_extracted_answer": native.get("extracted_answer"),
                    "candidate_extracted_answer": oscar.get("extracted_answer"),
                }
            )

    drop = baseline["accuracy"] - candidate["accuracy"]
    result = {
        "format_version": 1,
        "status": "passed" if drop <= max_drop + 1e-15 else "failed",
        "protocol_version": "official_v5",
        "scope": "current_stage_gsm8k",
        "total": len(baseline_by_id),
        "baseline_accuracy": baseline["accuracy"],
        "candidate_accuracy": candidate["accuracy"],
        "candidate_delta": candidate["accuracy"] - baseline["accuracy"],
        "accuracy_drop": drop,
        "maximum_allowed_drop": max_drop,
        "request_failures": 0,
        "categories": counts,
        "sample_differences": len(differences),
        "protocol_fingerprint": baseline["summary"]["protocol_fingerprint"],
        "final_full_evaluation_still_required": True,
    }
    return result, differences


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    baseline_dir = require_safe_path(args.baseline_dir, project_root)
    candidate_dir = require_safe_path(args.candidate_dir, project_root)
    output_dir = require_safe_path(args.output_dir, project_root)
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    config = read_json(args.config)
    expected_total = config["selection"]["total"]
    baseline = load_result(baseline_dir, "native", expected_total)
    candidate = load_result(candidate_dir, "candidate", expected_total)
    baseline_main_commit = baseline["runtime_manifest"]["main_commit"]
    candidate_main_commit = candidate["runtime_manifest"]["main_commit"]
    allowed_main_changes = verify_main_commit_compatibility(
        project_root,
        baseline_main_commit,
        candidate_main_commit,
    )
    result, differences = compare(
        baseline,
        candidate,
        config["thresholds"]["gsm8k_accuracy_max_drop"],
    )
    result["baseline_main_commit"] = baseline_main_commit
    result["candidate_main_commit"] = candidate_main_commit
    result["allowed_documentation_changes"] = allowed_main_changes

    output_dir.mkdir(parents=True)
    diff_path = output_dir / "sample_diff.jsonl"
    with diff_path.open("w", encoding="utf-8") as handle:
        for row in differences:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    result["sample_diff_sha256"] = sha256_file(diff_path)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
