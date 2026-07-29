#!/usr/bin/env python3
"""Compare the four Stage 7 fast-screening pilot cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_key_value_file(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


def require_safe_path(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    roots = ((project_root / "artifacts").resolve(), Path("/dev/shm"))
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"path is outside project artifacts/ and /dev/shm: {resolved}")
    return resolved


def load_result(
    directory: Path,
    *,
    role: str,
    concurrency: int,
    expected_total: int,
) -> dict[str, Any]:
    validation = read_json(directory / "validation.json")
    summary = read_json(directory / "summary.json")
    predictions = read_jsonl(directory / "predictions.jsonl")
    suite_identity = read_json(directory / "runtime_suite/fast_suite_identity.json")
    runtime_manifest_path = directory.parents[1] / "runtime_manifest.json"
    runtime_manifest = read_json(runtime_manifest_path)
    runner_environment = read_key_value_file(directory / "runner_environment.txt")
    expected = {
        "status": "passed",
        "evaluation_role": role,
        "protocol_version": "official_v5",
        "screening_protocol": "official_v5_fast_screen",
        "scope": f"stage7_fast_gsm8k_{expected_total}",
        "total": expected_total,
        "scored": expected_total,
        "request_failures": 0,
        "concurrency": concurrency,
        "server_max_model_len": 8192,
        "reasoning_effort": "high",
        "final_full_evaluation_still_required": True,
    }
    for name, value in expected.items():
        if validation.get(name) != value:
            raise ValueError(f"{role} c{concurrency} validation mismatch: {name}")
    evidence = {
        "summary_sha256": directory / "summary.json",
        "summary_by_benchmark_sha256": directory / "summary_by_benchmark.json",
        "summary_by_task_type_sha256": directory / "summary_by_task_type.json",
        "predictions_sha256": directory / "predictions.jsonl",
        "failed_cases_sha256": directory / "failed_cases.jsonl",
        "fast_runner_state_sha256": directory / "fast_runner_state.json",
        "runner_command_sha256": directory / "runner_command.txt",
        "runner_environment_sha256": directory / "runner_environment.txt",
        "runtime_manifest_sha256": runtime_manifest_path,
        "runtime_suite_manifest_sha256": (
            directory / "runtime_suite" / "manifest.jsonl"
        ),
        "runtime_suite_eval_config_sha256": (
            directory / "runtime_suite" / "eval_config.json"
        ),
        "fast_suite_identity_sha256": (
            directory / "runtime_suite" / "fast_suite_identity.json"
        ),
    }
    for name, path in evidence.items():
        if validation.get(name) != sha256_file(path):
            raise ValueError(f"{role} c{concurrency} evidence hash mismatch: {name}")
    if (
        summary.get("valid") is not True
        or summary.get("total") != expected_total
        or summary.get("scored") != expected_total
        or summary.get("status_counts") != {"scored": expected_total}
        or len(predictions) != expected_total
    ):
        raise ValueError(f"{role} c{concurrency} result is incomplete")
    ids = [row["id"] for row in predictions]
    if len(set(ids)) != expected_total:
        raise ValueError(f"{role} c{concurrency} IDs are not unique")
    if any(
        row.get("evaluator_status") != "scored" or row.get("score") not in {0.0, 1.0}
        for row in predictions
    ):
        raise ValueError(f"{role} c{concurrency} contains invalid predictions")
    accuracy = sum(row["score"] for row in predictions) / expected_total
    if abs(accuracy - validation["accuracy"]) > 1e-15:
        raise ValueError(f"{role} c{concurrency} validation accuracy mismatch")
    fingerprints = {row.get("protocol_fingerprint") for row in predictions}
    if fingerprints != {summary["protocol_fingerprint"]}:
        raise ValueError(f"{role} c{concurrency} protocol fingerprint mismatch")
    if (
        runner_environment.get("runtime_suite_manifest_sha256")
        != suite_identity["runtime_manifest_sha256"]
        or runner_environment.get("runtime_suite_eval_config_sha256")
        != suite_identity["runtime_eval_config_sha256"]
        or runner_environment.get("selection_identity_sha256")
        != validation["fast_suite_identity_sha256"]
        or runner_environment.get("runtime_manifest_sha256")
        != validation["runtime_manifest_sha256"]
    ):
        raise ValueError(f"{role} c{concurrency} runner environment mismatch")
    return {
        "validation": validation,
        "summary": summary,
        "predictions": predictions,
        "suite_identity": suite_identity,
        "runtime_manifest": runtime_manifest,
        "runner_environment": runner_environment,
    }


def pair_comparison(
    native: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_accuracy_drop: float,
) -> dict[str, Any]:
    native_by_id = {row["id"]: row for row in native["predictions"]}
    candidate_by_id = {row["id"]: row for row in candidate["predictions"]}
    if native_by_id.keys() != candidate_by_id.keys():
        raise ValueError("native and candidate sample IDs differ")
    categories = {
        "both_correct": 0,
        "native_only_correct": 0,
        "candidate_only_correct": 0,
        "both_incorrect": 0,
    }
    for sample_id in native_by_id:
        baseline = native_by_id[sample_id]
        oscar = candidate_by_id[sample_id]
        for name in ("prompt_hash", "gold", "task_type"):
            if baseline.get(name) != oscar.get(name):
                raise ValueError(f"sample identity mismatch: {sample_id} {name}")
        baseline_correct = baseline["score"] == 1.0
        oscar_correct = oscar["score"] == 1.0
        if baseline_correct and oscar_correct:
            categories["both_correct"] += 1
        elif baseline_correct:
            categories["native_only_correct"] += 1
        elif oscar_correct:
            categories["candidate_only_correct"] += 1
        else:
            categories["both_incorrect"] += 1
    native_accuracy = native["validation"]["accuracy"]
    candidate_accuracy = candidate["validation"]["accuracy"]
    accuracy_drop = native_accuracy - candidate_accuracy
    return {
        "status": "passed" if accuracy_drop <= max_accuracy_drop else "failed",
        "native_accuracy": native_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_drop": accuracy_drop,
        "native_truncation_rate": native["validation"]["truncation_rate"],
        "candidate_truncation_rate": candidate["validation"]["truncation_rate"],
        "native_completion_tokens_mean": native["validation"]["completion_tokens_mean"],
        "candidate_completion_tokens_mean": candidate["validation"][
            "completion_tokens_mean"
        ],
        "native_requests_per_hour": native["validation"]["requests_per_hour"],
        "candidate_requests_per_hour": candidate["validation"]["requests_per_hour"],
        "combined_duration_seconds": (
            native["summary"]["duration_seconds"]
            + candidate["summary"]["duration_seconds"]
        ),
        "categories": categories,
    }


def concurrency_differences(
    lower: dict[str, Any],
    higher: dict[str, Any],
) -> int:
    lower_by_id = {row["id"]: row for row in lower["predictions"]}
    higher_by_id = {row["id"]: row for row in higher["predictions"]}
    if lower_by_id.keys() != higher_by_id.keys():
        raise ValueError("concurrency cells have different sample IDs")
    return sum(
        any(
            lower_by_id[sample_id].get(name) != higher_by_id[sample_id].get(name)
            for name in ("score", "extracted_answer", "truncated")
        )
        for sample_id in lower_by_id
    )


def compare_matrix(
    cells: dict[tuple[str, int], dict[str, Any]],
    *,
    max_accuracy_drop: float,
) -> dict[str, Any]:
    fingerprints = {cell["summary"]["protocol_fingerprint"] for cell in cells.values()}
    if len(fingerprints) != 1:
        raise ValueError("fast pilot protocol fingerprints differ")
    for name in (
        "source_repository_commit",
        "runtime_source_commit",
        "model_filename_size_mtime_ns_manifest_sha256",
        "evaluation_protocol",
        "evaluation_scope",
        "evaluation_manifest_sha256",
    ):
        values = {cell["runtime_manifest"].get(name) for cell in cells.values()}
        if len(values) != 1:
            raise ValueError(f"fast pilot runtime provenance differs: {name}")
    for name in (
        "runtime_manifest_sha256",
        "runtime_eval_config_sha256",
        "selected_ids_sha256",
        "sample_count",
        "selection_seed",
    ):
        values = {cell["suite_identity"].get(name) for cell in cells.values()}
        if len(values) != 1:
            raise ValueError(f"fast pilot suite identity differs: {name}")
    for name in ("frozen_runner_sha256", "fast_runner_sha256"):
        values = {cell["runner_environment"].get(name) for cell in cells.values()}
        if len(values) != 1:
            raise ValueError(f"fast pilot runner identity differs: {name}")
    comparisons = {
        str(concurrency): pair_comparison(
            cells[("native", concurrency)],
            cells[("candidate", concurrency)],
            max_accuracy_drop=max_accuracy_drop,
        )
        for concurrency in (8, 16)
    }
    native_differences = concurrency_differences(
        cells[("native", 8)],
        cells[("native", 16)],
    )
    candidate_differences = concurrency_differences(
        cells[("candidate", 8)],
        cells[("candidate", 16)],
    )
    selected = 8
    reason = "concurrency 16 did not reduce paired duration by at least 5%"
    if native_differences or candidate_differences:
        reason = (
            "concurrency changed scored answers or truncation; choose conservative 8"
        )
    elif (
        comparisons["16"]["status"] == "passed"
        and comparisons["16"]["combined_duration_seconds"]
        < comparisons["8"]["combined_duration_seconds"] * 0.95
    ):
        selected = 16
        reason = "concurrency 16 reduced paired duration by at least 5%"
    return {
        "format_version": 1,
        "status": (
            "passed"
            if all(item["status"] == "passed" for item in comparisons.values())
            else "failed"
        ),
        "protocol": "official_v5_fast_screen",
        "sample_total": len(cells[("native", 8)]["predictions"]),
        "protocol_fingerprint": next(iter(fingerprints)),
        "comparisons": comparisons,
        "cross_concurrency_differences": {
            "native": native_differences,
            "candidate": candidate_differences,
        },
        "selected_concurrency": selected,
        "selection_reason": reason,
        "final_full_evaluation_still_required": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-c8", type=Path, required=True)
    parser.add_argument("--native-c16", type=Path, required=True)
    parser.add_argument("--candidate-c8", type=Path, required=True)
    parser.add_argument("--candidate-c16", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=256)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    cells = {
        ("native", 8): load_result(
            require_safe_path(args.native_c8, project_root),
            role="native",
            concurrency=8,
            expected_total=args.expected_total,
        ),
        ("native", 16): load_result(
            require_safe_path(args.native_c16, project_root),
            role="native",
            concurrency=16,
            expected_total=args.expected_total,
        ),
        ("candidate", 8): load_result(
            require_safe_path(args.candidate_c8, project_root),
            role="candidate",
            concurrency=8,
            expected_total=args.expected_total,
        ),
        ("candidate", 16): load_result(
            require_safe_path(args.candidate_c16, project_root),
            role="candidate",
            concurrency=16,
            expected_total=args.expected_total,
        ),
    }
    result = compare_matrix(cells, max_accuracy_drop=args.max_accuracy_drop)
    output = require_safe_path(args.output, project_root)
    if output.exists():
        raise SystemExit(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
