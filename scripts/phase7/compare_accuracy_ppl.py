#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_BENCHMARK_COUNTS = {
    "GSM8K": 1319,
    "IFEval": 541,
    "LiveCodeBench v6": 175,
    "MultiPL-E": 325,
}
EXPECTED_ACCURACY_ENVIRONMENT = {
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
EXPECTED_PPL_IDENTITY_SHA256 = (
    "fba1421ed512dd8551195e0856db69b9fbed75b71cffeddddb223e8ffbeaee44"
)
EXPECTED_PPL_MANIFEST_SHA256 = (
    "56e1caa23dd79555caec2fa8e8586bfc2a9622e26d7bf630764511ac5689423e"
)
EXPECTED_PPL_RUNNER_SHA256 = (
    "eec6b1a4be99a068f80b2e9c0d684392f1bf1a669cae4fbc881581d6baa25668"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_scoped_artifact_path(path: Path, project_root: Path) -> bool:
    roots = (project_root / "artifacts", Path("/dev/shm"))
    return any(path.is_relative_to(root) for root in roots)


def read_key_value_file(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            result[name] = value
    return result


def require_sha256(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"{label} changed: {actual} != {expected}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["evaluator_status"] == "scored"]
    invalid_scores = [
        row["id"]
        for row in scored
        if not isinstance(row.get("score"), (int, float))
        or not math.isfinite(row["score"])
        or row["score"] not in (0.0, 1.0)
    ]
    if invalid_scores:
        raise ValueError(f"invalid scored values: {invalid_scores[:20]}")
    correct = sum(row.get("score") == 1.0 for row in scored)
    return {
        "total": len(rows),
        "scored": len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "status_counts": dict(
            sorted(Counter(row["evaluator_status"] for row in rows).items())
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oscar-predictions", type=Path, required=True)
    parser.add_argument("--oscar-ppl-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_accuracy_evidence(predictions: Path) -> dict[str, Any]:
    output_dir = predictions.parent
    validation = read_json(output_dir / "validation.json")
    if (
        validation.get("status") != "passed"
        or validation.get("total") != 2360
        or validation.get("scored") != 2360
        or validation.get("predictions_rows") != 2360
    ):
        raise SystemExit(f"invalid accuracy validation: {validation}")
    require_sha256(
        predictions,
        validation["predictions_sha256"],
        "OSCAR predictions",
    )
    for filename, field in (
        ("summary.json", "summary_sha256"),
        ("runner_command.txt", "runner_command_sha256"),
        ("runner_environment.txt", "runner_environment_sha256"),
    ):
        require_sha256(output_dir / filename, validation[field], filename)
    environment = read_key_value_file(output_dir / "runner_environment.txt")
    for name, expected in EXPECTED_ACCURACY_ENVIRONMENT.items():
        if environment.get(name) != expected:
            raise SystemExit(
                f"accuracy environment mismatch for {name}: "
                f"{environment.get(name)!r} != {expected!r}"
            )
    runtime_config = output_dir / "runtime_suite/eval_config.json"
    require_sha256(
        runtime_config,
        environment["runtime_eval_config_sha256"],
        "runtime accuracy config",
    )
    return validation


def validate_ppl_evidence(summary_path: Path) -> dict[str, Any]:
    output_dir = summary_path.parent
    validation = read_json(output_dir / "validation.json")
    if (
        validation.get("status") != "passed"
        or validation.get("total") != 1
        or validation.get("scored") != 1
    ):
        raise SystemExit(f"invalid PPL validation: {validation}")
    require_sha256(summary_path, validation["summary_sha256"], "OSCAR PPL summary")
    expected = {
        "frozen_evaluator_identity_sha256": EXPECTED_PPL_IDENTITY_SHA256,
        "frozen_evaluator_manifest_sha256": EXPECTED_PPL_MANIFEST_SHA256,
        "frozen_ppl_runner_sha256": EXPECTED_PPL_RUNNER_SHA256,
    }
    for name, value in expected.items():
        if validation.get(name) != value:
            raise SystemExit(
                f"PPL validation mismatch for {name}: "
                f"{validation.get(name)!r} != {value!r}"
            )
    for filename, field in (
        ("runner_command.txt", "runner_command_sha256"),
        ("runtime_environment.txt", "runtime_environment_sha256"),
    ):
        require_sha256(output_dir / filename, validation[field], filename)
    return validation


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    manifest_path = args.manifest.resolve()
    expected_manifest_path = project_root / "configs/phase7/oscar_evaluation.json"
    if manifest_path != expected_manifest_path:
        raise SystemExit(f"unexpected Stage 7 manifest: {manifest_path}")
    output_dir = args.output_dir.resolve()
    if not is_scoped_artifact_path(output_dir, project_root):
        raise SystemExit(
            f"output must be under project artifacts/ or /dev/shm/: {output_dir}"
        )
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    oscar_predictions_path = args.oscar_predictions.resolve()
    oscar_ppl_summary_path = args.oscar_ppl_summary.resolve()
    for label, path in (
        ("OSCAR predictions", oscar_predictions_path),
        ("OSCAR PPL summary", oscar_ppl_summary_path),
    ):
        if not is_scoped_artifact_path(path, project_root):
            raise SystemExit(
                f"{label} must be under project artifacts/ or /dev/shm/: {path}"
            )

    manifest = read_json(manifest_path)
    baseline_path = project_root / manifest["baseline"]["accuracy"]["predictions"]
    expected_baseline_sha = manifest["baseline"]["accuracy"]["predictions_sha256"]
    require_sha256(baseline_path, expected_baseline_sha, "baseline predictions")
    baseline_accuracy_summary_path = (
        project_root / manifest["baseline"]["accuracy"]["summary"]
    )
    for section, field in (
        ("summary", "summary_sha256"),
        ("summary_by_benchmark", "summary_by_benchmark_sha256"),
    ):
        require_sha256(
            project_root / manifest["baseline"]["accuracy"][section],
            manifest["baseline"]["accuracy"][field],
            f"baseline accuracy {section}",
        )
    baseline_accuracy_summary = read_json(baseline_accuracy_summary_path)
    baseline_ppl_summary_path = project_root / manifest["baseline"]["ppl"]["summary"]
    require_sha256(
        baseline_ppl_summary_path,
        manifest["baseline"]["ppl"]["summary_sha256"],
        "baseline PPL summary",
    )
    baseline_ppl_summary = read_json(baseline_ppl_summary_path)
    if baseline_ppl_summary["total"] != 1 or baseline_ppl_summary["scored"] != 1:
        raise SystemExit("baseline PPL summary is incomplete")
    baseline_ppl_result = baseline_ppl_summary["results"][0]
    if (
        baseline_ppl_result["evaluator_status"] != "scored"
        or baseline_ppl_result["perplexity"]
        != manifest["baseline"]["ppl"]["perplexity"]
    ):
        raise SystemExit("baseline PPL value does not match the frozen summary")

    accuracy_validation = validate_accuracy_evidence(oscar_predictions_path)
    ppl_validation = validate_ppl_evidence(oscar_ppl_summary_path)
    baseline_rows = read_jsonl(baseline_path)
    oscar_rows = read_jsonl(oscar_predictions_path)
    if len(baseline_rows) != 2360 or len(oscar_rows) != 2360:
        raise SystemExit("accuracy inputs must contain exactly 2360 rows")
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    oscar_by_id = {row["id"]: row for row in oscar_rows}
    if len(baseline_by_id) != 2360 or set(baseline_by_id) != set(oscar_by_id):
        raise SystemExit("baseline and OSCAR prediction IDs differ")
    for label, rows in (("baseline", baseline_rows), ("OSCAR", oscar_rows)):
        counts = dict(Counter(row["benchmark"] for row in rows))
        if counts != EXPECTED_BENCHMARK_COUNTS:
            raise SystemExit(f"{label} benchmark counts changed: {counts}")

    sample_diff = []
    categories = Counter()
    by_benchmark: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"baseline": [], "oscar": []}
    )
    for baseline in baseline_rows:
        oscar = oscar_by_id[baseline["id"]]
        if baseline["prompt_hash"] != oscar["prompt_hash"]:
            raise SystemExit(f"prompt hash mismatch: {baseline['id']}")
        benchmark = baseline["benchmark"]
        if oscar["benchmark"] != benchmark:
            raise SystemExit(f"benchmark mismatch: {baseline['id']}")
        if oscar.get("task_type") != baseline.get("task_type"):
            raise SystemExit(f"task type mismatch: {baseline['id']}")
        if oscar.get("gold") != baseline.get("gold"):
            raise SystemExit(f"gold mismatch: {baseline['id']}")
        by_benchmark[benchmark]["baseline"].append(baseline)
        by_benchmark[benchmark]["oscar"].append(oscar)
        baseline_correct = (
            baseline["evaluator_status"] == "scored" and baseline.get("score") == 1.0
        )
        oscar_correct = (
            oscar["evaluator_status"] == "scored" and oscar.get("score") == 1.0
        )
        if baseline_correct and oscar_correct:
            category = "共同正确"
        elif baseline_correct:
            category = "仅 baseline 正确"
        elif oscar_correct:
            category = "仅 OSCAR 正确"
        else:
            category = "共同错误"
        categories[category] += 1
        sample_diff.append(
            {
                "id": baseline["id"],
                "benchmark": benchmark,
                "category": category,
                "baseline_status": baseline["evaluator_status"],
                "baseline_score": baseline.get("score"),
                "oscar_status": oscar["evaluator_status"],
                "oscar_score": oscar.get("score"),
            }
        )

    baseline_overall = summarize(baseline_rows)
    oscar_overall = summarize(oscar_rows)
    if (
        baseline_overall["total"] != 2360
        or baseline_overall["scored"] != 2360
        or baseline_overall["status_counts"] != {"scored": 2360}
        or baseline_overall["accuracy"] != baseline_accuracy_summary["accuracy"]
    ):
        raise SystemExit("baseline predictions do not match the frozen summary")
    if oscar_overall["accuracy"] != accuracy_validation["accuracy"]:
        raise SystemExit("OSCAR accuracy does not match validation.json")
    overall_delta = oscar_overall["accuracy"] - baseline_overall["accuracy"]
    thresholds = manifest["thresholds"]
    overall_passed = (
        oscar_overall["total"] == thresholds["total"]
        and oscar_overall["scored"] == thresholds["total"]
        and oscar_overall["status_counts"] == {"scored": thresholds["total"]}
        and overall_delta >= -thresholds["overall_accuracy_max_drop"]
    )

    benchmark_results = {}
    benchmark_passed = True
    for benchmark in sorted(by_benchmark):
        baseline_summary = summarize(by_benchmark[benchmark]["baseline"])
        oscar_summary = summarize(by_benchmark[benchmark]["oscar"])
        delta = oscar_summary["accuracy"] - baseline_summary["accuracy"]
        passed = (
            oscar_summary["scored"] == oscar_summary["total"]
            and oscar_summary["status_counts"] == {"scored": oscar_summary["total"]}
            and delta >= -thresholds["per_benchmark_accuracy_max_drop"]
        )
        benchmark_passed &= passed
        benchmark_results[benchmark] = {
            "baseline": baseline_summary,
            "oscar": oscar_summary,
            "delta": delta,
            "threshold": -thresholds["per_benchmark_accuracy_max_drop"],
            "passed": passed,
        }

    ppl_summary = read_json(oscar_ppl_summary_path)
    if ppl_summary["total"] != 1 or ppl_summary["scored"] != 1:
        raise SystemExit("OSCAR PPL result is incomplete")
    ppl_result = ppl_summary["results"][0]
    if ppl_result["evaluator_status"] != "scored":
        raise SystemExit("OSCAR PPL result is not scored")
    baseline_ppl = manifest["baseline"]["ppl"]["perplexity"]
    oscar_ppl = ppl_result["perplexity"]
    for label, value in (("baseline", baseline_ppl), ("OSCAR", oscar_ppl)):
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise SystemExit(f"invalid {label} PPL: {value}")
    for name in ("perplexity", "mean_nll", "evaluated_tokens", "windows"):
        if ppl_result[name] != ppl_validation[name]:
            raise SystemExit(f"OSCAR PPL {name} does not match validation.json")
    for name in ("evaluated_tokens", "windows"):
        if ppl_result[name] != baseline_ppl_result[name]:
            raise SystemExit(f"OSCAR PPL {name} differs from baseline")
    ppl_relative_increase = oscar_ppl / baseline_ppl - 1
    ppl_passed = ppl_relative_increase <= thresholds["ppl_max_relative_increase"]

    request_failures = sum(
        row["evaluator_status"] == "request_failed" for row in oscar_rows
    )
    request_failure_passed = request_failures == thresholds["request_failures"]
    passed = (
        overall_passed and benchmark_passed and ppl_passed and request_failure_passed
    )
    comparison = {
        "format_version": 1,
        "status": "passed" if passed else "failed",
        "provenance": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "accuracy_validation_sha256": sha256_file(
                oscar_predictions_path.parent / "validation.json"
            ),
            "ppl_validation_sha256": sha256_file(
                oscar_ppl_summary_path.parent / "validation.json"
            ),
            "oscar_ppl_summary_sha256": sha256_file(oscar_ppl_summary_path),
            "baseline_ppl_summary_sha256": sha256_file(baseline_ppl_summary_path),
        },
        "accuracy": {
            "baseline": baseline_overall,
            "oscar": oscar_overall,
            "delta": overall_delta,
            "threshold": -thresholds["overall_accuracy_max_drop"],
            "passed": overall_passed,
            "categories": {
                name: categories[name]
                for name in (
                    "共同正确",
                    "仅 baseline 正确",
                    "仅 OSCAR 正确",
                    "共同错误",
                )
            },
            "by_benchmark": benchmark_results,
            "request_failures": request_failures,
            "request_failure_passed": request_failure_passed,
            "baseline_predictions_sha256": sha256_file(baseline_path),
            "oscar_predictions_sha256": sha256_file(oscar_predictions_path),
        },
        "ppl": {
            "baseline": baseline_ppl,
            "oscar": oscar_ppl,
            "relative_increase": ppl_relative_increase,
            "threshold": thresholds["ppl_max_relative_increase"],
            "passed": ppl_passed,
            "evaluated_tokens": ppl_result["evaluated_tokens"],
            "windows": ppl_result["windows"],
            "mean_nll": ppl_result["mean_nll"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "sample_diff.jsonl").open("w", encoding="utf-8") as output:
        for row in sample_diff:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
