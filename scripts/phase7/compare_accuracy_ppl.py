#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["evaluator_status"] == "scored"]
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


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    manifest = read_json(args.manifest.resolve())
    baseline_path = project_root / manifest["baseline"]["accuracy"]["predictions"]
    expected_baseline_sha = manifest["baseline"]["accuracy"]["predictions_sha256"]
    if sha256_file(baseline_path) != expected_baseline_sha:
        raise SystemExit("baseline predictions changed")

    baseline_rows = read_jsonl(baseline_path)
    oscar_rows = read_jsonl(args.oscar_predictions.resolve())
    if len(baseline_rows) != 2360 or len(oscar_rows) != 2360:
        raise SystemExit("accuracy inputs must contain exactly 2360 rows")
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    oscar_by_id = {row["id"]: row for row in oscar_rows}
    if len(baseline_by_id) != 2360 or set(baseline_by_id) != set(oscar_by_id):
        raise SystemExit("baseline and OSCAR prediction IDs differ")

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

    ppl_summary = read_json(args.oscar_ppl_summary.resolve())
    if ppl_summary["total"] != 1 or ppl_summary["scored"] != 1:
        raise SystemExit("OSCAR PPL result is incomplete")
    ppl_result = ppl_summary["results"][0]
    if ppl_result["evaluator_status"] != "scored":
        raise SystemExit("OSCAR PPL result is not scored")
    baseline_ppl = manifest["baseline"]["ppl"]["perplexity"]
    oscar_ppl = ppl_result["perplexity"]
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
            "oscar_predictions_sha256": sha256_file(args.oscar_predictions.resolve()),
        },
        "ppl": {
            "baseline": baseline_ppl,
            "oscar": oscar_ppl,
            "relative_increase": ppl_relative_increase,
            "threshold": thresholds["ppl_max_relative_increase"],
            "passed": ppl_passed,
            "evaluated_tokens": ppl_result["evaluated_tokens"],
            "windows": ppl_result["windows"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "sample_diff.jsonl").open("w", encoding="utf-8") as output:
        for row in sample_diff:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
