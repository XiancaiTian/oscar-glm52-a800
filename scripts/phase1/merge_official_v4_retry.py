#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path


ACCURACY_BENCHMARKS = {
    "GSM8K",
    "IFEval",
    "LiveCodeBench v6",
    "MultiPL-E",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--retry-dir", type=Path, required=True)
    parser.add_argument("--merged-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.merged_dir.exists():
        raise SystemExit(f"merged directory already exists: {args.merged_dir}")

    spec = importlib.util.spec_from_file_location("official_v4_runner", args.runner)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)

    full_manifest = read_jsonl(args.manifest)
    manifest = [row for row in full_manifest if row["benchmark"] in ACCURACY_BENCHMARKS]
    excluded = [
        (row["id"], row["benchmark"])
        for row in full_manifest
        if row["benchmark"] not in ACCURACY_BENCHMARKS
    ]
    if len(manifest) != 2360:
        raise SystemExit(
            f"accuracy manifest does not contain 2360 rows: {len(manifest)}"
        )
    if excluded != [("wikitext2_perplexity:test", "WikiText-2")]:
        raise SystemExit(f"unexpected non-accuracy manifest rows: {excluded}")

    source_rows = read_jsonl(args.source_dir / "predictions.jsonl")
    retry_rows = read_jsonl(args.retry_dir / "predictions.jsonl")
    source_by_id = {row["id"]: row for row in source_rows}
    retry_by_id = {row["id"]: row for row in retry_rows}
    manifest_ids = [row["id"] for row in manifest]
    if len(source_rows) != 2360 or len(source_by_id) != 2360:
        raise SystemExit("source predictions do not contain 2360 unique rows")
    if set(source_by_id) != set(manifest_ids):
        raise SystemExit("source prediction IDs do not match accuracy manifest")
    if len(retry_rows) != 7 or len(retry_by_id) != 7:
        raise SystemExit("retry predictions do not contain 7 unique rows")

    retry_summary = json.loads(
        (args.retry_dir / "summary.json").read_text(encoding="utf-8")
    )
    if retry_summary["total"] != 7 or retry_summary["scored"] != 7:
        raise SystemExit(f"retry is incomplete: {retry_summary}")
    if retry_summary["status_counts"] != {"scored": 7}:
        raise SystemExit(f"retry has non-scored rows: {retry_summary['status_counts']}")

    source_failed_ids = {
        row["id"] for row in source_rows if row["evaluator_status"] != "scored"
    }
    if set(retry_by_id) != source_failed_ids:
        raise SystemExit("retry IDs do not exactly match source failed IDs")

    merged_rows = [
        retry_by_id.get(sample["id"], source_by_id[sample["id"]]) for sample in manifest
    ]
    for sample, row in zip(manifest, merged_rows, strict=True):
        prompt_hash = hashlib.sha256(sample["prompt"].encode("utf-8")).hexdigest()
        if row["prompt_hash"] != prompt_hash:
            raise SystemExit(f"merged prompt hash mismatch for {row['id']}")
        if row["evaluator_status"] != "scored" or row["score"] is None:
            raise SystemExit(f"merged row is not scored: {row['id']}")

    args.merged_dir.mkdir(parents=True)
    runner.write_jsonl(args.merged_dir / "predictions.jsonl", merged_rows)
    by_benchmark = runner.summarize(merged_rows, "benchmark")
    by_task_type = runner.summarize(merged_rows, "task_type")
    runner.write_json(args.merged_dir / "summary_by_benchmark.json", by_benchmark)
    runner.write_json(args.merged_dir / "summary_by_task_type.json", by_task_type)
    runner.write_csv(args.merged_dir / "summary_by_benchmark.csv", by_benchmark)
    runner.write_csv(args.merged_dir / "summary_by_task_type.csv", by_task_type)
    failed = [
        row
        for row in merged_rows
        if row["evaluator_status"] != "scored" or row.get("score") != 1.0
    ]
    runner.write_jsonl(args.merged_dir / "failed_cases.jsonl", failed)

    source_summary = json.loads(
        (args.source_dir / "summary.json").read_text(encoding="utf-8")
    )
    overall = {
        "total": 2360,
        "scored": 2360,
        "accuracy": sum(row["score"] for row in merged_rows) / 2360,
        "status_counts": dict(Counter(row["evaluator_status"] for row in merged_rows)),
        "started_at_unix": source_summary["started_at_unix"],
        "ended_at_unix": retry_summary["ended_at_unix"],
        "duration_seconds": (
            source_summary["duration_seconds"] + retry_summary["duration_seconds"]
        ),
    }
    runner.write_json(args.merged_dir / "summary.json", overall)

    provenance = {
        "merge_policy": "replace_only_source_request_failed_rows_by_exact_id",
        "manifest_policy": ("official_v4_accuracy_benchmarks_excluding_wikitext2_ppl"),
        "accuracy_benchmarks": sorted(ACCURACY_BENCHMARKS),
        "excluded_manifest_rows": excluded,
        "source_attempt": args.source_dir.name,
        "retry_attempt": args.retry_dir.name,
        "source_predictions_sha256": sha256(args.source_dir / "predictions.jsonl"),
        "retry_predictions_sha256": sha256(args.retry_dir / "predictions.jsonl"),
        "replaced_ids": sorted(source_failed_ids),
        "composite_duration_policy": ("sum_of_source_and_retry_runner_durations"),
    }
    runner.write_json(args.merged_dir / "merge_provenance.json", provenance)

    validation = {
        "status": "passed",
        "total": 2360,
        "scored": 2360,
        "accuracy": overall["accuracy"],
        "status_counts": overall["status_counts"],
        "predictions_rows": 2360,
        "predictions_sha256": sha256(args.merged_dir / "predictions.jsonl"),
        "summary_sha256": sha256(args.merged_dir / "summary.json"),
        "retry_total": 7,
        "retry_scored": 7,
        "replaced_ids": sorted(source_failed_ids),
    }
    runner.write_json(args.merged_dir / "validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
