#!/usr/bin/env python3
"""Build reproducible CPU-only paired accuracy attribution evidence."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUNS = {
    "current_ea8_default": {
        "predictions": ROOT / "artifacts/phase9-control/20260803T1245Z_ea8_fast256_launch_v5/final/predictions.jsonl",
        "source_commit": "ea8ae6b7758ae2b4db7cae44d638ae5de80148ac",
        "index_topk": 768,
        "decode_backend": "persistent",
        "prefill_sort_indices": False,
    },
    "c349_k1024_legacy": {
        "root": ROOT / "artifacts/phase9-control/20260801T2115Z_stage9_candidate_c349e32e9_topk1024_legacy_32k_b1_v1/formal_32k_b1_topk1024_legacy_accuracy_smoke_pass_v1",
        "source_commit": "c349e32e929279e0c7e20676d48d39cc4b5864b3",
        "index_topk": 1024,
        "decode_backend": "legacy",
        "prefill_sort_indices": True,
    },
    "c349_k1536_legacy": {
        "root": ROOT / "artifacts/phase9-control/20260801T1032Z_stage9_candidate_c349e32e9_32k_b1_v1/formal_32k_b1_topk1536_legacy_accuracy_smoke_pass_v1",
        "source_commit": "c349e32e929279e0c7e20676d48d39cc4b5864b3",
        "index_topk": 1536,
        "decode_backend": "legacy",
        "prefill_sort_indices": True,
    },
    "c349_k768_legacy": {
        "root": ROOT / "artifacts/phase9-control/20260802T0340Z_stage9_baseline_c349_source_32k_b1_v1/formal_32k_b1_topk768_legacy_accuracy_smoke_failed_v1",
        "source_commit": "c349e32e929279e0c7e20676d48d39cc4b5864b3",
        "index_topk": 768,
        "decode_backend": "legacy",
        "prefill_sort_indices": True,
    },
}
CURRENT_RUNTIME_ENV = Path("/dev/shm/oscar-glm-official-v5-ea8/phase7/20260803T1245Z_candidate_ea8_splitk_stride_fast256_c16_v5/runtime_environment.txt")
CURRENT_SERVER_ARGS = Path("/dev/shm/oscar-glm-official-v5-ea8/phase7/20260803T1245Z_candidate_ea8_splitk_stride_fast256_c16_v5/parsed_server_args.json")
CANONICAL_VALIDATION = ROOT / "artifacts/phase9-control/20260803T112618Z_ea8_active_identity_migration_v1/phase9_recursive_validation.json"
LAUNCH_SCRIPT = ROOT / "artifacts/phase9-control/20260803T1245Z_ea8_fast256_launch_v5/run_and_monitor.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["id"]: row for row in rows}


def is_right(row: dict) -> bool:
    return row["evaluator_status"] == "scored" and row["score"] == 1.0


def paired(current: dict[str, dict], other: dict[str, dict]) -> dict:
    result = {
        "prompt_hash_matches": 0,
        "gold_matches": 0,
        "both_right": 0,
        "current_only": 0,
        "other_only": 0,
        "both_wrong": 0,
        "truncation_transitions": {},
    }
    for sample_id in sorted(current):
        cur = current[sample_id]
        ref = other[sample_id]
        result["prompt_hash_matches"] += cur["prompt_hash"] == ref["prompt_hash"]
        result["gold_matches"] += cur["gold"] == ref["gold"]
        cur_right, ref_right = is_right(cur), is_right(ref)
        bucket = (
            "both_right" if cur_right and ref_right else
            "current_only" if cur_right else
            "other_only" if ref_right else
            "both_wrong"
        )
        result[bucket] += 1
        transition = f"current_{'truncated' if cur['truncated'] else 'nontruncated'}__other_{'truncated' if ref['truncated'] else 'nontruncated'}"
        counts = result["truncation_transitions"].setdefault(
            transition,
            {"total": 0, "both_right": 0, "current_only": 0, "other_only": 0, "both_wrong": 0},
        )
        counts["total"] += 1
        counts[bucket] += 1
    result["net_correct_gap_current_minus_other"] = result["current_only"] - result["other_only"]
    return result


def env_value(text: str, name: str) -> str | None:
    prefix = name + "="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):]
    return None


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


for name, run in RUNS.items():
    if "predictions" not in run:
        run["predictions"] = run["root"] / "predictions.jsonl"
    run["rows"] = load_rows(run["predictions"])

current_env_text = CURRENT_RUNTIME_ENV.read_text(encoding="utf-8")
current_server_args = json.loads(CURRENT_SERVER_ARGS.read_text(encoding="utf-8"))
canonical = json.loads(CANONICAL_VALIDATION.read_text(encoding="utf-8"))
canonical_checks = {check["name"]: check for check in canonical["checks"]}

snapshot_dir = OUT / "input_snapshots"
snapshot_dir.mkdir(exist_ok=True)
(snapshot_dir / "current_runtime_environment.txt").write_text(current_env_text, encoding="utf-8")
write_json(snapshot_dir / "current_parsed_server_args.json", current_server_args)

run_metrics = {}
for name, run in RUNS.items():
    rows = run["rows"]
    values = list(rows.values())
    completion = [row["token_usage"]["completion_tokens"] for row in values]
    run_metrics[name] = {
        "source_commit": run["source_commit"],
        "index_topk": run["index_topk"],
        "decode_backend": run["decode_backend"],
        "prefill_sort_indices": run["prefill_sort_indices"],
        "total": len(values),
        "unique_ids": len(rows),
        "correct": sum(is_right(row) for row in values),
        "accuracy": sum(is_right(row) for row in values) / len(values),
        "truncated": sum(bool(row["truncated"]) for row in values),
        "truncated_correct": sum(bool(row["truncated"]) and is_right(row) for row in values),
        "nontruncated_correct": sum(not row["truncated"] and is_right(row) for row in values),
        "completion_tokens_mean": sum(completion) / len(completion),
        "completion_tokens_median": statistics.median(completion),
        "protocol_fingerprints": sorted({row["protocol_fingerprint"] for row in values}),
        "predictions_path": str(run["predictions"].relative_to(ROOT)),
        "predictions_bytes": run["predictions"].stat().st_size,
        "predictions_sha256": sha256(run["predictions"]),
    }

current_rows = RUNS["current_ea8_default"]["rows"]
comparisons = {
    name: paired(current_rows, RUNS[name]["rows"])
    for name in ("c349_k1024_legacy", "c349_k1536_legacy", "c349_k768_legacy")
}

canonical_runtime = canonical_checks["performance.candidate_runtime_environment"]["actual"]
canonical_hf = canonical_checks["performance.candidate_hf_overrides"]["actual"]
identity = {
    "formal_v5_observed": {
        "decode_backend": env_value(current_env_text, "VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND"),
        "prefill_topk_tokens": env_value(current_env_text, "VLLM_SPARSE_INDEXER_PREFILL_TOPK_TOKENS"),
        "prefill_sort_indices": env_value(current_env_text, "VLLM_TOPK_PREFILL_SORT_INDICES"),
        "hf_overrides_json": env_value(current_env_text, "HF_OVERRIDES_JSON"),
        "parsed_hf_overrides": current_server_args["hf_overrides"],
        "launch_script_explicit_topk_env_count": sum(
            token in LAUNCH_SCRIPT.read_text(encoding="utf-8")
            for token in (
                "VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND",
                "VLLM_SPARSE_INDEXER_PREFILL_TOPK_TOKENS",
                "VLLM_TOPK_PREFILL_SORT_INDICES",
                "HF_OVERRIDES_JSON",
            )
        ),
    },
    "ea8_canonical_expected": {
        "runtime_environment": canonical_runtime,
        "hf_overrides": canonical_hf,
        "source_commit": canonical_checks["performance.source.commit"]["actual"],
    },
}

summary = {
    "format_version": 1,
    "analysis_type": "cpu_only_paired_accuracy_attribution",
    "runs": run_metrics,
    "paired_vs_current": comparisons,
    "identity_audit": identity,
    "observed_facts": {
        "current_correct_gap_vs_bf16_gate": run_metrics["current_ea8_default"]["correct"] - 105,
        "current_correct_gap_vs_k1024": comparisons["c349_k1024_legacy"]["net_correct_gap_current_minus_other"],
        "k1024_only_correct_count": comparisons["c349_k1024_legacy"]["other_only"],
        "k1024_only_correct_current_truncated_other_nontruncated": comparisons["c349_k1024_legacy"]["truncation_transitions"]["current_truncated__other_nontruncated"]["other_only"],
    },
    "inference_boundary": {
        "supported": "当前精度损失与新增超长输出/截断强相关，且正式v5运行身份偏离ea8 canonical候选身份。",
        "not_supported": "现有配对不能把精度差单独归因于index_topk、persistent decode、prefill排序或ea8源码中的任一变量。",
        "next_candidate": "在ea8源码上恢复canonical组合：index_topk=1024、legacy decode、prefill_topk_tokens=768、prefill sort=1，然后重新执行同一fixed256精度门禁。",
        "performance_retest_allowed": False,
    },
}
write_json(OUT / "summary.json", summary)

checks = []
def check(name: str, actual: object, expected: object) -> None:
    checks.append({"name": name, "status": "passed" if actual == expected else "failed", "actual": actual, "expected": expected})

for name, expected in {
    "current_ea8_default": (101, 140, 4541.55078125, 7974.0),
    "c349_k1024_legacy": (108, 122, 4029.11328125, 912.5),
    "c349_k1536_legacy": (106, 130, 4243.9765625, 7974.0),
    "c349_k768_legacy": (97, 121, 3996.51171875, 782.0),
}.items():
    got = run_metrics[name]
    check(f"{name}.total", got["total"], 256)
    check(f"{name}.unique_ids", got["unique_ids"], 256)
    check(f"{name}.correct", got["correct"], expected[0])
    check(f"{name}.truncated", got["truncated"], expected[1])
    check(f"{name}.completion_mean", got["completion_tokens_mean"], expected[2])
    check(f"{name}.completion_median", got["completion_tokens_median"], expected[3])
    check(f"{name}.protocol", got["protocol_fingerprints"], ["5bc5f1a00a7c48e86baf8a4e1e2b52b17ebbf2f0321b319a6e27b8ca0a404718"])

for name, expected in {
    "c349_k1024_legacy": (65, 36, 43, 112, -7),
    "c349_k1536_legacy": (66, 35, 40, 115, -5),
    "c349_k768_legacy": (59, 42, 38, 117, 4),
}.items():
    got = comparisons[name]
    check(f"paired.{name}.sample_ids_match", sorted(current_rows) == sorted(RUNS[name]["rows"]), True)
    check(f"paired.{name}.prompt_hash_matches", got["prompt_hash_matches"], 256)
    check(f"paired.{name}.gold_matches", got["gold_matches"], 256)
    check(f"paired.{name}.matrix", [got[x] for x in ("both_right", "current_only", "other_only", "both_wrong", "net_correct_gap_current_minus_other")], list(expected))

check("k1024.other_only.current_truncated_other_nontruncated", summary["observed_facts"]["k1024_only_correct_current_truncated_other_nontruncated"], 36)
check("formal_v5.decode_backend", identity["formal_v5_observed"]["decode_backend"], "persistent")
check("formal_v5.prefill_sort_absent", identity["formal_v5_observed"]["prefill_sort_indices"], None)
check("formal_v5.hf_overrides_empty", identity["formal_v5_observed"]["parsed_hf_overrides"], {})
check("formal_v5.launch_explicit_topk_env_count", identity["formal_v5_observed"]["launch_script_explicit_topk_env_count"], 0)
check("canonical.runtime", canonical_runtime, {
    "VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND": "legacy",
    "VLLM_SPARSE_INDEXER_PREFILL_TOPK_TOKENS": "768",
    "VLLM_TOPK_PREFILL_SORT_INDICES": "1",
})
check("canonical.hf_overrides", canonical_hf, {"index_topk": 1024})
check("performance_retest_allowed", summary["inference_boundary"]["performance_retest_allowed"], False)

validation = {
    "format_version": 1,
    "status": "passed" if all(item["status"] == "passed" for item in checks) else "failed",
    "passed": sum(item["status"] == "passed" for item in checks),
    "total": len(checks),
    "checks": checks,
}
write_json(OUT / "validation.json", validation)

manifest_paths = [
    OUT / "build_evidence.py",
    OUT / "summary.json",
    OUT / "validation.json",
    snapshot_dir / "current_runtime_environment.txt",
    snapshot_dir / "current_parsed_server_args.json",
]
(OUT / "evidence_manifest.sha256").write_text(
    "".join(f"{sha256(path)}  {path.relative_to(OUT)}\n" for path in manifest_paths),
    encoding="utf-8",
)
if validation["status"] != "passed":
    raise SystemExit(1)
print(json.dumps({"status": validation["status"], "passed": validation["passed"], "total": validation["total"]}, sort_keys=True))
