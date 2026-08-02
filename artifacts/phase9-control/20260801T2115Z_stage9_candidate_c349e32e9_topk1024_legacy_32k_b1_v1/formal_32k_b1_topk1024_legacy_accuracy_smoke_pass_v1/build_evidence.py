from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


PROJECT = Path("/project")
OUTPUT = PROJECT / (
    "artifacts/phase9-control/"
    "20260801T2115Z_stage9_candidate_c349e32e9_topk1024_legacy_32k_b1_v1/"
    "formal_32k_b1_topk1024_legacy_accuracy_smoke_pass_v1"
)
RUN_ID = "20260801T2134Z_candidate_topk1024_legacy_fast256_c16_v1"
RUN = Path("/host-dev-shm/oscar-glm-stage9/phase7") / RUN_ID
EVAL = RUN / "official_v5_fast_gsm8k_256_c16" / "20260801T214110Z"
INPUTS = {
    "gpu_first.txt": RUN / "gpu_first.txt",
    "gpu_second.txt": RUN / "gpu_second.txt",
    "network_isolation.txt": RUN / "network_isolation.txt",
    "fixed_environment_import.json": RUN / "fixed_environment_import.json",
    "static_preflight.json": RUN / "static_preflight.json",
    "parsed_server_args.json": RUN / "parsed_server_args.json",
    "runtime_environment.txt": RUN / "runtime_environment.txt",
    "runtime_manifest.json": RUN / "runtime_manifest.json",
    "serve_command.txt": RUN / "serve_command.txt",
    "server.log": RUN / "server.log",
    "accuracy_completed_at_utc.txt": RUN / "accuracy_completed_at_utc.txt",
    "server_progress_10min.log": RUN / "progress_10min.log",
    "summary.json": EVAL / "summary.json",
    "official_validation.json": EVAL / "validation.json",
    "predictions.jsonl": EVAL / "predictions.jsonl",
    "failed_cases.jsonl": EVAL / "failed_cases.jsonl",
    "fast_runner_state.json": EVAL / "fast_runner_state.json",
    "runner.log": EVAL / "runner.log",
    "runner_command.txt": EVAL / "runner_command.txt",
    "runner_environment.txt": EVAL / "runner_environment.txt",
    "runner_progress_10min.log": EVAL / "progress_10min.log",
    "summary_by_benchmark.json": EVAL / "summary_by_benchmark.json",
    "summary_by_task_type.json": EVAL / "summary_by_task_type.json",
    "runtime_suite_manifest.jsonl": EVAL / "runtime_suite" / "manifest.jsonl",
    "runtime_suite_eval_config.json": EVAL / "runtime_suite" / "eval_config.json",
    "fast_suite_identity.json": EVAL / "runtime_suite" / "fast_suite_identity.json",
}
EXPECTED_SHA256 = {
    "accuracy_completed_at_utc.txt": "9dfd71143131e61ad3fc7150b64608949ece01fe1e08fa0e9600b084f3b3aeb7",
    "failed_cases.jsonl": "a11cb5239fce9e24d6a50e90b31a7b73319c114d52b1173ebc9ada6f8a0280a0",
    "fast_runner_state.json": "5e6d01317f6e51a9bc84a1664d46a258977f0cd2ca2a52024defd426c5167af0",
    "fast_suite_identity.json": "9324c1d7cf2df73196dbe7353a7652c7201aa1655579f8e9e897c8e7d091cd3f",
    "fixed_environment_import.json": "c0ca8f9bb2b95b0a5477de746c93b245eb3dc810abeedab30cbe7d65d2c07ef0",
    "gpu_first.txt": "8d343ca65384ea57087413b14d0a4b8a7845a793f912c1a335f9374bcb774001",
    "gpu_second.txt": "c2be0a7eb2d0441b6b514e68b44341920940489ce381fbadaaaff9a7af9439ea",
    "network_isolation.txt": "103ec218430170e3dc21f5e789cff1f48dd3d602127d9aa946d3a3a5c13bb8a7",
    "official_validation.json": "4d99b45df8cdff09c3c0dc6b6e19242f38dcdfe221abca855b619b6ea810bf28",
    "parsed_server_args.json": "2aa321152f19117213a17d24c207856ad041abc607dfb1c4e7a55805eeffdb7e",
    "predictions.jsonl": "7ac8c847c1363767f66edaf007acfc261d16791dcc85b2f8eb5a07d6be6f6cbe",
    "runner.log": "abb6e5271107abd9c5e874b457376d6d05447be2c175a9d4109116948eb01b4b",
    "runner_command.txt": "4c4e60e284ae73ad09df436d994d5f738db5b84a79391cd23b43d24ea0d76c31",
    "runner_environment.txt": "c7b606357d4a03c081865e9458e7c1c4ef7f0abff39bdcccfac640f87078ce8f",
    "runner_progress_10min.log": "7f1f6030cf1167dfcd118940caa296378174d84bdc129ca8665ac809afdf73ad",
    "runtime_environment.txt": "c2260c8ffc15f93c8bc168a7b38798700fffa5aef0cd6104ce8352bafbf22350",
    "runtime_manifest.json": "64c2c98a56c6f94e8306b5ce2afe7d4ddb469ef5da52f4a139eb8a24a741d704",
    "runtime_suite_eval_config.json": "025b2ddec84bc65f0ab949025c941b5994b50f26403397edd71ae04aec773c95",
    "runtime_suite_manifest.jsonl": "fcd3079b2db727ef86c54f61fa63bc2431285f0aaa362cd2edb6a71e48f05dcf",
    "serve_command.txt": "f47f06dccefdf6de1d23c884ab41b16a9b769be894108a5906f801736e695311",
    "server.log": "81e26cadb028c15c79d4d687ba7b2a461f3ab76c26fd4d3b6b71fa6c06b295a8",
    "server_progress_10min.log": "95bbed8c60ab445b57d960f66ffd38d7a1d32ea3d55a510466bf7af1fd385df8",
    "static_preflight.json": "2f65fda88a50cac4dd288e28c20dccedde29a6347b5de6fd6bc2efe4ea596aa8",
    "summary.json": "af07359a27dcda1e864ec82e3aba7c50b3426da84928fde4afb8a0dbfeeb1ab1",
    "summary_by_benchmark.json": "b9ae21f99d19b46773aa3f9eeb8ae969a27ee36295864ef99d9956e995aeef75",
    "summary_by_task_type.json": "7dbf33fa5b804d60be5b96363e13eee6547798d0c6aab59de7265b5815bc09e8",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_prompt_hash(sample: dict) -> str:
    messages = sample.get("messages") or [
        {"role": "user", "content": sample["prompt"]}
    ]
    serialized = json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


assert set(INPUTS) == set(EXPECTED_SHA256)
for name, source in INPUTS.items():
    assert sha256(source) == EXPECTED_SHA256[name], name
    shutil.copyfile(source, OUTPUT / name)

static = json.loads((OUTPUT / "static_preflight.json").read_text())
parsed = json.loads((OUTPUT / "parsed_server_args.json").read_text())
runtime = (OUTPUT / "runtime_environment.txt").read_text().splitlines()
runtime_manifest = json.loads((OUTPUT / "runtime_manifest.json").read_text())
summary = json.loads((OUTPUT / "summary.json").read_text())
official = json.loads((OUTPUT / "official_validation.json").read_text())
state = json.loads((OUTPUT / "fast_runner_state.json").read_text())
predictions = read_jsonl(OUTPUT / "predictions.jsonl")
failed_cases = read_jsonl(OUTPUT / "failed_cases.jsonl")
suite_manifest = read_jsonl(OUTPUT / "runtime_suite_manifest.jsonl")
checkpoint_paths = sorted((EVAL / "prediction_checkpoints").glob("*.json"))
checkpoints = [json.loads(path.read_text()) for path in checkpoint_paths]
server_log = (OUTPUT / "server.log").read_text(errors="replace")
current_main = git(PROJECT, "rev-parse", "HEAD")
current_source = git(PROJECT / "glm52_oscar_vllm", "rev-parse", "HEAD")

ids = [row["id"] for row in predictions]
prompt_hashes = [row["prompt_hash"] for row in predictions]
status_counts = dict(collections.Counter(row["evaluator_status"] for row in predictions))
correct = sum(float(row.get("score", 0) or 0) > 0 for row in predictions)
truncated = sum(bool(row.get("truncated")) for row in predictions)
suite_by_id = {row["id"]: row for row in suite_manifest}
expected_failed_ids = {
    row["id"]
    for row in predictions
    if row["evaluator_status"] != "scored"
    or row.get("score") != 1.0
    or row.get("truncated")
}
fatal_patterns = [
    "Traceback",
    "EngineCore encountered a fatal error",
    "RuntimeError: k must be 2048",
    "CUDA out of memory",
    "OutOfMemoryError",
]

checks = [
    ("input_identity", all(sha256(OUTPUT / name) == digest for name, digest in EXPECTED_SHA256.items())),
    ("runtime_main_commit", runtime_manifest["main_commit"] == "c215df4245b6e51a32c2dc5ea28a182e6f43d5ad"),
    ("runtime_main_is_ancestor", subprocess.run(["git", "-C", str(PROJECT), "merge-base", "--is-ancestor", runtime_manifest["main_commit"], current_main]).returncode == 0),
    ("source_commit", current_source == "c349e32e929279e0c7e20676d48d39cc4b5864b3"),
    ("runtime_source_commit", runtime_manifest["runtime_source_commit"] == current_source),
    ("static_status", static["status"] == "passed"),
    ("static_checks", len(static["checks"]) == 44 and all(item["status"] == "passed" for item in static["checks"])),
    ("parsed_tp8", parsed["tensor_parallel_size"] == 8),
    ("parsed_max_model_len", parsed["max_model_len"] == 8192),
    ("parsed_max_batched_tokens", parsed["max_num_batched_tokens"] == 2048),
    ("parsed_max_seqs", parsed["max_num_seqs"] == 16),
    ("parsed_override", parsed["hf_overrides"] == {"index_topk": 1024}),
    ("parsed_cuda_false", parsed["cuda_initialized"] is False),
    ("runtime_override", 'HF_OVERRIDES_JSON={"index_topk":1024}' in runtime),
    ("runtime_prefill_sort", "VLLM_TOPK_PREFILL_SORT_INDICES=1" in runtime),
    ("runtime_decode_backend", "VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND=legacy" in runtime),
    ("runtime_protocol", "EVALUATION_PROTOCOL=official_v5_fast_screen" in runtime),
    ("summary_valid", summary["valid"] is True),
    ("summary_total", summary["total"] == 256),
    ("summary_scored", summary["scored"] == 256),
    ("summary_status", summary["status_counts"] == {"scored": 256}),
    ("summary_accuracy", summary["native_metrics"]["GSM8K"]["accuracy"] == 0.421875),
    ("summary_truncated", summary["truncated_count"] == 122),
    ("official_status", official["status"] == "passed"),
    ("official_request_failures", official["request_failures"] == 0),
    ("official_accuracy", official["accuracy"] == 0.421875),
    ("official_final_eval_boundary", official["final_full_evaluation_still_required"] is True),
    ("official_hashes", official["predictions_sha256"] == EXPECTED_SHA256["predictions.jsonl"] and official["summary_sha256"] == EXPECTED_SHA256["summary.json"] and official["failed_cases_sha256"] == EXPECTED_SHA256["failed_cases.jsonl"]),
    ("state_completed", state["status"] == "completed" and state["completed"] == 256),
    ("state_not_resumed", state["resumed"] == 0),
    ("prediction_rows", len(predictions) == 256),
    ("prediction_unique_ids", len(set(ids)) == 256),
    ("prediction_unique_prompt_hashes", len(set(prompt_hashes)) == 256),
    ("prediction_scored", status_counts == {"scored": 256}),
    ("prediction_correct", correct == 108),
    ("prediction_accuracy", correct / len(predictions) == 0.421875),
    ("prediction_truncated", truncated == 122),
    ("checkpoint_count", len(checkpoint_paths) == 256),
    ("checkpoint_names", [path.name for path in checkpoint_paths] == [f"{index:06d}.json" for index in range(256)]),
    ("checkpoint_records", checkpoints == predictions),
    ("suite_manifest_rows", len(suite_manifest) == 256 and len(suite_by_id) == 256),
    ("suite_prediction_ids", set(suite_by_id) == set(ids)),
    ("canonical_prompt_hashes", all(canonical_prompt_hash(suite_by_id[row["id"]]) == row["prompt_hash"] for row in predictions)),
    ("protocol_fingerprint", {row["protocol_fingerprint"] for row in predictions} == {official["protocol_fingerprint"]}),
    ("fixed_output_limit", {row["requested_max_tokens"] for row in predictions} == {7974}),
    ("server_model_len", {row["server_max_model_len"] for row in predictions} == {8192}),
    ("failed_cases_definition", len(failed_cases) == 156 and len({row["id"] for row in failed_cases}) == 156 and {row["id"] for row in failed_cases} == expected_failed_ids),
    ("server_no_fatal", not any(pattern in server_log for pattern in fatal_patterns)),
    ("accuracy_completed", (OUTPUT / "accuracy_completed_at_utc.txt").read_text().strip() == "2026-08-02T01:47:18Z"),
    ("candidate_gate", correct >= 105 and truncated <= 130 and status_counts == {"scored": 256}),
]
assert all(passed for _, passed in checks), [name for name, passed in checks if not passed]

source_contract = {
    "format_version": 1,
    "run_id": RUN_ID,
    "main_commit_at_run": runtime_manifest["main_commit"],
    "source_commit": current_source,
    "candidate": {
        "hf_overrides": {"index_topk": 1024},
        "prefill_sort_indices": "1",
        "decode_topk_backend": "legacy",
    },
    "screening_protocol": "official_v5_fast_screen",
    "input_sha256": EXPECTED_SHA256,
}
validation = {
    "format_version": 1,
    "status": "passed",
    "checks_passed": len(checks),
    "checks_total": len(checks),
    "checks": [{"name": name, "passed": passed} for name, passed in checks],
    "outcome": {
        "classification": "performance_candidate_screen_passed",
        "total": 256,
        "scored": 256,
        "correct": 108,
        "accuracy": 0.421875,
        "request_failed": 0,
        "truncated": 122,
        "truncation_rate": 0.4765625,
        "completion_tokens_mean": 4029.11328125,
        "requests_per_hour": 62.48502152715987,
    },
    "historical_conservative_gate": {
        "bf16_correct": 105,
        "oscar_topk2048_correct": 107,
        "oscar_topk1536_correct": 106,
        "candidate_minus_bf16_correct": 3,
        "candidate_minus_oscar_topk2048_correct": 1,
        "candidate_minus_oscar_topk1536_correct": 2,
        "paired_comparison": False,
    },
    "boundary": (
        "This run passed the conservative performance-candidate screening gate only. "
        "It used official_v5_fast_screen, had 122/256 truncated outputs, and the "
        "historical BF16/OSCAR protocol fingerprint is not paired with this run. "
        "The official validator states that a final full evaluation is still required."
    ),
}
(OUTPUT / "source_contract.json").write_text(
    json.dumps(source_contract, ensure_ascii=False, indent=2) + "\n"
)
(OUTPUT / "validation.json").write_text(
    json.dumps(validation, ensure_ascii=False, indent=2) + "\n"
)
manifest_paths = [
    OUTPUT / "build_evidence.py",
    *(OUTPUT / name for name in INPUTS),
    OUTPUT / "source_contract.json",
    OUTPUT / "validation.json",
]
(OUTPUT / "evidence_manifest.sha256").write_text(
    "".join(f"{sha256(path)}  {path.name}\n" for path in manifest_paths)
)
print(json.dumps({"status": "passed", "checks": len(checks), "manifest": len(manifest_paths)}))
