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
    "20260802T0340Z_stage9_baseline_c349_source_32k_b1_v1/"
    "formal_32k_b1_topk768_legacy_accuracy_smoke_failed_v1"
)
RUN_ID = "20260802T0600Z_candidate_topk768_legacy_fast256_c16_v1"
SHM = Path("/host-dev-shm/oscar-glm-stage9")
RUN = SHM / "phase7" / RUN_ID
EVAL = RUN / "official_v5_fast_gsm8k_256_c16" / "20260802T060633Z"
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
    "outer_exit.txt": SHM / f"{RUN_ID}.outer.exit",
    "outer.log": SHM / f"{RUN_ID}.outer.log",
    "accuracy_progress_10min.log": SHM / f"{RUN_ID}.accuracy_progress_10min.log",
}
EXPECTED_SHA256 = {
    "accuracy_completed_at_utc.txt": "80421b2bc17a27b9e1905ecfe5c28997c18aed2444a3a4fc04a68b13e0be4513",
    "accuracy_progress_10min.log": "67e7bb4f1d199ecac017fbec2b7a636e6017e97392a4c544315ebf145a64d2df",
    "failed_cases.jsonl": "c0f384f54572369a5fc771ab80469fc0e9b6115a3ba52cfb3cb128be430f7797",
    "fast_runner_state.json": "def71a12d9c2c25914928e0515709f13cf48d761e89a358a68f97b800fba8fef",
    "fast_suite_identity.json": "9324c1d7cf2df73196dbe7353a7652c7201aa1655579f8e9e897c8e7d091cd3f",
    "fixed_environment_import.json": "c0ca8f9bb2b95b0a5477de746c93b245eb3dc810abeedab30cbe7d65d2c07ef0",
    "gpu_first.txt": "7798f1ad1a37616effaf7b52a61b0738de6b013e1c6b18291c695189d1ff8f63",
    "gpu_second.txt": "129187f89bda89541be833052b7946c00c2d147f7b2946806f2bec6fc09c1485",
    "network_isolation.txt": "eff211575b0b1956c3348edaf6c02909a7d983f0902ec74a3fafc96cf2dc90b3",
    "official_validation.json": "fa239e7ca98b00b97642fab99c1b60552c320466bbdb3d8e6ba70dd15fed7ee1",
    "outer.log": "658092e4afd41ba33c0c8dc05279d7cc90ffa22962d8fcf58796357845d1a22f",
    "outer_exit.txt": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "parsed_server_args.json": "df7098fe351999cf85b6d9322afb33d896023ae6ca56a99eb67307e71ba83c0d",
    "predictions.jsonl": "c212cfadc3bf989fa95220ad38d6714af520358559b9da4c957eb7eefca5b4ab",
    "runner.log": "649d6114bf4fd6d2b99bb9ea6adc5a613f3bc40a85e922e9f28d5a70476ad689",
    "runner_command.txt": "c9413a93872c66383f0ae11412eded9804f6ccdf12fb601521717a65b5dfcc26",
    "runner_environment.txt": "828ac0aa5605b3acad37d051da9126029af66cc1e4ab235cc9987886a70fe69d",
    "runner_progress_10min.log": "cd8531b4b3f5214e02fbc7a512e7270b8b6c51c8de8636fcc1740f2b3dfd2b5b",
    "runtime_environment.txt": "ecaf3cc61035f6670ba29c50cbb966820dde50d34f0f3e089a6703e93e2ca1fa",
    "runtime_manifest.json": "a093ccd183aa358d3d70eaec6796731c83928e3012366327fd85eae5034536f6",
    "runtime_suite_eval_config.json": "025b2ddec84bc65f0ab949025c941b5994b50f26403397edd71ae04aec773c95",
    "runtime_suite_manifest.jsonl": "fcd3079b2db727ef86c54f61fa63bc2431285f0aaa362cd2edb6a71e48f05dcf",
    "serve_command.txt": "d501697003f28320b65f6a9830892cbb4a1e3c7835b1b9dcf96482f93de60ced",
    "server.log": "8d5e5ee91752ff80e6177139d73166f2df13ceb880cb8c0f38d79f80642bee36",
    "server_progress_10min.log": "2c276c7208beafc79a72a45568a35f0c5a28c8aec2a09f8b3f3003ddddb03f4f",
    "static_preflight.json": "2f65fda88a50cac4dd288e28c20dccedde29a6347b5de6fd6bc2efe4ea596aa8",
    "summary.json": "9180c02e64feb6153db19be6ee07f7c69aa7509291d555e55bd82f882f9ec91c",
    "summary_by_benchmark.json": "c2d0b75597e93fb588b91a32d51bd068e76e7aea780eec5cfbb6ee98f15bdc1c",
    "summary_by_task_type.json": "4c34ae580dfceb875006007f952d4cb2019a8d5c9026b7a7cb62dbf859aa6ef3",
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
accuracy_monitor = (OUTPUT / "accuracy_progress_10min.log").read_text().splitlines()
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
    ("outer_exit_zero", (OUTPUT / "outer_exit.txt").read_text().strip() == "0"),
    ("runtime_main_commit", runtime_manifest["main_commit"] == "88ad032b749a896c83085096eb4a3ecc0aaa6b7c"),
    ("runtime_main_is_ancestor", subprocess.run(["git", "-C", str(PROJECT), "merge-base", "--is-ancestor", runtime_manifest["main_commit"], current_main]).returncode == 0),
    ("source_commit", current_source == "c349e32e929279e0c7e20676d48d39cc4b5864b3"),
    ("runtime_source_commit", runtime_manifest["runtime_source_commit"] == current_source),
    ("static_status", static["status"] == "passed"),
    ("static_checks", len(static["checks"]) == 44 and all(item["status"] == "passed" for item in static["checks"])),
    ("parsed_tp8", parsed["tensor_parallel_size"] == 8),
    ("parsed_max_model_len", parsed["max_model_len"] == 8192),
    ("parsed_max_batched_tokens", parsed["max_num_batched_tokens"] == 2048),
    ("parsed_max_seqs", parsed["max_num_seqs"] == 16),
    ("parsed_override", parsed["hf_overrides"] == {"index_topk": 768}),
    ("parsed_cuda_false", parsed["cuda_initialized"] is False),
    ("runtime_override", 'HF_OVERRIDES_JSON={"index_topk":768}' in runtime),
    ("runtime_prefill_sort", "VLLM_TOPK_PREFILL_SORT_INDICES=1" in runtime),
    ("runtime_decode_backend", "VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND=legacy" in runtime),
    ("runtime_protocol", "EVALUATION_PROTOCOL=official_v5_fast_screen" in runtime),
    ("summary_valid", summary["valid"] is True),
    ("summary_total", summary["total"] == 256),
    ("summary_scored", summary["scored"] == 256),
    ("summary_status", summary["status_counts"] == {"scored": 256}),
    ("summary_accuracy", summary["native_metrics"]["GSM8K"]["accuracy"] == 0.37890625),
    ("summary_truncated", summary["truncated_count"] == 121),
    ("official_status", official["status"] == "passed"),
    ("official_request_failures", official["request_failures"] == 0),
    ("official_accuracy", official["accuracy"] == 0.37890625),
    ("official_final_eval_boundary", official["final_full_evaluation_still_required"] is True),
    ("official_hashes", official["predictions_sha256"] == EXPECTED_SHA256["predictions.jsonl"] and official["summary_sha256"] == EXPECTED_SHA256["summary.json"] and official["failed_cases_sha256"] == EXPECTED_SHA256["failed_cases.jsonl"]),
    ("state_completed", state["status"] == "completed" and state["completed"] == 256),
    ("state_not_resumed", state["resumed"] == 0),
    ("prediction_rows", len(predictions) == 256),
    ("prediction_unique_ids", len(set(ids)) == 256),
    ("prediction_unique_prompt_hashes", len(set(prompt_hashes)) == 256),
    ("prediction_scored", status_counts == {"scored": 256}),
    ("prediction_correct", correct == 97),
    ("prediction_accuracy", correct / len(predictions) == 0.37890625),
    ("prediction_truncated", truncated == 121),
    ("checkpoint_count", len(checkpoint_paths) == 256),
    ("checkpoint_names", [path.name for path in checkpoint_paths] == [f"{index:06d}.json" for index in range(256)]),
    ("checkpoint_records", checkpoints == predictions),
    ("suite_manifest_rows", len(suite_manifest) == 256 and len(suite_by_id) == 256),
    ("suite_prediction_ids", set(suite_by_id) == set(ids)),
    ("canonical_prompt_hashes", all(canonical_prompt_hash(suite_by_id[row["id"]]) == row["prompt_hash"] for row in predictions)),
    ("protocol_fingerprint", {row["protocol_fingerprint"] for row in predictions} == {official["protocol_fingerprint"]}),
    ("fixed_output_limit", {row["requested_max_tokens"] for row in predictions} == {7974}),
    ("server_model_len", {row["server_max_model_len"] for row in predictions} == {8192}),
    ("failed_cases_definition", len(failed_cases) == 161 and len({row["id"] for row in failed_cases}) == 161 and {row["id"] for row in failed_cases} == expected_failed_ids),
    ("server_no_fatal", not any(pattern in server_log for pattern in fatal_patterns)),
    ("accuracy_completed", (OUTPUT / "accuracy_completed_at_utc.txt").read_text().strip() == "2026-08-02T10:09:41Z"),
    ("accuracy_monitor_correction_retained", "failures=2" in accuracy_monitor[0] and "label=correction" in accuracy_monitor[1] and "request_failures=0" in accuracy_monitor[1] and "extraction_failures=2" in accuracy_monitor[1]),
    ("accuracy_monitor_final", "label=final completed=256/256 correct=97 accuracy=37.890625% request_failures=0 extraction_failures=126 truncated=121 checkpoint_read_errors=0" in accuracy_monitor[-1]),
    ("candidate_gate_failed", not (correct >= 105 and truncated <= 130 and status_counts == {"scored": 256})),
]
assert all(passed for _, passed in checks), [name for name, passed in checks if not passed]

source_contract = {
    "format_version": 1,
    "run_id": RUN_ID,
    "main_commit_at_run": runtime_manifest["main_commit"],
    "source_commit": current_source,
    "candidate": {
        "hf_overrides": {"index_topk": 768},
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
        "classification": "performance_candidate_screen_failed",
        "total": 256,
        "scored": 256,
        "correct": 97,
        "accuracy": 0.37890625,
        "request_failed": 0,
        "extraction_failed": 126,
        "truncated": 121,
        "truncation_rate": 0.47265625,
        "completion_tokens_mean": 3996.51171875,
        "requests_per_hour": 63.30784834991492,
    },
    "historical_conservative_gate": {
        "bf16_correct": 105,
        "oscar_topk2048_correct": 107,
        "oscar_topk1536_correct": 106,
        "oscar_topk1024_correct": 108,
        "candidate_minus_bf16_correct": -8,
        "candidate_minus_oscar_topk2048_correct": -10,
        "candidate_minus_oscar_topk1536_correct": -9,
        "candidate_minus_oscar_topk1024_correct": -11,
        "correctness_gate_passed": False,
        "truncation_gate_passed": True,
        "paired_comparison": False,
    },
    "boundary": (
        "This run failed the conservative performance-candidate screening gate: "
        "97 correct is below the required 105. It used official_v5_fast_screen, "
        "had 121/256 truncated outputs, and the historical BF16/OSCAR protocol "
        "fingerprints are not paired with this run. No 32K performance run is "
        "authorized for this K=768 candidate."
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
