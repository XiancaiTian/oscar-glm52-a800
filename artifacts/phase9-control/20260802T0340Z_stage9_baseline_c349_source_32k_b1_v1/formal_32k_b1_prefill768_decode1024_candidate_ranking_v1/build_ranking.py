from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


PROJECT = Path("/project")
OUTPUT = Path(__file__).resolve().parent
SOURCE = PROJECT / "glm52_oscar_vllm"
NUM_ROWS = 32768
TILE = 16
BF16_TTFT_MS = 12515.1053785036
K1024_TTFT_MS = 21032.01403375715
K1536_TTFT_MS = 25682.409651267033
K1024_STAGE1_MS = 10217.4636085
K1536_STAGE1_MS = 15068.884579500014


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def selected_token_instances(topk: int) -> int:
    return topk * (topk + 1) // 2 + (NUM_ROWS - topk) * topk


def active_tiles(topk: int) -> int:
    assert topk % TILE == 0
    tiles = topk // TILE
    return TILE * tiles * (tiles + 1) // 2 + (NUM_ROWS - topk) * tiles


def workload(topk: int) -> dict[str, int | float]:
    base_selected = selected_token_instances(1024)
    base_active = active_tiles(1024)
    return {
        "topk": topk,
        "selected_token_instances": selected_token_instances(topk),
        "selected_reduction_vs_k1024_fraction": 1
        - selected_token_instances(topk) / base_selected,
        "active_tiles": active_tiles(topk),
        "active_reduction_vs_k1024_fraction": 1
        - active_tiles(topk) / base_active,
    }


def linear_projection(topk: int) -> dict[str, float]:
    fraction = (topk - 1024) / (1536 - 1024)
    stage1 = K1024_STAGE1_MS + fraction * (K1536_STAGE1_MS - K1024_STAGE1_MS)
    ttft = K1024_TTFT_MS + fraction * (K1536_TTFT_MS - K1024_TTFT_MS)
    return {
        "topk": topk,
        "stage1_ms": stage1,
        "ttft_ms": ttft,
        "ttft_reduction_vs_k1024_ms": K1024_TTFT_MS - ttft,
        "ttft_regression_vs_bf16_percent": (ttft / BF16_TTFT_MS - 1) * 100,
    }


paths = {
    "model": SOURCE / "vllm/model_executor/models/glm4_moe_lite.py",
    "indexer": SOURCE / "vllm/model_executor/layers/sparse_attn_indexer.py",
    "attention": SOURCE / "vllm/v1/attention/backends/mla/triton_mla_sparse.py",
    "metadata": SOURCE / "vllm/v1/attention/backends/mla/xpu_mla_sparse.py",
    "reference": SOURCE / "vllm/v1/attention/backends/mla/flashmla_sparse.py",
    "config": PROJECT / "configs/phase9/performance_matrix.json",
}
texts = {name: path.read_text() for name, path in paths.items()}
config = json.loads(texts["config"])

result = {
    "status": "passed",
    "format_version": 1,
    "scope": "current_c349_prefill768_decode1024_candidate_ranking_v1",
    "mode": "cpu_only_source_contract_and_existing_measurement_ranking",
    "environment": {
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "image": "oscar-glm-stage9-runtime:c349e32e9",
        "network": "none",
    },
    "identity": {
        "main_commit": git(PROJECT, "rev-parse", "HEAD"),
        "source_commit": git(SOURCE, "rev-parse", "HEAD"),
    },
    "measured_boundary": {
        "bf16": {"ttft_ms": BF16_TTFT_MS, "tpot_ms": 153.7397446482396},
        "oscar_k1024": {
            "ttft_ms": K1024_TTFT_MS,
            "tpot_ms": 197.71881286406844,
            "fast256_correct": 108,
            "fast256_accuracy_percent": 42.1875,
            "fast256_truncated": 122,
            "screen_passed": True,
        },
        "oscar_k768": {
            "performance_run_executed": False,
            "fast256_correct": 97,
            "fast256_accuracy_percent": 37.890625,
            "fast256_truncated": 121,
            "screen_passed": False,
            "reason": "97 correct is below the frozen minimum of 105",
        },
    },
    "source_contract_now": {
        "shared_buffer_width_from_config": True,
        "indexer_prefill_decode_share_topk": True,
        "oscar_attention_has_prefill_slice": True,
        "oscar_attention_whole_batch_decode_test": True,
        "attention_metadata_has_split_counts": False,
        "attention_metadata_imports_split_helper": False,
        "flashmla_reference_has_split_counts": True,
        "current_control_index_topk": config["candidate_hf_overrides"]["index_topk"],
    },
    "workload": {
        "input_tokens": NUM_ROWS,
        "batch_size": 1,
        "output_tokens": 128,
        "prefill": [workload(k) for k in (1024, 960, 896, 768)],
        "decode_topk_for_selected_candidate": 1024,
        "boundary": "Exact causal selected-work accounting; not measured timing.",
    },
    "projection": {
        "basis": "linear interpolation/extrapolation through measured uniform K1536 and K1024",
        "uniform_or_prefill_only": [linear_projection(k) for k in (960, 896, 768)],
        "selected_candidate_tpot_ms": None,
        "boundary": (
            "The K768 prefill TTFT value is a projection only. Decode keeps K1024 "
            "algorithmically, but mixed-batch splitting overhead makes TPOT non-projectable."
        ),
    },
    "ranking": [
        {
            "rank": 1,
            "name": "prefill_topk768_decode_topk1024",
            "selected": True,
            "reason": (
                "Preserves the screened K1024 decode width while targeting the prefill "
                "stage that explains the material TTFT reduction; requires mixed-batch-safe "
                "metadata and two-width attention calls."
            ),
        },
        {
            "rank": 2,
            "name": "uniform_topk960",
            "selected": False,
            "reason": (
                "Lower accuracy risk than K768 but only about 581 ms projected TTFT gain "
                "and still about 63.4% slower than BF16."
            ),
        },
        {
            "rank": 3,
            "name": "uniform_topk896",
            "selected": False,
            "reason": (
                "About 1163 ms projected gain, but repeats the uniform decode-width accuracy "
                "risk before testing the more informative split-width candidate."
            ),
        },
        {
            "rank": 4,
            "name": "rotation_16384_rows_m32",
            "selected": False,
            "reason": "Existing stable inverse-only signal is about 85 ms and cannot close the gap.",
        },
    ],
    "selected_candidate_contract": {
        "model_index_topk": 1024,
        "prefill_index_topk": 768,
        "decode_index_topk": 1024,
        "default_unset_behavior_unchanged": True,
        "mixed_batch_required": True,
        "gpu_allowed_now": False,
        "implementation_scope": [
            "restore all five control-plane index_topk consumers from 768 to 1024",
            "add one fail-closed prefill-topk environment contract fixed to 768",
            "make indexer prefill write 768 columns while decode writes 1024",
            "add decode/prefill counts to sparse MLA metadata only when needed",
            "split OSCAR attention rows in mixed batches and consume 1024/768 columns",
        ],
        "gates": [
            "CPU-only TDD red/green for unset, pure-prefill, pure-decode, and mixed-batch contracts",
            "source diff and compile/static preflight with exact K1024/768 identity",
            "two 8-GPU idle checks at least 60 seconds apart",
            "CUDA indexer/attention correctness for pure and mixed batches",
            "same 256-question fast screen: 256 scored, 0 request failures, >=105 correct, <=130 truncated",
            "same 32K/batch1/output128/TP8 warm-up plus three formal rounds and profiler",
            "full 2360-example accuracy and PPL remain required before final promotion",
        ],
    },
}

checks = [
    ("cuda_hidden", result["environment"]["cuda_visible_devices"] in (None, "")),
    ("main_identity", result["identity"]["main_commit"] == "8eb2ee9005ef92514ac2bbcb655241a3001c3d03"),
    ("source_identity", result["identity"]["source_commit"] == "c349e32e929279e0c7e20676d48d39cc4b5864b3"),
    ("model_buffer_from_config", "topk_tokens = config.index_topk" in texts["model"] and "topk_tokens," in texts["model"]),
    ("indexer_unified_topk", "topk_indices_buffer[\n                    chunk.token_start : chunk.token_end, :topk_tokens\n                ]" in texts["indexer"]),
    ("indexer_decode_unified_topk", "topk_indices_buffer[:num_padded_tokens, :topk_tokens]" in texts["indexer"]),
    ("attention_prefill_slice", "self.topk_indices_buffer[:num_actual_toks, :topk_width]" in texts["attention"]),
    ("attention_whole_batch_test", "attn_metadata.max_query_len == 1" in texts["attention"]),
    ("metadata_no_split_counts", "num_decode_tokens:" not in texts["metadata"] and "num_prefill_tokens:" not in texts["metadata"]),
    ("metadata_missing_split_import", "split_decodes_and_prefills" not in texts["metadata"]),
    ("reference_split_counts", "num_decode_tokens: int = 0" in texts["reference"] and "num_prefill_tokens: int = 0" in texts["reference"]),
    ("control_currently_k768", result["source_contract_now"]["current_control_index_topk"] == 768),
    ("k1024_screen_passed", result["measured_boundary"]["oscar_k1024"]["screen_passed"]),
    ("k768_screen_failed", not result["measured_boundary"]["oscar_k768"]["screen_passed"]),
    ("hybrid_base_k1024", result["selected_candidate_contract"]["model_index_topk"] == 1024),
    ("hybrid_prefill_k768", result["selected_candidate_contract"]["prefill_index_topk"] == 768),
    ("hybrid_decode_k1024", result["selected_candidate_contract"]["decode_index_topk"] == 1024),
    ("mixed_batch_required", result["selected_candidate_contract"]["mixed_batch_required"]),
    ("default_unchanged", result["selected_candidate_contract"]["default_unset_behavior_unchanged"]),
    ("tpot_not_projected", result["projection"]["selected_candidate_tpot_ms"] is None),
    ("selected_hybrid", result["ranking"][0]["selected"] and result["ranking"][0]["name"] == "prefill_topk768_decode_topk1024"),
    ("gpu_forbidden", not result["selected_candidate_contract"]["gpu_allowed_now"]),
]
validation = {
    "status": "passed" if all(passed for _, passed in checks) else "failed",
    "checks_passed": sum(passed for _, passed in checks),
    "checks_total": len(checks),
    "checks": [{"name": name, "passed": passed} for name, passed in checks],
}
assert validation["status"] == "passed", [
    name for name, passed in checks if not passed
]

result_path = OUTPUT / "candidate_ranking.json"
validation_path = OUTPUT / "validation.json"
result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest_paths = [Path(__file__), result_path, validation_path]
(OUTPUT / "evidence_manifest.sha256").write_text(
    "".join(f"{sha256(path)}  {path.name}\n" for path in manifest_paths)
)
print(json.dumps({"status": "passed", "checks": validation["checks_total"]}))
