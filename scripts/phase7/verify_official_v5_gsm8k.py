#!/usr/bin/env python3
"""Fail-closed verifier for the frozen official_v5 GSM8K stage gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def equal(self, name: str, actual: Any, expected: Any) -> None:
        self.items.append(
            {
                "name": name,
                "status": "passed" if actual == expected else "failed",
                "actual": actual,
                "expected": expected,
            }
        )

    @property
    def passed(self) -> bool:
        return all(item["status"] == "passed" for item in self.items)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_sha256sums(checks: Checks, root: Path, expected_hash: str) -> None:
    sums = root / "SHA256SUMS"
    checks.equal("frozen.sha256sums.sha256", sha256_file(sums), expected_hash)
    mismatches: list[str] = []
    rows = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.removeprefix("*").removeprefix("./")
        path = root / relative
        rows += 1
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(relative)
    checks.equal(
        "frozen.sha256sums.contents",
        {"rows": rows, "mismatches": mismatches},
        {"rows": 161, "mismatches": []},
    )


def tree_sha256(root: Path) -> str:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(f"{sha256_file(path)}  {path.relative_to(root)}\n")
    return hashlib.sha256("".join(records).encode()).hexdigest()


def verify_environment(checks: Checks, root: Path) -> None:
    python = root / ".venv/bin/python"
    checks.equal("environment.python_exists", python.is_file(), True)
    completed = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import nltk, requests; "
                "nltk.data.find('tokenizers/punkt'); "
                "nltk.data.find('tokenizers/punkt_tab'); "
                "print(requests.__version__)"
            ),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "NLTK_DATA": str(root / "nltk_data")},
    )
    checks.equal("environment.imports", completed.returncode, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config = read_json(args.config)
    frozen = config["frozen_evaluator"]
    root = project_root / frozen["root"]
    suite = root / "accuracy_suites/model_agnostic_accuracy_official_v5"
    checks = Checks()

    checks.equal("config.status", config["status"], "ready")
    checks.equal("config.protocol", config["protocol"], "official_v5")
    checks.equal("config.scope", config["scope"], "current_stage_gsm8k")
    verify_sha256sums(checks, root, frozen["sha256sums_sha256"])

    for name, relative in (
        ("manifest", "manifest.jsonl"),
        ("suite_meta", "suite_meta.json"),
        ("eval_config", "eval_config.json"),
    ):
        checks.equal(
            f"suite.{name}.sha256",
            sha256_file(suite / relative),
            frozen[f"{name}_sha256"],
        )
    checks.equal(
        "runner.sha256",
        sha256_file(root / "tools/run_accuracy_suite.py"),
        frozen["runner_sha256"],
    )
    checks.equal(
        "requirements_lock.sha256",
        sha256_file(root / "requirements-lock.txt"),
        frozen["requirements_lock_sha256"],
    )
    checks.equal(
        "nltk.tree_sha256",
        tree_sha256(root / "nltk_data"),
        frozen["nltk_tree_sha256"],
    )

    source_identity = dict(
        line.split("=", 1)
        for line in (root / "SOURCE_IDENTITY.txt").read_text().splitlines()
        if "=" in line
    )
    checks.equal(
        "source.git_head",
        source_identity["source_git_head"],
        frozen["source_git_head"],
    )
    checks.equal(
        "source.git_status_sha256",
        source_identity["source_git_status_sha256"],
        frozen["source_git_status_sha256"],
    )
    checks.equal(
        "source.git_status_entries",
        int(source_identity["source_git_status_entries"]),
        frozen["source_git_status_entries"],
    )

    sample = suite / config["selection"]["sample_file"]
    checks.equal(
        "selection.sample_sha256",
        sha256_file(sample),
        config["selection"]["sample_sha256"],
    )
    sample_rows = read_jsonl(sample)
    selected_manifest = [
        row
        for row in read_jsonl(suite / "manifest.jsonl")
        if row["benchmark"] == "GSM8K"
    ]
    expected_total = config["selection"]["total"]
    checks.equal("selection.sample_rows", len(sample_rows), expected_total)
    checks.equal("selection.manifest_rows", len(selected_manifest), expected_total)
    checks.equal(
        "selection.sample_ids",
        [row["id"] for row in sample_rows],
        [row["id"] for row in selected_manifest],
    )

    eval_config = read_json(suite / "eval_config.json")
    checks.equal(
        "eval.protocol_version", eval_config["protocol_version"], "official_v5"
    )
    checks.equal("eval.decoding", eval_config["decoding"], config["decoding"])
    checks.equal(
        "eval.native_metrics_only",
        eval_config["reporting"],
        {"cross_benchmark_accuracy": False, "native_metrics_only": True},
    )
    checks.equal(
        "eval.math_timeout_seconds",
        eval_config["timeouts_seconds"]["math_reasoning"],
        300,
    )
    adaptation = config["transport_adaptation"]
    runtime_eval_config_path = project_root / adaptation["runtime_eval_config"]
    runtime_eval_config = read_json(runtime_eval_config_path)
    checks.equal(
        "runtime_eval_config.sha256",
        sha256_file(runtime_eval_config_path),
        adaptation["runtime_eval_config_sha256"],
    )
    expected_runtime_eval_config = json.loads(json.dumps(eval_config))
    expected_runtime_eval_config["timeouts_seconds"]["math_reasoning"] = 7200
    expected_runtime_eval_config["transport_adaptation"] = {
        "reason": (
            "The official_v5 runner fixes GSM8K output at 32550 tokens on the "
            "32768-token server; A800 TP8 non-streaming requests exceeded "
            "300, 900, and 1800-second client timeouts."
        ),
        "scope": "math_reasoning_client_timeout_only",
        "upstream_timeout_seconds": 300,
        "rejected_intermediate_timeout_seconds": [900, 1800],
    }
    checks.equal(
        "runtime_eval_config.only_math_timeout_changed",
        runtime_eval_config,
        expected_runtime_eval_config,
    )
    checks.equal(
        "runtime_eval_config.math_timeout_seconds",
        runtime_eval_config["timeouts_seconds"]["math_reasoning"],
        adaptation["runtime_math_timeout_seconds"],
    )
    checks.equal(
        "runtime_eval_config.rejected_intermediate_timeout_seconds",
        adaptation["rejected_intermediate_timeout_seconds"],
        [900, 1800],
    )
    checks.equal(
        "runtime_eval_config.samples_decoding_scoring_unchanged",
        adaptation["samples_decoding_scoring_unchanged"],
        True,
    )
    checks.equal("selection.benchmarks", config["selection"]["benchmarks"], ["GSM8K"])
    checks.equal(
        "final.current_result_not_substitute",
        config["final_acceptance"]["current_gsm8k_result_is_not_a_substitute"],
        True,
    )
    verify_environment(checks, root)

    result = {
        "format_version": 1,
        "status": "passed" if checks.passed else "failed",
        "config": str(args.config.resolve()),
        "checks": checks.items,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if checks.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
