#!/usr/bin/env python3
"""Fail-closed verifier for the Stage 7 official_v5 fast-screen protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("official_v5_fast_prepare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fast-suite preparer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def manifest_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def selected_ids_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(row["id"] for row in rows) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config = read_json(args.config)
    official = read_json(project_root / "configs/phase7/official_v5_gsm8k.json")
    implementation = config["implementation"]
    selection = config["selection"]
    eval_config_path = project_root / implementation["eval_config"]
    eval_config = read_json(eval_config_path)
    source_manifest = (
        project_root
        / "artifacts/phase7/frozen_evaluator_v5_20260728"
        / "accuracy_suites/model_agnostic_accuracy_official_v5/manifest.jsonl"
    )
    prepare_path = project_root / implementation["prepare_suite"]
    prepare = load_module(prepare_path)
    source_rows = read_jsonl(source_manifest)
    chat_template = (Path(config["model"]) / "chat_template.jinja").read_text(
        encoding="utf-8"
    )
    checks = Checks()

    checks.equal("config.status", config["status"], "ready")
    checks.equal("config.protocol", config["protocol"], "official_v5_fast_screen")
    checks.equal("config.source_protocol", config["source_protocol"], "official_v5")
    checks.equal("model.path", config["model"], official["model"]["path"])
    checks.equal(
        "model.served_name",
        config["served_model_name"],
        official["model"]["served_model_name"],
    )
    checks.equal(
        "source_manifest.sha256",
        sha256_file(source_manifest),
        selection["source_manifest_sha256"],
    )
    checks.equal(
        "source_manifest.rows",
        len([row for row in source_rows if row["benchmark"] == "GSM8K"]),
        selection["source_total"],
    )
    for name in ("prepare_suite", "fast_runner", "matrix_comparator", "eval_config"):
        path = project_root / implementation[name]
        checks.equal(
            f"implementation.{name}.sha256",
            sha256_file(path),
            implementation[f"{name}_sha256"],
        )
    checks.equal("eval.protocol", eval_config["protocol_version"], "official_v5")
    checks.equal(
        "eval.screening_protocol",
        eval_config["screening_protocol"],
        {
            "benchmark_max_prompt_tokens": 218,
            "name": "official_v5_fast_screen",
            "server_max_model_len": 8192,
            "final_full_evaluation_still_required": True,
        },
    )
    checks.equal("eval.decoding", eval_config["decoding"], config["decoding"])
    checks.equal(
        "eval.math_timeout",
        eval_config["timeouts_seconds"]["math_reasoning"],
        3600,
    )
    checks.equal("server.max_model_len", config["server"]["max_model_len"], 8192)
    checks.equal("server.max_num_seqs", config["server"]["max_num_seqs"], 16)
    checks.equal(
        "server.speculative_disabled",
        config["server"]["enable_speculative_decoding"],
        False,
    )
    checks.equal(
        "server.cuda_graph_disabled",
        config["server"]["enable_cuda_graph"],
        False,
    )
    checks.equal(
        "chat_template.high_is_high",
        (
            "effective_reasoning_effort = 'high' if reasoning_effort is defined "
            "and reasoning_effort == 'high' else 'max'"
        )
        in chat_template,
        True,
    )

    for name, count in (("pilot", selection["pilot_total"]), ("full", 1319)):
        rows = prepare.select_gsm8k(
            source_rows,
            count=count,
            seed=selection["seed"],
        )
        expected = selection[f"{name}_identity"]
        checks.equal(f"selection.{name}.count", len(rows), expected["sample_count"])
        checks.equal(
            f"selection.{name}.manifest_sha256",
            manifest_sha256(rows),
            expected["runtime_manifest_sha256"],
        )
        checks.equal(
            f"selection.{name}.ids_sha256",
            selected_ids_sha256(rows),
            expected["selected_ids_sha256"],
        )
        checks.equal(f"selection.{name}.first_id", rows[0]["id"], expected["first_id"])
        checks.equal(f"selection.{name}.last_id", rows[-1]["id"], expected["last_id"])

    checks.equal(
        "final.required",
        config["final_protocol"],
        {
            "max_model_len": 32768,
            "reasoning_effort": "max",
            "accuracy_total": 2360,
            "wikitext2_total": 1,
            "required": True,
        },
    )
    report = {
        "format_version": 1,
        "status": "passed" if checks.passed else "failed",
        "checks": checks.items,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if checks.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
