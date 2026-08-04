#!/usr/bin/env python3
"""Run a strictly aligned teacher-forced replay for hidden-state capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping
from urllib import request


MODEL_NAME = "glm-5.2-fp8-pruned-reap-e154"
CAPTURE_HOOK = "model.layers.36.post_attention_layernorm"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_ids_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_validated_replay(
    selection_path: Path,
    server_validation_path: Path,
) -> list[dict[str, Any]]:
    selection = _read_json(selection_path)
    if (
        selection.get("classification")
        != "hidden_similarity_teacher_forced_replay_selection"
        or selection.get("promotion_gate") is not False
        or selection.get("min_position") != 320
        or selection.get("eligible_count") != 9
    ):
        raise ValueError("invalid replay selection contract")
    samples = selection.get("samples")
    if not isinstance(samples, list) or len(samples) != 9:
        raise ValueError("replay selection must contain exactly nine samples")
    sample_ids = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("replay sample must be an object")
        sample_id = sample.get("id")
        token_ids = sample.get("token_ids")
        if not isinstance(sample_id, str) or sample_id in sample_ids:
            raise ValueError("replay sample IDs must be unique strings")
        if (
            not isinstance(token_ids, list)
            or not all(isinstance(value, int) for value in token_ids)
            or len(token_ids) < 320
            or sample.get("token_count") != len(token_ids)
            or sample.get("token_ids_sha256") != token_ids_sha256(token_ids)
        ):
            raise ValueError(f"invalid replay token contract: {sample_id}")
        sample_ids.append(sample_id)

    validation = _read_json(server_validation_path)
    if (
        validation.get("classification")
        != "hidden_similarity_server_token_ids_validation"
        or validation.get("status") != "passed"
        or validation.get("capture_ready") is not True
        or validation.get("promotion_gate") is not False
    ):
        raise ValueError("server validation is not capture-ready")
    if validation.get("selection_sha256") != sha256_file(selection_path):
        raise ValueError("server validation selection identity mismatch")
    server_samples = validation.get("samples")
    if not isinstance(server_samples, list) or len(server_samples) != 9:
        raise ValueError("server validation must contain exactly nine samples")
    indexed = {sample.get("id"): sample for sample in server_samples}
    if set(indexed) != set(sample_ids):
        raise ValueError("server validation sample identity mismatch")
    for sample in samples:
        server = indexed[sample["id"]]
        digest = sample["token_ids_sha256"]
        if (
            server.get("exact_token_ids") is not True
            or server.get("expected_count") != sample["token_count"]
            or server.get("server_count") != sample["token_count"]
            or server.get("server_max_model_len") != 8192
            or server.get("expected_token_ids_sha256") != digest
            or server.get("server_token_ids_sha256") != digest
        ):
            raise ValueError(
                f"server token identity is not exact: {sample['id']}"
            )
    return samples


def completion_request(*, model: str, token_ids: list[int]) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "seed": 42,
    }


def validate_capture_environment(
    actual: Mapping[str, str],
    expected: Mapping[str, str],
) -> None:
    mismatches = {
        name: {"actual": actual.get(name), "expected": value}
        for name, value in expected.items()
        if actual.get(name) != value
    }
    if mismatches:
        raise ValueError(f"capture environment mismatch: {mismatches}")


def _is_scoped(path: Path, project_root: Path) -> bool:
    roots = (project_root / "artifacts", Path("/dev/shm"))
    return any(path == root or path.is_relative_to(root) for root in roots)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(body, separators=(",", ":")).encode()
    http_request = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=300) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise RuntimeError("completion response must be an object")
    return value


def _validate_captures(
    capture_dir: Path,
    replay: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import torch

    paths = sorted(capture_dir.glob("*.pt"))
    if len(paths) != len(replay):
        raise RuntimeError(
            f"expected {len(replay)} capture files, found {len(paths)}"
        )
    indexed = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise RuntimeError(f"capture payload is not an object: {path}")
        counter = payload.get("counter")
        if not isinstance(counter, int) or counter in indexed:
            raise RuntimeError(f"invalid or duplicate capture counter: {path}")
        indexed[counter] = (path, payload)
    if set(indexed) != set(range(len(replay))):
        raise RuntimeError(f"capture counter sequence mismatch: {sorted(indexed)}")

    rows = []
    for counter, sample in enumerate(replay):
        path, payload = indexed[counter]
        hidden_states = payload.get("hidden_states")
        positions = payload.get("positions")
        expected_count = sample["token_count"]
        if (
            payload.get("hook") != CAPTURE_HOOK
            or payload.get("tp_rank") != 0
            or payload.get("pp_rank") != 0
            or payload.get("capture_mode") != "aux_runner"
            or not isinstance(hidden_states, torch.Tensor)
            or not isinstance(positions, torch.Tensor)
            or hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]
            != expected_count
            or not torch.equal(
                positions.reshape(-1).cpu(), torch.arange(expected_count)
            )
        ):
            raise RuntimeError(
                f"capture payload does not match replay sample: {sample['id']}"
            )
        rows.append(
            {
                "counter": counter,
                "id": sample["id"],
                "token_count": expected_count,
                "capture_path": str(path.resolve()),
                "capture_sha256": sha256_file(path),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--server-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    selection = args.selection.resolve()
    server_validation = args.server_validation.resolve()
    output_dir = args.output_dir.resolve()
    for label, path in (
        ("selection", selection),
        ("server validation", server_validation),
        ("output directory", output_dir),
    ):
        if not _is_scoped(path, project_root):
            raise SystemExit(f"{label} must be under artifacts/ or /dev/shm: {path}")
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    if args.base_url not in {
        "http://127.0.0.1:18083/v1",
        "http://127.0.0.1:18084/v1",
    }:
        raise SystemExit("base URL must be the isolated Stage 9 loopback endpoint")

    replay = load_validated_replay(selection, server_validation)
    capture_dir = output_dir / "captures"
    enable_file = output_dir / "capture.enable"
    expected_environment = {
        "VLLM_NORM_CAPTURE_MODE": "aux_runner",
        "VLLM_NORM_CAPTURE_DIR": str(capture_dir),
        "VLLM_NORM_CAPTURE_ENABLE_FILE": str(enable_file),
        "VLLM_NORM_CAPTURE_TP_RANK": "0",
        "VLLM_NORM_CAPTURE_LAYER": "36",
        "VLLM_NORM_CAPTURE_HOOK": CAPTURE_HOOK,
    }
    validate_capture_environment(os.environ, expected_environment)
    capture_dir.mkdir(parents=True)
    identity = {
        "format_version": 1,
        "classification": "hidden_similarity_teacher_forced_capture_run",
        "variant": args.variant,
        "model": args.model,
        "selection_sha256": sha256_file(selection),
        "server_validation_sha256": sha256_file(server_validation),
        "sample_count": len(replay),
        "capture_environment": expected_environment,
        "promotion_gate": False,
    }
    _atomic_json(output_dir / "identity.json", identity)

    responses = []
    enable_file.touch(exist_ok=False)
    try:
        for counter, sample in enumerate(replay):
            body = completion_request(model=args.model, token_ids=sample["token_ids"])
            response = _post_json(
                f"{args.base_url.rstrip('/')}/completions",
                body,
            )
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise RuntimeError(f"invalid completion response: {sample['id']}")
            responses.append(
                {
                    "counter": counter,
                    "id": sample["id"],
                    "token_count": sample["token_count"],
                    "finish_reason": choices[0].get("finish_reason"),
                    "usage": response.get("usage"),
                }
            )
    finally:
        enable_file.unlink(missing_ok=True)

    deadline = time.monotonic() + 30
    while len(list(capture_dir.glob("*.pt"))) < len(replay):
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out waiting for hidden-state capture files")
        time.sleep(0.1)
    captures = _validate_captures(capture_dir, replay)
    result = {
        **identity,
        "status": "passed",
        "capture_ready": True,
        "responses": responses,
        "captures": captures,
    }
    _atomic_json(output_dir / "summary.json", result)
    print(
        json.dumps(
            {
                "status": "passed",
                "variant": args.variant,
                "sample_count": len(replay),
                "capture_count": len(captures),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
