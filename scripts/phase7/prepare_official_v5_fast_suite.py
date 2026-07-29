#!/usr/bin/env python3
"""Build a deterministic GSM8K runtime suite for Stage 7 fast screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def select_gsm8k(
    manifest: list[dict[str, Any]],
    *,
    count: int,
    seed: str,
) -> list[dict[str, Any]]:
    indexed = [
        (index, row)
        for index, row in enumerate(manifest)
        if row.get("benchmark") == "GSM8K"
    ]
    if count <= 0 or count > len(indexed):
        raise ValueError(f"sample count must be in [1, {len(indexed)}]")
    if count == len(indexed):
        return [row for _, row in indexed]
    ranked = sorted(
        indexed,
        key=lambda item: (
            hashlib.sha256(f"{seed}\0{item[1]['id']}".encode("utf-8")).hexdigest(),
            item[1]["id"],
        ),
    )[:count]
    return [row for _, row in sorted(ranked, key=lambda item: item[0])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--selection-seed", required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output_dir}")
    manifest = read_jsonl(args.source_manifest)
    selected = select_gsm8k(
        manifest,
        count=args.sample_count,
        seed=args.selection_seed,
    )
    manifest_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected
    )
    selected_ids_text = "\n".join(row["id"] for row in selected) + "\n"
    output_manifest = args.output_dir / "manifest.jsonl"
    output_config = args.output_dir / "eval_config.json"
    atomic_write(output_manifest, manifest_text)
    atomic_write(output_config, args.eval_config.read_text(encoding="utf-8"))
    identity = {
        "format_version": 1,
        "protocol": "official_v5_fast_screen",
        "selection_method": "sha256(seed + NUL + sample_id), then source order",
        "selection_seed": args.selection_seed,
        "sample_count": len(selected),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "runtime_manifest_sha256": sha256_file(output_manifest),
        "runtime_eval_config_sha256": sha256_file(output_config),
        "selected_ids_sha256": hashlib.sha256(
            selected_ids_text.encode("utf-8")
        ).hexdigest(),
        "first_id": selected[0]["id"],
        "last_id": selected[-1]["id"],
    }
    atomic_write(
        args.output_dir / "fast_suite_identity.json",
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(identity, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
