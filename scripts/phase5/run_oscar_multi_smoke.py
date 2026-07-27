#!/usr/bin/env python3
"""Run eight concurrent >320-token OSCAR service requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def rendered_tokens(tokenizer: Any, text: str) -> int:
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort="high",
        enable_thinking=True,
    )
    if hasattr(tokens, "keys") and "input_ids" in tokens:
        tokens = tokens["input_ids"]
    return len(tokens)


def prompt_for_target(tokenizer: Any, target: int) -> str:
    prefix = "请阅读下面的固定文本，并只回答“完成”。\n"
    unit = "这是并发三池缓存隔离验证文本。 "
    low, high = 1, target
    best = prefix
    while low <= high:
        middle = (low + high) // 2
        candidate = prefix + unit * middle
        if rendered_tokens(tokenizer, candidate) <= target:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def post_json(
    *,
    url: str,
    body: dict[str, Any],
    barrier: threading.Barrier,
    timeout: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    barrier.wait()
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read().decode(errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw_error": raw}
    return {
        "http_status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "response": payload,
    }


def sha256_json(value: Any) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001",
    )
    parser.add_argument(
        "--served-model",
        default="glm-5.2-fp8-pruned-reap-e154",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.requests <= 1:
        raise ValueError("--requests must be greater than one")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    base_prompt = prompt_for_target(tokenizer, 500)
    url = f"{args.base_url.rstrip('/')}/chat/completions"
    barrier = threading.Barrier(args.requests)
    bodies = []
    for index in range(args.requests):
        prompt = f"{base_prompt}\n请求编号：{index}"
        bodies.append(
            {
                "model": args.served_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42 + index,
                "max_tokens": 64,
            }
        )

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.requests) as executor:
        futures = [
            executor.submit(
                post_json,
                url=url,
                body=body,
                barrier=barrier,
                timeout=1200,
            )
            for body in bodies
        ]
        next_progress = 600
        while any(not future.done() for future in futures):
            concurrent.futures.wait(futures, timeout=60)
            elapsed = int(time.monotonic() - started)
            if elapsed >= next_progress:
                completed = sum(future.done() for future in futures)
                print(
                    json.dumps(
                        {
                            "status": "running",
                            "completed": completed,
                            "total": len(futures),
                            "elapsed_seconds": elapsed,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                next_progress += 600
        responses = [future.result() for future in futures]

    results = []
    all_passed = True
    for index, (body, response) in enumerate(zip(bodies, responses, strict=True)):
        usage = response["response"].get("usage", {})
        passed = (
            response["http_status"] == 200
            and usage.get("prompt_tokens", 0) > 320
            and usage.get("completion_tokens", 0) > 0
        )
        all_passed = all_passed and passed
        results.append(
            {
                "request_index": index,
                "status": "passed" if passed else "failed",
                "request_sha256": sha256_json(body),
                "local_prompt_tokens": rendered_tokens(
                    tokenizer,
                    body["messages"][0]["content"],
                ),
                "http_status": response["http_status"],
                "elapsed_seconds": response["elapsed_seconds"],
                "usage": usage,
                "response": response["response"],
            }
        )

    output = {
        "format_version": 1,
        "status": "passed" if all_passed else "failed",
        "requests": args.requests,
        "base_url": args.base_url,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "requests": args.requests,
                "passed": sum(item["status"] == "passed" for item in results),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
