#!/usr/bin/env python3
"""Launch the pinned local model; importing this module needs no GPU libraries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
from typing import Mapping


MANIFEST_PATH = Path(__file__).with_name("model.json")


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for field in ("model_id", "revision", "served_model_name", "vllm_version", "model_directory"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"model.json must contain a nonempty {field}")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("Model revision must be an immutable full Hugging Face commit SHA")
    return manifest


def positive_int(value: str | int, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def build_command(manifest: dict, environ: Mapping[str, str] | None = None) -> list[str]:
    """Keep the submission alias fixed while allowing local one-GPU smoke runs."""
    env = os.environ if environ is None else environ
    parallel_size = positive_int(
        env.get("SWE_MERA_DATA_PARALLEL_SIZE", manifest["data_parallel_size"]),
        "SWE_MERA_DATA_PARALLEL_SIZE",
    )
    context_length = positive_int(
        env.get("SWE_MERA_MAX_MODEL_LEN", manifest["max_model_len"]),
        "SWE_MERA_MAX_MODEL_LEN",
    )
    if context_length > 262144:
        raise ValueError("The checkpoint's native context limit is 262144 tokens")
    sequences = positive_int(
        env.get("SWE_MERA_MAX_NUM_SEQS", manifest["max_num_seqs"]),
        "SWE_MERA_MAX_NUM_SEQS",
    )
    memory_fraction = float(
        env.get("SWE_MERA_GPU_MEMORY_UTILIZATION", manifest["gpu_memory_utilization"])
    )
    if not 0 < memory_fraction <= 1:
        raise ValueError("SWE_MERA_GPU_MEMORY_UTILIZATION must be in (0, 1]")
    command = [
        "vllm", "serve", manifest["model_directory"],
        "--host", "0.0.0.0", "--port", "8000", "--api-key", "dummy",
        "--served-model-name", manifest["served_model_name"],
        "--tokenizer", manifest["model_directory"],
        "--tensor-parallel-size", "1",
        "--data-parallel-size", str(parallel_size),
        "--dtype", "bfloat16",
        "--language-model-only",
        "--attention-backend", "flashinfer",
        "--block-size", "256",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-prompt-tokens-details",
        "--kv-cache-dtype", manifest["kv_cache_dtype"],
        "--max-model-len", str(context_length),
        "--max-num-seqs", str(sequences),
        "--max-num-batched-tokens", str(manifest["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(memory_fraction),
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        # The trusted organizer smoke check has a 256-token output budget.
        # Disable thinking by default so it can produce its required tool call;
        # the custom agent enables thinking explicitly when a task needs it.
        "--default-chat-template-kwargs",
        json.dumps(manifest["default_chat_template_kwargs"], separators=(",", ":")),
        "--generation-config", "vllm",
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print argv as JSON without loading vLLM")
    args = parser.parse_args()
    command = build_command(load_manifest())
    if args.dry_run:
        print(json.dumps(command))
        return
    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
    })
    print(f"Starting inference: {shlex.join(command)}", flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
