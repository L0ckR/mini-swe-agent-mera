#!/usr/bin/env python3
"""Launch the pinned local model; importing this module needs no GPU libraries."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
from typing import Mapping


MANIFEST_PATH = Path(__file__).with_name("model.json")


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for field in ("model_id", "revision", "served_model_name", "base_served_model_name", "vllm_version", "model_directory"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"model.json must contain a nonempty {field}")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("Model revision must be an immutable full Hugging Face commit SHA")
    adapter = manifest.get("adapter")
    if not isinstance(adapter, dict):
        raise ValueError("model.json requires the pinned Pi adapter")
    for field in ("model_id", "revision", "directory", "weights_sha256"):
        if not isinstance(adapter.get(field), str) or not adapter[field]:
            raise ValueError(f"model.json adapter requires {field}")
    if not re.fullmatch(r"[0-9a-f]{40}", adapter["revision"]):
        raise ValueError("Adapter revision must be an immutable full Hugging Face commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", adapter["weights_sha256"]):
        raise ValueError("Adapter weights require a full SHA256 digest")
    for field in ("rank", "alpha", "tensor_count", "weights_bytes"):
        positive_int(adapter.get(field), f"adapter.{field}")
    if adapter.get("source_dtype") != "F32" or set(adapter.get("target_modules") or []) != {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }:
        raise ValueError("The pinned Pi adapter requires its original F32 weights and seven target projections")
    if manifest["served_model_name"] == manifest["base_served_model_name"]:
        raise ValueError("Base and LoRA model aliases must differ to prevent accidental base-model inference")
    if manifest.get("dtype") != "bfloat16" or manifest["model_id"].endswith("-FP8"):
        raise ValueError("This Pi adapter must run on its BF16 base model")
    if manifest.get("default_chat_template_kwargs", {}).get("enable_thinking") is not False:
        raise ValueError("The Pi adapter was trained with thinking disabled")
    return manifest


def positive_int(value: str | int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a positive integer") from None
    if number <= 0 or isinstance(value, float) and value != number:
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
    if not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1:
        raise ValueError("SWE_MERA_GPU_MEMORY_UTILIZATION must be in (0, 1]")
    port = positive_int(env.get("SWE_MERA_INFERENCE_PORT", "8001"), "SWE_MERA_INFERENCE_PORT")
    if port > 65535:
        raise ValueError("SWE_MERA_INFERENCE_PORT must be at most 65535")
    adapter = manifest["adapter"]
    command = [
        "vllm", "serve", manifest["model_directory"],
        "--host", "127.0.0.1", "--port", str(port), "--api-key", "dummy",
        "--served-model-name", manifest["base_served_model_name"],
        "--tokenizer", adapter["directory"],
        "--chat-template", str(Path(adapter["directory"]) / "chat_template.jinja"),
        "--enable-lora", "--max-lora-rank", str(adapter["rank"]), "--max-loras", "1",
        "--lora-dtype", "auto",
        "--lora-modules", manifest["served_model_name"] + "=" + adapter["directory"],
        "--tensor-parallel-size", "1",
        "--data-parallel-size", str(parallel_size),
        "--dtype", manifest["dtype"],
        "--language-model-only",
        "--attention-backend", "flashinfer",
        "--block-size", "256",
        "--enable-prefix-caching",
        "--mamba-cache-mode", manifest["mamba_cache_mode"],
        "--enable-chunked-prefill",
        "--enable-prompt-tokens-details",
        "--kv-cache-dtype", manifest["kv_cache_dtype"],
        "--max-model-len", str(context_length),
        "--max-num-seqs", str(sequences),
        "--max-num-batched-tokens", str(manifest["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(memory_fraction),
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice", "--tool-call-parser", manifest["tool_call_parser"],
        # Preserve the Pi adapter's training mode and the organizer's short
        # tool-call smoke check. The agent must not turn thinking on.
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
