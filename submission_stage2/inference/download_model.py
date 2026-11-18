#!/usr/bin/env python3
"""Bake exactly the manifest's model revision into the inference image."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct
import time

from serve import load_manifest


def validate_snapshot(directory: Path) -> None:
    """Fail the image build if a tokenizer or any referenced weight shard is absent."""
    required = ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "model.safetensors.index.json")
    for filename in required:
        path = directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Incomplete model snapshot: missing {filename}")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    if not shards:
        raise ValueError("Model index has no weight shards")
    for filename in shards:
        path = directory / filename
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"Invalid model shard name: {filename!r}")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Incomplete model snapshot: missing weight shard {filename}")


def validate_adapter(directory: Path, manifest: dict) -> None:
    """Check the actual pinned adapter, not just the presence of a weight file."""
    adapter = manifest["adapter"]
    required = ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja", "README.md")
    for filename in required:
        path = directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Incomplete adapter snapshot: missing {filename}")
    config = json.loads((directory / "adapter_config.json").read_text())
    if (config.get("base_model_name_or_path") != manifest["model_id"]
            or config.get("peft_type") != "LORA"
            or config.get("r") != adapter["rank"]
            or config.get("lora_alpha") != adapter["alpha"]
            or set(config.get("target_modules") or []) != set(adapter["target_modules"])
            or config.get("use_dora") or config.get("use_rslora")
            or config.get("modules_to_save")):
        raise ValueError("Adapter configuration differs from the pinned Pi policy")
    weights = directory / "adapter_model.safetensors"
    if weights.stat().st_size != adapter["weights_bytes"]:
        raise ValueError("Adapter weight file size differs from the pinned artifact")
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Adapter safetensors header is missing")
        header_length = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_length <= 4 * 1024 * 1024:
            raise ValueError("Invalid adapter safetensors header length")
        header = json.loads(handle.read(header_length))
        handle.seek(0)
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != adapter["weights_sha256"]:
        raise ValueError("Adapter weight SHA256 differs from the pinned artifact")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if len(tensors) != adapter["tensor_count"]:
        raise ValueError("Adapter has an unexpected number of LoRA tensors")
    offsets, pairs = [], {}
    pattern = re.compile(r"^base_model\.model\.model\.language_model\.layers\.\d+\."
                         r"(?:mlp|self_attn)\.([a-z_]+)\.lora_([AB])\.weight$")
    for key, tensor in tensors.items():
        match = pattern.fullmatch(key)
        shape = tensor.get("shape", [])
        if (not match or match.group(1) not in adapter["target_modules"]
                or tensor.get("dtype") != adapter["source_dtype"]
                or len(shape) != 2 or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
                or shape[0 if match.group(2) == "A" else 1] != adapter["rank"]):
            raise ValueError(f"Unexpected LoRA tensor: {key}")
        span = tensor.get("data_offsets", [])
        if (len(span) != 2 or any(isinstance(n, bool) or not isinstance(n, int) for n in span)
                or span[0] < 0 or span[1] - span[0] != shape[0] * shape[1] * 4):
            raise ValueError(f"Invalid LoRA tensor offsets: {key}")
        offsets.append(span)
        stem = key.rsplit(".lora_", 1)[0]
        pairs.setdefault(stem, set()).add(match.group(2))
    expected = 0
    for start, end in sorted(offsets):
        if start != expected:
            raise ValueError("Adapter tensors contain a gap or overlapping data")
        expected = end
    if expected + 8 + header_length != weights.stat().st_size or any(pair != {"A", "B"} for pair in pairs.values()):
        raise ValueError("Adapter tensor pairs or safetensors payload are incomplete")


def download_snapshots(manifest: dict, snapshot_download) -> None:
    """All artifacts are fetched at image build time using immutable revisions."""
    directory = Path(manifest["model_directory"])
    adapter = manifest["adapter"]
    patterns = ["*.json", "*.jinja", "*.safetensors", "*.model", "*.txt", "LICENSE", "README.md"]
    jobs = [
        (manifest["model_id"], manifest["revision"], directory, patterns, lambda: validate_snapshot(directory)),
        (adapter["model_id"], adapter["revision"], Path(adapter["directory"]),
         ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE", "README.md"],
         lambda: validate_adapter(Path(adapter["directory"]), manifest)),
    ]
    for model_id, revision, target, allowed, validate in jobs:
        for attempt in range(3):
            try:
                snapshot_download(repo_id=model_id, revision=revision, local_dir=str(target),
                                  allow_patterns=allowed, max_workers=4)
                validate()
                (target / "swe_mera_snapshot.json").write_text(json.dumps({"model_id": model_id, "revision": revision}, indent=2) + "\n")
                print(f"Baked {model_id} at {revision}", flush=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(10 * (attempt + 1))


def main() -> None:
    from huggingface_hub import snapshot_download

    download_snapshots(load_manifest(), snapshot_download)


if __name__ == "__main__":
    main()
