#!/usr/bin/env python3
"""Bake exactly the manifest's model revision into the inference image."""

from __future__ import annotations

import json
from pathlib import Path
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


def main() -> None:
    from huggingface_hub import snapshot_download

    manifest = load_manifest()
    directory = Path(manifest["model_directory"])
    # Small metadata, tokenizer, model card/license and safetensors only. This
    # model is supported by vLLM itself and requires no downloaded Python code.
    patterns = ["*.json", "*.jinja", "*.safetensors", "*.model", "LICENSE", "README.md"]
    for attempt in range(3):
        try:
            snapshot_download(
                repo_id=manifest["model_id"],
                revision=manifest["revision"],
                local_dir=str(directory),
                allow_patterns=patterns,
                max_workers=4,
            )
            validate_snapshot(directory)
            (directory / "swe_mera_snapshot.json").write_text(
                json.dumps({key: manifest[key] for key in ("model_id", "revision")}, indent=2) + "\n"
            )
            print(f"Baked {manifest['model_id']} at {manifest['revision']}", flush=True)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))


if __name__ == "__main__":
    main()
