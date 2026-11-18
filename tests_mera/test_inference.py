"""CPU-only checks for the model build context and inference launch contract."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch


INFERENCE = Path(__file__).resolve().parents[1] / "submission_stage2" / "inference"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


serve = load_module("mera_inference_serve", INFERENCE / "serve.py")
with patch.dict(sys.modules, {"serve": serve}):
    download = load_module("mera_inference_download", INFERENCE / "download_model.py")


class InferenceContractTests(unittest.TestCase):
    def test_dry_run_works_without_gpu_or_vllm(self):
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("SWE_MERA_")}
        result = subprocess.run(
            [sys.executable, str(INFERENCE / "serve.py"), "--dry-run"],
            env=clean_env, text=True, capture_output=True, check=True,
        )
        command = json.loads(result.stdout)
        manifest = serve.load_manifest()
        self.assertEqual(command[:3], ["vllm", "serve", manifest["model_directory"]])
        self.assertEqual(command[command.index("--served-model-name") + 1], manifest["base_served_model_name"])
        self.assertEqual(command[command.index("--lora-modules") + 1], manifest["served_model_name"] + "=" + manifest["adapter"]["directory"])
        self.assertEqual(command[command.index("--tokenizer") + 1], manifest["adapter"]["directory"])
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "8001")
        self.assertEqual(command[command.index("--max-lora-rank") + 1], "32")
        self.assertEqual(command[command.index("--mamba-cache-mode") + 1], "align")
        self.assertEqual(command[command.index("--kv-cache-dtype") + 1], "auto")
        defaults = json.loads(command[command.index("--default-chat-template-kwargs") + 1])
        self.assertIs(defaults["enable_thinking"], False)
        self.assertIn("--enable-auto-tool-choice", command)
        self.assertEqual(command[command.index("--tool-call-parser") + 1], "qwen3_xml")

    def test_local_gpu_override_preserves_submitted_model_alias(self):
        manifest = serve.load_manifest()
        command = serve.build_command(manifest, {
            "SWE_MERA_DATA_PARALLEL_SIZE": "1",
            "SWE_MERA_MAX_MODEL_LEN": "32768",
            "SWE_MERA_MAX_NUM_SEQS": "2",
        })
        self.assertEqual(command[command.index("--data-parallel-size") + 1], "1")
        self.assertEqual(command[command.index("--max-model-len") + 1], "32768")
        self.assertEqual(command[command.index("--served-model-name") + 1], manifest["base_served_model_name"])
        self.assertEqual(command[command.index("--lora-modules") + 1].split("=", 1)[0], manifest["served_model_name"])

    def test_invalid_local_launch_settings_fail_before_loading_model(self):
        manifest = serve.load_manifest()
        for settings in (
            {"SWE_MERA_DATA_PARALLEL_SIZE": "0"},
            {"SWE_MERA_MAX_MODEL_LEN": "262145"},
            {"SWE_MERA_MAX_NUM_SEQS": "-1"},
            {"SWE_MERA_GPU_MEMORY_UTILIZATION": "1.1"},
            {"SWE_MERA_GPU_MEMORY_UTILIZATION": "nan"},
            {"SWE_MERA_INFERENCE_PORT": "65536"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                serve.build_command(manifest, settings)

    def test_floating_model_revision_is_rejected(self):
        manifest = serve.load_manifest()
        manifest["revision"] = "main"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "immutable"):
                serve.load_manifest(path)

    def test_adapter_pin_base_alias_and_thinking_are_checked_before_model_load(self):
        mutations = [
            lambda m: m["adapter"].update(revision="main"),
            lambda m: m["adapter"].update(weights_sha256="not-a-hash"),
            lambda m: m.update(base_served_model_name=m["served_model_name"]),
            lambda m: m.update(model_id="Qwen/Qwen3.8-27B-FP8"),
            lambda m: m["default_chat_template_kwargs"].update(enable_thinking=True),
        ]
        with tempfile.TemporaryDirectory() as temp:
            for change in mutations:
                manifest = serve.load_manifest()
                change(manifest)
                path = Path(temp) / "model.json"
                path.write_text(json.dumps(manifest))
                with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                    serve.load_manifest(path)

    def seed_adapter(self, directory, *, rank=32, tensor_rank=None):
        manifest = serve.load_manifest()
        adapter = manifest["adapter"]
        config = {"base_model_name_or_path": manifest["model_id"], "peft_type": "LORA",
                  "r": rank, "lora_alpha": adapter["alpha"], "target_modules": adapter["target_modules"]}
        (directory / "adapter_config.json").write_text(json.dumps(config))
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md"):
            (directory / name).write_text("fixture")
        size = rank if tensor_rank is None else tensor_rank
        stem = "base_model.model.model.language_model.layers.3.self_attn.q_proj.lora_"
        header = {
            stem + "A.weight": {"dtype": "F32", "shape": [size, 1], "data_offsets": [0, size * 4]},
            stem + "B.weight": {"dtype": "F32", "shape": [1, size], "data_offsets": [size * 4, size * 8]},
        }
        encoded = json.dumps(header).encode()
        weights = struct.pack("<Q", len(encoded)) + encoded + bytes(size * 8)
        (directory / "adapter_model.safetensors").write_bytes(weights)
        adapter.update(tensor_count=2, weights_bytes=len(weights), weights_sha256=hashlib.sha256(weights).hexdigest())
        return manifest

    def test_adapter_snapshot_checks_actual_tensor_pairs_and_exact_weight_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = self.seed_adapter(directory)
            download.validate_adapter(directory, manifest)
            weights = directory / "adapter_model.safetensors"
            raw = bytearray(weights.read_bytes()); raw[-1] = 1; weights.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                download.validate_adapter(directory, manifest)
            manifest = self.seed_adapter(directory, tensor_rank=16)
            with self.assertRaisesRegex(ValueError, "Unexpected LoRA tensor"):
                download.validate_adapter(directory, manifest)
            manifest = self.seed_adapter(directory, rank=16)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                download.validate_adapter(directory, manifest)

    def test_download_pins_base_and_adapter_and_includes_the_adapter_tokenizer(self):
        manifest = serve.load_manifest()
        with tempfile.TemporaryDirectory() as temp:
            manifest["model_directory"] = str(Path(temp) / "base")
            manifest["adapter"]["directory"] = str(Path(temp) / "adapter")
            requests = []

            def capture(**kwargs):
                requests.append(kwargs)
                Path(kwargs["local_dir"]).mkdir()

            with patch.object(download, "validate_snapshot"), patch.object(download, "validate_adapter"):
                download.download_snapshots(manifest, capture)
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0]["revision"], manifest["revision"])
            self.assertEqual(requests[1]["revision"], manifest["adapter"]["revision"])
            self.assertIn("adapter_model.safetensors", requests[1]["allow_patterns"])
            self.assertIn("tokenizer_config.json", requests[1]["allow_patterns"])
            self.assertIn("chat_template.jinja", requests[1]["allow_patterns"])
            self.assertFalse(any("*.py" in item["allow_patterns"] for item in requests))

    def test_snapshot_validation_rejects_missing_shards_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for filename in ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
                (directory / filename).write_text("{}")
            index = directory / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"layer.weight": "layer-0.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "missing weight shard"):
                download.validate_snapshot(directory)
            (directory / "layer-0.safetensors").write_bytes(b"test shard")
            download.validate_snapshot(directory)
            index.write_text(json.dumps({"weight_map": {"layer.weight": "../outside.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "Invalid model shard"):
                download.validate_snapshot(directory)


if __name__ == "__main__":
    unittest.main()
