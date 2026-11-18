"""CPU-only checks for the model build context and inference launch contract."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
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
        self.assertEqual(command[command.index("--served-model-name") + 1], manifest["served_model_name"])
        self.assertEqual(command[command.index("--tokenizer") + 1], manifest["model_directory"])
        defaults = json.loads(command[command.index("--default-chat-template-kwargs") + 1])
        self.assertIs(defaults["enable_thinking"], False)
        self.assertIn("--enable-auto-tool-choice", command)
        self.assertEqual(command[command.index("--tool-call-parser") + 1], "qwen3_coder")

    def test_local_gpu_override_preserves_submitted_model_alias(self):
        manifest = serve.load_manifest()
        command = serve.build_command(manifest, {
            "SWE_MERA_DATA_PARALLEL_SIZE": "1",
            "SWE_MERA_MAX_MODEL_LEN": "32768",
            "SWE_MERA_MAX_NUM_SEQS": "2",
        })
        self.assertEqual(command[command.index("--data-parallel-size") + 1], "1")
        self.assertEqual(command[command.index("--max-model-len") + 1], "32768")
        self.assertEqual(command[command.index("--served-model-name") + 1], manifest["served_model_name"])

    def test_invalid_local_launch_settings_fail_before_loading_model(self):
        manifest = serve.load_manifest()
        for settings in (
            {"SWE_MERA_DATA_PARALLEL_SIZE": "0"},
            {"SWE_MERA_MAX_MODEL_LEN": "262145"},
            {"SWE_MERA_MAX_NUM_SEQS": "-1"},
            {"SWE_MERA_GPU_MEMORY_UTILIZATION": "1.1"},
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
