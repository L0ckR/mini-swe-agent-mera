"""Contract, archive safety, and honest benchmark accounting regression tests."""

import json
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.package_submission import package_submission
from scripts.summarize_run import summarize_run
from scripts.validate_submission import ValidationError, validate_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "submission_stage2"
        (self.source / "inference").mkdir(parents=True)
        (self.source / "custom_agent").mkdir()
        (self.source / "submission.yaml").write_text("agent: custom_agent\nmodel: test-model\n")
        (self.source / "inference" / "Dockerfile").write_text("FROM example/server:1.0\n")
        (self.source / "inference" / ".dockerignore").write_text("__pycache__/\n")
        (self.source / "inference" / "model.json").write_text(json.dumps({
            "model_id": "Example/Model", "revision": "a" * 40,
            "served_model_name": "test-model", "vllm_version": "0.31.0",
        }))
        (self.source / "custom_agent" / "__init__.py").write_text("")
        (self.source / "custom_agent" / "react_bash_agent.py").write_text("class ReactBashAgent:\n    pass\n")

    def test_archive_is_deterministic_and_rooted_and_omits_generated_files(self):
        (self.source / "inference" / "test.sh").write_text("trusted organizer code")
        (self.source / "custom_agent" / "__pycache__").mkdir()
        (self.source / "custom_agent" / "__pycache__" / "agent.pyc").write_bytes(b"cache")
        first, second = self.root / "first.zip", self.root / "second.zip"
        one = package_submission(self.source, first)
        two = package_submission(self.source, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(one["sha256"], two["sha256"])
        with zipfile.ZipFile(first) as archive:
            self.assertIn("submission.yaml", archive.namelist())
            self.assertIn("inference/.dockerignore", archive.namelist())
            self.assertNotIn("inference/test.sh", archive.namelist())
            self.assertFalse(any("__pycache__" in name for name in archive.namelist()))
        self.assertEqual(validate_submission(first).model, "test-model")
        self.assertEqual(one["validation"], "static_only")

    def test_rejects_extra_yaml_keys_and_duplicate_keys(self):
        yaml_file = self.source / "submission.yaml"
        yaml_file.write_text("agent: custom_agent\nmodel: test-model\nusername: user\n")
        with self.assertRaisesRegex(ValidationError, "exactly"):
            validate_submission(self.source)
        yaml_file.write_text("agent: custom_agent\nmodel: test-model\nmodel: other\n")
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            validate_submission(self.source)

    def test_nested_fork_round_trips_with_license_provenance_and_async_import(self):
        vendor = self.source / "custom_agent" / "_vendor"
        package = vendor / "forklib"
        package.mkdir(parents=True)
        (vendor / "__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "core.py").write_text("class Solver:\n    async def run(self):\n        return 'fork-ready'\n")
        license_text = "MIT License\nCopyright (c) Upstream authors\n"
        (package / "LICENSE.md").write_text(license_text)
        (package / "NOTICE").write_text("Contains an adapted open-source component.\n")
        upstream = {"repository": "https://github.com/example/harness", "commit": "b" * 40}
        (package / "UPSTREAM.json").write_text(json.dumps(upstream))
        (package / "prompt.jinja").write_text("Solve {{ task }}")
        (package / "config.yaml").write_text("step_limit: 100\n")
        (self.source / "custom_agent" / "react_bash_agent.py").write_text(
            "from ._vendor.forklib.core import Solver as ReactBashAgent\n"
        )
        report = validate_submission(self.source)
        self.assertIn("custom_agent/_vendor/forklib/core.py", report.files)
        archive_path = self.root / "fork.zip"
        package_submission(self.source, archive_path)
        self.assertEqual(validate_submission(archive_path).files, report.files)
        extracted = self.root / "extracted"
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(archive.read("custom_agent/_vendor/forklib/LICENSE.md").decode(), license_text)
            self.assertEqual(json.loads(archive.read("custom_agent/_vendor/forklib/UPSTREAM.json")), upstream)
            archive.extractall(extracted)
        result = subprocess.run(
            [sys.executable, "-I", "-c",
             "import asyncio,sys; sys.path.insert(0,sys.argv[1]); "
             "from custom_agent.react_bash_agent import ReactBashAgent; "
             "assert asyncio.run(ReactBashAgent().run()) == 'fork-ready'", str(extracted)],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)

    def test_nested_agent_python_syntax_is_validated(self):
        package = self.source / "custom_agent" / "_vendor" / "forklib"
        package.mkdir(parents=True)
        (package / "broken.py").write_text("async def broken(:\n    pass\n")
        with self.assertRaisesRegex(ValidationError, "invalid Python syntax.*_vendor/forklib/broken.py"):
            validate_submission(self.source)

    def test_nested_credentials_and_unsafe_zip_members_remain_rejected(self):
        package = self.source / "custom_agent" / "_vendor" / "forklib"
        package.mkdir(parents=True)
        for name in ("credentials.json", ".env.json", "id_ed25519", "snapshot.safetensors"):
            with self.subTest(name=name):
                entry = package / name
                entry.write_text("do not upload\n")
                with self.assertRaises(ValidationError):
                    validate_submission(self.source)
                entry.unlink()
        output = self.root / "bad-fork.zip"
        for name in (
            "custom_agent/_vendor/forklib/__pycache__/core.pyc",
            "custom_agent/_vendor/forklib/.env/config.json",
            "custom_agent/_vendor/forklib/../../../../outside.py",
        ):
            with self.subTest(name=name):
                with zipfile.ZipFile(output, "w") as archive:
                    archive.writestr(name, "ignored")
                with self.assertRaises(ValidationError):
                    validate_submission(output)
    def test_rejects_model_alias_mismatch_and_unpinned_revision(self):
        manifest_file = self.source / "inference" / "model.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["served_model_name"] = "other-model"
        manifest_file.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValidationError, "alias differs"):
            validate_submission(self.source)
        manifest["served_model_name"] = "test-model"
        manifest["revision"] = "main"
        manifest_file.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValidationError, "pinned"):
            validate_submission(self.source)

    def test_rejects_symlinks_weights_and_embedded_secrets(self):
        unsafe = self.source / "inference" / "linked.py"
        unsafe.symlink_to(self.source / "inference" / "Dockerfile")
        with self.assertRaisesRegex(ValidationError, "regular files"):
            validate_submission(self.source)
        unsafe.unlink()
        weights = self.source / "inference" / "model.safetensors"
        weights.write_bytes(b"weights")
        with self.assertRaisesRegex(ValidationError, "weights"):
            validate_submission(self.source)
        weights.unlink()
        (self.source / "inference" / "download.py").write_text("token = 'hf_" + "X" * 35 + "'\n")
        with self.assertRaisesRegex(ValidationError, "credential"):
            validate_submission(self.source)

    def test_rejects_zip_traversal_symlinks_and_wrapper(self):
        output = self.root / "bad.zip"
        for name in ("../submission.yaml", "/submission.yaml", "wrapper/submission.yaml", "inference\\file.py"):
            with self.subTest(name=name):
                with zipfile.ZipFile(output, "w") as archive:
                    archive.writestr(name, "model: x\n")
                with self.assertRaises(ValidationError):
                    validate_submission(output)
        with zipfile.ZipFile(output, "w") as archive:
            info = zipfile.ZipInfo("inference/link.py")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "../../outside")
        with self.assertRaisesRegex(ValidationError, "symlink"):
            validate_submission(output)

    def test_rejects_archive_duplicate_entries_and_trusted_test(self):
        output = self.root / "bad.zip"
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("inference/test.sh", "trusted")
        with self.assertRaisesRegex(ValidationError, "trusted"):
            validate_submission(output)


class RunSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        (self.run / "config.json").write_text('{"n_attempts": 1}')
        (self.run / "result.json").write_text(json.dumps({
            "n_total_trials": 3, "finished_at": "2026-11-18T00:00:00Z", "stats": {},
        }))

    def trial(self, name, reward=None, error=None, finished=True):
        path = self.run / name
        (path / "verifier").mkdir(parents=True)
        (path / "result.json").write_text(json.dumps({
            "task_name": name.split("__", 1)[0],
            "finished_at": "2026-11-18T00:00:00Z" if finished else None,
            "exception_info": {"exception_type": error} if error else None,
        }))
        if reward is not None:
            (path / "verifier" / "reward.txt").write_text(str(reward))

    def test_denominator_includes_missing_trials_and_timeout_is_visible(self):
        self.trial("py-001__one", 1)
        self.trial("py-002__two", error="AgentTimeoutError")
        summary = summarize_run(self.run)
        self.assertAlmostEqual(summary["pass_at_1"], 1 / 3)
        self.assertEqual(summary["n_missing_trial_directories"], 1)
        self.assertEqual(summary["n_missing_rewards"], 1)
        self.assertEqual(summary["n_timeouts"], 1)
        self.assertFalse(summary["complete"])
        self.assertTrue(summary["score_is_provisional"])

    def test_complete_run_recomputes_rewards_and_rejects_invalid_reward(self):
        self.trial("a__one", 1)
        self.trial("b__two", 0)
        self.trial("c__three", 1)
        before = (self.run / "result.json").read_bytes()
        summary = summarize_run(self.run)
        self.assertTrue(summary["complete"])
        self.assertAlmostEqual(summary["pass_at_1"], 2 / 3)
        self.assertEqual((self.run / "result.json").read_bytes(), before)
        (self.run / "c__three" / "verifier" / "reward.txt").write_text("nan")
        summary = summarize_run(self.run)
        self.assertEqual(summary["n_invalid_rewards"], 1)
        self.assertFalse(summary["complete"])

    def test_multiple_attempts_and_duplicate_tasks_are_not_pass_at_one(self):
        self.trial("a__one", 1)
        self.trial("a__two", 1)
        self.assertIsNone(summarize_run(self.run)["pass_at_1"])
        (self.run / "config.json").write_text('{"n_attempts": 2}')
        self.assertIsNone(summarize_run(self.run)["pass_at_1"])


if __name__ == "__main__":
    unittest.main()
