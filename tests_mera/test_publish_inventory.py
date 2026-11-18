from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.manage_harness_fork import collect_publish_files, prepare_diagnostic_evidence


class PublishInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.write("submission_stage2/submission.yaml", "agent: custom_agent\nmodel: local\n")
        self.write("submission_stage2/inference/Dockerfile", "FROM scratch\n")
        self.write("submission_stage2/custom_agent/__init__.py", "")
        self.write("submission_stage2/custom_agent/react_bash_agent.py", "class ReactBashAgent: pass\n")

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_sources_fixtures_public_provenance_and_curated_reports_are_selected(self):
        statement = self.write("data/public_statements/py-001.md", "Public issue only.\n")
        self.write("data/public_task_inventory.json", json.dumps({"n_public_tasks": 1, "tasks": [{
            "task_id": "py-001", "instruction_path": "data/public_statements/py-001.md",
            "instruction_sha256": hashlib.sha256(statement.read_bytes()).hexdigest(),
        }]}))
        for name in ("experiments/diagnostics/cases.json", "experiments/diagnostics/public_statement_wrapper.json",
                     "experiments/results/diagnostics-live-001.json"):
            self.write(name, "{}")
        self.write("scripts/run_local_diagnostics.py", "# source\n")
        self.write("scripts/manage_harness_fork.py", "# source\n")
        self.write("tests/test_local_environment.py", "# test\n")
        self.write("jobs/private/result.json", '{"credential":"private"}')
        self.write("local_runs/repo/secret.py", "private\n")
        self.write("experiments/diagnostics/checkouts/repo/solution.py", "private\n")
        self.write("experiments/results/cache/trajectory.json", "private\n")
        self.write("scripts/.env", "private\n")
        names = [name for name, _, _ in collect_publish_files(self.root)]
        self.assertIn("scripts/run_local_diagnostics.py", names)
        self.assertIn("scripts/manage_harness_fork.py", names)
        self.assertIn("tests_mera/test_local_environment.py", names)
        self.assertIn("data/public_task_inventory.json", names)
        self.assertNotIn("data/public_statements/py-001.md", names)
        self.assertIn("experiments/diagnostics/cases.json", names)
        self.assertIn("experiments/results/diagnostics-live-001.json", names)
        self.assertFalse(any("jobs/" in name or "checkouts/" in name or "cache/" in name or name.endswith(".env") for name in names))
        self.assertEqual(names, sorted(set(names)))

    def test_external_statements_are_not_needed_for_publication(self):
        self.write("data/public_task_inventory.json", json.dumps({"n_public_tasks": 1, "tasks": [{
            "task_id": "py-001", "instruction_path": "data/public_statements/py-001.md", "instruction_sha256": "0" * 64,
        }]}))
        names = [name for name, _, _ in collect_publish_files(self.root)]
        self.assertIn("data/public_task_inventory.json", names)
        self.assertFalse(any(name.startswith("data/public_statements/") for name in names))

    def test_credential_in_an_approved_report_is_rejected(self):
        self.write("experiments/results/result.json", json.dumps({"token": "hf_" + "A" * 40}))
        with self.assertRaisesRegex(RuntimeError, "Possible credential"):
            collect_publish_files(self.root)

    def test_approved_source_symlink_is_rejected(self):
        outside = self.write("private.py", "private\n")
        (self.root / "scripts").mkdir()
        (self.root / "scripts/linked.py").symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            collect_publish_files(self.root)

    def test_promoted_evidence_is_compact_and_excludes_credentials_context_and_raw_logs(self):
        report = {"kind": "NON_HARBOR_SYNTHETIC_DIAGNOSTIC", "by_variant": {"after": {"resolved": 0, "total": 1}},
                  "model": "alias", "api_key": "private-top-level", "backend_metadata": {"model": "actual-model", "api_key": "private-backend"},
                  "trials": [{"variant": "after", "case": "synthetic", "resolved": False, "elapsed_sec": 2,
                              "context": {"secret": "private-context"}, "agent_error": "private-error",
                              "acceptance": {"passed": 0, "total": 1, "stderr": "private-log", "checks": [{"criterion": "behavior", "passed": False}]}}]}
        source = self.write("jobs/diagnostics-live-001/summary.json", json.dumps(report))
        self.write("jobs/private/summary.json", json.dumps(report))
        paths = prepare_diagnostic_evidence(self.root)
        self.assertEqual(len(paths), 1)
        saved = json.loads(paths[0].read_text())
        self.assertEqual(saved["backend_metadata"]["model"], "actual-model")
        self.assertEqual(saved["raw_summary_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertTrue(saved["trials"][0]["agent_failed"])
        self.assertIsNone(saved["pass_at_1"])
        text = paths[0].read_text()
        for excluded in ("private-top-level", "private-backend", "private-context", "private-error", "private-log"):
            self.assertNotIn(excluded, text)

    def test_incomplete_or_foreign_runs_are_not_promoted(self):
        self.write("jobs/diagnostics-live-001/summary.json", '{"kind":"NON_HARBOR_SYNTHETIC_DIAGNOSTIC","trials":[]}')
        self.write("jobs/diagnostics-live-002/summary.json", '{"kind":"PRIVATE_EVALUATION","by_variant":{"x":{}}}')
        self.assertEqual(prepare_diagnostic_evidence(self.root), [])

    def test_public_click_evidence_preserves_delivery_failures_without_raw_logs(self):
        report = {"kind": "NON_HARBOR_PUBLIC_CLICK_DIAGNOSTIC", "resolved_public_diagnostic": False,
                  "official_verifier": "NOT_RUN", "pass_at_1": None, "public_statement": "py-009",
                  "local_inference": {"model": "actual", "api_key": "private-backend"},
                  "agent_error": "private-error", "before": {"visible_tests": {"passed_count": 11, "stdout": "private-log"}},
                  "delivered_patch": {"valid": False, "nonempty": False, "stderr": "private-patch-error"}}
        self.write("jobs/public-click-live-001/result.json", json.dumps(report))
        paths = prepare_diagnostic_evidence(self.root)
        self.assertEqual(len(paths), 1)
        saved = json.loads(paths[0].read_text())
        self.assertFalse(saved["resolved_public_diagnostic"])
        self.assertFalse(saved["delivered_patch"]["valid"])
        self.assertEqual(saved["before"]["visible_tests"]["passed_count"], 11)
        self.assertTrue(saved["agent_failed"])
        self.assertIsNone(saved["pass_at_1"])
        self.assertNotIn("private-", paths[0].read_text())
