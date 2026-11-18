from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission_stage2"))
from custom_agent.recovery import LoopRecovery


MISSING = {"returncode": 1, "changed": False,
           "output": "File tool error: ValueError: old_text was not found; read the current file and use an exact match"}
AMBIGUOUS = {"returncode": 1, "changed": False,
             "output": "File tool error: ValueError: old_text matches more than once; include more surrounding context"}
OK = {"returncode": 0, "output": "observed"}


def edit(path="src/library.py", old="missing source", new="replacement"):
    return {"name": "edit_file", "arguments": {"path": path, "old_text": old, "new_text": new}}


class LoopRecoveryTests(unittest.TestCase):
    def test_alternating_reads_do_not_hide_the_second_failed_replacement(self):
        recovery = LoopRecovery()
        self.assertEqual(recovery.observe(edit(), MISSING), "")
        for start in (1, 21, 41):
            self.assertEqual(recovery.observe({"name": "read_file", "arguments": {
                "path": "src/library.py", "start_line": start}}, OK), "")
        advice = recovery.observe(edit(path="/testbed/src/library.py", new="another attempt"), MISSING)
        self.assertIn("2 failed exact-text edits", advice)
        self.assertIn("2 used this same old_text", advice)
        self.assertIn("not a claim that the file is still unchanged", advice)
        self.assertIn("Do not retry", advice)
        self.assertIn("grep -nF --", advice)

    def test_third_failure_escalates_even_when_the_guessed_old_text_changes(self):
        recovery = LoopRecovery()
        recovery.observe(edit(old="guess one"), MISSING)
        recovery.observe(edit(old="guess two"), MISSING)
        advice = recovery.observe(edit(old="guess three"), MISSING)
        self.assertIn("3 failed exact-text edits", advice)
        self.assertIn("1 used this same old_text", advice)
        self.assertIn("reconsider the target file and root-cause hypothesis", advice)
        self.assertIn("illustrative caller snippet", advice)

    def test_successful_file_operations_clear_only_their_path(self):
        for name in ("edit_file", "create_file"):
            with self.subTest(name=name):
                recovery = LoopRecovery()
                recovery.observe(edit(), MISSING)
                recovery.observe(edit(path="other.py"), MISSING)
                recovery.observe({"name": name, "arguments": {"path": "/testbed/src/library.py"}},
                                 {**OK, "changed": True})
                self.assertEqual(recovery.observe(edit(), MISSING), "")
                self.assertIn("2 failed exact-text edits", recovery.observe(edit(path="other.py"), MISSING))

    def test_history_expires_by_tool_calls_and_has_a_fixed_bound(self):
        recovery = LoopRecovery(max_history=4)
        recovery.observe(edit(), MISSING)
        for unused in range(4):
            recovery.observe({"name": "read_file", "arguments": {"path": "another.py"}}, OK)
        self.assertEqual(recovery.observe(edit(), MISSING), "")
        self.assertEqual(len(recovery._history), 4)

    def test_ambiguous_replacements_request_unique_surrounding_context(self):
        recovery = LoopRecovery()
        self.assertEqual(recovery.observe(edit(old="shared line"), AMBIGUOUS), "")
        advice = recovery.observe(edit(old="shared line"), AMBIGUOUS)
        self.assertIn("unique occurrence", advice)

    def test_failed_commands_unrelated_edits_and_successful_tests_are_not_counted(self):
        recovery = LoopRecovery()
        for unused in range(4):
            self.assertEqual(recovery.observe(edit(), {"returncode": 1, "output": "permission denied"}), "")
            self.assertEqual(recovery.observe(edit(), {"returncode": 1, "output": "file does not exist"}), "")
            self.assertEqual(recovery.observe({"name": "run_tests", "arguments": {
                "command": "python3 -m pytest"}}, OK), "")
        self.assertEqual(recovery.observe(edit(), MISSING), "")

    def test_openai_arguments_and_malformed_inputs_are_handled(self):
        recovery = LoopRecovery()
        action = {"function": {"name": "edit_file", "arguments": json.dumps(edit()["arguments"])}}
        self.assertEqual(recovery.observe(action, MISSING), "")
        self.assertIn("2 failed exact-text edits", recovery.observe(action, MISSING))
        for action in (None, {}, {"name": []}, {"function": []},
                       {"name": "edit_file", "arguments": "not JSON"},
                       {"name": "edit_file", "arguments": "[]"}):
            self.assertEqual(recovery.observe(action, MISSING), "")
        self.assertEqual(recovery.observe(edit(), None), "")

    def test_observation_does_not_change_actions_or_results_and_bounds_snippets(self):
        recovery = LoopRecovery()
        action = edit(old="very long\n" * 10000)
        before_action, before_result = json.dumps(action), json.dumps(MISSING)
        recovery.observe(action, MISSING)
        advice = recovery.observe(action, MISSING)
        self.assertLess(len(advice), 1200)
        self.assertEqual(json.dumps(action), before_action)
        self.assertEqual(json.dumps(MISSING), before_result)

    def test_grep_help_is_reported_as_usage_without_assuming_a_search_result(self):
        recovery = LoopRecovery()
        action = {"name": "bash", "arguments": {"command": "grep -r '--an-option' src/"}}
        result = {"returncode": 0, "output": "exit_code: 0\nstdout:\nUsage: grep [OPTION]... PATTERNS [FILE]...\n"}
        advice = recovery.observe(action, result)
        self.assertIn("printed grep usage", advice)
        self.assertIn("grep -nF --", advice)
        self.assertEqual(recovery.observe(action, {**OK, "output": "src/file.py: Usage: grep is discussed here"}), "")
        self.assertEqual(recovery.observe({"name": "read_file", "arguments": {
            "path": "docs.txt"}}, result), "")


if __name__ == "__main__":
    unittest.main()
