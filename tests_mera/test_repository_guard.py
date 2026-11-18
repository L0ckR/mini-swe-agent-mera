from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission_stage2"))
from custom_agent.tools import TaskTools, repository_command_error


class RepositoryGuardTests(unittest.TestCase):
    def test_accidental_root_and_metadata_erasure_is_rejected(self):
        commands = ["rm -rf /testbed", "rm -rf .", "rm -rf ./", "rm -rf *", "rm -rf /testbed/*",
                    "rm -rf /testbed/.*", "rm -rf .git", "rm .git/config", "rm -rf $PWD", "rm -rf ${PWD}/*",
                    "echo start; rm -rf /testbed", "echo start # comment\nrm -rf /testbed",
                    "env LC_ALL=C /bin/rm -rf /testbed", "bash -lc 'rm -rf /testbed'",
                    "git clone https://example.invalid/repo.git /testbed"]
        for command in commands:
            with self.subTest(command=command):
                self.assertIsNotNone(repository_command_error(command))

    def test_destructive_git_whole_tree_operations_are_rejected(self):
        for command in ("git reset --hard", "git reset --hard HEAD", "git -C /testbed reset --hard",
                        "git clean -fd", "git clean -f -d -x", "git clean --force -d",
                        "git checkout -- .", "git checkout HEAD -- .", "git restore .", "git restore --worktree /testbed"):
            with self.subTest(command=command):
                self.assertIsNotNone(repository_command_error(command))

    def test_targeted_cleanup_tests_and_literal_command_text_are_allowed(self):
        commands = ["rm -rf build .pytest_cache scratch/repro.py", "git diff HEAD && git status --short",
                    "git restore src/file.py", "git checkout -- src/file.py", "git clean -nd",
                    "cd build && rm -rf *", "python3 -m unittest -q test_solution",
                    "echo 'rm -rf /testbed'", "python3 -c \"print('git reset --hard')\"",
                    "python3 - <<'PY'\ntext = 'rm -rf /testbed'\nprint(text)\nPY\n",
                    "cat <<-EOF > scratch.txt\n\trm -rf /testbed\n\tEOF\n"]
        for command in commands:
            with self.subTest(command=command):
                self.assertIsNone(repository_command_error(command))

    def test_real_shell_after_heredoc_is_still_inspected(self):
        self.assertIsNotNone(repository_command_error("cat <<'EOF'\nrm -rf /testbed\nEOF\nrm -rf /testbed\n"))


class ToolGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_bash_and_run_tests_do_not_execute_rejected_commands(self):
        class Environment:
            calls = 0
            async def exec(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(return_code=0, stdout="ok", stderr="")

        environment = Environment()
        tools = TaskTools(environment)
        for name in ("bash", "run_tests"):
            result = await tools.execute({"name": name, "arguments": {"command": "rm -rf /testbed"}}, 10)
            self.assertTrue(result["guard_blocked"])
            self.assertNotEqual(result["returncode"], 0)
        self.assertEqual(environment.calls, 0)
