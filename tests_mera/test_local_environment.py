from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile
import unittest

from scripts.local_environment import IsolationUnavailable, LocalEnvironment


class LocalEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        self.repo.mkdir()
        self.env = LocalEnvironment(self.repo)

    def tearDown(self):
        self.temporary.cleanup()

    async def test_real_testbed_mount_and_python_only_write_inside_repository(self):
        result = await self.env.exec(
            "pwd\npython3 - <<'PY'\nfrom pathlib import Path\nPath('result.txt').write_text('local')\nprint(Path('/testbed/result.txt').read_text())\nprint(Path('/home/lockr').exists(), Path('/root/.ssh').exists())\ntry:\n    Path('/outside.txt').write_text('wrong')\nexcept OSError:\n    print('root is read only')\nPY",
        )
        self.assertEqual(result.return_code, 0, result.stderr)
        self.assertEqual((self.repo / "result.txt").read_text(), "local")
        self.assertIn("/testbed\nlocal\nFalse False", result.stdout)
        self.assertIn("root is read only", result.stdout)
        self.assertFalse((self.repo / "__pycache__").exists())

    async def test_environment_secrets_and_host_network_are_not_visible(self):
        old = os.environ.get("MERA_LOCAL_TEST_SECRET")
        os.environ["MERA_LOCAL_TEST_SECRET"] = "must-not-leak"
        try:
            result = await self.env.exec("python3 - <<'PY'\nimport os,socket\nprint(os.getenv('MERA_LOCAL_TEST_SECRET'))\nprint(os.getenv('HOME'))\nprint([name for _,name in socket.if_nameindex()])\ns=socket.socket()\ns.settimeout(0.1)\ntry:\n    s.connect(('127.0.0.1',11435))\n    print('connected')\nexcept OSError:\n    print('host service unreachable')\nPY")
        finally:
            if old is None:
                os.environ.pop("MERA_LOCAL_TEST_SECRET", None)
            else:
                os.environ["MERA_LOCAL_TEST_SECRET"] = old
        self.assertEqual(result.return_code, 0, result.stderr)
        self.assertIn("None\nNone\n['lo']", result.stdout)
        self.assertIn("host service unreachable", result.stdout)

    async def test_timeout_kills_background_descendants(self):
        command = "python3 -u - <<'PY' &\nimport time\nfrom pathlib import Path\nwhile True:\n    with Path('heartbeat.txt').open('a') as f:\n        f.write('x')\n    time.sleep(0.02)\nPY\nwait"
        result = await self.env.exec(command, timeout_sec=0.3)
        self.assertEqual(result.return_code, 124)
        self.assertTrue(result.timed_out)
        self.assertIn("Completion is unconfirmed", result.stderr)
        size = (self.repo / "heartbeat.txt").stat().st_size
        await asyncio.sleep(0.15)
        self.assertEqual((self.repo / "heartbeat.txt").stat().st_size, size)

    async def test_cancellation_kills_background_descendants_and_propagates(self):
        command = "python3 -u - <<'PY' &\nimport time\nfrom pathlib import Path\nwhile True:\n    with Path('cancel-heartbeat.txt').open('a') as f:\n        f.write('x')\n    time.sleep(0.02)\nPY\nwait"
        task = asyncio.create_task(self.env.exec(command, timeout_sec=60))
        for _ in range(100):
            if (self.repo / "cancel-heartbeat.txt").exists():
                break
            await asyncio.sleep(0.01)
        self.assertTrue((self.repo / "cancel-heartbeat.txt").exists())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        size = (self.repo / "cancel-heartbeat.txt").stat().st_size
        await asyncio.sleep(0.15)
        self.assertEqual((self.repo / "cancel-heartbeat.txt").stat().st_size, size)

    async def test_invalid_cwd_and_environment_fail_before_execution(self):
        for cwd in ("/tmp", "/testbed/../tmp", "relative"):
            with self.subTest(cwd=cwd), self.assertRaises(ValueError):
                await self.env.exec("true", cwd=cwd)
        with self.assertRaises(ValueError):
            await self.env.exec("true", env={"HOME": "/testbed"})
        with self.assertRaises(ValueError):
            await self.env.exec("true", timeout_sec=0)

    async def test_namespace_failure_has_no_unsafe_fallback(self):
        self.env.bwrap = "/bin/false"
        result = await self.env.exec("touch must-not-exist.txt")
        self.assertNotEqual(result.return_code, 0)
        self.assertFalse((self.repo / "must-not-exist.txt").exists())

    async def test_nested_cwd_and_literal_command_content(self):
        (self.repo / "subdir").mkdir()
        result = await self.env.exec("printf '%s' 'literal $(echo nope)' > literal.txt", cwd="/testbed/subdir")
        self.assertEqual(result.return_code, 0, result.stderr)
        self.assertEqual((self.repo / "subdir/literal.txt").read_text(), "literal $(echo nope)")


if __name__ == "__main__":
    unittest.main()
