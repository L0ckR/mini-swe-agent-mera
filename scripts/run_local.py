#!/usr/bin/env python3
"""Run our build context using the unmodified, pinned organizer runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def verify_vendor() -> None:
    manifest = json.loads((ROOT / "vendor/stage2/SOURCE.json").read_text())
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((ROOT / "vendor/stage2" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Organizer file was changed: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--tasks", type=Path, default=ROOT / "tasks")
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", args.username):
        parser.error("username: 1-64 lowercase letters/digits/._-, starting with letter/digit")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    tasks = args.tasks.resolve()
    if not tasks.is_dir() or not any(tasks.glob("*/task.toml")):
        parser.error(f"no Harbor tasks found in {tasks}")
    if not (ROOT / ".venv/bin/harbor").is_file():
        parser.error("install requirements.txt into .venv first")
    if shutil.which("docker") is None:
        parser.error("Docker CLI is missing")
    verify_vendor()
    from validate_submission import validate_directory

    validate_directory(ROOT / "submission_stage2")
    jobs = (args.jobs or ROOT / "jobs/stage2" / args.username).resolve()
    env = dict(os.environ)
    env.update({
        "SWE_MERA_TASKS_DIR": str(tasks),
        "SWE_MERA_JOBS_DIR": str(jobs),
        "SWE_MERA_N_CONCURRENT": str(args.concurrency),
    })
    # Its parent must be ROOT: the official runner resolves .venv relative to it.
    with tempfile.TemporaryDirectory(prefix=".stage2-local-", dir=ROOT) as temporary:
        staged = Path(temporary)
        shutil.copytree(ROOT / "submission_stage2", staged, dirs_exist_ok=True)
        shutil.copy2(ROOT / "vendor/stage2/run.sh", staged / "run.sh")
        shutil.copy2(ROOT / "vendor/stage2/inference/test.sh", staged / "inference/test.sh")
        try:
            return subprocess.call(["bash", str(staged / "run.sh"), "--username", args.username], env=env)
        except KeyboardInterrupt:
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
