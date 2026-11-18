#!/usr/bin/env python3
"""Read-only checks for a Linux GPU host before an expensive Stage II build."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def inspect_host(disk_path: Path, minimum_disk_gib: int = 140) -> dict:
    checks = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    record("platform", platform.system() == "Linux" and platform.machine() in {"x86_64", "amd64"}, f"{platform.system()} {platform.machine()}")
    free = shutil.disk_usage(disk_path).free / 1024**3
    record("disk", free >= minimum_disk_gib, f"{free:.1f} GiB free at {disk_path}; recommended at least {minimum_disk_gib} GiB for model build and task images")

    def command(name: str, args: list[str]) -> str | None:
        if shutil.which(args[0]) is None:
            record(name, False, f"missing executable: {args[0]}")
            return None
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            record(name, False, str(exc))
            return None
        text = (result.stdout or result.stderr).strip()
        record(name, result.returncode == 0, text[:1000])
        return result.stdout if result.returncode == 0 else None

    command("docker", ["docker", "info", "--format", "{{.ServerVersion}} {{.OSType}} {{.Architecture}}"])
    command("compose", ["docker", "compose", "version"])
    gpu_data = command("gpu_visibility", ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if gpu_data is not None:
        lines = [line.rsplit(",", 1) for line in gpu_data.strip().splitlines() if line.strip()]
        try:
            suitable = sum(float(parts[-1]) >= 90000 for parts in lines)
        except (ValueError, IndexError):
            suitable = 0
        record("gpu_capacity", suitable >= 3, f"{suitable} GPUs with at least 90000 MiB; submission uses 3 replicas on 96GB Blackwell GPUs")
    command("harbor", [str(ROOT / ".venv/bin/harbor"), "--version"])
    return {
        "passed": all(item["passed"] for item in checks), "checks": checks,
        "remaining_checks": "Docker GPU passthrough, Harbor allowlist kernel support, model startup/tool protocol, throughput, and scored tasks require actual container runs.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disk-path", type=Path, default=ROOT)
    parser.add_argument("--minimum-disk-gib", type=int, default=140)
    args = parser.parse_args()
    result = inspect_host(args.disk_path.resolve(), args.minimum_disk_gib)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
