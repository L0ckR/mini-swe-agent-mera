"""Keep the private vLLM backend and public Pi/API bridge in one container."""
from __future__ import annotations

import os
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent
    model = json.loads((root / 'model.json').read_text())
    environment = dict(os.environ, SWE_MERA_INFERENCE_HOST="127.0.0.1", SWE_MERA_INFERENCE_PORT="8001",
                       MERA_VLLM_BASE_URL="http://127.0.0.1:8001", MERA_PI_MODEL=model['served_model_name'],
                       MERA_PI_ALLOWED_MODELS=','.join([model['served_model_name'], model['base_served_model_name']]))
    children = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        for process in children:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        children.append(subprocess.Popen([sys.executable, str(root / "serve.py")], env=environment, start_new_session=True))
        children.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "pi_service:app", "--app-dir", str(root),
                                         "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "10"],
                                        env=environment, start_new_session=True))
        while not stopping and all(process.poll() is None for process in children):
            time.sleep(0.2)
        return next((process.returncode for process in children if process.returncode not in {None, 0}), 1)
    finally:
        stop()
        for process in children:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
