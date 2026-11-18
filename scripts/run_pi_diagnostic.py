#!/usr/bin/env python3
"""Run native Pi through its actual HTTP/Harbor bridge on a synthetic task.

This checks integration with the supplied live model, not competition pass@1.
The selected production model may also be passed as --model when available.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.local_environment import LocalEnvironment
from scripts.run_local_diagnostics import seed, grade_script, render_statement, isolated_repository_state, safe_test_state


async def run(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Output already exists; use a fresh directory')
    output.mkdir(parents=True)
    source = output / 'source'
    shutil.copytree(args.source, source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    case = json.loads((ROOT / 'experiments/diagnostics/cases.json').read_text())[args.case]
    repo = output / 'repo'
    head = seed(case, repo)
    original = {name: {'sha256': hashlib.sha256((repo / name).read_bytes()).hexdigest(), 'mode': 0o644}
                for name in case['files'] if name.startswith('test_')}
    statement = render_statement(case)
    (output / 'statement.txt').write_text(statement)
    sys.path.insert(0, str(source))
    from custom_agent.pi_agent import PiAgent
    from harbor.models.agent.context import AgentContext
    spec = importlib.util.spec_from_file_location('mera_pi_diagnostic_service', source / 'inference/pi_service.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = args.base_url.rstrip('/').removesuffix('/v1')
    app = module.create_app(backend_url=backend, model_alias=args.model,
                            worker_path=source / 'inference/pi/worker.mjs', node_binary=args.node_bin)
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    service = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='on'))
    running = asyncio.create_task(service.serve(sockets=[listener]))
    report = {'kind': 'LIVE_NATIVE_PI_BRIDGE_DIAGNOSTIC', 'official_verifier': 'NOT_RUN', 'pass_at_1': None,
              'case': args.case, 'model': args.model, 'backend': backend, 'attempts': 1,
              'settings': {'max_steps': args.max_steps, 'time_budget_sec': args.time_budget_sec,
                           'max_tokens': args.max_tokens, 'context_window': args.context_window, 'thinking': False},
              'source_sha256': {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(source.rglob('*')) if path.is_file()},
              'notes': ['Uses real native Pi SDK and loop, HTTP operation relay, and isolated task commands.',
                        'Synthetic development task; no official task verifier, hidden tests or gold patch used.',
                        'Results apply only to the actual backend recorded below, not another model.']}
    try:
        for _ in range(100):
            if service.started:
                break
            if running.done():
                await running
                raise RuntimeError('Pi service did not start')
            await asyncio.sleep(0.02)
        else:
            raise TimeoutError('Pi service startup timed out')
        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
            try:
                info = await client.get(backend + '/local-info')
                info.raise_for_status()
                report['actual_backend'] = info.json()
            except Exception as exc:
                report['actual_backend'] = {'metadata_unavailable': type(exc).__name__}
        env = LocalEnvironment(repo)
        context = AgentContext()
        agent = PiAgent(logs_dir=output / 'agent', model_name='openai/' + args.model,
            extra_env={'REACT_AGENT_BASE_URL': f'http://127.0.0.1:{port}/v1', 'REACT_AGENT_API_KEY': 'dummy'},
            max_steps=args.max_steps, max_tokens=args.max_tokens, time_budget_sec=args.time_budget_sec,
            context_window=args.context_window)
        print('START native Pi live bridge diagnostic: ' + args.case, flush=True)
        start = time.monotonic()
        error = None
        try:
            await asyncio.wait_for(agent.run(statement, env, context), timeout=args.time_budget_sec + 55)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        report.update(elapsed_sec=round(time.monotonic()-start, 3), agent_error=error,
                      context=context.model_dump(mode='json'))
        visible = await env.exec('python3 -B -m unittest -q test_solution', timeout_sec=20)
        encoded = base64.b64encode(grade_script(case).encode()).decode()
        acceptance = await env.exec("python3 -B -c \"import base64;exec(base64.b64decode('" + encoded + "'))\"", timeout_sec=20)
        try:
            graded = json.loads(acceptance.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            graded = {'passed': 0, 'total': None, 'error': acceptance.stderr or acceptance.stdout}
        state = await isolated_repository_state(repo, original)
        tests = {name: safe_test_state(repo, name, expected) for name, expected in original.items()}
        report.update(visible_tests={'returncode': visible.return_code, 'stdout': visible.stdout, 'stderr': visible.stderr},
                      acceptance={'returncode': acceptance.return_code, **graded},
                      original_tests=tests, head_changed=state['head'] != head, patch_nonempty=bool(state['patch'].strip()))
        report['resolved_synthetic_diagnostic'] = bool(error is None and acceptance.return_code == 0
            and visible.return_code == 0 and graded.get('passed') == graded.get('total') and graded.get('total', 0)
            and not any(item['modified'] for item in tests.values()) and state['head'] == head and state['patch'].strip())
        (output / 'model_patch.diff').write_text(state['patch'])
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({key: report[key] for key in ('kind', 'elapsed_sec', 'agent_error', 'resolved_synthetic_diagnostic')}, indent=2), flush=True)
    finally:
        service.should_exit = True
        await asyncio.wait_for(running, timeout=15)
        listener.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'submission_stage2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:18000/v1')
    parser.add_argument('--model', default='qwen3-8b-local')
    parser.add_argument('--node-bin', default=shutil.which('node') or '/tmp/mera-node/bin/node')
    parser.add_argument('--case', choices=['options_merge', 'intervals', 'query_params'], default='options_merge')
    parser.add_argument('--max-steps', type=int, default=25)
    parser.add_argument('--time-budget-sec', type=int, default=240)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--context-window', type=int, default=16384)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
