#!/usr/bin/env python3
"""Exercise native Pi + HTTP relay + isolated task I/O with scripted model replies.

This verifies integration, not the selected model's quality or GPU startup.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

from fastapi import FastAPI, Request
from fastapi.responses import Response
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'submission_stage2'))
from scripts.local_environment import LocalEnvironment
from scripts.ollama_bridge import buffered_sse
from custom_agent.pi_agent import PiAgent
from harbor.models.agent.context import AgentContext


@asynccontextmanager
async def serve(app):
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='on'))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.02)
        else:
            raise TimeoutError('Local test service did not start')
        yield f'http://127.0.0.1:{port}'
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 15)
        listener.close()


async def check(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Use a fresh output directory')
    repo = output / 'repo'
    repo.mkdir(parents=True)
    (repo / 'calc.py').write_text('def add(a,b):\n    return a-b\n')
    (repo / 'test_calc.py').write_text('import unittest\nfrom calc import add\nclass TestAdd(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2,3),5)\n')
    original_test = (repo / 'test_calc.py').read_bytes()
    for command in (['git', 'init', '-q'], ['git', 'add', '.'],
                    ['git', '-c', 'user.name=Smoke', '-c', 'user.email=smoke@example.invalid', 'commit', '-qm', 'initial']):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    steps = [('read', {'path': 'calc.py'}),
             ('edit', {'path': 'calc.py', 'edits': [{'oldText': 'return a-b', 'newText': 'return a+b'}]}),
             ('bash', {'command': 'python3 -B -m unittest -q test_calc', 'timeout': 10}),
             (None, 'Fixed add and ran the existing unittest.')]
    requests = []
    model = FastAPI()

    @model.post('/v1/chat/completions')
    async def completion(request: Request):
        payload = await request.json()
        assert payload['stream'] is True
        assert {tool['function']['name'] for tool in payload['tools']} == {'read', 'bash', 'edit', 'write'}
        assert payload['chat_template_kwargs']['enable_thinking'] is False
        number = len(requests)
        requests.append(payload)
        if number >= len(steps):
            return Response('unexpected extra model request', status_code=500)
        name, arguments = steps[number]
        message = {'role': 'assistant', 'content': arguments if name is None else None}
        if name:
            message['tool_calls'] = [{'id': f'smoke_{number}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments)}}]
        value = {'id': f'chatcmpl-smoke-{number}', 'created': 1, 'object': 'chat.completion', 'model': 'scripted',
                 'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if name else 'stop'}],
                 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        return Response(buffered_sse(value), media_type='text/event-stream')

    spec = importlib.util.spec_from_file_location('mera_pi_smoke_service', ROOT / 'submission_stage2/inference/pi_service.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = AgentContext()
    environment = LocalEnvironment(repo)
    async with serve(model) as model_url:
        app = module.create_app(backend_url=model_url, model_alias='scripted', node_binary=args.node_bin,
                                worker_path=args.worker_path)
        async with serve(app) as service_url:
            agent = PiAgent(logs_dir=output / 'agent', model_name='openai/scripted', time_budget_sec=90,
                extra_env={'REACT_AGENT_BASE_URL': service_url + '/v1', 'REACT_AGENT_API_KEY': 'dummy'})
            await agent.run('<issue_description>Fix add(a,b) so it adds both arguments.</issue_description>\n'
                'Change non-test files only.\n## Submission\nWrite the complete change as a unified diff to `/testbed/change.diff`.',
                environment, context)
    verified = await environment.exec('python3 -B -m unittest -q test_calc', timeout_sec=10)
    patch = repo / 'change.diff'
    assert verified.return_code == 0 and 'Ran 1 test' in verified.stderr
    assert (repo / 'test_calc.py').read_bytes() == original_test
    assert patch.is_file() and patch.stat().st_size
    assert len(requests) == 4
    assert context.metadata['patch_export']['apply_check_passed']
    assert context.n_input_tokens == 40 and context.n_output_tokens == 20
    report = {'kind': 'SCRIPTED_NATIVE_PI_END_TO_END', 'passed': True, 'model_quality': 'NOT_MEASURED',
              'selected_model_gpu_startup': 'NOT_RUN', 'native_pi_version': '1.5.2', 'model_turns': len(requests),
              'task_isolation': 'bubblewrap', 'existing_tests_passed': 1, 'existing_tests_unchanged': True,
              'change_diff_applies': True, 'context': context.model_dump(mode='json')}
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('kind', 'passed', 'model_turns', 'existing_tests_passed', 'change_diff_applies')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--node-bin', default=shutil.which('node') or '/tmp/mera-node/bin/node')
    parser.add_argument('--worker-path', type=Path, default=ROOT / 'submission_stage2/inference/pi/worker.mjs')
    asyncio.run(check(parser.parse_args()))


if __name__ == '__main__':
    main()
