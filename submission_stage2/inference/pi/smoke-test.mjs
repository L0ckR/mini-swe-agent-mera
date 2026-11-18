/** Scripted transport test of the real Pi loop and native tools; no model score. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
import { OperationBridge, createNativePiSession, validateConfig } from './worker.mjs';
const expectedTools = JSON.parse(await readFile(new URL('./tool-schema.json', import.meta.url), 'utf8'));

const original = 'def add(a, b):\n    return a - b\n';
const files = new Map([['/testbed/calc.py', original]]);
const operations = [];
const requests = [];
const events = [];
const calls = [
  ['read', { path: 'calc.py' }],
  ['edit', { path: 'calc.py', edits: [{ oldText: 'return a - b', newText: 'return a + b' }] }],
  ['bash', { command: 'python3 -B -m unittest -q test_calc', timeout: 20 }],
  ['write', { path: 'notes.txt', content: 'native Pi write\n' }],
];

const server = createServer(async (request, response) => {
  try {
    assert.equal(request.url, '/v1/chat/completions');
    let body = '';
    for await (const chunk of request) body += chunk;
    const payload = JSON.parse(body);
    requests.push(payload);
    assert.equal(payload.model, 'scripted-model');
    assert.equal(payload.stream, true);
    assert.deepEqual(payload.tools.map((tool) => tool.function.name).sort(), ['bash', 'edit', 'read', 'write']);
    for (const tool of payload.tools) {
      const expected = expectedTools.find((entry) => entry.function.name === tool.function.name);
      assert.equal(tool.function.description, expected.function.description);
      assert.deepEqual(tool.function.parameters, expected.function.parameters);
    }
    assert.equal(payload.chat_template_kwargs?.enable_thinking, false);
    const number = requests.length;
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    const chunk = (choices, usage) => response.write('data: ' + JSON.stringify({
      id: 'chatcmpl-test-' + number, object: 'chat.completion.chunk', created: 1,
      model: 'scripted-model', choices, ...(usage ? { usage } : {}),
    }) + '\n\n');
    chunk([{ index: 0, delta: { role: 'assistant' }, finish_reason: null }]);
    if (number <= calls.length) {
      const [name, arguments_] = calls[number - 1];
      chunk([{ index: 0, delta: { tool_calls: [{ index: 0, id: 'call_' + number,
        type: 'function', function: { name, arguments: JSON.stringify(arguments_) } }] }, finish_reason: null }]);
      chunk([{ index: 0, delta: {}, finish_reason: 'tool_calls' }]);
    } else {
      assert.ok(number === calls.length + 1 || number === calls.length + 2);
      chunk([{ index: 0, delta: { content: 'Fixed the implementation; the observed check returned 0.' }, finish_reason: null }]);
      chunk([{ index: 0, delta: {}, finish_reason: 'stop' }]);
    }
    chunk([], { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 });
    response.end('data: [DONE]\n\n');
  } catch (error) {
    response.writeHead(500); response.end(String(error?.stack ?? error));
  }
});
server.listen(0, '127.0.0.1');
await once(server, 'listening');

function operationResult(message) {
  operations.push(message);
  const { operation, args, id } = message;
  try {
    let result = {};
    if (operation === 'access') assert.ok(files.has(args.path));
    else if (operation === 'read_file') result = { data_base64: Buffer.from(files.get(args.path)).toString('base64') };
    else if (operation === 'write_file') files.set(args.path, Buffer.from(args.data_base64, 'base64').toString());
    else if (operation === 'mkdir') assert.equal(args.path, '/testbed');
    else if (operation === 'exec') {
      assert.equal(args.command, 'python3 -B -m unittest -q test_calc');
      assert.equal(args.cwd, '/testbed');
      assert.ok(files.get('/testbed/calc.py').includes('return a + b'));
      result = { returncode: 0, stdout: 'scripted validation observation\n', stderr: '' };
    } else throw new Error('Unexpected operation ' + operation);
    return { type: 'operation_result', id, result };
  } catch (error) {
    return { type: 'operation_result', id, error: String(error.message) };
  }
}
const bridge = new OperationBridge((message) => {
  events.push(message);
  if (message.type !== 'operation') return;
  const response = operationResult(message);
  queueMicrotask(() => bridge.receive(response));
});

try {
  const instruction = 'Fix add(a,b), validate it and write notes.txt.';
  const controller = await createNativePiSession({
    instruction, model: 'scripted-model',
    base_url: `http://127.0.0.1:${server.address().port}/v1`, time_budget_sec: 30,
    max_turns: 10, context_window: 65536, max_output_tokens: 2048,
  }, bridge, (event) => events.push(event));
  const result = await controller.prompt(instruction);
  assert.equal(result.stop_reason, 'completed');
  if (result.turns !== 5) console.error(JSON.stringify({result, requests: requests.map((request) => ({messages: request.messages, tools: request.tools}))}, null, 2));
  assert.equal(result.turns, 5);
  assert.equal(result.usage.input, 50);
  assert.equal(result.usage.output, 25);
  assert.ok(files.get('/testbed/calc.py').includes('return a + b'));
  assert.equal(files.get('/testbed/notes.txt'), 'native Pi write\n');
  assert.ok(events.some((event) => event.type === 'ready' && event.pi_version === '1.5.2'));
  assert.equal(operations.filter((entry) => entry.operation === 'exec').length, 1);
  assert.ok(requests[4].messages.some((message) => message.role === 'tool' && message.content.includes('Successfully wrote')));
  assert.ok(requests[0].messages.some((message) => message.role === 'system' && message.content.includes('coding assistant running inside pi')));
  assert.ok(!requests[0].tools.some((tool) => ['finish', 'edit_file', 'run_tests'].includes(tool.function.name)));
  assert.ok(requests[0].messages.some((message) => message.role === 'system' && message.content.includes('Use edit for targeted changes')));
  const continued = await controller.prompt('Inspect the exported patch contract once more.');
  assert.equal(continued.turns, 6);
  assert.equal(continued.usage.input, 60);
  assert.equal(continued.usage.output, 30);
  const usageEvents = events.filter((entry) => entry.type === 'event' && entry.event.type === 'usage_totals');
  assert.equal(usageEvents.at(-1).event.usage.input, 60);
  assert.equal(usageEvents.at(-1).event.usage.output, 30);
  assert.equal(requests[5].messages.filter((message) => message.role === 'user').length, 2);
  assert.ok(requests[5].messages.some((message) => message.role === 'tool' && message.content.includes('Successfully wrote')));
  controller.dispose();

  for (const config of [{ max_turns: true }, { temperature: 3 }, { workdir: '/models' }, { max_output_tokens: 65536 }]) {
    assert.throws(() => validateConfig({ instruction: 'x', model: 'x', base_url: 'http://local/v1', ...config }));
  }
  const pendingBridge = new OperationBridge(() => {});
  const pending = pendingBridge.call('read_file', { path: '/testbed/calc.py' });
  pendingBridge.close('cancelled');
  await assert.rejects(pending, /cancelled/);

  // Exercise the executable JSONL protocol, including two prompts in the
  // same child process. Closing stdin after the second completion must exit.
  requests.length = 0;
  files.set('/testbed/calc.py', 'def add(a, b):\n    return a - b\n');
  const child = spawn(process.execPath, [new URL('./worker.mjs', import.meta.url).pathname], { stdio: ['pipe', 'pipe', 'pipe'] });
  const childLines = createInterface({ input: child.stdout });
  let childError = '';
  let callbackError;
  const completions = [];
  child.stderr.on('data', (chunk) => { childError += chunk; });
  childLines.on('line', (line) => {
    try {
      const message = JSON.parse(line);
      if (message.type === 'operation') child.stdin.write(JSON.stringify(operationResult(message)) + '\n');
      if (message.type === 'error') throw new Error(message.error);
      if (message.type === 'complete') {
        completions.push(message);
        assert.ok(!('messages' in message));
        if (completions.length === 1) {
          child.stdin.write(JSON.stringify({ type: 'prompt', message: 'Review the export contract.' }) + '\n');
        } else child.stdin.end();
      }
    } catch (error) { callbackError = error; child.kill('SIGTERM'); }
  });
  const childTimeout = setTimeout(() => child.kill('SIGKILL'), 15000);
  child.stdin.write(JSON.stringify({ instruction, model: 'scripted-model',
    base_url: `http://127.0.0.1:${server.address().port}/v1`, max_turns: 10,
    context_window: 65536, max_output_tokens: 2048, time_budget_sec: 20 }) + '\n');
  const [childCode] = await once(child, 'exit');
  clearTimeout(childTimeout);
  if (callbackError) throw callbackError;
  assert.equal(childCode, 0, childError);
  assert.equal(completions.length, 2);
  assert.equal(completions[1].usage.input, 60);
  assert.equal(completions[1].usage.output, 30);
  assert.equal(requests[5].messages.filter((message) => message.role === 'user').length, 2);

  // The executable protocol must also emit one terminal failure for bad config.
  const worker = spawn(process.execPath, [new URL('./worker.mjs', import.meta.url).pathname], { stdio: ['pipe', 'pipe', 'pipe'] });
  let stdout = '';
  worker.stdout.on('data', (chunk) => { stdout += chunk; });
  worker.stdin.end(JSON.stringify({ instruction: '' }) + '\n');
  const [code] = await once(worker, 'exit');
  assert.equal(code, 1);
  assert.ok(stdout.split('\n').filter(Boolean).some((line) => JSON.parse(line).type === 'error'));
  console.log(JSON.stringify({ passed: true, kind: 'SCRIPTED_NATIVE_PI_SDK_SMOKE',
    pi_version: '1.5.2', native_model_turns: continued.turns, prompts_in_same_session: 2,
    executable_jsonl_continuation: true,
    operation_names: [...new Set(operations.map((item) => item.operation))],
    model_quality: 'NOT_MEASURED' }));
} finally {
  server.closeAllConnections(); server.close();
}
