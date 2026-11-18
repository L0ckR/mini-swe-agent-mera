/**
 * Native Pi SDK agent with task operations delegated to the Harbor worker.
 * stdin/stdout are a JSON-lines control protocol, never a shell transport.
 * Pi keeps its actual loop, prompts, schemas, edit behavior and compaction.
 */
import { createInterface } from 'node:readline';
import {
  ModelRuntime, SessionManager, SettingsManager,
  createAgentSession, createExtensionRuntime,
} from '@earendil-works/pi-coding-agent';
import { InMemoryCredentialStore } from '@earendil-works/pi-ai';
// The pinned SDK's public entry exports AgentTool wrappers but drops three
// native prompt contributions. Use its unmodified definition factories so
// names, descriptions, schemas, argument preparation AND prompt stay native.
const { createCodingToolDefinitions } = await import(new URL(
  './core/tools/index.js', import.meta.resolve('@earendil-works/pi-coding-agent'),
));

process.env.PI_OFFLINE = '1';

export const PI_VERSION = '1.5.2';
export const TOOL_NAMES = ['read', 'bash', 'edit', 'write'];

const DEFAULT_GUIDANCE = `This is one SWE MERA repository repair attempt. The repository is already populated at /testbed.
Read the complete issue and submission requirements. Make minimal changes to implementation files only;
preserve existing tests, repository history and pre-existing work. Dependencies are already installed.
Use project configuration to discover focused tests for the relevant language: Go, PHP or Python.
Treat the reporter's proposed cause as a hypothesis, locate the responsible implementation and verify the fix.
Keep every task command and file operation inside /testbed. Do not inspect hidden verifiers or ready-made solutions.
Run real focused tests or a reproduction, read the result, and inspect the final diff.
Summarize the actual changes, observed checks and unresolved limitations when finished.
The host exports any required change.diff after the run; do not include temporary scripts, tests or generated files in the patch.`;

function positive(value, fallback, name) {
  const result = value ?? fallback;
  if (typeof result !== 'number' || !Number.isFinite(result) || result <= 0) {
    throw new Error(`${name} must be a positive finite number`);
  }
  return result;
}

export function validateConfig(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('config must be an object');
  if (typeof input.instruction !== 'string' || !input.instruction.trim()) throw new Error('instruction is required');
  if (input.workdir !== undefined && input.workdir !== '/testbed') throw new Error('workdir must be /testbed');
  if (typeof input.model !== 'string' || !input.model.trim()) throw new Error('model is required');
  const url = new URL(input.base_url);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error('base_url must be a plain HTTP(S) endpoint');
  }
  const maxTurns = positive(input.max_turns, 100, 'max_turns');
  const maxTokens = positive(input.max_output_tokens, 8192, 'max_output_tokens');
  const contextWindow = positive(input.context_window, 65536, 'context_window');
  if (![maxTurns, maxTokens, contextWindow].every(Number.isInteger)) throw new Error('token/turn limits must be integers');
  if (maxTokens >= contextWindow) throw new Error('max_output_tokens must be below context_window');
  if (input.append_system_prompt !== undefined && typeof input.append_system_prompt !== 'string') {
    throw new Error('append_system_prompt must be text');
  }
  const temperature = input.temperature ?? 0.8;
  if (typeof temperature !== 'number' || !Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
    throw new Error('temperature must be between 0 and 2');
  }
  return {
    ...input, workdir: '/testbed', base_url: url.href.replace(/\/$/, ''),
    api_key: input.api_key || 'dummy', max_turns: maxTurns,
    max_output_tokens: maxTokens, context_window: contextWindow,
    time_budget_sec: positive(input.time_budget_sec, 1680, 'time_budget_sec'),
    temperature,
  };
}

export class OperationBridge {
  constructor(emit) { this.emit = emit; this.pending = new Map(); this.sequence = 0; this.closed = false; }
  call(operation, args) {
    if (this.closed) return Promise.reject(new Error('operation bridge is closed'));
    const id = `op_${++this.sequence}`;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.emit({ type: 'operation', id, operation, args });
    });
  }
  receive(message) {
    if (message?.type !== 'operation_result') return false;
    const pending = this.pending.get(message.id);
    if (!pending) return false;
    this.pending.delete(message.id);
    if (message.error) pending.reject(new Error(String(message.error)));
    else pending.resolve(message.result ?? {});
    return true;
  }
  close(reason = 'operation bridge closed') {
    this.closed = true;
    for (const [id, pending] of this.pending) {
      pending.reject(new Error(reason));
      this.emit({ type: 'operation_cancel', id, reason });
    }
    this.pending.clear();
  }
}

function nativeRemoteTools(cwd, bridge) {
  const readFile = async (path) => {
    const result = await bridge.call('read_file', { path });
    if (typeof result.data_base64 !== 'string') throw new Error('read_file returned no data_base64');
    return Buffer.from(result.data_base64, 'base64');
  };
  const access = async (path, write = false) => { await bridge.call('access', { path, write }); };
  const writeFile = async (path, content) => {
    await bridge.call('write_file', { path, data_base64: Buffer.from(content, 'utf8').toString('base64') });
  };
  const operations = {
    readFile, writeFile, access,
    mkdir: async (path) => { await bridge.call('mkdir', { path }); },
  };
  return createCodingToolDefinitions(cwd, {
    read: { operations: { readFile, access, detectImageMimeType: async () => null } },
    bash: { operations: { exec: async (command, taskCwd, { onData, signal, timeout }) => {
      signal?.throwIfAborted();
      const result = await bridge.call('exec', { command, cwd: taskCwd, timeout_sec: timeout ?? 120 });
      if (result.stdout) onData(Buffer.from(result.stdout));
      if (result.stderr) onData(Buffer.from(result.stderr));
      signal?.throwIfAborted();
      if (result.timed_out) throw new Error(`timeout:${timeout ?? 120}`);
      if (!Number.isInteger(result.returncode)) throw new Error('exec returned no integer returncode');
      return { exitCode: result.returncode };
    } } },
    edit: { operations: { readFile, writeFile, access: (path) => access(path, true) } },
    write: { operations },
  });
}

export async function createNativePiSession(input, bridge, emit) {
  const config = validateConfig(input);
  // Prevent catalog downloads, background updates, auth commands or project resource discovery.
  process.env.PI_OFFLINE = '1';
  const modelRuntime = await ModelRuntime.create({
    credentials: new InMemoryCredentialStore(), modelsPath: null, allowModelNetwork: false, refreshOnCreate: false,
  });
  modelRuntime.registerProvider('mera-local', {
    api: 'openai-completions', baseUrl: config.base_url, apiKey: config.api_key,
    models: [{
      id: config.model, name: config.model, reasoning: true, input: ['text'],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: config.context_window, maxTokens: config.max_output_tokens,
      samplingParams: { temperature: config.temperature },
      compat: { supportsStore: false, supportsDeveloperRole: false, maxTokensField: 'max_tokens',
        supportsReasoningEffort: false, thinkingFormat: 'qwen-chat-template',
        chatTemplateKwargs: { enable_thinking: false } },
    }],
  });
  const model = modelRuntime.getModel('mera-local', config.model);
  if (!model) throw new Error('configured local model was not registered');
  const resourceLoader = {
    getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
    getSkills: () => ({ skills: [], diagnostics: [] }),
    getPrompts: () => ({ prompts: [], diagnostics: [] }),
    getThemes: () => ({ themes: [], diagnostics: [] }),
    getAgentsFiles: () => ({ agentsFiles: [] }),
    getSystemPrompt: () => undefined, getSystemPromptSource: () => undefined,
    getAppendSystemPrompt: () => [DEFAULT_GUIDANCE, config.append_system_prompt ?? ''].filter(Boolean),
    getAppendSystemPromptSources: () => [], extendResources: () => {}, reload: async () => {},
  };
  const { session } = await createAgentSession({
    cwd: config.workdir, agentDir: '/tmp/mera-pi', modelRuntime, model, thinkingLevel: 'off',
    sessionManager: SessionManager.inMemory(config.workdir),
    settingsManager: SettingsManager.inMemory({
      compaction: { enabled: true, reserveTokens: config.max_output_tokens + 2048,
        keepRecentTokens: Math.min(12000, Math.floor(config.context_window / 4)) },
      retry: { enabled: true, maxRetries: 2 }, cacheWarming: { enabled: false },
    }),
    resourceLoader, tools: TOOL_NAMES, customTools: nativeRemoteTools(config.workdir, bridge),
  });
  let turns = 0;
  let stopReason = 'completed';
  const cumulativeUsage = () => {
    // Native stats also account for model calls made by compaction and keep
    // entries that have left the active model context.
    const tokens = session.getSessionStats().tokens;
    return { input: tokens.input, output: tokens.output, cacheRead: tokens.cacheRead,
      cacheWrite: tokens.cacheWrite, totalTokens: tokens.total };
  };
  let lastAssistant = '';
  let lastError;
  const abort = (reason) => {
    if (stopReason === 'completed') stopReason = reason;
    bridge.close(reason);
    void session.abort();
  };
  const unsubscribe = session.subscribe((event) => {
    if (['message_end', 'turn_end', 'tool_execution_start', 'tool_execution_end', 'agent_end',
      'auto_compaction_start', 'auto_compaction_end', 'auto_retry_start', 'auto_retry_end'].includes(event.type)) {
      emit({ type: 'event', event });
    }
    if (event.type === 'message_end' && event.message?.role === 'assistant') {
      const message = event.message;
      lastAssistant = (message.content ?? []).filter((block) => block.type === 'text').map((block) => block.text).join('\n');
      lastError = message.stopReason === 'error' ? (message.errorMessage ?? 'model request failed') : undefined;
    }
    if (event.type === 'turn_end' && ++turns >= config.max_turns) abort('max_turns');
    if (event.type === 'turn_end' || event.type === 'auto_compaction_end') {
      emit({ type: 'event', event: { type: 'usage_totals', usage: cumulativeUsage(), turns } });
    }
  });
  const timer = setTimeout(() => abort('time_budget'), config.time_budget_sec * 1000);
  emit({ type: 'ready', pi_version: PI_VERSION, tools: session.getActiveToolNames(), thinking: 'off',
    model: config.model, context_window: config.context_window });
  return {
    abort,
    async prompt(message) {
      if (typeof message !== 'string' || !message.trim()) throw new Error('prompt message must be nonempty text');
      if (stopReason !== 'completed') throw new Error(`session already stopped: ${stopReason}`);
      lastError = undefined;
      await session.prompt(message, { expandPromptTemplates: false });
      return { type: 'complete', pi_version: PI_VERSION, stop_reason: lastError ? 'model_error' : stopReason,
        turns, usage: cumulativeUsage(), final_answer: lastAssistant.slice(0, 65536),
        ...(lastError ? { error: lastError.slice(0, 4000) } : {}) };
    },
    dispose() { clearTimeout(timer); unsubscribe(); session.dispose(); bridge.close('session ended'); },
  };
}

export async function runNativePi(input, bridge, emit) {
  const controller = await createNativePiSession(input, bridge, emit);
  try { return await controller.prompt(input.instruction); }
  finally { controller.dispose(); }
}

export async function main() {
  const emit = (message) => process.stdout.write(JSON.stringify(message) + '\n');
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const bridge = new OperationBridge(emit);
  let started = false;
  let active;
  let inputEnded = false;
  let run = Promise.resolve();
  const fail = (error) => {
    emit({ type: 'error', error: String(error?.message ?? error) });
    process.exitCode = 1;
    active?.dispose();
    lines.close(); process.stdin.pause();
  };
  lines.on('line', (line) => {
    let message;
    try { message = JSON.parse(line); }
    catch { emit({ type: 'error', error: 'invalid JSON control message' }); return; }
    if (!started) {
      started = true;
      run = createNativePiSession(message, bridge, emit).then((controller) => {
        active = controller;
        if (inputEnded) controller.abort('transport_closed');
        return controller.prompt(message.instruction);
      }).then(emit).catch(fail);
    } else if (message?.type === 'abort') active?.abort(message.reason ?? 'cancelled');
    else if (message?.type === 'prompt') {
      run = run.then(() => active.prompt(message.message)).then(emit).catch(fail);
    }
    else bridge.receive(message);
  });
  lines.on('close', () => { inputEnded = true; if (active && !bridge.closed) active.abort('transport_closed'); });
  await new Promise((resolve) => lines.once('close', resolve));
  await run;
  active?.dispose();
}

if (import.meta.url === `file://${process.argv[1]}` && process.argv.includes('--dump-tools')) {
  const tools = nativeRemoteTools('/testbed', { call: () => { throw new Error('schema dump must not execute operations'); } });
  process.stdout.write(JSON.stringify(tools.map((tool) => ({ type: 'function', function: {
    name: tool.name, description: tool.description, parameters: tool.parameters,
  } })), null, 2) + '\n');
} else if (import.meta.url === `file://${process.argv[1]}` && process.argv.includes('--check')) {
  process.stdout.write(JSON.stringify({ passed: true, kind: 'NATIVE_PI_SDK_IMPORT', pi_version: PI_VERSION,
    tools: TOOL_NAMES, thinking: 'off', model_quality: 'NOT_MEASURED' }) + '\n');
} else if (import.meta.url === `file://${process.argv[1]}`) main().catch((error) => {
  process.stdout.write(JSON.stringify({ type: 'error', error: String(error?.message ?? error) }) + '\n');
  process.exitCode = 1;
});
