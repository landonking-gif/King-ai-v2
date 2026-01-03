import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const redisStore = new Map<string, Record<string, unknown>>();

type StreamEntry = [string, string[]];
const streamStore = new Map<string, StreamEntry[]>();
let streamSeq = 0;

const getByPath = (value: Record<string, unknown>, path: string): unknown => {
    if (path === '$') return value;
    const pathParts = path.replace('$.', '').split('.');
    return pathParts.reduce<unknown>((acc, part) => {
        if (acc && typeof acc === 'object' && part in (acc as Record<string, unknown>)) {
            return (acc as Record<string, unknown>)[part];
        }
        return undefined;
    }, value);
};

const setByPath = (value: Record<string, unknown>, path: string, next: unknown): void => {
    if (path === '$') {
        Object.assign(value, next as Record<string, unknown>);
        return;
    }
    const pathParts = path.replace('$.', '').split('.');
    let target = value;
    for (let i = 0; i < pathParts.length - 1; i++) {
        const part = pathParts[i]!;
        if (!target[part] || typeof target[part] !== 'object') {
            target[part] = {};
        }
        target = target[part] as Record<string, unknown>;
    }
    target[pathParts[pathParts.length - 1]!] = next;
};

const mockRedisJSON = {
    get: vi.fn(<T>(key: string, path: string = '$'): T | null => {
        const value = redisStore.get(key);
        if (!value) return null;
        return (getByPath(value, path) as T) ?? null;
    }),
    set: vi.fn((key: string, path: string, value: unknown) => {
        const current = redisStore.get(key) ?? {};
        setByPath(current, path, value);
        redisStore.set(key, current);
        return 'OK';
    }),
    del: vi.fn((key: string) => {
        const existed = redisStore.has(key);
        redisStore.delete(key);
        return existed ? 1 : 0;
    }),
    keys: vi.fn((pattern: string) => {
        const regex = new RegExp(`^${pattern.replace('*', '.*')}$`);
        return Array.from(redisStore.keys()).filter(key => regex.test(key));
    }),
    arrAppend: vi.fn((key: string, path: string, ...values: unknown[]) => {
        const current = redisStore.get(key) ?? {};
        const existing = getByPath(current, path);
        const next = Array.isArray(existing) ? [...existing, ...values] : [...values];
        setByPath(current, path, next);
        redisStore.set(key, current);
        return next.length;
    }),
};

function xadd(streamKey: string, _id: string, ...args: string[]): string {
    const id = `${Date.now()}-${streamSeq++}`;
    const entries = streamStore.get(streamKey) ?? [];
    entries.push([id, args]);
    streamStore.set(streamKey, entries);
    return id;
}

function xrange(streamKey: string, _start: string, _end: string, ...args: Array<string | number>): StreamEntry[] {
    const entries = streamStore.get(streamKey) ?? [];
    const countIndex = args.findIndex(v => v === 'COUNT');
    if (countIndex >= 0) {
        const raw = args[countIndex + 1];
        const limit = typeof raw === 'number' ? raw : Number.parseInt(String(raw), 10);
        return entries.slice(0, limit);
    }
    return entries;
}

function xrevrange(streamKey: string, _end: string, _start: string, ...args: Array<string | number>): StreamEntry[] {
    const entries = streamStore.get(streamKey) ?? [];
    const reversed = [...entries].reverse();
    const countIndex = args.findIndex(v => v === 'COUNT');
    if (countIndex >= 0) {
        const raw = args[countIndex + 1];
        const limit = typeof raw === 'number' ? raw : Number.parseInt(String(raw), 10);
        return reversed.slice(0, limit);
    }
    return reversed;
}

const mockRedisClient = {
    ping: vi.fn(() => 'PONG'),
    xadd: vi.fn((...args: any[]) => xadd(args[0], args[1], ...args.slice(2))),
    xrange: vi.fn((...args: any[]) => xrange(args[0], args[1], args[2], ...args.slice(3))),
    xrevrange: vi.fn((...args: any[]) => xrevrange(args[0], args[1], args[2], ...args.slice(3))),
};

const mockLLMClient = {
    json: vi.fn(async () => ({
        data: null,
        raw: {
            content: '',
            model: 'test',
            tokens_used: { prompt: 0, completion: 0, total: 0 },
            duration_ms: 0,
            finish_reason: 'stop',
        },
    })),
    complete: vi.fn(async () => ({
        content: '',
        model: 'test',
        tokens_used: { prompt: 0, completion: 0, total: 0 },
        duration_ms: 0,
        finish_reason: 'stop',
    })),
    embed: vi.fn(async () => ({
        embeddings: [new Array(768).fill(0)],
        model: 'test',
        dimensions: 768,
    })),
};

vi.mock('@mother-harness/shared', async () => {
    const actual = await vi.importActual<typeof import('@mother-harness/shared')>('@mother-harness/shared');
    return {
        ...actual,
        getRedisJSON: () => mockRedisJSON,
        getRedisClient: () => mockRedisClient,
        getLLMClient: () => mockLLMClient,
        getResourceBudgetGuard: () => ({
            checkAllScopes: vi.fn(async () => ({ allowed: true, remaining: 9999 })),
            recordUsageAllScopes: vi.fn(async () => undefined),
        }),
        getRoleRegistry: () => ({
            getContract: vi.fn(async () => ({
                role: 'test',
                description: 'test',
                default_action: 'rag_retrieval',
                required_inputs: [],
                optional_inputs: [],
                required_outputs: [],
                optional_outputs: [],
                required_artifacts: [],
                optional_artifacts: [],
                allowed_artifact_types: [],
                max_cost_usd: 1.0,
            })),
        }),
        getContractEnforcer: () => ({
            validateAllowlist: vi.fn(async () => ({ valid: true, errors: [] })),
            validateRequiredArtifacts: vi.fn(async () => ({ valid: true, errors: [] })),
        }),
    };
});

import { Orchestrator, clearAgentExecutors, registerAgentExecutor } from './orchestrator.js';

describe('Orchestrator run replay timeline', () => {
    beforeEach(() => {
        redisStore.clear();
        streamStore.clear();
        streamSeq = 0;
        clearAgentExecutors();
        vi.clearAllMocks();
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('records per-run activity events and exposes them via getRunReplay', async () => {
        // Force workflow calls to fail so we exercise local executors.
        const fetchMock = vi.fn(async () => ({
            ok: false,
            status: 503,
            text: async () => 'unavailable',
        }));
        vi.stubGlobal('fetch', fetchMock);

        // Register simple executors so the run completes without approvals,
        // regardless of which agents the planner selects.
        const okExecutor = async () => ({
            success: true,
            outputs: { answer: 'ok' },
            explanation: 'done',
            tokens_used: 1,
            duration_ms: 1,
            model_used: 'test',
        });

        const agentTypes = [
            'researcher',
            'analyst',
            'coder',
            'critic',
            'skeptic',
            'designer',
            'librarian',
            'toolsmith',
            'update',
            'vision',
        ] as const;

        for (const agentType of agentTypes) {
            registerAgentExecutor(agentType as any, okExecutor);
        }

        const orchestrator = new Orchestrator();
        const { run, task } = await orchestrator.createRun('user-1', 'Research user onboarding');

        await orchestrator.executeRun(run.id);

        const replay = await orchestrator.getRunReplay(run.id);
        expect(replay).not.toBeNull();
        expect(replay?.run.id).toBe(run.id);
        expect(replay?.task.id).toBe(task.id);

        expect(replay?.task.status).toBe('completed');
        expect(replay?.run.status).toBe('terminated');
        expect(replay?.run.termination_reason).toBe('completed');

        const eventTypes = (replay?.events ?? []).map(e => e.fields['event_type']);
        expect(eventTypes).toContain('run_created');
        expect(eventTypes).toContain('run_started');
        expect(eventTypes).toContain('run_terminated');

        const firstRunCreated = eventTypes.indexOf('run_created');
        const firstRunStarted = eventTypes.indexOf('run_started');
        const firstRunTerminated = eventTypes.indexOf('run_terminated');
        expect(firstRunCreated).toBeGreaterThanOrEqual(0);
        expect(firstRunStarted).toBeGreaterThan(firstRunCreated);
        expect(firstRunTerminated).toBeGreaterThan(firstRunStarted);

        // If step events exist, they must occur between run_started and run_terminated.
        const stepStartedIndex = eventTypes.indexOf('step_started');
        if (stepStartedIndex >= 0) {
            expect(stepStartedIndex).toBeGreaterThan(firstRunStarted);
            expect(stepStartedIndex).toBeLessThan(firstRunTerminated);
        }

        // Ensure events are coming from the per-run stream.
        const runStreamKey = `stream:activity:run:${run.id}`;
        expect(streamStore.get(runStreamKey)?.length).toBeGreaterThan(0);
    });
});
