import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const redisStore = new Map<string, Record<string, unknown>>();

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
        return getByPath(value, path) as T ?? null;
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

const mockRedisClient = {
    ping: vi.fn(() => 'PONG'),
    xadd: vi.fn(async () => 'test-event-id'),
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

import { N8nAdapter } from './n8n-adapter.js';
import { Orchestrator, clearAgentExecutors, registerAgentExecutor } from './orchestrator.js';

describe('Orchestrator ↔ n8n ↔ agents ↔ Redis integration', () => {
    beforeEach(() => {
        redisStore.clear();
        clearAgentExecutors();
        vi.clearAllMocks();
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('executes an agent workflow through n8n and persists task state', async () => {
        const fetchMock = vi.fn(async () => ({
            ok: true,
            json: async () => ({ result: 'ok' }),
            text: async () => 'ok',
        }));

        vi.stubGlobal('fetch', fetchMock);

        const adapter = new N8nAdapter();

        registerAgentExecutor('researcher', async (inputs, context) => {
            const result = await adapter.triggerWorkflow('agent-run', { inputs, context });
            return {
                success: result.success,
                outputs: { data: result.data },
                explanation: 'Workflow executed',
                tokens_used: 12,
                duration_ms: result.duration_ms,
            };
        });

        registerAgentExecutor('critic', async (inputs, context) => {
            const result = await adapter.triggerWorkflow('agent-run', { inputs, context });
            return {
                success: result.success,
                outputs: { data: result.data },
                explanation: 'Workflow reviewed',
                tokens_used: 8,
                duration_ms: result.duration_ms,
            };
        });

        const orchestrator = new Orchestrator();
        const task = await orchestrator.createTask('user-1', 'Research onboarding metrics');

        await orchestrator.executeTask(task.id);

        const stored = await orchestrator.getTask(task.id);
        expect(stored?.status).toBe('completed');
        expect(stored?.steps_completed.length).toBe(stored?.todo_list.length);

        expect(fetchMock).toHaveBeenCalled();
        expect(fetchMock).toHaveBeenCalledWith(
            expect.stringContaining('/webhook/researcher'),
            expect.objectContaining({ method: 'POST' })
        );
        expect(fetchMock).toHaveBeenCalledWith(
            expect.stringContaining('/webhook/analyst'),
            expect.objectContaining({ method: 'POST' })
        );
        expect(fetchMock).toHaveBeenCalledWith(
            expect.stringContaining('/webhook/critic'),
            expect.objectContaining({ method: 'POST' })
        );
    });

    it('fails when workflows error and no executor is registered', async () => {
        const fetchMock = vi.fn(async () => ({
            ok: false,
            status: 500,
            text: async () => 'workflow down',
        }));

        vi.stubGlobal('fetch', fetchMock);

        const orchestrator = new Orchestrator();
        const task = await orchestrator.createTask('user-1', 'Research onboarding metrics');

        await orchestrator.executeTask(task.id);

        const stored = await orchestrator.getTask(task.id);
        expect(stored?.status).toBe('failed');
        expect(stored?.todo_list[0]?.status).toBe('failed');
        expect(fetchMock).toHaveBeenCalled();
    });
});
