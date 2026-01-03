import { beforeEach, describe, expect, it, vi } from 'vitest';

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
    };
});

import { Orchestrator } from './orchestrator.js';

describe('Orchestrator state machine', () => {
    beforeEach(() => {
        redisStore.clear();
        vi.clearAllMocks();
    });

    it('moves tasks into approval_needed and creates approvals', async () => {
        const orchestrator = new Orchestrator();

        const task = await orchestrator.createTask('user-1', 'Build a pipeline');

        await orchestrator.executeTask(task.id);

        const stored = await orchestrator.getTask(task.id);
        expect(stored?.status).toBe('approval_needed');

        const approvals = await orchestrator.getPendingApprovals('user-1');
        expect(approvals).toHaveLength(1);
        expect(approvals[0]?.task_id).toBe(task.id);
    });

    it('completes tasks that do not require approval', async () => {
        const orchestrator = new Orchestrator();

        const task = await orchestrator.createTask('user-1', 'Research user onboarding');

        await orchestrator.executeTask(task.id);

        const stored = await orchestrator.getTask(task.id);
        expect(stored?.status).toBe('completed');
        expect(stored?.steps_completed.length).toBe(stored?.todo_list.length);
    });
});
