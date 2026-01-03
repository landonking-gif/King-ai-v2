/**
 * Integration tests for local agent execution fallback
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mock Redis and dependencies
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

vi.mock('@mother-harness/shared', async () => {
    const actual = await vi.importActual('@mother-harness/shared');
    return {
        ...actual,
        getRedisJSON: () => ({
            get: async <T>(key: string, path: string = '$') => {
                const value = redisStore.get(key);
                if (!value) return null;
                return (getByPath(value, path) as T) ?? null;
            },
            set: async (key: string, path: string, value: unknown) => {
                const current = redisStore.get(key) ?? {};
                setByPath(current, path, value);
                redisStore.set(key, current);
            },
            del: async (key: string) => {
                const existed = redisStore.has(key);
                redisStore.delete(key);
                return existed ? 1 : 0;
            },
            keys: async (pattern: string) => {
                const allKeys = Array.from(redisStore.keys());
                const regex = new RegExp('^' + pattern.replace(/\*/g, '.*') + '$');
                return allKeys.filter(key => regex.test(key));
            },
            arrAppend: async (key: string, path: string, ...values: unknown[]) => {
                const current = redisStore.get(key) ?? {};
                const existing = getByPath(current, path);
                const next = Array.isArray(existing) ? [...existing, ...values] : [...values];
                setByPath(current, path, next);
                redisStore.set(key, current);
                return next.length;
            },
        }),
        getRedisClient: () => ({
            xadd: vi.fn(async () => 'test-event-id'),
            quit: vi.fn(async () => undefined),
        }),
        getLLMClient: () => ({
            chat: vi.fn(),
            embed: vi.fn(async () => ({
                embeddings: [new Array(768).fill(0)],
                model: 'test',
                dimensions: 768,
            })),
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
            validateInputAllowlist: vi.fn(async () => ({ valid: true, errors: [] })),
            validateAllowlist: vi.fn(async () => ({ valid: true, errors: [] })),
            validateRequiredArtifacts: vi.fn(async () => ({ valid: true, errors: [] })),
            validateOutputs: vi.fn(async () => ({ valid: true, errors: [] })),
        }),
        getResourceBudgetGuard: () => ({
            checkAllScopes: vi.fn(async () => ({ allowed: true, remaining: 9999 })),
            recordUsageAllScopes: vi.fn(async () => undefined),
        }),
    };
});

// Keep these tests focused on fallback behavior, not approval gating.
vi.mock('./approval-service.js', async () => {
    const actual = await vi.importActual<typeof import('./approval-service.js')>('./approval-service.js');
    return {
        ...actual,
        getApprovalService: () => ({
            shouldRequireApproval: () => ({
                required: false,
                assessment: {
                    level: 'low',
                    factors: [],
                    requires_manual_approval: false,
                    auto_approvable: true,
                },
            }),
        }),
    };
});

function agentsMockFactory() {
    return {
    createAgent: (_agentType: string) => ({
        execute: vi.fn(async () => ({
            result: {
                answer: 'Test result from local agent',
                sources: ['local'],
            },
            artifacts: [],
            explanation: 'Local agent execution',
            tokens_used: 100,
            duration_ms: 50,
            model_used: 'local-model',
        })),
    }),
    };
}

// Mock both the package specifier and the aliased source path.
vi.mock('@mother-harness/agents', agentsMockFactory);
vi.mock('../../agents/src/index.ts', agentsMockFactory);

import { Orchestrator, clearAgentExecutors, hasAgentExecutor } from './orchestrator.js';

describe('Orchestrator local agent fallback', () => {
    beforeEach(() => {
        redisStore.clear();
        clearAgentExecutors();
        vi.clearAllMocks();
    });

    it('executes agent locally when no executor is registered and workflow fails', async () => {
        // Mock fetch to simulate workflow failure
        const fetchMock = vi.fn(async () => ({
            ok: false,
            status: 500,
            text: async () => 'Workflow unavailable',
        }));
        vi.stubGlobal('fetch', fetchMock);

        const orchestrator = new Orchestrator();
        const task = await orchestrator.createTask('user-local-test', 'Test local execution');

        expect(hasAgentExecutor('researcher')).toBe(false);

        await orchestrator.executeTask(task.id);

        const result = await orchestrator.getTask(task.id);

        expect(result?.status).toBe('completed');
        expect(result?.todo_list[0]?.status).toBe('completed');
        expect(result?.todo_list[0]?.result?.outputs).toMatchObject({
            answer: 'Test result from local agent',
        });
    });

    it('uses workflow when available before falling back to local agent', async () => {
        const workflowResponse = {
            success: true,
            outputs: { answer: 'Workflow result' },
            artifacts: [],
            tokens_used: 200,
        };

        const fetchMock = vi.fn(async () => ({
            ok: true,
            json: async () => workflowResponse,
        }));
        vi.stubGlobal('fetch', fetchMock);

        const orchestrator = new Orchestrator();
        const task = await orchestrator.createTask('user-workflow-test', 'Test workflow priority');

        await orchestrator.executeTask(task.id);

        const result = await orchestrator.getTask(task.id);

        expect(fetchMock).toHaveBeenCalled();
        expect(result?.status).toBe('completed');
        expect(result?.todo_list[0]?.result?.outputs).toMatchObject({
            answer: 'Workflow result',
        });
    });

    it('falls back to local agent when workflow returns error', async () => {
        const fetchMock = vi.fn(async () => ({
            ok: true,
            json: async () => ({
                success: false,
                error: { message: 'Workflow execution failed' },
            }),
        }));
        vi.stubGlobal('fetch', fetchMock);

        const orchestrator = new Orchestrator();
        const task = await orchestrator.createTask('user-fallback-test', 'Test error fallback');

        await orchestrator.executeTask(task.id);

        const result = await orchestrator.getTask(task.id);

        expect(fetchMock).toHaveBeenCalled();
        expect(result?.status).toBe('completed');
        expect(result?.todo_list[0]?.status).toBe('completed');
        expect(result?.todo_list[0]?.result?.outputs).toMatchObject({
            answer: 'Test result from local agent',
        });
    });

    it('uses local agent for all supported agent types', async () => {
        const fetchMock = vi.fn(async () => ({
            ok: false,
            status: 503,
            text: async () => 'Service unavailable',
        }));
        vi.stubGlobal('fetch', fetchMock);

        const orchestrator = new Orchestrator();

        // Test different agent types
        const agentTypes = ['researcher', 'coder', 'analyst', 'critic'];

        for (const agentType of agentTypes) {
            redisStore.clear();
            const task = await orchestrator.createTask(
                'user-test',
                `Test ${agentType} local execution`
            );

            // Override the planner-generated steps so this test stays focused on
            // local fallback behavior and avoids approval-gated steps (e.g. coder).
            const storedTask = redisStore.get(`task:${task.id}`);
            if (storedTask) {
                storedTask.todo_list = [
                    {
                        id: 'step-1',
                        description: `Local execution for ${agentType}`,
                        agent: agentType,
                        status: 'pending',
                        depends_on: [],
                    },
                ];
                storedTask.steps_completed = [];
                storedTask.status = 'planning';
            }

            await orchestrator.executeTask(task.id);

            const result = await orchestrator.getTask(task.id);
            expect(result?.status).toBe('completed');
            expect(result?.todo_list[0]?.result?.outputs).toBeDefined();
        }
    });
});
