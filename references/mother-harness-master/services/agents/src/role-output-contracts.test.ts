import { describe, it, expect, vi } from 'vitest';

// Minimal shared mocks so we can call agent.run() directly.
vi.mock('@mother-harness/shared', () => {
    return {
        getLLMClient: () => ({
            json: vi.fn(async (prompt: string) => {
                // UpdateAgent: inventory extraction / update analysis
                if (prompt.includes('Extract software items from this input')) {
                    return {
                        data: [{ name: 'vitest', type: 'dependency', current_version: '1.0.0', source: 'npm' }],
                        raw: { tokens_used: { total: 1 } },
                    };
                }

                if (prompt.includes('Analyze these software items for updates')) {
                    return {
                        data: [{
                            name: 'vitest',
                            current_version: '1.0.0',
                            latest_version: '1.6.1',
                            update_urgency: 'high',
                            breaking_changes: false,
                            security_related: false,
                            summary: 'Bugfixes',
                            migration_notes: ['Run pnpm install'],
                            rollback_plan: 'Revert lockfile',
                        }],
                        raw: { tokens_used: { total: 1 } },
                    };
                }

                // ToolsmithAgent: tool design
                if (prompt.includes('Design a deterministic tool based on this request')) {
                    return {
                        data: {
                            name: 'string_reverse',
                            description: 'Reverse a string input',
                            parameters: {
                                input: { type: 'string', description: 'string to reverse', required: true },
                            },
                            return_type: 'string',
                            implementation: "return String(params.input ?? '').split('').reverse().join('');",
                            examples: ['string_reverse({input: "abc"}) -> "cba"'],
                        },
                        raw: { tokens_used: { total: 1 } },
                    };
                }

                // LibrarianAgent: parse command
                if (prompt.includes('Parse this librarian command')) {
                    return {
                        data: {
                            command: 'create_library',
                            library_name: 'Test Library',
                            documents: [],
                            metadata: { description: 'desc', folder_path: '/tmp' },
                        },
                        raw: { tokens_used: { total: 1 } },
                    };
                }

                return { data: null, raw: { tokens_used: { total: 0 } } };
            }),
            embed: vi.fn(async () => ({ embeddings: [[0.1, 0.2]], raw: { tokens_used: { total: 1 } } })),
            complete: vi.fn(async () => ({ finish_reason: 'stop', content: 'ok' })),
        }),
        getToolRegistry: () => ({
            register: vi.fn(async () => undefined),
        }),
        getRedisJSON: () => ({
            get: vi.fn(async () => null),
            set: vi.fn(async () => undefined),
            exists: vi.fn(async () => 0),
            keys: vi.fn(async () => []),
        }),
        getRedisClient: () => ({
            xadd: vi.fn(async () => 'id'),
            call: vi.fn(async () => []),
        }),
        getRoleRegistry: () => ({}),
        getContractEnforcer: () => ({}),
    };
});

import type { AgentContext } from './base-agent.js';
import { UpdateAgent } from './update.js';
import { ToolsmithAgent } from './toolsmith.js';
import { LibrarianAgent } from './librarian.js';

describe('Role-required outputs are emitted', () => {
    const ctx: AgentContext = {
        task_id: 't',
        step_id: 's',
        project_id: 'p',
        user_id: 'u',
        recent_context: 'c',
    };

    it('UpdateAgent emits update_recommendations', async () => {
        const agent = new UpdateAgent();
        const result = await (agent as any).run('check updates', ctx);

        expect(result.outputs).toHaveProperty('update_recommendations');
        expect(Array.isArray(result.outputs.update_recommendations)).toBe(true);
    });

    it('ToolsmithAgent emits tool_definition and tool_code', async () => {
        const agent = new ToolsmithAgent();
        const result = await (agent as any).run('make a tool', ctx);

        expect(result.outputs).toHaveProperty('tool_definition');
        expect(result.outputs).toHaveProperty('tool_code');
        expect(typeof result.outputs.tool_code).toBe('string');
    });

    it('LibrarianAgent emits ingestion_report', async () => {
        const agent = new LibrarianAgent();
        const result = await (agent as any).run('create a library', ctx);

        expect(result.outputs).toHaveProperty('ingestion_report');
        expect(result.outputs.ingestion_report).toMatchObject({
            command: 'create_library',
            success: true,
        });
    });
});
