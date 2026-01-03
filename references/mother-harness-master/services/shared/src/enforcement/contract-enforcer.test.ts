import { describe, it, expect, vi } from 'vitest';

vi.mock('../registry/role-registry.js', () => {
    return {
        RoleRegistry: class RoleRegistry {},
        getRoleRegistry: () => ({
            validateOutputs: vi.fn(async (_agentType: string, outputs: Record<string, unknown>) => {
                if ('review_report' in outputs && 'issues_found' in outputs) {
                    return { valid: true, missing: [] };
                }
                const missing: string[] = [];
                if (!('review_report' in outputs)) missing.push('review_report');
                if (!('issues_found' in outputs)) missing.push('issues_found');
                return { valid: false, missing };
            }),
        }),
    };
});

vi.mock('../redis/index.js', () => {
    return {
        getRedisJSON: () => ({
            get: vi.fn(async () => null),
            set: vi.fn(async () => undefined),
        }),
    };
});

import { ContractEnforcer } from './contract-enforcer.js';

describe('ContractEnforcer.validatePhaseExit', () => {
    it('rejects missing required outputs', async () => {
        const enforcer = new ContractEnforcer();

        const result = await enforcer.validatePhaseExit('critic' as never, {
            review_report: { approval: 'approved', summary: 'ok' },
        });

        expect(result.valid).toBe(false);
        expect(result.errors).toContain('Missing required output: issues_found');
    });

    it('accepts when required outputs exist', async () => {
        const enforcer = new ContractEnforcer();

        const result = await enforcer.validatePhaseExit('critic' as never, {
            review_report: { approval: 'approved', summary: 'ok' },
            issues_found: [],
        });

        expect(result.valid).toBe(true);
        expect(result.errors).toEqual([]);
    });
});
