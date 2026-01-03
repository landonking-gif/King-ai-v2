import { describe, expect, it, vi } from 'vitest';

vi.mock('@mother-harness/shared', () => ({
    getLLMClient: () => ({}),
    getRedisJSON: () => ({}),
    getRoleRegistry: () => ({}),
    getContractEnforcer: () => ({}),
}));

import { CoderAgent } from './coder.js';
import type { AgentContext } from './base-agent.js';

describe('CoderAgent runTests', () => {
    it('returns failure when ToolExecutor is missing', async () => {
        const agent = new CoderAgent();
        const changes = [
            { file_path: 'src/example.ts', change_type: 'modify' as const },
        ];

        const result = await (agent as { runTests: Function }).runTests(
            changes,
            {} as AgentContext
        );

        expect(result.passed).toBe(false);
        expect(result.output).toContain('ToolExecutor not available');
    });
});
