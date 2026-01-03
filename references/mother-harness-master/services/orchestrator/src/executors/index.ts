/**
 * Production executor registration
 */

import { DEFAULT_AGENT_MODELS, type AgentType } from '@mother-harness/shared';
import { registerAgentExecutor } from '../orchestrator.js';

type AgentExecutor = (inputs: string, context: Record<string, unknown>) => Promise<{
    success: boolean;
    outputs: Record<string, unknown>;
    artifacts?: string[];
    explanation?: string;
    tokens_used?: number;
    duration_ms?: number;
    model_used?: string;
}>;

const workflowOnlyExecutor = (agent: AgentType): AgentExecutor => {
    return async () => {
        throw new Error(`Direct executor not configured for agent: ${agent}. Workflow execution is required.`);
    };
};

const allAgents = Object.keys(DEFAULT_AGENT_MODELS) as AgentType[];

export function registerProductionExecutors(): void {
    for (const agent of allAgents) {
        registerAgentExecutor(agent, workflowOnlyExecutor(agent));
    }
}
