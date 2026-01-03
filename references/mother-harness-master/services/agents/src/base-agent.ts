/**
 * Base Agent
 * Abstract base class for all specialist agents
 */

import type { AgentRequest, AgentResponse, AgentType } from '@mother-harness/shared';
import { getRedisJSON, getRoleRegistry, getContractEnforcer } from '@mother-harness/shared';
import type { RoleDefinition } from '@mother-harness/shared';

/** Agent execution context */
export interface AgentContext {
    task_id: string;
    step_id: string;
    project_id: string;
    user_id: string;
    recent_context?: string;
    rag_context?: string;
    library_ids?: string[];
}

/** Agent execution result */
export interface AgentResult {
    success: boolean;
    outputs: Record<string, unknown>;
    explanation?: string;
    sources?: string[];
    artifacts?: string[];
    tokens_used: number;
    duration_ms: number;
}

export abstract class BaseAgent {
    protected redis = getRedisJSON();
    protected registry = getRoleRegistry();
    protected enforcer = getContractEnforcer();
    protected role: RoleDefinition | null = null;

    abstract readonly agentType: AgentType;

    /**
     * Initialize the agent
     */
    async initialize(): Promise<void> {
        this.role = await this.registry.getRole(this.agentType);
        if (!this.role) {
            throw new Error(`Role not found for agent: ${this.agentType}`);
        }
        if (!this.role.enabled) {
            throw new Error(`Agent ${this.agentType} is disabled`);
        }
    }

    /**
     * Execute the agent's task
     */
    async execute(request: AgentRequest, context: AgentContext): Promise<AgentResponse> {
        const startTime = Date.now();

        // Ensure initialized
        if (!this.role) {
            await this.initialize();
        }

        try {
            // Execute the agent-specific logic
            const result = await this.run(request.inputs, context);

            const duration = Date.now() - startTime;

            // Validate outputs
            const validation = await this.enforcer.validatePhaseExit(
                this.agentType,
                result.outputs
            );

            if (!validation.valid) {
                console.warn(
                    `[${this.agentType}] Output validation warnings:`,
                    validation.errors
                );
            }

            return {
                agent: this.agentType,
                model_used: request.model ?? this.role!.preferred_local_model,
                result: result.outputs,
                ...(result.explanation !== undefined && { explanation: result.explanation }),
                ...(result.sources !== undefined && { sources: result.sources }),
                ...(result.artifacts !== undefined && { artifacts: result.artifacts }),
                tokens_used: result.tokens_used,
                duration_ms: duration,
                timestamp: new Date().toISOString(),
            };
        } catch (error) {
            const duration = Date.now() - startTime;
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';

            return {
                agent: this.agentType,
                model_used: request.model ?? this.role!.preferred_local_model,
                result: { error: errorMessage },
                tokens_used: 0,
                duration_ms: duration,
                timestamp: new Date().toISOString(),
            };
        }
    }

    /**
     * Agent-specific execution logic - must be implemented by each agent
     */
    protected abstract run(inputs: string, context: AgentContext): Promise<AgentResult>;

    /**
     * Get the agent's preferred model
     */
    getPreferredModel(useCloud: boolean = false): string {
        if (!this.role) {
            return 'gpt-oss:20b';
        }
        return useCloud ? this.role.preferred_cloud_model : this.role.preferred_local_model;
    }

    /**
     * Check if agent has a capability
     */
    async hasCapability(capability: RoleDefinition['capabilities'][number]): Promise<boolean> {
        if (!this.role) {
            await this.initialize();
        }
        return this.role!.capabilities.includes(capability);
    }
}
