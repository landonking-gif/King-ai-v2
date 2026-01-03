/**
 * Contract Enforcer
 * Validates agent actions against role contracts
 */

import type { AgentType } from '../types/agent.js';
import type { Task, TodoItem } from '../types/task.js';
import type { Approval } from '../types/approval.js';
import { createApproval } from '../types/approval.js';
import { getRedisJSON } from '../redis/index.js';
import { RoleRegistry, getRoleRegistry } from '../registry/role-registry.js';
import type { AgentCapability } from '../types/role.js';
import { nanoid } from 'nanoid';

/** Result of contract validation */
export interface ContractValidation {
    valid: boolean;
    errors: string[];
    warnings: string[];
    requires_approval?: {
        type: Approval['type'];
        risk_level: Approval['risk_level'];
        description: string;
    };
}

/** Retry accounting for a phase */
export interface RetryAccount {
    task_id: string;
    step_id: string;
    agent: AgentType;
    attempts: number;
    max_retries: number;
    last_error?: string;
    last_attempt_at?: string;
}

export class ContractEnforcer {
    private registry: RoleRegistry;
    private redis = getRedisJSON();
    private readonly retryKeyPrefix = 'retry:';

    constructor() {
        this.registry = getRoleRegistry();
    }

    /**
     * Validate that an agent can perform the requested action
     */
    async validateAction(
        agentType: AgentType,
        action: string,
        capabilities_needed: AgentCapability[]
    ): Promise<ContractValidation> {
        const errors: string[] = [];
        const warnings: string[] = [];

        // Get role definition
        const role = await this.registry.getRole(agentType);
        if (!role) {
            return {
                valid: false,
                errors: [`Role not found for agent: ${agentType}`],
                warnings: [],
            };
        }

        // Check if role is enabled
        if (!role.enabled) {
            return {
                valid: false,
                errors: [`Agent ${agentType} is disabled`],
                warnings: [],
            };
        }

        // Check required capabilities
        for (const cap of capabilities_needed) {
            if (!role.capabilities.includes(cap)) {
                errors.push(`Agent ${agentType} lacks capability: ${cap}`);
            }
        }

        // Check if action requires approval
        const approvalReq = await this.registry.getApprovalRequirements(agentType, action);

        if (errors.length > 0) {
            return { valid: false, errors, warnings };
        }

        if (approvalReq) {
            return {
                valid: true,
                errors: [],
                warnings: [`Action ${action} requires ${approvalReq.risk_level} approval`],
                requires_approval: {
                    type: approvalReq.approval_type,
                    risk_level: approvalReq.risk_level,
                    description: approvalReq.description,
                },
            };
        }

        return { valid: true, errors: [], warnings };
    }

    /**
     * Validate action allowlist from the agent contract
     */
    async validateAllowlist(
        agentType: AgentType,
        action: string
    ): Promise<ContractValidation> {
        const contract = await this.registry.getContract(agentType);

        if (!contract) {
            return {
                valid: false,
                errors: [`Contract not found for agent: ${agentType}`],
                warnings: [],
            };
        }

        if (!contract.action_allowlist.includes(action)) {
            return {
                valid: false,
                errors: [`Action ${action} is not allowlisted for agent: ${agentType}`],
                warnings: [],
            };
        }

        return { valid: true, errors: [], warnings: [] };
    }

    /**
     * Validate required artifacts from the agent contract
     */
    async validateRequiredArtifacts(
        agentType: AgentType,
        artifacts: string[]
    ): Promise<ContractValidation> {
        const contract = await this.registry.getContract(agentType);

        if (!contract) {
            return {
                valid: false,
                errors: [`Contract not found for agent: ${agentType}`],
                warnings: [],
            };
        }

        const missing = contract.required_artifacts.filter(
            artifact => !artifacts.includes(artifact)
        );

        if (missing.length > 0) {
            return {
                valid: false,
                errors: missing.map(m => `Missing required artifact: ${m}`),
                warnings: [],
            };
        }

        return { valid: true, errors: [], warnings: [] };
    }

    /**
     * Validate agent outputs meet phase exit criteria
     */
    async validatePhaseExit(
        agentType: AgentType,
        outputs: Record<string, unknown>
    ): Promise<ContractValidation> {
        const result = await this.registry.validateOutputs(agentType, outputs);

        if (!result.valid) {
            return {
                valid: false,
                errors: result.missing.map(m => `Missing required output: ${m}`),
                warnings: [],
            };
        }

        return { valid: true, errors: [], warnings: [] };
    }

    /**
     * Check if retry is allowed for a step
     */
    async canRetry(task: Task, step: TodoItem): Promise<boolean> {
        const role = await this.registry.getRole(step.agent);
        if (!role) return false;

        const retryKey = `${this.retryKeyPrefix}${task.id}:${step.id}`;
        const account = await this.redis.get<RetryAccount>(retryKey);

        if (!account) {
            // First attempt
            return true;
        }

        return account.attempts < role.max_retries;
    }

    /**
     * Record a retry attempt
     */
    async recordRetry(
        task: Task,
        step: TodoItem,
        error?: string
    ): Promise<RetryAccount> {
        const role = await this.registry.getRole(step.agent);
        const maxRetries = role?.max_retries ?? 3;

        const retryKey = `${this.retryKeyPrefix}${task.id}:${step.id}`;
        const existing = await this.redis.get<RetryAccount>(retryKey);

        const account: RetryAccount = {
            task_id: task.id,
            step_id: step.id,
            agent: step.agent,
            attempts: (existing?.attempts ?? 0) + 1,
            max_retries: maxRetries,
            ...(error !== undefined && { last_error: error }),
            last_attempt_at: new Date().toISOString(),
        };

        await this.redis.set(retryKey, '$', account);
        return account;
    }

    /**
     * Get retry account for a step
     */
    async getRetryAccount(taskId: string, stepId: string): Promise<RetryAccount | null> {
        return await this.redis.get<RetryAccount>(`${this.retryKeyPrefix}${taskId}:${stepId}`);
    }

    /**
     * Create approval request if action requires it
     */
    async createApprovalIfNeeded(
        task: Task,
        step: TodoItem,
        runId: string,
        action: string,
        preview: Approval['preview']
    ): Promise<Approval | null> {
        const validation = await this.validateAction(step.agent, action, []);

        if (!validation.requires_approval) {
            return null;
        }

        const approval = createApproval(
            `approval-${nanoid()}`,
            runId,
            task.id,
            task.project_id,
            step.id,
            task.user_id,
            validation.requires_approval.type,
            validation.requires_approval.description,
            validation.requires_approval.risk_level,
            preview
        );

        await this.redis.set(`approval:${approval.id}`, '$', approval);
        return approval;
    }

    /**
     * Get timeout for an agent
     */
    async getTimeout(agentType: AgentType): Promise<number> {
        const role = await this.registry.getRole(agentType);
        return role?.timeout_ms ?? 60000;
    }

    /**
     * Get max tokens for an agent
     */
    async getMaxTokens(agentType: AgentType): Promise<number> {
        const role = await this.registry.getRole(agentType);
        return role?.max_tokens_per_call ?? 8000;
    }
}

// Singleton instance
let enforcerInstance: ContractEnforcer | null = null;

export function getContractEnforcer(): ContractEnforcer {
    if (!enforcerInstance) {
        enforcerInstance = new ContractEnforcer();
    }
    return enforcerInstance;
}
