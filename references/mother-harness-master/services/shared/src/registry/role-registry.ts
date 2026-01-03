/**
 * Role Registry
 * Manages agent role definitions in Redis
 */

import type { AgentType } from '../types/agent.js';
import type { AgentContract } from '../types/contract.js';
import type { RoleRegistryEntry } from '../types/registry.js';
import { getRedisJSON } from '../redis/index.js';
import { type RoleDefinition, DEFAULT_ROLES } from '../types/role.js';
import { DEFAULT_AGENT_CONTRACTS } from '../types/contract.js';

export class RoleRegistry {
    private redis = getRedisJSON();
    private readonly keyPrefix = 'role:';
    private readonly registryPrefix = 'role_registry:';

    /**
     * Initialize registry with default roles
     */
    async initialize(): Promise<void> {
        for (const [agentType, role] of Object.entries(DEFAULT_ROLES)) {
            const exists = await this.redis.exists(`${this.keyPrefix}${agentType}`);
            if (!exists) {
                await this.registerRole(role);
            }
        }
        for (const [agentType, contract] of Object.entries(DEFAULT_AGENT_CONTRACTS)) {
            const exists = await this.redis.exists(`${this.registryPrefix}${agentType}`);
            if (!exists) {
                await this.registerContract(contract);
            }
        }
        console.log('[RoleRegistry] Initialized with default roles');
    }

    /**
     * Register a new role or update existing
     */
    async registerRole(role: RoleDefinition): Promise<void> {
        await this.redis.set(`${this.keyPrefix}${role.type}`, '$', role);
    }

    /**
     * Register an agent contract
     */
    async registerContract(contract: AgentContract): Promise<void> {
        await this.redis.set(`${this.registryPrefix}${contract.agent}`, '$', {
            ...contract,
            registered_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
        });
    }

    /**
     * Register full role registry entry
     */
    async registerEntry(entry: RoleRegistryEntry): Promise<void> {
        await this.redis.set(`${this.registryPrefix}${entry.agent}`, '$', entry);
    }

    /**
     * Get role definition
     */
    async getRole(agentType: AgentType): Promise<RoleDefinition | null> {
        return await this.redis.get<RoleDefinition>(`${this.keyPrefix}${agentType}`);
    }

    /**
     * Get agent contract
     */
    async getContract(agentType: AgentType): Promise<AgentContract | null> {
        return await this.redis.get<AgentContract>(`${this.registryPrefix}${agentType}`);
    }

    /**
     * Get all registered roles
     */
    async getAllRoles(): Promise<RoleDefinition[]> {
        const keys = await this.redis.keys(`${this.keyPrefix}*`);
        const roles: RoleDefinition[] = [];

        for (const key of keys) {
            const role = await this.redis.get<RoleDefinition>(key);
            if (role) roles.push(role);
        }

        return roles;
    }

    /**
     * Get all registered contracts
     */
    async getAllContracts(): Promise<AgentContract[]> {
        const keys = await this.redis.keys(`${this.registryPrefix}*`);
        const contracts: AgentContract[] = [];

        for (const key of keys) {
            const contract = await this.redis.get<AgentContract>(key);
            if (contract) contracts.push(contract);
        }

        return contracts;
    }

    /**
     * Get enabled roles only
     */
    async getEnabledRoles(): Promise<RoleDefinition[]> {
        const roles = await this.getAllRoles();
        return roles.filter(r => r.enabled);
    }

    /**
     * Enable or disable a role
     */
    async setRoleEnabled(agentType: AgentType, enabled: boolean): Promise<void> {
        await this.redis.set(`${this.keyPrefix}${agentType}`, '$.enabled', enabled);
    }

    /**
     * Update contract allowlist
     */
    async setContractAllowlist(agentType: AgentType, allowlist: string[]): Promise<void> {
        await this.redis.set(`${this.registryPrefix}${agentType}`, '$.action_allowlist', allowlist);
        await this.redis.set(`${this.registryPrefix}${agentType}`, '$.updated_at', new Date().toISOString());
    }

    /**
     * Update contract required artifacts
     */
    async setRequiredArtifacts(agentType: AgentType, required: string[]): Promise<void> {
        await this.redis.set(`${this.registryPrefix}${agentType}`, '$.required_artifacts', required);
        await this.redis.set(`${this.registryPrefix}${agentType}`, '$.updated_at', new Date().toISOString());
    }

    /**
     * Check if agent has a specific capability
     */
    async hasCapability(
        agentType: AgentType,
        capability: RoleDefinition['capabilities'][number]
    ): Promise<boolean> {
        const role = await this.getRole(agentType);
        if (!role) return false;
        return role.capabilities.includes(capability);
    }

    /**
     * Get approval requirements for an action
     */
    async getApprovalRequirements(
        agentType: AgentType,
        action: string
    ): Promise<RoleDefinition['approval_requirements'][number] | null> {
        const role = await this.getRole(agentType);
        if (!role) return null;

        return role.approval_requirements.find(r => r.action === action) ?? null;
    }

    /**
     * Get agents with a specific capability
     */
    async getAgentsWithCapability(
        capability: RoleDefinition['capabilities'][number]
    ): Promise<AgentType[]> {
        const roles = await getEnabledRoles();
        return roles
            .filter(r => r.capabilities.includes(capability))
            .map(r => r.type);
    }

    /**
     * Validate that required outputs are present
     */
    async validateOutputs(
        agentType: AgentType,
        outputs: Record<string, unknown>
    ): Promise<{ valid: boolean; missing: string[] }> {
        const role = await this.getRole(agentType);
        if (!role) return { valid: false, missing: ['role_not_found'] };

        const missing = role.required_outputs.filter(
            output => !(output in outputs)
        );

        return {
            valid: missing.length === 0,
            missing,
        };
    }
}

// Helper function used in getAgentsWithCapability
async function getEnabledRoles(): Promise<RoleDefinition[]> {
    const registry = new RoleRegistry();
    return registry.getEnabledRoles();
}

// Singleton instance
let registryInstance: RoleRegistry | null = null;

export function getRoleRegistry(): RoleRegistry {
    if (!registryInstance) {
        registryInstance = new RoleRegistry();
    }
    return registryInstance;
}
