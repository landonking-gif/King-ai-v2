/**
 * Role Registry Types
 * Stored records for role definitions and contracts
 */

import type { AgentType } from './agent.js';
import type { AgentContract } from './contract.js';
import type { RoleDefinition } from './role.js';

export interface RoleRegistryEntry {
    agent: AgentType;
    role: RoleDefinition;
    contract: AgentContract;
    registered_at: string;
    updated_at: string;
}
