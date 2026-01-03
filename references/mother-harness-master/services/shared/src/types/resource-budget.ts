/**
 * Resource Budget Types and Counters
 * Shared budget definitions for resource consumption tracking
 */

/** Budget scope */
export type BudgetScope = 'run' | 'user' | 'global';

/** Resource types */
export type ResourceType =
    | 'tokens'            // LLM tokens
    | 'api_calls'         // External API calls
    | 'tool_executions'   // Tool runs
    | 'agent_invocations' // Agent calls
    | 'embeddings'        // Embedding generations
    | 'storage_bytes'     // Storage used
    | 'cost';             // Monetary spend (USD)

/** Resource counters */
export interface ResourceBudgetCounters {
    tokens: number;
    api_calls: number;
    tool_executions: number;
    agent_invocations: number;
    embeddings: number;
    storage_bytes: number;
    cost: number;
}

/** Budget definition */
export interface ResourceBudget {
    scope: BudgetScope;
    scope_id: string;    // run_id, user_id, or 'global'
    limits: ResourceBudgetCounters;
    used: ResourceBudgetCounters;
    warnings_sent: ResourceType[];
    created_at: string;
    updated_at: string;
}

export interface ResourceUsageReportItem {
    used: number;
    limit: number;
    percent: number;
}

export type ResourceUsageReport = Record<ResourceType, ResourceUsageReportItem>;
