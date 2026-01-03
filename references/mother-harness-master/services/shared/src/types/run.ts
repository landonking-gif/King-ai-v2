/**
 * Run Types and Schemas
 * Execution lifecycle tracking for orchestrator runs
 */

import type { TerminationReason } from './result.js';

/** Run lifecycle status */
export type RunStatus =
    | 'created'
    | 'executing'
    | 'waiting_approval'
    | 'terminated';

/** Run record stored in Redis as run:{id} */
export interface Run {
    id: string;
    task_id: string;
    project_id: string;
    user_id: string;
    status: RunStatus;

    created_at: string;
    updated_at: string;
    started_at?: string;
    terminated_at?: string;

    termination_reason?: TerminationReason;
    termination_details?: string;
    total_tokens?: number;
    total_duration_ms?: number;
}
