/**
 * Result Types and Schemas
 * Termination and result envelope structures (v4 robustness)
 */

import type { AgentType } from './agent.js';

/** Reasons a run can terminate */
export type TerminationReason =
    | 'completed'           // Task finished successfully
    | 'user_cancelled'      // User cancelled the run
    | 'approval_rejected'   // User rejected an approval request
    | 'approval_timeout'    // Approval request expired
    | 'budget_exhausted'    // Resource budget exceeded
    | 'rate_limited'        // Rate limit hit
    | 'timeout'             // Overall timeout exceeded
    | 'max_retries'         // Max retry attempts reached
    | 'agent_error'         // Agent threw an error
    | 'validation_error'    // Schema validation failed
    | 'dependency_failed'   // A prerequisite step failed
    | 'conflict_unresolved' // Agent disagreement not resolved
    | 'circuit_breaker';    // Global limit triggered

/** Termination record for audit - stored in Redis as termination:{run_id} */
export interface TerminationRecord {
    run_id: string;
    task_id: string;
    project_id: string;
    user_id: string;

    reason: TerminationReason;
    details: string;               // Human-readable explanation

    // Context at termination
    last_step_id?: string;
    last_agent?: AgentType;
    error_stack?: string;

    // Metrics
    total_steps_planned: number;
    steps_completed: number;
    total_tokens: number;
    total_duration_ms: number;

    // Timestamps
    started_at: string;
    terminated_at: string;
}

/** Conflict types between agents */
export type ConflictType =
    | 'answer_disagreement'   // Agents gave different answers
    | 'approach_disagreement' // Different approaches suggested
    | 'risk_assessment'       // Different risk evaluations
    | 'evidence_conflict';    // Contradicting evidence

/** Conflict resolution strategies */
export type ResolutionStrategy =
    | 'arbiter'         // Use a third agent to decide
    | 'escalate'        // Escalate to user
    | 'prefer_evidence' // Prefer the answer with more citations
    | 'prefer_critic'   // Prefer the more conservative answer
    | 'majority';       // Use majority vote

/** Conflict resolution record */
export interface ConflictResolution {
    id: string;
    task_id: string;
    step_id: string;

    conflict_type: ConflictType;
    agents_involved: AgentType[];
    positions: {
        agent: AgentType;
        position: string;
        evidence?: string[];
    }[];

    strategy_used: ResolutionStrategy;
    resolution: string;
    resolved_by: AgentType | 'user';

    created_at: string;
    resolved_at: string;
}

/** Result envelope - standard wrapper for all task results */
export interface ResultEnvelope<T = unknown> {
    success: boolean;
    data?: T;
    error?: {
        code: string;
        message: string;
        details?: unknown;
    };

    // Metadata
    task_id: string;
    run_id: string;

    // Termination info
    termination: {
        reason: TerminationReason;
        details: string;
    };

    // Metrics
    metrics: {
        total_tokens: number;
        total_duration_ms: number;
        agents_used: AgentType[];
        steps_completed: number;
    };

    // Context summary (v4: no silent truncation)
    context_summary?: {
        original_tokens: number;
        summarized_tokens: number;
        summarization_method: 'extractive' | 'llm' | 'none';
    };

    timestamp: string;
}

/** Create a success result envelope */
export function createSuccessResult<T>(
    data: T,
    taskId: string,
    runId: string,
    metrics: ResultEnvelope['metrics']
): ResultEnvelope<T> {
    return {
        success: true,
        data,
        task_id: taskId,
        run_id: runId,
        termination: {
            reason: 'completed',
            details: 'Task completed successfully',
        },
        metrics,
        timestamp: new Date().toISOString(),
    };
}

/** Create an error result envelope */
export function createErrorResult(
    error: { code: string; message: string; details?: unknown },
    taskId: string,
    runId: string,
    reason: TerminationReason,
    metrics: ResultEnvelope['metrics']
): ResultEnvelope {
    return {
        success: false,
        error,
        task_id: taskId,
        run_id: runId,
        termination: {
            reason,
            details: error.message,
        },
        metrics,
        timestamp: new Date().toISOString(),
    };
}
