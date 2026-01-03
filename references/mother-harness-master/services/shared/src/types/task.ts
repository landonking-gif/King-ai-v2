/**
 * Task Types and Schemas
 * Core task management structures for Mother-Harness orchestration
 */

import type { AgentType } from './agent.js';

/** Task type classification */
export type TaskType = 'research' | 'code' | 'design' | 'analysis' | 'mixed';

/** Task execution status */
export type TaskStatus =
    | 'planning'        // Creating execution plan
    | 'executing'       // Running agent steps
    | 'approval_needed' // Waiting for user approval
    | 'completed'       // Successfully finished
    | 'failed';         // Error occurred

/** Individual step in the execution plan */
export interface TodoItem {
    id: string;                    // e.g., 'step-1'
    description: string;           // Human-readable task description
    agent: AgentType;              // Agent responsible for this step
    status: 'pending' | 'in_progress' | 'completed' | 'failed';
    depends_on: string[];          // IDs of prerequisite steps
    require_approval?: boolean;    // Pause for user approval after completion
    approval_type?: ApprovalType;  // Type of approval if required
    risk?: RiskLevel;              // Risk level for approval decisions
    result?: unknown;              // Step result after completion
    error?: string;                // Error message if failed
    started_at?: string;
    completed_at?: string;
}

/** Approval types that require user intervention */
export type ApprovalType =
    | 'file_write'
    | 'code_execution'
    | 'git_push'
    | 'workflow_creation'
    | 'api_call';

/** Risk level for approval decisions */
export type RiskLevel = 'low' | 'medium' | 'high';

/** Execution plan for a task */
export interface ExecutionPlan {
    steps: TodoItem[];
    estimated_duration?: string;   // e.g., '30-45 minutes'
    created_at: string;
    updated_at: string;
}

/** Artifact produced by a task */
export interface Artifact {
    id: string;
    task_id: string;
    type: 'report' | 'code' | 'diagram' | 'data' | 'other';
    name: string;
    content: string;
    format: 'markdown' | 'json' | 'typescript' | 'mermaid' | 'text';
    created_at: string;
}

/** Task result after completion */
export interface TaskResult {
    summary: string;
    findings?: string[];
    recommendations?: string[];
    artifacts: Artifact[];
    agents_used: AgentType[];
    total_tokens: number;
    total_duration_ms: number;
}

/** Main Task interface - stored in Redis as task:{id} */
export interface Task {
    id: string;                    // task-uuid
    project_id: string;
    user_id: string;
    type: TaskType;
    query: string;                 // Original user query
    status: TaskStatus;

    // Planning
    todo_list: TodoItem[];
    execution_plan: ExecutionPlan;

    // Execution progress
    current_step?: string;         // Current step ID
    steps_completed: string[];     // Completed step IDs

    // Results
    result?: TaskResult;
    artifacts: Artifact[];

    // Metadata
    created_at: string;
    updated_at: string;
    completed_at?: string;
}

/** Create a new task with defaults */
export function createTask(
    id: string,
    projectId: string,
    userId: string,
    query: string,
    type: TaskType = 'mixed'
): Task {
    const now = new Date().toISOString();
    return {
        id,
        project_id: projectId,
        user_id: userId,
        type,
        query,
        status: 'planning',
        todo_list: [],
        execution_plan: {
            steps: [],
            created_at: now,
            updated_at: now,
        },
        steps_completed: [],
        artifacts: [],
        created_at: now,
        updated_at: now,
    };
}
