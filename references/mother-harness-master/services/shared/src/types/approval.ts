/**
 * Approval Types and Schemas
 * User approval workflow for high-risk operations
 */

import type { ApprovalType, RiskLevel } from './task.js';

/** Approval request status */
export type ApprovalStatus = 'pending' | 'approved' | 'rejected';

/** Approval request - stored in Redis as approval:{id} */
export interface Approval {
    id: string;
    run_id: string;
    task_id: string;
    project_id: string;
    step_id: string;
    user_id: string;

    // Request details
    type: ApprovalType;
    description: string;
    risk_level: RiskLevel;

    // Preview of what will be done
    preview: ApprovalPreview;

    // Response
    status: ApprovalStatus;
    response_notes?: string;       // User explanation for approval/rejection

    // Timestamps
    created_at: string;
    responded_at?: string;
    expires_at?: string;           // Auto-reject after this time
}

/** Preview of the operation requiring approval */
export interface ApprovalPreview {
    files?: string[];              // Files that will be modified
    commands?: string[];           // Commands that will be executed
    workflow?: Record<string, unknown>; // Workflow configuration
    api_calls?: ApiCallPreview[];  // External API calls
}

/** Preview of an API call */
export interface ApiCallPreview {
    method: string;
    url: string;
    description: string;
}

/** Create a new approval request */
export function createApproval(
    id: string,
    runId: string,
    taskId: string,
    projectId: string,
    stepId: string,
    userId: string,
    type: ApprovalType,
    description: string,
    riskLevel: RiskLevel,
    preview: ApprovalPreview
): Approval {
    const now = new Date().toISOString();
    return {
        id,
        run_id: runId,
        task_id: taskId,
        project_id: projectId,
        step_id: stepId,
        user_id: userId,
        type,
        description,
        risk_level: riskLevel,
        preview,
        status: 'pending',
        created_at: now,
    };
}

/** Check if an approval is still pending and not expired */
export function isApprovalPending(approval: Approval): boolean {
    if (approval.status !== 'pending') return false;
    if (approval.expires_at) {
        return new Date(approval.expires_at) > new Date();
    }
    return true;
}
