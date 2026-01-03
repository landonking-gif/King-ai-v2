/**
 * Approval Workflow Implementation Guide
 * 
 * This module provides the implementation details for the approval gating system
 * that ensures risky operations require human review before execution.
 */

import type { TodoItem, Task, RiskLevel } from '@mother-harness/shared';

/**
 * APPROVAL WORKFLOW OVERVIEW
 * ==========================
 * 
 * The approval system provides human-in-the-loop control for high-risk operations.
 * When a task step requires approval, execution pauses until an authorized user
 * reviews and approves/rejects the action.
 * 
 * Flow:
 * 1. Planner marks risky steps with `require_approval: true`
 * 2. Orchestrator creates approval request when reaching gated step
 * 3. Task transitions to 'approval_needed' status
 * 4. User reviews via API/Dashboard and approves or rejects
 * 5. On approval: execution resumes
 * 6. On rejection: task is terminated
 */

/**
 * RISK CLASSIFICATION
 * ===================
 */

export const RISK_LEVELS: Record<RiskLevel, { description: string; auto_approve: boolean }> = {
    low: {
        description: 'Read-only operations, data retrieval, analysis',
        auto_approve: true,
    },
    medium: {
        description: 'Code generation, content creation, API calls',
        auto_approve: false,
    },
    high: {
        description: 'Data modifications, deployments, external integrations, deletions, security changes',
        auto_approve: false,
    },
};

/**
 * APPROVAL TYPES
 * ===============
 * 
 * Different types of operations require different approval workflows
 */

export const APPROVAL_CATEGORIES = {
    // Code execution approvals
    code_execution: {
        description: 'Running generated code',
        required_role: 'developer',
        timeout_hours: 24,
    },

    // API call approvals
    external_api: {
        description: 'Calling external APIs',
        required_role: 'admin',
        timeout_hours: 4,
    },

    // Data modification approvals
    data_modification: {
        description: 'Modifying or deleting data',
        required_role: 'admin',
        timeout_hours: 2,
    },

    // Deployment approvals
    deployment: {
        description: 'Deploying changes to infrastructure',
        required_role: 'admin',
        timeout_hours: 8,
    },

    // Cost approvals
    budget_override: {
        description: 'Exceeding budget limits',
        required_role: 'admin',
        timeout_hours: 1,
    },
} as const;

/**
 * IMPLEMENTATION CHECKLIST
 * =========================
 * 
 * [ ] Risk assessment in planner
 * [ ] Approval request creation in orchestrator
 * [ ] Approval review API endpoints
 * [ ] Approval timeout handling
 * [ ] Role-based approval authorization
 * [ ] Approval notification system
 * [ ] Approval audit logging
 */

/**
 * Risk Assessment Rules
 * 
 * Applied by the planner when generating execution plans
 */
export function assessStepRisk(step: TodoItem, query: string): RiskLevel {
    const lowerQuery = query.toLowerCase();
    const lowerDesc = step.description.toLowerCase();

    // Critical/High risk indicators
    const criticalKeywords = ['delete', 'remove', 'drop', 'destroy', 'purge', 'production'];
    if (criticalKeywords.some(kw => lowerQuery.includes(kw) || lowerDesc.includes(kw))) {
        return 'high';
    }

    // High risk indicators
    const highKeywords = ['deploy', 'publish', 'create', 'update', 'modify', 'write'];
    if (highKeywords.some(kw => lowerQuery.includes(kw) || lowerDesc.includes(kw))) {
        return 'high';
    }

    // Medium risk for code generation
    if (step.agent === 'coder') {
        return 'medium';
    }

    // Default to low
    return 'low';
}

/**
 * Determine if a step requires approval based on risk level and configuration
 */
export function requiresApproval(
    step: TodoItem,
    _task: Task,
    userPreferences?: { auto_approve_low?: boolean; auto_approve_medium?: boolean }
): boolean {
    // Explicit approval flag takes precedence
    if (step.require_approval !== undefined) {
        return step.require_approval;
    }

    // Check risk level
    const risk = step.risk ?? 'low';
    const config = RISK_LEVELS[risk];

    // Never auto-approve high
    if (risk === 'high') {
        return true;
    }

    // Check user preferences for medium/low
    if (risk === 'medium' && userPreferences?.auto_approve_medium) {
        return false;
    }

    if (risk === 'low' && userPreferences?.auto_approve_low === false) {
        return true;
    }

    return config ? !config.auto_approve : true;
}

/**
 * Create approval preview data for display to users
 */
export interface ApprovalPreview {
    step_id: string;
    agent: string;
    description: string;
    risk_level: RiskLevel;
    approval_type: string;
    impact_summary: string;
    rollback_available: boolean;
}

export function createApprovalPreview(step: TodoItem, task: Task): ApprovalPreview {
    const risk = step.risk ?? 'low';
    const approvalType = step.approval_type ?? 'general';

    return {
        step_id: step.id,
        agent: step.agent,
        description: step.description,
        risk_level: risk,
        approval_type: approvalType,
        impact_summary: generateImpactSummary(step, task),
        rollback_available: isRollbackAvailable(step),
    };
}

function generateImpactSummary(step: TodoItem, _task: Task): string {
    const agent = step.agent;
    const risk = step.risk ?? 'low';

    switch (agent) {
        case 'coder':
            return `Will generate and potentially execute code. Risk: ${risk}`;
        case 'analyst':
            return 'Will analyze data and generate reports. Read-only operation.';
        case 'researcher':
            return 'Will search and compile information. Read-only operation.';
        default:
            return `Will perform ${agent} operations with risk level: ${risk}`;
    }
}

function isRollbackAvailable(step: TodoItem): boolean {
    // Code and design operations can typically be rolled back
    return ['coder', 'design'].includes(step.agent);
}

/**
 * INTEGRATION POINTS
 * ==================
 */

/**
 * Example: Adding approval gate to planner
 * 
 * In planner.ts:
 * ```typescript
 * import { assessStepRisk, requiresApproval } from './approval-workflow.js';
 * 
 * private generateSteps(query: string, agents: AgentType[]): TodoItem[] {
 *     const steps: TodoItem[] = [];
 *     
 *     // ... create steps ...
 *     
 *     // Add risk assessment
 *     for (const step of steps) {
 *         step.risk = assessStepRisk(step, query);
 *         step.require_approval = requiresApproval(step, task);
 *     }
 *     
 *     return steps;
 * }
 * ```
 */

/**
 * Example: Handling approval in orchestrator
 * 
 * In orchestrator.ts executeTask:
 * ```typescript
 * if (step.require_approval) {
 *     const approvalPreview = createApprovalPreview(step, task);
 *     await this.createApprovalRequest(task, step, runId, approvalPreview);
 *     await this.updateTaskStatus(taskId, 'approval_needed');
 *     return { status: 'approval_needed', ... };
 * }
 * ```
 */

/**
 * Example: API endpoint for approval
 * 
 * In server.ts:
 * ```typescript
 * app.post('/approvals/:id/respond', async (req, res) => {
 *     const { approved, notes } = req.body;
 *     const userId = req.user.id;
 *     
 *     // Check user has approval permission
 *     if (!canApprove(userId, approval)) {
 *         return res.status(403).json({ error: 'Insufficient permissions' });
 *     }
 *     
 *     await orchestrator.respondToApproval(approvalId, userId, approved, notes);
 *     res.json({ success: true });
 * });
 * ```
 */

/**
 * SECURITY CONSIDERATIONS
 * ========================
 * 
 * 1. Role-Based Access Control (RBAC)
 *    - Only authorized users can approve specific types of requests
 *    - Admin role required for critical operations
 *    
 * 2. Audit Logging
 *    - All approval decisions must be logged with user ID and timestamp
 *    - Include justification/notes in audit trail
 *    
 * 3. Timeout Handling
 *    - Approvals should expire after configured timeout
 *    - Expired approvals automatically reject the operation
 *    
 * 4. Rate Limiting
 *    - Prevent approval request flooding
 *    - Track approval patterns per user
 */

export const SECURITY_GUIDELINES = {
    // Minimum wait time before auto-rejection
    min_approval_timeout_minutes: 5,

    // Maximum pending approvals per user
    max_pending_approvals: 10,

    // Required approval fields
    required_approval_fields: ['user_id', 'timestamp', 'decision', 'notes'],

    // Approval authorization matrix
    authorization_matrix: {
        code_execution: ['developer', 'admin'],
        external_api: ['admin'],
        data_modification: ['admin'],
        deployment: ['admin'],
        budget_override: ['admin'],
    },
};

/**
 * TESTING GUIDELINES
 * ==================
 * 
 * Required test cases:
 * 1. Risk classification for different query types
 * 2. Approval gate triggers correctly
 * 3. Task pauses at approval step
 * 4. Approval approval resumes execution
 * 5. Rejection terminates task
 * 6. Timeout handling
 * 7. Authorization checks
 * 8. Audit logging
 */
