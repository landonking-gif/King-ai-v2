/**
 * Event Taxonomy
 * Shared taxonomy for activity events and structured logging
 */

export type ActivityEventType =
    | 'run_created'
    | 'run_started'
    | 'run_waiting_approval'
    | 'run_terminated'
    | 'step_started'
    | 'step_completed'
    | 'step_failed'
    | 'approval_requested'
    | 'approval_approved'
    | 'approval_rejected';

export type EventCategory = 'run' | 'step' | 'approval';
export type EventOutcome = 'success' | 'failure' | 'neutral';
export type EventSeverity = 'low' | 'medium' | 'high';

export interface ActivityEventTaxonomy {
    name: ActivityEventType;
    category: EventCategory;
    action: string;
    outcome: EventOutcome;
    severity: EventSeverity;
}

const ACTIVITY_EVENT_TAXONOMY: Record<ActivityEventType, ActivityEventTaxonomy> = {
    run_created: {
        name: 'run_created',
        category: 'run',
        action: 'create',
        outcome: 'success',
        severity: 'low',
    },
    run_started: {
        name: 'run_started',
        category: 'run',
        action: 'start',
        outcome: 'neutral',
        severity: 'low',
    },
    run_waiting_approval: {
        name: 'run_waiting_approval',
        category: 'run',
        action: 'await_approval',
        outcome: 'neutral',
        severity: 'low',
    },
    run_terminated: {
        name: 'run_terminated',
        category: 'run',
        action: 'terminate',
        outcome: 'failure',
        severity: 'high',
    },
    step_started: {
        name: 'step_started',
        category: 'step',
        action: 'start',
        outcome: 'neutral',
        severity: 'low',
    },
    step_completed: {
        name: 'step_completed',
        category: 'step',
        action: 'complete',
        outcome: 'success',
        severity: 'low',
    },
    step_failed: {
        name: 'step_failed',
        category: 'step',
        action: 'fail',
        outcome: 'failure',
        severity: 'high',
    },
    approval_requested: {
        name: 'approval_requested',
        category: 'approval',
        action: 'request',
        outcome: 'neutral',
        severity: 'medium',
    },
    approval_approved: {
        name: 'approval_approved',
        category: 'approval',
        action: 'approve',
        outcome: 'success',
        severity: 'low',
    },
    approval_rejected: {
        name: 'approval_rejected',
        category: 'approval',
        action: 'reject',
        outcome: 'failure',
        severity: 'medium',
    },
};

export function getActivityEventTaxonomy(type: ActivityEventType): ActivityEventTaxonomy {
    return ACTIVITY_EVENT_TAXONOMY[type];
}
