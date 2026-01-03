/**
 * Audit Logging Utilities
 */

import { randomUUID } from 'crypto';
import { getRedisJSON } from '../redis/client.js';
import { redactPIIFromObject } from './redaction.js';

export type AuditEventType = 'access' | 'approval_requested' | 'approval_responded';

export interface AuditActor {
    user_id: string;
    email?: string;
    roles: string[];
}

export interface AuditResource {
    type: string;
    id?: string;
    attributes?: Record<string, unknown>;
}

export interface AuditEvent {
    id: string;
    type: AuditEventType;
    action: string;
    actor: AuditActor;
    resource?: AuditResource;
    metadata?: Record<string, unknown>;
    status?: string;
    ip?: string;
    user_agent?: string;
    created_at: string;
}

export async function logAuditEvent(
    event: Omit<AuditEvent, 'id' | 'created_at'> & Partial<Pick<AuditEvent, 'id' | 'created_at'>>
): Promise<AuditEvent> {
    const redis = getRedisJSON();
    const id = event.id ?? `audit-${randomUUID()}`;

    const sanitizedEvent: AuditEvent = {
        id,
        created_at: event.created_at ?? new Date().toISOString(),
        type: event.type,
        action: event.action,
        actor: event.actor,
        ...(event.resource !== undefined && { resource: event.resource }),
        ...(event.metadata !== undefined && { metadata: redactPIIFromObject(event.metadata) }),
        ...(event.status !== undefined && { status: event.status }),
        ...(event.ip !== undefined && { ip: event.ip }),
        ...(event.user_agent !== undefined && { user_agent: event.user_agent }),
    };

    await redis.set(`audit:${id}`, '$', sanitizedEvent);

    return sanitizedEvent;
}
