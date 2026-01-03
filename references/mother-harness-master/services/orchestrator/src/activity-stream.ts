/**
 * Activity Stream Logger
 * Writes run lifecycle events to Redis Streams
 */

import { getRedisClient } from '@mother-harness/shared';
import { getActivityEventTaxonomy, type ActivityEventType } from './event-taxonomy.js';
import { logger } from './logger.js';

export interface ActivityEvent {
    type: ActivityEventType;
    run_id: string;
    task_id: string;
    project_id: string;
    user_id: string;
    timestamp?: string;
    details?: Record<string, unknown>;
}

const STREAM_KEY = 'stream:activity';
const RUN_STREAM_PREFIX = 'stream:activity:run:';

export interface ActivityStreamEntry {
    id: string;
    fields: Record<string, string>;
    details: Record<string, unknown>;
}

function parseStreamEntry(entry: [string, string[]]): ActivityStreamEntry {
    const [id, rawFields] = entry;
    const fields: Record<string, string> = {};
    for (let i = 0; i < rawFields.length; i += 2) {
        fields[rawFields[i] ?? ''] = rawFields[i + 1] ?? '';
    }

    let details: Record<string, unknown> = {};
    const rawDetails = fields['details'];
    if (rawDetails) {
        try {
            const parsed = JSON.parse(rawDetails);
            if (parsed && typeof parsed === 'object') {
                details = parsed as Record<string, unknown>;
            }
        } catch {
            // Ignore invalid details payloads; callers still get fields.
        }
    }

    return { id, fields, details };
}

function getRunStreamKey(runId: string): string {
    return `${RUN_STREAM_PREFIX}${runId}`;
}

export async function listRunActivityEvents(
    runId: string,
    options: { limit?: number; direction?: 'forward' | 'backward' } = {}
): Promise<ActivityStreamEntry[]> {
    const redis = getRedisClient();
    const limit = options.limit ?? 500;
    const direction = options.direction ?? 'forward';
    const streamKey = getRunStreamKey(runId);

    const entries = direction === 'backward'
        ? await redis.xrevrange(streamKey, '+', '-', 'COUNT', limit)
        : await redis.xrange(streamKey, '-', '+', 'COUNT', limit);

    const parsed = (entries as [string, string[]][]).map(parseStreamEntry);
    // Ensure chronological ordering for consumers.
    return direction === 'backward' ? parsed.reverse() : parsed;
}

export async function logActivity(event: ActivityEvent): Promise<void> {
    const redis = getRedisClient();
    const timestamp = event.timestamp ?? new Date().toISOString();
    const payload = JSON.stringify(event.details ?? {});
    const taxonomy = getActivityEventTaxonomy(event.type);

    const runStreamKey = getRunStreamKey(event.run_id);

    await redis.xadd(
        STREAM_KEY,
        '*',
        'event_type',
        event.type,
        'event_category',
        taxonomy.category,
        'event_action',
        taxonomy.action,
        'event_outcome',
        taxonomy.outcome,
        'event_severity',
        taxonomy.severity,
        'run_id',
        event.run_id,
        'task_id',
        event.task_id,
        'project_id',
        event.project_id,
        'user_id',
        event.user_id,
        'timestamp',
        timestamp,
        'details',
        payload
    );

    // Also write to a per-run stream for efficient replay/timeline queries.
    await redis.xadd(
        runStreamKey,
        '*',
        'event_type',
        event.type,
        'event_category',
        taxonomy.category,
        'event_action',
        taxonomy.action,
        'event_outcome',
        taxonomy.outcome,
        'event_severity',
        taxonomy.severity,
        'run_id',
        event.run_id,
        'task_id',
        event.task_id,
        'project_id',
        event.project_id,
        'user_id',
        event.user_id,
        'timestamp',
        timestamp,
        'details',
        payload
    );

    logger.event('info', {
        name: event.type,
        category: taxonomy.category,
        action: taxonomy.action,
        outcome: taxonomy.outcome,
        severity: taxonomy.severity,
    }, 'Activity event logged', {
        run_id: event.run_id,
        task_id: event.task_id,
        project_id: event.project_id,
        user_id: event.user_id,
    });
}
