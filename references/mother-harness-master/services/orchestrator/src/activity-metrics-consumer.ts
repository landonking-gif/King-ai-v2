/**
 * Activity Metrics Consumer
 * Consumes activity stream events and aggregates metrics for dashboards
 */

import { getRedisClient } from '@mother-harness/shared';
import { nanoid } from 'nanoid';
import { getActivityEventTaxonomy } from './event-taxonomy.js';
import { logger } from './logger.js';

interface ActivityMetricsConsumerOptions {
    streamKey?: string;
    groupName?: string;
    consumerName?: string;
    blockMs?: number;
    batchSize?: number;
}

interface ActivityStreamEntry {
    id: string;
    fields: Record<string, string>;
}

const DEFAULT_STREAM_KEY = 'stream:activity';
const DEFAULT_GROUP_NAME = 'activity-metrics';

function parseStreamEntry(entry: [string, string[]]): ActivityStreamEntry {
    const [id, rawFields] = entry;
    const fields: Record<string, string> = {};
    for (let i = 0; i < rawFields.length; i += 2) {
        fields[rawFields[i] ?? ''] = rawFields[i + 1] ?? '';
    }
    return { id, fields };
}

function toDateKey(timestamp?: string): string {
    if (!timestamp) return new Date().toISOString().split('T')[0] ?? '';
    return timestamp.split('T')[0] ?? timestamp;
}

async function ensureConsumerGroup(streamKey: string, groupName: string) {
    const redis = getRedisClient();
    try {
        await redis.xgroup('CREATE', streamKey, groupName, '$', 'MKSTREAM');
    } catch (error) {
        const message = (error as Error).message ?? '';
        if (!message.includes('BUSYGROUP')) {
            throw error;
        }
    }
}

export function startActivityMetricsConsumer(options: ActivityMetricsConsumerOptions = {}) {
    const redis = getRedisClient();
    const streamKey = options.streamKey ?? DEFAULT_STREAM_KEY;
    const groupName = options.groupName ?? DEFAULT_GROUP_NAME;
    const consumerName = options.consumerName ?? `metrics-${process.pid}-${nanoid(6)}`;
    const blockMs = options.blockMs ?? 5000;
    const batchSize = options.batchSize ?? 50;
    let running = true;

    const processEntry = async (entry: ActivityStreamEntry) => {
        const eventType = entry.fields['event_type'];
        const userId = entry.fields['user_id'];

        if (!eventType || !userId) {
            logger.warn('Skipping activity event with missing fields', {
                entry_id: entry.id,
                fields: entry.fields,
            });
            return;
        }

        const taxonomy = getActivityEventTaxonomy(eventType as Parameters<typeof getActivityEventTaxonomy>[0]);
        const dateKey = toDateKey(entry.fields['timestamp']);

        const activityKey = `metrics:activity:user:${userId}:daily:${dateKey}`;
        const errorsKey = `metrics:errors:user:${userId}:daily:${dateKey}`;
        const runKey = `metrics:runs:user:${userId}:daily:${dateKey}`;

        await redis.hincrby(activityKey, eventType, 1);
        await redis.expire(activityKey, 30 * 24 * 60 * 60);

        if (taxonomy.outcome === 'failure') {
            await redis.hincrby(errorsKey, eventType, 1);
            await redis.expire(errorsKey, 30 * 24 * 60 * 60);
        }

        if (taxonomy.category === 'run') {
            await redis.hincrby(runKey, eventType, 1);
            await redis.expire(runKey, 30 * 24 * 60 * 60);
        }
    };

    const loop = async () => {
        await ensureConsumerGroup(streamKey, groupName);
        logger.info(`Activity metrics consumer started (${consumerName})`, {
            stream_key: streamKey,
            group_name: groupName,
        });

        while (running) {
            const response = await redis.call(
                'XREADGROUP',
                'GROUP',
                groupName,
                consumerName,
                'BLOCK',
                String(blockMs),
                'COUNT',
                String(batchSize),
                'STREAMS',
                streamKey,
                '>'
            ) as [string, [string, string[]][]][];

            if (!response) continue;

            for (const [, entries] of response) {
                for (const entry of entries as [string, string[]][]) {
                    const parsed = parseStreamEntry(entry);
                    try {
                        await processEntry(parsed);
                        await redis.xack(streamKey, groupName, parsed.id);
                    } catch (error) {
                        logger.error('Failed to process activity stream entry', error as Error, {
                            entry_id: parsed.id,
                            event_type: parsed.fields['event_type'],
                        });
                    }
                }
            }
        }
    };

    loop().catch((error) => {
        logger.error('Activity metrics consumer crashed', error as Error);
    });

    return {
        stop: async () => {
            running = false;
        },
    };
}
