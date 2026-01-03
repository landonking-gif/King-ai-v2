/**
 * Activity Metrics Access
 * Reads aggregated metrics for dashboards
 */

import { getRedisClient } from '@mother-harness/shared';

interface DailyMetrics {
    date: string;
    activity: Record<string, number>;
    errors: Record<string, number>;
    runs: Record<string, number>;
}

export interface ActivityMetricsSnapshot {
    user_id: string;
    days: DailyMetrics[];
}

function toNumberHash(hash: Record<string, string>): Record<string, number> {
    const result: Record<string, number> = {};
    for (const [key, value] of Object.entries(hash)) {
        result[key] = Number.parseFloat(value) || 0;
    }
    return result;
}

function getDateList(days: number): string[] {
    const list: string[] = [];
    const now = new Date();
    for (let i = days - 1; i >= 0; i -= 1) {
        const date = new Date(now);
        date.setDate(now.getDate() - i);
        list.push(date.toISOString().split('T')[0] ?? '');
    }
    return list;
}

export async function getActivityMetrics(userId: string, days: number = 7): Promise<ActivityMetricsSnapshot> {
    const redis = getRedisClient();
    const dates = getDateList(days);
    const snapshot: DailyMetrics[] = [];

    for (const date of dates) {
        const activityKey = `metrics:activity:user:${userId}:daily:${date}`;
        const errorsKey = `metrics:errors:user:${userId}:daily:${date}`;
        const runKey = `metrics:runs:user:${userId}:daily:${date}`;

        const [activity, errors, runs] = await Promise.all([
            redis.hgetall(activityKey),
            redis.hgetall(errorsKey),
            redis.hgetall(runKey),
        ]);

        snapshot.push({
            date,
            activity: toNumberHash(activity),
            errors: toNumberHash(errors),
            runs: toNumberHash(runs),
        });
    }

    return {
        user_id: userId,
        days: snapshot,
    };
}
