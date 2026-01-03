import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const redisStore = new Map<string, unknown>();
const zsetStore = new Map<string, Map<string, number>>();

const mockRedisJSON = {
    get: vi.fn(<T>(key: string, path: string = '$'): T | null => {
        const value = redisStore.get(key) as T | undefined;
        if (!value) return null;
        if (path === '$') return value;
        const pathParts = path.replace('$.', '').split('.');
        return pathParts.reduce<unknown>((acc, part) => {
            if (acc && typeof acc === 'object' && part in (acc as Record<string, unknown>)) {
                return (acc as Record<string, unknown>)[part];
            }
            return undefined;
        }, value) as T ?? null;
    }),
    set: vi.fn((key: string, path: string, value: unknown) => {
        if (path === '$') {
            redisStore.set(key, value);
            return 'OK';
        }
        const current = (redisStore.get(key) as Record<string, unknown>) ?? {};
        const pathParts = path.replace('$.', '').split('.');
        let target = current;
        for (let i = 0; i < pathParts.length - 1; i++) {
            const part = pathParts[i]!;
            if (!target[part] || typeof target[part] !== 'object') {
                target[part] = {};
            }
            target = target[part] as Record<string, unknown>;
        }
        target[pathParts[pathParts.length - 1]!] = value;
        redisStore.set(key, current);
        return 'OK';
    }),
    del: vi.fn((key: string) => {
        const existed = redisStore.has(key);
        redisStore.delete(key);
        return existed ? 1 : 0;
    }),
    keys: vi.fn((pattern: string) => {
        const regex = new RegExp(`^${pattern.replace('*', '.*')}$`);
        return Array.from(redisStore.keys()).filter(key => regex.test(key));
    }),
};

const mockRedisClient = {
    zadd: vi.fn((key: string, score: number, member: string) => {
        const set = zsetStore.get(key) ?? new Map<string, number>();
        set.set(member, score);
        zsetStore.set(key, set);
        return 1;
    }),
    zrem: vi.fn((key: string, member: string) => {
        const set = zsetStore.get(key);
        if (!set) return 0;
        const existed = set.delete(member);
        return existed ? 1 : 0;
    }),
    zrangebyscore: vi.fn((key: string, min: string | number, max: string | number, ...args: Array<string | number>) => {
        const set = zsetStore.get(key) ?? new Map<string, number>();
        const minScore = min === '-inf' ? Number.NEGATIVE_INFINITY : Number(min);
        const maxScore = max === '+inf' ? Number.POSITIVE_INFINITY : Number(max);

        let offset = 0;
        let count = set.size;
        if (args[0] === 'LIMIT') {
            offset = Number(args[1] ?? 0);
            count = Number(args[2] ?? count);
        }

        return Array.from(set.entries())
            .filter(([, score]) => score >= minScore && score <= maxScore)
            .sort((a, b) => a[1] - b[1])
            .slice(offset, offset + count)
            .map(([member]) => member);
    }),
};

vi.mock('@mother-harness/shared', async () => {
    const actual = await vi.importActual<typeof import('@mother-harness/shared')>('@mother-harness/shared');
    return {
        ...actual,
        getRedisJSON: () => mockRedisJSON,
        getRedisClient: () => mockRedisClient,
    };
});

import { Scheduler } from './scheduler.js';

describe('Scheduler state transitions', () => {
    beforeEach(() => {
        redisStore.clear();
        zsetStore.clear();
        vi.clearAllMocks();
        vi.useFakeTimers();
        vi.setSystemTime(new Date('2024-01-01T00:00:00Z'));
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('schedules interval tasks and advances on execution', async () => {
        const scheduler = new Scheduler();

        const task = await scheduler.createScheduledTask('user-1', {
            name: 'Interval Task',
            schedule_type: 'interval',
            interval_ms: 60000,
            query: 'Do something',
        });

        expect(task.next_run_at).toBe('2024-01-01T00:01:00.000Z');

        await scheduler.recordExecution(task.id, { success: true, created_task_id: 'task-1' });

        const updated = await scheduler.getScheduledTask(task.id);
        expect(updated?.run_count).toBe(1);
        expect(updated?.last_run_status).toBe('success');
        expect(updated?.next_run_at).toBe('2024-01-01T00:01:00.000Z');
    });

    it('disables tasks and removes them from the queue', async () => {
        const scheduler = new Scheduler();

        const task = await scheduler.createScheduledTask('user-1', {
            name: 'Once Task',
            schedule_type: 'once',
            run_at: '2024-01-01T00:00:00.000Z',
            query: 'Run once',
        });

        const dueBefore = await scheduler.getDueTasks();
        expect(dueBefore.map(item => item.id)).toContain(task.id);

        await scheduler.setEnabled(task.id, false);

        const dueAfter = await scheduler.getDueTasks();
        expect(dueAfter.map(item => item.id)).not.toContain(task.id);
    });

    it('respects max runs and stops scheduling', async () => {
        const scheduler = new Scheduler();

        const task = await scheduler.createScheduledTask('user-1', {
            name: 'Limited Task',
            schedule_type: 'interval',
            interval_ms: 60000,
            query: 'Run twice',
            max_runs: 1,
        });

        await scheduler.recordExecution(task.id, { success: true });

        const updated = await scheduler.getScheduledTask(task.id);
        expect(updated?.enabled).toBe(false);
    });
});
