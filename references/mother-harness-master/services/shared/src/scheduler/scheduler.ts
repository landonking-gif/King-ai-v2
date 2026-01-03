/**
 * Scheduler
 * Scheduled and recurring task execution
 */

import { getRedisClient, getRedisJSON } from '../redis/index.js';
import { nanoid } from 'nanoid';

/** Schedule types */
export type ScheduleType = 'once' | 'cron' | 'interval';

/** Scheduled task */
export interface ScheduledTask {
    id: string;
    user_id: string;
    name: string;
    description?: string;

    // Schedule configuration
    schedule_type: ScheduleType;
    cron_expression?: string;    // For cron type
    interval_ms?: number;        // For interval type
    run_at?: string;             // For once type

    // Task payload
    query: string;               // The query to execute
    project_id?: string;
    template_id?: string;        // Template to use
    template_vars?: Record<string, string | number | boolean>;

    // Execution state
    last_run_at?: string;
    next_run_at: string;
    last_run_status?: 'success' | 'failure';
    last_run_task_id?: string;
    run_count: number;

    // Control
    enabled: boolean;
    max_runs?: number;           // Stop after N runs

    // Metadata
    created_at: string;
    updated_at: string;
}

/** Upcoming run */
export interface UpcomingRun {
    task: ScheduledTask;
    scheduled_at: string;
}

export class Scheduler {
    private redis = getRedisJSON();
    private redisClient = getRedisClient();
    private readonly prefix = 'scheduled:';
    private readonly queueKey = 'queue:scheduled';

    /**
     * Create a scheduled task
     */
    async createScheduledTask(
        userId: string,
        data: {
            name: string;
            description?: string;
            schedule_type: ScheduleType;
            cron_expression?: string;
            interval_ms?: number;
            run_at?: string;
            query: string;
            project_id?: string;
            template_id?: string;
            template_vars?: Record<string, string | number | boolean>;
            max_runs?: number;
        }
    ): Promise<ScheduledTask> {
        const now = new Date().toISOString();
        const nextRun = await this.calculateNextRun(data);

        const task: ScheduledTask = {
            id: `sched-${nanoid()}`,
            user_id: userId,
            name: data.name,
            ...(data.description !== undefined && { description: data.description }),
            schedule_type: data.schedule_type,
            ...(data.cron_expression !== undefined && { cron_expression: data.cron_expression }),
            ...(data.interval_ms !== undefined && { interval_ms: data.interval_ms }),
            ...(data.run_at !== undefined && { run_at: data.run_at }),
            query: data.query,
            ...(data.project_id !== undefined && { project_id: data.project_id }),
            ...(data.template_id !== undefined && { template_id: data.template_id }),
            ...(data.template_vars !== undefined && { template_vars: data.template_vars }),
            next_run_at: nextRun,
            run_count: 0,
            enabled: true,
            ...(data.max_runs !== undefined && { max_runs: data.max_runs }),
            created_at: now,
            updated_at: now,
        };

        await this.redis.set(`${this.prefix}${task.id}`, '$', task);

        // Add to sorted set for scheduling
        await this.redisClient.zadd(this.queueKey, new Date(nextRun).getTime(), task.id);

        return task;
    }

    /**
     * Get a scheduled task
     */
    async getScheduledTask(taskId: string): Promise<ScheduledTask | null> {
        return await this.redis.get<ScheduledTask>(`${this.prefix}${taskId}`);
    }

    /**
     * Enable/disable a scheduled task
     */
    async setEnabled(taskId: string, enabled: boolean): Promise<ScheduledTask | null> {
        const task = await this.getScheduledTask(taskId);
        if (!task) return null;

        task.enabled = enabled;
        task.updated_at = new Date().toISOString();

        if (enabled) {
            // Recalculate next run
            task.next_run_at = await this.calculateNextRun(task);
            await this.redisClient.zadd(this.queueKey, new Date(task.next_run_at).getTime(), taskId);
        } else {
            // Remove from queue
            await this.redisClient.zrem(this.queueKey, taskId);
        }

        await this.redis.set(`${this.prefix}${taskId}`, '$', task);

        return task;
    }

    /**
     * Record task execution
     */
    async recordExecution(
        taskId: string,
        result: { success: boolean; created_task_id?: string }
    ): Promise<void> {
        const task = await this.getScheduledTask(taskId);
        if (!task) return;

        task.last_run_at = new Date().toISOString();
        task.last_run_status = result.success ? 'success' : 'failure';
        if (result.created_task_id !== undefined) {
            task.last_run_task_id = result.created_task_id;
        }
        task.run_count++;

        // Check max runs
        if (task.max_runs && task.run_count >= task.max_runs) {
            task.enabled = false;
            await this.redisClient.zrem(this.queueKey, taskId);
        } else if (task.schedule_type !== 'once') {
            // Schedule next run
            task.next_run_at = await this.calculateNextRun(task);
            await this.redisClient.zadd(this.queueKey, new Date(task.next_run_at).getTime(), taskId);
        } else {
            // One-time task, disable
            task.enabled = false;
            await this.redisClient.zrem(this.queueKey, taskId);
        }

        task.updated_at = new Date().toISOString();
        await this.redis.set(`${this.prefix}${taskId}`, '$', task);
    }

    /**
     * Get tasks due for execution
     */
    async getDueTasks(): Promise<ScheduledTask[]> {
        const now = Date.now();

        // Get tasks with score <= now
        const taskIds = await this.redisClient.zrangebyscore(this.queueKey, '-inf', now);
        const tasks: ScheduledTask[] = [];

        for (const id of taskIds) {
            const task = await this.getScheduledTask(id);
            if (task && task.enabled) {
                tasks.push(task);
            }
        }

        return tasks;
    }

    /**
     * Get upcoming runs
     */
    async getUpcomingRuns(userId: string, limit: number = 10): Promise<UpcomingRun[]> {
        const taskIds = await this.redisClient.zrangebyscore(this.queueKey, Date.now(), '+inf', 'LIMIT', 0, limit * 2);
        const runs: UpcomingRun[] = [];

        for (const id of taskIds) {
            const task = await this.getScheduledTask(id);
            if (task && task.user_id === userId && task.enabled) {
                runs.push({
                    task,
                    scheduled_at: task.next_run_at,
                });
            }

            if (runs.length >= limit) break;
        }

        return runs;
    }

    /**
     * Calculate next run time
     */
    private async calculateNextRun(config: {
        schedule_type: ScheduleType;
        cron_expression?: string;
        interval_ms?: number;
        run_at?: string;
        last_run_at?: string;
    }): Promise<string> {
        const now = new Date();

        switch (config.schedule_type) {
            case 'once':
                return config.run_at ?? now.toISOString();

            case 'interval':
                if (config.interval_ms) {
                    const base = config.last_run_at ? new Date(config.last_run_at) : now;
                    return new Date(base.getTime() + config.interval_ms).toISOString();
                }
                return now.toISOString();

            case 'cron':
                if (config.cron_expression) {
                    try {
                        // Dynamic import to avoid circular dependencies
                        const { parseCron, getNextCronRun } = await import('./cron-parser.js');
                        const parsed = parseCron(config.cron_expression);
                        return getNextCronRun(parsed, now).toISOString();
                    } catch (error) {
                        console.error('[Scheduler] Failed to parse cron:', error);
                        // Fallback to 1 hour from now
                        return new Date(now.getTime() + 60 * 60 * 1000).toISOString();
                    }
                }
                return new Date(now.getTime() + 60 * 60 * 1000).toISOString();

            default:
                return now.toISOString();
        }
    }

    /**
     * Get user's scheduled tasks
     */
    async getUserTasks(userId: string): Promise<ScheduledTask[]> {
        const keys = await this.redis.keys(`${this.prefix}*`);
        const tasks: ScheduledTask[] = [];

        for (const key of keys) {
            const task = await this.redis.get<ScheduledTask>(key);
            if (task && task.user_id === userId) {
                tasks.push(task);
            }
        }

        return tasks.sort((a, b) =>
            new Date(a.next_run_at).getTime() - new Date(b.next_run_at).getTime()
        );
    }

    /**
     * Delete a scheduled task
     */
    async deleteTask(taskId: string): Promise<void> {
        await this.redis.del(`${this.prefix}${taskId}`);
        await this.redisClient.zrem(this.queueKey, taskId);
    }
}

// Singleton
let schedulerInstance: Scheduler | null = null;

export function getScheduler(): Scheduler {
    if (!schedulerInstance) {
        schedulerInstance = new Scheduler();
    }
    return schedulerInstance;
}
