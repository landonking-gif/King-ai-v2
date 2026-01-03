/**
 * Metrics Module
 * Prometheus-compatible metrics for observability
 */

// import { getRedisClient } from '@mother-harness/shared';

/** Metric types */
export type MetricType = 'counter' | 'gauge' | 'histogram';

/** Metric definition */
export interface MetricDefinition {
    name: string;
    type: MetricType;
    help: string;
    labels?: string[];
}

/** Collected metrics */
const metrics: Map<string, { definition: MetricDefinition; values: Map<string, number> }> = new Map();

/** Histogram buckets for latency */
const LATENCY_BUCKETS = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10];

/**
 * Define application metrics
 */
export const METRICS = {
    // Request metrics
    http_requests_total: {
        name: 'http_requests_total',
        type: 'counter' as MetricType,
        help: 'Total HTTP requests',
        labels: ['method', 'path', 'status'],
    },
    http_request_duration_seconds: {
        name: 'http_request_duration_seconds',
        type: 'histogram' as MetricType,
        help: 'HTTP request duration in seconds',
        labels: ['method', 'path'],
    },

    // Task metrics
    tasks_created_total: {
        name: 'tasks_created_total',
        type: 'counter' as MetricType,
        help: 'Total tasks created',
        labels: ['project'],
    },
    tasks_completed_total: {
        name: 'tasks_completed_total',
        type: 'counter' as MetricType,
        help: 'Total tasks completed',
        labels: ['project', 'status'],
    },
    tasks_in_progress: {
        name: 'tasks_in_progress',
        type: 'gauge' as MetricType,
        help: 'Tasks currently in progress',
        labels: ['project'],
    },
    task_duration_seconds: {
        name: 'task_duration_seconds',
        type: 'histogram' as MetricType,
        help: 'Task duration in seconds',
        labels: ['project'],
    },

    // Agent metrics
    agent_invocations_total: {
        name: 'agent_invocations_total',
        type: 'counter' as MetricType,
        help: 'Total agent invocations',
        labels: ['agent', 'model'],
    },
    agent_duration_seconds: {
        name: 'agent_duration_seconds',
        type: 'histogram' as MetricType,
        help: 'Agent execution duration in seconds',
        labels: ['agent'],
    },
    agent_tokens_used: {
        name: 'agent_tokens_used',
        type: 'counter' as MetricType,
        help: 'Total tokens used by agents',
        labels: ['agent', 'model'],
    },

    // Memory metrics
    memory_usage_bytes: {
        name: 'memory_usage_bytes',
        type: 'gauge' as MetricType,
        help: 'Memory usage in bytes',
        labels: ['type'],
    },

    // Redis metrics
    redis_operations_total: {
        name: 'redis_operations_total',
        type: 'counter' as MetricType,
        help: 'Total Redis operations',
        labels: ['operation'],
    },
    redis_latency_seconds: {
        name: 'redis_latency_seconds',
        type: 'histogram' as MetricType,
        help: 'Redis operation latency in seconds',
        labels: ['operation'],
    },

    // Budget metrics
    budget_usage_percent: {
        name: 'budget_usage_percent',
        type: 'gauge' as MetricType,
        help: 'Budget usage percentage',
        labels: ['scope', 'resource'],
    },
    budget_limit_exceeded_total: {
        name: 'budget_limit_exceeded_total',
        type: 'counter' as MetricType,
        help: 'Total budget limit exceeded events',
        labels: ['scope', 'resource'],
    },
};

/**
 * Initialize metrics
 */
export function initMetrics(): void {
    for (const def of Object.values(METRICS)) {
        metrics.set(def.name, { definition: def, values: new Map() });
    }
}

/**
 * Increment a counter
 */
export function incCounter(name: string, labels: Record<string, string> = {}, value: number = 1): void {
    const metric = metrics.get(name);
    if (!metric || metric.definition.type !== 'counter') return;

    const key = serializeLabels(labels);
    const current = metric.values.get(key) ?? 0;
    metric.values.set(key, current + value);
}

/**
 * Set a gauge value
 */
export function setGauge(name: string, labels: Record<string, string> = {}, value: number): void {
    const metric = metrics.get(name);
    if (!metric || metric.definition.type !== 'gauge') return;

    const key = serializeLabels(labels);
    metric.values.set(key, value);
}

/**
 * Observe a histogram value
 */
export function observeHistogram(name: string, labels: Record<string, string> = {}, value: number): void {
    const metric = metrics.get(name);
    if (!metric || metric.definition.type !== 'histogram') return;

    // Increment bucket counters
    for (const bucket of LATENCY_BUCKETS) {
        if (value <= bucket) {
            const bucketKey = serializeLabels({ ...labels, le: String(bucket) });
            const current = metric.values.get(bucketKey) ?? 0;
            metric.values.set(bucketKey, current + 1);
        }
    }

    // Increment +Inf bucket
    const infKey = serializeLabels({ ...labels, le: '+Inf' });
    const infCurrent = metric.values.get(infKey) ?? 0;
    metric.values.set(infKey, infCurrent + 1);

    // Track sum and count
    const sumKey = serializeLabels({ ...labels, type: 'sum' });
    const sumCurrent = metric.values.get(sumKey) ?? 0;
    metric.values.set(sumKey, sumCurrent + value);

    const countKey = serializeLabels({ ...labels, type: 'count' });
    const countCurrent = metric.values.get(countKey) ?? 0;
    metric.values.set(countKey, countCurrent + 1);
}

/**
 * Serialize labels to a key
 */
function serializeLabels(labels: Record<string, string>): string {
    return Object.entries(labels)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([k, v]) => `${k}="${v}"`)
        .join(',');
}

/**
 * Export metrics in Prometheus format
 */
export function exportMetrics(): string {
    const lines: string[] = [];

    for (const [name, metric] of metrics) {
        lines.push(`# HELP ${name} ${metric.definition.help}`);
        lines.push(`# TYPE ${name} ${metric.definition.type}`);

        for (const [labelKey, value] of metric.values) {
            if (labelKey) {
                lines.push(`${name}{${labelKey}} ${value}`);
            } else {
                lines.push(`${name} ${value}`);
            }
        }

        lines.push('');
    }

    return lines.join('\n');
}

/**
 * Reset all metrics (for testing)
 */
export function resetMetrics(): void {
    for (const metric of metrics.values()) {
        metric.values.clear();
    }
}
