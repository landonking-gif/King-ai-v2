/**
 * Structured Logging
 * JSON-formatted logs for observability
 */

import type { EventCategory, EventOutcome, EventSeverity } from './event-taxonomy.js';

/** Log levels */
export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

/** Log entry structure */
interface LogEntry {
    timestamp: string;
    level: LogLevel;
    message: string;
    service: string;

    // Event taxonomy
    event?: {
        name: string;
        category?: EventCategory;
        action?: string;
        outcome?: EventOutcome;
        severity?: EventSeverity;
    };

    // Optional context
    task_id?: string;
    project_id?: string;
    user_id?: string;
    agent?: string;

    // Error details
    error?: {
        message: string;
        stack?: string;
        code?: string;
    };

    // Performance
    duration_ms?: number;

    // Additional data
    data?: Record<string, unknown>;
}

/** Logger configuration */
const config = {
    service: process.env['SERVICE_NAME'] ?? 'mother-harness',
    level: (process.env['LOG_LEVEL'] ?? 'info') as LogLevel,
    pretty: process.env['NODE_ENV'] !== 'production',
};

/** Level priority */
const LEVEL_PRIORITY: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
};

/**
 * Create a logger instance
 */
export function createLogger(context: {
    task_id?: string;
    project_id?: string;
    user_id?: string;
    agent?: string;
} = {}) {
    return {
        debug: (message: string, data?: Record<string, unknown>) => {
            const entry: Partial<LogEntry> = { ...context };
            if (data) entry.data = data;
            log('debug', message, entry);
        },

        info: (message: string, data?: Record<string, unknown>) => {
            const entry: Partial<LogEntry> = { ...context };
            if (data) entry.data = data;
            log('info', message, entry);
        },

        warn: (message: string, data?: Record<string, unknown>) => {
            const entry: Partial<LogEntry> = { ...context };
            if (data) entry.data = data;
            log('warn', message, entry);
        },

        error: (message: string, error?: Error, data?: Record<string, unknown>) => {
            const entry: Partial<LogEntry> = {
                ...context,
            };
            if (error) {
                entry.error = {
                    message: error.message,
                    ...(error.stack !== undefined && { stack: error.stack }),
                    ...((error as any).code !== undefined && { code: (error as any).code }),
                };
            }
            if (data) entry.data = data;
            log('error', message, entry);
        },

        event: (
            level: LogLevel,
            event: {
                name: string;
                category?: EventCategory;
                action?: string;
                outcome?: EventOutcome;
                severity?: EventSeverity;
            },
            message: string,
            data?: Record<string, unknown>
        ) => {
            const entry: Partial<LogEntry> = {
                ...context,
                event,
            };
            if (data) entry.data = data;
            log(level, message, entry);
        },

        child: (additionalContext: Record<string, string>) =>
            createLogger({ ...context, ...additionalContext }),
    };
}

/**
 * Log a message
 */
function log(
    level: LogLevel,
    message: string,
    context: Partial<LogEntry> = {}
): void {
    // Check log level
    if (LEVEL_PRIORITY[level] < LEVEL_PRIORITY[config.level]) {
        return;
    }

    const entry: LogEntry = {
        timestamp: new Date().toISOString(),
        level,
        message,
        service: config.service,
        ...context,
    };

    const output = config.pretty ? formatPretty(entry) : JSON.stringify(entry);

    if (level === 'error') {
        console.error(output);
    } else if (level === 'warn') {
        console.warn(output);
    } else {
        console.log(output);
    }
}

/**
 * Format log entry for human readability
 */
function formatPretty(entry: LogEntry): string {
    const levelColors: Record<LogLevel, string> = {
        debug: '\x1b[90m', // gray
        info: '\x1b[36m',  // cyan
        warn: '\x1b[33m',  // yellow
        error: '\x1b[31m', // red
    };

    const reset = '\x1b[0m';
    const color = levelColors[entry.level];
    const time = entry.timestamp.split('T')[1]?.slice(0, 12) ?? entry.timestamp;

    let line = `${color}[${entry.level.toUpperCase().padEnd(5)}]${reset} ${time} ${entry.message}`;

    if (entry.task_id) {
        line += ` ${'\x1b[90m'}task=${entry.task_id}${reset}`;
    }

    if (entry.agent) {
        line += ` ${'\x1b[90m'}agent=${entry.agent}${reset}`;
    }

    if (entry.duration_ms !== undefined) {
        line += ` ${'\x1b[90m'}(${entry.duration_ms}ms)${reset}`;
    }

    if (entry.error) {
        line += `\n  ${color}Error: ${entry.error.message}${reset}`;
        if (entry.error.stack && entry.level === 'error') {
            line += `\n  ${'\x1b[90m'}${entry.error.stack.split('\n').slice(1, 4).join('\n  ')}${reset}`;
        }
    }

    if (entry.event) {
        line += `\n  ${'\x1b[90m'}event=${entry.event.name}`;
        if (entry.event.category) line += ` category=${entry.event.category}`;
        if (entry.event.action) line += ` action=${entry.event.action}`;
        if (entry.event.outcome) line += ` outcome=${entry.event.outcome}`;
        if (entry.event.severity) line += ` severity=${entry.event.severity}`;
        line += `${reset}`;
    }

    if (entry.data && Object.keys(entry.data).length > 0) {
        line += `\n  ${'\x1b[90m'}${JSON.stringify(entry.data)}${reset}`;
    }

    return line;
}

/** Default logger */
export const logger = createLogger();
