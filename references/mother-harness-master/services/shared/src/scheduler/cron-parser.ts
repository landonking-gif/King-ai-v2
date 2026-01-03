/**
 * Cron Expression Parser
 * Parses and evaluates 5-field cron expressions
 * Format: minute hour day-of-month month day-of-week
 */

/** Parsed cron field */
interface CronField {
    type: 'all' | 'list' | 'range' | 'step';
    values: number[];
}

/** Parsed cron expression */
export interface ParsedCron {
    minute: CronField;
    hour: CronField;
    dayOfMonth: CronField;
    month: CronField;
    dayOfWeek: CronField;
    raw: string;
}

/** Field limits */
const FIELD_LIMITS = {
    minute: { min: 0, max: 59 },
    hour: { min: 0, max: 23 },
    dayOfMonth: { min: 1, max: 31 },
    month: { min: 1, max: 12 },
    dayOfWeek: { min: 0, max: 6 }, // 0 = Sunday
};

/** Named values */
const DAY_NAMES: Record<string, number> = {
    sun: 0, mon: 1, tue: 2, wed: 3, thu: 4, fri: 5, sat: 6,
};

const MONTH_NAMES: Record<string, number> = {
    jan: 1, feb: 2, mar: 3, apr: 4, may: 5, jun: 6,
    jul: 7, aug: 8, sep: 9, oct: 10, nov: 11, dec: 12,
};

/**
 * Parse a cron expression
 */
export function parseCron(expression: string): ParsedCron {
    const parts = expression.trim().split(/\s+/);

    if (parts.length !== 5) {
        throw new Error(`Invalid cron expression: expected 5 fields, got ${parts.length}`);
    }

    return {
        minute: parseField(parts[0]!, FIELD_LIMITS.minute),
        hour: parseField(parts[1]!, FIELD_LIMITS.hour),
        dayOfMonth: parseField(parts[2]!, FIELD_LIMITS.dayOfMonth),
        month: parseField(parts[3]!, FIELD_LIMITS.month, MONTH_NAMES),
        dayOfWeek: parseField(parts[4]!, FIELD_LIMITS.dayOfWeek, DAY_NAMES),
        raw: expression,
    };
}

/**
 * Parse a single cron field
 */
function parseField(
    field: string,
    limits: { min: number; max: number },
    names?: Record<string, number>
): CronField {
    // Handle * (all values)
    if (field === '*') {
        const values = [];
        for (let i = limits.min; i <= limits.max; i++) {
            values.push(i);
        }
        return { type: 'all', values };
    }

    // Handle */step (step values)
    if (field.startsWith('*/')) {
        const step = parseInt(field.substring(2), 10);
        if (isNaN(step) || step < 1) {
            throw new Error(`Invalid step value: ${field}`);
        }
        const values = [];
        for (let i = limits.min; i <= limits.max; i += step) {
            values.push(i);
        }
        return { type: 'step', values };
    }

    // Handle comma-separated list
    if (field.includes(',')) {
        const values = field.split(',').map(v => parseValue(v.trim(), limits, names));
        return { type: 'list', values: [...new Set(values)].sort((a, b) => a - b) };
    }

    // Handle range (e.g., 1-5)
    if (field.includes('-') && !field.startsWith('-')) {
        const [startStr, endStr] = field.split('-');
        const start = parseValue(startStr!, limits, names);
        const end = parseValue(endStr!, limits, names);

        if (start > end) {
            throw new Error(`Invalid range: ${field}`);
        }

        const values = [];
        for (let i = start; i <= end; i++) {
            values.push(i);
        }
        return { type: 'range', values };
    }

    // Single value
    const value = parseValue(field, limits, names);
    return { type: 'list', values: [value] };
}

/**
 * Parse a single value
 */
function parseValue(
    value: string,
    limits: { min: number; max: number },
    names?: Record<string, number>
): number {
    // Check for named value
    if (names) {
        const named = names[value.toLowerCase()];
        if (named !== undefined) return named;
    }

    const num = parseInt(value, 10);
    if (isNaN(num) || num < limits.min || num > limits.max) {
        throw new Error(`Invalid value: ${value} (expected ${limits.min}-${limits.max})`);
    }
    return num;
}

/**
 * Get the next run time for a cron expression
 */
export function getNextCronRun(cron: ParsedCron, after: Date = new Date()): Date {
    // Start from the next minute
    const next = new Date(after);
    next.setSeconds(0, 0);
    next.setMinutes(next.getMinutes() + 1);

    // Search for up to 2 years
    const maxIterations = 366 * 24 * 60 * 2;

    for (let i = 0; i < maxIterations; i++) {
        if (matchesCron(cron, next)) {
            return next;
        }
        next.setMinutes(next.getMinutes() + 1);
    }

    throw new Error('Could not find next run time within 2 years');
}

/**
 * Check if a date matches a cron expression
 */
export function matchesCron(cron: ParsedCron, date: Date): boolean {
    const minute = date.getMinutes();
    const hour = date.getHours();
    const dayOfMonth = date.getDate();
    const month = date.getMonth() + 1; // 1-indexed
    const dayOfWeek = date.getDay();

    return (
        cron.minute.values.includes(minute) &&
        cron.hour.values.includes(hour) &&
        cron.dayOfMonth.values.includes(dayOfMonth) &&
        cron.month.values.includes(month) &&
        cron.dayOfWeek.values.includes(dayOfWeek)
    );
}

/**
 * Common cron presets
 */
export const CRON_PRESETS: Record<string, string> = {
    'every_minute': '* * * * *',
    'every_5_minutes': '*/5 * * * *',
    'every_15_minutes': '*/15 * * * *',
    'every_30_minutes': '*/30 * * * *',
    'every_hour': '0 * * * *',
    'every_6_hours': '0 */6 * * *',
    'daily': '0 0 * * *',
    'weekly': '0 0 * * 0',
    'monthly': '0 0 1 * *',
    'weekdays': '0 9 * * 1-5',
    'weekends': '0 9 * * 0,6',
};

/**
 * Describe a cron expression in human readable form
 */
export function describeCron(expression: string): string {
    try {
        const cron = parseCron(expression);

        const parts: string[] = [];

        // Minute
        if (cron.minute.type === 'all') {
            parts.push('every minute');
        } else if (cron.minute.type === 'step') {
            parts.push(`every ${60 / cron.minute.values.length} minutes`);
        } else {
            parts.push(`at minute ${cron.minute.values.join(', ')}`);
        }

        // Hour
        if (cron.hour.type !== 'all') {
            if (cron.hour.values.length === 1) {
                parts.push(`at ${cron.hour.values[0]}:00`);
            } else {
                parts.push(`during hours ${cron.hour.values.join(', ')}`);
            }
        }

        // Day of week
        if (cron.dayOfWeek.type !== 'all') {
            const days = cron.dayOfWeek.values.map(d =>
                ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][d]
            );
            parts.push(`on ${days.join(', ')}`);
        }

        // Day of month
        if (cron.dayOfMonth.type !== 'all') {
            parts.push(`on day ${cron.dayOfMonth.values.join(', ')}`);
        }

        // Month
        if (cron.month.type !== 'all') {
            const months = cron.month.values.map(m =>
                ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m]
            );
            parts.push(`in ${months.join(', ')}`);
        }

        return parts.join(' ');
    } catch {
        return 'Invalid cron expression';
    }
}
