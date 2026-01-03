/**
 * PII Redaction Utilities
 */

const PII_PATTERNS: Array<{ pattern: RegExp; replacement: string }> = [
    // Email addresses
    {
        pattern: /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi,
        replacement: '[REDACTED_EMAIL]',
    },
    // SSN - must be checked before phone to avoid conflicts
    {
        pattern: /\b\d{3}-\d{2}-\d{4}\b/g,
        replacement: '[REDACTED_SSN]',
    },
    // Credit card numbers - 13-16 digits with optional spaces/dashes
    {
        pattern: /\b(?:\d[ -]*?){13,16}\b/g,
        replacement: '[REDACTED_CARD]',
    },
    // Phone numbers - various formats including international
    // Matches: (555) 123-4567, 555-123-4567, +1 555 123 4567, +44 20 7946 0958, etc.
    {
        pattern: /(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}(?:[\s.-]?\d{1,4})?/g,
        replacement: '[REDACTED_PHONE]',
    },
];

/**
 * Redact PII from a string
 */
export function redactPII(input: string): string {
    return PII_PATTERNS.reduce(
        (value, { pattern, replacement }) => value.replace(pattern, replacement),
        input
    );
}

/**
 * Redact PII from an unknown structure
 */
export function redactPIIFromObject<T>(value: T): T {
    if (typeof value === 'string') {
        return redactPII(value) as T;
    }

    if (Array.isArray(value)) {
        return value.map(item => redactPIIFromObject(item)) as T;
    }

    if (value && typeof value === 'object') {
        const sanitized: Record<string, unknown> = {};
        for (const [key, entry] of Object.entries(value)) {
            sanitized[key] = redactPIIFromObject(entry);
        }
        return sanitized as T;
    }

    return value;
}
