/**
 * Security Utilities
 * Input validation, sanitization, and security helpers
 */

/** Allowed characters for various input types */
const PATTERNS = {
    alphanumeric: /^[a-zA-Z0-9]+$/,
    alphanumeric_dash: /^[a-zA-Z0-9_-]+$/,
    email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    path_safe: /^[a-zA-Z0-9_\-./]+$/,
};

/** Dangerous patterns to block */
const BLOCKED_PATTERNS = [
    /\.\./,                    // Path traversal
    /<script/i,                // XSS
    /javascript:/i,            // XSS
    /on\w+\s*=/i,              // Event handlers
    /data:/i,                  // Data URLs
    /eval\s*\(/,               // Code injection
    /exec\s*\(/,               // Code injection
    /\$\{/,                    // Template injection
    /{{/,                      // Template injection
];

/**
 * Validate input against a pattern
 */
export function validateInput(
    input: string,
    type: keyof typeof PATTERNS
): boolean {
    return PATTERNS[type].test(input);
}

/**
 * Sanitize string input
 */
export function sanitizeString(input: string, maxLength: number = 1000): string {
    // Trim whitespace
    let sanitized = input.trim();

    // Truncate if too long
    if (sanitized.length > maxLength) {
        sanitized = sanitized.substring(0, maxLength);
    }

    // Remove null bytes
    sanitized = sanitized.replace(/\0/g, '');

    // HTML encode special characters
    sanitized = sanitized
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#x27;');

    return sanitized;
}

/**
 * Check for dangerous patterns
 */
export function containsDangerousPattern(input: string): boolean {
    return BLOCKED_PATTERNS.some(pattern => pattern.test(input));
}

/**
 * Sanitize file path
 */
export function sanitizePath(path: string, basePath: string): string | null {
    // Normalize the path
    const normalized = path
        .replace(/\\/g, '/')
        .replace(/\/+/g, '/');

    // Check for path traversal
    if (normalized.includes('..')) {
        return null;
    }

    // Ensure it's within the base path
    if (!normalized.startsWith(basePath)) {
        return null;
    }

    // Validate characters
    if (!PATTERNS.path_safe.test(normalized)) {
        return null;
    }

    return normalized;
}

/**
 * Validate JSON structure
 */
export function validateJSON(
    json: unknown,
    schema: {
        required?: string[];
        maxDepth?: number;
        maxSize?: number;
    }
): { valid: boolean; error?: string } {
    if (typeof json !== 'object' || json === null) {
        return { valid: false, error: 'Input must be an object' };
    }

    const obj = json as Record<string, unknown>;

    // Check required fields
    if (schema.required) {
        for (const field of schema.required) {
            if (!(field in obj)) {
                return { valid: false, error: `Missing required field: ${field}` };
            }
        }
    }

    // Check depth
    if (schema.maxDepth) {
        const depth = getObjectDepth(obj);
        if (depth > schema.maxDepth) {
            return { valid: false, error: `Object too deeply nested (max ${schema.maxDepth})` };
        }
    }

    // Check size
    if (schema.maxSize) {
        const size = JSON.stringify(obj).length;
        if (size > schema.maxSize) {
            return { valid: false, error: `Object too large (max ${schema.maxSize} bytes)` };
        }
    }

    return { valid: true };
}

/**
 * Get depth of nested object
 */
function getObjectDepth(obj: unknown, currentDepth: number = 0): number {
    if (typeof obj !== 'object' || obj === null) {
        return currentDepth;
    }

    let maxDepth = currentDepth;

    for (const value of Object.values(obj)) {
        const depth = getObjectDepth(value, currentDepth + 1);
        if (depth > maxDepth) {
            maxDepth = depth;
        }
    }

    return maxDepth;
}

/**
 * Generate secure random string
 */
export function generateSecureToken(length: number = 32): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';

    // Use crypto if available
    if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
        const array = new Uint8Array(length);
        crypto.getRandomValues(array);
        for (let i = 0; i < length; i++) {
            result += chars[array[i]! % chars.length];
        }
    } else {
        // Fallback (less secure)
        for (let i = 0; i < length; i++) {
            result += chars[Math.floor(Math.random() * chars.length)];
        }
    }

    return result;
}

/**
 * Mask sensitive data for logging
 */
export function maskSensitiveData(obj: Record<string, unknown>): Record<string, unknown> {
    const sensitiveKeys = ['password', 'token', 'secret', 'key', 'apikey', 'api_key', 'auth'];
    const masked: Record<string, unknown> = {};

    for (const [key, value] of Object.entries(obj)) {
        const lowerKey = key.toLowerCase();
        const isSensitive = sensitiveKeys.some(sk => lowerKey.includes(sk));

        if (isSensitive && typeof value === 'string') {
            masked[key] = value.length > 4
                ? `${value.substring(0, 2)}***${value.substring(value.length - 2)}`
                : '***';
        } else if (typeof value === 'object' && value !== null) {
            masked[key] = maskSensitiveData(value as Record<string, unknown>);
        } else {
            masked[key] = value;
        }
    }

    return masked;
}

/**
 * Validate and sanitize query string
 */
export function sanitizeQuery(query: string, maxLength: number = 5000): {
    valid: boolean;
    sanitized?: string;
    error?: string;
} {
    if (!query || typeof query !== 'string') {
        return { valid: false, error: 'Query is required' };
    }

    if (query.length > maxLength) {
        return { valid: false, error: `Query too long (max ${maxLength} characters)` };
    }

    if (containsDangerousPattern(query)) {
        return { valid: false, error: 'Query contains disallowed patterns' };
    }

    return { valid: true, sanitized: query.trim() };
}
