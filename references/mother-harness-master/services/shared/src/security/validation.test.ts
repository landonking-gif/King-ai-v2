import { describe, it, expect } from 'vitest';
import {
    validateInput,
    sanitizeString,
    containsDangerousPattern,
    sanitizePath,
    validateJSON,
    generateSecureToken,
    maskSensitiveData,
    sanitizeQuery,
} from '../security/validation.js';

describe('validateInput', () => {
    it('should validate alphanumeric strings', () => {
        expect(validateInput('abc123', 'alphanumeric')).toBe(true);
        expect(validateInput('abc-123', 'alphanumeric')).toBe(false);
        expect(validateInput('abc 123', 'alphanumeric')).toBe(false);
    });

    it('should validate alphanumeric with dashes', () => {
        expect(validateInput('abc-123_xyz', 'alphanumeric_dash')).toBe(true);
        expect(validateInput('abc 123', 'alphanumeric_dash')).toBe(false);
    });

    it('should validate email format', () => {
        expect(validateInput('user@example.com', 'email')).toBe(true);
        expect(validateInput('invalid-email', 'email')).toBe(false);
    });
});

describe('sanitizeString', () => {
    it('should trim whitespace', () => {
        expect(sanitizeString('  hello  ')).toBe('hello');
    });

    it('should truncate long strings', () => {
        const long = 'a'.repeat(2000);
        expect(sanitizeString(long, 100).length).toBe(100);
    });

    it('should HTML encode special characters', () => {
        expect(sanitizeString('<script>')).toBe('&lt;script&gt;');
        expect(sanitizeString('a & b')).toBe('a &amp; b');
        expect(sanitizeString('"quoted"')).toBe('&quot;quoted&quot;');
    });
});

describe('containsDangerousPattern', () => {
    it('should detect path traversal', () => {
        expect(containsDangerousPattern('../etc/passwd')).toBe(true);
        expect(containsDangerousPattern('normal/path')).toBe(false);
    });

    it('should detect script tags', () => {
        expect(containsDangerousPattern('<script>alert(1)</script>')).toBe(true);
    });

    it('should detect template injection', () => {
        expect(containsDangerousPattern('${process.env.SECRET}')).toBe(true);
        expect(containsDangerousPattern('{{constructor.constructor}}')).toBe(true);
    });
});

describe('sanitizePath', () => {
    it('should reject path traversal', () => {
        expect(sanitizePath('../etc/passwd', '/safe')).toBeNull();
        expect(sanitizePath('/safe/../etc', '/safe')).toBeNull();
    });

    it('should reject paths outside base', () => {
        expect(sanitizePath('/other/path', '/safe')).toBeNull();
    });

    it('should allow valid paths', () => {
        expect(sanitizePath('/safe/file.txt', '/safe')).toBe('/safe/file.txt');
    });
});

describe('validateJSON', () => {
    it('should check required fields', () => {
        const result = validateJSON({ name: 'test' }, { required: ['name', 'value'] });
        expect(result.valid).toBe(false);
        expect(result.error).toContain('value');
    });

    it('should check object depth', () => {
        const deep = { a: { b: { c: { d: { e: 1 } } } } };
        const result = validateJSON(deep, { maxDepth: 3 });
        expect(result.valid).toBe(false);
    });

    it('should check object size', () => {
        const large = { data: 'x'.repeat(1000) };
        const result = validateJSON(large, { maxSize: 100 });
        expect(result.valid).toBe(false);
    });
});

describe('generateSecureToken', () => {
    it('should generate tokens of specified length', () => {
        expect(generateSecureToken(16).length).toBe(16);
        expect(generateSecureToken(32).length).toBe(32);
    });

    it('should generate unique tokens', () => {
        const tokens = new Set([
            generateSecureToken(),
            generateSecureToken(),
            generateSecureToken(),
        ]);
        expect(tokens.size).toBe(3);
    });
});

describe('maskSensitiveData', () => {
    it('should mask password fields', () => {
        const masked = maskSensitiveData({ password: 'FAKE_TEST_SECRET' });
        expect(masked.password).not.toBe('FAKE_TEST_SECRET');
        expect(masked.password).toContain('***');
    });

    it('should mask nested sensitive fields', () => {
        const masked = maskSensitiveData({
            config: { api_key: 'FAKE_TEST_KEY' },
        });
        expect((masked.config as Record<string, string>).api_key).toContain('***');
    });

    it('should preserve non-sensitive fields', () => {
        const masked = maskSensitiveData({ name: 'test', count: 42 });
        expect(masked.name).toBe('test');
        expect(masked.count).toBe(42);
    });
});

describe('sanitizeQuery', () => {
    it('should accept valid queries', () => {
        const result = sanitizeQuery('How do I build a web app?');
        expect(result.valid).toBe(true);
        expect(result.sanitized).toBe('How do I build a web app?');
    });

    it('should reject empty queries', () => {
        expect(sanitizeQuery('').valid).toBe(false);
    });

    it('should reject queries with dangerous patterns', () => {
        expect(sanitizeQuery('<script>alert(1)</script>').valid).toBe(false);
    });

    it('should reject overly long queries', () => {
        const long = 'a'.repeat(6000);
        expect(sanitizeQuery(long).valid).toBe(false);
    });
});
