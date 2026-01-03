/**
 * PII Redaction Validation Tests
 * Tests redaction rules against sample documents with realistic PII
 */

import { describe, it, expect } from 'vitest';
import { redactPII, redactPIIFromObject } from './redaction.js';

describe('PII Redaction - Email Addresses', () => {
    it('should redact simple email addresses', () => {
        const input = 'Contact john.doe@example.com for details';
        const output = redactPII(input);
        expect(output).toBe('Contact [REDACTED_EMAIL] for details');
        expect(output).not.toContain('@example.com');
    });

    it('should redact multiple email addresses', () => {
        const input = 'Send to alice@corp.com and bob@company.org';
        const output = redactPII(input);
        expect(output).toBe('Send to [REDACTED_EMAIL] and [REDACTED_EMAIL]');
    });

    it('should redact emails with complex formats', () => {
        const input = 'Email: user+tag@subdomain.example.co.uk';
        const output = redactPII(input);
        expect(output).not.toContain('@subdomain.example.co.uk');
        expect(output).toContain('[REDACTED_EMAIL]');
    });
});

describe('PII Redaction - Phone Numbers', () => {
    it('should redact US phone numbers', () => {
        const input = 'Call me at (555) 123-4567 tomorrow';
        const output = redactPII(input);
        expect(output).toBe('Call me at [REDACTED_PHONE] tomorrow');
    });

    it('should redact phone numbers in various formats', () => {
        const formats = [
            '555-123-4567',
            '(555) 123-4567',
            '555.123.4567',
            '5551234567',
            '+1 555 123 4567',
        ];

        formats.forEach(phone => {
            const output = redactPII(`Contact: ${phone}`);
            expect(output).toContain('[REDACTED_PHONE]');
            expect(output).not.toContain(phone);
        });
    });

    it('should redact international phone numbers', () => {
        const input = 'Call +44 20 7946 0958 for UK office';
        const output = redactPII(input);
        expect(output).toContain('[REDACTED_PHONE]');
    });
});

describe('PII Redaction - Social Security Numbers', () => {
    it('should redact SSN format', () => {
        const input = 'SSN: 123-45-6789';
        const output = redactPII(input);
        expect(output).toBe('SSN: [REDACTED_SSN]');
    });

    it('should redact multiple SSNs in text', () => {
        const input = 'Employee 1: 111-22-3333, Employee 2: 444-55-6666';
        const output = redactPII(input);
        expect(output).not.toContain('111-22-3333');
        expect(output).not.toContain('444-55-6666');
        expect(output.match(/\[REDACTED_SSN\]/g)?.length).toBe(2);
    });
});

describe('PII Redaction - Credit Card Numbers', () => {
    it('should redact credit card numbers', () => {
        const input = 'Card: 4532-1234-5678-9010';
        const output = redactPII(input);
        expect(output).toContain('[REDACTED_CARD]');
        expect(output).not.toContain('4532');
    });

    it('should redact cards without dashes', () => {
        const input = 'Payment: 4532123456789010';
        const output = redactPII(input);
        expect(output).toContain('[REDACTED_CARD]');
    });

    it('should redact various card formats', () => {
        const cards = [
            '4532 1234 5678 9010', // Visa
            '5425-2334-3010-9903', // MasterCard
            '3782 822463 10005',   // Amex
        ];

        cards.forEach(card => {
            const output = redactPII(`Card number: ${card}`);
            expect(output).toContain('[REDACTED_CARD]');
        });
    });
});

describe('PII Redaction - Complex Documents', () => {
    it('should redact all PII from customer support ticket', () => {
        const ticket = `
Customer: John Smith
Email: john.smith@gmail.com
Phone: (555) 234-5678
SSN: 987-65-4321
Payment Method: Card ending in 1234 (full: 4532-1234-5678-9010)

Issue: Cannot access account. Please verify my identity.
        `.trim();

        const redacted = redactPII(ticket);

        expect(redacted).not.toContain('john.smith@gmail.com');
        expect(redacted).not.toContain('(555) 234-5678');
        expect(redacted).not.toContain('987-65-4321');
        expect(redacted).not.toContain('4532-1234-5678-9010');
        expect(redacted).toContain('[REDACTED_EMAIL]');
        expect(redacted).toContain('[REDACTED_PHONE]');
        expect(redacted).toContain('[REDACTED_SSN]');
        expect(redacted).toContain('[REDACTED_CARD]');
    });

    it('should redact PII from API error logs', () => {
        const log = `
[ERROR] Failed to process payment for user alice@example.com
User phone: 555-111-2222
Card: 4111111111111111
Transaction ID: tx_12345
        `.trim();

        const redacted = redactPII(log);

        expect(redacted).not.toContain('alice@example.com');
        expect(redacted).not.toContain('555-111-2222');
        expect(redacted).not.toContain('4111111111111111');
        expect(redacted).toContain('tx_12345'); // Non-PII should remain
    });

    it('should redact PII from medical records excerpt', () => {
        const record = `
Patient: Sarah Johnson
DOB: 01/15/1985
SSN: 456-78-9012
Contact: sarah.j@healthcare.org
Emergency: (555) 987-6543
Insurance: Policy #INS-789456
        `.trim();

        const redacted = redactPII(record);

        expect(redacted).not.toContain('456-78-9012');
        expect(redacted).not.toContain('sarah.j@healthcare.org');
        expect(redacted).not.toContain('(555) 987-6543');
        expect(redacted).toContain('Sarah Johnson'); // Names not redacted (not in scope)
        expect(redacted).toContain('01/15/1985'); // Dates not redacted
    });
});

describe('PII Redaction - Object Structure', () => {
    it('should redact PII from nested objects', () => {
        const data = {
            user: {
                name: 'John Doe',
                email: 'john@example.com',
                phone: '555-123-4567',
                address: {
                    street: '123 Main St',
                    contact: 'support@example.org',
                },
            },
            payment: {
                card: '4532-1234-5678-9010',
                ssn: '123-45-6789',
            },
        };

        const redacted = redactPIIFromObject(data);

        expect(redacted.user.email).toBe('[REDACTED_EMAIL]');
        expect(redacted.user.phone).toBe('[REDACTED_PHONE]');
        expect(redacted.user.address.contact).toBe('[REDACTED_EMAIL]');
        expect(redacted.payment.card).toBe('[REDACTED_CARD]');
        expect(redacted.payment.ssn).toBe('[REDACTED_SSN]');
        expect(redacted.user.name).toBe('John Doe'); // Non-PII preserved
    });

    it('should redact PII from arrays', () => {
        const contacts = [
            { name: 'Alice', email: 'alice@example.com' },
            { name: 'Bob', email: 'bob@company.org' },
        ];

        const redacted = redactPIIFromObject(contacts);

        expect(redacted[0].email).toBe('[REDACTED_EMAIL]');
        expect(redacted[1].email).toBe('[REDACTED_EMAIL]');
        expect(redacted[0].name).toBe('Alice');
        expect(redacted[1].name).toBe('Bob');
    });

    it('should handle mixed data types', () => {
        const mixed = {
            count: 42,
            active: true,
            message: 'Contact support@example.com',
            items: ['item1', 'Call 555-123-4567', 'item3'],
            metadata: null,
            timestamp: new Date('2024-01-01'),
        };

        const redacted = redactPIIFromObject(mixed);

        expect(redacted.count).toBe(42);
        expect(redacted.active).toBe(true);
        expect(redacted.message).toBe('Contact [REDACTED_EMAIL]');
        expect(redacted.items[1]).toBe('Call [REDACTED_PHONE]');
        expect(redacted.metadata).toBeNull();
    });
});

describe('PII Redaction - Edge Cases', () => {
    it('should handle empty strings', () => {
        expect(redactPII('')).toBe('');
    });

    it('should handle strings with no PII', () => {
        const input = 'This is a regular sentence with no sensitive data.';
        expect(redactPII(input)).toBe(input);
    });

    it('should not over-redact non-PII numbers', () => {
        const input = 'Order #12345 for $99.99 placed on 2024-01-01';
        const output = redactPII(input);
        expect(output).toBe(input); // No PII to redact
    });

    it('should handle malformed email-like strings', () => {
        const input = 'Not an email: user@';
        const output = redactPII(input);
        expect(output).toBe(input); // Should not match incomplete pattern
    });

    it('should preserve redaction markers when re-redacting', () => {
        const input = 'Email: test@example.com';
        const redacted1 = redactPII(input);
        const redacted2 = redactPII(redacted1);
        expect(redacted1).toBe(redacted2);
    });
});

describe('PII Redaction - Production Scenarios', () => {
    it('should redact user task query with personal info', () => {
        const query = `
Help me draft an email to john.doe@company.com about the contract.
My phone is 555-888-9999 if they need to reach me.
Reference account #123-45-6789 in the discussion.
        `.trim();

        const redacted = redactPII(query);

        expect(redacted).not.toContain('john.doe@company.com');
        expect(redacted).not.toContain('555-888-9999');
        expect(redacted).not.toContain('123-45-6789');
    });

    it('should redact document content before vector storage', () => {
        const document = {
            title: 'Employee Handbook',
            content: `
For benefits questions, contact hr@company.com or call (555) 100-2000.
Report incidents to security@company.com within 24 hours.
Employee ID format: XXX-XX-XXXX (e.g., 789-01-2345)
            `.trim(),
            metadata: {
                author: 'HR Department',
                contact: 'benefits@company.com',
            },
        };

        const redacted = redactPIIFromObject(document);

        expect(redacted.content).not.toContain('hr@company.com');
        expect(redacted.content).not.toContain('security@company.com');
        expect(redacted.content).not.toContain('(555) 100-2000');
        expect(redacted.metadata.contact).toBe('[REDACTED_EMAIL]');
    });

    it('should redact approval request preview with sensitive data', () => {
        const approval = {
            action: 'send_email',
            preview: 'Sending invoice to customer@example.com with payment link for card 4532123456789010',
            risk_factors: ['contains email', 'contains payment info'],
        };

        const redacted = redactPIIFromObject(approval);

        expect(redacted.preview).not.toContain('customer@example.com');
        expect(redacted.preview).not.toContain('4532123456789010');
        expect(redacted.preview).toContain('[REDACTED_EMAIL]');
        expect(redacted.preview).toContain('[REDACTED_CARD]');
    });
});

describe('PII Redaction - Performance', () => {
    it('should handle large documents efficiently', () => {
        const largeDoc = Array(1000)
            .fill('Some text with user@example.com and 555-123-4567. ')
            .join('');

        const start = performance.now();
        const redacted = redactPII(largeDoc);
        const duration = performance.now() - start;

        expect(redacted).not.toContain('user@example.com');
        expect(duration).toBeLessThan(100); // Should complete in under 100ms
    });

    it('should handle deeply nested objects', () => {
        const deep = {
            level1: {
                level2: {
                    level3: {
                        level4: {
                            email: 'deep@example.com',
                            phone: '555-999-8888',
                        },
                    },
                },
            },
        };

        const redacted = redactPIIFromObject(deep);

        expect(redacted.level1.level2.level3.level4.email).toBe('[REDACTED_EMAIL]');
        expect(redacted.level1.level2.level3.level4.phone).toBe('[REDACTED_PHONE]');
    });
});
