import { describe, it, expect } from 'vitest';

import { normalizeReviewOutputs } from './normalize.js';

describe('normalizeReviewOutputs', () => {
    it('normalizes critic outputs (area + recommendation + severity)', () => {
        const normalized = normalizeReviewOutputs('critic', {
            review_report: { approval: 'changes_requested', summary: 'Needs work' },
            issues_found: [
                {
                    severity: 'high',
                    category: 'security',
                    description: 'SQL injection risk',
                    suggestion: 'Use parameterized queries',
                    location: 'src/db.ts:12',
                },
            ],
        });

        expect(normalized.decision).toBe('changes_requested');
        expect(normalized.summary).toBe('Needs work');
        expect(normalized.issues).toHaveLength(1);
        expect(normalized.issues[0]).toMatchObject({
            severity: 'high',
            area: 'security',
            description: 'SQL injection risk',
            recommendation: 'Use parameterized queries',
            location: 'src/db.ts:12',
            source_agent: 'critic',
        });
    });

    it('normalizes skeptic legacy levels into shared severity vocabulary', () => {
        const normalized = normalizeReviewOutputs('skeptic', {
            assessment: 'Looks risky',
            verdict: 'proceed_with_changes',
            concerns: [
                {
                    level: 'significant',
                    area: 'security',
                    issue: 'Missing rate limiting',
                    potential_impact: 'Abuse risk',
                    mitigation: 'Add throttling',
                },
            ],
        });

        expect(normalized.decision).toBe('proceed_with_changes');
        expect(normalized.summary).toBe('Looks risky');
        expect(normalized.issues).toHaveLength(1);
        expect(normalized.issues[0]).toMatchObject({
            severity: 'high',
            area: 'security',
            description: 'Missing rate limiting',
            recommendation: 'Add throttling',
            source_agent: 'skeptic',
        });
    });
});
