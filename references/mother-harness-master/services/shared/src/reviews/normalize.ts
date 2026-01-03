import type { AgentType } from '../types/agent.js';

export type ReviewSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';
export type ReviewSourceAgent = Extract<AgentType, 'critic' | 'skeptic'>;

export interface NormalizedReviewIssue {
    severity: ReviewSeverity;
    area: string;
    description: string;
    recommendation?: string;
    location?: string;
    source_agent: ReviewSourceAgent;
    raw: unknown;
}

export interface NormalizedReview {
    summary?: string;
    decision?: string;
    issues: NormalizedReviewIssue[];
}

export function normalizeReviewOutputs(
    agent: ReviewSourceAgent,
    outputs: Record<string, unknown>
): NormalizedReview {
    if (agent === 'critic') {
        return normalizeCritic(outputs);
    }

    return normalizeSkeptic(outputs);
}

function normalizeCritic(outputs: Record<string, unknown>): NormalizedReview {
    const reviewReport = isRecord(outputs.review_report) ? outputs.review_report : undefined;
    const summary = typeof reviewReport?.summary === 'string'
        ? reviewReport.summary
        : (typeof outputs.summary === 'string' ? outputs.summary : undefined);

    const decision = typeof reviewReport?.approval === 'string'
        ? reviewReport.approval
        : (typeof outputs.approval === 'string' ? outputs.approval : undefined);

    const issuesRaw = Array.isArray(outputs.issues_found)
        ? outputs.issues_found
        : (Array.isArray(outputs.issues) ? outputs.issues : []);

    const issues: NormalizedReviewIssue[] = issuesRaw
        .filter(isRecord)
        .map((issue): NormalizedReviewIssue => {
            const severity = coerceSeverity(issue.severity);
            const area = coerceArea(issue.area, issue.category);
            const description = typeof issue.description === 'string'
                ? issue.description
                : 'Issue reported by critic';
            const recommendation = typeof issue.recommendation === 'string'
                ? issue.recommendation
                : (typeof issue.suggestion === 'string' ? issue.suggestion : undefined);
            const location = typeof issue.location === 'string' ? issue.location : undefined;

            return {
                severity,
                area,
                description,
                source_agent: 'critic',
                raw: issue,
                ...(recommendation ? { recommendation } : {}),
                ...(location ? { location } : {}),
            };
        });

    return {
        issues,
        ...(summary ? { summary } : {}),
        ...(decision ? { decision } : {}),
    };
}

function normalizeSkeptic(outputs: Record<string, unknown>): NormalizedReview {
    const summary = typeof outputs.assessment === 'string'
        ? outputs.assessment
        : (typeof outputs.overall_assessment === 'string' ? outputs.overall_assessment : undefined);

    const decision = typeof outputs.verdict === 'string' ? outputs.verdict : undefined;

    const concernsRaw = Array.isArray(outputs.concerns) ? outputs.concerns : [];

    const issues: NormalizedReviewIssue[] = concernsRaw
        .filter(isRecord)
        .map((concern): NormalizedReviewIssue => {
            const severity = coerceSeverity(concern.severity, concern.level);
            const area = coerceArea(concern.area);
            const description = typeof concern.description === 'string'
                ? concern.description
                : (typeof concern.issue === 'string' ? concern.issue : 'Concern raised by skeptic');
            const recommendation = typeof concern.recommendation === 'string'
                ? concern.recommendation
                : (typeof concern.mitigation === 'string' ? concern.mitigation : undefined);

            return {
                severity,
                area,
                description,
                source_agent: 'skeptic',
                raw: concern,
                ...(recommendation ? { recommendation } : {}),
            };
        });

    return {
        issues,
        ...(summary ? { summary } : {}),
        ...(decision ? { decision } : {}),
    };
}

function coerceSeverity(...values: unknown[]): ReviewSeverity {
    for (const v of values) {
        if (typeof v !== 'string') continue;
        switch (v) {
            case 'critical':
            case 'high':
            case 'medium':
            case 'low':
            case 'info':
                return v;
            // legacy skeptic vocab
            case 'significant':
                return 'high';
            case 'minor':
                return 'low';
            case 'consideration':
                return 'info';
        }
    }
    return 'info';
}

function coerceArea(...values: unknown[]): string {
    for (const v of values) {
        if (typeof v === 'string' && v.trim().length > 0) return v;
    }
    return 'general';
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}
