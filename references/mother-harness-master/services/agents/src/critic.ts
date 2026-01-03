/**
 * Critic Agent
 * Reviews code and artifacts for quality, security, and correctness
 */

import type { AgentType } from '@mother-harness/shared';
import { getLLMClient } from '@mother-harness/shared';
import { BaseAgent, type AgentContext, type AgentResult } from './base-agent.js';

/** Issue severity levels */
export type IssueSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';

/** Review issue */
export interface ReviewIssue {
    severity: IssueSeverity;
    category: 'security' | 'performance' | 'correctness' | 'style' | 'documentation';
    /** Normalized alias for chaining across review agents */
    area?: string;
    location?: string;
    description: string;
    suggestion: string;
    /** Normalized alias for chaining across review agents */
    recommendation?: string;
}

/** Review result */
interface ReviewResult {
    approval: 'approved' | 'changes_requested' | 'blocked';
    summary: string;
    issues: ReviewIssue[];
    positive_notes: string[];
    tokens: number;
}

const CRITIC_SYSTEM_PROMPT = `You are an expert code reviewer and QA specialist. Your role is to:
1. Identify bugs, security vulnerabilities, and logic errors
2. Check for performance issues and anti-patterns
3. Ensure code follows best practices
4. Verify documentation and type safety
5. Provide constructive, actionable feedback

Be thorough but fair. Acknowledge good work while identifying areas for improvement.
Focus on issues that matter - don't nitpick minor style preferences.`;

const REVIEW_PROMPT = `Review the following work:

Type: {review_type}
Content:
{content}

Context:
{context}

Provide a thorough review. Return a JSON object:
{
  "approval": "approved" | "changes_requested" | "blocked",
  "summary": "Brief summary of the review (2-3 sentences)",
  "issues": [
    {
      "severity": "critical" | "high" | "medium" | "low" | "info",
      "category": "security" | "performance" | "correctness" | "style" | "documentation",
      "location": "file:line or section (if applicable)",
      "description": "What the issue is",
      "suggestion": "How to fix it"
    }
  ],
  "positive_notes": [
    "Things that were done well"
  ],
  "grounded": true/false (is this review based on verifiable information?)
}

Approval criteria:
- "approved": No critical/high issues, code is production-ready
- "changes_requested": Has medium+ issues that should be addressed
- "blocked": Has critical security or correctness issues`;

export class CriticAgent extends BaseAgent {
    readonly agentType: AgentType = 'critic';
    private llm = getLLMClient();

    protected async run(inputs: string, context: AgentContext): Promise<AgentResult> {
        const startTime = Date.now();

        // Determine what type of content is being reviewed
        const reviewType = this.detectReviewType(inputs);

        // Conduct the review
        const reviewResult = await this.conductReview(inputs, reviewType, context);

        const criticalCount = reviewResult.issues.filter(i => i.severity === 'critical').length;
        const highCount = reviewResult.issues.filter(i => i.severity === 'high').length;
        const totalCount = reviewResult.issues.length;

        const issues = reviewResult.issues.map(issue => ({
            ...issue,
            area: issue.area ?? issue.category,
            recommendation: issue.recommendation ?? issue.suggestion,
        }));

        return {
            success: true,
            outputs: {
                review_report: {
                    approval: reviewResult.approval,
                    summary: reviewResult.summary,
                    positives: reviewResult.positive_notes,
                    counts: {
                        critical: criticalCount,
                        high: highCount,
                        total: totalCount,
                    },
                },
                issues_found: issues,
                approval: reviewResult.approval,
                summary: reviewResult.summary,
                issues,
                positive_notes: reviewResult.positive_notes,
                critical_issues: criticalCount,
                high_issues: highCount,
                total_issues: totalCount,
            },
            explanation: `Review ${reviewResult.approval}: ${criticalCount} critical, ${highCount} high, ${totalCount} total issues`,
            tokens_used: reviewResult.tokens,
            duration_ms: Date.now() - startTime,
        };
    }

    private detectReviewType(inputs: string): string {
        const lower = inputs.toLowerCase();

        if (lower.includes('```') || lower.includes('function ') || lower.includes('class ')) {
            return 'code';
        }
        if (lower.includes('security') || lower.includes('vulnerability')) {
            return 'security';
        }
        if (lower.includes('design') || lower.includes('architecture')) {
            return 'design';
        }
        if (lower.includes('research') || lower.includes('report')) {
            return 'research';
        }

        return 'general';
    }

    private async conductReview(
        content: string,
        reviewType: string,
        context: AgentContext
    ): Promise<ReviewResult> {
        const prompt = REVIEW_PROMPT
            .replace('{review_type}', reviewType)
            .replace('{content}', content)
            .replace('{context}', context.recent_context ?? 'No additional context');

        const result = await this.llm.json<{
            approval: 'approved' | 'changes_requested' | 'blocked';
            summary: string;
            issues: ReviewIssue[];
            positive_notes: string[];
            grounded?: boolean;
        }>(prompt, {
            system: CRITIC_SYSTEM_PROMPT,
            temperature: 0.3, // Lower temperature for consistent reviews
            max_tokens: 4096,
        });

        if (result.data) {
            return {
                approval: result.data.approval,
                summary: result.data.summary,
                issues: (result.data.issues || []).map(issue => ({
                    ...issue,
                    area: issue.area ?? issue.category,
                    recommendation: issue.recommendation ?? issue.suggestion,
                })),
                positive_notes: result.data.positive_notes || [],
                tokens: result.raw.tokens_used.total,
            };
        }

        // Fallback if LLM fails
        console.warn('[CriticAgent] Review failed');
        return {
            approval: 'changes_requested',
            summary: 'Review could not be completed due to a processing error.',
            issues: [{
                severity: 'info',
                category: 'documentation',
                area: 'documentation',
                description: 'Automated review failed',
                suggestion: 'Please retry or conduct manual review',
                recommendation: 'Please retry or conduct manual review',
            }],
            positive_notes: [],
            tokens: result.raw.tokens_used.total,
        };
    }
}
