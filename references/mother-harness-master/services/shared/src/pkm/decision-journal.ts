/**
 * Decision Journal
 * Track important decisions with context, reasoning, and outcomes
 */

import { getRedisJSON } from '../redis/index.js';
import { nanoid } from 'nanoid';

/** Decision status */
export type DecisionStatus = 'pending' | 'implemented' | 'evaluated' | 'revised' | 'abandoned';

/** Decision record */
export interface Decision {
    id: string;
    user_id: string;
    project_id?: string;
    task_id?: string;

    // Core fields
    title: string;
    context: string;           // Situation that led to decision
    options_considered: Array<{
        name: string;
        pros: string[];
        cons: string[];
    }>;
    decision: string;          // What was decided
    reasoning: string;         // Why this option was chosen

    // Stakeholders
    stakeholders?: string[];

    // Timeline
    made_at: string;
    review_at?: string;        // When to review the decision
    reviewed_at?: string;

    // Outcomes
    status: DecisionStatus;
    actual_outcome?: string;
    lessons_learned?: string;

    // Tags
    tags: string[];

    // Metadata
    created_at: string;
    updated_at: string;
}

export class DecisionJournal {
    private redis = getRedisJSON();
    private readonly prefix = 'decision:';

    /**
     * Record a new decision
     */
    async recordDecision(
        userId: string,
        data: {
            title: string;
            context: string;
            options_considered: Decision['options_considered'];
            decision: string;
            reasoning: string;
            project_id?: string;
            task_id?: string;
            stakeholders?: string[];
            review_at?: string;
            tags?: string[];
        }
    ): Promise<Decision> {
        const now = new Date().toISOString();

        const decision: Decision = {
            id: `decision-${nanoid()}`,
            user_id: userId,
            ...(data.project_id !== undefined && { project_id: data.project_id }),
            ...(data.task_id !== undefined && { task_id: data.task_id }),
            title: data.title,
            context: data.context,
            options_considered: data.options_considered,
            decision: data.decision,
            reasoning: data.reasoning,
            ...(data.stakeholders !== undefined && { stakeholders: data.stakeholders }),
            made_at: now,
            ...(data.review_at !== undefined && { review_at: data.review_at }),
            status: 'pending',
            tags: data.tags ?? [],
            created_at: now,
            updated_at: now,
        };

        await this.redis.set(`${this.prefix}${decision.id}`, '$', decision);

        return decision;
    }

    /**
     * Get a decision by ID
     */
    async getDecision(decisionId: string): Promise<Decision | null> {
        return await this.redis.get<Decision>(`${this.prefix}${decisionId}`);
    }

    /**
     * Update decision status
     */
    async updateStatus(
        decisionId: string,
        status: DecisionStatus,
        outcome?: { actual_outcome?: string; lessons_learned?: string }
    ): Promise<Decision | null> {
        const decision = await this.getDecision(decisionId);
        if (!decision) return null;

        decision.status = status;
        decision.updated_at = new Date().toISOString();

        if (status === 'evaluated' || status === 'revised') {
            decision.reviewed_at = decision.updated_at;
        }

        if (outcome?.actual_outcome) {
            decision.actual_outcome = outcome.actual_outcome;
        }

        if (outcome?.lessons_learned) {
            decision.lessons_learned = outcome.lessons_learned;
        }

        await this.redis.set(`${this.prefix}${decisionId}`, '$', decision);

        return decision;
    }

    /**
     * Get decisions for a user
     */
    async getUserDecisions(
        userId: string,
        options: {
            status?: DecisionStatus;
            project_id?: string;
            limit?: number;
        } = {}
    ): Promise<Decision[]> {
        const keys = await this.redis.keys(`${this.prefix}*`);
        const decisions: Decision[] = [];

        for (const key of keys) {
            const decision = await this.redis.get<Decision>(key);
            if (!decision) continue;

            if (decision.user_id !== userId) continue;
            if (options.status && decision.status !== options.status) continue;
            if (options.project_id && decision.project_id !== options.project_id) continue;

            decisions.push(decision);
        }

        return decisions
            .sort((a, b) => new Date(b.made_at).getTime() - new Date(a.made_at).getTime())
            .slice(0, options.limit ?? 50);
    }

    /**
     * Get decisions due for review
     */
    async getDueForReview(userId: string): Promise<Decision[]> {
        const now = new Date();
        const decisions = await this.getUserDecisions(userId);

        return decisions.filter(d => {
            if (d.status !== 'pending' && d.status !== 'implemented') return false;
            if (!d.review_at) return false;
            return new Date(d.review_at) <= now;
        });
    }

    /**
     * Get decision statistics
     */
    async getStats(userId: string): Promise<{
        total: number;
        by_status: Record<DecisionStatus, number>;
        pending_reviews: number;
        avg_time_to_evaluate_days: number;
    }> {
        const decisions = await this.getUserDecisions(userId);

        const stats = {
            total: decisions.length,
            by_status: {
                pending: 0,
                implemented: 0,
                evaluated: 0,
                revised: 0,
                abandoned: 0,
            },
            pending_reviews: 0,
            avg_time_to_evaluate_days: 0,
        };

        const now = new Date();
        let evaluatedCount = 0;
        let totalEvalDays = 0;

        for (const d of decisions) {
            stats.by_status[d.status]++;

            if (d.review_at && new Date(d.review_at) <= now && d.status === 'pending') {
                stats.pending_reviews++;
            }

            if (d.reviewed_at) {
                const days = (new Date(d.reviewed_at).getTime() - new Date(d.made_at).getTime()) / (1000 * 60 * 60 * 24);
                totalEvalDays += days;
                evaluatedCount++;
            }
        }

        if (evaluatedCount > 0) {
            stats.avg_time_to_evaluate_days = Math.round(totalEvalDays / evaluatedCount);
        }

        return stats;
    }
}

// Singleton
let journalInstance: DecisionJournal | null = null;

export function getDecisionJournal(): DecisionJournal {
    if (!journalInstance) {
        journalInstance = new DecisionJournal();
    }
    return journalInstance;
}
