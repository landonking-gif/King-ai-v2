/**
 * Cost Tracker
 * Monitors cloud model usage costs and enforces budgets
 */

import { getRedisClient } from '@mother-harness/shared';

/** Cost per 1K tokens for each model */
const MODEL_COSTS: Record<string, number> = {
    'gpt-oss:20b-q4': 0,
    'gpt-oss:20b-q5': 0,
    'gpt-oss:20b-fp8': 0,
    'devstral-2:123b-cloud': 0.003,
    'deepseek-v3.1:671b-cloud': 0.004,
    'qwen3-next:80b-cloud': 0.002,
    'gemini-3-flash-preview-cloud': 0.002,
};

/** Budget thresholds */
const BUDGETS = {
    daily: { limit: 10.00, warning: 8.00 },
    monthly: { limit: 100.00, warning: 80.00 },
};

export interface UsageRecord {
    user_id: string;
    model: string;
    tokens: number;
    cost: number;
    timestamp: string;
}

export interface BudgetStatus {
    daily_spend: number;
    monthly_spend: number;
    daily_remaining: number;
    monthly_remaining: number;
    daily_warning: boolean;
    monthly_warning: boolean;
    can_use_cloud: boolean;
}

export class CostTracker {
    private redis = getRedisClient();

    /**
     * Track token usage for a user
     */
    async trackUsage(userId: string, model: string, tokensUsed: number): Promise<void> {
        const cost = (tokensUsed / 1000) * (MODEL_COSTS[model] ?? 0);

        if (cost === 0) {
            // Local model, no cost tracking needed
            return;
        }

        const today = new Date().toISOString().split('T')[0] ?? '';
        const month = today.substring(0, 7);

        // Increment daily spend
        await this.redis.hincrbyfloat(`cost:${userId}:daily:${today}`, model, cost);
        await this.redis.hincrbyfloat(`cost:${userId}:daily:${today}`, 'total', cost);

        // Increment monthly spend
        await this.redis.hincrbyfloat(`cost:${userId}:monthly:${month}`, model, cost);
        await this.redis.hincrbyfloat(`cost:${userId}:monthly:${month}`, 'total', cost);

        // Set expiration (7 days for daily, 60 days for monthly)
        await this.redis.expire(`cost:${userId}:daily:${today}`, 7 * 24 * 60 * 60);
        await this.redis.expire(`cost:${userId}:monthly:${month}`, 60 * 24 * 60 * 60);

        // Check for budget alerts
        const status = await this.getBudgetStatus(userId);

        if (status.daily_warning || status.monthly_warning) {
            await this.sendBudgetAlert(userId, status);
        }
    }

    /**
     * Get current budget status for a user
     */
    async getBudgetStatus(userId: string): Promise<BudgetStatus> {
        const today = new Date().toISOString().split('T')[0] ?? '';
        const month = today.substring(0, 7);

        const dailySpend = await this.getSpend(userId, 'daily', today);
        const monthlySpend = await this.getSpend(userId, 'monthly', month);

        return {
            daily_spend: dailySpend,
            monthly_spend: monthlySpend,
            daily_remaining: BUDGETS.daily.limit - dailySpend,
            monthly_remaining: BUDGETS.monthly.limit - monthlySpend,
            daily_warning: dailySpend >= BUDGETS.daily.warning,
            monthly_warning: monthlySpend >= BUDGETS.monthly.warning,
            can_use_cloud: dailySpend < BUDGETS.daily.limit && monthlySpend < BUDGETS.monthly.limit,
        };
    }

    /**
     * Get spend for a period
     */
    private async getSpend(
        userId: string,
        period: 'daily' | 'monthly',
        date: string
    ): Promise<number> {
        const key = `cost:${userId}:${period}:${date}`;
        const total = await this.redis.hget(key, 'total');
        return total ? parseFloat(total) : 0;
    }

    /**
     * Send budget alert using AlertManager
     */
    private async sendBudgetAlert(userId: string, status: BudgetStatus): Promise<void> {
        const { getAlertManager } = await import('@mother-harness/shared');
        const alert = getAlertManager();

        if (status.daily_warning) {
            await alert.budgetWarning(userId, status.daily_spend, BUDGETS.daily.limit, 'daily');
        }
        if (status.monthly_warning) {
            await alert.budgetWarning(userId, status.monthly_spend, BUDGETS.monthly.limit, 'monthly');
        }
    }

    /**
     * Get usage report for a user
     */
    async getUsageReport(userId: string): Promise<{
        daily: Record<string, number>;
        monthly: Record<string, number>;
        by_model: Record<string, number>;
    }> {
        const today = new Date().toISOString().split('T')[0] ?? '';
        const month = today.substring(0, 7);

        const dailyKey = `cost:${userId}:daily:${today}`;
        const monthlyKey = `cost:${userId}:monthly:${month}`;

        const daily = await this.redis.hgetall(dailyKey);
        const monthly = await this.redis.hgetall(monthlyKey);

        // Parse values to numbers
        const parseHash = (hash: Record<string, string>) => {
            const result: Record<string, number> = {};
            for (const [key, value] of Object.entries(hash)) {
                result[key] = parseFloat(value) || 0;
            }
            return result;
        };

        const removeTotal = (hash: Record<string, number>) => {
            const { total, ...rest } = hash;
            return rest;
        };

        return {
            daily: parseHash(daily),
            monthly: parseHash(monthly),
            by_model: removeTotal(parseHash(monthly)),
        };
    }

    /**
     * Check if a model can be used (within budget)
     */
    async canUseModel(userId: string, model: string, estimatedTokens: number): Promise<boolean> {
        const cost = (estimatedTokens / 1000) * (MODEL_COSTS[model] ?? 0);

        if (cost === 0) return true; // Local models always allowed

        const status = await this.getBudgetStatus(userId);
        return status.daily_remaining >= cost && status.monthly_remaining >= cost;
    }
}

// Singleton
let trackerInstance: CostTracker | null = null;

export function getCostTracker(): CostTracker {
    if (!trackerInstance) {
        trackerInstance = new CostTracker();
    }
    return trackerInstance;
}
