/**
 * Alert Manager
 * Centralized alerting for budget warnings, errors, and notifications
 */

import { getRedisJSON } from '../redis/client.js';

/** Alert severity levels */
export type AlertSeverity = 'info' | 'warning' | 'error' | 'critical';

/** Alert categories */
export type AlertCategory = 'budget' | 'error' | 'security' | 'system' | 'task';

/** Alert configuration */
interface AlertConfig {
    webhookUrl: string | undefined;
    emailEnabled: boolean;
    emailRecipients: string[];
    slackWebhook: string | undefined;
    minSeverity: AlertSeverity;
    cooldownMinutes: number;
}

/** Alert record */
export interface AlertRecord {
    id: string;
    severity: AlertSeverity;
    category: AlertCategory;
    title: string;
    message: string;
    context: Record<string, unknown>;
    timestamp: string;
    acknowledged: boolean;
    notificationsSent: string[];
}

/** Default configuration */
const DEFAULT_CONFIG: AlertConfig = {
    webhookUrl: process.env['ALERT_WEBHOOK_URL'],
    emailEnabled: false,
    emailRecipients: [],
    slackWebhook: process.env['SLACK_WEBHOOK_URL'],
    minSeverity: 'warning',
    cooldownMinutes: 5,
};

/** Severity levels ordered by importance */
const SEVERITY_ORDER: Record<AlertSeverity, number> = {
    info: 0,
    warning: 1,
    error: 2,
    critical: 3,
};

export class AlertManager {
    private redis = getRedisJSON();
    private config: AlertConfig;
    private readonly alertPrefix = 'alert:';
    private readonly cooldownPrefix = 'alert_cooldown:';

    constructor(config?: Partial<AlertConfig>) {
        this.config = { ...DEFAULT_CONFIG, ...config };
    }

    /**
     * Send an alert
     */
    async sendAlert(
        severity: AlertSeverity,
        category: AlertCategory,
        title: string,
        message: string,
        context: Record<string, unknown> = {}
    ): Promise<string> {
        // Check if severity meets minimum threshold
        if (SEVERITY_ORDER[severity] < SEVERITY_ORDER[this.config.minSeverity]) {
            return '';
        }

        // Check cooldown for similar alerts
        const cooldownKey = `${this.cooldownPrefix}${category}:${title}`;
        const onCooldown = await this.redis.get(cooldownKey);
        if (onCooldown) {
            console.log(`[AlertManager] Alert on cooldown: ${title}`);
            return '';
        }

        // Create alert record
        const alertId = `alert-${Date.now()}-${Math.random().toString(36).substring(7)}`;
        const alert: AlertRecord = {
            id: alertId,
            severity,
            category,
            title,
            message,
            context,
            timestamp: new Date().toISOString(),
            acknowledged: false,
            notificationsSent: [],
        };

        // Store alert
        await this.redis.set(`${this.alertPrefix}${alertId}`, '$', alert);

        // Set cooldown
        await this.redis.set(cooldownKey, '$', { timestamp: alert.timestamp });
        // Note: Would set expire via client.expire() for actual cooldown

        // Send notifications
        const notifications: string[] = [];

        // Webhook notification
        if (this.config.webhookUrl) {
            const sent = await this.sendWebhook(alert);
            if (sent) notifications.push('webhook');
        }

        // Slack notification
        if (this.config.slackWebhook) {
            const sent = await this.sendSlack(alert);
            if (sent) notifications.push('slack');
        }

        // Console log (always)
        this.logAlert(alert);

        // Update alert with notifications sent
        alert.notificationsSent = notifications;
        await this.redis.set(`${this.alertPrefix}${alertId}`, '$', alert);

        return alertId;
    }

    /**
     * Send webhook notification
     */
    private async sendWebhook(alert: AlertRecord): Promise<boolean> {
        if (!this.config.webhookUrl) return false;

        try {
            const response = await fetch(this.config.webhookUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    type: 'alert',
                    alert: {
                        id: alert.id,
                        severity: alert.severity,
                        category: alert.category,
                        title: alert.title,
                        message: alert.message,
                        timestamp: alert.timestamp,
                        context: alert.context,
                    },
                }),
                signal: AbortSignal.timeout(5000),
            });

            return response.ok;
        } catch (error) {
            console.error('[AlertManager] Webhook failed:', error);
            return false;
        }
    }

    /**
     * Send Slack notification
     */
    private async sendSlack(alert: AlertRecord): Promise<boolean> {
        if (!this.config.slackWebhook) return false;

        const emoji = {
            info: 'ℹ️',
            warning: '⚠️',
            error: '❌',
            critical: '🚨',
        }[alert.severity];

        const color = {
            info: '#36a64f',
            warning: '#daa038',
            error: '#cc0000',
            critical: '#ff0000',
        }[alert.severity];

        try {
            const response = await fetch(this.config.slackWebhook, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    attachments: [{
                        color,
                        pretext: `${emoji} *${alert.severity.toUpperCase()}* - ${alert.category}`,
                        title: alert.title,
                        text: alert.message,
                        fields: Object.entries(alert.context).slice(0, 5).map(([key, value]) => ({
                            title: key,
                            value: String(value),
                            short: true,
                        })),
                        ts: Math.floor(new Date(alert.timestamp).getTime() / 1000),
                    }],
                }),
                signal: AbortSignal.timeout(5000),
            });

            return response.ok;
        } catch (error) {
            console.error('[AlertManager] Slack notification failed:', error);
            return false;
        }
    }

    /**
     * Log alert to console
     */
    private logAlert(alert: AlertRecord): void {
        const prefix = {
            info: '\x1b[36m[INFO]\x1b[0m',
            warning: '\x1b[33m[WARN]\x1b[0m',
            error: '\x1b[31m[ERROR]\x1b[0m',
            critical: '\x1b[41m[CRITICAL]\x1b[0m',
        }[alert.severity];

        console.log(`${prefix} [${alert.category}] ${alert.title}: ${alert.message}`);
    }

    /**
     * Acknowledge an alert
     */
    async acknowledgeAlert(alertId: string): Promise<void> {
        await this.redis.set(`${this.alertPrefix}${alertId}`, '$.acknowledged', true);
        await this.redis.set(`${this.alertPrefix}${alertId}`, '$.acknowledged_at', new Date().toISOString());
    }

    /**
     * Get recent alerts
     */
    async getRecentAlerts(limit: number = 20): Promise<AlertRecord[]> {
        const keys = await this.redis.keys(`${this.alertPrefix}*`);
        const alerts: AlertRecord[] = [];

        for (const key of keys.slice(-limit)) {
            const alert = await this.redis.get(key) as AlertRecord | null;
            if (alert) alerts.push(alert);
        }

        return alerts.sort((a, b) =>
            new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );
    }

    /**
     * Get unacknowledged alerts
     */
    async getUnacknowledgedAlerts(): Promise<AlertRecord[]> {
        const alerts = await this.getRecentAlerts(100);
        return alerts.filter(a => !a.acknowledged);
    }

    // Convenience methods for common alert types

    async budgetWarning(userId: string, spend: number, limit: number, period: string): Promise<string> {
        return this.sendAlert('warning', 'budget', 'Budget Warning',
            `User ${userId} has reached ${Math.round(spend / limit * 100)}% of ${period} budget`,
            { userId, spend, limit, period, remaining: limit - spend }
        );
    }

    async budgetExceeded(userId: string, spend: number, limit: number, period: string): Promise<string> {
        return this.sendAlert('error', 'budget', 'Budget Exceeded',
            `User ${userId} has exceeded ${period} budget limit`,
            { userId, spend, limit, period, overage: spend - limit }
        );
    }

    async taskError(taskId: string, error: string, context?: Record<string, unknown>): Promise<string> {
        return this.sendAlert('error', 'task', 'Task Execution Error',
            `Task ${taskId} failed: ${error}`,
            { taskId, error, ...context }
        );
    }

    async systemError(component: string, error: string, context?: Record<string, unknown>): Promise<string> {
        return this.sendAlert('error', 'system', `System Error: ${component}`,
            error,
            { component, ...context }
        );
    }

    async securityAlert(type: string, details: string, context?: Record<string, unknown>): Promise<string> {
        return this.sendAlert('critical', 'security', `Security Alert: ${type}`,
            details,
            context ?? {}
        );
    }
}

// Singleton
let alertManagerInstance: AlertManager | null = null;

export function getAlertManager(config?: Partial<AlertConfig>): AlertManager {
    if (!alertManagerInstance) {
        alertManagerInstance = new AlertManager(config);
    }
    return alertManagerInstance;
}
