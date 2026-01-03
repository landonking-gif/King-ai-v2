/**
 * N8N Adapter
 * Interfaces with n8n workflows for agent execution
 */

import { config } from './config.js';

/** Workflow execution result */
export interface WorkflowResult {
    success: boolean;
    data?: unknown;
    error?: WorkflowError;
    execution_id?: string;
    duration_ms: number;
}

export type WorkflowErrorType = 'http' | 'timeout' | 'execution' | 'unknown';

export interface WorkflowError {
    type: WorkflowErrorType;
    message: string;
    status_code?: number;
    details?: unknown;
}

/** Workflow trigger options */
export interface WorkflowOptions {
    timeout?: number;
    retries?: number;
    fallback_to_direct?: boolean;
    poll_interval_ms?: number;
}

export class N8nAdapter {
    private readonly baseUrl: string;
    private readonly apiKey: string;
    private readonly defaultTimeout = 300000; // 5 minutes
    private readonly defaultRetries = 2;
    private readonly defaultPollInterval = 5000;

    constructor() {
        this.baseUrl = config.n8nUrl;
        this.apiKey = config.n8nApiKey;
    }

    /**
     * Trigger an n8n workflow
     */
    async triggerWorkflow(
        workflowName: string,
        data: Record<string, unknown>,
        options: WorkflowOptions = {}
    ): Promise<WorkflowResult> {
        const startTime = Date.now();
        const timeout = options.timeout ?? this.defaultTimeout;
        const maxRetries = options.retries ?? (options.fallback_to_direct ? 0 : this.defaultRetries);
        const pollInterval = options.poll_interval_ms ?? this.defaultPollInterval;

        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                const triggerResult = await Promise.race([
                    this.executeWorkflow(workflowName, data),
                    this.timeoutPromise(timeout),
                ]);
                const executionId = this.extractExecutionId(triggerResult);
                const elapsedMs = Date.now() - startTime;
                const remainingTimeout = Math.max(timeout - elapsedMs, 0);
                const workflowData = executionId
                    ? await this.pollExecution(executionId, remainingTimeout, pollInterval)
                    : triggerResult;

                return {
                    success: true,
                    data: workflowData,
                    duration_ms: Date.now() - startTime,
                    ...(executionId && { execution_id: executionId }),
                };
            } catch (error) {
                const workflowError = this.normalizeError(error);

                if (attempt === maxRetries) {
                    // Final failure
                    console.error(
                        `[N8nAdapter] Workflow ${workflowName} failed after ${attempt + 1} attempts:`,
                        workflowError.message
                    );

                    if (options.fallback_to_direct) {
                        console.log(`[N8nAdapter] Falling back to direct execution`);
                        return {
                            success: false,
                            error: this.createError('unknown', 'Unknown error'),
                            duration_ms: Date.now() - startTime,
                        };
                    }

                    throw workflowError;
                }

                // Retry with exponential backoff
                const backoff = 5000 * Math.pow(2, attempt);
                console.log(`[N8nAdapter] Retry ${attempt + 1}/${maxRetries} after ${backoff}ms`);
                await this.sleep(backoff);
            }
        }

        // Should never reach here
        return {
            success: false,
            error: this.createError('unknown', 'Unknown error'),
            duration_ms: Date.now() - startTime,
        };
    }

    /**
     * Execute workflow via webhook
     */
    private async executeWorkflow(
        workflowName: string,
        data: Record<string, unknown>
    ): Promise<unknown> {
        const webhookUrl = `${this.baseUrl}/webhook/${workflowName}`;

        const response = await fetch(webhookUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(this.apiKey && { 'Authorization': `Bearer ${this.apiKey}` }),
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw this.createError('http', `Workflow failed: ${response.status} - ${errorText}`, response.status);
        }

        return response.json();
    }

    /**
     * Create a timeout promise
     */
    private timeoutPromise(ms: number): Promise<never> {
        return new Promise((_, reject) => {
            setTimeout(() => reject(this.createError('timeout', `Workflow timeout after ${ms}ms`)), ms);
        });
    }

    /**
     * Sleep helper
     */
    private sleep(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Get workflow execution status
     */
    async getExecutionStatus(executionId: string): Promise<{
        status: 'running' | 'success' | 'error';
        data?: unknown;
        error?: WorkflowError;
    }> {
        const url = `${this.baseUrl}/executions/${executionId}`;

        const response = await fetch(url, {
            headers: {
                ...(this.apiKey && { 'Authorization': `Bearer ${this.apiKey}` }),
            },
        });

        if (!response.ok) {
            throw this.createError('http', `Failed to get execution status: ${response.status}`, response.status);
        }

        const result = await response.json() as {
            finished?: boolean;
            data?: unknown;
            stoppedAt?: string;
            status?: string;
            error?: unknown;
        };

        if (result.status === 'error' || result.error) {
            return {
                status: 'error',
                error: this.createError('execution', 'Execution reported error state', undefined, result.error),
            };
        }

        if (!result.finished && result.status !== 'success') {
            return { status: 'running' };
        }

        if (result.stoppedAt) {
            return {
                status: 'error',
                error: this.createError('execution', 'Execution stopped', undefined, result),
            };
        }

        return {
            status: 'success',
            data: result.data,
        };
    }

    /**
     * List available workflows
     */
    async listWorkflows(): Promise<Array<{ id: string; name: string; active: boolean }>> {
        const url = `${this.baseUrl}/workflows`;

        const response = await fetch(url, {
            headers: {
                ...(this.apiKey && { 'Authorization': `Bearer ${this.apiKey}` }),
            },
        });

        if (!response.ok) {
            throw this.createError('http', `Failed to list workflows: ${response.status}`, response.status);
        }

        const result = await response.json() as { data: Array<{ id: string; name: string; active: boolean }> };
        return result.data;
    }

    private extractExecutionId(result: unknown): string | null {
        if (result && typeof result === 'object') {
            const payload = result as { executionId?: string; execution_id?: string; id?: string };
            return payload.executionId ?? payload.execution_id ?? payload.id ?? null;
        }
        return null;
    }

    private async pollExecution(
        executionId: string,
        timeoutMs: number,
        pollIntervalMs: number
    ): Promise<unknown> {
        const startTime = Date.now();
        while (Date.now() - startTime < timeoutMs) {
            const status = await this.getExecutionStatus(executionId);

            if (status.status === 'success') {
                return status.data;
            }

            if (status.status === 'error') {
                throw status.error ?? this.createError('execution', 'Execution failed without details');
            }

            await this.sleep(pollIntervalMs);
        }

        throw this.createError('timeout', `Workflow timeout after ${timeoutMs}ms`);
    }

    private normalizeError(error: unknown): WorkflowError {
        if (this.isWorkflowError(error)) {
            return error;
        }

        if (error instanceof Error) {
            return this.createError('unknown', error.message);
        }

        return this.createError('unknown', 'Unknown error', undefined, error);
    }

    private createError(
        type: WorkflowErrorType,
        message: string,
        statusCode?: number,
        details?: unknown
    ): WorkflowError {
        return {
            type,
            message,
            ...(statusCode !== undefined && { status_code: statusCode }),
            ...(details !== undefined && { details }),
        };
    }

    private isWorkflowError(error: unknown): error is WorkflowError {
        return Boolean(
            error
            && typeof error === 'object'
            && 'type' in error
            && 'message' in error
        );
    }
}

// Singleton
let adapterInstance: N8nAdapter | null = null;

export function getN8nAdapter(): N8nAdapter {
    if (!adapterInstance) {
        adapterInstance = new N8nAdapter();
    }
    return adapterInstance;
}
