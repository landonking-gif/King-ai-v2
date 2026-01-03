/**
 * ToolExecutor
 * Convenience wrapper around tool registry/execution.
 */

import { executeTool } from './executor.js';
import { getToolRegistry } from './tool-registry.js';

export interface ToolExecutionRequest {
    tool: string;
    parameters?: Record<string, unknown>;
}

export interface ToolExecutionResponse {
    success: boolean;
    result?: unknown;
    error?: string;
}

export class ToolExecutor {
    async executeTool(request: ToolExecutionRequest): Promise<ToolExecutionResponse> {
        const registry = getToolRegistry();
        const tool = await registry.getTool(request.tool);

        if (!tool) {
            return {
                success: false,
                error: `Tool not found: ${request.tool}`,
            };
        }

        const result = await executeTool(tool, request.parameters ?? {});
        return {
            success: result.success,
            ...(result.output !== undefined && { result: result.output }),
            ...(result.error !== undefined && { error: result.error }),
        };
    }
}

let executorInstance: ToolExecutor | null = null;

export function getToolExecutor(): ToolExecutor {
    if (!executorInstance) {
        executorInstance = new ToolExecutor();
    }
    return executorInstance;
}
