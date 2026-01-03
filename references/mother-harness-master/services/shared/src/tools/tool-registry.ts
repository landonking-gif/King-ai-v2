/**
 * Tool Registry
 * Manages deterministic tools that execute before LLM calls
 */

import { getRedisJSON } from '../redis/index.js';
import { nanoid } from 'nanoid';

/** Tool input/output parameter definition */
export interface ToolParameter {
    name: string;
    type: 'string' | 'number' | 'boolean' | 'object' | 'array';
    required: boolean;
    description: string;
    default?: unknown;
}

/** Deterministic tool definition */
export interface DeterministicTool {
    id: string;
    name: string;
    description: string;
    version: string;

    // Routing
    patterns: string[];  // Regex patterns that trigger this tool
    keywords: string[];  // Keywords that suggest this tool
    priority: number;    // Higher = checked first

    // Interface
    inputs: ToolParameter[];
    outputs: ToolParameter[];

    // Execution
    handler: string;     // Handler function name
    timeout_ms: number;

    // Stats
    success_count: number;
    failure_count: number;
    avg_duration_ms: number;

    // Metadata
    enabled: boolean;
    created_at: string;
    updated_at: string;
}

/** Tool execution result */
export interface ToolResult {
    success: boolean;
    output?: unknown;
    error?: string;
    duration_ms: number;
    tool_id: string;
}

/** Built-in tools */
export const BUILTIN_TOOLS: Partial<DeterministicTool>[] = [
    {
        name: 'git_status',
        description: 'Get current git repository status',
        patterns: ['git\\s+status', 'repo\\s+status', 'what.*changed'],
        keywords: ['git', 'status', 'changes', 'modified'],
        priority: 100,
        inputs: [
            { name: 'path', type: 'string', required: false, description: 'Repository path' },
        ],
        outputs: [
            { name: 'branch', type: 'string', required: true, description: 'Current branch' },
            { name: 'modified', type: 'array', required: true, description: 'Modified files' },
            { name: 'staged', type: 'array', required: true, description: 'Staged files' },
        ],
        handler: 'handleGitStatus',
        timeout_ms: 10000,
    },
    {
        name: 'file_read',
        description: 'Read contents of a file',
        patterns: ['read\\s+file', 'show\\s+me\\s+.*file', 'contents?\\s+of'],
        keywords: ['read', 'file', 'contents', 'show'],
        priority: 90,
        inputs: [
            { name: 'path', type: 'string', required: true, description: 'File path to read' },
        ],
        outputs: [
            { name: 'content', type: 'string', required: true, description: 'File contents' },
            { name: 'size', type: 'number', required: true, description: 'File size in bytes' },
        ],
        handler: 'handleFileRead',
        timeout_ms: 5000,
    },
    {
        name: 'run_tests',
        description: 'Run test suite in a project',
        patterns: ['run\\s+tests?', 'execute\\s+tests?', 'test\\s+(suite|runner)'],
        keywords: ['test', 'tests', 'testing', 'jest', 'vitest', 'pytest'],
        priority: 80,
        inputs: [
            { name: 'path', type: 'string', required: false, description: 'Project path' },
            { name: 'filter', type: 'string', required: false, description: 'Test filter pattern' },
        ],
        outputs: [
            { name: 'passed', type: 'boolean', required: true, description: 'All tests passed' },
            { name: 'total', type: 'number', required: true, description: 'Total tests' },
            { name: 'failures', type: 'array', required: true, description: 'Failed tests' },
        ],
        handler: 'handleRunTests',
        timeout_ms: 300000,
    },
    {
        name: 'lint_check',
        description: 'Run linter on code',
        patterns: ['lint', 'check\\s+style', 'eslint', 'prettier'],
        keywords: ['lint', 'linting', 'eslint', 'prettier', 'style'],
        priority: 70,
        inputs: [
            { name: 'path', type: 'string', required: true, description: 'Path to lint' },
        ],
        outputs: [
            { name: 'issues', type: 'array', required: true, description: 'Lint issues' },
            { name: 'fixable', type: 'number', required: true, description: 'Auto-fixable count' },
        ],
        handler: 'handleLintCheck',
        timeout_ms: 60000,
    },
];

export class ToolRegistry {
    private redis = getRedisJSON();
    private readonly keyPrefix = 'tool:';

    /**
     * Initialize registry with built-in tools
     */
    async initialize(): Promise<void> {
        for (const tool of BUILTIN_TOOLS) {
            const exists = await this.redis.exists(`${this.keyPrefix}${tool.name}`);
            if (!exists) {
                await this.registerTool(tool as DeterministicTool);
            }
        }
        console.log('[ToolRegistry] Initialized with built-in tools');
    }

    /**
     * Register a new tool
     */
    async registerTool(tool: Partial<DeterministicTool>): Promise<DeterministicTool> {
        const now = new Date().toISOString();
        const fullTool: DeterministicTool = {
            id: tool.id ?? `tool-${nanoid()}`,
            name: tool.name ?? 'unnamed',
            description: tool.description ?? '',
            version: tool.version ?? '1.0.0',
            patterns: tool.patterns ?? [],
            keywords: tool.keywords ?? [],
            priority: tool.priority ?? 50,
            inputs: tool.inputs ?? [],
            outputs: tool.outputs ?? [],
            handler: tool.handler ?? '',
            timeout_ms: tool.timeout_ms ?? 30000,
            success_count: 0,
            failure_count: 0,
            avg_duration_ms: 0,
            enabled: tool.enabled ?? true,
            created_at: now,
            updated_at: now,
        };

        await this.redis.set(`${this.keyPrefix}${fullTool.name}`, '$', fullTool);
        return fullTool;
    }

    /**
     * Get tool by name
     */
    async getTool(name: string): Promise<DeterministicTool | null> {
        return await this.redis.get<DeterministicTool>(`${this.keyPrefix}${name}`);
    }

    /**
     * Get all enabled tools sorted by priority
     */
    async getEnabledTools(): Promise<DeterministicTool[]> {
        const keys = await this.redis.keys(`${this.keyPrefix}*`);
        const tools: DeterministicTool[] = [];

        for (const key of keys) {
            const tool = await this.redis.get<DeterministicTool>(key);
            if (tool?.enabled) {
                tools.push(tool);
            }
        }

        return tools.sort((a, b) => b.priority - a.priority);
    }

    /**
     * Find matching tool for a query
     */
    async findMatchingTool(query: string): Promise<DeterministicTool | null> {
        const tools = await this.getEnabledTools();
        const lowerQuery = query.toLowerCase();

        for (const tool of tools) {
            // Check patterns
            for (const pattern of tool.patterns) {
                if (new RegExp(pattern, 'i').test(query)) {
                    return tool;
                }
            }

            // Check keywords
            const keywordMatches = tool.keywords.filter(kw =>
                lowerQuery.includes(kw.toLowerCase())
            );
            if (keywordMatches.length >= 2) {
                return tool;
            }
        }

        return null;
    }

    /**
     * Update tool statistics
     */
    async updateStats(
        toolName: string,
        success: boolean,
        duration_ms: number
    ): Promise<void> {
        const tool = await this.getTool(toolName);
        if (!tool) return;

        if (success) {
            tool.success_count++;
        } else {
            tool.failure_count++;
        }

        // Update rolling average
        const totalExecutions = tool.success_count + tool.failure_count;
        tool.avg_duration_ms = (tool.avg_duration_ms * (totalExecutions - 1) + duration_ms) / totalExecutions;
        tool.updated_at = new Date().toISOString();

        await this.redis.set(`${this.keyPrefix}${toolName}`, '$', tool);
    }

    /**
     * Get tool success rate
     */
    async getSuccessRate(toolName: string): Promise<number> {
        const tool = await this.getTool(toolName);
        if (!tool) return 0;

        const total = tool.success_count + tool.failure_count;
        if (total === 0) return 1; // No executions = 100% (no failures)

        return tool.success_count / total;
    }
}

// Singleton
let registryInstance: ToolRegistry | null = null;

export function getToolRegistry(): ToolRegistry {
    if (!registryInstance) {
        registryInstance = new ToolRegistry();
    }
    return registryInstance;
}
