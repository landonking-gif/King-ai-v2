/**
 * Tool Executor
 * Executes deterministic tools with real implementations
 */

import { type DeterministicTool, type ToolResult, getToolRegistry } from './tool-registry.js';
import * as fs from 'fs/promises';
import * as path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

/** Tool handler function type */
type ToolHandler = (inputs: Record<string, unknown>) => Promise<Record<string, unknown>>;

/** Registered handlers */
const handlers: Map<string, ToolHandler> = new Map();

/** Allowed base directories for file operations (security) */
const ALLOWED_BASE_PATHS = [
    process.cwd(),
    process.env.WORKSPACE_PATH ?? '/workspace',
];

/**
 * Check if a path is safe to access
 */
function isPathSafe(targetPath: string): boolean {
    const resolved = path.resolve(targetPath);
    return ALLOWED_BASE_PATHS.some(base =>
        resolved.startsWith(path.resolve(base))
    );
}

/**
 * Register a tool handler
 */
export function registerHandler(name: string, handler: ToolHandler): void {
    handlers.set(name, handler);
}

/**
 * Execute a tool
 */
export async function executeTool(
    tool: DeterministicTool,
    inputs: Record<string, unknown>
): Promise<ToolResult> {
    const startTime = Date.now();
    const registry = getToolRegistry();

    try {
        // Validate required inputs
        for (const param of tool.inputs) {
            if (param.required && !(param.name in inputs)) {
                throw new Error(`Missing required input: ${param.name}`);
            }
        }

        // Get handler
        const handler = handlers.get(tool.handler);
        if (!handler) {
            throw new Error(`Handler not found: ${tool.handler}`);
        }

        // Execute with timeout
        const result = await Promise.race([
            handler(inputs),
            new Promise<never>((_, reject) =>
                setTimeout(() => reject(new Error('Tool timeout')), tool.timeout_ms)
            ),
        ]);

        const duration = Date.now() - startTime;

        // Update stats
        await registry.updateStats(tool.name, true, duration);

        return {
            success: true,
            output: result,
            duration_ms: duration,
            tool_id: tool.id,
        };
    } catch (error) {
        const duration = Date.now() - startTime;
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';

        // Update stats
        await registry.updateStats(tool.name, false, duration);

        return {
            success: false,
            error: errorMessage,
            duration_ms: duration,
            tool_id: tool.id,
        };
    }
}

// ============ Built-in Handlers ============

/** Git status handler - REAL IMPLEMENTATION */
registerHandler('handleGitStatus', async (inputs) => {
    const repoPath = (inputs['path'] as string) ?? process.cwd();

    if (!isPathSafe(repoPath)) {
        throw new Error('Path is outside allowed directories');
    }

    try {
        // Get current branch
        const { stdout: branchOut } = await execAsync('git branch --show-current', { cwd: repoPath });
        const branch = branchOut.trim();

        // Get status with porcelain format for parsing
        const { stdout: statusOut } = await execAsync('git status --porcelain', { cwd: repoPath });

        const modified: string[] = [];
        const staged: string[] = [];
        const untracked: string[] = [];
        const deleted: string[] = [];

        for (const line of statusOut.split('\n').filter(l => l.trim())) {
            const code = line.substring(0, 2);
            const file = line.substring(3);

            // Index status (first char)
            if (code[0] === 'M' || code[0] === 'A' || code[0] === 'D' || code[0] === 'R') {
                staged.push(file);
            }

            // Working tree status (second char)
            if (code[1] === 'M') {
                modified.push(file);
            } else if (code[1] === 'D') {
                deleted.push(file);
            } else if (code === '??') {
                untracked.push(file);
            }
        }

        // Get ahead/behind info
        let ahead = 0;
        let behind = 0;
        try {
            const { stdout: trackingOut } = await execAsync(
                'git rev-list --left-right --count @{upstream}...HEAD 2>/dev/null || echo "0\t0"',
                { cwd: repoPath }
            );
            const [behindStr, aheadStr] = trackingOut.trim().split('\t');
            behind = parseInt(behindStr ?? '0', 10);
            ahead = parseInt(aheadStr ?? '0', 10);
        } catch {
            // No upstream configured
        }

        return {
            branch,
            modified,
            staged,
            untracked,
            deleted,
            ahead,
            behind,
            clean: modified.length === 0 && staged.length === 0 && untracked.length === 0,
        };
    } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error';
        throw new Error(`Git status failed: ${message}`);
    }
});

/** Git diff handler */
registerHandler('handleGitDiff', async (inputs) => {
    const repoPath = (inputs['path'] as string) ?? process.cwd();
    const file = inputs['file'] as string | undefined;
    const staged = inputs['staged'] as boolean ?? false;

    if (!isPathSafe(repoPath)) {
        throw new Error('Path is outside allowed directories');
    }

    try {
        const stagedFlag = staged ? '--staged' : '';
        const fileArg = file ? `-- "${file}"` : '';
        const { stdout } = await execAsync(
            `git diff ${stagedFlag} ${fileArg}`.trim(),
            { cwd: repoPath, maxBuffer: 1024 * 1024 * 5 } // 5MB buffer
        );

        return {
            diff: stdout,
            has_changes: stdout.trim().length > 0,
        };
    } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error';
        throw new Error(`Git diff failed: ${message}`);
    }
});

/** File read handler - REAL IMPLEMENTATION */
registerHandler('handleFileRead', async (inputs) => {
    const filePath = inputs['path'] as string;
    if (!filePath) {
        throw new Error('File path is required');
    }

    if (!isPathSafe(filePath)) {
        throw new Error('Path is outside allowed directories');
    }

    try {
        const resolvedPath = path.resolve(filePath);
        const stats = await fs.stat(resolvedPath);

        if (!stats.isFile()) {
            throw new Error('Path is not a file');
        }

        // Limit file size to 1MB for safety
        if (stats.size > 1024 * 1024) {
            throw new Error('File too large (max 1MB)');
        }

        const content = await fs.readFile(resolvedPath, 'utf-8');

        return {
            content,
            size: stats.size,
            modified: stats.mtime.toISOString(),
            extension: path.extname(filePath),
        };
    } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error';
        throw new Error(`File read failed: ${message}`);
    }
});

/** File write handler */
registerHandler('handleFileWrite', async (inputs) => {
    const filePath = inputs['path'] as string;
    const content = inputs['content'] as string;
    const createDirs = inputs['create_dirs'] as boolean ?? true;

    if (!filePath) {
        throw new Error('File path is required');
    }
    if (content === undefined) {
        throw new Error('Content is required');
    }

    if (!isPathSafe(filePath)) {
        throw new Error('Path is outside allowed directories');
    }

    try {
        const resolvedPath = path.resolve(filePath);

        if (createDirs) {
            await fs.mkdir(path.dirname(resolvedPath), { recursive: true });
        }

        await fs.writeFile(resolvedPath, content, 'utf-8');
        const stats = await fs.stat(resolvedPath);

        return {
            written: true,
            path: resolvedPath,
            size: stats.size,
        };
    } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error';
        throw new Error(`File write failed: ${message}`);
    }
});

/** List directory handler */
registerHandler('handleListDir', async (inputs) => {
    const dirPath = inputs['path'] as string;
    const recursive = inputs['recursive'] as boolean ?? false;
    const pattern = inputs['pattern'] as string | undefined;

    if (!dirPath) {
        throw new Error('Directory path is required');
    }

    if (!isPathSafe(dirPath)) {
        throw new Error('Path is outside allowed directories');
    }

    try {
        const resolvedPath = path.resolve(dirPath);

        async function listRecursive(dir: string, depth = 0): Promise<Array<{
            name: string;
            path: string;
            type: 'file' | 'directory';
            size?: number;
        }>> {
            const entries = await fs.readdir(dir, { withFileTypes: true });
            const results: Array<{
                name: string;
                path: string;
                type: 'file' | 'directory';
                size?: number;
            }> = [];

            for (const entry of entries) {
                // Skip hidden files and node_modules
                if (entry.name.startsWith('.') || entry.name === 'node_modules') {
                    continue;
                }

                const entryPath = path.join(dir, entry.name);
                const relativePath = path.relative(resolvedPath, entryPath);

                if (pattern && !entry.name.match(new RegExp(pattern))) {
                    continue;
                }

                if (entry.isDirectory()) {
                    results.push({
                        name: entry.name,
                        path: relativePath,
                        type: 'directory',
                    });

                    if (recursive && depth < 5) { // Max depth 5
                        const subEntries = await listRecursive(entryPath, depth + 1);
                        results.push(...subEntries);
                    }
                } else if (entry.isFile()) {
                    const stats = await fs.stat(entryPath);
                    results.push({
                        name: entry.name,
                        path: relativePath,
                        type: 'file',
                        size: stats.size,
                    });
                }
            }

            return results;
        }

        const entries = await listRecursive(resolvedPath);

        return {
            directory: resolvedPath,
            entries,
            count: entries.length,
        };
    } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error';
        throw new Error(`List directory failed: ${message}`);
    }
});

/** Run tests handler - REAL IMPLEMENTATION */
registerHandler('handleRunTests', async (inputs) => {
    const projectPath = (inputs['path'] as string) ?? process.cwd();
    const filter = inputs['filter'] as string | undefined;
    const framework = inputs['framework'] as string | undefined;

    if (!isPathSafe(projectPath)) {
        throw new Error('Path is outside allowed directories');
    }

    try {
        // Detect test framework
        let testCommand = 'npm test';

        // Check for common test configurations
        const packageJsonPath = path.join(projectPath, 'package.json');
        try {
            const packageJson = JSON.parse(await fs.readFile(packageJsonPath, 'utf-8')) as {
                scripts?: Record<string, string>;
                devDependencies?: Record<string, string>;
            };

            if (framework) {
                testCommand = framework;
            } else if (packageJson.devDependencies?.vitest) {
                testCommand = 'npx vitest run';
            } else if (packageJson.devDependencies?.jest) {
                testCommand = 'npx jest';
            } else if (packageJson.devDependencies?.mocha) {
                testCommand = 'npx mocha';
            }
        } catch {
            // No package.json, use default
        }

        if (filter) {
            testCommand += ` --grep "${filter}"`;
        }

        const { stdout, stderr } = await execAsync(testCommand, {
            cwd: projectPath,
            timeout: 120000, // 2 minute timeout
            maxBuffer: 1024 * 1024 * 10, // 10MB buffer
        });

        // Basic parsing of results
        const output = stdout + stderr;
        const passed = !output.includes('FAIL') && !output.includes('failed');

        // Try to extract counts
        const passMatch = output.match(/(\d+)\s*(?:tests?\s*)?pass/i);
        const failMatch = output.match(/(\d+)\s*(?:tests?\s*)?fail/i);

        return {
            passed,
            output: output.slice(0, 5000), // Limit output size
            passed_count: passMatch ? parseInt(passMatch[1] ?? '0', 10) : 0,
            failed_count: failMatch ? parseInt(failMatch[1] ?? '0', 10) : 0,
            command: testCommand,
        };
    } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error';
        const stderr = (error as { stderr?: string }).stderr ?? '';

        return {
            passed: false,
            output: stderr.slice(0, 5000),
            error: message,
            passed_count: 0,
            failed_count: 1,
        };
    }
});

/** Lint check handler - REAL IMPLEMENTATION */
registerHandler('handleLintCheck', async (inputs) => {
    const targetPath = inputs['path'] as string;
    const fix = inputs['fix'] as boolean ?? false;

    if (!targetPath) {
        throw new Error('Path is required');
    }

    if (!isPathSafe(targetPath)) {
        throw new Error('Path is outside allowed directories');
    }

    try {
        // Try ESLint first
        let lintCommand = fix ? 'npx eslint --fix' : 'npx eslint';
        lintCommand += ` "${targetPath}" --format json`;

        const { stdout, stderr } = await execAsync(lintCommand, {
            cwd: process.cwd(),
            timeout: 60000, // 1 minute timeout
            maxBuffer: 1024 * 1024 * 5,
        }).catch(async (error) => {
            // ESLint exits with non-zero when there are issues
            return { stdout: (error as { stdout?: string }).stdout ?? '', stderr: (error as { stderr?: string }).stderr ?? '' };
        });

        // Parse ESLint JSON output
        try {
            const results = JSON.parse(stdout) as Array<{
                filePath: string;
                messages: Array<{
                    severity: number;
                    message: string;
                    line: number;
                    column: number;
                    ruleId: string;
                }>;
                errorCount: number;
                warningCount: number;
                fixableErrorCount: number;
                fixableWarningCount: number;
            }>;

            let totalErrors = 0;
            let totalWarnings = 0;
            let totalFixable = 0;
            const issues: Array<{
                file: string;
                line: number;
                column: number;
                severity: 'error' | 'warning';
                message: string;
                rule: string;
            }> = [];

            for (const result of results) {
                totalErrors += result.errorCount;
                totalWarnings += result.warningCount;
                totalFixable += result.fixableErrorCount + result.fixableWarningCount;

                for (const msg of result.messages.slice(0, 50)) { // Limit issues
                    issues.push({
                        file: path.relative(process.cwd(), result.filePath),
                        line: msg.line,
                        column: msg.column,
                        severity: msg.severity === 2 ? 'error' : 'warning',
                        message: msg.message,
                        rule: msg.ruleId ?? 'unknown',
                    });
                }
            }

            return {
                issues,
                errors: totalErrors,
                warnings: totalWarnings,
                fixable: totalFixable,
                clean: totalErrors === 0 && totalWarnings === 0,
            };
        } catch {
            // Couldn't parse as JSON
            return {
                issues: [],
                errors: 0,
                warnings: 0,
                fixable: 0,
                output: (stdout + stderr).slice(0, 2000),
            };
        }
    } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error';
        throw new Error(`Lint check failed: ${message}`);
    }
});

/** Execute shell command handler (sandboxed) */
registerHandler('handleShellCommand', async (inputs) => {
    const command = inputs['command'] as string;
    const cwd = (inputs['cwd'] as string) ?? process.cwd();
    const timeoutMs = (inputs['timeout'] as number) ?? 30000;

    if (!command) {
        throw new Error('Command is required');
    }

    // Block dangerous commands
    const blockedPatterns = [
        /\brm\s+-rf?\s+[\/~]/i,
        /\bsudo\b/i,
        /\bchmod\b.*777/i,
        /\bcurl\b.*\|\s*sh/i,
        /\bwget\b.*\|\s*sh/i,
        /\beval\b/i,
        />>\s*\/etc\//i,
        /\bdd\b.*if=/i,
    ];

    for (const pattern of blockedPatterns) {
        if (pattern.test(command)) {
            throw new Error('Command contains blocked pattern');
        }
    }

    if (!isPathSafe(cwd)) {
        throw new Error('Working directory is outside allowed paths');
    }

    try {
        const { stdout, stderr } = await execAsync(command, {
            cwd,
            timeout: Math.min(timeoutMs, 60000), // Max 1 minute
            maxBuffer: 1024 * 1024 * 5,
        });

        return {
            stdout: stdout.slice(0, 10000),
            stderr: stderr.slice(0, 2000),
            success: true,
        };
    } catch (error) {
        const exitCode = (error as { code?: number }).code;
        const stderr = (error as { stderr?: string }).stderr ?? '';

        return {
            stdout: '',
            stderr: stderr.slice(0, 2000),
            success: false,
            exit_code: exitCode,
            error: error instanceof Error ? error.message : 'Command failed',
        };
    }
});
