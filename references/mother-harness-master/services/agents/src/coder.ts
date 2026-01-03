/**
 * Coder Agent
 * Generates code, creates patches, runs tests
 */

import type { AgentType } from '@mother-harness/shared';
import { getLLMClient } from '@mother-harness/shared';
import { BaseAgent, type AgentContext, type AgentResult } from './base-agent.js';

/** Code change structure */
export interface CodeChange {
    file_path: string;
    change_type: 'create' | 'modify' | 'delete';
    content?: string;
    diff?: string;
    language?: string;
}

/** Analysis result from LLM */
interface CodeAnalysis {
    task_type: 'create' | 'modify' | 'fix' | 'refactor';
    files_involved: string[];
    explanation: string;
    approach: string;
    should_test: boolean;
    docs_needed: boolean;
    tokens: number;
}

const CODER_SYSTEM_PROMPT = `You are an expert software engineer. You write clean, well-documented, production-ready code.
When given a coding task, analyze it carefully and generate appropriate code.
Always follow best practices for the language and framework being used.
Include helpful comments and type annotations where appropriate.`;

const ANALYSIS_PROMPT = `Analyze this coding request and determine the approach:

Request: {request}

Context: {context}

Respond with a JSON object containing:
{
  "task_type": "create" | "modify" | "fix" | "refactor",
  "files_involved": ["array of file paths that will be affected"],
  "explanation": "brief explanation of what needs to be done",
  "approach": "step-by-step approach to implement this",
  "should_test": true/false,
  "docs_needed": true/false
}`;

const CODE_GENERATION_PROMPT = `Generate code for this task:

Task: {task_type} - {explanation}
Approach: {approach}
Files: {files}

Previous context: {context}

For each file, generate the complete code. Return a JSON array:
[
  {
    "file_path": "path/to/file.ts",
    "change_type": "create" | "modify" | "delete",
    "content": "full file content here",
    "language": "typescript"
  }
]

Generate production-ready code with proper error handling and types.`;

export class CoderAgent extends BaseAgent {
    readonly agentType: AgentType = 'coder';
    private llm = getLLMClient();

    protected async run(inputs: string, context: AgentContext): Promise<AgentResult> {
        const startTime = Date.now();
        let totalTokens = 0;

        // Analyze the coding request
        const analysis = await this.analyzeRequest(inputs, context);
        totalTokens += analysis.tokens;

        // Generate code changes
        const codeResult = await this.generateCode(analysis, context);
        totalTokens += codeResult.tokens;

        // Optionally run tests if configured
        const testResults = analysis.should_test
            ? await this.runTests(codeResult.changes, context)
            : undefined;

        // Generate documentation if needed
        const docs = analysis.docs_needed
            ? await this.generateDocs(codeResult.changes, analysis)
            : undefined;

        return {
            success: codeResult.changes.length > 0,
            outputs: {
                code_changes: codeResult.changes,
                explanation: analysis.explanation,
                approach: analysis.approach,
                tests: testResults,
                documentation: docs,
            },
            explanation: `Generated ${codeResult.changes.length} code change(s): ${analysis.explanation}`,
            artifacts: codeResult.changes.map(c => c.file_path),
            tokens_used: totalTokens,
            duration_ms: Date.now() - startTime,
        };
    }

    private async analyzeRequest(
        inputs: string,
        context: AgentContext
    ): Promise<CodeAnalysis> {
        const prompt = ANALYSIS_PROMPT
            .replace('{request}', inputs)
            .replace('{context}', context.recent_context ?? 'No previous context');

        const result = await this.llm.json<CodeAnalysis>(prompt, {
            system: CODER_SYSTEM_PROMPT,
            temperature: 0.3, // Lower temperature for analysis
        });

        if (result.data) {
            return {
                ...result.data,
                tokens: result.raw.tokens_used.total,
            };
        }

        // Fallback if LLM fails
        console.warn('[CoderAgent] Analysis failed, using defaults');
        return {
            task_type: 'create',
            files_involved: [],
            explanation: `Processing request: "${inputs}"`,
            approach: 'Generate code based on the request',
            should_test: false,
            docs_needed: false,
            tokens: result.raw.tokens_used.total,
        };
    }

    private async generateCode(
        analysis: CodeAnalysis,
        context: AgentContext
    ): Promise<{ changes: CodeChange[]; tokens: number }> {
        const prompt = CODE_GENERATION_PROMPT
            .replace('{task_type}', analysis.task_type)
            .replace('{explanation}', analysis.explanation)
            .replace('{approach}', analysis.approach)
            .replace('{files}', analysis.files_involved.join(', ') || 'Determine appropriate file paths')
            .replace('{context}', context.recent_context ?? 'No previous context');

        const result = await this.llm.json<CodeChange[]>(prompt, {
            system: CODER_SYSTEM_PROMPT,
            temperature: 0.2, // Low temperature for code generation
            max_tokens: 8192, // Allow for longer code output
        });

        if (result.data && Array.isArray(result.data)) {
            return {
                changes: result.data,
                tokens: result.raw.tokens_used.total,
            };
        }

        // Fallback
        console.warn('[CoderAgent] Code generation failed');
        return {
            changes: [],
            tokens: result.raw.tokens_used.total,
        };
    }

    private async runTests(
        changes: CodeChange[],
        _context: AgentContext
    ): Promise<{ passed: boolean; output: string }> {
        // Check for testable files
        const testableFiles = changes.filter(c =>
            c.file_path.endsWith('.ts') || c.file_path.endsWith('.js')
        );

        if (testableFiles.length === 0) {
            return {
                passed: true,
                output: 'No testable files in changes',
            };
        }

        try {
            // Use ToolExecutor to run actual tests
            const { getToolExecutor } = await import('@mother-harness/shared');
            const executor = getToolExecutor();

            const result = await executor.executeTool({
                tool: 'run_tests',
                parameters: {
                    path: process.cwd(),
                    testFilter: testableFiles.map(f => f.file_path).join(','),
                },
            });

            if (result.success && result.result) {
                const testResult = result.result as { passed: number; failed: number; output: string };
                return {
                    passed: testResult.failed === 0,
                    output: testResult.output ?? `Tests: ${testResult.passed} passed, ${testResult.failed} failed`,
                };
            }

            return {
                passed: false,
                output: result.error ?? 'Test execution failed',
            };
        } catch (error) {
            // Fallback if tool executor not available
            return {
                passed: false,
                output: `Test execution failed for ${testableFiles.length} file(s). ToolExecutor not available.`,
            };
        }
    }

    private async generateDocs(
        changes: CodeChange[],
        analysis: CodeAnalysis
    ): Promise<string> {
        const docPrompt = `Generate brief documentation for these code changes:

Task: ${analysis.explanation}
Files changed:
${changes.map(c => `- ${c.change_type}: ${c.file_path}`).join('\n')}

Write a concise markdown summary of what was implemented.`;

        const result = await this.llm.complete(docPrompt, {
            system: 'You are a technical writer. Write clear, concise documentation.',
            temperature: 0.4,
            max_tokens: 1024,
        });

        if (result.finish_reason !== 'error') {
            return result.content;
        }

        // Fallback
        return `## Code Changes\n\n${changes.map(c => `- **${c.change_type}**: \`${c.file_path}\``).join('\n')}\n\n${analysis.explanation}`;
    }
}
