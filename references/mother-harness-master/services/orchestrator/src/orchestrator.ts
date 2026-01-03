/**
 * Mother Orchestrator
 * Main orchestration class for task planning, routing, and execution
 */

import { nanoid } from 'nanoid';
import {
    type Task,
    type Project,
    type Library,
    type Approval,
    type TodoItem,
    type AgentType,
    type Run,
    type TerminationReason,
    type TerminationRecord,
    createTask,
    createProject,
    createLibrary,
    getRedisJSON,
    getResourceBudgetGuard,
    getContractEnforcer,
    getRoleRegistry,
    logAuditEvent,
    TaskSchema,
    ArtifactSchema,
    ProjectSchema,
    ApprovalSchema,
    RunSchema,
    LibrarySchema,
    type AgentRequest,
} from '@mother-harness/shared';
import { createAgent, type AgentContext as AgentExecutionContext } from '@mother-harness/agents';
import { TaskPlanner } from './planner.js';
import { N8nAdapter, type WorkflowResult } from './n8n-adapter.js';
import { Tier1Memory } from './memory/tier1-recent.js';
import { Tier2Memory } from './memory/tier2-summaries.js';
import { Tier3Memory } from './memory/tier3-longterm.js';
import { listRunActivityEvents, logActivity } from './activity-stream.js';
import { getCostTracker } from './cost-tracker.js';
import { getApprovalService } from './approval-service.js';


/** Agent dispatch result */
interface AgentResult {
    success: boolean;
    outputs: Record<string, unknown>;
    artifacts?: string[];
    explanation?: string;
    tokens_used?: number;
    duration_ms?: number;
    model_used?: string;
}

/** Agent executor interface */
type AgentExecutor = (inputs: string, context: Record<string, unknown>) => Promise<AgentResult>;

/** Registered agent executors */
const agentExecutors: Map<AgentType, AgentExecutor> = new Map();
const localAgentExecutors: Map<AgentType, AgentExecutor> = new Map();

function normalizeAgentOutputs(result: unknown): Record<string, unknown> {
    if (typeof result === 'object' && result !== null) {
        return result as Record<string, unknown>;
    }

    return { result };
}

function agentResultHasError(result: Record<string, unknown>): boolean {
    const errorValue = result['error'];
    return typeof errorValue === 'string' && errorValue.length > 0;
}

function getLocalAgentExecutor(agentType: AgentType): AgentExecutor {
    const cached = localAgentExecutors.get(agentType);
    if (cached) return cached;

    const executor: AgentExecutor = async (inputs, context) => {
        const agent = createAgent(agentType);
        const agentContext: AgentExecutionContext = {
            task_id: String(context.task_id ?? ''),
            step_id: String(context.step_id ?? ''),
            project_id: String(context.project_id ?? ''),
            user_id: String(context.user_id ?? ''),
            ...(typeof context.recent_context === 'string' && { recent_context: context.recent_context }),
            ...(typeof context.rag_context === 'string' && { rag_context: context.rag_context }),
            ...(Array.isArray(context.library_ids) ? {
                library_ids: context.library_ids.filter((id): id is string => typeof id === 'string')
            } : {}),
        };
        const request: AgentRequest = {
            task_id: agentContext.task_id,
            step_id: agentContext.step_id,
            agent: agentType,
            inputs,
            context,
        };
        const response = await agent.execute(request, agentContext);
        const outputs = normalizeAgentOutputs(response.result);

        return {
            success: !agentResultHasError(outputs),
            outputs,
            ...(response.artifacts && { artifacts: response.artifacts }),
            ...(response.explanation && { explanation: response.explanation }),
            ...(response.tokens_used !== undefined && { tokens_used: response.tokens_used }),
            ...(response.duration_ms !== undefined && { duration_ms: response.duration_ms }),
            ...(response.model_used && { model_used: response.model_used }),
        };
    };

    localAgentExecutors.set(agentType, executor);
    return executor;
}

/**
 * Register an agent executor
 */
export function registerAgentExecutor(agentType: AgentType, executor: AgentExecutor): void {
    agentExecutors.set(agentType, executor);
}

export function clearAgentExecutors(): void {
    agentExecutors.clear();
}

export function hasAgentExecutor(agentType: AgentType): boolean {
    return agentExecutors.has(agentType);
}

export class Orchestrator {
    private redis = getRedisJSON();
    private planner = new TaskPlanner();
    private n8nAdapter = new N8nAdapter();
    private tier1 = new Tier1Memory();
    private tier2 = new Tier2Memory();
    private tier3 = new Tier3Memory();
    private enforcer = getContractEnforcer();
    private registry = getRoleRegistry();
    private budgetGuard = getResourceBudgetGuard();
    private costTracker = getCostTracker();

    private getModelUsed(result: AgentResult): string | undefined {
        return typeof result.model_used === 'string' && result.model_used.length > 0
            ? result.model_used
            : undefined;
    }

    /**
     * Create a run with a new task
     */
    async createRun(
        userId: string,
        query: string,
        projectId?: string
    ): Promise<{ run: Run; task: Task }> {
        const task = await this.createTask(userId, query, projectId);
        const runId = `run-${nanoid()}`;
        const now = new Date().toISOString();

        const run: Run = {
            id: runId,
            task_id: task.id,
            project_id: task.project_id,
            user_id: task.user_id,
            status: 'created',
            created_at: now,
            updated_at: now,
        };

        await this.redis.set(`run:${runId}`, '$', run);
        await logActivity({
            type: 'run_created',
            run_id: runId,
            task_id: task.id,
            project_id: task.project_id,
            user_id: task.user_id,
            details: { query },
        });

        return { run, task };
    }

    /**
     * Get run by ID
     */
    async getRun(runId: string): Promise<Run | null> {
        const run = await this.redis.get(`run:${runId}`) as Run | null;
        if (!run) return null;
        const result = RunSchema.safeParse(run);
        if (!result.success) {
            console.error('Invalid run data:', result.error);
            return null;
        }
        return result.data as unknown as Run;
    }

    /**
     * List runs for a user
     */
    async listRuns(userId: string): Promise<Run[]> {
        const keys = await this.redis.keys('run:*');
        const runs: Run[] = [];

        for (const key of keys) {
            const run = await this.redis.get(key) as Run | null;
            if (run?.user_id !== userId) continue;
            const result = RunSchema.safeParse(run);
            if (result.success) {
                runs.push(result.data as unknown as Run);
            }
        }

        return runs.sort((a, b) => b.created_at.localeCompare(a.created_at));
    }

    /**
     * List document libraries
     */
    async listLibraries(search?: string): Promise<Library[]> {
        const keys = await this.redis.keys('library:*');
        const libraries: Library[] = [];
        const query = search?.trim().toLowerCase();

        for (const key of keys) {
            const library = await this.redis.get(key) as Library | null;
            if (!library) continue;
            const result = LibrarySchema.safeParse(library);
            if (!result.success) {
                console.error('Invalid library data:', result.error);
                continue;
            }

            if (query) {
                const haystack = [
                    result.data.name,
                    result.data.id,
                    result.data.folder_path,
                    result.data.description ?? '',
                ].join(' ').toLowerCase();
                if (!haystack.includes(query)) continue;
            }

            libraries.push(result.data as unknown as Library);
        }

        return libraries.sort((a, b) => a.name.localeCompare(b.name));
    }

    /**
     * Create a new document library
     */
    async createLibrary(
        name: string,
        folderPath: string,
        description?: string,
        autoScan: boolean = true
    ): Promise<Library> {
        const libraryId = `lib-${nanoid(10)}`;
        const library = createLibrary(libraryId, name, folderPath, autoScan);
        if (description) {
            library.description = description;
        }

        await this.redis.set(`library:${libraryId}`, '$', library);
        return library;
    }

    /**
     * Trigger a rescan for a library
     */
    async rescanLibrary(libraryId: string): Promise<Library | null> {
        const library = await this.redis.get(`library:${libraryId}`) as Library | null;
        if (!library) return null;

        const now = new Date().toISOString();
        const updated = {
            ...library,
            scan_status: 'scanning' as const,
            last_scanned: now,
            updated_at: now,
            processed_count: 0,
        };

        await this.redis.set(`library:${libraryId}`, '$', updated);
        return updated;
    }

    /**
     * Get artifacts for a run
     */
    async getRunArtifacts(runId: string): Promise<Task['artifacts'] | null> {
        const run = await this.getRun(runId);
        if (!run) return null;
        const task = await this.getTask(run.task_id);
        if (!task) return null;
        return task.artifacts;
    }

    /**
     * Fetch a replay/timeline view for a run.
     * Read-only: does not mutate run/task state.
     */
    async getRunReplay(
        runId: string,
        options: { limit?: number; direction?: 'forward' | 'backward' } = {}
    ): Promise<{
        run: Run;
        task: Task;
        events: Awaited<ReturnType<typeof listRunActivityEvents>>;
    } | null> {
        const run = await this.getRun(runId);
        if (!run) return null;
        const task = await this.getTask(run.task_id);
        if (!task) return null;
        const events = await listRunActivityEvents(runId, options);
        return { run, task, events };
    }

    /**
     * Get a specific artifact by ID
     */
    async getArtifact(artifactId: string): Promise<Task['artifacts'][number] | null> {
        const artifact = await this.redis.get(`artifact:${artifactId}`) as Task['artifacts'][number] | null;
        if (!artifact) return null;
        const result = ArtifactSchema.safeParse(artifact);
        if (!result.success) {
            console.error('Invalid artifact data:', result.error);
            return null;
        }
        return result.data;
    }

    /**
     * Create a new task from user query
     */
    async createTask(
        userId: string,
        query: string,
        projectId?: string
    ): Promise<Task> {
        const taskId = `task-${nanoid()}`;

        // Create or get project
        const actualProjectId = projectId ?? await this.getOrCreateDefaultProject(userId);

        // Create task
        const task = createTask(taskId, actualProjectId, userId, query);

        // Add user message to tier 1 memory
        await this.tier1.addUserMessage(actualProjectId, query);

        // Get context from memory for planning
        const recentContext = await this.tier1.getContextString(actualProjectId);
        const summaryContext = await this.tier2.getContextString(actualProjectId, 3);
        const longTermContext = await this.tier3.getContextString(actualProjectId, query);

        // Generate execution plan with context
        const plan = await this.planner.createPlan(query, {
            project_id: actualProjectId,
            user_id: userId,
            recent_context: `${recentContext}\n\n${summaryContext}\n\n${longTermContext}`,
        });

        task.todo_list = plan.steps;
        task.execution_plan = plan;
        task.status = 'planning';

        // Store in Redis
        await this.redis.set(`task:${taskId}`, '$', task);

        // Add task to project threads
        await this.redis.arrAppend(`project:${actualProjectId}`, '$.threads', taskId);

        return task;
    }

    /**
     * Get task by ID
     */
    async getTask(taskId: string): Promise<Task | null> {
        const task = await this.redis.get(`task:${taskId}`) as Task | null;
        if (!task) return null;

        // Validate schema
        const result = TaskSchema.safeParse(task);
        if (!result.success) {
            console.error('Invalid task data:', result.error);
            return null;
        }

        return result.data as unknown as Task;
    }

    /**
     * Execute a run
     */
    async executeRun(runId: string): Promise<void> {
        const run = await this.getRun(runId);
        if (!run) throw new Error(`Run not found: ${runId}`);

        const task = await this.getTask(run.task_id);
        if (!task) throw new Error(`Task not found: ${run.task_id}`);

        const startedAt = run.started_at ?? new Date().toISOString();
        await this.updateRunStatus(runId, 'executing', { started_at: startedAt });
        await logActivity({
            type: 'run_started',
            run_id: runId,
            task_id: task.id,
            project_id: task.project_id,
            user_id: task.user_id,
        });

        const startTime = Date.now();
        let outcome: Awaited<ReturnType<Orchestrator['executeTask']>>;

        try {
            outcome = await this.executeTask(task.id, runId);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            await this.terminateRun(runId, {
                reason: 'agent_error',
                details: errorMessage,
                total_tokens: 0,
                total_duration_ms: Date.now() - startTime,
            });
            return;
        }

        if (outcome.status === 'approval_needed') {
            await this.updateRunStatus(runId, 'waiting_approval');
            await logActivity({
                type: 'run_waiting_approval',
                run_id: runId,
                task_id: task.id,
                project_id: task.project_id,
                user_id: task.user_id,
                details: { step_id: outcome.last_step_id },
            });
            return;
        }

        const totalDuration = outcome.total_duration_ms ?? (Date.now() - startTime);
        await this.terminateRun(runId, {
            reason: outcome.termination_reason,
            details: outcome.termination_details,
            total_tokens: outcome.total_tokens,
            total_duration_ms: totalDuration,
            ...(outcome.last_step_id && { last_step_id: outcome.last_step_id }),
            ...(outcome.last_agent && { last_agent: outcome.last_agent }),
        });
    }

    /**
     * Execute a task
     */
    private async executeTask(
        taskId: string,
        runId: string
    ): Promise<{
        status: 'completed' | 'failed' | 'approval_needed';
        termination_reason: TerminationReason;
        termination_details: string;
        total_tokens: number;
        total_duration_ms: number;
        last_step_id?: string;
        last_agent?: AgentType;
    }> {
        const task = await this.getTask(taskId);
        if (!task) throw new Error(`Task not found: ${taskId}`);

        // Update status to executing
        await this.updateTaskStatus(taskId, 'executing');

        const agentsInvoked: Array<{ agent: AgentType; step_id: string; tokens: number; models: string[]; duration: number }> = [];
        let totalTokens = 0;
        const startTime = Date.now();
        let lastStepId: string | undefined;
        let lastAgent: AgentType | undefined;

        // Execute steps in order
        for (const step of task.todo_list) {
            // Check dependencies
            const depsMet = await this.checkDependencies(task, step);
            if (!depsMet) continue;

            const invocationCheck = await this.budgetGuard.checkAllScopes(
                runId,
                task.user_id,
                'agent_invocations',
                1
            );
            if (!invocationCheck.allowed) {
                const warning = invocationCheck.warning ?? 'Agent invocation budget exhausted';
                await this.updateStepStatus(taskId, step.id, 'failed', warning);
                await this.updateTaskStatus(taskId, 'failed');
                await logActivity({
                    type: 'step_failed',
                    run_id: runId,
                    task_id: task.id,
                    project_id: task.project_id,
                    user_id: task.user_id,
                    details: { step_id: step.id, agent: step.agent, error: warning },
                });
                return {
                    status: 'failed',
                    termination_reason: 'budget_exhausted',
                    termination_details: warning,
                    total_tokens: totalTokens,
                    total_duration_ms: Date.now() - startTime,
                    last_step_id: step.id,
                    last_agent: step.agent,
                };
            }

            // Update step status
            await this.updateStepStatus(taskId, step.id, 'in_progress');
            await logActivity({
                type: 'step_started',
                run_id: runId,
                task_id: task.id,
                project_id: task.project_id,
                user_id: task.user_id,
                details: { step_id: step.id, agent: step.agent },
            });

            try {
                // Check if approval is required
                if (step.require_approval) {
                    await this.createApprovalRequest(task, step, runId);
                    await this.updateTaskStatus(taskId, 'approval_needed');
                    await logActivity({
                        type: 'approval_requested',
                        run_id: runId,
                        task_id: task.id,
                        project_id: task.project_id,
                        user_id: task.user_id,
                        details: { step_id: step.id, agent: step.agent },
                    });
                    return {
                        status: 'approval_needed',
                        termination_reason: 'approval_timeout',
                        termination_details: 'Awaiting approval',
                        total_tokens: totalTokens,
                        total_duration_ms: Date.now() - startTime,
                        last_step_id: step.id,
                        last_agent: step.agent,
                    };
                }

                // Execute the step
                const result = await this.executeStep(task, step);

                // Track agent invocation
                agentsInvoked.push({
                    agent: step.agent,
                    step_id: step.id,
                    tokens: result.tokens_used ?? 0,
                    models: result.model_used ? [result.model_used] : [],
                    duration: result.duration_ms ?? 0,
                });
                totalTokens += result.tokens_used ?? 0;
                lastStepId = step.id;
                lastAgent = step.agent;

                const tokensUsed = result.tokens_used ?? 0;
                await this.budgetGuard.recordUsageAllScopes(runId, task.user_id, 'agent_invocations', 1);
                if (tokensUsed > 0) {
                    const tokenCheck = await this.budgetGuard.checkAllScopes(
                        runId,
                        task.user_id,
                        'tokens',
                        tokensUsed
                    );
                    await this.budgetGuard.recordUsageAllScopes(runId, task.user_id, 'tokens', tokensUsed);

                    if (!tokenCheck.allowed) {
                        const warning = tokenCheck.warning ?? 'Token budget exhausted';
                        await this.updateStepStatus(taskId, step.id, 'failed', warning);
                        await this.updateTaskStatus(taskId, 'failed');
                        await logActivity({
                            type: 'step_failed',
                            run_id: runId,
                            task_id: task.id,
                            project_id: task.project_id,
                            user_id: task.user_id,
                            details: { step_id: step.id, agent: step.agent, error: warning },
                        });
                        return {
                            status: 'failed',
                            termination_reason: 'budget_exhausted',
                            termination_details: warning,
                            total_tokens: totalTokens,
                            total_duration_ms: Date.now() - startTime,
                            last_step_id: step.id,
                            last_agent: step.agent,
                        };
                    }
                }

                const modelUsed = this.getModelUsed(result);
                if (modelUsed && tokensUsed > 0) {
                    await this.costTracker.trackUsage(task.user_id, modelUsed, tokensUsed);
                }

                // Add assistant response to tier 1 memory
                if (result.explanation) {
                    await this.tier1.addAssistantMessage(task.project_id, result.explanation, step.agent);
                }

                // Update step with result
                await this.updateStepResult(taskId, step.id, result);

                // Keep the in-memory copy in sync so dependency checks work.
                if (!task.steps_completed.includes(step.id)) {
                    task.steps_completed.push(step.id);
                }

                // Check if approval is needed based on the result
                const approvalService = getApprovalService();
                const approvalCheck = approvalService.shouldRequireApproval(step, task, result);

                if (approvalCheck.required) {
                    await this.createApprovalRequest(task, step, runId, result);
                    await this.updateTaskStatus(taskId, 'approval_needed');
                    await logActivity({
                        type: 'approval_requested',
                        run_id: runId,
                        task_id: task.id,
                        project_id: task.project_id,
                        user_id: task.user_id,
                        details: {
                            step_id: step.id,
                            agent: step.agent,
                            reason: approvalCheck.reason,
                            risk_level: approvalCheck.assessment.level,
                        },
                    });
                    return {
                        status: 'approval_needed',
                        termination_reason: 'agent_error', // Placeholder, ignored by executeRun when status is approval_needed
                        termination_details: approvalCheck.reason ?? 'Approval required for this step',
                        total_tokens: totalTokens,
                        total_duration_ms: Date.now() - startTime,
                        last_step_id: step.id,
                        last_agent: step.agent,
                    };
                }

                // Mark step as completed
                await this.updateStepStatus(taskId, step.id, 'completed');
                await logActivity({
                    type: 'step_completed',
                    run_id: runId,
                    task_id: task.id,
                    project_id: task.project_id,
                    user_id: task.user_id,
                    details: { step_id: step.id, agent: step.agent },
                });

            } catch (error) {
                const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                await this.updateStepStatus(taskId, step.id, 'failed', errorMessage);
                await this.updateTaskStatus(taskId, 'failed');
                await logActivity({
                    type: 'step_failed',
                    run_id: runId,
                    task_id: task.id,
                    project_id: task.project_id,
                    user_id: task.user_id,
                    details: { step_id: step.id, agent: step.agent, error: errorMessage },
                });
                return {
                    status: 'failed',
                    termination_reason: 'agent_error',
                    termination_details: errorMessage,
                    total_tokens: totalTokens,
                    total_duration_ms: Date.now() - startTime,
                    last_step_id: step.id,
                    last_agent: step.agent,
                };
            }
        }

        // Update token usage
        await this.redis.set(`task:${taskId}`, '$.tokens_used', totalTokens);

        // All steps completed - finalize with memory updates
        await this.finalizeTask(taskId, agentsInvoked);

        return {
            status: 'completed',
            termination_reason: 'completed',
            termination_details: 'Task completed successfully',
            total_tokens: totalTokens,
            total_duration_ms: Date.now() - startTime,
            ...(lastStepId && { last_step_id: lastStepId }),
            ...(lastAgent && { last_agent: lastAgent }),
        };
    }

    /**
     * Execute a single step by dispatching to the appropriate agent
     */
    private async executeStep(task: Task, step: TodoItem): Promise<AgentResult> {
        const contract = await this.registry.getContract(step.agent);

        if (!contract) {
            throw new Error(`Contract not found for agent: ${step.agent}`);
        }

        const allowlistValidation = await this.enforcer.validateAllowlist(
            step.agent,
            contract.default_action
        );

        if (!allowlistValidation.valid) {
            throw new Error(allowlistValidation.errors.join('; '));
        }

        // Build context for agent
        const recentContext = await this.tier1.getContextString(task.project_id);
        const longTermContext = await this.tier3.getContextString(task.project_id, step.description);

        const context = {
            task_id: task.id,
            project_id: task.project_id,
            user_id: task.user_id,
            step_id: step.id,
            recent_context: recentContext,
            rag_context: longTermContext,
            library_ids: [], // Would be populated from project settings
        };

        const workflowResult = await this.n8nAdapter.triggerWorkflow(
            step.agent,
            {
                task_id: task.id,
                step_id: step.id,
                agent: step.agent,
                inputs: step.description,
                context,
            },
            {
                fallback_to_direct: true,
            }
        );

        if (workflowResult.success) {
            const normalized = this.normalizeWorkflowResult(step, workflowResult);
            // If the workflow ran but reported failure in its payload, treat it as a failure
            // so we fall back to direct/local execution.
            if (normalized.success) {
                return normalized;
            }

            workflowResult.success = false;
            workflowResult.error = {
                type: 'execution',
                message: 'Workflow reported success=false',
                details: workflowResult.data,
            };
        }

        // Execute the agent directly
        const directExecutor = agentExecutors.get(step.agent) ?? getLocalAgentExecutor(step.agent);
        const startTime = Date.now();
        const result = await directExecutor(step.description, context);
        result.duration_ms = Date.now() - startTime;
        result.outputs = {
            ...(result.outputs ?? {}),
            workflow_error: workflowResult.error,
        };

        const requiredArtifactsValidation = await this.enforcer.validateRequiredArtifacts(
            step.agent,
            result.artifacts ?? []
        );

        if (!requiredArtifactsValidation.valid) {
            throw new Error(requiredArtifactsValidation.errors.join('; '));
        }

        return result;
    }

    private normalizeWorkflowResult(step: TodoItem, workflowResult: WorkflowResult): AgentResult {
        const outputs: Record<string, unknown> = {
            workflow_data: workflowResult.data,
            workflow_execution_id: workflowResult.execution_id,
        };

        const agentResult: AgentResult = {
            success: true,
            outputs,
            explanation: `Executed step via workflow: ${step.description}`,
            tokens_used: 0,
            duration_ms: workflowResult.duration_ms,
        };

        if (workflowResult.data && typeof workflowResult.data === 'object') {
            const payload = workflowResult.data as {
                outputs?: Record<string, unknown>;
                explanation?: string;
                tokens_used?: number;
                duration_ms?: number;
                success?: boolean;
            };

            if (payload.outputs && typeof payload.outputs === 'object') {
                agentResult.outputs = {
                    ...outputs,
                    ...payload.outputs,
                };
            }

            if (typeof payload.explanation === 'string') {
                agentResult.explanation = payload.explanation;
            }

            if (typeof payload.tokens_used === 'number') {
                agentResult.tokens_used = payload.tokens_used;
            }

            if (typeof payload.duration_ms === 'number') {
                agentResult.duration_ms = payload.duration_ms;
            }

            if (typeof payload.success === 'boolean') {
                agentResult.success = payload.success;
            }
        }

        return agentResult;
    }

    /**
     * Finalize completed task with memory updates
     */
    private async finalizeTask(
        taskId: string,
        agentsInvoked: Array<{ agent: AgentType; step_id: string; tokens: number; models: string[]; duration: number }>
    ): Promise<void> {
        const task = await this.getTask(taskId);
        if (!task) return;

        await this.updateTaskStatus(taskId, 'completed');
        await this.redis.set(`task:${taskId}`, '$.completed_at', new Date().toISOString());

        // Compile results for memory
        const results = task.todo_list
            .filter(s => s.result)
            .map(s => {
                const result = s.result as { explanation?: string; outputs?: Record<string, unknown> };
                return result.explanation || JSON.stringify(result.outputs || {}).substring(0, 200);
            })
            .join('\n');

        // Generate session summary (Tier 2 memory)
        try {
            await this.tier2.createSummary(task.project_id, taskId, {
                goal: task.query,
                outcome: task.status === 'completed' ? 'Successfully completed' : 'Failed',
                agents_invoked: agentsInvoked,

            });
        } catch (error) {
            console.error('[Orchestrator] Failed to create tier 2 summary:', error);
        }

        // Extract and embed important findings (Tier 3 memory)
        try {
            const memoriesStored = await this.tier3.extractAndStore(task.project_id, taskId, results);
            console.log(`[Orchestrator] Stored ${memoriesStored} long-term memories from task ${taskId}`);
        } catch (error) {
            console.error('[Orchestrator] Failed to extract tier 3 memories:', error);
        }
    }

    /**
     * Get or create default project for user
     */
    private async getOrCreateDefaultProject(_userId: string): Promise<string> {
        // Check for existing default project
        const existingKeys = await this.redis.keys('project:*');
        for (const key of existingKeys) {
            const project = await this.redis.get(key) as Project | null;
            if (project?.status === 'active') {
                return project.id;
            }
        }

        // Create new project
        const projectId = `proj-${nanoid()}`;
        const project = createProject(projectId, 'Default Project');
        await this.redis.set(`project:${projectId}`, '$', project);

        return projectId;
    }

    /**
     * Update task status
     */
    private async updateTaskStatus(taskId: string, status: Task['status']): Promise<void> {
        await this.redis.set(`task:${taskId}`, '$.status', status);
        await this.redis.set(`task:${taskId}`, '$.updated_at', new Date().toISOString());
    }

    /**
     * Update step status
     */
    private async updateStepStatus(
        taskId: string,
        stepId: string,
        status: TodoItem['status'],
        error?: string
    ): Promise<void> {
        const task = await this.getTask(taskId);
        if (!task) return;

        const stepIndex = task.todo_list.findIndex(s => s.id === stepId);
        if (stepIndex === -1) return;

        task.todo_list[stepIndex]!.status = status;
        if (status === 'in_progress') {
            task.todo_list[stepIndex]!.started_at = new Date().toISOString();
        }
        if (status === 'completed' || status === 'failed') {
            task.todo_list[stepIndex]!.completed_at = new Date().toISOString();
        }
        if (error) {
            task.todo_list[stepIndex]!.error = error;
        }

        await this.redis.set(`task:${taskId}`, '$.todo_list', task.todo_list);
    }

    /**
     * Update step result
     */
    private async updateStepResult(taskId: string, stepId: string, result: AgentResult): Promise<void> {
        const task = await this.getTask(taskId);
        if (!task) return;

        const stepIndex = task.todo_list.findIndex(s => s.id === stepId);
        if (stepIndex === -1) return;

        task.todo_list[stepIndex]!.result = result;
        await this.redis.set(`task:${taskId}`, '$.todo_list', task.todo_list);

        // Add to completed steps
        await this.redis.arrAppend(`task:${taskId}`, '$.steps_completed', stepId);
    }

    /**
     * Check if step dependencies are met
     */
    private async checkDependencies(task: Task, step: TodoItem): Promise<boolean> {
        if (step.depends_on.length === 0) return true;

        const completedSteps = task.steps_completed;
        return step.depends_on.every(dep => completedSteps.includes(dep));
    }

    /**
     * Create approval request for a step
     */
    private async createApprovalRequest(task: Task, step: TodoItem, runId: string, stepResult?: unknown): Promise<Approval> {
        const approvalService = getApprovalService();
        const approval = approvalService.createApprovalRequest(step, task, runId, stepResult);

        await this.redis.set(`approval:${approval.id}`, '$', approval);
        await logAuditEvent({
            type: 'approval_requested',
            action: approval.type,
            actor: {
                user_id: task.user_id,
                roles: ['user'],
            },
            resource: {
                type: 'approval',
                id: approval.id,
                attributes: {
                    task_id: task.id,
                    step_id: step.id,
                },
            },
            metadata: {
                description: step.description,
                risk_level: approval.risk_level,
            },
            status: 'pending',
        });
        return approval;
    }

    /**
     * Update run status
     */
    private async updateRunStatus(
        runId: string,
        status: Run['status'],
        updates: Partial<Pick<Run, 'started_at' | 'terminated_at' | 'termination_reason' | 'termination_details' | 'total_tokens' | 'total_duration_ms'>> = {}
    ): Promise<void> {
        const now = new Date().toISOString();
        await this.redis.set(`run:${runId}`, '$.status', status);
        await this.redis.set(`run:${runId}`, '$.updated_at', now);

        const entries = Object.entries(updates);
        for (const [key, value] of entries) {
            if (value === undefined) continue;
            await this.redis.set(`run:${runId}`, `$.${key}`, value);
        }
    }

    /**
     * Terminate run with record
     */
    private async terminateRun(
        runId: string,
        payload: {
            reason: TerminationReason;
            details: string;
            total_tokens: number;
            total_duration_ms: number;
            last_step_id?: string;
            last_agent?: AgentType;
        }
    ): Promise<void> {
        const run = await this.getRun(runId);
        if (!run) throw new Error(`Run not found: ${runId}`);

        const task = await this.getTask(run.task_id);
        if (!task) throw new Error(`Task not found: ${run.task_id}`);

        const terminatedAt = new Date().toISOString();
        const termination: TerminationRecord = {
            run_id: runId,
            task_id: run.task_id,
            project_id: run.project_id,
            user_id: run.user_id,
            reason: payload.reason,
            details: payload.details,
            ...(payload.last_step_id && { last_step_id: payload.last_step_id }),
            ...(payload.last_agent && { last_agent: payload.last_agent }),
            total_steps_planned: task.todo_list.length,
            steps_completed: task.steps_completed.length,
            total_tokens: payload.total_tokens,
            total_duration_ms: payload.total_duration_ms,
            started_at: run.started_at ?? run.created_at,
            terminated_at: terminatedAt,
        };

        await this.redis.set(`termination:${runId}`, '$', termination);
        await this.updateRunStatus(runId, 'terminated', {
            terminated_at: terminatedAt,
            termination_reason: payload.reason,
            termination_details: payload.details,
            total_tokens: payload.total_tokens,
            total_duration_ms: payload.total_duration_ms,
        });
        await logActivity({
            type: 'run_terminated',
            run_id: runId,
            task_id: run.task_id,
            project_id: run.project_id,
            user_id: run.user_id,
            details: {
                reason: payload.reason,
                details: payload.details,
                steps_completed: task.steps_completed.length,
            },
        });
    }

    /**
     * List projects for a user
     */
    async listProjects(_userId: string): Promise<Project[]> {
        const keys = await this.redis.keys('project:*');
        const projects: Project[] = [];

        for (const key of keys) {
            const project = await this.redis.get(key) as Project | null;
            if (project) {
                const result = ProjectSchema.safeParse(project);
                if (result.success) {
                    projects.push(result.data as unknown as Project);
                }
            }
        }

        return projects;
    }

    /**
     * Get pending approvals for user
     */
    async getPendingApprovals(userId: string): Promise<Approval[]> {
        const keys = await this.redis.keys('approval:*');
        const approvals: Approval[] = [];

        for (const key of keys) {
            const approval = await this.redis.get(key) as Approval | null;
            if (approval?.status === 'pending' && approval.user_id === userId) {
                const result = ApprovalSchema.safeParse(approval);
                if (result.success) {
                    approvals.push(result.data as unknown as Approval);
                }
            }
        }

        return approvals;
    }

    /**
     * Respond to an approval request
     */
    async respondToApproval(approvalId: string, approved: boolean, notes?: string): Promise<void> {
        const approval = await this.redis.get(`approval:${approvalId}`) as Approval | null;
        if (!approval) throw new Error(`Approval not found: ${approvalId}`);

        await this.redis.set(`approval:${approvalId}`, '$.status', approved ? 'approved' : 'rejected');
        await this.redis.set(`approval:${approvalId}`, '$.responded_at', new Date().toISOString());
        if (notes) {
            await this.redis.set(`approval:${approvalId}`, '$.response_notes', notes);
        }

        // Resume task execution if approved
        if (approved) {
            await logActivity({
                type: 'approval_approved',
                run_id: approval.run_id,
                task_id: approval.task_id,
                project_id: approval.project_id,
                user_id: approval.user_id,
                details: { step_id: approval.step_id },
            });
            await this.updateStepStatus(approval.task_id, approval.step_id, 'completed');
            // Continue task execution in background
            this.executeRun(approval.run_id).catch(console.error);
        } else {
            await this.updateTaskStatus(approval.task_id, 'failed');
            await logActivity({
                type: 'approval_rejected',
                run_id: approval.run_id,
                task_id: approval.task_id,
                project_id: approval.project_id,
                user_id: approval.user_id,
                details: { step_id: approval.step_id, notes },
            });
            await this.terminateRun(approval.run_id, {
                reason: 'approval_rejected',
                details: notes ?? 'Approval rejected by user',
                total_tokens: 0,
                total_duration_ms: 0,
                last_step_id: approval.step_id,
            });
        }
    }

    /**
     * Get memory context for a task
     */
    async getMemoryContext(_taskId: string, projectId: string, query: string): Promise<{
        recent: string;
        summaries: string;
        longTerm: string;
    }> {
        return {
            recent: await this.tier1.getContextString(projectId),
            summaries: await this.tier2.getContextString(projectId),
            longTerm: await this.tier3.getContextString(projectId, query),
        };
    }
}
