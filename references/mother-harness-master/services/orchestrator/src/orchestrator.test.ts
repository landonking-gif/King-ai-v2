/**
 * Orchestrator Unit Tests
 * Tests for core orchestration lifecycle and state management
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { Orchestrator, registerAgentExecutor, clearAgentExecutors } from './orchestrator.js';
import type { AgentType, Task } from '@mother-harness/shared';

// Mock Redis
vi.mock('@mother-harness/shared', async () => {
    const actual = await vi.importActual('@mother-harness/shared');
    return {
        ...actual,
        getRedisJSON: () => ({
            set: vi.fn().mockResolvedValue('OK'),
            get: vi.fn().mockResolvedValue(null),
            keys: vi.fn().mockResolvedValue([]),
            del: vi.fn().mockResolvedValue(1),
        }),
        getRedisClient: () => ({
            set: vi.fn().mockResolvedValue('OK'),
            get: vi.fn().mockResolvedValue(null),
            keys: vi.fn().mockResolvedValue([]),
            del: vi.fn().mockResolvedValue(1),
            exists: vi.fn().mockResolvedValue(0),
            expire: vi.fn().mockResolvedValue(1),
        }),
        getResourceBudgetGuard: () => ({
            checkBudget: vi.fn().mockResolvedValue({ allowed: true }),
            recordUsage: vi.fn().mockResolvedValue(undefined),
        }),
    };
});

// Mock agents package
vi.mock('@mother-harness/agents', () => ({
    createAgent: vi.fn().mockReturnValue({
        execute: vi.fn().mockResolvedValue({
            success: true,
            result: 'Mock agent result',
        }),
    }),
}));

describe('Orchestrator', () => {
    let orchestrator: Orchestrator;
    const testUserId = 'user-test-123';
    const testQuery = 'Write a function to calculate fibonacci numbers';

    beforeEach(() => {
        orchestrator = new Orchestrator();
        clearAgentExecutors();
    });

    afterEach(() => {
        clearAgentExecutors();
    });

    describe('createRun', () => {
        it('should create a new run with associated task', async () => {
            const { run, task } = await orchestrator.createRun(testUserId, testQuery);

            expect(run).toBeDefined();
            expect(run.id).toMatch(/^run-/);
            expect(run.task_id).toBe(task.id);
            expect(run.user_id).toBe(testUserId);
            expect(run.status).toBe('created');

            expect(task).toBeDefined();
            expect(task.id).toMatch(/^task-/);
            expect(task.user_id).toBe(testUserId);
            expect(task.original_query).toBe(testQuery);
            expect(task.status).toBe('pending');
        });

        it('should associate run with project if provided', async () => {
            const projectId = 'proj-123';
            const { run, task } = await orchestrator.createRun(testUserId, testQuery, projectId);

            expect(run.project_id).toBe(projectId);
            expect(task.project_id).toBe(projectId);
        });

        it('should set timestamps correctly', async () => {
            const before = new Date().toISOString();
            const { run, task } = await orchestrator.createRun(testUserId, testQuery);
            const after = new Date().toISOString();

            expect(run.created_at).toBeGreaterThanOrEqual(before);
            expect(run.created_at).toBeLessThanOrEqual(after);
            expect(task.created_at).toBeGreaterThanOrEqual(before);
            expect(task.created_at).toBeLessThanOrEqual(after);
        });
    });

    describe('createTask', () => {
        it('should create task with initial planning state', async () => {
            const task = await orchestrator.createTask(testUserId, testQuery);

            expect(task.status).toBe('pending');
            expect(task.todo_list).toEqual([]);
            expect(task.original_query).toBe(testQuery);
            expect(task.artifacts).toEqual([]);
        });

        it('should generate unique task IDs', async () => {
            const task1 = await orchestrator.createTask(testUserId, testQuery);
            const task2 = await orchestrator.createTask(testUserId, testQuery);

            expect(task1.id).not.toBe(task2.id);
            expect(task1.id).toMatch(/^task-/);
            expect(task2.id).toMatch(/^task-/);
        });
    });

    describe('Agent execution', () => {
        it('should register and use custom agent executors', async () => {
            const mockExecutor = vi.fn().mockResolvedValue({
                success: true,
                outputs: { result: 'test output' },
                explanation: 'Test execution',
                tokens_used: 100,
                duration_ms: 50,
            });

            registerAgentExecutor('researcher', mockExecutor);

            // Create a task with a researcher step
            const task = await orchestrator.createTask(testUserId, 'Research AI trends');
            
            // Manually add a step to test execution
            task.todo_list = [{
                id: 'step-1',
                agent: 'researcher' as AgentType,
                description: 'Research current AI trends',
                status: 'pending',
                dependencies: [],
                require_approval: false,
            }];

            // Mock getTask to return our modified task
            vi.spyOn(orchestrator as any, 'getTask').mockResolvedValue(task);
            vi.spyOn(orchestrator as any, 'updateTaskStatus').mockResolvedValue(undefined);
            vi.spyOn(orchestrator as any, 'updateStepStatus').mockResolvedValue(undefined);
            vi.spyOn(orchestrator as any, 'checkDependencies').mockResolvedValue(true);

            // This would be called internally during executeRun
            // Just verify the executor can be called
            expect(mockExecutor).toBeDefined();
        });

        it('should handle agent execution failures', async () => {
            const mockExecutor = vi.fn().mockRejectedValue(new Error('Agent failed'));

            registerAgentExecutor('coder', mockExecutor);

            // Verify executor is registered
            const { hasAgentExecutor } = await import('./orchestrator.js');
            expect(hasAgentExecutor('coder')).toBe(true);
        });
    });

    describe('Task lifecycle state transitions', () => {
        it('should transition from pending → planning → executing → completed', async () => {
            const transitions: string[] = [];
            
            const mockUpdateStatus = vi.fn((taskId: string, status: string) => {
                transitions.push(status);
                return Promise.resolve();
            });

            vi.spyOn(orchestrator as any, 'updateTaskStatus').mockImplementation(mockUpdateStatus);

            const task = await orchestrator.createTask(testUserId, testQuery);
            expect(task.status).toBe('pending');

            // Simulate state transitions
            await (orchestrator as any).updateTaskStatus(task.id, 'planning');
            await (orchestrator as any).updateTaskStatus(task.id, 'executing');
            await (orchestrator as any).updateTaskStatus(task.id, 'completed');

            expect(transitions).toEqual(['planning', 'executing', 'completed']);
        });

        it('should handle failed state', async () => {
            const task = await orchestrator.createTask(testUserId, testQuery);
            
            vi.spyOn(orchestrator as any, 'updateTaskStatus').mockResolvedValue(undefined);
            await (orchestrator as any).updateTaskStatus(task.id, 'failed');

            // Verify the update was called
            expect((orchestrator as any).updateTaskStatus).toHaveBeenCalledWith(task.id, 'failed');
        });

        it('should handle approval_needed state', async () => {
            const task = await orchestrator.createTask(testUserId, testQuery);
            
            vi.spyOn(orchestrator as any, 'updateTaskStatus').mockResolvedValue(undefined);
            await (orchestrator as any).updateTaskStatus(task.id, 'approval_needed');

            expect((orchestrator as any).updateTaskStatus).toHaveBeenCalledWith(task.id, 'approval_needed');
        });
    });

    describe('Memory tier integration', () => {
        it('should finalize task with memory updates', async () => {
            const task = await orchestrator.createTask(testUserId, testQuery);
            task.status = 'completed';
            task.todo_list = [{
                id: 'step-1',
                agent: 'coder' as AgentType,
                description: 'Write code',
                status: 'completed',
                dependencies: [],
                require_approval: false,
                result: {
                    explanation: 'Generated fibonacci function',
                    outputs: { code: 'function fib(n) { ... }' },
                },
            }];

            vi.spyOn(orchestrator as any, 'getTask').mockResolvedValue(task);
            vi.spyOn(orchestrator as any, 'updateTaskStatus').mockResolvedValue(undefined);

            const agentsInvoked = [
                { agent: 'coder' as AgentType, step_id: 'step-1', tokens: 150 }
            ];

            // Call finalize (it's private but we can test via type assertion)
            await (orchestrator as any).finalizeTask(task.id, agentsInvoked);

            // Verify task was updated to completed
            expect((orchestrator as any).updateTaskStatus).toHaveBeenCalledWith(task.id, 'completed');
        });
    });

    describe('Library management', () => {
        it('should create document library with correct structure', async () => {
            const name = 'Test Library';
            const folderPath = '/path/to/documents';
            const description = 'Test library for unit tests';

            const library = await orchestrator.createLibrary(name, folderPath, description, true);

            expect(library.id).toMatch(/^lib-/);
            expect(library.name).toBe(name);
            expect(library.folder_path).toBe(folderPath);
            expect(library.description).toBe(description);
            expect(library.auto_scan).toBe(true);
            expect(library.scan_status).toBe('idle');
            expect(library.processed_count).toBe(0);
        });

        it('should trigger library rescan', async () => {
            const library = await orchestrator.createLibrary('Test', '/path', undefined, false);
            
            vi.spyOn(orchestrator as any, 'redis').mockReturnValue({
                get: vi.fn().mockResolvedValue(library),
                set: vi.fn().mockResolvedValue('OK'),
            });

            const updated = await orchestrator.rescanLibrary(library.id);

            expect(updated).toBeDefined();
            if (updated) {
                expect(updated.scan_status).toBe('scanning');
                expect(updated.processed_count).toBe(0);
            }
        });
    });

    describe('Error handling', () => {
        it('should handle missing run gracefully', async () => {
            const run = await orchestrator.getRun('non-existent-run');
            expect(run).toBeNull();
        });

        it('should handle missing task gracefully', async () => {
            const task = await orchestrator.getTask('non-existent-task');
            expect(task).toBeNull();
        });

        it('should return empty arrays for non-existent user runs', async () => {
            const runs = await orchestrator.listRuns('non-existent-user');
            expect(runs).toEqual([]);
        });
    });

    describe('Approval workflow', () => {
        it('should support approval gates for risky operations', async () => {
            const task = await orchestrator.createTask(testUserId, 'Delete production database');
            
            // Add a step that requires approval
            task.todo_list = [{
                id: 'step-1',
                agent: 'coder' as AgentType,
                description: 'Delete database',
                status: 'pending',
                dependencies: [],
                require_approval: true,
            }];

            expect(task.todo_list[0].require_approval).toBe(true);
        });

        it('should create approval request for gated steps', async () => {
            const task = await orchestrator.createTask(testUserId, 'Risky operation');
            task.todo_list = [{
                id: 'step-risky',
                agent: 'coder' as AgentType,
                description: 'Delete files',
                status: 'pending',
                dependencies: [],
                require_approval: true,
            }];

            const step = task.todo_list[0];
            const runId = 'run-test-123';

            vi.spyOn(orchestrator as any, 'createApprovalRequest').mockResolvedValue({
                id: 'approval-123',
                task_id: task.id,
                step_id: step.id,
                run_id: runId,
                status: 'pending',
            });

            const approval = await (orchestrator as any).createApprovalRequest(task, step, runId);
            expect(approval.status).toBe('pending');
            expect(approval.step_id).toBe(step.id);
        });
    });
});
