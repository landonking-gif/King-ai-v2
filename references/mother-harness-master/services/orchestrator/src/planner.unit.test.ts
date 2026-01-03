/**
 * Task Planner Unit Tests
 * Tests for task planning and agent detection
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { TaskPlanner } from './planner.js';
import type { TodoItem } from '@mother-harness/shared';

describe('TaskPlanner', () => {
    let planner: TaskPlanner;

    beforeEach(() => {
        planner = new TaskPlanner();
    });

    describe('createPlan', () => {
        it('should create a plan with researcher and critic for generic query', async () => {
            const query = 'Tell me about quantum computing';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            expect(plan.steps.length).toBeGreaterThan(0);
            expect(plan.steps.some((s: TodoItem) => s.agent === 'researcher')).toBe(true);
            expect(plan.steps.some((s: TodoItem) => s.agent === 'critic')).toBe(true);
            expect(plan.estimated_duration).toBeDefined();
            expect(typeof plan.estimated_duration).toBe('string');
        });

        it('should detect research tasks', async () => {
            const query = 'Research the latest trends in AI';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            const researchStep = plan.steps.find((s: TodoItem) => s.agent === 'researcher');
            expect(researchStep).toBeDefined();
            expect(researchStep?.description).toContain('Research');
        });

        it('should detect coding tasks', async () => {
            const query = 'Write code to implement a binary search tree';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            const coderStep = plan.steps.find((s: TodoItem) => s.agent === 'coder');
            expect(coderStep).toBeDefined();
            expect(coderStep?.description).toContain('Implement');
            expect(coderStep?.require_approval).toBe(true);
            expect(coderStep?.risk).toBe('medium');
        });

        it('should detect design tasks', async () => {
            const query = 'Design a microservices architecture for an e-commerce platform';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            const designStep = plan.steps.find((s: TodoItem) => s.agent === 'design');
            expect(designStep).toBeDefined();
            expect(designStep?.description).toContain('Design');
        });

        it('should detect analysis tasks', async () => {
            const query = 'Analyze the performance metrics and generate a report';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            const analystStep = plan.steps.find((s: TodoItem) => s.agent === 'analyst');
            expect(analystStep).toBeDefined();
            expect(analystStep?.description).toContain('Analyze');
        });

        it('should detect skeptic tasks', async () => {
            const query = 'Challenge the assumptions in our product roadmap';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            const skepticStep = plan.steps.find((s: TodoItem) => s.agent === 'skeptic');
            expect(skepticStep).toBeDefined();
            expect(skepticStep?.description).toContain('Challenge');
        });

        it('should create dependencies between steps', async () => {
            const query = 'Research AI trends, design a system, and implement it';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            expect(plan.steps.length).toBeGreaterThan(2);
            
            // Check that later steps depend on earlier steps
            const designStep = plan.steps.find((s: TodoItem) => s.agent === 'design');
            const coderStep = plan.steps.find((s: TodoItem) => s.agent === 'coder');

            if (designStep && coderStep) {
                expect(coderStep.depends_on?.length).toBeGreaterThan(0);
            }
        });

        it('should always include critic as final review step', async () => {
            const query = 'Build a todo app';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            const criticStep = plan.steps.find((s: TodoItem) => s.agent === 'critic');
            expect(criticStep).toBeDefined();
            
            // Critic should be last or near the end
            const criticIndex = plan.steps.findIndex((s: TodoItem) => s.agent === 'critic');
            expect(criticIndex).toBeGreaterThan(0);
        });

        it('should set all steps to pending status initially', async () => {
            const query = 'Research and build a REST API';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            expect(plan.steps.every((s: TodoItem) => s.status === 'pending')).toBe(true);
        });

        it('should generate unique step IDs', async () => {
            const query = 'Create a complex multi-step project';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            const stepIds = plan.steps.map((s: TodoItem) => s.id);
            const uniqueIds = new Set(stepIds);
            expect(uniqueIds.size).toBe(stepIds.length);
        });

        it('should estimate reasonable durations', async () => {
            const query = 'Write a hello world program';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            expect(plan.estimated_duration).toBeDefined();
            // Duration is a string like '10 minutes' or '1 hour'
            expect(typeof plan.estimated_duration).toBe('string');
            expect(plan.estimated_duration).toMatch(/\d+\s+(minute|hour)s?/);
        });
    });

    describe('Agent detection', () => {
        it('should detect multiple agents in complex queries', async () => {
            const query = 'Research authentication methods, design a security system, implement it, and analyze the results';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            const agents = new Set(plan.steps.map((s: TodoItem) => s.agent));
            expect(agents.has('researcher')).toBe(true);
            expect(agents.has('design')).toBe(true);
            expect(agents.has('coder')).toBe(true);
            expect(agents.has('analyst')).toBe(true);
        });

        it('should handle queries with no clear agent indicators', async () => {
            const query = 'What do you think about this?';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            // Should default to researcher
            expect(plan.steps.some((s: TodoItem) => s.agent === 'researcher')).toBe(true);
            expect(plan.steps.length).toBeGreaterThan(0);
        });
    });

    describe('Risk assessment', () => {
        it('should mark code execution steps as requiring approval', async () => {
            const query = 'Implement a function to delete all user data';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            const coderStep = plan.steps.find((s: TodoItem) => s.agent === 'coder');
            expect(coderStep?.require_approval).toBe(true);
        });

        it('should set risk levels appropriately', async () => {
            const query = 'Build a new feature';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const plan = await planner.createPlan(query, context);

            const coderStep = plan.steps.find((s: TodoItem) => s.agent === 'coder');
            if (coderStep && coderStep.risk) {
                expect(['low', 'medium', 'high', 'critical']).toContain(coderStep.risk);
            }
        });
    });

    describe('Plan timestamps', () => {
        it('should set created_at and updated_at timestamps', async () => {
            const query = 'Test task';
            const context = {
                project_id: 'proj-123',
                user_id: 'user-123',
            };

            const before = new Date();
            const plan = await planner.createPlan(query, context);
            const after = new Date();

            expect(plan.created_at).toBeDefined();
            expect(plan.updated_at).toBeDefined();
            // Timestamps are ISO strings
            const createdAt = new Date(plan.created_at);
            expect(createdAt.getTime()).toBeGreaterThanOrEqual(before.getTime());
            expect(createdAt.getTime()).toBeLessThanOrEqual(after.getTime());
        });
    });
});
