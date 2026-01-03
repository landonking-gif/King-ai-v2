import { describe, expect, it } from 'vitest';
import { TaskPlanner } from './planner.js';

describe('TaskPlanner routing logic', () => {
    it('routes to coder and critic for build requests', async () => {
        const planner = new TaskPlanner();

        const plan = await planner.createPlan('Build a data pipeline', {
            project_id: 'project-1',
            user_id: 'user-1',
        });

        const agents = plan.steps.map(step => step.agent);

        expect(agents).toContain('coder');
        expect(agents).toContain('critic');
        const coderStep = plan.steps.find(step => step.agent === 'coder');
        expect(coderStep?.require_approval).toBe(true);
    });

    it('defaults to researcher when no keywords match', async () => {
        const planner = new TaskPlanner();

        const plan = await planner.createPlan('Summarize this topic', {
            project_id: 'project-1',
            user_id: 'user-1',
        });

        const agents = plan.steps.map(step => step.agent);

        expect(agents).toContain('researcher');
        expect(agents).toContain('critic');
    });

    it('chains dependencies between sequential steps', async () => {
        const planner = new TaskPlanner();

        const plan = await planner.createPlan('Research and analyze metrics', {
            project_id: 'project-1',
            user_id: 'user-1',
        });

        expect(plan.steps.length).toBeGreaterThan(1);
        const secondStep = plan.steps[1]!;
        expect(secondStep.depends_on.length).toBe(1);
        expect(secondStep.depends_on[0]).toBe(plan.steps[0]!.id);
    });
});
