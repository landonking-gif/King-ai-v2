import { describe, it, expect } from 'vitest';
import {
    TaskSchema,
    ProjectSchema,
    TodoItemSchema,
    ApprovalSchema,
} from './schemas.js';
import { createTask } from '../types/task.js';
import { createProject } from '../types/project.js';

describe('Task Schema', () => {
    it('validates a task created from defaults', () => {
        const task = createTask('task-123', 'project-1', 'user-1', 'Test query');

        const result = TaskSchema.safeParse(task);

        expect(result.success).toBe(true);
        expect(result.success && result.data.id).toBe('task-123');
        expect(result.success && result.data.status).toBe('planning');
        expect(result.success && result.data.todo_list).toEqual([]);
    });

    it('rejects invalid task status', () => {
        const task = createTask('task-123', 'project-1', 'user-1', 'Test query');

        const result = TaskSchema.safeParse({
            ...task,
            status: 'unknown',
        });

        expect(result.success).toBe(false);
    });
});

describe('Project Schema', () => {
    it('validates a project created from defaults', () => {
        const project = createProject('proj-1', 'Test Project');

        const result = ProjectSchema.safeParse(project);

        expect(result.success).toBe(true);
        expect(result.success && result.data.name).toBe('Test Project');
        expect(result.success && result.data.threads).toEqual([]);
    });

    it('rejects invalid project status', () => {
        const project = createProject('proj-1', 'Test Project');

        const result = ProjectSchema.safeParse({
            ...project,
            status: 'paused',
        });

        expect(result.success).toBe(false);
    });
});

describe('TodoItem Schema', () => {
    it('validates a minimal todo item', () => {
        const todo = {
            id: 'step-1',
            description: 'Test description',
            agent: 'researcher',
            status: 'pending',
            depends_on: [],
        };

        const result = TodoItemSchema.safeParse(todo);

        expect(result.success).toBe(true);
        expect(result.success && result.data.description).toBe('Test description');
    });

    it('rejects invalid risk level', () => {
        const todo = {
            id: 'step-1',
            description: 'Test description',
            agent: 'coder',
            status: 'pending',
            depends_on: [],
            risk: 'extreme',
        };

        const result = TodoItemSchema.safeParse(todo);

        expect(result.success).toBe(false);
    });
});

describe('Approval Schema', () => {
    it('validates an approval payload', () => {
        const approval = {
            id: 'approval-1',
            run_id: 'run-1',
            task_id: 'task-1',
            project_id: 'project-1',
            step_id: 'step-1',
            user_id: 'user-1',
            type: 'code_execution',
            description: 'Approve execution',
            risk_level: 'medium',
            preview: {},
            status: 'pending',
            created_at: new Date().toISOString(),
        };

        const result = ApprovalSchema.safeParse(approval);

        expect(result.success).toBe(true);
    });
});
