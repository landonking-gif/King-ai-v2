/**
 * Test Utilities
 * Shared testing helpers and fixtures
 */

import { vi, type Mock } from 'vitest';
import type { Task, Project, AgentType, TodoItem, TaskStatus, TaskType, ProjectType, ProjectStatus } from '../types/index.js';

/** Create a mock task for testing */
export function createMockTask(overrides: Partial<Task> = {}): Task {
    const now = new Date().toISOString();
    return {
        id: 'task-test-123',
        project_id: 'project-test-123',
        user_id: 'user-test-123',
        type: 'mixed' as TaskType,
        query: 'Test query',
        status: 'planning' as TaskStatus,
        todo_list: [],
        execution_plan: {
            steps: [],
            created_at: now,
            updated_at: now,
        },
        steps_completed: [],
        artifacts: [],
        created_at: now,
        updated_at: now,
        ...overrides,
    };
}

/** Create a mock project for testing */
export function createMockProject(overrides: Partial<Project> = {}): Project {
    const now = new Date().toISOString();
    return {
        id: 'project-test-123',
        name: 'Test Project',
        type: 'mixed' as ProjectType,
        status: 'active' as ProjectStatus,
        threads: [],
        recent_messages: [],
        session_summaries: [],
        created_at: now,
        updated_at: now,
        last_activity: now,
        ...overrides,
    };
}

/** Create a mock todo item for testing */
export function createMockTodoItem(overrides: Partial<TodoItem> = {}): TodoItem {
    return {
        id: 'todo-test-123',
        description: 'Test todo item',
        agent: 'researcher' as AgentType,
        status: 'pending',
        depends_on: [],
        ...overrides,
    };
}

/** Wait for a condition to be true */
export async function waitFor(
    condition: () => boolean | Promise<boolean>,
    options: { timeout?: number; interval?: number } = {}
): Promise<void> {
    const timeout = options.timeout ?? 5000;
    const interval = options.interval ?? 100;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
        if (await condition()) {
            return;
        }
        await new Promise(resolve => setTimeout(resolve, interval));
    }

    throw new Error('Timeout waiting for condition');
}

/** Mock Redis client for testing */
export function createMockRedis(): Record<string, Mock | Map<string, string> | ((...args: unknown[]) => unknown)> {
    const store = new Map<string, string>();

    return {
        get: vi.fn((key: string) => store.get(key) ?? null),
        set: vi.fn((key: string, value: string) => {
            store.set(key, value);
            return 'OK';
        }),
        del: vi.fn((key: string) => {
            const existed = store.has(key);
            store.delete(key);
            return existed ? 1 : 0;
        }),
        keys: vi.fn((pattern: string) => {
            const regex = new RegExp(pattern.replace('*', '.*'));
            return Array.from(store.keys()).filter(k => regex.test(k));
        }),
        exists: vi.fn((key: string) => store.has(key) ? 1 : 0),
        expire: vi.fn(() => 1),
        hget: vi.fn(() => null),
        hset: vi.fn(() => 1),
        hgetall: vi.fn(() => ({})),
        hincrbyfloat: vi.fn(() => 0),
        xadd: vi.fn(() => 'stream-id'),
        xread: vi.fn(() => null),
        xreadgroup: vi.fn(() => null),
        xack: vi.fn(() => 1),
        xgroup: vi.fn(() => 'OK'),
        ping: vi.fn(() => 'PONG'),
        quit: vi.fn(() => 'OK'),
        _store: store, // Expose for assertions
    };
}

/** Mock Redis JSON client for testing */
export function createMockRedisJSON(): Record<string, Mock | Map<string, unknown> | ((...args: unknown[]) => unknown)> {
    const store = new Map<string, unknown>();

    return {
        get: vi.fn(<T>(key: string): T | null => {
            return (store.get(key) as T) ?? null;
        }),
        set: vi.fn((key: string, path: string, value: unknown) => {
            if (path === '$') {
                store.set(key, value);
            } else {
                // Handle nested paths (simplified)
                const obj = store.get(key) as Record<string, unknown> | undefined;
                if (obj) {
                    const pathParts = path.replace('$.', '').split('.');
                    let target = obj;
                    for (let i = 0; i < pathParts.length - 1; i++) {
                        target = target[pathParts[i]!] as Record<string, unknown>;
                    }
                    target[pathParts[pathParts.length - 1]!] = value;
                }
            }
            return 'OK';
        }),
        del: vi.fn((key: string) => {
            const existed = store.has(key);
            store.delete(key);
            return existed ? 1 : 0;
        }),
        keys: vi.fn((pattern: string) => {
            const regex = new RegExp(pattern.replace('*', '.*'));
            return Array.from(store.keys()).filter(k => regex.test(k));
        }),
        exists: vi.fn((key: string) => store.has(key) ? 1 : 0),
        _store: store,
    };
}
