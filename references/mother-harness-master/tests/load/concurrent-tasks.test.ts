/**
 * Load Test: 10 Concurrent Task Executions
 * Validates system can handle multiple simultaneous task runs
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';

const ORCHESTRATOR_URL = process.env.ORCHESTRATOR_URL || 'http://localhost:8000';
const TEST_TOKEN = process.env.TEST_JWT_TOKEN || '';
const CONCURRENT_TASKS = 10;

interface TaskResponse {
  id: string;
  status: string;
  description: string;
  todo_list?: Array<{ id: string; status: string }>;
}

interface ExecuteResponse {
  run_id: string;
  status: string;
}

describe('Load Test: 10 Concurrent Tasks', () => {
  let taskIds: string[] = [];
  let runIds: string[] = [];

  beforeAll(() => {
    if (!TEST_TOKEN) {
      console.warn('TEST_JWT_TOKEN not set - tests may fail authentication');
    }
  });

  afterAll(() => {
    console.log('\n=== Load Test Summary ===');
    console.log(`Tasks created: ${taskIds.length}`);
    console.log(`Tasks executed: ${runIds.length}`);
  });

  it('should handle health check under load', async () => {
    const requests = Array.from({ length: 100 }, () =>
      fetch(`${ORCHESTRATOR_URL}/health`)
    );

    const responses = await Promise.all(requests);
    const allOk = responses.every(r => r.ok);

    expect(allOk).toBe(true);
  }, 30000);

  it('should create 10 tasks concurrently', async () => {
    const taskQueries = [
      'Research React testing best practices',
      'Analyze API performance metrics',
      'Design user authentication system',
      'Implement caching strategy',
      'Review security vulnerabilities',
      'Research microservices patterns',
      'Optimize database queries',
      'Create API rate limiter',
      'Build monitoring dashboard',
      'Implement CI/CD pipeline',
    ];

    const createTask = async (query: string): Promise<string> => {
      const response = await fetch(`${ORCHESTRATOR_URL}/api/tasks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${TEST_TOKEN}`,
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error(`Failed to create task: ${response.status} ${await response.text()}`);
      }

      const data = await response.json() as TaskResponse;
      return data.id;
    };

    const startTime = Date.now();
    taskIds = await Promise.all(taskQueries.map(createTask));
    const duration = Date.now() - startTime;

    console.log(`\nCreated ${taskIds.length} tasks in ${duration}ms`);
    console.log(`Average: ${(duration / taskIds.length).toFixed(2)}ms per task`);

    expect(taskIds).toHaveLength(CONCURRENT_TASKS);
  }, 60000);

  it('should retrieve all created tasks', async () => {
    const getTasks = taskIds.map(id =>
      fetch(`${ORCHESTRATOR_URL}/api/tasks/${id}`, {
        headers: { 'Authorization': `Bearer ${TEST_TOKEN}` },
      })
    );

    const responses = await Promise.all(getTasks);
    const allOk = responses.every(r => r.ok);

    expect(allOk).toBe(true);

    const tasks = await Promise.all(
      responses.map(r => r.json() as Promise<TaskResponse>)
    );

    console.log('\nTask statuses:');
    tasks.forEach((task, i) => {
      console.log(`  Task ${i + 1}: ${task.status}`);
    });
  }, 30000);

  it('should execute 10 tasks concurrently', async () => {
    const executeTask = async (taskId: string): Promise<string> => {
      const response = await fetch(`${ORCHESTRATOR_URL}/api/tasks/${taskId}/execute`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${TEST_TOKEN}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to execute task ${taskId}: ${response.status}`);
      }

      const data = await response.json() as ExecuteResponse;
      return data.run_id;
    };

    const startTime = Date.now();
    runIds = await Promise.all(taskIds.map(executeTask));
    const duration = Date.now() - startTime;

    console.log(`\nExecuted ${runIds.length} tasks in ${duration}ms`);
    console.log(`Average: ${(duration / runIds.length).toFixed(2)}ms per execution`);

    expect(runIds).toHaveLength(CONCURRENT_TASKS);
  }, 60000);

  it('should monitor concurrent task execution', async () => {
    const monitorTask = async (taskId: string): Promise<TaskResponse> => {
      // Poll task status for up to 2 minutes
      const maxAttempts = 40;
      const pollInterval = 3000; // 3 seconds

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const response = await fetch(`${ORCHESTRATOR_URL}/api/tasks/${taskId}`, {
          headers: { 'Authorization': `Bearer ${TEST_TOKEN}` },
        });

        if (!response.ok) {
          throw new Error(`Failed to get task status: ${response.status}`);
        }

        const task = await response.json() as TaskResponse;

        if (task.status === 'completed' || task.status === 'failed') {
          return task;
        }

        await new Promise(resolve => setTimeout(resolve, pollInterval));
      }

      throw new Error(`Task ${taskId} did not complete within timeout`);
    };

    const startTime = Date.now();
    const results = await Promise.allSettled(taskIds.map(monitorTask));
    const duration = Date.now() - startTime;

    const completed = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;

    console.log(`\n=== Execution Results ===`);
    console.log(`Total duration: ${(duration / 1000).toFixed(2)}s`);
    console.log(`Completed: ${completed}/${CONCURRENT_TASKS}`);
    console.log(`Failed: ${failed}/${CONCURRENT_TASKS}`);

    results.forEach((result, i) => {
      if (result.status === 'fulfilled') {
        console.log(`  Task ${i + 1}: ${result.value.status}`);
      } else {
        console.log(`  Task ${i + 1}: ERROR - ${result.reason}`);
      }
    });

    // At least 70% should complete successfully
    expect(completed).toBeGreaterThanOrEqual(CONCURRENT_TASKS * 0.7);
  }, 180000); // 3 minute timeout

  it('should verify system health after concurrent load', async () => {
    const response = await fetch(`${ORCHESTRATOR_URL}/health`);
    expect(response.ok).toBe(true);

    const health = await response.json();
    console.log('\nPost-load health check:', health);

    expect(health.status).toMatch(/ok|healthy/i);
  }, 10000);

  it('should measure response times under concurrent load', async () => {
    const measureRequest = async (): Promise<number> => {
      const start = Date.now();
      await fetch(`${ORCHESTRATOR_URL}/health`);
      return Date.now() - start;
    };

    const concurrentRequests = 50;
    const times = await Promise.all(
      Array.from({ length: concurrentRequests }, measureRequest)
    );

    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const p95 = times.sort((a, b) => a - b)[Math.floor(times.length * 0.95)] || 0;
    const p99 = times.sort((a, b) => a - b)[Math.floor(times.length * 0.99)] || 0;

    console.log(`\n=== Response Time Metrics (${concurrentRequests} concurrent requests) ===`);
    console.log(`Average: ${avg.toFixed(2)}ms`);
    console.log(`P95: ${p95}ms`);
    console.log(`P99: ${p99}ms`);
    console.log(`Min: ${Math.min(...times)}ms`);
    console.log(`Max: ${Math.max(...times)}ms`);

    // P95 should be under 2 seconds
    expect(p95).toBeLessThan(2000);
  }, 60000);
});

describe('Database Performance Under Load', () => {
  it('should handle rapid task creation and retrieval', async () => {
    const iterations = 20;
    const results: number[] = [];

    for (let i = 0; i < iterations; i++) {
      const start = Date.now();

      // Create task
      const createResponse = await fetch(`${ORCHESTRATOR_URL}/api/tasks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${TEST_TOKEN}`,
        },
        body: JSON.stringify({ query: `Performance test ${i}` }),
      });

      if (!createResponse.ok) continue;

      const { id } = await createResponse.json() as TaskResponse;

      // Immediately retrieve it
      const getResponse = await fetch(`${ORCHESTRATOR_URL}/api/tasks/${id}`, {
        headers: { 'Authorization': `Bearer ${TEST_TOKEN}` },
      });

      if (getResponse.ok) {
        results.push(Date.now() - start);
      }
    }

    const avgTime = results.reduce((a, b) => a + b, 0) / results.length;
    console.log(`\nAverage create+retrieve time: ${avgTime.toFixed(2)}ms`);
    console.log(`Successful operations: ${results.length}/${iterations}`);

    // Average should be under 500ms
    expect(avgTime).toBeLessThan(500);
  }, 60000);
});
