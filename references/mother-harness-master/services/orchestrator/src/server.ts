/**
 * Mother Orchestrator Server
 * Main entry point for the orchestration service
 */

import Fastify from 'fastify';
import cors from '@fastify/cors';
import helmet from '@fastify/helmet';
import websocket from '@fastify/websocket';
import swagger from '@fastify/swagger';
import swaggerUi from '@fastify/swagger-ui';
import { readFileSync } from 'fs';
import { parse as parseYaml } from 'yaml';
import {
    getRedisClient,
    closeRedisClient,
    createAllIndexes,
    checkIndexesExist,
    logAuditEvent,
    redactPIIFromObject,
    resolveLibraryAccess,
} from '@mother-harness/shared';
import { Orchestrator } from './orchestrator.js';
import { config } from './config.js';
import { registerAuth, requireRole, type UserSession } from './auth.js';
import { registerProductionExecutors } from './executors/index.js';
import { startActivityMetricsConsumer } from './activity-metrics-consumer.js';
import { getActivityEventTaxonomy } from './event-taxonomy.js';

// Create Fastify instance
const app = Fastify({
    logger: {
        level: config.logLevel,
        transport: config.nodeEnv === 'development'
            ? { target: 'pino-pretty' }
            : undefined,
    } as any, // Cast to any to avoid strict definition mismatch
});

// Orchestrator instance
let orchestrator: Orchestrator;
let metricsConsumer: ReturnType<typeof startActivityMetricsConsumer> | null = null;

const INVALID_SECRET_VALUES = new Set([
    '',
    'CHANGE_ME',
    'CHANGEME',
    'change_me',
    'changeme',
    'default',
    'password',
    'development-secret-change-in-production',
]);

function isInvalidSecret(value: string | undefined): boolean {
    if (!value) {
        return true;
    }

    const trimmed = value.trim();
    if (!trimmed) {
        return true;
    }

    return INVALID_SECRET_VALUES.has(trimmed);
}

function validateSecrets() {
    const requiredSecrets = [
        'JWT_SECRET',
        'REDIS_PASSWORD',
        'REDIS_ACL_ORCHESTRATOR_PASSWORD',
        'REDIS_ACL_DOCLING_PASSWORD',
        'REDIS_ACL_AGENTS_PASSWORD',
        'REDIS_ACL_DASHBOARD_PASSWORD',
        'N8N_PASSWORD',
    ];

    const invalidSecrets = requiredSecrets.filter((key) => isInvalidSecret(process.env[key]));

    const redisUrl = process.env['REDIS_URL'] ?? '';
    if (/CHANGE_ME|CHANGEME|changeme|change_me/.test(redisUrl)) {
        invalidSecrets.push('REDIS_URL');
    }

    if (invalidSecrets.length > 0) {
        throw new Error(
            `Refusing to start with default or empty secrets: ${invalidSecrets.join(', ')}`
        );
    }
}

// Register plugins
async function registerPlugins() {
    // Load OpenAPI specification
    const openApiPath = new URL('../openapi.yaml', import.meta.url);
    const openApiContent = readFileSync(openApiPath, 'utf-8');
    const openApiSpec = parseYaml(openApiContent);

    // Register Swagger
    await app.register(swagger, {
        mode: 'static',
        specification: {
            document: openApiSpec,
        },
    });

    // Register Swagger UI
    await app.register(swaggerUi, {
        routePrefix: '/documentation',
        uiConfig: {
            docExpansion: 'list',
            deepLinking: true,
        },
        staticCSP: true,
    });

    // Register security headers
    await app.register(helmet, {
        contentSecurityPolicy: {
            directives: {
                defaultSrc: ["'self'"],
                scriptSrc: ["'self'", "'unsafe-inline'"], // unsafe-inline needed for Swagger UI
                styleSrc: ["'self'", "'unsafe-inline'"],
                imgSrc: ["'self'", 'data:', 'https:'],
                connectSrc: ["'self'"],
                fontSrc: ["'self'"],
                objectSrc: ["'none'"],
                mediaSrc: ["'self'"],
                frameSrc: ["'none'"],
            },
        },
        crossOriginEmbedderPolicy: false, // Disable for API server
        hsts: {
            maxAge: 31536000, // 1 year
            includeSubDomains: true,
            preload: true,
        },
    });

    await app.register(cors, {
        origin: config.corsOrigin,
    });

    await app.register(websocket);

    // WebSocket endpoint - MUST be registered here after websocket plugin
    // so that @fastify/websocket provides the WebSocket as first param
    app.register(async function (fastify) {
        fastify.get('/ws', { websocket: true }, (socket: any, request: any) => {
            const url = new URL(request?.url || '', 'http://localhost');
            const taskId = url.searchParams.get('task_id');
            const userId = request?.user?.user_id || url.searchParams.get('user_id') || 'anonymous';

            app.log.info({ task_id: taskId, user_id: userId }, 'WebSocket connection established');

            socket.on('message', (message: Buffer | string) => {
                try {
                    const data = JSON.parse(message.toString());
                    app.log.debug({ data }, 'WebSocket message received');

                    // Handle ping/pong for keepalive
                    if (data.type === 'ping') {
                        socket.socket.send(JSON.stringify({ type: 'pong' }));
                    }
                } catch (error) {
                    app.log.error(error, 'Failed to parse WebSocket message');
                }
            });

            socket.on('close', () => {
                app.log.info({ task_id: taskId }, 'WebSocket connection closed');
            });

            socket.on('error', (error: Error) => {
                app.log.error({ error, task_id: taskId }, 'WebSocket error');
            });

            // Send initial connection confirmation
            socket.socket.send(JSON.stringify({
                type: 'connected',
                task_id: taskId
            }));
        });
    });

    await registerAuth(app as any);
}

// Health check endpoint
app.get('/health', async () => {
    const redisOk = await checkRedisHealth();
    return {
        status: redisOk ? 'ok' : 'degraded',
        version: config.version,
        timestamp: new Date().toISOString(),
        redis: redisOk ? 'connected' : 'disconnected',
    };
});

// API routes
app.post('/api/ask', { preHandler: requireRole('user', 'admin') }, async (request, reply) => {
    const body = request.body as { query: string; project_id?: string; library_ids?: string[] };
    const user = (request as any).user as UserSession;
    const requestedLibraries = body.library_ids ?? [];
    const allowedLibraryIds = process.env['LIBRARY_ALLOWED_IDS']
        ?.split(',')
        .map(entry => entry.trim())
        .filter(Boolean);

    const access = resolveLibraryAccess(requestedLibraries, user.roles, {
        ...(allowedLibraryIds && { allowedLibraryIds }),
    });

    if (access.denied.length > 0) {
        reply.status(403);
        return { error: 'Library access denied', denied: access.denied };
    }

    try {
        const { run, task } = await orchestrator.createRun(
            user.user_id,
            body.query,
            body.project_id
        );

        // Start execution in background
        orchestrator.executeRun(run.id).catch((err) => {
            app.log.error({ run_id: run.id, error: err }, 'Run execution failed');
        });

        return { run_id: run.id, task_id: task.id, status: run.status };
    } catch (error) {
        app.log.error(error, 'Failed to create task');
        reply.status(500);
        return { error: 'Failed to create task' };
    }
});

app.post('/api/runs', async (request, reply) => {
    const body = request.body as { query: string; project_id?: string; user_id: string };

    try {
        const { run, task } = await orchestrator.createRun(
            body.user_id,
            body.query,
            body.project_id
        );

        orchestrator.executeRun(run.id).catch((err) => {
            app.log.error({ run_id: run.id, error: err }, 'Run execution failed');
        });

        return { run_id: run.id, task_id: task.id, status: run.status };
    } catch (error) {
        app.log.error(error, 'Failed to submit run');
        reply.status(500);
        return { error: 'Failed to submit run' };
    }
});

app.get('/api/runs', async (request) => {
    const { user_id } = request.query as { user_id: string };
    return await orchestrator.listRuns(user_id);
});

app.get('/api/runs/:id', async (request, reply) => {
    const { id } = request.params as { id: string };

    try {
        const run = await orchestrator.getRun(id);
        if (!run) {
            reply.status(404);
            return { error: 'Run not found' };
        }
        return run;
    } catch (error) {
        app.log.error(error, 'Failed to get run');
        reply.status(500);
        return { error: 'Failed to get run' };
    }
});

app.get('/api/runs/:id/artifacts', async (request, reply) => {
    const { id } = request.params as { id: string };

    try {
        const artifacts = await orchestrator.getRunArtifacts(id);
        if (!artifacts) {
            reply.status(404);
            return { error: 'Run not found' };
        }
        return artifacts;
    } catch (error) {
        app.log.error(error, 'Failed to get run artifacts');
        reply.status(500);
        return { error: 'Failed to get run artifacts' };
    }
});

app.get('/api/runs/:id/replay', async (request, reply) => {
    const { id } = request.params as { id: string };
    const { limit, direction } = request.query as { limit?: string; direction?: 'forward' | 'backward' };

    const parsedLimit = limit ? Number.parseInt(limit, 10) : undefined;
    const safeLimit = typeof parsedLimit === 'number' && Number.isFinite(parsedLimit) ? parsedLimit : undefined;

    try {
        const replay = await orchestrator.getRunReplay(id, {
            ...(safeLimit !== undefined && { limit: safeLimit }),
            ...(direction && { direction }),
        });
        if (!replay) {
            reply.status(404);
            return { error: 'Run not found' };
        }
        return replay;
    } catch (error) {
        app.log.error(error, 'Failed to get run replay');
        reply.status(500);
        return { error: 'Failed to get run replay' };
    }
});

app.get('/api/artifacts/:id', async (request, reply) => {
    const { id } = request.params as { id: string };

    try {
        const artifact = await orchestrator.getArtifact(id);
        if (!artifact) {
            reply.status(404);
            return { error: 'Artifact not found' };
        }
        return artifact;
    } catch (error) {
        app.log.error(error, 'Failed to get artifact');
        reply.status(500);
        return { error: 'Failed to get artifact' };
    }
});

app.get('/api/task/:id', async (request, reply) => {
    const { id } = request.params as { id: string };

    try {
        const task = await orchestrator.getTask(id);
        if (!task) {
            reply.status(404);
            return { error: 'Task not found' };
        }
        return task;
    } catch (error) {
        app.log.error(error, 'Failed to get task');
        reply.status(500);
        return { error: 'Failed to get task' };
    }
});

app.get('/api/projects', { preHandler: requireRole('user', 'admin') }, async (request) => {
    const { user_id } = request.query as { user_id?: string };
    const user = (request as any).user as UserSession;
    const requestedUserId = user.roles.includes('admin') && user_id ? user_id : user.user_id;
    return await orchestrator.listProjects(requestedUserId);
});

app.get('/api/approvals/pending', { preHandler: requireRole('user', 'approver', 'admin') }, async (request) => {
    const user = (request as any).user as UserSession;
    return await orchestrator.getPendingApprovals(user.user_id);
});

app.post('/api/approvals/:id/respond', { preHandler: requireRole('approver', 'admin') }, async (request, reply) => {
    const { id } = request.params as { id: string };
    const { approved, notes } = request.body as { approved: boolean; notes?: string };
    const user = (request as any).user as UserSession;

    try {
        await orchestrator.respondToApproval(id, approved, notes);
        await logAuditEvent({
            type: 'approval_responded',
            action: approved ? 'approved' : 'rejected',
            actor: {
                user_id: user.user_id,
                ...(user.email !== undefined && { email: user.email }),
                roles: user.roles,
            },
            resource: {
                type: 'approval',
                id,
            },
            metadata: {
                notes,
            },
            status: 'success',
        });
        return { success: true };
    } catch (error) {
        app.log.error(error, 'Failed to respond to approval');
        reply.status(500);
        return { error: 'Failed to respond to approval' };
    }
});

app.get('/api/libraries', async (request, reply) => {
    const { search } = request.query as { search?: string };

    try {
        return await orchestrator.listLibraries(search);
    } catch (error) {
        app.log.error(error, 'Failed to list libraries');
        reply.status(500);
        return { error: 'Failed to list libraries' };
    }
});

app.post('/api/libraries', async (request, reply) => {
    const body = request.body as {
        name: string;
        folder_path: string;
        description?: string;
        auto_scan?: boolean;
    };

    try {
        const library = await orchestrator.createLibrary(
            body.name,
            body.folder_path,
            body.description,
            body.auto_scan ?? true
        );
        reply.status(201);
        return library;
    } catch (error) {
        app.log.error(error, 'Failed to create library');
        reply.status(500);
        return { error: 'Failed to create library' };
    }
});

app.post('/api/libraries/:id/rescan', async (request, reply) => {
    const { id } = request.params as { id: string };

    try {
        const library = await orchestrator.rescanLibrary(id);
        if (!library) {
            reply.status(404);
            return { error: 'Library not found' };
        }
        return library;
    } catch (error) {
        app.log.error(error, 'Failed to rescan library');
        reply.status(500);
        return { error: 'Failed to rescan library' };
    }
});

// Activity Metrics API
app.get('/api/metrics/activity', { preHandler: requireRole('user', 'admin') }, async (request, reply) => {
    const { user_id, days = '7' } = request.query as { user_id?: string; days?: string };
    const user = (request as any).user as UserSession;

    // Users can only view their own metrics unless they're admin
    const targetUserId = user.roles.includes('admin') && user_id ? user_id : user.user_id;
    const daysToFetch = Math.min(parseInt(days), 90); // Max 90 days

    try {
        const redis = getRedisClient();
        const metricsData: Array<{
            date: string;
            activity: Record<string, number>;
            errors: Record<string, number>;
            runs: Record<string, number>;
        }> = [];

        // Fetch last N days of metrics
        for (let i = 0; i < daysToFetch; i++) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            const dateKey = date.toISOString().split('T')[0] ?? '';

            const activityKey = `metrics:activity:user:${targetUserId}:daily:${dateKey}`;
            const errorsKey = `metrics:errors:user:${targetUserId}:daily:${dateKey}`;
            const runsKey = `metrics:runs:user:${targetUserId}:daily:${dateKey}`;

            const [activity, errors, runs] = await Promise.all([
                redis.hgetall(activityKey),
                redis.hgetall(errorsKey),
                redis.hgetall(runsKey),
            ]);

            // Convert string values to numbers
            const activityCounts: Record<string, number> = {};
            const errorCounts: Record<string, number> = {};
            const runCounts: Record<string, number> = {};

            for (const [key, value] of Object.entries(activity)) {
                activityCounts[key] = parseInt(value) || 0;
            }
            for (const [key, value] of Object.entries(errors)) {
                errorCounts[key] = parseInt(value) || 0;
            }
            for (const [key, value] of Object.entries(runs)) {
                runCounts[key] = parseInt(value) || 0;
            }

            metricsData.push({
                date: dateKey,
                activity: activityCounts,
                errors: errorCounts,
                runs: runCounts,
            });
        }

        return {
            user_id: targetUserId,
            days: metricsData.reverse(), // Oldest first
        };
    } catch (error) {
        app.log.error(error, 'Failed to fetch activity metrics');
        reply.status(500);
        return { error: 'Failed to fetch activity metrics' };
    }
});

app.get('/api/metrics/summary', { preHandler: requireRole('user', 'admin') }, async (request, reply) => {
    const user = (request as any).user as UserSession;
    const { user_id } = request.query as { user_id?: string };
    const targetUserId = user.roles.includes('admin') && user_id ? user_id : user.user_id;

    try {
        const redis = getRedisClient();
        const today = new Date().toISOString().split('T')[0] ?? '';

        const activityKey = `metrics:activity:user:${targetUserId}:daily:${today}`;
        const errorsKey = `metrics:errors:user:${targetUserId}:daily:${today}`;
        const runsKey = `metrics:runs:user:${targetUserId}:daily:${today}`;

        const [activity, errors, runs] = await Promise.all([
            redis.hgetall(activityKey),
            redis.hgetall(errorsKey),
            redis.hgetall(runsKey),
        ]);

        // Calculate totals
        const totalActivity = Object.values(activity).reduce((sum, val) => sum + (parseInt(val) || 0), 0);
        const totalErrors = Object.values(errors).reduce((sum, val) => sum + (parseInt(val) || 0), 0);
        const totalRuns = Object.values(runs).reduce((sum, val) => sum + (parseInt(val) || 0), 0);

        return {
            user_id: targetUserId,
            date: today,
            total_events: totalActivity,
            total_errors: totalErrors,
            total_runs: totalRuns,
            error_rate: totalActivity > 0 ? (totalErrors / totalActivity) * 100 : 0,
            top_events: Object.entries(activity)
                .map(([event, count]) => ({
                    event,
                    count: parseInt(count) || 0,
                    taxonomy: getActivityEventTaxonomy(event as any),
                }))
                .sort((a, b) => b.count - a.count)
                .slice(0, 10),
        };
    } catch (error) {
        app.log.error(error, 'Failed to fetch metrics summary');
        reply.status(500);
        return { error: 'Failed to fetch metrics summary' };
    }
});


app.addHook('onResponse', async (request, reply) => {
    if (request.url === '/health' || request.url.startsWith('/ws')) {
        return;
    }

    const user = (request as any).user as UserSession | undefined;
    if (!user) return;

    try {
        await logAuditEvent({
            type: 'access',
            action: request.method,
            actor: {
                user_id: user.user_id,
                ...(user.email !== undefined && { email: user.email }),
                roles: user.roles,
            },
            resource: {
                type: 'http',
                id: request.routerPath ?? request.url,
                attributes: {
                    path: request.url,
                },
            },
            metadata: {
                status_code: reply.statusCode,
                query: redactPIIFromObject(request.query),
                body: request.body ? redactPIIFromObject(request.body) : undefined,
            },
            status: reply.statusCode < 400 ? 'success' : 'error',
            ip: request.ip,
            ...(request.headers['user-agent'] && { user_agent: request.headers['user-agent'] }),
        });
    } catch (error) {
        app.log.warn({ error }, 'Failed to write audit log');
    }
});

// Check Redis health
async function checkRedisHealth(): Promise<boolean> {
    try {
        const redis = getRedisClient();
        await redis.ping();
        return true;
    } catch {
        return false;
    }
}

// Initialize Redis indexes if needed
async function initializeRedis() {
    const redis = getRedisClient({ url: config.redisUrl });

    const indexesExist = await checkIndexesExist(redis);
    if (!indexesExist) {
        app.log.info('Creating Redis indexes...');
        await createAllIndexes(redis);
        app.log.info('Redis indexes created');
    }
}

// Start server
async function start() {
    try {
        validateSecrets();
        await registerPlugins();
        await initializeRedis();
        registerProductionExecutors();

        // Initialize orchestrator
        orchestrator = new Orchestrator();
        metricsConsumer = startActivityMetricsConsumer();

        await app.listen({
            port: config.port,
            host: '0.0.0.0'
        });

        app.log.info(`Mother Orchestrator running on port ${config.port}`);
    } catch (error) {
        app.log.error(error);
        process.exit(1);
    }
}

// Graceful shutdown
async function shutdown() {
    app.log.info('Shutting down...');
    if (metricsConsumer) {
        await metricsConsumer.stop();
    }
    await app.close();
    await closeRedisClient();
    process.exit(0);
}

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

start();
