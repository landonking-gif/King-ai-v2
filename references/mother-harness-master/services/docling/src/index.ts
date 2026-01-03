/**
 * Docling Service Entry Point
 * Starts the document ingestion service
 */

import * as http from 'http';
import { getRedisClient, getRedisJSON, type Library } from '@mother-harness/shared';
import { getLibraryWatcher } from './watcher.js';
import { getDocumentProcessor } from './processor.js';

const STREAM_KEY = 'stream:docling';
const CONSUMER_GROUP = 'docling-processors';
const CONSUMER_NAME = `processor-${process.pid}`;

/**
 * Start health check HTTP server
 */
function startHealthServer(): http.Server {
    const server = http.createServer((req, res) => {
        if (req.url === '/health') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                status: 'healthy',
                service: 'docling',
                pid: process.pid,
                uptime: process.uptime(),
            }));
        } else {
            res.writeHead(404);
            res.end();
        }
    });

    const port = process.env['HEALTH_PORT'] ? parseInt(process.env['HEALTH_PORT']) : 8080;
    server.listen(port, () => {
        console.log(`[Docling] Health check server listening on port ${port}`);
    });

    return server;
}

/**
 * Start the Docling service
 */
async function start(): Promise<void> {
    console.log('[Docling] Starting document ingestion service...');

    // Start health check server
    const healthServer = startHealthServer();

    const redis = getRedisClient();
    const redisJson = getRedisJSON();
    const watcher = getLibraryWatcher();
    const processor = getDocumentProcessor();

    // Create consumer group if it doesn't exist
    try {
        await redis.xgroup('CREATE', STREAM_KEY, CONSUMER_GROUP, '0', 'MKSTREAM');
        console.log('[Docling] Created consumer group');
    } catch (error) {
        // Group already exists, that's fine
    }

    // Start watching all auto-scan libraries
    await startWatchers(redisJson, watcher);

    // Store health server reference for shutdown
    (global as { healthServer?: http.Server }).healthServer = healthServer;

    // Process jobs from stream
    await processStream(redis, processor);
}

/**
 * Start watchers for all auto-scan libraries
 */
async function startWatchers(
    redis: ReturnType<typeof getRedisJSON>,
    watcher: ReturnType<typeof getLibraryWatcher>
): Promise<void> {
    const keys = await redis.keys('library:*');

    for (const key of keys) {
        const library = await redis.get<Library>(key);
        if (library?.auto_scan) {
            await watcher.watchLibrary(library);
        }
    }

    console.log(`[Docling] Started ${watcher.getActiveWatcherCount()} library watchers`);
}

/**
 * Process jobs from Redis Stream
 */
async function processStream(
    redis: ReturnType<typeof getRedisClient>,
    processor: ReturnType<typeof getDocumentProcessor>
): Promise<void> {
    console.log('[Docling] Listening for ingestion jobs...');

    while (true) {
        try {
            // Read from stream with blocking
            const messages = await redis.call(
                'XREADGROUP',
                'GROUP', CONSUMER_GROUP, CONSUMER_NAME,
                'BLOCK', '5000',
                'COUNT', '1',
                'STREAMS', STREAM_KEY, '>'
            ) as Array<[string, Array<[string, string[]]>]> | null;

            if (!messages || messages.length === 0) {
                continue;
            }

            for (const [_stream, entries] of messages) {
                for (const [id, fields] of entries) {
                    const jobIndex = fields.findIndex((field) => field === 'job');
                    const jobData = jobIndex >= 0 ? fields[jobIndex + 1] : undefined;

                    if (jobData) {
                        const job = JSON.parse(jobData);
                        console.log(`[Docling] Processing job: ${job.id}`);

                        const result = await processor.processJob(job);

                        if (result.success) {
                            console.log(`[Docling] Job ${job.id} completed: ${result.chunks} chunks`);
                        } else {
                            console.error(`[Docling] Job ${job.id} failed: ${result.error}`);
                        }

                        // Acknowledge the message
                        await redis.xack(STREAM_KEY, CONSUMER_GROUP, id);
                    }
                }
            }
        } catch (error) {
            console.error('[Docling] Stream processing error:', error);
            await new Promise(resolve => setTimeout(resolve, 5000));
        }
    }
}

/**
 * Graceful shutdown
 */
async function shutdown(): Promise<void> {
    console.log('[Docling] Shutting down...');

    // Close health server
    const healthServer = (global as { healthServer?: http.Server }).healthServer;
    if (healthServer) {
        healthServer.close();
    }

    const watcher = getLibraryWatcher();
    await watcher.stopAll();

    const redis = getRedisClient();
    await redis.quit();

    process.exit(0);
}

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

// Start the service
start().catch(error => {
    console.error('[Docling] Fatal error:', error);
    process.exit(1);
});
