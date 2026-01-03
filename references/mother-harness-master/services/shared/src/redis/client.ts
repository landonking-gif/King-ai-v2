/**
 * Redis Client
 * Singleton Redis connection with JSON operations
 */
import { Redis, type RedisOptions } from 'ioredis';

/** Redis connection configuration */
export interface RedisConfig {
    url?: string;                  // Full Redis URL (overrides other options)
    host?: string;
    port?: number;
    password?: string;
    db?: number;
    keyPrefix?: string;
}

/** Default configuration */
const DEFAULT_CONFIG: RedisConfig = {
    host: 'localhost',
    port: 6379,
    db: 0,
};

/** Redis client singleton */
let client: Redis | null = null;

/**
 * Parse Redis URL into discrete options
 * Handles the redis://:password@host:port format (empty username)
 */
function parseRedisUrl(url: string): RedisOptions {
    try {
        const parsed = new URL(url);
        const options: RedisOptions = {
            host: parsed.hostname || 'localhost',
            port: parsed.port ? parseInt(parsed.port, 10) : 6379,
        };

        // Handle password - could be in password field or as username (for :password@host format)
        if (parsed.password) {
            options.password = decodeURIComponent(parsed.password);
        } else if (parsed.username && !parsed.password) {
            // Some URL formats put password as username when using :password@host
            options.password = decodeURIComponent(parsed.username);
        }

        // Parse database from pathname (e.g., /0)
        if (parsed.pathname && parsed.pathname.length > 1) {
            const db = parseInt(parsed.pathname.slice(1), 10);
            if (!isNaN(db)) {
                options.db = db;
            }
        }

        console.log('[Redis] Parsed URL:', {
            host: options.host,
            port: options.port,
            db: options.db ?? 0,
            hasPassword: !!options.password,
        });

        return options;
    } catch (error) {
        console.error('[Redis] Failed to parse URL:', error);
        throw new Error(`Invalid Redis URL: ${error}`);
    }
}

/**
 * Get or create Redis client instance
 */
export function getRedisClient(config: RedisConfig = {}): Redis {
    if (client) return client;

    // Check for REDIS_URL environment variable
    const envUrl = process.env.REDIS_URL;
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };

    let options: RedisOptions;

    if (mergedConfig.url || envUrl) {
        // Parse URL into discrete options for better compatibility
        options = parseRedisUrl(mergedConfig.url || envUrl!);
    } else {
        options = {
            host: mergedConfig.host ?? 'localhost',
            port: mergedConfig.port ?? 6379,
            db: mergedConfig.db ?? 0,
        };
        if (mergedConfig.password !== undefined) {
            options.password = mergedConfig.password;
        }
    }

    // Add key prefix if specified
    if (mergedConfig.keyPrefix !== undefined) {
        options.keyPrefix = mergedConfig.keyPrefix;
    }

    // Add robust connection settings
    options.retryStrategy = (times: number) => {
        const delay = Math.min(times * 200, 3000);
        console.log(`[Redis] Retry attempt ${times}, next retry in ${delay}ms`);
        if (times > 10) {
            console.error('[Redis] Max retries exceeded, giving up');
            return null; // Stop retrying
        }
        return delay;
    };

    options.maxRetriesPerRequest = 3;
    options.connectTimeout = 10000;
    options.commandTimeout = 10000; // Must be > XREADGROUP BLOCK time (5000ms)
    options.enableReadyCheck = true;
    options.lazyConnect = false;

    console.log('[Redis] Creating client with options:', {
        host: options.host,
        port: options.port,
        db: options.db,
        hasPassword: !!options.password,
        connectTimeout: options.connectTimeout,
        maxRetriesPerRequest: options.maxRetriesPerRequest,
    });

    client = new Redis(options);

    // Detailed error handling
    client.on('error', (err: Error) => {
        console.error('[Redis] Connection error:', err.message);
        console.error('[Redis] Error details:', {
            name: err.name,
            stack: err.stack?.split('\n').slice(0, 3).join('\n'),
        });
    });

    client.on('connect', () => {
        console.log('[Redis] TCP connection established');
    });

    client.on('ready', () => {
        console.log('[Redis] Client ready and authenticated');
    });

    client.on('close', () => {
        console.log('[Redis] Connection closed');
    });

    client.on('reconnecting', () => {
        console.log('[Redis] Attempting to reconnect...');
    });

    return client;
}

/**
 * Close Redis connection
 */
export async function closeRedisClient(): Promise<void> {
    if (client) {
        await client.quit();
        client = null;
        console.log('[Redis] Connection closed');
    }
}

/**
 * RedisJSON operations wrapper
 * Note: Requires Redis Stack with RedisJSON module
 */
export class RedisJSON {
    private redis: Redis;

    constructor(redisClient?: Redis) {
        this.redis = redisClient ?? getRedisClient();
    }

    /**
     * Set a JSON value at key
     */
    async set<T>(key: string, path: string, value: T): Promise<'OK'> {
        const result = await this.redis.call(
            'JSON.SET',
            key,
            path,
            JSON.stringify(value)
        );
        return result as 'OK';
    }

    /**
     * Get a JSON value from key
     */
    async get<T>(key: string, path: string = '$'): Promise<T | null> {
        const result = await this.redis.call('JSON.GET', key, path);
        if (!result) return null;
        const parsed = JSON.parse(result as string);
        // JSON.GET with $ returns an array, extract first element
        return path === '$' ? parsed[0] : parsed;
    }

    /**
     * Delete a key or path within a key
     */
    async del(key: string, path: string = '$'): Promise<number> {
        const result = await this.redis.call('JSON.DEL', key, path);
        return result as number;
    }

    /**
     * Append to an array at path
     */
    async arrAppend<T>(key: string, path: string, ...values: T[]): Promise<number> {
        const args = values.map(v => JSON.stringify(v));
        const result = await this.redis.call('JSON.ARRAPPEND', key, path, ...args);
        return result as number;
    }

    /**
     * Check if key exists
     */
    async exists(key: string): Promise<boolean> {
        const result = await this.redis.exists(key);
        return result === 1;
    }

    /**
     * Get all keys matching pattern
     */
    async keys(pattern: string): Promise<string[]> {
        return await this.redis.keys(pattern);
    }
}

/**
 * Get RedisJSON instance
 */
export function getRedisJSON(redisClient?: Redis): RedisJSON {
    return new RedisJSON(redisClient);
}

/** Re-export the Redis type for external use */
export type { Redis };
