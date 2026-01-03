/**
 * Orchestrator Configuration
 * Environment-based configuration for the orchestrator service
 */

export interface Config {
    // Server
    port: number;
    nodeEnv: string;
    logLevel: string;
    corsOrigin: string | string[];
    version: string;

    // Redis
    redisUrl: string;

    // Ollama
    ollamaLocalUrl: string;
    ollamaCloudApiKey: string;

    // n8n
    n8nUrl: string;
    n8nApiKey: string;
}

function getEnv(key: string, defaultValue?: string): string {
    const value = process.env[key];
    if (value === undefined) {
        if (defaultValue !== undefined) return defaultValue;
        throw new Error(`Missing required environment variable: ${key}`);
    }
    return value;
}

export const config: Config = {
    // Server
    port: parseInt(getEnv('PORT', '8000'), 10),
    nodeEnv: getEnv('NODE_ENV', 'development'),
    logLevel: getEnv('LOG_LEVEL', 'info'),
    corsOrigin: getEnv('CORS_ORIGIN', '*'),
    version: getEnv('VERSION', '0.1.0'),

    // Redis
    redisUrl: getEnv('REDIS_URL', 'redis://localhost:6379'),

    // Ollama
    ollamaLocalUrl: getEnv('OLLAMA_LOCAL_URL', 'http://localhost:11434'),
    ollamaCloudApiKey: getEnv('OLLAMA_CLOUD_API_KEY', ''),

    // n8n
    n8nUrl: getEnv('N8N_URL', 'http://localhost:5678'),
    n8nApiKey: getEnv('N8N_API_KEY', ''),
};
