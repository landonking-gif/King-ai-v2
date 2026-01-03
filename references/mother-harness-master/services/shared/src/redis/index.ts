/**
 * Redis Module Index
 * Re-exports all Redis utilities
 */

export {
    getRedisClient,
    closeRedisClient,
    RedisJSON,
    getRedisJSON,
    type RedisConfig,
} from './client.js';

export {
    INDEX_DEFINITIONS,
    createAllIndexes,
    checkIndexesExist,
    getIndexInfo,
} from './indexes.js';
