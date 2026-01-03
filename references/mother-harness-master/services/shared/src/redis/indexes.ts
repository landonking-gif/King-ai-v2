/**
 * Redis Indexes
 * FT.CREATE definitions for RediSearch indexes
 */

import { getRedisClient } from './client.js';
import type { Redis } from 'ioredis';

/** Index definitions for RediSearch */
export const INDEX_DEFINITIONS = {
    /**
     * Document chunks index for vector similarity search
     * Supports: semantic search, library filtering, chunk type filtering
     */
    'idx:chunks': `
    FT.CREATE idx:chunks ON JSON PREFIX 1 chunk:
    SCHEMA
      $.embedding AS embedding VECTOR FLAT 6 TYPE FLOAT32 DIM 768 DISTANCE_METRIC COSINE
      $.library AS library TAG
      $.document_id AS document_id TAG
      $.document_name AS document_name TEXT
      $.file_path AS file_path TEXT
      $.chunk_type AS chunk_type TAG
      $.chunk_index AS chunk_index NUMERIC SORTABLE
      $.page_number AS page_number NUMERIC SORTABLE
      $.section_title AS section_title TEXT
      $.content AS content TEXT
      $.searchable AS searchable TAG
  `,

    /**
     * Tasks index for task search and filtering
     * Supports: project filtering, status filtering, type filtering, text search
     */
    'idx:tasks': `
    FT.CREATE idx:tasks ON JSON PREFIX 1 task:
    SCHEMA
      $.project_id AS project_id TAG
      $.user_id AS user_id TAG
      $.status AS status TAG
      $.type AS type TAG
      $.query AS query TEXT
      $.created_at AS created_at NUMERIC SORTABLE
  `,

    /**
     * Projects index for project search
     * Supports: name search, status filtering, type filtering
     */
    'idx:projects': `
    FT.CREATE idx:projects ON JSON PREFIX 1 project:
    SCHEMA
      $.name AS name TEXT
      $.status AS status TAG
      $.type AS type TAG
      $.user_id AS user_id TAG
      $.created_at AS created_at NUMERIC SORTABLE
  `,

    /**
     * Approvals index for pending approval retrieval
     * Supports: status filtering, user filtering, task filtering
     */
    'idx:approvals': `
    FT.CREATE idx:approvals ON JSON PREFIX 1 approval:
    SCHEMA
      $.status AS status TAG
      $.user_id AS user_id TAG
      $.task_id AS task_id TAG
      $.project_id AS project_id TAG
      $.risk_level AS risk_level TAG
      $.created_at AS created_at NUMERIC SORTABLE
  `,

    /**
     * Libraries index for library management
     * Supports: name search, status filtering
     */
    'idx:libraries': `
    FT.CREATE idx:libraries ON JSON PREFIX 1 library:
    SCHEMA
      $.name AS name TEXT
      $.folder_path AS folder_path TAG
      $.scan_status AS scan_status TAG
      $.auto_scan AS auto_scan TAG
  `,

    /**
     * Terminations index for analytics
     * Supports: reason filtering, date range queries
     */
    'idx:terminations': `
    FT.CREATE idx:terminations ON JSON PREFIX 1 termination:
    SCHEMA
      $.reason AS reason TAG
      $.user_id AS user_id TAG
      $.task_id AS task_id TAG
      $.terminated_at AS terminated_at NUMERIC SORTABLE
  `,
};

/**
 * Create all indexes
 * Drops existing indexes first (idempotent operation)
 */
export async function createAllIndexes(redis?: Redis): Promise<void> {
    const client = redis ?? getRedisClient();

    for (const [indexName, definition] of Object.entries(INDEX_DEFINITIONS)) {
        try {
            // Try to drop existing index
            await client.call('FT.DROPINDEX', indexName);
            console.log(`[Redis] Dropped existing index: ${indexName}`);
        } catch {
            // Index doesn't exist, that's fine
        }

        try {
            // Create the index
            const parts = definition.trim().split(/\s+/);
            await client.call(parts[0]!, ...parts.slice(1));
            console.log(`[Redis] Created index: ${indexName}`);
        } catch (error) {
            console.error(`[Redis] Failed to create index ${indexName}:`, error);
            throw error;
        }
    }
}

/**
 * Check if all indexes exist
 */
export async function checkIndexesExist(redis?: Redis): Promise<boolean> {
    const client = redis ?? getRedisClient();

    for (const indexName of Object.keys(INDEX_DEFINITIONS)) {
        try {
            await client.call('FT.INFO', indexName);
        } catch {
            return false;
        }
    }
    return true;
}

/**
 * Get index info
 */
export async function getIndexInfo(indexName: string, redis?: Redis): Promise<unknown> {
    const client = redis ?? getRedisClient();
    return await client.call('FT.INFO', indexName);
}
