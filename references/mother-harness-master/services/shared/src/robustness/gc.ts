/**
 * Artifact Garbage Collector
 * Manages artifact lifecycle and cleanup
 */

import { getRedisClient, getRedisJSON } from '../redis/index.js';

/** Artifact retention policy */
export interface ArtifactRetention {
    type: string;           // Artifact type
    default_ttl_days: number;
    max_ttl_days: number;
    archive_after_days?: number;
}

/** Artifact metadata for GC */
export interface ArtifactMetadata {
    id: string;
    type: string;
    task_id: string;
    project_id: string;
    user_id: string;
    size_bytes: number;

    // Lifecycle
    created_at: string;
    last_accessed_at: string;
    expires_at?: string;

    // User controls
    starred: boolean;
    deletion_requested: boolean;

    // Status
    archived: boolean;
    archive_path?: string;
}

/** Default retention policies */
const DEFAULT_POLICIES: ArtifactRetention[] = [
    { type: 'report', default_ttl_days: 30, max_ttl_days: 365, archive_after_days: 7 },
    { type: 'code', default_ttl_days: 90, max_ttl_days: 365 },
    { type: 'diagram', default_ttl_days: 60, max_ttl_days: 365, archive_after_days: 14 },
    { type: 'data', default_ttl_days: 14, max_ttl_days: 90, archive_after_days: 7 },
    { type: 'other', default_ttl_days: 7, max_ttl_days: 30 },
];

export class ArtifactGarbageCollector {
    private redis = getRedisClient();
    private redisJson = getRedisJSON();
    private readonly metadataPrefix = 'artifact_meta:';
    private policies: Map<string, ArtifactRetention> = new Map();

    constructor() {
        // Initialize policies
        for (const policy of DEFAULT_POLICIES) {
            this.policies.set(policy.type, policy);
        }
    }

    /**
     * Register artifact for GC tracking
     */
    async registerArtifact(metadata: Omit<ArtifactMetadata, 'last_accessed_at' | 'starred' | 'deletion_requested' | 'archived'>): Promise<void> {
        const policy = this.policies.get(metadata.type) ?? DEFAULT_POLICIES.find(p => p.type === 'other')!;

        const fullMetadata: ArtifactMetadata = {
            ...metadata,
            last_accessed_at: metadata.created_at,
            starred: false,
            deletion_requested: false,
            archived: false,
            expires_at: this.calculateExpiry(new Date(metadata.created_at), policy.default_ttl_days),
        };

        await this.redisJson.set(`${this.metadataPrefix}${metadata.id}`, '$', fullMetadata);
    }

    /**
     * Record artifact access (extends TTL)
     */
    async recordAccess(artifactId: string): Promise<void> {
        const key = `${this.metadataPrefix}${artifactId}`;
        const metadata = await this.redisJson.get<ArtifactMetadata>(key);

        if (metadata) {
            metadata.last_accessed_at = new Date().toISOString();
            await this.redisJson.set(key, '$', metadata);
        }
    }

    /**
     * Star an artifact (prevents deletion)
     */
    async starArtifact(artifactId: string): Promise<void> {
        const key = `${this.metadataPrefix}${artifactId}`;
        await this.redisJson.set(key, '$.starred', true);
        await this.redisJson.set(key, '$.expires_at', null); // Never expires
    }

    /**
     * Unstar an artifact
     */
    async unstarArtifact(artifactId: string): Promise<void> {
        const key = `${this.metadataPrefix}${artifactId}`;
        const metadata = await this.redisJson.get<ArtifactMetadata>(key);

        if (metadata) {
            const policy = this.policies.get(metadata.type) ?? DEFAULT_POLICIES.find(p => p.type === 'other')!;
            metadata.starred = false;
            metadata.expires_at = this.calculateExpiry(new Date(), policy.default_ttl_days);
            await this.redisJson.set(key, '$', metadata);
        }
    }

    /**
     * Request artifact deletion
     */
    async requestDeletion(artifactId: string): Promise<void> {
        const key = `${this.metadataPrefix}${artifactId}`;
        await this.redisJson.set(key, '$.deletion_requested', true);
    }

    /**
     * Run garbage collection scan
     */
    async runGC(): Promise<{ scanned: number; archived: number; deleted: number }> {
        const keys = await this.redisJson.keys(`${this.metadataPrefix}*`);
        const now = new Date();

        let scanned = 0;
        let archived = 0;
        let deleted = 0;

        for (const key of keys) {
            scanned++;
            const metadata = await this.redisJson.get<ArtifactMetadata>(key);

            if (!metadata) continue;

            // Skip starred artifacts
            if (metadata.starred) continue;

            // Delete if requested
            if (metadata.deletion_requested) {
                await this.deleteArtifact(metadata);
                deleted++;
                continue;
            }

            // Check expiry
            if (metadata.expires_at && new Date(metadata.expires_at) < now) {
                await this.deleteArtifact(metadata);
                deleted++;
                continue;
            }

            // Check for archival
            const policy = this.policies.get(metadata.type);
            if (policy?.archive_after_days && !metadata.archived) {
                const createdAt = new Date(metadata.created_at);
                const archiveAfter = new Date(createdAt.getTime() + policy.archive_after_days * 24 * 60 * 60 * 1000);

                if (now > archiveAfter) {
                    await this.archiveArtifact(metadata);
                    archived++;
                }
            }
        }

        console.log(`[GC] Scan complete: ${scanned} scanned, ${archived} archived, ${deleted} deleted`);

        return { scanned, archived, deleted };
    }

    /**
     * Archive an artifact using gzip compression
     */
    private async archiveArtifact(metadata: ArtifactMetadata): Promise<void> {
        const key = `${this.metadataPrefix}${metadata.id}`;

        try {
            // Get the artifact data
            const artifactData = await this.redisJson.get(`artifact:${metadata.id}`);

            if (artifactData) {
                // Compress the data using gzip
                const { gzipSync } = await import('zlib');
                const jsonData = JSON.stringify(artifactData);
                const compressed = gzipSync(Buffer.from(jsonData, 'utf-8'));

                // Store compressed data with archive prefix
                const archivePath = `archive:${metadata.id}`;
                await this.redis.setex(
                    archivePath,
                    365 * 24 * 60 * 60, // 1 year retention for archives
                    compressed.toString('base64')
                );

                // Delete original artifact data (keep metadata)
                await this.redisJson.del(`artifact:${metadata.id}`);

                // Update metadata
                await this.redisJson.set(key, '$.archived', true);
                await this.redisJson.set(key, '$.archive_path', archivePath);
                await this.redisJson.set(key, '$.archived_at', new Date().toISOString());

                console.log(`[GC] Archived artifact: ${metadata.id} (${compressed.length} bytes compressed)`);
            }
        } catch (error) {
            console.error(`[GC] Failed to archive artifact ${metadata.id}:`, error);
        }
    }

    /**
     * Restore an archived artifact
     */
    async restoreArtifact(artifactId: string): Promise<unknown | null> {
        const key = `${this.metadataPrefix}${artifactId}`;
        const metadata = await this.redisJson.get<ArtifactMetadata>(key);

        if (!metadata?.archived || !metadata.archive_path) {
            return null;
        }

        try {
            // Get compressed data
            const compressedBase64 = await this.redis.get(metadata.archive_path);
            if (!compressedBase64) return null;

            // Decompress
            const { gunzipSync } = await import('zlib');
            const compressed = Buffer.from(compressedBase64, 'base64');
            const jsonData = gunzipSync(compressed).toString('utf-8');
            const data = JSON.parse(jsonData);

            // Restore artifact
            await this.redisJson.set(`artifact:${artifactId}`, '$', data);

            // Update metadata
            await this.redisJson.set(key, '$.archived', false);
            await this.redisJson.set(key, '$.archive_path', null);
            await this.redisJson.set(key, '$.restored_at', new Date().toISOString());

            // Delete archive
            await this.redis.del(metadata.archive_path);

            console.log(`[GC] Restored artifact: ${artifactId}`);
            return data;
        } catch (error) {
            console.error(`[GC] Failed to restore artifact ${artifactId}:`, error);
            return null;
        }
    }

    /**
     * Delete an artifact
     */
    private async deleteArtifact(metadata: ArtifactMetadata): Promise<void> {
        // Delete the artifact data
        await this.redisJson.del(`artifact:${metadata.id}`);

        // Delete the metadata
        await this.redisJson.del(`${this.metadataPrefix}${metadata.id}`);

        console.log(`[GC] Deleted artifact: ${metadata.id}`);
    }

    /**
     * Calculate expiry date
     */
    private calculateExpiry(from: Date, days: number): string {
        const expiry = new Date(from.getTime() + days * 24 * 60 * 60 * 1000);
        return expiry.toISOString();
    }

    /**
     * Get GC statistics
     */
    async getStats(): Promise<{
        total_artifacts: number;
        starred: number;
        archived: number;
        pending_deletion: number;
        by_type: Record<string, number>;
    }> {
        const keys = await this.redisJson.keys(`${this.metadataPrefix}*`);
        const stats = {
            total_artifacts: keys.length,
            starred: 0,
            archived: 0,
            pending_deletion: 0,
            by_type: {} as Record<string, number>,
        };

        for (const key of keys) {
            const metadata = await this.redisJson.get<ArtifactMetadata>(key);
            if (!metadata) continue;

            if (metadata.starred) stats.starred++;
            if (metadata.archived) stats.archived++;
            if (metadata.deletion_requested) stats.pending_deletion++;

            stats.by_type[metadata.type] = (stats.by_type[metadata.type] ?? 0) + 1;
        }

        return stats;
    }
}

// Singleton
let gcInstance: ArtifactGarbageCollector | null = null;

export function getArtifactGC(): ArtifactGarbageCollector {
    if (!gcInstance) {
        gcInstance = new ArtifactGarbageCollector();
    }
    return gcInstance;
}
