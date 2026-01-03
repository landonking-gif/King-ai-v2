/**
 * Tier 3 Memory - Long-term Vector Memory
 * Semantic search over embedded memories with LLM extraction
 */

import type { LongTermMemory, Project } from '@mother-harness/shared';
import { getRedisJSON, getLLMClient } from '@mother-harness/shared';
import { nanoid } from 'nanoid';

/** Memory categories */
export type MemoryCategory =
    | 'decision'     // Important decisions made
    | 'finding'      // Key research findings
    | 'preference'   // User preferences learned
    | 'process'      // Process/workflow information
    | 'entity'       // Important entities mentioned
    | 'other';

/** Memory with embedding */
// EmbeddedMemory extends LongTermMemory from shared
interface EmbeddedMemory extends LongTermMemory {
    importance: number; // 0-1 scale
    access_count: number;
    last_accessed: string;
    // We will store category in metadata.type
}

/** Extraction result from LLM */
interface ExtractionResult {
    memories: Array<{
        content: string;
        category: MemoryCategory;
        importance: number;
    }>;
}

const EXTRACTION_PROMPT = `Extract important information worth remembering from this content.
Focus on: decisions made, key findings, user preferences, process information, and important entities.

Content:
{content}

Return a JSON object:
{
  "memories": [
    {
      "content": "A concise statement of the important information",
      "category": "decision" | "finding" | "preference" | "process" | "entity",
      "importance": 0.0-1.0 (how important to remember)
    }
  ]
}

Only include truly important information. Return empty array if nothing worth remembering.`;

export class Tier3Memory {
    private redis = getRedisJSON();
    // private redisClient = getRedisClient(); // Unused
    private llm = getLLMClient();
    private readonly memoryPrefix = 'memory:';

    /**
     * Store a long-term memory with embedding
     */
    async storeMemory(
        projectId: string,
        content: string,
        metadata: {
            source: string;
            category?: MemoryCategory;
            importance?: number;
        }
    ): Promise<LongTermMemory> {
        // Generate embedding using LLM client
        const embedding = await this.generateEmbedding(content);

        const memoryId = `mem-${nanoid()}`;

        // Map category to allowed types or default to 'finding'
        const type = (['decision', 'finding', 'code', 'artifact'].includes(metadata.category ?? '')
            ? metadata.category
            : 'finding') as LongTermMemory['metadata']['type'];

        const memory: EmbeddedMemory = {
            chunk_id: memoryId,
            project_id: projectId,
            content,
            embedding,
            metadata: {
                type,
                session_id: 'unknown', // Tier 3 doesn't always know session
                timestamp: new Date().toISOString(),
                tags: [metadata.source, metadata.category ?? 'other'],
            },
            importance: metadata.importance ?? 0.5,
            access_count: 0,
            last_accessed: new Date().toISOString(),
        };

        // Store in Redis with vector
        // Store in Redis with vector
        await this.redis.set(`${this.memoryPrefix}${memory.chunk_id}`, '$', memory);

        // Update project's long-term memory reference
        await this.addToProject(projectId, memory);

        return memory;
    }

    /**
     * Retrieve relevant memories using semantic search
     */
    async retrieveRelevant(
        projectId: string,
        query: string,
        limit: number = 5
    ): Promise<LongTermMemory[]> {
        // Generate query embedding
        const queryEmbedding = await this.generateEmbedding(query);

        // Get project memories
        const project = await this.redis.get(`project:${projectId}`) as (Project & { long_term_memory?: LongTermMemory[] }) | null;
        if (!project?.long_term_memory || project.long_term_memory.length === 0) {
            return [];
        }

        // Score memories by cosine similarity
        const scoredMemories: Array<{ memory: LongTermMemory; score: number }> = [];

        for (const memRef of project.long_term_memory) {
            const fullMem = await this.redis.get(`${this.memoryPrefix}${memRef.chunk_id}`) as EmbeddedMemory | null;
            if (!fullMem?.embedding) continue;

            const score = this.cosineSimilarity(queryEmbedding, fullMem.embedding);

            // Apply importance boost
            const boostedScore = score * (0.7 + 0.3 * fullMem.importance);

            scoredMemories.push({
                memory: fullMem,
                score: boostedScore,
            });
        }

        // Sort by score and return top results
        const topMemories = scoredMemories
            .sort((a, b) => b.score - a.score)
            .slice(0, limit)
            .filter(m => m.score > 0.3) // Minimum relevance threshold
            .map(m => m.memory);

        // Update access stats for retrieved memories
        for (const mem of topMemories) {
            await this.recordAccess(mem.chunk_id);
        }

        return topMemories;
    }

    /**
     * Calculate cosine similarity between two embeddings
     */
    private cosineSimilarity(a: number[], b: number[]): number {
        if (a.length !== b.length) return 0;

        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < a.length; i++) {
            dotProduct += (a[i] ?? 0) * (b[i] ?? 0);
            normA += (a[i] ?? 0) ** 2;
            normB += (b[i] ?? 0) ** 2;
        }

        const denominator = Math.sqrt(normA) * Math.sqrt(normB);
        return denominator === 0 ? 0 : dotProduct / denominator;
    }

    /**
     * Get formatted context from long-term memory
     */
    async getContextString(projectId: string, query: string): Promise<string> {
        const memories = await this.retrieveRelevant(projectId, query);

        if (memories.length === 0) {
            return 'No relevant long-term memories.';
        }

        return 'Relevant long-term memories:\n' +
            memories
                .map((m, i) => {
                    const mem = m as EmbeddedMemory;
                    return `${i + 1}. [${mem.metadata.type}] ${m.content}`;
                })
                .join('\n');
    }

    /**
     * Extract and store important information from task results using LLM
     */
    async extractAndStore(
        projectId: string,
        taskId: string,
        content: string
    ): Promise<number> {
        // Skip if content is too short
        if (content.length < 50) {
            return 0;
        }

        // Truncate very long content
        const truncatedContent = content.length > 4000
            ? content.substring(0, 4000) + '...'
            : content;

        const prompt = EXTRACTION_PROMPT.replace('{content}', truncatedContent);

        const result = await this.llm.json<ExtractionResult>(prompt, {
            system: 'You are a memory extraction system. Extract important information concisely.',
            temperature: 0.2,
            max_tokens: 1024,
        });

        if (!result.data?.memories || result.data.memories.length === 0) {
            return 0;
        }

        // Store each extracted memory
        let stored = 0;
        for (const mem of result.data.memories.slice(0, 5)) { // Max 5 per extraction
            if (mem.content && mem.content.length > 10) {
                await this.storeMemory(projectId, mem.content, {
                    source: `task:${taskId}`,
                    category: mem.category || 'finding',
                    importance: Math.min(1, Math.max(0, mem.importance || 0.5)),
                });
                stored++;
            }
        }

        return stored;
    }

    /**
     * Generate embedding for content using LLM client
     */
    private async generateEmbedding(content: string): Promise<number[]> {
        try {
            const result = await this.llm.embed(content);
            return result.embeddings[0] ?? new Array(768).fill(0);
        } catch (error) {
            console.error('[Tier3Memory] Embedding generation failed:', error);
            return new Array(768).fill(0);
        }
    }

    /**
     * Add memory reference to project
     */
    private async addToProject(projectId: string, memory: LongTermMemory): Promise<void> {
        const project = await this.redis.get(`project:${projectId}`) as (Project & { long_term_memory?: LongTermMemory[] }) | null;
        if (!project) return;

        const memories = [...(project.long_term_memory ?? []), memory];

        // Keep only the most recent 100 references
        const trimmed = memories.slice(-100);

        await this.redis.set(`project:${projectId}`, '$.long_term_memory', trimmed);
    }

    /**
     * Record memory access for ranking
     */
    private async recordAccess(memoryId: string): Promise<void> {
        const key = `${this.memoryPrefix}${memoryId}`;
        try {
            const memory = await this.redis.get(key) as EmbeddedMemory | null;

            if (memory) {
                memory.access_count++;
                memory.last_accessed = new Date().toISOString();
                await this.redis.set(key, '$', memory);
            }
        } catch (error) {
            console.warn('[Tier3Memory] Failed to record access:', error);
        }
    }

    /**
     * Forget memories older than a threshold (with low importance)
     */
    async forget(
        projectId: string,
        options: {
            older_than_days?: number;
            importance_below?: number;
        } = {}
    ): Promise<number> {
        const threshold = options.older_than_days ?? 90;
        const importanceThreshold = options.importance_below ?? 0.3;
        const cutoff = new Date(Date.now() - threshold * 24 * 60 * 60 * 1000);

        const project = await this.redis.get(`project:${projectId}`) as (Project & { long_term_memory?: LongTermMemory[] }) | null;
        if (!project?.long_term_memory) return 0;

        let forgotten = 0;
        const kept: LongTermMemory[] = [];

        for (const mem of project.long_term_memory) {
            const memFull = await this.redis.get(`${this.memoryPrefix}${mem.chunk_id}`) as EmbeddedMemory | null;

            const isOld = new Date(mem.metadata.timestamp) < cutoff;
            const isLowImportance = (memFull?.importance ?? 1) < importanceThreshold;
            const isLowAccess = (memFull?.access_count ?? 0) < 3;

            if (isOld && isLowImportance && isLowAccess) {
                // Forget this memory
                await this.redis.del(`${this.memoryPrefix}${mem.chunk_id}`);
                forgotten++;
            } else {
                kept.push(mem);
            }
        }

        await this.redis.set(`project:${projectId}`, '$.long_term_memory', kept);

        return forgotten;
    }
}
