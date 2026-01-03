/**
 * RAG Agent
 * Retrieval-Augmented Generation - answers questions using embedded documents
 */

import type { AgentType, RetrievalChunkSummary, RetrievalReport } from '@mother-harness/shared';
import { getLLMClient, getRedisClient } from '@mother-harness/shared';
import { BaseAgent, type AgentContext, type AgentResult } from './base-agent.js';

/** Retrieved chunk */
interface RetrievedChunk {
    id: string;
    content: string;
    source: string;
    score: number;
    metadata?: Record<string, unknown>;
}

/** RAG Agent Result */
export interface RAGResult {
    answer: string;
    sources: RetrievedChunk[];
    confidence: 'high' | 'medium' | 'low';
    reasoning: string;
    related_topics: string[];
    chunks_retrieved: number;
    retrieval_report: RetrievalReport;
}

const RAG_SYSTEM_PROMPT = `You are a helpful assistant with access to a knowledge base. Your role is to:
1. Answer questions accurately using the provided context
2. Cite sources when making claims
3. Acknowledge when information isn't in the provided context
4. Synthesize information from multiple sources when relevant
5. Be clear about what you know vs what you're inferring

IMPORTANT: Only use information from the provided context. If the context doesn't contain the answer, say so clearly.`;

const RAG_PROMPT = `Answer this question using ONLY the provided context:

Question: {question}

Context from knowledge base:
{context}

Previous conversation:
{conversation}

Return a JSON object:
{
  "answer": "Your comprehensive answer based on the context",
  "sources_used": ["list of source IDs used"],
  "confidence": "high" | "medium" | "low",
  "reasoning": "brief explanation of how you derived the answer",
  "related_topics": ["topics the user might want to explore next"],
  "context_sufficient": true/false
}`;

export class RAGAgent extends BaseAgent {
    readonly agentType: AgentType = 'rag';
    private llm = getLLMClient();
    private redisClient = getRedisClient();

    private readonly QUERY_CHUNK_SIZE = 1200;
    private readonly QUERY_CHUNK_OVERLAP = 200;
    private readonly DEFAULT_TOP_K = 8;

    protected async run(inputs: string, context: AgentContext): Promise<AgentResult> {
        const startTime = Date.now();
        let totalTokens = 0;

        // Retrieve relevant chunks from the knowledge base
        const chunks = await this.retrieveChunks(inputs, context.library_ids ?? []);
        const retrievalReport = await this.buildRetrievalReport(inputs, context.library_ids ?? [], chunks);

        // Generate answer using retrieved context
        const answer = await this.generateAnswer(inputs, chunks, context);
        totalTokens += answer.tokens;

        return {
            success: answer.context_sufficient,
            outputs: {
                answer: answer.answer,
                sources: chunks.filter(c => answer.sources_used.includes(c.id)),
                confidence: answer.confidence,
                reasoning: answer.reasoning,
                related_topics: answer.related_topics,
                chunks_retrieved: chunks.length,
                retrieval_report: retrievalReport,
            },
            explanation: answer.context_sufficient
                ? `Answered using ${answer.sources_used.length} sources (${answer.confidence} confidence)`
                : 'Insufficient context in knowledge base',
            sources: chunks.map(c => c.source),
            tokens_used: totalTokens,
            duration_ms: Date.now() - startTime,
        };
    }

    private async retrieveChunks(
        query: string,
        libraryIds: string[]
    ): Promise<RetrievedChunk[]> {
        const queryChunks = this.chunkQuery(query);
        if (queryChunks.length === 0) {
            return [];
        }
        const queryEmbedding = await this.llm.embed(queryChunks);

        if (queryEmbedding.embeddings.length === 0 ||
            queryEmbedding.embeddings.every(vec => vec.every(v => v === 0))) {
            console.warn('[RAGAgent] Failed to generate query embedding');
            return [];
        }

        const retrieved = await this.searchEmbeddings(
            queryEmbedding.embeddings,
            libraryIds,
            this.DEFAULT_TOP_K
        );

        return retrieved.map(chunk => ({
            id: chunk.chunk_id,
            content: chunk.content ?? chunk.content_preview,
            source: chunk.file_path || chunk.document_name || chunk.library_id,
            score: chunk.score,
            metadata: {
                document_id: chunk.document_id,
                page_number: chunk.page_number,
                chunk_index: chunk.chunk_index,
                library_id: chunk.library_id,
                document_name: chunk.document_name,
                file_path: chunk.file_path,
            },
        }));
    }

    private async generateAnswer(
        question: string,
        chunks: RetrievedChunk[],
        context: AgentContext
    ): Promise<{
        answer: string;
        sources_used: string[];
        confidence: 'high' | 'medium' | 'low';
        reasoning: string;
        related_topics: string[];
        context_sufficient: boolean;
        tokens: number;
    }> {
        const chunkContext = chunks.length > 0
            ? chunks.map((c, i) => `[Source ${i + 1}: ${c.id}]\n${c.content}`).join('\n\n')
            : 'No relevant context found in the knowledge base.';

        const prompt = RAG_PROMPT
            .replace('{question}', question)
            .replace('{context}', chunkContext)
            .replace('{conversation}', context.recent_context ?? 'No previous conversation');

        const result = await this.llm.json<{
            answer: string;
            sources_used: string[];
            confidence: 'high' | 'medium' | 'low';
            reasoning: string;
            related_topics: string[];
            context_sufficient: boolean;
        }>(prompt, {
            system: RAG_SYSTEM_PROMPT,
            temperature: 0.3,
            max_tokens: 2048,
        });

        if (result.data) {
            return {
                ...result.data,
                tokens: result.raw.tokens_used.total,
            };
        }

        return {
            answer: 'Unable to generate an answer.',
            sources_used: [],
            confidence: 'low',
            reasoning: 'Answer generation failed.',
            related_topics: [],
            context_sufficient: false,
            tokens: result.raw.tokens_used.total,
        };
    }

    private chunkQuery(query: string): string[] {
        const trimmed = query.trim();
        if (!trimmed) return [];

        const chunks: string[] = [];
        let index = 0;

        while (index < trimmed.length) {
            const end = Math.min(trimmed.length, index + this.QUERY_CHUNK_SIZE);
            const chunk = trimmed.slice(index, end).trim();
            if (chunk) {
                chunks.push(chunk);
            }
            if (end >= trimmed.length) break;
            index = end - this.QUERY_CHUNK_OVERLAP;
        }

        return chunks;
    }

    private async searchEmbeddings(
        embeddings: number[][],
        libraryIds: string[],
        topK: number
    ): Promise<RetrievalChunkSummary[]> {
        if (embeddings.length === 0) {
            return [];
        }

        const aggregated = new Map<string, RetrievalChunkSummary>();

        for (const embedding of embeddings) {
            if (embedding.every(value => value === 0)) {
                continue;
            }

            const results = await this.vectorSearch(embedding, libraryIds, topK);
            for (const result of results) {
                const existing = aggregated.get(result.chunk_id);
                if (!existing || result.score > existing.score) {
                    aggregated.set(result.chunk_id, result);
                }
            }
        }

        return Array.from(aggregated.values())
            .sort((a, b) => b.score - a.score)
            .slice(0, topK);
    }

    private async vectorSearch(
        embedding: number[],
        libraryIds: string[],
        topK: number
    ): Promise<RetrievalChunkSummary[]> {
        const libraryFilter = libraryIds.length
            ? `@library:{${libraryIds.join('|')}}`
            : '*';
        const queryFilter = `${libraryFilter} @searchable:{true}`;

        const vector = Buffer.from(Float32Array.from(embedding).buffer);

        const returnFields = [
            'score',
            '$.id',
            '$.library',
            '$.document_id',
            '$.document_name',
            '$.file_path',
            '$.page_number',
            '$.chunk_index',
            '$.content',
        ];

        try {
            const response = await this.redisClient.call(
                'FT.SEARCH',
                'idx:chunks',
                `${queryFilter}=>[KNN ${topK} @embedding $vector AS score]`,
                'PARAMS',
                '2',
                'vector',
                vector,
                'RETURN',
                returnFields.length.toString(),
                ...returnFields,
                'SORTBY',
                'score',
                'ASC',
                'DIALECT',
                '2'
            ) as Array<string | number | Array<string>>;

            return this.parseSearchResults(response);
        } catch (error) {
            console.warn('[RAGAgent] Vector search failed:', error);
            return [];
        }
    }

    private parseSearchResults(
        response: Array<string | number | Array<string>>
    ): RetrievalChunkSummary[] {
        const results: RetrievalChunkSummary[] = [];
        if (response.length < 2) {
            return results;
        }

        for (let i = 1; i < response.length; i += 2) {
            const docId = response[i] as string;
            const fields = response[i + 1] as string[] | undefined;
            if (!fields) continue;

            const fieldMap = this.parseFieldMap(fields);
            const rawScore = Number(fieldMap.score ?? 0);
            const similarity = Number.isFinite(rawScore) ? 1 - rawScore : 0;
            const chunkId = (fieldMap['$.id'] as string | undefined) ?? docId;

            results.push({
                chunk_id: chunkId,
                library_id: (fieldMap['$.library'] as unknown as string) ?? '',
                document_id: (fieldMap['$.document_id'] as unknown as string) ?? '',
                document_name: (fieldMap['$.document_name'] as unknown as string) ?? '',
                file_path: (fieldMap['$.file_path'] as unknown as string) ?? '',
                ...(fieldMap['$.page_number'] !== undefined && { page_number: fieldMap['$.page_number'] as number }),
                ...(fieldMap['$.chunk_index'] !== undefined && { chunk_index: fieldMap['$.chunk_index'] as number }),
                score: similarity,
                content_preview: this.previewContent(fieldMap['$.content'] as unknown as string | undefined),
                ...(fieldMap['$.content'] !== undefined && { content: fieldMap['$.content'] as unknown as string }),
            });
        }

        return results;
    }

    private parseFieldMap(fields: string[]): Record<string, string | number> {
        const fieldMap: Record<string, string | number> = {};
        for (let i = 0; i < fields.length; i += 2) {
            const key = fields[i] as string;
            const value = fields[i + 1] as string | undefined;
            if (!key) continue;
            fieldMap[key] = this.parseJsonField(value);
        }
        return fieldMap;
    }

    private parseJsonField(value?: string): string | number {
        if (value === undefined) return '';
        try {
            const parsed = JSON.parse(value);
            if (typeof parsed === 'string' || typeof parsed === 'number') {
                return parsed;
            }
            return JSON.stringify(parsed);
        } catch {
            const numeric = Number(value);
            return Number.isFinite(numeric) ? numeric : value;
        }
    }

    private previewContent(content?: string): string {
        if (!content) return '';
        const trimmed = content.trim();
        return trimmed.length > 240 ? `${trimmed.slice(0, 240)}…` : trimmed;
    }

    private async buildRetrievalReport(
        query: string,
        libraryIds: string[],
        chunks: RetrievedChunk[]
    ): Promise<RetrievalReport> {
        const queryChunks = this.chunkQuery(query);
        const embeddingResult = queryChunks.length > 0
            ? await this.llm.embed(queryChunks)
            : { model: 'unknown', embeddings: [], dimensions: 0 };

        return {
            query,
            query_chunks: queryChunks,
            library_ids: libraryIds,
            embedding_model: embeddingResult.model,
            requested_k: this.DEFAULT_TOP_K,
            retrieved_at: new Date().toISOString(),
            results: chunks.map((chunk) => {
                const pageNumber = chunk.metadata?.page_number as number | undefined;
                const chunkIndex = chunk.metadata?.chunk_index as number | undefined;

                return {
                    chunk_id: chunk.id,
                    library_id: (chunk.metadata?.library_id as string) ?? '',
                    document_id: (chunk.metadata?.document_id as string) ?? '',
                    document_name: (chunk.metadata?.document_name as string) ?? chunk.source,
                    file_path: (chunk.metadata?.file_path as string) ?? chunk.source,
                    ...(pageNumber !== undefined && { page_number: pageNumber }),
                    ...(chunkIndex !== undefined && { chunk_index: chunkIndex }),
                    score: chunk.score,
                    content_preview: this.previewContent(chunk.content),
                };
            }),
        };
    }
}
