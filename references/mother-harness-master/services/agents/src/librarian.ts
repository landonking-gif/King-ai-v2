/**
 * Librarian Agent
 * Manages document libraries, ingestion, and organization
 */

import type { AgentType, DoclingJob, Library, RetrievalChunkSummary, RetrievalReport } from '@mother-harness/shared';
import { getLLMClient, getRedisClient } from '@mother-harness/shared';
import { BaseAgent, type AgentContext, type AgentResult } from './base-agent.js';
import { nanoid } from 'nanoid';

/** Librarian command types */
type LibrarianCommand = 'create_library' | 'add_document' | 'search' | 'organize' | 'status';

/** Ingestion Report */
export interface IngestionReport {
    command: LibrarianCommand;
    success: boolean;
    message: string;
    library: string;
    document_count: number;
    timestamp: string;
}

const LIBRARIAN_SYSTEM_PROMPT = `You are a knowledge management specialist. Your role is to:
1. Organize and categorize documents effectively
2. Create meaningful tags and metadata
3. Ensure documents are easily discoverable
4. Maintain library structure and organization
5. Help users find relevant information

Be systematic and consistent in your organization approach.`;

const PARSE_COMMAND_PROMPT = `Parse this librarian command:

Input: {input}

Return a JSON object:
{
  "command": "create_library" | "add_document" | "search" | "organize" | "status",
  "library_name": "name if creating/targeting a library",
  "documents": ["paths to documents if adding"],
  "search_query": "search query if searching",
  "metadata": { "any additional metadata" }
}`;

const ORGANIZE_PROMPT = `Suggest organization for these documents:

Documents: {documents}
Current Library: {library}

Return a JSON object:
{
  "suggested_tags": ["array of suggested tags"],
  "suggested_categories": ["array of categories"],
  "document_assignments": [
    { "document": "doc name", "tags": ["tag1"], "category": "category" }
  ],
  "summary": "brief summary of the organization scheme"
}`;

export class LibrarianAgent extends BaseAgent {
    readonly agentType: AgentType = 'librarian';
    private llm = getLLMClient();
    private redisClient = getRedisClient();

    private readonly QUERY_CHUNK_SIZE = 1200;
    private readonly QUERY_CHUNK_OVERLAP = 200;
    private readonly DEFAULT_TOP_K = 8;

    protected async run(inputs: string, context: AgentContext): Promise<AgentResult> {
        const startTime = Date.now();
        let totalTokens = 0;

        // Parse the command
        const command = await this.parseCommand(inputs);
        totalTokens += command.tokens;

        // Execute the command
        const result = await this.executeCommand(command, context);
        totalTokens += result.tokens;

        const ingestionReport = {
            command: command.command,
            success: result.success,
            message: result.message,
            library: command.library_name ?? context.library_ids?.[0] ?? 'default',
            document_count: Array.isArray(command.documents) ? command.documents.length : 0,
            timestamp: new Date().toISOString(),
        };

        return {
            success: result.success,
            outputs: {
                ...result.data,
                ingestion_report: ingestionReport,
            },
            explanation: result.message,
            tokens_used: totalTokens,
            duration_ms: Date.now() - startTime,
        };
    }

    private async parseCommand(inputs: string): Promise<{
        command: LibrarianCommand;
        library_name?: string;
        documents?: string[];
        search_query?: string;
        metadata?: Record<string, unknown>;
        tokens: number;
    }> {
        const prompt = PARSE_COMMAND_PROMPT.replace('{input}', inputs);

        const result = await this.llm.json<{
            command: LibrarianCommand;
            library_name?: string;
            documents?: string[];
            search_query?: string;
            metadata?: Record<string, unknown>;
        }>(prompt, {
            system: LIBRARIAN_SYSTEM_PROMPT,
            temperature: 0.2,
        });

        if (result.data) {
            return {
                ...result.data,
                tokens: result.raw.tokens_used.total,
            };
        }

        // Default to status if parsing fails
        return {
            command: 'status',
            tokens: result.raw.tokens_used.total,
        };
    }

    private async executeCommand(
        command: Awaited<ReturnType<typeof this.parseCommand>>,
        context: AgentContext
    ): Promise<{
        success: boolean;
        data: Record<string, unknown>;
        message: string;
        tokens: number;
    }> {
        switch (command.command) {
            case 'create_library':
                return this.createLibrary(command.library_name ?? 'Untitled Library', command.metadata);

            case 'add_document':
                return this.addDocuments(
                    command.library_name ?? context.library_ids?.[0] ?? 'default',
                    command.documents ?? []
                );

            case 'search':
                return this.searchLibrary(
                    command.library_name ?? context.library_ids?.[0],
                    command.search_query ?? ''
                );

            case 'organize':
                return this.organizeLibrary(command.library_name ?? context.library_ids?.[0]);

            case 'status':
            default:
                return this.getStatus(context.library_ids);
        }
    }

    private async createLibrary(
        name: string,
        metadata?: Record<string, unknown>
    ): Promise<{ success: boolean; data: Record<string, unknown>; message: string; tokens: number }> {
        const libraryId = `lib-${nanoid(10)}`;
        const now = new Date().toISOString();
        const library: Library = {
            id: libraryId,
            name,
            description: (metadata?.description as string) ?? '',
            folder_path: (metadata?.folder_path as string) ?? `/libraries/${libraryId}`,
            auto_scan: false,
            document_count: 0,
            chunk_count: 0,
            total_size_bytes: 0,
            last_scanned: now,
            scan_status: 'idle',
            created_at: now,
            updated_at: now,
        };

        await this.redis.set(`library:${libraryId}`, '$', library);

        return {
            success: true,
            data: { library },
            message: `Created library "${name}" with ID ${libraryId}`,
            tokens: 0,
        };
    }

    private async addDocuments(
        libraryId: string,
        documents: string[]
    ): Promise<{ success: boolean; data: Record<string, unknown>; message: string; tokens: number }> {
        if (documents.length === 0) {
            return {
                success: false,
                data: {},
                message: 'No documents specified',
                tokens: 0,
            };
        }

        const library = await this.redis.get(`library:${libraryId}`) as Library | null;
        const libraryName = library?.name ?? libraryId;

        // Queue documents for processing
        const jobs: DoclingJob[] = documents.map(doc => ({
            id: `job-${nanoid(10)}`,
            library_id: libraryId,
            library_name: libraryName,
            file_path: doc,
            operation: 'ingest',
            priority: 'normal',
            status: 'pending',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
        }));

        for (const job of jobs) {
            await this.redis.set(`docling_job:${job.id}`, '$', job);
            await this.redisClient.xadd(
                'stream:docling',
                '*',
                'job',
                JSON.stringify(job)
            );
        }

        return {
            success: true,
            data: { jobs, count: jobs.length },
            message: `Queued ${jobs.length} document(s) for processing`,
            tokens: 0,
        };
    }

    private async searchLibrary(
        libraryId: string | undefined,
        query: string
    ): Promise<{ success: boolean; data: Record<string, unknown>; message: string; tokens: number }> {
        if (!libraryId) {
            return {
                success: false,
                data: {},
                message: 'No library specified',
                tokens: 0,
            };
        }

        const queryChunks = this.chunkQuery(query);
        if (queryChunks.length === 0) {
            return {
                success: false,
                data: {},
                message: 'Search query is empty',
                tokens: 0,
            };
        }
        const embeddingResult = await this.llm.embed(queryChunks);
        const retrieved = await this.searchEmbeddings(
            embeddingResult.embeddings,
            [libraryId],
            this.DEFAULT_TOP_K
        );

        const report: RetrievalReport = {
            query,
            query_chunks: queryChunks,
            library_ids: [libraryId],
            embedding_model: embeddingResult.model,
            requested_k: this.DEFAULT_TOP_K,
            retrieved_at: new Date().toISOString(),
            results: retrieved,
        };

        return {
            success: true,
            data: {
                query,
                library_id: libraryId,
                results: retrieved,
                retrieval_report: report,
            },
            message: `Searched library ${libraryId} for "${query}"`,
            tokens: 0,
        };
    }

    private async organizeLibrary(
        libraryId: string | undefined
    ): Promise<{ success: boolean; data: Record<string, unknown>; message: string; tokens: number }> {
        if (!libraryId) {
            return {
                success: false,
                data: {},
                message: 'No library specified',
                tokens: 0,
            };
        }

        const library = await this.redis.get(`library:${libraryId}`) as Library | null;
        if (!library) {
            return {
                success: false,
                data: {},
                message: `Library ${libraryId} not found`,
                tokens: 0,
            };
        }

        const prompt = ORGANIZE_PROMPT
            .replace('{documents}', `${library.document_count} documents`)
            .replace('{library}', library.name);

        const result = await this.llm.json<{
            suggested_tags: string[];
            suggested_categories: string[];
            summary: string;
        }>(prompt, {
            system: LIBRARIAN_SYSTEM_PROMPT,
            temperature: 0.4,
        });

        return {
            success: true,
            data: result.data ?? {},
            message: `Organization suggestions for ${library.name}`,
            tokens: result.raw.tokens_used.total,
        };
    }

    private async getStatus(
        libraryIds?: string[]
    ): Promise<{ success: boolean; data: Record<string, unknown>; message: string; tokens: number }> {
        const libraries: Library[] = [];

        if (libraryIds && libraryIds.length > 0) {
            for (const id of libraryIds) {
                const lib = await this.redis.get(`library:${id}`) as Library | null;
                if (lib) libraries.push(lib);
            }
        }

        return {
            success: true,
            data: {
                libraries,
                total_count: libraries.length,
            },
            message: `Found ${libraries.length} library(ies)`,
            tokens: 0,
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
            console.warn('[Librarian] Vector search failed:', error);
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
}
