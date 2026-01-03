/**
 * Document Processor
 * Handles document extraction, chunking, and embedding
 */

import type { DocumentChunk, Library, DoclingJob } from '@mother-harness/shared';
import { getRedisJSON, getLLMClient } from '@mother-harness/shared';
import { nanoid } from 'nanoid';
import pdfParse from 'pdf-parse';
import * as path from 'path';
import * as fs from 'fs/promises';
import * as crypto from 'crypto';

/** Docling API configuration */
interface DoclingConfig {
    apiUrl: string;
    timeout: number;
    maxRetries: number;
    embeddingModel: string;
}

/** Extraction result from document processing */
interface ExtractionResult {
    text: string;
    pages: Array<{
        page_number: number;
        content: string;
        tables?: Array<{ content: string; caption?: string }>;
        images?: Array<{ path: string; caption?: string }>;
    }>;
    metadata: {
        title?: string;
        author?: string;
        created_date?: string;
        page_count: number;
        file_type: string;
        file_size: number;
    };
}

/** Chunk with embedding */
interface ProcessedChunk {
    content: string;
    page_number?: number;
    section_title?: string;
    hierarchy: string[];
    images: DocumentChunk['images'];
    tables: DocumentChunk['tables'];
}

export class DocumentProcessor {
    private redis = getRedisJSON();
    private llm = getLLMClient();
    private config: DoclingConfig;

    /** Chunking parameters */
    private readonly CHUNK_SIZE = 450; // tokens (approx 4 chars per token)
    private readonly CHUNK_OVERLAP = 80;
    private readonly CHARS_PER_TOKEN = 4;

    /** Supported file types */
    private readonly SUPPORTED_EXTENSIONS = [
        '.txt', '.md', '.markdown',
        '.json', '.yaml', '.yml',
        '.ts', '.js', '.py', '.go', '.rs', '.java', '.c', '.cpp', '.h',
        '.html', '.css', '.xml',
        '.csv', '.tsv',
    ];

    constructor(config?: Partial<DoclingConfig>) {
        this.config = {
            apiUrl: config?.apiUrl ?? process.env['DOCLING_API_URL'] ?? 'http://localhost:8000',
            timeout: config?.timeout ?? 300000, // 5 minutes
            maxRetries: config?.maxRetries ?? 3,
            embeddingModel: config?.embeddingModel ?? 'nomic-embed-text',
        };
    }

    /**
     * Process a document ingestion job
     */
    async processJob(job: DoclingJob): Promise<{ success: boolean; chunks: number; error?: string }> {
        const startTime = Date.now();

        try {
            // Update job status to processing
            await this.updateJobStatus(job.id, 'processing', { started_at: new Date().toISOString() });

            if (job.operation === 'delete') {
                const removedChunks = await this.deleteDocumentChunks(job.library_id, job.file_path);
                await this.updateLibraryStats(job.library_id, -removedChunks, -1);
                await this.updateJobStatus(job.id, 'completed', {
                    duration_ms: Date.now() - startTime,
                    completed_at: new Date().toISOString(),
                });
                return { success: true, chunks: 0 };
            }

            // Extract content from document
            const extraction = await this.extractDocument(job.file_path);

            // Chunk the content
            const chunks = this.chunkDocument(extraction, job);

            if (chunks.length === 0) {
                throw new Error('No content extracted from document');
            }

            // Generate embeddings using LLM client
            const chunksWithEmbeddings = await this.generateEmbeddings(chunks);

            const removedChunks = job.operation === 'update'
                ? await this.deleteDocumentChunks(job.library_id, job.file_path)
                : 0;

            // Store chunks in Redis
            await this.storeChunks(job.library_id, job.file_path, extraction, chunksWithEmbeddings);

            // Update library stats
            const documentDelta = job.operation === 'ingest' ? 1 : 0;
            const chunkDelta = chunks.length - removedChunks;
            await this.updateLibraryStats(job.library_id, chunkDelta, documentDelta);

            // Update job status to completed
            await this.updateJobStatus(job.id, 'completed', {
                chunks_created: chunks.length,
                duration_ms: Date.now() - startTime,
                completed_at: new Date().toISOString(),
            });

            console.log(`[Processor] Successfully processed ${job.file_path}: ${chunks.length} chunks`);
            return { success: true, chunks: chunks.length };

        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';

            // Update job status to failed
            await this.updateJobStatus(job.id, 'failed', {
                error: errorMessage,
                completed_at: new Date().toISOString(),
            });

            // Handle failure (move to failed folder, alert)
            await this.handleFailure(job, errorMessage);

            return { success: false, chunks: 0, error: errorMessage };
        }
    }

    /**
     * Extract content from document
     */
    private async extractDocument(filePath: string): Promise<ExtractionResult> {
        const resolvedPath = path.resolve(filePath);
        const extension = path.extname(filePath).toLowerCase();


        // Check if file exists
        let stats: Awaited<ReturnType<typeof fs.stat>>;
        try {
            stats = await fs.stat(resolvedPath);
        } catch {
            throw new Error(`File not found: ${filePath}`);
        }

        // Check file size (max 10MB)
        if (stats.size > 10 * 1024 * 1024) {
            throw new Error('File too large (max 10MB)');
        }

        // For text-based files, read directly
        if (this.SUPPORTED_EXTENSIONS.includes(extension)) {
            const content = await fs.readFile(resolvedPath, 'utf-8');

            return {
                text: content,
                pages: [{
                    page_number: 1,
                    content: content,
                }],
                metadata: {
                    title: path.basename(filePath, extension),
                    page_count: 1,
                    file_type: extension.slice(1),
                    file_size: Number(stats.size),
                },
            };
        }

        // For PDFs and other binary formats, try Docling API then fallback to local parsing
        if (extension === '.pdf') {
            return this.extractPdf(resolvedPath, stats);
        }

        // Fallback: try to read as text
        try {
            const content = await fs.readFile(resolvedPath, 'utf-8');
            return {
                text: content,
                pages: [{ page_number: 1, content }],
                metadata: {
                    title: path.basename(filePath, extension),
                    page_count: 1,
                    file_type: extension.slice(1) || 'unknown',
                    file_size: Number(stats.size),
                },
            };
        } catch {
            throw new Error(`Unsupported file type: ${extension}`);
        }
    }

    /**
     * Extract PDF using Docling API
     */
    private async extractWithDoclingApi(
        filePath: string,
        stats: Awaited<ReturnType<typeof fs.stat>>
    ): Promise<ExtractionResult> {
        try {
            const fileBuffer = await fs.readFile(filePath);
            const formData = new FormData();
            formData.append('file', new Blob([fileBuffer]), path.basename(filePath));

            const response = await fetch(`${this.config.apiUrl}/convert`, {
                method: 'POST',
                body: formData,
                signal: AbortSignal.timeout(this.config.timeout),
            });

            if (!response.ok) {
                throw new Error(`Docling API error: ${response.status}`);
            }

            const result = await response.json() as {
                pages?: Array<{ text: string; tables?: Array<{ content: string }> }>;
                metadata?: { title?: string; author?: string };
            };

            const pages = (result.pages ?? []).map((p, i) => ({
                page_number: i + 1,
                content: p.text ?? '',
                tables: p.tables?.map(t => ({ content: t.content })),
            }));

            return {
                text: pages.map(p => p.content).join('\n\n'),
                pages,
                metadata: {
                    title: result.metadata?.title ?? path.basename(filePath, '.pdf'),
                    author: result.metadata?.author,
                    page_count: pages.length,
                    file_type: 'pdf',
                    file_size: Number(stats.size),
                },
            } as ExtractionResult;
        } catch (error) {
            // If Docling API fails, throw error
            const message = error instanceof Error ? error.message : 'Unknown error';
            throw new Error(`PDF extraction failed: ${message}. Ensure Docling API is running.`);
        }
    }

    /**
     * Extract PDF using Docling API with fallback to local parsing
     */
    private async extractPdf(
        filePath: string,
        stats: Awaited<ReturnType<typeof fs.stat>>
    ): Promise<ExtractionResult> {
        try {
            return await this.extractWithDoclingApi(filePath, stats);
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Unknown error';
            console.warn(`[Processor] Docling API failed for ${filePath}: ${message}. Falling back to local PDF parsing.`);
            return this.extractWithPdfParse(filePath, stats);
        }
    }

    /**
     * Extract PDF using local pdf-parse fallback
     */
    private async extractWithPdfParse(
        filePath: string,
        stats: Awaited<ReturnType<typeof fs.stat>>
    ): Promise<ExtractionResult> {
        const fileBuffer = await fs.readFile(filePath);
        const parsed = await pdfParse(fileBuffer);
        const rawText = parsed.text?.trim() ?? '';
        const pageTexts = rawText.split('\f').map(page => page.trim()).filter(Boolean);
        const resolvedPages = pageTexts.length > 0 ? pageTexts : (rawText ? [rawText] : ['']);

        const pages = resolvedPages.map((content, index) => ({
            page_number: index + 1,
            content,
        }));

        return {
            text: pages.map(p => p.content).join('\n\n'),
            pages,
            metadata: {
                title: parsed.info?.Title ?? path.basename(filePath, '.pdf'),
                author: parsed.info?.Author,
                page_count: parsed.numpages ?? pages.length,
                file_type: 'pdf',
                file_size: Number(stats.size),
            },
        };
    }

    /**
     * Chunk document content using overlapping windows
     */
    private chunkDocument(extraction: ExtractionResult, _job: DoclingJob): ProcessedChunk[] {
        const chunks: ProcessedChunk[] = [];
        const chunkSize = this.CHUNK_SIZE * this.CHARS_PER_TOKEN;
        const overlap = this.CHUNK_OVERLAP * this.CHARS_PER_TOKEN;

        for (const page of extraction.pages) {
            const content = page.content.trim();
            if (!content) continue;

            // Split by paragraphs first for better semantic boundaries
            const paragraphs = content.split(/\n\n+/);
            let buffer = '';

            for (const para of paragraphs) {
                if (buffer.length + para.length < chunkSize) {
                    buffer += (buffer ? '\n\n' : '') + para;
                } else {
                    // Save current buffer as chunk
                    if (buffer.trim()) {
                        chunks.push(this.createChunk(buffer, page, extraction));
                    }

                    // Start new buffer with overlap from previous
                    const overlapText = buffer.slice(-overlap);
                    buffer = overlapText + (overlapText ? '\n\n' : '') + para;
                }
            }

            // Don't forget the last buffer
            if (buffer.trim()) {
                chunks.push(this.createChunk(buffer, page, extraction));
            }
        }

        return chunks;
    }

    private createChunk(
        content: string,
        page: ExtractionResult['pages'][0],
        extraction: ExtractionResult
    ): ProcessedChunk {
        return {
            content,
            page_number: page.page_number,
            hierarchy: extraction.metadata.title ? [extraction.metadata.title] : [],
            images: (page.images ?? []).map(img => ({
                id: `img-${nanoid()}`,
                file_path: img.path,
                ...(img.caption !== undefined && { caption: img.caption }),
                page_number: page.page_number,
            })),
            tables: (page.tables ?? []).map(tbl => ({
                id: `tbl-${nanoid()}`,
                content: tbl.content,
                ...(tbl.caption !== undefined && { caption: tbl.caption }),
                page_number: page.page_number,
            })),
        };
    }

    /**
     * Generate embeddings for chunks using LLM client
     */
    private async generateEmbeddings(
        chunks: ProcessedChunk[]
    ): Promise<Array<ProcessedChunk & { embedding: number[] }>> {
        const results: Array<ProcessedChunk & { embedding: number[] }> = [];

        // Process in batches of 10
        const batchSize = 10;
        for (let i = 0; i < chunks.length; i += batchSize) {
            const batch = chunks.slice(i, i + batchSize);
            const texts = batch.map(c => c.content);

            try {
                const embedResult = await this.llm.embed(texts, this.config.embeddingModel);

                for (let j = 0; j < batch.length; j++) {
                    results.push({
                        ...batch[j]!,
                        embedding: embedResult.embeddings[j] ?? new Array(768).fill(0),
                    });
                }
            } catch (error) {
                console.error('[Processor] Embedding generation failed:', error);
                // Use zero vectors as fallback
                for (const chunk of batch) {
                    results.push({
                        ...chunk,
                        embedding: new Array(768).fill(0),
                    });
                }
            }
        }

        return results;
    }

    /**
     * Store chunks in Redis with proper indexing
     */
    private async storeChunks(
        libraryId: string,
        filePath: string,
        extraction: ExtractionResult,
        chunks: Array<ProcessedChunk & { embedding: number[] }>
    ): Promise<void> {
        const documentId = `doc-${nanoid()}`;
        const contentHash = this.hashContent(extraction.text);

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i]!;
            const chunkId = `chunk-${nanoid()}`;

            const docChunk: DocumentChunk = {
                id: chunkId,
                library: libraryId,
                document_id: documentId,
                document_name: extraction.metadata.title ?? path.basename(filePath),
                file_path: filePath,
                content: chunk.content,
                embedding: chunk.embedding,
                images: chunk.images,
                tables: chunk.tables,

                ...(chunk.page_number !== undefined && { page_number: chunk.page_number }),
                ...(chunk.section_title !== undefined && { section_title: chunk.section_title }),
                hierarchy: chunk.hierarchy,
                chunk_type: 'text',
                chunk_index: i,
                total_chunks: chunks.length,
                content_hash: contentHash,
                indexed_at: new Date().toISOString(),
                source_modified_at: new Date().toISOString(),
                searchable: chunk.embedding.some(v => v !== 0),
            };

            // Store chunk
            await this.redis.set(`chunk:${libraryId}:${chunkId}`, '$', docChunk);
        }

        // Store document metadata
        await this.redis.set(`document:${documentId}`, '$', {
            id: documentId,
            library_id: libraryId,
            file_path: filePath,
            title: extraction.metadata.title,
            author: extraction.metadata.author,
            file_type: extraction.metadata.file_type,
            file_size: extraction.metadata.file_size,
            page_count: extraction.metadata.page_count,
            chunk_count: chunks.length,
            content_hash: contentHash,
            indexed_at: new Date().toISOString(),
        });
    }

    /**
     * Update library statistics
     */
    private async updateLibraryStats(
        libraryId: string,
        chunkDelta: number,
        documentDelta: number
    ): Promise<void> {
        try {
            const library = await this.redis.get(`library:${libraryId}`) as Library | null;
            if (library) {
                await this.redis.set(
                    `library:${libraryId}`,
                    '$.document_count',
                    Math.max(0, library.document_count + documentDelta)
                );
                await this.redis.set(
                    `library:${libraryId}`,
                    '$.chunk_count',
                    Math.max(0, library.chunk_count + chunkDelta)
                );
                await this.redis.set(`library:${libraryId}`, '$.last_scanned', new Date().toISOString());
            }
        } catch (error) {
            console.warn('[Processor] Failed to update library stats:', error);
        }
    }

    /**
     * Update job status
     */
    private async updateJobStatus(
        jobId: string,
        status: DoclingJob['status'],
        metadata?: Record<string, unknown>
    ): Promise<void> {
        const key = `docling_job:${jobId}`;
        try {
            if (!(await this.redis.exists(key))) {
                await this.redis.set(key, '$', {
                    id: jobId,
                    status,
                    updated_at: new Date().toISOString(),
                    ...(metadata ?? {}),
                });
                return;
            }
            await this.redis.set(key, '$.status', status);
            await this.redis.set(key, '$.updated_at', new Date().toISOString());

            if (metadata) {
                for (const [k, v] of Object.entries(metadata)) {
                    await this.redis.set(key, `$.${k}`, v);
                }
            }
        } catch (error) {
            console.warn('[Processor] Failed to update job status:', error);
        }
    }

    private async deleteDocumentChunks(libraryId: string, filePath: string): Promise<number> {
        const chunkKeys = await this.redis.keys(`chunk:${libraryId}:*`);
        let removed = 0;
        for (const key of chunkKeys) {
            const chunk = await this.redis.get<{ file_path?: string }>(key);
            if (chunk?.file_path === filePath) {
                await this.redis.del(key);
                removed += 1;
            }
        }
        return removed;
    }

    /**
     * Handle processing failure
     */
    private async handleFailure(job: DoclingJob, error: string): Promise<void> {
        console.error(`[Processor] Job ${job.id} failed: ${error}`);

        // Move file to _failed folder if it exists
        try {
            const failedDir = path.join(path.dirname(job.file_path), '_failed');
            await fs.mkdir(failedDir, { recursive: true });

            const failedPath = path.join(failedDir, path.basename(job.file_path));
            await fs.rename(job.file_path, failedPath);

            // Write error log
            const logPath = failedPath + '.error.txt';
            await fs.writeFile(logPath, `Error: ${error}\nTimestamp: ${new Date().toISOString()}\nJob ID: ${job.id}`);

            console.log(`[Processor] Moved failed file to: ${failedPath}`);
        } catch (moveError) {
            console.warn('[Processor] Could not move failed file:', moveError);
        }
    }

    /**
     * Generate content hash for deduplication
     */
    private hashContent(content: string): string {
        return crypto.createHash('sha256').update(content).digest('hex').substring(0, 16);
    }
}

// Singleton
let processorInstance: DocumentProcessor | null = null;

export function getDocumentProcessor(): DocumentProcessor {
    if (!processorInstance) {
        processorInstance = new DocumentProcessor();
    }
    return processorInstance;
}
