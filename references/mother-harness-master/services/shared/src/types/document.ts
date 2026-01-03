/**
 * Document Types and Schemas
 * RAG document library structures for vector search
 */

/** Document chunk types */
export type ChunkType = 'text' | 'table' | 'figure' | 'code';

/** Library scan status */
export type ScanStatus = 'idle' | 'scanning' | 'processing';

/** Image reference extracted from document */
export interface ImageReference {
    id: string;
    file_path: string;             // /core4/libraries/_images/{hash}.png
    caption?: string;
    page_number?: number;
}

/** Table reference extracted from document */
export interface TableReference {
    id: string;
    content: string;               // Markdown table content
    caption?: string;
    page_number?: number;
}

/** Document chunk - stored in Redis as chunk:{libraryId}:{id} */
export interface DocumentChunk {
    id: string;                    // chunk-uuid
    library: string;               // Library ID this belongs to
    document_id: string;
    document_name: string;
    file_path: string;             // Original file path

    // Content
    content: string;
    embedding: number[];           // 768-dim vector

    // Multimodal content
    images: ImageReference[];
    tables: TableReference[];

    // Metadata from Docling
    page_number?: number;
    section_title?: string;
    hierarchy: string[];           // e.g., ['Chapter 1', 'Section 1.1']
    chunk_type: ChunkType;
    chunk_index: number;
    total_chunks: number;
    content_hash: string;

    // Timestamps
    indexed_at: string;
    source_modified_at: string;

    // Flags
    searchable: boolean;           // False if embedding failed
}

/** Document library - stored in Redis as library:{id} */
export interface Library {
    id: string;
    name: string;                  // Display name
    folder_path: string;           // /core4/libraries/coding
    description?: string;

    // Stats
    document_count: number;
    chunk_count: number;
    total_size_bytes: number;
    last_scanned: string;
    scan_status: ScanStatus;
    processed_count?: number;      // For progress tracking
    total_files?: number;

    // Configuration
    auto_scan: boolean;            // File watcher enabled
    scan_schedule?: string;        // Cron: '0 2 * * *' (2am daily)

    // Timestamps
    created_at: string;
    updated_at: string;
}

/** Docling processing job */
export interface DoclingJob {
    id: string;
    library_id: string;
    library_name: string;
    file_path: string;
    operation: 'ingest' | 'update' | 'delete';
    priority: 'high' | 'normal' | 'low';
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress?: number;             // 0-100
    error?: string;
    created_at: string;
    started_at?: string;
    completed_at?: string;
    updated_at?: string;
}

/** Create a new library */
export function createLibrary(
    id: string,
    name: string,
    folderPath: string,
    autoScan: boolean = true
): Library {
    const now = new Date().toISOString();
    return {
        id,
        name,
        folder_path: folderPath,
        document_count: 0,
        chunk_count: 0,
        total_size_bytes: 0,
        last_scanned: now,
        scan_status: 'idle',
        auto_scan: autoScan,
        created_at: now,
        updated_at: now,
    };
}
