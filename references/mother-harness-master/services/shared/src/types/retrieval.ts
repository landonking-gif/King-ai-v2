/**
 * Retrieval Reporting Types
 * Captures RAG retrieval diagnostics for artifacts and auditing
 */

export interface RetrievalChunkSummary {
    chunk_id: string;
    library_id: string;
    document_id: string;
    document_name: string;
    file_path: string;
    page_number?: number;
    chunk_index?: number;
    score: number;
    content_preview: string;
    content?: string;
}

export interface RetrievalReport {
    query: string;
    query_chunks: string[];
    library_ids: string[];
    embedding_model: string;
    requested_k: number;
    retrieved_at: string;
    results: RetrievalChunkSummary[];
}
