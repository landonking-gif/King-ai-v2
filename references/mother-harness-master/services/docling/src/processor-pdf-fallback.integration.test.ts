/**
 * Integration tests for PDF fallback parsing
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as fs from 'fs/promises';
import * as path from 'path';
import { DocumentProcessor } from './processor.js';

// Mock Redis and LLM dependencies
vi.mock('@mother-harness/shared', () => ({
    getRedisJSON: () => ({
        get: vi.fn(async () => null),
        set: vi.fn(async () => undefined),
    }),
    getLLMClient: () => ({
        embedText: vi.fn(async (text: string) => {
            // Return a simple mock embedding
            return new Array(384).fill(0).map((_, i) => Math.sin(i * text.length));
        }),
    }),
}));

// Mock pdf-parse to simulate PDF extraction
vi.mock('pdf-parse', () => ({
    default: vi.fn(async (buffer: Buffer) => {
        // Simple mock: extract text based on buffer content
        const content = buffer.toString('utf-8');

        if (content.includes('MOCK_PDF_ERROR')) {
            throw new Error('PDF parsing failed');
        }

        return {
            text: 'Sample PDF content from fallback parser.\n\fPage 2 content here.',
            numpages: 2,
            info: {
                Title: 'Test Document',
                Author: 'Test Author',
            },
        };
    }),
}));

describe('DocumentProcessor PDF fallback', () => {
    let processor: DocumentProcessor;
    let tempDir: string;

    beforeEach(async () => {
        processor = new DocumentProcessor();
        tempDir = path.join(process.cwd(), 'test-temp-pdf');
        await fs.mkdir(tempDir, { recursive: true });
    });

    afterEach(async () => {
        try {
            await fs.rm(tempDir, { recursive: true, force: true });
        } catch {
            // Ignore cleanup errors
        }
    });

    it('falls back to pdf-parse when Docling API is unavailable', async () => {
        // Mock fetch to simulate Docling API failure
        const fetchMock = vi.fn(async () => {
            throw new Error('Docling API unavailable');
        });
        vi.stubGlobal('fetch', fetchMock);

        // Create a mock PDF file
        const pdfPath = path.join(tempDir, 'test.pdf');
        await fs.writeFile(pdfPath, Buffer.from('MOCK_PDF_CONTENT'));

        const result = await (processor as any).extractPdf(
            pdfPath,
            await fs.stat(pdfPath)
        );

        expect(result.text).toContain('Sample PDF content');
        expect(result.pages).toHaveLength(2);
        expect(result.metadata.title).toBe('Test Document');
        expect(result.metadata.author).toBe('Test Author');
        expect(result.metadata.page_count).toBe(2);
    });

    it('uses Docling API when available', async () => {
        const doclingResponse = {
            text: 'Content from Docling API',
            pages: [
                { page_number: 1, content: 'Page 1 from API' },
            ],
            metadata: {
                title: 'API Document',
                page_count: 1,
            },
        };

        const fetchMock = vi.fn(async () => ({
            ok: true,
            json: async () => doclingResponse,
        }));
        vi.stubGlobal('fetch', fetchMock);

        const pdfPath = path.join(tempDir, 'test-api.pdf');
        await fs.writeFile(pdfPath, Buffer.from('MOCK_PDF_CONTENT'));

        const result = await (processor as any).extractWithDoclingApi(
            pdfPath,
            await fs.stat(pdfPath)
        );

        expect(fetchMock).toHaveBeenCalled();
        expect(result.text).toBe('Content from Docling API');
        expect(result.pages).toHaveLength(1);
        expect(result.metadata.title).toBe('API Document');
    });

    it('extracts multiple pages correctly with fallback parser', async () => {
        const fetchMock = vi.fn(async () => {
            throw new Error('API down');
        });
        vi.stubGlobal('fetch', fetchMock);

        const pdfPath = path.join(tempDir, 'multipage.pdf');
        await fs.writeFile(pdfPath, Buffer.from('MOCK_PDF_CONTENT'));

        const result = await (processor as any).extractPdf(
            pdfPath,
            await fs.stat(pdfPath)
        );

        expect(result.pages.length).toBeGreaterThan(0);
        expect(result.pages[0]).toHaveProperty('page_number');
        expect(result.pages[0]).toHaveProperty('content');
        expect(result.metadata.page_count).toBe(result.pages.length);
    });

    it('handles empty PDF gracefully', async () => {
        const pdfParseMock = await import('pdf-parse');
        vi.mocked(pdfParseMock.default).mockResolvedValueOnce({
            text: '',
            numpages: 0,
            info: {},
        } as any);

        const pdfPath = path.join(tempDir, 'empty.pdf');
        await fs.writeFile(pdfPath, Buffer.from('MOCK_EMPTY_PDF'));

        const result = await (processor as any).extractWithPdfParse(
            pdfPath,
            await fs.stat(pdfPath)
        );

        expect(result.pages).toHaveLength(1);
        expect(result.pages[0]?.content).toBe('');
        expect(result.metadata.page_count).toBe(0);
    });

    it('preserves metadata from pdf-parse', async () => {
        const pdfParseMock = await import('pdf-parse');
        vi.mocked(pdfParseMock.default).mockResolvedValueOnce({
            text: 'Document content',
            numpages: 1,
            info: {
                Title: 'Custom Title',
                Author: 'Custom Author',
                Subject: 'Test Subject',
                Keywords: 'test, pdf',
            },
        } as any);

        const pdfPath = path.join(tempDir, 'metadata.pdf');
        await fs.writeFile(pdfPath, Buffer.from('MOCK_PDF'));

        const result = await (processor as any).extractWithPdfParse(
            pdfPath,
            await fs.stat(pdfPath)
        );

        expect(result.metadata.title).toBe('Custom Title');
        expect(result.metadata.author).toBe('Custom Author');
        expect(result.metadata.file_type).toBe('pdf');
    });
});
