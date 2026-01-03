/**
 * Personal Knowledge Management (PKM) Module
 * Graph-based note storage with bidirectional links
 */

import { getRedisJSON, getRedisClient } from '../redis/index.js';
import { nanoid } from 'nanoid';

/** PKM Note */
export interface PKMNote {
    id: string;
    user_id: string;
    title: string;
    content: string;
    tags: string[];

    // Links
    outgoing_links: string[];  // Notes this note links to
    backlinks: string[];        // Notes that link to this note

    // Metadata
    created_at: string;
    updated_at: string;
    accessed_at: string;

    // Source tracking
    source_type?: 'manual' | 'agent' | 'research' | 'decision';
    source_task_id?: string;

    // Status
    archived: boolean;
    starred: boolean;
}

/** PKM Search result */
export interface PKMSearchResult {
    note: PKMNote;
    score: number;
    highlights: string[];
}

export class PKMManager {
    private redis = getRedisJSON();
    private redisClient = getRedisClient();
    private readonly notePrefix = 'pkm:note:';
    private readonly indexPrefix = 'pkm:index:';

    /**
     * Create a new note
     */
    async createNote(
        userId: string,
        title: string,
        content: string,
        options: {
            tags?: string[];
            source_type?: PKMNote['source_type'];
            source_task_id?: string;
        } = {}
    ): Promise<PKMNote> {
        const now = new Date().toISOString();

        const note: PKMNote = {
            id: `note-${nanoid()}`,
            user_id: userId,
            title,
            content,
            tags: options.tags ?? [],
            outgoing_links: [],
            backlinks: [],
            created_at: now,
            updated_at: now,
            accessed_at: now,
            source_type: options.source_type ?? 'manual',
            ...(options.source_task_id !== undefined && { source_task_id: options.source_task_id }),
            archived: false,
            starred: false,
        };

        // Extract links from content [[note-id]] or [[Note Title]]
        note.outgoing_links = this.extractLinks(content);

        // Save note
        await this.redis.set(`${this.notePrefix}${note.id}`, '$', note);

        // Update backlinks in linked notes
        await this.updateBacklinks(note.id, note.outgoing_links);

        // Update tag index
        await this.updateTagIndex(userId, note.id, note.tags);

        return note;
    }

    /**
     * Update a note
     */
    async updateNote(
        noteId: string,
        updates: Partial<Pick<PKMNote, 'title' | 'content' | 'tags'>>
    ): Promise<PKMNote | null> {
        const note = await this.getNote(noteId);
        if (!note) return null;

        const oldLinks = note.outgoing_links;

        if (updates.title !== undefined) {
            note.title = updates.title;
        }

        if (updates.content !== undefined) {
            note.content = updates.content;
            note.outgoing_links = this.extractLinks(updates.content);
        }

        if (updates.tags !== undefined) {
            // Remove from old tags
            await this.removeFromTagIndex(note.user_id, noteId, note.tags);
            note.tags = updates.tags;
            await this.updateTagIndex(note.user_id, noteId, note.tags);
        }

        note.updated_at = new Date().toISOString();

        // Update backlinks if links changed
        const removedLinks = oldLinks.filter(l => !note.outgoing_links.includes(l));
        const addedLinks = note.outgoing_links.filter(l => !oldLinks.includes(l));

        await this.removeBacklinks(noteId, removedLinks);
        await this.updateBacklinks(noteId, addedLinks);

        await this.redis.set(`${this.notePrefix}${noteId}`, '$', note);

        return note;
    }

    /**
     * Get a note by ID
     */
    async getNote(noteId: string): Promise<PKMNote | null> {
        const note = await this.redis.get<PKMNote>(`${this.notePrefix}${noteId}`);

        if (note) {
            // Update access time
            note.accessed_at = new Date().toISOString();
            await this.redis.set(`${this.notePrefix}${noteId}`, '$.accessed_at', note.accessed_at);
        }

        return note;
    }

    /**
     * Search notes
     */
    async searchNotes(
        userId: string,
        query: string,
        options: {
            limit?: number;
            tags?: string[];
            include_archived?: boolean;
        } = {}
    ): Promise<PKMSearchResult[]> {
        const limit = options.limit ?? 20;

        // Note: When RediSearch is configured, this should use FT.SEARCH for full-text search
        // For now, using in-memory search with TF-IDF-like scoring

        const keys = await this.redis.keys(`${this.notePrefix}*`);
        const results: PKMSearchResult[] = [];
        const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);

        for (const key of keys) {
            const note = await this.redis.get<PKMNote>(key);
            if (!note) continue;

            // Filter by user
            if (note.user_id !== userId) continue;

            // Filter archived
            if (!options.include_archived && note.archived) continue;

            // Filter by tags
            if (options.tags && options.tags.length > 0) {
                const hasTags = options.tags.every(t => note.tags.includes(t));
                if (!hasTags) continue;
            }

            // TF-IDF-like scoring
            let score = 0;
            const lowerTitle = note.title.toLowerCase();
            const lowerContent = note.content.toLowerCase();

            for (const term of queryTerms) {
                // Title matches score higher
                if (lowerTitle.includes(term)) score += 3;
                // Content matches
                const contentMatches = (lowerContent.match(new RegExp(term, 'gi')) || []).length;
                score += Math.min(contentMatches, 5); // Cap contribution per term
            }

            // Boost starred notes
            if (note.starred) score *= 1.2;

            // Boost recently accessed
            const daysSinceAccess = (Date.now() - new Date(note.accessed_at).getTime()) / (1000 * 60 * 60 * 24);
            if (daysSinceAccess < 7) score *= 1.1;

            if (score > 0) {
                results.push({
                    note,
                    score,
                    highlights: this.extractHighlights(note.content, query),
                });
            }
        }

        return results
            .sort((a, b) => b.score - a.score)
            .slice(0, limit);
    }

    /**
     * Get notes by tag
     */
    async getNotesByTag(userId: string, tag: string): Promise<PKMNote[]> {
        const noteIds = await this.redisClient.smembers(`${this.indexPrefix}${userId}:tag:${tag}`);
        const notes: PKMNote[] = [];

        for (const id of noteIds) {
            const note = await this.getNote(id);
            if (note && !note.archived) {
                notes.push(note);
            }
        }

        return notes;
    }

    /**
     * Get related notes (via links and backlinks)
     */
    async getRelatedNotes(noteId: string, depth: number = 1): Promise<PKMNote[]> {
        const note = await this.getNote(noteId);
        if (!note) return [];

        const related = new Set<string>();
        const toVisit = [...note.outgoing_links, ...note.backlinks];
        let currentDepth = 0;

        while (currentDepth < depth && toVisit.length > 0) {
            const visiting = [...toVisit];
            toVisit.length = 0;

            for (const id of visiting) {
                if (related.has(id) || id === noteId) continue;
                related.add(id);

                const relatedNote = await this.getNote(id);
                if (relatedNote && currentDepth < depth - 1) {
                    toVisit.push(...relatedNote.outgoing_links, ...relatedNote.backlinks);
                }
            }

            currentDepth++;
        }

        const notes: PKMNote[] = [];
        for (const id of related) {
            const n = await this.getNote(id);
            if (n) notes.push(n);
        }

        return notes;
    }

    /**
     * Extract [[wiki-style]] links from content
     */
    private extractLinks(content: string): string[] {
        const linkPattern = /\[\[([^\]]+)\]\]/g;
        const links: string[] = [];
        let match;

        while ((match = linkPattern.exec(content)) !== null) {
            const link = match[1];
            if (link) {
                links.push(link);
            }
        }

        return [...new Set(links)];
    }

    /**
     * Extract highlighted snippets around query matches
     */
    private extractHighlights(content: string, query: string): string[] {
        const highlights: string[] = [];
        const lowerContent = content.toLowerCase();
        const lowerQuery = query.toLowerCase();

        let pos = 0;
        while ((pos = lowerContent.indexOf(lowerQuery, pos)) !== -1) {
            const start = Math.max(0, pos - 50);
            const end = Math.min(content.length, pos + query.length + 50);
            highlights.push('...' + content.slice(start, end) + '...');
            pos += query.length;

            if (highlights.length >= 3) break;
        }

        return highlights;
    }

    /**
     * Update backlinks in linked notes
     */
    private async updateBacklinks(sourceId: string, targetIds: string[]): Promise<void> {
        for (const targetId of targetIds) {
            const target = await this.redis.get<PKMNote>(`${this.notePrefix}${targetId}`);
            if (target && !target.backlinks.includes(sourceId)) {
                target.backlinks.push(sourceId);
                await this.redis.set(`${this.notePrefix}${targetId}`, '$.backlinks', target.backlinks);
            }
        }
    }

    /**
     * Remove backlinks from notes
     */
    private async removeBacklinks(sourceId: string, targetIds: string[]): Promise<void> {
        for (const targetId of targetIds) {
            const target = await this.redis.get<PKMNote>(`${this.notePrefix}${targetId}`);
            if (target) {
                target.backlinks = target.backlinks.filter(id => id !== sourceId);
                await this.redis.set(`${this.notePrefix}${targetId}`, '$.backlinks', target.backlinks);
            }
        }
    }

    /**
     * Update tag index
     */
    private async updateTagIndex(userId: string, noteId: string, tags: string[]): Promise<void> {
        for (const tag of tags) {
            await this.redisClient.sadd(`${this.indexPrefix}${userId}:tag:${tag}`, noteId);
        }
    }

    /**
     * Remove from tag index
     */
    private async removeFromTagIndex(userId: string, noteId: string, tags: string[]): Promise<void> {
        for (const tag of tags) {
            await this.redisClient.srem(`${this.indexPrefix}${userId}:tag:${tag}`, noteId);
        }
    }
}

// Singleton
let pkmInstance: PKMManager | null = null;

export function getPKM(): PKMManager {
    if (!pkmInstance) {
        pkmInstance = new PKMManager();
    }
    return pkmInstance;
}
